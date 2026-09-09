"""Explicitly activated, finite prefix-only regeneration; never edits originals.

The live generation supervisor does not import or activate this module. A later
coordinator may call run_regenerations while owning the actual inference lock.
The standalone CLI acquires that lock itself and reads its source journal only.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sqlite3
import threading
import time
import urllib.request

from .core import canonical, digest
from .corpus_source import file_sha256
from .corpus_worker import (Journal, PrefixReviewProvider, SourceChanged, _budget,
    _code_identity, _retryable, _stat, now, review_policy, validate_review_controls)
from .provider import ContextOverflow, NoRedirect
from .regeneration import (PrefixRegenerationProvider, RegenerationLedger,
    _receipt_matches, reason_codes, regeneration_messages, request_regeneration, revalidate_review)

VERSION = "cot.prefix-regeneration-worker/v1"


def implementation_identity():
    return {**_code_identity(), **{name: file_sha256(Path(__file__).with_name(name))
            for name in ("regeneration.py", "regeneration_worker.py")}}


def bind_worker(ledger, identity):
    expected = canonical({"schema": VERSION, "implementation": implementation_identity(), **identity})
    row = ledger.db.execute("SELECT value FROM cot_regen_meta WHERE key='worker_identity'").fetchone()
    if row is not None and row[0] != expected:
        raise ValueError("Regeneration worker/source/config identity changed; use an explicit new version")
    with ledger.db:
        ledger.db.execute("INSERT OR IGNORE INTO cot_regen_meta VALUES('worker_identity',?)", (expected,))


class ReadOnlyRejectedSource(Journal):
    """Reuse exact row/boundary verification without calling mutating Journal.__init__."""

    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("BEGIN")  # Fixed read-only snapshot for this finite queue.
            self.identity = json.loads(self.db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
            if canonical(self.identity.get("implementation_sha256")) != canonical(_code_identity()):
                raise ValueError("Original journal implementation differs from the loaded exact renderer/reviewer")
            if not self.db.execute("SELECT 1 FROM meta WHERE key='index_complete'").fetchone():
                raise ValueError("Original source index is incomplete")
            self.source = Path(self.identity["source_path"]).resolve(strict=True)
            self.signature = _stat(self.source)
            if file_sha256(self.source) != self.identity["source_sha256"] or _stat(self.source) != self.signature:
                raise SourceChanged("Original frozen source changed")
            self.cache = OrderedDict()
        except BaseException:
            self.db.close()
            raise

    def rejected_rows(self, limit, after_ordinal=0):
        if type(limit) is not int or limit < 1 or type(after_ordinal) is not int or after_ordinal < 0:
            raise ValueError("Finite queue needs a positive limit and nonnegative ordinal cursor")
        return self.db.execute("""SELECT g.* FROM gaps g JOIN candidates c ON c.gap_id=g.id
            JOIN reviews r ON r.gap_id=g.id WHERE g.state='rejected' AND g.ordinal>?
            ORDER BY g.ordinal LIMIT ?""", (after_ordinal, limit))

    def entry(self, row):
        gap = self.load_gap(row)
        candidate = self.db.execute("SELECT * FROM candidates WHERE gap_id=?", (gap["id"],)).fetchone()
        review = self.db.execute("SELECT data FROM reviews WHERE gap_id=?", (gap["id"],)).fetchone()
        if candidate is None or review is None:
            raise ValueError("A rejected source candidate or review is missing")
        for key in ("prefix_hash", "lookahead_hash", "prompt_hash"):
            if candidate[key] != gap[key]:
                raise SourceChanged("Original candidate no longer matches its exact gap")
        record = json.loads(review[0])
        return gap, candidate["text"], record


def _strict_receipt(config, receipt):
    """Check a fresh *live callback* result, never accept a boolean/config file."""
    if not isinstance(receipt, dict) or receipt.get("schema") != "cot.live-strict-backends/v1":
        raise ValueError("Strict thinking requires an independently verified live backend callback")
    checked = receipt.get("checked_at_unix")
    if type(checked) not in (int, float) or not 0 <= time.time() - checked <= 30:
        raise ValueError("Strict backend verification is stale or not a live timestamp")
    if receipt.get("config_hash") != digest(config):
        raise ValueError("Live backend verification is bound to a different configuration")
    routes, backends = receipt.get("routed_backend_ids"), receipt.get("backends")
    if (not isinstance(routes, list) or not routes or any(not isinstance(i, str) for i in routes)
            or len(set(routes)) != len(routes) or not isinstance(backends, list)):
        raise ValueError("Live routing inventory is missing")
    if len(backends) != len(routes) or set(b.get("id") for b in backends) != set(routes):
        raise ValueError("Every currently routable backend must be checked")
    for backend in backends:
        args = backend.get("server_args", {})
        if (args.get("enable_strict_thinking") is not True
                or args.get("grammar_backend") != "xgrammar"
                or args.get("served_model_name") != config["model"]
                or type(args.get("context_length")) is not int
                or args["context_length"] < int(config["context_limit"])):
            raise ValueError("A routed backend does not satisfy the actual strict model/context settings")
        for key in ("model_path", "revision"):
            if key in config and args.get(key) != config[key]:
                raise ValueError("A routed backend differs from the pinned model path/revision")
    # Store only the identity/hash of the server arguments, never arbitrary server settings.
    return {"schema": receipt["schema"], "config_hash": receipt["config_hash"],
            "checked_at_unix": checked, "routed_backend_ids": routes,
            "server_args_hashes": {b["id"]: digest(b["server_args"]) for b in backends}}


def verify_direct_strict(config):
    """Live verifier for a single direct SGLang endpoint, never a fleet router."""
    base = config["base_url"].rstrip("/")
    if not base.endswith("/v1"):
        raise ValueError("Direct server verification expects an explicit /v1 endpoint")
    wire = urllib.request.Request(base[:-3] + "/get_server_info")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(wire, timeout=10) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("Server identity response exceeds its size bound")
        data = json.loads(raw)
    args = data.get("server_args", {})
    return {"schema": "cot.live-strict-backends/v1", "config_hash": digest(config),
            "checked_at_unix": time.time(), "routed_backend_ids": ["direct"],
            "backends": [{"id": "direct", "server_args": args}]}


def _readiness(config, callback):
    if config.get("require_strict_thinking_server") is not True:
        return None
    if callback is None:
        raise ValueError("No live strict-backend readiness callback: no inference permitted")
    return _strict_receipt(config, callback(config))


def _validate_review(gap, candidate, raw, metadata, provider):
    config = provider.config
    _receipt_matches(metadata, config)
    if metadata.get("response_model") != config["model"]:
        raise ValueError("Actual review response model differs from the configured model")
    expected = {"model": config["model"], "max_tokens": int(config["max_output_tokens"]),
                "temperature": config.get("temperature", 0.0),
                "reasoning_effort": config.get("reasoning_effort")}
    if any(canonical(metadata["parameters"].get(k)) != canonical(v) for k, v in expected.items()):
        raise ValueError("Actual review model/output/temperature/reasoning settings changed")
    if metadata.get("tokenizer_sha256") != provider.tokenizer.identity:
        raise ValueError("Actual reviewer tokenizer identity changed")
    if metadata.get("prompt_tokens_local") != provider.tokenizer.count(
            review_policy(config).review_messages(gap, candidate), config.get("chat_template_kwargs", {})):
        raise ValueError("Review receipt differs from the exact current local prompt rendering")
    record = review_policy(config).validate_review(gap, candidate, raw, reviewer=metadata)
    record.update(prompt_token_parity_verified=True, configured_template_options_verified=True,
                  configured_decoding_controls_verified=True, semantic_certainty_claimed=False)
    return record


def _failure_category(exc):
    if isinstance(exc, ContextOverflow):
        return "context_overflow"
    if _retryable(exc):
        return "transport"
    return "malformed"


def _attempt(ledger, gap, request, generator, reviewer, readiness, stop):
    aid = request["attempt_id"]
    if ledger._event(aid, "failure"):
        return "excluded"
    if ledger._event(aid, "review"):
        return ledger._event(aid, "review")["state"]
    generated = ledger._event(aid, "generation")
    if generated is None:
        response = ledger._event(aid, "generation_response")
        if response is None:
            if ledger._event(aid, "generate_started"):
                ledger.failure(aid, "interrupted")
                return "excluded"  # Unknown request outcome is never silently replayed.
            if stop.is_set():
                return "pending"
            # The downstream reviewer must be genuinely ready before spending
            # any generation capacity, including a non-strict generator call.
            live = {"generator": _readiness(generator.config, readiness),
                    "reviewer": _readiness(reviewer.config, readiness)}
            _budget(generator, regeneration_messages(gap, request))
            ledger._append(aid, "generate_started", {"at": now(), "live_readiness": live})
            text, metadata = generator.generate(gap, request)
            response = {"text": text, "metadata": metadata}
            ledger._append(aid, "generation_response", response)
        if response["metadata"].get("generator", {}).get("tokenizer_sha256") != generator.tokenizer.identity:
            raise ValueError("Actual generation tokenizer identity changed")
        if response["metadata"].get("prompt_tokens_local") != generator.tokenizer.count(
                regeneration_messages(gap, request), generator.config.get("chat_template_kwargs", {})):
            raise ValueError("Generation receipt differs from the exact current local prompt rendering")
        ledger.generation_done(gap, aid, response["text"], response["metadata"])
        generated = ledger._event(aid, "generation")
    if not generated["valid"]:
        return "generation_invalid"
    response = ledger._event(aid, "review_response")
    if response is None:
        if ledger._event(aid, "review_started"):
            ledger.failure(aid, "interrupted")
            return "excluded"
        if stop.is_set():
            return "pending"
        live = _readiness(reviewer.config, readiness)
        _budget(reviewer, review_policy(reviewer.config).review_messages(gap, generated["text"]))
        ledger._append(aid, "review_started", {"at": now(), "live_readiness": live})
        raw, metadata = reviewer.review(gap, generated["text"])
        response = {"raw": raw, "metadata": metadata}
        ledger._append(aid, "review_response", response)
    record = _validate_review(gap, generated["text"], response["raw"], response["metadata"], reviewer)
    return ledger.review_done(gap, aid, record)


def run_regenerations(entries, ledger, generator, reviewer, *, max_gaps, readiness=None,
                      stop=None, progress=None):
    """Caller owns shared inference and ledger locks; every entry is an original gap.

    Sequential admission deliberately bounds this optional retry path to one
    model request at a time. A queue is finite and each gap gets at most two
    replacement attempts. No transport retries or new approval of originals.
    """
    if type(max_gaps) is not int or max_gaps < 1:
        raise ValueError("max_gaps must be a positive finite integer")
    for actual, expected in ((generator.config, ledger.generator_config), (reviewer.config, ledger.reviewer_config)):
        if canonical(actual) != canonical(expected):
            raise ValueError("Provider differs from immutable regeneration configuration")
    validate_review_controls(reviewer.config)
    for config in (generator.config, reviewer.config):
        if config.get("require_strict_thinking_server") is True and readiness is None:
            raise ValueError("Strict backend live readiness is required before any model call")
    bind_worker(ledger, {"generator_tokenizer": generator.tokenizer.identity,
                         "reviewer_tokenizer": reviewer.tokenizer.identity})
    stop = stop or threading.Event()
    summary = {"schema": VERSION, "selected": 0, "states": {}, "semantic_certainty_claimed": False}
    for gap, candidate, review in entries:
        if stop.is_set() or summary["selected"] >= max_gaps:
            break
        summary["selected"] += 1
        # Recheck original lineage even when only resuming an already committed attempt.
        try:
            revalidate_review(gap, candidate, review, ledger.reviewer_config)
        except (ValueError, KeyError, TypeError):
            # Malformed/unfinished originals never reserve a model attempt.
            state = "invalid_original_review"
            summary["states"][state] = summary["states"].get(state, 0) + 1
            continue
        if not reason_codes(review):
            summary["states"]["ineligible"] = summary["states"].get("ineligible", 0) + 1
            continue
        rows = ledger.db.execute("SELECT id FROM cot_regen_attempts WHERE gap_id=? ORDER BY attempt_number", (gap["id"],)).fetchall()
        if rows and canonical(ledger._request(rows[0][0])) != canonical(request_regeneration(
                gap, candidate, review, ledger.reviewer_config, attempt_number=1)):
            raise ValueError("Original rejection lineage changed on regeneration resume")
        request = ledger._request(rows[-1][0]) if rows else ledger.reserve(gap, candidate, review)
        state = "ineligible"
        while request is not None:
            try:
                state = _attempt(ledger, gap, request, generator, reviewer, readiness, stop)
            except Exception as exc:
                if not ledger._event(request["attempt_id"], "failure"):
                    ledger.failure(request["attempt_id"], _failure_category(exc))
                state = "excluded"
            if state != "rejected" or stop.is_set():
                break
            generated = ledger._event(request["attempt_id"], "generation")
            reviewed = ledger._event(request["attempt_id"], "review")
            if not reason_codes(reviewed["record"]):
                break
            request = ledger.reserve(gap, generated["text"], reviewed["record"])
        summary["states"][state] = summary["states"].get(state, 0) + 1
        if progress:
            progress(summary)
    summary["stopped"] = stop.is_set()
    return summary


def accepted_selections(ledger):
    """Separate append-only selections; this does not export or replace originals."""
    for row in ledger.db.execute("SELECT a.id,a.gap_id,e.payload FROM cot_regen_attempts a JOIN cot_regen_events e ON e.attempt_id=a.id WHERE e.kind='review' ORDER BY a.rowid"):
        reviewed = json.loads(row[2])
        if reviewed["state"] == "accepted":
            yield {"gap_id": row[1], "attempt_id": row[0], "candidate_hash": reviewed["candidate_hash"]}


def _cohort_result(result, queue, after_ordinal):
    complete = (not result["stopped"] and result["selected"] == len(queue)
                and result["states"].get("pending", 0) == 0)
    return {**result, "cohort_complete": complete, "resume_same_ledger_required": not complete,
            "next_after_ordinal": max((row["ordinal"] for row in queue), default=after_ordinal) if complete else None}


def _load_hook(path, expected_sha):
    file = Path(path).resolve(strict=True)
    if not expected_sha or file_sha256(file) != expected_sha:
        raise ValueError("Live readiness hook code hash differs from the explicit expected hash")
    spec = importlib.util.spec_from_file_location("cot_regeneration_live_readiness", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    callback = getattr(module, "verify_live", None)
    if not callable(callback):
        raise ValueError("Live readiness hook must implement verify_live(config)")
    return callback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-journal", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--inference-lock", required=True)
    parser.add_argument("--max-gaps", type=int, required=True)
    parser.add_argument("--after-ordinal", type=int, default=0, help="Select only rejected original ordinals after this cursor; immutable per ledger")
    parser.add_argument("--confirm-run", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--direct-strict-server", action="store_true", help="Only for an actual direct SGLang endpoint; never a fleet router")
    group.add_argument("--live-readiness-hook", help="Pinned local Python code that freshly inspects all routed backends on every call")
    parser.add_argument("--live-readiness-hook-sha256")
    args = parser.parse_args(argv)
    if not args.confirm_run or args.max_gaps < 1 or args.after_ordinal < 0:
        parser.error("Actual generation/review requires --confirm-run and a finite positive --max-gaps")
    original, output, lockpath = [Path(p).resolve() for p in (args.original_journal, args.ledger, args.inference_lock)]
    if len({original, output, lockpath}) != 3:
        parser.error("Original journal, new ledger and shared lock must be distinct")
    with open(lockpath, "r+") as shared, open(str(output) + ".worker.lock", "a+") as own:
        try:
            fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Actual inference or ledger lock is occupied; no requests sent")
        source = ReadOnlyRejectedSource(original)
        try:
            if output in {source.source, Path(str(original) + "-wal"), Path(str(original) + "-shm")}:
                raise ValueError("Regeneration ledger must not overwrite source artifacts")
            genconfig, revconfig = source.identity["generator_config"], source.identity["reviewer_config"]
            generator, reviewer = PrefixRegenerationProvider(genconfig), PrefixReviewProvider(revconfig)
            for key, value in (("generator_tokenizer", generator.tokenizer.identity), ("reviewer_tokenizer", reviewer.tokenizer.identity)):
                expected = source.identity.get(key)
                if expected is not None and expected != value:
                    raise ValueError("Original configured tokenizer bytes changed")
            callback = verify_direct_strict if args.direct_strict_server else None
            if args.live_readiness_hook:
                callback = _load_hook(args.live_readiness_hook, args.live_readiness_hook_sha256)
            db = sqlite3.connect(output)
            try:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=FULL")
                ledger = RegenerationLedger(db, genconfig, revconfig)
                queue = list(source.rejected_rows(args.max_gaps, args.after_ordinal))
                signature = []
                for row in queue:
                    candidate = source.db.execute("SELECT text FROM candidates WHERE gap_id=?", (row["id"],)).fetchone()[0]
                    record = source.db.execute("SELECT data FROM reviews WHERE gap_id=?", (row["id"],)).fetchone()[0]
                    signature.append({"gap_id": row["id"], "ordinal": row["ordinal"], "candidate_hash": digest(candidate), "review_hash": digest(json.loads(record))})
                activation = {"original_journal": str(original), "original_identity_hash": digest(source.identity),
                              "queue": signature, "max_gaps": args.max_gaps,
                              "after_ordinal": args.after_ordinal,
                              "readiness_hook_sha256": args.live_readiness_hook_sha256,
                              "direct_strict_server": args.direct_strict_server}
                expected = canonical(activation)
                previous = db.execute("SELECT value FROM cot_regen_meta WHERE key='activation'").fetchone()
                if previous is not None and previous[0] != expected:
                    raise ValueError("Finite source queue or activation changed; use a new ledger")
                with db:
                    db.execute("INSERT OR IGNORE INTO cot_regen_meta VALUES('activation',?)", (expected,))
                stop = threading.Event()
                handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
                for sig in handlers:
                    signal.signal(sig, lambda *_: stop.set())
                try:
                    result = run_regenerations((source.entry(row) for row in queue), ledger, generator, reviewer,
                        max_gaps=args.max_gaps, readiness=callback, stop=stop,
                        progress=lambda item: print(canonical(item), flush=True))
                    result = _cohort_result(result, queue, args.after_ordinal)
                    print(canonical(result), flush=True)
                finally:
                    for sig, handler in handlers.items():
                        signal.signal(sig, handler)
            finally:
                db.close()
        finally:
            source.close()


if __name__ == "__main__":
    main()
