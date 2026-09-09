"""Bounded, resumable original-source CoT generation and prefix-only review.

SQLite journals source offsets and event IDs, not copied historical prefixes.
All mutations occur in the coordinator thread; workers make bounded model calls.
Actual evaluator decisions are retained. Uncertain/rejected or structurally
invalid candidates never become approved. Model review is not a leakage proof.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid

from .core import BLOCKING_FLAGS, PROMPT_VERSION, canonical, digest, prompt_messages, validate_candidate, validate_trace
from .corpus_source import file_sha256
from .grounding_review_v3 import REVIEW_VERSION, review_messages, validate_review
from . import grounding_review_v3, grounding_review_v4
from .provider import ContextOverflow, NoRedirect, OpenAICompatibleProvider, check_budget


WORKER_VERSION = "cot.compact-corpus-worker/v5-generation-staging"


def review_policy(config):
    version = config.get("review_version", REVIEW_VERSION)
    policies = {grounding_review_v3.REVIEW_VERSION: grounding_review_v3,
                grounding_review_v4.REVIEW_VERSION: grounding_review_v4}
    if version not in policies:
        raise ValueError("Unknown configured prefix-review version")
    return policies[version]


def validate_review_controls(config):
    """Allow only explicit budget/schema controls, never arbitrary API overrides."""
    if "custom_params" in config:
        params = config["custom_params"]
        if not isinstance(params, dict) or set(params) != {"thinking_budget"}:
            raise ValueError("Review custom_params supports only thinking_budget")
        budget = params["thinking_budget"]
        if type(budget) is not int or not 1 <= budget < int(config["max_output_tokens"]):
            raise ValueError("Thinking budget must be a positive integer below the total output limit")
        if config.get("chat_template_kwargs", {}).get("thinking") is not True:
            raise ValueError("A bounded-thinking reviewer must keep thinking enabled")
        if config.get("require_strict_thinking_server") is not True:
            raise ValueError("Bounded thinking requires explicit strict server configuration requirement")
    if "response_format" in config:
        policy = review_policy(config)
        expected = policy.response_format() if hasattr(policy, "response_format") else None
        if expected is None or canonical(config["response_format"]) != canonical(expected):
            raise ValueError("Response format must match the exact configured review schema")


def now():
    return datetime.now(timezone.utc).isoformat()


def _stat(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class SourceChanged(ValueError):
    pass


class Interrupted(ValueError):
    pass


class AdmissionOverflow(ValueError):
    pass


def gap_at(trace, event_index, source_digest=None):
    """Construct one exact v4 window without constructing all earlier gaps."""
    source_digest = source_digest or digest(trace)
    event = trace["events"][event_index]
    marker = next((g for g in trace["gap_targets"] if g["event_id"] == event["event_id"]), None)
    if event["role"] != "assistant" or marker is None:
        raise ValueError("Journal target is not an original assistant reasoning boundary")
    prefix = copy.deepcopy(trace["events"][:event_index])
    lookahead = copy.deepcopy(trace["events"][event_index:event_index + 5])
    gap = {
        "id": digest({"source_digest": source_digest, "event_id": event["event_id"], "window_version": "original-visible/v1"}),
        "trace_id": trace["trace_id"], "source_digest": source_digest,
        "source_format": trace.get("source_format", "canonical"), "event_index": event_index,
        "event_id": event["event_id"], "target": copy.deepcopy(event),
        "prefix": prefix, "lookahead": lookahead, "prefix_hash": digest(prefix),
        "lookahead_hash": digest(lookahead), "gap_policy": "markers", "marker_provenance": marker,
        "flags": ["synthetic", "lookahead_conditioned", "semantic_review_required"], "prompt_version": PROMPT_VERSION,
    }
    gap["prompt_hash"] = digest(prompt_messages(gap))
    return gap


class TokenAdmission:
    def __init__(self, max_requests, max_tokens):
        if type(max_requests) is not int or max_requests < 1 or type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Admission limits must be positive integers")
        self.max_requests, self.max_tokens = max_requests, max_tokens
        self.requests = self.tokens = self.peak_requests = self.peak_tokens = 0
        self.condition = threading.Condition()

    @contextmanager
    def reserve(self, weight, stop):
        if weight > self.max_tokens:
            raise AdmissionOverflow("Request reservation exceeds the configured token-admission budget")
        with self.condition:
            while self.requests >= self.max_requests or self.tokens + weight > self.max_tokens:
                if stop.is_set():
                    raise Interrupted("Stopped before request admission")
                self.condition.wait(timeout=0.2)
            if stop.is_set():
                raise Interrupted("Stopped before request admission")
            self.requests += 1
            self.tokens += weight
            self.peak_requests = max(self.peak_requests, self.requests)
            self.peak_tokens = max(self.peak_tokens, self.tokens)
        try:
            yield
        finally:
            with self.condition:
                self.requests -= 1
                self.tokens -= weight
                self.condition.notify_all()


class PrefixReviewProvider(OpenAICompatibleProvider):
    """Existing provider configuration/tokenizer validation, dedicated review input."""
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer=tokenizer)
        self.policy = review_policy(config)
        validate_review_controls(config)

    def review(self, gap, candidate):
        messages = self.policy.review_messages(gap, candidate)
        config = self.config
        options = config.get("chat_template_kwargs", {})
        count = self.tokenizer.count(messages, options)
        check_budget(count, int(config["max_output_tokens"]), int(config["context_limit"]), int(config.get("safety_margin", 256)))
        body = {"model": config["model"], "messages": messages, "max_tokens": int(config["max_output_tokens"]),
                "temperature": config.get("temperature", 0.0), "stream": False}
        for key in ("reasoning_effort", "chat_template_kwargs", "custom_params", "response_format"):
            if key in config:
                body[key] = config[key]
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(config.get("api_key_env", "COT_TEACHER_API_KEY"))
        if key:
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(config["base_url"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            with opener.open(request, timeout=int(config.get("timeout_seconds", 600))) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise ValueError("Review response exceeds size bound")
                result = json.loads(raw)
        except urllib.error.HTTPError as exc:
            raise ValueError(f"Teacher returned HTTP {exc.code}") from None
        except urllib.error.URLError:
            raise ValueError("Teacher connection failed; inspect endpoint access separately") from None
        choices = result.get("choices", [])
        if len(choices) != 1 or not isinstance(choices[0].get("message", {}).get("content"), str):
            raise ValueError("Reviewer returned no single textual JSON result")
        choice = choices[0]
        provenance = {"provider": "openai-compatible", "model": config["model"], "response_model": result.get("model"),
                      "finish_reason": choice.get("finish_reason"), "tokenizer_sha256": self.tokenizer.identity,
                      "prompt_tokens_local": count, "usage": {k: v for k, v in result.get("usage", {}).items() if isinstance(v, (int, float))},
                      "parameters": {k: v for k, v in body.items() if k not in {"messages", "stream"}},
                      "review_version": self.policy.REVIEW_VERSION}
        return choice["message"]["content"], provenance


def _retryable(exc):
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    message = str(exc)
    return message == "Teacher connection failed; inspect endpoint access separately" or message in {
        "Teacher returned HTTP " + str(code) for code in (408, 429, 500, 502, 503, 504)
    }


def _category(exc):
    if isinstance(exc, SourceChanged):
        return "source_changed"
    if isinstance(exc, Interrupted):
        return "interrupted"
    if isinstance(exc, ContextOverflow):
        return "context_overflow"
    if isinstance(exc, AdmissionOverflow):
        return "admission_overflow"
    if _retryable(exc):
        return "provider_unavailable"
    return "invalid_or_failed"


class Journal:
    def __init__(self, path, source, source_sha256, identity):
        self.path, self.source = Path(path), Path(source).resolve(strict=True)
        self.signature = _stat(self.source)
        if file_sha256(self.source) != source_sha256 or self.signature != _stat(self.source):
            raise SourceChanged("Source SHA-256 or file identity changed")
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sources(id INTEGER PRIMARY KEY,trace_id TEXT UNIQUE NOT NULL,offset INTEGER NOT NULL,
          length INTEGER NOT NULL,row_sha256 TEXT NOT NULL,source_digest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS gaps(ordinal INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,source_id INTEGER NOT NULL REFERENCES sources(id),
          event_index INTEGER NOT NULL,event_id TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'queued',invocation TEXT,updated TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS gap_state ON gaps(state,ordinal);
        CREATE INDEX IF NOT EXISTS gap_invocation ON gaps(state,invocation,ordinal);
        CREATE TABLE IF NOT EXISTS candidates(gap_id TEXT PRIMARY KEY REFERENCES gaps(id),text TEXT NOT NULL,data TEXT NOT NULL,
          prefix_hash TEXT NOT NULL,lookahead_hash TEXT NOT NULL,prompt_hash TEXT NOT NULL,created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS reviews(gap_id TEXT PRIMARY KEY REFERENCES candidates(gap_id),data TEXT NOT NULL,created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS responses(id INTEGER PRIMARY KEY,gap_id TEXT NOT NULL REFERENCES gaps(id),stage TEXT NOT NULL,
          text TEXT NOT NULL,data TEXT NOT NULL,created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS failures(id INTEGER PRIMARY KEY,gap_id TEXT REFERENCES gaps(id),stage TEXT,category TEXT,created TEXT);
        """)
        policy = review_policy(identity.get("reviewer_config", {}))
        if identity.get("review_version", policy.REVIEW_VERSION) != policy.REVIEW_VERSION:
            raise ValueError("Journal review version differs from its reviewer configuration")
        full_identity = {"source_path": str(self.source), "source_sha256": source_sha256, "worker_version": WORKER_VERSION,
                         "prompt_version": PROMPT_VERSION, "review_version": policy.REVIEW_VERSION, **identity}
        expected = canonical(full_identity)
        # Keep the same immutable configuration represented in the journal,
        # independent of later mutations to caller/provider dictionaries.
        self.identity = json.loads(expected)
        row = self.db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if row and row[0] != expected:
            self.db.close()
            raise ValueError("Journal run identity changed; use a fresh journal instead of rewriting provenance")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta VALUES('identity',?)", (expected,))
        self.cache = OrderedDict()
        self._index()
        with self.db:
            # Only this process holds the run lock. In-flight requests from a
            # previous process have no journal receipt and may need regeneration.
            self.db.execute("UPDATE gaps SET state='queued' WHERE state='generating'")
            self.db.execute("UPDATE gaps SET state='review_pending' WHERE state='reviewing'")

    def close(self):
        self.db.close()

    def verify_file(self):
        if _stat(self.source) != self.signature:
            raise SourceChanged("Original frozen source changed during the run")

    def _index(self):
        if self.db.execute("SELECT 1 FROM meta WHERE key='index_complete'").fetchone():
            return
        self.verify_file()
        with self.source.open("rb") as stream:
            while True:
                offset = stream.tell()
                raw = stream.readline(64 * 1024 * 1024 + 1)
                if not raw:
                    break
                if len(raw) > 64 * 1024 * 1024:
                    raise ValueError("Canonical source row exceeds 64 MiB bound")
                if not raw.strip():
                    continue
                trace = validate_trace(json.loads(raw))
                sha = hashlib.sha256(raw).hexdigest()
                source_digest = digest(trace)
                previous = self.db.execute("SELECT * FROM sources WHERE trace_id=?", (trace["trace_id"],)).fetchone()
                if previous:
                    if previous["offset"] != offset or previous["row_sha256"] != sha:
                        raise ValueError("Duplicate or changed trace identity in canonical source")
                    continue
                positions = {event["event_id"]: i for i, event in enumerate(trace["events"])}
                with self.db:
                    cursor = self.db.execute("INSERT INTO sources(trace_id,offset,length,row_sha256,source_digest) VALUES(?,?,?,?,?)",
                                             (trace["trace_id"], offset, len(raw), sha, source_digest))
                    for marker in trace.get("gap_targets", []):
                        event_id = marker["event_id"]
                        gap_id = digest({"source_digest": source_digest, "event_id": event_id, "window_version": "original-visible/v1"})
                        self.db.execute("INSERT INTO gaps(id,source_id,event_index,event_id,updated) VALUES(?,?,?,?,?)",
                                        (gap_id, cursor.lastrowid, positions[event_id], event_id, now()))
        self.verify_file()
        with self.db:
            self.db.execute("INSERT INTO meta VALUES('index_complete',?)", (now(),))

    def load_gap(self, row):
        self.verify_file()
        source = self.db.execute("SELECT * FROM sources WHERE id=?", (row["source_id"],)).fetchone()
        sid = source["id"]
        if sid not in self.cache:
            with self.source.open("rb") as stream:
                stream.seek(source["offset"])
                raw = stream.read(source["length"])
            if hashlib.sha256(raw).hexdigest() != source["row_sha256"]:
                raise SourceChanged("Source row bytes changed")
            trace = validate_trace(json.loads(raw))
            if digest(trace) != source["source_digest"]:
                raise SourceChanged("Source trace digest changed")
            self.cache[sid] = trace
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(sid)
        gap = gap_at(self.cache[sid], row["event_index"], source["source_digest"])
        if gap["id"] != row["id"] or gap["event_id"] != row["event_id"]:
            raise SourceChanged("Gap location changed")
        return gap

    def claim(self, invocation, allow_new, *, generation_only=False):
        if generation_only:
            if not allow_new:
                return None
            row = self.db.execute("SELECT * FROM gaps WHERE state='queued' ORDER BY ordinal LIMIT 1").fetchone()
            if row is None:
                return None
            with self.db:
                self.db.execute("UPDATE gaps SET state='generating',invocation=?,updated=? WHERE id=?", (invocation, now(), row["id"]))
            return dict(row), "generate", True
        row = self.db.execute("SELECT * FROM gaps WHERE state='review_pending' AND invocation=? ORDER BY ordinal LIMIT 1", (invocation,)).fetchone()
        new = False
        if row is None and allow_new:
            row = self.db.execute("SELECT * FROM gaps WHERE state='review_pending' ORDER BY ordinal LIMIT 1").fetchone()
            if row is None:
                row = self.db.execute("SELECT * FROM gaps WHERE state='queued' ORDER BY ordinal LIMIT 1").fetchone()
            new = row is not None
        if row is None:
            return None
        stage = "review" if row["state"] == "review_pending" else "generate"
        with self.db:
            self.db.execute("UPDATE gaps SET state=?,invocation=?,updated=? WHERE id=?", ("reviewing" if stage == "review" else "generating", invocation, now(), row["id"]))
        return dict(row), stage, new

    def candidate(self, gap_id):
        return self.db.execute("SELECT * FROM candidates WHERE gap_id=?", (gap_id,)).fetchone()

    def receipt(self, gap_id, stage, text, metadata):
        # Keep even invalid/truncated structured evaluator output for later audit.
        with self.db:
            self.db.execute("INSERT INTO responses(gap_id,stage,text,data,created) VALUES(?,?,?,?,?)", (gap_id, stage, text, canonical(metadata), now()))

    def generation_done(self, gap, text, data):
        self.verify_file()
        if data.get("generator", {}).get("prompt_version") != PROMPT_VERSION or data.get("prompt_hash") != gap["prompt_hash"]:
            raise ValueError("Generator receipt does not match exact current prompt")
        flags = sorted(set(validate_candidate(text) + data.get("flags", [])))
        valid = not BLOCKING_FLAGS.intersection(flags) and data.get("finish_reason") == "stop"
        if data.get("generator", {}).get("provider") == "fake" or "demo_fixture_not_quality_evidence" in flags:
            valid = False
        metadata = {**data, "flags": flags, "candidate_hash": digest(text), "synthetic": True, "lookahead_conditioned": True}
        with self.db:
            self.db.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?,?)", (gap["id"], text, canonical(metadata), gap["prefix_hash"], gap["lookahead_hash"], gap["prompt_hash"], now()))
            self.db.execute("UPDATE gaps SET state=?,updated=? WHERE id=?", ("review_pending" if valid else "generation_invalid", now(), gap["id"]))

    def review_done(self, gap, text, result, reviewer):
        self.verify_file()
        candidate = self.candidate(gap["id"])
        if candidate is None or candidate["text"] != text or candidate["prefix_hash"] != gap["prefix_hash"] or candidate["prompt_hash"] != gap["prompt_hash"]:
            raise ValueError("Reviewer candidate/source identity changed")
        if reviewer.get("provider") != "openai-compatible" or reviewer.get("model", "").startswith("fixture"):
            raise ValueError("Only an actual configured teacher reviewer may authorize bulk approval")
        config = self.identity.get("reviewer_config")
        if not isinstance(config, dict):
            raise ValueError("Journal has no configured reviewer receipt contract")
        usage = reviewer.get("usage")
        local = reviewer.get("prompt_tokens_local")
        remote = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if type(local) is not int or type(remote) is not int or local < 1 or remote != local:
            raise ValueError("Actual review prompt count differs from exact local rendering")
        parameters = reviewer.get("parameters")
        expected_options = config.get("chat_template_kwargs", {})
        actual_options = parameters.get("chat_template_kwargs", {}) if isinstance(parameters, dict) else None
        if (not isinstance(expected_options, dict) or not isinstance(actual_options, dict)
                or canonical(actual_options) != canonical(expected_options)):
            raise ValueError("Review template options differ from the immutable configured reviewer")
        validate_review_controls(config)
        for key in ("custom_params", "response_format"):
            if canonical(parameters.get(key)) != canonical(config.get(key)):
                raise ValueError("Review decoding controls differ from the immutable configured reviewer")
        record = review_policy(config).validate_review(gap, text, result, reviewer=reviewer)
        state = {"pass": "approved", "reject": "rejected", "uncertain": "uncertain"}[record["decision"]]
        record["automatic_approval"] = state == "approved"
        record["approval_policy"] = "user-authorized-prefix-model-review/v1" if state == "approved" else None
        record["semantic_certainty_claimed"] = False
        record["prompt_token_parity_verified"] = True
        record["configured_template_options_verified"] = True
        record["configured_decoding_controls_verified"] = True
        with self.db:
            self.db.execute("INSERT INTO reviews VALUES(?,?,?)", (gap["id"], canonical(record), now()))
            self.db.execute("UPDATE gaps SET state=?,updated=? WHERE id=?", (state, now(), gap["id"]))
        return state

    def failure(self, gap_id, stage, category):
        if category == "interrupted":
            state = "queued" if stage == "generate" else "review_pending"
        else:
            state = category if category in {"context_overflow", "admission_overflow"} else "failed"
        with self.db:
            self.db.execute("INSERT INTO failures(gap_id,stage,category,created) VALUES(?,?,?,?)", (gap_id, stage, category, now()))
            self.db.execute("UPDATE gaps SET state=?,updated=? WHERE id=?", (state, now(), gap_id))

    def summary(self):
        return {row[0]: row[1] for row in self.db.execute("SELECT state,COUNT(*) FROM gaps GROUP BY state")}


def _budget(provider, messages):
    config = provider.config
    count = provider.tokenizer.count(messages, config.get("chat_template_kwargs", {}))
    return check_budget(count, int(config["max_output_tokens"]), int(config["context_limit"]), int(config.get("safety_margin", 256)))


def run_stage(stage, gap, candidate, provider, admission, stop, retries):
    messages = prompt_messages(gap) if stage == "generate" else review_policy(provider.config).review_messages(gap, candidate)
    weight = _budget(provider, messages)
    for attempt in range(retries + 1):
        try:
            with admission.reserve(weight, stop):
                text, metadata = provider.generate(gap) if stage == "generate" else provider.review(gap, candidate)
                return text, {**metadata, "request_attempts": attempt + 1}
        except Exception as exc:
            if attempt >= retries or not _retryable(exc):
                raise
            if stop.wait(min(2 ** attempt, 8)):
                raise Interrupted("Stopped during provider retry backoff") from None


def run_corpus(journal, generator, reviewer, *, max_gaps, workers=8, token_budget=2097152, retries=2,
               provider_failure_limit=3, validation_failure_limit=5, stop=None, progress=None, generation_only=False):
    if type(generation_only) is not bool:
        raise ValueError("generation_only must be an explicit boolean")
    if type(max_gaps) is not int or max_gaps < 1:
        raise ValueError("max_gaps must be a positive finite bound")
    if type(workers) is not int or not 1 <= workers <= 64 or type(retries) is not int or not 0 <= retries <= 5:
        raise ValueError("workers must be 1..64; retries must be 0..5")
    if type(provider_failure_limit) is not int or provider_failure_limit < 1:
        raise ValueError("Provider failure limit must be positive")
    if type(validation_failure_limit) is not int or validation_failure_limit < 1:
        raise ValueError("Validation failure limit must be positive")
    stop = stop or threading.Event()
    admission = TokenAdmission(workers, token_budget)
    invocation = uuid.uuid4().hex
    selected = completed_stages = unavailable = 0
    invalid = {"generate": 0, "review": 0}
    pending = {}
    started = now()
    stop_reason = None
    last_update = 0.0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cot-request") as executor:
        while True:
            if not stop.is_set():
                while len(pending) < workers:
                    claimed = journal.claim(invocation, selected < max_gaps, generation_only=generation_only)
                    if claimed is None:
                        break
                    row, stage, is_new = claimed
                    selected += int(is_new)
                    try:
                        gap = journal.load_gap(row)
                        text = journal.candidate(row["id"])["text"] if stage == "review" else None
                        provider = reviewer if stage == "review" else generator
                        future = executor.submit(run_stage, stage, gap, text, provider, admission, stop, retries)
                        pending[future] = (gap, stage, text)
                    except Exception as exc:
                        journal.failure(row["id"], stage, _category(exc))
                        stop_reason = _category(exc)
                        stop.set()
                        break
            if not pending:
                break
            finished, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in finished:
                gap, stage, text = pending.pop(future)
                try:
                    response, metadata = future.result()
                    journal.receipt(gap["id"], stage, response, metadata)
                    if stage == "generate":
                        journal.generation_done(gap, response, metadata)
                        state = journal.db.execute("SELECT state FROM gaps WHERE id=?", (gap["id"],)).fetchone()[0]
                        invalid[stage] = invalid[stage] + 1 if state == "generation_invalid" else 0
                    else:
                        journal.review_done(gap, text, response, metadata)
                        invalid[stage] = 0
                    unavailable = 0
                except Exception as exc:
                    category = _category(exc)
                    journal.failure(gap["id"], stage, category)
                    unavailable = unavailable + 1 if category == "provider_unavailable" else 0
                    invalid[stage] = invalid[stage] + 1 if category == "invalid_or_failed" else 0
                    if category == "source_changed" or unavailable >= provider_failure_limit:
                        stop_reason = category
                        stop.set()
                if invalid[stage] >= validation_failure_limit:
                    stop_reason = "repeated_invalid_model_output"
                    stop.set()
                completed_stages += 1
            if progress and time.monotonic() - last_update >= 30:
                progress({"invocation": invocation, "selected_gaps": selected, "completed_stages": completed_stages,
                          "mode": "generation_only" if generation_only else "generation_and_review",
                          "active_stages": len(pending), "admitted_requests": admission.requests,
                          "reserved_tokens": admission.tokens, "states": journal.summary(), "updated": now()})
                last_update = time.monotonic()
    return {"schema": WORKER_VERSION, "invocation": invocation, "started": started, "ended": now(),
            "mode": "generation_only" if generation_only else "generation_and_review",
            "selected_gaps": selected, "completed_stages": completed_stages,
            "states": journal.summary(), "peak_requests": admission.peak_requests, "peak_reserved_tokens": admission.peak_tokens,
            "stopped": stop.is_set(), "stop_reason": stop_reason or ("signal_or_requested_stop" if stop.is_set() else None),
            "semantic_review_method": journal.identity["review_version"], "semantic_certainty_claimed": False}


def _code_identity():
    names = ("corpus_worker.py", "core.py", "provider.py", "grounding_review.py", "grounding_review_ids.py", "grounding_review_v3.py", "grounding_review_v4.py")
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--review-config", required=True)
    parser.add_argument("--inference-lock", required=True, help="Actual shared manager-coordinator inference lock; must already exist")
    parser.add_argument("--max-gaps", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=2097152)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--provider-failure-limit", type=int, default=3)
    parser.add_argument("--validation-failure-limit", type=int, default=5)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--generation-only", action="store_true", help="Stage candidates as review_pending; never call the reviewer or approve/export candidates")
    args = parser.parse_args(argv)
    if not args.confirm_run:
        parser.error("--confirm-run is required for actual teacher and review requests")
    if args.max_gaps < 1:
        parser.error("--max-gaps must be positive")
    paths = [Path(p).resolve() for p in (args.source, args.journal, args.config, args.review_config, args.inference_lock)]
    if paths[0] == paths[1] or paths[1] in paths[2:]:
        parser.error("Journal must not overwrite source, configs, or the shared lock")
    journal_path = Path(args.journal)
    with open(args.inference_lock, "r+") as shared_lock, open(str(journal_path) + ".worker.lock", "a+") as run_lock:
        try:
            fcntl.flock(shared_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Shared inference reservation or journal is occupied; no request sent")
        config = json.loads(Path(args.config).read_text())
        review_config = json.loads(Path(args.review_config).read_text())
        generator = OpenAICompatibleProvider(config)
        reviewer = PrefixReviewProvider(review_config)
        # Config files contain environment-variable names, never credential values.
        identity = {"generator_config": config, "reviewer_config": review_config,
                    "generator_tokenizer": generator.tokenizer.identity, "reviewer_tokenizer": reviewer.tokenizer.identity,
                    "implementation_sha256": _code_identity()}
        journal = Journal(args.journal, args.source, args.source_sha256, identity)
        stop = threading.Event()
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in previous:
            signal.signal(sig, lambda *_: stop.set())
        try:
            result = run_corpus(journal, generator, reviewer, max_gaps=args.max_gaps, workers=args.workers, token_budget=args.token_budget,
                                retries=args.retries, provider_failure_limit=args.provider_failure_limit,
                                validation_failure_limit=args.validation_failure_limit, stop=stop,
                                generation_only=args.generation_only,
                                progress=lambda value: print(canonical(value), flush=True))
            print(canonical(result), flush=True)
            if result["stop_reason"] and result["stop_reason"] != "signal_or_requested_stop":
                raise SystemExit(1)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            journal.close()


if __name__ == "__main__":
    main()
