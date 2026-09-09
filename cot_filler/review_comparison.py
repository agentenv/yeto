"""Review a bounded frozen candidate set without issuing any generation call.

Original source/candidate/generation receipts are copied and hash bound; prior
review outcomes are never imported as approvals. Actual new evaluations use a
fresh journal and the same shared inference reservation as the corpus worker.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import signal
import sqlite3
import threading

from .core import BLOCKING_FLAGS, PROMPT_VERSION, canonical, digest, validate_candidate
from .corpus_source import file_sha256
from .corpus_worker import (Journal, PrefixReviewProvider, _budget, _code_identity,
                            run_corpus)
from .grounding_review_v3 import REVIEW_VERSION, review_messages

COMPARISON_VERSION = "cot.persisted-candidate-review-comparison/v2-configured-thinking"
# Independently inspected concrete unsupported/hindsight claims in this pilot.
KNOWN_NEGATIVES = (7, 11, 13, 17, 23, 25, 41, 47)
KNOWN_AMBIGUOUS = 37
EXPECTED_SOURCE = "fb667c4732a40a2f1fdf08107023fbea77b0901930177351242a344ea9158112"


class NoGeneration:
    def generate(self, *args, **kwargs):
        raise RuntimeError("Review comparison prohibits generation")


class ComparisonJournal(Journal):
    """The shared Journal enforces the configured review receipt invariants."""


def snapshot_candidates(path, *, expected_count=48):
    """Read a stable completed journal transaction and keep exact prior receipts."""
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute("BEGIN")
        identity = json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
        rows = []
        for row in db.execute("SELECT g.*,s.row_sha256,s.source_digest,s.trace_id FROM gaps g JOIN sources s ON s.id=g.source_id ORDER BY g.ordinal"):
            if row["state"] not in {"approved", "rejected", "failed", "uncertain", "context_overflow"}:
                raise ValueError("Original comparison run must be fully terminal")
            candidate = db.execute("SELECT * FROM candidates WHERE gap_id=?", (row["id"],)).fetchone()
            receipts = db.execute("SELECT stage,text,data,created FROM responses WHERE gap_id=? AND stage='generate' ORDER BY id", (row["id"],)).fetchall()
            if candidate is None or len(receipts) != 1:
                raise ValueError("Every comparison gap requires one persisted real generation receipt")
            candidate, receipt = dict(candidate), dict(receipts[0])
            metadata = json.loads(receipt["data"])
            if (receipt["text"] != candidate["text"] or metadata.get("finish_reason") != "stop"
                    or metadata.get("generator", {}).get("provider") != "openai-compatible"
                    or metadata.get("generator", {}).get("prompt_version") != PROMPT_VERSION
                    or BLOCKING_FLAGS.intersection(validate_candidate(candidate["text"]))) :
                raise ValueError("Original generation identity or structural validation failed")
            rows.append({"gap": dict(row), "candidate": candidate, "generation_receipt": receipt})
        if len(rows) != expected_count:
            raise ValueError("Comparison must include the exact expected finite candidate count")
        return {"original_journal": str(Path(path).resolve()), "original_identity": identity,
                "rows": rows, "snapshot_sha256": digest({"identity": identity, "rows": rows})}
    finally:
        db.close()


def import_candidates(journal, snapshot):
    """Copy exact generation records once; preserve completed comparison reviews."""
    original = snapshot["original_identity"]
    if original["source_path"] != str(journal.source) or original["source_sha256"] != file_sha256(journal.source):
        raise ValueError("Comparison source differs from the original generator source")
    if journal.db.execute("SELECT count(*) FROM gaps").fetchone()[0] != len(snapshot["rows"]):
        raise ValueError("Comparison source contains an unexpected gap set")
    for saved in snapshot["rows"]:
        old, candidate, receipt = saved["gap"], saved["candidate"], saved["generation_receipt"]
        row = journal.db.execute("SELECT * FROM gaps WHERE ordinal=?", (old["ordinal"],)).fetchone()
        if row is None or any(row[key] != old[key] for key in ("id", "event_id", "event_index")):
            raise ValueError("Comparison gap order or original boundary changed")
        gap = journal.load_gap(dict(row))
        metadata = json.loads(candidate["data"])
        if (gap["source_digest"] != old["source_digest"] or gap["prefix_hash"] != candidate["prefix_hash"]
                or gap["lookahead_hash"] != candidate["lookahead_hash"] or gap["prompt_hash"] != candidate["prompt_hash"]
                or metadata.get("candidate_hash") != digest(candidate["text"])
                or json.loads(receipt["data"]).get("prompt_hash") != gap["prompt_hash"]):
            raise ValueError("Saved candidate/generator source provenance changed")
        present = journal.candidate(gap["id"])
        if present is not None:
            if dict(present) != candidate:
                raise ValueError("Previously imported comparison candidate changed")
            continue
        with journal.db:
            journal.db.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?,?)", tuple(candidate[key] for key in
                ("gap_id", "text", "data", "prefix_hash", "lookahead_hash", "prompt_hash", "created")))
            journal.db.execute("INSERT INTO responses(gap_id,stage,text,data,created) VALUES(?,?,?,?,?)",
                (gap["id"], receipt["stage"], receipt["text"], receipt["data"], receipt["created"]))
            journal.db.execute("UPDATE gaps SET state='review_pending',invocation=NULL WHERE id=?", (gap["id"],))
    if journal.db.execute("SELECT count(*) FROM gaps WHERE state='queued'").fetchone()[0]:
        raise ValueError("No generation stage may remain in a review-only comparison")


def comparison_gate(journal):
    states = dict(journal.db.execute("SELECT ordinal,state FROM gaps"))
    required = (KNOWN_AMBIGUOUS,) + KNOWN_NEGATIVES
    detected = {str(i): states.get(i) in {"rejected", "uncertain"} for i in required}
    return {"known_case_states": {str(i): states.get(i) for i in required},
            "all_known_bad_cases_detected": all(detected.values()),
            "known_case_detection": detected,
            "automatic_bulk_approval": False, "semantic_certainty_claimed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-journal", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--review-config", required=True)
    parser.add_argument("--inference-lock", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=2097152)
    parser.add_argument("--review-mode", choices=("thinking", "nonthinking"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.preflight_only and not args.confirm_run:
        parser.error("Choose CPU --preflight-only or explicit --confirm-run")
    paths = [Path(p).resolve() for p in (args.original_journal, args.journal, args.review_config, args.inference_lock)]
    if len(set(paths)) != len(paths):
        parser.error("Original journal, new journal, config and lock must be distinct")
    snapshot = snapshot_candidates(args.original_journal)
    original = snapshot["original_identity"]
    if original["source_sha256"] != EXPECTED_SOURCE:
        parser.error("Known-case gates apply only to the exact v5 diverse48 source")
    config = json.loads(Path(args.review_config).read_text())
    thinking = args.review_mode == "thinking"
    if config.get("chat_template_kwargs") != {"thinking": thinking} or config.get("max_output_tokens") != (16384 if thinking else 8192):
        parser.error("Comparison mode requires exact template options and the qualified finite output allowance")
    if config.get("model") != original["generator_config"]["model"]:
        parser.error("Comparison must use the same actual teacher model")
    reviewer = PrefixReviewProvider(config)
    identity = {"comparison_version": COMPARISON_VERSION,
                "original_snapshot_sha256": snapshot["snapshot_sha256"],
                "original_journal": snapshot["original_journal"],
                "reviewer_config": config, "reviewer_tokenizer": reviewer.tokenizer.identity,
                "implementation_sha256": {**_code_identity(), "review_comparison.py": file_sha256(__file__)}}
    # CPU preflight does no HTTP and creates only the new comparison journal.
    with open(str(paths[1]) + ".worker.lock", "a+") as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = ComparisonJournal(args.journal, original["source_path"], original["source_sha256"], identity)
        try:
            import_candidates(journal, snapshot)
            reservations = []
            for row in journal.db.execute("SELECT * FROM gaps ORDER BY ordinal").fetchall():
                gap = journal.load_gap(dict(row));candidate = journal.candidate(gap["id"])["text"]
                reservations.append(_budget(reviewer, review_messages(gap, candidate)))
            preflight = {"comparison_version": COMPARISON_VERSION, "source_sha256": original["source_sha256"],
                         "snapshot_sha256": snapshot["snapshot_sha256"], "count": len(reservations),
                         "reservation_min": min(reservations), "reservation_max": max(reservations),
                         "review_version": REVIEW_VERSION, "inference_started": False}
            print(canonical(preflight), flush=True)
            if args.preflight_only:
                return
            with open(args.inference_lock, "r+") as shared:
                fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
                stop = threading.Event()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(sig, lambda *_: stop.set())
                result = run_corpus(journal, NoGeneration(), reviewer, max_gaps=48,
                    workers=args.workers, token_budget=args.token_budget, retries=1,
                    validation_failure_limit=8, stop=stop,
                    progress=lambda value: print(canonical(value), flush=True))
                result["comparison_gate"] = comparison_gate(journal)
                print(canonical(result), flush=True)
        finally:
            journal.close()


if __name__ == "__main__":
    main()
