"""Separate, immutable revalidation or bounded retries of persisted reviews.

Offline revalidation does not create model judgments: it replays the exact saved
text and receipt through the current parser/strict validators. Missing or invalid
judgments remain failed. Retries create a fresh journal with an explicit subset,
new configured output allowance, and new actual provider receipts.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import signal
import sqlite3
import threading

from .core import canonical, digest
from .corpus_source import file_sha256
from .corpus_worker import Journal, PrefixReviewProvider, _category, _code_identity, run_corpus
from .review_comparison import NoGeneration, import_candidates, snapshot_candidates

RECOVERY_VERSION = "cot.persisted-review-recovery/v1"


def saved_reviews(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return [{"gap_id": g, "text": t, "data": d, "created": c}
                for g,t,d,c in db.execute("SELECT gap_id,text,data,created FROM responses WHERE stage='review' ORDER BY id")]
    finally:
        db.close()


def revalidate(journal, reviews):
    by_gap = {}
    for review in reviews:
        if review["gap_id"] in by_gap:
            raise ValueError("Multiple original review responses require explicit selection")
        by_gap[review["gap_id"]] = review
    for row in journal.db.execute("SELECT * FROM gaps ORDER BY ordinal").fetchall():
        gap = journal.load_gap(dict(row));candidate = journal.candidate(gap["id"])["text"]
        review = by_gap.get(gap["id"])
        if review is None:
            journal.failure(gap["id"], "review", "missing_actual_review")
            continue
        with journal.db:
            journal.db.execute("INSERT INTO responses(gap_id,stage,text,data,created) VALUES(?,?,?,?,?)",
                (gap["id"], "review", review["text"], review["data"], review["created"]))
        try:
            journal.review_done(gap, candidate, review["text"], json.loads(review["data"]))
        except Exception as exc:
            journal.failure(gap["id"], "review", _category(exc))
    return {"schema": RECOVERY_VERSION, "mode": "offline_exact_receipt_revalidation",
            "states": journal.summary(), "inference_requests": 0,
            "automatic_bulk_approval": False, "semantic_certainty_claimed": False}


def restrict_retry(journal, ordinals):
    if not ordinals or len(ordinals) != len(set(ordinals)) or any(type(x) is not int or x < 1 for x in ordinals):
        raise ValueError("Retry requires an explicit nonempty unique ordinal list")
    existing = {row[0] for row in journal.db.execute("SELECT ordinal FROM gaps")}
    if not set(ordinals) <= existing:
        raise ValueError("Requested retry ordinal is absent from the source")
    with journal.db:
        for ordinal in existing - set(ordinals):
            journal.db.execute("UPDATE gaps SET state='recovery_not_selected' WHERE ordinal=?", (ordinal,))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("revalidate", "retry"))
    parser.add_argument("--original-journal", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--review-config")
    parser.add_argument("--ordinals", help="Explicit comma-separated finite retry set")
    parser.add_argument("--inference-lock")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    if Path(args.journal).exists():
        parser.error("Recovery uses a fresh journal; preserve previous artifacts")
    snapshot = snapshot_candidates(args.original_journal)
    original = snapshot["original_identity"]
    reviews = saved_reviews(args.original_journal)
    if args.mode == "retry" and (not args.confirm_run or not args.inference_lock or not args.ordinals or not args.review_config):
        parser.error("Actual retry requires config, --confirm-run, --inference-lock and finite --ordinals")
    config = original["reviewer_config"] if args.mode == "revalidate" else json.loads(Path(args.review_config).read_text())
    if config.get("model") != original["reviewer_config"]["model"] or config.get("chat_template_kwargs") != {"thinking": True}:
        parser.error("Recovery keeps the same model and thinking-enabled prefix reviewer")
    ordinals = [] if args.mode == "revalidate" else [int(x) for x in args.ordinals.split(",")]
    identity = {"recovery_version": RECOVERY_VERSION, "recovery_mode": args.mode,
                "original_journal": str(Path(args.original_journal).resolve()),
                "original_snapshot_sha256": snapshot["snapshot_sha256"], "original_reviews_sha256": digest(reviews),
                "selected_ordinals": ordinals, "reviewer_config": config,
                "implementation_sha256": {**_code_identity(), "review_recovery.py": file_sha256(__file__)}}
    with open(args.journal + ".worker.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = Journal(args.journal, original["source_path"], original["source_sha256"], identity)
        try:
            import_candidates(journal, snapshot)
            if args.mode == "revalidate":
                result = revalidate(journal, reviews)
            else:
                restrict_retry(journal, ordinals)
                reviewer = PrefixReviewProvider(config)
                with open(args.inference_lock, "r+") as shared:
                    fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    stop = threading.Event()
                    for sig in (signal.SIGINT, signal.SIGTERM):signal.signal(sig, lambda *_: stop.set())
                    result = run_corpus(journal, NoGeneration(), reviewer, max_gaps=len(ordinals),
                        workers=min(16, len(ordinals)), token_budget=2097152, retries=1, stop=stop,
                        progress=lambda value: print(canonical(value), flush=True))
                result.update(recovery_mode="actual_bounded_retry", selected_ordinals=ordinals,
                              automatic_bulk_approval=False)
            print(canonical(result), flush=True)
        finally:
            journal.close()


if __name__ == "__main__":main()
