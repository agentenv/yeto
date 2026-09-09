"""Finite review-only comparison with an enforced-thinking server configuration.

Copies exact saved candidates/generation receipts into a fresh identity. Does
not regenerate, export, or infer semantic correctness from JSON/schema validity.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import signal
import threading

from .core import canonical
from .corpus_source import file_sha256
from .corpus_worker import (Journal, PrefixReviewProvider, _budget, _code_identity,
                            review_policy, run_corpus)
from .grounding_review_v4 import REVIEW_VERSION
from .provider import ContextOverflow
from .review_comparison import NoGeneration, import_candidates, snapshot_candidates
from .review_recovery import restrict_retry

COMPARISON_VERSION = "cot.bounded-thinking-review-comparison/v1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-journal", required=True)
    parser.add_argument("--original-count", required=True, type=int)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--review-config", required=True)
    parser.add_argument("--ordinals", required=True, help="Explicit finite comma-separated ordinal list, or all")
    parser.add_argument("--inference-lock", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=2097152)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.original_count <= 256:
        parser.error("Original candidate count must be finite and within 1..256")
    if not args.preflight_only and not args.confirm_run:
        parser.error("Choose CPU --preflight-only or explicit --confirm-run")
    paths = [Path(p).resolve() for p in (args.original_journal, args.journal, args.review_config, args.inference_lock)]
    if len(set(paths)) != len(paths):
        parser.error("Original journal, comparison journal, config and shared lock must differ")
    snapshot = snapshot_candidates(args.original_journal, expected_count=args.original_count)
    original = snapshot["original_identity"]
    config = json.loads(Path(args.review_config).read_text())
    if config.get("review_version") != REVIEW_VERSION or not {"custom_params", "response_format"} <= config.keys():
        parser.error("Bounded comparison requires v4 reviewer and explicit thinking budget/JSON schema")
    if config.get("model") != original["generator_config"]["model"]:
        parser.error("Comparison must retain the original generator model")
    reviewer = PrefixReviewProvider(config)
    ordinals = list(range(1, args.original_count + 1)) if args.ordinals == "all" else [int(s) for s in args.ordinals.split(",")]
    identity = {"comparison_version": COMPARISON_VERSION,
                "original_journal": snapshot["original_journal"],
                "original_snapshot_sha256": snapshot["snapshot_sha256"],
                "selected_ordinals": ordinals, "reviewer_config": config,
                "reviewer_tokenizer": reviewer.tokenizer.identity,
                "implementation_sha256": {**_code_identity(),
                    "review_bounded.py": file_sha256(__file__),
                    "review_comparison.py": file_sha256(Path(__file__).with_name("review_comparison.py")),
                    "review_recovery.py": file_sha256(Path(__file__).with_name("review_recovery.py"))}}
    with open(args.journal + ".worker.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = Journal(args.journal, original["source_path"], original["source_sha256"], identity)
        try:
            import_candidates(journal, snapshot)
            restrict_retry(journal, ordinals)
            if args.preflight_only:
                reservations, overflow = [], []
                for row in journal.db.execute("SELECT * FROM gaps WHERE state='review_pending' ORDER BY ordinal"):
                    gap = journal.load_gap(dict(row));candidate = journal.candidate(gap["id"])["text"]
                    try:
                        reservations.append(_budget(reviewer, review_policy(config).review_messages(gap, candidate)))
                    except ContextOverflow:
                        overflow.append(row["ordinal"])
                result = {"schema": COMPARISON_VERSION, "mode": "cpu_preflight", "selected_ordinals": ordinals,
                          "fitting_requests": len(reservations), "context_overflow_ordinals": overflow,
                          "maximum_reservation": max(reservations, default=0), "inference_requests": 0}
            else:
                with open(args.inference_lock, "r+") as shared:
                    fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    stop = threading.Event()
                    for sig in (signal.SIGINT, signal.SIGTERM):
                        signal.signal(sig, lambda *_: stop.set())
                    result = run_corpus(journal, NoGeneration(), reviewer, max_gaps=len(ordinals),
                        workers=args.workers, token_budget=args.token_budget, retries=1, stop=stop,
                        progress=lambda value: print(canonical(value), flush=True))
                result.update(mode="actual_bounded_review", selected_ordinals=ordinals)
            result.update(automatic_bulk_approval=False, semantic_certainty_claimed=False)
            print(canonical(result), flush=True)
        finally:
            journal.close()


if __name__ == "__main__":main()
