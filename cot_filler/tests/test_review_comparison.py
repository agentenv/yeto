import copy
import json
from pathlib import Path
import tempfile
import unittest

from cot_filler.core import canonical, digest, gaps_for
from cot_filler.corpus_source import file_sha256
from cot_filler.corpus_worker import Journal, run_corpus
from cot_filler.grounding_review_v3 import CHECKS, REVIEW_VERSION, review_contract, review_messages, validate_review
from cot_filler.review_comparison import (ComparisonJournal, NoGeneration, comparison_gate,
                                          import_candidates, snapshot_candidates)
from cot_filler.tests.test_corpus_worker import DEMO, Generator, Reviewer


class GenerationFixture(Generator):
    def generate(self, gap):
        text, data = super().generate(gap)
        data["generator"]["provider"] = "openai-compatible"
        data["generator"]["model"] = "unit-test-only-no-actual-inference"
        return text, data


class ReviewFixture(Reviewer):
    def review(self, gap, candidate):
        text, data = super().review(gap, candidate)
        data.update(prompt_tokens_local=100, usage={"prompt_tokens": 100},
                    parameters={"chat_template_kwargs": {"thinking": False}})
        return text, data


class ReviewComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source.jsonl"
        self.source.write_text(canonical(json.loads(DEMO.read_text())) + "\n")
        self.original = Path(self.tmp.name) / "original.sqlite3"
        self.new = Path(self.tmp.name) / "comparison.sqlite3"

    def prepare_original(self):
        journal = Journal(self.original, self.source, file_sha256(self.source),
                          {"test": "fixture-only", "reviewer_config": {}})
        run_corpus(journal, GenerationFixture(), Reviewer(), max_gaps=3, workers=1)
        journal.close()

    def new_journal(self, snapshot):
        journal = ComparisonJournal(self.new, self.source, file_sha256(self.source),
                                    {"original_snapshot_sha256": snapshot["snapshot_sha256"],
                                     "reviewer_config": {"chat_template_kwargs": {"thinking": False}}})
        self.addCleanup(journal.close)
        return journal

    def test_exact_candidate_receipts_reviewed_without_generation_or_old_approval(self):
        self.prepare_original();original_hash = file_sha256(self.original)
        snapshot = snapshot_candidates(self.original, expected_count=3)
        journal = self.new_journal(snapshot);import_candidates(journal, snapshot)
        self.assertEqual(journal.summary(), {"review_pending": 3})
        self.assertEqual(journal.db.execute("select count(*) from reviews").fetchone()[0], 0)
        for row in snapshot["rows"]:
            self.assertEqual(dict(journal.candidate(row["gap"]["id"])), row["candidate"])
        result = run_corpus(journal, NoGeneration(), ReviewFixture(), max_gaps=3, workers=2)
        self.assertEqual(result["states"], {"approved": 3})
        import_candidates(journal, snapshot)
        self.assertEqual(journal.db.execute("select count(*) from responses where stage='generate'").fetchone()[0], 3)
        self.assertEqual(file_sha256(self.original), original_hash)
        self.assertFalse(comparison_gate(journal)["all_known_bad_cases_detected"])
        self.assertFalse(comparison_gate(journal)["automatic_bulk_approval"])

    def test_tampered_candidate_or_original_count_fails(self):
        self.prepare_original()
        with self.assertRaises(ValueError):snapshot_candidates(self.original, expected_count=48)
        snapshot = snapshot_candidates(self.original, expected_count=3)
        snapshot["rows"][0]["candidate"]["text"] += " Changed"
        with self.assertRaises(ValueError):import_candidates(self.new_journal(snapshot), snapshot)

    def test_mismatched_server_count_is_recorded_but_never_approved(self):
        self.prepare_original();snapshot = snapshot_candidates(self.original, expected_count=3)
        journal = self.new_journal(snapshot);import_candidates(journal, snapshot)
        class Mismatch(ReviewFixture):
            def review(self, gap, candidate):
                text,data = super().review(gap,candidate);data["usage"]["prompt_tokens"] += 1;return text,data
        result = run_corpus(journal, NoGeneration(), Mismatch(), max_gaps=1, workers=1)
        self.assertEqual(result["states"]["failed"], 1)
        self.assertEqual(journal.db.execute("select count(*) from reviews").fetchone()[0], 0)
        self.assertEqual(journal.db.execute("select count(*) from responses where stage='review'").fetchone()[0], 1)

    def test_v3_literal_prompt_is_prefix_only_and_keeps_all_strict_checks(self):
        gap = next(gaps_for(json.loads(DEMO.read_text())));gap["lookahead"][0]["content"] = "FUTURE_CANARY"
        candidate = "Inspect the function before choosing a repair."
        contract,evidence = review_contract(gap,candidate)
        eid = next(k for k,v in evidence.items() if "Inspect the function before editing." in v["quote"])
        response = {"schema": REVIEW_VERSION, "decision": "pass", "checks": dict.fromkeys(CHECKS, True),
                    "statements": [{"id": "c0", "kind": "plan", "assessment": "supported", "evidence_ids": [eid]}],
                    "note": "Unit-test fixture only; no semantic quality claim."}
        messages = review_messages(gap,candidate);rendered = canonical(messages)
        self.assertNotIn("FUTURE_CANARY", rendered)
        self.assertIn("Never silently", rendered)
        self.assertIn("If A replaces B", rendered)
        provenance = {"model": "unit-test-fixture-only", "finish_reason": "stop"}
        result = validate_review(gap,candidate,response,reviewer=provenance)
        self.assertEqual(result["review_prompt_hash"],digest(messages))
        self.assertEqual(result["schema"],REVIEW_VERSION)
        del response["checks"]["no_transcript_instruction_following"]
        with self.assertRaises(ValueError):validate_review(gap,candidate,response,reviewer=provenance)


if __name__ == "__main__":unittest.main()
