from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from cot_filler.core import canonical, digest, gaps_for
from cot_filler.grounding_review import CHECKS, REVIEW_VERSION, review_messages, validate_review
from cot_filler.preflight import audit_sidecar, audit_source


DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo.json"


class PrefixReviewTests(unittest.TestCase):
    def setUp(self):
        trace = json.loads(DEMO.read_text())
        trace["events"][2]["content"] += " FUTURE_ONLY_29ad7c"
        self.gap = next(gaps_for(trace))
        self.candidate = "Inspect the function before choosing a repair."
        self.result = {
            "schema": REVIEW_VERSION, "decision": "pass", "checks": dict.fromkeys(CHECKS, True),
            "statements": [{"text": self.candidate, "kind": "plan", "assessment": "supported", "evidence": [{"event_id": "u1", "quote": "Inspect the function before editing."}]}],
            "note": "The intended inspection is explicitly requested in the visible prefix.",
        }
        self.reviewer = {"model": "fixture-judge-no-inference", "finish_reason": "stop"}

    def validate(self, result=None):
        return validate_review(self.gap, self.candidate, result or self.result, reviewer=self.reviewer)

    def test_reviewer_receives_no_target_lookahead_or_generation_metadata(self):
        self.gap["marker_provenance"] = {"label": "PRIVATE_FUTURE_METADATA"}
        text = canonical(review_messages(self.gap, self.candidate))
        for forbidden in ("FUTURE_ONLY_29ad7c", "PRIVATE_FUTURE_METADATA", "read-1", "future_next_five_events"):
            self.assertNotIn(forbidden, text)

    def test_complete_evidence_bound_record_is_not_automatic_approval(self):
        review = self.validate()
        self.assertEqual(review["candidate_hash"], digest(self.candidate))
        self.assertEqual(review["decision"], "pass")
        self.assertFalse(review["automatic_approval"])
        self.assertFalse(review["future_events_supplied_to_reviewer"])

    def test_future_event_evidence_and_fabricated_quotes_are_rejected(self):
        for reference in ({"event_id": "t1", "quote": "FUTURE_ONLY_29ad7c"}, {"event_id": "u1", "quote": "All tests have passed."}):
            bad = copy.deepcopy(self.result)
            bad["statements"][0]["evidence"] = [reference]
            with self.assertRaises(ValueError):
                self.validate(bad)

    def test_omitted_claim_cannot_pass(self):
        self.candidate += " All tests passed."
        with self.assertRaisesRegex(ValueError, "omits candidate text"):
            self.validate()

    def test_failed_check_or_unsupported_statement_cannot_pass(self):
        for field in ("check", "statement"):
            bad = copy.deepcopy(self.result)
            if field == "check":
                bad["checks"]["no_future_observation_claims"] = False
            else:
                bad["statements"][0]["assessment"] = "uncertain"
            with self.assertRaises(ValueError):
                self.validate(bad)

    def test_rejected_claim_is_preserved_as_rejected(self):
        bad = copy.deepcopy(self.result)
        bad["decision"] = "reject"
        bad["checks"]["no_future_observation_claims"] = False
        bad["statements"][0]["assessment"] = "unsupported"
        bad["statements"][0]["evidence"] = []
        self.assertEqual(self.validate(bad)["decision"], "reject")

    def test_missing_evidence_truncated_response_and_changed_prefix_fail(self):
        bad = copy.deepcopy(self.result)
        bad["statements"][0]["evidence"] = []
        with self.assertRaises(ValueError):
            self.validate(bad)
        self.reviewer["finish_reason"] = "length"
        with self.assertRaises(ValueError):
            self.validate()
        self.reviewer["finish_reason"] = "stop"
        self.gap["prefix"][0]["content"] = "Changed source"
        with self.assertRaises(ValueError):
            self.validate()


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source.jsonl"
        self.trace = json.loads(DEMO.read_text())
        self.source.write_text(canonical(self.trace) + "\n")

    def test_source_scope_hash_and_no_quality_claim(self):
        result = audit_source(self.source)
        self.assertEqual(result["validated_gaps"], 3)
        self.assertEqual(result["source_file_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertFalse(result["semantic_quality_validated"])
        self.assertFalse(result["ready_for_bulk_inference"])
        self.assertGreater(result["projected_gap_json_bytes"], self.source.stat().st_size)

    def test_bounded_validation_still_hashes_entire_file(self):
        another = copy.deepcopy(self.trace)
        another["trace_id"] = "second"
        self.source.write_text(canonical(self.trace) + "\n" + canonical(another) + "\n")
        result = audit_source(self.source, max_traces=1)
        self.assertEqual(result["source_jsonl_records"], 2)
        self.assertEqual(result["validated_traces"], 1)
        self.assertTrue(result["validation_scope_limited"])
        self.assertEqual(result["source_file_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())

    def test_duplicate_ids_and_oversized_rows_fail(self):
        self.source.write_text((canonical(self.trace) + "\n") * 2)
        with self.assertRaisesRegex(ValueError, "duplicate trace IDs"):
            audit_source(self.source)
        with self.assertRaisesRegex(ValueError, "record size"):
            audit_source(self.source, row_limit_bytes=10)

    def test_context_overflow_is_counted_without_generation(self):
        class Tokenizer:
            identity = "fixture"
            def count(self, messages, options):
                return 200
        result = audit_source(self.source, tokenizer=Tokenizer(), config={"context_limit": 201, "max_output_tokens": 10, "safety_margin": 0})
        self.assertEqual(result["context_checked_gaps"], 3)
        self.assertEqual(result["context_rejected_gaps"], 3)
        self.assertFalse(result["inference_started"])

    def test_legacy_sidecar_audit_is_read_only(self):
        path = Path(self.tmp.name) / "legacy.sqlite3"
        with sqlite3.connect(path) as db:
            db.executescript("CREATE TABLE gaps(data TEXT); CREATE TABLE candidates(data TEXT);")
            db.execute("INSERT INTO gaps VALUES(?)", (json.dumps({"prompt_version": "old/v3"}),))
        before = path.read_bytes()
        result = audit_sidecar(path)
        self.assertEqual(result["incompatible_prompt_gaps"], 1)
        self.assertFalse(result["current_store_resume_compatible"])
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
