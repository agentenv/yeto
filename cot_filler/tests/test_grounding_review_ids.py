import copy
import json
from pathlib import Path
import unittest

from cot_filler.core import canonical, digest, gaps_for
from cot_filler.grounding_review_ids import CHECKS, REVIEW_VERSION, SNIPPET_CHARS, review_contract, review_messages, validate_review


class BoundReviewTests(unittest.TestCase):
    def setUp(self):
        trace = json.loads((Path(__file__).resolve().parents[1] / "examples/demo.json").read_text())
        trace["events"][2]["content"] += " FUTURE_SECRET"
        self.gap = next(gaps_for(trace))
        self.candidate = "Inspect the function.\n\nThen decide whether it needs a repair."
        contract, evidence = review_contract(self.gap, self.candidate)
        self.evidence_id = next(k for k, v in evidence.items() if "Inspect the function before editing." in v["quote"])
        self.result = {"schema": REVIEW_VERSION, "decision": "pass", "checks": dict.fromkeys(CHECKS, True),
                       "statements": [{"id": x["id"], "kind": "plan", "assessment": "supported", "evidence_ids": [self.evidence_id]}
                                      for x in contract["candidate_units"]],
                       "note": "Deterministic fixture only; no real reviewer has run."}
        self.provenance = {"model": "fixture-unit-test", "finish_reason": "stop"}

    def validate(self, result=None):
        return validate_review(self.gap, self.candidate, result or self.result, reviewer=self.provenance)

    def test_bound_ids_resolve_exact_text_and_hash_without_transcription(self):
        record = self.validate()
        self.assertEqual(record["schema"], REVIEW_VERSION)
        self.assertEqual(record["candidate_hash"], digest(self.candidate))
        self.assertEqual(record["review_prompt_hash"], digest(review_messages(self.gap, self.candidate)))
        self.assertEqual(record["statements"][0]["text"], "Inspect the function.")
        self.assertEqual(record["statements"][1]["text"], "Then decide whether it needs a repair.")
        self.assertEqual(record["statements"][0]["evidence"][0]["quote"], self.gap["prefix"][0]["content"])
        self.assertFalse(record["automatic_approval"])
        self.assertFalse(record["future_events_supplied_to_reviewer"])

    def test_prefix_only_contract_preserves_all_long_string_bytes(self):
        original = "prefix\n" + "x" * (SNIPPET_CHARS * 2 + 7) + "\nend"
        self.gap["prefix"][0]["content"] = original
        self.gap["prefix_hash"] = digest(self.gap["prefix"])
        self.gap["marker_provenance"] = {"secret": "FUTURE_METADATA"}
        contract, evidence = review_contract(self.gap, self.candidate)
        snippets = contract["prefix_events_with_evidence_ids"][0]["content"]["snippets"]
        self.assertEqual("".join(x["text"] for x in snippets), original)
        self.assertTrue(all(evidence[x["id"]]["quote"] == x["text"] for x in snippets))
        prompt = canonical(review_messages(self.gap, self.candidate))
        for forbidden in ("FUTURE_SECRET", "FUTURE_METADATA", "read-1"):
            self.assertNotIn(forbidden, prompt)

    def test_missing_reordered_duplicate_candidate_or_unknown_evidence_ids_fail(self):
        cases = []
        bad = copy.deepcopy(self.result); bad["statements"].pop(); cases.append(bad)
        bad = copy.deepcopy(self.result); bad["statements"].reverse(); cases.append(bad)
        bad = copy.deepcopy(self.result); bad["statements"][1]["id"] = "c0"; cases.append(bad)
        bad = copy.deepcopy(self.result); bad["statements"][0]["evidence_ids"] = ["future-p0"]; cases.append(bad)
        bad = copy.deepcopy(self.result); bad["statements"][0]["evidence_ids"] *= 2; cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):
                self.validate(bad)

    def test_failed_check_unsupported_missing_evidence_and_incomplete_cannot_pass(self):
        cases = []
        bad = copy.deepcopy(self.result); bad["checks"]["no_future_observation_claims"] = False; cases.append(bad)
        bad = copy.deepcopy(self.result); bad["statements"][0]["assessment"] = "uncertain"; cases.append(bad)
        bad = copy.deepcopy(self.result); bad["statements"][0]["evidence_ids"] = []; cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):
                self.validate(bad)
        self.provenance["finish_reason"] = "length"
        with self.assertRaises(ValueError):
            self.validate()

    def test_uncertain_model_decision_remains_uncertain(self):
        self.result["decision"] = "uncertain"
        self.result["checks"]["all_factual_claims_supported_by_prefix"] = False
        self.result["statements"][0]["assessment"] = "uncertain"
        self.result["statements"][0]["evidence_ids"] = []
        self.assertEqual(self.validate()["decision"], "uncertain")


if __name__ == "__main__":
    unittest.main()
