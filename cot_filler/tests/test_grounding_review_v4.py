import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import unittest

from cot_filler.core import canonical, digest, gaps_for
from cot_filler.grounding_review_v4 import (CHECKS, REVIEW_VERSION, SYSTEM,
    response_format, review_contract, review_messages, validate_review)


class EmbeddedFactReviewTests(unittest.TestCase):
    def setUp(self):
        trace = json.loads((Path(__file__).resolve().parents[1] / "examples/demo.json").read_text())
        trace["events"][2]["content"] += " FUTURE_SENTINEL"
        self.gap = next(gaps_for(trace))
        self.candidate = "I will inspect the function before deciding whether to edit it."
        contract, evidence = review_contract(self.gap, self.candidate)
        evidence_id = next(k for k, v in evidence.items() if "Inspect the function before editing." in v["quote"])
        self.result = {"schema": REVIEW_VERSION, "decision": "pass", "checks": dict.fromkeys(CHECKS, True),
                       "statements": [{"id": unit["id"], "kind": "plan", "assessment": "supported", "evidence_ids": [evidence_id]}
                                      for unit in contract["candidate_units"]],
                       "note": "Test fixture only, not an actual semantic judgment."}
        self.reviewer = {"model": "fixture-test", "finish_reason": "stop"}

    def test_new_prompt_remains_prefix_only(self):
        prompt = canonical(review_messages(self.gap, self.candidate))
        self.assertNotIn("FUTURE_SENTINEL", prompt)
        self.assertIn("expected or predicted failure", SYSTEM)
        self.assertIn("embedded factual claims", SYSTEM)
        record = validate_review(self.gap, self.candidate, self.result, reviewer=self.reviewer)
        self.assertEqual(record["review_prompt_hash"], digest(review_messages(self.gap, self.candidate)))
        self.assertFalse(record["future_events_supplied_to_reviewer"])

    def test_existing_strict_guards_and_whole_fence_preserved(self):
        wrapped = "```json\n" + json.dumps(self.result) + "\n```"
        record = validate_review(self.gap, self.candidate, wrapped, reviewer=self.reviewer)
        self.assertEqual(record["review_response_format"], "whole_json_fence")
        bad = copy.deepcopy(self.result)
        bad["statements"][0]["evidence_ids"] = ["not-an-evidence-id"]
        with self.assertRaises(ValueError):
            validate_review(self.gap, self.candidate, bad, reviewer=self.reviewer)
        with self.assertRaises(ValueError):
            validate_review(self.gap, self.candidate, self.result, reviewer={**self.reviewer, "finish_reason": "length"})


    def test_schema_identity_is_stable_across_python_hash_seeds(self):
        command = ("from cot_filler.grounding_review_v4 import response_format; "
                   "from cot_filler.core import digest; print(digest(response_format()))")
        repo = Path(__file__).resolve().parents[2]
        identities = [subprocess.check_output(
            [sys.executable, "-c", command], cwd=repo,
            env={**os.environ, "PYTHONHASHSEED": str(seed)}, text=True).strip()
            for seed in (0, 1, 2)]
        self.assertEqual(len(set(identities)), 1)
        self.assertEqual(identities[0], digest(response_format()))


    def test_schema_constrains_format_but_does_not_encode_case_answers(self):
        schema = response_format()["json_schema"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["properties"]), set(self.result))
        self.assertEqual(schema["properties"]["schema"]["enum"], [REVIEW_VERSION])
        self.assertEqual(set(schema["properties"]["checks"]["required"]), set(CHECKS))
        self.assertEqual(schema["properties"]["statements"]["items"]["properties"]["id"], {"type": "string"})

    def test_structural_wrapper_is_not_sole_support_and_arguments_stay_visible(self):
        self.gap["prefix"][0]["data"] = {"block": {"type": "input_text"},
                                             "arguments": {"type": "actual-business-value"}}
        self.gap["prefix_hash"] = digest(self.gap["prefix"])
        contract, evidence = review_contract(self.gap, self.candidate)
        wrapper_id = next(k for k, v in evidence.items() if v["quote"] == "input_text")
        self.assertIn("actual-business-value", canonical(contract))
        bad = copy.deepcopy(self.result)
        bad["statements"][0]["evidence_ids"] = [wrapper_id]
        with self.assertRaisesRegex(ValueError, "structural wrapper"):
            validate_review(self.gap, self.candidate, bad, reviewer=self.reviewer)


if __name__ == "__main__":
    unittest.main()
