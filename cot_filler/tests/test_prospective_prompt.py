import json
from pathlib import Path
import unittest

from cot_filler.core import PROMPT_VERSION, canonical, gaps_for, prompt_messages
from cot_filler.grounding_review_ids import review_messages


class ProspectivePromptTests(unittest.TestCase):
    def test_v5_preserves_full_prefix_and_five_event_orientation(self):
        trace = json.loads((Path(__file__).resolve().parents[1] / "examples/demo.json").read_text())
        gap = next(gaps_for(trace))
        messages = prompt_messages(gap)
        blob = messages[1]["content"].split("\n", 1)[1].rsplit("\n", 1)[0]
        data = json.loads(blob)
        expected = lambda events: [event for event in events if event.get("kind") != "tool_definition"]
        self.assertEqual(data["original_prefix_events"], expected(gap["prefix"]))
        self.assertEqual(data["future_next_five_events"], expected(gap["lookahead"]))
        self.assertIn("v5-prospective-english", PROMPT_VERSION)
        self.assertIn("1-3 concise sentences in ENGLISH", messages[0]["content"])
        self.assertIn("FIRST PERSON", messages[0]["content"])
        self.assertIn("If a detail is visible only in the continuation, omit it", messages[0]["content"])

    def test_reviewer_stays_prefix_only_after_generator_revision(self):
        trace = json.loads((Path(__file__).resolve().parents[1] / "examples/demo.json").read_text())
        trace["events"][2]["content"] += " FUTURE_SENTINEL_V5"
        gap = next(gaps_for(trace))
        self.assertIn("FUTURE_SENTINEL_V5", canonical(prompt_messages(gap)))
        self.assertNotIn("FUTURE_SENTINEL_V5", canonical(review_messages(gap, "I will inspect the function before changing it.")))


if __name__ == "__main__":
    unittest.main()
