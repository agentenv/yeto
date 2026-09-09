import copy
import unittest

from cot_filler.corpus_source import SOURCE_VERSION, normalize_archive


class TypedAssistantSourceTests(unittest.TestCase):
    def archive(self, blocks):
        return [
            {"event_id": "u1", "role": "user", "kind": "message", "content": "Read the file.", "data": {}},
            {"event_id": "r1", "role": "assistant", "kind": "reasoning", "content": "private", "data": {}},
            {"event_id": "a1", "role": "assistant", "kind": "message", "content": "I will inspect it.",
             "data": {"block": {"type": "agent_message", "author": "assistant", "recipient": "all", "content": blocks}}},
            {"event_id": "a2", "role": "assistant", "kind": "tool_call", "content": "",
             "data": {"type": "custom_tool_call", "name": "run", "input": "cat source.py", "call_id": "c1"}},
        ]

    def test_clean_typed_fields_preserves_blocks_tools_and_original_boundary(self):
        blocks = [{"type": "input_text", "text": "<think>private</think>I will inspect it.", "annotation": "retained"},
                  {"type": "output_text", "text": "Check the contents."},
                  {"type": "refusal", "refusal": "No unsafe action."}]
        archive = self.archive(blocks)
        original = copy.deepcopy(archive)
        trace = normalize_archive(archive, {"capture_id": "t"})
        self.assertEqual(archive, original)
        self.assertEqual(trace["source_format"], SOURCE_VERSION)
        actual = trace["events"][1]["data"]["block"]["content"]
        self.assertEqual(actual[0], {"type": "input_text", "text": "I will inspect it.", "annotation": "retained"})
        self.assertEqual(actual[1:], blocks[1:])
        self.assertEqual(trace["events"][2], archive[3])
        self.assertEqual(trace["gap_targets"][0]["event_id"], "a1")
        self.assertEqual(trace["gap_targets"][0]["source_reasoning_events"], [{"event_id": "r1", "archive_position": 1}])

    def test_unknown_reasoning_and_nested_nontext_blocks_fail_closed(self):
        for block in ({"type": "image_url", "image_url": "opaque"},
                      {"type": "reasoning", "text": "private"},
                      {"type": "input_text", "text": ["unknown"]},
                      {"type": "input_text"},
                      {"type": "input_text", "text": "visible", "reasoning": "private"}):
            with self.subTest(block=block):
                with self.assertRaises(ValueError):
                    normalize_archive(self.archive([block]), {"capture_id": "t"})


if __name__ == "__main__":
    unittest.main()
