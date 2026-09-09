import json
from pathlib import Path
import tempfile
import unittest

from cot_filler.core import canonical
from cot_filler.corpus_audit import audit_prepared_source
from cot_filler.corpus_source import file_sha256, prepare_source


class CorpusAuditTests(unittest.TestCase):
    def test_full_selected_source_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archive.jsonl"
            events = [
                {"event_id": "u1", "kind": "message", "role": "user", "content": "Inspect the function", "data": {}},
                {"event_id": "r1", "kind": "reasoning", "role": "assistant", "content": "", "data": {}},
                {"event_id": "a1", "kind": "tool_call", "role": "assistant", "content": "", "data": {"name": "read", "arguments": {}}},
            ]
            archive.write_text("\n".join(map(canonical, events)) + "\n")
            record = {"capture_id": "c1", "archive_path": str(archive), "archive_file_sha256": file_sha256(archive),
                      "quality_score": 4, "overall_confidence_score": 3}
            manifest = root / "manifest.jsonl"
            manifest.write_text(canonical(record) + "\n")
            source, report = root / "source.jsonl", root / "report.json"
            prepare_source(manifest, source, report, expected_manifest_sha256=file_sha256(manifest))
            result = audit_prepared_source(report)
            self.assertTrue(result["structural_audit_passed"])
            self.assertEqual(result["validated_gaps"], 1)
            self.assertEqual(result["context_counted_gaps"], 0)
            self.assertFalse(result["semantic_quality_validated"])
            trace = json.loads(source.read_text())
            trace["metadata"]["selection"]["quality_score"] = 3
            source.write_text(canonical(trace) + "\n")
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                audit_prepared_source(report)


if __name__ == "__main__":
    unittest.main()
