import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cot_filler.core import digest
from cot_filler.prepare_diverse_pilot import prepare, selected_trace


class DiversePilotTests(unittest.TestCase):
    def test_selection_preserves_full_events_and_original_boundary_provenance(self):
        trace = json.loads((Path(__file__).resolve().parents[1] / "examples/demo.json").read_text())
        trace.setdefault("metadata", {})
        original = copy.deepcopy(trace)
        target = trace["gap_targets"][-1]
        selected = selected_trace(trace, [target], {"source_sha256": "unit-test-only"})
        self.assertEqual(trace, original)
        self.assertEqual(selected["events"], trace["events"])
        self.assertEqual(selected["gap_targets"], [target])
        self.assertEqual(selected["metadata"]["pilot_selection"]["original_trace_digest"], digest(trace))
        self.assertNotEqual(digest(selected), digest(trace))
        modified = {**target, "reason": "invented"}
        with self.assertRaises(ValueError):
            selected_trace(trace, [modified], {})
        with self.assertRaises(ValueError):
            selected_trace(trace, [target, target], {})

    def test_new_pilot_excludes_every_previous_group_and_preserves_source(self):
        demo = json.loads((Path(__file__).resolve().parents[1] / "examples/demo.json").read_text())
        traces = []
        for index, group in enumerate(("previous", "previous", "fresh-a", "fresh-b")):
            trace = copy.deepcopy(demo)
            trace["trace_id"] = "fixture-" + str(index)
            trace["metadata"]["selection"] = {"group_id": group}
            traces.append(trace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            previous = root / "previous.jsonl"
            raw = b"".join((json.dumps(t) + "\n").encode() for t in traces)
            source.write_bytes(raw)
            previous.write_text(json.dumps(traces[0]) + "\n")
            kwargs = dict(traces=2, pool_size=2, max_scan=4, exclude_source=previous,
                          exclude_source_sha256=hashlib.sha256(previous.read_bytes()).hexdigest())
            with patch("cot_filler.prepare_diverse_pilot._budget", return_value=1000):
                report = prepare(source, hashlib.sha256(raw).hexdigest(), root / "out.jsonl", root / "report.json",
                                 object(), object(), **kwargs)
            self.assertEqual(report["selected_gaps"], 4)
            self.assertEqual(report["selection_exclusions"]["previous_pilot_group"], 2)
            self.assertEqual({r["group_id"] for r in report["selections"]}, {"fresh-a", "fresh-b"})
            self.assertEqual(source.read_bytes(), raw)
            selected = [json.loads(line) for line in (root / "out.jsonl").read_bytes().splitlines()]
            self.assertEqual([t["events"] for t in selected], [t["events"] for t in traces[2:]])
            with self.assertRaisesRegex(ValueError, "Excluded pilot source hash mismatch"):
                prepare(source, hashlib.sha256(raw).hexdigest(), root / "bad.jsonl", root / "bad-report.json",
                        object(), object(), **{**kwargs, "exclude_source_sha256": "0" * 64})


if __name__ == "__main__":
    unittest.main()
