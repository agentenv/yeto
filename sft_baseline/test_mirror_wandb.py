import copy
import json
import unittest

from sft_baseline.mirror_wandb import clean_row, ingest, parse_bytes, parse_stdout, send_pending, source_identity, verify_state_source, remote_script


def metric(**updates):
    return {"step": 0, "epoch": 0, "timestamp": "2026-09-09T00:30:00+00:00", "loss": 0.5, **updates}


class MirrorTests(unittest.TestCase):
    def test_new_run_cannot_reuse_previous_source_journal(self):
        source = source_identity("a" * 64, "/data/sft_baseline_20260908/runs/new-run/checkpoints", "new123")
        state = {"schema": 2, "source": source}
        verify_state_source(state, source)
        for key, value in [("container", "b" * 64), ("directory", "/different"), ("run_id", "other")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_state_source(state, dict(source, **{key: value}))
        with self.assertRaises(ValueError):
            verify_state_source({"schema": 1, "source_container": "a" * 64}, source)
        with self.assertRaises(ValueError):
            verify_state_source(dict(state, finished=True), source)
        for directory in ["/etc/checkpoints", "/data/sft_baseline_20260908/runs/new/../old/checkpoints"]:
            with self.assertRaises(ValueError): source_identity("a" * 64, directory, "")
        script = remote_script(source)
        compile(script, "remote_numeric_reader", "exec")
        self.assertIn(source["container"], script)
        self.assertIn(source["directory"], script)

    def test_stdout_fallback_and_later_jsonl_deduplicate(self):
        line = "2026-09-09 00:22:10 | INFO | root | step 0 | epoch 0 | loss 0.7507 | grad_norm 3.2170 | lr 1.00e-05 | mem 92.01 GiB | tps 1163.40(145.43/gpu) | num_label_tokens 97495"
        rows = parse_stdout("unrelated private text\n" + line + "\n")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["payload"]["num_label_tokens"], 97495)
        self.assertEqual(parse_stdout(line + " SECRET"), [])
        self.assertEqual(parse_stdout(line.replace("0.7507", "nan")), [])
        progress = "\rTraining:   0%|          | 0/2068 [00:00<?, ?step/s]"
        self.assertEqual(parse_stdout(progress + line), rows)
        self.assertEqual(parse_stdout("unrelated private prefix " + line), [])
        exact_event = {"event": "verified_optimizer_update", "step": 0,
                       "measurements": {"loss": .7507123, "max_rank_peak_allocated_gib": 100.123,
                                        "max_rank_peak_reserved_gib": 104.456,
                                        "private": "must not leave node"}}
        exact_rows = parse_stdout(line + "\n" + json.dumps(exact_event))
        self.assertEqual(exact_rows[0]["payload"]["loss"], .7507123)
        self.assertEqual(exact_rows[0]["payload"]["max_rank_peak_allocated_gib"], 100.123)
        self.assertNotIn("private", exact_rows[0]["payload"])
        exact_event["measurements"]["loss"] = .9
        with self.assertRaises(ValueError): parse_stdout(line + "\n" + json.dumps(exact_event))
        state = {"events": [], "cursor": {}}
        snapshot = {"streams": {k: parse_bytes(b"", k) for k in ["train", "validation"]}, "stdout_rows": rows}
        self.assertEqual(ingest(state, snapshot), 1)
        precise = dict(rows[0]["payload"], loss=.7507123, grad_norm=3.2170123)
        snapshot["streams"]["train"] = parse_bytes(json.dumps(precise).encode()+b"\n", "train")
        self.assertEqual(ingest(state, snapshot), 0)
        self.assertTrue(state["events"][0]["jsonl_confirmed"])
        self.assertEqual(len(state["events"]), 1)
        wrong = copy.deepcopy(snapshot)
        wrong["stdout_rows"][0]["payload"]["loss"] = .1
        with self.assertRaises(ValueError): ingest(state, wrong)

    def test_whitelist_and_complete_lines(self):
        raw = json.dumps(metric(secret="never exported", config={"hidden": True})).encode() + b"\n"
        parsed = parse_bytes(raw + b'{"unfinished":', "train")
        self.assertEqual(parsed["complete_bytes"], len(raw))
        self.assertEqual(len(parsed["rows"]), 1)
        self.assertNotIn("secret", parsed["rows"][0]["payload"])
        self.assertNotIn("config", parsed["rows"][0]["payload"])

    def test_nonfinite_boolean_and_malformed_are_rejected(self):
        for value in [float("nan"), float("inf"), True, "1"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                clean_row(metric(loss=value), "train")
        with self.assertRaises(ValueError):
            parse_bytes(b"bad json\n", "train")
        with self.assertRaises(ValueError):
            clean_row(metric(timestamp="bad"), "train")

    def test_resume_keeps_axes_and_does_not_duplicate(self):
        train = parse_bytes(json.dumps(metric()).encode() + b"\n", "train")
        val = parse_bytes(json.dumps({"step": 0, "epoch": 0, "timestamp": "2026-09-09T00:31:00Z", "val_loss": .4}).encode() + b"\n", "validation")
        snapshot = {"streams": {"train": train, "validation": val}}
        state = {"events": [], "cursor": {}}
        self.assertEqual(ingest(state, snapshot), 2)
        self.assertEqual(ingest(state, snapshot), 0)

        class Run:
            def __init__(self, step): self.step, self.calls = step, []
            def log(self, payload, step, commit):
                self.calls.append((payload, step)); self.step = step + 1

        # Simulate crash after first event reached W&B but before local progress.
        run = Run(1)
        send_pending(run, state)
        self.assertEqual(len(run.calls), 1)
        self.assertEqual(run.calls[0][0]["validation/step"], 0)
        self.assertEqual(run.calls[0][1], 1)
        send_pending(run, state)
        self.assertEqual(len(run.calls), 1)
        # Simulate logs never delivered: durable outbox replays them both.
        run = Run(0)
        send_pending(run, state)
        self.assertEqual(len(run.calls), 2)

    def test_source_mutation_and_truncation_fail(self):
        data = json.dumps(metric()).encode() + b"\n"
        snapshot = {"streams": {"train": parse_bytes(data, "train"), "validation": parse_bytes(b"", "validation")}}
        state = {"events": [], "cursor": {}}
        ingest(state, snapshot)
        changed = copy.deepcopy(snapshot)
        changed["streams"]["train"]["rows"][0]["sha256"] = "different"
        with self.assertRaises(ValueError): ingest(state, changed)
        changed = copy.deepcopy(snapshot)
        changed["streams"]["train"]["complete_bytes"] = 0
        with self.assertRaises(ValueError): ingest(state, changed)


if __name__ == "__main__":
    unittest.main()
