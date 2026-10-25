"""S27 G-EA end to end on CPU: real syncer, one learner, a stand-in spot sampler.

The spot sampler is the real entry (``python -m yeto.rl.spot_rollout.sampler``)
with the fake engine and an identity Megatron -> HF converter. The learner
publishes its fragment layout on the relay board and trains (pushes) while
the sampler subscribes, generates and sends groups through the relay. The
trainer side is the relay pump into a local sample buffer, read with the
trainer's ``BatchReader`` (policy age gate). The sampler is killed with
SIGKILL (a reclaim without notice) and a new incarnation starts.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.test_spot_rollout_syncer import (  # noqa: F401 - fixtures
    KEY,
    P,
    ROOT,
    Learner,
    layout,
    needs_cargo,
    start_syncer,
    syncer_binary,
    wait_for,
)


def half_reward(item, completion) -> float:
    return 0.5


def _spec(tmp_path: Path, port: int) -> Path:
    from yeto.rl.engine.sampler_checks import GSM8K_FIXED_PROMPTS

    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    (base / "config.json").write_text(json.dumps({"model_type": "fake", "hidden_size": 4,
                                                  "num_attention_heads": 1, "vocab_size": 8,
                                                  "num_hidden_layers": 1}))
    (base / "tokenizer.json").write_text("{}")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"q": f"question {i}", "a": str(i)}) + "\n" for i in range(8)))
    spec = {
        "sampler_id": "spot0",
        "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "fake", "kwargs": {
            "answers": {p.prompt: f"The answer is {p.answer}." for p in GSM8K_FIXED_PROMPTS},
            "max_model_len": 64}},
        "sampling": {"max_tokens": 6, "n": 2},
        "max_backlog_groups": 4, "max_prompt_tokens": 32,
        "prompts": {"path": str(prompts), "input_key": "q", "label_key": "a", "groups_per_round": 2},
        "reward": "tests.test_spot_rollout_e2e:half_reward",
        "event_tape": str(tmp_path / "tape" / "events.jsonl"),
        "syncer": f"127.0.0.1:{port}", "channel": 0, "telemetry": False,
        "base_model": str(base), "work_dir": str(tmp_path / "work"),
        "converter": "yeto.rl.spot_rollout.materialize:identity_converter",
        "num_streams": 1, "poll_s": 0.05, "layout_timeout_s": 60, "syncer_lost_s": 60,
    }
    path = tmp_path / "spot-spec.json"
    path.write_text(json.dumps(spec))
    return path


def _start_sampler(spec: Path, log: Path) -> subprocess.Popen:
    env = {**os.environ, "YETO_ISLAND_HMAC_KEY": KEY, "PYTHONPATH": str(ROOT)}
    with log.open("w") as out:
        return subprocess.Popen([sys.executable, "-m", "yeto.rl.spot_rollout.sampler", "--spec", str(spec)],
                                cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)


class PacedLearner(Learner):
    """A learner whose pushes wait ``delay`` seconds (the test speeds it up at the end)."""

    def __init__(self, port: int, **kw) -> None:
        self.delay = 0.15
        self.pace = threading.Event()
        self.pace.set()
        super().__init__(port, hold=self.pace, on_merge=self._merged, **kw)
        self.merges: list[int] = []

    def _merged(self, fid: int, version: int) -> None:
        self.merges.append(version)
        if self.delay:
            self.pace.clear()
            threading.Timer(self.delay, self.pace.set).start()

    def merged_version(self) -> int:
        from yeto.rl.engine.full_parameter_sync import merged_version

        return merged_version([self.held[f][0] for f in range(P)])


@needs_cargo
def test_spot_sampler_feeds_the_trainer_and_survives_a_reclaim(syncer_binary, tmp_path, monkeypatch):
    from yeto.rl.adapters.miles.buffer_rollout import BatchReader
    from yeto.rl.engine.resume import CheckpointStore
    from yeto.rl.engine.sample_buffer import FileSampleBuffer
    from yeto.rl.spot_rollout import layout_manifest
    from yeto.rl.spot_rollout.relay import RelayBoard, RelayConsumer
    from yeto.rl.spot_rollout.relay_buffer import RelayPump

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", KEY)
    total = 200 * P
    process, port, syncer_log = start_syncer(
        syncer_binary, tmp_path, total_steps=total,
        extra=("--max-subscribers", "2", "--sample-relay", "--sample-relay-max-items", "64"))
    addr = ("127.0.0.1", port)
    learner = PacedLearner(port).start()
    layout_manifest.publish(RelayBoard(addr, layout_manifest.LAYOUT_BOARD), layout(), {"role": "actor"})
    # Trainer side: the relay pump fills a local buffer that the trainer's reader reads.
    store = CheckpointStore(tmp_path / "trainer-buffer")
    reader_buffer = FileSampleBuffer(store, batch_size=2)
    pump = RelayPump(RelayConsumer(addr, 0), FileSampleBuffer(store, batch_size=1),
                     reader=reader_buffer, prefetch_groups=8, wait_s=0.5).start()
    reader = BatchReader(reader_buffer, max_policy_age=2, wait_s=60, poll_s=0.05, read_workers=1,
                         prefetch=False)
    spec = _spec(tmp_path, port)
    first = _start_sampler(spec, tmp_path / "sampler-1.log")
    taken: list[tuple[int, int, int]] = []  # (rollout id, behavior version, trainer merged version)
    stale = 0
    try:
        rollout = 0

        def take(rounds: int) -> None:
            nonlocal rollout, stale
            for _ in range(rounds):
                current = learner.merged_version()
                groups, report = reader.take(rollout, 2, current_version=current)
                stale += report.stale_groups
                for _position, group in groups:
                    versions = {int(s["metadata"]["weight_version"]) for s in group}
                    assert len(versions) == 1
                    taken.append((rollout, versions.pop(), current))
                rollout += 1

        take(4)
        assert max(v for _, v, _ in taken) >= 1, "the sampler never served merged weights"
        # Reclaim without notice: SIGKILL. The learner keeps merging.
        first.send_signal(signal.SIGKILL)
        first.wait(timeout=10)
        merges_at_kill = len(learner.merges)
        version_at_kill = learner.merged_version()
        wait_for(lambda: len(learner.merges) >= merges_at_kill + 2 * P, "merges stopped after the reclaim")
        second = _start_sampler(spec, tmp_path / "sampler-2.log")
        before = len(taken)
        take(4)
        after_restart = [v for _, v, _ in taken[before:]]
        assert max(after_restart) >= version_at_kill
        learner.delay = 0.0
        learner.pace.set()
        learner.join(timeout=120)
        assert process.wait(timeout=60) == 0
        assert second.wait(timeout=60) == 0
    finally:
        pump.stop()
        for proc in (first,):
            if proc.poll() is None:
                proc.kill()
        if process.poll() is None:
            process.kill()
    # Every merge happened: the reclaim and the subscribers never blocked a round.
    assert learner.pushed == list(range(1, total + 1))
    # The age gate held: every trained group is at most 2 merged versions old.
    assert all(current - version <= 2 for _, version, current in taken)
    # Two incarnations wrote groups; sample ids never repeat in the trainer's buffer.
    reader_buffer.refresh()
    ids = [e.sample_id for e in reader_buffer.entries()]
    assert len(ids) == len(set(ids))
    incarnations = {i.split("-")[1] for i in ids}
    assert len(incarnations) == 2
    tapes = [[json.loads(line) for line in tape.read_text().splitlines()]
             for tape in (tmp_path / "tape").glob("events.*.jsonl")]
    assert len(tapes) == 2
    left = [t for t in tapes if any(e["event"] == "rl_sampler_leave" for e in t)]
    assert len(left) == 1  # the killed incarnation left no record; the second one finished
    [leave] = [e for e in left[0] if e["event"] == "rl_sampler_leave"]
    assert leave["reason"] == "FINISHED"
    for events in tapes:
        names = [e["event"] for e in events]
        assert "rl_spot_sampler_channels" in names and "rl_sampler_join" in names
        loads = [e for e in events if e["event"] == "rl_sampler_policy_loaded"]
        assert loads and loads[0]["policy_version"] >= 1
        rounds = [e for e in events if e["event"] == "rl_sampler_engine_round"]
        assert rounds and all(r["groups"] >= 1 for r in rounds)
    assert "subscriber connected" in syncer_log.read_text()
