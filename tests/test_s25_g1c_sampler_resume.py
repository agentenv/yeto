"""S25 T-G1C: a replaced engine sampler island continues where the buffer says
and never overwrites a payload or an earlier incarnation's tape (CPU)."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import pytest

from tests.test_stage1_gaps import _sampler_spec
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer, SampleBufferError
from yeto.rl.engine.sampler_resume import MEMBERS_DIR, RECLAIM_MARKER_ENV, SamplerResumeError

STATE_ENV = "YETO_TEST_G1C_REWARD_STATE"  # the reward is imported by name: share state via a file


def counting_reward(item, completion) -> float:
    """Fails (a lost container) or writes the reclaim marker after the limits in the state file."""
    path = Path(os.environ[STATE_ENV])
    state = json.loads(path.read_text())
    state["n"] += 1
    path.write_text(json.dumps(state))
    if state.get("limit") is not None and state["n"] > state["limit"]:
        raise RuntimeError("container lost")
    if state.get("mark_at") is not None and state["n"] == state["mark_at"]:
        Path(state["marker"]).write_text("{}")
    return 0.5


class _Calls:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.update(n=0)

    def update(self, **fields) -> None:
        state = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.path.write_text(json.dumps({**state, **fields}))


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    global CALLS
    monkeypatch.setenv(STATE_ENV, str(tmp_path / "reward-state.json"))
    CALLS = _Calls(tmp_path / "reward-state.json")
    monkeypatch.setenv(RECLAIM_MARKER_ENV, str(tmp_path / "reclaim-marker"))
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in handlers.items():
        signal.signal(sig, handler)


def _spec(tmp_path: Path, rounds: int = 4) -> dict:
    return _sampler_spec(tmp_path, num_rollouts=rounds, reward="tests.test_s25_g1c_sampler_resume:counting_reward")


def _buffer(tmp_path: Path) -> FileSampleBuffer:
    buffer = FileSampleBuffer(CheckpointStore(tmp_path / "store"), batch_size=1)
    buffer.refresh()
    return buffer


def _payloads(tmp_path: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in (tmp_path / "store" / "sample-buffer" / "payloads").iterdir()}


def _tape(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_replacement_continues_after_the_written_rounds_and_overwrites_nothing(tmp_path):
    spec = _spec(tmp_path)
    CALLS.update(limit=8)  # 2 prompts x n=2 per round: the third round fails before it writes
    with pytest.raises(RuntimeError, match="container lost"):
        entry.run(spec, resource_sampler=lambda emit: None, incarnation="aaaa0001")
    first = _payloads(tmp_path)
    assert len(first) == 4 and all(name.startswith("ppu4-aaaa0001-r") for name in first)

    CALLS.update(n=0, limit=None)
    assert entry.run(spec, resource_sampler=lambda emit: None, incarnation="bbbb0002") == 2
    after = _payloads(tmp_path)
    assert {k: after[k] for k in first} == first  # the first incarnation's payloads are untouched
    assert len(after) == 8
    buffer = _buffer(tmp_path)
    entries = buffer.entries()
    assert buffer.data_ended and len({e.sample_id for e in entries}) == len(entries) == 8
    rounds = [(e.sampler_round["incarnation"], e.sampler_round["rollout_id"]) for e in entries]
    assert rounds == [("aaaa0001", 0)] * 2 + [("aaaa0001", 1)] * 2 + [("bbbb0002", 2)] * 2 \
        + [("bbbb0002", 3)] * 2
    assert all(buffer.read_payload(e.payload_ref) for e in entries)

    tapes = {p.name: _tape(p) for p in (tmp_path / "tape").glob("events.*.jsonl")}
    assert set(tapes) == {"events.aaaa0001.jsonl", "events.bbbb0002.jsonl"}
    old = {e["event"]: e for e in tapes["events.aaaa0001.jsonl"]}
    assert old["rl_sampler_leave"]["reason"] == "INFRA_FAILED"
    new = {e["event"]: e for e in tapes["events.bbbb0002.jsonl"]}
    assert new["rl_sampler_join"]["next_rollout_id"] == 2
    assert new["rl_sampler_join"]["incarnations"] == ["aaaa0001"]
    assert new["rl_sampler_leave"]["reason"] == "FINISHED"
    members = sorted(p.name for p in (tmp_path / "store" / MEMBERS_DIR / "ppu4").iterdir())
    assert members == ["aaaa0001.json", "bbbb0002.json"]


def test_a_finished_sampler_restarted_generates_nothing(tmp_path):
    spec = _spec(tmp_path, rounds=2)
    assert entry.run(spec, resource_sampler=lambda emit: None, incarnation="aaaa0001") == 2
    before = _payloads(tmp_path)
    assert entry.run(spec, resource_sampler=lambda emit: None, incarnation="bbbb0002") == 0
    assert _payloads(tmp_path) == before


def test_a_written_round_is_refused_to_a_new_incarnation(tmp_path):
    spec = _spec(tmp_path)
    CALLS.update(limit=4)
    with pytest.raises(RuntimeError):
        entry.run(spec, resource_sampler=lambda emit: None, incarnation="aaaa0001")
    from yeto.rl.engine.engine_sampler import EngineRound
    from yeto.rl.engine.sampler_resume import resume_point

    resume = resume_point(_buffer(tmp_path).entries(), "ppu4")
    assert resume.next_rollout_id == 1
    round_ = EngineRound.__new__(EngineRound)
    from yeto.rl.engine.sampler_resume import RoundGuard

    round_.guard = RoundGuard(resume)
    with pytest.raises(SamplerResumeError, match="round 0 is already"):
        EngineRound.__call__(round_, 0)


def test_a_payload_is_never_overwritten(tmp_path):
    buffer = FileSampleBuffer(CheckpointStore(tmp_path / "store"), batch_size=1)
    buffer.write_payload("ppu4-a-r0-row0", b"one")
    with pytest.raises(SampleBufferError, match="never overwritten"):
        buffer.write_payload("ppu4-a-r0-row0", b"two")
    assert (tmp_path / "store" / "sample-buffer" / "payloads" / "ppu4-a-r0-row0.bin").read_bytes() == b"one"


def test_reclaim_marker_stops_writes_and_leaves_preempted(tmp_path):
    spec = _spec(tmp_path)
    CALLS.update(marker=str(tmp_path / "reclaim-marker"), mark_at=6)  # inside round 1
    # round 1 was generated (2 rounds), but none of its groups is written
    assert entry.run(spec, resource_sampler=lambda emit: None, incarnation="aaaa0001") == 2
    buffer = _buffer(tmp_path)
    assert not buffer.data_ended
    assert [e.sampler_round["rollout_id"] for e in buffer.entries()] == [0, 0]
    [tape] = (tmp_path / "tape").glob("events.*.jsonl")
    leave = [e for e in _tape(tape) if e["event"] == "rl_sampler_leave"]
    assert len(leave) == 1 and leave[0]["reason"] == "PREEMPTED"
    # the next incarnation continues: this tape is not complete (S26 E9)
    assert not any(e["event"] == "rl_learner_finalized" for e in _tape(tape))


def test_caller_stop_ends_the_island_finished_and_on_event_sees_the_tape(tmp_path):
    """T-E2V: the eval harness's end condition (trainers done) stops the loop
    without a reclaim: the round in progress is still written and the island
    leaves FINISHED; on_event sees every record written to the tape."""
    spec = _spec(tmp_path)
    seen = []
    rounds = entry.run(spec, resource_sampler=lambda emit: None, incarnation="aaaa0001",
                       stop=lambda: any(e == "rl_sampler_round" for e, _ in seen),
                       on_event=lambda event, **fields: seen.append((event, fields)))
    assert rounds == 1
    assert [e.sampler_round["rollout_id"] for e in _buffer(tmp_path).entries()] == [0, 0]
    [tape] = (tmp_path / "tape").glob("events.*.jsonl")
    assert [e["event"] for e in _tape(tape)] == [e for e, _ in seen]
    assert [f["reason"] for e, f in seen if e == "rl_sampler_leave"] == ["FINISHED"]
    # S26 E9 G1 (e9-g1-i, launcher exit 3): a finished island's tape ends with the
    # record the launcher requires of every island tape
    assert seen[-1] == ("rl_learner_finalized", {"rounds": 1})


def test_hold_after_reclaim_only_when_the_marker_is_there(tmp_path):
    """A reclaimed engine sampler process does not exit cleanly at once (the cloud
    could end its input as finished instead of running it again)."""
    env = {RECLAIM_MARKER_ENV: str(tmp_path / "marker")}
    slept = []
    assert entry.hold_after_reclaim(env, hold_s=5.0, sleep=slept.append) is False and slept == []
    (tmp_path / "marker").write_text("{}")
    assert entry.hold_after_reclaim(env, hold_s=5.0, sleep=slept.append) is True and slept == [5.0]


def test_a_launcher_island_tape_is_recovered_complete(tmp_path):
    """S26 E9 G1 (e9-g1-j, exit 3): with the spec's island_id every record carries
    it, so the launcher's tape recovery accepts the file and sees the finalized
    record; without it the records are discarded as not island tape lines."""
    from yeto.rl.event_echo import TapeCollector

    spec = {**_spec(tmp_path, rounds=1), "island_id": 0}
    assert entry.run(spec, resource_sampler=lambda emit: None, incarnation="aaaa0001") == 1
    [tape] = (tmp_path / "tape").glob("events.*.jsonl")
    assert all(r["island_id"] == 0 for r in _tape(tape))
    collector = TapeCollector(tmp_path / "events" / "l0.jsonl", fresh=False)
    assert collector.recover_from_file(tape) is True
    assert collector.discarded == 0 and collector.count == len(_tape(tape))
