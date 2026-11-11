"""A replacement sampler with its own shard continues the prompts of the shard it
replaces (``continues_shard``, sampler_resume handoff), in the same data slot (CPU)."""

from __future__ import annotations

import json
from pathlib import Path

from tests.test_stage1_gaps import _sampler_spec
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer, shard_index
from yeto.rl.engine.sampler_resume import (MEMBERS_DIR, handoff_requested, handoff_start, request_handoff,
                                           resume_point)


def _spec(tmp_path: Path, shard: str, rounds: int, **over) -> dict:
    spec = _sampler_spec(tmp_path, sampler_id=shard, num_rollouts=rounds, buffer_shards=["a", "b"], **over)
    (tmp_path / "prompts.jsonl").write_text(
        "".join(json.dumps({"q": f"question {i}", "a": str(i)}) + "\n" for i in range(20)))
    spec["prompts"] = {**spec["prompts"], "data_slot": "1/2"}
    spec["event_tape"] = str(tmp_path / f"tape-{shard}" / "events.jsonl")
    return spec


def _rows(tmp_path: Path, shard: str) -> list[tuple[int, str]]:
    buffer = FileSampleBuffer(CheckpointStore(tmp_path / "store"), batch_size=1, index=shard_index(shard),
                              create_index=False)
    out = []
    for e in buffer.entries():
        group = json.loads(buffer.read_payload(e.payload_ref))
        out.append((e.sampler_round["rollout_id"], json.dumps(group)))
    return out


def _prompt_ids(tmp_path: Path, shard: str) -> list[int]:
    buffer = FileSampleBuffer(CheckpointStore(tmp_path / "store"), batch_size=1, index=shard_index(shard),
                              create_index=False)
    ids = []
    for e in buffer.entries():
        text = buffer.read_payload(e.payload_ref).decode()
        ids.append(next(i for i in range(20) if f"question {i}\\" in text or f"question {i}\"" in text))
    return ids


def test_replacement_continues_the_predecessor_prompts(tmp_path):
    assert entry.run(_spec(tmp_path, "a", 2), resource_sampler=lambda emit: None) == 2  # rounds 0, 1
    assert entry.run(_spec(tmp_path, "b", 4, continues_shard="a", handoff_wait_s=0),
                     resource_sampler=lambda emit: None) == 2  # rounds 2, 3
    a, b = _prompt_ids(tmp_path, "a"), _prompt_ids(tmp_path, "b")
    assert a == [1, 3, 5, 7] and b == [9, 11, 13, 15]  # slot 1/2 rows, contiguous, no repeat
    rounds = [r for r, _ in _rows(tmp_path, "b")]
    assert min(rounds) == 2  # continues after the predecessor, never at round 0
    store = CheckpointStore(tmp_path / "store")
    assert handoff_requested(store, "a") and not handoff_requested(store, "b")
    [tape] = (tmp_path / "tape-b").glob("events.*.jsonl")
    handoff = [json.loads(l) for l in tape.read_text().splitlines() if '"rl_sampler_handoff"' in l][0]
    assert handoff["predecessor_left"] is True and handoff["start_rollout_id"] == 2
    merged = FileSampleBuffer(store, batch_size=1, merge_shards=True, expected_shards=["a", "b"])
    assert len(merged) == 8 and merged.shard_positions(8) == {"a": 4, "b": 4}


def test_a_handed_off_shard_ends_so_the_merged_data_can_end(tmp_path):
    store = CheckpointStore(tmp_path / "store")
    request_handoff(store, "a", "b")
    entry.run(_spec(tmp_path, "a", 3), resource_sampler=lambda emit: None)
    entry.run(_spec(tmp_path, "b", 2, continues_shard="a", handoff_wait_s=0), resource_sampler=lambda emit: None)
    assert FileSampleBuffer(store, batch_size=1, merge_shards=True, expected_shards=["a", "b"]).data_ended


def test_a_handed_off_sampler_starts_no_new_round(tmp_path):
    store = CheckpointStore(tmp_path / "store")
    request_handoff(store, "a", "b")
    assert entry.run(_spec(tmp_path, "a", 3), resource_sampler=lambda emit: None) == 0


def test_a_predecessor_still_running_skips_its_round_in_flight(tmp_path):
    assert entry.run(_spec(tmp_path, "a", 2), resource_sampler=lambda emit: None) == 2
    store = CheckpointStore(tmp_path / "store")
    members = tmp_path / "store" / MEMBERS_DIR / "a"
    for path in members.glob("*.json"):  # as if the incarnation were still running
        record = json.loads(path.read_text())
        record["left_at"] = None
        path.write_text(json.dumps(record))
    entries = lambda: FileSampleBuffer(store, batch_size=1, index=shard_index("a"), create_index=False).entries()  # noqa: E731
    start, detail = handoff_start(store, "a", "b", entries, wait_s=0, sleep=lambda s: None)
    assert resume_point(entries(), "a").next_rollout_id == 2
    assert start == 3 and detail["predecessor_left"] is False
