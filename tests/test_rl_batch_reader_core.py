"""The trainer's sample-buffer reader lives in the core (``yeto.rl.engine.batch_reader``).

Miles, verl and ms-swift read through it, and the trainer breakers it owns
(``trainer.sample_wait``, ``policy_age``, ``buffer_damaged``, ``group_integrity``,
``sample_alignment``, ``compaction_round``) trip under the same names and events on a
verl buffer trainer as on a Miles one. CPU only, no verl, no Miles.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from yeto.breakers.registry import load_layers
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.verl import buffer_feed as bf
from yeto.rl.adapters.verl.fully_async_translate import VersionMap
from yeto.rl.engine import batch_reader
from yeto.rl.engine.buffer_payload import DamagedPayloadLimit, GroupWriter
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer

REPO = Path(__file__).resolve().parents[1]
READER_BREAKERS = ("trainer.sample_wait", "trainer.policy_age", "trainer.buffer_damaged",
                   "trainer.group_integrity", "trainer.sample_alignment", "trainer.compaction_round")
SHAPE = bf.RowShape(prompt_length=6, response_length=8, pad_token_id=0)


def _sample(version, *, prompt=(11, 12), response=(21, 22, 23), reward=1.0, index=0, compaction=None):
    meta = {"weight_version": str(version), "prompt_tokens_known": True, "finish_reason": "stop",
            "provenance": {"engine": "fake"}}
    if compaction is not None:
        meta["compaction"] = compaction
    return {"index": index, "prompt": "p", "tokens": list(prompt) + list(response),
            "response_length": len(response), "rollout_log_probs": [-0.5] * len(response),
            "reward": reward, "metadata": meta}


def _buffer(tmp_path):
    return FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)


def _feed(tmp_path, events, records=None, **kwargs):
    vmap = VersionMap()
    vmap.record(0, 0)
    return bf.BufferFeed(_buffer(tmp_path), vmap=lambda: vmap, shape=SHAPE, add_group=lambda rows: None,
                         record=(records if records is not None else []).append, sleep=lambda s: None,
                         emit=lambda event, **f: events.append({"event": event, **f}), **kwargs)


def _breakers(events):
    return [(e["name"], e["action"]) for e in events if e["event"] == "rl_breaker"]


# ---------------------------------------------------------------------- one reader
def test_the_reader_breakers_are_owned_by_the_core_reader():
    entries = {b.name: b for b in load_layers(("trainer",))}
    for name in READER_BREAKERS:
        assert entries[name].owner == "yeto.rl.engine.batch_reader", name


def test_miles_and_swift_read_through_the_core_reader():
    # the Miles module re-exports the core reader under its old names (one class, not a copy)
    assert br.BatchReader is batch_reader.BatchReader
    assert br.open_buffer is batch_reader.open_buffer and br.TakeReport is batch_reader.TakeReport
    assert br.WAIT_ENV == batch_reader.WAIT_ENV and br.TRAINER_CONSUMER == bf.TRAINER_CONSUMER
    tree = ast.parse((REPO / "yeto/swift_megatron/rl_buffer.py").read_text())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "yeto.rl.engine.batch_reader" in imported
    assert not any(m and m.startswith("yeto.rl.adapters.miles") for m in imported)


def test_the_core_reader_imports_no_backend():
    tree = ast.parse((REPO / "yeto/rl/engine/batch_reader.py").read_text())
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m.startswith(("yeto.rl.adapters", "miles", "verl", "megatron", "torch"))]


# ---------------------------------------------------------------------- verl: same breakers
def test_verl_sample_wait_trips_the_same_breaker_as_miles(tmp_path):
    clock = iter(range(0, 100, 10))
    events: list = []
    feed = _feed(tmp_path, events, wait_s=15, clock=lambda: next(clock))
    with pytest.raises(bf.BufferFeedError, match="no buffer group within 15 s"):
        feed.pull()
    [trip] = [e for e in events if e["event"] == "rl_breaker"]
    assert (trip["name"], trip["action"], trip["limit"]) == ("trainer.sample_wait", "stop", 15.0)
    assert trip["legacy_event"] == "SampleBufferRolloutError"

    miles: list = []
    reader = br.BatchReader(_buffer(tmp_path / "m"), max_policy_age=1, wait_s=15, sleep=lambda s: None,
                            clock=iter(range(0, 100, 10)).__next__, prefetch=False, incarnation_gate=None)
    reader.emit = lambda event, **f: miles.append({"event": event, **f})
    with pytest.raises(Exception):
        reader.take(0, 1, current_version=None)
    assert _breakers(miles) == _breakers(events)


def _damage(tmp_path, entry):
    path = tmp_path / "sample-buffer" / "payloads" / Path(entry.payload_ref).name
    if not path.exists():
        path = next(p for p in tmp_path.rglob("*.bin") if p.name == Path(entry.payload_ref).name)
    path.write_bytes(b"not a group")


def test_verl_passes_over_a_damaged_payload_and_trips_buffer_damaged_at_the_bound(tmp_path):
    writer = GroupWriter(_buffer(tmp_path), "s0")
    for key in "abcd":
        writer.write([_sample(0, index=0), _sample(0, index=1)], group_key=key)
    entries = _buffer(tmp_path).entries()
    _damage(tmp_path, entries[0])
    events, records = [], []
    feed = _feed(tmp_path, events, records, max_damaged_run=2)
    meta = feed.pull()  # position 0 is damaged: recorded, dropped, the next group is pulled
    assert meta["uid"] == "buf1"
    assert records[0]["dropped"] == bf.DAMAGED_PAYLOAD_REASON and records[0]["position"] == 0
    assert feed.dropped == {bf.DAMAGED_PAYLOAD_REASON: 1} and not _breakers(events)
    feed.keep()
    feed.close_round(0)  # the bound counts per round (the Miles reader: per take), as there
    _damage(tmp_path, entries[2])
    _damage(tmp_path, entries[3])
    with pytest.raises(DamagedPayloadLimit, match=r"\(2 in a row, 0 decoded\)"):
        feed.pull()
    assert _breakers(events) == [("trainer.buffer_damaged", "stop")]


def test_verl_reports_group_integrity_and_sample_alignment_once_per_round(tmp_path):
    writer = GroupWriter(_buffer(tmp_path), "s0")
    writer.write([_sample(0, index=0), _sample(0, index=1, reward=float("nan"))], group_key="nan")
    writer.write([_sample(0, index=0, prompt=tuple(range(1, 9))), _sample(0, index=1)], group_key="wide")
    writer.write([_sample(0, index=0)], group_key="short")
    writer.write([_sample(0, index=0), _sample(0, index=1)], group_key="good")
    events: list = []
    feed = _feed(tmp_path, events, group_size=2)
    assert feed.pull()["uid"] == "buf3"
    feed.keep()
    assert not _breakers(events)  # reported when the round closes, as the Miles reader does per take
    feed.close_round(0)
    trips = {e["name"]: e for e in events if e["event"] == "rl_breaker"}
    assert set(trips) == {"trainer.group_integrity", "trainer.sample_alignment"}
    assert trips["trainer.group_integrity"]["count"] == 2  # the short group and the NaN reward
    assert trips["trainer.group_integrity"]["detail"] == "rollout 0: 1 dropped_small, 1 nonfinite_reward"
    assert trips["trainer.sample_alignment"]["count"] == 1 and trips["trainer.sample_alignment"]["action"] == "drop"
    feed.close_round(1)  # a round that dropped nothing reports nothing
    assert len(_breakers(events)) == 2


def test_verl_refuses_a_round_of_compacted_trials(tmp_path):
    seg = {"trial": 0, "segment": 0, "segments": 2, "context_budget": 64}
    writer = GroupWriter(_buffer(tmp_path), "s0")
    writer.write([_sample(0, index=0, compaction=seg), _sample(0, index=0, compaction={**seg, "segment": 1})],
                 group_key="compacted")
    events: list = []
    feed = _feed(tmp_path, events)
    with pytest.raises(bf.SampleBufferRolloutError, match="compacted trials"):
        feed.pull()
    assert _breakers(events) == [("trainer.compaction_round", "stop")]


def test_verl_buffer_trainer_reports_policy_age_like_the_miles_reader():
    events: list = []
    batch_reader.report_round_guards(lambda event, **f: events.append({"event": event, **f}), rollout_id=3,
                                     max_policy_age=2, stale_groups=4)
    [trip] = events
    assert (trip["name"], trip["count"], trip["legacy_event"]) == ("trainer.policy_age", 4,
                                                                   "rl_buffer_stale_skipped")
    assert trip["detail"] == "rollout 3: 4 groups older than 2 versions"
    src = (REPO / "yeto/rl/adapters/verl/fully_async_ports.py").read_text()
    assert "report_round_guards(isl.emit" in src and 'carry["over_age_discarded_groups"]' in src
