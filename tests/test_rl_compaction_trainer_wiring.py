"""Trainer-side compaction metrics wired end to end (C10's trainer hook points).

* ``build_metadata`` adds ``compaction`` (``timeline.compaction_stats`` of the trained
  samples); ``handle_from_metadata`` keeps it; ``IslandDriver._train_fields`` puts it on
  ``rl_round_trained``; W&B shows it as ``train/compaction/<key>``.
* the reader counts the stale trials by segment count; the driver tapes them as
  ``rl_buffer_stale_skipped.stale_by_segments``.
* the ledger group record's reward statistics count each trial once.
* the xhw downweight reaches the trained samples, so the xhw compaction metrics show.

Without compaction none of these fields appears.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_rl_buffer_rollout import _import_plugin_with_stub_miles
from test_rl_compaction_grpo_trainer import N, _buffer, _reader, _write, group
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles import rollout_meta_hook as hook
from yeto.rl.adapters.miles.rollout import handle_from_metadata
from yeto.rl.engine.driver import IslandDriver, _stale_by_segments
from yeto.rl.engine.timeline import compaction_stats
from yeto.rl.wandb_rl import event_metrics


def _miles(groups):
    """Converted groups as Miles-like samples (attribute access, as the hooks see them)."""
    converted, problems = br.miles_groups(list(enumerate(groups)), samples_per_group=N)
    assert problems == []
    return [[SimpleNamespace(**{**d, "status": SimpleNamespace(value=d["status"]), "remove_sample": False,
                                "weight_versions": None}) for d in g] for _, g in converted]


def _metadata(monkeypatch, data):
    monkeypatch.setattr(hook, "current_round_id", lambda *a, **kw: 3)
    args = SimpleNamespace()
    hook.record_trained_groups(args, data)
    return hook.build_metadata(args, data)


def test_build_metadata_reports_the_trained_compaction_and_nothing_without(monkeypatch):
    data = _miles([group(0, (1.0, 0.0, 0.0, 1.0), (1, 3, 1, 2)), group(1, (1.0, 1.0, 0.0, 0.5), (1, 1, 1, 1))])
    meta = _metadata(monkeypatch, data)
    stats = meta["compaction"]
    assert stats["trials"] == 8 and stats["samples"] == 11 and stats["compacted_trials"] == 2
    assert stats == compaction_stats([s for g in data for s in g])
    plain = _miles([group(0, (1.0, 0.0, 0.0, 1.0), (1, 1, 1, 1))])
    assert "compaction" not in _metadata(monkeypatch, plain)


def _payload(**extra):
    return {"schema": hook.METADATA_SCHEMA, "rollout_id": 3, "completed": 1, "filtered": 0, "aborted": 0,
            "groups": [{"group_id": "g0", "sample_ids": ["s0", "s0", "s1", "s2"], "policy_token": "yeto:3:" + "a" * 64,
                        "reward_mean": 0.5, "reward_std": 0.5, "token_count": 12}], **extra}


def _handle(payload):
    return handle_from_metadata(payload, rollout_id=3, policy_version=3, policy_hash="a" * 64, data_pack=None)


def test_the_batch_handle_driver_and_wandb_carry_the_compaction_summary(monkeypatch):
    data = _miles([group(0, (1.0, 0.0, 0.0, 1.0), (1, 3, 1, 2))])
    stats = compaction_stats([s for g in data for s in g])
    batch = _handle(_payload(compaction=stats, buffer_stale_by_segments={"2": 3, "unknown": 4}))
    assert batch.compaction == stats and batch.buffer_stale_by_segments == {"2": 3, "unknown": 4}
    driver = SimpleNamespace(trainer=SimpleNamespace())
    fields = IslandDriver._train_fields(driver, batch, SimpleNamespace(train_step=7), 1.0)
    assert fields["compaction"] == stats
    metrics = event_metrics({"event": "rl_round_trained", "rollout_id": 3, **fields})
    assert metrics["train/compaction/trials"] == 4 and metrics["train/compaction/samples"] == 7
    assert metrics["train/compaction/samples_per_trial"] == pytest.approx(7 / 4)
    assert _stale_by_segments(batch) == {"stale_by_segments": {"2": 3, "unknown": 4}}
    # without compaction: no field anywhere
    plain = _handle(_payload())
    assert plain.compaction is None and plain.buffer_stale_by_segments is None
    assert "compaction" not in IslandDriver._train_fields(driver, plain, SimpleNamespace(train_step=7), 1.0)
    assert _stale_by_segments(plain) == {}


def test_the_reader_counts_stale_trials_by_segments(tmp_path):
    # g1 overflows round 0 and is carried over; at version 9 it has aged out (decoded:
    # its trials are bucketed exactly) and g2 is passed over unread (its trials unknown)
    _write(tmp_path, [group(0, (1.0, 0.0, 1.0), (3, 1, 2)), group(1, (1.0, 0.0), (4, 4)),
                      group(2, (1.0, 0.0, 0.0, 1.0), (1, 1, 1, 1))])
    reader = _reader(tmp_path)
    taken, report = reader.take(0, 1, current_version=6)
    assert [p for p, _ in taken] == [0] and report.carried_over_groups == 1
    assert report.stale_by_segments == {} and not report.stale_compacted
    br.GroupWriter(_buffer(tmp_path), "s1").write(group(3, (1.0, 0.0, 0.0, 1.0), (2, 1, 1, 1), version=9),
                                                   group_key="g3")
    taken, report = reader.take(1, 1, current_version=9)
    assert [p for p, _ in taken] == [3] and report.stale_groups == 2
    assert report.stale_by_segments == {"4+": 2, "unknown": N} and report.stale_compacted
    assert br.stale_segment_buckets(report) == {"4+": 2, "unknown": N}


def test_stale_buckets_are_reported_only_with_compaction():
    report = br.TakeReport(rollout_id=1, groups=1, stale_groups=2, start=0, end=2,
                           stale_by_segments={"unknown": 8})
    assert br.stale_segment_buckets(report) == {}  # an uncompacted run: the old record
    assert br.stale_segment_buckets(report, compacted=True) == {"unknown": 8}
    pruned = [group(0, (1.0, 0.0), (2, 1))]
    assert br.stale_segment_buckets(report, pruned) == {"1": 1, "2": 1, "unknown": 8}


def test_the_reader_round_reports_stale_buckets_and_xhw_metrics_show(monkeypatch):
    monkeypatch.setenv("YETO_RL_XHW_GUARD", "action=downweight")
    calls = []
    plugin = _import_plugin_with_stub_miles(monkeypatch, {})
    monkeypatch.setattr(plugin, "record_round_metadata", lambda args, rid, **counts: calls.append(counts))
    monkeypatch.setattr(plugin, "Sample", SimpleNamespace(from_dict=lambda d: d))
    monkeypatch.setattr(plugin, "RolloutFnTrainOutput", lambda **kw: kw)
    monkeypatch.setattr(plugin, "offpolicy_source_event", lambda *a, **kw: {})
    monkeypatch.setattr(plugin, "flat_metrics", lambda source: {})
    g0 = group(0, (1.0, 0.0, 0.0, 0.0), (1, 3, 1, 1))
    for d in g0:
        d["metadata"]["provenance"] = {"sampler": "a"}
    groups = [(0, g0), (1, group(1, (1.0, 1.0, 0.0, 0.0), (1, 1, 1, 1)))]
    report = br.TakeReport(rollout_id=2, groups=2, stale_groups=1, start=0, end=2,
                           position_sources={0: "src-a"}, stale_by_segments={"3": 2})
    reader = plugin._BufferRound.__new__(plugin._BufferRound)
    reader.args = SimpleNamespace(rollout_sample_filter_path=None, rollout_all_samples_process_path=None,
                                  use_dynamic_batch_size=True, seq_length=64, grpo_std_normalization=False)
    reader.groups, reader.samples_per_group = 2, N
    reader.reader = SimpleNamespace(take=lambda rid, n, current_version: (groups, report),
                                    xhw_actions={"src-a": {"action": "downweight", "weight": 0.5}})
    out = reader(SimpleNamespace(evaluation=False, rollout_id=2, weight_version=6))
    assert calls[0]["buffer_stale_by_segments"] == {"3": 2}
    trained = [d for g in out["samples"] for d in g]
    stats = compaction_stats(trained)
    assert stats["xhw_downweighted_groups"] == 1 and stats["xhw_downweighted_samples"] == 6
    assert stats["xhw_trial_mean_shift"] is not None


def test_group_record_reward_counts_trials_and_equals_main_without_compaction():
    def samples(dicts):
        return [SimpleNamespace(**{**d, "weight_versions": [], "effective_response_length": d["response_length"]})
                for d in dicts]

    converted, _ = br.miles_groups([(0, group(0, (1.0, 0.0, 0.0, 0.0), (1, 3, 3, 3)))], samples_per_group=N)
    record = hook.group_record(SimpleNamespace(), samples(converted[0][1]))
    assert record["reward_mean"] == pytest.approx(0.25) and record["reward_std"] == pytest.approx(0.4330127)
    plain, _ = br.miles_groups([(1, group(1, (1.0, 0.0, 0.0, 0.5), (1, 1, 1, 1)))], samples_per_group=N)
    import statistics

    rewards = [1.0, 0.0, 0.0, 0.5]
    record = hook.group_record(SimpleNamespace(), samples(plain[0][1]))
    assert (record["reward_mean"], record["reward_std"]) == (statistics.fmean(rewards), statistics.pstdev(rewards))


def test_the_compaction_fields_travel_the_metadata_sink(tmp_path, monkeypatch):
    """The real hook path: the all-samples hook's payload and the reader's round record,
    merged as the driver reads them, give the batch handle both fields."""
    import json

    from yeto.rl.adapters.miles.rollout import merge_round_metadata

    data = _miles([group(0, (1.0, 0.0, 0.0, 1.0), (1, 3, 1, 2))])
    args = SimpleNamespace()
    hook.record_trained_groups(args, data)
    monkeypatch.setenv(hook.META_SINK_ENV, f"dir:{tmp_path}")
    monkeypatch.setattr(hook, "current_round_id", lambda *a, **kw: 3)
    hook.extract_rollout_metadata(args, data, SimpleNamespace(sample_group_index=6))
    payload = json.loads((tmp_path / "rollout-3.json").read_text())
    hook.record_round_metadata(None, 3, sink=f"dir:{tmp_path / 'r'}", buffer_stale_by_segments={"3": 1})
    record = json.loads(next((tmp_path / "r").iterdir()).read_text())
    batch = _handle(merge_round_metadata(payload, [record], 3))
    assert batch.compaction["compacted_trials"] == 2 and batch.buffer_stale_by_segments == {"3": 1}
    fields = IslandDriver._train_fields(SimpleNamespace(trainer=SimpleNamespace()), batch,
                                        SimpleNamespace(train_step=1), 1.0)
    assert event_metrics({"event": "rl_round_trained", "rollout_id": 3, **fields})[
        "train/compaction/compacted_trials"] == 2
