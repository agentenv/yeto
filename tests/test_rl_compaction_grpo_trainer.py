"""Trainer side of async GRPO with compacted trials (compaction-1118 C3).

A compacted trial reaches the sample buffer as several segment samples of one group
(same ``group_index`` and ``index``, the trial's reward, ``metadata.compaction``). The
reader counts trials, not samples; the conversion gives the segments of a trial one
Miles ``rollout_id``; Miles then centers one reward per trial and keeps a trial's
segments in one training step. A group without segments converts exactly as before.

The Miles side is checked two ways:

* :class:`MilesRef` copies the Miles functions involved line for line (fork commit
  ``MILES_NEXT_COMMIT`` c351a33c, ``miles/rollout/reward_shaping.py`` L34-69,
  ``miles/ray/rollout/train_data_conversion.py`` L125 and L365-423 and L466-479,
  ``miles/ray/rollout/rollout_data_conversion.py`` L12-47, ``miles/utils/dp_schedule.py``
  L79-111);
* with ``YETO_MILES_REPO`` set (a git checkout of the Miles fork that holds the pinned
  commit), the pinned source itself runs in a subprocess (``git archive`` of the
  commit; the real ``Sample``, ``reward_shaping`` and ``seqlen_balancing`` modules,
  the other functions compiled from the pinned files) and must give the same numbers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import io
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles.pins import MILES_NEXT_COMMIT
from yeto.rl.engine import buffer_payload as bp
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer

N = 4  # n_samples_per_prompt: trials per group


# ---------------------------------------------------------------------- samples (contract 2)
def trial(group: int, index: int, reward: float, segments: int, *, version: int = 5,
          prompt: int = 3, top_k: int = 0, layers: int = 0, truncated: bool = False) -> list[dict]:
    """One trial as its buffer samples: one sample without compaction, else ``segments``
    segment samples (``metadata.compaction``); the response grows by one token per
    segment so every segment has its own lengths."""
    out = []
    for s in range(segments):
        response = 2 + s
        tokens = list(range(1, prompt + response + 1))
        meta = {"prompt_id": f"p{group}", "trajectory_id": f"traj-{group}-{index}",
                "weight_version": str(version + s), "final_weight_version": str(version + s),
                "finish_reason": "stop", "prompt_tokens_known": True}
        record = {"group_index": group, "index": index, "prompt": "p", "response": "", "label": None,
                  "tokens": tokens, "response_length": response,
                  "rollout_log_probs": [-0.1 * (t + 1) for t in range(response)],
                  "loss_mask": [1] * response, "reward": float(reward), "status": "completed",
                  "metadata": meta}
        if segments > 1:
            meta["compaction"] = {"schema": 1, "trajectory_id": meta["trajectory_id"], "segment_index": s,
                                  "num_segments": segments, "context_budget": 64, "summary_ok": True,
                                  "kept_steps": 2, "segment_tokens": response,
                                  "tokens_after": sum(2 + k for k in range(s + 1, segments))}
        if top_k:
            record[bp.ROLLOUT_TOP_LOGPROBS_IDS] = numpy.arange(response * top_k, dtype=numpy.int32).reshape(
                response, top_k)
            record[bp.ROLLOUT_TOP_LOGPROBS] = numpy.full((response, top_k), -1.5, dtype=numpy.float32)
        if layers:
            record[bp.ROLLOUT_ROUTED_EXPERTS] = numpy.full((len(tokens) - 1, layers, 2), 7, dtype=numpy.uint16)
        out.append(record)
    if truncated:
        out[-1]["status"] = "truncated"
    return out


def group(g: int, rewards, segments, **kwargs) -> list[dict]:
    return [d for j, (r, k) in enumerate(zip(rewards, segments, strict=True))
            for d in trial(g, g * N + j, r, k, **kwargs)]


def _buffer(root) -> FileSampleBuffer:
    return FileSampleBuffer(CheckpointStore(root), batch_size=2)


def _reader(root, **kwargs) -> br.BatchReader:
    return br.BatchReader(_buffer(root), max_policy_age=2, wait_s=0.0, poll_s=0.0, group_size=N, **kwargs)


# ---------------------------------------------------------------------- Miles, copied
class MilesRef:
    """Miles fork c351a33c (MILES_NEXT_COMMIT), copied line for line on
    ``SimpleNamespace`` samples (module doc for the file and line ranges)."""

    @staticmethod
    def reward_group_segments(args, samples, prompt_group_sizes=None):  # reward_shaping.py L34-59
        if prompt_group_sizes is not None:
            groups, start = [], 0
            for size in prompt_group_sizes:
                if size > 0:
                    groups.append(list(range(start, start + size)))
                start += size
            return groups
        if all(sample.group_index is not None for sample in samples):
            by_prompt: dict[int, list[int]] = {}
            for row, sample in enumerate(samples):
                by_prompt.setdefault(int(sample.group_index), []).append(row)
            return list(by_prompt.values())
        if len(samples) == args.n_samples_per_prompt * args.rollout_batch_size:
            return [list(range(start, start + args.n_samples_per_prompt))
                    for start in range(0, len(samples), args.n_samples_per_prompt)]
        return [list(range(len(samples)))]

    @staticmethod
    def rollout_segments(samples, rows):  # reward_shaping.py L62-69
        by_rollout: dict = {}
        for row in rows:
            sample = samples[row]
            key = sample.rollout_id if sample.rollout_id is not None else sample.index
            by_rollout.setdefault(key if key is not None else ("row", row), []).append(row)
        return list(by_rollout.values())

    @classmethod
    def normalize(cls, args, samples, raw_rewards):  # train_data_conversion.py L375-423
        import torch

        if not samples:
            return []
        normalized_rewards = torch.empty(len(raw_rewards), dtype=torch.float)
        for prompt_segments in cls.reward_group_segments(args, samples, None):
            rollout_segment_groups = cls.rollout_segments(samples, prompt_segments)
            shared_rewards = []
            for sibling_rows in rollout_segment_groups:
                sibling_rewards = [raw_rewards[i] for i in sibling_rows]
                if any(reward != sibling_rewards[0] for reward in sibling_rewards[1:]):
                    raise ValueError("all samples in a rollout must share one reward")
                shared_rewards.append(sibling_rewards[0])
            rollout_rewards = torch.tensor(shared_rewards, dtype=torch.float)
            normalized_rollout_rewards = rollout_rewards - rollout_rewards.mean()
            if (args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization
                    and len(rollout_rewards) > 1):
                rollout_std = rollout_rewards.std()
                if rollout_std > 0:
                    normalized_rollout_rewards = normalized_rollout_rewards / (rollout_std + 1e-6)
            for sibling_rows, value in zip(rollout_segment_groups, normalized_rollout_rewards.tolist(),
                                           strict=True):
                for i in sibling_rows:
                    normalized_rewards[i] = value
        return normalized_rewards.tolist()

    @staticmethod
    def rollout_ids(samples):  # train_data_conversion.py L125
        return [s.rollout_id if s.rollout_id is not None else s.index for s in samples]

    @staticmethod
    def rollout_mask_sums(rollout_ids, loss_masks):  # train_data_conversion.py L365-372
        totals: dict = {}
        for rid, mask in zip(rollout_ids, loss_masks, strict=True):
            totals[rid] = totals.get(rid, 0) + sum(mask)
        return [totals[rid] for rid in rollout_ids]

    @staticmethod
    def kept_after_trim(samples, global_batch_size):  # rollout_data_conversion.py L24-44
        if any(s.rollout_id is not None for s in samples):
            return len(samples)  # compact: not trimmed by sample count
        return (len(samples) // global_batch_size) * global_batch_size

    @staticmethod
    def can_precompute(rollout_ids, global_batch_size):  # train_data_conversion.py L478-479
        return len(set(rollout_ids)) >= global_batch_size

    @staticmethod
    def steps(rollout_ids, global_batch_size):  # dp_schedule.py L79-111: the samples of each step
        by_rollout: dict = {}
        for pos, rid in enumerate(rollout_ids):
            by_rollout.setdefault(rid, []).append(pos)
        order = list(by_rollout)
        out = []
        for start in range(0, (len(order) // global_batch_size) * global_batch_size, global_batch_size):
            out.append([pos for rid in order[start:start + global_batch_size] for pos in by_rollout[rid]])
        return out


def _ns(d: dict) -> SimpleNamespace:
    """The fields Miles reads of a converted sample (``Sample.from_dict`` defaults)."""
    return SimpleNamespace(group_index=d.get("group_index"), index=d.get("index"),
                           rollout_id=d.get("rollout_id"), reward=d.get("reward"),
                           loss_mask=d.get("loss_mask") or [1] * int(d["response_length"]),
                           status=d.get("status"), metadata=d.get("metadata") or {},
                           remove_sample=bool(d.get("remove_sample", False)),
                           response_length=int(d["response_length"]))


def _args(std: bool) -> SimpleNamespace:
    return SimpleNamespace(advantage_estimator="grpo", grpo_std_normalization=std, n_samples_per_prompt=N,
                           rollout_batch_size=2, reward_key=None, rewards_normalization=True, multi_lora=False)


def _trial_grpo(rewards, std):
    import torch

    t = torch.tensor(rewards, dtype=torch.float)
    out = t - t.mean()
    if std and len(rewards) > 1 and t.std() > 0:
        out = out / (t.std() + 1e-6)
    return out.tolist()


# ---------------------------------------------------------------------- the round
REWARDS_A, SEGMENTS_A = (1.0, 0.0, 0.0, 1.0), (1, 3, 1, 2)  # 4 trials, 7 samples
REWARDS_B, SEGMENTS_B = (1.0, 1.0, 0.0, 0.5), (1, 1, 1, 1)  # no compaction


def _write(root, groups) -> None:
    writer = br.GroupWriter(_buffer(root), "s0")
    for g, samples in enumerate(groups):
        writer.write(samples, group_key=f"g{g}")


def _round(tmp_path, groups=None):
    """Write two groups, read a round of two, convert it: (taken, report, converted)."""
    groups = groups or [group(0, REWARDS_A, SEGMENTS_A), group(1, REWARDS_B, SEGMENTS_B)]
    _write(tmp_path, groups)
    taken, report = _reader(tmp_path).take(0, 2, current_version=6)
    converted, problems = br.miles_groups(taken, samples_per_group=N)
    assert problems == []
    return taken, report, converted


@pytest.mark.parametrize("std", [True, False])
def test_segments_of_a_trial_share_one_advantage_from_trial_level_grpo(tmp_path, std):
    taken, report, converted = _round(tmp_path)
    assert [len(g) for _, g in taken] == [7, 4]
    assert (report.incomplete_groups, report.dropped_small_groups, report.padded_samples) == (0, 0, 0)
    samples = [_ns(d) for _, g in converted for d in g]
    out = MilesRef.normalize(_args(std), samples, [s.reward for s in samples])
    rows_a = [s for s in samples if s.group_index == converted[0][0]]
    expected_a = _trial_grpo(list(REWARDS_A), std)
    trial_of = [t for t, k in enumerate(SEGMENTS_A) for _ in range(k)]
    assert out[:7] == pytest.approx([expected_a[t] for t in trial_of], abs=0)
    # the segments of a trial: one rollout, one advantage
    for t in range(N):
        mine = {out[i] for i, tt in enumerate(trial_of) if tt == t}
        assert len(mine) == 1
    assert len({s.rollout_id for s in rows_a}) == N and {s.index for s in rows_a} == {s.rollout_id for s in rows_a}
    # the uncompacted group: per-sample GRPO, as on main
    assert out[7:] == pytest.approx(_trial_grpo(list(REWARDS_B), std), abs=0)


def test_an_uncompacted_group_converts_exactly_as_on_main(tmp_path):
    raw = group(1, REWARDS_B, SEGMENTS_B)
    out = br.renumber_group(raw, position=9, samples_per_group=N)
    # main's renumber_group: index = position * n + j, the sampler's identity kept, nothing else
    expected = []
    for j, d in enumerate(raw):
        d = dict(d)
        d["metadata"] = {**d["metadata"], "yeto_sampler_identity": {"group_index": d["group_index"],
                                                                    "index": d["index"]}}
        d.update(group_index=9, index=9 * N + j)
        expected.append(d)
    assert bp.encode_group(out) == bp.encode_group(expected)
    assert all("rollout_id" not in d for d in out) and not bp.has_siblings(raw)
    assert bp.trial_count(raw) == len(raw)
    # a group of distinct-index samples without compaction keeps the sample count rule
    with pytest.raises(bp.SampleBufferRolloutError, match="5 samples > samples per group 4"):
        br.renumber_group(raw + [dict(raw[0], index=99)], position=9, samples_per_group=N)


def test_rollout_id_siblings_without_compaction_keep_mains_numbering():
    """Only metadata.compaction makes a group a group of trials: a Miles sampler's
    rollout_id siblings keep main's per-sample numbering and size rule."""
    raw = [dict(d, rollout_id=7) for d in group(1, (1.0, 1.0), (1, 1))]
    assert not bp.has_siblings(raw) and bp.trial_count(raw) == 2
    out = br.renumber_group(raw, position=2, samples_per_group=N)
    assert [d["index"] for d in out] == [8, 9] and [d["rollout_id"] for d in out] == [7, 7]
    assert out[0]["metadata"]["yeto_sampler_identity"] == {"group_index": 1, "index": 4}


def test_the_reader_counts_trials_not_samples(tmp_path):
    groups = [group(0, REWARDS_A, SEGMENTS_A),                      # 4 trials / 7 samples: trained
              group(1, (1.0, 0.0, 1.0, 0.0, 1.0), (2, 1, 1, 1, 1)),  # 5 trials: more than n, dropped
              group(2, (1.0,), (3,)),                               # 1 trial / 3 samples: no spread
              group(3, (0.0, 1.0, 1.0), (2, 2, 1))]                 # 3 trials / 5 samples: short
    _write(tmp_path, groups)
    taken, report = _reader(tmp_path).take(0, 2, current_version=6)
    assert [pos for pos, _ in taken] == [0, 3]
    assert (report.dropped_small_groups, report.incomplete_groups, report.padded_samples) == (2, 1, 1)
    assert [bp.trial_count(g) for _, g in taken] == [4, 3]


def test_a_round_is_topped_up_and_carried_over_by_trials(tmp_path):
    groups = [group(0, (1.0, 0.0, 1.0), (3, 1, 2)),      # 3 trials: 1 missing
              group(1, (1.0, 0.0), (4, 4)),              # 2 trials: would overflow, carried over
              group(2, (1.0, 0.0, 0.0, 1.0), (1, 1, 1, 1))]
    _write(tmp_path, groups)
    reader = _reader(tmp_path)
    taken, report = reader.take(0, 1, current_version=6)
    assert [pos for pos, _ in taken] == [0]
    assert (report.carried_over_groups, report.padded_samples, report.end) == (1, 1, 1)
    taken, report = reader.take(1, 2, current_version=6)
    assert [pos for pos, _ in taken] == [1, 2] and report.padded_samples == 2


def test_the_reader_round_pads_missing_trials_one_rollout_each(monkeypatch, capsys):
    from test_rl_buffer_rollout import _import_plugin_with_stub_miles

    calls = []
    registry = {"all": lambda args, samples, source: calls.append(sum(len(g) for g in samples))}
    plugin = _import_plugin_with_stub_miles(monkeypatch, registry)
    monkeypatch.setattr(plugin, "record_round_metadata", lambda args, rid, **counts: calls.append(counts))
    monkeypatch.setattr(plugin, "Sample", SimpleNamespace(from_dict=lambda d: d))
    monkeypatch.setattr(plugin, "RolloutFnTrainOutput", lambda **kw: kw)
    monkeypatch.setattr(plugin, "offpolicy_source_event", lambda *a, **kw: {})
    monkeypatch.setattr(plugin, "flat_metrics", lambda source: {})
    groups = [(0, group(0, REWARDS_A, SEGMENTS_A)), (1, group(1, (0.0, 1.0, 1.0), (2, 2, 1)))]
    report = br.TakeReport(rollout_id=2, groups=2, stale_groups=0, start=0, end=2, incomplete_groups=1)
    reader = plugin._BufferRound.__new__(plugin._BufferRound)
    reader.args = SimpleNamespace(rollout_sample_filter_path=None, rollout_all_samples_process_path="all",
                                  use_dynamic_batch_size=True, seq_length=64)
    reader.groups, reader.samples_per_group = 2, N
    reader.reader = SimpleNamespace(take=lambda rid, n, current_version: (groups, report))
    out = reader(SimpleNamespace(evaluation=False, rollout_id=2, weight_version=6))
    assert calls[0]["buffer_padded_samples"] == 1 and calls[1] == 12  # the hooks see the 12 real samples
    assert {k: calls[0][k] for k in ("buffer_compacted_groups", "buffer_compacted_trials",
                                     "buffer_compacted_samples")} == {
        "buffer_compacted_groups": 2, "buffer_compacted_trials": 4, "buffer_compacted_samples": 3 + 2 + 2 + 2}
    flat = [d for g in out["samples"] for d in g]
    assert [len(g) for g in out["samples"]] == [7, 5, 1]
    assert ("[sample-buffer] compaction_batch: rollout=2 trials=7 segments=12 per_trial={1:3,2:3,3:1}"
            in capsys.readouterr().out)  # the padding is no trial
    rollouts = MilesRef.rollout_ids([_ns(d) for d in flat])
    assert len(set(rollouts)) == 2 * N  # rollout_batch_size * n rollouts, as Miles' global batch counts
    pad = out["samples"][2][0]
    assert pad["remove_sample"] and pad["rollout_id"] is None and "compaction" not in pad["metadata"]
    assert out["metrics"]["sample_buffer/padded_fraction"] == 1 / 8
    # a trainer whose Miles would split the round by sample count refuses it
    reader.args.use_dynamic_batch_size = False
    with pytest.raises(br.SampleBufferRolloutError, match="use-dynamic-batch-size"):
        reader(SimpleNamespace(evaluation=False, rollout_id=3, weight_version=6))
    reader.args.use_dynamic_batch_size, reader.args.indep_dp = True, True
    with pytest.raises(br.SampleBufferRolloutError, match="indep-dp"):
        reader(SimpleNamespace(evaluation=False, rollout_id=3, weight_version=6))
    # a trainer whose --seq-length does not hold a segment of the sampler's budget stops
    reader.args.indep_dp, reader.args.seq_length = False, 32
    with pytest.raises(br.SampleBufferRolloutError, match="context budget 64"):
        reader(SimpleNamespace(evaluation=False, rollout_id=3, weight_version=6))


def test_an_uncompacted_round_reports_no_compaction_keys(monkeypatch, capsys):
    from test_rl_buffer_rollout import _import_plugin_with_stub_miles

    calls = []
    plugin = _import_plugin_with_stub_miles(monkeypatch, {})
    monkeypatch.setattr(plugin, "record_round_metadata", lambda args, rid, **counts: calls.append(counts))
    monkeypatch.setattr(plugin, "Sample", SimpleNamespace(from_dict=lambda d: d))
    monkeypatch.setattr(plugin, "RolloutFnTrainOutput", lambda **kw: kw)
    monkeypatch.setattr(plugin, "offpolicy_source_event", lambda *a, **kw: {})
    monkeypatch.setattr(plugin, "flat_metrics", lambda source: {})
    groups = [(0, group(0, REWARDS_B, SEGMENTS_B))]
    report = br.TakeReport(rollout_id=2, groups=1, stale_groups=0, start=0, end=1)
    reader = plugin._BufferRound.__new__(plugin._BufferRound)
    reader.args = SimpleNamespace(rollout_sample_filter_path=None, rollout_all_samples_process_path=None)
    reader.groups, reader.samples_per_group = 1, N
    reader.reader = SimpleNamespace(take=lambda rid, n, current_version: (groups, report))
    out = reader(SimpleNamespace(evaluation=False, rollout_id=2, weight_version=6))
    assert not any(k.startswith("buffer_compacted") for k in calls[0])
    assert "compaction_batch" not in capsys.readouterr().out
    assert all("rollout_id" not in d for d in out["samples"][0])


@pytest.mark.parametrize("global_batch_size", [8, 4, 2])
def test_every_trial_trains_whole_in_one_step(tmp_path, global_batch_size):
    """Miles' global batch counts rollouts on its rollout-side schedule: with a
    rollout_id on the round nothing is trimmed by sample count, and each step takes
    ``global_batch_size`` whole trials."""
    _taken, _report, converted = _round(tmp_path)
    samples = [_ns(d) for _, g in converted for d in g]
    assert len(samples) == 11
    assert MilesRef.kept_after_trim(samples, global_batch_size) == 11  # 11 % 8 != 0, yet nothing cut
    rollouts = MilesRef.rollout_ids(samples)
    assert MilesRef.can_precompute(rollouts, global_batch_size)
    steps = MilesRef.steps(rollouts, global_batch_size)
    assert len(steps) == 8 // global_batch_size and sorted(p for s in steps for p in s) == list(range(11))
    for rid in set(rollouts):
        rows = {p for p, r in enumerate(rollouts) if r == rid}
        assert sum(1 for s in steps if rows & set(s)) == 1  # one step holds all of a trial's segments
    # the loss reducer's per-rollout token total spans all of a trial's segments
    sums = MilesRef.rollout_mask_sums(rollouts, [s.loss_mask for s in samples])
    assert sums[:7] == [2, 2 + 3 + 4, 2 + 3 + 4, 2 + 3 + 4, 2, 2 + 3, 2 + 3]


def test_without_rollout_ids_miles_would_cut_segments(tmp_path):
    """Why renumber_group sets rollout_id: by index alone the round is not compact, and
    Miles trims it to a multiple of the global batch by sample count."""
    _taken, _report, converted = _round(tmp_path)
    samples = [_ns({**d, "rollout_id": None}) for _, g in converted for d in g]
    assert MilesRef.kept_after_trim(samples, 8) == 8  # 3 of the 11 samples cut


@pytest.mark.parametrize("bad, message", [
    (lambda g: g[1].update(reward=0.5), "do not share one reward"),
    (lambda g: g.pop(2), "num_segments 3 != 2"),
    (lambda g: [d["metadata"]["compaction"].update(trajectory_id="traj-0-1") for d in g[5:7]], "spans sample indexes"),
    (lambda g: g[1]["metadata"].pop("compaction"), "lack compaction metadata"),
    (lambda g: g[2].update(status="truncated"), "has status 'truncated'"),
])
def test_a_malformed_segmented_group_is_left_out(bad, message):
    """The contract check (``contracts.validate_compaction_group``) leaves it out (padded)."""
    g = group(0, REWARDS_A, SEGMENTS_A)
    bad(g)
    converted, problems = br.miles_groups([(3, g)], samples_per_group=N)
    assert converted == [] and len(problems) == 1 and message in problems[0]


def test_score_centering_rows_are_per_segment():
    """Top-128 and R3 stay per sample: each segment converts with its own lengths."""
    g = group(0, REWARDS_A, SEGMENTS_A, top_k=128, layers=3)
    converted, problems = br.miles_groups([(0, g)], samples_per_group=N, r3_layers=(4, 1), r3_num_experts=8)
    assert problems == []
    for d in converted[0][1]:
        n = d["response_length"]
        assert d[bp.MILES_TOPK_TOKEN_IDS].shape == (n, 128) and d[bp.MILES_TOPK_LOG_PROBS].shape == (n, 128)
        assert d[bp.ROLLOUT_ROUTED_EXPERTS].shape == (len(d["tokens"]) - 1, 4, 2)
        assert len(d["rollout_log_probs"]) == n
        spans = d["weight_versions"][0]["spans"]
        assert spans == [{"version": d["metadata"]["weight_version"], "abs_start": len(d["tokens"]) - n,
                          "abs_end": len(d["tokens"])}]


def test_the_yeto_reward_dispatcher_centers_per_trial_like_miles(tmp_path):
    """score_centering takes GRPO advantages (std off); the ports path's reward dispatcher
    (``reward_pipeline.grpo_default``) is Miles' per-rollout normalization."""
    from yeto.rl.algos import reward_pipeline as rp

    _taken, _report, converted = _round(tmp_path)
    samples = [_ns(d) for _, g in converted for d in g]
    rewards = [s.reward for s in samples]
    groups = rp.reward_groups(_args(False), samples)
    ours = rp.grpo_default(_args(False), samples, rewards, groups, {})
    assert ours == MilesRef.normalize(_args(False), samples, rewards)


def test_ledger_group_record_counts_trials():
    from yeto.rl.adapters.miles import rollout_meta_hook as hook

    def as_samples(dicts):
        return [SimpleNamespace(**{**_ns(d).__dict__, "weight_versions": [],
                                   "effective_response_length": d["response_length"]}) for d in dicts]

    converted, _ = br.miles_groups([(0, group(0, (1.0, 0.0, 0.0, 0.0), (1, 3, 3, 3)))], samples_per_group=N)
    record = hook.group_record(SimpleNamespace(), as_samples(converted[0][1]))
    assert record["reward_mean"] == pytest.approx(0.25) and len(set(record["sample_ids"])) == N
    assert len(record["sample_ids"]) == 10 and record["token_count"] == 2 + 3 * (2 + 3 + 4)
    plain, _ = br.miles_groups([(1, group(1, (1.0, 0.0, 0.0, 0.0), (1, 1, 1, 1)))], samples_per_group=N)
    assert hook.group_record(SimpleNamespace(), as_samples(plain[0][1]))["reward_mean"] == pytest.approx(0.25)


def test_segment_counts_follow_the_final_loss_masks():
    from yeto.rl.adapters.miles.rollout_meta_hook import refresh_compaction_counts
    from yeto.rl.compaction import refresh_segment_counts

    converted, _ = br.miles_groups([(0, group(0, REWARDS_A, SEGMENTS_A))], samples_per_group=N)
    samples = [_ns(d) for d in converted[0][1]]
    trial_rows = samples[1:4]  # trial 1: segments of 2, 3, 4 response tokens
    before = trial_rows[0].metadata["compaction"]
    trial_rows[2].loss_mask = [1, 0, 0, 1]  # a later hook masked two tokens of the last segment
    assert refresh_segment_counts(samples) == 0  # the PPO CompactionRL keys: none here
    assert refresh_compaction_counts(samples) == 3  # every segment of trial 1
    assert [s.metadata["compaction"]["tokens_after"] for s in trial_rows] == [3 + 2, 2, 0]
    assert trial_rows[2].metadata["compaction"]["segment_tokens"] == 2
    assert before["tokens_after"] == 3 + 4  # the decoded payload's dict is not touched
    assert all(k not in s.metadata for s in samples for k in ("segment_tokens", "tokens_after", "gae_length"))


def test_xhw_downweight_scales_a_compacted_groups_advantages_exactly(tmp_path):
    """c + w (r - c) with the sample mean c: Miles' per-trial centering cancels c."""
    from yeto.rl.engine.xhw_guard import downweight_group

    g = group(0, (1.0, 0.0, 0.0, 0.0), (1, 3, 1, 1))  # sample mean 1/6, trial mean 1/4
    down = downweight_group(g, 0.25)
    plain, _ = br.miles_groups([(0, g)], samples_per_group=N)
    scaled, _ = br.miles_groups([(0, down)], samples_per_group=N)
    before = MilesRef.normalize(_args(False), [_ns(d) for d in plain[0][1]], [d["reward"] for d in plain[0][1]])
    after = MilesRef.normalize(_args(False), [_ns(d) for d in scaled[0][1]], [d["reward"] for d in scaled[0][1]])
    assert after == pytest.approx([0.25 * a for a in before])


# ---------------------------------------------------------------------- the pinned Miles source
_MILES_SCRIPT = r'''
import __future__, ast, json, logging, os, sys
from types import SimpleNamespace
from typing import Any, NamedTuple
src, path = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
import torch
from miles.utils.types import Sample
from miles.rollout import reward_shaping
from miles.utils import seqlen_balancing


def compiled(rel, names, scope):
    tree = ast.parse(open(os.path.join(src, rel)).read())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert sorted(n.name for n in nodes) == sorted(names), rel
    exec(compile(ast.Module(body=nodes, type_ignores=[]), rel, "exec",
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), scope)
    return scope


common = {"Any": Any, "Sample": Sample, "logger": logging.getLogger("miles"), "torch": torch,
          "NamedTuple": NamedTuple, "TrainParallelConfig": object}
tdc = compiled("miles/ray/rollout/train_data_conversion.py",
               ["_normalize_rewards_by_rollout", "_compute_rollout_mask_sums", "can_precompute_dp_schedule"],
               {**common, "reward_group_segments": reward_shaping.reward_group_segments,
                "rollout_segments": reward_shaping.rollout_segments,
                "is_correct_reward": reward_shaping.is_correct_reward,
                "is_multi_lora_enabled": lambda args: args.multi_lora})
rdc = compiled("miles/ray/rollout/rollout_data_conversion.py",
               ["postprocess_rollout_data", "validate_compact_rollout_ids", "_iter_samples",
                "_compute_dynamic_global_batch_size"],
               {**common, "record_dropped_samples": lambda samples: None,
                "SampleOwnershipRecorder": SimpleNamespace(log_dropped_groups=lambda **kw: None)})
dps = compiled("miles/utils/dp_schedule.py", ["DPSchedule", "build_dp_schedule"],
               {**common, "first_fit_decreasing_pack": seqlen_balancing.first_fit_decreasing_pack,
                "expand_bins_by_splitting": seqlen_balancing.expand_bins_by_splitting,
                "get_seqlen_balanced_partitions": seqlen_balancing.get_seqlen_balanced_partitions})
case = json.load(open(path))
groups = [[Sample.from_dict(d) for d in g] for g in case["groups"]]
out = {}
for gbs in case["global_batch_sizes"]:
    args = SimpleNamespace(advantage_estimator="grpo", grpo_std_normalization=case["std"],
                           n_samples_per_prompt=case["n"], rollout_batch_size=case["groups_per_round"],
                           reward_key=None, rewards_normalization=True, multi_lora=False,
                           disable_rollout_trim_samples=False, use_dynamic_global_batch_size=False,
                           global_batch_size=gbs, use_dynamic_batch_size=True, max_tokens_per_gpu=64,
                           balance_by_flops=False, balance_data=False, allow_partial_train_step=False,
                           micro_batch_size=None)
    config = SimpleNamespace(dp_size=2, cp_size=1, vpp_size=None, microbatch_group_size_per_vp_stage=None,
                             independent_dp=False, supports_precomputed_schedule=True)
    data, _meta = rdc["postprocess_rollout_data"](args, groups, config)
    raw = [s.get_reward_value(args) for s in data]
    rollout_ids = [s.rollout_id if s.rollout_id is not None else s.index for s in data]  # L125
    masks = [s.loss_mask for s in data]
    sched = dps["build_dp_schedule"](args, config, [len(s.tokens) for s in data], global_batch_size=gbs,
                                     rollout_indices=rollout_ids)
    out[str(gbs)] = {
        "kept": len(data),
        "rewards": tdc["_normalize_rewards_by_rollout"](args, data, raw, None),
        "rollout_ids": rollout_ids,
        "mask_sums": tdc["_compute_rollout_mask_sums"](rollout_ids, masks),
        "precompute": tdc["can_precompute_dp_schedule"](args, {"rollout_ids": rollout_ids}, config),
        "num_rollouts": list(sched.num_rollouts),
        "partitions": [list(p) for p in sched.partitions],
        "micro_batch_indices": [[list(m) for m in r] for r in sched.micro_batch_indices],
        "num_microbatches": list(sched.num_microbatches),
    }
print(json.dumps(out))
'''


def _miles_tree(tmp_path) -> Path:
    repo = os.environ.get("YETO_MILES_REPO")
    if not repo or shutil.which("git") is None:
        pytest.skip("YETO_MILES_REPO (a git checkout of the Miles fork) is not set")
    archive = subprocess.run(["git", "-C", repo, "archive", MILES_NEXT_COMMIT, "miles"], capture_output=True)
    if archive.returncode != 0:
        pytest.skip(f"the Miles checkout has no commit {MILES_NEXT_COMMIT[:8]}")
    root = tmp_path / "miles-src"
    root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(root, filter="data")
    return root


def _jsonable(d: dict) -> dict:
    keep = ("group_index", "index", "rollout_id", "tokens", "response_length", "loss_mask", "reward",
            "status", "remove_sample")
    return {k: d[k] for k in keep if k in d} | {"metadata": {}}


@pytest.mark.parametrize("std", [True, False])
def test_the_pinned_miles_source_agrees(tmp_path, std):
    tree = _miles_tree(tmp_path)
    _taken, _report, converted = _round(tmp_path / "buffer")
    groups = [[_jsonable(d) for d in g] for _, g in converted]
    case = {"groups": groups, "std": std, "n": N, "groups_per_round": 2, "global_batch_sizes": [8, 4, 2]}
    (tmp_path / "case.json").write_text(json.dumps(case))
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts" / "miles_side.py"
    script.write_text(_MILES_SCRIPT)
    run = subprocess.run([sys.executable, "-I", str(script), str(tree), str(tmp_path / "case.json")],
                         capture_output=True, text=True, cwd=tmp_path)
    assert run.returncode == 0, run.stderr[-2000:]
    got = json.loads(run.stdout.strip().splitlines()[-1])
    samples = [_ns(d) for g in groups for d in g]
    ref = MilesRef.normalize(_args(std), samples, [s.reward for s in samples])
    rollouts = MilesRef.rollout_ids(samples)
    for gbs in (8, 4, 2):
        miles = got[str(gbs)]
        assert miles["kept"] == 11 and miles["precompute"] is True
        assert miles["rewards"] == ref  # element-wise: the same float32 values
        assert miles["rollout_ids"] == rollouts
        assert miles["mask_sums"] == MilesRef.rollout_mask_sums(rollouts, [s.loss_mask for s in samples])
        assert miles["num_rollouts"] == [gbs] * (8 // gbs)
        assert sorted(p for part in miles["partitions"] for p in part) == list(range(11))
        # Miles' own schedule: the step of every sample, and a trial's segments in one step
        step_of = {}
        for rank, partition in enumerate(miles["partitions"]):
            start = 0
            for step, count in enumerate(miles["num_microbatches"]):
                for micro in miles["micro_batch_indices"][rank][start:start + count]:
                    for local in micro:
                        step_of[partition[local]] = step
                start += count
        assert sorted(step_of) == list(range(11))
        for rid in set(rollouts):
            assert len({step_of[p] for p, r in enumerate(rollouts) if r == rid}) == 1
        assert [sorted(s) for s in MilesRef.steps(rollouts, gbs)] == [
            sorted(p for p, step in step_of.items() if step == k) for k in range(8 // gbs)]
