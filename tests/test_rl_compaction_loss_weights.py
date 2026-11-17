"""Compaction segments under the GRPO loss.

A trial split into ``k`` segment samples (shared reward, siblings of one
rollout) against the same trial as one sample (the segments' responses and
loss masks concatenated): advantages and per-token loss weights must agree
under every aggregation Miles offers, and the filters / shapers must decide
per trial.

Reference implementations follow Miles ``MILES_NEXT_COMMIT`` c351a33c line by
line (CPU, ``cp_size == 1``, one optimizer step, DP 1):

* ``_compute_rollout_mask_sums``: ``miles/ray/rollout/train_data_conversion.py:365-372``;
  ``rollout_ids``: ``:125`` (``rollout_id``, else ``index``);
* ``get_sum_of_sample_mean``: ``miles/backends/training_utils/data/context_parallel.py:247-317``
  (``denominators=rollout_mask_sums``: ``loss/objective.py:226``);
* step normalizer: ``loss/objective.py:255-279`` (per-token: ``/ num_step_tokens``;
  else ``/ num_rollouts``, the rollout-side DP schedule's distinct rollouts);
* GRPO returns: ``loss/hub/math_utils.py:628`` (reward broadcast to every token);
* reward normalization: ``yeto.rl.algos.reward_pipeline.grpo_default``, element-wise
  equal to ``_normalize_rewards_by_rollout`` (``train_data_conversion.py:375-423``;
  ``tests/test_rl_reward_pipeline_equivalence.py``).

With Miles importable (``PYTHONPATH`` with a Miles checkout) the parity tests run
the Miles functions themselves (source taken from the checkout, no Miles runtime).
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from yeto.rl.algos import grpo_knobs as gk  # noqa: E402
from yeto.rl.algos import reward_pipeline as rp  # noqa: E402
from yeto.rl.algos import sample_filters as sf  # noqa: E402
from yeto.rl.algos import segment_weights as sw  # noqa: E402
from yeto.rl.engine.algorithm import AlgorithmSpec, AlgorithmSpecError, load_extensions  # noqa: E402
from yeto.rl.engine.xhw_guard import downweight_group  # noqa: E402
from yeto.rl.rewards.builtin import BoundedNonzeroStdFilter  # noqa: E402
from yeto.rl.rewards.types import Trajectory  # noqa: E402

N = 3  # n_samples_per_prompt
# (group, trial ordinal, reward, segment response lengths); one entry per segment length
TRIALS = [
    (0, 0, 1.0, [37, 21, 44]),  # compacted twice
    (0, 1, 0.0, [58]),
    (0, 2, 0.0, [12]),
    (1, 0, 1.0, [30, 25]),  # compacted once
    (1, 1, 1.0, [70]),
    (1, 2, 0.0, [9]),
]


def _masks(seed=7):
    """Per-segment 0/1 loss masks (tool output = 0), each with a trainable token."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _, _, _, lengths in TRIALS:
        segs = []
        for n in lengths:
            m = (torch.rand(n, generator=g) < 0.7).to(torch.int64)
            m[-1] = 1  # the segment ends on a sampled turn (summary or final answer)
            segs.append(m.tolist())
        out.append(segs)
    return out


MASKS = _masks()


def _meta(tid, s, k, n):
    return {"compaction": {"schema": 1, "trajectory_id": tid, "segment_index": s, "num_segments": k,
                           "context_budget": 4096, "summary_ok": True, "kept_steps": 2,
                           "segment_tokens": n, "tokens_after": 0}}


def build(segmented: bool, *, sibling_ids: bool = True, rewards=None):
    """Samples (SimpleNamespace) in trial order; ``trial_rows[t]`` = rows of trial t."""
    samples, trial_rows = [], []
    for t, ((g, j, reward, _), segs) in enumerate(zip(TRIALS, MASKS)):
        reward = reward if rewards is None else rewards[t]
        parts = segs if segmented else [sum(segs, [])]
        rows = []
        for s, mask in enumerate(parts):
            index = g * N + j
            rollout_id = None if sibling_ids else 1000 * index + s
            meta = _meta(f"traj-{t}", s, len(parts), len(mask)) if len(parts) > 1 else None
            rows.append(len(samples))
            samples.append(SimpleNamespace(group_index=g, index=index, rollout_id=rollout_id, reward=reward,
                                           loss_mask=list(mask), response_length=len(mask), metadata=meta,
                                           status=SimpleNamespace(value="completed"), remove_sample=False))
        trial_rows.append(rows)
    return samples, trial_rows


def _args(std=False):
    return SimpleNamespace(advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=std,
                           n_samples_per_prompt=N, rollout_batch_size=2)


def advantages(samples, std=False):
    """Per-sample normalized reward (Miles ``_normalize_rewards_by_rollout``)."""
    args = _args(std)
    rewards = [s.reward for s in samples]
    return rp.grpo_default(args, samples, rewards, rp.reward_groups(args, samples), {})


# ----------------------------------------------------------- Miles reductions (reference)


def rollout_keys(samples):  # train_data_conversion.py:125
    return [s.rollout_id if s.rollout_id is not None else s.index for s in samples]


def compute_rollout_mask_sums(rollout_ids, loss_masks):  # train_data_conversion.py:365-372
    totals = {}
    for rid, mask in zip(rollout_ids, loss_masks, strict=True):
        totals[rid] = totals.get(rid, 0) + sum(mask)
    return [totals[rid] for rid in rollout_ids]


def sum_of_sample_mean(response_lengths, loss_masks, per_token, denominators=None):
    """context_parallel.py:247-317, cp_size == 1."""
    if denominators is None:
        denominators = [m.sum() for m in loss_masks]

    def sample_mean(x):
        return sum((x_i * m).sum() / torch.clamp_min(d, 1)
                   for x_i, m, d in zip(x.split(response_lengths, dim=0), loss_masks, denominators, strict=True))

    def token_sum(x):
        return sum((x_i * m).sum() for x_i, m in zip(x.split(response_lengths, dim=0), loss_masks, strict=True))

    return token_sum if per_token else sample_mean


def token_weights(samples, aggregation, *, rollout_mask_sums=True, constant=100.0):
    """d(step loss)/d(token loss) per sample: the weight each response token gets.

    aggregation: ``token`` (--calculate-per-token-loss), ``default`` (per-sample
    mean; with Miles' rollout_mask_sums and num_rollouts unless
    ``rollout_mask_sums=False``: the per-sample mean / sample count of a Miles
    without sibling support), ``constant`` (Dr.GRPO reducer, sum / D).
    """
    lengths = [s.response_length for s in samples]
    masks = [torch.tensor(s.loss_mask, dtype=torch.float64) for s in samples]
    keys = rollout_keys(samples)
    x = torch.ones(sum(lengths), dtype=torch.float64, requires_grad=True)
    if aggregation == "token":
        loss = sum_of_sample_mean(lengths, masks, True)(x) / sum(m.sum() for m in masks)  # objective.py:279
    elif aggregation == "default":
        if rollout_mask_sums:
            den = [torch.tensor(float(v), dtype=torch.float64) for v in compute_rollout_mask_sums(keys, masks)]
            loss = sum_of_sample_mean(lengths, masks, False, den)(x) / len(set(keys))  # objective.py:257,270
        else:
            loss = sum_of_sample_mean(lengths, masks, False)(x) / len(samples)
    elif aggregation == "constant":  # yeto.rl.algos.reducers.constant_denominator_reducer
        loss = sum((x_i * m).sum() / constant for x_i, m in zip(x.split(lengths), masks)) / len(set(keys))
    else:
        raise ValueError(aggregation)
    loss.backward()
    return list(x.grad.split(lengths))


def per_trial(values_per_sample, trial_rows):
    """Concatenate per-sample tensors (or broadcast scalars) of each trial in segment order."""
    out = []
    for rows in trial_rows:
        parts = [values_per_sample[r] for r in rows]
        out.append(torch.cat(parts) if isinstance(parts[0], torch.Tensor) else parts)
    return out


def coefficients(samples, aggregation, std=False, **kw):
    """Per-token gradient coefficient A_t * w_t (GRPO returns: get_grpo_returns broadcast)."""
    adv = advantages(samples, std)
    w = token_weights(samples, aggregation, **kw)
    return [a * w_i for a, w_i in zip(adv, w)]


# ----------------------------------------------------------- 1. aggregation


@pytest.mark.parametrize("aggregation", ["token", "default", "constant"])
@pytest.mark.parametrize("std", [False, True])
def test_segmented_equals_unsegmented(aggregation, std):
    seg, seg_rows = build(True)
    whole, whole_rows = build(False)
    a = per_trial(coefficients(seg, aggregation, std), seg_rows)
    b = per_trial(coefficients(whole, aggregation, std), whole_rows)
    for t, (x, y) in enumerate(zip(a, b)):
        assert torch.allclose(x, y, rtol=0, atol=1e-15), (aggregation, t)
    # every segment of a trial carries the trial's advantage
    adv = advantages(seg, std)
    for rows, (whole_adv,) in zip(seg_rows, per_trial(advantages(whole, std), whole_rows)):
        assert {adv[r] for r in rows} == {whole_adv}


def test_sample_mean_without_rollout_sums_weights_trial_by_segment_count():
    """Per-sample mean without Miles' rollout_mask_sums: a k-segment trial weighs k x."""
    seg, seg_rows = build(True)
    whole, whole_rows = build(False)
    w_seg = per_trial(token_weights(seg, "default", rollout_mask_sums=False), seg_rows)
    w_whole = per_trial(token_weights(whole, "default", rollout_mask_sums=False), whole_rows)
    ratios = [float(a.sum() / b.sum()) for a, b in zip(w_seg, w_whole)]
    k = [len(rows) for rows in seg_rows]
    scale = len(whole) / len(seg)  # the step divides by the sample count
    assert ratios == pytest.approx([n * scale for n in k], rel=1e-12)
    # with rollout_mask_sums every trial weighs the same, compacted or not
    totals = [float(w.sum()) for w in per_trial(token_weights(seg, "default"), seg_rows)]
    assert totals == pytest.approx([1 / len(TRIALS)] * len(TRIALS), rel=1e-12)


def test_token_weights_are_segment_free():
    """Per-token: each token weighs 1/N_step, no matter how the trial is split."""
    seg, _ = build(True)
    n_tokens = sum(sum(s.loss_mask) for s in seg)
    for s, w in zip(seg, token_weights(seg, "token")):
        assert torch.allclose(w, torch.tensor(s.loss_mask, dtype=torch.float64) / n_tokens, rtol=0, atol=1e-18)


def test_distinct_rollout_keys_bias_the_baseline():
    """Siblings not recognized (each segment its own rollout): the group mean counts segments."""
    seg, seg_rows = build(True, sibling_ids=False)
    whole, whole_rows = build(False)
    adv = per_trial(advantages(seg), seg_rows)
    ok = per_trial(advantages(whole), whole_rows)
    assert adv[0][0] != ok[0][0]
    # group 0: segment rewards [1,1,1,0,0] -> mean 0.6 (trial mean 1/3)
    assert adv[0][0] == pytest.approx(0.4) and ok[0][0] == pytest.approx(2 / 3)
    assert sw.sibling_problems(seg)  # the trainer-side check refuses this batch


# ----------------------------------------------------------- 2. score centering (Miles)


def _sc_inputs(n_tokens, k=4, vocab=12, seed=3):
    g = torch.Generator().manual_seed(seed)
    train = torch.log_softmax(torch.randn(n_tokens, vocab, generator=g, dtype=torch.float64), -1)
    rollout = torch.log_softmax(train + 0.3 * torch.randn(n_tokens, vocab, generator=g, dtype=torch.float64), -1)
    sampled = torch.multinomial(rollout.exp(), 1, generator=g).squeeze(-1)
    head = rollout.topk(k, dim=-1).indices
    return train, rollout, sampled, head


def test_score_centering_tis_is_token_local_and_segment_invariant():
    est = pytest.importorskip("miles.backends.training_utils.loss.hub.score_centering")
    seg, seg_rows = build(True)
    whole, whole_rows = build(False)
    n = sum(s.response_length for s in whole)
    train, rollout, sampled, head = _sc_inputs(n)

    def losses(samples, trial_rows):
        # token positions of trial t, in sample order -> the same tokens in both layouts
        order, offsets, start = [], [], 0
        for t, rows in enumerate(trial_rows):
            offsets.append(start)
            start += sum(samples[r].response_length for r in rows)
        flat_adv, flat_mask = [], []
        adv = advantages(samples)
        for t, rows in enumerate(trial_rows):
            pos = offsets[t]
            for r in rows:
                length = samples[r].response_length
                order.extend(range(pos, pos + length))
                pos += length
                flat_adv.extend([adv[r]] * length)
                flat_mask.extend(samples[r].loss_mask)
        idx = torch.tensor(order)
        active = torch.tensor(flat_mask, dtype=torch.bool)
        logp = train.gather(-1, sampled[:, None]).squeeze(-1)[idx].clone().requires_grad_(True)
        inputs = est.ScoreCenteringInputs(
            train_log_probs=logp,
            train_head_log_probs=train.gather(-1, head)[idx],
            rollout_log_probs=torch.where(active, rollout.gather(-1, sampled[:, None]).squeeze(-1)[idx], 0.0),
            rollout_head_log_probs=rollout.gather(-1, head)[idx],
            head_mask=torch.ones(len(idx), head.shape[1], dtype=torch.bool) & active[:, None],
            advantages=torch.where(active, torch.tensor(flat_adv, dtype=torch.float64), 0.0),
            mode="tis", tis_clip=2.0)
        token_loss, _ = est.score_centering_loss(inputs)
        lengths = [s.response_length for s in samples]
        masks = [torch.tensor(s.loss_mask, dtype=torch.float64) for s in samples]
        step = sum_of_sample_mean(lengths, masks, True)(token_loss) / sum(m.sum() for m in masks)
        step.backward()
        return token_loss.detach(), logp.grad, step.detach()

    seg_loss, seg_grad, seg_step = losses(seg, seg_rows)
    whole_loss, whole_grad, whole_step = losses(whole, whole_rows)
    assert torch.allclose(seg_loss, whole_loss, rtol=0, atol=1e-15)
    assert torch.allclose(seg_grad, whole_grad, rtol=0, atol=1e-15)
    assert torch.allclose(seg_step, whole_step, rtol=0, atol=1e-15)
    # token-local: one segment alone gives the same per-token loss as inside the batch
    first = seg[0].response_length
    inputs_one = est.ScoreCenteringInputs(
        train_log_probs=train.gather(-1, sampled[:, None]).squeeze(-1)[:first],
        train_head_log_probs=train.gather(-1, head)[:first],
        rollout_log_probs=rollout.gather(-1, sampled[:, None]).squeeze(-1)[:first],
        rollout_head_log_probs=rollout.gather(-1, head)[:first],
        head_mask=torch.ones(first, head.shape[1], dtype=torch.bool),
        advantages=torch.full((first,), 0.5, dtype=torch.float64), mode="tis", tis_clip=2.0)
    inputs_all = est.ScoreCenteringInputs(
        train_log_probs=train.gather(-1, sampled[:, None]).squeeze(-1),
        train_head_log_probs=train.gather(-1, head),
        rollout_log_probs=rollout.gather(-1, sampled[:, None]).squeeze(-1),
        rollout_head_log_probs=rollout.gather(-1, head),
        head_mask=torch.ones(n, head.shape[1], dtype=torch.bool),
        advantages=torch.full((n,), 0.5, dtype=torch.float64), mode="tis", tis_clip=2.0)
    assert torch.equal(est.score_centering_loss(inputs_one)[0], est.score_centering_loss(inputs_all)[0][:first])


# ----------------------------------------------------------- 3. shaping and filters


def _miles_samples(samples):
    types = pytest.importorskip("miles.utils.types")
    out = []
    for s in samples:
        out.append(types.Sample(group_index=s.group_index, index=s.index, rollout_id=s.rollout_id,
                                reward=s.reward, loss_mask=list(s.loss_mask), response_length=s.response_length,
                                tokens=[0] * (s.response_length + 1), metadata=dict(s.metadata or {})))
    return out


def test_miles_length_penalty_sums_siblings():
    rs = pytest.importorskip("miles.rollout.reward_shaping")
    args = SimpleNamespace(length_penalty_coeff=0.5, length_penalty_min_tokens=1, n_samples_per_prompt=N,
                           rollout_batch_size=2)
    for build_kw in ({}, {"rewards": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]}):
        seg, seg_rows = build(True, **build_kw)
        whole, whole_rows = build(False, **build_kw)
        ms, mw = _miles_samples(seg), _miles_samples(whole)
        a = rs.apply_length_penalty(args, ms, [s.reward for s in ms], rs.reward_group_segments(args, ms))
        b = rs.apply_length_penalty(args, mw, [s.reward for s in mw], rs.reward_group_segments(args, mw))
        assert per_trial(a, seg_rows) == [[v[0]] * len(rows) for v, rows in zip(per_trial(b, whole_rows), seg_rows)]


def test_yeto_overlong_penalty_sums_siblings():
    seg, seg_rows = build(True)
    whole, whole_rows = build(False)
    params = {"max_length": 80, "cache_length": 40}
    a = rp.overlong_penalty(None, seg, [s.reward for s in seg], params)
    b = rp.overlong_penalty(None, whole, [s.reward for s in whole], params)
    assert per_trial(a, seg_rows) == [[v[0]] * len(rows) for v, rows in zip(per_trial(b, whole_rows), seg_rows)]


GROUP_REWARDS = {"all_correct": [1.0] * 6, "all_wrong": [0.0] * 6, "mixed": None}


@pytest.mark.parametrize("case", sorted(GROUP_REWARDS))
def test_nonzero_std_filters_decide_per_trial(case):
    rewards = GROUP_REWARDS[case]
    seg, _ = build(True, rewards=rewards)
    whole, _ = build(False, rewards=rewards)
    for g in (0, 1):
        gs = [s for s in seg if s.group_index == g]
        gw = [s for s in whole if s.group_index == g]
        decide = BoundedNonzeroStdFilter()
        a = decide([Trajectory(index=s.index, reward=s.reward) for s in gs], round_id=0, max_replacements=None)
        b = BoundedNonzeroStdFilter()([Trajectory(index=s.index, reward=s.reward) for s in gw], round_id=0,
                                      max_replacements=None)
        assert a.keep == b.keep == (case == "mixed" and len({s.reward for s in gw}) > 1)


@pytest.mark.parametrize("case", sorted(GROUP_REWARDS))
@pytest.mark.parametrize("coeff", [0.0, 0.5])
def test_miles_nonzero_std_filter_decides_per_trial(case, coeff):
    cf = pytest.importorskip("miles.rollout.filter_hub.common_filters")
    args = SimpleNamespace(length_penalty_coeff=coeff, length_penalty_min_tokens=1, reward_key=None)
    rewards = GROUP_REWARDS[case]
    seg, _ = build(True, rewards=rewards)
    whole, _ = build(False, rewards=rewards)
    for g in (0, 1):
        a = cf.apply_reward_nonzero_std_filter(args, _miles_samples([s for s in seg if s.group_index == g]))
        b = cf.apply_reward_nonzero_std_filter(args, _miles_samples([s for s in whole if s.group_index == g]))
        assert bool(a.keep) == bool(b.keep)


def test_xhw_downweight_scales_segment_advantages_exactly():
    seg, seg_rows = build(True)
    whole, whole_rows = build(False)
    weight = 0.25

    def down(samples):
        out = []
        for g in (0, 1):
            rows = [s for s in samples if s.group_index == g]
            dicts = downweight_group([{"reward": s.reward, "metadata": s.metadata} for s in rows], weight)
            for s, d in zip(rows, dicts):
                out.append(SimpleNamespace(**{**vars(s), "reward": d["reward"]}))
        return out

    a = per_trial(advantages(down(seg)), seg_rows)
    b = per_trial(advantages(down(whole)), whole_rows)
    base = per_trial(advantages(whole), whole_rows)
    for x, y, z in zip(a, b, base):
        assert x == pytest.approx([weight * z[0]] * len(x), abs=1e-7) and y == pytest.approx([weight * z[0]], abs=1e-7)
        assert len(set(x)) == 1  # siblings keep one reward


def _status_group(statuses, metas):
    return [SimpleNamespace(index=i, status=SimpleNamespace(value=s), remove_sample=False, metadata=m)
            for i, (s, m) in enumerate(zip(statuses, metas))]


def test_overlong_filter_drops_every_segment_of_a_truncated_trial():
    spec = gk.with_pipeline_plugins(AlgorithmSpec(sampling={"overlong_filter": True}))
    args = SimpleNamespace(**gk.runtime_attrs(spec))
    metas = [_meta("t0", 0, 3, 5), _meta("t0", 1, 3, 5), _meta("t0", 2, 3, 5), None, None,
             _meta("t1", 0, 2, 5), _meta("t1", 1, 2, 5)]
    statuses = ["completed", "completed", "truncated", "truncated", "completed", "completed", "completed"]
    data = [_status_group(statuses, metas)]
    assert sf.apply_sample_filters(args, data) == {"overlong_filter": 4}
    assert [s.remove_sample for s in data[0]] == [True, True, True, True, False, False, False]
    assert sf.group_filtered_samples(data[0]) == 4


def test_overlong_filter_uncompacted_rollout_siblings_unchanged():
    """No compaction metadata: only the truncated sample itself (as without compaction), even if
    two samples share a rollout key."""
    group = _status_group(["completed", "truncated"], [None, None])
    for s in group:
        s.index, s.rollout_id = 5, 5
    assert sf.overlong_filter([group]) == 1
    assert [s.remove_sample for s in group] == [False, True]


# ----------------------------------------------------------- 4./5. checks


def test_sibling_problems_accepts_contract_layout_and_rollout_id_layout():
    seg, _ = build(True)
    assert sw.sibling_problems(seg) == []
    # alternative trainer layout: unique index per row, rollout_id per trial
    for row, s in enumerate(seg):
        s.rollout_id, s.index = s.index, 10_000 + row
    assert sw.sibling_problems(seg) == []
    sw.check_segment_siblings(seg)


def test_sibling_problems_rejects_broken_layouts():
    seg, rows = build(True)
    seg[rows[0][1]].reward = 0.0
    assert any("rewards" in p for p in sw.sibling_problems(seg))
    seg, rows = build(True)
    del seg[rows[0][2]]
    assert any("segment_index" in p for p in sw.sibling_problems(seg))
    seg, rows = build(True)
    for s in seg:  # a renumbering that restarts per group: keys collide across groups
        s.index = s.index % N
    assert any("shared by" in p for p in sw.sibling_problems(seg))
    with pytest.raises(sw.SegmentSiblingError):
        sw.check_segment_siblings(seg)


def test_compaction_algorithm_checks():
    load_extensions()
    from yeto.rl.algos.compactionrl import compactionrl_spec
    from yeto.rl.algos.score_centering import score_centering_spec

    ok = [score_centering_spec(importance_sampling="tis"), AlgorithmSpec(),
          AlgorithmSpec(loss={"aggregation": "token"}),
          AlgorithmSpec(advantage={"estimator": "reinforce_plus_plus", "whiten": True})]
    for spec in ok:
        assert sw.check_compaction_algorithm(spec) == []
    refused = {
        "gspo": AlgorithmSpec(advantage={"estimator": "gspo"}),
        "critic": compactionrl_spec(),
        "gmpo": AlgorithmSpec(loss={"policy_loss_variant": "gmpo"}),
        "OPSM": AlgorithmSpec(correction={"method": "opsm", "opsm_delta": 0.1}),
        "level": AlgorithmSpec(correction={"method": "tis", "ratio_mask": "mis", "level": "sequence",
                                           "upper": 2.0, "lower": 0.0}),
        "reward KL": AlgorithmSpec(advantage={"estimator": "reinforce_plus_plus", "whiten": True},
                                   kl={"placement": "reward", "coef": 0.01}),
    }
    for word, spec in refused.items():
        with pytest.raises(AlgorithmSpecError, match=word):
            sw.check_compaction_algorithm(spec)


# ----------------------------------------------------------- parity with the Miles source


def _miles_functions(relpath, names, namespace):
    spec = importlib.util.find_spec("miles")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("miles not importable")
    path = Path(next(iter(spec.submodule_search_locations))) / relpath
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(n.name for n in nodes) == sorted(names), relpath
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return [namespace[n] for n in names]


def test_parity_with_miles_reward_normalization_and_reducer():
    rs = pytest.importorskip("miles.rollout.reward_shaping")
    from typing import Any, Callable

    ns = {"torch": torch, "Any": Any, "Callable": Callable, "Sample": object,
          "reward_group_segments": rs.reward_group_segments, "rollout_segments": rs.rollout_segments,
          "is_correct_reward": rs.is_correct_reward,
          "get_parallel_state": lambda: SimpleNamespace(cp=SimpleNamespace(size=1))}
    normalize, mask_sums = _miles_functions("ray/rollout/train_data_conversion.py",
                                            ["_normalize_rewards_by_rollout", "_compute_rollout_mask_sums"], ns)
    (miles_reducer,) = _miles_functions("backends/training_utils/data/context_parallel.py",
                                        ["get_sum_of_sample_mean"], ns)
    for sibling_ids in (True, False):
        seg, _ = build(True, sibling_ids=sibling_ids)
        for std in (False, True):
            ours = advantages(seg, std)
            theirs = normalize(_args(std), seg, [s.reward for s in seg], None)
            assert ours == theirs
        keys = rollout_keys(seg)
        masks = [s.loss_mask for s in seg]
        assert mask_sums(keys, masks) == compute_rollout_mask_sums(keys, masks)
        lengths = [s.response_length for s in seg]
        tmasks = [torch.tensor(m, dtype=torch.float64) for m in masks]
        den = [torch.tensor(float(v), dtype=torch.float64) for v in compute_rollout_mask_sums(keys, masks)]
        x = torch.randn(sum(lengths), dtype=torch.float64)
        for per_token in (False, True):
            got = miles_reducer([0] * len(seg), lengths, tmasks, per_token, denominators=den)(x)
            assert torch.equal(got, sum_of_sample_mean(lengths, tmasks, per_token, den)(x))


def test_segment_weight_stats():
    seg, _ = build(True)
    whole, _ = build(False)
    a, b = sw.segment_weight_stats(seg), sw.segment_weight_stats(whole)
    assert a["loss_tokens"] == b["loss_tokens"] and a["trials"] == b["trials"] == len(TRIALS)
    assert a["compacted_trials"] == 2 and b["compacted_trials"] == 0
    tokens = [sum(map(sum, segs)) for segs in MASKS]
    share3 = tokens[0] / sum(tokens)
    assert a["token_share_by_segments"][3] == pytest.approx(share3)
    assert sum(a["token_share_by_segments"].values()) == pytest.approx(1.0)
    assert sum(a["abs_adv_share_by_segments"].values()) == pytest.approx(1.0)
    assert a["summary_tokens_upper"] == sum(MASKS[0][0]) + sum(MASKS[0][1]) + sum(MASKS[3][0])
    seg[0].metadata["compaction"]["summary_ok"] = False  # trial 0 is correct: A > 0
    assert sw.segment_weight_stats(seg)["bad_summary_positive_tokens"] == sum(MASKS[0][0])


def test_xhw_reward_downweight_is_exact_only_without_shapers():
    """The reward-side form r' = c + w (r - c) scales A by w; a shaper added after it is not scaled."""
    whole, _ = build(False)
    weight, params = 0.25, {"max_length": 40, "cache_length": 30}

    def adv(samples, down):
        rows = [SimpleNamespace(**vars(s)) for s in samples]
        if down:
            for g in (0, 1):
                group = [s for s in rows if s.group_index == g]
                for s, d in zip(group, downweight_group([{"reward": s.reward} for s in group], weight)):
                    s.reward = d["reward"]
        shaped = rp.overlong_penalty(None, rows, [s.reward for s in rows], params)
        for s, r in zip(rows, shaped):
            s.reward = r
        return advantages(rows)

    base, down = adv(whole, False), adv(whole, True)
    assert any(abs(d - weight * b) > 1e-3 for d, b in zip(down, base))
