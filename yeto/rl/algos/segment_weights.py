"""How compaction segments enter the GRPO loss (compaction-1118 C15).

With ``agentic.compaction`` on, the records proxy turns one trial into ``k``
segment samples that share the trial's reward (``metadata["compaction"]``,
CONTRACT.zh.md section 2). The trainer (Miles, pin ``MILES_NEXT_COMMIT``)
already treats such samples as *siblings of one rollout*, keyed by
``Sample.rollout_id`` (``None``: ``Sample.index``):

* rewards: ``_normalize_rewards_by_rollout`` centres one reward per rollout
  inside the prompt group (``reward_group_segments`` by ``group_index``,
  ``rollout_segments`` by the rollout key) and broadcasts it to the siblings;
  the yeto dispatcher (``reward_pipeline.grpo_default``, MaxRL / MAPO / GDPO)
  does the same;
* loss aggregation (``loss.aggregation``):

  ``token`` (``--calculate-per-token-loss``; the K3.1 recipe)
      every response token weighs ``1 / N_step`` (the step's loss-token
      count). Splitting a trial changes no token's weight: equivalent to the
      unsplit trial except for the compacted context itself;
  ``default`` (per-sample mean)
      Miles divides each sample's token sum by ``rollout_mask_sums`` (the
      loss-token count of the whole rollout) and the step by ``num_rollouts``
      (rollout-side DP schedule): one token-weighted mean per *rollout*, so
      equivalent as well. Without the shared rollout key (or on a Miles
      without ``rollout_mask_sums``) a trial of ``k`` segments weighs ``k``
      times as much;
  ``constant`` (Dr.GRPO reducer)
      token sum ``/ D``, then ``/ num_rollouts``: per-token, equivalent.

  All three are equivalent *only if* the siblings share the rollout key and
  different trials do not (``rollout_mask_sums`` and the DP schedule key the
  whole batch, not one prompt group): :func:`sibling_problems` checks this
  on the samples the trainer converts.
* score centering: the advantage comes from the GRPO path above
  (``compute_advantages`` -> ``get_grpo_returns``); the loss itself is
  token-local (TIS weight, top-K residual per token) and has no group
  logic of its own.

What is *not* segment-invariant, and refused with compaction on
(:func:`compaction_algorithm_problems`): per-sample *sequence* quantities
(GSPO's sequence ratio, GMPO's geometric mean, OPSM's sequence mask,
sequence-level / geometric MIS, the MIS veto) would be taken per segment
instead of per trial; a critic estimator (PPO / CompactionRL) needs the
cross-segment GAE layout this path does not produce; REINFORCE++ with a
reward-side KL accumulates its KL return per sample, so it would stop at
each segment boundary.

Not a registration module (it is not in ``EXTENSION_MODULES``): compaction is
a sampler-spec property, not part of the ``AlgorithmSpec``, so the launch
check that knows compaction is on calls :func:`check_compaction_algorithm`,
and the trainer-side conversion calls :func:`check_segment_siblings`.

Import-light: no torch / miles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from yeto.rl.engine.algorithm import CRITIC_ESTIMATORS, AlgorithmSpec, AlgorithmSpecError

COMPACTION_KEY = "compaction"  # sample.metadata key written by the records proxy (contract 2)

# Plugin paths whose sibling handling is known (token-local or rollout-aware).
KNOWN_CORRECTION_FUNCTIONS = frozenset({
    "yeto.rl.algos.mismatch_observe.observe_mismatch",
    "miles.backends.training_utils.loss.hub.corrections.icepop_function",
    "yeto.rl.algos.vendor.miles_mis.compute_mis_weights_with_cp",  # level checked below
    "yeto.rl.algos.adaptive_tis.ais_tis_function",
})
KNOWN_REDUCERS = frozenset({"yeto.rl.algos.reducers.constant_denominator_reducer"})
KNOWN_REWARD_POSTPROCESS = frozenset({"yeto.rl.algos.reward_pipeline.post_process"})
SEQUENCE_LEVELS = ("sequence", "geometric")


class SegmentSiblingError(ValueError):
    """Segment samples of one trial are not siblings of one rollout for the trainer."""


# --------------------------------------------------------------------------
# sample access (Miles ``Sample`` objects and buffer sample dicts)
# --------------------------------------------------------------------------


def _get(sample: Any, name: str) -> Any:
    if isinstance(sample, Mapping):
        return sample.get(name)
    return getattr(sample, name, None)


def compaction_meta(sample: Any) -> Mapping[str, Any] | None:
    """``metadata["compaction"]`` of a segment sample (None: not compacted)."""

    metadata = _get(sample, "metadata")
    meta = metadata.get(COMPACTION_KEY) if isinstance(metadata, Mapping) else None
    return meta if isinstance(meta, Mapping) else None


def trajectory_id(sample: Any) -> str | None:
    meta = compaction_meta(sample)
    value = None if meta is None else meta.get("trajectory_id")
    return None if value is None else str(value)


def miles_rollout_key(sample: Any, row: int) -> Any:
    """The key Miles groups siblings by (``reward_shaping.rollout_segments``)."""

    key = _get(sample, "rollout_id")
    if key is None:
        key = _get(sample, "index")
    return ("row", row) if key is None else key


def sibling_key(sample: Any, row: int) -> Any:
    """One value per trial: the compaction trajectory id, else the row itself.

    Uncompacted samples are never merged (keeps their behaviour as on main).
    The overlong filter uses the same key (defined there: its source is the
    filter's plugin identity).
    """

    from .sample_filters import trial_key

    return trial_key(sample, row)


# --------------------------------------------------------------------------
# trainer-side check (C3: Sample conversion)
# --------------------------------------------------------------------------


def sibling_problems(samples: Sequence[Any]) -> list[str]:
    """Why Miles would not see each compacted trial as exactly one rollout.

    ``samples``: the whole training batch (Miles keys ``rollout_mask_sums``
    and the DP schedule over the batch, not per prompt group). Checks, per
    trajectory id: one ``group_index``, one Miles rollout key, one reward, and
    segment indices ``0..num_segments-1`` each once; across the batch: no
    Miles rollout key shared by two trials (compacted or not).
    """

    problems: list[str] = []
    by_trajectory: dict[str, list[int]] = {}
    owners: dict[Any, set[Any]] = {}
    for row, sample in enumerate(samples):
        key = miles_rollout_key(sample, row)
        try:
            hash(key)
        except TypeError:
            problems.append(f"row {row}: rollout key {key!r} is not hashable")
            continue
        owners.setdefault(key, set()).add(sibling_key(sample, row))
        tid = trajectory_id(sample)
        if tid is not None:
            by_trajectory.setdefault(tid, []).append(row)
    for tid, rows in by_trajectory.items():
        segs = [samples[r] for r in rows]
        groups = {_get(s, "group_index") for s in segs}
        keys = {miles_rollout_key(s, r) for s, r in zip(segs, rows)}
        rewards = [_get(s, "reward") for s in segs]
        if len(groups) != 1:
            problems.append(f"trajectory {tid}: segments in groups {sorted(map(str, groups))}")
        if len(keys) != 1:
            problems.append(f"trajectory {tid}: segments have rollout keys {sorted(map(str, keys))}; "
                            "Miles would treat each as its own rollout (set one Sample.rollout_id "
                            "per trial)")
        if any(r != rewards[0] for r in rewards[1:]):
            problems.append(f"trajectory {tid}: segments have rewards {rewards}")
        metas = [compaction_meta(s) or {} for s in segs]
        totals = {m.get("num_segments") for m in metas}
        indices = sorted(m.get("segment_index", -1) for m in metas)
        if len(totals) != 1 or indices != list(range(len(rows))) or totals != {len(rows)}:
            problems.append(f"trajectory {tid}: segment_index {indices} / num_segments "
                            f"{sorted(map(str, totals))} (every segment exactly once)")
    for key, trials in owners.items():
        if len(trials) > 1:
            problems.append(f"rollout key {key!r} is shared by {len(trials)} trials; "
                            "rollout_mask_sums and the DP schedule would merge them")
    return problems


def check_segment_siblings(samples: Sequence[Any]) -> None:
    problems = sibling_problems(samples)
    if problems:
        raise SegmentSiblingError("compaction segments are not rollout siblings: " + "; ".join(problems))


# --------------------------------------------------------------------------
# launch check (C2: check_compaction_launch, compaction on)
# --------------------------------------------------------------------------


def _plugin_path(ref: Any) -> str | None:
    return None if ref is None else getattr(ref, "path", ref)


def compaction_algorithm_problems(spec: AlgorithmSpec) -> list[str]:
    """Algorithm settings whose per-sample math changes when a trial is split."""

    problems = []
    adv, loss, corr = spec.advantage, spec.loss, spec.correction
    if adv.estimator in CRITIC_ESTIMATORS or spec.execution.needs_critic:
        problems.append(f"advantage.estimator={adv.estimator!r}: a critic needs the cross-segment GAE "
                        "layout (CompactionRL), not the shared GRPO advantage of this path")
    if adv.estimator == "gspo":
        problems.append("advantage.estimator='gspo': the sequence ratio would be per segment, not per trial")
    if adv.estimator == "reinforce_plus_plus" and spec.kl.placement == "reward" and (spec.kl.coef or 0) > 0:
        problems.append("advantage.estimator='reinforce_plus_plus' with a reward KL: the KL return stops "
                        "at every segment end")
    if getattr(loss, "policy_loss_variant", "policy_loss") == "gmpo":
        problems.append("loss.policy_loss_variant='gmpo': the geometric mean would be per segment")
    if getattr(loss, "policy_objective", None) is not None:
        problems.append(f"loss.policy_objective={loss.policy_objective!r} (SAO) needs a critic")
    if corr.method == "opsm" or corr.opsm_delta is not None:
        problems.append("correction OPSM: the sequence mask would be per segment")
    levels = {name: getattr(corr, name, None) for name in ("level", "mis_level")}
    for name, level in levels.items():
        if level in SEQUENCE_LEVELS and (corr.method != "none" or corr.ratio_mask != "none"):
            problems.append(f"correction.{name}={level!r}: the sequence weight would be per segment")
    if corr.veto is not None and corr.method != "none":
        problems.append("correction.veto: a vetoed token would drop its segment only, not the trial")
    return problems


def compaction_algorithm_warnings(spec: AlgorithmSpec) -> list[str]:
    """Settings this module cannot vouch for (plugins), or that see summary turns."""

    out = []
    loss, adv, corr = spec.loss, spec.advantage, spec.correction
    if loss.variant == "custom_loss":
        out.append("loss.variant='custom_loss': its reduction over segment siblings is not checked")
    reducer = _plugin_path(loss.reducer)
    if reducer is not None and reducer not in KNOWN_REDUCERS:
        out.append(f"loss.reducer {reducer!r}: a custom pg_loss reducer gets no rollout_mask_sums "
                   "(a per-sample mean would weigh a trial by its segment count)")
    post = _plugin_path(adv.reward_postprocess)
    if post is not None and post not in KNOWN_REWARD_POSTPROCESS:
        out.append(f"advantage.reward_postprocess {post!r}: must give every segment of a trial one "
                   "advantage (Miles rollout_segments)")
    function = _plugin_path(corr.function)
    if function is not None and function not in KNOWN_CORRECTION_FUNCTIONS:
        out.append(f"correction.function {function!r}: segment handling not checked")
    if any(dict(s).get("name") == "overlong_penalty" for s in adv.reward_shapers):
        out.append("overlong_penalty counts the summary turns into the trial length (sum of the "
                   "segments' response_length)")
    return out


def check_compaction_algorithm(spec: AlgorithmSpec) -> list[str]:
    """Refuse :func:`compaction_algorithm_problems`; return the warnings."""

    problems = compaction_algorithm_problems(spec)
    if problems:
        raise AlgorithmSpecError("agentic.compaction is on, but " + "; ".join(problems))
    return compaction_algorithm_warnings(spec)


# --------------------------------------------------------------------------
# monitoring (C10: per training step, before the loss)
# --------------------------------------------------------------------------


def _loss_tokens(sample: Any) -> int:
    mask = _get(sample, "loss_mask")
    if mask is None:
        return int(_get(sample, "response_length") or 0)
    return int(sum(mask))


def segment_weight_stats(samples: Sequence[Any]) -> dict[str, Any]:
    """Where a token-aggregated step's weight goes when trials are compacted.

    Under ``loss.aggregation='token'`` a token weighs ``1 / loss_tokens``, so a
    trial's share of the step is its loss-token share. Reported per step:

    * ``token_share_by_segments`` {k: share of loss tokens in trials of k segments};
    * ``abs_adv_share_by_segments`` {k: share of sum |A| * tokens}, with A the
      group-centered trial reward (GRPO without std normalization; with it only
      the scale differs per group) -- the share of the gradient signal;
    * ``summary_tokens_upper`` (loss tokens of segments that end in a summary,
      i.e. every segment but the last: an upper bound, they also hold actions),
      ``bad_summary_positive_tokens`` (of those, segments with
      ``summary_ok=False`` in trials with A > 0: a malformed summary reinforced).
    """

    total = sum(_loss_tokens(s) for s in samples)
    trials: dict[Any, dict[str, Any]] = {}
    for row, sample in enumerate(samples):
        meta = compaction_meta(sample) or {}
        trial = trials.setdefault(sibling_key(sample, row), {
            "group": _get(sample, "group_index"), "reward": _get(sample, "reward"),
            "k": int(meta.get("num_segments") or 1), "tokens": 0, "summary": 0, "bad_summary": 0})
        n = _loss_tokens(sample)
        trial["tokens"] += n
        if meta and int(meta.get("segment_index", 0)) < int(meta.get("num_segments") or 1) - 1:
            trial["summary"] += n
            if meta.get("summary_ok") is False:
                trial["bad_summary"] += n
    by_group: dict[Any, list[float]] = {}
    for trial in trials.values():
        reward = trial["reward"]
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            by_group.setdefault(trial["group"], []).append(float(reward))
    mean = {g: sum(v) / len(v) for g, v in by_group.items()}
    tok: dict[int, int] = {}
    adv: dict[int, float] = {}
    summary = bad_positive = 0
    for trial in trials.values():
        reward = trial["reward"]
        a = float(reward) - mean[trial["group"]] if trial["group"] in mean and isinstance(
            reward, (int, float)) and not isinstance(reward, bool) else 0.0
        tok[trial["k"]] = tok.get(trial["k"], 0) + trial["tokens"]
        adv[trial["k"]] = adv.get(trial["k"], 0.0) + abs(a) * trial["tokens"]
        summary += trial["summary"]
        if a > 0:
            bad_positive += trial["bad_summary"]
    adv_total = sum(adv.values())
    return {
        "loss_tokens": total,
        "trials": len(trials),
        "compacted_trials": sum(1 for t in trials.values() if t["k"] > 1),
        "token_share_by_segments": {k: v / total if total else 0.0 for k, v in sorted(tok.items())},
        "abs_adv_share_by_segments": {k: v / adv_total if adv_total else 0.0 for k, v in sorted(adv.items())},
        "summary_tokens_upper": summary,
        "bad_summary_positive_tokens": bad_positive,
    }
