"""Miles training and observation options on a verl full-parameter island (S27 G-EC).

The launcher sends these Miles island options to every island.  The verl
FSDP2 full-parameter island maps the ones verl can express; each mapping
applies only when its option is given, so the overrides and plans of every
other run stay unchanged.  The options without a verl mapping stay refused
before any cloud work (``launch_flags.IGNORED_ON_VERL_FULL_PARAMETER`` and
``launch_flags.NOT_MAPPED_REASONS``).

Training options (:func:`apply_training_options`, verl hydra overrides):

* ``--rl-lr-schedule constant``: ``actor.optim.lr_scheduler_type=constant``
  (verl's default; the explicit key is asserted against the resolved config).
  ``linear`` is refused: verl's FSDP2 optimizer offers ``constant`` and
  ``cosine`` only (``verl/workers/config/optimizer.py``), so Miles' linear
  decay has no exact counterpart;
* ``--rl-max-tokens-per-gpu N``: the two dynamic-batching token bounds the
  island already sets (16384 otherwise): ``actor.ppo_max_token_len_per_gpu``
  and ``rollout.log_prob_max_token_len_per_gpu``.  ``N`` must hold one whole
  sample (``max_prompt_len + max_response_len``), the rule of Miles'
  ``--max-tokens-per-gpu >= --seq-len``;
* ``--rl-activation-recompute``: Miles' activation recompute is verl's
  gradient checkpointing, which the island always turns on
  (``config.build_overrides``); the option asserts it.

Observation options (:func:`observation_plan`, plan keys the driver reads in
:mod:`.driver_extras`), the engine driver's own features, as on a Miles
island (``adapters/miles/island_entry.py``):

* ``--rl-observe-timeline``: ``IslandDriver.observe`` (timeline spans, load
  samples), with the heartbeat and resource sampler at their defaults;
* ``--rl-resource-sample-interval S``: the driver's NVML resource sampler
  period (0: off).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

LR_SCHEDULER_KEY = "actor_rollout_ref.actor.optim.lr_scheduler_type"
ACTOR_TOKENS_KEY = "actor_rollout_ref.actor.ppo_max_token_len_per_gpu"
LOG_PROB_TOKENS_KEY = "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu"
CHECKPOINTING_KEY = "actor_rollout_ref.model.enable_gradient_checkpointing"
VERL_LR_SCHEDULES = ("constant",)


def add_island_flags(parser) -> None:
    """The options above, in Miles island spellings.  An island parser that
    already defines one (the verl Megatron island, S27 G-DC) keeps its own."""

    for flag, kwargs in (
        ("--rl-lr-schedule", {"default": None}),
        ("--rl-max-tokens-per-gpu", {"type": int, "default": None}),
        ("--rl-activation-recompute", {"action": "store_true"}),
        ("--rl-observe-timeline", {"action": "store_true"}),
        ("--rl-resource-sample-interval", {"type": float, "default": None}),
        # S27 G-EU (VERL-PARITY #25, #26): the publication checks (verify_plan)
        ("--rl-verify-every-round", {"action": "store_true"}),
        ("--rl-verify-every", {"type": int, "default": None}),
        # S27 G-EU (VERL-PARITY #35): Miles' debug dump (debug_plan)
        ("--rl-debug-dump", {"action": "store_true"}),
    ):
        if flag not in parser._option_string_actions:
            parser.add_argument(flag, **kwargs)


def _replace(overrides: Sequence[str], key: str, value: Any) -> list[str]:
    """``overrides`` with ``key`` set to ``value`` (replaced in place, else appended)."""

    item = f"{key}={value}"
    out, found = [], False
    for override in overrides:
        if override.startswith(key + "="):
            out.append(item)
            found = True
        else:
            out.append(override)
    return out if found else [*out, item]


def apply_training_options(args: Any, overrides: Sequence[str], asserted_keys: Sequence[str], *,
                           max_prompt_len: int, max_response_len: int) -> tuple[list[str], tuple[str, ...]]:
    """``(overrides, asserted keys)`` with the training options of ``args``
    mapped; unchanged when none is given.  Raises ValueError for a value verl
    cannot run."""

    out, asserted = list(overrides), list(asserted_keys)
    schedule = getattr(args, "rl_lr_schedule", None)
    if schedule not in (None, "auto"):
        if schedule not in VERL_LR_SCHEDULES:
            raise ValueError(f"verl 全参数岛只接 --rl-lr-schedule constant（得到 {schedule}）："
                             "verl FSDP2 的学习率调度只有 constant 与 cosine，没有 Miles 的线性衰减")
        out = _replace(out, LR_SCHEDULER_KEY, schedule)
        asserted.append(LR_SCHEDULER_KEY)
    tokens = getattr(args, "rl_max_tokens_per_gpu", None)
    if tokens is not None:
        need = int(max_prompt_len) + int(max_response_len)
        if int(tokens) < need:
            raise ValueError(f"--rl-max-tokens-per-gpu {tokens} 放不下一条样本"
                             f"（max_prompt_len + max_response_len = {need}）")
        for key in (ACTOR_TOKENS_KEY, LOG_PROB_TOKENS_KEY):
            out = _replace(out, key, int(tokens))
            asserted.append(key)
    if getattr(args, "rl_activation_recompute", False):
        out = _replace(out, CHECKPOINTING_KEY, True)
        asserted.append(CHECKPOINTING_KEY)
    return out, tuple(dict.fromkeys(asserted))


def observation_plan(args: Any) -> dict[str, Any]:
    """The plan keys of the observation options (absent when off): ``observe_timeline``
    and the driver's ``heartbeat_interval_s`` / ``resource_sample_interval_s``,
    with Miles' defaults (30 s / 60 s) when the timeline is observed."""

    from yeto.rl.engine.telemetry import DEFAULT_HEARTBEAT_INTERVAL_S, DEFAULT_RESOURCE_INTERVAL_S

    observe = bool(getattr(args, "rl_observe_timeline", False))
    out: dict[str, Any] = {"observe_timeline": True} if observe else {}
    sample = getattr(args, "rl_resource_sample_interval", None)
    if sample is not None and float(sample) < 0:
        raise ValueError("--rl-resource-sample-interval must be >= 0")
    resource = (DEFAULT_RESOURCE_INTERVAL_S if observe else None) if sample is None else float(sample)
    if observe:
        out["heartbeat_interval_s"] = DEFAULT_HEARTBEAT_INTERVAL_S
    if resource:
        out["resource_sample_interval_s"] = resource
    return out


def wire_observation(driver: Any, plan: dict) -> None:
    """The driver side of :func:`observation_plan` (nothing when the plan has none)."""

    if plan.get("observe_timeline"):
        driver.observe = True
    for key in ("heartbeat_interval_s", "resource_sample_interval_s"):
        if plan.get(key):
            setattr(driver, key, float(plan[key]))


# S27 G-EU (VERL-PARITY #25): why a full-parameter verl island refuses --rl-verify-every-round
VERIFY_EVERY_ROUND_REFUSAL = (
    "--rl-verify-every-round: a full-parameter verl publication does not read vLLM's weights back "
    "(VERL-PARITY #25; verl's update_weights / NCCL checkpoint engine report no per-tensor checksum, "
    "and the LoRA read-back hook vllm_readback reads a registered adapter only). Every round already "
    "checks the training/rollout log-prob gap (the criteria); drop the option")


def verify_plan(args: Any, *, full: bool) -> dict[str, Any]:
    """The plan keys of the publication checks (absent when off), as Miles' island
    reads them (``adapters/miles/island_entry.py``, S27 C5):

    * ``--rl-verify-every-round``: a LoRA island verifies every publication anyway
      (the driver hashes the adapter, ``VerlPublisher`` reads vLLM's copy back), so
      the key only records it; a full-parameter island (FSDP2 or Megatron) refuses
      it (:data:`VERIFY_EVERY_ROUND_REFUSAL`);
    * ``--rl-verify-every N`` (a no-sync island, ``run_flags.check_verify_every``):
      a full-parameter island hashes its policy on the first, every N-th, the last
      and every cut round only (``driver.verify_every``, :func:`wire_verify`); a
      LoRA island has nothing to skip (its adapter digest is small and every
      publication is read back), so the option is recorded and a warning printed.
    Raises ValueError on a refused combination."""

    out: dict[str, Any] = {}
    every_round = bool(getattr(args, "rl_verify_every_round", False))
    every = getattr(args, "rl_verify_every", None)
    if every is not None:
        from yeto.rl.engine.run_flags import check_verify_every

        check_verify_every(args)
    if every_round:
        if full:
            raise ValueError(VERIFY_EVERY_ROUND_REFUSAL)
        out["verify_every_round"] = True
    if every is not None:
        if full:
            out["verify_every"] = int(every)
        else:
            note = (f"--rl-verify-every {int(every)}: a verl LoRA island verifies every publication "
                    "(adapter digest + vLLM read-back); nothing is skipped")
            import sys

            print(f"[yeto-verl] WARNING: {note}", file=sys.stderr, flush=True)
            out["verify_every_note"] = note
    return out


def wire_verify(driver: Any, policy_state: Any, plan: dict) -> None:
    """The driver side of :func:`verify_plan` (nothing when the plan has none)."""

    if plan.get("verify_every_round"):
        driver.verify_every_round = True
    every = plan.get("verify_every")
    if every is not None:
        driver.verify_every = int(every)
        allow = getattr(policy_state, "allow_unverified_rounds", None)
        if not callable(allow):
            raise ValueError("--rl-verify-every: this policy state cannot skip its digest")
        allow()


def debug_plan(args: Any) -> dict[str, Any]:
    """S27 G-EU (VERL-PARITY #35): ``--rl-debug-dump`` as on a Miles island, the
    round's rollout samples and train data next to the event tape: the island
    already writes the log-prob tensors of every step (``verl-dump-<id>-stepNNN.pt``,
    the mismatch criteria); with the option every round also writes
    ``verl-debug-<id>-rNNNN.pt`` (uid, prompt and response tokens, reward, advantage
    and the response mask, read from the transfer queue after the advantage step;
    ``ports_impl.debug_dump``). Absent when off."""

    return {"debug_dump": True} if getattr(args, "rl_debug_dump", False) else {}


def wire_debug(island: Any, plan: dict) -> None:
    """The trainer side of :func:`debug_plan` (``VerlTrainerGroup.train_step`` reads it)."""

    island.debug_dump = bool(plan.get("debug_dump"))


def refuse_unmapped_argv(ignored: Sequence[str], *, full: bool) -> None:
    """S27 G-EU: a full-parameter island refuses the Miles options it has no mapping
    for when they reach it anyway (the launcher checks them on a synced full-parameter
    preset only; a no-sync island, e.g. the Megatron island, got them dropped by
    ``parse_known_args``). Raises ValueError with each option's reason."""

    if not full:
        return
    from .launch_flags import NOT_MAPPED_REASONS

    seen = [option for option in NOT_MAPPED_REASONS
            if any(token == option or token.startswith(option + "=") for token in ignored)]
    if seen:
        raise ValueError("verl full-parameter island: no mapping for " + "; ".join(
            f"{option} ({NOT_MAPPED_REASONS[option]})" for option in seen))
