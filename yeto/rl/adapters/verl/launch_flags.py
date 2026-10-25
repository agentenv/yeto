"""verl island-entry spellings the launcher emits (rl-verl-backend; decoupling 5.5 role).

A verl island is one node with any number of cards (S26 E9 G3): the
island-local Ray sees every card of the node and verl's colocated hybrid
engine puts the trainer and the rollout replicas on all of them
(``island_entry.island_gpus``).  A single-node multi-card island therefore
needs no layout flag: the launcher emits a bundle map only for a
fixed-partition placement (``--rl-placement fixed-partition`` with a
``placement`` in the initial config, ``launch.topology.rl_island_layout``),
and verl places its own workers, so it cannot honour a role -> card map.
Such a map and the cross-node switches have no verl spelling and are refused
instead of being silently dropped.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .pins import VERL_COMMIT, VERL_IMAGE

ISLAND_ENTRY_MODULE = "yeto.rl.adapters.verl.island_entry"
NEEDS_MILES_SOURCE = False
DEFAULT_RL_IMAGE = VERL_IMAGE
_IMAGE = re.compile(r"verl-build:(?P<commit>[0-9a-f]{40})")


def rl_image_ok(rl_image: str) -> bool:
    """``verl-build:<pinned commit>`` (Modal-built, see ``image``); other commits are refused."""
    match = _IMAGE.fullmatch(rl_image or "")
    return bool(match) and match["commit"] == VERL_COMMIT


def bundle_map_flag(bundle_map: Mapping[str, Sequence[int]]) -> str:
    """Refuse a role -> card map: verl places its trainer and rollout workers itself.

    Only a fixed-partition placement reaches here (the launcher emits no map
    for a single-node colocated island, whatever its card count)."""
    roles = ", ".join(f"{role}={list(slots)}" for role, slots in sorted(bundle_map.items()))
    raise ValueError(
        "verl 岛是单节点同卡共置（colocated）：训练与 rollout 用本节点全部卡，由 verl 自己放置，"
        f"不接受按角色指定卡号的 bundle map（收到 {roles}）。单节点多卡直接用 --gpu <provider>:<N>x<卡型>，"
        "不要给 fixed-partition 的 placement")


def cross_node_flags(allow_cross_node_tp: bool, allow_cross_node_engine_tp: bool) -> str:
    if allow_cross_node_tp or allow_cross_node_engine_tp:
        raise ValueError("verl 岛只在一个节点内：不支持跨节点 TP")
    return ""


# ----------------------------------------------------------------------------- full-parameter preset
# S27 G-DN: the shape of ``--rl-sync-preset full-parameter`` on verl islands
# (launch.rl_args._prepare_dense_full_sweep asks for it; the Miles backend
# provides none and keeps its fixed-partition shape).  A verl full-parameter
# island is one colocated node: verl's hybrid engine trains and rolls out on
# the same cards, so there is no dedicated rollout partition.  The layout hash
# and fragment count come from ``python -m yeto.rl.adapters.verl.full_parameter_plan``
# (the model's HF config on the meta device), not from the Miles layout probe.
FULL_PARAMETER_PLACEMENT = "colocated"
# The stream refuses peer recovery (full_parameter_stream.refuse_unsupported),
# so the async mode must not turn it on by default (rl_args._peer_recovery_applies).
FULL_PARAMETER_PEER_RECOVERY = False
FULL_PARAMETER_PLAN_COMMAND = (
    "python -m yeto.rl.adapters.verl.full_parameter_plan --model <M> --model-revision <R> "
    "--fragments <P> --matrix-merge <avg|rda>")


def full_parameter_problems(args) -> list[str]:
    """What a verl full-parameter launch must change (empty: the shape is right).

    Replaces the Miles placement rule of the preset (fixed-partition engines);
    every other rule of the preset (layout hash, fragments, pipeline, push
    options) stays the launcher's.  Each refused option is one the verl island
    does not run; the island refuses the same options again before any GPU work."""
    from yeto.rl.engine.sample_roles import is_synced_trainer

    problems = []
    if getattr(args, "rl_placement", "colocated") != FULL_PARAMETER_PLACEMENT:
        problems.append("--rl-placement colocated (a verl island trains and rolls out on the same "
                        "cards; no fixed-partition)")
    if is_synced_trainer(args):
        problems.append("no --rl-sample-role trainer (a verl full-parameter island generates its own samples)")
    if int(getattr(args, "rl_max_policy_age", 0) or 0):
        problems.append("--rl-max-policy-age 0 (the verl full-parameter island runs the synchronous trainer)")
    if int(getattr(args, "rl_rollout_quant_tier", 0) or 0):
        problems.append("no --rl-rollout-quant-tier")
    if int(getattr(args, "expert_full_count", 0) or 0):
        problems.append("no --expert-full-count (DeepSeek V4 expert-full is a Miles recipe)")
    for dest, option in (("rl_sync_overlap", "--rl-sync-overlap"),
                         ("rl_sync_expert_token_weights", "--rl-sync-expert-token-weights"),
                         ("rl_peer_recovery", "--rl-peer-recovery")):
        if getattr(args, dest, False):
            problems.append(f"no {option} (not offered by the verl stream)")
    if not re.fullmatch(r"[0-9a-f]{64}", getattr(args, "rl_parameter_layout_sha256", None) or ""):
        problems.append(f"the verl layout hash and fragment count from `{FULL_PARAMETER_PLAN_COMMAND}`")
    ignored = [option for dest, off, option in IGNORED_ON_VERL_FULL_PARAMETER
               if getattr(args, dest, off) not in (off, None)]
    if ignored:
        problems.append("no " + ", ".join(ignored) + " (the verl full-parameter island has no such option; "
                        "it would ignore it: " + "; ".join(f"{option}: {NOT_MAPPED_REASONS[option]}"
                                                           for option in ignored) + ")")
    schedule = getattr(args, "rl_lr_schedule", None)
    if schedule not in (None, "auto", "constant"):  # S27 G-EC: verl.train_options
        problems.append(f"--rl-lr-schedule constant (got {schedule}; verl's FSDP2 optimizer schedules are "
                        "constant and cosine, not Miles' linear decay)")
    return problems


# S27 G-DS: options the launcher forwards to every island that the verl island
# entry does not parse (parse_known_args would drop them without a word):
# (args dest, value when off, option). Refused instead, before any cloud work.
# S27 G-EC mapped five of the first fourteen (verl.train_options: --rl-lr-schedule
# constant, --rl-max-tokens-per-gpu, --rl-activation-recompute,
# --rl-observe-timeline, --rl-resource-sample-interval); the rest stay here.
IGNORED_ON_VERL_FULL_PARAMETER = (
    ("rl_verify_every_round", False, "--rl-verify-every-round"),
    ("rl_eval_interval", None, "--rl-eval-interval"),
    ("rl_deterministic_trainer", False, "--rl-deterministic-trainer"),
    ("rl_gradient_accumulation_fusion", False, "--rl-gradient-accumulation-fusion"),
    ("rl_async_buffer_rounds", None, "--rl-async-buffer-rounds"),
    ("rl_lora_dropout", None, "--rl-lora-dropout"),
    ("rl_lora_expert_rank", None, "--rl-lora-expert-rank"),
    ("rl_profile_trainer", False, "--rl-profile-trainer"),
    ("rl_debug_dump", False, "--rl-debug-dump"),
)
# Why each option above has no verl mapping yet (the launcher's refusal says it).
NOT_MAPPED_REASONS = {
    "--rl-verify-every-round": "the full-parameter publication does not read vLLM's weights back "
                               "(VERL-PARITY #25)",
    "--rl-eval-interval": "no verl eval port yet (the driver's evaluate hook over verl's _validate is "
                          "not wired or GPU-checked)",
    "--rl-deterministic-trainer": "yeto.rl.engine.determinism has no measured verl row",
    "--rl-gradient-accumulation-fusion": "a Megatron kernel option; FSDP2 has none",
    "--rl-async-buffer-rounds": "needs --rl-max-policy-age > 0, which the full-parameter island refuses",
    "--rl-lora-dropout": "a full-parameter island has no LoRA adapter",
    "--rl-lora-expert-rank": "a full-parameter island has no LoRA adapter",
    "--rl-profile-trainer": "the trainer step runs in verl's Ray workers; the step profiler is not hooked "
                            "there",
    "--rl-debug-dump": "Miles' rollout/train data dump has no verl counterpart (the island writes its own "
                       "verl-dump criteria tensors)",
}
