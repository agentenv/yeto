"""verl island-entry spellings the launcher emits.

A verl island is one node with any number of cards: the
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
from .ray_resources import trainer_cpus as _trainer_cpus

ISLAND_ENTRY_MODULE = "yeto.rl.adapters.verl.island_entry"
# The island Ray's CPU slots the launcher gives a verl island: max(node vCPUs, this)
# (``launch.topology.rl_island_ray_cpus``). verl needs more slots than a small node has
# (TQ storage units, agent / reward workers, the 10-slot trainer actor: ray_resources);
# Ray CPUs are scheduling tokens, so the slots are raised, not the machine.
ISLAND_RAY_MIN_CPUS = 32
# The slots the trainer actor and its task runner hold at once (the dry run's
# ``island_ray`` record; ``ray_resources.trainer_cpus``)
ISLAND_TRAINER_RAY_CPUS = int(_trainer_cpus())
NEEDS_MILES_SOURCE = False
DEFAULT_RL_IMAGE = VERL_IMAGE
# ``verl-megatron-build`` is the same image plus the Megatron engine layers
_IMAGE = re.compile(r"verl(?:-megatron)?-build:(?P<commit>[0-9a-f]{40})")


def rl_image_ok(rl_image: str) -> bool:
    """``verl-build:<pinned commit>`` or ``verl-megatron-build:<pinned commit>``
    (Modal-built, see ``image``), or the pinned registry image of that recipe on any
    registry (``<registry>/yeto-verl-fsdp@sha256:<pinned digest>``, pins); other
    commits are refused."""
    from .pins import is_verl_registry_image

    match = _IMAGE.fullmatch(rl_image or "")
    return (bool(match) and match["commit"] == VERL_COMMIT) or is_verl_registry_image(rl_image)


# The offline layout (yeto.launch.fragment_preflight): a verl full-parameter layout names
# no card or shard (full_parameter module doc), so one model plans one layout on any island
# shape; the launcher checks --fragments and --rl-parameter-layout-sha256 against it.
OFFLINE_LAYOUT_CARD_FREE = True


def offline_full_parameter_layout(config: Mapping, *, tp: int = 1, pp: int = 1, minimum_fragments: int,
                                  max_fragment_bytes: int, matrix_merge: str = "avg", model_revision: str = "",
                                  config_sha256: str = ""):
    """The full-parameter layout a verl island of the model in ``config`` (its
    ``config.json``) plans, computed on the control host (``full_parameter_plan``: the
    HF model on the meta device, no weights): ``fragments``, ``layout_hash`` (None
    without a commit ``model_revision`` and the config's SHA-256, which the hash binds),
    ``fragment_bytes``, ``ranks`` and ``exact``. ``tp`` / ``pp`` do not change it. None
    for a model the meta-device build does not take; ImportError without torch /
    transformers on this host."""
    import json as _json
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from yeto.rl.local_learner import ComponentIdentity

    from . import full_parameter as fp
    from .full_parameter_plan import model_shapes

    del tp, pp  # card-free: the same layout on every shape
    import torch  # noqa: F401 - ImportError: the check is out of scope on this host
    import transformers  # noqa: F401

    revision = str(model_revision or "").lower()
    identified = bool(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision)
                      and re.fullmatch(r"[0-9a-f]{64}", str(config_sha256 or "")))
    try:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(_json.dumps(dict(config)))
            shapes = model_shapes(directory)
        component = ComponentIdentity(role="actor", model_revision=revision if identified else "0" * 40,
                                      config_hash=config_sha256 if identified else "0" * 64)
        layout = fp.plan_layout(fp.island_specs(shapes), component=component,
                                minimum_fragments=int(minimum_fragments),
                                max_fragment_bytes=int(max_fragment_bytes), matrix_merge=matrix_merge)
    except Exception:  # noqa: BLE001 - a model the meta-device build or the planner does not take
        return None
    return SimpleNamespace(fragments=layout.fragments.num_fragments,
                           layout_hash=layout.layout_hash if identified else None,
                           fragment_bytes=tuple(f.numel * 4 for f in layout.fragments.fragments),
                           ranks=1, exact=True)


def per_learner_release_flags() -> str:
    """The island flag of a synced trainer whose samplers follow it (the launcher gives it
    its own sub-store ``trainer<k>`` and lane): it releases its own local versions there
    (``buffer_ports`` per-learner mode) instead of learner 0's merged versions."""
    return " --rl-release-mode per_learner"


def buffer_prompt_width(*, seq_len: int, response_len: int, pool_max_prompt_len: int | None = None) -> int:
    """The verl buffer trainer's prompt row width (``buffer_feed.buffer_prompt_width``):
    the launcher's check of an island fleet's trainers before anything is ordered."""
    from .buffer_feed import buffer_prompt_width as width

    return width(seq_len=seq_len, response_len=response_len, pool_max_prompt_len=pool_max_prompt_len)


def buffer_prompt_width_flags(width: int) -> str:
    """The island flag of a trainer's prompt row width (its pools' ``agentic.max_prompt_len``)."""
    return f" --rl-buffer-max-prompt-len {int(width)}"


def image_pin_problem(rl_image: str) -> str | None:
    """Why ``rl_image`` is not this yeto's verl image (yeto.launch.trainer_image_check):
    a ``verl-build`` / ``verl-megatron-build`` image of another commit than the pin
    (``pins.VERL_COMMIT``). None: the pinned image, or not a verl-built image."""
    match = _IMAGE.fullmatch(rl_image or "")
    if not match or match["commit"] == VERL_COMMIT:
        return None
    pinned = rl_image[:match.start("commit")] + VERL_COMMIT
    return (f"trainer image {rl_image} is built from verl commit {match['commit'][:12]}, but this yeto pins verl "
            f"{VERL_COMMIT[:12]} (yeto.rl.adapters.verl.pins.VERL_COMMIT; the island checks its package versions "
            f"against that pin): use {pinned}")


def pulled_image(rl_image: str) -> dict | None:
    """The registry image an island that pulls its image (a sky island) runs in place
    of the Modal-built ``verl-build:<pinned commit>``: the same pin and recipe on the
    registry ``$YETO_VERL_IMAGE_REGISTRY`` names. ``problem`` (and no ``image``) when no
    registry is configured: there is no default. None for any other image (an explicit
    registry image is used as given; the Megatron variant has no registry image)."""
    from .pins import VERL_REGISTRY_TAG, registry_image_for, registry_missing_problem

    if rl_image != VERL_IMAGE:
        return None
    image = registry_image_for(rl_image)
    if image is None:
        return {"image": None, "tag": VERL_REGISTRY_TAG, "replaces": rl_image, "problem": registry_missing_problem()}
    return {"image": image, "tag": VERL_REGISTRY_TAG, "replaces": rl_image}


def bundle_map_flag(bundle_map: Mapping[str, Sequence[int]]) -> str:
    """Refuse a role -> card map: verl places its trainer and rollout workers itself.

    Only a fixed-partition placement reaches here (the launcher emits no map
    for a single-node colocated island, whatever its card count)."""
    roles = ", ".join(f"{role}={list(slots)}" for role, slots in sorted(bundle_map.items()))
    raise ValueError(
        "a verl island is single-node and colocated: training and rollout use all of the node's cards, "
        f"placed by verl itself, and a per-role card bundle map is not accepted (got {roles}). For several "
        "cards on one node use --gpu <provider>:<N>x<card type> without a fixed-partition placement")


def cross_node_flags(allow_cross_node_tp: bool, allow_cross_node_engine_tp: bool) -> str:
    if allow_cross_node_tp or allow_cross_node_engine_tp:
        raise ValueError("a verl island stays on one node: cross-node TP is not supported")
    return ""


# ----------------------------------------------------------------------------- full-parameter preset
# The shape of ``--rl-sync-preset full-parameter`` on verl islands
# (launch.rl_args._prepare_dense_full_sweep asks for it; the Miles backend
# provides none and keeps its fixed-partition shape).  A verl full-parameter
# island is colocated by default: verl's hybrid engine trains and rolls out on
# the same cards.  ``--rl-placement fixed-partition`` with
# ``--rl-rollout-gpus N`` runs verl's separate mode instead (``separate``): the
# trainer keeps its own cards, standalone vLLM servers take the last N cards
# and the weights go over verl's NCCL checkpoint engine.  The layout hash
# and fragment count come from ``python -m yeto.rl.adapters.verl.full_parameter_plan``
# (the model's HF config on the meta device), not from the Miles layout probe.
FULL_PARAMETER_PLACEMENT = "colocated"  # the default
FULL_PARAMETER_PLACEMENTS = ("colocated", "fixed-partition")
# The verl stream recovers from a peer (adapters/verl/peer_recovery.py),
# so the async mode turns it on by default as on Miles (rl_args._peer_recovery_applies).
FULL_PARAMETER_PEER_RECOVERY = True


def full_parameter_peer_recovery(args) -> bool:
    """Whether the async mode's default peer recovery applies to this run:
    not on the Megatron engine island, whose donor has no FSDP2 rank files
    (``megatron_engine.sync_problems``); it rejoins by catch-up."""
    from .run_config_rules import megatron_run

    return not megatron_run(args)
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
    placement = getattr(args, "rl_placement", "colocated")
    if placement not in FULL_PARAMETER_PLACEMENTS:
        problems.append("--rl-placement colocated or fixed-partition (a verl island trains and rolls out "
                        "on the same cards, or in verl's separate mode on its own trainer and rollout cards)")
    if placement == "fixed-partition" and int(getattr(args, "rl_standby_gpus", 0) or 0):
        problems.append("no --rl-standby-gpus (verl's separate mode has no standby cards)")
    if is_synced_trainer(args) and placement == "fixed-partition":
        problems.append("no --rl-placement fixed-partition with --rl-sample-role trainer (a verl buffer "
                        "trainer starts no engine; the sampler islands generate)")
    # A synced buffer trainer runs verl's fully_async trainer over the sample buffer
    # (--rl-max-policy-age >= 1, sample_roles). With --rl-rollout-ahead 1 the limit
    # bounds the one-version lag of the separate mode's synchronous trainer;
    # without it a limit > 0 runs verl's fully_async trainer and rollouter,
    # the rollouter on its own cards
    if (int(getattr(args, "rl_max_policy_age", 0) or 0) and not is_synced_trainer(args)
            and placement != "fixed-partition"):
        problems.append("--rl-placement fixed-partition with --rl-max-policy-age > 0 (the fully_async "
                        "rollouter, or the rollout servers of --rl-rollout-ahead, generate on their own cards)")
    if int(getattr(args, "rl_rollout_quant_tier", 0) or 0):
        problems.append("no --rl-rollout-quant-tier")
    if int(getattr(args, "expert_full_count", 0) or 0):
        problems.append("no --expert-full-count (DeepSeek V4 expert-full is a Miles recipe)")
    for dest, option in (("rl_sync_overlap", "--rl-sync-overlap"),
                         ("rl_sync_expert_token_weights", "--rl-sync-expert-token-weights")):
        if getattr(args, dest, False):
            problems.append(f"no {option} (not offered by the verl stream)")
    if not re.fullmatch(r"[0-9a-f]{64}", getattr(args, "rl_parameter_layout_sha256", None) or ""):
        problems.append(f"the verl layout hash and fragment count from `{FULL_PARAMETER_PLAN_COMMAND}`"
                        + (" (--rl-parameter-layout-sha256 auto is Miles-only: verl plans its layout "
                           "offline)" if getattr(args, "rl_parameter_layout_sha256", None) == "auto" else ""))
    ignored = [option for dest, off, option in IGNORED_ON_VERL_FULL_PARAMETER
               if getattr(args, dest, off) not in (off, None)]
    if ignored:
        problems.append("no " + ", ".join(ignored) + " (the verl full-parameter island has no such option; "
                        "it would ignore it: " + "; ".join(f"{option}: {NOT_MAPPED_REASONS[option]}"
                                                           for option in ignored) + ")")
    schedule = getattr(args, "rl_lr_schedule", None)
    if schedule not in (None, "auto", "constant"):  # verl.train_options
        problems.append(f"--rl-lr-schedule constant (got {schedule}; verl's FSDP2 optimizer schedules are "
                        "constant and cosine, not Miles' linear decay)")
    return problems


# Options the launcher forwards to every island that the verl island
# entry does not parse (parse_known_args would drop them without a word):
# (args dest, value when off, option). Refused instead, before any cloud work.
# The island refuses them too when they reach a full-parameter island the
# launcher's preset check does not cover (a no-sync island:
# train_options.refuse_unmapped_argv); --rl-debug-dump is mapped (train_options.debug_plan).
# Mapped elsewhere: verl.train_options (--rl-lr-schedule constant,
# --rl-max-tokens-per-gpu, --rl-activation-recompute, --rl-observe-timeline,
# --rl-resource-sample-interval) and verl.eval_port (--rl-eval-interval:
# verl's _validate behind the driver's evaluate hook,
# verl.eval_port); the rest stay here.
IGNORED_ON_VERL_FULL_PARAMETER = (
    ("rl_verify_every_round", False, "--rl-verify-every-round"),
    ("rl_deterministic_trainer", False, "--rl-deterministic-trainer"),
    ("rl_gradient_accumulation_fusion", False, "--rl-gradient-accumulation-fusion"),
    ("rl_async_buffer_rounds", None, "--rl-async-buffer-rounds"),
    ("rl_lora_dropout", None, "--rl-lora-dropout"),
    ("rl_lora_expert_rank", None, "--rl-lora-expert-rank"),
    ("rl_profile_trainer", False, "--rl-profile-trainer"),
    ("rl_log_probs_chunk_size", "auto", "--rl-log-probs-chunk-size"),
)
# Why each option above has no verl mapping yet (the launcher's refusal says it).
NOT_MAPPED_REASONS = {
    "--rl-verify-every-round": "the full-parameter publication does not read vLLM's weights back "
                               "(the island refuses it as well, train_options.verify_plan)",
    "--rl-deterministic-trainer": "yeto.rl.engine.determinism has no measured verl row",
    "--rl-gradient-accumulation-fusion": "a Megatron kernel option; FSDP2 has none",
    "--rl-async-buffer-rounds": "Miles' fully_async buffer depth; verl's fully_async rollouter has no such "
                                "option (its staleness_threshold follows --rl-max-policy-age)",
    "--rl-lora-dropout": "a full-parameter island has no LoRA adapter",
    "--rl-lora-expert-rank": "a full-parameter island has no LoRA adapter",
    "--rl-profile-trainer": "the trainer step runs in verl's Ray workers; the step profiler is not hooked "
                            "there",
    "--rl-log-probs-chunk-size": "Miles' log-prob row chunks; verl's actor computes log-probs in its own "
                                 "workers and no chunk option is mapped there",
}
# The options above that do not apply to a full-parameter
# FSDP2 island at all (the rest wait for a mapping)
NOT_APPLICABLE_ON_FULL_PARAMETER = ("--rl-gradient-accumulation-fusion", "--rl-lora-dropout",
                                    "--rl-lora-expert-rank")
