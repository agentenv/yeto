"""RL launch flag schema: every RL flag is defined exactly once (P0 4.7).

A flag is an :class:`RlFlag` with the layers that accept it:

* ``cli``    -- ``yeto launch`` (the RL argument group of ``yeto/cli.py``);
* ``island`` -- the island entrypoint the launcher starts on every island.

A ``shared_flag`` is parsed by both layers with the same name, type, choices and
action. Where the island must differ (a value the launcher always sends is
``required`` there), the difference is an explicit ``island={...}`` override,
listed in ``tests/test_run_flags.py``. The launcher names the flags it checks or
forwards through :func:`rl_option_pairs` / :func:`rl_forward_pairs`, so a renamed
flag fails at import instead of being sent under a stale name.

This module holds the backend-neutral flags. The training backend contributes
its own flags (engine, image, inference-server and trainer knobs, the island
entrypoint's internal flags) through the ``run_flags`` role of
``yeto.rl.engine.backends``: a module whose ``RL_FLAGS`` maps theme -> flags.
Flags are grouped by theme; :data:`THEMES` fixes the help order, and within a
theme the neutral flags come before the backend's.

Generic launch flags (``--model``, ``--data``, ``--seed``, parallel sizes, LoRA
shape, ...) stay in their ``yeto/cli.py`` groups. Recommend-mode and W&B flags
keep their own registration hooks.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from yeto.rl.engine.backends import DEFAULT_BACKEND, KNOWN_BACKENDS
from yeto.rl.engine.data_split import AUTO, DATA_SPLITS
from yeto.rl.engine.selection import (
    FULL_PARAMETER_ROUNDS,
    RL_MATRIX_MERGES,
    RL_SYNC_MODES,
    SYNC_PRESETS,
    ASYNC_GRACE_CAP_MS,
    ASYNC_NESTEROV_MIN_H,
    ASYNC_TERMINAL_GRACE_MS,
    canonical_sync_preset,
    sync_mode_default,
)

CLI = "cli"
# S27 asynchronous full-parameter rounds: the default and the largest number
# of fragment rounds in flight (--rl-sync-fragment-pipeline / island
# --pipeline).  The syncer and the island pairing take any depth up to P; the
# bound limits the retained submissions and staged broadcasts (one fragment
# each) an island holds at once.  60 lets the probed P=60 layouts push a whole
# model round at once (``--total-steps 1``: one boundary submits every
# fragment).
FULL_PARAMETER_ASYNC_DEFAULT_PIPELINE = 2
FULL_PARAMETER_ASYNC_MAX_PIPELINE = 60
# The island's wait for a paired broadcast, the terminal barrier and the FINAL
# cut (--rl-sync-wait-timeout-s / island --sweep-wait-timeout-s).
FULL_PARAMETER_DEFAULT_WAIT_TIMEOUT_S = 900.0
# S27 G-AG lead cap of the asynchronous rounds (--rl-sync-max-local-rounds-ahead
# / island --sweep-max-local-rounds-ahead): the largest accepted K.
FULL_PARAMETER_MAX_LOCAL_ROUNDS_AHEAD = 1000

ISLAND = "island"
LAYERS = (CLI, ISLAND)

# Help order of the flag themes; "island" holds the island-internal flags.
THEMES = (
    "mode", "model", "reward", "algorithm", "placement", "runtime", "elastic", "resume",
    "scheduling", "eval", "run_control", "rollout_engine", "agent", "sync", "island",
)

_EMPTY: Mapping[str, Any] = MappingProxyType({})


class ArgumentTarget(Protocol):
    """What flags are registered on: an ``ArgumentParser`` or an argument group."""

    def add_argument(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class EnvDefault:
    """A default read from the environment when the parser is built."""

    variable: str
    fallback: str
    convert: Callable[[str], Any] = str

    def resolve(self) -> Any:
        import os

        return self.convert(os.environ.get(self.variable, self.fallback))


@dataclass(frozen=True)
class RlFlag:
    """One RL flag: its option strings, argparse keywords and accepting layers."""

    options: tuple[str, ...]
    kwargs: Mapping[str, Any]
    layers: frozenset[str]
    island: Mapping[str, Any] = _EMPTY
    cli_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.layers or not self.layers <= set(LAYERS):
            raise ValueError(f"{self.option}: unknown layers {sorted(self.layers)}")
        if self.island and ISLAND not in self.layers:
            raise ValueError(f"{self.option}: island overrides on a flag the island lacks")
        if self.cli_aliases and CLI not in self.layers:
            raise ValueError(f"{self.option}: CLI aliases on a flag the CLI lacks")

    @property
    def option(self) -> str:
        return self.options[0]

    @property
    def dest(self) -> str:
        return self.kwargs.get("dest") or self.option.lstrip("-").replace("-", "_")

    def argparse_call(self, layer: str) -> tuple[tuple[str, ...], dict[str, Any]]:
        """The ``add_argument`` positional and keyword arguments for ``layer``."""

        if layer not in self.layers:
            raise ValueError(f"{self.option} is not a {layer} flag")
        kwargs = dict(self.kwargs) if layer == CLI else {**self.kwargs, **self.island}
        if isinstance(kwargs.get("default"), EnvDefault):
            kwargs["default"] = kwargs["default"].resolve()
        options = self.options + self.cli_aliases if layer == CLI else self.options
        return options, kwargs


def _flag(layers: frozenset[str], options: tuple[str, ...], kwargs: dict[str, Any], *,
          island: Mapping[str, Any] | None = None,
          cli_aliases: tuple[str, ...] = ()) -> RlFlag:
    return RlFlag(options, MappingProxyType(kwargs), layers,
                  MappingProxyType(dict(island or {})), cli_aliases)


def cli_flag(*options: str, **kwargs: Any) -> RlFlag:
    """A flag only ``yeto launch`` accepts (the launcher consumes it)."""

    return _flag(frozenset((CLI,)), options, kwargs)


def island_flag(*options: str, **kwargs: Any) -> RlFlag:
    """A flag only the island entrypoint accepts (the launcher generates it)."""

    return _flag(frozenset((ISLAND,)), options, kwargs)


def shared_flag(*options: str, island: Mapping[str, Any] | None = None,
                cli_aliases: tuple[str, ...] = (), **kwargs: Any) -> RlFlag:
    """A flag both layers accept; ``island`` lists the island's keyword overrides."""

    return _flag(frozenset(LAYERS), options, kwargs, island=island, cli_aliases=cli_aliases)


def sha256_hex_arg(value: str) -> str:
    """argparse type: a lowercase 64-hex-digit SHA256."""

    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise argparse.ArgumentTypeError("must be 64 lowercase hex chars")
    return value


def backend_flags(backend: str = DEFAULT_BACKEND) -> Mapping[str, tuple[RlFlag, ...]]:
    """The flags ``backend`` contributes, by theme (its ``run_flags`` role)."""

    from yeto.rl.engine import backends

    contributed = backends.module("run_flags", backend).RL_FLAGS
    unknown = set(contributed) - set(THEMES)
    if unknown:
        raise ValueError(f"backend {backend!r} flags use unknown themes {sorted(unknown)}")
    return contributed


def rl_flags(backend: str = DEFAULT_BACKEND) -> tuple[RlFlag, ...]:
    """Every RL flag in help order: per theme, the neutral flags then the backend's."""

    contributed = backend_flags(backend)
    return tuple(flag for theme in THEMES
                 for flag in _NEUTRAL_FLAGS.get(theme, ()) + tuple(contributed.get(theme, ())))


def flags_for(layer: str) -> Iterator[RlFlag]:
    return (flag for flag in rl_flags() if layer in flag.layers)


def add_rl_flags(parser: ArgumentTarget, layer: str) -> None:
    """Register every ``layer`` flag on ``parser`` (a parser or an argument group)."""

    for flag in flags_for(layer):
        args, kwargs = flag.argparse_call(layer)
        parser.add_argument(*args, **kwargs)


def rl_option(dest: str, layer: str = ISLAND) -> str:
    """The option string of the ``layer`` flag stored in ``dest`` (KeyError if none)."""

    for flag in flags_for(layer):
        if flag.dest == dest:
            return flag.option
    raise KeyError(f"no {layer} RL flag stores {dest!r}")


def rl_option_pairs(*dests: str, layer: str = ISLAND) -> tuple[tuple[str, str], ...]:
    """``(dest, option)`` pairs, in the order given, for check tables."""

    return tuple((dest, rl_option(dest, layer)) for dest in dests)


def rl_forward_pairs(*dests: str | tuple[str, str]) -> tuple[tuple[str, str], ...]:
    """``(cli_dest, island_option)`` pairs for flags the launcher forwards to the island.

    Each item is a dest both layers share, or ``(cli_dest, island_dest)`` when the
    island spells it differently. Both must exist, else KeyError at import.
    """

    pairs = []
    for item in dests:
        cli_dest, island_dest = (item, item) if isinstance(item, str) else item
        rl_option(cli_dest, CLI)
        pairs.append((cli_dest, rl_option(island_dest, ISLAND)))
    return tuple(pairs)


# --- the neutral schema, by theme --------------------------------------------
# Generated once from the hand-written parsers; the parameter lists before and
# after were diffed to zero. Edit entries here (or in the backend's run_flags
# module); never add an RL flag to a parser directly.

_MODE = (
    cli_flag(
        "--training-mode",
        choices=["sft", "rl"],
        default="sft",
        help="training workflow (default: sft)",
    ),
    cli_flag(
        "--rl-backend",
        choices=list(KNOWN_BACKENDS),
        default=DEFAULT_BACKEND,
        help=f"training backend adapter (yeto/rl/engine/backends.py; default {DEFAULT_BACKEND}). "
        "A backend without a registered adapter is refused before launch.",
    ),
    cli_flag(
        "--rl-image-private",
        action="store_true",
        help=(
            "pull --rl-image with a registry login: SKYPILOT_DOCKER_USERNAME/"
            "PASSWORD/SERVER from the environment, else the registry's "
            "~/.docker/config.json entry (error if neither). Default: anonymous "
            "pull; a login is injected only when those variables are set"
        ),
    ),
)

_MODEL = (
    shared_flag(
        "--expert-full-count",
        type=int,
        default=0,
        help="DeepSeek V4 only: number of attested clone experts per layer to tune fully",
    ),
    shared_flag("--expert-full-lr", type=float, default=1e-6),
    shared_flag("--expert-selection-sha256", default=None),
    shared_flag("--expert-selection-contract-sha256", default=None),
    shared_flag(
        "--rollout-model-revision",
        default=None,
        help="immutable Hugging Face commit represented by --rollout-model",
    ),
)

_REWARD = (
    shared_flag(
        "--reward-function",
        default=None,
        help="RL reward callable as package.module:function",
        island={"required": True},
    ),
    shared_flag(
        "--rl-prompt-column",
        default=None,
        help="dataset column holding the prompt (default: messages, prompt or input)",
    ),
    shared_flag(
        "--rl-label-column",
        default=None,
        help="dataset column passed to the reward function as the label (default: label)",
    ),
    cli_flag(
        "--cybergym-url",
        default=EnvDefault("CYBERGYM_URL", "http://127.0.0.1:8666"),
    ),
    cli_flag(
        "--cybergym-agent-id",
        default=EnvDefault("CYBERGYM_AGENT_ID", "yeto_agent"),
    ),
    cli_flag(
        "--cybergym-timeout",
        type=float,
        default=EnvDefault("CYBERGYM_TIMEOUT", "60", float),
    ),
)

_ALGORITHM = (
    cli_flag(
        "--advantage-estimator", choices=["grpo"], default="grpo"
    ),
    cli_flag("--n-samples-per-prompt", type=int, default=4),
    cli_flag("--rollout-batch-size", type=int, default=32),
    shared_flag("--over-sampling-batch-size", type=int, default=None,
                island={"required": True}),
    shared_flag(
        "--dynamic-sampling-max-replacements",
        type=int,
        default=None,
        help=(
            "bound zero-variance rollout replacements; with the stock "
            "nonzero-variance filter, automatically use Yeto's bounded "
            "fallback after this many rejected groups"
        ),
    ),
    cli_flag(  # refused since S26 E0.5 (removed Codex/SecRLEnv agents)
        "--secrlenv-max-infrastructure-replacements",
        type=int,
        default=None,
        help=(
            "bound authenticated SecRLEnv infrastructure-group retries; "
            "the signed SecRLEnv contract requires exactly one same-task retry"
        ),
    ),
    cli_flag(
        "--rl-allow-local-data",
        action="store_true",
        help=(
            "ports engine: accept a local prompt file for --data (shipped to the "
            "island via file mounts; provenance records source=local, no revision). "
            "Default: RL requires a revision-pinned Hugging Face dataset"
        ),
    ),
    # rl-algorithm-capabilities D8/D9/D11 (ports only; refused on legacy).
    shared_flag(
        "--rl-algorithm-spec",
        default=None,
        metavar="PATH",
        help=(
            "AlgorithmSpec JSON (v1 or v2) for --rl-engine ports; the launcher "
            "sends each island its canonical form and expected hash"
        ),
    ),
    # rl-algo-critic-family 5.3 (design D5): critic warm-up stage on the island
    # rl-algo-critic-family D5: the critic warm-up stage product the main stage loads
    shared_flag(
        "--rl-critic-load",
        default=None,
        metavar="DIR",
        help=(
            "ports critic with a warm-up: an existing stage-W product directory on the "
            "island (with --rl-critic-init-sha256); without it the island runs stage W "
            "itself before the main stage"
        ),
    ),
    shared_flag(
        "--rl-critic-init-sha256",
        default=None,
        metavar="HEX",
        help="content SHA256 of --rl-critic-load (critic_warmup.checkpoint_sha256)",
    ),
    shared_flag(
        "--rl-critic-baseline-rounds",
        type=int,
        default=0,
        metavar="N",
        help=(
            "ports critic with a warm-up, --rl-single-island-no-sync: first run N rounds "
            "of the same algorithm without the warm-up (explained-variance baseline)"
        ),
    ),
)

_PLACEMENT = (
    # rl-infra-spec 2.1 (ports only): LoRA fixed partition + reserved standby
    # GPUs, read by engine.run_config.resolve_rl_run_config.
    shared_flag(
        "--rl-placement",
        choices=["colocated", "fixed-partition"],
        default="colocated",
        help="ports: colocated (default) or a LoRA fixed partition (rl-infra-spec 2.1)",
    ),
    cli_flag(
        "--rl-rollout-gpus",
        dest="rollout_num_gpus",
        type=int,
        default=None,
        help="ports fixed partition: dedicated rollout GPUs per island (rl-infra-spec 2.1; "
        "on a multi-node island the island total, laid out after the trainer's nodes). "
        "Note: on `launch`, a bare --rollout-num-gpus is an argparse abbreviation of "
        "--rollout-num-gpus-per-engine, not this option",
    ),
    shared_flag(
        "--rl-standby-gpus",
        type=int,
        default=0,
        help="ports fixed partition: reserved standby GPUs never started by any role",
    ),
    cli_flag(
        "--rl-min-nodes-per-learner",
        type=int,
        default=0,
        help="rl-multinode-island: refuse a learner island with fewer nodes (0 = the "
        "recipe-derived minimum: one TP*PP replica + one rollout engine + standby)",
    ),
    # ruling 2026-10-04 v2: explicit cross-node TP opt-ins (default: TP stays in a node)
    shared_flag("--rl-allow-cross-node-tp", action="store_true",
                help="rl-multinode-island (ruling 2026-10-04 v2): let the trainer TP*CP group "
                "span nodes (default: TP stays inside a node, refused otherwise); same as cfg "
                "parallel.allow_cross_node_tp; journaled as topology.layout.cross_node_tp"),
    # rl-mixed-island-placement: per-island placement values (yeto.rl.engine.island_placement)
    cli_flag("--rl-island-placement", action="append", default=None,
             metavar="ISLAND:KEY=VALUE",
             help="give island ISLAND its own placement value (repeatable); KEY is "
             "placement (colocated | fixed-partition), standby-gpus or use-gpus-per-node; "
             "other islands keep --rl-placement / --rl-standby-gpus / "
             "--rl-island-use-gpus-per-node. Different placements need island pool "
             "options (e.g. --rl-sync-island-lease-s)"),
    cli_flag("--rl-island-use-gpus-per-node", type=int, default=None, metavar="M",
             help="rl-multinode-island: provision the --gpu machines (N GPUs/node, billed) but "
             "give the island only local GPUs 0..M-1 on every node (M <= N): ray start "
             "--num-gpus=M, CUDA_VISIBLE_DEVICES=0..M-1, and every layout/placement/gpu_pool "
             "rule uses M (e.g. m5 TP8 over 2x4 on 2x8xH100 machines); unset = all N"),
    cli_flag("--rl-island-ray-object-store-gib", type=int, default=None, metavar="GIB",
             help="RL island: Ray object store size per node in GiB (ray start "
             "--object-store-memory); unset = Ray's default (30%% of memory, capped by "
             "/dev/shm). Full-parameter islands keep whole-model f32 cuts there; what does "
             "not fit spills to disk (see --rl-island-ray-spill-dir)"),
    cli_flag("--rl-island-ray-spill-dir", default=None, metavar="PATH",
             help="RL island: directory for Ray object spilling (default: the island's Ray "
             "temp dir on the container/VM disk). Point it at the largest local disk; on "
             "Modal size the disk with --modal-ephemeral-disk-gib"),
    cli_flag("--rl-node-inventory", default=None, metavar="PATH",
             help="rl-ssh-node-pool: YAML inventory of the nodes yeto reaches by SSH "
             "(per node: ssh target, optional jump host, device family, card, card count, "
             "HBM GiB, host memory GiB, fabric, mounts, image). The launcher validates the "
             "file before any cloud or SSH work and refuses an unknown card or a missing "
             "device family; the SSH harness consumes it"),
    cli_flag("--model-store", default=None, metavar="URI",
             help="RL island: cloud-local pinned HF snapshots (COLDSTART-PLAN #4). "
             "nebius-fs://<computefilesystem-id> attaches that Nebius shared filesystem "
             "at /mnt/yeto-models on every node (needs nebius@<region>) and points "
             "HF_HUB_CACHE at it when it holds a completed --model@--model-revision "
             "(scripts/populate_nebius_model_store.sh); otherwise WARNING + Hub download"),
    cli_flag("--rl-island-network-tier", choices=("auto", "none"), default="auto",
             help="multi-node RL island: auto = ask sky for network_tier=best where the cloud "
             "honors it (Nebius H100:8/H200:8 -> InfiniBand GPU cluster on a fixed fabric); "
             "none = plain VMs (NCCL stays on TCP sockets, NCCL_IB_DISABLE=1 by default anyway)"),
)

_RUNTIME = (
    shared_flag("--rl-elastic-state-dir", default=None, metavar="ISLAND_PATH",
                help="--rl-elastic: controller journal/ledger/cut dir ON THE ISLAND (e.g. a "
                "persistent volume mount); default ~/yeto-rl/elastic-state"),
    cli_flag("--rl-elastic-restart-attempts", type=int, default=None, metavar="N",
             help="--rl-elastic: re-run the learner in place (same args and state dir) "
             "after a non-zero exit, at most N times; off by default"),
    cli_flag("--rl-test-inject-lora-perturb", type=float, default=None, metavar="EPS",
             help="--rl-elastic, TEST ONLY (3.5 E1-B): the first member update ships the "
             "LoRA adapter + EPS (trainer restored right after), so check_weights must "
             "refuse the new engines"),
    cli_flag("--rl-test-hold-before-check-s", type=float, default=None, metavar="S",
             help="--rl-elastic, TEST ONLY (A4 E1-B (a)): the first member publication "
             "stays S seconds after end_update_weights and before check_weights/"
             "admit_cells (observation window); needs another --rl-test-* injection"),
    cli_flag("--rl-test-inject-tool-wait-s", type=float, default=None, metavar="S",
             help="--rl-elastic, TEST ONLY (A4b): every drain executes an artificial tool "
             "that waits S seconds (drain timeout -> CANCELLED -> undrain); a drain "
             "while it still waits is journaled, not re-armed; needs "
             "--rl-elastic-tool-wait-board"),
    cli_flag("--rl-test-tool-side-effect-log", action="store_true", default=False,
             help="--rl-elastic, TEST ONLY (A4bc / 3.3 X5): journal every execution of "
             "the injected tool (--rl-test-inject-tool-wait-s) in <elastic state dir>/"
             "side_effects.jsonl, one tool_side_effect record per (trajectory, tool call) "
             "before the wait starts; the judge proves a drain-timeout cancel never "
             "replays a tool call"),
    cli_flag("--rl-test-inject-undrain-fail", type=int, default=None, metavar="N",
             help="--rl-elastic, TEST ONLY (A4b / E1-C): the next N undrain calls fail "
             "inside the adapter, so a drain timeout cannot restore the old routing "
             "and the island must end in RECOVERY_REQUIRED (never CANCELLED)"),
    cli_flag("--rl-test-inject-stop-failures", type=int, default=None, metavar="N",
             help="--rl-elastic, TEST ONLY (3.7 E1-D ③④): the next N fork stop_cells fail "
             "inside the fork after deregistration (incomplete)"),
    cli_flag("--rl-test-kill-learner-at", default=None, metavar="PHASE",
             help="--rl-elastic, TEST ONLY (3.7 E1-D ⑤⑥): hard-kill the learner once when a "
             "transaction journals PHASE (needs --rl-elastic-restart-attempts)"),
    # ports LoRA training-time dropout (default 0 = unchanged argv)
    shared_flag("--rl-lora-dropout", type=float, default=None, metavar="P",
                help="ports LoRA: training-time LoRA dropout (default 0). Trainer DP-change "
                "edges refuse dropout > 0; same-shape rebuild restores its RNG"),
    # Flash-Next native recipe: routed-expert LoRA rank (default: profile's 8)
    shared_flag("--rl-lora-expert-rank", type=int, default=None, metavar="R",
                help="ports Qwen3.8-Flash-Next: routed-expert LoRA rank r_e (0 <= r_e <= "
                "--lora-r; default the profile's 8)"),
    cli_flag("--rl-island-pre-run", default=None, metavar="FILE",
             help="ports, single island no-sync: a local bash snippet embedded into the island "
             "run script and executed on the node (inside the island image, GPUs idle) "
             "BEFORE the island's Ray starts; a non-zero exit fails the job. For node-side "
             "preparation the launcher has no flag for, e.g. an HF -> torch_dist conversion "
             "on a cloud that cannot mount a prepared model volume (Verda G0)"),
    cli_flag("--rl-test-inject-rebuild-fail", action="store_true",
             help="--rl-elastic, TEST ONLY (4.4 REBUILD_OLD): the first same-shape trainer "
             "rebuild fails in the fork at create_training_models"),
    cli_flag("--rl-test-inject-cut-save-kill-rank", type=int, default=None, metavar="RANK",
             help="--rl-elastic, TEST ONLY (E2 G-4.5): that trainer rank exits while "
             "writing its save_cut shard"),
    cli_flag("--rl-test-inject-cut-restore-kill-rank", type=int, default=None, metavar="RANK",
             help="--rl-elastic, TEST ONLY (E2 G-4.5): that trainer rank exits during "
             "restore_cut (after the adapter write)"),
    cli_flag("--rl-test-inject-cut-restore-sleep", default=None, metavar="RANK:SECONDS",
             help="--rl-elastic, TEST ONLY (E2 G-4.5): that trainer rank sleeps before "
             "restore_cut"),
    cli_flag("--rl-test-inject-rebuild-cursor-shift", type=int, default=None, metavar="GROUPS",
             help="--rl-elastic, TEST ONLY (E2 G-4.5): before a trainer rebuild, place a "
             "dataset state advanced by GROUPS where rollout_executor.load reads it"),
    # 1.7 observation: per-round timeline labels (entry observe=...), off by default.
    shared_flag("--rl-observe-timeline", action="store_true",
                help="ports: record per-round timeline labels (rl-infra-spec 1.7); off by default"),
    shared_flag("--rl-verify-every-round", action="store_true",
                help="ports: hash the policy and read the engine weight checksums back at every "
                "publication (default: on the first publication and on every --rl-cut-every-th "
                "round, which includes every round that keeps a round cut)"),
    shared_flag("--rl-profile-trainer", action="store_true",
                help="ports, observation only (E1): per training step, print where the trainer "
                "process spends its time (stack samples, GC pauses, CUDA allocator counters)"),
    shared_flag("--rl-resource-sample-interval", type=float, default=None, metavar="SECONDS",
                help="ports: NVML rl_resource_sample period forwarded to the learner "
                "(default: learner default, 60 s with --rl-observe-timeline; 0 = off)"),
)

_ELASTIC = (
    # 3.3 X5: wire the island's ToolWaitBoard actor into the elastic pool (needs a
    # rollout workload that records tool waits, e.g. yeto.rl.tool_wait_workload).
    shared_flag("--rl-elastic-tool-wait-board", action="store_true",
                help="--rl-elastic: feed the island's tool-wait board into the drain check "
                "(3.3; needs a workload that records tool waits)"),
    shared_flag(
        "--rl-elastic",
        action="store_true",
        help="ports: enable the E1 elastic rollout controller (rl-infra-spec 3.x); "
        "needs --rl-elastic-resources/-initial-config/-cells; off by default",
    ),
    shared_flag("--rl-elastic-resources", default=None, metavar="PATH",
                help="--rl-elastic: resources manifest JSON (configs/edges)"),
    shared_flag("--rl-elastic-attestation", default=None, metavar="PATH",
                help="--rl-elastic: capability attestation JSON"),
    shared_flag("--rl-elastic-initial-config", default=None, metavar="NAME",
                help="--rl-elastic: initial config id in the manifest"),
    shared_flag("--rl-elastic-cells", default=None, metavar="ID[,ID...]",
                help="--rl-elastic: the rollout cells to manage: yeto names declared to the fork "
                "with --rl-elastic-declare-cells (F-R1), else the fork's own cell ids; "
                "default: every cell the fork declares (needs describe_cells, F-R1)"),
)

_RESUME = (
    cli_flag("--rl-checkpoint-store", default=None, metavar="PATH|URI",
             help="--rl-elastic, or --rl-single-island-no-sync (resume across launches, "
             "rl-resume-from-checkpoint): off-island copy of the island state dir (journal, cuts, "
             "ledger) for a rebuild after node loss (rl-multinode-island Q4): a bucket "
             "URI (s3://, gs://, ...) mounted on the island, modal-volume://NAME[/PREFIX] on "
             "a Modal island (the only form Modal accepts), or a path already shared "
             "across machines (NFS, persistent volume). The learner syncs the state dir "
             "there after every commit point and restores from it when its state dir "
             "is empty (machine replaced). Without it the state stays on node0's local "
             "disk (warning on a multi-node island)"),
    # S27 G-CO: a single-island store that holds only the HF exports
    cli_flag("--rl-checkpoint-store-export-only", action="store_true", default=False,
             help="with --rl-single-island-no-sync, --rl-checkpoint-store and "
             "--rl-export-hf-rounds: the store holds only the HF exports. The launcher "
             "forwards no --rl-resume-store (no resume ledger, no round cuts), so the run "
             "does not resume across launches; --rl-rollout-ahead accepts this store. "
             "Refused with the cut and resume flags"),
    # rl-identity-tiering L9.4-L9.6
    cli_flag("--rl-code-pack", choices=["auto", "off", "git-patch", "release"], default="auto",
             help="git-patch: pack the local git tree (commit + dirty patch with untracked "
             "files) once at the first start, store it in --rl-checkpoint-store and ship it "
             "as every island's workdir; resumed and replacement islands get the same pack. "
             "release: --rl-image is a yeto release image (scripts/build_yeto_release_image.sh); "
             "its digest is the pack, islands run its /opt/yeto/src and no local tree is shipped. "
             "auto (default): git-patch when --rl-checkpoint-store is a bucket or "
             "modal-volume:// URI, else off"),
    cli_flag("--rl-code-pack-upgrade", default=None, metavar="REASON",
             help="with --rl-code-pack git-patch or release on an existing run: switch the run "
             "to a pack of the current local tree (or to the release --rl-image); refused unless the algorithm semantic hash is "
             "unchanged; the manifest records the old -> new pack and REASON"),
    # rl-resume-from-checkpoint (S17 C4)
    shared_flag("--rl-cut-every", type=int, default=None,
                help="with --rl-checkpoint-store: keep a round cut every N rounds (plus the "
                "last round and our own stop). Default: a timed interval, at most 5%% of the "
                "wall time in cuts and sqrt(2 * C * 21600) s between them (C = the measured "
                "cost of a cut), where the engine wires timed cuts; else every round. The "
                "publication checks (policy "
                "hash, engine weight read-back) run at the same interval and on the first "
                "publication; see --rl-verify-every-round"),
    # S27 R-2: a timed cut interval
    shared_flag("--rl-cut-auto-mtbf-s", type=float, default=None, metavar="S",
                help="with --rl-checkpoint-store, instead of --rl-cut-every: keep a round cut "
                "once sqrt(2 * C * S) seconds have passed since the previous one (C = "
                "the measured seconds a cut costs the round, S = the expected seconds between "
                "interruptions, e.g. 21600 for spot, 604800 on demand); off by default"),
    shared_flag("--rl-cut-max-interval-s", type=float, default=None, metavar="S",
                help="with --rl-cut-auto-mtbf-s: at most S seconds between cuts (default 3600)"),
    shared_flag("--rl-cut-keep", type=int, default=None,
                help="with --rl-checkpoint-store and no --rl-elastic: keep the newest K cuts "
                "(default 2); older ones are deleted only after the new LATEST is committed"),
    shared_flag("--rl-resume-allow-config-change", action="store_true",
                help="resume a cut even though the run configuration differs (the "
                "differences are written to the rl_resume event); refused by default"),
    shared_flag("--rl-stop-after-rounds", type=int, default=None,
                help="with --rl-checkpoint-store and no --rl-elastic: stop after N rounds "
                "with a final cut (our own stop; the next launch resumes)"),
    # S27 closing-run evaluation (yeto.rl.engine.hf_export)
    shared_flag("--rl-export-hf-rounds", default=None, metavar="R[,R...][,final]",
                help="with --rl-checkpoint-store (full parameter, no --rl-elastic): at "
                "the sync boundary of round R (the island's R-th trained round; final: the "
                "round that stops the island, before its final ACK), write the island's own "
                "model as an HF directory to <store>/hf/island<k>/round-R (k = learner id; "
                "complete when it holds .complete; <store>/hf/island<k>/final.json names the "
                "final one); tape event rl_hf_export. Off by default"),
    # rl-trainer-start-stop L4.6: sampler / trainer islands over the durable sample buffer
    shared_flag("--rl-sample-role", choices=("both", "sampler", "trainer", "convertible"),
                default="both",
                help="both (default): the island generates and trains as before. sampler: the "
                "island only generates and writes each group to the sample buffer in "
                "--rl-checkpoint-store (no trainer model, no trainer GPU) and loads every policy "
                "the trainer releases. trainer: the island trains on groups read from the buffer "
                "(needs --rl-max-policy-age >= 1) and releases every published policy. Both need "
                "--rl-single-island-no-sync and the same --rl-checkpoint-store; full-parameter "
                "mode only. convertible: set per island by the launcher for islands declared "
                "--rl-node-class islandM=convertible (colocated LoRA; generates until the "
                "head's scheduler converts it to trainer)"),
    shared_flag("--rl-convertible-node", default=None, metavar="NAME",
                help="--rl-sample-role convertible: this island's node name (set by the "
                "launcher per island; buffer shard and convertible/<NAME>/ channel)"),
    shared_flag("--rl-sampler-max-backlog-rounds", type=int, default=None, metavar="N",
                help="--rl-sample-role sampler: wait while the buffer holds N rounds the trainer "
                "has not read (as of its last release); default 4. Convertible pool: a node "
                "does not start a round while the trainable backlog the head counted is N "
                "rounds; default max(4, --rl-max-policy-age + 2), at least "
                "--rl-max-policy-age + 1"),
    shared_flag("--rl-sampler-min-round-seconds", type=float, default=None, metavar="S",
                help="--rl-sample-role sampler: start one round at most every S seconds (a "
                "throttle; a check run uses it to make the trainer idle); default no throttle"),
    shared_flag("--rl-sampler-fused-stage", action="store_true", default=False,
                help="--rl-sample-role sampler with --tuning lora: check each release's digest "
                "while copying the adapter to local disk (one read of the store instead of "
                "two); default off"),
    shared_flag("--rl-release-reload-latest", action="store_true", default=False,
                help="--rl-sample-role trainer: reload the store and read LATEST before every "
                "policy release and commit LATEST in a second commit (the behavior before "
                "S26 E5); default: the trainer keeps the LATEST it wrote and commits once"),
    shared_flag("--rl-release-inline", action="store_true", default=False,
                help="--rl-sample-role trainer: write each full-weight policy release into the "
                "store before the next training step (the behavior before S27 G-AT); default: "
                "stage it on local disk (every rank writes a share) and upload it while the "
                "next step trains"),
    shared_flag("--rl-release-delta-full-every", type=int, default=0, metavar="N",
                help="--rl-sample-role trainer with staged full-weight releases: upload "
                "each release as the bf16 words that changed since the previous release; every "
                "N-th release (and a delta above half the full size) goes in full. The samplers "
                "rebuild the weights on local disk and check every file's sha256. 0 (default): "
                "every release in full"),
    shared_flag("--rl-sample-wait-s", type=float, default=None, metavar="S",
                help="--rl-sample-role trainer: max seconds to wait for one full round from "
                "the buffer (default 3600)"),
    shared_flag("--rl-sampler-node", default=None, metavar="NAME",
                help="--rl-sample-role sampler: this island's node name (set by the launcher "
                "per island of --rl-island-sample-role; the sampler id that a replacement "
                "incarnation of the island resumes under)"),
    shared_flag("--rl-buffer-shards", default=None, metavar="A,B",
                type=lambda text: [s for s in text.split(",") if s],
                help="--rl-island-sample-role with several samplers: the sampler shards of the "
                "buffer (set by the launcher); each sampler writes its own, the trainer merges them"),
    shared_flag("--rl-sampler-data-slot", default=None, metavar="I/N",
                help="--rl-sample-role sampler with several samplers: draw only the prompt rows "
                "i with i %% N == I (set by the launcher; the samplers draw disjoint prompts)"),
)

_SCHEDULING = (
    # rl-multinode-island Q6: accept a changed GPU uuid pool (rebind) at startup
    shared_flag("--rl-elastic-accept-rebind", action="store_true",
                help="--rl-elastic, multi-node: accept a GPU uuid pool that differs from the "
                "cfg / journal binding (machine replaced) and rebind; off by default the "
                "learner refuses to start on changed GPUs (Q6)"),
    # fork F-R1: declare --rl-elastic-cells as the fork's rollout cells (map rollout_cells).
    shared_flag("--rl-elastic-declare-cells", action="store_true",
                help="--rl-elastic: declare the --rl-elastic-cells names to the fork as its "
                "rollout engine cells (placement map rollout_cells: started on the rollout "
                "GPUs, then stopped on standby GPUs, then stopped unbound; needs fork F-R1)"),
    cli_flag("--rl-island-contract-sha256", default=None,
             type=sha256_hex_arg,
             help="secret-handling-hardening: island contract v3 (backend identity bound to "
             "the LR schedule, the AlgorithmSpec semantic hash and the run code pack "
             "digest) pinned by the head; the syncer refuses islands whose "
             "contract differs, even when they connect first. Default: first island decides"),
    shared_flag("--rl-max-policy-age", type=int, default=0,
                help="agentic-rollout-utilization: how many published policy versions a "
                "trained sample may lag (0, default: deterministic, every sample from the "
                "policy just published). Part of the island contract; a backend that does "
                "not support the requested limit is refused before launch"),
    # S27 rollout ahead (yeto.rl.engine.rollout_ahead): whole batches, weight writes
    # between them; the age limit must cover the lag.
    shared_flag("--rl-rollout-ahead", type=int, choices=(0, 1), default=0, metavar="N",
                help="fixed-partition island: the engines generate the next round's batch "
                "with the published policy while the trainer trains this round and runs "
                "its sync boundary; the weight update runs between two batches, so every "
                "sample lags exactly N versions (round 0: none). Needs "
                "--rl-max-policy-age >= N and a TIS correction; replaces the backend's own "
                "in-batch overlap (fully_async). 0 (default): serial rounds"),
    # S26 E2: the overlap buffer bounds how far generation runs ahead, so it bounds the
    # realised policy age (the backend maps K to its own buffer setting).
    shared_flag("--rl-async-buffer-rounds", type=int, default=None, metavar="K",
                help="--rl-max-policy-age N > 0 with generation that overlaps training "
                "(partitioned island): the finished-group buffer holds K rounds of groups, "
                "so generation runs at most about K rounds ahead and the realised policy "
                "age stays near min(K, N). Default: K = N (a deeper buffer only adds groups "
                "that are too old when trained). Refused without overlap"),
    # S17 N16: LR schedule choice (yeto.rl.engine.run_config.resolve_lr_schedule)
    shared_flag("--rl-lr-schedule", choices=("auto", "linear", "constant"), default="auto",
                help="island optimizer LR schedule: auto (default) = constant for decoupled, "
                "island-pool and asynchronous full-parameter islands (they train until the "
                "syncer stops them), linear decay over global_rounds x optimizer_steps "
                "otherwise; constant = fixed LR in every mode (single island included); "
                "linear is refused where islands train until the syncer stops them"),
    cli_flag("--rl-sync-quorum-theta", type=float, default=None, metavar="F",
             help="membership: merge a round once the arrived islands hold F of the "
             "capacity the round expects (replaces --rl-sync-quorum). Any membership "
             "option (this, --rl-sync-q-min, --rl-sync-soft-deadline-s, "
             "--rl-sync-carry-gamma, --rl-sync-island-lease-s, --rl-sync-max-learners, "
             "--rl-sync-final-grace-s) puts the islands in a pool with dynamic membership; "
             "unset ones stay off. strict-avg, decoupled and full-parameter with "
             "--rl-sync-full-parameter-rounds async"),
    cli_flag("--rl-sync-q-min", type=int, default=None, metavar="N",
             help="membership: minimum arrived islands for a capacity-quorum or "
             "soft-deadline merge (default 1)"),
    cli_flag("--rl-sync-soft-deadline-s", type=int, default=None, metavar="S",
             help="membership: merge what arrived after S seconds even below quorum "
             "(default off)"),
    cli_flag("--rl-sync-carry-gamma", type=float, default=None, metavar="F",
             help="membership: weight discount F^lag of a delta whose base lags the round "
             "(default 1: no discount); the largest lag is --rl-sync-max-base-lag"),
    shared_flag("--rl-sync-island-lease-s", type=float, default=None, metavar="S",
                help="membership: island lease on the syncer (--island-lease-s); an island "
                "silent this long is removed and re-JOINs; islands heartbeat every S/3. "
                "Default 150 in a pool"),
    cli_flag("--rl-sync-max-learners", type=int, default=None, metavar="N",
             help="membership: the syncer admits island ids 0..N-1, so islands may join "
             "after the launch (default: the launched islands; not with the "
             "full-parameter preset, whose session binds the roster)"),
    cli_flag("--rl-sync-final-grace-s", type=int, default=None, metavar="S",
             help="membership: after the last step the syncer waits this long for the "
             "islands still connected to take the final cut, then finishes without the "
             "missing ones (default: --rl-sync-soft-deadline-s; unset both: wait for all)"),
    cli_flag("--rl-elastic-debug-delay-s", default=None, metavar="ISLAND:S[,ISLAND:S]",
             help="membership TEST switch: the given island(s) wait S seconds before "
             "submitting each round's delta (bare S = every island); used to trigger "
             "the soft deadline and carried_over on real hardware"),
    cli_flag("--rl-elastic-debug-pause", default=None, metavar="ISLAND:AFTER_V:S[,...]",
             help="membership TEST switch: after the island applied outer version >= AFTER_V "
             "it stops heartbeating and holds its pushes for S seconds (training continues) "
             "to exercise lease expiry and automatic re-JOIN on real hardware"),
    shared_flag("--rl-syncer-epoch", type=int, default=None, metavar="N",
                help="island pool, or full-parameter async with --rl-peer-recovery: syncer "
                "term (incremented on each syncer restart); passed to the syncer as "
                "--syncer-epoch and to every island's HELLO (default: read from the running "
                "syncer, else 0)",
                island={"default": 0}),
    cli_flag("--rl-peer-recovery", action=argparse.BooleanOptionalAction, default=None,
             help="full-parameter async: an island relaunched after it lost its trainer "
             "state restores the inner optimizer state from a peer island's snapshot, "
             "then takes every fragment from the syncer. Default: on in --rl-sync-mode "
             "async where it applies (full-parameter async rounds, one syncer process, "
             "no broadcast deltas, a snapshot store from --rl-peer-recovery-store or "
             "--rl-checkpoint-store; a fixed roster or an island pool), else off. Same as "
             "--rl-island-state-source peer (on) or catch-up (off)"),
    cli_flag("--rl-island-state-source", choices=("peer", "catch-up", "cut"), default=None,
             help="full-parameter async: where an island that replaces a lost island (a "
             "relaunch, or a new member of an island pool) takes its trainer state. peer: "
             "parameters and inner optimizer state from a healthy peer, then the newer "
             "fragments from the syncer (--rl-peer-recovery); catch-up: every fragment "
             "from the syncer, fresh optimizer state; cut: the island's own latest round "
             "cut (not implemented for synced islands yet, refused). Default: peer where "
             "--rl-peer-recovery applies, else catch-up"),
    cli_flag("--rl-recovery-bound-steps", type=int, default=None, metavar="H",
             help="--rl-peer-recovery: bound on one recovery in syncer steps; the syncer "
             "records a recovery past it as missed (default: measured, not checked)"),
    cli_flag("--rl-recovery-max-donor-tries", type=int, default=None, metavar="N",
             help="--rl-peer-recovery: donors the syncer asks before the island restarts "
             "with fresh optimizer state (default 3)"),
    cli_flag("--rl-recovery-donor-timeout-s", type=int, default=None, metavar="S",
             help="--rl-peer-recovery: wait for one donor snapshot before asking the next "
             "(default 600)"),
    shared_flag("--rl-peer-recovery-store", default=None, metavar="PATH",
                help="--rl-peer-recovery: checkpoint store of the peer snapshots, reached "
                "by every island (a shared path, a Modal Volume or bucket mount, or an "
                "object store through the store environment); default: the "
                "--rl-checkpoint-store"),
    shared_flag("--rl-peer-recovery-ttl-s", type=float, default=None, metavar="S",
                help="--rl-peer-recovery: snapshots older than this are removed (a donor "
                "or recovering island died); longer than a snapshot transfer (default 3600)"),
    shared_flag("--rl-elastic-quorum-timeout-s", type=float, default=None, metavar="S",
                help="--rl-elastic: syncer --quorum-timeout-s (whole seconds), also the "
                "island's strict pause budget input (default: syncer default 900)"),
    shared_flag("--rl-elastic-idle-flow-timeout-s", type=float, default=None, metavar="S",
                help="--rl-elastic: measured network idle-flow timeout capping the pause"),
    cli_flag("--rl-test-inject-start-delay-s", type=float, default=None, metavar="S",
             help="--rl-elastic, TEST ONLY (GPU acceptance fault injection): sleep S "
             "seconds before the island's first fork start_cells; off by default"),
    cli_flag("--rl-test-inject-update-weights-block-s", type=float, default=None,
             metavar="S",
             help="--rl-elastic, TEST ONLY (watchdog fault injection): block up to S "
             "seconds before the island's first member update_weights, failing as soon "
             "as a target engine dies; off by default"),
    # controller D4 timeouts (defaults: controller.Timeouts, T_drain 120 s, T_recovery 900 s)
    shared_flag("--rl-elastic-drain-timeout-s", type=float, default=None, metavar="S",
                help="--rl-elastic: controller T_drain (default 120)"),
    shared_flag("--rl-elastic-recovery-timeout-s", type=float, default=None, metavar="S",
                help="--rl-elastic: controller T_recovery, the REBUILD_OLD budget beyond the "
                "transaction deadline (default 900)"),
    # 3.7 restart recovery: consecutive unverified recoveries before RECOVERY_REQUIRED
    shared_flag("--rl-elastic-max-recovery-attempts", type=int, default=None, metavar="N",
                help="--rl-elastic: consecutive learner-restart recoveries of the committed "
                "membership that may stay unverified before the island is RECOVERY_REQUIRED "
                "(default 3)"),
    shared_flag("--rl-elastic-pause-margin", type=float, default=None, metavar="X",
                help="--rl-elastic: pause budget = X * quorum timeout (default 0.5; "
                "X6 cross-quorum runs only)"),
    # rl-per-island-tasks: each training island trains on its own task subset.
    cli_flag("--rl-island-tasks", default=None, metavar="SPEC",
             help="per-island task lists from --data (a local JSONL whose rows carry "
             "metadata.task_id): 'l0=web1-10,pwn1-10;l1=web11-20,pwn11-20' or a JSON file "
             "{\"0\": [...], \"1\": [...]}; the launcher writes one prompt file per island; "
             "every training island needs a list; eval runs on the eval island only; a JSON "
             "spec file path must not contain '='"),
    # S27 task assignment (P1): which training island reads which rows of --data.
    cli_flag("--rl-data-split", choices=DATA_SPLITS, default=AUTO,
             help="which rows of --data each training island reads (islands that generate "
             "and train; samplers always read their own slots). auto (default): with N >= 2 "
             "such islands, the island with learner id k reads the rows i with i %% N == k, "
             "in data order; after its last row it starts the next pass over its own rows, "
             "reshuffled; a relaunched island keeps its rows. --rl-island-tasks replaces this. "
             "stride: the same as auto, and refused where auto would not split. dispense "
             "(S27 P2, opt-in): the actor syncer hands out the rows, a block per island round; "
             "no row goes to two islands, a fast island claims more, the rows of an island "
             "that leaves go to the others, a relaunched or added island just claims. shared: "
             "every island reads every row in the same order (the old behaviour; controlled "
             "experiments only, the launch warns)"),
    cli_flag("--rl-data-passes", type=int, default=None, metavar="N",
             help="--rl-data-split dispense: the data ends after N passes over the rows "
             "(the islands stop with a clear error); default: unlimited, every pass after the "
             "first reshuffled"),
    cli_flag("--rl-dispense-max-redo", type=int, default=1, metavar="N",
             help="--rl-data-split dispense: how often one row may be given out again after "
             "its group was too stale to train (default 1)"),
    cli_flag("--rl-dispense-lease-s", type=float, default=300.0, metavar="S",
             help="--rl-data-split dispense: an island process silent this long loses its "
             "unreported rows to the other islands (default 300; a relaunched process "
             "takes over at once)"),
    shared_flag("--rl-data-dispense", action="store_true", default=False,
                help="this training island claims its rows from the actor syncer's dispenser "
                "(set by the launcher from --rl-data-split dispense; not a launch option)"),
    shared_flag("--rl-dispense-prefetch-blocks", type=int, default=2, metavar="N",
                help="--rl-data-split dispense: blocks (rounds of rows) an island keeps ready "
                "ahead, so it trains on through a short head outage (default 2)"),
    shared_flag("--rl-data-slot", default=None, metavar="I/N",
                help="this training island reads only the rows i of --data with i %% N == I "
                "(set by the launcher from --rl-data-split; not a launch option)"),
    shared_flag("--rl-data-position-store", default=None, metavar="PATH",
                help="full-parameter island: keep the data position after every round under "
                "PATH, so a relaunched island continues at its next row (set by the launcher "
                "with --rl-checkpoint-store; not a launch option)"),
)

_EVAL = (
    # Ports LoRA training-time heldout eval (forwarded to the learner's
    # --eval-*; the file is shipped inline and checked by SHA256 there).
    cli_flag("--rl-eval-interval", type=int, default=None, metavar="N",
             help="ports: heldout eval every N rollouts (learner --eval-interval); "
             "needs --rl-eval-data/-dataset-name/-samples-per-prompt; off by default"),
    cli_flag("--rl-eval-data", default=None, metavar="PATH",
             help="--rl-eval-interval: local heldout prompt JSONL (distinct from --data)"),
    cli_flag("--rl-eval-dataset-name", default=None, metavar="NAME",
             help="--rl-eval-interval: eval dataset name"),
    cli_flag("--rl-eval-samples-per-prompt", type=int, default=None, metavar="N",
             help="--rl-eval-interval: samples per eval prompt"),
    # Optional eval sampling/length knobs, forwarded verbatim (learner --eval-*).
    cli_flag("--rl-eval-temperature", type=float, default=None, metavar="T",
             help="--rl-eval-interval: eval sampling temperature (0 = greedy)"),
    cli_flag("--rl-eval-top-p", type=float, default=None, metavar="P",
             help="--rl-eval-interval: eval top-p"),
    cli_flag("--rl-eval-max-prompt-len", type=int, default=None, metavar="N"),
    cli_flag("--rl-eval-max-response-len", type=int, default=None, metavar="N"),
    cli_flag("--rl-eval-max-context-len", type=int, default=None, metavar="N"),
)

_RUN_CONTROL = (
    cli_flag(
        "--dry-run",
        action="store_true",
        help=(
            "validate the launch (arguments, provenance, ports algorithm and "
            "capability checks) and print the resource request, algorithm hash "
            "and learner command as JSON; creates no cloud resource"
        ),
    ),
    cli_flag(
        "--rl-stall-timeout",
        type=float,
        default=None,
        help=(
            "ports RL: stop the run (exit 6) when no island event arrives for this many "
            "seconds and not every island finalized (default 900; 0 disables)"
        ),
    ),
    cli_flag(
        "--rl-optimizer-steps",
        type=int,
        default=1,
        help=(
            "ports: optimizer steps per RL round (default 1); must divide "
            "rollout-batch-size * n-samples-per-prompt"
        ),
    ),
    shared_flag(
        "--rl-single-island-no-sync",
        action="store_true",
        help=(
            "ports only: launch exactly one island with no syncer and no outer "
            "sync (the G1 smoke entry for --rl-allow-unverified-mechanism)"
        ),
    ),
    cli_flag(
        "--rl-island-post-cmd",
        default=None,
        metavar="CMD",
        help=(
            "ports single-island smoke only: shell command run on the island's "
            "rank 0 after the learner exits (whatever its exit code), with "
            "stdout+stderr in ~/yeto-output/post-cmd.txt (mirrored into the Modal "
            "tape Volume); the learner's exit code is preserved (S14 A17: run the "
            "fork's Ray tests in the same container)"
        ),
    ),
    shared_flag(
        "--rl-allow-unverified-mechanism",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "ports single-island smoke only: admit an expressible but "
            "undeclared mechanism (recorded in events and provenance)"
        ),
    ),
    shared_flag(
        "--rl-offload-train",
        action="store_true",
        help=(
            "offload the trainer between colocated rollout steps; recommended "
            "for large multi-GPU LoRA runs"
        ),
    ),
)

_ROLLOUT_ENGINE = (
    shared_flag("--use-rollout-routing-replay", action="store_true"),
    shared_flag(
        "--observe-training-entropy",
        action="store_true",
        help="log the training-side policy entropy; metrics only, the loss is "
        "unchanged while the entropy coefficient is 0",
    ),
    shared_flag(
        "--mamba-training-ssm-states-dtype",
        choices=("fp32", "bf16"),
        default=None,
        help="Mamba/hybrid models: dtype of the inter-chunk SSM states in training "
        "(default: the params dtype, bf16; rollouts keep fp32 states). "
        "Changes the training numerics, so it is part of the training contract",
    ),
    shared_flag("--rollout-max-response-len", type=int, default=32768,
                island={"default": None, "required": True}),
    # S26 E6: quantized rollout tiers (yeto.rl.quant.tiers)
    shared_flag("--rl-rollout-quant-tier", type=int, choices=(0, 1, 2, 3), default=0,
                help="sampler precision: 0 BF16 (default); 1 the engine quantizes FFN linears "
                "(dense) or routed experts (MoE) to FP8 itself, FP8 KV cache; 2 the trainer side "
                "converts each release to FP8 W8A8 on the same linears and the sampler island "
                "loads it (needs --rl-island-sample-role and --tuning full); 3 4-bit, interface "
                "only (refused). Router, lm_head, embeddings, norms and attention stay BF16; the "
                "KV cache stays BF16. Needs FP8 cards (H100, H200, L4)"),
    shared_flag("--rl-rollout-quant-fp8-source", choices=("engine", "trainer"), default=None,
                help="tier 1 only: who quantizes (default: trainer where the backend supports it, else engine). "
                "trainer: the engines start on a 128x128 block-FP8 conversion of the base "
                "and the trainer block-quantizes every weight update. engine: the rollout engine "
                "quantizes at load (online FP8). Same ignore list"),
)

_AGENT = (
    shared_flag("--apply-chat-template-kwargs", type=json.loads, default=None),
    shared_flag("--custom-generate-function-path", default=None),
    cli_flag(  # refused since S26 E0.5 (removed Codex/SecRLEnv agents)
        "--codex-reasoning-effort",
        choices=["xhigh"],
        default=None,
        help=(
            "stock Codex harness only: pin Codex reasoning effort to xhigh; "
            "the attested binary is supplied by the direct SSH harness"
        ),
    ),
    shared_flag("--use-session-server", action="store_true"),
    shared_flag("--session-server-ip", default=None),
    shared_flag("--session-server-port", type=int, nargs="+", default=None),
    shared_flag("--tito-model", default=None),
    cli_flag(  # refused since S26 E0.5 (removed Codex/SecRLEnv agents)
        "--codex-exclude-reasoning-from-loss",
        action="store_true",
        help=(
            "stock Codex harness only: generated reasoning (<think> blocks) "
            "gets loss_mask 0; default off = reasoning counts toward the loss"
        ),
    ),
    shared_flag(
        "--agent-max-seq-len",
        type=int,
        default=None,
        help="hard total-token cap for one multi-turn agent trajectory",
    ),
)

_SYNC = (
    cli_flag("--local-rl-rounds-per-sync", type=int, default=1),
    cli_flag(
        "--rl-sync-mode",
        choices=RL_SYNC_MODES,
        default="async",
        help="how a run with two or more islands fills the sync options left unset "
        "(strict-avg, decoupled and full-parameter; one island, SAO, a critic and "
        "--experimental-rl-sync keep the strict values). async (default): "
        "full-parameter asynchronous sharded rounds (--rl-sync-full-parameter-rounds "
        "async, at most 2 fragment rounds in flight), --rl-sync-quorum 1 with the "
        "adaptive grace window (--rl-sync-grace-ms %d cap; --grace-gamma/--grace-tau keep "
        "their defaults, no gamma value is prescribed), --rl-sync-max-base-lag 1, "
        "tokens2-over-steps island weights, rda matrix merge (decoupled, full-parameter), "
        "outer step by local rounds per sync H: plain averaging (lr 1.0, momentum 0) "
        "below H=%d, Nesterov lr 0.7 momentum 0.9 from H=%d on, broadcast overwrite "
        "(merge alpha 0), "
        "int4 push (full-parameter, block 128). strict: the control, every island "
        "per round, equal weights, plain average, f32; its commands equal the "
        "commands from before the modes. An option given explicitly always wins. A "
        "session started in one mode resumes only with the same settings"
        % (ASYNC_GRACE_CAP_MS, ASYNC_NESTEROV_MIN_H, ASYNC_NESTEROV_MIN_H),
    ),
    cli_flag(
        "--rl-sync-preset",
        choices=SYNC_PRESETS,
        type=canonical_sync_preset,
        default="strict-avg",
        help="full-parameter (alias dense-full): full-parameter GRPO as strict "
        "whole-model fragment sweeps "
        "(ports engine, --rl-placement fixed-partition, --rl-parameter-layout-sha256); "
        "with --expert-full-count the swept state is the DeepSeek V4 LoRA hybrid; "
        "sao-streaming-full: full-parameter SAO, actor and critic streams "
        "(ports engine, --rl-placement fixed-partition, --rl-sao-contracts)",
    ),
    cli_flag(
        "--rl-sync-fragment-pipeline",
        type=int,
        default=sync_mode_default(1),
        help="full-parameter only: fragment rounds the syncer gathers at once "
        "(1..--fragments; asynchronous rounds: 1..min(60, --fragments)). Above 1, fragment k+1 uploads "
        "while fragment k merges and downloads (default: min(2, --fragments) with "
        "asynchronous rounds, else 1 = serial)",
    ),
    cli_flag(
        "--rl-sync-push-dtype",
        choices=("f32", "bf16", "int8", "int4", "int4-log"),
        default=sync_mode_default("f32"),
        help="full-parameter only: wire dtype of the islands' pushed outer deltas "
        "(pseudo-gradients). bf16 halves upload bytes, int8 (block absmax scale, see "
        "--rl-sync-push-int8-block) quarters them, int4 (linear, 15 levels per block "
        "absmax scale, see --rl-sync-push-int4-block) and int4-log (the SFT q4 codec: "
        "256-value blocks, log2 levels) cut them to ~1/7.5-1/7.8; the syncer keeps "
        "f32 masters, momentum and f32 FINAL (default: int4 in --rl-sync-mode async, "
        "f32 = exact in --rl-sync-mode strict)",
    ),
    cli_flag(
        "--rl-sync-overlap",
        action="store_true",
        help="full-parameter only (E2 option): train the next round while the "
        "sweep syncs, then rebase the trainer onto the new global weights "
        "(theta <- G_k + local progress). The local weights lag the global policy "
        "by one outer step; rollouts stay on-policy, so --rl-max-policy-age stays 0. "
        "Needs --rl-sync-fragment-pipeline equal to --fragments",
    ),
    cli_flag(
        "--rl-sync-fragment-max-bytes",
        type=int,
        default=2 << 30,
        help="full-parameter only: f32 byte bound of one sync fragment (multiple of 4, "
        "4..2147483648). A parameter above the bound spans several fragments as "
        "dimension-0 row slices, each merged as its own tensor (RDA per slice). "
        "A smaller bound changes the layout hash, so the probe must use the same value",
    ),
    cli_flag(
        "--rl-sync-max-local-rounds-ahead",
        type=int,
        default=None,
        help="full-parameter async rounds only: lead cap K >= 1 (default off). At a "
        "boundary an island waits, still pushing and applying, instead of training a "
        "K-th local round on top of an unmerged push of its own. K=1 starts no round "
        "while one of its pushes is unmerged: with a fixed roster a fast island stays "
        "at most one local round ahead of the slowest island. The wait gives up (and "
        "trains on) after half of --rl-sync-wait-timeout-s since that push. Island-local: "
        "the layout, syncer profile and training contract hashes do not change",
    ),
    cli_flag(
        "--rl-sync-wait-timeout-s",
        type=float,
        default=FULL_PARAMETER_DEFAULT_WAIT_TIMEOUT_S,
        help="full-parameter only: seconds an island waits for a paired broadcast "
        "after its last push of that round, for the terminal barrier and for the "
        "FINAL cut; a value other than 900 also sets the syncer's final-ACK "
        "timeout (which changes the syncer profile hash, not the layout hash)",
    ),
    cli_flag(
        "--rl-sync-push-int8-block",
        type=int,
        default=128,
        help="full-parameter only, with --rl-sync-push-dtype int8: values per f32 "
        "scale (32..1048576; error <= block absmax / 254 per value)",
    ),
    cli_flag(
        "--rl-sync-push-int4-block",
        type=int,
        default=128,
        help="full-parameter only, with --rl-sync-push-dtype int4: values per f32 "
        "scale (even, 64..256; nearest rounding errs <= block absmax / 14 per value)",
    ),
    cli_flag(
        "--rl-sync-push-no-tensor-blocks",
        action="store_true",
        help="full-parameter only, with --rl-sync-push-dtype int8, int4 or int4-log: let "
        "quantization blocks run across a whole fragment (the layout before tensor blocks). "
        "By default blocks restart at every tensor so a tensor replicated on several ranks "
        "quantizes the same on each. A resumed run must repeat the choice it started with",
    ),
    cli_flag(
        "--rl-sync-push-int4-stochastic",
        action="store_true",
        help="full-parameter only, with --rl-sync-push-dtype int4: unbiased stochastic "
        "rounding instead of round-to-nearest (default off)",
    ),
    cli_flag(
        "--rl-sync-push-int4-error-feedback",
        choices=("off", "bf16", "f32"),
        default="off",
        help="full-parameter only, with --rl-sync-push-dtype int4 (nearest rounding): "
        "each island adds the previous int4 rounding error of a fragment to its next "
        "delta, kept in this dtype (bf16 = 2 bytes per pushed parameter on the island "
        "host). Dropped after skipped rounds, re-JOIN and resume (default off)",
    ),
    cli_flag(
        "--rl-sync-bcast-delta-full-every",
        type=int,
        default=0,
        help="full-parameter only: above 0, the syncer sends each new fragment "
        "version as a bf16 delta against the islands' shared f32 anchor, with an "
        "exact f32 refresh every N-th version (N >= 2; digest-checked, a mismatch "
        "reconnects for f32). 0 = always f32",
    ),
    cli_flag(
        "--rl-sync-checkpoint-every",
        type=int,
        default=1,
        help="full-parameter only: syncer checkpoint interval in fragment steps "
        "(1, or a multiple of --fragments). Above 1, only checkpointed steps are "
        "persisted before their broadcast; a crash resumes at the newest checkpoint",
    ),
    cli_flag(
        "--rl-sync-async-checkpoint",
        action="store_true",
        help="full-parameter only: the syncer writes each per-round checkpoint off "
        "its scheduler and broadcasts that round only once the checkpoint is durable",
    ),
    # S27 P1: the outer-merge settings each RL preset used to fix. Unset keeps
    # the preset's value, so the syncer command and profile stay unchanged.
    cli_flag(
        "--rl-sync-init-mode",
        choices=("auto", "local", "upload"),
        default="auto",
        help="full-parameter only (S27): how islands get the version-0 weights. "
        "local = every island keeps the base checkpoint it loaded and reports "
        "per-fragment digests; the syncer checks they agree and sends no initial "
        "broadcast (learner 0 still uploads INIT for the syncer's f32 masters). "
        "upload = learner 0 uploads INIT and the syncer broadcasts it to every "
        "island. auto (default) = local with --rl-sync-full-parameter-rounds async",
    ),
    cli_flag(
        "--rl-sync-full-parameter-rounds",
        choices=FULL_PARAMETER_ROUNDS,
        default=sync_mode_default("sweep"),
        help="full-parameter only (S27 P2): sweep = strict policy sweeps (every "
        "island per fragment round, plain average; the control, default in "
        "--rl-sync-mode strict); async = (default in --rl-sync-mode async) "
        "asynchronous fragment rounds: H = --local-rl-rounds-per-sync, at most "
        "--rl-sync-fragment-pipeline (1 or 2) fragment rounds in flight, the "
        "decoupled preset's outer Nesterov (0.7, 0.9), and the --rl-sync-quorum "
        "/ grace / learner-weight / max-base-lag / outer / matrix-merge options",
    ),
    cli_flag(
        "--rl-sync-quorum",
        type=int,
        default=None,
        metavar="K",
        help="strict-avg / decoupled / full-parameter async: islands whose pushes close a round "
        "(1..islands; default 1 in --rl-sync-mode async, every island in strict)",
    ),
    cli_flag(
        "--rl-sync-grace-ms",
        type=int,
        default=None,
        metavar="MS",
        help="strict-avg / decoupled / full-parameter async: cap of the adaptive post-quorum grace window "
        "(--grace-gamma, --grace-tau; the window adapts to the islands' step time, "
        "the cap only bounds it); default %d in --rl-sync-mode async, 0 = no grace in strict"
        % ASYNC_GRACE_CAP_MS,
    ),
    cli_flag(
        "--rl-sync-grace-min-ms",
        type=int,
        default=None,
        metavar="MS",
        help="strict-avg / decoupled / full-parameter async: lower bound on the post-quorum grace "
        "window (still capped by --rl-sync-grace-ms); default off",
    ),
    cli_flag(
        "--rl-sync-grace-arrival-quantile",
        type=float,
        default=None,
        metavar="Q",
        help="strict-avg / decoupled / full-parameter async: the grace window is at least this "
        "quantile (0..1] of the measured gaps between the islands' pushes of one round, late "
        "pushes included (still capped by --rl-sync-grace-ms); default off",
    ),
    cli_flag(
        "--rl-sync-terminal-grace-ms",
        type=int,
        default=None,
        metavar="MS",
        help="full-parameter async (and other presets with carried late pushes): the last round "
        "of each fragment waits up to this long for every island, so a late delta still reaches "
        "the final parameters (not capped by --rl-sync-grace-ms); default %d in --rl-sync-mode "
        "async with the full-parameter preset, 0 = off" % ASYNC_TERMINAL_GRACE_MS,
    ),
    cli_flag(
        "--rl-sync-learner-weight",
        choices=("equal", "tokens2-over-steps"),
        default=sync_mode_default("equal"),
        help="strict-avg / decoupled / full-parameter async: island weight in the outer merge; "
        "tokens2-over-steps weighs by c_tokens^2 / c_steps (default tokens2-over-steps in "
        "--rl-sync-mode async, equal in strict)",
    ),
    cli_flag(
        "--rl-sync-final-cut",
        choices=("auto", "ack", "download"),
        default="auto",
        help="full-parameter preset: what islands take from the final cut. ack = the final "
        "version manifest only (an island ACKs once its pushes drained; the final parameters "
        "stay in the syncer checkpoint, no island downloads the f32 model); download = every "
        "f32 fragment, applied to the trainer before the ACK (island-side eval or a bitwise "
        "check). auto (default) = ack for --rl-sync-full-parameter-rounds async in "
        "--rl-sync-mode async without --rl-eval-interval, else download",
    ),
    cli_flag(
        "--rl-sync-verify-ahead",
        choices=("auto", "off"),
        default="auto",
        help="full-parameter preset: auto (default) = each island checks a received broadcast's "
        "chunks (finite values, chunk hashes, fragment digest) on a background thread and the "
        "trainer's apply skips that check, when the trainer backend supports it; "
        "off = the trainer checks every chunk in its apply (the applied values are the same)",
    ),
    cli_flag(
        "--rl-sync-expert-token-weights",
        action="store_true",
        help="full-parameter async, MoE models only: weight each expert's tensors in the "
        "outer merge by the tokens the island routed to that expert during the outer "
        "interval, relative to an even split over the layer's experts (an expert an island "
        "never routed to gets weight 0 from that island); multiplies the island weight of "
        "--rl-sync-learner-weight. Islands count routed tokens with a router hook (default off)",
    ),
    cli_flag(
        "--rl-sync-max-base-lag",
        type=int,
        default=sync_mode_default(0),
        metavar="N",
        help="strict-avg / decoupled / full-parameter async: how many rounds a push's base "
        "may lag the fragment being merged (default 1 in --rl-sync-mode async, so a delta "
        "that misses the quorum merges into the next round; 0 = late pushes are refused, "
        "the strict default). One round "
        "merges every fragment once; the launcher passes N x fragments to the syncer, whose "
        "--max-base-lag counts versions",
    ),
    cli_flag(
        "--rl-sync-outer-lr",
        type=float,
        default=None,
        help="strict-avg / decoupled / full-parameter async: outer Nesterov learning rate "
        "(default in --rl-sync-mode async: 1.0 below %d local rounds per sync, 0.7 "
        "from there; strict: the preset's, 1.0 strict-avg, 0.7 decoupled and "
        "full-parameter async)" % ASYNC_NESTEROV_MIN_H,
    ),
    cli_flag(
        "--rl-sync-outer-momentum",
        type=float,
        default=None,
        help="strict-avg / decoupled / full-parameter async: outer Nesterov momentum in [0, 1) "
        "(default in --rl-sync-mode async: 0 below %d local rounds per sync, 0.9 "
        "from there; strict: the preset's, 0.0 strict-avg, 0.9 decoupled and "
        "full-parameter async)" % ASYNC_NESTEROV_MIN_H,
    ),
    shared_flag(
        "--rl-sync-matrix-merge",
        choices=RL_MATRIX_MERGES,
        default=sync_mode_default("avg"),
        island={"default": "avg"},
        help="decoupled and full-parameter async only: avg merges every fragment by weighted averaging "
        "(default in --rl-sync-mode strict); rda (default in --rl-sync-mode async) merges "
        "fragments without embedding tensors by radial-directional averaging, embedding "
        "fragments stay averaged",
    ),
    cli_flag(
        "--rl-initial-adapter",
        default=None,
        help="local final PEFT adapter for a fresh Decoupled RL phase",
    ),
    cli_flag(
        "--rl-initial-adapter-sha256",
        default=None,
        help="optional expected SHA256 for --rl-initial-adapter",
    ),
    cli_flag(
        "--rl-policy-version", choices=["strict"], default="strict"
    ),
    cli_flag(
        "--rl-completed-groups-path",
        default="~/yeto-rl/island-checkpoint.pt",
    ),
    cli_flag("--experimental-rl-sync", action="store_true"),
)


_NEUTRAL_FLAGS: Mapping[str, tuple[RlFlag, ...]] = MappingProxyType({
    "mode": _MODE, "model": _MODEL, "reward": _REWARD, "algorithm": _ALGORITHM,
    "placement": _PLACEMENT, "runtime": _RUNTIME, "elastic": _ELASTIC, "resume": _RESUME,
    "scheduling": _SCHEDULING, "eval": _EVAL, "run_control": _RUN_CONTROL,
    "rollout_engine": _ROLLOUT_ENGINE, "agent": _AGENT, "sync": _SYNC,
})
