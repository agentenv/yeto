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

CLI = "cli"
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
    shared_flag(
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
    cli_flag("--rl-island-use-gpus-per-node", type=int, default=None, metavar="M",
             help="rl-multinode-island: provision the --gpu machines (N GPUs/node, billed) but "
             "give the island only local GPUs 0..M-1 on every node (M <= N): ray start "
             "--num-gpus=M, CUDA_VISIBLE_DEVICES=0..M-1, and every layout/placement/gpu_pool "
             "rule uses M (e.g. m5 TP8 over 2x4 on 2x8xH100 machines); unset = all N"),
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
    # rl-resume-from-checkpoint (S17 C4)
    shared_flag("--rl-cut-every", type=int, default=None,
                help="with --rl-checkpoint-store: keep a round cut every N rounds (plus the "
                "last round and our own stop); default 1"),
    shared_flag("--rl-cut-keep", type=int, default=None,
                help="with --rl-checkpoint-store and no --rl-elastic: keep the newest K cuts "
                "(default 2); older ones are deleted only after the new LATEST is committed"),
    shared_flag("--rl-resume-allow-config-change", action="store_true",
                help="resume a cut even though the run configuration differs (the "
                "differences are written to the rl_resume event); refused by default"),
    shared_flag("--rl-stop-after-rounds", type=int, default=None,
                help="with --rl-checkpoint-store and no --rl-elastic: stop after N rounds "
                "with a final cut (our own stop; the next launch resumes)"),
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
    # rl-inter-island-scheduling 0.13: explicit old/new outer-scheduling choice.
    # 3.8 strict pause budget inputs (defaults: syncer 900 s, margin 0.5).
    # rl-inter-island-scheduling 0.13 (launcher forwards only "elastic")
    shared_flag("--rl-island-scheduling", choices=("legacy", "elastic"), default="legacy",
                help="legacy (default): existing syncer behaviour, fixed members, every "
                "island must arrive; elastic: inter-island scheduling (capacity-weighted "
                "stepping, late deltas carried over with a discount, join/leave)"),
    cli_flag("--rl-island-contract-sha256", default=None,
             type=sha256_hex_arg,
             help="secret-handling-hardening: island contract (backend identity bound to "
             "the LR schedule and the AlgorithmSpec hash) pinned by the head; the syncer refuses islands whose "
             "contract differs, even when they connect first. Default: first island decides"),
    shared_flag("--rl-max-policy-age", type=int, default=0,
                help="agentic-rollout-utilization: how many published policy versions a "
                "trained sample may lag (0, default: deterministic, every sample from the "
                "policy just published). Part of the island contract; a backend that does "
                "not support the requested limit is refused before launch"),
    # S17 N16: LR schedule choice (yeto.rl.engine.run_config.resolve_lr_schedule)
    shared_flag("--rl-lr-schedule", choices=("auto", "linear", "constant"), default="auto",
                help="island optimizer LR schedule: auto (default) = constant for decoupled "
                "and elastic islands, linear decay over global_rounds x optimizer_steps "
                "otherwise; constant = fixed LR in every mode (single island included); "
                "linear is refused for decoupled/elastic"),
    cli_flag("--rl-quorum-theta", type=float, default=None, metavar="F",
             help="elastic: step when arrived capacity >= F x total (default 0.75)"),
    cli_flag("--rl-carry-gamma", type=float, default=None, metavar="F",
             help="elastic: discount per outer step for a late delta (default 0.5)"),
    cli_flag("--rl-soft-deadline-s", type=int, default=None, metavar="S",
             help="elastic: soft deadline of an outer round (default: the quorum "
             "timeout, syncer default 900)"),
    cli_flag("--rl-elastic-debug-delay-s", default=None, metavar="ISLAND:S[,ISLAND:S]",
             help="elastic TEST switch: the given island(s) wait S seconds before "
             "submitting each round's delta (bare S = every island); used to trigger "
             "the soft deadline and carried_over on real hardware"),
    cli_flag("--rl-elastic-debug-pause", default=None, metavar="ISLAND:AFTER_V:S[,...]",
             help="elastic TEST switch: after the island applied outer version >= AFTER_V "
             "its link to the syncer goes silent for S seconds (no heartbeats, sends and "
             "received frames held; training continues) to exercise lease expiry and "
             "automatic re-JOIN on real hardware"),
    cli_flag("--rl-final-grace-s", type=int, default=None, metavar="S",
             help="elastic: after total steps the syncer keeps answering FINISHED for "
             "this long so late islands end cleanly (default: the soft deadline)"),
    cli_flag("--rl-island-lease-s", type=float, default=None, metavar="S",
             help="elastic: island lease on the syncer (--island-lease-s); an island "
             "silent this long is removed. Default 150 (heartbeats every 50 s on their own thread)"),
    cli_flag("--rl-q-min", type=int, default=None, metavar="N",
             help="elastic: minimum arrived islands for a step (default 1)"),
    shared_flag("--rl-syncer-epoch", type=int, default=None, metavar="N",
                help="elastic: syncer term (incremented on each syncer restart); passed to "
                "the syncer as --syncer-epoch and to every island for JOIN (default 0)",
                island={"default": 0}),
    cli_flag("--rl-max-carry-lag", type=int, default=None, metavar="N",
             help="elastic: oldest late delta still carried over, in outer steps (default 2)"),
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
    shared_flag("--rollout-max-response-len", type=int, default=32768,
                island={"default": None, "required": True}),
)

_AGENT = (
    shared_flag("--apply-chat-template-kwargs", type=json.loads, default=None),
    shared_flag("--custom-generate-function-path", default=None),
    shared_flag(
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
    shared_flag(
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
        "--rl-sync-preset",
        choices=["strict-avg", "decoupled", "dense-full", "sao-streaming-full"],
        default="strict-avg",
        help="dense-full: full-parameter GRPO as strict whole-model fragment sweeps "
        "(ports engine, --rl-placement fixed-partition, --rl-parameter-layout-sha256); "
        "with --expert-full-count the swept state is the DeepSeek V4 LoRA hybrid; "
        "sao-streaming-full: full-parameter SAO, actor and critic streams "
        "(ports engine, --rl-placement fixed-partition, --rl-sao-contracts)",
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
