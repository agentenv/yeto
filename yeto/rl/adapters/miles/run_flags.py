"""Miles-backend RL flags (the ``run_flags`` role of ``yeto.rl.engine.backends``).

Flags that name the Miles engine, its image, SGLang or Megatron, and every
flag internal to the Miles island entrypoint. Each is defined only here; the
neutral flags and the flag machinery live in ``yeto.rl.engine.run_flags``.
``RL_FLAGS`` maps a theme of ``run_flags.THEMES`` to this backend's flags.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from types import MappingProxyType

from yeto.rl.adapters.miles.pins import MILES_NEXT_IMAGE
from yeto.rl.engine.run_flags import RlFlag, cli_flag, island_flag, shared_flag
from yeto.rl.engine.selection import (
    FULL_PARAMETER_ROUNDS,
    RL_ENGINES,
    SYNC_PRESETS,
    canonical_sync_preset,
)

_MODE = (
    cli_flag("--rl-runtime", choices=["miles"], default="miles"),
    shared_flag(
        "--rl-engine",
        choices=RL_ENGINES,
        default="ports",
        help=(
            "RL engine path: ports, the yeto-owned island loop over upstream "
            "Miles and the only one left (the legacy engine is removed); the "
            "flag stays so existing commands parse"
        ),
    ),
    cli_flag(
        "--rl-image",
        default=None,
        help=(
            "digest-pinned RL image (docker:REPO@sha256:DIGEST); defaults to "
            f"{MILES_NEXT_IMAGE}, upstream Miles with the pinned forks"
        ),
    ),
    shared_flag(
        "--rl-model-recipe",
        choices=["generic", "deepseek-v4-flash", "glm5_next"],
        default="generic",
        help="model-specific Miles memory/kernel contract",
    ),
    shared_flag(
        "--rl-transformer-impl",
        choices=["transformer_engine", "local"],
        default="transformer_engine",
        help="Megatron layer implementation of the glm5_next recipe: transformer_engine "
        "(default) or local, the TE-free fallback (no grouped GEMM, no permute fusion)",
    ),
    shared_flag(
        "--rl-training-attention-backend",
        choices=["auto", "flash", "fused", "unfused"],
        default="auto",
        help="override the recipe's Megatron --attention-backend (auto = recipe "
        "decides); use unfused when the flash-attn cute JIT cannot compile on "
        "the target GPU. Recipes with flash-only kernels refuse non-flash.",
    ),
)

_MODEL = (
    shared_flag(
        "--rollout-model",
        default=None,
        help=(
            "optional inference checkpoint for Miles/SGLang; defaults to --model. "
            "Use this to pair an FP8 rollout checkpoint with a BF16 --model"
        ),
    ),
)

_ALGORITHM = (
    shared_flag(
        "--dynamic-sampling-filter-path",
        default=None,
        help="optional Miles group filter; use the nonzero-reward-variance "
        "filter with oversampling for variance-aware GRPO",
    ),
)

_PLACEMENT = (
    shared_flag("--rl-allow-cross-node-engine-tp", action="store_true",
                help="rl-multinode-island (ruling 2026-10-04 v2): let a rollout engine's TP span "
                "nodes as whole-node replicas (e.g. SGLang TP8 over 2x4); an engine then scales "
                "up/down as one replica; same as cfg parallel.allow_cross_node_engine_tp"),
)

_RUNTIME = (
    # rl-infra-spec 2.3 / 3.x (ports only, off by default; the default island
    # and its Miles argv are unchanged): eval||train overlap and the E1
    # elastic rollout controller.
    shared_flag(
        "--rl-overlap-eval",
        action="store_true",
        help="ports fixed partition: run eval overlapped with train/outer sync "
        "(rl-infra-spec 2.3; needs --rl-eval-interval); off by default",
    ),
    cli_flag("--rl-test-tool-delay-s", type=float, default=None, metavar="S",
             help="ports, TEST ONLY: every training trajectory waits S seconds on a fake "
             "tool call (needs --custom-generate-function-path "
             "yeto.rl.adapters.miles.harness_glue.tool_wait.generate); off by default"),
    # Print the attestation runtime_fingerprint (same Miles argv as the island)
    # and exit before Ray/GPU (ports only).
    shared_flag("--rl-print-attestation-fingerprint", action="store_true",
                help="ports: each island learner builds its Miles argv exactly as a real run, "
                "prints the attestation runtime_fingerprint as one JSON line and exits "
                "before Ray/GPU work. NOTE: the launcher still provisions the island as declared, "
                "so pass CPU-only resources to avoid paying for GPUs"),
    shared_flag("--rl-boot-only", action="store_true",
                help="ports, single island no-sync: the learner runs every check that needs no "
                "torch_dist (model provider view, Miles argv build + parse, --rl-megatron-ref-load "
                "probe), writes ~/yeto-rl/boot_only.json + a FN_BOOT_ONLY_OK line and exits 0 "
                "WITHOUT training, so the job SUCCEEDS and --keep keeps the cluster (S11 fnboot)"),
    cli_flag("--rl-megatron-ref-load", default=None, metavar="DIR",
             help="ports: absolute node-side Megatron torch_dist checkpoint for --ref-load "
             "(required by the Qwen3.8-Flash-Next raw recipe, e.g. "
             "/mnt/yeto-models/torch_dist/qwen3.8-flash-next_torch_dist)"),
    # E2 plan-v2 §0 determinism: Megatron --deterministic-mode + DETERMINISM_ENV
    # in the learner and every Ray worker; off by default.
    shared_flag("--rl-deterministic-trainer", action="store_true",
                dest="rl_deterministic_trainer",
                help="(--deterministic is the backend-neutral name, decoupling 4.9; the "
                "environment comes from yeto.rl.engine.determinism per backend x device) "
                "ports: Megatron --deterministic-mode plus NCCL_ALGO=Ring, "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8, NVIDIA_TF32_OVERRIDE=0 on the learner and "
                "every Ray worker (E2 plan-v2 §0); off by default. SGLang deterministic "
                "inference is --sglang-deterministic-inference (on by default)",
                cli_aliases=("--deterministic",)),
    # ports: Miles rollout/train debug dumps next to the event tape (flat .pt
    # names, so the Modal tape mirror picks them up).
    shared_flag("--rl-debug-dump", action="store_true",
                help="ports: Miles writes each round's rollout samples and train data "
                "(advantages, values, returns) as ~/yeto-output/miles-rollout-<r>.pt / "
                "miles-train-<r>_<rank>.pt (mirrored into the Modal tape Volume); "
                "evidence for offline checks, off by default"),
)

_ELASTIC = (
    # E3 4.7: enable trainer DP-change / role-transfer edges (drops --balance-data,
    # wires MilesTrainerOps and the pool GPU ids)
    shared_flag("--rl-elastic-trainer-edges", action="store_true",
                help="--rl-elastic: enable trainer DP-change / role-transfer edges (4.7): "
                "drops --balance-data (refused by the DP certification) and wires the "
                "trainer ops and pool GPU ids; off by default"),
)

_RUN_CONTROL = (
    cli_flag(
        "--rl-miles-overlay",
        choices=("auto", "off", "critic-c357"),
        default="auto",
        help=(
            "ports only: apply the critic-family Miles patch (fork yeto-critic-c357, "
            "not pushed) over the image's Miles c35702e at island setup. auto (default): "
            "only when the algorithm spec uses a fork-only Miles flag; recorded in the "
            "run manifest as 'image + overlay <sha256>'"
        ),
    ),
)

_ROLLOUT_ENGINE = (
    shared_flag(
        "--rl-activation-recompute",
        action="store_true",
        help="ports: Megatron full uniform activation recompute (one layer per "
        "chunk); trades about one extra forward for activation memory. Off by default",
    ),
    shared_flag(
        "--rl-max-tokens-per-gpu",
        type=int,
        default=None,
        help="ports: Miles dynamic batching, packing samples into micro batches of "
        "at most this many tokens per GPU (must be >= --seq-len). Default: fixed "
        "micro batches",
    ),
    shared_flag(
        "--rl-gradient-accumulation-fusion",
        action="store_true",
        help="ports: Megatron gradient-accumulation fusion (the Miles recipes' "
        "default). Off by default (--no-gradient-accumulation-fusion)",
    ),
    shared_flag(
        "--rl-optimizer-offload-fraction",
        type=float,
        default=0.0,
        help="ports: share (0, 1] of the Adam state (FP32 masters and moments) "
        "that Megatron keeps and updates on the host (--optimizer-cpu-offload, "
        "with the distributed optimizer). Default 0: all on the device",
    ),
    shared_flag(
        "--rl-distributed-timeout-minutes",
        type=int,
        default=10,
        help="bounded Miles distributed/Gloo timeout (default: 10 minutes)",
    ),
    shared_flag(
        "--rollout-num-gpus-per-engine",
        type=int,
        default=1,
        help="GPUs assigned to each colocated SGLang rollout engine",
    ),
    shared_flag("--sglang-tp-size", type=int, default=None),
    shared_flag("--sglang-dp-size", type=int, default=None),
    shared_flag("--sglang-ep-size", type=int, default=None),
    shared_flag("--sglang-mem-fraction-static", type=float, default=0.4),
    shared_flag("--sglang-attention-backend", default=None),
    shared_flag(
        "--sglang-deterministic-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "enable deterministic SGLang kernels when supported; disable for "
            "DeepSeek V4 dsv4/compressed attention"
        ),
    ),
    shared_flag("--sglang-page-size", type=int, default=None),
    shared_flag("--sglang-max-running-requests", type=int, default=None),
    shared_flag("--sglang-chunked-prefill-size", type=int, default=None),
    shared_flag(
        "--rl-sampler-snapshot",
        action="store_true",
        help=(
            "ports, fixed partition: start the sampler (rollout) SGLang engines with "
            "--enable-memory-saver --enable-weights-cpu-backup (GPU snapshot mode); "
            "off by default. Refused unless sampler_snapshot_compat.json records a "
            "pass for the pinned SGLang commit and the NCCL weight-update path"
        ),
    ),
)

_AGENT = (
    shared_flag(
        "--custom-agent-function-path",
        default=None,
        help=(
            "Miles async agent callable used by an agentic custom generate "
            "function (package.module.function)"
        ),
    ),
    shared_flag(
        "--session-server-workers",
        type=int,
        default=None,
        help="Miles session server instance count (default: Miles' 32); lower it "
        "on few-core single-GPU islands to avoid the parallel-import storm "
        "(32 sglang imports can miss the 300 s ready window)",
    ),
    shared_flag(
        "--codex-backend-profile",
        default=None,
        help=(
            "exact signed Codex backend profile; independent from the "
            "family-level Miles --tito-model tokenizer setting"
        ),
    ),
    shared_flag(
        "--tito-allowed-append-roles",
        nargs="+",
        choices=["tool", "user", "system"],
        default=None,
        help="message roles the Miles TITO session may append after generation",
    ),
)

_ISLAND = (
    # Set by the launcher when any island pool option is given (dynamic membership).
    island_flag("--rl-island-pool", action="store_true",
                help="this island joins a dynamic island pool on the syncer: it heartbeats "
                "its lease, re-JOINs after an expiry and may see the outer version jump"),
    island_flag("--model", required=True),
    island_flag("--model-revision", required=True),
    island_flag("--data", required=True),
    # rl-per-island-tasks: SHA256 of this island's own prompt file (checked before training)
    island_flag("--data-sha256", default=None),
    island_flag("--data-revision", default=None),
    island_flag("--eval-data", default=None),
    island_flag("--eval-dataset-name", default=None),
    island_flag("--eval-data-sha256", default=None),
    island_flag("--eval-only", action="store_true"),
    island_flag("--eval-interval", type=int, default=None),
    island_flag("--eval-samples-per-prompt", type=int, default=None),
    island_flag("--eval-temperature", type=float, default=None),
    island_flag("--eval-top-p", type=float, default=None),
    island_flag("--eval-max-prompt-len", type=int, default=None),
    island_flag("--eval-max-response-len", type=int, default=None),
    island_flag("--eval-max-context-len", type=int, default=None),
    # Required unless --rl-single-island-no-sync (checked after parsing).
    island_flag("--syncer", default=None),
    # rl-algo-critic-family 4.2.1: the critic channel's syncer (critic algorithms only).
    island_flag("--critic-syncer", default=None),
    island_flag(
        "--rl-echo-events",
        action="store_true",
        help=(
            "ports: echo every event-tape record to stdout as 'YETO_RL_EVENT <json>' "
            "(the launcher sets it for Modal islands, whose ~/yeto-output is not fetchable)"
        ),
    ),
    island_flag("--learner-id", type=int, required=True),
    island_flag("--num-learners", type=int, default=1),
    island_flag("--learner-generation", type=int, default=0),
    island_flag("--reward-sha256", required=True),
    island_flag("--source-sha256", required=True),
    # rl-identity-tiering L9.5: the run code pack digest; the island refuses to start
    # when its workdir is not that pack.
    island_flag("--code-pack-digest", default=None),
    # L9.12: "release" -- the code is the island image's /opt/yeto/src (checked against
    # the image's release manifest; the digest is the image digest).
    island_flag("--code-pack-mode", choices=["git-patch", "release"], default=None),
    island_flag("--global-rounds", type=int, required=True),
    island_flag(
        "--parameter-mode",
        choices=["lora", "full"],
        default="lora",
    ),
    island_flag(
        "--sync-preset",
        choices=SYNC_PRESETS,
        type=canonical_sync_preset,
        default="strict-avg",
    ),
    island_flag("--fragments", type=int, default=1),
    island_flag("--parameter-layout-sha256", default=None),
    island_flag(
        "--full-parameter-rounds",
        choices=FULL_PARAMETER_ROUNDS,
        default="sweep",
        help="full-parameter preset: sweep = strict policy sweeps; async = "
        "asynchronous sharded rounds (H = --local-horizon, a quorum below the "
        "island count may close a round without this island)",
    ),
    island_flag(
        "--init-mode",
        choices=("auto", "local", "upload"),
        default="auto",
        help="full-parameter: local = keep the loaded base weights as version 0 "
        "after the syncer confirms every island's fragment digests agree; upload "
        "= wait for the syncer's initial broadcast; auto = local for async rounds",
    ),
    island_flag(
        "--sweep-push-dtype",
        choices=("f32", "bf16", "int8", "int4", "int4-log"),
        default="f32",
        help="full-parameter sweep: wire dtype of pushed outer deltas (syncer masters stay f32)",
    ),
    island_flag(
        "--sweep-fragment-max-bytes",
        type=int,
        default=2 << 30,
        help="full-parameter: f32 byte bound of one sync fragment; a larger "
        "parameter spans several fragments as dimension-0 row slices",
    ),
    island_flag(
        "--sweep-max-local-rounds-ahead",
        type=int,
        default=None,
        help="full-parameter async rounds: lead cap K >= 1, wait at a boundary instead of "
        "training a K-th local round on top of an unmerged push (default off)",
    ),
    island_flag(
        "--sweep-wait-timeout-s",
        type=float,
        default=900.0,
        help="full-parameter sync: seconds to wait for a paired broadcast after its "
        "last push, for the terminal barrier and for the FINAL cut",
    ),
    island_flag(
        "--sweep-push-int8-block",
        type=int,
        default=128,
        help="full-parameter sweep, int8 pushes: values per f32 absmax scale",
    ),
    island_flag(
        "--sweep-push-int4-block",
        type=int,
        default=128,
        help="full-parameter sweep, int4 pushes: values per f32 absmax scale",
    ),
    island_flag(
        "--sweep-push-no-tensor-blocks",
        action="store_true",
        help="full-parameter sweep, int8/int4/int4-log pushes: blocks run across the whole fragment",
    ),
    island_flag(
        "--sweep-push-int4-stochastic",
        action="store_true",
        help="full-parameter sweep, int4 pushes: stochastic instead of nearest rounding",
    ),
    island_flag(
        "--sweep-push-int4-error-feedback",
        choices=("off", "bf16", "f32"),
        default="off",
        help="full-parameter sweep, int4 pushes: residual dtype of island-side error feedback",
    ),
    island_flag(
        "--sweep-expert-token-weights",
        action="store_true",
        help="full-parameter async, MoE: count the tokens each router sends to each "
        "expert and send them with every push (the syncer runs --expert-token-weights)",
    ),
    island_flag(
        "--sweep-overlap",
        action="store_true",
        help="full-parameter sweep: train the next round while the sweep syncs, then "
        "rebase onto the new global weights (rollout policy age stays 0)",
    ),
    island_flag(
        "--bcast-delta",
        action="store_true",
        help="strict-avg / full-parameter sweep: accept bf16 downlink deltas against the "
        "shared f32 anchor (the syncer runs --bcast-delta-full-every)",
    ),
    island_flag(
        "--syncer-profile-sha256",
        default=None,
        help="full-parameter mode: the launcher-computed syncer semantic profile hash "
        "(yeto.syncer_profile), bound into every role's HELLO",
    ),
    # The SAO streaming run contract (P0-1.7b.3).
    island_flag(
        "--sao-streaming-runtime",
        default=None,
        metavar="PATH",
        help="--sync-preset sao-streaming-full: the immutable SAO streaming runtime "
        "contract (yeto.rl.sao_streaming_runtime)",
    ),
    island_flag("--sao-streaming-runtime-sha256", default=None, metavar="HEX"),
    island_flag("--sao-context-sha256", default=None, metavar="HEX"),
    island_flag("--pipeline", type=int, default=1),
    island_flag("--local-horizon", type=int, default=1),
    island_flag("--total-fragment-steps", type=int, default=None),
    island_flag("--initial-adapter", default=None),
    island_flag("--initial-adapter-sha256", default=None),
    island_flag("--groups-per-round", type=int, required=True),
    island_flag("--samples-per-group", type=int, required=True),
    island_flag("--optimizer-steps", type=int, required=True),
    island_flag("--codex-harness-contract", type=json.loads, default=None),
    island_flag("--completed-groups-path", required=True),
    island_flag("--event-tape", required=True),
    island_flag("--audit-dir", default=None),
    island_flag("--actor-num-nodes", type=int, required=True),
    island_flag("--actor-num-gpus-per-node", type=int, required=True),
    island_flag("--tensor-parallel", type=int, default=1),
    island_flag("--pipeline-parallel", type=int, default=1),
    island_flag("--expert-parallel", type=int, default=None),
    island_flag("--context-parallel", type=int, default=1),
    island_flag("--rollout-num-gpus", type=int, default=None),
    # fleet-dashboard 2.1/2.2 (ports only): rl_heartbeat / rl_resource_sample
    # periods in seconds; default None = 30 / 60 with --rl-observe-timeline,
    # else off (0 disables explicitly).
    island_flag("--rl-heartbeat-interval", type=float, default=None),
    # rl-multinode-island: GPUs per island node when the island spans nodes (the
    # launcher sends it for --gpu cloud:NxGxgpu with N > 1); None = single node.
    island_flag("--rl-island-gpus-per-node", type=int, default=None),
    # set by the launcher on every island of a --rl-island-placement launch:
    # the placement flags then stay out of the full-parameter training contract
    island_flag("--rl-per-island-placement", action="store_true"),
    # rl-multinode-island Q2 (mixed rollout/trainer nodes): the role -> logical
    # bundle map derived by the launcher from the elastic cfg placement; None =
    # the leading-bundle layout (trainer first, then rollout, then standby).
    island_flag("--rl-island-bundle-map", default=None, metavar="JSON"),
    # decoupling 7.7a: "<vendor>-<card>" from the launcher's card catalog
    # (yeto.hw.catalog); falls back to $YETO_RL_COMPAT_GROUP, unset -> error.
    island_flag("--rl-compat-group", default=None),
    # rl-multinode-island Q4 (C5): off-island copy of the state dir (journal/cuts/ledger),
    # synced after every commit point and restored on an empty state dir (machine replaced)
    island_flag("--rl-elastic-checkpoint-store", default=None, metavar="PATH"),
    # rl-resume-from-checkpoint (S17 C4): resume across launches WITHOUT --rl-elastic
    # (single island, no sync). The store path is on the island (a Modal Volume mount, a
    # bucket mount or a shared path); round cuts + a hashed state snapshot + LATEST go there.
    island_flag("--rl-resume-store", default=None, metavar="PATH"),
    island_flag("--rl-resume-state-dir", default="~/yeto-rl/resume-state", metavar="PATH"),
    # S27 R-2: round cuts of a synced island (strict / decoupled outer sync), written only
    # (the syncer stays authoritative at start): the island's cuts go to
    # <PATH>/island<learner id>; the launcher forwards it from --rl-checkpoint-store
    island_flag("--rl-island-cut-store", default=None, metavar="PATH"),
    # S27 E-3: this trainer lost its state and recovers from a peer snapshot (the
    # launcher sets YETO_RL_PEER_RECOVER=1 on a relaunch instead); local staging
    # directory of object-store snapshot transfers (node-shared on a multi-node island)
    island_flag("--rl-peer-recover", action="store_true", default=False),
    island_flag("--rl-peer-recovery-staging", default=None, metavar="PATH"),
    # S25 C3: a synced trainer island (--rl-sample-role trainer with a syncer): the
    # store of the sample buffer and the policy releases (no round cuts), and the
    # number of synced trainers (its buffer lane is learner id of that count)
    island_flag("--rl-sample-store", default=None, metavar="PATH"),
    island_flag("--rl-sample-lanes", type=int, default=None, metavar="N"),
    island_flag("--lora-r", type=int, default=None),
    island_flag(
        "--lora-targets",
        choices=[
            "auto",
            "attention",
            "attention-routed-experts",
            "all-linear",
        ],
        default=None,
    ),
    island_flag("--inner-lr", type=float, required=True),
    island_flag("--seq-len", type=int, required=True),
    island_flag("--seed", type=int, required=True),
    island_flag("--wan-streams", type=int, default=4),
    island_flag("--wan-stream-schedule", choices=("rr", "adaptive"), default="rr"),
    island_flag("--wan-stall-timeout", type=float, default=0.0),
    island_flag("--wan-probe-min-mbps", type=float, default=0.0),
    island_flag("--wan-probe-min-down-mbps", type=float, default=0.0),
    island_flag("--wan-probe-mib", type=int, default=256),
    island_flag("--miles-root", required=True),
    island_flag(
        "--miles-commit",
        default=None,
        metavar="SHA",
        help="Full commit the --miles-root checkout must be at (default: the pinned "
        "MILES_NEXT_COMMIT); a run whose image carries another Miles commit names it here",
    ),
    island_flag(
        "--rl-expected-algorithm-sha256",
        default=None,
        help="algorithm hash the launcher expects every island to build (ports)",
    ),
    # rl-algo-critic-family 5.3: without --rl-critic-load, a critic algorithm with
    # a warm-up runs stage W itself before the main stage (critic_warmup.run_ports_warmup)
    island_flag("--rl-critic-warmup-dir", default="~/yeto-rl/critic-warmup",
                metavar="DIR",
                help="stage-W product cache (one product per algorithm + initial actor)"),
    # set by run_critic_baseline: no tape echo
    island_flag("--rl-critic-baseline-run", action="store_true",
                help=argparse.SUPPRESS),
    island_flag("--megatron-ref-load", default=None),
    island_flag("--trust-remote-code", action="store_true"),
)


# Full-parameter run inputs produced by Miles probes (P0-1.7b.1, P0-1.7b.3).
_SYNC = (
    cli_flag(
        "--rl-sao-contracts",
        default=None,
        metavar="MANIFEST",
        help="sao-streaming-full only: the final manifest written by "
        "tools/probes/build_tbench21_sao_streaming_contracts.py finalize (frozen "
        "syncer profile, one runtime contract and layout attestation per island)",
    ),
    cli_flag(
        "--rl-parameter-layout-sha256",
        default=None,
        metavar="HEX",
        help="full-parameter preset only: the full-parameter layout hash reported by the Miles "
        "layout probe (yeto.rl.adapters.miles.models.full_parameter_manifest_probe); "
        "--fragments must equal that layout's fragment count",
    ),
)


RL_FLAGS: Mapping[str, tuple[RlFlag, ...]] = MappingProxyType({
    "mode": _MODE, "model": _MODEL, "algorithm": _ALGORITHM, "placement": _PLACEMENT,
    "runtime": _RUNTIME, "elastic": _ELASTIC, "run_control": _RUN_CONTROL,
    "rollout_engine": _ROLLOUT_ENGINE, "agent": _AGENT, "sync": _SYNC, "island": _ISLAND,
})
