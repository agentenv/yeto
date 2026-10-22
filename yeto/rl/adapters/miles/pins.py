"""Miles / SGLang pins, bundle and image constants (yeto-framework-decoupling 5.3, audit L4).

Moved from ``yeto/rl/__init__.py``; ``yeto.rl`` re-exports every name so
``from yeto.rl import MILES_NEXT_IMAGE`` keeps working.
"""

from typing import NamedTuple


class MilesRevisionPins(NamedTuple):
    """Expected Miles checkout identity (``revision.verify_miles_revision``)."""

    repository: str
    commit: str

MILES_PEFT_VERSION = "0.20.0"

# Ports engine (the only Miles engine): upstream Miles plus yeto compatibility
# commits on michaellchung/miles ``yeto/ports``; fetched directly, no bundle.
MILES_NEXT_REPOSITORY = "https://github.com/michaellchung/miles"
MILES_NEXT_UPSTREAM_COMMIT = "9e4260de047a704208535c0e90c531929879ab40"
# yeto/ports (= yeto-elastic-m1-m6 after four review rounds): run_plugin,
# --worker-dynamic-port-start, the elastic M1-M6 changes and the LoRA bridge
# calculate_per_token_loss fix on top of the base, plus (5c1b49eb) the 2b
# loss variants: --policy-loss-variant {policy_loss,cispo,sapo,gmpo},
# --sapo-tau-{pos,neg}, --gmpo-log-clip-{low,high}; plus (2f23a0fc) F-R1:
# placement map rollout_cells, deferred/unbound cells, describe_cells,
# public slice_pg_info, bundle-free check before any cell starts; plus
# (fb04d6ff) M5: LoRA DP-invariant restore no longer drops DistOpt exp_avg/exp_avg_sq,
# and (e3a11ab3) its cross-optimizer error text/docstring and tests.
# image-m3a27b (c35702e = merge of m3-qwen4exp-lora ab904f43a [Qwen3.8-Next LoRA layout]
# + fork-a27-worker-loss 857fc9592 [workers_lost cells, bounded weight-update group
# rendezvous --update-weight-group-timeout-s; A27-2 ExternalFailureError /
# RolloutEngineJoinError: an engine-side failure no longer kills the trainer]
# on yeto/ports e3a11ab38).
# s16-raw-lora-disagg (8bc52237a = c35702e + broadcast LoRA gathers the adapter across PP onto one sender,
# actor allows --megatron-to-hf-mode raw for non-colocated LoRA; MILES-RAW-LORA-DISAGG-S16.md plan A).
# s18-abort-discard-stats (efbbc63ea = 8bc52237a + rollout abort tallies discarded
# groups/samples/response tokens without partial rollout; S18 agentic-rollout-util).
# 2f7871fb2 = efbbc63ea + the same tally on the train path actually in use
# (inference_rollout_train.abort; efbbc63ea only patched sglang_rollout.abort) and a
# failed surplus group no longer fails the abort (S17 N17 A/B).
# s19-critic-algosup (64b591a4b = ddce20992 + critic family + algo-supplement metrics; S19 #1,
# local fork branch, not pushed; built into the image from the local clone).
# s23-full-parameter-rpc (56068bd56 = 64b591a4b + full-parameter shard state,
# transport and TrainerController methods ported from agentenv/miles e25048edd;
# P0-1.7b.1, needed by full-parameter DiLoCo on the ports engine).
# s23-sao-streaming-roles (386ae3b07 = 56068bd56 + LoRA-hybrid full-parameter
# transport (95533a0cd, 5f7ac4e32, 587bd258a; P0-1.7b.2) + SAO streaming
# primitives ported from agentenv/miles d5eb5e82a: content identity, per-role
# chunks, critic group RPC (P0-1.7b.3)).
# s24-miles-import2 (44d41efd0 = 17aeb5cb5 + the megatron_utils package
# import stays light, so RayWorkerManager no longer loads megatron.bridge
# (rl-trainer-cold-start-cost L2.2)).  17aeb5cb5 = 386ae3b07 + light Ray actors
# stay off megatron_utils (L2.1) + the master port is held until use (S23 defect 9).
MILES_NEXT_COMMIT = "ed5f64fd7a183cccc8ab7ca3da90fba570108c2b"
# sgl-project/sglang ``sglang-miles`` head when radixark/miles@9e4260d was
# committed (upstream's Dockerfile follows that branch unpinned).
SGLANG_NEXT_REPOSITORY = "https://github.com/michaellchung/sglang"
SGLANG_NEXT_UPSTREAM_COMMIT = "571212b636baca45e10fa3b4da11a289123f3235"
# yeto/lora-checksum (a1240c530 = 9f29303 + WeightChecker checksum covers LoRA adapter A/B);
# yeto/ports 9f29303: the ported agentenv/sglang patches (see sglang-patch-port.md).
# m3-qwen4exp-lora (4e4148f1b = a1240c530 + Qwen4ExpForConditionalGeneration LoRA hooks).
# n17-qwen3coder-single-call (2fa880182 = 4e4148f1b + qwen3_coder honors
# parallel_tool_calls=false under tool_choice=auto: at most one tool call; S17 N17).
SGLANG_NEXT_COMMIT = "2fa880182eefdb31e64d8ea70317be27f9f32121"
# Fork lineage on michaellchung/miles yeto/ports, oldest first; the single
# source for every "this fork commit carries X" set (rl-code-quality-cleanup
# 0.7).  Each entry is the previous one plus the change named here; the newest
# is MILES_NEXT_COMMIT.  Append a commit here and the loss-variant set
# (algos/loss_variants.FORK_COMMITS) and the Miles declaration pins
# (adapters/miles/entry.py) include it with no other edit.
MILES_FORK_LINEAGE: tuple[str, ...] = (
    # 0af62f4d: LoRA bridge honours calculate_per_token_loss (g1c).
    "0af62f4d48ed6a5b185c257578d8f7e22312aa87",
    # 5c1b49eb: fast-forward of 0af62f4d with the 2b loss variants
    # (--policy-loss-variant, --sapo-tau-*, --gmpo-log-clip-*).
    "5c1b49ebccbc7508c1d9ef89eacc2db3e448b6ba",
    "2f23a0fca9b80f6a7300da401703c343014b03c0",  # + F-R1 (placement/worker files only)
    "fb04d6ffa30edc28c7ba0a2e88802a84bbbd28f9",  # + M5 (LoRA dp_invariant_state.py only)
    "e3a11ab38cbb7fd911b23fdd62a4eb6dfbb1c841",  # + M5 error text/docstring/tests
    # 1023269 = e3a11ab3 + M3 (Qwen3.8-Next LoRA layout) + A27 (worker loss / bounded
    # weight-update group); no loss_hub file.
    "1023269412bf4e54a1d95c8d2deaee795871aa72",
    # c35702e = + A27-2 (857fc9592) engine-failure propagation; no loss path.
    "c35702eefcf2862cee155e46870e6ad30568d2c6",
    # 8bc52237a = + s16-raw-lora-disagg; no loss path.
    "8bc52237a1102306abd8f89a2ea2090aa2df6850",
    # efbbc63ea = + s18-abort-discard-stats; rollout bookkeeping only.
    "efbbc63ea0e8c6dcbc200ad5df5bc56badfce7bf",
    # 2f7871fb2 = + inference_rollout_train.abort tally; rollout bookkeeping only.
    "2f7871fb2e80d37420b16ef81f001c7b0ba8e32e",
    # ddce20992 = + s18-agentic-suspend; rollout scheduling only.
    "ddce20992c9ee82270a22e259e9e2bf55f3053d1",
    # 64b591a4b = + critic family + rl-algo-supplement metrics (S19 #1); new flags
    # default off, loss bitwise unchanged with defaults.
    "64b591a4bec1ffa37fb089d3e3b99c84773b1ffc",
    # 56068bd56 = + full-parameter shard state and transport (P0-1.7b.1); no loss path.
    "56068bd561df6658e929fe1b2ca051b5385d3d9b",
    # 95533a0cd / 5f7ac4e32 / 587bd258a = + LoRA actor opt-in to the full-parameter
    # transport, base-weight publish for a LoRA hybrid (P0-1.7b.2); no loss path.
    "95533a0cd2097175980afa4a9c43a817512ebca6",
    "5f7ac4e32ba3131ebf92a77bd6571c467bfa6220",
    "587bd258a350cc4757e3e774436eb48085d808cc",
    # 386ae3b07 = + SAO streaming primitives (P0-1.7b.3); no loss path.
    "386ae3b07138ccd45ce4a07fc0383587f7968a3a",
    # 17aeb5cb5 = + lazy imports for light actors, master port held (S24 L2.1); no loss path.
    "17aeb5cb51635b9ba6ef4ca10db95762695c7e03",
    # 44d41efd0 = + megatron_utils runtime patches moved out of the package init (S24 L2.2); no loss path.
    "44d41efd0a0428cb12b4ee34422de6f5e1ccb9d9",
    # 9462de532 = + FP8 rollout honors the FP8 ignore list, --rollout-hf-checkpoint (S26 E6); no loss path.
    "9462de532f747cf688e090bbaf6b24f80e378fe8",
    # ed5f64fd7 = + DP-invariant state export/restore carries the CPU-offload optimizer state (S26 E6 B8); no loss path.
    "ed5f64fd7a183cccc8ab7ca3da90fba570108c2b",
)

# First Miles commit whose full-parameter sync reads distributed-optimizer
# masters (and the HybridDeviceOptimizer's offloaded FP32 copies) at DP=1
# (S27 G-B; local fork branch s27-distopt-fp on ed5f64fd7, no image yet).
# Append it to MILES_FORK_LINEAGE together with the image re-pin.
MILES_DISTOPT_FULL_PARAMETER_COMMIT = "56190a5a86d14f1eb96d592038d76a9ad8651b0c"


def miles_fork_lineage_from(commit: str) -> frozenset[str]:
    """``commit`` and every later commit of :data:`MILES_FORK_LINEAGE`."""
    return frozenset(MILES_FORK_LINEAGE[MILES_FORK_LINEAGE.index(commit):])


# Critic-family fork (a separate chain, local, on c35702e), oldest first; the
# newest entry is the critic fork pin (algos/critic_fork.CRITIC_FORK_PIN).
CRITIC_FORK_LINEAGE: tuple[str, ...] = (
    "6e7365b602ecf5d83d16169e87f4d78f4daf5f86",  # yeto-critic-c357
    "6574a9c82edeed17f4b00bc64db6c7cd6b45cb39",  # yeto-critic-c357-tied: + value head on language_model
)

# torch_memory_saver preload patch used by the SSH harness (was a separate pin
# in yeto/rl/ssh_harness.py).
TMS_PRELOAD_PATCH_REPOSITORY = "https://github.com/fzyzcjy/torch_memory_saver"
TMS_PRELOAD_PATCH_BASE_COMMIT = "fdb09ad342f8d150e11afe719f798453cef742ad"
TMS_PRELOAD_PATCH_COMMIT = "d7a3a51bce723a80736dabf233017b927de03df8"
TMS_PRELOAD_BASE_BINARY_SHA256 = (
    "c26551247616cabb0bb79dbd26abb61a688a4bb47c9b6a6f9873cfc98093640d"
)
MILES_NEXT_PINS = MilesRevisionPins(MILES_NEXT_REPOSITORY, MILES_NEXT_COMMIT)
# radixark/miles:dev multi-arch index (upstream docker/Dockerfile at
# radixark 9e4260d, sglang v0.5.20 base): the base MILES_NEXT_IMAGE extends
# (its linux/amd64 manifest).  Public.
MILES_NEXT_BASE_IMAGE = (
    "docker:docker.io/radixark/miles@sha256:"
    "90940828dcd4d54fd907ff668b43537cbd94778047580e4160d6560af548b74d"
)
# The base plus the pinned forks: MILES_NEXT_COMMIT at /root/miles and
# SGLANG_NEXT_COMMIT at /sgl-workspace/sglang, both the base's editable
# installs (pure-Python overlay, scripts/build_miles_ports_image.sh;
# docker/miles-ports/Dockerfile).  /opt/yeto/image-manifest.json records
# every SHA.  PUBLIC on ghcr.io/michaellchung since 2026-10-07 (anonymous
# pull); a private image needs SKYPILOT_DOCKER_{USERNAME,PASSWORD,SERVER} or
# --rl-image-private (yeto.launcher.registry_login_for; read:packages token).
# Tag ed5f64f-2fa8801; (S27 IMG: Miles ed5f64fd7, host-Adam round cuts (B8); previous 44d41ef-2fa8801 @sha256:5b9dc2a9...; build record openspec/changes/rl-infra-spec/evidence/ports-image/2026-10-22-ed5f64f-2fa8801/). Earlier: tag 44d41ef-2fa8801; (S24 L2.2: Miles s24-miles-import2 and the lazy-modelopt
# Megatron-LM patch in the layer; previous 17aeb5c-2fa8801 @sha256:b304f857...;
# build record openspec/changes/rl-infra-spec/evidence/ports-image/
# 2026-10-14-44d41ef-2fa8801/). Earlier: tag 17aeb5c-2fa8801; (S24 L2.1-L2.4: Miles s24-miles-import, peft 0.20.0 and the
# lazy-modelopt Megatron-Bridge __init__ in the layer; previous 386ae3b-2fa8801
# @sha256:bcd5ffd3...; build record openspec/changes/rl-infra-spec/evidence/
# ports-image/2026-10-14-17aeb5c-2fa8801/). Earlier: tag 386ae3b-2fa8801; (S23 P0-1.11 GPU run: Miles s23-sao-streaming-roles; previous 64b591a-2fa8801 @sha256:fa2413be...; build record openspec/changes/rl-infra-spec/evidence/ports-image/2026-10-14-386ae3b-2fa8801/). Earlier: (S19 #1: critic family + algo-supplement metrics baked in; previous ddce209-2fa8801 @sha256:9c252c38...) (S18 ARU-3: Miles s18-agentic-suspend = 2f7871fb2 + --agentic-suspend-between-turns; previous 2f7871f-2fa8801 @sha256:62b4f164..., efbbc63-2fa8801 @sha256:a7990076..., 8bc5223-4e4148f @sha256:4aeafd77...) ghcr.io/michaellchung (public package); linux/amd64 only.
MILES_NEXT_IMAGE = (
    "docker:ghcr.io/michaellchung/yeto-miles-ports@sha256:"
    "a476925466dad4df51db6955006a2230f9f88e800e44d529ac5d3f8820b19d58"
)
MILES_NEXT_IMAGE_MANIFEST = "/opt/yeto/image-manifest.json"
# Nebius VM images whose /var/lib/docker already holds a docker image's
# layers (scripts/bake_nebius_image.sh; COLDSTART-PLAN.md #3), keyed by that
# image's digest, then region.  sky still runs `docker pull <digest>`, which
# is then a no-op.  Only an exact digest match is used: a stale entry for an
# older pin would just cost a full pull on top of a non-default base disk, so
# the launcher warns and keeps the stock image instead.  Base disk: Nebius
# public family ubuntu24.04-cuda13.0 (sky's own GPU default), so the NVIDIA
# driver and container toolkit are unchanged.
NEBIUS_BAKED_IMAGES: dict[str, dict[str, str]] = {
    # MILES_NEXT_IMAGE c35702e-4e4148f; baked 2026-10-05 (cs2-bake.log).
    "sha256:37ac689e29caeecf9faf8587a3ad58c154ecffc7798711d5bd59d792d002b9f9": {
        "eu-north1": "computeimage-e00xts577c333r08gv",
    },
}
# Where MILES_NEXT_IMAGE installed the SGLang fork (editable).
MILES_NEXT_IMAGE_SGLANG_ROOT = "/sgl-workspace/sglang"


def default_rl_image() -> str:
    """The digest-pinned ``--rl-image`` default.

    MILES_NEXT_IMAGE: upstream Miles' image with the pinned michaellchung
    forks preinstalled, public on ghcr.io/michaellchung (a registry login is
    injected only via SKYPILOT_DOCKER_* or --rl-image-private; see
    yeto.launcher.registry_login_for).
    """

    return MILES_NEXT_IMAGE
