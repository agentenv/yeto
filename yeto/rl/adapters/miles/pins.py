"""Miles / SGLang pins, bundle and image constants (yeto-framework-decoupling 5.3, audit L4).

Moved from ``yeto/rl/__init__.py``; ``yeto.rl`` re-exports every name so
``from yeto.rl import MILES_NEXT_IMAGE`` keeps working.
"""

from typing import NamedTuple


class MilesRevisionPins(NamedTuple):
    """Expected Miles checkout identity for one ``--rl-engine`` path."""

    repository: str
    commit: str

MILES_REPOSITORY = "https://github.com/agentenv/miles"
MILES_BASE_COMMIT = "6062afe0a9d5d6471e8395dedc81c78dd9f4a84f"
MILES_COMMIT = "ae475060fa670145aef75d678809039ae999cb97"
MILES_BUNDLE_PATH = "yeto/rl/vendor/miles-qwen38.bundle"
MILES_BUNDLE_SHA256 = "da3464d3c389f7e2cb3e390119b3c42c2f130d94a0711f9c95608f3243826cc6"
MILES_PEFT_VERSION = "0.20.0"
SGLANG_REPOSITORY = "https://github.com/agentenv/sglang"
SGLANG_COMMIT = "e1b57eb8e7749235c987cc6b1b2824ce3265369b"
MILES_IMAGE = (
    "docker:ghcr.io/agentenv/miles@sha256:"
    "80c20538b63f76defde06ad5d4cfa564ae6f261110696eb1864470cb835e1590"
)

# Ports engine (``--rl-engine ports``): upstream Miles plus yeto compatibility
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
MILES_NEXT_COMMIT = "64b591a4bec1ffa37fb089d3e3b99c84773b1ffc"
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
MILES_LEGACY_PINS = MilesRevisionPins(MILES_REPOSITORY, MILES_COMMIT)
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
# Tag 64b591a-2fa8801; (S19 #1: critic family + algo-supplement metrics baked in; previous ddce209-2fa8801 @sha256:9c252c38...) (S18 ARU-3: Miles s18-agentic-suspend = 2f7871fb2 + --agentic-suspend-between-turns; previous 2f7871f-2fa8801 @sha256:62b4f164..., efbbc63-2fa8801 @sha256:a7990076..., 8bc5223-4e4148f @sha256:4aeafd77...) ghcr.io/michaellchung (public package); linux/amd64 only.
MILES_NEXT_IMAGE = (
    "docker:ghcr.io/michaellchung/yeto-miles-ports@sha256:"
    "fa2413be4c4fcf066437f946365f01392d6f884002f8fa19c770c12c08f2acf0"
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


def default_rl_image(rl_engine: str) -> str:
    """The digest-pinned ``--rl-image`` default for an RL engine.

    Legacy keeps the agentenv fork image (private ghcr.io/agentenv).  Ports
    uses MILES_NEXT_IMAGE: upstream Miles' image with the pinned
    michaellchung forks preinstalled, public on ghcr.io/michaellchung (a
    registry login is injected only via SKYPILOT_DOCKER_* or
    --rl-image-private; see yeto.launcher.registry_login_for).
    """

    return MILES_NEXT_IMAGE if rl_engine == "ports" else MILES_IMAGE
