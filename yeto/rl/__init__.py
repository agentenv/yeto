"""Pinned Miles reinforcement-learning integration."""

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
MILES_NEXT_COMMIT = "c35702eefcf2862cee155e46870e6ad30568d2c6"
# sgl-project/sglang ``sglang-miles`` head when radixark/miles@9e4260d was
# committed (upstream's Dockerfile follows that branch unpinned).
SGLANG_NEXT_REPOSITORY = "https://github.com/michaellchung/sglang"
SGLANG_NEXT_UPSTREAM_COMMIT = "571212b636baca45e10fa3b4da11a289123f3235"
# yeto/lora-checksum (a1240c530 = 9f29303 + WeightChecker checksum covers LoRA adapter A/B);
# yeto/ports 9f29303: the ported agentenv/sglang patches (see sglang-patch-port.md).
# m3-qwen4exp-lora (4e4148f1b = a1240c530 + Qwen4ExpForConditionalGeneration LoRA hooks).
SGLANG_NEXT_COMMIT = "4e4148f1b4fe9f05973da0d1e5cfe237512d5155"
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
# Tag c35702e-4e4148f; linux/amd64 only.
MILES_NEXT_IMAGE = (
    "docker:ghcr.io/michaellchung/yeto-miles-ports@sha256:"
    "37ac689e29caeecf9faf8587a3ad58c154ecffc7798711d5bd59d792d002b9f9"
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

SECRLENV_AGENT_PATH = "yeto/rl/harness/codex/agent.py"
SECRLENV_AGENT_SHA256 = (
    "fef958c32d27af124827b17369bb82557e96946c4b6f9b74c67f084a533f7c81"
)
SECRLENV_AGENT = "yeto.rl.harness.codex.agent.run"
SECRLENV_REWARD = "yeto.rl.harness.codex.reward:reward_func"
SECRLENV_GROUP_FILTER = "yeto.rl.harness.codex.reward.check_group"
SECRLENV_GENERATE = "yeto.rl.harness.codex.generate.generate"
SECRLENV_GENERATE_SHA256 = (
    "1c79b0e678b8681b5bd6221b5a4bbc6adbe7a1413b688e248cb930eb3e456cca"
)
SECRLENV_ZERO_VARIANCE_REPLACEMENTS = 0
SECRLENV_INFRASTRUCTURE_REPLACEMENTS = 1

# Stock Codex is part of the signed Yeto security-environment harness.  These
# pins identify the official Linux artifact, not the controller's host binary.
CODEX_HARNESS_AGENT = "yeto.rl.harness.codex.codex_harness_agent.run"
CODEX_HARNESS_AGENT_PATH = "yeto/rl/harness/codex/codex_harness_agent.py"
CODEX_HARNESS_AGENT_SHA256 = (
    "24fd17e2027393b2a414f190acdfe92e103fd75429729e2e06f28bc31b8572e8"
)
CODEX_BASE_INSTRUCTIONS_SHA256 = (
    "1c183656ca1319142cba9e76baa199b7ab59f770a51a76660622a087e74ba846"
)
# 2026-10-07 (S15 stage 2): signed TB2 system prompt (codex_harness_agent.
# TB2_BASE_INSTRUCTIONS); the legacy CODEX_BASE_INSTRUCTIONS_SHA256 is unchanged.
CODEX_TB2_BASE_INSTRUCTIONS_SHA256 = (
    "707e144f4c41dd791f5875ea6693be4738d21e7dec73a3ea09ccbf92106bdeb3"
)
CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256 = (
    "868dbbff9fe2f5a57573826cae1ae1f4ceac04eff8689a522d9af7ef1b589c5a"
)
CODEX_SUBMIT_TOOL_SCHEMA_SHA256 = (
    "162980cf1de2346e6a246a739c10a31d0f5bd30c62b27c080887b298ecad1a6f"
)
CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256 = (
    "06142f7664a668c11149b9410af6438423654ac7ee85fa382226bc7fbbf101af"
)
CODEX_CLI_VERSION = "codex-cli 0.145.0"
CODEX_NPM_PACKAGE = "@openai/codex@0.145.0-linux-x64"
CODEX_LINUX_TARGET = "x86_64-unknown-linux-musl"
CODEX_LINUX_BINARY_SHA256 = (
    "a2a05dafaa1acb002a45eaec0a462de5b13694fcfcd7bc43305f14781ce7be14"
)
CODEX_LINUX_BINARY_SIZE_BYTES = 310_730_800
CODEX_NPM_TARBALL_SHA256 = (
    "11239480f8e3efd1430f23bbe91c1a397856b8bbe6185ccbaee2382d25e03df2"
)
CODEX_PACKAGE_MANIFEST_SHA256 = (
    "8da5349aa5a4242f5e11c5ca8ff4a16d8f9f912cb8accebea4def94edbf30aee"
)
CODEX_APP_SERVER_PROTOCOL_REVISION = "v2"
CODEX_APP_SERVER_SCHEMA_SHA256 = (
    "a88d865c3ca41fc63672baf423e28b3bfdb85b989e1c95e6e268932daaac91a0"
)
CODEX_CONTAINER_BINARY_PATH = "/opt/yeto/codex/codex-x86_64-unknown-linux-musl"
CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH = (
    "/opt/yeto/codex/codex_app_server_protocol.v2.schemas.json"
)

# Terminal-Bench uses a thin isolated-process wrapper around the same attested
# stock Codex runtime.  It is intentionally not a SecRLEnv agent: its reward,
# retry, and cleanup evidence contracts are Terminal-Bench-specific.
CODEX_OPENENV_AGENT = "yeto.rl.harness.codex.codex_openenv_subprocess_agent_function.run"
CODEX_OPENENV_AGENT_MODULES = (
    "codex_openenv_subprocess_agent_function.py",
    "codex_openenv_agent_worker.py",
    "codex_openenv_agent_function.py",
    # CompactionRL bridge (opt-in, YETO_CODEX_COMPACTIONRL); imported by
    # codex_openenv_agent_function.  Name list only: no per-module hash pin.
    "compaction_bridge.py",
)
CODEX_OPENENV_IDENTITY_ENV = {
    "YETO_CODEX_OPENENV_BACKEND_PROFILE": "qwen35_08b",
    "YETO_CODEX_OPENENV_MODEL_ID": "Qwen/Qwen3.5-0.8B",
    "YETO_CODEX_OPENENV_MODEL_REVISION": ("2fc06364715b967f1860aea9cf38778875588b17"),
    "YETO_CODEX_OPENENV_BASE_INSTRUCTIONS_SHA256": (
        "1c183656ca1319142cba9e76baa199b7ab59f770a51a76660622a087e74ba846"
    ),
    "YETO_CODEX_OPENENV_TERMINAL_EXEC_TOOL_SCHEMA_SHA256": (
        "868dbbff9fe2f5a57573826cae1ae1f4ceac04eff8689a522d9af7ef1b589c5a"
    ),
    "YETO_CODEX_OPENENV_SUBMIT_TOOL_SCHEMA_SHA256": (
        "162980cf1de2346e6a246a739c10a31d0f5bd30c62b27c080887b298ecad1a6f"
    ),
    "YETO_CODEX_OPENENV_DYNAMIC_TOOLS_SCHEMA_SHA256": (
        "06142f7664a668c11149b9410af6438423654ac7ee85fa382226bc7fbbf101af"
    ),
}
SIGNED_CODEX_AGENTS = frozenset((CODEX_HARNESS_AGENT, CODEX_OPENENV_AGENT))

SECRLENV_AGENTS = frozenset((SECRLENV_AGENT, CODEX_HARNESS_AGENT))
