"""Pinned Miles reinforcement-learning integration."""

# Miles pins moved to ``yeto.rl.adapters.miles.pins`` (decoupling 5.3, audit L4).
# Re-exported lazily so importing the neutral core never loads the adapter.
_MILES_PIN_NAMES = frozenset((
    "MILES_REPOSITORY",
    "MILES_BASE_COMMIT",
    "MILES_COMMIT",
    "MILES_BUNDLE_PATH",
    "MILES_BUNDLE_SHA256",
    "MILES_PEFT_VERSION",
    "SGLANG_REPOSITORY",
    "SGLANG_COMMIT",
    "MILES_IMAGE",
    "MILES_NEXT_REPOSITORY",
    "MILES_NEXT_UPSTREAM_COMMIT",
    "MILES_NEXT_COMMIT",
    "SGLANG_NEXT_REPOSITORY",
    "SGLANG_NEXT_UPSTREAM_COMMIT",
    "SGLANG_NEXT_COMMIT",
    "MILES_LEGACY_PINS",
    "MILES_NEXT_PINS",
    "MILES_NEXT_BASE_IMAGE",
    "MILES_NEXT_IMAGE",
    "MILES_NEXT_IMAGE_MANIFEST",
    "NEBIUS_BAKED_IMAGES",
    "MILES_NEXT_IMAGE_SGLANG_ROOT",
    "MilesRevisionPins",
    "default_rl_image",
))


def __getattr__(name: str):
    if name in _MILES_PIN_NAMES:
        from .adapters.miles import pins

        return getattr(pins, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
