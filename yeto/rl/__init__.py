"""Pinned Miles reinforcement-learning integration."""

MILES_REPOSITORY = "https://github.com/agentenv/miles"
MILES_BASE_COMMIT = "6062afe0a9d5d6471e8395dedc81c78dd9f4a84f"
MILES_COMMIT = "e2ad83d84a6b32a0f7d79ff196ad8c64fc67586a"
MILES_BUNDLE_PATH = "yeto/rl/vendor/miles-qwen38.bundle"
MILES_BUNDLE_SHA256 = (
    "ff0e2ed7de75e06926c4637545a8a918c2e3ae1d5ceeb2b49d6b87511c0598b2"
)
MILES_PEFT_VERSION = "0.20.0"
SGLANG_REPOSITORY = "https://github.com/agentenv/sglang"
SGLANG_COMMIT = "e1b57eb8e7749235c987cc6b1b2824ce3265369b"
MILES_IMAGE = (
    "docker:ghcr.io/agentenv/miles@sha256:"
    "80c20538b63f76defde06ad5d4cfa564ae6f261110696eb1864470cb835e1590"
)

SECRLENV_AGENT_PATH = "yeto_miles_secrlenv/agent.py"
SECRLENV_AGENT_SHA256 = "0f76c7fbd81135bc5b02cab2488629aaff1bb58dc59eae9228ca317583d90c26"
SECRLENV_AGENT = "yeto_miles_secrlenv.agent.run"
SECRLENV_REWARD = "yeto_miles_secrlenv.reward:reward_func"
SECRLENV_GROUP_FILTER = "yeto_miles_secrlenv.reward.check_group"
SECRLENV_GENERATE = "yeto_miles_secrlenv.generate.generate"
SECRLENV_GENERATE_SHA256 = (
    "9e034d6b2e9fec642501ea4a638a8fe196819dacde614ce2903359fc54ea1713"
)
SECRLENV_ZERO_VARIANCE_REPLACEMENTS = 0
SECRLENV_INFRASTRUCTURE_REPLACEMENTS = 1

# Stock Codex is part of the signed Yeto security-environment harness.  These
# pins identify the official Linux artifact, not the controller's host binary.
CODEX_HARNESS_AGENT = "yeto_miles_secrlenv.codex_harness_agent.run"
CODEX_HARNESS_AGENT_PATH = "yeto_miles_secrlenv/codex_harness_agent.py"
CODEX_HARNESS_AGENT_SHA256 = (
    "a8f6f89a0090dce7401d6f59441fbe34fc8359deb34e355970c9fa47250ca51d"
)
CODEX_BASE_INSTRUCTIONS_SHA256 = (
    "1c183656ca1319142cba9e76baa199b7ab59f770a51a76660622a087e74ba846"
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
    "f2415ee36b3c9fa16617c800910cd65b8086ce7c7fecee3dac5f7089eb5973b9"
)
CODEX_CONTAINER_BINARY_PATH = (
    "/opt/yeto/codex/codex-x86_64-unknown-linux-musl"
)
CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH = (
    "/opt/yeto/codex/codex_app_server_protocol.v2.schemas.json"
)

SECRLENV_AGENTS = frozenset((SECRLENV_AGENT, CODEX_HARNESS_AGENT))
