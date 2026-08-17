"""Pinned Miles reinforcement-learning integration."""

MILES_REPOSITORY = "https://github.com/agentenv/miles"
MILES_COMMIT = "6062afe0a9d5d6471e8395dedc81c78dd9f4a84f"
MILES_PEFT_VERSION = "0.20.0"
SGLANG_REPOSITORY = "https://github.com/agentenv/sglang"
SGLANG_COMMIT = "e1b57eb8e7749235c987cc6b1b2824ce3265369b"
MILES_IMAGE = (
    "docker:ghcr.io/alexeisie/miles@sha256:"
    "5be3e0722c7b0174c3c1a5526064872987c7bc367af700117a3589efbd6b19bd"
)

SECRLENV_AGENT_PATH = "yeto_miles_secrlenv/agent.py"
SECRLENV_AGENT_SHA256 = "750e9432f81b9467d5600423a8b2c80f31156d25b2ae0db0e4f73c4d080afadb"

# Stock Codex is part of the signed Yeto security-environment harness.  These
# pins identify the official Linux artifact, not the controller's host binary.
CODEX_HARNESS_AGENT = "yeto_miles_secrlenv.codex_harness_agent.run"
CODEX_HARNESS_AGENT_PATH = "yeto_miles_secrlenv/codex_harness_agent.py"
CODEX_HARNESS_AGENT_SHA256 = (
    "f8b0e8a6c8e13b96a8027c262ce2450b6ae8c6a03acedab4e1a5b53f26b3aecc"
)
CODEX_BASE_INSTRUCTIONS_SHA256 = (
    "55622eb2d7246eb199fd95b0bbb97b34698feb550aca5c0a62c4557242e5f8b1"
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
