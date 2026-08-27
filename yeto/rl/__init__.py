"""Pinned Miles reinforcement-learning integration."""

MILES_REPOSITORY = "https://github.com/agentenv/miles"
MILES_BASE_COMMIT = "6062afe0a9d5d6471e8395dedc81c78dd9f4a84f"
MILES_COMMIT = "e2ad83d84a6b32a0f7d79ff196ad8c64fc67586a"
MILES_BUNDLE_PATH = "yeto/rl/vendor/miles-qwen38.bundle"
MILES_BUNDLE_SHA256 = "ff0e2ed7de75e06926c4637545a8a918c2e3ae1d5ceeb2b49d6b87511c0598b2"
MILES_PEFT_VERSION = "0.20.0"
SGLANG_REPOSITORY = "https://github.com/agentenv/sglang"
SGLANG_COMMIT = "e1b57eb8e7749235c987cc6b1b2824ce3265369b"
MILES_IMAGE = (
    "docker:ghcr.io/agentenv/miles@sha256:"
    "80c20538b63f76defde06ad5d4cfa564ae6f261110696eb1864470cb835e1590"
)

SECRLENV_AGENT_PATH = "yeto_miles_secrlenv/agent.py"
SECRLENV_AGENT_SHA256 = (
    "0f76c7fbd81135bc5b02cab2488629aaff1bb58dc59eae9228ca317583d90c26"
)
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
    "995e48f0e2817191314f19e794e25fd70e738aec5957213b51914f24d552f7b7"
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
CODEX_CONTAINER_BINARY_PATH = "/opt/yeto/codex/codex-x86_64-unknown-linux-musl"
CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH = (
    "/opt/yeto/codex/codex_app_server_protocol.v2.schemas.json"
)

# Terminal-Bench uses a thin isolated-process wrapper around the same attested
# stock Codex runtime.  It is intentionally not a SecRLEnv agent: its reward,
# retry, and cleanup evidence contracts are Terminal-Bench-specific.
CODEX_OPENENV_AGENT = "codex_openenv_subprocess_agent_function.run"
CODEX_OPENENV_AGENT_MODULES = (
    "codex_openenv_subprocess_agent_function.py",
    "codex_openenv_agent_worker.py",
    "codex_openenv_agent_function.py",
)
CODEX_OPENENV_IDENTITY_ENV = {
    "YETO_CODEX_OPENENV_BACKEND_PROFILE": "qwen35_08b",
    "YETO_CODEX_OPENENV_MODEL_ID": "Qwen/Qwen3.5-0.8B",
    "YETO_CODEX_OPENENV_MODEL_REVISION": ("2fc06364715b967f1860aea9cf38778875588b17"),
    "YETO_CODEX_OPENENV_BASE_INSTRUCTIONS_SHA256": (
        "49f65bcd88cfe5848f1fd448524dca097eec86b67900f3d212e2d5c8609346e2"
    ),
    "YETO_CODEX_OPENENV_TERMINAL_EXEC_TOOL_SCHEMA_SHA256": (
        "7e21b8634834b5c24eaf07f10bcd47e3b0a3d75d153a379cec36ff7d0acedb7e"
    ),
    "YETO_CODEX_OPENENV_SUBMIT_TOOL_SCHEMA_SHA256": (
        "c4df0e3dfae83fa3a05b142a6635a27838a4f268fd505fb95f1878d5d1646614"
    ),
    "YETO_CODEX_OPENENV_DYNAMIC_TOOLS_SCHEMA_SHA256": (
        "c41c53ef0ded04efb790e74a48eaccc9489c3b39d24d01c81d7031dc11539187"
    ),
}
SIGNED_CODEX_AGENTS = frozenset((CODEX_HARNESS_AGENT, CODEX_OPENENV_AGENT))

SECRLENV_AGENTS = frozenset((SECRLENV_AGENT, CODEX_HARNESS_AGENT))
