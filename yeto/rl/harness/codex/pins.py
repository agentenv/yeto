"""Self-attestation pins for the Codex harness package (ports path).

Provenance: moved from yeto commit 5bfc011 ``yeto_miles_secrlenv/`` (see
openspec change rl-codex-harness-rollout, design R-CTX). The legacy pins in
``yeto.rl`` (``SECRLENV_*_SHA256``, ``CODEX_HARNESS_AGENT_SHA256``) are owned by
the image line and still describe the legacy import path; the values below
describe the moved files. ``LEGACY_SOURCE_SHA256`` is kept for comparison.
"""

AGENT_SHA256 = "fef958c32d27af124827b17369bb82557e96946c4b6f9b74c67f084a533f7c81"
GENERATE_SHA256 = "1c79b0e678b8681b5bd6221b5a4bbc6adbe7a1413b688e248cb930eb3e456cca"

LEGACY_SOURCE_SHA256 = {
    "agent.py": "0f76c7fbd81135bc5b02cab2488629aaff1bb58dc59eae9228ca317583d90c26",
    "generate.py": "9e034d6b2e9fec642501ea4a638a8fe196819dacde614ce2903359fc54ea1713",
    "codex_harness_agent.py": "94fa4c245b719d236ec1007b70d395adb12456b3ef04278592ae2d3c0d843947",
}
LEGACY_SOURCE_COMMIT = "5bfc011"

OPENENV_BACKEND_PROFILE = "qwen35_08b"
