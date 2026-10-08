"""Self-attestation pins for the Codex harness package (ports path).

Provenance: moved from yeto commit 5bfc011 ``yeto_miles_secrlenv/`` (see
openspec change rl-codex-harness-rollout, design R-CTX). The legacy pins in
``yeto.rl`` (``SECRLENV_*_SHA256``, ``CODEX_HARNESS_AGENT_SHA256``) are owned by
the image line and still describe the legacy import path; the values below
describe the moved files. ``LEGACY_SOURCE_SHA256`` is kept for comparison.
"""

AGENT_SHA256 = "fef958c32d27af124827b17369bb82557e96946c4b6f9b74c67f084a533f7c81"
GENERATE_SHA256 = "1df1ded9c6c81404a6101e1821aefc3129d207a3150241980306d6494ee4a2be"  # adapters/miles/harness_glue/codex_generate.py (decoupling 5.7; was 1c79b0e6…)

LEGACY_SOURCE_SHA256 = {
    "agent.py": "0f76c7fbd81135bc5b02cab2488629aaff1bb58dc59eae9228ca317583d90c26",
    "generate.py": "9e034d6b2e9fec642501ea4a638a8fe196819dacde614ce2903359fc54ea1713",
    "codex_harness_agent.py": "94fa4c245b719d236ec1007b70d395adb12456b3ef04278592ae2d3c0d843947",
}
LEGACY_SOURCE_COMMIT = "5bfc011"

# Build-time default backend profile of the Codex OpenEnv adapter.  This is the
# value the image line bakes into the container as
# ``YETO_CODEX_OPENENV_BACKEND_PROFILE`` (``yeto/rl/__init__.py``
# CODEX_OPENENV_IDENTITY_ENV) and is a *record of the image build*, not the
# profile a launch must use.  rl-fn-codex-rollout 1.0: the runtime profile is
# declared per launch (``--codex-backend-profile``) and only has to belong to
# OPENENV_BACKEND_PROFILES; see codex_openenv_agent_function.resolve_backend_profile.
OPENENV_BACKEND_PROFILE = "qwen35_08b"
# Profiles the OpenEnv adapter may be driven with at runtime.  Every member must
# declare ``model_identifier``/``model_revision`` in ``yeto.rl.codex_backend``
# (checked by ``codex_openenv_agent_function.profile_identity``).  The tool
# surface pins (``*_SHA256`` identity env) are profile-independent, so adding a
# profile here needs no image rebuild.
OPENENV_BACKEND_PROFILES = ("qwen35_08b", "qwen38_next", "qwen38_next_4layer")
