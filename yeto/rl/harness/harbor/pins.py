"""Pinned Harbor build for agentic RL (S25 N10.1).

Miles' Harbor example (``examples/experimental/harbor``) requires the
``harbor-miles-v0.20.0`` branch of harbor-framework/harbor: upstream 0.24.0
lacks what TITO needs (terminus-2 ``response_length_exceeded_policy``,
``MaxSeqLenExceededError`` / ``SingleTurnMaxSeqLenExceededError``, the
``max_seq_len`` check in ``Chat``, mini-swe-agent ``poll_steps``).  The branch
moves, so yeto pins a commit of it, not the branch name.  See README.md in
this package for the 0.20.0-branch vs 0.24.0 comparison.
"""

from __future__ import annotations

import json
from importlib import metadata

HARBOR_REPO = "https://github.com/harbor-framework/harbor"
HARBOR_BRANCH = "harbor-miles-v0.20.0"
# Branch head on 2026-10-19 ("E2B: reconnect to a running command when its
# output stream drops", #3684); package version string is 0.20.0.
HARBOR_COMMIT = "8f88560ce9daae9b4cc0f6ff7bac8779ae9cfa15"
HARBOR_VERSION = "0.20.0"
# Miles' example ported its harness-binding table from this branch commit.
MILES_BINDINGS_COMMIT = "53a6e92a"

# Codex build Harbor's built-in codex agent installs (``npm install -g
# @openai/codex@<version>``).  Same build as the S24 TB2 long run; whether
# Harbor's codex agent drives it unchanged is not verified yet (N10.7).
CODEX_VERSION = "0.145.0"

# The linux-x64 build of that version as npm ships it (a static musl binary
# plus its bundled rg), for the boat Codex cache (S26 E8).  sha512 = npm
# ``dist.integrity`` of the tarball, in hex.
CODEX_LINUX_X64_URL = "https://registry.npmjs.org/@openai/codex/-/codex-0.145.0-linux-x64.tgz"
CODEX_LINUX_X64_SHA512 = (
    "bbcc3c2cbbf70efb1fac30a8b302c87a66744a83445f28b9d755ac7c5b1289851"
    "ace4f6a32c07f36d53c37dbe101f37b9990312c5a0c7b1e2feba2631ee656c0"
)

# Node for the boat Codex cache (S26 E8).  Harbor's codex install runs
# ``nvm install 22``, i.e. the newest 22.x on the day it runs; on 2026-10-21
# that is v22.23.3 (nodejs.org/dist/index.json).  sha256 from SHASUMS256.txt.
NODE_VERSION = "22.23.3"
NODE_LINUX_X64_URL = f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-linux-x64.tar.gz"
NODE_LINUX_X64_SHA256 = "1084aa36196bba4c3a5e69a1ee388a6e4ff729dad09445fbcd434b28fe3c24af"


def install_spec(extra: str = "modal") -> str:
    """``uv pip install`` requirement for the pinned build (uv: the branch has a uv-workspace dependency)."""
    return f"harbor[{extra}] @ git+{HARBOR_REPO}@{HARBOR_COMMIT}"


def installed_problem() -> str | None:
    """``None`` when the importable Harbor is the pinned build, else what differs."""
    try:
        dist = metadata.distribution("harbor")
    except metadata.PackageNotFoundError:
        return "harbor is not installed"
    if dist.version != HARBOR_VERSION:
        return f"harbor {dist.version} installed, {HARBOR_VERSION} pinned"
    raw = dist.read_text("direct_url.json")
    commit = (json.loads(raw).get("vcs_info") or {}).get("commit_id") if raw else None
    if commit != HARBOR_COMMIT:
        return f"harbor commit {commit or 'unknown (not a git install)'} installed, {HARBOR_COMMIT} pinned"
    return None
