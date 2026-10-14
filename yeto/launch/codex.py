"""Codex harness and reward-sandbox environment for RL islands."""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path


# --- Stock Codex run bundle (ports engine; design R-SCOPE / T3 G2) ---------------
#
# The legacy direct SSH harness freezes the attested Codex artifact into the run
# bundle and ``--volume``-mounts it at /opt/yeto/codex (ssh_harness.py).  The
# sky / Modal launch paths do the same from a local bundle directory
# (``scripts/fetch_codex_bundle.py``): the launcher attests it with the same
# ``_codex_harness_contract`` the SSH harness uses, mounts the directory at
# ``CODEX_CONTAINER_DIR`` (sky file_mounts / Modal add_local_dir) and injects
# the ``YETO_CODEX_*`` environment the container preflights check
# (``yeto.rl.adapters.miles.island_entry._preflight_codex_harness`` expected_env and
# ``codex_harness_agent._attest_runtime``); every value comes from the contract.
CODEX_BUNDLE_DIR_ENV = "YETO_CODEX_BUNDLE_DIR"
CODEX_CONTAINER_DIR = "/opt/yeto/codex"
CODEX_BUNDLE_FILES = (
    "codex_harness_binary",
    "codex_package_manifest",
    "codex_app_server_schema",
)
# Harness hook / provider env the ports island preflight resolves
# (entry.HARNESS_PREFLIGHT_ENV, preflight.ENVIRONMENT_PROVIDER_ENV); the
# provider must come from the launching environment (no in-tree default).
HARNESS_PREFLIGHT_ENV = "YETO_HARNESS_PREFLIGHT"
HARNESS_ENVIRONMENT_PROVIDER_ENV = "YETO_HARNESS_ENVIRONMENT_PROVIDER"
HARNESS_PASSTHROUGH_ENV = (
    HARNESS_ENVIRONMENT_PROVIDER_ENV,
    # tb2_provider Modal Sandbox backend: the island creates task sandboxes
    # itself. secret-handling-hardening: it gets a separate sandbox-only Modal
    # token (YETO_SANDBOX_MODAL_TOKEN_*), never the main MODAL_TOKEN_*.
    "YETO_SANDBOX_MODAL_TOKEN_ID",
    "YETO_SANDBOX_MODAL_TOKEN_SECRET",
    "OPENENV_RUN_ID",
    "SECRLENV_MAX_TURNS",
)
HARNESS_PASSTHROUGH_ENV_PREFIXES = ("YETO_HARNESS_TB2_", "YETO_RLF_")
# tb2_provider's Modal Sandbox backend runs inside the island (rollout
# workers create the task sandboxes), so the island python needs the Modal
# client.  The ports image does not ship it; install it in setup without
# touching the image's protobuf 7 (modal pins <7 but works; verified in-image
# on ghcr 12fcd9e5: Sandbox create/exec/terminate, ray/sglang/miles import).
MODAL_SANDBOX_PROVIDER = "yeto.rl.harness.codex.tb2_provider:modal_provider"
# rl-agentic-reward-env: same TB2 provider, sandboxes from prebaked images.
PREBAKED_MODAL_SANDBOX_PROVIDER = "yeto.cloud.modal_reward_env:modal_provider"
RLF_MODAL_SANDBOX_PROVIDER = "yeto.cloud.modal_reward_env:rlf_modal_provider"
MODAL_SANDBOX_PROVIDERS = frozenset({MODAL_SANDBOX_PROVIDER, PREBAKED_MODAL_SANDBOX_PROVIDER, RLF_MODAL_SANDBOX_PROVIDER})
SANDBOX_MODAL_TOKEN_ENVS = ("YETO_SANDBOX_MODAL_TOKEN_ID", "YETO_SANDBOX_MODAL_TOKEN_SECRET")


def require_sandbox_modal_token(envs) -> None:
    """secret-handling-hardening: a Modal Sandbox provider on the island needs
    the sandbox-only token; the main MODAL_TOKEN_* never goes to an island."""
    missing = [n for n in SANDBOX_MODAL_TOKEN_ENVS if not envs.get(n)]
    if missing:
        raise ValueError(
            f"the Modal Sandbox provider needs {', '.join(missing)} in the launching "
            "environment: a separate Modal token used only for task sandboxes (the main "
            "MODAL_TOKEN_ID/SECRET is never shipped to a learner island)")
# --modal-sandbox-secret: in-island fail-closed check (names only, never values).
SANDBOX_TOKEN_IN_ISLAND_CHECK = (
    "for _v in " + " ".join(SANDBOX_MODAL_TOKEN_ENVS) + "; do "
    '[ -n "$(printenv "$_v")" ] || { echo "[yeto-setup] $_v missing in the island: '
    'the --modal-sandbox-secret Modal Secret must hold it" >&2; exit 1; }; done'
)


def modal_sandbox_secret_name(args, spec) -> str | None:
    """--modal-sandbox-secret NAME, or None when unset.  Only Modal islands can
    attach a named Modal Secret: any other cloud is refused, never ignored."""
    name = getattr(args, "modal_sandbox_secret", None)
    if not name:
        return None
    if getattr(spec, "cloud", None) != "modal":
        raise ValueError(
            f"--modal-sandbox-secret attaches a Modal Secret to a Modal island; "
            f"island cloud {getattr(spec, 'cloud', None)!r} cannot use it")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", str(name)):
        raise ValueError(f"--modal-sandbox-secret {name!r} is not a Modal Secret name")
    return str(name)


MODAL_CLIENT_SETUP = (
    # --ignore-installed: in a Modal Function container the setup shell
    # already sees Modal's runtime copies (/pkg, /__modal/deps), so a plain
    # install reports "already satisfied" and writes nothing into the island
    # python's site-packages, which is all its own processes can import
    # (codex-smoke-20261003-8/-9).
    "python3 -m pip install -q --no-deps --ignore-installed "
    "'modal==1.5.5' 'grpclib>=0.4.7,<0.4.10' 'synchronicity~=0.12.5' cbor2 toml "
    "types-certifi types-toml watchfiles\n"
    # Verify the packages physically landed in site-packages and import under
    # the run script's PYTHONPATH (what the learner and Ray workers see).
    'PYTHONPATH="$HOME/miles:$HOME/sglang/python:$HOME/sky_workdir${PYTHONPATH:+:$PYTHONPATH}" '
    "python3 -c 'import os, sys, sysconfig; import modal, grpclib, synchronicity; "
    "sp = sysconfig.get_paths()[\"purelib\"]; "
    "missing = [m for m in (\"modal\", \"grpclib\", \"synchronicity\") if not os.path.isdir(os.path.join(sp, m))]; "
    "sys.exit(f\"not in {sp}: {missing}\" if missing else 0)' "
    "|| { echo '[yeto-setup] Modal client unusable in the island python' >&2; exit 1; }"
)
CODEX_COMPACTION_ENV = "YETO_CODEX_COMPACTION_ENABLED"


def codex_bundle_paths(bundle_dir: str) -> dict[str, str]:
    """The three attested files inside a fetched bundle dir, keyed like the SSH
    harness CLI (``--codex-harness-binary`` ...)."""
    from ..rl import CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH, CODEX_CONTAINER_BINARY_PATH

    root = Path(bundle_dir).expanduser()
    return {
        "codex_harness_binary": str(root / os.path.basename(CODEX_CONTAINER_BINARY_PATH)),
        "codex_package_manifest": str(root / "codex-package.json"),
        "codex_app_server_schema": str(
            root / os.path.basename(CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH)
        ),
    }


def codex_bundle_contract(args, bundle_dir: str) -> dict:
    """Attest ``bundle_dir`` against the Yeto pins (same checks as the SSH harness)."""
    from argparse import Namespace

    from ..rl.ssh_harness import HarnessError, _codex_harness_contract

    if not os.path.isdir(os.path.expanduser(bundle_dir)):
        raise ValueError(f"{CODEX_BUNDLE_DIR_ENV}={bundle_dir!r} is not a directory")
    try:
        return _codex_harness_contract(Namespace(**codex_bundle_paths(bundle_dir)), args)
    except HarnessError as exc:
        raise ValueError(f"Codex bundle {bundle_dir}: {exc}") from exc


def codex_container_env(contract: dict) -> dict[str, str]:
    """``YETO_CODEX_*`` for the island container, mirroring the SSH harness
    ``codex_env`` template and the learner's expected_env, from the contract."""
    from ..rl.ssh_harness import _canonical_json, _plan_digest

    backend = contract["backend"]
    env = {
        "YETO_CODEX_BINARY_PATH": contract["container_binary_path"],
        "YETO_CODEX_BINARY_SHA256": contract["binary_sha256"],
        "YETO_CODEX_BINARY_SIZE_BYTES": str(contract["binary_size_bytes"]),
        "YETO_CODEX_VERSION": contract["cli_version"],
        "YETO_CODEX_APP_SERVER_PROTOCOL_REVISION": contract["app_server_protocol_revision"],
        "YETO_CODEX_APP_SERVER_SCHEMA_SHA256": contract["app_server_schema_sha256"],
        "YETO_CODEX_BASE_INSTRUCTIONS_SHA256": contract["base_instructions_sha256"],
        "YETO_CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256": contract["terminal_exec_tool_schema_sha256"],
        "YETO_CODEX_SUBMIT_TOOL_SCHEMA_SHA256": contract["submit_tool_schema_sha256"],
        "YETO_CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256": contract["dynamic_tools_schema_sha256"],
        "YETO_CODEX_REASONING_EFFORT": contract["reasoning_effort"],
        "YETO_CODEX_BACKEND_MAX_TOKENS": str(backend["max_tokens"]),
        "YETO_CODEX_BACKEND_REASONING_EFFORT": backend["reasoning_effort"],
        "YETO_CODEX_BACKEND_THINKING": backend["thinking"]["type"],
        "YETO_CODEX_CHAT_TEMPLATE": backend["chat_template"],
        "YETO_CODEX_CHAT_TEMPLATE_KWARGS": _canonical_json(backend["chat_template_kwargs"]),
        "YETO_CODEX_TITO_ALLOWED_APPEND_ROLES": "tool,user",
    }
    env.update({str(k): str(v) for k, v in contract.get("openenv_identity_env", {}).items()})
    env["YETO_CODEX_HARNESS_CONTRACT_SHA256"] = _plan_digest(contract)
    return env


def rl_gae_variant(args) -> str | None:
    """``advantage.gae_variant`` of the prepared RL algorithm spec (None if absent)."""
    raw = getattr(args, "rl_algorithm_spec_json", None)
    if getattr(args, "training_mode", "sft") != "rl" or not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    value = ((payload or {}).get("advantage") or {}).get("gae_variant")
    return None if value is None else str(value)


def compactionrl_launch_env(args, environ, harness_compacts: bool) -> dict[str, str]:
    """Rollout-worker env for CompactionRL (design D8; progress "S13 Codex 压缩接线").

    A ``cross_segment_per_sample`` spec switches ``YETO_CODEX_COMPACTIONRL=1`` on
    (and forwards ``YETO_CODEX_COMPACTIONRL_T_COMP`` from the launching
    environment when set); the island preflight forwards every ``YETO_CODEX_*``
    to the Ray rollout workers.  Any other spec returns {} (env unchanged).
    Raises when spec, harness and the launching environment disagree
    (``compactionrl.check_rollout_compaction``)."""
    from ..rl.algos import compactionrl as crl

    variant = rl_gae_variant(args)
    raw_switch = environ.get(crl.COMPACTION_SWITCH_ENV)
    t_comp = environ.get(crl.COMPACTION_T_COMP_ENV)
    if variant is None and not raw_switch and not t_comp:
        return {}
    try:
        explicit = crl.compaction_switch(raw_switch)
        wanted = variant == crl.COMPACTED_GAE_VARIANT
        if raw_switch not in (None, "") and explicit != wanted:
            if wanted:
                raise ValueError(
                    f"{crl.COMPACTION_SWITCH_ENV}={raw_switch!r} in the launching environment "
                    f"contradicts the CompactionRL spec ({crl.COMPACTED_GAE_VARIANT})"
                )
        crl.check_rollout_compaction(
            variant, wanted or explicit, harness_compacts=harness_compacts, t_comp=t_comp
        )
    except ValueError as error:
        raise ValueError(f"CompactionRL launch: {error}") from error
    if not wanted:
        return {}
    generate = getattr(args, "custom_generate_function_path", None)
    if getattr(args, "custom_agent_function_path", None) and generate != crl.CODEX_OPENENV_GENERATE:
        # s19-compaction-g1-20261010b: with upstream agentic_tool_call.generate the
        # segment sessions are never collected, samples carry no tokens_after and
        # the trainer fails at round 0 (cross_segment_per_sample needs tokens_after).
        raise ValueError(
            f"CompactionRL launch: {crl.COMPACTED_GAE_VARIANT} needs "
            f"--custom-generate-function-path {crl.CODEX_OPENENV_GENERATE} (got {generate!r})")
    env = {crl.COMPACTION_SWITCH_ENV: "1"}
    if t_comp:
        env[crl.COMPACTION_T_COMP_ENV] = str(t_comp).strip()
    return env


def codex_harness_launch(args, environ=None) -> tuple[str, dict[str, str], dict[str, str]] | None:
    """(learner flags, container envs, file_mounts) for a signed Codex agent on
    the ports engine, or None when the run is not a Codex run.

    Fails closed: no bundle dir, a bundle that does not match the pins, no
    environment provider, or trainable compaction switched on in the launching
    environment (R-D5a) all raise before any cloud resource is requested."""
    from ..rl import CODEX_OPENENV_AGENT, SIGNED_CODEX_AGENTS
    from ..rl.harness.codex.preflight import HARNESS_PREFLIGHT_SPEC, _FALSE

    environ = os.environ if environ is None else environ
    custom_agent = getattr(args, "custom_agent_function_path", None)
    from ..rl.synthetic_segments import SYNTHETIC_SEGMENTS_REWARD

    # TEST ONLY (task 6.4 G1): the synthetic two-segment reward stands in for a
    # compacting harness; it is accepted only with cross_segment_per_sample.
    synthetic = getattr(args, "reward_function", None) == SYNTHETIC_SEGMENTS_REWARD
    if synthetic and rl_gae_variant(args) != "cross_segment_per_sample":
        raise ValueError(f"{SYNTHETIC_SEGMENTS_REWARD} is only valid with gae_variant cross_segment_per_sample")
    compaction_env = compactionrl_launch_env(
        args, environ, custom_agent == CODEX_OPENENV_AGENT or (synthetic and custom_agent is None))
    if custom_agent not in SIGNED_CODEX_AGENTS:
        return None
    bundle_dir = getattr(args, "codex_bundle_dir", None) or environ.get(CODEX_BUNDLE_DIR_ENV)
    if not bundle_dir:
        raise ValueError(
            f"the signed Codex harness on the ports engine needs {CODEX_BUNDLE_DIR_ENV} "
            "(a bundle from scripts/fetch_codex_bundle.py) to mount at "
            f"{CODEX_CONTAINER_DIR}"
        )
    if environ.get(CODEX_COMPACTION_ENV, "").strip().lower() not in _FALSE:
        raise ValueError(
            f"{CODEX_COMPACTION_ENV} is set in the launching environment; the fork pin "
            "session server has no trainable compaction (R-D5a)"
        )
    contract = codex_bundle_contract(args, bundle_dir)
    flags = f" --codex-harness-contract {shlex.quote(_json_compact(contract))}"
    envs = codex_container_env(contract)
    if custom_agent == CODEX_OPENENV_AGENT:
        envs[HARNESS_PREFLIGHT_ENV] = environ.get(HARNESS_PREFLIGHT_ENV) or HARNESS_PREFLIGHT_SPEC
        if not environ.get(HARNESS_ENVIRONMENT_PROVIDER_ENV):
            raise ValueError(
                f"{HARNESS_ENVIRONMENT_PROVIDER_ENV}=module:callable is required: the "
                "Codex Terminal-Bench preflight fails closed without an environment provider"
            )
        for name in HARNESS_PASSTHROUGH_ENV:
            if environ.get(name):
                envs[name] = environ[name]
        for name, value in environ.items():
            if name.startswith(HARNESS_PASSTHROUGH_ENV_PREFIXES) and value:
                envs[name] = value
    envs.update(compaction_env)  # {} unless the spec is CompactionRL
    mounts = {CODEX_CONTAINER_DIR: str(Path(bundle_dir).expanduser().resolve())}
    return flags, envs, mounts


HEAD_CODEX_BUNDLE_PATH = "~/yeto-codex-bundle"


def codex_head_staging(args, environ=None) -> tuple[dict[str, str], dict[str, str]]:
    """(head file_mounts, head-job secret envs) for a signed Codex run under
    ``--controller head`` (S17 M1).

    The head VM replays the launch and builds every island itself, so it needs
    what ``codex_harness_launch`` reads on the submitter: the bundle directory
    (staged at ``HEAD_CODEX_BUNDLE_PATH``; the caller points
    ``args.codex_bundle_dir`` there) and the harness env (provider, TB2 knobs,
    reward HMAC key, Modal token).  The env goes as sky secrets (redacted),
    never as plain envs.  Returns ({}, {}) for a non-Codex run; fails closed
    exactly like ``codex_harness_launch`` (attests the local bundle first).
    """
    environ = os.environ if environ is None else environ
    if codex_harness_launch(args, environ) is None:
        return {}, {}
    bundle_dir = getattr(args, "codex_bundle_dir", None) or environ.get(CODEX_BUNDLE_DIR_ENV)
    mounts = {HEAD_CODEX_BUNDLE_PATH: str(Path(bundle_dir).expanduser().resolve())}
    secrets = {name: environ[name] for name in (HARNESS_PREFLIGHT_ENV, *HARNESS_PASSTHROUGH_ENV)
               if environ.get(name)}
    secrets.update({name: value for name, value in environ.items()
                    if name.startswith(HARNESS_PASSTHROUGH_ENV_PREFIXES) and value})
    return mounts, secrets


def _json_compact(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
