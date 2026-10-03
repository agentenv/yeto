"""Stock Codex run bundle on the ports launch paths (T3 G2; design R-SCOPE).

CPU only.  The contract attestation is exercised against the real bundle when
``YETO_CODEX_BUNDLE_DIR`` points at a ``scripts/fetch_codex_bundle.py`` output
(skipped otherwise); everything else uses a synthetic contract built from the
Yeto pins so the launcher wiring (learner flag, ``YETO_CODEX_*`` env, sky
file_mounts, Modal add_local_dir) is checked without a 310 MB binary.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import types
from pathlib import Path

import pytest

from yeto import launcher as L
from yeto import modal_runner as mr
from yeto.gpu_spec import parse_gpu_spec
from yeto.launcher import _prepare_rl_args, build_modal_island_config, make_miles_island_task
from yeto.rl import (
    CODEX_CONTAINER_BINARY_PATH,
    CODEX_OPENENV_AGENT,
    CODEX_OPENENV_IDENTITY_ENV,
)
from yeto.rl.codex_backend import QWEN35_08B_MODEL, QWEN35_08B_REVISION
from yeto.rl.ssh_harness import _plan_digest

from test_modal_runner import fake_modal
from test_rl_engine_selection import _cli as _args  # ports engine (CLI default)
from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task

PROVIDER = "yeto.rl.harness.codex.tb2_provider:provider"


def _codex_args():
    args = _args(
        (
            "--model", QWEN35_08B_MODEL,
            "--model-revision", QWEN35_08B_REVISION,
            "--lora-targets", "attention",
            "--custom-generate-function-path",
            "miles.rollout.generate_hub.agentic_tool_call.generate",
            "--custom-agent-function-path", CODEX_OPENENV_AGENT,
            "--codex-reasoning-effort", "xhigh",
            "--codex-backend-profile", "qwen35_08b",
            "--use-session-server",
            "--tito-model", "qwen35",
            "--tito-allowed-append-roles", "tool", "user",
            "--agent-max-seq-len", "128",
        )
    )
    args.data_revision = "d" * 40
    args.source_sha256 = "e" * 64
    args.reward_sha256 = "f" * 64
    _prepare_rl_args(args)
    return args


def _synthetic_contract(args) -> dict:
    """What ``ssh_harness._codex_harness_contract`` returns for a pin-exact bundle."""
    from yeto.rl import ssh_harness as sh
    from yeto.rl.codex_backend import stock_codex_backend_contract

    identity = sh._pinned_codex_adapter_identity()
    return {
        "agent_function_path": sh.CODEX_HARNESS_AGENT,
        "agent_source_sha256": sh.CODEX_HARNESS_AGENT_SHA256,
        "controller_binary_path": "/bundle/codex/codex-x86_64-unknown-linux-musl",
        "controller_package_manifest_path": "/bundle/codex/codex-package.json",
        "controller_app_server_schema_path": "/bundle/codex/codex_app_server_protocol.v2.schemas.json",
        "bundle_binary_path": sh.CODEX_REMOTE_BUNDLE_PATH,
        "bundle_package_manifest_path": sh.CODEX_REMOTE_MANIFEST_PATH,
        "bundle_app_server_schema_path": sh.CODEX_REMOTE_SCHEMA_PATH,
        "container_binary_path": sh.CODEX_CONTAINER_BINARY_PATH,
        "container_app_server_schema_path": sh.CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH,
        "binary_sha256": sh.CODEX_LINUX_BINARY_SHA256,
        "binary_size_bytes": sh.CODEX_LINUX_BINARY_SIZE_BYTES,
        "cli_version": sh.CODEX_CLI_VERSION,
        "npm_package": sh.CODEX_NPM_PACKAGE,
        "target": sh.CODEX_LINUX_TARGET,
        "npm_tarball_sha256": sh.CODEX_NPM_TARBALL_SHA256,
        "package_manifest_sha256": sh.CODEX_PACKAGE_MANIFEST_SHA256,
        "app_server_protocol_revision": sh.CODEX_APP_SERVER_PROTOCOL_REVISION,
        "app_server_schema_sha256": sh.CODEX_APP_SERVER_SCHEMA_SHA256,
        **identity,
        "reasoning_effort": "xhigh",
        "backend": stock_codex_backend_contract("qwen35_08b", args.rollout_max_response_len),
        "openenv_identity_env": dict(CODEX_OPENENV_IDENTITY_ENV),
    }


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """A fake bundle dir + the contract attestation stubbed to the pin values."""
    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    root = tmp_path / "codex-bundle" / "codex"
    root.mkdir(parents=True)
    for name in ("codex-x86_64-unknown-linux-musl", "codex-package.json",
                 "codex_app_server_protocol.v2.schemas.json"):
        (root / name).write_bytes(b"x")
    seen = {}

    def fake_contract(namespace, args):
        seen["namespace"] = vars(namespace)
        return _synthetic_contract(args)

    import yeto.rl.ssh_harness as sh

    monkeypatch.setattr(sh, "_codex_harness_contract", fake_contract)
    monkeypatch.setenv(L.CODEX_BUNDLE_DIR_ENV, str(root))
    monkeypatch.setenv(L.HARNESS_ENVIRONMENT_PROVIDER_ENV, PROVIDER)
    monkeypatch.delenv(L.CODEX_COMPACTION_ENV, raising=False)
    monkeypatch.delenv(L.HARNESS_PREFLIGHT_ENV, raising=False)
    return root, seen


# The learner's container-side expected_env keys (yeto.rl.learner._preflight_codex_harness)
# plus what codex_harness_agent._attest_runtime / preflight.assert_openenv_identity read.
LEARNER_EXPECTED_ENV = {
    "YETO_CODEX_BINARY_PATH", "YETO_CODEX_BINARY_SHA256", "YETO_CODEX_BINARY_SIZE_BYTES",
    "YETO_CODEX_VERSION", "YETO_CODEX_APP_SERVER_PROTOCOL_REVISION",
    "YETO_CODEX_APP_SERVER_SCHEMA_SHA256", "YETO_CODEX_BASE_INSTRUCTIONS_SHA256",
    "YETO_CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256", "YETO_CODEX_SUBMIT_TOOL_SCHEMA_SHA256",
    "YETO_CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256", "YETO_CODEX_REASONING_EFFORT",
    "YETO_CODEX_BACKEND_MAX_TOKENS", "YETO_CODEX_BACKEND_REASONING_EFFORT",
    "YETO_CODEX_BACKEND_THINKING", "YETO_CODEX_CHAT_TEMPLATE", "YETO_CODEX_CHAT_TEMPLATE_KWARGS",
    "YETO_CODEX_TITO_ALLOWED_APPEND_ROLES", "YETO_CODEX_HARNESS_CONTRACT_SHA256",
}


def test_non_codex_runs_are_untouched():
    args = _args()
    args.data_revision = "d" * 40
    args.source_sha256 = "e" * 64
    args.reward_sha256 = "f" * 64
    assert L.codex_harness_launch(args, environ={}) is None


def test_codex_on_legacy_engine_still_needs_the_ssh_harness():
    args = _codex_args()
    args.rl_engine = "legacy"
    with pytest.raises(ValueError, match="direct SSH harness"):
        L.codex_harness_launch(args, environ={L.CODEX_BUNDLE_DIR_ENV: "/x"})


def test_codex_on_ports_fails_closed_without_bundle_provider_or_with_compaction(bundle, monkeypatch):
    args = _codex_args()
    with pytest.raises(ValueError, match=L.CODEX_BUNDLE_DIR_ENV):
        L.codex_harness_launch(args, environ={})
    root, _ = bundle
    with pytest.raises(ValueError, match=L.HARNESS_ENVIRONMENT_PROVIDER_ENV):
        L.codex_harness_launch(args, environ={L.CODEX_BUNDLE_DIR_ENV: str(root)})
    with pytest.raises(ValueError, match=L.CODEX_COMPACTION_ENV):
        L.codex_harness_launch(args, environ={
            L.CODEX_BUNDLE_DIR_ENV: str(root), L.HARNESS_ENVIRONMENT_PROVIDER_ENV: PROVIDER,
            L.CODEX_COMPACTION_ENV: "1"})
    with pytest.raises(ValueError, match="not a directory"):
        L.codex_harness_launch(args, environ={
            L.CODEX_BUNDLE_DIR_ENV: str(root / "missing"), L.HARNESS_ENVIRONMENT_PROVIDER_ENV: PROVIDER})


def test_codex_launch_flags_env_and_mount_come_from_the_contract(bundle):
    root, seen = bundle
    args = _codex_args()
    flags, envs, mounts = L.codex_harness_launch(args)
    # The attestation sees the three bundle files, named as the SSH harness CLI does.
    assert seen["namespace"] == {
        "codex_harness_binary": str(root / "codex-x86_64-unknown-linux-musl"),
        "codex_package_manifest": str(root / "codex-package.json"),
        "codex_app_server_schema": str(root / "codex_app_server_protocol.v2.schemas.json"),
    }
    contract = _synthetic_contract(args)
    parts = shlex.split(flags)
    assert parts[0] == "--codex-harness-contract" and json.loads(parts[1]) == contract
    assert mounts == {L.CODEX_CONTAINER_DIR: str(root.resolve())}
    assert L.CODEX_CONTAINER_DIR == str(Path(CODEX_CONTAINER_BINARY_PATH).parent)
    # Every env the container preflights compare is present, with contract values.
    assert LEARNER_EXPECTED_ENV <= set(envs)
    assert set(CODEX_OPENENV_IDENTITY_ENV) <= set(envs)
    assert envs["YETO_CODEX_BINARY_PATH"] == CODEX_CONTAINER_BINARY_PATH
    assert envs["YETO_CODEX_BINARY_SIZE_BYTES"] == "310730800"
    assert envs["YETO_CODEX_VERSION"] == "codex-cli 0.145.0"
    assert envs["YETO_CODEX_BACKEND_MAX_TOKENS"] == str(contract["backend"]["max_tokens"])
    assert envs["YETO_CODEX_CHAT_TEMPLATE_KWARGS"] == json.dumps(
        contract["backend"]["chat_template_kwargs"], sort_keys=True, separators=(",", ":"))
    assert envs["YETO_CODEX_HARNESS_CONTRACT_SHA256"] == _plan_digest(contract)
    assert envs["YETO_CODEX_REASONING_EFFORT"] == "xhigh"
    # Harness hook + provider reach the island; compaction stays unset (R-D5a).
    assert envs[L.HARNESS_PREFLIGHT_ENV] == "yeto.rl.harness.codex.preflight:harness_preflight"
    assert envs[L.HARNESS_ENVIRONMENT_PROVIDER_ENV] == PROVIDER
    assert L.CODEX_COMPACTION_ENV not in envs


def test_sky_island_task_mounts_the_bundle_and_exports_the_env(bundle):
    root, _ = bundle
    args = _codex_args()
    task = make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:29400")
    assert "--codex-harness-contract " in task.run
    assert f"--custom-agent-function-path {CODEX_OPENENV_AGENT}" in task.run
    assert "--agent-max-seq-len 128" in task.run and "--use-session-server" in task.run
    assert "--tito-model qwen35" in task.run and "--codex-backend-profile qwen35_08b" in task.run
    assert "--partial-rollout" not in task.run
    assert task.file_mounts[L.CODEX_CONTAINER_DIR] == str(root.resolve())
    assert task.envs["YETO_CODEX_BINARY_PATH"] == CODEX_CONTAINER_BINARY_PATH
    assert task.envs[L.HARNESS_PREFLIGHT_ENV].endswith(":harness_preflight")
    assert task.envs[L.HARNESS_ENVIRONMENT_PROVIDER_ENV] == PROVIDER
    assert "YETO_CODEX_COMPACTION_ENABLED" not in task.envs


def test_modal_island_mounts_the_bundle_through_the_image(bundle, monkeypatch):
    root, _ = bundle
    args = _codex_args()
    args.gpu = "modal:1xh100"
    args.rl_image = "docker:ghcr.io/x/miles@sha256:" + "c" * 64
    spec = parse_gpu_spec(args.gpu)[0]
    task = make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400")
    cfg = build_modal_island_config(args, spec, 0, task, "1.2.3.4:29400")
    assert cfg.codex_dir == str(root.resolve()) and cfg.codex_mount == L.CODEX_CONTAINER_DIR
    assert cfg.envs["YETO_CODEX_BINARY_PATH"] == CODEX_CONTAINER_BINARY_PATH
    state = fake_modal(monkeypatch)
    monkeypatch.setattr(mr, "registry_credentials", lambda *_a, **_k: None)
    mr.ModalOps("yeto-run").define(cfg)
    (img,) = state["images"]
    names = [c[0] for c in img.calls]
    assert names == ["from_registry", "env", "add_local_dir", "add_local_dir"]
    assert img.calls[-1][1] == (cfg.codex_dir, L.CODEX_CONTAINER_DIR) and img.calls[-1][2] == {"copy": False}


def test_modal_config_requires_an_existing_bundle_dir(tmp_path):
    with pytest.raises(ValueError, match="go together"):
        mr.ModalIslandConfig(app_name="a", learner_id=0, training_mode="sft", gpu="H100",
                             gpus_per_node=8, num_nodes=1, run_script="x", codex_dir=str(tmp_path)).validate()
    with pytest.raises(ValueError, match="not a directory"):
        mr.ModalIslandConfig(app_name="a", learner_id=0, training_mode="sft", gpu="H100",
                             gpus_per_node=8, num_nodes=1, run_script="x",
                             codex_dir=str(tmp_path / "nope"), codex_mount="/opt/yeto/codex").validate()


@pytest.mark.skipif(
    not os.environ.get(L.CODEX_BUNDLE_DIR_ENV) or not os.path.isdir(os.environ.get(L.CODEX_BUNDLE_DIR_ENV, "")),
    reason="YETO_CODEX_BUNDLE_DIR not set to a fetched bundle (scripts/fetch_codex_bundle.py)",
)
def test_real_bundle_attests_against_the_pins():
    args = _codex_args()
    try:
        contract = L.codex_bundle_contract(args, os.environ[L.CODEX_BUNDLE_DIR_ENV])
    except ValueError as exc:
        if "app-server schema" in str(exc):
            pytest.xfail(f"pinned app-server schema file not reproducible from the npm artifact: {exc}")
        raise
    assert contract["binary_size_bytes"] == 310_730_800
    assert contract["openenv_identity_env"] == CODEX_OPENENV_IDENTITY_ENV


def test_modal_island_mounts_local_data_and_passes_tb2_env(bundle, monkeypatch, tmp_path):
    root, _ = bundle
    data = tmp_path / "smoke.jsonl"
    data.write_text('{"prompt": "x", "metadata": {"task_id": "fix-git"}}\n')
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-test")
    monkeypatch.setenv("YETO_HARNESS_TB2_TASKS_DIR", "/opt/yeto/codex/tb2-tasks")
    monkeypatch.setenv("YETO_HARNESS_TB2_FAULT", "create_fail:2")
    args = _codex_args()
    args.data = str(data)
    args.gpu = "modal:1xl40s"
    args.rl_image = "docker:ghcr.io/x/miles@sha256:" + "c" * 64
    spec = parse_gpu_spec(args.gpu)[0]
    task = make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400")
    cfg = build_modal_island_config(args, spec, 0, task, "1.2.3.4:29400")
    assert cfg.extra_mounts == {"/root/yeto-data.jsonl": str(data)}
    for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "YETO_HARNESS_TB2_TASKS_DIR", "YETO_HARNESS_TB2_FAULT"):
        assert cfg.envs[name] == os.environ[name]
    assert mr.ModalIslandConfig.from_json(cfg.to_json()).extra_mounts == cfg.extra_mounts
    state = fake_modal(monkeypatch)
    monkeypatch.setattr(mr, "registry_credentials", lambda *_a, **_k: None)
    mr.ModalOps("yeto-run").define(cfg)
    (img,) = state["images"]
    assert img.calls[-1][0] == "add_local_file"
    assert img.calls[-1][1] == (str(data), "/root/yeto-data.jsonl") and img.calls[-1][2] == {"copy": False}


def test_learner_command_forwards_the_codex_reasoning_effort(bundle):
    args = _codex_args()
    spec = parse_gpu_spec(args.gpu)[0]
    task = make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400")
    assert " --codex-reasoning-effort xhigh" in task.run
    assert " --codex-backend-profile qwen35_08b" in task.run
