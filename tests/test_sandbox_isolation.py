"""rl-agentic-reward-env 5.1/5.2: sandbox egress closed by default, minimal local env."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from yeto.rl.harness.codex import tb2_provider as tb2
from yeto.rl.harness.codex.tb2_network_grants import TB2_OPEN_EGRESS_TASKS


def _task(tmp_path: Path, task_id: str = "t1") -> tb2.Tb2Task:
    return tb2.Tb2Task(task_id=task_id, task_dir=tmp_path, docker_image="img:1", cpus=1, memory_mb=2048,
                       workdir="/app", agent_timeout_s=60, verifier_timeout_s=60)


def test_default_policy_is_closed():
    grant = tb2.NetworkPolicy().grant("anything")
    assert grant.closed
    assert grant.modal_kwargs() == {"block_network": True}


def test_grant_forms():
    policy = tb2.NetworkPolicy.from_mapping({"tasks": {
        "pip": {"domains": ["pypi.org", "*.pythonhosted.org"]},
        "net": {"cidrs": ["10.0.0.0/8"]},
        "web": "open",
        "off": "closed",
    }})
    assert policy.grant("pip").modal_kwargs() == {"outbound_domain_allowlist": ["pypi.org", "*.pythonhosted.org"]}
    assert policy.grant("net").modal_kwargs() == {"outbound_cidr_allowlist": ["10.0.0.0/8"]}
    assert policy.grant("web").modal_kwargs() == {}
    assert policy.grant("off").modal_kwargs() == {"block_network": True}
    assert policy.grant("unlisted").modal_kwargs() == {"block_network": True}


@pytest.mark.parametrize("bad", [
    {"tasks": {"x": {"domains": ["*"]}}},
    {"tasks": {"x": {"cidrs": ["not-a-cidr"]}}},
    {"tasks": {"x": {"hosts": ["a"]}}},
    {"tasks": {"x": "yes"}},
    {"tasks": [], "default": "closed"},
    {"extra": 1},
])
def test_bad_policy_fails_closed(bad):
    with pytest.raises(ValueError):
        tb2.NetworkPolicy.from_mapping(bad)


def test_builtin_grants_every_tb2_task_open_explicitly():
    assert len(TB2_OPEN_EGRESS_TASKS) == 89 == len(set(TB2_OPEN_EGRESS_TASKS))
    policy = tb2.NetworkPolicy.builtin()
    assert policy.default.closed
    for task_id in ("adaptive-rejection-sampler", "build-pmars", "qemu-startup"):
        assert policy.grant(task_id).open
    assert policy.grant("not-a-tb2-task").closed


def test_policy_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv(tb2.NETWORK_POLICY_ENV, raising=False)
    assert tb2.NetworkPolicy.from_env() == tb2.NetworkPolicy.builtin()
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"tasks": {"pip": {"domains": ["pypi.org"]}}}))
    monkeypatch.setenv(tb2.NETWORK_POLICY_ENV, str(path))
    policy = tb2.NetworkPolicy.from_env()
    assert policy.grant("pip").domains == ("pypi.org",)
    assert policy.grant("adaptive-rejection-sampler").closed


def _fake_modal(monkeypatch, calls):
    mod = types.ModuleType("modal")

    class Sandbox:
        @staticmethod
        def create(*args, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(object_id="sb-1")

    class Image:
        @staticmethod
        def from_registry(name):
            return types.SimpleNamespace(name=name, run_commands=lambda *a: "img")

    class App:
        @staticmethod
        def lookup(name, create_if_missing=False):
            return "app"

    mod.Sandbox, mod.Image, mod.App = Sandbox, Image, App
    monkeypatch.setitem(sys.modules, "modal", mod)


def test_modal_backend_passes_network_kwargs(tmp_path, monkeypatch):
    calls: list[dict] = []
    _fake_modal(monkeypatch, calls)
    tb2.ModalSandboxBackend().create(_task(tmp_path), "traj")
    assert calls[-1]["block_network"] is True
    policy = tb2.NetworkPolicy.from_mapping({"tasks": {"t1": {"domains": ["pypi.org"]}}})
    tb2.ModalSandboxBackend(network_policy=policy).create(_task(tmp_path), "traj")
    assert calls[-1]["outbound_domain_allowlist"] == ["pypi.org"]
    assert "block_network" not in calls[-1]


def test_prebaked_modal_backend_passes_network_kwargs(tmp_path, monkeypatch):
    from yeto.cloud import modal_reward_env

    calls: list[dict] = []
    _fake_modal(monkeypatch, calls)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test.sh").write_text("#!/bin/bash\necho 1\n")
    modal_reward_env.PrebakedModalSandboxBackend(prebake=False).create(_task(tmp_path), "traj")
    assert calls[-1]["block_network"] is True
    backend = modal_reward_env.PrebakedModalSandboxBackend(prebake=False, network_policy=tb2.NetworkPolicy.builtin())
    backend.create(_task(tmp_path, "adaptive-rejection-sampler"), "traj")
    assert not {"block_network", "outbound_domain_allowlist", "outbound_cidr_allowlist"} & set(calls[-1])


def test_minimal_env_drops_tokens():
    parent = {"PATH": "/x/bin", "MODAL_TOKEN_SECRET": "s", "HF_TOKEN": "h", "TBENCH_REWARD_HMAC_KEY": "k",
              "WANDB_API_KEY": "w", "LANG": "en_US.UTF-8", "HOME": "/home/u"}
    assert tb2.minimal_env(parent) == {"PATH": "/x/bin", "LANG": "en_US.UTF-8"}
    assert tb2.minimal_env({})["PATH"] == tb2.DEFAULT_PATH


def test_local_sandbox_does_not_inherit_parent_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "secret-value")
    monkeypatch.setenv("YETO_FAKE_TOKEN", "x")
    backend = tb2.LocalProcessBackend(tmp_path)
    sandbox = backend.create(_task(tmp_path), "traj")
    try:
        result = sandbox.exec("env", timeout_s=10)
    finally:
        sandbox.terminate()
    assert result.exit_code == 0
    names = {line.split("=", 1)[0] for line in result.output.splitlines() if "=" in line}
    assert "MODAL_TOKEN_SECRET" not in names and "YETO_FAKE_TOKEN" not in names
    assert {"PATH", "HOME", "TB2_TESTS_DIR", "TB2_VERIFIER_LOGS_DIR"} <= names
    assert sandbox.network.closed
