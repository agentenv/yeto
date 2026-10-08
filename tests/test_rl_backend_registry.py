"""yeto-framework-decoupling 5.4 + 4.11: backend registry and the neutral
rollout-metadata port (design D11 方案 A; unset YETO_RL_BACKEND = miles)."""

from __future__ import annotations

import argparse
import types

import pytest

from yeto.rl.engine import backends, rollout_meta


def test_default_backend_is_miles_when_env_unset(monkeypatch):
    monkeypatch.delenv(backends.BACKEND_ENV, raising=False)
    assert backends.active_name() == "miles"
    assert rollout_meta.port().__name__ == "yeto.rl.adapters.miles.rollout_meta_hook"


def test_env_selects_backend_and_unknown_is_refused(monkeypatch):
    monkeypatch.setenv(backends.BACKEND_ENV, "verl")
    with pytest.raises(backends.UnknownBackend, match="未注册"):
        rollout_meta.current_policy_token()
    with pytest.raises(backends.UnknownBackend, match="未注册"):
        backends.get("nope")
    with pytest.raises(backends.UnknownBackend, match="没有提供"):
        backends.module("no-such-role", "miles")


def test_registered_backend_supplies_the_rollout_port(monkeypatch):
    calls = []
    fake = types.ModuleType("fake_rollout_meta")
    fake.sink_available = lambda: True
    fake.current_policy_token = lambda: "yeto:3:abc"
    fake.expected_policy_version = lambda sample: "v"
    fake.record_round_metadata = lambda args, rid, **c: calls.append((rid, c))
    import sys

    monkeypatch.setitem(sys.modules, "fake_rollout_meta", fake)
    backends.register(backends.BackendEntry("fake", "fake", {"rollout_meta": "fake_rollout_meta"}))
    try:
        monkeypatch.setenv(backends.BACKEND_ENV, "fake")
        assert rollout_meta.sink_available() is True
        assert rollout_meta.current_policy_token() == "yeto:3:abc"
        assert rollout_meta.expected_policy_version(None) == "v"
        rollout_meta.record_round_metadata(None, 3, nonzero_advantages=2)
        assert calls == [(3, {"nonzero_advantages": 2})]
    finally:
        backends.unregister("fake")


def test_miles_port_keeps_key_names_and_counter_semantics():
    from yeto.rl.adapters.miles import rollout_meta_hook as hook

    assert hook.EXPECTED_POLICY_VERSION_KEY == "expected_policy_version"
    assert hook.POLICY_AGE_VIOLATION_KEY == "policy_age_violation"
    assert hook.TITO_SESSION_MISMATCH_KEY == "tito_session_mismatch"
    assert hook.TITO_CHAIN_BREAKS_KEY == "tito_chain_breaks"
    assert hook.counter_value is rollout_meta.counter_value
    assert [rollout_meta.counter_value(v) for v in (None, "", True, 3, [1, 2], {"a": 1, "b": 0})] \
        == [0, 0, 1, 3, 2, 1]


def test_miles_port_reads_the_dir_sink(monkeypatch, tmp_path):
    from yeto.rl.adapters.miles import rollout_meta_hook as hook

    monkeypatch.delenv(backends.BACKEND_ENV, raising=False)
    monkeypatch.setenv(hook.META_SINK_ENV, f"dir:{tmp_path}")
    assert rollout_meta.sink_available()
    hook.put_policy_token("yeto:7:h")
    assert rollout_meta.current_policy_token() == "yeto:7:h"
    sample = types.SimpleNamespace(metadata={"expected_policy_version": "yeto:1:x"})
    assert rollout_meta.expected_policy_version(sample) == "yeto:1:x"
    assert rollout_meta.expected_policy_version(None) == "yeto:7:h"


def test_island_ray_job_env_names_the_backend():
    from yeto.rl.adapters.miles.entry import connect_island_ray

    seen = {}

    class FakeRay:
        @staticmethod
        def is_initialized():
            return False

        @staticmethod
        def init(address, runtime_env):
            seen.update(runtime_env["env_vars"])

    connect_island_ray(environ={"RAY_ADDRESS": "10.0.0.1:6379"}, ray_module=FakeRay)
    assert seen[backends.BACKEND_ENV] == "miles"


def test_cli_rl_backend_flag_and_launcher_refuses_unregistered():
    from yeto import launcher

    args = argparse.Namespace(training_mode="rl", rl_backend="verl", rl_image="x")
    with pytest.raises(ValueError, match="未注册"):
        launcher.resolve_default_rl_image(args)
    args = argparse.Namespace(training_mode="rl", rl_backend="miles", rl_image="x")
    launcher.resolve_default_rl_image(args)
    assert launcher._rl_backend_module(args, "placement").__name__ == \
        "yeto.rl.adapters.miles.placement"
