"""worker_setup: one Ray hook runs the chain, in order, and survives a bad hook."""

from __future__ import annotations

import os

import pytest

from yeto.rl import worker_setup

MARK = "yeto_worker_setup_marks"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(worker_setup.HOOKS_ENV, raising=False)


def test_add_hook_is_ordered_and_deduplicated():
    env_vars: dict = {}
    worker_setup.add_hook(env_vars, "a:one")
    worker_setup.add_hook(env_vars, "b:two")
    worker_setup.add_hook(env_vars, "a:one")
    assert env_vars[worker_setup.HOOKS_ENV] == "a:one,b:two"
    assert os.environ[worker_setup.HOOKS_ENV] == "a:one,b:two"


def test_setup_worker_runs_every_hook_and_skips_a_failing_one(monkeypatch):
    calls: list[str] = []
    module = type(os)("yeto_fake_hooks")
    module.ok = lambda: calls.append("ok")  # type: ignore[attr-defined]
    module.boom = lambda: (_ for _ in ()).throw(RuntimeError("no"))  # type: ignore[attr-defined]
    module.last = lambda: calls.append("last")  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "yeto_fake_hooks", module)
    monkeypatch.setenv(worker_setup.HOOKS_ENV,
                       "yeto_fake_hooks:ok,yeto_fake_hooks:boom, yeto_fake_hooks:last ,")
    worker_setup.setup_worker()
    assert calls == ["ok", "last"]


def test_gloo_hook_name_resolves():
    from yeto.swift_megatron import cross_node_gloo

    module, _, attr = cross_node_gloo.SETUP_HOOK.partition(":")
    assert getattr(__import__(module, fromlist=[attr]), attr) is cross_node_gloo.install
