"""worker_setup: one Ray hook runs the chain in order, resolves hook names as Ray does, fails fast."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from yeto.rl import worker_setup

MARK = "yeto_worker_setup_marks"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(worker_setup.HOOKS_ENV, raising=False)


def test_add_hook_is_ordered_deduplicated_and_leaves_process_env_alone():
    env_vars: dict = {}
    worker_setup.add_hook(env_vars, "a:one")
    worker_setup.add_hook(env_vars, "b.two")
    worker_setup.add_hook(env_vars, "a:one")
    assert env_vars[worker_setup.HOOKS_ENV] == "a:one,b.two"
    assert worker_setup.HOOKS_ENV not in os.environ


def _fake(monkeypatch, calls):
    module = type(os)("yeto_fake_hooks")
    module.ok = lambda: calls.append("ok")  # type: ignore[attr-defined]
    module.boom = lambda: (_ for _ in ()).throw(RuntimeError("no"))  # type: ignore[attr-defined]
    module.last = lambda: calls.append("last")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yeto_fake_hooks", module)


def test_setup_worker_runs_colon_and_dotted_hooks_in_order(monkeypatch):
    calls: list[str] = []
    _fake(monkeypatch, calls)
    monkeypatch.setenv(worker_setup.HOOKS_ENV, "yeto_fake_hooks:ok, yeto_fake_hooks.last ,")
    worker_setup.setup_worker()
    assert calls == ["ok", "last"]


def test_failing_hook_raises(monkeypatch):
    calls: list[str] = []
    _fake(monkeypatch, calls)
    monkeypatch.setenv(worker_setup.HOOKS_ENV, "yeto_fake_hooks:ok,yeto_fake_hooks:boom,yeto_fake_hooks:last")
    with pytest.raises(RuntimeError):
        worker_setup.setup_worker()
    assert calls == ["ok"]


@pytest.mark.parametrize("hook", ["no_such_module_xyz.fn", "yeto_fake_hooks.missing", "nodots"])
def test_unresolvable_hook_raises(monkeypatch, hook):
    _fake(monkeypatch, [])
    monkeypatch.setenv(worker_setup.HOOKS_ENV, hook)
    with pytest.raises(Exception):
        worker_setup.setup_worker()


def test_real_hooks_resolve_like_ray():
    from yeto.rl import ascend_plugin_stack
    from yeto.rl.adapters.verl import agentic_main
    from yeto.swift_megatron import cross_node_gloo

    assert worker_setup.resolve(agentic_main.SETUP_HOOK) is agentic_main.setup_worker
    assert worker_setup.resolve(ascend_plugin_stack.SETUP_HOOK) is ascend_plugin_stack.setup_worker
    assert worker_setup.resolve(cross_node_gloo.SETUP_HOOK) is cross_node_gloo.install


def _fake_verl(root):
    """A stand-in for verl's core_algos registry (verl is not installed in CI)."""
    pkg = root / "verl" / "trainer" / "ppo"
    pkg.mkdir(parents=True)
    for d in (root / "verl", root / "verl" / "trainer", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "core_algos.py").write_text(
        "REG = {}\n"
        "def register_adv_est(name):\n"
        "    def deco(fn):\n"
        "        REG[getattr(name, 'value', name)] = fn\n"
        "        return fn\n"
        "    return deco\n")
    (root / "verl" / "trainer" / "constants_ppo.py").write_text(
        "def get_ppo_ray_runtime_env(_=None):\n    return {'env_vars': {}}\n")


@pytest.mark.parametrize("ascend", [False, True])
def test_real_hook_strings_run_and_register_yeto_grpo(tmp_path, ascend):
    """The real, dotted hook strings of agentic_main and ascend_plugin_stack, run
    through the chain in a fresh worker-like interpreter: agentic_main.setup_worker
    really runs, and importing core_algos afterwards registers yeto_grpo (the
    real estimator module, not a stand-in)."""
    _fake_verl(tmp_path)
    code = textwrap.dedent("""
        import os, sys
        from yeto.rl import ascend_plugin_stack, worker_setup
        from yeto.rl.adapters.verl import agentic_main
        ran = []
        real = ascend_plugin_stack.setup_worker
        ascend_plugin_stack.setup_worker = lambda: ran.append("ascend")
        hooks = [agentic_main.SETUP_HOOK]
        if os.environ.get("ASCEND") == "1":
            hooks.insert(0, ascend_plugin_stack.SETUP_HOOK)
        os.environ[worker_setup.HOOKS_ENV] = ",".join(hooks)
        assert "." in agentic_main.SETUP_HOOK and ":" not in agentic_main.SETUP_HOOK
        worker_setup.setup_worker()
        assert ran == (["ascend"] if os.environ.get("ASCEND") == "1" else []), ran
        assert any(isinstance(f, agentic_main._AfterImport) for f in sys.meta_path)
        from verl.trainer.ppo import core_algos
        assert "yeto_grpo" in core_algos.REG, core_algos.REG
        print("ok")
    """)
    env = dict(os.environ, PYTHONPATH=f"{tmp_path}:{os.getcwd()}", ASCEND="1" if ascend else "0")
    env.pop(worker_setup.HOOKS_ENV, None)
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, env=env)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_real_ascend_hook_runs_through_chain(monkeypatch):
    from yeto.rl import ascend_plugin_stack

    called: list[int] = []
    monkeypatch.setattr(ascend_plugin_stack, "setup_worker", lambda: called.append(1))
    monkeypatch.setenv(worker_setup.HOOKS_ENV, ascend_plugin_stack.SETUP_HOOK)
    worker_setup.setup_worker()
    assert called == [1]


def test_agentic_runtime_env_without_gloo_matches_main(tmp_path, monkeypatch):
    _fake_verl(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [k for k in sys.modules if k == "verl" or k.startswith("verl.")]:
        monkeypatch.delitem(sys.modules, name)
    from yeto.rl.adapters.verl import agentic_main

    env = agentic_main.runtime_env({"PATH": "/bin"})
    assert env["worker_process_setup_hook"] == agentic_main.SETUP_HOOK  # exactly as on main
    assert worker_setup.HOOKS_ENV not in env["env_vars"]
    env = agentic_main.runtime_env({"PATH": "/bin", "YETO_CROSS_NODE_GLOO": "1"})
    assert env["worker_process_setup_hook"] == worker_setup.SETUP_HOOK
    assert env["env_vars"][worker_setup.HOOKS_ENV].split(",") == [
        "yeto.swift_megatron.cross_node_gloo:install", agentic_main.SETUP_HOOK]
