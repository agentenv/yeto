"""S25 N10.9: the verl agentic entry's Ray setup hook imports nothing CUDA-related (CPU)."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_setup_hook_defers_the_estimator_until_core_algos_loads(tmp_path):
    # Stand-ins for verl's core_algos and yeto's estimator module, in a fresh interpreter.
    (tmp_path / "fakeverl").mkdir()
    (tmp_path / "fakeverl" / "__init__.py").write_text("")
    (tmp_path / "fakeverl" / "core_algos.py").write_text("REG = []\n")
    (tmp_path / "fakeest.py").write_text("import fakeverl.core_algos as c\nc.REG.append('yeto_grpo')\n")
    code = textwrap.dedent("""
        import sys
        from yeto.rl.adapters.verl import agentic_main as m
        m.ESTIMATOR_HOST, m.ESTIMATOR_MODULE = "fakeverl.core_algos", "fakeest"
        m.setup_worker()
        assert "fakeverl.core_algos" not in sys.modules and "torch" not in sys.modules
        assert not any(k == "verl" or k.startswith("verl.") for k in sys.modules)
        import fakeverl.core_algos as c
        assert c.REG == ["yeto_grpo"], c.REG
        m.setup_worker()  # idempotent once loaded
        assert c.REG == ["yeto_grpo"]
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
                         env={"PYTHONPATH": f"{tmp_path}:{__import__('os').getcwd()}"})
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_main_selects_the_fully_async_entry_and_leaves_transfer_queue_to_it(monkeypatch):
    import types

    from yeto.rl.adapters.verl import agentic_main as m

    calls = []
    fake_ray = types.SimpleNamespace(is_initialized=lambda: False,
                                     init=lambda **kw: calls.append(("init", kw["runtime_env"])))
    entries = {name: types.SimpleNamespace(main=lambda name=name: calls.append(("main", name, list(sys.argv[1:]))))
               for name in ("verl.trainer.main_ppo", "verl.experimental.fully_async_policy.fully_async_main")}
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(m, "runtime_env", lambda fully_async=False: {"fully_async": fully_async})
    monkeypatch.setattr("yeto.launch.image_check.check_image_at_start", lambda: ())
    monkeypatch.setattr(m.importlib, "import_module", entries.get)
    monkeypatch.setattr(sys, "argv", ["agentic_main"])
    monkeypatch.setattr(m.runpy, "run_module", lambda name, run_name, alter_sys: calls.append(
        ("main", name, run_name, list(sys.argv[1:]))))

    m.main(["--fully-async", "a=1"])
    m.main(["a=2"])
    assert calls == [("init", {"fully_async": True}),
                     ("main", "verl.experimental.fully_async_policy.fully_async_main", "__main__", ["a=1"]),
                     ("init", {"fully_async": False}), ("main", "verl.trainer.main_ppo", ["a=2"])]
