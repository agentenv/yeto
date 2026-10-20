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
