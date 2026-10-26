"""S27 G-FL: the Ascend recipe's Megatron plugin stack as a verl-island opt-in.

The real plugins live on the Ascend host, so these tests install a fake directory and
check the table, the refusals and the per-plugin report. ``YETO_ASCEND_RECIPE_DIR``
(the recipe's ``plugins`` directory) turns on the one test that compares the table
with the delivery itself; it is skipped otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from yeto.rl import ascend_plugin_stack as stack

REPO = Path(__file__).resolve().parents[1]


def test_the_switch_set_is_the_recipes_and_never_enables_clamp():
    assert stack.RECIPE_ENV["STACK_REWRITES"] == "cumsum,squeeze" == stack.STACK_REWRITES
    assert not [key for key, value in stack.RECIPE_ENV.items() if "clamp" in value.lower()]
    assert stack.recipe_env()["ASCEND_GRAD_SCALE_FIX"] == "1"
    assert stack.recipe_env({"ASCEND_MOE_GMM_SPLITS": "0"})["ASCEND_MOE_GMM_SPLITS"] == "0"
    for value in ("cumsum,squeeze,clamp", "clamp", " clamp , cumsum"):
        with pytest.raises(stack.AscendPluginStackError, match="clamp"):
            stack.recipe_env({"STACK_REWRITES": value})
    # the recipe's own variables that this switch deliberately leaves alone
    assert set(stack.UNSET) == {"ASCEND_RECOMP_POLICY", "HCCL_BUFFSIZE", "ASCEND_BENCH_NO_SAVE",
                                "ASCEND_STAGE_TIMER", "ASCEND_OUTSIDE_PROBE", "ASCEND_COMM_CENSUS",
                                "ASCEND_RECOMP_MEMLOG"}
    assert set(stack.UNSET).isdisjoint(stack.RECIPE_ENV)
    assert all(reason for reason in stack.UNSET.values()) and all(reason for reason in stack.EXCLUDED.values())
    assert set(stack.PLUGINS).isdisjoint(stack.EXCLUDED) and len(set(stack.PLUGINS)) == len(stack.PLUGINS)


def test_the_plan_reports_which_plugin_files_are_there(tmp_path):
    (tmp_path / stack.PLUGINS[0]).write_text("")
    report = stack.plan(str(tmp_path) + "/")
    assert report["root"] == str(tmp_path)
    assert [row["plugin"] for row in report["plugins"]] == list(stack.PLUGINS)
    assert report["plugins"][0] == {"plugin": stack.PLUGINS[0],
                                    "path": f"{tmp_path}/{stack.PLUGINS[0]}", "present": True}
    assert [row["present"] for row in report["plugins"][1:]] == [False] * (len(stack.PLUGINS) - 1)
    assert report["env"] == stack.recipe_env() and report["excluded"] == stack.EXCLUDED


def test_install_sets_the_switches_first_then_imports_every_plugin_and_reports_each(tmp_path, monkeypatch):
    """A plugin installs itself on import and reads its switch there, so the switches
    must be in the process environment before the first import. One plugin raises and
    one file is missing: both are reported and the plugins after them still run."""
    for key in stack.RECIPE_ENV:  # also undoes what install() sets
        monkeypatch.delenv(key, raising=False)
    marker = tmp_path / "seen.txt"
    (tmp_path / "ok_first.py").write_text(
        "import os, pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text(os.environ['STACK_REWRITES'] + '|' "
        "+ os.environ['ASCEND_MOE_FAST_SORT'])\n")
    (tmp_path / "boom.py").write_text("raise RuntimeError('no kernel here')\n")
    (tmp_path / "ok_last.py").write_text("VALUE = 1\n")
    lines: list[str] = []
    report = stack.install(str(tmp_path), log=lines.append,
                           plugins=["ok_first.py", "boom.py", "gone.py", "ok_last.py"])
    assert marker.read_text() == "cumsum,squeeze|1"  # the switches were set before the import
    assert os.environ["STACK_REWRITES"] == "cumsum,squeeze"
    assert report["installed"] == ["ok_first.py", "ok_last.py"]
    assert report["failed"] == {"boom.py": "RuntimeError: no kernel here", "gone.py": "file not found"}
    assert stack.LAST_REPORT is report
    # the island log names every switch, every refusal and every plugin
    text = "\n".join(lines)
    for key, value in stack.RECIPE_ENV.items():
        assert f"set {key}={value}" in text
    for name in (*stack.UNSET, *stack.EXCLUDED):
        assert name in text
    assert "NOT installed boom.py: RuntimeError: no kernel here" in text
    assert "NOT installed gone.py: file not found" in text
    assert "installed ok_last.py" in text


def test_install_can_write_the_switches_somewhere_other_than_the_process(tmp_path):
    """``environ`` is the mapping the switches go into; the process is left alone."""
    before = dict(os.environ)
    environ: dict[str, str] = {}
    stack.install(str(tmp_path), environ=environ, plugins=[], log=lambda line: None)
    assert environ == stack.recipe_env()
    assert dict(os.environ) == before


def test_install_refuses_a_switch_set_that_names_clamp(tmp_path):
    with pytest.raises(stack.AscendPluginStackError, match="clamp"):
        stack.install(str(tmp_path), environ={}, extra_env={"STACK_REWRITES": "cumsum,clamp"},
                      plugins=[], log=lambda line: None)


def test_the_worker_hook_does_nothing_without_the_switch(monkeypatch, tmp_path):
    monkeypatch.delenv(stack.ENV, raising=False)
    monkeypatch.setattr(stack, "LAST_REPORT", None)
    stack.setup_worker()
    assert stack.LAST_REPORT is None
    monkeypatch.setenv(stack.ENV, str(tmp_path))
    stack.setup_worker()  # the hook installs from the island's directory
    assert stack.LAST_REPORT is not None and stack.LAST_REPORT["root"] == str(tmp_path)
    assert sorted(stack.LAST_REPORT["failed"]) == sorted(stack.PLUGINS)  # the fake dir is empty
    # the hook is the name Ray is given, and it is importable
    import importlib

    module, _, name = stack.SETUP_HOOK.rpartition(".")
    assert getattr(importlib.import_module(module), name) is stack.setup_worker


_RECIPE_DIR = os.environ.get("YETO_ASCEND_RECIPE_DIR")


@pytest.mark.skipif(not _RECIPE_DIR, reason="needs YETO_ASCEND_RECIPE_DIR (the recipe's plugins dir)")
def test_the_table_covers_the_recipes_own_plugin_directory():
    """Every plugin this switch installs exists in the delivery, and every plugin of the
    recipe's load order is either installed or in ``EXCLUDED`` with a reason."""
    present = {path.name for path in Path(_RECIPE_DIR).glob("*.py")}
    assert set(stack.PLUGINS) <= present, sorted(set(stack.PLUGINS) - present)
    # EXCLUDED also names a file outside the directory (the optimizer timing injector)
    assert set(stack.EXCLUDED) - present == {"optimizer_phase_timing_inject.py"}
