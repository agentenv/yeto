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


def test_install_overwrites_an_inherited_stack_rewrites_that_names_clamp(tmp_path, monkeypatch):
    """S27 G-FW: the write is the layer that keeps ``clamp`` out, so it must win.

    A Ray worker of an already-running cluster inherits the raylet's environment, and
    the delivery reads the variable with a default that DOES name clamp
    (``kern-stack/mamba2_stack.py:87``: ``cumsum,squeeze,clamp``). So an inherited
    value must be replaced, and replaced before the first plugin import reads it."""
    monkeypatch.setenv("STACK_REWRITES", "cumsum,squeeze,clamp")
    marker = tmp_path / "read_by_plugin.txt"
    (tmp_path / "reader.py").write_text(
        "import os, pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text(os.environ['STACK_REWRITES'])\n")
    report = stack.install(str(tmp_path), plugins=["reader.py"], log=lambda line: None)
    assert marker.read_text() == "cumsum,squeeze"
    assert os.environ["STACK_REWRITES"] == "cumsum,squeeze"
    assert report["env"]["STACK_REWRITES"] == "cumsum,squeeze"
    # the entry has to be in the table at all; without it nothing is written
    assert "STACK_REWRITES" in stack.RECIPE_ENV


def test_the_worker_hook_does_nothing_without_the_switch(monkeypatch, tmp_path):
    monkeypatch.delenv(stack.ENV, raising=False)
    monkeypatch.setattr(stack, "LAST_REPORT", None)
    stack.setup_worker()
    assert stack.LAST_REPORT is None
    monkeypatch.setenv(stack.ENV, str(tmp_path))
    monkeypatch.setenv(stack.RANK_ENV, "3")  # a process that already owns a card installs at once
    stack.setup_worker()  # the hook installs from the island's directory
    assert stack.LAST_REPORT is not None and stack.LAST_REPORT["root"] == str(tmp_path)
    assert sorted(stack.LAST_REPORT["failed"]) == sorted(stack.PLUGINS)  # the fake dir is empty
    # the hook is the name Ray is given, and it is importable
    import importlib

    module, _, name = stack.SETUP_HOOK.rpartition(".")
    assert getattr(importlib.import_module(module), name) is stack.setup_worker


# --------------------------------------------------------------------------- S27 G-FY
# The defect these tests hold down (S27 G-FW, measured on npu1 n2, 8 x 910B3):
# ``ascend_mamba2_kernels.apply`` calls ``torch.npu.set_device(_local_device())``
# (plugins/ascend_mamba2_kernels.py:352-356) and ``_local_device`` returns 0 when
# LOCAL_RANK is unset (:62). Ray runs ``worker_process_setup_hook`` before it leases
# the process to an actor, so LOCAL_RANK did not exist yet and all eight trainer ranks
# bound card 0: the island log read ``(Device:0, RankID:0)`` .. ``(Device:0, RankID:7)``,
# cards 1-7 took no allocation at all, and HCCL could not build the eight-rank domain.
# So the thing to assert is NOT "the plugins were imported" -- they were, that was the
# bug -- but "nothing was imported while this process did not own a card".


class _FakeVerlWorker:
    """Stands in for ``verl.single_controller.base.worker.Worker``: its ``__init__``
    writes LOCAL_RANK and binds the card, as verl's does (worker.py:275-280)."""

    seen: list[str] = []

    def __init__(self, local_rank: str = "5"):
        os.environ["LOCAL_RANK"] = local_rank
        type(self).seen.append(local_rank)


@pytest.fixture
def armable(monkeypatch):
    """A fresh process-local state plus a fake verl Worker to defer onto."""
    import sys
    import types

    monkeypatch.setattr(stack, "LAST_REPORT", None)
    monkeypatch.delenv(stack.RANK_ENV, raising=False)
    module = types.ModuleType("fake_verl_worker_module")
    module.Worker = type("Worker", (_FakeVerlWorker,), {"seen": []})
    monkeypatch.setitem(sys.modules, "fake_verl_worker_module", module)
    monkeypatch.setattr(stack, "VERL_WORKER", ("fake_verl_worker_module", "Worker"))
    return module.Worker


def _device_touching_plugin(directory: Path, log: Path, name: str = "device_binder.py") -> str:
    """A stand-in for the delivery's Mamba2 plugin: on import it picks a device from
    LOCAL_RANK with the delivery's own fallback (0) and records which one it bound.

    ``name`` differs per stand-in process on purpose: rewriting one file keeps its size
    and (within the second) its mtime, so ``SourceFileLoader`` would serve the previous
    body out of ``__pycache__`` and the test would read a stale device."""
    (directory / name).write_text(
        "import os, pathlib\n"
        "device = int(os.environ.get('LOCAL_RANK', 0))  # ascend_mamba2_kernels.py:62\n"
        f"pathlib.Path({str(log)!r}).write_text(str(device))  # torch.npu.set_device(...)\n")
    return name


def test_the_hook_imports_no_plugin_while_this_process_owns_no_card(tmp_path, monkeypatch, armable):
    """The regression: with no LOCAL_RANK, the hook must not import a plugin at all.

    Mutation check: make :func:`setup_worker` call ``install(root)`` directly again and
    this fails on the first assert -- the plugin is imported and binds card 0."""
    bound = tmp_path / "bound_device.txt"
    name = _device_touching_plugin(tmp_path, bound)
    monkeypatch.setattr(stack, "PLUGINS", (name,))
    monkeypatch.setenv(stack.ENV, str(tmp_path))

    stack.setup_worker()  # Ray's setup hook, in a process Ray has not leased to an actor
    assert not bound.exists(), f"a plugin bound card {bound.read_text()} before this process had one"
    assert stack.LAST_REPORT is None  # nothing installed yet, so the island log says so

    armable(local_rank="5")  # verl leases the process: LOCAL_RANK appears, the card is bound
    assert bound.read_text() == "5"  # and only now was the plugin imported
    assert stack.LAST_REPORT is not None and stack.LAST_REPORT["installed"] == [name]


def test_every_rank_binds_its_own_card_not_card_zero(tmp_path, monkeypatch, armable):
    """Eight worker processes, each leased a different card: each must bind its own.

    This is the shape of the failure itself. Under the old hook every one of these
    recorded 0 (S27 G-FW's ``(Device:0, RankID:0..7)``)."""
    monkeypatch.setenv(stack.ENV, str(tmp_path))
    bound: list[str] = []
    for rank in range(8):
        log = tmp_path / f"bound_{rank}.txt"
        name = _device_touching_plugin(tmp_path, log, name=f"device_binder_{rank}.py")
        monkeypatch.setattr(stack, "PLUGINS", (name,))
        monkeypatch.setattr(stack, "LAST_REPORT", None)  # a separate worker process
        monkeypatch.delenv(stack.RANK_ENV, raising=False)
        stack.setup_worker()
        assert not log.exists()
        armable(local_rank=str(rank))
        bound.append(log.read_text())
    assert bound == [str(rank) for rank in range(8)]
    assert len(set(bound)) == 8, f"the ranks did not spread over the cards: {bound}"


def test_a_worker_that_names_no_card_installs_nothing_even_once_armed(tmp_path, monkeypatch, armable):
    """The second line of defence, and the reason the check sits in the install and not
    only in the hook: the deferral fires when the verl worker is built, but a worker
    that still carries no LOCAL_RANK must make the install refuse, not import a plugin
    that would then bind card 0."""
    bound = tmp_path / "bound_device.txt"
    name = _device_touching_plugin(tmp_path, bound)
    monkeypatch.setattr(stack, "PLUGINS", (name,))
    monkeypatch.setenv(stack.ENV, str(tmp_path))
    silent = armable
    monkeypatch.setattr(silent, "__init__", lambda self: None, raising=False)  # writes no LOCAL_RANK
    stack.setup_worker()
    lines: list[str] = []
    assert stack._install_once(str(tmp_path), log=lines.append) is None
    assert not bound.exists() and stack.LAST_REPORT is None
    assert f"NOT installed: no {stack.RANK_ENV}" in "\n".join(lines)


def test_the_deferred_install_runs_once_per_process(tmp_path, monkeypatch, armable):
    """``WorkerDict.__init__`` builds one verl worker per role in the same process
    (verl/single_controller/ray/base.py:1028-1040), so the plugins must be imported
    once: a second import re-applies every monkeypatch and re-runs the prewarm."""
    counter = tmp_path / "imports.txt"
    (tmp_path / "counting.py").write_text(
        "import pathlib\n"
        f"path = pathlib.Path({str(counter)!r})\n"
        "path.write_text(str(int(path.read_text() or 0) + 1) if path.exists() else '1')\n")
    monkeypatch.setattr(stack, "PLUGINS", ("counting.py",))
    monkeypatch.setenv(stack.ENV, str(tmp_path))
    stack.setup_worker()
    worker = armable
    worker("2"), worker("2"), worker("2")
    assert worker.seen == ["2", "2", "2"] and counter.read_text() == "1"


def test_without_the_verl_seam_nothing_is_installed_rather_than_pinned(tmp_path, monkeypatch):
    """Not installing is the safe failure; installing without a card is not.

    Without the stack the island runs without the recipe's kernels. With the stack
    installed blind, every rank binds card 0 and the island cannot start at all."""
    monkeypatch.setattr(stack, "LAST_REPORT", None)
    monkeypatch.delenv(stack.RANK_ENV, raising=False)
    monkeypatch.setattr(stack, "VERL_WORKER", ("yeto_no_such_verl_module", "Worker"))
    bound = tmp_path / "bound_device.txt"
    name = _device_touching_plugin(tmp_path, bound)
    monkeypatch.setattr(stack, "PLUGINS", (name,))
    lines: list[str] = []
    assert stack.install_when_rank_known(str(tmp_path), log=lines.append) == "unavailable"
    assert not bound.exists() and stack.LAST_REPORT is None
    assert "NOT installed" in "\n".join(lines) and stack.RANK_ENV in "\n".join(lines)


def test_the_deferred_install_still_overwrites_an_inherited_clamp_rewrite(tmp_path, monkeypatch, armable):
    """The clamp red line must survive the deferral: ``install`` writes STACK_REWRITES
    before the first plugin import, and the deferral only moves *when* that pair runs."""
    monkeypatch.setenv("STACK_REWRITES", "cumsum,squeeze,clamp")
    read = tmp_path / "read_by_plugin.txt"
    (tmp_path / "reader.py").write_text(
        "import os, pathlib\n"
        f"pathlib.Path({str(read)!r}).write_text(os.environ['STACK_REWRITES'])\n")
    monkeypatch.setattr(stack, "PLUGINS", ("reader.py",))
    monkeypatch.setenv(stack.ENV, str(tmp_path))
    stack.setup_worker()
    assert not read.exists() and os.environ["STACK_REWRITES"] == "cumsum,squeeze,clamp"
    armable(local_rank="1")
    assert read.read_text() == "cumsum,squeeze" and os.environ["STACK_REWRITES"] == "cumsum,squeeze"


def test_the_seam_this_fix_waits_for_is_the_one_verl_has():
    """The deferral is only as good as the name it waits for, so check the name against
    verl itself when verl is importable, and against its source otherwise."""
    import importlib

    module_name, class_name = stack.VERL_WORKER
    try:
        worker_cls = getattr(importlib.import_module(module_name), class_name)
    except Exception:  # noqa: BLE001 - verl is not installed in the unit-test image
        root = os.environ.get("YETO_VERL_ROOT") or str(REPO.parent / "verl-fork-acad9875")
        source = Path(root) / (module_name.replace(".", "/") + ".py")
        if not source.is_file():
            pytest.skip(f"needs verl importable or its source at {source}")
        text = source.read_text()
        assert f"class {class_name}" in text
        # the two writes the deferral depends on, in Worker.__init__'s own body
        assert 'os.environ["LOCAL_RANK"] = local_rank' in text
        assert "def __init__(self, cuda_visible_devices=None)" in text
        return
    assert "LOCAL_RANK" in worker_cls.env_keys()
    assert callable(worker_cls.__init__)


_RECIPE_DIR = os.environ.get("YETO_ASCEND_RECIPE_DIR")


@pytest.mark.skipif(not _RECIPE_DIR, reason="needs YETO_ASCEND_RECIPE_DIR (the recipe's plugins dir)")
def test_the_table_covers_the_recipes_own_plugin_directory():
    """Every plugin this switch installs exists in the delivery, and every plugin of the
    recipe's load order is either installed or in ``EXCLUDED`` with a reason."""
    present = {path.name for path in Path(_RECIPE_DIR).glob("*.py")}
    assert set(stack.PLUGINS) <= present, sorted(set(stack.PLUGINS) - present)
    # EXCLUDED also names a file outside the directory (the optimizer timing injector)
    assert set(stack.EXCLUDED) - present == {"optimizer_phase_timing_inject.py"}


@pytest.fixture
def fake_torch_npu(monkeypatch):
    """A stand-in ``torch`` whose ``npu.set_device`` records the devices it is given.

    The real call is what the S28 G-PIN contrast switches off
    (``plugins/ascend_mamba2_kernels.py:352-356`` and ``:290``), so the test needs a
    seam with the same name that it can watch."""
    import sys
    import types

    torch = types.ModuleType("torch")
    torch.npu = types.SimpleNamespace()
    torch.npu.bound: list[object] = []
    torch.npu.set_device = lambda device=None: torch.npu.bound.append(device)
    real = torch.npu.set_device
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    return torch, real


def _set_device_calling_plugin(directory: Path, name: str = "device_setter.py") -> str:
    """A stand-in for the delivery's plugin: on import it calls ``torch.npu.set_device``
    with this rank's card, the way ``ascend_mamba2_kernels.apply`` does."""
    (directory / name).write_text(
        "import os, torch\n"
        "torch.npu.set_device(int(os.environ.get('LOCAL_RANK', 0)))\n")
    return name


def test_the_plugins_still_bind_their_card_when_the_switch_is_not_set(
        tmp_path, monkeypatch, armable, fake_torch_npu):
    """Default behaviour is the delivery's: the switch is off by default, so a plugin's
    ``set_device`` reaches torch. The contrast must change one thing, not two."""
    torch, _ = fake_torch_npu
    monkeypatch.delenv(stack.SET_DEVICE_ENV, raising=False)
    monkeypatch.setattr(stack, "PLUGINS", (_set_device_calling_plugin(tmp_path),))
    monkeypatch.setenv(stack.ENV, str(tmp_path))
    stack.setup_worker()
    armable(local_rank="6")
    assert torch.npu.bound == [6]


def test_the_switch_keeps_the_plugins_off_set_device_and_puts_it_back(
        tmp_path, monkeypatch, armable, fake_torch_npu):
    """With the switch at 0 no plugin reaches ``torch.npu.set_device`` during the
    import, the plugins still install, and the real function is in place afterwards."""
    torch, real = fake_torch_npu
    monkeypatch.setenv(stack.SET_DEVICE_ENV, "0")
    name = _set_device_calling_plugin(tmp_path)
    monkeypatch.setattr(stack, "PLUGINS", (name,))
    monkeypatch.setenv(stack.ENV, str(tmp_path))
    stack.setup_worker()
    armable(local_rank="6")
    assert torch.npu.bound == [], "the switch was set to 0 and a plugin still bound a card"
    assert stack.LAST_REPORT is not None and stack.LAST_REPORT["installed"] == [name]
    assert torch.npu.set_device is real, "the real set_device was not restored"
    torch.npu.set_device(3)
    assert torch.npu.bound == [3]


def test_the_switch_reports_every_call_it_swallowed(tmp_path, monkeypatch, fake_torch_npu):
    """The island log has to say whether the switch changed anything, otherwise a run
    with no plugin calling ``set_device`` is indistinguishable from a working bypass."""
    import torch

    monkeypatch.setenv(stack.SET_DEVICE_ENV, "0")
    lines: list[str] = []
    with stack._set_device_neutralised(lines.append):
        torch.npu.set_device(4)
    text = "\n".join(lines)
    assert "swallowed torch.npu.set_device(4)" in text
    assert "restored torch.npu.set_device after 1 swallowed call(s) [4]" in text


def test_only_the_documented_values_turn_the_switch_off(monkeypatch):
    """An unreadable value must keep the delivery's behaviour, not silently bypass it."""
    for value in ("0", "false", "FALSE", "no"):
        assert stack.set_device_is_allowed({stack.SET_DEVICE_ENV: value}) is False
    for value in ("1", "true", "", "yes", "nope"):
        assert stack.set_device_is_allowed({stack.SET_DEVICE_ENV: value}) is True
    assert stack.set_device_is_allowed({}) is True


def _three_plugins(directory: Path) -> list[str]:
    names = ["a_one.py", "b_two.py", "c_three.py"]
    for name in names:
        (directory / name).write_text("value = 1\n")
    return names


def test_only_imports_the_plugins_it_names(tmp_path, monkeypatch):
    """The bisect knob: ONLY narrows the import list and keeps the recipe's order."""
    names = _three_plugins(tmp_path)
    monkeypatch.setattr(stack, "PLUGINS", tuple(names))
    monkeypatch.setenv(stack.ONLY_ENV, "c_three.py, a_one.py")
    report = stack.install(str(tmp_path), environ={}, log=lambda line: None)
    assert report["installed"] == ["a_one.py", "c_three.py"]
    assert [row["plugin"] for row in report["plugins"]] == ["a_one.py", "c_three.py"]


def test_skip_leaves_out_the_plugins_it_names(tmp_path, monkeypatch):
    names = _three_plugins(tmp_path)
    monkeypatch.setattr(stack, "PLUGINS", tuple(names))
    monkeypatch.setenv(stack.SKIP_ENV, "b_two.py")
    report = stack.install(str(tmp_path), environ={}, log=lambda line: None)
    assert report["installed"] == ["a_one.py", "c_three.py"]


def test_only_wins_over_skip_and_an_unknown_name_is_reported(tmp_path, monkeypatch):
    """A typo during a bisect must not read as 'that plugin is innocent'."""
    names = _three_plugins(tmp_path)
    monkeypatch.setattr(stack, "PLUGINS", tuple(names))
    monkeypatch.setenv(stack.ONLY_ENV, "a_one.py,typo.py")
    monkeypatch.setenv(stack.SKIP_ENV, "a_one.py")
    lines: list[str] = []
    report = stack.install(str(tmp_path), environ={}, log=lines.append)
    assert report["installed"] == ["a_one.py"]
    assert f"{stack.ONLY_ENV} names typo.py, which is not in this list" in "\n".join(lines)


def test_neither_name_set_imports_the_whole_list(tmp_path, monkeypatch):
    names = _three_plugins(tmp_path)
    monkeypatch.setattr(stack, "PLUGINS", tuple(names))
    monkeypatch.delenv(stack.ONLY_ENV, raising=False)
    monkeypatch.delenv(stack.SKIP_ENV, raising=False)
    report = stack.install(str(tmp_path), environ={}, log=lambda line: None)
    assert report["installed"] == names
