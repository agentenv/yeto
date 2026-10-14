"""rl-trainer-cold-start-cost L0 (start phases), L2.3/L2.4 (image manifest
instead of imports; peft in the image) and L2.5 (model prefetch).  CPU only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto import rl
from yeto.launch import images, manifest as launch_manifest
from yeto.launch.remote_setup import model_prefetch
from yeto.rl.engine import start_phases as sp

REPO = Path(__file__).resolve().parents[1]


class _Collect:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def __call__(self, event, **fields):
        self.events.append((event, fields))

    def named(self, name):
        return [f for e, f in self.events if e == name]


# ------------------------------------------------------------ L0.1 phases


def _fake_start(tmp_path, *, with_shell=True):
    """A fake launcher + setup + run: real shell marks, then the island replay."""

    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HOME": str(home)}
    lines = [sp.shell_reset(), sp.shell_mark("container_ready"), "sleep 0.05",
             sp.shell_mark("setup_done"), "sleep 0.05", sp.shell_mark("ray_ready")]
    submitted = time.time()
    if with_shell:
        subprocess.run(["bash", "-c", "\n".join(lines)], env=env, check=True)
    emit = _Collect()
    phases = sp.StartPhases(emit)
    phases.replay({sp.SUBMITTED_ENV: f"{submitted:.3f}"}, home / ".yeto-start-phases")
    return phases, emit


def test_fake_launcher_start_has_seven_phases_in_order_and_the_total(tmp_path):
    phases, emit = _fake_start(tmp_path)
    phases.mark("actor_import_done")
    phases.mark("actor_init_done")
    phases.mark_once("first_step_done")
    phases.mark_once("first_step_done")  # later steps record nothing
    records = emit.named(sp.START_PHASE_EVENT)
    assert [r["phase"] for r in records] == list(sp.START_PHASES)
    assert [r["index"] for r in records] == list(range(7))
    times = [r["t_unix"] for r in records]
    assert times == sorted(times)
    [summary] = emit.named(sp.START_SUMMARY_EVENT)
    assert summary["missing"] == []
    assert summary["total_s"] == pytest.approx(times[-1] - times[0], abs=1e-3)
    assert summary["phases"] == dict(zip(sp.START_PHASES, times))


def test_phase_order_is_enforced():
    phases = sp.StartPhases(_Collect(), clock=lambda: 1.0)
    phases.mark("submitted")
    phases.mark("setup_done")  # a skipped phase is allowed ...
    with pytest.raises(ValueError, match="out of order"):
        phases.mark("container_ready")  # ... going back is not
    with pytest.raises(ValueError, match="twice"):
        phases.mark("setup_done")
    with pytest.raises(ValueError, match="unknown"):
        phases.mark("warm")


def test_missing_marks_are_named_in_the_summary(tmp_path):
    phases, emit = _fake_start(tmp_path, with_shell=False)
    phases.missing("actor_import_done", "no Ray record")
    phases.mark("actor_init_done")
    phases.mark("first_step_done")
    [summary] = emit.named(sp.START_SUMMARY_EVENT)
    assert summary["missing"] == ["container_ready", "setup_done", "ray_ready", "actor_import_done"]
    assert summary["total_s"] is not None


def test_trainer_actor_import_time_from_ray_state():
    from yeto.rl.adapters.miles.entry import trainer_actor_import_done_unix

    tasks = [SimpleNamespace(name="MegatronTrainRayActor.__init__", end_time_ms=2000),
             SimpleNamespace(name="MegatronTrainRayActor.__init__", end_time_ms=3500),
             SimpleNamespace(name="RolloutManager.__init__", end_time_ms=9000)]
    assert trainer_actor_import_done_unix(lambda **kw: tasks) == 3.5
    assert trainer_actor_import_done_unix(lambda **kw: []) is None

    def broken(**kw):
        raise RuntimeError("no dashboard")

    assert trainer_actor_import_done_unix(broken) is None


def test_island_task_scripts_carry_the_marks(monkeypatch):
    from tests.test_rl_engine_selection import _cli, _island_task
    from yeto.launcher import _prepare_rl_args

    args = _cli(["--gpu", "nebius:2x1xl40s@eu-north1"])
    _prepare_rl_args(args)
    task = _island_task(args, monkeypatch)
    setup = task.setup
    assert setup.index(sp.shell_reset()) < setup.index(sp.shell_mark("container_ready"))
    assert setup.index(sp.shell_mark("container_ready")) < setup.index("SGLANG_IN_IMAGE=0")
    assert setup.rstrip().endswith(sp.shell_mark("setup_done"))
    run = task.run
    assert run.index("ray start --head") < run.index(sp.shell_mark("ray_ready")) < run.index("python3 -m ")
    assert abs(float(task.envs[sp.SUBMITTED_ENV]) - time.time()) < 600


# ------------------------------------------------------- L0.2 run manifest


def test_summary_goes_into_the_run_manifest_and_t_s_reads_back(tmp_path, monkeypatch):
    from yeto import runs

    run_dir = tmp_path / "run"
    (run_dir / "events").mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({"rl_engine": "ports"}))
    monkeypatch.setattr(runs, "run_dir", lambda prefix: run_dir)
    for island, total in ((0, 81.5), (1, 120.25)):
        tape = run_dir / "events" / f"island-{island}.jsonl"
        lines = [{"event": "rl_driver_start", "island_id": island},
                 {"event": sp.START_SUMMARY_EVENT, "island_id": island, "total_s": total,
                  "missing": [], "phases": {"submitted": 1.0}}]
        tape.write_text("".join(json.dumps(l) + "\n" for l in lines) + "{torn")
    summaries = launch_manifest._record_trainer_start(
        SimpleNamespace(cluster_prefix="r"), run_dir / "events")
    assert set(summaries) == {"0", "1"}
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["rl_engine"] == "ports"
    assert manifest[sp.MANIFEST_KEY]["1"]["total_s"] == 120.25
    assert sp.t_s_from_manifest(manifest) == 120.25
    assert sp.t_s_from_manifest({"rl_engine": "ports"}) is None


def test_no_summary_leaves_the_manifest_alone(tmp_path, monkeypatch):
    from yeto import runs

    (tmp_path / "events").mkdir()
    monkeypatch.setattr(runs, "run_dir", lambda prefix: tmp_path)
    assert launch_manifest._record_trainer_start(SimpleNamespace(cluster_prefix="r"),
                                                 tmp_path / "events") is None
    assert not (tmp_path / "run_manifest.json").exists()



def test_sky_islands_record_t_s_from_teardown_diagnostics(tmp_path, monkeypatch):
    """Non-echo sky islands: T_s comes from the island tapes the pre-teardown
    diagnostics fetched into <diag root>/<cluster>/ (other tapes ignored)."""
    from yeto import runs

    monkeypatch.setattr(runs, "run_dir", lambda prefix: tmp_path)
    diag = tmp_path / "diag"
    for cluster, island, total in (("r-0", 0, 90.0), ("r-1", 1, 140.5)):
        (diag / cluster).mkdir(parents=True)
        (diag / cluster / f"rl-island-{island}.jsonl").write_text(json.dumps(
            {"event": sp.START_SUMMARY_EVENT, "island_id": island, "total_s": total,
             "missing": [], "phases": {}}) + "\n")
    (diag / "r-1" / "yeto-tape.jsonl").write_text(json.dumps(
        {"event": sp.START_SUMMARY_EVENT, "island_id": 7, "total_s": 1.0}) + "\n")
    args = SimpleNamespace(cluster_prefix="r")
    assert set(launch_manifest._record_sky_trainer_start(args, ["r-0", "r-1"], str(diag))) == {"0", "1"}
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert sp.t_s_from_manifest(manifest) == 140.5
    assert launch_manifest._record_sky_trainer_start(args, ["r-9"], str(diag)) is None


# --------------------------------------------- L2.3 manifest, not imports


def _run_check(tmp_path, manifest, package="peft", key="version", want=rl.MILES_PEFT_VERSION,
               path=""):
    """Run image_manifest_check against ``manifest`` (dict, or None = no file)
    with a python3 shim that refuses to import peft/sglang."""

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    manifest_path = tmp_path / "image-manifest.json"
    if manifest is not None:
        manifest_path.write_text(json.dumps(manifest))
    poison = tmp_path / "poison"
    for name in ("peft", "sglang"):
        (poison / name).mkdir(parents=True, exist_ok=True)
        (poison / name / "__init__.py").write_text(
            f"import pathlib; pathlib.Path({str(tmp_path / 'imported')!r}).write_text({name!r})\n"
            "__version__ = 'poison'\n__file__ = '/nowhere'\n")
    script = images.image_manifest_check(
        package, key, want, path=path, on_pass="echo PASS",
        fallback="echo FALLBACK",
    ).replace(rl.MILES_NEXT_IMAGE_MANIFEST, str(manifest_path))
    env = {**os.environ, "HOME": str(home), "PYTHONPATH": str(poison)}
    proc = subprocess.run(["bash", "-c", "set -e\n" + script + "\necho AFTER"], env=env,
                          text=True, capture_output=True)
    events = home / ".yeto-start-phases" / sp.SETUP_EVENTS_FILE
    return proc, (sp.read_tape_records([events]) if events.exists() else []), \
        (tmp_path / "imported").exists()


def test_manifest_present_passes_without_importing(tmp_path):
    proc, events, imported = _run_check(
        tmp_path, {"peft": {"version": rl.MILES_PEFT_VERSION}})
    assert proc.returncode == 0, proc.stderr
    assert "PASS" in proc.stdout and "FALLBACK" not in proc.stdout
    assert events == [] and not imported


def test_sglang_manifest_checks_commit_and_path(tmp_path):
    want = {"sglang": {"commit": rl.SGLANG_NEXT_COMMIT, "path": "/sgl-workspace/sglang"}}
    proc, _, imported = _run_check(tmp_path, want, "sglang", "commit", rl.SGLANG_NEXT_COMMIT,
                                   "/sgl-workspace/sglang")
    assert proc.returncode == 0 and "PASS" in proc.stdout and not imported
    want["sglang"]["path"] = "/elsewhere"
    proc, _, _ = _run_check(tmp_path, want, "sglang", "commit", rl.SGLANG_NEXT_COMMIT,
                            "/sgl-workspace/sglang")
    assert proc.returncode != 0 and "AFTER" not in proc.stdout
    assert "sglang path /elsewhere != /sgl-workspace/sglang" in proc.stderr


@pytest.mark.parametrize("manifest", [None, {"miles": {"commit": "x"}}], ids=["no-file", "no-entry"])
def test_manifest_missing_falls_back_with_an_event(tmp_path, manifest):
    proc, events, _ = _run_check(tmp_path, manifest)
    assert proc.returncode == 0, proc.stderr
    assert "FALLBACK" in proc.stdout and "PASS" not in proc.stdout
    assert [e["event"] for e in events] == ["image_manifest_fallback"]
    assert events[0]["package"] == "peft" and events[0]["check"] == "import"
    assert '"event": "image_manifest_fallback"' in proc.stderr


def test_manifest_mismatch_fails_naming_package_and_both_versions(tmp_path):
    proc, _, imported = _run_check(tmp_path, {"peft": {"version": "0.13.2"}})
    assert proc.returncode == 1
    assert "AFTER" not in proc.stdout and "PASS" not in proc.stdout
    assert f"peft version 0.13.2 != pinned {rl.MILES_PEFT_VERSION}" in proc.stderr
    assert not imported


def test_setup_events_reach_the_tape_once(tmp_path):
    proc, _, _ = _run_check(tmp_path, None)
    emit = _Collect()
    phases = sp.StartPhases(emit)
    mark_dir = tmp_path / "home" / ".yeto-start-phases"
    phases.replay({}, mark_dir)
    assert phases.forward_setup_events(mark_dir) == 0
    [event] = emit.named("image_manifest_fallback")
    assert event["source"] == "setup" and event["package"] == "peft"


# ------------------------------------------------- L2.4 peft in the image


def test_build_script_puts_peft_into_the_image_and_the_manifest():
    script = (REPO / "scripts" / "build_miles_ports_image.sh").read_text()
    assert '"MILES_PEFT_VERSION"' in script
    assert 'download -q --no-deps --only-binary=:all:' in script
    assert '"peft": {"version": "$MILES_PEFT_VERSION"' in script
    assert '--wheel "$PEFT_WHEEL"' in script
    dockerfile = (REPO / "docker" / "miles-ports" / "Dockerfile").read_text()
    assert f"ARG PEFT_VERSION={rl.MILES_PEFT_VERSION}" in dockerfile


def _wheel(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def test_layer_unpacks_a_pure_python_wheel(tmp_path):
    sys.path.insert(0, str(REPO / "docker" / "miles-ports"))
    try:
        import make_layer
    finally:
        sys.path.pop(0)
    wheel = _wheel(tmp_path / "peft-0.20.0-py3-none-any.whl", {
        "peft/__init__.py": b"__version__ = '0.20.0'\n",
        "peft-0.20.0.dist-info/METADATA": b"Name: peft\n",
        "peft-0.20.0.dist-info/RECORD": b"",
    })
    files = make_layer.wheel_files(wheel)
    assert files["peft/__init__.py"].startswith(b"__version__")
    assert files["peft-0.20.0.dist-info/INSTALLER"] == b"yeto-make-layer\n"
    with pytest.raises(SystemExit, match="pure-Python"):
        make_layer.wheel_files(_wheel(tmp_path / "x-1-cp312-cp312-linux_x86_64.whl", {}))
    with pytest.raises(SystemExit, match=".data"):
        make_layer.wheel_files(_wheel(tmp_path / "y-1-py3-none-any.whl",
                                      {"y-1.data/scripts/y": b"", "y-1.dist-info/RECORD": b""}))


# --------------------------------------------------------- L2.5 prefetch


def _run_prefetch(tmp_path, exit_code: int):
    home = tmp_path / "home"
    home.mkdir()
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "python3").write_text(f"#!/bin/sh\necho \"$@\" > {tmp_path / 'argv'}\nexit {exit_code}\n")
    (shim / "python3").chmod(0o755)
    command = model_prefetch("org/model", "a" * 40)
    env = {**os.environ, "HOME": str(home), "PATH": f"{shim}:{os.environ['PATH']}"}
    proc = subprocess.run(["bash", "-c", f"set -e\n{command}\necho CONTINUES"], env=env,
                          text=True, capture_output=True, timeout=30)
    events_file = home / ".yeto-start-phases" / sp.SETUP_EVENTS_FILE
    deadline = time.time() + 10
    while time.time() < deadline and not (tmp_path / "argv").exists():
        time.sleep(0.05)
    time.sleep(0.3)  # the background shell writes the event after python3 exits
    return proc, sp.read_tape_records([events_file]), (tmp_path / "argv").read_text()


def test_prefetch_uses_snapshot_download_not_the_removed_cli():
    command = model_prefetch("org/model", "a" * 40)
    assert "huggingface-cli" not in command and "|| true" not in command.split("sh -c")[0]
    assert "snapshot_download(repo_id=" in command and "org/model" in command and "a" * 40 in command


def test_prefetch_failure_is_recorded_and_the_setup_continues(tmp_path):
    proc, events, argv = _run_prefetch(tmp_path, 5)
    assert proc.returncode == 0 and "CONTINUES" in proc.stdout
    assert "snapshot_download" in argv
    assert events == [{"event": "model_prefetch_failed", "exit_code": 5,
                       "log": "/tmp/hf-prefetch.log", "model": "org/model"}]


def test_prefetch_success_records_nothing(tmp_path):
    proc, events, _ = _run_prefetch(tmp_path, 0)
    assert proc.returncode == 0 and events == []
