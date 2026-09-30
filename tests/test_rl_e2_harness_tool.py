"""tools/probes/e2_cut_harness.py: dry-run artifacts only (CPU)."""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e2_cut_harness", REPO / "tools/probes/e2_cut_harness.py")
tool = importlib.util.module_from_spec(spec)
sys.modules["e2_cut_harness"] = tool  # dataclasses resolve the module
spec.loader.exec_module(tool)


def test_pin_check_is_against_plan_v4(tmp_path):
    init = tmp_path / "yeto" / "rl" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text(f'MILES_NEXT_COMMIT = "{tool.MILES_COMMIT}"\nX = "{tool.IMAGE_DIGEST.split(":")[1]}"\n')
    assert tool.PLAN_VERSION == "plan-v4" and tool.check_pins(tmp_path) == []
    init.write_text('MILES_NEXT_COMMIT = "5c1b49ebccbc7508c1d9ef89eacc2db3e448b6ba"\n')
    assert len(tool.check_pins(tmp_path)) == 2


def test_execute_is_refused(tmp_path, capsys):
    assert tool.main(["--root", str(tmp_path), "--yeto-sha", "abc", "--execute"]) == 2
    assert not any(tmp_path.iterdir())


def test_dry_run_writes_every_run_in_plan_order(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "check_pins", lambda repo: [])  # pins: see test_pin_check_is_against_plan_v4
    assert tool.main(["--root", str(tmp_path), "--yeto-sha", "abc1234", "--prefix", "p"]) == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    names = [r["run"] for r in plan["runs"]]
    assert names[:4] == ["c1", "c2", "c3-b1", "c3-rb"]
    assert plan["miles_commit"] == tool.MILES_COMMIT and tool.IMAGE_DIGEST in plan["image"]
    for run in plan["runs"]:
        d = Path(run["dir"])
        args = shlex.split((d / "args.txt").read_text())
        assert args[args.index("--model-revision") + 1] == tool.MODELS[run["model"]]
        assert "--modal-gpu-exact" in args and "--rl-deterministic-trainer" in args
        if run["config"] in ("C1", "C2", "C3"):
            assert args[args.index("--rl-lora-dropout") + 1] == "0.05"
        script = (d / "run.sh").read_text()
        assert tool.GUARD in script and "app stop -y" in script  # guard + watchdog
        if run["config"] == "C3":
            assert json.loads((d / "resources.json").read_text())["configs"]["T2R1S0"]["trainer"] == 2
    c1 = json.loads((tmp_path / "p-c1" / "harness.json").read_text())
    assert c1["config"] == "C1" and c1["expected_dp"] == 1 and c1["lora_dropout"] == 0.05
    assert "G-4.2(f)" in json.loads((tmp_path / "p-c2" / "spec.json").read_text())["criteria"]
    blocked = {r["run"]: r["blocked"] for r in plan["runs"] if r["blocked"]}
    assert blocked == {}


def test_harness_plan_is_valid_for_the_in_learner_harness():
    from yeto.rl.engine.miles_adapter.e2_harness import plan_problems

    for run in tool.plan_runs():
        if run.harness:
            assert plan_problems(run.harness) == []


def test_run_scripts_parse_and_carry_the_strict_guards(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(tool, "check_pins", lambda repo: [])
    assert tool.main(["--root", str(tmp_path), "--yeto-sha", "abc1234", "--prefix", "p"]) == 0
    for d in sorted(tmp_path.glob("p-*")):
        script = (d / "run.sh").read_text()
        assert subprocess.run(["bash", "-n", str(d / "run.sh")]).returncode == 0, d
        assert tool.GPU_NAME in script and tool.MILES_COMMIT in script and "2f23a0f-9f29303" in script
        assert "guard.fail" in script and "app stop -y" in script
        args = (d / "args.txt").read_text()
        assert "--rl-elastic-cells" not in args
        if json.loads((d / "spec.json").read_text())["rebuild_trigger"]:
            assert "e2_inwatch.py train 2 rb1 0 900" in script
