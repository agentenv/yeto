from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tms_post_pause_hold_wraps_only_the_trainer_sleep(monkeypatch, capsys):
    calls = []

    class FakeMegatronTrainRayActor:
        def sleep(self):
            calls.append("sleep")
            return "paused"

    modules = {
        "miles": types.ModuleType("miles"),
        "miles.backends": types.ModuleType("miles.backends"),
        "miles.backends.megatron_utils": types.ModuleType(
            "miles.backends.megatron_utils"
        ),
        "miles.backends.megatron_utils.actor": types.ModuleType(
            "miles.backends.megatron_utils.actor"
        ),
    }
    modules["miles.backends.megatron_utils.actor"].MegatronTrainRayActor = (
        FakeMegatronTrainRayActor
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.delenv("YETO_DSV4_EXPERT_CLONE", raising=False)
    monkeypatch.setenv("TMS_INIT_ENABLE", "1")
    monkeypatch.setenv("YETO_TMS_POST_PAUSE_IDLE_S", "0")
    runpy.run_path(str(ROOT / "sitecustomize.py"), run_name="_sitecustomize_test")

    actor = FakeMegatronTrainRayActor()
    assert actor.sleep() == "paused"
    assert calls == ["sleep"]
    output = capsys.readouterr().out
    assert "[yeto-tms-post-pause] phase=start" in output
    assert "[yeto-tms-post-pause] phase=end" in output


def test_tms_post_pause_hold_is_not_installed_without_tms(monkeypatch):
    class FakeMegatronTrainRayActor:
        def sleep(self):
            return "plain"

    original = FakeMegatronTrainRayActor.sleep
    monkeypatch.delenv("YETO_DSV4_EXPERT_CLONE", raising=False)
    monkeypatch.delenv("TMS_INIT_ENABLE", raising=False)
    monkeypatch.setenv("YETO_TMS_POST_PAUSE_IDLE_S", "30")
    runpy.run_path(str(ROOT / "sitecustomize.py"), run_name="_sitecustomize_no_tms_test")

    assert FakeMegatronTrainRayActor.sleep is original
