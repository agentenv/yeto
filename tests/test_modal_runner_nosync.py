"""Standalone modal_runner CLI: --rl-single-island-no-sync has no syncer."""

import pytest

from yeto import modal_runner as mr

IMAGE = "docker:ghcr.io/x/miles@sha256:" + "a" * 64


@pytest.fixture
def cli(monkeypatch, tmp_path):
    script = tmp_path / "run.sh"
    script.write_text("echo hi")
    calls = {"resolve": [], "defined": []}
    monkeypatch.setattr(mr, "modal_available", lambda: True)
    monkeypatch.setattr(mr, "resolve_syncer_for_modal", lambda addr, o: calls["resolve"].append(addr) or addr)

    class Ops:
        def __init__(self, name):
            pass

        def define(self, cfg):
            calls["defined"].append(cfg)

        def deploy(self):
            pass

        def spawn(self, cfg):
            return "fc-1"

        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(mr, "ModalOps", Ops)
    base = ["--learner-id", "0", "--num-learners", "1", "--run-script", str(script)]
    return calls, base


def test_no_sync_skips_the_syncer_resolution(cli):
    calls, base = cli
    mr.main([*base, "--training-mode", "rl", "--rl-image", IMAGE, "--rl-single-island-no-sync"])
    assert calls["resolve"] == []
    (cfg,) = calls["defined"]
    assert "SYNCER_ADDR" not in cfg.envs and cfg.training_mode == "rl"


def test_sync_still_resolves_the_syncer(cli):
    calls, base = cli
    mr.main([*base, "--training-mode", "rl", "--rl-image", IMAGE, "--syncer-addr", "8.8.8.8:5000"])
    assert calls["resolve"] == ["8.8.8.8:5000"]
    assert calls["defined"][0].envs["SYNCER_ADDR"] == "8.8.8.8:5000"


@pytest.mark.parametrize(
    "extra",
    [
        [],  # neither flag: --syncer-addr required
        ["--rl-single-island-no-sync"],  # sft
        ["--training-mode", "rl", "--rl-image", IMAGE, "--rl-single-island-no-sync", "--syncer-addr", "1.2.3.4:5"],
    ],
)
def test_bad_combinations_exit_2_before_any_modal_work(cli, extra):
    calls, base = cli
    assert mr.main([*base, *extra]) == 2
    assert calls == {"resolve": [], "defined": []}


def test_no_sync_needs_one_learner(cli):
    calls, base = cli
    argv = [a if a != "1" else "2" for a in base]
    assert mr.main([*argv, "--training-mode", "rl", "--rl-image", IMAGE, "--rl-single-island-no-sync"]) == 2
    assert calls["defined"] == []
