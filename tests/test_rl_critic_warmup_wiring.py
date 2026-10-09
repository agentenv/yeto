"""rl-algo-critic-family 5.3: stage W / baseline wired into the ports learner and launcher (CPU)."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from yeto import launcher
from yeto.rl import critic_warmup as cw
from yeto.rl import learner as rl_learner
from yeto.rl.engine.algorithm import AlgorithmSpec

from test_rl_engine_selection import _learner_argv
from test_rl_launcher import _args, _prepare_rl_args
from test_rl_miles_adapter_config import make_config

WARM = AlgorithmSpec(advantage={"estimator": "ppo"}, execution={"needs_critic": True},
                     critic={"warmup_steps": 50, "critic_lr": 3e-6})
ALLOW = ("--rl-allow-unverified-mechanism", "execution:critic",
         "--rl-allow-unverified-mechanism", "advantage_estimators:ppo")
NO_SYNC = [a for a in _learner_argv() if a not in ("--syncer", "127.0.0.1:29400")] + [
    "--rl-single-island-no-sync"]


def _spec_file(tmp_path, spec=WARM):
    path = tmp_path / "spec.json"
    path.write_text(spec.canonical_json())
    return str(path)


# -- learner ---------------------------------------------------------------------------


def test_learner_baseline_needs_no_sync_ports_and_no_product(tmp_path):
    spec = _spec_file(tmp_path)
    args = rl_learner.parse_args(NO_SYNC + ["--rl-algorithm-spec", spec, *ALLOW,
                                            "--rl-critic-baseline-rounds", "3"])
    assert args.rl_critic_baseline_rounds == 3
    for bad in (["--rl-critic-baseline-rounds", "-1"],
                ["--rl-critic-baseline-rounds", "3", "--rl-critic-load", "/w",
                 "--rl-critic-init-sha256", "a" * 64],
                ["--rl-critic-load", "/w"]):
        with pytest.raises(SystemExit):
            rl_learner.parse_args(NO_SYNC + ["--rl-algorithm-spec", spec, *ALLOW, *bad])
    with pytest.raises(SystemExit):  # outer sync
        rl_learner.parse_args(_learner_argv(("--rl-critic-baseline-rounds", "3")))


def _warm_args(tmp_path, **extra):
    args = rl_learner.parse_args(NO_SYNC + ["--rl-algorithm-spec", _spec_file(tmp_path), *ALLOW,
                                            "--rl-critic-warmup-dir", str(tmp_path / "cache")])
    for key, value in extra.items():
        setattr(args, key, value)
    return args


def _ppo_run_config():
    cfg = make_config()
    return dataclasses.replace(cfg, algorithm=dataclasses.replace(
        cfg.algorithm, advantage_estimator="ppo"))


def test_learner_runs_stage_w_and_loads_its_product(tmp_path, monkeypatch):
    seen = {}

    def fake_warmup(spec, *, main_argv, actor_checkpoint, cache_root, miles_root):
        seen.update(main_argv=list(main_argv), actor=actor_checkpoint, cache=cache_root, spec=spec)
        return cw.WarmupProduct(algorithm_sha256=spec.sha256(), warmup_steps=50,
                                actor_checkpoint=actor_checkpoint, actor_sha256="b" * 64,
                                critic_checkpoint="/cache/k", critic_sha256="c" * 64)

    monkeypatch.setattr(cw, "run_ports_warmup", fake_warmup)
    args = _warm_args(tmp_path)
    config = _ppo_run_config()
    out = rl_learner._run_critic_warmup_stage(args, config)
    assert seen["spec"].sha256() == WARM.sha256()
    assert seen["actor"] == config.ref_load and seen["cache"] == str(tmp_path / "cache")
    assert "--critic-load" in seen["main_argv"]
    assert out.algorithm.critic.critic_load == "/cache/k"
    assert out.algorithm.critic.init_sha256 == "c" * 64
    assert (args.rl_critic_load, args.rl_critic_init_sha256) == ("/cache/k", "c" * 64)
    argv = rl_learner.build_ports_launch(args, out).argv
    assert cw.main_stage_critic_args(argv) == {"--num-critic-only-steps": "0",
                                              "--critic-load": "/cache/k"}


def test_learner_skips_stage_w_with_a_product_or_without_warmup(tmp_path, monkeypatch):
    monkeypatch.setattr(cw, "run_ports_warmup", lambda *a, **k: pytest.fail("stage W ran"))
    config = _ppo_run_config()
    args = _warm_args(tmp_path, rl_critic_load="/w", rl_critic_init_sha256="a" * 64)
    assert rl_learner._run_critic_warmup_stage(args, config) is config
    grpo = rl_learner.parse_args(NO_SYNC)
    plain = make_config()
    assert rl_learner._run_critic_warmup_stage(grpo, plain) is plain


def test_learner_baseline_run_is_the_same_learner_without_warmup(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    argv = NO_SYNC + ["--rl-algorithm-spec", _spec_file(tmp_path), *ALLOW,
                      "--rl-critic-baseline-rounds", "3"]
    args = rl_learner.parse_args(argv)
    calls = []
    envs = []
    monkeypatch.setenv("YETO_RL_ECHO_EVENTS", "1")
    rl_learner.run_critic_baseline(
        args, argv, run=lambda cmd, check, env: (calls.append(cmd), envs.append(env)))
    (cmd,) = calls
    assert cmd[-1] == "--rl-critic-baseline-run" and "YETO_RL_ECHO_EVENTS" not in envs[0]
    assert cmd[1:3] == ["-m", "yeto.rl.adapters.miles.island_entry"]
    spec_path = cmd[cmd.index("--rl-algorithm-spec") + 1]
    base = AlgorithmSpec.from_json_file(spec_path)
    assert base.critic.warmup_steps == 0
    assert cmd[cmd.index("--rl-expected-algorithm-sha256") + 1] == base.sha256()
    assert cmd[cmd.index("--global-rounds") + 1] == "3"
    assert cmd[cmd.index("--event-tape") + 1] == "/tmp/x.critic-baseline.jsonl"
    assert cmd[cmd.index("--completed-groups-path") + 1] == "/tmp/x.critic-baseline.pt"
    assert "--rl-critic-baseline-rounds" not in cmd
    assert rl_learner.parse_args(cmd[3:]).rl_critic_baseline_run  # a valid learner argv
    calls.clear()
    rl_learner.run_critic_baseline(rl_learner.parse_args(NO_SYNC), NO_SYNC,
                                   run=lambda cmd, check, env: calls.append(cmd))
    assert calls == []


def test_baseline_run_does_not_echo_its_tape(monkeypatch):
    echoed = []
    monkeypatch.setattr(rl_learner, "install_event_echo", lambda: echoed.append(1))
    parsed = [(rl_learner.parse_args(NO_SYNC), [1]),
              (rl_learner.parse_args(NO_SYNC + ["--rl-critic-baseline-run"]), [])]
    monkeypatch.setattr(rl_learner, "_require_ports_supported",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    for args, expect in parsed:
        echoed.clear()
        with pytest.raises(RuntimeError, match="stop"):
            rl_learner.run_miles(args, model_path="/m", prompt_path="/p", yeto_policy_sync=False)
        assert echoed == expect


def test_baseline_run_flag_needs_no_sync():
    with pytest.raises(SystemExit):
        rl_learner.parse_args(_learner_argv(("--rl-critic-baseline-run",)))


# -- launcher --------------------------------------------------------------------------


def _launch_args(tmp_path, *extra, gpu="aws:1xa100@us-east-1"):
    args = _args(("--rl-engine", "ports", "--rl-single-island-no-sync",
                  "--rl-algorithm-spec", _spec_file(tmp_path), *ALLOW, *extra))
    args.gpu = gpu
    _prepare_rl_args(args)
    return args


def test_launcher_forwards_critic_warmup_options_only_when_set(tmp_path):
    _, flags = launcher._ports_algorithm_flags(_launch_args(tmp_path))
    assert "--rl-critic" not in flags
    _, flags = launcher._ports_algorithm_flags(
        _launch_args(tmp_path, "--rl-critic-baseline-rounds", "3"))
    assert " --rl-critic-baseline-rounds 3" in flags
    _, flags = launcher._ports_algorithm_flags(_launch_args(
        tmp_path, "--rl-critic-load", "/w", "--rl-critic-init-sha256", "a" * 64))
    assert f" --rl-critic-load /w --rl-critic-init-sha256 {'a' * 64}" in flags


@pytest.mark.parametrize("extra", [
    ("--rl-critic-load", "/w"),
    ("--rl-critic-load", "/w", "--rl-critic-init-sha256", "XYZ"),
    ("--rl-critic-baseline-rounds", "3", "--rl-critic-load", "/w",
     "--rl-critic-init-sha256", "a" * 64),
])
def test_launcher_refuses_bad_critic_warmup_options(tmp_path, extra):
    with pytest.raises(ValueError):
        _launch_args(tmp_path, *extra)


def test_launcher_refuses_critic_options_without_a_warm_critic(tmp_path):
    cold = AlgorithmSpec(advantage={"estimator": "ppo"}, execution={"needs_critic": True})
    args = _args(("--rl-engine", "ports", "--rl-single-island-no-sync",
                  "--rl-algorithm-spec", _spec_file(tmp_path, cold), *ALLOW,
                  "--rl-critic-baseline-rounds", "3"))
    args.gpu = "aws:1xa100@us-east-1"
    with pytest.raises(ValueError, match="warmup_steps > 0"):
        _prepare_rl_args(args)
