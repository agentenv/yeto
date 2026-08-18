"""The telemetry sink's contract: off by default, silent when broken."""

import argparse
import sys
import types

import pytest

from yeto import wandb_logger
from yeto.wandb_logger import NullRun, WandbRun, build_config, init


class _FakeRun:
    def __init__(self):
        self.summary = {}
        self.finished = None
        self.url = "https://wandb.ai/fake/run"

    def finish(self, exit_code=0):
        self.finished = exit_code


class _FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.logged = []
        self.init_kwargs = None
        self.defined = []
        self.run = _FakeRun()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def log(self, metrics):
        self.logged.append(metrics)

    def define_metric(self, name, step_metric=None):
        self.defined.append((name, step_metric))


@pytest.fixture
def fake_wandb(monkeypatch):
    module = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", module)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    return module


def _args(**overrides):
    base = {
        "wandb": True,
        "wandb_project": "yeto",
        "wandb_entity": None,
        "wandb_mode": "online",
        "cluster_prefix": "my-run",
        "model": "qwen35-9b",
        "inner_lr": 1e-5,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_disabled_without_the_flag(fake_wandb):
    run = init(_args(wandb=False), job_type="learner", name="learner-0")
    assert isinstance(run, NullRun)
    assert fake_wandb.init_kwargs is None


def test_only_rank_zero_logs(fake_wandb):
    run = init(_args(), job_type="learner", name="learner-0", rank=1)
    assert isinstance(run, NullRun)
    assert fake_wandb.init_kwargs is None


def test_missing_package_degrades_to_a_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # import raises ImportError
    run = init(_args(), job_type="learner", name="learner-0")
    assert isinstance(run, NullRun)


def test_islands_share_a_group_and_get_a_resumable_id(fake_wandb):
    run = init(_args(), job_type="learner", name="learner-2")
    assert isinstance(run, WandbRun)
    kwargs = fake_wandb.init_kwargs
    assert kwargs["group"] == "my-run"
    assert kwargs["job_type"] == "learner"
    assert kwargs["name"] == "learner-2"
    # Deterministic id + resume: a preempted spot island reattaches to the
    # curve it left behind instead of starting a second run.
    assert kwargs["id"] == "my-run-learner-2"
    assert kwargs["resume"] == "allow"


def test_run_group_env_overrides_the_local_prefix(fake_wandb, monkeypatch):
    monkeypatch.setenv("YETO_RUN_GROUP", "fleet-from-the-head")
    init(_args(cluster_prefix="stale"), job_type="learner", name="learner-0")
    assert fake_wandb.init_kwargs["group"] == "fleet-from-the-head"


def test_step_metrics_are_declared(fake_wandb):
    init(_args(), job_type="learner", name="learner-0")
    assert ("train/*", "local_step") in fake_wandb.defined
    assert ("sync/*", "global_step") in fake_wandb.defined
    # Both axes must be declared as metrics in their own right.
    assert ("local_step", None) in fake_wandb.defined
    assert ("global_step", None) in fake_wandb.defined


def test_no_credential_forces_offline(fake_wandb, monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.netrc
    init(_args(), job_type="learner", name="learner-0")
    assert fake_wandb.init_kwargs["mode"] == "offline"


def test_netrc_entry_keeps_the_run_online(fake_wandb, monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".netrc").write_text("machine api.wandb.ai login user password x\n")
    init(_args(), job_type="learner", name="learner-0")
    assert fake_wandb.init_kwargs["mode"] == "online"


def test_a_failing_init_never_reaches_the_caller(fake_wandb, monkeypatch, caplog):
    def boom(**kwargs):
        raise RuntimeError("private response from backend")

    monkeypatch.setattr(fake_wandb, "init", boom)
    run = init(_args(), job_type="learner", name="learner-0")
    assert isinstance(run, NullRun)
    assert "RuntimeError" in caplog.text
    assert "private response" not in caplog.text


def test_a_failing_log_disables_telemetry_instead_of_raising(
    fake_wandb, monkeypatch, caplog
):
    run = init(_args(), job_type="learner", name="learner-0")
    calls = []

    def boom(metrics):
        calls.append(metrics)
        raise RuntimeError("private tool result")

    monkeypatch.setattr(fake_wandb, "log", boom)
    run.log({"train/loss_per_token": 1.0})  # must not raise
    run.log({"train/loss_per_token": 2.0})
    # One failed attempt, then the run stops trying for the rest of training.
    assert len(calls) == 1
    assert "RuntimeError" in caplog.text
    assert "private tool result" not in caplog.text


def test_config_drops_credentials_and_private_scratch():
    config = build_config(
        _args(hf_token="secret", api_key="secret", _training_recipe={"a": 1}),
        extra={"island_backend": "torch"},
    )
    assert "hf_token" not in config
    assert "api_key" not in config
    assert "_training_recipe" not in config
    assert config["model"] == "qwen35-9b"
    assert config["island_backend"] == "torch"
    # The wandb flags themselves are noise in a config table.
    assert "wandb_project" not in config


def test_privacy_bounded_config_override_does_not_copy_the_namespace(fake_wandb):
    init(
        _args(
            prompt="private prompt",
            response="private response",
            tools=[{"name": "private"}],
            api_key="private-key",
        ),
        job_type="rl-learner",
        name="learner-0",
        config_override={
            "island_backend": "rl-miles",
            "learner_id": 0,
            "rl_sync_preset": "decoupled",
        },
    )
    assert fake_wandb.init_kwargs["config"] == {
        "island_backend": "rl-miles",
        "learner_id": 0,
        "rl_sync_preset": "decoupled",
    }


def test_private_config_override_fails_closed_without_hurting_training(fake_wandb):
    run = init(
        _args(),
        job_type="rl-learner",
        name="learner-0",
        config_override={"WANDB_API_KEY": "private"},
    )
    assert isinstance(run, NullRun)
    assert fake_wandb.init_kwargs is None


@pytest.mark.parametrize(
    "value",
    (
        {"prompt": "private"},
        ["private response"],
        float("nan"),
        "x" * 257,
    ),
)
def test_config_override_accepts_only_bounded_scalars(fake_wandb, value):
    run = init(
        _args(),
        job_type="rl-learner",
        name="learner-0",
        config_override={"island_backend": value},
    )
    assert isinstance(run, NullRun)
    assert fake_wandb.init_kwargs is None


def test_config_survives_unserializable_values():
    config = build_config(_args(device=object()))
    assert isinstance(config["device"], str)


def test_null_run_absorbs_every_call():
    run = NullRun()
    assert run.enabled is False
    run.log({"a": 1})
    run.summary({"b": 2})
    run.finish(exit_code=1)


def test_add_arguments_matches_the_learner_defaults():
    p = argparse.ArgumentParser()
    wandb_logger.add_arguments(p)
    args = p.parse_args([])
    assert args.wandb is False
    assert args.wandb_project == "yeto"
    assert args.wandb_mode == "online"
    args = p.parse_args(
        ["--wandb", "--wandb-mode", "offline", "--wandb-entity", "acme"]
    )
    assert (args.wandb, args.wandb_mode, args.wandb_entity) == (True, "offline", "acme")
