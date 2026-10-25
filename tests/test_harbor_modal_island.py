"""Harbor agent on a Modal island (S27 G-DA): rollout function, example imports,
boat token, Harbor image layer, gate relay tunnel, abort hook."""

from __future__ import annotations

import asyncio
import contextlib
import os
from types import SimpleNamespace

import pytest

from tests.test_agentic_stage3_island_argv import _island_argv, _miles_argv, _value
from yeto import modal_runner
from yeto.launch import harness_env
from yeto.launch.credentials import SECRET_ENV_NAMES
from yeto.modal_runner import ModalIslandConfig
from yeto.rl.adapters.miles import config as miles_config
from yeto.rl.adapters.miles import harbor_agent, island_entry

IMAGE = "ghcr.io/x/yeto-miles-ports@sha256:" + "a" * 64


# ---------------------------------------------------------------- Miles argv

def test_harbor_agent_island_uses_the_harbor_rollout_function(tmp_path, monkeypatch):
    args = island_entry.parse_args(_island_argv(tmp_path))
    argv = _miles_argv(args, tmp_path, monkeypatch)
    assert _value(argv, "--rollout-function-path") == harbor_agent.ROLLOUT_FUNCTION


def test_other_agents_keep_miles_rollout_function():
    assert miles_config.harbor_rollout_argv("pkg.agent.run") == ()
    assert miles_config.harbor_rollout_argv(harbor_agent.AGENT_FUNCTION) == (
        "--rollout-function-path", harbor_agent.ROLLOUT_FUNCTION)


def test_harbor_rollout_function_and_sample_buffer_are_refused_together():
    with pytest.raises(miles_config.MilesConfigError, match="sample buffer"):
        miles_config.harbor_rollout_argv(harbor_agent.AGENT_FUNCTION, sample_buffer=SimpleNamespace(role="trainer"))


# ---------------------------------------------------------------- launch side

def _args(agent: str | None = harbor_agent.AGENT_FUNCTION) -> SimpleNamespace:
    return SimpleNamespace(custom_agent_function_path=agent, training_mode="rl")


def test_example_dirs_go_on_the_island_pythonpath_only_for_the_harbor_agent():
    assert harness_env.harbor_example_pythonpath(_args()) == (
        "$HOME/miles/examples/experimental/harbor:$HOME/miles/examples/swe-agent-harbor-docker:")
    assert harness_env.harbor_example_pythonpath(_args("pkg.agent.run")) == ""
    assert harness_env.harbor_example_pythonpath(_args(None)) == ""


def test_example_dirs_exist_in_the_pinned_miles_layout():
    # the module names harbor_agent and harbor_rollout import
    assert harbor_agent.EXAMPLE_MODULE == "harbor_agent_function"
    assert harness_env.HARBOR_EXAMPLE_DIRS == ("examples/experimental/harbor", "examples/swe-agent-harbor-docker")


def test_boat_island_gets_the_launching_token_and_a_relay_port():
    envs = {"HARBOR_ENV_TYPE": "boat"}
    out = harness_env.harbor_island_envs(envs, load_boat_token=lambda: "tok-123")
    assert out == {"BOAT_API_KEY": "tok-123", harbor_agent.RELAY_PORT_ENV: "31500"}
    assert envs == {"HARBOR_ENV_TYPE": "boat"}  # input unchanged
    assert "BOAT_API_KEY" in SECRET_ENV_NAMES  # travels as a secret, never a plain env


def test_island_envs_keep_what_the_launch_set():
    def no_token():
        raise AssertionError("token must not be read")

    given = {"HARBOR_ENV_TYPE": "boat", "BOAT_API_KEY": "given", harbor_agent.RELAY_PORT_ENV: "31700"}
    assert harness_env.harbor_island_envs(given, load_boat_token=no_token) == {}
    other = {"HARBOR_ENV_TYPE": "docker", harbor_agent.RELAY_PUBLIC_BASE_ENV: "https://relay.example"}
    assert harness_env.harbor_island_envs(other, load_boat_token=no_token) == {}


def test_tunnel_port_is_one_fixed_port_unless_the_launch_gave_a_public_base():
    port = harbor_agent.RELAY_PORT_ENV
    assert harness_env.harbor_relay_tunnel_port({port: "31500"}) == 31500
    assert harness_env.harbor_relay_tunnel_port({port: "31500",
                                                 harbor_agent.RELAY_PUBLIC_BASE_ENV: "https://r"}) is None
    assert harness_env.harbor_relay_tunnel_port({}) is None
    with pytest.raises(ValueError, match="one fixed port"):
        harness_env.harbor_relay_tunnel_port({port: "31500-31510"})


# ---------------------------------------------------------------- Modal config

def _cfg(**kw) -> ModalIslandConfig:
    base = dict(app_name="yeto-x", learner_id=0, training_mode="rl", gpu="H200", gpus_per_node=8,
                num_nodes=2, run_script="true", image_ref=IMAGE, workdir=None)
    return ModalIslandConfig(**{**base, **kw})


def test_harbor_fields_survive_the_json_round_trip_and_validate():
    cfg = _cfg(harbor_layer=True, harbor_relay_port=31500)
    cfg.validate()
    back = ModalIslandConfig.from_json(cfg.to_json())
    assert back.harbor_layer is True and back.harbor_relay_port == 31500


@pytest.mark.parametrize("kw, match", [
    (dict(harbor_relay_port=31500), "Harbor agent islands"),
    (dict(harbor_layer=True, harbor_relay_port=70000), "not a TCP port"),
    (dict(harbor_layer=True, training_mode="sft", image_ref=None), "RL islands only"),
])
def test_bad_harbor_fields_are_refused(kw, match):
    with pytest.raises(ValueError, match=match):
        _cfg(**kw).validate()


class _Image:
    def __init__(self, steps=()):
        self.steps = list(steps)

    def __getattr__(self, name):
        return lambda *a, **k: _Image(self.steps + [name])


def test_build_image_adds_the_harbor_layer_on_the_registry_image(monkeypatch):
    calls = []
    fake_modal = SimpleNamespace(Image=SimpleNamespace(from_registry=lambda ref, **k: _Image(["from_registry"])))
    import yeto.rl.harness.harbor.image_layer as image_layer

    def layer(image, backend, python):
        calls.append((backend, python))
        return _Image(image.steps + ["harbor_layer"])

    monkeypatch.setattr(image_layer, "modal_harbor_layer", layer)
    runner = modal_runner.ModalOps("yeto-x")
    monkeypatch.setattr(runner, "_modal", lambda: fake_modal)
    image = runner.build_image(_cfg(harbor_layer=True))
    assert image.steps[:2] == ["from_registry", "harbor_layer"]
    assert calls == [("miles", modal_runner.HARBOR_LAYER_PYTHON)]
    plain = runner.build_image(_cfg())
    assert "harbor_layer" not in plain.steps


class _Tunnel:
    url = "https://abc.modal.host"


def _forward(opened):
    @contextlib.contextmanager
    def forward(port):
        opened.append(port)
        yield _Tunnel()
    return forward


def test_rank0_tunnels_the_relay_and_writes_a_private_secret(tmp_path, monkeypatch):
    secret = tmp_path / "secret"
    monkeypatch.setattr(modal_runner, "HARBOR_RELAY_SECRET_PATH", str(secret))
    opened, env = [], {}
    with modal_runner.harbor_relay_tunnel(_cfg(harbor_layer=True, harbor_relay_port=31500), 0, env,
                                          forward=_forward(opened)):
        assert opened == [31500]
        assert env[harbor_agent.RELAY_PORT_ENV] == "31500"
        assert env[harbor_agent.RELAY_PUBLIC_BASE_ENV] == _Tunnel.url
        assert env[harbor_agent.RELAY_SECRET_FILE_ENV] == str(secret)
        assert len(secret.read_text()) >= 24 and oct(os.stat(secret).st_mode & 0o777) == "0o600"
        # the relay settings the agent reads from this environment
        settings = harbor_agent.relay_settings(env)
        assert settings.public_base == _Tunnel.url and settings.ports == (31500,)
        assert settings.path_secret == secret.read_text()


@pytest.mark.parametrize("rank, port", [(1, 31500), (0, None)])
def test_no_tunnel_on_other_ranks_or_islands(rank, port):
    opened, env = [], {}
    cfg = _cfg(harbor_layer=port is not None, harbor_relay_port=port)
    with modal_runner.harbor_relay_tunnel(cfg, rank, env, forward=_forward(opened)):
        pass
    assert opened == [] and env == {}


# ---------------------------------------------------------------- abort hook

def test_abort_cancels_the_trials_in_flight():
    async def scenario():
        started = asyncio.Event()

        async def long_trial():
            started.set()
            await asyncio.sleep(3600)

        runner = asyncio.ensure_future(harbor_agent._run_trial(long_trial()))
        await started.wait()
        await harbor_agent.abort(None)
        with pytest.raises(asyncio.CancelledError):
            await runner
        assert harbor_agent._trials.get() == set()

    asyncio.run(scenario())


def test_abort_without_trials_is_a_no_op():
    asyncio.run(harbor_agent.abort(None))


def test_miles_finds_the_abort_hook_next_to_the_agent_function():
    module, _, _ = harbor_agent.AGENT_FUNCTION.rpartition(".")
    assert module == harbor_agent.__name__ and callable(getattr(harbor_agent, "abort"))
