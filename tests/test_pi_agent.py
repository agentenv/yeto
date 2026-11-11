"""yeto's pi agent: pi's models.json / settings.json behind the records proxy, and the trial wiring."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl.agentic import AgentTask, HarnessConfigError
from yeto.rl.harness.harbor import pi_config as pc
from yeto.rl.harness.harbor.trial import HarborSettings, agent_env


def test_setup_command_writes_both_files(tmp_path):
    env = {"HOME": str(tmp_path), "OPENAI_BASE_URL": "https://relay.example/k/s3cret/t/traj-1/v1",
           "OPENAI_API_KEY": "dummy", "PATH": "/usr/bin:/bin"}
    subprocess.run(["bash", "-c", pc.setup_command("Qwen3-4B", 4096)], env=env, check=True)
    agent_dir = tmp_path / ".pi" / "agent"
    provider = json.loads((agent_dir / "models.json").read_text())["providers"]["yeto"]
    assert provider["baseUrl"] == env["OPENAI_BASE_URL"]
    assert provider["apiKey"] == "$OPENAI_API_KEY"  # pi resolves it; the key is not written
    assert provider["api"] == "openai-completions"
    assert provider["compat"]["maxTokensField"] == "max_tokens"
    assert provider["compat"]["supportsDeveloperRole"] is False
    (model,) = provider["models"]
    assert (model["id"], model["maxTokens"], model["reasoning"]) == ("Qwen3-4B", 4096, False)
    assert model["contextWindow"] == pc.UNBOUNDED_CONTEXT_WINDOW
    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings["compaction"] == {"enabled": False}
    assert settings["defaultProjectTrust"] == "never"
    assert settings["httpIdleTimeoutMs"] == 1_800_000


def test_setup_command_honours_the_agent_dir_variable(tmp_path):
    env = {"HOME": "/nonexistent", "PI_CODING_AGENT_DIR": str(tmp_path / "pi"), "OPENAI_BASE_URL": "http://r/v1",
           "PATH": "/usr/bin:/bin"}
    subprocess.run(["bash", "-c", pc.setup_command("m", 1)], env=env, check=True)
    assert (tmp_path / "pi" / "models.json").is_file()


def test_bad_values_are_refused():
    with pytest.raises(HarnessConfigError):
        pc.models_json("m", 0)
    with pytest.raises(HarnessConfigError):
        pc.setup_command("m`id`", 16)
    agent = SimpleNamespace(name="pi", import_path=None, kwargs={})
    with pytest.raises(HarnessConfigError):
        pc.use_yeto_pi(agent, max_tokens=None)


def test_use_yeto_pi_switches_the_agent_class():
    agent = SimpleNamespace(name="pi", import_path=None, kwargs={"version": "1.5.1"})
    pc.use_yeto_pi(agent, max_tokens=2048)
    assert (agent.name, agent.import_path) == (None, pc.PI_AGENT_IMPORT_PATH)
    assert agent.kwargs == {"version": "1.5.1", "max_tokens": 2048}


def test_agent_env_runs_pi_offline():
    env = agent_env("pi", "http://relay/v1")
    assert env["OPENAI_BASE_URL"] == "http://relay/v1" and env["PI_OFFLINE"] == "1"
    assert "PI_OFFLINE" not in agent_env("codex", "http://relay/v1")


def _task_root(tmp_path: Path) -> Path:
    (tmp_path / "tasks" / "hello").mkdir(parents=True)
    return tmp_path / "tasks"


def test_build_trial_config_for_pi(tmp_path):
    pytest.importorskip("harbor.models.trial.config")
    from yeto.rl.harness.harbor.pins import PI_VERSION
    from yeto.rl.harness.harbor.trial import build_trial_config

    settings = HarborSettings(tasks_dir=_task_root(tmp_path), trials_dir=tmp_path / "trials", turn_max_tokens=1024)
    task = AgentTask(task_id="hello", agent="pi", provider="docker", model_name="Qwen3-4B")
    config = build_trial_config(task, "http://relay/v1", settings)
    assert config.agent.import_path == pc.PI_AGENT_IMPORT_PATH and config.agent.name is None
    assert config.agent.model_name == "yeto/Qwen3-4B"
    assert config.agent.kwargs == {"version": PI_VERSION, "max_tokens": 1024}
    assert config.agent.env["OPENAI_BASE_URL"] == "http://relay/v1"
    with pytest.raises(HarnessConfigError):
        build_trial_config(task, "http://relay/v1", HarborSettings(tasks_dir=settings.tasks_dir))


def test_yeto_pi_class(tmp_path):
    pytest.importorskip("harbor.agents.installed.pi")
    from yeto.rl.harness.harbor.pi_agent import YetoPi

    agent = YetoPi(logs_dir=tmp_path, model_name="yeto/Qwen3-4B", max_tokens=512)
    assert YetoPi.name() == "pi"
    command = agent.config_command()
    assert '"maxTokens": 512' in command and '"id": "Qwen3-4B"' in command
    with pytest.raises(ValueError):
        YetoPi(logs_dir=tmp_path, model_name="Qwen3-4B", max_tokens=512).config_command()


def test_install_is_skipped_when_the_image_has_the_pinned_pi(tmp_path):
    pytest.importorskip("harbor.agents.installed.pi")
    import asyncio

    from yeto.rl.harness.harbor.pi_agent import YetoPi

    class Env:
        def __init__(self, code):
            self.code, self.commands = code, []

        async def exec(self, command, **kwargs):
            self.commands.append(command)
            code = self.code if "pi --version 2>/dev/null" in command else 0  # only the check fails
            return SimpleNamespace(return_code=code, stdout="", stderr="")

    agent = YetoPi(logs_dir=tmp_path, model_name="yeto/m", max_tokens=8, version="1.5.1")
    assert '= "1.5.1" ]' in agent.preinstalled_command()
    ready = Env(0)
    asyncio.run(agent.install(ready))
    assert len(ready.commands) == 1, "only the version check runs"
    bare = Env(0)
    bare.code = 1
    asyncio.run(agent.install(bare))
    assert any("npm install -g @earendil-works/pi-coding-agent@1.5.1" in c for c in bare.commands)
