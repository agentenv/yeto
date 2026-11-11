"""remote-docker: Harbor's docker environment on another host's daemon, without host bind mounts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from yeto.rl.agentic import AgentTask, HarnessConfigError

pytest.importorskip("harbor.environments.docker.docker")

from harbor.environments.docker.docker import DockerEnvironment  # noqa: E402
from harbor.models.task.config import EnvironmentConfig  # noqa: E402
from harbor.models.trial.config import ServiceVolumeConfig  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402

from yeto.rl.harness.harbor.remote_docker_env import RemoteDockerEnvironment  # noqa: E402


def _env(tmp_path: Path, cls: type = RemoteDockerEnvironment) -> DockerEnvironment:
    env_dir = tmp_path / "environment"
    env_dir.mkdir(exist_ok=True)
    (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    paths = TrialPaths(trial_dir=tmp_path / "trial")
    paths.mkdir()
    mounts = [ServiceVolumeConfig(type="bind", source=str(paths.agent_dir), target="/logs/agent")]
    with patch.object(DockerEnvironment, "_detect_windows_containers", return_value=False):
        return cls(environment_dir=env_dir, environment_name="t", session_id="t__1", trial_paths=paths,
                   task_env_config=EnvironmentConfig(docker_image="ubuntu:24.04"), mounts=mounts)


def test_not_mounted_and_no_bind_volume(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "ssh://root@sandbox")
    env = _env(tmp_path)
    assert env.capabilities.mounted is False
    assert env.capabilities.docker_compose is True  # everything else is Harbor's docker environment
    compose = json.loads(env._write_mounts_compose_file().read_text())
    assert compose == {"services": {"main": {"volumes": []}}}
    assert env._mount_targets(writable_only=True) == ["/logs/agent"], "start() still creates the target"


def test_local_docker_keeps_its_bind_mounts(tmp_path):
    env = _env(tmp_path, DockerEnvironment)
    assert env.capabilities.mounted is True
    assert json.loads(env._write_mounts_compose_file().read_text())["services"]["main"]["volumes"]


def test_needs_a_docker_host(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    with pytest.raises(HarnessConfigError, match="DOCKER_HOST"):
        _env(tmp_path)


def test_trial_config_loads_it_by_import_path(tmp_path):
    from yeto.rl.harness.harbor.trial import HarborSettings, build_trial_config

    (tmp_path / "tasks" / "hello").mkdir(parents=True)
    task = AgentTask(task_id="hello", agent="nop", provider="remote-docker")
    config = build_trial_config(task, "http://relay/v1", HarborSettings(tasks_dir=tmp_path / "tasks"))
    assert config.environment.import_path == "yeto.rl.harness.harbor.remote_docker_env:RemoteDockerEnvironment"
