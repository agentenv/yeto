"""S26 R3: yeto's Codex model provider (idle timeout above the gate relay's hold)."""

from __future__ import annotations

import subprocess
import tomllib
from types import SimpleNamespace

import pytest

from yeto.rl.agentic import HarnessConfigError
from yeto.rl.harness.harbor import codex_provider as cp


def test_timeout_env_default_and_override():
    assert cp.stream_idle_timeout_from_env({}) == 1800
    assert cp.stream_idle_timeout_from_env({cp.STREAM_IDLE_TIMEOUT_ENV: "3600"}) == 3600
    for bad in ("x", "0", "-5"):
        with pytest.raises(HarnessConfigError):
            cp.stream_idle_timeout_from_env({cp.STREAM_IDLE_TIMEOUT_ENV: bad})


def test_setup_command_writes_a_valid_provider_after_harbors_base_url(tmp_path):
    # Harbor's own line first, then ours, then a later MCP table: one valid TOML file.
    script = "\n".join((
        'cat >>"$CODEX_HOME/config.toml" <<TOML\nopenai_base_url = "${OPENAI_BASE_URL}"\nTOML',
        cp.provider_setup_command(1800),
        "echo '[mcp_servers.x]\nurl = \"http://m\"' >> \"$CODEX_HOME/config.toml\"",
    ))
    env = {"CODEX_HOME": str(tmp_path), "OPENAI_BASE_URL": "http://relay:1/v1", "PATH": "/usr/bin:/bin"}
    subprocess.run(["bash", "-c", script], env=env, check=True)
    config = tomllib.loads((tmp_path / "config.toml").read_text())
    assert config["model_provider"] == "yeto"
    assert config["model_providers"]["yeto"] == {
        "name": "yeto", "base_url": "http://relay:1/v1", "env_key": "OPENAI_API_KEY",
        "wire_api": "responses", "stream_idle_timeout_ms": 1_800_000,
    }
    assert config["openai_base_url"] == "http://relay:1/v1"
    assert "x" in config["mcp_servers"]


def test_use_yeto_codex_switches_the_agent_class(monkeypatch):
    agent = SimpleNamespace(name="codex", import_path=None, kwargs={"version": "0.145.0"})
    cp.use_yeto_codex(agent, {cp.STREAM_IDLE_TIMEOUT_ENV: "900"})
    assert (agent.name, agent.import_path) == (None, cp.CODEX_AGENT_IMPORT_PATH)
    assert agent.kwargs == {"version": "0.145.0", "stream_idle_timeout_s": 900}


def test_yeto_codex_class_adds_the_provider(tmp_path):
    pytest.importorskip("harbor.agents.installed.codex")
    from yeto.rl.harness.harbor.codex_agent import YetoCodex

    agent = YetoCodex(logs_dir=tmp_path, model_name="openai/m", stream_idle_timeout_s=60,
                      extra_env={"OPENAI_BASE_URL": "http://relay/v1"})
    assert YetoCodex.name() == "codex"
    assert "stream_idle_timeout_ms = 60000" in agent._build_register_mcp_servers_command()
    bare = YetoCodex(logs_dir=tmp_path, model_name="openai/m")
    assert bare._build_register_mcp_servers_command() is None
