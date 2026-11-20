"""AgentENV microVM sandboxes through ``yeto run up``: ``sandbox.provider: agentenv`` skips the
docker-host steps, and ``agentic.setup`` with provider agentenv hands the sampler its gateway,
caps and key file without ssh / docker / tunnel steps (and never shows the key)."""

from __future__ import annotations

import json

import pytest

from tests.test_agentic_setup import FakeNode, _node
from yeto.launch import agentic_setup as su
from yeto.rl import up
from yeto.rl.harness.harbor import agentenv_client

FAKE_KEY = "fake-agentenv-key-" + "0" * 24
PUBLIC = "https://records.example.org"
KEY_FILE = "/".join(["/mnt/nvme/work/harbor", "home", "agentenv.key"])


def _spec(tmp_path, *, public=PUBLIC, **setup):
    key = tmp_path / "agentenv.env"
    key.write_text(f"# auth\nexport AGENTENV_API_KEY='{FAKE_KEY}'\n")
    key.chmod(0o600)
    tasks = tmp_path / "tasks"
    for t in ("a", "b"):
        (tasks / t).mkdir(parents=True, exist_ok=True)
        (tasks / t / "task.toml").write_text("[task]\n")
    doc = {"provider": "agentenv",
           "agentenv": {"gateway_url": "https://gw.example.org/", "sandbox_url": "https://data.example.org",
                        "api_key": {"path": str(key)}, "machine_type": "m-2c", "max_sandboxes": 16, "pool": True},
           "records_proxy": {"public_base": public},
           "tasks": {"source": str(tasks), "ids": ["a", "b"]}, "records_secret": "generate", **setup}
    return {"sampler_id": "s-0", "agentic": {"proxy": {"host": "127.0.0.1", "port": 18080},
                                             "task_defaults": {"provider": "agentenv"}, "setup": doc}}


def test_agentenv_setup_parses_into_the_sampler_env(tmp_path):
    s = su.parse(_spec(tmp_path))
    assert s.provider == "agentenv" and s.sandbox is None and s.tunnel is None
    env = s.container_env()
    assert env["HARBOR_ENV_TYPE"] == "agentenv" and "DOCKER_HOST" not in env
    assert env["E2B_API_URL"] == "https://gw.example.org"
    assert env["AGENTENV_SANDBOX_URL"] == env["E2B_SANDBOX_URL"] == "https://data.example.org"
    assert json.loads(env["HARBOR_ENV_KWARGS"]) == {"machine_type": "m-2c"}
    assert json.loads(env["HARBOR_MAX_SANDBOXES"]) == {"agentenv": 16}
    assert env["YETO_AGENTENV_POOL"] == "1" and env["HARBOR_TASKS_DIR"] == "/work/harbor/tasks"
    assert env["YETO_RECORDS_PROXY_PUBLIC_BASE"] == PUBLIC
    assert env["AGENTENV_API_KEY_FILE"] == s.agentenv_key_file
    assert su.local_problems(s) == []
    lines, problems = su.dry_run_report(_spec(tmp_path))
    assert problems == [] and not any("docker_cli" in ln or "tunnel:" in ln for ln in lines)
    assert FAKE_KEY not in "\n".join(lines) + json.dumps(env)


def test_a_placeholder_records_base_warns_in_a_dry_run_and_stops_the_setup(tmp_path):
    spec = _spec(tmp_path, public="__TODO_NETWORK_DESIGN_RECORDS_PROXY_BASE__")
    lines, problems = su.dry_run_report(spec)
    assert problems == [] and any("WARNING" in ln and "records_proxy.public_base" in ln for ln in lines)
    p = su.Provisioner(su.parse(spec), _node(FakeNode()), sleep=lambda _s: None)
    with pytest.raises(su.SetupError, match="records_proxy"):
        p.run()


@pytest.mark.parametrize("change, match", [
    ({"sandbox": {"docker_host": "ssh://u@h", "key": {"path": "k"}}}, "no docker host"),
    ({"tunnel": {"listen": "192.0.2.1:1"}}, "no docker host"),
    ({"agentenv": {"api_key": {"env": "K"}}}, "gateway_url"),
    ({"agentenv": {"gateway_url": "__TODO_AGENTENV_GATEWAY_URL__", "api_key": {"env": "K"}}}, "gateway_url"),
    ({"agentenv": {"gateway_url": "https://gw.example.org"}}, "api_key is required"),
    ({"agentenv": {"gateway_url": "https://gw.example.org", "api_key": {"env": "K"}, "max_sandboxes": 0}},
     "max_sandboxes"),
    ({"agentenv": {"gateway_url": "https://gw.example.org", "api_key": {"env": "K"}, "bogus": 1}}, "unknown keys"),
    ({"provider": "remote-docker"}, "is not task_defaults.provider"),
    ({"provider": "daytona"}, "must be one of"),
])
def test_agentenv_setup_refusals(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        su.parse(_spec(tmp_path, **change))


def test_a_key_file_without_a_key_is_a_local_problem_without_the_value(tmp_path):
    s = su.parse(_spec(tmp_path))
    open(s.agentenv.api_key.path, "w").write("OTHER_VAR=not-it\n")
    problems = su.local_problems(s)
    assert len(problems) == 1 and "agentenv.api_key" in problems[0] and "not-it" not in problems[0]


def test_provision_skips_docker_steps_and_uploads_the_key_file(tmp_path):
    fake = FakeNode()
    p = su.Provisioner(su.parse(_spec(tmp_path)), _node(fake), sleep=lambda _s: None, local_run=fake.run)
    p.node.stream_tar = lambda *a, **k: None
    report = p.run()
    assert report == {"layout": "ok", "harbor": "changed", "records_secret": "changed", "tasks": "changed",
                      "agentenv_key": "changed"}
    assert fake.files[KEY_FILE][1] == "600" and FAKE_KEY in fake.files[KEY_FILE][0]
    assert not any("docker -H ssh://" in c or "docker-compose" in c for c in fake.calls)
    assert any("pip install" in c and su.E2B_REQUIREMENT in c for c in fake.calls)
    assert not any(FAKE_KEY in c for c in fake.calls)  # the key travels on stdin only


def test_load_token_reads_the_key_file(tmp_path, monkeypatch):
    for name in ("AGENTENV_API_KEY", "E2B_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "key"
    path.write_text(f"AGENTENV_API_KEY=\"{FAKE_KEY}\"\n")
    monkeypatch.setenv("AGENTENV_API_KEY_FILE", str(path))
    assert agentenv_client.load_token() == FAKE_KEY
    path.write_text(FAKE_KEY + "\n")
    assert agentenv_client.load_token() == FAKE_KEY
    path.write_text("NOPE=1\n")
    with pytest.raises(agentenv_client.AgentenvError) as info:
        agentenv_client.load_token()
    assert FAKE_KEY not in str(info.value)


def _run_cfg(tmp_path, sandbox):
    cfg = {"run": "r", "store": "s3://b/p", "trainer": {"argv": []}, "pool": {"config": "p.json"},
           "sandbox": sandbox}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def test_run_up_sandbox_provider_agentenv_skips_the_docker_host(tmp_path):
    cfg = up.load_config(_run_cfg(tmp_path, {"provider": "agentenv", "envs": "envs.json"}))
    run = up.Run(cfg, state_root=tmp_path / "state")
    assert run.envs is None and run.sandbox_hosts() == [] and run.sandbox_prepare() is True
    plain = up.load_config(_run_cfg(tmp_path, {"envs": "envs.json"}))
    assert up.Run(plain, state_root=tmp_path / "state2").envs == "envs.json"


@pytest.mark.parametrize("sandbox, match", [
    ({"provider": "boat"}, "sandbox.provider"),
    ({"provider": "agentenv", "docker_host": "ssh://h"}, "no docker host"),
])
def test_run_up_sandbox_provider_refusals(tmp_path, sandbox, match):
    with pytest.raises(ValueError, match=match):
        up.load_config(_run_cfg(tmp_path, sandbox))
