"""Customer sandbox providers by ``module:Class`` (``HARBOR_PROVIDERS``), end to
end with the records proxy: the environment's agent gets a base URL that holds
its trajectory's own token.  CPU only (Harbor's Trial is replaced by a fake that
drives the environment the way Harbor does: start, agent exec, verifier, stop)."""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest

from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.agentic import AgentTask, HarnessConfigError
from yeto.rl.harness.gateway.proxy import RecordsProxy
from yeto.rl.harness.gateway.recorder import ProxyTokenRecorder
from yeto.rl.harness.harbor.trial import (CUSTOM_PROVIDERS, HarborHarness, agent_env, provider_import_path,
                                          registered_providers)

SIGNED = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 16}
SECRET = "q" * 32

# ------------------------------------------------------------- provider import paths
def test_registered_providers_and_import_paths():
    env = {"HARBOR_PROVIDERS": json.dumps({"acme": "acme_sbx.env:AcmeEnvironment"})}
    assert provider_import_path("acme", env) == "acme_sbx.env:AcmeEnvironment"
    assert provider_import_path("acme_sbx.env:AcmeEnvironment", env) == "acme_sbx.env:AcmeEnvironment"
    assert provider_import_path("remote-docker", env) == CUSTOM_PROVIDERS["remote-docker"]
    assert provider_import_path("docker", env) is None  # Harbor's own
    with pytest.raises(HarnessConfigError, match="not registered"):
        provider_import_path("os:system", env)  # a task row cannot pick a module
    for bad in ('{"x": "no-colon"}', '{"x": "a b:C"}', '["a:B"]', "{", '{"boat": "evil:Env"}'):
        with pytest.raises(HarnessConfigError):
            registered_providers({"HARBOR_PROVIDERS": bad})


def test_trial_config_uses_a_registered_import_path(tmp_path, monkeypatch):
    pytest.importorskip("harbor.models.trial.config")
    from yeto.rl.harness.harbor.trial import HarborSettings, build_trial_config

    monkeypatch.setenv("HARBOR_PROVIDERS", json.dumps({"acme": "acme_sbx.env:AcmeEnvironment"}))
    (tmp_path / "tasks" / "hello").mkdir(parents=True)
    task = AgentTask(task_id="hello", agent="nop", provider="acme")
    config = build_trial_config(task, "http://relay/v1", HarborSettings(tasks_dir=tmp_path / "tasks"))
    assert config.environment.import_path == "acme_sbx.env:AcmeEnvironment"


FAKE_PROVIDER = textwrap.dedent("""
    import asyncio
    import json
    import urllib.request

    EVENTS = []


    class FakeEnvironment:
        \"\"\"The methods of Harbor's BaseEnvironment that a trial calls (ENV_PROVIDER.md).\"\"\"

        def __init__(self, *, session_id, **kwargs):
            self.session_id, self.kwargs, self.reward = session_id, kwargs, None

        async def start(self, force_build):
            EVENTS.append(("start", self.session_id))

        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            # the agent inside the sandbox: one model call to its records proxy
            base = env["OPENAI_BASE_URL"]
            body = json.dumps({"model": "policy", "messages": [{"role": "user", "content": "ls"}]}).encode()
            req = urllib.request.Request(base + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            def call():
                with urllib.request.urlopen(req, timeout=10) as reply:
                    return reply.status, json.loads(reply.read())

            status, answer = await asyncio.to_thread(call)  # the agent is another process in real life
            EVENTS.append(("agent", self.session_id, base, status, answer))
            self.reward = 1.0  # the verifier's result
            return {"return_code": 0}

        async def stop(self, delete):
            EVENTS.append(("stop", self.session_id, delete))
""")


class _HarborLikeTrial:
    """Harbor's Trial in small: import the environment by import_path, start, agent, verifier, stop."""

    def __init__(self, config: Any) -> None:
        self.config, self.paths = config, None

    @classmethod
    async def create(cls, config: Any) -> _HarborLikeTrial:
        return cls(config)

    async def run(self) -> Any:
        import importlib

        module, _, name = self.config.environment.import_path.partition(":")
        env = getattr(importlib.import_module(module), name)(session_id=self.config.session_id,
                                                             **self.config.environment.kwargs)
        await env.start(force_build=False)
        try:
            await env.exec("agent", env=self.config.agent.env)
        finally:
            await env.stop(delete=True)
        return SimpleNamespace(exception_info=None, verifier_result=SimpleNamespace(rewards={"reward": env.reward}))


def test_a_registered_fake_provider_runs_a_trial_against_the_records_proxy(tmp_path, monkeypatch):
    (tmp_path / "acme_fake_env.py").write_text(FAKE_PROVIDER)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("HARBOR_PROVIDERS", json.dumps({"acme": "acme_fake_env:FakeEnvironment"}))

    engine = FakeEngine()
    engine.start(None, 1)
    proxy = RecordsProxy(engine, SimpleNamespace(spec=SimpleNamespace(keeps_history_reasoning=False)),
                         model="policy", signed_sampling=SIGNED)

    async def handle(entry: str, body: dict[str, Any], *, trajectory_id: str) -> dict[str, Any]:
        return {"object": "chat.completion", "choices": [], "trajectory": trajectory_id}

    proxy.handle = handle  # type: ignore[method-assign]  # the access layer is under test, not the gateway
    recorder = ProxyTokenRecorder(proxy, path_secret=SECRET)

    def build(task: AgentTask, endpoint: str) -> Any:
        return SimpleNamespace(session_id=task.trajectory_id,
                               environment=SimpleNamespace(import_path=provider_import_path(task.provider),
                                                           kwargs={"site": "x"}),
                               agent=SimpleNamespace(env=agent_env(task.agent, endpoint)))

    monkeypatch.delenv("HARBOR_MAX_SANDBOXES", raising=False)
    harness = HarborHarness(build, trial_cls=_HarborLikeTrial)  # "acme" uncapped

    async def go():
        try:
            out = []
            for session in ("traj-a", "traj-b"):
                endpoint = await recorder.open(session, {})
                task = AgentTask(task_id="t", agent="nop", provider="acme", trajectory_id=session)
                out.append((session, endpoint, await harness.run(task, endpoint)))
            return out
        finally:
            await recorder.aclose()

    runs = asyncio.run(go())
    events = sys.modules["acme_fake_env"].EVENTS
    agents = [e for e in events if e[0] == "agent"]
    assert [e[0] for e in events] == ["start", "agent", "stop"] * 2
    for (session, endpoint, verdict), agent in zip(runs, agents):
        token = proxy.token(session)
        assert agent[2] == endpoint and f"/k/{SECRET}/t/{session}/{token}/v1" in endpoint
        assert agent[3] == 200 and agent[4]["trajectory"] == session
        assert verdict.rewards == {"reward": 1.0}
    assert proxy.token("traj-a") != proxy.token("traj-b")
    sys.modules.pop("acme_fake_env", None)

