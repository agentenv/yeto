"""S26 E9.5 in the pinned Miles image: Miles itself drives the Harbor suspend/resume hooks (CPU).

Skipped without the ``miles`` package (local venvs, yeto-cpu-test); run in the
Miles image (``yeto-miles-ports``).  Real pieces: Miles' agent-hook loader
(``miles.utils.misc.call_agent_hook``), Miles' session URL helper, the yeto
hooks, gate and relay.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("miles")
pytest.importorskip("aiohttp")


def test_miles_hook_loader_finds_and_runs_the_harbor_hooks(monkeypatch):
    from miles.utils.misc import call_agent_hook, load_agent_hook

    from yeto.rl.adapters.miles import harbor_agent

    args = SimpleNamespace(custom_agent_function_path=harbor_agent.AGENT_FUNCTION)
    assert load_agent_hook(args, "suspend") is harbor_agent.suspend
    assert load_agent_hook(args, "resume") is harbor_agent.resume

    async def go():
        harbor_agent._turn_gates.clear()
        monkeypatch.delenv(harbor_agent.RELAY_PORT_ENV, raising=False)
        with pytest.raises(harbor_agent.HarnessConfigError):
            await call_agent_hook(args, "suspend")  # fail closed: Miles propagates it
        monkeypatch.setenv(harbor_agent.RELAY_PORT_ENV, "0")
        await call_agent_hook(args, "suspend")
        assert harbor_agent.turn_gate().closed
        await call_agent_hook(args, "resume")
        assert not harbor_agent.turn_gate().closed

    asyncio.run(go())


def test_relay_forwards_below_the_miles_session_url():
    from aiohttp import web
    from miles.rollout.agentic.session import openai_session_url

    from yeto.rl.agentic.suspend import TurnGate
    from yeto.rl.harness.gateway.gate_relay import GateRelay

    async def go():
        seen = []

        async def handle(request):
            seen.append(request.path)
            return web.json_response({"ok": True})

        app = web.Application()
        app.router.add_post("/{tail:.*}", handle)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        base = f"http://127.0.0.1:{runner.addresses[0][1]}/sessions/abc"
        relay = GateRelay(TurnGate())
        url = await relay.open(openai_session_url(base))
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{url}/chat/completions", json={}) as resp:
                assert resp.status == 200
        await relay.aclose()
        await runner.cleanup()
        return seen

    assert asyncio.run(go()) == ["/sessions/abc/v1/chat/completions"]


def test_miles_fork_has_the_suspend_flags_policy_age_emits():
    """Read the sources: importing the rollout module needs CUDA (libcuda) in the image."""
    from importlib.util import find_spec
    from pathlib import Path

    from yeto.rl.adapters.miles.policy_age import policy_age_argv

    arguments = Path(find_spec("miles.utils.arguments").origin).read_text()
    rollout = Path(find_spec("miles.rollout.inference_rollout.inference_rollout_train").origin).read_text()
    assert all(flag in arguments for flag in policy_age_argv(1, agentic=True) if flag.startswith("--"))
    assert "async def suspend(" in rollout and "async def resume(" in rollout
    assert 'call_agent_hook(args, "suspend")' in rollout and 'call_agent_hook(state.args, "resume")' in rollout
