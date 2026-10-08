"""codex harness against the session-service protocol test double (decoupling 3.8).

The trusted-layer session code (``codex_openenv_agent_function``) talks real
HTTP to ``tests/session_service_double.py`` instead of a Miles session server;
the protocol is ``yeto/rl/harness/session_protocol.md``.  CPU only, localhost.
"""

from __future__ import annotations

import asyncio

import aiohttp
import pytest

from session_service_double import SessionServiceDouble
from yeto.rl.harness.codex import codex_openenv_agent_function as agent_fn
from yeto.rl.harness.codex import compaction_bridge


def _run(coro):
    return asyncio.run(coro)


async def _main_session(double: SessionServiceDouble) -> str:
    async with aiohttp.ClientSession() as http:
        async with http.post(f"{double.router}/sessions", json={}) as reply:
            assert reply.status == 200
            return double.session_url((await reply.json())["session_id"])


def test_create_chat_collect_delete_through_the_protocol():
    async def scenario():
        async with SessionServiceDouble() as double:
            base = await _main_session(double)
            ids, urls = await agent_fn.create_segment_sessions(
                base, {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "max_tokens": 5}, 2)
            assert urls == [double.session_url(i) for i in ids]
            for sid in ids:  # sampling defaults went into CreateSessionRequest, typed
                assert double.sessions[sid].params == {"evaluation": False, "temperature": 0.7,
                                                       "top_p": 0.9, "top_k": 20}
            async with aiohttp.ClientSession() as http:
                async with http.post(f"{urls[0]}/v1/chat/completions",
                                     json={"messages": [{"role": "user", "content": "hi"}]}) as r:
                    body = await r.json()
                async with http.post(f"{urls[0]}/samples", json={"max_seq_len": 64}) as r:
                    samples = await r.json()
            choice = body["choices"][0]
            assert [t[1] for t in choice["meta_info"]["output_token_logprobs"]] == [11, 12, 13]
            assert double.sessions[ids[0]].records[0]["request"]["logprobs"] is True
            assert samples["samples"][0]["tokens"] == [11, 12, 13]
            # Unreturned segments are released; a second release sees 404 and is still fine.
            segments = {compaction_bridge.SESSIONS_METADATA_KEY: ids}
            assert await agent_fn.release_unreturned_segments(base, segments) == []
            assert double.deleted == ids
            assert await agent_fn.delete_segment_sessions(base, ids) == []
    _run(scenario())


def test_half_failed_pre_creation_deletes_what_it_made():
    async def scenario():
        async with SessionServiceDouble() as double:
            base = await _main_session(double)
            calls = {"n": 0}

            async def flaky_post(url, body):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("router went away")
                return await agent_fn._post_json(url, body)

            with pytest.raises(RuntimeError, match="router went away"):
                await agent_fn.create_segment_sessions(base, {}, 3, post=flaky_post)
            assert double.deleted == ["s2"]  # the one segment created before the failure
            assert set(double.sessions) == {"s1"}  # only the main session remains
    _run(scenario())


def test_bad_create_body_is_rejected_by_the_double():
    async def scenario():
        async with SessionServiceDouble() as double:
            base = await _main_session(double)
            with pytest.raises(RuntimeError, match="HTTP 400"):
                await agent_fn._post_json(f"{double.router}/sessions", {"top_k": "20"})
            with pytest.raises(RuntimeError, match="no session_id"):
                async def no_id(url, body):
                    return {}
                await agent_fn.create_segment_sessions(base, {}, 1, post=no_id)
    _run(scenario())
