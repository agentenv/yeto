"""Records proxy: one random token per trajectory.  A sandbox's policy knows the
shared path secret and can guess other session ids; it must not read, close or
write any trajectory but its own.  CPU only."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.harness.gateway.proxy import TOKEN_BYTES, RecordsProxy, TrajectoryForbidden
from yeto.rl.harness.gateway.recorder import ProxyTokenRecorder

SIGNED = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 16}
SECRET = "p" * 32
TEMPLATE = SimpleNamespace(spec=SimpleNamespace(keeps_history_reasoning=False))


def _proxy() -> RecordsProxy:
    """A proxy whose gateway is replaced by echoes: these tests check the access layer only."""
    engine = FakeEngine()
    engine.start(None, 1)
    proxy = RecordsProxy(engine, TEMPLATE, model="policy", signed_sampling=SIGNED)
    proxy.calls = []  # type: ignore[attr-defined]

    async def handle(entry: str, body: dict[str, Any], *, trajectory_id: str) -> dict[str, Any]:
        proxy.calls.append((entry, trajectory_id))
        return {"id": "x", "object": "chat.completion", "choices": [], "trajectory": trajectory_id}

    proxy.handle = handle  # type: ignore[method-assign]
    proxy.records = lambda tid: {"trajectory": tid, "sessions": []}  # type: ignore[method-assign]
    proxy.closed = []  # type: ignore[attr-defined]
    proxy.close = proxy.closed.append  # type: ignore[method-assign]
    return proxy


def _client(app: Any):
    from aiohttp.test_utils import TestClient, TestServer

    return TestClient(TestServer(app))


CHAT = {"model": "policy", "messages": [{"role": "user", "content": "hi"}]}


# ------------------------------------------------------------- records proxy
def test_another_trajectorys_routes_refuse_a_foreign_token():
    proxy = _proxy()
    a, b = proxy.open("sess-a"), proxy.open("sess-b")
    ta, tb = proxy.token(a), proxy.token(b)
    assert ta != tb and len(ta) >= 43  # 256 random bits each

    async def go():
        async with _client(proxy.app(control=True)) as client:
            assert (await client.post(f"/t/{a}/{ta}/v1/chat/completions", json=CHAT)).status == 200
            # A's policy knows B's id but only its own token: read, close, pollute all refused
            assert (await client.post(f"/t/{b}/{ta}/v1/chat/completions", json=CHAT)).status == 403
            assert (await client.post(f"/t/{b}/{ta}/v1/responses", json={})).status == 403
            assert (await client.get(f"/t/{b}/{ta}/records")).status == 403
            assert (await client.post(f"/t/{b}/{ta}/close")).status == 403
            # the old token-less routes are gone
            assert (await client.get(f"/t/{b}/records")).status == 404
            assert (await client.post(f"/t/{b}/close")).status == 404
            assert (await client.post(f"/t/{b}/v1/chat/completions", json=CHAT)).status == 404
            # unknown id and wrong token answer the same 403 (no oracle for open ids)
            unknown = await client.post(f"/t/sess-zz/{ta}/v1/chat/completions", json=CHAT)
            wrong = await client.post(f"/t/{a}/{'x' * 43}/v1/chat/completions", json=CHAT)
            assert unknown.status == wrong.status == 403
            assert await unknown.json() == await wrong.json()
            assert (await client.get(f"/t/{b}/{tb}/records")).status == 200

    asyncio.run(go())
    assert proxy.calls == [("chat", a)] and proxy.closed == []


def test_a_leaked_token_reaches_one_trajectory_and_dies_with_it():
    proxy = _proxy()
    ids = [proxy.open(f"s{i}") for i in range(5)]
    leaked = proxy.token(ids[0])

    async def go():
        async with _client(proxy.app(control=True)) as client:
            statuses = [(await client.post(f"/t/{tid}/{leaked}/v1/chat/completions", json=CHAT)).status
                        for tid in ids]
            assert statuses == [200, 403, 403, 403, 403]
            proxy.forget(ids[0])
            after = await client.post(f"/t/{ids[0]}/{leaked}/v1/chat/completions", json=CHAT)
            assert after.status == 403
            proxy.open(ids[0])  # the id is reused: a new token, the leaked one stays dead
            again = await client.post(f"/t/{ids[0]}/{leaked}/v1/chat/completions", json=CHAT)
            assert again.status == 403

    asyncio.run(go())
    assert [tid for _, tid in proxy.calls] == [ids[0]]


def test_control_routes_need_a_loopback_peer_without_a_forwarding_hop():
    proxy = _proxy()
    tid = proxy.open("s1")
    tok = proxy.token(tid)

    async def go():
        async with _client(proxy.app(control=True)) as client:
            assert (await client.get(f"/t/{tid}/{tok}/records")).status == 200
            hop = {"X-Forwarded-For": "203.0.113.7"}
            assert (await client.get(f"/t/{tid}/{tok}/records", headers=hop)).status == 403
            assert (await client.post(f"/t/{tid}/{tok}/close", headers={"Forwarded": "for=1.2.3.4"})).status == 403
            assert (await client.post("/trajectories", headers=hop)).status == 403
            # the model route itself is not peer-restricted (the sandbox is remote)
            assert (await client.post(f"/t/{tid}/{tok}/v1/chat/completions", json=CHAT, headers=hop)).status == 200
        async with _client(proxy.app(control=True, control_peers=("10.0.0.0/8",))) as client:  # loopback not allowed
            assert (await client.get(f"/t/{tid}/{tok}/records")).status == 403
            assert (await client.post("/trajectories")).status == 403

    asyncio.run(go())
    assert proxy.closed == []


def test_created_trajectory_url_carries_its_own_token():
    proxy = _proxy()

    async def go():
        async with _client(proxy.app(control=True)) as client:
            first = await (await client.post("/trajectories")).json()
            second = await (await client.post("/trajectories")).json()
        return first, second

    first, second = asyncio.run(go())
    for created in (first, second):
        assert created["base_url"].endswith(f"/t/{created['trajectory_id']}/{proxy.token(created['trajectory_id'])}/v1")
    assert first["trajectory_id"] != second["trajectory_id"]


def test_recorder_serves_only_model_routes_behind_the_path_secret():
    proxy = _proxy()
    recorder = ProxyTokenRecorder(proxy, path_secret=SECRET)

    async def go():
        import aiohttp

        try:
            url_a = await recorder.open("traj-1", {})
            url_b = await recorder.open("traj-2", {})
            root = recorder._base_url.split("/k/")[0]
            ta = proxy.token("traj-1")
            async with aiohttp.ClientSession() as http:
                async def status(method: str, url: str, **kw: Any) -> int:
                    async with http.request(method, url, **kw) as reply:
                        return reply.status

                ok = await status("POST", f"{url_a}/chat/completions", json=CHAT)
                # the path secret is still the outer layer
                no_secret = await status("POST", f"{root}/t/traj-1/{ta}/v1/chat/completions", json=CHAT)
                # control routes are not served at all, even with the right token
                records = await status("GET", f"{root}/k/{SECRET}/t/traj-1/{ta}/records")
                close = await status("POST", f"{root}/k/{SECRET}/t/traj-1/{ta}/close")
                create = await status("POST", f"{root}/k/{SECRET}/trajectories")
                foreign = await status("POST", f"{url_b.rsplit('/t/', 1)[0]}/t/traj-2/{ta}/v1/chat/completions",
                                       json=CHAT)
            return url_a, url_b, ok, no_secret, records, close, create, foreign
        finally:
            await recorder.aclose()

    url_a, url_b, ok, no_secret, records, close, create, foreign = asyncio.run(go())
    assert f"/k/{SECRET}/t/traj-1/{proxy.token('traj-1')}/v1" in url_a
    assert ok == 200 and no_secret == 404 and foreign == 403
    assert records == close == create == 404 or {records, close, create} <= {404, 405}


def test_open_is_thread_safe_and_tokens_are_unique():
    proxy = _proxy()
    ids: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(n: int) -> None:
        try:
            got = [proxy.open(f"w{n}-{i}") for i in range(200)]
            with lock:
                ids.extend(got)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(ids) == 1600
    tokens = {proxy.token(tid) for tid in ids}
    assert len(tokens) == 1600 and TOKEN_BYTES == 32
    with pytest.raises(TrajectoryForbidden, match="already open"):
        proxy.open(ids[0])
    with pytest.raises(TrajectoryForbidden):
        proxy.open("a/b")


def test_concurrent_requests_stay_on_their_own_trajectory():
    proxy = _proxy()
    ids = [proxy.open(f"c{i}") for i in range(16)]

    async def go():
        async with _client(proxy.app(control=True)) as client:
            async def call(tid: str, tok: str) -> tuple[int, str | None]:
                reply = await client.post(f"/t/{tid}/{tok}/v1/chat/completions", json=CHAT)
                return reply.status, (await reply.json()).get("trajectory")

            own = [call(tid, proxy.token(tid)) for tid in ids for _ in range(4)]
            cross = [call(ids[i], proxy.token(ids[(i + 1) % len(ids)])) for i in range(len(ids))]
            return await asyncio.gather(*own), await asyncio.gather(*cross)

    own, cross = asyncio.run(go())
    assert [r for r in own] == [(200, tid) for tid in ids for _ in range(4)]
    assert all(status == 403 for status, _ in cross)
    assert sorted(tid for _, tid in proxy.calls) == sorted(tid for tid in ids for _ in range(4))


def test_the_default_app_serves_no_control_route():
    """A deployment that serves ``proxy.app()`` on a sandbox-reachable port exposes model routes only."""
    proxy = _proxy()
    tid = proxy.open("s1")
    tok = proxy.token(tid)

    async def go():
        async with _client(proxy.app()) as client:
            assert (await client.post(f"/t/{tid}/{tok}/v1/chat/completions", json=CHAT)).status == 200
            for method, path in (("GET", f"/t/{tid}/{tok}/records"), ("POST", f"/t/{tid}/{tok}/close"),
                                 ("POST", "/trajectories"), ("GET", f"/t/{tid}/records"),
                                 ("POST", f"/t/{tid}/close")):
                assert (await client.request(method, path)).status in (404, 405), path

    asyncio.run(go())
    assert proxy.closed == []
