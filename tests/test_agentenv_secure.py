"""SdkAgentenvClient creates secure sandboxes and never leaks the envd access token."""

from __future__ import annotations

import logging
import sys
import types

import httpx
import pytest

from yeto.rl.harness.harbor import agentenv_client as ac

TOKEN = "envd-secret-token-123"
API_KEY = "api-key-xyz"


class _SandboxException(Exception):
    pass


class _FakeBox:
    def __init__(self, sandbox_id: str, token: str | None) -> None:
        self.sandbox_id = sandbox_id
        self._envd_access_token = token


class _FakeSandbox:
    connect_token: str | None = TOKEN
    connects: list[str] = []
    kills: list[str] = []

    @classmethod
    def connect(cls, sandbox_id: str, **opts):
        cls.connects.append(sandbox_id)
        return _FakeBox(sandbox_id, cls.connect_token)

    @classmethod
    def kill(cls, sandbox_id: str, **opts):
        cls.kills.append(sandbox_id)


@pytest.fixture
def fake_e2b(monkeypatch):
    e2b = types.ModuleType("e2b")
    e2b.Sandbox = _FakeSandbox
    exc = types.ModuleType("e2b.exceptions")
    exc.SandboxException = _SandboxException
    monkeypatch.setitem(sys.modules, "e2b", e2b)
    monkeypatch.setitem(sys.modules, "e2b.exceptions", exc)
    monkeypatch.delenv(ac.INSECURE_ENV, raising=False)
    _FakeSandbox.connect_token = TOKEN
    _FakeSandbox.connects = []
    _FakeSandbox.kills = []
    return _FakeSandbox


@pytest.fixture
def posts(monkeypatch):
    calls: list[dict] = []
    reply = {"json": {"sandboxID": "sb-1", "envdAccessToken": TOKEN}, "status": 201}

    def fake_post(url, *, json, headers, timeout, trust_env):
        calls.append({"url": url, "json": json, "headers": headers})
        return httpx.Response(reply["status"], json=reply["json"], request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls, reply


def _client(**kw) -> ac.SdkAgentenvClient:
    return ac.SdkAgentenvClient(api_key=API_KEY, api_url="http://gw.test", sandbox_url="http://gw.test", **kw)


@pytest.mark.parametrize("cold", [True, False])
def test_create_requests_secure_and_keeps_the_token(fake_e2b, posts, cold):
    calls, _ = posts
    client = _client(cold_start=cold)
    sid = client.create(template="img:1", ttl_seconds=60, metadata={"k": "v"}, env=None, idempotency_key="idem")
    assert sid == "sb-1"
    (call,) = calls
    assert call["json"]["secure"] is True
    assert call["url"].endswith("/sandboxes-cold" if cold else "/sandboxes")
    assert call["headers"]["X-API-Key"] == API_KEY
    if not cold:
        assert call["json"]["templateID"] == "img:1"
        assert call["headers"]["Idempotency-Key"] == "idem"
    assert client._tokens["sb-1"] == TOKEN


def test_data_plane_connection_carries_the_token(fake_e2b, posts):
    client = _client(cold_start=True)
    sid = client.create(template="img:1", ttl_seconds=60, metadata=None, env=None, idempotency_key="i")
    box = client._box(sid)
    assert box._envd_access_token == TOKEN
    client.delete(sid)
    assert sid not in client._tokens and fake_e2b.kills == [sid]


def test_connection_without_a_token_is_refused(fake_e2b, posts):
    client = _client(cold_start=True)
    sid = client.create(template="img:1", ttl_seconds=60, metadata=None, env=None, idempotency_key="i")
    fake_e2b.connect_token = None
    with pytest.raises(ac.AgentenvError, match="no envd access token"):
        client._box(sid)


def test_gateway_ignoring_secure_fails_closed_and_deletes(fake_e2b, posts):
    _, reply = posts
    reply["json"] = {"sandboxID": "sb-2"}
    client = _client(cold_start=True)
    with pytest.raises(ac.AgentenvError, match="no envd access token"):
        client.create(template="img:1", ttl_seconds=60, metadata=None, env=None, idempotency_key="i")
    assert fake_e2b.kills == ["sb-2"]


def test_insecure_is_an_explicit_opt_out(fake_e2b, posts, monkeypatch):
    calls, reply = posts
    reply["json"] = {"sandboxID": "sb-3"}
    monkeypatch.setenv(ac.INSECURE_ENV, "1")
    client = _client(cold_start=True)
    fake_e2b.connect_token = None
    sid = client.create(template="img:1", ttl_seconds=60, metadata=None, env=None, idempotency_key="i")
    assert calls[0]["json"]["secure"] is False
    client._box(sid)  # no token required
    assert _client(cold_start=True, secure=True)._secure is True


def test_token_never_reaches_logs_or_errors(fake_e2b, posts, caplog):
    calls, reply = posts
    caplog.set_level(logging.DEBUG)
    client = _client(cold_start=True)
    sid = client.create(template="img:1", ttl_seconds=60, metadata=None, env=None, idempotency_key="i")
    client._box(sid)
    errors = []
    reply["status"], reply["json"] = 500, {"message": "boom", "envdAccessToken": TOKEN}
    with pytest.raises(ac.AgentenvError) as info:
        client.create(template="img:1", ttl_seconds=60, metadata=None, env=None, idempotency_key="i")
    errors.append(str(info.value))
    fake_e2b.connect_token = None
    client._boxes.clear()
    with pytest.raises(ac.AgentenvError) as info:
        client._box(sid)
    errors.append(str(info.value))
    assert TOKEN not in caplog.text and API_KEY not in caplog.text
    for text in errors:
        assert TOKEN not in text and API_KEY not in text
