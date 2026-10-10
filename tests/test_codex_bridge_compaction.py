"""CompactionRL compaction at the Codex Responses bridge (design D8; CPU only).

Fake Codex client (direct POSTs to the bridge, as Codex app-server would send
them) + fake session server (aiohttp, per-session routes).  No Ray, no Miles.
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from test_secrlenv_codex_harness import _codex_body, _completion, _fake_miles_sse, _sse_events
from yeto.rl.compaction import RESUME_TEMPLATE, SUMMARY_PROMPT
from yeto.rl.harness.codex import codex_harness_agent as harness
from yeto.rl.harness.codex import codex_openenv_agent_function as adapter
from yeto.rl.harness.codex import codex_openenv_generate as generate_wrapper
from yeto.rl.harness.codex import compaction_bridge as cb

C, T = 8000, 4000


def _run(coro):
    return asyncio.run(coro)


def _summary(text: str, total: int, finish: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-summary",
        "object": "chat.completion",
        "created": 1,
        "model": harness.BACKEND_MODEL,
        "choices": [{"index": 0, "finish_reason": finish, "message": {
            "role": "assistant", "content": text, "reasoning_content": "compress"}}],
        "usage": {"prompt_tokens": total - 10, "completion_tokens": 10, "total_tokens": total},
    }


class FakeSessionServer:
    """Per-session Chat Completions; records (session, body, headers)."""

    def __init__(self, replies: list[dict[str, Any]]):
        self.replies = list(replies)
        self.requests: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    async def __aenter__(self):
        async def chat(request: web.Request) -> web.Response:
            self.requests.append((request.match_info["sid"], await request.json(), dict(request.headers)))
            return web.Response(body=_fake_miles_sse(self.replies.pop(0)), content_type="text/event-stream")

        app = web.Application()
        app.router.add_post("/sessions/{sid}/v1/chat/completions", chat)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.router = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *_):
        await self.runner.cleanup()

    def url(self, sid: str) -> str:
        return f"{self.router}/sessions/{sid}"


class FakeCodex:
    """Sends what Codex app-server sends: the full append-only history every turn."""

    def __init__(self, bridge: harness._ResponsesBridge, http: aiohttp.ClientSession):
        self.bridge, self.http, self.raw = bridge, http, []

    async def turn(self, tool_output: str | None = None) -> tuple[int, dict[str, Any] | None]:
        if tool_output is not None:
            self.bridge.expect_tool_output(self.bridge._pending.call_id, tool_output)
            history = copy.deepcopy(self.bridge._expected_input)
        else:
            history = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "solve target"}]}]
        async with self.http.post(f"{self.bridge.url}/v1/responses", json=_codex_body(history),
                                  headers={"Authorization": f"Bearer {self.bridge.token}"}) as response:
            text = await response.text()
            self.raw.append((response.status, text))
            if response.status != 200:
                return response.status, None
            return 200, _sse_events(text)[-1]["response"]


def _env(monkeypatch, *, on: bool = True, t_comp: int = T, max_tokens: int = 512):
    monkeypatch.setenv("YETO_CODEX_BACKEND_MAX_TOKENS", str(max_tokens))
    monkeypatch.delenv("YETO_CODEX_COMPACTION_ENABLED", raising=False)
    if on:
        monkeypatch.setenv(cb.COMPACTIONRL_ENV, "1")
        monkeypatch.setenv(cb.T_COMP_ENV, str(t_comp))
    else:
        monkeypatch.delenv(cb.COMPACTIONRL_ENV, raising=False)
        monkeypatch.delenv(cb.T_COMP_ENV, raising=False)


def _exec(i: int, total: int) -> dict[str, Any]:
    return _completion("terminal.exec", json.dumps({"command": f"step {i}"}), f"call-{i}", f"think {i}", total)


def _submit(total: int) -> dict[str, Any]:
    return _completion("submit", '{"evidence":"done"}', "call-submit", "finish", total)


def _bridge(server, metrics, *, cls=cb.CompactionRLBridge, **kw):
    if cls is cb.CompactionRLBridge:
        kw.setdefault("segment_base_urls", [server.url(f"S{i}") for i in (1, 2, 3)])
        kw.setdefault("config", cb.compaction_config(C))
    return cls(server.url("S0"), "solve target", {"max_tokens": 512}, metrics, max_seq_len=C, **kw)


def _by_session(server):
    out: dict[str, list[dict[str, Any]]] = {}
    for sid, body, _headers in server.requests:
        out.setdefault(sid, []).append(body)
    return out


# --- trigger, trainable summary, rebuild, session switch --------------------

def test_trigger_summary_rebuild_and_new_session(monkeypatch):
    _env(monkeypatch)
    replies = [_exec(1, 500), _exec(2, 3900), _summary("<analysis>scratch</analysis><summary>S-ONE</summary>", 4300),
               _submit(900)]

    async def scenario():
        metrics = harness.legacy.AgentMetrics()
        async with FakeSessionServer(replies) as server:
            async with _bridge(server, metrics) as bridge, aiohttp.ClientSession() as http:
                codex = FakeCodex(bridge, http)
                assert (await codex.turn())[0] == 200
                assert (await codex.turn('{"output":"a"}'))[0] == 200  # 500+bound: no trigger
                status, resp = await codex.turn('{"output":"b"}')  # 3900+bound > C-T: compaction
                assert status == 200 and resp["output"][1]["name"] == "submit"
                assert bridge._expected_input is not None and len(bridge._history_after_model) == 1 + 3 * 2 + 2  # Codex keeps the full history
            return server, metrics

    server, metrics = _run(scenario())
    sessions = _by_session(server)
    assert [len(sessions[s]) for s in ("S0", "S1")] == [3, 1] and set(sessions) == {"S0", "S1"}
    first, second, summary_req = sessions["S0"]
    # Summary: same policy, same session (trainable turn with logprobs in the session record).
    assert summary_req["messages"][:-1] == second["messages"] + summary_req["messages"][len(second["messages"]):-1]
    assert summary_req["messages"][-1] == {"role": "user", "content": SUMMARY_PROMPT}
    assert summary_req["tool_choice"] == "none" and summary_req["model"] == second["model"]
    # Explicit D8 choice: the summary request carries no tool table (execution turns do).
    assert summary_req["tools"] == list(cb.SUMMARY_REQUEST_TOOLS) == []
    assert summary_req["tool_choice"] == cb.SUMMARY_REQUEST_TOOL_CHOICE and second["tools"]
    assert all(not k.lower().startswith("x-miles-compaction") for _s, _b, h in server.requests for k in h)
    # Rebuilt context = system + u_resume(summary without <analysis>) + last k=2 atomic steps, in S1.
    resumed = sessions["S1"][0]["messages"]
    assert resumed[0] == {"role": "system", "content": harness.BASE_INSTRUCTIONS}
    assert resumed[1] == {"role": "user", "content": RESUME_TEMPLATE.format(summary="S-ONE")}
    assert resumed[2:] == summary_req["messages"][2:-1]
    assert len(resumed[2:]) == 4 and "scratch" not in json.dumps(resumed)
    record = cb.compaction_counters(metrics)[cb.METRICS_KEY]
    assert record["compactions"] == 1 and record["t_comp"] == T and record["context_budget"] == C
    assert record["segments"][0]["summary_ok"] is True and record["segments"][0]["kept_steps"] == 2
    assert metrics.turns == 3  # the summary is not a Codex turn


def test_trigger_boundary_is_remaining_context_below_t_comp(monkeypatch):
    _env(monkeypatch)
    bridge = cb.CompactionRLBridge("http://127.0.0.1:1/sessions/S0", "p", {"max_tokens": 512},
                                   harness.legacy.AgentMetrics(), max_seq_len=C,
                                   segment_base_urls=["http://127.0.0.1:1/sessions/S1"] * 3)
    bridge._atomic_steps = [({}, {})]
    assert not bridge._should_compact(C - T)  # C-|h| == T_comp: no trigger
    assert bridge._should_compact(C - T + 1)
    bridge._compaction_count = 3
    assert not bridge._should_compact(C)  # cap reached
    with pytest.raises(harness.CodexHarnessError, match="one pre-created session"):
        cb.CompactionRLBridge("http://h/sessions/S0", "p", {"max_tokens": 512}, harness.legacy.AgentMetrics(),
                              max_seq_len=C, segment_base_urls=["http://h/sessions/S1"])
    with pytest.raises(harness.CodexHarnessError, match="config rejected"):
        cb.compaction_config(T)  # C must exceed T_comp


def test_kept_steps_shrink_when_rebuilt_context_would_trigger(monkeypatch):
    _env(monkeypatch)
    big = "x" * 1500
    replies = [_exec(1, 300), _exec(2, 2500), _summary("<summary>S</summary>", 5900), _submit(1200)]

    async def scenario():
        metrics = harness.legacy.AgentMetrics()
        async with FakeSessionServer(replies) as server:
            async with _bridge(server, metrics) as bridge, aiohttp.ClientSession() as http:
                codex = FakeCodex(bridge, http)
                await codex.turn()
                await codex.turn(big)
                assert (await codex.turn(big))[0] == 200
            return server, metrics

    server, metrics = _run(scenario())
    resumed = _by_session(server)["S1"][0]["messages"]
    summary_req = _by_session(server)["S0"][-1]["messages"]
    k = cb.compaction_counters(metrics)[cb.METRICS_KEY]["segments"][0]["kept_steps"]
    assert k < 2  # two 1.5 kB steps would make the rebuilt context trigger again
    assert resumed[2:] == (summary_req[2:-1][-2 * k:] if k else [])


def test_cap_of_three_then_run_to_full_truncation(monkeypatch):
    _env(monkeypatch)
    replies = [_exec(1, 3900)]
    for i in range(3):
        replies += [_summary(f"<summary>S{i}</summary>", 4200), _exec(i + 2, 3900)]
    replies += [_exec(9, 7995)]  # 4th trigger is ignored; the context then fills up

    async def scenario():
        metrics = harness.legacy.AgentMetrics()
        async with FakeSessionServer(replies) as server:
            async with _bridge(server, metrics) as bridge, aiohttp.ClientSession() as http:
                codex = FakeCodex(bridge, http)
                assert (await codex.turn())[0] == 200
                for _ in range(4):
                    assert (await codex.turn('{"output":"o"}'))[0] == 200
                status, _resp = await codex.turn('{"output":"o"}')
                fatal = bridge.fatal.result() if bridge.fatal.done() else None
            return server, metrics, status, fatal

    server, metrics, status, fatal = _run(scenario())
    assert status == 400 and isinstance(fatal, harness.CodexSequenceLimit)
    assert metrics.max_seq_len_hit == 1  # -> policy status max_seq_len (truncated, still rewarded)
    sessions = _by_session(server)
    assert [len(sessions[f"S{i}"]) for i in range(4)] == [2, 2, 2, 2]
    assert sessions["S3"][0]["messages"][1]["content"] == RESUME_TEMPLATE.format(summary="S2")
    record = cb.compaction_counters(metrics)[cb.METRICS_KEY]
    assert record["compactions"] == 3 and len(record["segments"]) == 3


def test_summary_without_tag_is_kept_and_flagged(monkeypatch):
    _env(monkeypatch)
    replies = [_exec(1, 3900), _summary("<analysis>a</analysis>plain state", 4400, finish="length"), _submit(500)]

    async def scenario():
        metrics = harness.legacy.AgentMetrics()
        async with FakeSessionServer(replies) as server:
            async with _bridge(server, metrics) as bridge, aiohttp.ClientSession() as http:
                codex = FakeCodex(bridge, http)
                await codex.turn()
                assert (await codex.turn('{"output":"o"}'))[0] == 200
            return server, metrics

    server, metrics = _run(scenario())
    seg = cb.compaction_counters(metrics)[cb.METRICS_KEY]["segments"][0]
    assert seg["summary_ok"] is False and seg["summary_finish_reason"] == "length"
    assert _by_session(server)["S1"][0]["messages"][1]["content"] == RESUME_TEMPLATE.format(summary="plain state")


def test_no_room_for_summary_truncates(monkeypatch):
    _env(monkeypatch)
    replies = [_exec(1, 7000)]

    async def scenario():
        async with FakeSessionServer(replies) as server:
            async with _bridge(server, harness.legacy.AgentMetrics()) as bridge, aiohttp.ClientSession() as http:
                codex = FakeCodex(bridge, http)
                await codex.turn()
                status, _ = await codex.turn("y" * 800)
                return status, bridge.fatal.result(), len(server.requests)

    status, fatal, n = _run(scenario())
    assert status == 400 and isinstance(fatal, harness.CodexSequenceLimit) and n == 1


def test_codex_argv_disables_codex_auto_compaction(monkeypatch):
    _env(monkeypatch)
    bridge = SimpleNamespace(url="http://127.0.0.1:9")
    argv = cb.codex_argv(harness.Path("/codex"), bridge)
    stock = harness._codex_argv(harness.Path("/codex"), bridge)
    assert argv == [*stock, "-c", "model_auto_compact_token_limit=9223372036854775807"]
    assert not any("auto_compact" in a for a in stock)


# --- compaction off: stock path byte-identical ------------------------------

def _script():
    return [_exec(1, 500), _exec(2, 2900), _exec(3, 3300), _submit(3600)]


async def _drive(cls, monkeypatch, **kw):
    async with FakeSessionServer(_script()) as server:
        metrics = harness.legacy.AgentMetrics()
        async with _bridge(server, metrics, cls=cls, **kw) as bridge, aiohttp.ClientSession() as http:
            codex = FakeCodex(bridge, http)
            await codex.turn()
            for out in ('{"o":1}', '{"o":2}', '{"o":3}'):
                await codex.turn(out)
        return [(s, b) for s, b, _h in server.requests], codex.raw, asdict(metrics)


def test_off_stock_bridge_unchanged_and_compactionrl_without_trigger_is_byte_identical(monkeypatch):
    _env(monkeypatch, on=False)
    stock_requests, stock_raw, stock_metrics = _run(_drive(harness._ResponsesBridge, monkeypatch))
    _env(monkeypatch, on=True, t_comp=10)  # never triggers below C-10
    rl_requests, rl_raw, rl_metrics = _run(_drive(cb.CompactionRLBridge, monkeypatch,
                                                  config=cb.compaction_config(C, {cb.T_COMP_ENV: "10"})))
    stock_metrics.pop("total_generation_time"), rl_metrics.pop("total_generation_time")  # wall clock
    assert rl_requests == stock_requests and rl_raw == stock_raw and rl_metrics == stock_metrics


def test_off_drive_untrusted_trusted_side_and_metrics_are_stock(monkeypatch):
    _env(monkeypatch, on=False)
    calls = []

    async def fake_drive(*args, **kwargs):
        calls.append((args[1], args[4], kwargs))
        return "completed"

    monkeypatch.setattr(harness, "_drive_codex", fake_drive)
    job = {"base_url": "http://r/sessions/S0", "prompt": "p", "request_kwargs": {"temperature": 1.0},
           "episode_id": "ep-1", "max_seq_len": C}
    before = copy.deepcopy(job)
    assert _run(adapter.prepare_segment_sessions(job, post=None)) == {} and job == before
    result = _run(adapter.drive_untrusted(job, SimpleNamespace(), binary=harness.Path("/codex")))
    assert calls == [("http://r/sessions/S0", {"temperature": 1.0}, {"max_seq_len": C})]
    metrics = harness.legacy.AgentMetrics()
    assert set(result["metrics"]) == set(asdict(metrics)) | set(harness.tito_counters(metrics))
    with pytest.raises(harness.CodexHarnessError, match="without CompactionRL"):
        _run(adapter.drive_untrusted({**job, cb.SEGMENT_URLS_KEY: ["u"] * 3}, SimpleNamespace(),
                                     binary=harness.Path("/codex")))


def test_on_trusted_side_precreates_sessions_and_worker_requires_them(monkeypatch):
    _env(monkeypatch)
    posts = []

    async def post(url, body):
        posts.append((url, body))
        return {"session_id": f"seg{len(posts)}"}

    job = {"base_url": "http://r:1/sessions/S0", "prompt": "p",
           "request_kwargs": {"temperature": 0.7, "top_p": 1, "top_k": 20, "max_tokens": 9}, "episode_id": "e",
           "max_seq_len": C}
    meta = _run(adapter.prepare_segment_sessions(job, post=post))
    assert meta == {cb.SESSIONS_METADATA_KEY: ["seg1", "seg2", "seg3"], cb.SESSIONS_ROUTER_METADATA_KEY: "http://r:1"}
    assert cb.SESSIONS_ROUTER_METADATA_KEY == generate_wrapper.ROUTER_KEY
    assert job[cb.SEGMENT_URLS_KEY] == [f"http://r:1/sessions/seg{i}" for i in (1, 2, 3)]
    assert posts == [("http://r:1/sessions", {"evaluation": False, "temperature": 0.7, "top_p": 1.0, "top_k": 20})] * 3
    with pytest.raises(harness.CodexHarnessError, match="max_seq_len"):
        _run(adapter.prepare_segment_sessions({**job, "max_seq_len": None}, post=post))
    with pytest.raises(harness.CodexHarnessError, match="no pre-created segment sessions"):
        _run(adapter.drive_untrusted({k: v for k, v in job.items() if k != cb.SEGMENT_URLS_KEY},
                                     SimpleNamespace(), binary=harness.Path("/codex")))


# --- generate wrapper: one sample per segment -------------------------------

def _sample(n_prompt, n_gen, version="pv", **meta):
    tokens = list(range(n_prompt + n_gen))
    span = SimpleNamespace(version=version, abs_start=n_prompt, abs_end=n_prompt + n_gen)
    return SimpleNamespace(tokens=tokens, loss_mask=[1] * n_gen, rollout_log_probs=[-0.1] * n_gen,
                           response_length=n_gen, weight_versions=[SimpleNamespace(spans=[span])],
                           metadata=dict(meta), status=None)


def _seg0_meta(compactions, sessions=("a", "b", "c"), **extra):
    return {"expected_policy_version": "pv", "session_server_id": "10.0.0.1:9", "exit_status": "completed",
            "tbench_trusted_outcome": {"reward": 1.0}, "tbench_trusted_outcome_hmac": "mac",
            "chain_index": 0, "chains_total": 1, "segment_id": 0, "chain_break_reason": None,
            generate_wrapper.SESSIONS_KEY: list(sessions),
            "agent_metrics": {cb.METRICS_KEY: {"compactions": compactions}}, **extra}


@dataclass(frozen=True)
class _Out:
    samples: Any


def _wrap(seg0, collected, *, raise_on=None):
    inp = SimpleNamespace(sample=SimpleNamespace(group_index=7, index=11, metadata={"expected_policy_version": "pv"}))
    seen = []

    async def upstream(_input):
        return _Out(samples=seg0)

    async def collect(_input, router, sid):
        seen.append((router, sid))
        if sid == raise_on:
            raise TimeoutError("boom")
        return collected.get(sid, ([], {}))

    return _run(generate_wrapper.generate(inp, upstream=upstream, collect=collect)), seen


def test_wrapper_splits_segments_with_metadata_and_shared_reward():
    seg0 = _sample(10, 30, **_seg0_meta(2))
    s1, s2 = _sample(8, 20), _sample(8, 5)
    out, seen = _wrap(seg0, {"a": ([s1], {"sm": 1}), "b": ([s2], {})})
    assert seen == [("http://10.0.0.1:9", "a"), ("http://10.0.0.1:9", "b"), ("http://10.0.0.1:9", "c")]
    assert out.samples == [seg0, s1, s2] and all(s.status is None for s in out.samples)
    md = [s.metadata for s in out.samples]
    assert [m["segment_index"] for m in md] == [0, 1, 2] and {m["num_segments"] for m in md} == {3}
    assert [m["segment_tokens"] for m in md] == [30, 20, 5]
    assert [m["tokens_after"] for m in md] == [25, 5, 0] and {m["gae_length"] for m in md} == {55}
    assert [m["chain_index"] for m in md] == [0, 1, 2] and {m["chains_total"] for m in md} == {3}
    assert [m["chain_break_reason"] for m in md] == [None, "compaction_window", "compaction_window"]
    assert {json.dumps(m["tbench_trusted_outcome"]) for m in md} == {'{"reward": 1.0}'}
    assert {s.rollout_id for s in out.samples} == {11} and {s.group_index for s in out.samples} == {7}
    assert md[1]["sm"] == 1 and all(m["truncated"] is False and m["compactions"] == 2 for m in md)


def test_wrapper_no_compaction_keeps_single_sample_and_drains_sessions():
    seg0 = _sample(10, 30, **_seg0_meta(0, exit_status="max_seq_len"))
    out, seen = _wrap(seg0, {})
    assert out.samples is seg0 and len(seen) == 3
    assert seg0.metadata["num_segments"] == 1 and seg0.metadata["tokens_after"] == 0
    assert seg0.metadata["truncated"] is True


@pytest.mark.parametrize("case", ["mismatch_used_empty", "mismatch_unused_nonempty", "collect_error", "no_record"])
def test_wrapper_aborts_on_inconsistent_segments(case):
    meta = _seg0_meta(1)
    collected = {"a": ([_sample(4, 4)], {})}
    raise_on = None
    if case == "mismatch_used_empty":
        collected = {}
    elif case == "mismatch_unused_nonempty":
        collected["b"] = ([_sample(4, 4)], {})
    elif case == "collect_error":
        raise_on = "a"
    else:
        meta["agent_metrics"] = {}
    seg0 = _sample(10, 30, **meta)
    out, seen = _wrap(seg0, collected, raise_on=raise_on)
    assert len(seen) == 3  # every pre-created session is still collected (deleted)
    assert out.samples is seg0 and seg0.status == "ABORTED"
    assert "tbench_trusted_outcome" not in seg0.metadata


def test_wrapper_infrastructure_failure_only_drains():
    seg0 = _sample(10, 30, **_seg0_meta(1, **{generate_wrapper.INFRASTRUCTURE_KEY: "x"}))
    out, seen = _wrap(seg0, {"a": ([_sample(4, 4)], {})})
    assert len(seen) == 3 and out.samples is seg0 and "segment_index" not in seg0.metadata


def test_wrapper_without_compaction_metadata_is_untouched():
    seg0 = _sample(10, 30, expected_policy_version="pv")
    before = copy.deepcopy(seg0.metadata)
    out, seen = _wrap(seg0, {})
    assert seen == [] and out.samples is seg0
    assert {k: v for k, v in seg0.metadata.items() if k not in before} == {
        "chain_index": 0, "chains_total": 1, "policy_versions_actual": ["pv"]}


def test_wrapper_collects_on_the_recorded_router_and_aborts_without_one():
    """s19-compaction-g1-20261010d: the returned metadata had no session_server_id;
    the collect URL was "http:///sessions/<id>" and retried forever."""
    meta = _seg0_meta(1, **{generate_wrapper.ROUTER_KEY: "http://10.0.0.2:7"})
    del meta["session_server_id"]
    seg0, s1 = _sample(10, 30, **meta), _sample(8, 20)
    out, seen = _wrap(seg0, {"a": ([s1], {})})
    assert {r for r, _ in seen} == {"http://10.0.0.2:7"} and [m.metadata["tokens_after"] for m in out.samples] == [20, 0]
    bare = _seg0_meta(1)
    del bare["session_server_id"]
    seg0 = _sample(10, 30, **bare)
    out, seen = _wrap(seg0, {"a": ([s1], {})})
    assert seen == [] and generate_wrapper.INFRASTRUCTURE_KEY in seg0.metadata


def test_wrapper_bounds_each_segment_collect(monkeypatch):
    """A hanging collect (Miles retries) aborts the rollout after the budget."""
    import asyncio

    monkeypatch.setenv(generate_wrapper.SEGMENT_COLLECT_TIMEOUT_ENV, "0.05")
    seg0 = _sample(10, 30, **_seg0_meta(1))
    inp = SimpleNamespace(sample=SimpleNamespace(group_index=7, index=11, metadata={"expected_policy_version": "pv"}))

    async def upstream(_input):
        return _Out(samples=seg0)

    async def hang(_input, _router, _sid):
        await asyncio.sleep(30)

    out = _run(generate_wrapper.generate(inp, upstream=upstream, collect=hang))
    assert "timed out" in seg0.metadata[generate_wrapper.INFRASTRUCTURE_KEY]
    assert generate_wrapper.segment_collect_timeout_s() == 0.05
    monkeypatch.setenv(generate_wrapper.SEGMENT_COLLECT_TIMEOUT_ENV, "bad")
    assert generate_wrapper.segment_collect_timeout_s() == generate_wrapper.DEFAULT_SEGMENT_COLLECT_TIMEOUT_S
