"""Context compaction and the agentic round's weight switch, abandon, infra retry and close (CPU).

Real pieces: ``AgenticRound`` (``run_agent_task``, ``finish_sample``, the drain
timers, detach, infra retry, close), ``ProxyTokenRecorder`` serving a compacting
``RecordsProxy`` over HTTP on 127.0.0.1, ``FileSampleBuffer``. Fakes: the engine
(a scripted ``FakeEngine``; ``LiveEngine`` stamps each completion with the
versions served at its start and end) and the agent (three Codex-like Responses
turns; the third one triggers the compaction, so the proxy samples a summary turn
and then the action turn of the next segment inside that one request).
Tokenizer: GLM-4.7-Flash (the ``glm5`` template keeps history reasoning, so the
summary request extends the recorded stream) from the local HF cache.

What a compacted trial does at a weight switch, in short:

* drain cut / detach / close while the trial is in its summary turn: the trial is
  lost (``drain_timeout`` / ``closed``) with every segment; its recorder state is
  forgotten, and the proxy request still in the engine samples nothing more;
* a reload inside the summary turn (a forced load in ``drain``): the proxy refuses
  the turn (409) and records nothing of it; the trial never trains on it;
* ``live``: each segment's versions are its own turns' (the summary turn belongs
  to the segment it ends);
* infra retry still means "no model call recorded": a trial that failed after its
  summary turn is not run again.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import gc
import logging
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import pytest
from test_records_proxy import SCRIPT, ScriptedModel, _template, assistant, echo, engine_for, responses_body
from test_records_proxy import run_codex, tool_output, user_item

from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.agentic import TrialVerdict
from yeto.rl.compaction import CompactionConfig
from yeto.rl.engine.agentic_round import (CLOSE_WAIT_S, DRAIN_EVENT, GROUP_EVENT, LOST_DRAIN, LOST_VERSION,
                                          RETRY_EVENT, AgenticRound, LoopThread, infra_failure)
from yeto.rl.engine.buffer_payload import GroupWriter, decode_group
from yeto.rl.engine.engine_sampler import PromptItem
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_engine import Provenance, SamplingParams
from yeto.rl.harness.gateway.proxy import RecordsProxy
from yeto.rl.harness.gateway.recorder import ProxyTokenRecorder

aiohttp = pytest.importorskip("aiohttp")

SIGNED = {"max_tokens": 256, "temperature": 1.0, "top_p": 1.0}
TASK = {"instance_id": "fix-git", "agent_name": "codex"}
SUMMARY = assistant("check", content="<summary>1. listed files</summary>")


@pytest.fixture(scope="module")
def glm():
    return _template("glm5")


@pytest.fixture(scope="module")
def budget(glm):
    """A context budget the third request of a trial (after two tool steps) reaches, the second not."""
    dry = engine_for(ScriptedModel(glm, SCRIPT))
    run_codex(RecordsProxy(dry, glm, model="policy", signed_sampling=SIGNED), "dry")
    return len(dry.token_requests[2][0][0])


class Model:
    """``token_reply``: per trial the scripted replies SCRIPT[0], SCRIPT[1], the summary,
    SCRIPT[2]. The summary call blocks until ``release`` is set (set by default) and
    runs ``during_summary`` (a reload) before it replies."""

    def __init__(self, template, *, trials: int = 1, summaries: int = 1) -> None:
        self.scripted = ScriptedModel(template, [SCRIPT[0], SCRIPT[1], *[SUMMARY] * summaries, SCRIPT[2]] * trials)
        self.summary_started = threading.Event()
        self.summary_done = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.during_summary = lambda: None

    def __call__(self, prompt, index):
        summary = self.scripted.replies[0] is SUMMARY
        if not summary:
            return self.scripted(prompt, index)
        self.summary_started.set()
        assert self.release.wait(30), "the summary call was never released"
        self.during_summary()
        try:
            return self.scripted(prompt, index)
        finally:
            self.summary_done.set()


class LiveEngine(FakeEngine):
    """An engine that updates during a generation: each completion carries the versions
    served when it started (its first token) and when it ended."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.updates_during_generation = True

    def generate_tokens(self, prompts, sampling, *, stop_token_ids=()):
        first = self.served_version()
        groups = super().generate_tokens(prompts, sampling, stop_token_ids=stop_token_ids)
        last = self.served_version()
        return [[dataclasses.replace(c, weight_version=first, final_weight_version=last) for c in g] for g in groups]


class Agent:
    """Three Responses turns against the recorder URL (the Codex loop of
    ``test_records_proxy``), reward 1. ``retry_refused``: send a request the proxy
    refused once more; ``stubborn``: a model call ignores cancellation (blocked in
    a sandbox); ``fail_first``: that many trials fail before any model call;
    ``fail_after``: a trial fails after that many model requests."""

    def __init__(self, *, retry_refused: bool = False, stubborn: bool = False, fail_first: int = 0,
                 fail_after: int | None = None) -> None:
        self.retry_refused = retry_refused
        self.stubborn = stubborn
        self.fail_first = fail_first
        self.fail_after = fail_after
        self.statuses: list[int] = []
        self.cancels = 0
        self.ended = threading.Event()

    async def run(self, task, endpoint):
        try:
            if self.fail_first > 0:
                self.fail_first -= 1
                raise RuntimeError("sandbox lost")  # infrastructure, before the policy ran
            items = [user_item("read a.txt")]
            async with aiohttp.ClientSession() as http:
                for turn in range(3):
                    response = await self._post(http, f"{endpoint}/responses", items)
                    items += echo(response["output"])
                    items += [tool_output(o["call_id"], f"result {turn}")
                              for o in response["output"] if o["type"] == "function_call"]
                    if self.fail_after is not None and turn + 1 >= self.fail_after:
                        raise RuntimeError("sandbox lost mid-trial")  # infrastructure, after model calls
            return TrialVerdict(rewards={"reward": 1.0})
        finally:
            self.ended.set()

    async def _post(self, http, url, items):
        for attempt in range(2):
            status, body = await self._call(http, url, responses_body(copy.deepcopy(items)))
            self.statuses.append(status)
            if status == 200:
                return body
            if not self.retry_refused or attempt:
                raise RuntimeError(f"model call refused ({status}): {body}")
        raise AssertionError("unreachable")

    async def _call(self, http, url, body):
        async def call():
            async with http.post(url, json=body) as reply:
                return reply.status, (await reply.json() if reply.status == 200 else await reply.text())

        if not self.stubborn:
            return await call()
        request = asyncio.ensure_future(call())
        while True:
            try:
                return await asyncio.shield(request)
            except asyncio.CancelledError:
                self.cancels += 1
                if request.cancelled():  # cancelled itself: awaiting it again raises at once, so yield
                    await asyncio.sleep(0.01)


class World:
    """A compacting records proxy and its recorder over ``engine``, with what the
    trials' closes read (``closed``) and how each backend chat call ended (``chats``)."""

    def __init__(self, template, engine, budget: int) -> None:
        self.engine = engine
        cfg = CompactionConfig(context_budget=budget, t_comp=1, keep_recent_steps=1, max_compactions=1)
        self.proxy = RecordsProxy(engine, template, model="policy", signed_sampling=SIGNED, sampler_id="modal-0",
                                  compaction=cfg)
        self.recorder = ProxyTokenRecorder(self.proxy, allow_compaction=True)
        self.closed: list[dict[str, Any]] = []
        self.chats: list[str] = []
        records, chat = self.proxy.records, self.proxy.backend.chat

        def read(trajectory_id):
            out = records(trajectory_id)
            self.closed.append(copy.deepcopy({k: v for k, v in out.items() if k != "gateway"}))
            return out

        async def traced(session_id, messages, sampling):
            try:
                reply = await chat(session_id, messages, sampling)
            except BaseException as exc:
                self.chats.append(type(exc).__name__)
                raise
            self.chats.append("ok")
            return reply

        self.proxy.records = read
        self.proxy.backend.chat = traced

    def held(self) -> dict[str, int]:
        """What the proxy and the recorder still hold per trajectory (all 0 once every trial ended)."""
        backend, gateway = self.proxy.backend, self.proxy.gateway
        return {"sessions": len(backend.sessions), "tools": len(backend.tools), "counters": len(backend.counters),
                "chains": len(gateway.registries), "invalidated": len(gateway.invalidated),
                "tokens": len(self.proxy._tokens), "open": len(self.recorder._open)}

    def engine_calls(self) -> int:
        return len(self.engine.token_requests)


EMPTY = {"sessions": 0, "tools": 0, "counters": 0, "chains": 0, "invalidated": 0, "tokens": 0, "open": 0}


def _world(glm, budget, model: Model, *, live: bool = False) -> World:
    engine = (LiveEngine if live else FakeEngine)(token_reply=model, max_model_len=8192)
    engine.start(None, 0)
    return World(glm, engine, budget)


def _round(tmp_path: Path, world: World, agent: Agent, *, n: int = 1, max_in_flight: int = 2, live: bool = False,
           rounds: tuple[int, ...] = (0,), **kw):
    events: list[tuple[str, dict]] = []
    sampling = SamplingParams(max_tokens=8, n=n)

    def prompts(rollout_id):
        if rollout_id not in rounds:
            return []
        return [PromptItem(f"p{rollout_id}", f"task {rollout_id}", metadata={"agent_task": dict(TASK)})]

    rnd = AgenticRound(world.engine, GroupWriter(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1), "s"),
                       recorder=world.recorder, harness=agent, max_in_flight=max_in_flight,
                       max_response_len=4096, max_prompt_len=4096, min_samples=1,
                       task_defaults={"provider": "docker"}, prompts=prompts, sampling=sampling,
                       provenance=Provenance.of(world.engine.identity, sampler_id="s", sampling=sampling),
                       emit=lambda name, **f: events.append((name, f)),
                       weight_switch="live" if live else "drain", release_poll_s=0.0, wait_s=0.005, **kw)
    return rnd, events


def _samples(tmp_path: Path) -> list[dict[str, Any]]:
    reader = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    reader.refresh()
    entries = [x.entry for x in reader.select("t", 100)]
    return [s for e in entries for s in decode_group(reader.read_payload(e.payload_ref))]


def _wait(condition, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def _segments(records: dict[str, Any]) -> list[list[tuple[str, int | None]]]:
    """(kind, policy version) of each turn, per segment, of one trajectory's records."""
    return [[(t["kind"], t["policy_version"]) for t in sample["turns"]]
            for session in records["sessions"] for sample in session["samples"]]


def _events(events, name):
    return [f for n, f in events if n == name]


# ---------------------------------------------------------------------- 1. drain while in the summary turn
def test_a_drain_cut_during_the_summary_turn_loses_the_whole_trial_and_forgets_it(tmp_path, glm, budget):
    """The cut cancels the trial while the proxy samples its summary: the trial is lost as
    drain_timeout with every segment (no verifier reward, so no segment can be trained), its
    recorder state is forgotten, and the proxy request still in the engine (aiohttp does not
    cancel a handler whose client left) samples nothing more after the loop's load."""
    model = Model(glm)
    model.release.clear()
    world = _world(glm, budget, model)
    pending = {"seq": None}
    rnd, events = _round(tmp_path, world, Agent(), drain_timeout_s=0.05)
    rnd.release_pending = lambda: pending["seq"]
    try:
        rnd(0)
        _wait(model.summary_started.is_set)
        assert world.engine_calls() == 3  # two action turns, then the summary turn (in the engine)
        pending["seq"] = 3
        rnd(1)  # the drain: cut at 0.05 s, the cancelled trial ends, the drain ends
        (drain,) = _events(events, DRAIN_EVENT)
        assert drain["timed_out"] and drain["abandoned_trials"] == 1 and drain["detached_trials"] == 0
        assert not rnd.holds_weights() and world.held() == EMPTY
        # what the cancelled trial's close read: segment 0 without the summary still being sampled
        (records,) = world.closed
        assert _segments(records) == [[("action", 0), ("action", 0)]]
        world.engine.load(Path("r1"), 1)  # the loop loads the release
        model.release.set()
        _wait(lambda: len(world.chats) == 3)
        assert world.chats[-1] == "RecordsMismatch"  # served version changed during the summary turn
        assert world.engine_calls() == 3 and world.held() == EMPTY  # no action turn on the new weights
        rnd.flush()
    finally:
        model.release.set()
        rnd.close()
    (group,) = _events(events, GROUP_EVENT)
    assert not group["written"] and group["lost"] == {LOST_DRAIN: 1} and group["dropped_reason"] == "below_min_samples"
    assert _samples(tmp_path) == []


def test_a_cut_trial_whose_summary_ends_on_the_same_weights_samples_nothing_more(tmp_path, glm, budget):
    """As above, but the summary ends before any load: the forgotten trajectory's request
    still gets no next-segment turn from the engine and recreates no state."""
    model = Model(glm)
    model.release.clear()
    world = _world(glm, budget, model)
    pending = {"seq": None}
    rnd, events = _round(tmp_path, world, Agent(), drain_timeout_s=0.05)
    rnd.release_pending = lambda: pending["seq"]
    try:
        rnd(0)
        _wait(model.summary_started.is_set)
        pending["seq"] = 3
        rnd(1)
        model.release.set()
        _wait(lambda: len(world.chats) == 3)
        # the request fails (today a KeyError of the forgotten counters in _compact, by accident)
        assert world.chats[-1] != "ok"
        assert world.engine_calls() == 3 and world.held() == EMPTY
    finally:
        model.release.set()
        rnd.close()
    assert _events(events, GROUP_EVENT)[0]["lost"] == {LOST_DRAIN: 1}


def test_a_trial_stuck_in_its_summary_turn_is_detached_and_forgotten(tmp_path, glm, budget):
    """A trial that ignores the cut (its model call is blocked in a sandbox) is detached
    after abandon_grace_s: counted lost, its recorder session closed and forgotten while the
    summary is still in the engine; its late end is dropped."""
    model = Model(glm)
    model.release.clear()
    world = _world(glm, budget, model)
    agent = Agent(stubborn=True)
    pending = {"seq": None}
    rnd, events = _round(tmp_path, world, agent, drain_timeout_s=0.05, abandon_grace_s=0.05)
    rnd.release_pending = lambda: pending["seq"]
    try:
        rnd(0)
        _wait(model.summary_started.is_set)
        pending["seq"] = 3
        rnd(1)  # returns although the trial never honours the cancel
        (drain,) = _events(events, DRAIN_EVENT)
        assert drain["timed_out"] and drain["detached_trials"] == 1 and agent.cancels >= 1
        _wait(lambda: world.held() == EMPTY)  # the detach closed the session (a task of the loop)
        assert _segments(world.closed[0]) == [[("action", 0), ("action", 0)]]
        world.engine.load(Path("r1"), 1)
        model.release.set()
        _wait(lambda: agent.ended.is_set() and not rnd._detached)
        assert world.chats[-1] == "RecordsMismatch" and agent.statuses[-1] == 409
        assert world.engine_calls() == 3 and world.held() == EMPTY and rnd.detached_total == 1
        rnd.flush()
    finally:
        model.release.set()
        rnd.close()
    (group,) = _events(events, GROUP_EVENT)
    assert not group["written"] and group["lost"] == {LOST_DRAIN: 1}
    assert _samples(tmp_path) == []


def test_a_forced_load_inside_the_summary_turn_is_refused_and_the_trial_not_trained(tmp_path, glm, budget):
    """drain, no cut yet: the follower forces a load (reload_max_defer_s) while the proxy
    samples the summary. The proxy refuses the turn (409) and records nothing of it; an
    agent that gives up is ABORTED after model calls, so it is lost and not retried."""
    model = Model(glm)
    world = _world(glm, budget, model)
    model.during_summary = lambda: world.engine.load(Path("r1"), 1)
    rnd, events = _round(tmp_path, world, Agent(), infra_retries=2)
    try:
        rnd(0)
        rnd.flush()
    finally:
        rnd.close()
    assert world.chats == ["ok", "ok", "RecordsMismatch"]
    (records,) = world.closed
    assert _segments(records) == [[("action", 0), ("action", 0)]]  # no summary turn, no second segment
    assert "compactions" not in records["counters"]
    (group,) = _events(events, GROUP_EVENT)
    assert not group["written"] and group["lost"] == {"RuntimeError": 1} and group["retried_trials"] == 0
    assert not _events(events, RETRY_EVENT) and world.held() == EMPTY


def test_an_agent_that_resends_the_refused_request_compacts_on_the_new_weights_and_is_lost_in_drain(
        tmp_path, glm, budget):
    """The agent sends the refused request again: the proxy compacts again, now on v1, so
    the trial's turns span two versions and drain mode leaves it out of its group."""
    model = Model(glm, summaries=2)  # the refused summary consumes its scripted reply
    world = _world(glm, budget, model)
    loads = iter([lambda: world.engine.load(Path("r1"), 1)])
    model.during_summary = lambda: next(loads, lambda: None)()
    rnd, events = _round(tmp_path, world, Agent(retry_refused=True))
    try:
        rnd(0)
        rnd.flush()
    finally:
        rnd.close()
    assert world.chats == ["ok", "ok", "RecordsMismatch", "ok"]
    (records,) = world.closed
    assert _segments(records) == [[("action", 0), ("action", 0), ("summary", 1)], [("action", 1)]]
    (group,) = _events(events, GROUP_EVENT)
    assert not group["written"] and group["kept"] == 0
    assert group["lost"] == {LOST_VERSION: 1}  # the version rule drops the trial
    assert world.held() == EMPTY


# ---------------------------------------------------------------------- 2. live: versions per segment
def test_live_records_each_segment_with_its_own_turn_versions(glm, budget):
    """A live reload inside the summary turn: the summary keeps its first token's version
    and stays in the segment it ends; the next segment's turns are of the new version."""
    model = Model(glm)
    world = _world(glm, budget, model, live=True)
    model.during_summary = lambda: world.engine.load(Path("r1"), 1)
    asyncio.run(_drive(world))
    (records,) = world.closed
    assert _segments(records) == [[("action", 0), ("action", 0), ("summary", 0)], [("action", 1)]]


async def _drive(world: World) -> None:
    endpoint = await world.recorder.open("t", {"temperature": 1.0, "top_p": 1.0})
    try:
        await Agent().run(None, endpoint)
        await world.recorder.close("t")
    finally:
        await world.recorder.aclose()


def test_live_writes_each_segment_at_its_own_oldest_version(tmp_path, glm, budget):
    """live round, reload inside the summary turn: segment 0 is v0 (its turns and the
    trial's start), segment 1 is v1 (its own turns only, not the trial's start version)."""
    model = Model(glm)
    world = _world(glm, budget, model, live=True)
    model.during_summary = lambda: world.engine.load(Path("r1"), 1)
    rnd, events = _round(tmp_path, world, Agent(), live=True)
    try:
        rnd(0)
        rnd.flush()
    finally:
        rnd.close()
    samples = sorted(_samples(tmp_path), key=lambda s: s["metadata"]["compaction"]["segment_index"])
    assert [s["metadata"]["weight_version"] for s in samples] == ["0", "1"]
    assert samples[1]["metadata"]["final_weight_version"] == "1"
    assert {s["index"] for s in samples} == {0} and {s["reward"] for s in samples} == {1.0}
    (group,) = _events(events, GROUP_EVENT)
    assert group["written"] and group["versions"] == [0, 1]
    assert world.held() == EMPTY


# ---------------------------------------------------------------------- 3. memory: every segment forgotten
def test_finished_compacted_trials_leave_nothing_in_the_recorder(tmp_path, glm, budget):
    model = Model(glm, trials=4)
    world = _world(glm, budget, model)
    rnd, events = _round(tmp_path, world, Agent(), n=2, max_in_flight=1, rounds=(0, 1))
    try:
        rnd(0)
        rnd(1)
        rnd.flush()
    finally:
        rnd.close()
    assert len(world.closed) == 4 and all(len(_segments(r)) == 2 for r in world.closed)
    assert all(r["counters"]["compactions"] == 1 for r in world.closed)
    assert world.held() == EMPTY
    assert len(_samples(tmp_path)) == 8  # two groups of two trials, two segments each


def test_closing_the_island_during_a_summary_turn_forgets_the_trial(tmp_path, glm, budget, caplog):
    """No suspension on this path: a sampler that stops (or restarts) with a trial in its
    summary turn loses it with every segment; nothing of it is left in the recorder. Nor on
    the round's loop: the recorder request still in the engine is cancelled, the recorder's
    server stopped and the loop closed, so the garbage collector, in whatever test runs
    next, finds no pending task ("Task was destroyed but it is pending!")."""
    model = Model(glm)
    model.release.clear()
    world = _world(glm, budget, model)
    rnd, events = _round(tmp_path, world, Agent())
    try:
        rnd(0)
        _wait(model.summary_started.is_set)
    finally:
        rnd.close()  # cancels the trial and waits up to CLOSE_WAIT_S for it
        model.release.set()
    assert world.held() == EMPTY and _samples(tmp_path) == []
    assert not _events(events, GROUP_EVENT)  # never settled: its round is not in the buffer
    assert rnd.loop.loop.is_closed() and world.recorder._runner is None
    caplog.set_level(logging.DEBUG, logger="asyncio")
    del rnd, world
    gc.collect()
    assert [r.getMessage() for r in caplog.records if r.name == "asyncio"] == []


class CleanupAgent(Agent):
    """An agent whose cancelled trial still awaits its own cleanup (a sandbox teardown)
    on the round's loop before it ends."""

    def __init__(self) -> None:
        super().__init__()
        self.cleaned = threading.Event()

    async def run(self, task, endpoint):
        try:
            return await super().run(task, endpoint)
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)  # on the loop the close is about to close
            self.cleaned.set()
            raise


def _listening(url: str) -> bool:
    import socket
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    with socket.socket() as sock:
        sock.settimeout(2)
        return sock.connect_ex((parts.hostname, parts.port)) == 0


def test_close_after_finished_rounds_stops_the_recorder_server_and_closes_the_loop(tmp_path, glm, budget, caplog):
    model = Model(glm)
    world = _world(glm, budget, model)
    rnd, events = _round(tmp_path, world, Agent())
    try:
        rnd(0)
        rnd.flush()
        url = world.recorder._base_url
        assert url and _listening(url) and not rnd.loop.loop.is_closed()  # running: nothing closed early
    finally:
        deleted = rnd.close()
    assert deleted == 0 and len(_samples(tmp_path)) == 2  # the finished trial's two segments were written
    assert rnd.loop.loop.is_closed() and world.recorder._runner is None and not _listening(url)
    assert [f["trials_total"] for f in _events(events, "rl_agentic_close")] == [1]
    caplog.set_level(logging.DEBUG, logger="asyncio")
    del rnd, world
    gc.collect()
    assert [r.getMessage() for r in caplog.records if r.name == "asyncio"] == []


def test_close_waits_for_a_cancelled_trials_cleanup_before_it_closes_the_loop(tmp_path, glm, budget):
    model = Model(glm)
    model.release.clear()
    world = _world(glm, budget, model)
    agent = CleanupAgent()
    rnd, _ = _round(tmp_path, world, agent)
    try:
        rnd(0)
        _wait(model.summary_started.is_set)
    finally:
        rnd.close()
        model.release.set()
    assert agent.cleaned.is_set()  # its cleanup ran to the end on the loop before the loop closed
    assert rnd.loop.loop.is_closed() and world.held() == EMPTY


def test_close_closes_the_loop_when_stopping_the_recorder_or_deleting_the_pools_fails(tmp_path, glm, budget,
                                                                                       monkeypatch):
    from yeto.rl.harness.harbor import boat_pool

    world = _world(glm, budget, Model(glm))
    rnd, events = _round(tmp_path, world, Agent())
    rnd(0)
    rnd.flush()

    async def broken_aclose():
        raise OSError("the server did not stop")

    monkeypatch.setattr(world.recorder, "aclose", broken_aclose)
    assert rnd.close() == 0  # the recorder's failure does not fail the close
    assert rnd.loop.loop.is_closed()

    world = _world(glm, budget, Model(glm))
    rnd, events = _round(tmp_path / "b", world, Agent())
    rnd(0)
    rnd.flush()

    def broken_pools():
        raise RuntimeError("boat API down")

    monkeypatch.setattr(boat_pool, "close_pools_now", broken_pools)
    with pytest.raises(RuntimeError, match="boat API down"):  # as before: the pool error is the caller's
        rnd.close()
    assert rnd.loop.loop.is_closed() and world.recorder._runner is None


def test_a_second_close_is_safe_and_changes_nothing(tmp_path, glm, budget, monkeypatch):
    from yeto.rl.harness.harbor import boat_pool

    deletes = []
    monkeypatch.setattr(boat_pool, "close_pools_now", lambda: deletes.append("pools") or 3)
    world = _world(glm, budget, Model(glm))
    rnd, events = _round(tmp_path, world, Agent())
    rnd(0)
    rnd.flush()
    assert rnd.close() == 3
    gc.collect()  # what earlier tests left for the collector is not this close's
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert rnd.close() == 0  # the loop is closed already: nothing to cancel, stop or close
        gc.collect()
    assert [str(w.message) for w in caught] == []  # no coroutine handed to the closed loop
    assert deletes == ["pools"]  # the pooled sandboxes are deleted once
    assert [f["trials_total"] for f in _events(events, "rl_agentic_close")] == [1]
    assert rnd.loop.loop.is_closed() and world.recorder._runner is None
    assert len(_samples(tmp_path)) == 2


def test_closing_the_loop_thread_cancels_its_tasks_and_closes_the_loop(caplog):
    loop = LoopThread(workers=1)
    started = threading.Event()

    async def pending():
        started.set()
        await asyncio.sleep(3600)

    task = asyncio.run_coroutine_threadsafe(pending(), loop.loop)
    assert started.wait(10)
    loop.close()
    assert loop.loop.is_closed() and task.cancelled()
    loop.close()  # a second close does nothing
    caplog.set_level(logging.DEBUG, logger="asyncio")
    del loop
    gc.collect()
    assert [r.getMessage() for r in caplog.records if r.name == "asyncio"] == []



class ShieldedCleanupAgent(Agent):
    """A cancelled trial's two-step cleanup under ``asyncio.shield``, as a sandbox close: a
    slow read in a thread, then the delete in a thread; it outlives the trials' grace."""

    def __init__(self, read_s: float) -> None:
        super().__init__()
        self.read_s = read_s
        self.deleted = threading.Event()

    async def run(self, task, endpoint):
        try:
            return await super().run(task, endpoint)
        except asyncio.CancelledError:
            async def read_and_delete():
                await asyncio.to_thread(time.sleep, self.read_s)
                await asyncio.to_thread(self.deleted.set)

            await asyncio.shield(read_and_delete())
            raise


def test_a_shielded_cleanup_that_outlives_the_grace_still_runs_while_the_pools_are_deleted(tmp_path, glm, budget,
                                                                                           monkeypatch):
    """The loop runs, and nothing on it is cancelled, while close_pools_now deletes the
    pooled sandboxes (here 1.5 s): a shielded delete that needs 2.5 s after the cancel ends."""
    from yeto.rl.harness.harbor import boat_pool

    pools = []
    monkeypatch.setattr(boat_pool, "close_pools_now", lambda: pools.append(time.monotonic()) or time.sleep(1.5) or 0)
    model = Model(glm)
    model.release.clear()
    world = _world(glm, budget, model)
    agent = ShieldedCleanupAgent(read_s=2.5)
    rnd, events = _round(tmp_path, world, agent)
    try:
        rnd(0)
        _wait(model.summary_started.is_set)
    finally:
        start = time.monotonic()
        rnd.close()
        model.release.set()
    assert agent.deleted.is_set(), "the shielded delete was cancelled before it ran"
    assert pools[0] - start < CLOSE_WAIT_S + 1.0  # the pools are deleted right after the trials' grace
    assert rnd.loop.loop.is_closed() and world.recorder._runner is None
    assert len(_events(events, "rl_agentic_close")) == 1


def test_a_trial_that_ignores_its_cancel_ends_with_the_recorder_and_the_close_stays_bounded(tmp_path, glm, budget):
    """A trial whose model call ignores cancellation (blocked in the engine) at close: the
    recorder's server cancels the request at the end of the trials' grace, so the trial ends
    then; the close returns soon after, the loop's thread exits and no CPU is spent spinning."""
    model = Model(glm)
    model.release.clear()
    world = _world(glm, budget, model)
    agent = Agent(stubborn=True)
    rnd, events = _round(tmp_path, world, agent)
    rnd(0)
    _wait(model.summary_started.is_set)
    start, cpu = time.monotonic(), time.process_time()
    try:
        rnd.close()
        took, spent = time.monotonic() - start, time.process_time() - cpu
    finally:
        model.release.set()
    assert agent.cancels >= 1 and agent.ended.is_set()  # it ignored the cancel; the request's end ended it
    assert took < 2 * CLOSE_WAIT_S, took
    assert spent < CLOSE_WAIT_S, spent  # waited, did not spin
    assert not rnd.loop._thread.is_alive() and rnd.loop.loop.is_closed()
    assert len(_events(events, "rl_agentic_close")) == 1


def test_a_trial_in_flight_at_close_is_cancelled_before_the_recorder_stops(tmp_path, glm, budget):
    """The trials are cancelled first, and only then is the recorder stopped: the trial in its
    summary turn ends as closed, never as a failed model call (no retry, no loss reported)."""
    model = Model(glm)
    model.release.clear()
    world = _world(glm, budget, model)
    agent = Agent()
    errors = []
    run = agent.run

    async def recorded(task, endpoint):
        try:
            return await run(task, endpoint)
        except BaseException as error:
            errors.append(type(error).__name__)
            raise

    agent.run = recorded
    rnd, events = _round(tmp_path, world, agent)
    try:
        rnd(0)
        _wait(model.summary_started.is_set)
    finally:
        rnd.close()
        model.release.set()
    assert errors == ["CancelledError"]
    assert not _events(events, RETRY_EVENT) and not _events(events, "rl_agentic_trial_lost")


def test_a_close_interrupted_while_deleting_the_pools_still_stops_the_recorder_and_the_loop(tmp_path, glm, budget,
                                                                                            monkeypatch):
    from yeto.rl.harness.harbor import boat_pool

    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(boat_pool, "close_pools_now", interrupt)
    world = _world(glm, budget, Model(glm))
    rnd, events = _round(tmp_path, world, Agent())
    rnd(0)
    rnd.flush()
    with pytest.raises(KeyboardInterrupt):
        rnd.close()
    assert rnd.loop.loop.is_closed() and world.recorder._runner is None
    assert not _events(events, "rl_agentic_close")
    assert rnd.close() == 0  # the close ran: a second one does nothing


def test_a_second_close_after_the_loops_thread_did_not_stop_returns_at_once(tmp_path, glm, budget, monkeypatch):
    """The loop's thread was blocked in a synchronous call at close, so the loop could not be
    closed; a second close does not hand anything to that stopped, unclosed loop."""
    import yeto.rl.engine.agentic_round as agentic_round

    monkeypatch.setattr(agentic_round, "LOOP_CLOSE_WAIT_S", 0.2)
    monkeypatch.setattr(agentic_round, "CLOSE_WAIT_S", 0.2)
    world = _world(glm, budget, Model(glm))
    rnd, events = _round(tmp_path, world, Agent())
    rnd(0)
    rnd.flush()
    gate = threading.Event()
    rnd.loop.loop.call_soon_threadsafe(lambda: gate.wait(40))
    first = threading.Thread(target=rnd.close, daemon=True)
    first.start()
    first.join(35)
    try:
        assert not first.is_alive(), "the first close hung"
        start = time.monotonic()
        assert rnd.close() == 0
        assert time.monotonic() - start < 0.5
    finally:
        gate.set()
    assert len(_events(events, "rl_agentic_close")) == 1


def test_closing_a_loop_thread_blocked_in_a_sync_call_returns_and_leaves_the_loop_open(monkeypatch):
    """A loop thread blocked in a synchronous call: close() gives up after its bounded waits
    (the cancel times out; the error does not escape) and does not close a loop that still
    runs (that raises); once the call returned, a second close closes it."""
    import yeto.rl.engine.agentic_round as agentic_round

    monkeypatch.setattr(agentic_round, "LOOP_CLOSE_WAIT_S", 0.2)
    loop = LoopThread(workers=1)
    gate = threading.Event()
    loop.loop.call_soon_threadsafe(lambda: gate.wait(40))
    try:
        loop.close()
        assert not loop.loop.is_closed() and loop._thread.is_alive()
    finally:
        gate.set()
    loop._thread.join(10)
    loop.close()
    assert loop.loop.is_closed()


def test_closing_the_loop_thread_waits_for_a_cancelled_tasks_cleanup_and_closes_async_generators():
    loop = LoopThread(workers=1)
    cleaned, generator_closed, started = threading.Event(), threading.Event(), threading.Event()

    async def task_with_cleanup():
        try:
            started.set()
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)  # its cleanup awaits on the loop
            cleaned.set()
            raise

    async def generator():
        try:
            yield 1
            yield 2
        finally:
            generator_closed.set()

    held = []

    async def start_generator():
        gen = generator()
        held.append(gen)
        await gen.__anext__()  # suspended at its first yield: only shutdown_asyncgens closes it

    loop.run(start_generator())
    asyncio.run_coroutine_threadsafe(task_with_cleanup(), loop.loop)
    assert started.wait(10)
    loop.close()
    assert cleaned.is_set() and generator_closed.is_set()
    assert loop.loop.is_closed()

def test_a_request_for_a_forgotten_trajectory_recreates_no_state(glm, budget):
    world = _world(glm, budget, Model(glm))
    world.proxy.open("t")
    world.proxy.forget("t")  # the detach / cancel path closed the session meanwhile
    with pytest.raises(Exception):
        asyncio.run(world.proxy.handle("responses", responses_body([user_item("read a.txt")]), trajectory_id="t"))
    assert world.held() == EMPTY and world.engine_calls() == 0


# ---------------------------------------------------------------------- 4. infra retry
def test_a_trial_that_failed_before_any_model_call_is_retried_with_compaction_on(tmp_path, glm, budget):
    model = Model(glm)
    world = _world(glm, budget, model)
    rnd, events = _round(tmp_path, world, Agent(fail_first=1), infra_retries=2)
    try:
        rnd(0)
        rnd.flush()
    finally:
        rnd.close()
    (retry,) = _events(events, RETRY_EVENT)
    assert retry["reason"] == "RuntimeError" and retry["attempt"] == 1
    assert len(world.closed) == 2 and _segments(world.closed[0]) == []  # the first attempt recorded no call
    assert len(_segments(world.closed[1])) == 2 and world.held() == EMPTY
    (group,) = _events(events, GROUP_EVENT)
    assert group["retried_trials"] == 1
    assert group["written"] and len(_samples(tmp_path)) == 2


def test_a_trial_that_failed_after_its_summary_turn_is_not_retried(tmp_path, glm, budget):
    model = Model(glm, trials=2)
    world = _world(glm, budget, model)
    rnd, events = _round(tmp_path, world, Agent(fail_after=3), infra_retries=2)
    try:
        rnd(0)
        rnd.flush()
    finally:
        rnd.close()
    assert not _events(events, RETRY_EVENT)
    (records,) = world.closed
    assert [k for seg in _segments(records) for k, _ in seg] == ["action", "action", "summary", "action"]
    (group,) = _events(events, GROUP_EVENT)
    assert group["lost"] == {"RuntimeError": 1} and group["retried_trials"] == 0
    assert world.held() == EMPTY


def test_infra_failure_reads_the_calls_of_every_segment():
    """``infra_failure`` asks whether any model call was recorded; a compacted trial's
    ``turns`` are the calls of all its segments (``close_segments`` parts keep the same ``turns``)."""
    from yeto.rl.agentic import AgentTask, AgentTrajectory, Outcome, TokenTurn
    from yeto.rl.agentic.finish import FinishedSample

    task = AgentTask.from_mapping(dict(TASK), defaults={"provider": "docker"})
    aborted = AgentTrajectory(task=task, outcome=Outcome.ABORTED, reward=0.0, reason="RuntimeError", verdict=None)
    turn = TokenTurn(prompt_ids=(1,), completion_ids=(2,), completion_logprobs=(-0.1,), policy_version=0)
    assert infra_failure(FinishedSample(aborted, None, ())) == "RuntimeError"
    assert infra_failure(FinishedSample(aborted, None, (turn,))) is None


# ---------------------------------------------------------------------- 6. drain settings
def test_compaction_with_a_drain_switch_is_warned():
    from yeto.rl.engine.agentic_round import drain_timeout_warning

    compaction = {"context_budget": 262144}
    within = {"reload_max_defer_s": 600, "agentic": {"weight_switch": "drain", "drain_timeout_s": 600}}
    assert drain_timeout_warning(within) is None
    warning = drain_timeout_warning({**within, "agentic": {**within["agentic"], "compaction": compaction}})
    assert warning is not None and "compaction" in warning and "600 s" in warning and "2400 s" in warning
    above = {"reload_max_defer_s": 300, "agentic": {"weight_switch": "drain", "drain_timeout_s": 3600,
                                                    "compaction": compaction}}
    both = drain_timeout_warning(above)
    assert "3600 s" in both and "compaction" in both
    # live, no cut at all, or compaction off: nothing more than before
    assert drain_timeout_warning({**above, "agentic": {**above["agentic"], "weight_switch": "live"}}) is None
    assert drain_timeout_warning({"reload_max_defer_s": None, "agentic": {"weight_switch": "drain",
                                                                          "compaction": compaction}}) is None
    assert drain_timeout_warning({**within, "agentic": {**within["agentic"], "compaction": None}}) is None
