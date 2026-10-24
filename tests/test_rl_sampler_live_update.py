"""Updates of the sampler weights during a generation (S27 R-3 follow-up).

The ``sglang-server`` adapter against a fake SGLang HTTP server
(:mod:`fake_sglang_server`) that models the update lock and pause/continue,
and the sampler loop's in-round refresh. CPU only."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from fake_sglang_server import TOKEN_BASE, FakeSGLang

from yeto.rl.adapters.sampler_engines.server import ServerConfig
from yeto.rl.adapters.sampler_engines.sglang_server import SGLangOptions, SGLangServerEngine
from yeto.rl.engine.engine_sampler import PromptItem, sample_record
from yeto.rl.engine.policy_release import FULL, PolicyFollower, PolicyReleaseStore, ReleaseNotLoaded
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sampler_engine import SamplerEngineError, SamplingParams, updates_during_generation
from yeto.rl.engine.sampler_loop import SamplerLoop

TOKENS = 150  # x 2 ms per token: a round of about 0.3 s


class _Process:
    def poll(self):
        return None

    def send_signal(self, sig):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


def _engine(tmp_path: Path, fake: FakeSGLang, *, live: bool, spawned: list | None = None) -> SGLangServerEngine:
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    config = ServerConfig(base_dir=base, card_model="H100", device_family="cuda", engine_version="0.5.20",
                          precision="bf16", quant_scheme="bf16", topology="tp1", max_model_len=4096,
                          port=fake.port, in_place_reload=True, request_timeout_s=30.0)
    spawn = (lambda a, e: spawned.append(a) or _Process()) if spawned is not None else (lambda a, e: _Process())
    return SGLangServerEngine(config, SGLangOptions(tp=1, live_update=live), spawn=spawn,
                              sleep=lambda s: None)


def _release_dir(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir(exist_ok=True)
    return path


def _round(engine: SGLangServerEngine, out: list) -> threading.Thread:
    thread = threading.Thread(target=lambda: out.extend(
        engine.generate(["a", "b"], SamplingParams(max_tokens=TOKENS, n=2))))
    thread.start()
    return thread


def _wait_tokens(fake: FakeSGLang, count: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with fake._lock:
            if fake._running and min(len(r.tokens) for r in fake._running) >= count:
                return
        time.sleep(0.002)
    raise AssertionError("the fake server generated nothing")


def test_live_update_pauses_loads_and_continues_the_running_round(tmp_path):
    spawned: list = []
    with FakeSGLang() as fake:
        engine = _engine(tmp_path, fake, live=True, spawned=spawned)
        assert updates_during_generation(engine)
        engine.start(None, 0)
        assert "--disable-radix-cache" in spawned[0]
        out: list = []
        thread = _round(engine, out)
        _wait_tokens(fake, 20)
        report = engine.load(_release_dir(tmp_path, "v1"), 1)
        with fake._lock:
            unfinished = sum(1 for r in fake._running if len(r.tokens) < TOKENS)
        thread.join(10)
    assert unfinished == 4  # the update did not wait for the round
    assert report.reload_mode == "in_place" and report.detail["live"] is True
    paths = [p for p, _ in fake.calls if p != "/generate"]
    assert paths == ["/pause_generation", "/update_weights_from_disk", "/continue_generation"]
    update = dict(fake.calls)["/update_weights_from_disk"]
    assert update["flush_cache"] is False and update["weight_version"] == "1"
    assert dict(fake.calls)["/pause_generation"] == {"mode": "retract"}
    assert fake.updates[0]["paused"] and engine.served_version() == 1
    for completion in (c for group in out for c in group):
        assert completion.weight_version == 0 and completion.final_weight_version == 1
        versions = {t - TOKEN_BASE for t in completion.token_ids}
        assert versions == {0, 1}  # tokens before and after the update
        assert len(completion.token_ids) == len(completion.logprobs) == TOKENS


def test_without_live_update_the_update_waits_for_the_round(tmp_path):
    """The E5 behavior: the release waits behind the whole generation."""
    with FakeSGLang() as fake:
        engine = _engine(tmp_path, fake, live=False)
        assert not updates_during_generation(engine)
        engine.start(None, 0)
        out: list = []
        thread = _round(engine, out)
        _wait_tokens(fake, 20)
        engine.load(_release_dir(tmp_path, "v1"), 1)
        with fake._lock:
            running = len(fake._running)
        thread.join(10)
    assert running == 0  # it returned only after every request ended
    assert "/pause_generation" not in [p for p, _ in fake.calls]
    for completion in (c for group in out for c in group):
        assert completion.weight_version is None and {t - TOKEN_BASE for t in completion.token_ids} == {0}


def test_a_failed_live_update_continues_generation_on_the_served_weights(tmp_path):
    with FakeSGLang() as fake:
        engine = _engine(tmp_path, fake, live=True)
        engine.start(None, 0)
        out: list = []
        thread = _round(engine, out)
        _wait_tokens(fake, 10)
        with pytest.raises(SamplerEngineError, match="load failed"):
            engine.load(_release_dir(tmp_path, "broken"), 1)
        thread.join(10)
    assert not fake.paused and [p for p, _ in fake.calls][-1] == "/continue_generation"
    assert engine.served_version() == 0
    assert all(c.final_weight_version == 0 and len(c.token_ids) == TOKENS for g in out for c in g)


def test_live_update_needs_in_place_reload(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    config = ServerConfig(base_dir=base, card_model="H100", device_family="cuda", engine_version="0.5.20",
                          precision="bf16", quant_scheme="bf16", topology="tp1", max_model_len=4096)
    with pytest.raises(SamplerEngineError, match="in_place_reload"):
        SGLangServerEngine(config, SGLangOptions(tp=1, live_update=True))


def test_sample_record_carries_both_versions():
    from yeto.rl.engine.sampler_engine import Completion

    completion = Completion(token_ids=(1,), logprobs=(-0.1,), text="x", finish_reason="stop",
                            weight_version=3, final_weight_version=4)
    record = sample_record(PromptItem("p", "q"), completion, reward=1.0, version=3, provenance={},
                           group_index=0, index=0)
    assert record["metadata"]["weight_version"] == "3"
    assert record["metadata"]["final_weight_version"] == "4"


# ------------------------------------------------------------------ the loop
def _store(tmp_path: Path) -> PolicyReleaseStore:
    return PolicyReleaseStore(CheckpointStore(str(tmp_path / "store")), keep=8)


def _release(store: PolicyReleaseStore, version: int):
    return store.release(version, f"h{version}",
                         lambda d: (d / "model.safetensors").write_bytes(b"w" * (version + 1)),
                         weights=FULL)


class _Buffer:
    groups = 0
    ended = False

    def refresh(self):
        pass

    def __len__(self):
        return self.groups

    def mark_data_end(self):
        self.ended = True


def _loop(follower, generate, events, **kw) -> SamplerLoop:
    return SamplerLoop(follower=follower, buffer=_Buffer(), generate=generate, num_rollouts=1,
                       max_backlog_groups=100, emit=lambda e, **f: events.append((e, f)),
                       sleep=lambda s: None, live_poll_s=0.005, **kw)


def _wait(predicate) -> None:
    deadline = time.monotonic() + 10
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.002)


def test_live_loop_loads_a_release_that_lands_during_the_round(tmp_path):
    store, loaded, events = _store(tmp_path), [], []
    follower = PolicyFollower(store, lambda r, d: loaded.append(r.version), live_update=True)

    def generate(rollout_id):
        _release(store, 1)
        _wait(lambda: loaded == [1])  # the round goes on only after the in-round load

    loop = _loop(follower, generate, events)
    assert loop.run() == 1 and not loop.unchecked_in_flight
    in_round = [f for e, f in events if e == "rl_sampler_policy_loaded"]
    assert in_round[0]["in_flight"] is True and follower.counters.deferred_in_flight == 0


def test_live_loop_marks_a_round_whose_in_round_load_failed_a_check(tmp_path):
    store, events = _store(tmp_path), []
    ready = {"ok": True}

    def load(release, directory):
        ready["ok"] = False  # the post-load checks failed: the engine holds unchecked weights
        raise ReleaseNotLoaded("probe", reason="load_check_failed")

    follower = PolicyFollower(store, load, live_update=True)

    def generate(rollout_id):
        _release(store, 1)
        _wait(lambda: any(e == "rl_sampler_release_not_loaded" for e, _ in events))
        _wait(lambda: loop.unchecked_in_flight)

    loop = _loop(follower, generate, events, ready=lambda: ready["ok"])
    assert loop.run() == 1 and loop.unchecked_in_flight


def test_live_loop_reraises_an_in_round_load_error(tmp_path):
    store = _store(tmp_path)

    def load(release, directory):
        raise SamplerEngineError("server died")

    follower = PolicyFollower(store, load, live_update=True)

    def generate(rollout_id):
        _release(store, 1)
        time.sleep(0.1)

    with pytest.raises(SamplerEngineError, match="server died"):
        _loop(follower, generate, []).run()


def test_engine_round_drops_a_round_generated_on_unchecked_weights():
    from yeto.rl.engine.engine_sampler import EngineRound
    from yeto.rl.engine.sampler_engine import Completion, EngineIdentity, Provenance

    class _Live:
        updates_during_generation = True

        def served_version(self):
            return 2

        def generate(self, prompts, sampling):
            return [[Completion((1,), (-0.1,), "x", "stop", weight_version=1, final_weight_version=2)]
                    for _ in prompts]

    class _Writer:
        sampler_id = "s"
        written = 0

        def write(self, *a, **k):
            self.written += 1

    identity = EngineIdentity("sglang", "0.5.20", "cuda", "H100", "bf16", "bf16", "tp1")
    sampling = SamplingParams(max_tokens=4)
    events, writer, flag = [], _Writer(), {"bad": True}
    engine_round = EngineRound(_Live(), writer, prompts=lambda r: [PromptItem("p", "q")],
                               reward=lambda i, c: 1.0, sampling=sampling,
                               provenance=Provenance.of(identity, sampler_id="s", sampling=sampling),
                               emit=lambda e, **f: events.append(f),
                               unchecked_in_flight=lambda: flag["bad"])
    engine_round(0)
    assert writer.written == 0 and events[-1]["dropped_reason"] == "unchecked_weights_in_flight"
    flag["bad"] = False
    engine_round(1)
    assert writer.written == 1 and events[-1]["groups"] == 1


def test_completion_versions_from_spans_or_the_sent_version():
    from yeto.rl.adapters.sampler_engines.sglang_server import _completion_versions

    spans = [{"version": "default", "start": 0, "end": 4}, {"version": "6", "start": 4, "end": 9}]
    assert _completion_versions({"weight_versions": spans}, 5, 5) == {"weight_version": 5,
                                                                     "final_weight_version": 6}
    queued = [{"version": "6", "start": 0, "end": 9}]  # queued at the update: all tokens are v6
    assert _completion_versions({"weight_versions": queued}, 5, 5)["weight_version"] == 6
    assert _completion_versions({"weight_version": "6"}, 5, 5) == {"weight_version": 5,
                                                                  "final_weight_version": 6}
    assert _completion_versions({}, 5, 3) == {"weight_version": 5, "final_weight_version": 5}


def test_live_update_without_spans_uses_the_sent_version(tmp_path):
    with FakeSGLang(spans=False) as fake:
        engine = _engine(tmp_path, fake, live=True)
        engine.start(None, 0)
        out: list = []
        thread = _round(engine, out)
        _wait_tokens(fake, 20)
        engine.load(_release_dir(tmp_path, "v1"), 1)
        thread.join(10)
    assert all((c.weight_version, c.final_weight_version) == (0, 1) for g in out for c in g)
