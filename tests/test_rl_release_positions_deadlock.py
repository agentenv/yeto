"""A staged policy release never waits for the rollout executor (release deadlock).

The trainer's buffer reader lives in the rollout executor, a Ray asyncio actor:
a take is synchronous and holds the actor's event loop until its groups exist,
and ``__ray_call__`` runs on that loop only when it is free. A sampler counts
its backlog from the read position of the newest release (0 before the first)
and stops at ``max_backlog_groups``. A release record that reads the position
through ``__ray_call__`` after its upload waits for the take, the take waits for
groups, the samplers wait for the record: none moves. The record now reads the
position the driver kept after the reader's last finished round
(``MilesRolloutPool.buffer_read_positions``)."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_rl_miles_adapter_rollout import H, FakeController, group, run_hooks
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles import rollout_meta_hook as hook
from yeto.rl.adapters.miles import sampler_island as si
from yeto.rl.adapters.miles.rollout import DirMetadataSource, MilesRolloutPool
from yeto.rl.engine import sample_roles
from yeto.rl.engine.policy_release import LATEST, RELEASE_DIR, PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore

G = 8  # groups per round (--rollout-batch-size)
CAP = 16  # the sampler's max_backlog_groups
ROUNDS = 6
DEADLINE_S = 30.0
UPLOAD_THREAD = "policy-release-upload"  # BackgroundReleaser


class _State:
    def __init__(self, version: int) -> None:
        self.policy_version = version

    def policy_tensor_hash(self) -> str:
        return f"h{self.policy_version}"


class Buffer:
    """The groups the sampler wrote and the trainer reader's position."""

    def __init__(self) -> None:
        self.written = 0
        self.read_cursor = 0
        self.stop = threading.Event()

    def shard_positions(self, cursor: int) -> dict[str, int]:
        return {"s0": int(cursor)}


class Loop:
    """The executor's one event-loop thread: a take holds it; ``__ray_call__``
    runs on it once it is free (ray ``actor.py``: no concurrency group)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: list[tuple[str, str]] = []


class Ref:
    """A ``__ray_call__`` reference: ``ray.get`` / ``await`` run it on the loop."""

    def __init__(self, loop: Loop, fn, args) -> None:
        self.loop, self.fn, self.args = loop, fn, args

    def result(self):
        with self.loop.lock:
            self.loop.calls.append((threading.current_thread().name, self.fn.__name__))
            return self.fn(None, *self.args)

    def __await__(self):
        yield from ()
        return self.result()


class Executor:
    """Miles' ``RayWorkerHandle`` around the rollout executor actor: ``get`` runs
    the buffer reader's round (a synchronous take on the loop), ``__ray_call__``
    on ``_actor_handle``."""

    def __init__(self, loop: Loop, buffer: Buffer, sink: str) -> None:
        self._actor_handle = SimpleNamespace(
            __ray_call__=SimpleNamespace(remote=lambda fn, *a: Ref(loop, fn, a)))
        self.loop, self.buffer, self.sink = loop, buffer, sink
        self.args = SimpleNamespace()
        self.waiting: dict[int, threading.Event] = {}

    def waiting_on(self, rollout_id: int) -> threading.Event:
        return self.waiting.setdefault(rollout_id, threading.Event())

    async def get(self, rollout_id: int):
        with self.loop.lock:  # BufferReaderRolloutFn.__call__ never yields the loop
            want = (rollout_id + 1) * G
            while self.buffer.written < want:
                self.waiting_on(rollout_id).set()
                if self.buffer.stop.is_set():
                    raise RuntimeError(f"rollout {rollout_id}: {self.buffer.written}/{want} groups, stopped")
                time.sleep(0.002)
            self.buffer.read_cursor = want
            cursor, shards = br.reader_positions(SimpleNamespace(buffer=self.buffer))
            hook.record_round_metadata(self.args, rollout_id, sink=self.sink,
                                       buffer_read_cursor=cursor, buffer_read_shards=shards or {})
            kept = [group(0, [1.0, 0.0])]
            run_hooks(self.args, kept, [], self.sink)
        return SimpleNamespace(sample_indices=[s.index for g in kept for s in g], data_ref="REF")


def _latest(store: Path) -> dict | None:
    path = store / RELEASE_DIR / LATEST
    return json.loads(path.read_text()) if path.is_file() else None


def _sampler(buffer: Buffer, store: Path, stop: threading.Event) -> None:
    """One sampler at max_backlog_groups CAP, counting from the newest release's
    read position (0 before the first one; SamplerLoop.backlog)."""
    while not stop.is_set():
        latest = _latest(store)
        read = 0 if latest is None or latest.get("buffer_selected") is None else int(latest["buffer_selected"])
        if buffer.written - read < CAP:
            buffer.written += 1
        else:
            time.sleep(0.002)


def test_a_staged_release_lands_while_a_take_holds_the_executor(tmp_path, monkeypatch):
    """The ring of the 1123 e2e run: cap 16, no release yet, 8 groups a round,
    skip-busy, an upload slower than two rounds. r1-v1's record is written while
    round 2's take holds the executor's loop, the sampler goes on from it, and
    the trainer trains every round. Before the fix the record waited for the take
    through ``__ray_call__`` and nothing moved."""
    loop, buffer = Loop(), Buffer()
    store, staging, meta = tmp_path / "store", tmp_path / "staging", tmp_path / "meta"
    meta.mkdir()
    sink = f"dir:{meta}"
    monkeypatch.setenv(hook.META_SINK_ENV, sink)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda ref: ref.result()))
    monkeypatch.setitem(br._ACTIVE, br.TRAINER, SimpleNamespace(
        buffer=buffer, trained_before=lambda rollout_id: int(rollout_id) * G))
    executor = Executor(loop, buffer, sink)
    pool = MilesRolloutPool(inference_controller=FakeController(), rollout_executor=executor,
                            metadata=DirMetadataSource(meta), expected_policy=lambda: (3, H))
    events: list[tuple[str, dict]] = []
    driver = SimpleNamespace(publisher=SimpleNamespace(publish=lambda state: None), rollout=pool,
                             emit=lambda event, **fields: events.append((event, fields)))
    args = SimpleNamespace(yeto_rl_sample_buffer=sample_roles.SampleBufferConfig(
        role="trainer", store=str(store), release_every=1, release_skip_busy=True))

    def run_plugin(fn_path, kwargs):
        Path(kwargs["path"], "model.safetensors").write_bytes(b"w" * (kwargs["rollout_id"] + 1))
        return [{"writers": "all"}]

    stop, recorded, failure = threading.Event(), [], []

    def train() -> None:
        try:
            wired = si.wire_policy_releases(driver, miles_args=args, actor_model=SimpleNamespace(run_plugin=run_plugin),
                                            runner=SimpleNamespace(run=lambda value: value),
                                            environ={si.STAGING_ENV: str(staging)})
            original, put = wired.releases.release_staged, wired.releases.store.put_files

            def slow_upload(*a, **k):  # the weights go up while round 2 waits for groups
                assert executor.waiting_on(2).wait(DEADLINE_S)
                return original(*a, **k)

            def put_files(files):  # each record's read position and the reader's when it is written
                for rel, data in files.items():
                    if rel.endswith("/" + LATEST):
                        recorded.append((json.loads(data)["buffer_selected"], buffer.read_cursor))
                return put(files)

            wired.releases.release_staged = slow_upload
            wired.releases.store.put_files = put_files
            for rollout_id in range(ROUNDS):
                pool.generate(rollout_id)
                wired.publish(_State(rollout_id + 1))
            wired.drain_releases()
        except BaseException as error:  # noqa: BLE001 - reported by the test thread
            failure.append(error)

    sampler = threading.Thread(target=_sampler, args=(buffer, store, stop), daemon=True)
    trainer = threading.Thread(target=train, name="driver", daemon=True)
    sampler.start()
    trainer.start()
    try:
        trainer.join(DEADLINE_S)
        hung = trainer.is_alive()
        first = _latest(store)
    finally:
        buffer.stop.set()  # a hung take raises: the loop and every thread waiting on it go free
        stop.set()
        trainer.join(DEADLINE_S)
        sampler.join(DEADLINE_S)
    assert not hung, (f"release deadlock: no release record after {DEADLINE_S:.0f} s (LATEST {first}); "
                      f"sampler at {buffer.written} groups, reader at {buffer.read_cursor}")
    assert not failure, failure
    assert not [c for c in loop.calls if c[0].startswith(UPLOAD_THREAD)], loop.calls
    released = [f for e, f in events if e == "rl_policy_released"]
    assert [f["policy_version"] for f in released][:1] == [1] and released[-1]["policy_version"] == ROUNDS
    # r1-v1: trained on round 0 (8 groups); written while round 2 waited: rounds 0-1 were read
    assert (released[0]["buffer_trained"], released[0]["buffer_selected"]) == (G, 2 * G), released[0]
    # every record: never past the reader, never behind what its weights were trained on
    assert recorded and all(selected is not None and selected <= live for selected, live in recorded), recorded
    for f in released:
        assert f["buffer_trained"] <= f["buffer_selected"], f
    assert buffer.read_cursor == ROUNDS * G


def test_releasing_publisher_records_the_cached_positions_and_never_the_executor(tmp_path):
    """With ``read_positions`` the record never calls the executor reads (inline or
    from the upload thread); nothing cached yet keeps the newest record's positions."""
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))

    def executor_read():
        raise AssertionError("the release read the rollout executor")

    cache = {"value": None}
    publisher = si.ReleasingPublisher(
        SimpleNamespace(publish=lambda state: None), releases,
        lambda version, directory: Path(directory, "model.safetensors").write_bytes(b"w"),
        read_position=executor_read, read_shards=executor_read, emit=lambda event, **fields: None,
        read_position_from_thread=executor_read, read_shards_from_thread=executor_read,
        read_positions=lambda: cache["value"])
    publisher.publish(_State(1))
    assert (releases.latest().buffer_selected, releases.latest().buffer_shards) == (None, None)
    cache["value"] = (16, {"s0": 10, "s1": 6})
    publisher.publish(_State(2))
    assert (releases.latest().buffer_selected, releases.latest().buffer_shards) == (16, {"s0": 10, "s1": 6})
    cache["value"] = None  # e.g. a restore forgot it: the newest record's positions stay
    publisher.publish(_State(3))
    assert (releases.latest().version, releases.latest().buffer_selected) == (3, 16)


def test_the_pool_keeps_the_positions_of_each_finished_round_and_of_a_restore(monkeypatch):
    """``_batch_of`` keeps the round's reported positions; a restore (``BufferCutLink``)
    moves the cursor back and the cache with it, read once on the driver's thread."""
    pool = MilesRolloutPool.__new__(MilesRolloutPool)
    assert pool.buffer_read_positions() is None
    pool.note_buffer_read_positions((24, {"s0": 24}))
    got = pool.buffer_read_positions()
    got[1]["s0"] = 0  # a copy
    assert pool.buffer_read_positions() == (24, {"s0": 24})
    buffer = Buffer()
    buffer.read_cursor = 24
    reader = SimpleNamespace(buffer=buffer, reset=lambda watermark: setattr(buffer, "read_cursor", watermark)
                             or watermark)
    monkeypatch.setitem(br._ACTIVE, br.TRAINER, reader)
    link = br.BufferCutLink(lambda fn, *a: fn(None, *a), pool.note_buffer_read_positions)
    assert link.resume(8) == 8
    assert pool.buffer_read_positions() == (8, {"s0": 8})
    pool.note_buffer_read_positions(None)
    assert pool.buffer_read_positions() is None
    assert br.BufferCutLink(lambda fn, *a: fn(None, *a)).resume(4) == 4  # no cache: nothing told


def test_the_round_metadata_carries_the_reader_positions(tmp_path, monkeypatch):
    """The positions travel as per-round metadata (validated like the other keys)."""
    sink = f"dir:{tmp_path}"
    hook.record_round_metadata(SimpleNamespace(), 5, sink=sink, buffer_read_cursor=40,
                               buffer_read_shards={"s0": 30, "s1": 10})
    with pytest.raises(RuntimeError, match="non-negative int"):
        hook.record_round_metadata(SimpleNamespace(), 5, sink=sink, buffer_read_cursor=-1)
    with pytest.raises(RuntimeError, match="str keys"):
        hook.record_round_metadata(SimpleNamespace(), 5, sink=sink, buffer_read_shards={0: 1})
    buffer = Buffer()
    buffer.read_cursor = 40
    assert br.reader_positions(SimpleNamespace(buffer=buffer)) == (40, {"s0": 40})
    buffer.shard_positions = lambda cursor: {}
    assert br.reader_positions(SimpleNamespace(buffer=buffer)) == (40, None)
    monkeypatch.delitem(br._ACTIVE, br.TRAINER, raising=False)
    assert br.executor_read_positions(None) is None  # no reader in this process yet
    monkeypatch.setitem(br._ACTIVE, br.TRAINER, SimpleNamespace(buffer=buffer))
    assert br.executor_read_positions(None) == (40, None)
