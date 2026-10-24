"""S27 G-AP: background pushes of the asynchronous full-parameter rounds.

With every role able to encode ahead, the pairing hands each pushed pair to a
worker thread (delta, wire encoding, enqueue) and returns to the driver, which
trains the next local round meanwhile.  The pairing keeps every table on its
own thread: it records a push only when it harvests the finished job.

Covered on fakes: the boundary does not wait for the push; a PULL seen again
while its push runs is not pushed twice; a harvested push makes the pair
complete (its broadcast is the paired apply); a round closed while its push
runs keeps the operands pinned until the push ends, and a push not started is
cancelled; a worker error fail-stops; a push that was not queued waits for
the replayed PULL; close cancels and waits; the conditions that keep the
synchronous push.  The session applies broadcasts once per boundary with
background pushes and reports where the boundary time went.

End to end (real Miles role stream, CPU trainer fake, in-process syncer that
delivers between local rounds): background pushes send the same deltas and
leave every trainer and syncer value bit-identical to synchronous pushes.
"""

from __future__ import annotations

# ruff: noqa: I001 -- the Miles test helpers install a bridge stub before import.

import asyncio
import threading
import time
from concurrent.futures import wait as wait_for_futures
from types import SimpleNamespace

import pytest
import torch
from test_rl_full_parameter_sync import _PairedRole

from yeto.protocol import PullRequest
from yeto.rl.adapters.miles.full_parameter.role_stream import RoleSubmissionCandidate, SentPush
from yeto.rl.engine import full_parameter_sync as sync_module
from yeto.rl.engine.full_parameter_sync import FullParameterSync, FullParameterSyncProfile, RolePairing


def _candidate(global_step: int, fragments: int = 2) -> RoleSubmissionCandidate:
    return RoleSubmissionCandidate(
        "actor", PullRequest((global_step - 1) % fragments, global_step, 1),
        max(0, global_step - fragments), 1, 1, 7, 8, object(), object())


class _BackgroundRole(_PairedRole):
    """A role with the background push methods; ``gate`` holds every push."""

    encodes_ahead = True

    def __init__(self, candidates, *, gate=None, queued=True, fail_step=None, **kwargs):
        super().__init__("actor", candidates, [], release_broadcasts=False, **kwargs)
        self.gate = gate
        self.queued = queued  # bool, or a callable(step) -> bool
        self.fail_step = fail_step
        self.sent: list[tuple[int, str]] = []
        self.noted: list[int] = []
        self.retained: list[int] = []
        self.released: list[int] = []
        self.timed: list[int] = []
        self.lock = threading.Lock()

    def retain_candidate(self, candidate):
        self.retained.append(candidate.permit.global_step)

    def release_candidate(self, candidate):
        self.released.append(candidate.permit.global_step)

    def pushable(self, candidate):
        return any(value.logical_key == candidate.logical_key for value in self.candidates)

    def prepare_push(self, candidate):
        if self.gate is not None:
            assert self.gate.wait(10), "the test never opened the gate"
        if candidate.permit.global_step == self.fail_step:
            raise ValueError("encode failed")
        return ("bytes", candidate.permit.global_step)

    def send_prepared(self, candidate, prepared):
        assert prepared == ("bytes", candidate.permit.global_step)
        step = candidate.permit.global_step
        with self.lock:
            self.sent.append((step, threading.current_thread().name))
        queued = self.queued(step) if callable(self.queued) else self.queued
        return SentPush(queued, 1, {"prepare_s": 0.01, "push_s": 0.02}, time.monotonic())

    def note_push_timing(self, candidate, timings):
        self.timed.append(candidate.permit.global_step)

    def note_pushed(self, candidate, link_generation):
        assert threading.current_thread() is threading.main_thread()
        assert link_generation == 1
        self.noted.append(candidate.permit.global_step)
        return super().push_candidate(candidate)

    def push_candidate(self, candidate, prepared=None):  # pragma: no cover - must not be used
        raise AssertionError("a background role was pushed on the coordinator thread")


def _pairing(role, *, depth=2, fragments=2, **overrides):
    role.fragment_versions = (0,) * fragments
    values = {"roles": ("actor",), "total_fragment_steps": 10 * fragments, "fragments": fragments,
              "local_horizon": 1, "pipeline_depth": depth, "wait_timeout": 900.0, "poll_seconds": 0.001,
              "partial_quorum": True}
    values.update(overrides)
    return RolePairing({"actor": role}, FullParameterSyncProfile(**values))


def _wait_jobs(pairing):
    futures = [pair.job.future for pair in (*pairing.inflight.values(), *pairing._releasing)
               if pair.job is not None]
    _done, pending = wait_for_futures(futures, timeout=10)
    assert not pending


def _steps(progress):
    return [submission.global_step for submission in progress["actor"].submissions]


def test_the_boundary_does_not_wait_for_the_push_and_harvests_it_once():
    gate = threading.Event()
    role = _BackgroundRole([_candidate(1), _candidate(2)], gate=gate)
    pairing = _pairing(role)
    started = time.monotonic()
    assert _steps(pairing.submit_ready()) == []
    assert time.monotonic() - started < 1.0
    assert pairing.pushes_in_background  # decided at the first submit
    assert pairing.push_jobs_running == 2 and sorted(role.retained) == [1, 2]
    # The PULLs stay visible while the workers push: no second push.
    for _ in range(3):
        assert _steps(pairing.submit_ready()) == []
    gate.set()
    _wait_jobs(pairing)
    assert sorted(_steps(pairing.submit_ready())) == [1, 2]
    assert sorted(step for step, _ in role.sent) == [1, 2]
    assert all(name.startswith("yeto-push-bg") for _, name in role.sent)
    assert sorted(role.noted) == [1, 2] and sorted(role.timed) == [1, 2]
    assert all(pair.complete and pair.job is None for pair in pairing.inflight.values())
    assert pairing.push_job_counters()["harvested"] == 2
    assert _steps(pairing.submit_ready()) == []  # nothing is pushed twice
    assert len(role.sent) == 2


def test_a_harvested_push_makes_its_broadcast_the_paired_apply():
    role = _BackgroundRole([_candidate(1)])
    pairing = _pairing(role)
    pairing.submit_ready()
    _wait_jobs(pairing)
    role._staged_steps.append(1)  # the broadcast arrived while the island trained
    asyncio.run(pairing.commit_staged_broadcasts())
    assert role.fragment_versions == (1, 0)
    assert pairing.overwritten_rounds == 0
    assert [list(entry) for entry in pairing.journal] == [["actor"]]
    assert role.released == [1] and not pairing.inflight


def test_a_round_closed_while_its_push_runs_keeps_the_operands_until_the_push_ends():
    gate = threading.Event()
    role = _BackgroundRole([_candidate(1)], gate=gate)
    pairing = _pairing(role)
    pairing.submit_ready()
    role._staged_steps.append(1)  # the other islands closed round 1 meanwhile
    asyncio.run(pairing.commit_staged_broadcasts())
    assert role.fragment_versions == (1, 0) and pairing.overwritten_rounds == 1
    assert not pairing.inflight and role.released == []  # the worker still reads them
    assert pairing.push_jobs_running == 1
    gate.set()
    _wait_jobs(pairing)
    pairing.submit_ready()
    assert role.released == [1]
    assert role.noted == [1]  # the late push was queued (the syncer may carry it)
    assert pairing.journal == [] and not pairing._releasing


def test_a_push_not_started_is_cancelled_when_its_round_closes(monkeypatch):
    monkeypatch.setenv(sync_module.PUSH_WORKERS_ENV, "1")
    gate = threading.Event()
    role = _BackgroundRole([_candidate(1), _candidate(2)], gate=gate)
    pairing = _pairing(role)
    pairing.submit_ready()
    role._staged_steps.append(2)  # round 2 closes before its job could start
    asyncio.run(pairing.commit_staged_broadcasts())
    assert role.released == [2]
    gate.set()
    _wait_jobs(pairing)
    pairing.submit_ready()
    assert [step for step, _ in role.sent] == [1]


def test_a_worker_error_fail_stops_the_pairing_at_the_harvest():
    role = _BackgroundRole([_candidate(1), _candidate(2)], fail_step=2)
    pairing = _pairing(role)
    pairing.submit_ready()
    _wait_jobs(pairing)
    with pytest.raises(ValueError, match="encode failed"):
        pairing.submit_ready()
    assert pairing.failure is not None and role.closed
    with pytest.raises(RuntimeError, match="fail-stopped"):
        pairing.submit_ready()


def test_a_push_that_was_not_queued_waits_for_the_replayed_pull():
    attempts = []

    def queued(step):
        attempts.append(step)
        return len(attempts) > 1  # the first enqueue meets an outage

    role = _BackgroundRole([_candidate(1)], queued=queued)
    pairing = _pairing(role)
    pairing.submit_ready()
    _wait_jobs(pairing)
    assert _steps(pairing.submit_ready()) == []  # harvested: not queued; the PULL is replayed
    _wait_jobs(pairing)
    assert _steps(pairing.submit_ready()) == [1]
    assert attempts == [1, 1] and role.noted == [1]
    assert next(iter(pairing.inflight.values())).complete


def test_close_cancels_queued_pushes_and_waits_for_running_ones(monkeypatch):
    monkeypatch.setenv(sync_module.PUSH_WORKERS_ENV, "1")
    gate = threading.Event()
    role = _BackgroundRole([_candidate(1), _candidate(2)], gate=gate)
    pairing = _pairing(role)
    pairing.submit_ready()
    threading.Timer(0.2, gate.set).start()
    assert pairing.close_all() == []
    assert role.closed
    assert [step for step, _ in role.sent] == [1]  # the second never started


@pytest.mark.parametrize("change", ["env", "sweep", "no-method", "no-encode-ahead", "strict-rounds"])
def test_the_synchronous_push_stays_when_a_condition_is_missing(monkeypatch, change):
    role = _BackgroundRole([_candidate(1)])
    overrides = {}
    if change == "env":
        monkeypatch.setenv(sync_module.BACKGROUND_PUSH_ENV, "0")
    elif change == "sweep":
        overrides = {"partial_quorum": False, "sweep": True}
    elif change == "strict-rounds":
        overrides = {"partial_quorum": False}
    elif change == "no-method":
        role.send_prepared = None
    else:
        role.encodes_ahead = False
    role.push_candidate = lambda candidate, prepared=None: _PairedRole.push_candidate(role, candidate)
    pairing = _pairing(role, **overrides)
    pairing.submit_ready()
    assert not pairing.pushes_in_background
    assert role.sent == []


def test_finish_drains_the_running_pushes_before_the_final_apply():
    gate = threading.Event()
    role = _BackgroundRole([_candidate(1)], gate=gate)
    pairing = _pairing(role)
    pairing.submit_ready()
    role.finalizing = True
    threading.Timer(0.2, gate.set).start()
    asyncio.run(pairing.finish_without_more_training())
    assert role.noted == [1] and not pairing.inflight and pairing.push_jobs_running == 0


# -- the session ----------------------------------------------------------------------


class _CountingRole(_BackgroundRole):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_calls = 0

    async def stage_available_broadcasts(self):
        self.stage_calls += 1
        return await super().stage_available_broadcasts()


@pytest.mark.parametrize("background", [True, False])
def test_the_session_applies_once_per_boundary_with_background_pushes(monkeypatch, background):
    monkeypatch.setenv(sync_module.BACKGROUND_PUSH_ENV, "1" if background else "0")
    role = _CountingRole([_candidate(1)])
    if not background:
        role.push_candidate = lambda candidate, prepared=None: _PairedRole.push_candidate(role, candidate)
    pairing = _pairing(role)
    session = FullParameterSync(pairing, groups={"actor": object()}, run=asyncio.run,
                                trajectories=lambda rollout_id: SimpleNamespace(rollout_id=rollout_id))
    events = []
    driver = SimpleNamespace(emit=lambda name, **fields: events.append((name, fields)), outer_version=None)
    session._driver = driver
    session.boundary(driver, rollout_id=0, stats=None)
    assert role.stage_calls == (1 if background else 2)
    (fields,) = [fields for name, fields in events if name == "rl_full_parameter_round"]
    assert fields["sync/push_background"] is background
    assert {"sync/boundary_capture_s", "sync/boundary_apply_s", "sync/boundary_materialize_s",
            "sync/boundary_submit_s", "sync/boundary_wait_s"} <= set(fields)
    assert ("sync/boundary_apply_after_submit_s" in fields) is not background
    if background:
        assert fields["sync/push_jobs_submitted"] == 1
        assert fields["sync/push_jobs_running"] + fields["sync/push_jobs_harvested"] <= 1


# -- end to end: the real Miles role stream ----------------------------------------------

try:
    import miles  # noqa: F401
    HAVE_MILES = True
except ImportError:
    HAVE_MILES = False


@pytest.mark.skipif(not HAVE_MILES, reason="needs Miles")
@pytest.mark.parametrize(("workers", "horizon"), [("1", 1), ("3", 1), ("3", 2)])
def test_background_pushes_keep_every_value_bit_identical_to_synchronous_pushes(monkeypatch, workers, horizon):
    """``horizon`` is H (``--local-rl-rounds-per-sync``): a fragment is
    pushed only after H local rounds since its anchor."""
    monkeypatch.setenv(sync_module.PUSH_WORKERS_ENV, workers)
    synchronous = _run_miles(monkeypatch, background=False, horizon=horizon)
    background = _run_miles(monkeypatch, background=True, horizon=horizon)
    assert background.pairing.push_job_counters()["harvested"] > 0
    assert not synchronous.pairing.pushes_in_background
    for run in (synchronous, background):
        assert run.stops[-1] and run.stream.final_applied and run.stream.final_acknowledged
        assert sorted(run.syncer.committed) == list(range(1, len(run.syncer.committed) + 1))
    assert background.stops == synchronous.stops

    def bits(values):
        return {f: value.view(torch.int32).tolist() for f, value in values.items()}

    def pushes(run):
        return sorted((step, attempt, delta.view(torch.int32).tolist()) for step, attempt, delta in run.syncer.pushed)

    assert pushes(background) == pushes(synchronous)
    assert bits(background.trainer.values) == bits(synchronous.trainer.values)
    assert bits(background.syncer.global_) == bits(synchronous.syncer.global_)
    assert len(background.trainer.trained_from) == len(synchronous.trainer.trained_from)
    for run in (synchronous, background):
        assert min(c_steps for _step, c_steps, _local in run.client.meta) >= horizon
        by_fragment = {}
        for step, _c_steps, local_step in sorted(run.client.meta):
            by_fragment.setdefault((step - 1) % len(run.syncer.versions), []).append(local_step)
        for local_steps in by_fragment.values():  # H local rounds between a fragment's pushes
            assert all(b - a >= horizon for a, b in zip(local_steps, local_steps[1:]))
    for got, want in zip(background.trainer.trained_from, synchronous.trainer.trained_from, strict=True):
        assert bits(got) == bits(want)


def _run_miles(monkeypatch, *, background: bool, horizon: int = 1):
    import test_s27_sync_fast_miles_role_stream as fast

    from yeto.rl.adapters.miles.full_parameter import role_stream as role_stream_module
    from yeto.rl.adapters.miles.full_parameter.role_stream import MilesFullParameterRoleStream
    from yeto.rl.adapters.miles.models import chunked_full_parameter as adapter_module
    from yeto.rl.adapters.miles.models.chunked_full_parameter import (
        DEFER_CAPTURE_ENV,
        SKIP_RESIDENT_ENV,
        MilesChunkedFullParameterAdapter,
    )

    monkeypatch.setenv(sync_module.BACKGROUND_PUSH_ENV, "1" if background else "0")
    for env in (DEFER_CAPTURE_ENV, sync_module.BATCH_APPLY_ENV, SKIP_RESIDENT_ENV):
        monkeypatch.setenv(env, "1")
    monkeypatch.setattr(adapter_module, "_local_transport_features",
                        lambda: frozenset({"skip_resident", "deferred_export", "optimizer_state_check_model"}))
    rounds = 4
    generator = torch.Generator().manual_seed(41)

    def rand(scale=1.0):
        return {f: torch.randn(n, generator=generator) * scale for f, n in enumerate(fast.SIZES)}

    initial = rand()
    steps = {k: rand(0.1) for k in range(4 * rounds)}
    peer = {k: rand(0.1) for k in range(4 * rounds)}
    ray = fast.FakeRay()
    adapter = fast._adapter()
    holder = SimpleNamespace(pairing=None)

    def delivered_before_training():
        """The network delivers while the island trains: every push of the
        previous boundary has reached the syncer by the next local step."""
        pairing = holder.pairing
        if pairing is not None:
            _wait_jobs(pairing)
        client.deliver()

    class _Trainer(fast._MilesContractTrainer):
        async def record_full_parameter_local_step(self, **kwargs):
            delivered_before_training()
            return await super().record_full_parameter_local_step(**kwargs)

    trainer = _Trainer(adapter, ray, initial, lambda k: steps[k], fast=True)
    syncer = fast._AsyncSyncer(initial, rounds, lambda k: peer[k])
    client = _EncodeAheadClient(syncer, holder)

    async def initialize(cls, group, *, policy_version, **_kwargs):
        cut = await group.export_full_parameter_chunked_cut(policy_version, 0,
                                                            max_chunk_bytes=adapter.max_chunk_bytes)
        return adapter, adapter.validate_cut(cut, policy_version, 0)

    monkeypatch.setattr(MilesChunkedFullParameterAdapter, "initialize", classmethod(initialize))
    monkeypatch.setattr(role_stream_module, "SyncerClient", lambda *_args, **_kwargs: client)
    config = fast._config()
    from dataclasses import replace

    stream = MilesFullParameterRoleStream(replace(config, total_fragment_steps=rounds * fast.P,
                                                  local_horizon=horizon),
                                          ray_module=ray)
    profile = FullParameterSyncProfile.from_configs({"actor": stream.config})
    session = FullParameterSync(
        RolePairing({"actor": stream}, profile), groups={"actor": trainer}, run=asyncio.run,
        trajectories=lambda rollout_id: fast._evidence(rollout_id=rollout_id,
                                                       behavior_policy_hash=stream.current.policy_hash))
    holder.pairing = session.pairing
    driver = fast._Driver()
    session.start(driver)
    stops = []
    for rollout_id in range(4 * rounds):
        boundary = session.boundary(driver, rollout_id=rollout_id, stats=None)
        stops.append(boundary.stop)
        if boundary.stop:
            session.published(driver, rollout_id=rollout_id, policy_hash=boundary.state.policy_tensor_hash())
            break
    session.finish(driver)
    return SimpleNamespace(trainer=trainer, syncer=syncer, stream=stream, stops=stops,
                           pairing=session.pairing, client=client)


class _EncodeAheadClient:
    """The fake client of ``test_s27_sync_fast_miles_role_stream`` with a
    deterministic encode-ahead codec (f32 bytes) and thread-safe tables.

    Pushes are delivered to the syncer only by :meth:`deliver` (called before
    each local step) or, once the island stopped training, on every poll."""

    can_encode_ahead = True

    def __init__(self, syncer, holder) -> None:
        import test_s27_sync_fast_miles_role_stream as fast

        self._base = fast._AsyncClient(syncer)
        self.syncer = syncer
        self.holder = holder
        self.lock = threading.RLock()
        self.meta: list[tuple[int, int, int]] = []  # (global step, c_steps, local step) per push

    def __getattr__(self, name):
        return getattr(self._base, name)

    def encode_push_parts(self, fragment_id, parts, *, timings=None):
        data = b"".join(bytes(part) for part in parts)
        return SimpleNamespace(fragment_id=fragment_id, push_base=0, nbytes=len(data), parts=(data,))

    def push_fragment_parts(self, fragment_id, global_step, round_attempt, base_version, local_step,
                            c_steps, c_tokens, parts, *, encoded=None, expert_tags=None, timings=None):
        payload = b"".join(bytes(part) for part in (encoded.parts if encoded is not None else parts))
        delta = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()
        with self.lock:
            assert c_steps >= 1 and c_tokens >= 1
            assert base_version == self.syncer.versions[fragment_id], "PUSH on a stale base"
            self.syncer.pushed.append((global_step, round_attempt, delta.clone()))
            self.meta.append((global_step, c_steps, local_step))
            self.syncer.received[global_step] = delta
        return True

    def deliver(self) -> None:
        with self.lock:
            self.syncer.commit_ready()

    def _deliver_when_stopped(self) -> None:
        pairing = self.holder.pairing
        if pairing is not None and pairing.any_finalizing and all(
                pair.job is None or pair.job.future.done()
                for pair in (*pairing.inflight.values(), *pairing._releasing)):
            self.deliver()

    def drain_updates(self):
        with self.lock:
            self._deliver_when_stopped()
            return self._base.drain_updates()

    def drain_pulls(self):
        with self.lock:
            self._deliver_when_stopped()
            return self._base.drain_pulls()
