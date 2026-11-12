"""The participants of a simulated run: real yeto store code, stub compute.

:class:`SamplerProc` is one sampler process (one incarnation of a shard). It
builds and drives the real objects in the order of
``engine_sampler_entry._run_on``: shard buffer, resume point, store dispenser
(``recover``), membership, incarnation lease, policy follower; a "generation"
is a record that is turned into a group some scheduler steps later, written the
way ``agentic_round`` settles a group (lease check, ``still_holds``,
``GroupWriter.write``, ``written``).

:class:`TrainerProc` is one trainer process: ``open_trainer_buffer`` (the shard
merge), ``BatchReader`` with an ``IncarnationGate`` and ``max_policy_age``, a
fake train step, ``PolicyReleaseStore.release_staged`` through a staged
``S3Store``, and a cut record holding the buffer watermark that a restarted
trainer resets to.
"""

from __future__ import annotations

import os

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from botocore.exceptions import BotoCoreError, ClientError

from yeto.cloud import s3_store
from yeto.rl.adapters.miles.buffer_rollout import BatchReader, SampleBufferExhausted
from yeto.rl.engine.buffer_payload import GroupWriter
from yeto.rl.engine.incarnation_lease import IncarnationGate, IncarnationLease, read_lease
from yeto.rl.engine.policy_release import PolicyFollower, PolicyReleaseStore, ReleaseNotLoaded
from yeto.rl.engine.rollout_data_path import r2_classes
from yeto.rl.engine.sample_buffer import ShardFenced  # noqa: E402
from yeto.rl.engine.trainer_lease import TrainerLease  # noqa: E402
from yeto.rl.engine.sample_buffer import open_trainer_buffer, shard_index
from yeto.rl.engine.sampler_resume import (
    RoundGuard,
    SamplerMembership,
    SamplerRound,
    handoff_requested,
    resume_point,
)
from yeto.rl.engine.store_dispense import ATTEMPT_KEY, StoreDispensedPrompts
from yeto.rl.engine.task_dispense import DISPENSE_POSITION_KEY, DispenseEnded, DispenseFencedEnd

from .fault_s3 import ActorClock, Crash, FaultyS3Client

if TYPE_CHECKING:
    from .sim import Simulation

URI = "s3://store-sim"
NAMESPACE = "task-dispense"
CUT_KEY = "sim-trainer-cut/LATEST.json"
STORE_ENV = {s3_store.PART_CONCURRENCY_ENV: "1", s3_store.FILE_CONCURRENCY_ENV: "1",
             s3_store.DOWNLOAD_PROCESSES_ENV: "1", s3_store.PART_CHECKSUM_ENV: "0"}
TRANSPORT_ERRORS = (BotoCoreError, ClientError)


class Blocked(BaseException):
    """A real loop would sleep here; the scheduler takes the turn back instead."""


def make_store(sim: "Simulation", actor: str, node: str, *, stage: Path | None = None) -> s3_store.S3Store:
    """The run's store for one process: the real ``S3Store`` / ``S3ObjectStore`` on
    this process's fault-injecting client. Parts are a few bytes so that small
    files still move as multipart uploads and ranged, ETag-pinned downloads."""
    client = FaultyS3Client(sim.server, actor, sim.injector, node=node)
    store = s3_store.S3Store(sim.tmp / "mount" / actor.replace("/", "-"), URI, environ=STORE_ENV,
                             client=client, stage=stage)
    object.__setattr__(store.objects.transfer, "part_bytes", sim.config.part_bytes)  # below S3's 5 MiB floor
    return store


def release_files(version: int, policy_hash: str) -> dict[str, bytes]:
    """The "weights" of a release: every file names the version and hash it belongs to."""
    out = {}
    for name in ("config.json", "weights-00000.bin", "weights-00001.bin"):
        tag = json.dumps({"version": int(version), "policy_hash": policy_hash, "file": name}, sort_keys=True)
        out[name] = (tag + "\n").encode() + bytes(160)
    return out


def file_tag(data: bytes) -> dict[str, Any]:
    return json.loads(data.split(b"\n", 1)[0])


@dataclass
class Generation:
    """A group being "generated": claimed, not yet written."""

    slot: int
    attempt: int
    rollout_id: int
    prompt_id: str
    version: int  # the version the engine served when the trial started
    policy_hash: str | None
    cursor: dict[str, int]
    lost: bool = False  # a forced reload fell inside it (served by two versions)


class SamplerProc:
    def __init__(self, sim: "Simulation", sid: str, incarnation: str, node: str) -> None:
        self.sim, self.sid, self.incarnation, self.node = sim, sid, incarnation, node
        self.actor = f"{sid}/{incarnation}"
        self.kind = "sampler"
        self.clock = ActorClock(sim.clock, sim.rng.uniform(-1, 1) * sim.config.faults.clock_skew_s)
        self.state = "new"  # new -> running -> done | fenced | dead
        self.inflight: dict[int, Generation] = {}
        self.open_rounds: dict[int, int] = {}
        self.next_rollout_id = 0
        self.watcher_alive = True
        self.next_renew = self.next_lease_poll = 0.0
        self.written = 0
        self.exit_reason: str | None = None

    # ------------------------------------------------------------------ start
    def start(self) -> None:
        sim, cfg = self.sim, self.sim.config
        self.store = make_store(sim, self.actor, self.node)
        R2Release, R2Buffer = r2_classes()
        self.releases = R2Release(self.store, cache_dir=sim.tmp / "cache" / self.actor.replace("/", "-"),
                                  clock=self.clock.time, timer=self.clock.monotonic)
        self.buffer = R2Buffer(self.store, batch_size=1, index=shard_index(self.sid), clock=self.clock.time)
        self.buffer.refresh()
        self.resume = resume_point(self.buffer.entries(), self.sid)
        self.prompts = StoreDispensedPrompts.open(
            sim.prompt_path, sim.dispense_config, store=self.store, holder=self.sid,
            incarnation=self.incarnation, input_key="prompt", label_key=None,
            groups_per_round=cfg.capacity, lease_of_holder=self._lease_of_holder,
            sleep=self._block, clock=self.clock.monotonic, renew=False, emit=self._emit)
        self.prompts.recover(self.buffer.entries())
        self.membership = SamplerMembership(self.store, self.sid, self.incarnation, self.resume, self._emit,
                                            clock=self.clock.time)
        self.lease = IncarnationLease(self.store, self.sid, self.incarnation, write_check_s=cfg.write_check_s,
                                      clock=self.clock.time, monotonic=self.clock.monotonic, buffer=self.buffer)
        self.follower = PolicyFollower(self.releases, self._load, timer=self.clock.monotonic,
                                       wall=self.clock.time, max_defer_s=cfg.reload_max_defer_s)
        self.follower.hold = lambda: bool(self.inflight)
        self.writer = GroupWriter(self.buffer, self.sid)
        self.guard = RoundGuard(self.resume)
        latest = self.releases.latest()
        self.membership.join(latest_release=None if latest is None else
                             {"seq": latest.seq, "policy_version": latest.version})
        self.lease.claim(self.buffer.entries(), earlier=self.resume.incarnations)
        self.prompts.fenced = lambda: self.lease.fenced
        self.next_rollout_id = self.resume.next_rollout_id
        self.state = "running"
        self.next_renew = self.sim.clock.now + self.prompts.renew_period()
        self.next_lease_poll = self.sim.clock.now + cfg.lease_poll_s
        if self.buffer.data_ended:  # ``rounds = 0 if buffer.data_ended``
            self._exit("FINISHED", "done")

    def _lease_of_holder(self, sampler: str) -> str | None:
        record = read_lease(self.store, sampler)
        return None if not record or record.get("incarnation") is None else str(record["incarnation"])

    def _emit(self, event: str, **fields: Any) -> None:
        if event in ("rl_dispense_expired", "rl_dispense_lost", "rl_dispense_requeue"):
            self.sim.metric(event)
        elif event == "rl_dispense_release":  # a claim given back unwritten (closing, unwritten, fenced)
            self.sim.metric(f"claims_released_{fields.get('reason') or 'other'}")

    def _block(self, seconds: float) -> None:
        raise Blocked()

    def _load(self, release: Any, directory: Path) -> None:
        """The engine load: read every file and check it belongs to this release."""
        tags = []
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file() and path.name != "RELEASE.json":
                tags.append(file_tag(path.read_bytes()))
        bad = [t for t in tags if (t["version"], t["policy_hash"]) != (release.version, release.policy_hash)]
        if bad or len(tags) != len(release_files(0, "")):
            self.sim.violation("release_files_of_another_version", actor=self.actor, seq=release.seq,
                               version=release.version, policy_hash=release.policy_hash, files=tags)
        self.sim.trace("load", actor=self.actor, seq=release.seq, version=release.version)

    # --------------------------------------------------------------- background
    def renew(self) -> None:
        """The dispenser's renew thread (``_renew_loop``: an error waits for the next period)."""
        self.next_renew = self.sim.clock.now + self.prompts.renew_period()
        try:
            self.prompts.renew()
        except Exception:  # noqa: BLE001
            self.sim.metric("renew_failed")

    def lease_poll(self) -> None:
        """The lease watcher thread (``ReclaimWatcher``: an error in ``lease.poll`` ends the thread)."""
        self.next_lease_poll = self.sim.clock.now + self.sim.config.lease_poll_s
        if not self.watcher_alive:
            return
        try:
            self.lease.poll()
        except Exception:  # noqa: BLE001
            self.watcher_alive = False
            self.sim.metric("lease_watcher_thread_died")

    # --------------------------------------------------------------- foreground
    def actions(self) -> list[Any]:
        out = [self.act_refresh_policy]
        if len(self.inflight) < self.sim.config.capacity:
            out += [self.act_claim, self.act_claim]
        out += [lambda slot=slot: self.act_finish(slot) for slot in sorted(self.inflight)]
        return out

    def act_refresh_policy(self) -> None:
        """``SamplerLoop._load_newest``."""
        try:
            loaded = self.follower.refresh()
        except ReleaseNotLoaded:
            self.sim.metric("release_not_loaded")
            return
        if loaded is not None and self.follower.last_forced:
            for gen in self.inflight.values():  # trials served by two versions are lost
                gen.lost = True

    def act_claim(self) -> None:
        """The start of a round: ``should_stop``, then the prompts (the dispenser's claims)."""
        if self.lease.fenced or handoff_requested(self.store, self.sid):
            self._stop()
            return
        rollout_id = self.next_rollout_id
        self.guard.check(rollout_id)
        try:
            items = self.prompts(rollout_id)
        except Blocked:
            return
        except DispenseFencedEnd:
            self._exit("FENCED", "fenced")
            return
        except DispenseEnded:
            self.buffer.refresh()
            try:
                self.buffer.mark_data_end()
            except ShardFenced:
                self._exit("FENCED", "fenced")
                return
            self._stop(finished=True)
            return
        if not items:
            return
        self.next_rollout_id += 1
        cursor = dict(self.prompts.cursor(rollout_id))
        release = self.follower.release
        self.open_rounds[rollout_id] = len(items)
        for item in items:
            slot = int(item.metadata[DISPENSE_POSITION_KEY])
            self.inflight[slot] = Generation(
                slot=slot, attempt=int(item.metadata[ATTEMPT_KEY]), rollout_id=rollout_id,
                prompt_id=item.prompt_id, version=self.follower.version,
                policy_hash=None if release is None else release.policy_hash, cursor=cursor)
            self.sim.trace("claim", actor=self.actor, slot=slot, attempt=int(item.metadata[ATTEMPT_KEY]),
                           version=self.follower.version)

    def act_finish(self, slot: int) -> None:
        """A group's trials ended: write it or drop it (``AgenticRound`` settle order)."""
        gen = self.inflight.pop(slot)
        self.open_rounds[gen.rollout_id] -= 1
        last = self.open_rounds[gen.rollout_id] == 0
        if last:
            del self.open_rounds[gen.rollout_id]
        reason = None
        if gen.lost:
            reason = "below_min_samples"
        elif self.lease.check_before_write():
            reason = "fenced"
        elif not self.prompts.still_holds(slot):
            reason = "dispense_lost"
        if reason is None:
            sample_id = f"{self.sid}-{self.incarnation}-r{gen.rollout_id}-{gen.prompt_id}"
            self.sim.intent(sample_id, self, gen)
            meta = {"weight_version": gen.version,
                    "sim": {"slot": slot, "attempt": gen.attempt, "sampler": self.sid,
                            "incarnation": self.incarnation, "version": gen.version,
                            "policy_hash": gen.policy_hash}}
            samples = [{"index": j, "prompt": gen.prompt_id, "tokens": [1, 2], "response_length": 1,
                        "rollout_log_probs": [0.0], "reward": float(j), "metadata": meta} for j in range(2)]
            record = SamplerRound(self.sid, self.incarnation, gen.rollout_id, data_cursor=gen.cursor,
                                  last=last, dispense_position=slot).record()
            try:
                self.writer.write(samples, group_key=f"{self.incarnation}-r{gen.rollout_id}-{gen.prompt_id}",
                                  sampler_round=record)
            except ShardFenced:
                reason = "fenced"
                self.sim.intents.pop(sample_id, None)
            else:
                self.guard.written(gen.rollout_id)
                self.written += 1
                self.sim.trace("write", actor=self.actor, slot=slot, sample_id=sample_id, version=gen.version)
        if reason is not None:
            self.sim.metric(f"group_dropped_{reason}")
            self.sim.trace("drop", actor=self.actor, slot=slot, reason=reason)
        requeue = [slot] if reason == "below_min_samples" else []
        self.prompts.written(gen.rollout_id, [slot] if reason is None else [], keep=sorted(self.inflight),
                             requeue=requeue)

    # --------------------------------------------------------------------- exit
    def _stop(self, finished: bool = False) -> None:
        """The loop returned: the tail of ``_run_on``."""
        if not self.lease.fenced:
            self.lease.poll()
        if handoff_requested(self.store, self.sid) and not self.buffer.data_ended and not self.lease.fenced:
            try:
                self.buffer.mark_data_end()
            except ShardFenced:
                pass
        if self.lease.fenced:
            self._exit("FENCED", "fenced")
        else:
            self._exit("FINISHED", "done")

    def _exit(self, reason: str, state: str) -> None:
        self.exit_reason = reason
        try:
            self.membership.leave(reason, rounds=self.next_rollout_id, policy_version=self.follower.version,
                                  buffer_end=len(self.buffer))
        finally:
            self._close(state)

    def _close(self, state: str) -> None:
        try:
            self.prompts.close()
        except Exception:  # noqa: BLE001 - "the lease returns them anyway"
            self.sim.metric("dispense_close_failed")
        self.inflight.clear()
        self.state = state
        self.sim.trace("exit", actor=self.actor, state=state, reason=self.exit_reason)

    def failed(self, error: BaseException) -> None:
        """An error left the island's loop: LEAVE as INFRA_FAILED, close, exit non-zero."""
        self.exit_reason = f"INFRA_FAILED {type(error).__name__}"
        if self.state == "new":  # died while it was being built: nothing to close
            self.state = "dead"
            return
        try:
            self.membership.leave("FENCED" if self.lease.fenced else "INFRA_FAILED",
                                  error=f"{type(error).__name__}: {error}"[:300])
        except Exception:  # noqa: BLE001 - "the original error is the one to report"
            pass
        self._close("dead")

    def killed(self) -> None:
        self.exit_reason = "killed"
        self.inflight.clear()
        self.state = "dead"

    @property
    def alive(self) -> bool:
        return self.state in ("new", "running")


class TrainerProc:
    def __init__(self, sim: "Simulation", number: int) -> None:
        self.sim, self.number = sim, number
        self.actor = f"trainer/t{number}"
        self.node = f"trainer-node-{number}"
        self.kind = "trainer"
        self.clock = ActorClock(sim.clock, sim.rng.uniform(-1, 1) * sim.config.faults.clock_skew_s)
        self.state = "new"  # new -> running -> done | dead
        self.zombie = False  # superseded by a restarted trainer, still running
        self.version = 0
        self.rollout_id = 0
        self.hashes: dict[int, str] = {}
        self.exit_reason: str | None = None

    def start(self) -> None:
        sim, cfg = self.sim, self.sim.config
        self.store = make_store(sim, self.actor, self.node, stage=sim.tmp / "stage" / f"t{self.number}")
        if os.environ.get("SIM_OWNED", "1") == "1":  # a trainer that reads shard ownership records
            TrainerLease.claim(self.store.objects, incarnation=f"t{self.number}")
        self.buffer = open_trainer_buffer(self.store, batch_size=cfg.groups_per_step, shards=sim.shards,
                                          clock=self.clock.time)
        gate = IncarnationGate(self.store, self.buffer.entries, monotonic=self.clock.monotonic,
                               owner_epoch=self.buffer.shard_epoch_at)
        self.reader = BatchReader(self.buffer, max_policy_age=cfg.max_policy_age, wait_s=cfg.trainer_wait_s,
                                  poll_s=2.0, clock=self.clock.monotonic, sleep=self._sleep, read_workers=1,
                                  prefetch=False, incarnation_gate=gate)
        self.releases = PolicyReleaseStore(self.store, keep=cfg.keep_releases, clock=self.clock.time,
                                           timer=self.clock.monotonic,
                                           cache_dir=sim.tmp / "cache" / f"trainer-t{self.number}")
        raw = self.store.objects.read(CUT_KEY)
        cut = json.loads(raw) if raw else {"watermark": 0, "version": 0, "rollout_id": 0, "hashes": {}}
        self.reader.reset(int(cut["watermark"]))
        self.version, self.rollout_id = int(cut["version"]), int(cut["rollout_id"])
        self.hashes = {int(k): v for k, v in cut["hashes"].items()}
        sim.restored(self, cut)
        self.state = "running"

    def _sleep(self, seconds: float) -> None:
        self.sim.clock.advance(seconds)
        self.sim.yield_from(self)

    def actions(self) -> list[Any]:
        return [self.act_step]

    def act_step(self) -> None:
        """One trainer round: wait for a batch, take it, "train", release, maybe cut."""
        cfg = self.sim.config
        want = cfg.groups_per_step
        if len(self.buffer.peek(want)) < want and not self.buffer.data_ended:
            return  # the reader would wait here
        start = self.buffer.read_cursor
        try:
            groups, report = self.reader.take(self.rollout_id, want, current_version=self.version)
        except SampleBufferExhausted:
            # the data ended and what was left was passed over (too old, a fenced incarnation's)
            self.sim.took(self, self.rollout_id, [], None, start=start, end=self.buffer.read_cursor)
            if not self.zombie:
                self._cut()
            self.state = "done"
            self.sim.trace("trainer_done", actor=self.actor, read=self.buffer.read_cursor)
            return
        self.sim.took(self, self.rollout_id, groups, report)
        self.rollout_id += 1
        self.version += 1
        policy_hash = hashlib.sha256(f"{self.actor}:{self.version}:{self.rollout_id}".encode()).hexdigest()[:12]
        self.hashes[self.version] = policy_hash
        if self.rollout_id % cfg.release_every == 0:
            self._release(policy_hash)
        if self.rollout_id % cfg.cut_every == 0 and not self.zombie:
            self._cut()

    def _release(self, policy_hash: str) -> None:
        staged = self.sim.tmp / "export" / f"t{self.number}" / f"v{self.version}"
        staged.mkdir(parents=True, exist_ok=True)
        for name, data in release_files(self.version, policy_hash).items():
            (staged / name).write_bytes(data)
        try:
            release = self.releases.release_staged(
                self.version, policy_hash, staged,
                position=lambda: (self.buffer.read_cursor, self.buffer.shard_positions(self.buffer.read_cursor)))
        finally:
            shutil.rmtree(staged, ignore_errors=True)
        if release is not None:
            self.sim.trace("release", actor=self.actor, seq=release.seq, version=release.version)

    def _cut(self) -> None:
        """The round cut: the consumed watermark (and what the stub needs to continue)."""
        watermark = self.reader.watermark_before(self.rollout_id)
        cut = {"watermark": watermark, "version": self.version, "rollout_id": self.rollout_id,
               "trainer": self.number, "hashes": {str(k): v for k, v in self.hashes.items()}}
        self.store.objects.write(CUT_KEY, json.dumps(cut, sort_keys=True).encode())
        self.sim.trace("cut", actor=self.actor, watermark=watermark, rollout_id=self.rollout_id)

    def failed(self, error: BaseException) -> None:
        self.exit_reason = f"{type(error).__name__}"
        self.state = "dead"

    def killed(self) -> None:
        self.exit_reason = "killed"
        self.state = "dead"

    @property
    def alive(self) -> bool:
        return self.state in ("new", "running")


__all__ = ["Blocked", "Crash", "CUT_KEY", "NAMESPACE", "SamplerProc", "TrainerProc", "TRANSPORT_ERRORS", "URI",
           "file_tag", "make_store", "release_files"]
