"""The participants of a simulated run: real yeto store code, stub compute.

:class:`SamplerProc` is one sampler process (one incarnation of a shard). It
builds the real objects in the order of ``engine_sampler_entry._run_on``: shard
buffer, resume point, store dispenser (``recover``), membership, incarnation
lease, policy follower, the round and the loop. What the simulation adds is
when things happen: a claimed task is a :class:`Generation` that ends some
scheduler steps later. Ending it is the product's own code:

* ``agentic`` samplers (the default) end one group at a time with
  ``AgenticRound._settle`` (trial results are stubs; the version filter, the
  write decision, the group write and the dispenser report are real);
* ``engine`` samplers end a whole round at once with ``EngineRound._write_scored``.

The served version is what the stub engine loaded: the content tag of the release
files the follower handed to ``load``, never a number the code under test reports.

:class:`TrainerProc` is one trainer process: ``open_trainer_buffer`` (the shard
merge), ``BatchReader`` with an ``IncarnationGate`` and ``max_policy_age``, a
fake train step, ``PolicyReleaseStore.release_staged`` through a staged
``S3Store``, and a cut record holding the buffer watermark that a restarted
trainer resets to.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from botocore.exceptions import BotoCoreError, ClientError

from yeto.cloud import s3_store
from yeto.rl.adapters.miles.buffer_rollout import BatchReader, SampleBufferExhausted
from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.engine import sample_buffer
from yeto.rl.agentic import Outcome
from yeto.rl.engine.agentic_round import AgenticRound, _Group, _RoundState, _Tally
from yeto.rl.engine.buffer_payload import GroupWriter, encode_group
from yeto.rl.engine.engine_sampler import EngineRound, PromptItem, unique_prompt_ids
from yeto.rl.engine.incarnation_lease import IncarnationGate, IncarnationLease, read_lease
from yeto.rl.engine.policy_release import PolicyFollower, PolicyReleaseStore
from yeto.rl.engine.rollout_data_path import r2_classes
from yeto.rl.engine.sample_buffer import open_trainer_buffer, shard_index
from yeto.rl.engine.sampler_engine import Provenance, SamplingParams
from yeto.rl.engine.sampler_loop import SamplerLoop
from yeto.rl.engine.sampler_resume import ReclaimWatcher, SamplerMembership, handoff_requested, resume_point
from yeto.rl.engine.store_dispense import ATTEMPT_KEY, StoreDispensedPrompts
from yeto.rl.engine.task_dispense import DISPENSE_POSITION_KEY, DispenseEnded, DispenseFencedEnd

from .fault_s3 import ActorClock, Crash, FaultyS3Client

try:  # the trainer generation lease, where this tree has it
    from yeto.rl.engine import trainer_lease
except ImportError:
    trainer_lease = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .sim import Simulation

URI = "s3://store-sim"
NAMESPACE = "task-dispense"
CUT_KEY = "sim-trainer-cut/LATEST.json"
STORE_ENV = {s3_store.PART_CONCURRENCY_ENV: "1", s3_store.FILE_CONCURRENCY_ENV: "1",
             s3_store.DOWNLOAD_PROCESSES_ENV: "1", s3_store.PART_CHECKSUM_ENV: "0"}
TRANSPORT_ERRORS = (BotoCoreError, ClientError)


SIM_KEY = "sim"  # item / sample metadata: what the simulation knows about the group


class SimEngine(FakeEngine):
    """Generates nothing here; it serves what :meth:`SamplerProc._load` loaded."""

    served: int = 0

    def served_version(self) -> int:
        return self.served


class SimWriter(GroupWriter):
    """The real writer; the simulation notes the truth at the moment of each write."""

    def __init__(self, buffer: Any, sampler_id: str, proc: "SamplerProc") -> None:
        super().__init__(buffer, sampler_id)
        self.proc = proc

    def write(self, samples: Any, *, group_key: str, provenance: Any = None, sampler_round: Any = None) -> Any:
        slot = int((sampler_round or {})["dispense_position"])
        sample_id = f"{self.sampler_id}-{group_key}"
        self.proc.sim.intent(sample_id, self.proc, self.proc.gens[slot])
        entry = super().write(samples, group_key=group_key, provenance=provenance, sampler_round=sampler_round)
        self.proc.wrote.add(slot)
        self.proc.sim.trace("write", actor=self.proc.actor, slot=slot, sample_id=sample_id)
        return entry


def _trial(gen: "Generation", j: int, turn_version: int) -> Any:
    """A finished agentic trial, as ``AgenticRound._settle`` and ``agentic_sample`` read it."""
    task = SimpleNamespace(task_id=f"t{gen.slot}", trajectory_id=f"{gen.item.prompt_id}-{j}")
    trajectory = SimpleNamespace(
        outcome=Outcome.OK, reason="", reward=float(j), task=task,
        to_metadata=lambda: {"yeto_agent_outcome": "ok", "outcome_reason": "", "eval_report": None,
                             "agent_metrics": {}, "trial_dir": None})
    segments = SimpleNamespace(prompt_ids=[1], response_ids=[2], response_logprobs=[0.0], response_mask=[1],
                               truncated=False, rebased_turns=0)
    return SimpleNamespace(trajectory=trajectory, segments=segments,
                           turns=[SimpleNamespace(policy_version=turn_version, finish_reason="stop")])


def _accepts(call: Any, name: str) -> bool:
    """Whether ``call`` takes the keyword ``name`` (a capability of a fix that is merged or not)."""
    try:
        return name in inspect.signature(call).parameters
    except (TypeError, ValueError):
        return False


# a write refused by the shard's ownership records (only where the buffer has them)
SHARD_FENCED: tuple[type[BaseException], ...] = tuple(
    c for c in (getattr(sample_buffer, "ShardFenced", None),) if c is not None)
WATCHER_SURVIVES_A_FAILED_POLL = hasattr(ReclaimWatcher(lambda: False, lambda: None), "poll_failures")


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
    item: Any  # the PromptItem the dispenser gave out
    version: int  # the version the engine served when the trial started
    policy_hash: str | None
    cursor: dict[str, int]
    group: Any = None  # the round's own record of it (agentic: its ``_Group``)
    owned_at_check: bool = True  # the ledger's truth when its write decision began


class SamplerProc:
    def __init__(self, sim: "Simulation", sid: str, incarnation: str, node: str, *, legacy: bool = False,
                 mode: str = "agentic") -> None:
        self.sim, self.sid, self.incarnation, self.node = sim, sid, incarnation, node
        self.mode = mode  # "agentic": groups end one by one; "engine": a round ends whole
        self.namespace = sim.namespace  # the dispenser this process was started with
        self.gens: dict[int, Generation] = {}  # slot -> the generation being settled or in flight
        self.wrote: set[int] = set()  # slots whose group this process wrote
        self.engine = SimEngine()
        self.served: dict[str, Any] = {"version": 0, "policy_hash": None, "path": None}  # what was loaded
        self.actor = f"{sid}/{incarnation}"
        self.kind = "sampler"
        self.legacy = legacy  # runs the rules of the code before shard ownership records
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
            sim.prompt_path, {**sim.dispense_config, "namespace": self.namespace}, store=self.store,
            holder=self.sid,
            incarnation=self.incarnation, input_key="prompt", label_key=None,
            groups_per_round=cfg.capacity, lease_of_holder=self._lease_of_holder,
            sleep=self._block, clock=self.clock.monotonic, renew=False, emit=self._emit)
        self.prompts.recover(self.buffer.entries())
        self.membership = SamplerMembership(self.store, self.sid, self.incarnation, self.resume, self._emit,
                                            clock=self.clock.time)
        lease_kwargs: dict[str, Any] = {}
        if _accepts(IncarnationLease, "buffer") and not self.legacy:  # shard ownership records are the fence
            lease_kwargs["buffer"] = self.buffer
        self.lease = IncarnationLease(self.store, self.sid, self.incarnation, write_check_s=cfg.write_check_s,
                                      clock=self.clock.time, monotonic=self.clock.monotonic, **lease_kwargs)
        self.follower = PolicyFollower(self.releases, self._load, timer=self.clock.monotonic,
                                       wall=self.clock.time, max_defer_s=cfg.reload_max_defer_s)
        self.follower.hold = lambda: bool(self.inflight)
        # the real round and loop objects: the write decision, the group write, the
        # dispenser report and the loop's store polls are theirs
        sampling = SamplingParams(max_tokens=4, n=2)
        self.provenance = Provenance.of(self.engine.identity, sampler_id=self.sid, sampling=sampling)
        common = dict(prompts=self.prompts, sampling=sampling, provenance=self.provenance, emit=self._emit,
                      incarnation=self.incarnation, resume=self.resume, data_cursor=self.prompts.cursor,
                      reclaimed=self._fenced)
        writer = SimWriter(self.buffer, self.sid, self)
        if self.mode == "agentic":
            self.round = AgenticRound(self.engine, writer, recorder=None, harness=None,
                                      max_in_flight=cfg.capacity, max_response_len=8, loop=object(),
                                      weight_switch="drain", clock=self.clock.monotonic, **common)
        else:
            self.round = EngineRound(self.engine, writer, reward=lambda item, completion: 0.0, **common)
        if hasattr(self.round, "sleep"):  # a write that waits for a store read: the others go on
            self.round.sleep = self._wait
        self.loop = SamplerLoop(follower=self.follower, buffer=self.buffer, generate=self.round,
                                num_rollouts=None, max_backlog_groups=1 << 30, emit=self._emit, poll_s=5.0,
                                sleep=self._block, should_stop=self._should_stop, clock=self.clock.monotonic)
        latest = self.releases.latest()
        self.membership.join(latest_release=None if latest is None else
                             {"seq": latest.seq, "policy_version": latest.version})
        self.lease.claim(self.buffer.entries(), earlier=self.resume.incarnations)
        self.round.write_fenced = self.lease.check_before_write
        if hasattr(self.round, "write_unknown") and hasattr(self.lease, "unknown"):
            self.round.write_unknown = self.lease.unknown
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
        if event in ("rl_dispense_expired", "rl_dispense_lost", "rl_dispense_requeue",
                     "rl_sampler_store_read_failed", "rl_sampler_write_deferred"):
            self.sim.metric(event)
        elif event == "rl_dispense_release":  # a claim given back unwritten (closing, unwritten, fenced)
            self.sim.metric(f"claims_released_{fields.get('reason') or 'other'}")

    def _block(self, seconds: float) -> None:
        raise Blocked()

    def _wait(self, seconds: float) -> None:
        self.sim.clock.advance(seconds)
        self.sim.yield_from(self)

    def _fenced(self) -> bool:
        return self.lease.fenced or bool(getattr(self.buffer, "fenced", False))

    def _handed_off(self) -> bool:
        if _accepts(handoff_requested, "on_error"):  # a failed read is no answer
            return handoff_requested(self.store, self.sid,
                                     on_error=lambda error: self.sim.metric("handoff_unread"))
        return handoff_requested(self.store, self.sid)

    def _should_stop(self) -> bool:
        return self._fenced() or self._handed_off()

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
        truth = tags[0] if tags else {"version": release.version, "policy_hash": release.policy_hash}
        self.served = {"version": int(truth["version"]), "policy_hash": truth["policy_hash"], "path": release.path}
        self.engine.served = self.served["version"]
        self.sim.loaded(self, release)
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
        """The lease watcher thread (``ReclaimWatcher``). Where the watcher counts a
        failed poll (``poll_failures``) it polls again; before that an error in
        ``lease.poll`` ended the thread."""
        self.next_lease_poll = self.sim.clock.now + self.sim.config.lease_poll_s
        if not self.watcher_alive:
            return
        try:
            self.lease.poll()
        except Exception:  # noqa: BLE001
            if WATCHER_SURVIVES_A_FAILED_POLL:
                self.sim.metric("lease_poll_failed")
            else:
                self.watcher_alive = False
                self.sim.metric("lease_watcher_thread_died")

    # --------------------------------------------------------------- foreground
    def actions(self) -> list[Any]:
        out = [self.act_refresh_policy]
        if self.mode == "agentic":
            if len(self.inflight) < self.sim.config.capacity:
                out += [self.act_claim, self.act_claim]
            out += [lambda slot=slot: self.act_finish(slot) for slot in sorted(self.inflight)]
        elif self.inflight:
            out += [self.act_finish_round, self.act_finish_round]
        else:
            out += [self.act_claim, self.act_claim]
        return out

    def act_refresh_policy(self) -> None:
        """``SamplerLoop._load_newest`` (a release that is not loaded keeps the served one).
        An engine round takes no load while it generates (``follower.generating``)."""
        if self.mode == "engine" and self.inflight:
            return
        self.loop._load_newest()
        if self.follower.version != self.served["version"]:
            self.sim.violation("follower_reports_another_version_than_was_loaded", actor=self.actor,
                               reports=self.follower.version, loaded=self.served["version"])

    def act_claim(self) -> None:
        """One iteration of ``SamplerLoop.run`` up to the round's draw: ``should_stop``,
        the admission (its store polls), then the prompts (the dispenser's claims)."""
        rollout_id = self.next_rollout_id
        if self.loop.should_stop() or not self.loop._admit(rollout_id):
            self._stop()
            return
        self.round.guard.check(rollout_id)
        try:
            items = unique_prompt_ids(list(self.prompts(rollout_id)))
        except DispenseFencedEnd:
            self._exit("FENCED", "fenced")
            return
        except DispenseEnded:
            read = getattr(self.loop, "_read", None)  # the round's groups are written; this counts them
            read("buffer", self.buffer.refresh) if callable(read) else self.buffer.refresh()
            self.buffer.mark_data_end()
            self._stop()
            return
        if not items:
            return
        self.next_rollout_id += 1
        cursor = dict(self.prompts.cursor(rollout_id))
        if self.mode == "agentic":
            self.round._rounds[rollout_id] = _RoundState(cursor, len(items))
        for position, item in enumerate(items):
            slot, attempt = int(item.metadata[DISPENSE_POSITION_KEY]), int(item.metadata[ATTEMPT_KEY])
            truth = {"slot": slot, "attempt": attempt, "sampler": self.sid, "incarnation": self.incarnation,
                     "namespace": self.namespace, "version": self.served["version"],
                     "policy_hash": self.served["policy_hash"]}
            item = PromptItem(prompt_id=item.prompt_id, prompt=item.prompt, label=item.label,
                              metadata={**item.metadata, SIM_KEY: truth})
            gen = Generation(slot=slot, attempt=attempt, rollout_id=rollout_id, item=item,
                             version=self.served["version"], policy_hash=self.served["policy_hash"],
                             cursor=cursor)
            if self.mode == "agentic":
                gen.group = _Group(rollout_id, position, item, [f"{item.prompt_id}-{j}" for j in range(2)])
                gen.group.version = gen.version
                gen.group.start_versions = [gen.version, gen.version]
                gen.group.started_at = self.clock.monotonic()
                self.round._unsettled.append(gen.group)
            self.inflight[slot] = self.gens[slot] = gen
            self.sim.claimed(self, gen)
            self.sim.trace("claim", actor=self.actor, slot=slot, attempt=attempt, version=gen.version)

    def _check_begins(self, gen: Generation) -> None:
        gen.owned_at_check = self.sim.truth_holds(self.namespace, gen.slot, gen.attempt, self.sid,
                                                  self.incarnation, settled_counts=False)

    def _settled(self, gens: list[Generation]) -> None:
        for gen in gens:
            self.gens.pop(gen.slot, None)
            if gen.slot not in self.wrote:
                self.sim.metric("groups_not_written")
                self.sim.trace("drop", actor=self.actor, slot=gen.slot)

    def act_finish(self, slot: int) -> None:
        """A group's trials ended: ``AgenticRound._settle`` filters, decides, writes and
        reports it. A trial that a forced reload fell into is served by two versions."""
        gen = self.inflight.pop(slot)
        self._check_begins(gen)
        gen.group.results = [_trial(gen, j, self.served["version"]) for j in range(2)]
        self.round._settle(gen.group, _Tally())
        self._settled([gen])

    def act_finish_round(self) -> None:
        """The round's generation returned: ``EngineRound._write_scored`` writes its groups."""
        gens = [self.inflight[slot] for slot in sorted(self.inflight)]
        self.inflight.clear()
        scored = []
        for position, gen in enumerate(gens):
            self._check_begins(gen)
            meta = {"weight_version": gen.version, SIM_KEY: gen.item.metadata[SIM_KEY]}
            scored.append((gen.item, [
                {"index": position * 2 + j, "group_index": position, "prompt": gen.item.prompt_id,
                 "tokens": [1, 2], "response_length": 1, "rollout_log_probs": [0.0], "reward": float(j),
                 "metadata": meta} for j in range(2)]))
        self.round._write_scored(gens[0].rollout_id, gens[0].version, scored, finish={"stop": 2 * len(gens)},
                                 tokens=2 * len(gens), record=self.provenance.to_record())
        self._settled(gens)

    def rogue_write(self) -> int:
        """A writer that skips every check of the sampler (a process that passed them
        before it was fenced, or code that has none): each group in flight goes to the
        shard as a payload and a record at the shard's next position."""
        objects, written = self.store.objects, 0
        for slot, gen in sorted(self.inflight.items()):
            sample_id = f"{self.sid}-{self.incarnation}-r{gen.rollout_id}-{gen.item.prompt_id}"
            samples = [{"index": j, "prompt": gen.item.prompt_id, "tokens": [1, 2], "response_length": 1,
                        "rollout_log_probs": [0.0], "reward": float(j),
                        "metadata": {"weight_version": gen.version, SIM_KEY: gen.item.metadata[SIM_KEY]}}
                       for j in range(2)]
            ref = f"{sample_buffer.PAYLOADS}/{sample_buffer.payload_name(sample_id)}"
            prefix = sample_buffer.shard_prefix(self.sid)
            record = {"schema": sample_buffer.SCHEMA, "pos": len(objects.list(prefix)), "sample_id": sample_id,
                      "behavior_version": gen.version, "payload_ref": ref, "tokens": 2,
                      "appended_at": self.clock.time(),
                      "sampler_round": {"sampler": self.sid, "incarnation": self.incarnation,
                                        "rollout_id": gen.rollout_id, "last": False, "data_cursor": gen.cursor,
                                        "dispense_position": slot}}
            # the store dispenser's fields, as the engine sampler records them (``_round_record``):
            # the attempt, and the plan of a dispenser that names one (``StoreDispensedPrompts.plan_id``)
            if gen.attempt is not None:
                record["sampler_round"]["dispense_attempt"] = gen.attempt
            plan = getattr(self.prompts, "plan_id", None)
            if isinstance(plan, str) and plan:
                record["sampler_round"]["dispense_plan"] = plan
            self._check_begins(gen)
            self.sim.intent(sample_id, self, gen, rogue=True)
            objects.write(f"{sample_buffer.BUFFER_DIR}/{ref}", encode_group(samples))
            key = f"{prefix}{record['pos']:012d}.json"
            if objects.create(key, (json.dumps(record, sort_keys=True) + "\n").encode()):
                written += 1
                self.sim.trace("rogue_write", actor=self.actor, slot=slot, sample_id=sample_id)
            else:
                self.sim.intents.pop(sample_id, None)
        self.inflight.clear()  # they are written: this process does not write them again
        return written

    # --------------------------------------------------------------------- exit
    def _stop(self) -> None:
        """The loop returned: the tail of ``_run_on``."""
        owned = hasattr(self.lease, "owned")  # shard ownership records: a fenced incarnation ends no shard
        if owned and not self.lease.fenced:
            self.lease.poll()
        if self._handed_off() and not self.buffer.data_ended and not (owned and self._fenced()):
            try:
                self.buffer.mark_data_end()
            except SHARD_FENCED:
                pass
        if self._fenced():
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
        self.gens.clear()
        self.state = state
        self.sim.trace("exit", actor=self.actor, state=state, reason=self.exit_reason)

    def failed(self, error: BaseException) -> None:
        """An error left the island's loop: LEAVE as INFRA_FAILED, close, exit non-zero."""
        self.exit_reason = f"INFRA_FAILED {type(error).__name__}"
        if self.state == "new":  # died while it was being built: nothing to close
            self.state = "dead"
            return
        try:
            self.membership.leave("FENCED" if self._fenced() else "INFRA_FAILED",
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
        self.lease: Any = None

    def start(self) -> None:
        sim, cfg = self.sim, self.sim.config
        self.store = make_store(sim, self.actor, self.node, stage=sim.tmp / "stage" / f"t{self.number}")
        self.buffer = open_trainer_buffer(self.store, batch_size=cfg.groups_per_step, shards=sim.shards,
                                          clock=self.clock.time)
        gate_kwargs: dict[str, Any] = {}
        if _accepts(IncarnationGate.__init__, "owner_epoch"):  # as default_incarnation_gate builds it
            gate_kwargs["owner_epoch"] = getattr(self.buffer, "shard_epoch_at", None)
        gate = IncarnationGate(self.store, self.buffer.entries, monotonic=self.clock.monotonic, **gate_kwargs)
        self.reader = BatchReader(self.buffer, max_policy_age=cfg.max_policy_age, wait_s=cfg.trainer_wait_s,
                                  poll_s=2.0, clock=self.clock.monotonic, sleep=self._sleep, read_workers=1,
                                  prefetch=False, incarnation_gate=gate)
        release_kwargs: dict[str, Any] = {}
        if (trainer_lease is not None and cfg.trainer_lease
                and _accepts(PolicyReleaseStore.__init__, "lease")):
            # this trainer's generation: its releases carry it, a higher one fences it
            self.lease = trainer_lease.TrainerLease.claim(
                self.store, incarnation=f"t{self.number}", host=self.node, clock=self.clock.time,
                monotonic=self.clock.monotonic)
            release_kwargs["lease"] = self.lease
            sim.trace("trainer_generation", actor=self.actor, generation=self.lease.generation)
        self.releases = PolicyReleaseStore(self.store, keep=cfg.keep_releases, clock=self.clock.time,
                                           timer=self.clock.monotonic,
                                           cache_dir=sim.tmp / "cache" / f"trainer-t{self.number}",
                                           **release_kwargs)
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
        except SampleBufferExhausted as error:
            # the data ended and what was left was passed over (too old, a fenced incarnation's)
            self.sim.took(self, self.rollout_id, [], None, start=start, end=self.buffer.read_cursor,
                          duplicates=(getattr(error, "passed_over", None) or {}).get("duplicate_tasks"))
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
        # a trainer that lost its generation stops for good: nothing restarts it
        self.state = "fenced" if type(error).__name__ == "TrainerFenced" else "dead"

    def killed(self) -> None:
        self.exit_reason = "killed"
        self.state = "dead"

    @property
    def alive(self) -> bool:
        return self.state in ("new", "running")


__all__ = ["Blocked", "Crash", "CUT_KEY", "NAMESPACE", "SamplerProc", "TrainerProc", "TRANSPORT_ERRORS", "URI",
           "file_tag", "make_store", "release_files"]
