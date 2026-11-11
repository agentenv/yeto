"""Randomized (fuzz) test of the trainer's read-and-train chain over the sample buffer.

CPU only, no Ray, no Miles: a real ``FileSampleBuffer`` and ``BatchReader``, the real
``_BufferRound`` of the Miles reader rollout function (Miles stubbed), the real yeto
metadata hooks (``record_trained_groups`` / ``extract_rollout_metadata`` /
``record_round_metadata``) over a ``dir:`` sink, ``handle_from_metadata``, the
driver's batch acceptance (``IslandDriver._accept_batch``) and a real ``BatchLedger``.

Every seed runs a random history of 20+ rounds whose buffer mixes whole, short,
too-small, oversized, stale, fenced, quarantined and non-finite-reward groups; the
batch size G changes from round to round; releases advance the policy version
between rounds; cuts, reader resets, cut restores (a new trainer process) and
restarts from the base happen at random. After every round:

* nothing raised (the data end is the only normal stop);
* no buffer position is trained twice in the surviving history and every position
  below the committed watermark is trained or passed over for a reason; the
  watermark never moves back except at a restore;
* the trained samples are exactly the metadata's samples (padding aside) and the
  batch is exactly G * n samples;
* the ledger's states are legal and agree with the history.

``YETO_FUZZ_SEEDS`` (default 40) and ``YETO_FUZZ_ROUNDS`` (default 24) scale it.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import random
import shutil
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy
import pytest

from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles import rollout_meta_hook as hook
from yeto.rl.adapters.miles.rollout import handle_from_metadata, merge_round_metadata
from yeto.rl.engine.driver import IslandDriver
from yeto.rl.engine.ledger import BatchLedger
from yeto.rl.engine.policy_age import age_floor
from yeto.rl.engine.policy_versions import PublishedPolicies, policy_label
from yeto.rl.engine.resume import CheckpointStore, consumed_digest
from yeto.rl.engine.sample_buffer import FileSampleBuffer, shard_index
from yeto.rl.engine.xhw_guard import source_key

SEEDS = int(os.environ.get("YETO_FUZZ_SEEDS", "40"))
ROUNDS = int(os.environ.get("YETO_FUZZ_ROUNDS", "24"))
SEED_BASE = int(os.environ.get("YETO_FUZZ_SEED_BASE", "0"))
WAIT_S = 50.0  # the reader's wait (fake seconds; the samplers write while it waits)
GOOD, BAD = "good", "bad"  # sampler ids; BAD is the one a quarantine names


def _hash(version: int) -> str:
    return f"{version:064x}"


# ------------------------------------------------------------------ Miles stand-ins
class _Span(SimpleNamespace):
    pass


class FakeSample:
    """The ``miles.utils.types.Sample`` fields the yeto hooks read (``from_dict``)."""

    def __init__(self, d: dict[str, Any]) -> None:
        self.__dict__.update(d)
        self.metadata = dict(d.get("metadata") or {})
        self.weight_versions = [SimpleNamespace(spans=[_Span(**s) for s in call.get("spans", ())])
                                for call in d.get("weight_versions") or ()]
        self.status = SimpleNamespace(value=d.get("status", "completed"))
        self.remove_sample = bool(d.get("remove_sample", False))
        self.rollout_id = d.get("rollout_id")
        self.loss_mask = d.get("loss_mask")


def _plugin(monkeypatch, registry):
    stubs = {
        "miles.rollout.base_types": dict(BaseRolloutFn=object, RolloutFnConstructorInput=object,
                                         RolloutFnInput=object, RolloutFnTrainOutput=object),
        "miles.rollout.inference_rollout.inference_rollout_common": dict(InferenceRolloutFn=object),
        "miles.utils.function_registry": dict(load_function=lambda path: registry.get(path) if path else None),
        "miles.utils.types": dict(Sample=SimpleNamespace(from_dict=FakeSample)),
    }
    for name, attrs in stubs.items():
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))
    name = "yeto.rl.adapters.miles.buffer_rollout_miles"
    monkeypatch.setitem(sys.modules, name, None)
    del sys.modules[name]
    plugin = importlib.import_module(name)
    monkeypatch.setattr(plugin, "RolloutFnTrainOutput", lambda **kw: kw)
    return plugin


class FakeGate:
    """``incarnation_lease.IncarnationGate``: the sample ids a fenced incarnation wrote."""

    def __init__(self) -> None:
        self.fenced: set[str] = set()

    def verdict(self, entry, position):
        return "fenced" if entry.sample_id in self.fenced else None


class _Driver:
    """The ``IslandDriver`` state ``_accept_batch`` reads (its methods, unbound)."""

    _accept_batch = IslandDriver._accept_batch
    _with_policy_labels = IslandDriver._with_policy_labels
    _discard_over_age = IslandDriver._discard_over_age
    _finish_accept = IslandDriver._finish_accept
    _policy_age_basis = IslandDriver._policy_age_basis

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.external_policy = None
        self.eval_overlap = None
        self.published_policies = PublishedPolicies()
        self.published_version: int | None = None
        self.rollout = SimpleNamespace()  # no discard_groups: an over-age group fails the round
        self.events: list[tuple[str, dict]] = []

    def _max_policy_age(self) -> int:
        return self.limit

    def emit(self, kind, **fields):
        self.events.append((kind, fields))

    @property
    def expected_token(self) -> str:
        return policy_label(self.published_version, _hash(self.published_version))

    def publish(self, version: int) -> None:
        self.published_policies.record(version, _hash(version))
        self.published_policies.retain_from(version - self.limit)
        self.published_version = version


# ------------------------------------------------------------------ the world
@dataclass
class Truth:
    """What the fuzz wrote at a buffer position (the oracle)."""
    size: int
    version: int
    fenced: bool
    source: str
    nonfinite: bool
    malformed: bool = False  # left out of the round after the take (not a reader pass-over)


@dataclass
class Cut:
    next_rollout: int
    watermark: int
    ledger_dir: Path
    lineage: dict
    version: int


@dataclass
class World:
    seed: int
    root: Path
    n: int
    limit: int
    rng: random.Random
    truth: dict[str, Truth] = field(default_factory=dict)  # by sample id (positions: merge order)
    sharded: bool = False  # each sampler writes its own shard; the trainer merges them
    clock: float = 0.0
    # while the trainer waits: the samplers write (None: nothing, e.g. stalled)
    on_wait: Any = None

    def now(self) -> float:
        return self.clock

    def sleep(self, seconds: float) -> None:
        self.clock += seconds
        if self.on_wait is not None:
            self.on_wait()


def _sample(index: int, group: int, version: int, reward: float, length: int) -> dict[str, Any]:
    prompt = [1, 2, 3]
    return {"group_index": group, "index": index, "prompt": "p", "tokens": prompt + list(range(10, 10 + length)),
            "response_length": length, "rollout_log_probs": [-0.1] * length, "reward": reward,
            "status": "completed", "metadata": {"weight_version": str(version), "prompt_tokens_known": True}}


def _write_groups(world: World, writers: dict, gate: FakeGate, version: int, count: int) -> None:
    rng, n = world.rng, world.n
    for _ in range(count):
        roll = rng.random()
        size = n if roll < 0.55 else rng.randint(2, n - 1) if roll < 0.8 and n > 2 else \
            1 if roll < 0.88 else n + 1 if roll < 0.92 else n
        behind = rng.choice([0, 0, 0, 1, 1, world.limit, world.limit + 1, world.limit + 3])
        v = max(0, version - behind)
        sampler = BAD if rng.random() < 0.1 else GOOD
        nonfinite = rng.random() < 0.04
        rewards = [float(rng.randint(0, 1)) for _ in range(size)]
        if nonfinite:
            rewards[rng.randrange(size)] = math.nan
        position = len(world.truth)
        # a sampler that hot-updated mid-round: some samples of the group are one version newer
        mixed = v < version and rng.random() < 0.2
        samples = [_sample(1000 * position + j, rng.randint(0, 5), v + (mixed and j % 2), rewards[j],
                           rng.randint(1, 6)) for j in range(size)]
        if rng.random() < 0.03:  # an update in the middle of one generation: no per-token versions
            samples[0]["metadata"]["final_weight_version"] = str(v + 1)
        if rng.random() < 0.05:  # an engine that stopped at once: an empty response
            samples[-1].update(tokens=samples[-1]["tokens"][:3], response_length=0, rollout_log_probs=[])
        malformed = rng.random() < 0.03
        if malformed:  # an R3 routing row that mixes pads and expert ids
            routing = numpy.zeros((len(samples[0]["tokens"]), 2, 2), dtype=numpy.uint16)
            routing[0, 0, 0] = 65535
            samples[0]["rollout_routed_experts"] = routing
        writers[sampler].write(samples, group_key=f"k{position}",
                               provenance={"sampler_id": sampler, "engine": "fake"})
        entry = writers[sampler].buffer.entries()[-1]
        fenced = rng.random() < 0.05
        if fenced:
            gate.fenced.add(entry.sample_id)
        world.truth[entry.sample_id] = Truth(size, v, fenced, source_key(entry.provenance, None), nonfinite,
                                             malformed)


def _passable(t: Truth, floor: int | None, n: int, quarantined: set[str]) -> bool:
    """A reason the reader may pass this position over (consumed, never trained)."""
    return ((floor is not None and t.version < floor) or t.fenced or t.source in quarantined
            or not 2 <= t.size <= n or t.nonfinite or t.malformed)


class Trainer:
    """One trainer process: its reader, rollout function, driver view and ledger."""

    def __init__(self, world: World, monkeypatch, plugin, gate: FakeGate, ledger_dir: Path) -> None:
        self.world = world
        buffer = FileSampleBuffer(CheckpointStore(world.root / "store"), batch_size=1,
                                  merge_shards=world.sharded,
                                  expected_shards=(GOOD, BAD) if world.sharded else None)
        self.selected: list[int] = []
        select = buffer.select

        def logged_select(consumer, limit):
            out = select(consumer, limit)
            self.selected.extend(int(s.position) for s in out)
            return out

        buffer.select = logged_select
        rng = world.rng
        self.reader = br.BatchReader(buffer, max_policy_age=world.limit, wait_s=WAIT_S, poll_s=1.0,
                                     group_size=world.n, read_workers=rng.choice([1, 4]),
                                     prefetch=rng.random() < 0.5, incarnation_gate=gate,
                                     clock=world.now, sleep=world.sleep)
        self.round = plugin._BufferRound.__new__(plugin._BufferRound)
        self.round.args = SimpleNamespace(rollout_sample_filter_path="filter",
                                          rollout_all_samples_process_path="all",
                                          grpo_std_normalization=True, hf_checkpoint=None,
                                          yeto_rl_max_policy_age=world.limit)
        self.round.samples_per_group = world.n
        self.round.reader = self.reader
        self.round._tokenizer = None
        self.ledger = BatchLedger(ledger_dir)

    def close(self) -> None:
        self.reader._stop_prefetching()
        self.ledger.close()


class Run:
    """One seed: a random history of rounds, restores and restarts (module doc)."""

    def __init__(self, seed: int, tmp_path: Path, monkeypatch) -> None:
        self.seed = seed
        self.rng = rng = random.Random(seed)
        self.world = World(seed=seed, root=tmp_path / f"s{seed}", n=rng.choice([2, 3, 4, 8]),
                           limit=rng.choice([1, 2, 3]), rng=rng)
        self.sink = self.world.root / "sink"
        monkeypatch.setenv(hook.META_SINK_ENV, f"dir:{self.sink}")
        self.monkeypatch = monkeypatch
        self.plugin = _plugin(monkeypatch, {"filter": hook.record_trained_groups,
                                            "all": hook.extract_rollout_metadata})
        self.gate = FakeGate()
        self.world.sharded = rng.random() < 0.5
        store = CheckpointStore(self.world.root / "store")
        if self.world.sharded:
            self.wbuffers = [FileSampleBuffer(store, batch_size=1, index=shard_index(s)) for s in (GOOD, BAD)]
        else:
            self.wbuffers = [FileSampleBuffer(store, batch_size=1)]
        self.writers = {s: br.GroupWriter(self.wbuffers[i if self.world.sharded else 0], s)
                        for i, s in enumerate((GOOD, BAD))}
        self.state_dir = self.world.root / "state"
        self.version = 0  # the release account (continues over restores: the store's releases)
        self.ended = False
        self.stalled = False
        self.world.on_wait = self._sampler_writes_while_waiting
        self.quarantined: set[str] = set()
        self.stats = {k: 0 for k in ("rounds", "restores", "resets", "base_restarts", "crash_restores",
                                     "padded_rounds", "carried", "short_groups", "stale", "fenced",
                                     "quarantined", "dropped", "waits", "releases", "ahead_cuts")}
        self.trainer: Trainer | None = None
        self.snaps = 0
        self._new_history()

    # ---------------------------------------------------------------- the samplers
    def write(self, count: int) -> None:
        if not self.ended:
            _write_groups(self.world, self.writers, self.gate, self.version, count)

    def _sampler_writes_while_waiting(self) -> None:
        self.stats["waits"] += 1
        if not self.stalled and self.rng.random() < 0.7:
            self.write(self.rng.randint(1, 3))

    # ---------------------------------------------------------------- trainer processes
    def _new_trainer(self) -> None:
        if self.trainer is not None:
            self.trainer.close()
        self.trainer = Trainer(self.world, self.monkeypatch, self.plugin, self.gate, self.state_dir)
        self.trainer.reader.set_xhw_actions({q: {"action": "quarantine"} for q in self.quarantined})
        self.driver = _Driver(self.world.limit)
        for v in range(max(0, self.version - self.world.limit), self.version + 1):
            self.driver.publish(v)  # the account continues at the newest release vK

    def _new_history(self) -> None:
        """A start from the base: rollout 0, buffer position 0, an empty ledger."""
        self._new_trainer()
        if self.trainer.ledger.recorded_rollouts():
            self.trainer.ledger.set_aside(f"fuzz{self.stats['base_restarts']}")
        self.trainer.reader.reset(0)
        self.trainer.ledger.rebase(0)
        self.lineage: dict[int, dict[str, list[int]]] = {}
        self.cuts: list[Cut] = []
        self.committed = 0
        self.rollout_id = 0

    def restore(self) -> None:
        """A new trainer process from the newest cut (from the base without one)."""
        if not self.cuts:
            self.stats["base_restarts"] += 1
            self._new_history()
            return
        cut = self.cuts[-1]
        self.trainer.close()
        shutil.rmtree(self.state_dir)
        shutil.copytree(cut.ledger_dir, self.state_dir)
        self.trainer = None
        self._new_trainer()
        ledger = self.trainer.ledger
        if cut.next_rollout:
            last = ledger.batch(cut.next_rollout - 1)
            assert last and last["state"] == "outer_recorded", (self.seed, last)
        self.trainer.reader.reset(cut.watermark)
        ledger.rebase(cut.next_rollout)
        self.lineage = {k: dict(v) for k, v in cut.lineage.items()}
        self.committed = cut.watermark
        self.rollout_id = cut.next_rollout
        self.stats["restores"] += 1

    def check_watermark(self, watermark: int, before: int) -> None:
        trained = [p for rid, r in self.lineage.items() if rid < before for p in r["trained"]]
        assert all(p < watermark for p in trained), (self.seed, "trained past the watermark", watermark, trained)
        done = set(trained) | {p for rid, r in self.lineage.items() if rid < before for p in r["passed"]}
        missing = [p for p in range(watermark) if p not in done]
        assert not missing, (self.seed, "positions below the watermark neither trained nor passed over", missing)

    def cut(self) -> None:
        watermark = self.trainer.reader.watermark_before(self.rollout_id)
        assert watermark >= self.committed, (self.seed, "watermark moved back", self.committed, watermark)
        self.committed = watermark
        self.check_watermark(watermark, self.rollout_id)
        self.snaps += 1
        snap = self.world.root / f"cut-{self.snaps}"
        shutil.copytree(self.state_dir, snap)
        self.cuts.append(Cut(self.rollout_id, watermark, snap, {k: dict(v) for k, v in self.lineage.items()},
                             self.version))

    # ---------------------------------------------------------------- one round
    def round(self) -> bool:
        """One training round; False once the data ended."""
        rng, world, trainer, driver = self.rng, self.world, self.trainer, self.driver
        rollout_id = self.rollout_id
        G = rng.choice([1, 2, 3, 4, 6])
        trainer.round.groups = G
        hook.put_rollout_round(rollout_id)
        hook.put_policy_token(driver.expected_token)
        floor = age_floor(driver.published_version, world.limit)
        before = len(trainer.selected)
        carried_in = trainer.reader._carry[0] if trainer.reader._carry else None
        self.stalled = rng.random() < 0.05
        try:
            out = trainer.round(SimpleNamespace(evaluation=False, rollout_id=rollout_id,
                                                weight_version=driver.published_version))
        except br.SampleBufferExhausted:
            assert self.ended, (self.seed, "exhausted before the data end")
            return False
        except br.SampleBufferRolloutError as error:
            # the samplers stalled past the reader's wait: the trainer fails, a new
            # process continues from the newest cut
            assert "groups after" in str(error) and self.stalled and not self.ended, (self.seed, str(error))
            self.stats["crash_restores"] += 1
            self.restore()
            return True
        finally:
            self.stalled = False
        if rng.random() < 0.15:
            # rollout ahead: this round was taken before the cut that precedes it
            self.cut()
            self.stats["ahead_cuts"] += 1
        samples = [s for group in out["samples"] for s in group]
        real = [s for s in samples if not br.is_padding(s)]
        pads = [s for s in samples if br.is_padding(s)]
        assert len(samples) == G * world.n, (self.seed, "batch is not G * n", len(samples), G, world.n)
        assert all(len(g) <= world.n for g in out["samples"])
        assert len({s.index for s in samples}) == len(samples), (self.seed, "duplicate sample index")
        assert all(int(s.group_index) >= br.PADDING_GROUP_BASE for s in pads)
        assert all(len(s.loss_mask or [1] * s.response_length) == s.response_length for s in samples)
        assert all(math.isfinite(s.reward) for s in real), (self.seed, "a non-finite reward reaches Miles")
        self.stats["padded_rounds"] += bool(pads)
        main = json.loads((self.sink / f"rollout-{rollout_id}.json").read_text())
        extra = json.loads((self.sink / f"round-{rollout_id}.json").read_text())
        (self.sink / f"rollout-{rollout_id}.json").unlink()
        (self.sink / f"round-{rollout_id}.json").unlink()
        payload = merge_round_metadata(main, [extra], rollout_id)
        handle = handle_from_metadata(payload, rollout_id=rollout_id, policy_version=driver.published_version,
                                      policy_hash=_hash(driver.published_version),
                                      data_pack=SimpleNamespace(sample_indices=[s.index for s in samples]))
        assert sorted(f"g{s.group_index}/s{s.index}" for s in real) == sorted(
            f"{g.group_id}/{sid}" for g in handle.groups for sid in g.sample_ids), (self.seed, "metadata != batch")
        batch = driver._accept_batch(rollout_id, handle)
        assert len(batch.groups) == len(handle.groups), (self.seed, "the driver dropped groups")
        # positions: each trained once in the surviving history; the others passed over for a reason
        trained = sorted({int(s.group_index) for s in real})
        taken = set(trainer.selected[before:]) | ({carried_in} if carried_in is not None else set())
        carry_out = trainer.reader._carry[0] if trainer.reader._carry else None
        passed = sorted(taken - set(trained) - ({carry_out} if carry_out is not None else set()))
        earlier = {p for r in self.lineage.values() for p in r["trained"]}
        assert not earlier & set(trained), (self.seed, rollout_id, "trained twice", sorted(earlier & set(trained)))
        entries = trainer.reader.buffer.entries()
        for p in trained:
            t = world.truth[entries[p].sample_id]
            quarantined = self.quarantined if p != carried_in else set()  # selected before the action
            assert not _passable(t, floor, world.n, quarantined), (self.seed, rollout_id, "trained", p, t, floor)
        for p in passed:
            t = world.truth[entries[p].sample_id]
            assert _passable(t, floor, world.n, self.quarantined), (self.seed, rollout_id, "passed over", p, t, floor)
        ledger = trainer.ledger
        ledger.prepare(batch, policy_token=batch.groups[0].policy_token)
        ledger.optimizer_applied(rollout_id)
        ledger.outer_recorded(rollout_id, next_policy_version=rollout_id + 1)
        assert ledger.state(rollout_id) == "outer_recorded" and ledger.unconsumed() == []
        self.lineage[rollout_id] = {"trained": trained, "passed": passed}
        expected = [f"g{p}" for r in self.lineage.values() for p in r["trained"]]
        assert sorted(ledger._consumed_groups) == sorted(expected), (self.seed, "ledger != history")
        assert consumed_digest(ledger._consumed_groups) == consumed_digest(expected)
        metrics = out["metrics"]
        self.stats["rounds"] += 1
        for key, name in (("carried", "carried_over_groups"), ("short_groups", "incomplete_groups"),
                          ("stale", "stale_groups"), ("fenced", "stale_incarnation_groups"),
                          ("quarantined", "xhw_quarantined_groups"), ("dropped", "dropped_small_groups")):
            self.stats[key] += int(metrics[f"sample_buffer/{name}"])
        self.rollout_id += 1
        return True

    def run(self, rounds: int) -> dict[str, int]:
        rng = self.rng
        for step in range(rounds):
            self.write(rng.choice([0, 1, 2, 3, 4, 6, 10]))
            if step == rounds - 4 and not self.ended:
                for buffer in self.wbuffers:
                    buffer.mark_data_end()
                self.ended = True
            event = rng.random()
            if self.cuts and event < 0.08:
                self.restore()
            elif event < 0.12 and step < rounds - 6:
                self.stats["base_restarts"] += 1
                self._new_history()
            if rng.random() < 0.15:  # the xhw guard quarantines (or frees) the bad sampler
                self.quarantined = set() if self.quarantined else {source_key({"sampler_id": BAD, "engine": "fake"})}
                self.trainer.reader.set_xhw_actions({q: {"action": "quarantine"} for q in self.quarantined})
            if rng.random() < 0.3:
                self.cut()
                if self.trainer.reader._carry is not None or rng.random() < 0.2:
                    # the reader goes back to the cut just taken (same process): a carried
                    # group is dropped and selected again
                    self.trainer.reader.reset(self.cuts[-1].watermark)
                    self.stats["resets"] += 1
            if not self.round():
                break
            if rng.random() < 0.85:  # a release between rounds: the next round trains a newer policy
                self.version += 1
                self.driver.publish(self.version)
                self.stats["releases"] += 1
        self.cut()
        self.trainer.close()
        return self.stats


def _run_seed(seed: int, tmp_path: Path, monkeypatch) -> dict[str, int]:
    return Run(seed, tmp_path, monkeypatch).run(ROUNDS)


def test_trainer_read_chain_fuzz(tmp_path, monkeypatch):
    totals: dict[str, int] = {}
    for seed in range(SEED_BASE, SEED_BASE + SEEDS):
        with monkeypatch.context() as m:
            stats = _run_seed(seed, tmp_path, m)
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        shutil.rmtree(tmp_path / f"s{seed}", ignore_errors=True)
    print(f"[fuzz] seeds={SEEDS} rounds/seed={ROUNDS} totals={totals}")
    assert totals["rounds"] > SEEDS * 5
    for key in ("restores", "resets", "base_restarts", "crash_restores", "padded_rounds", "carried",
                "short_groups", "stale",
                "fenced", "quarantined", "dropped"):
        assert totals[key] > 0, (key, totals)
