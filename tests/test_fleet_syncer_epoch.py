"""The syncer_epoch of a fleet launched island by island, from the syncer launch to each
island's HELLO: the syncer launch records the term without waiting for the syncer's
status.json (the Rust syncer writes it only once its first learners are in, and the
island launches wait for the record), every island launch takes the record's term (it
never reads status.json: it has no syncer cluster), and a syncer launch adopting a
running syncer reads its term from status.json. A fake syncer: status.json after the
first HELLO only, a HELLO below its term fenced (no cloud, no Rust)."""

from __future__ import annotations

import json
import re
import subprocess
import time
import types

import pytest

from test_fleet_island_launches import _prepared, example
from test_rl_algorithm_provenance import _fake_sky  # noqa: F401 (fixture)
from test_rl_launcher_island_failure import Clock, _setup
from yeto import launcher
from yeto.launch import fleet_syncer
from yeto.launch import syncer as syncer_mod
from yeto.rl import fleet, island_launches
from yeto.rl.engine.island_status import STATUS_SCHEMA
from yeto.rl.engine.object_store import DirectoryObjectStore


@pytest.fixture(autouse=True)
def _island_env(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "fleet-t5b-test-key")
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")


class LateStatusSyncer:
    """A running syncer at term ``term``: its status.json exists only once a learner's
    HELLO is in (the Rust syncer's scheduler), read as ``ssh <syncer cluster> cat``."""

    def __init__(self, term: int, *, joined: bool = False) -> None:
        self.term, self.joined, self.reads = term, joined, 0

    def run(self, argv, **kw):  # subprocess.run of the status reader
        if not all(isinstance(a, str) for a in argv):
            raise TypeError("expected str, bytes or os.PathLike object, not NoneType")
        self.reads += 1
        if not self.joined:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="cat: status.json: No such file")
        return types.SimpleNamespace(returncode=0, stdout=json.dumps({"schema": STATUS_SCHEMA,
                                                                       "syncer_epoch": self.term}), stderr="")

    def hello(self, epoch: int) -> None:
        if epoch < self.term:
            raise AssertionError(f"fenced: syncer_epoch {epoch} < {self.term}")
        self.joined = True


@pytest.fixture
def fleet_run(tmp_path, monkeypatch):
    """The per-island example fleet, its run store on a local directory, the syncer
    launch's args and the syncer's start term (its --syncer-epoch)."""
    cfg = example(tmp_path)
    objects = DirectoryObjectStore(tmp_path / "store")
    monkeypatch.setattr(fleet_syncer, "_store_objects", lambda args: objects)
    args = _prepared([*cfg["trainer"]["argv"], "--controller", "local"])
    assert args.rl_fleet_role == "syncer" and syncer_mod.island_pool(args) and args.rl_syncer_epoch is None
    term = int(re.search(r"--syncer-epoch (\d+)", syncer_mod._syncer_island_pool(args)).group(1))
    return cfg, objects, args, term


def _wire(monkeypatch, syncer: LateStatusSyncer) -> None:
    monkeypatch.setattr(syncer_mod, "subprocess",
                        types.SimpleNamespace(run=syncer.run, TimeoutExpired=subprocess.TimeoutExpired))


def _syncer_launch(args, objects, *, adopted: bool, timeout_s: float | None = None) -> dict:
    launcher._resolve_launch_syncer_epoch(args, member_role="syncer", no_sync=False, head_mode=False,
                                          syncer_cluster="r-syncer", adopted=adopted, timeout_s=timeout_s)
    return fleet_syncer.record(args, "10.0.0.9:29400", 2, objects=objects)


def _islands_join(cfg, objects, syncer: LateStatusSyncer) -> list[int]:
    """Each island launch as the trainer worker starts it (the syncer record's argv), up to
    its epoch step, then its HELLO."""
    record = island_launches.read_syncer_record(objects)
    epochs = []
    for member in island_launches.trainer_members(cfg, fleet.plan_of(cfg)):
        args = _prepared(island_launches.launch_argv(member, syncer=record))
        assert args.rl_fleet_role == "island" and syncer_mod.island_pool(args)
        launcher._resolve_launch_syncer_epoch(args, member_role="island", no_sync=False, head_mode=False,
                                              syncer_cluster=None, adopted=False)
        syncer.hello(syncer_mod.island_syncer_epoch(args))
        epochs.append(args.rl_syncer_epoch)
    return epochs


def test_a_fresh_per_island_fleet_comes_up_without_waiting_for_status_json(fleet_run, monkeypatch):
    cfg, objects, args, term = fleet_run
    syncer = LateStatusSyncer(term)
    _wire(monkeypatch, syncer)
    began = time.monotonic()
    record = _syncer_launch(args, objects, adopted=False)
    assert time.monotonic() - began < 5 and syncer.reads == 0  # nothing waits on a status.json not yet there
    assert record["syncer_epoch"] == term
    assert _islands_join(cfg, objects, syncer) == [term, term]
    assert syncer.reads == 0 and syncer.joined


def test_an_adopted_syncer_s_term_is_read_from_its_status_json(fleet_run, monkeypatch):
    cfg, objects, args, _ = fleet_run
    syncer = LateStatusSyncer(5, joined=True)  # running, its learners in: status.json says term 5
    _wire(monkeypatch, syncer)
    record = _syncer_launch(args, objects, adopted=True)
    assert syncer.reads >= 1 and record["syncer_epoch"] == 5
    assert _islands_join(cfg, objects, syncer) == [5, 5]


def test_an_adopted_syncer_without_status_json_keeps_the_fleet_s_recorded_term(fleet_run, monkeypatch):
    """An adopted syncer no learner has joined yet (no status.json): the term the fleet's
    last syncer record gave, never lower than the launch's own --syncer-epoch."""
    cfg, objects, args, term = fleet_run
    island_launches.write_syncer_record(objects, addr="10.0.0.9:29400", syncer_epoch=term + 2, learners=2,
                                        max_learners=4, run="r")
    syncer = LateStatusSyncer(term + 2)
    _wire(monkeypatch, syncer)
    record = _syncer_launch(args, objects, adopted=True, timeout_s=0.0)
    assert record["syncer_epoch"] == term + 2
    assert _islands_join(cfg, objects, syncer) == [term + 2, term + 2]


def test_an_island_launch_from_a_record_without_an_epoch_joins_at_0_and_reads_nothing(fleet_run, monkeypatch):
    cfg, objects, _, _ = fleet_run
    island_launches.write_syncer_record(objects, addr="10.0.0.9:29400", syncer_epoch=None, learners=2,
                                        max_learners=4, run="r")
    syncer = LateStatusSyncer(0)
    _wire(monkeypatch, syncer)
    assert _islands_join(cfg, objects, syncer) == [0, 0] and syncer.reads == 0


def test_the_syncer_launch_of_a_fresh_fleet_records_its_term_without_reading_status_json(tmp_path, monkeypatch):
    """launcher.run of a fleet's syncer launch (fake sky): its syncer just started (nothing
    adopted), so the record it serves carries the syncer's start term and status.json is
    never read (it would not appear before the islands this record starts)."""
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    monkeypatch.setattr(launcher, "prepare_run_code_pack", lambda args: None)  # no git here
    objects = DirectoryObjectStore(tmp_path / "store")
    monkeypatch.setattr(fleet_syncer, "_store_objects", lambda args: objects)
    syncer = LateStatusSyncer(0)
    _wire(monkeypatch, syncer)
    served = []
    monkeypatch.setattr(fleet_syncer, "serve", lambda args, addr, n, **kw: served.append(
        fleet_syncer.record(args, addr, n, objects=objects)) or 0)
    from yeto.cli import parse_args

    args = parse_args([*example(tmp_path)["trainer"]["argv"], "--controller", "local"])
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    args.keep, args.recover_timeout, args.controller_poll = False, 0.0, 30.0
    began = time.monotonic()
    assert launcher.run(args) == 0
    assert time.monotonic() - began < 30 and syncer.reads == 0
    assert [r["syncer_epoch"] for r in served] == [0]


class FlakyStore(DirectoryObjectStore):
    """A run store whose reads fail ``failures`` times first."""

    def __init__(self, root, failures: int) -> None:
        super().__init__(root)
        self.failures = failures

    def read(self, key):
        if self.failures > 0:
            self.failures -= 1
            raise OSError("the run store does not answer")
        return super().read(key)


@pytest.fixture
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(fleet_syncer, "RECORD_READ_RETRY_S", 0.0)


def test_a_record_read_that_fails_is_retried_and_never_lowers_the_term(fleet_run, tmp_path, monkeypatch,
                                                                       no_retry_wait):
    cfg, _, args, term = fleet_run
    flaky = FlakyStore(tmp_path / "flaky", failures=fleet_syncer.RECORD_READ_ATTEMPTS - 1)
    island_launches.write_syncer_record(DirectoryObjectStore(tmp_path / "flaky"), addr="10.0.0.9:29400",
                                        syncer_epoch=term + 3, learners=2, max_learners=4, run="r")
    _wire(monkeypatch, LateStatusSyncer(term + 3))
    monkeypatch.setattr(fleet_syncer, "_store_objects", lambda args: flaky)
    assert _syncer_launch(args, flaky, adopted=False)["syncer_epoch"] == term + 3  # answered on the last try
    down = FlakyStore(tmp_path / "flaky", failures=fleet_syncer.RECORD_READ_ATTEMPTS)
    monkeypatch.setattr(fleet_syncer, "_store_objects", lambda args: down)
    args.rl_syncer_epoch = None
    with pytest.raises(fleet_syncer.SyncerRecordUnreadable, match="never written lower"):
        _syncer_launch(args, down, adopted=False)
    assert island_launches.read_syncer_record(DirectoryObjectStore(tmp_path / "flaky"))["syncer_epoch"] == term + 3


def test_a_store_that_raises_not_found_has_no_record(fleet_run, monkeypatch):
    _, _, args, term = fleet_run

    class NotThere:
        def read(self, key):
            raise FileNotFoundError(key)

    assert fleet_syncer._recorded_epoch(args, NotThere(), log=lambda m: None) is None
    monkeypatch.setattr(fleet_syncer, "_store_objects", lambda args: NotThere())
    assert fleet_syncer.resolve_epoch(args, "syncer", adopted=False, reader=None, log=lambda m: None) == term


def test_a_record_that_does_not_parse_is_refused_not_overwritten(fleet_run, monkeypatch):
    _, objects, args, _ = fleet_run
    objects.write(island_launches.SYNCER_RECORD, b"{truncated")
    _wire(monkeypatch, LateStatusSyncer(0))
    with pytest.raises(fleet_syncer.SyncerRecordUnreadable, match="does not parse"):
        _syncer_launch(args, objects, adopted=False)
    assert objects.read(island_launches.SYNCER_RECORD) == b"{truncated"


def test_an_explicit_or_adopted_term_below_the_record_is_raised_to_it(fleet_run, monkeypatch):
    cfg, objects, args, term = fleet_run
    island_launches.write_syncer_record(objects, addr="10.0.0.9:29400", syncer_epoch=term + 4, learners=2,
                                        max_learners=4, run="r")
    syncer = LateStatusSyncer(term + 4)
    _wire(monkeypatch, syncer)
    args.rl_syncer_epoch = term + 1  # an explicit --rl-syncer-epoch below the fleet's record
    assert _syncer_launch(args, objects, adopted=False)["syncer_epoch"] == term + 4
    args.rl_syncer_epoch = None
    syncer.joined = True  # an adopted syncer that reports a lower term than the record (record kept)
    syncer.term = term
    assert _syncer_launch(args, objects, adopted=True)["syncer_epoch"] == term + 4


def test_a_syncer_record_that_does_not_read_fails_the_syncer_launch_before_its_syncer_is_ordered(
        tmp_path, monkeypatch, no_retry_wait):
    import sys

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    monkeypatch.setattr(launcher, "prepare_run_code_pack", lambda args: None)
    broken = DirectoryObjectStore(tmp_path / "store")
    broken.write(island_launches.SYNCER_RECORD, b"{truncated")
    monkeypatch.setattr(fleet_syncer, "_store_objects", lambda args: broken)
    ordered = []
    sky = sys.modules["sky"]
    real = sky.launch
    monkeypatch.setattr(sky, "launch", lambda task, **kw: ordered.append(kw.get("cluster_name")) or real(task, **kw))
    from yeto.cli import parse_args

    args = parse_args([*example(tmp_path)["trainer"]["argv"], "--controller", "local"])
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    args.keep, args.recover_timeout, args.controller_poll = False, 0.0, 30.0
    with pytest.raises(fleet_syncer.SyncerRecordUnreadable, match="does not parse"):
        launcher.run(args)
    assert ordered == []  # no syncer VM ordered


def test_a_status_reader_without_a_syncer_says_why():
    with pytest.raises(ValueError, match="no syncer to read status.json from"):
        syncer_mod.syncer_status_reader(ssh_host=None)


def test_the_syncer_vm_a_fleet_syncer_launch_orders_is_priced_in_its_ledger(tmp_path, monkeypatch):
    """The syncer launch's own sky VM (AWS) is recorded in the run's resource ledger with the
    price table's syncer row of its cloud: run.budget and `yeto cost` bill it from its order
    to its confirmed release, never as unpriced (0 in the run's spend)."""
    from yeto import runs
    from yeto.cost import prices
    from yeto.cost.billing import Rates, resource_costs
    from yeto.cost.ledger import ResourceLedger

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), clock=clock, record=record)
    monkeypatch.setattr(launcher, "prepare_run_code_pack", lambda args: None)  # no git here
    objects = DirectoryObjectStore(tmp_path / "store")
    monkeypatch.setattr(fleet_syncer, "_store_objects", lambda args: objects)
    _wire(monkeypatch, LateStatusSyncer(0))
    monkeypatch.setattr(fleet_syncer, "serve", lambda args, addr, n, **kw: 0)
    from yeto.cli import parse_args

    args = parse_args([*example(tmp_path)["trainer"]["argv"], "--controller", "local"])
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    args.keep, args.recover_timeout, args.controller_poll = False, 0.0, 30.0
    assert launcher.run(args) == 0
    ledgers = sorted(runs.RUNS_DIR.glob("*/resources.jsonl"))
    syncers = [r for path in ledgers for r in ResourceLedger(path).resources()
               if r.kind == "sky_cluster" and r.attrs.get("role") == "syncer"]
    assert len(syncers) == 1
    vm = syncers[0]
    assert vm.attrs["cloud"] == "aws" and vm.attrs["usd_per_hour"] == prices.syncer_usd_h("aws") > 0
    (item,) = resource_costs([vm], vm.created_at + 3600.0, rates=Rates()).items
    assert item.priced and item.rate_usd_per_hour == pytest.approx(prices.syncer_usd_h("aws"))
