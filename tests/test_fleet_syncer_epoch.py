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


def test_a_status_reader_without_a_syncer_says_why():
    with pytest.raises(ValueError, match="no syncer to read status.json from"):
        syncer_mod.syncer_status_reader(ssh_host=None)
