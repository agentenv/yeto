"""Strict in-place restart resumes the rollout data source (rl-infra-spec 3.6/3.7), CPU.

GPU evidence a4s8-2r2 r6 (2026-10-02): the learner was killed at QUIESCING with the
syncer at version 2; every restart attempt rebased the ledger to 2 correctly but the
restarted rollout process drew its prompts from offset 0 again, so rollout 2 came back
with Miles' group ids ``g0..g3`` and ``BatchLedger.prepare`` refused it (``were already
trained in rollout 0``) -> the island never trained again. Fakes name groups
``r<rollout>-g<i>`` and report no cursor, so the C1/C3' tests never saw it.

``MilesLikePool`` wraps the elastic fake with Miles' real identity rule: group ids are
the data source's monotonic ``sample_group_index`` and a *fresh pool instance* is a
restarted rollout process (counter back to 0). The ledger now records each batch's
``data_cursor`` and the driver seeks the pool back to it on restart.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.engine.bridges import StrictAvgSync
from yeto.rl.engine.driver import DriverError
from yeto.rl.engine.fake import FakeEngine
from yeto.rl.engine.journal import read_journal
from yeto.rl.engine.ledger import BatchLedger, LedgerError
from yeto.rl.adapters.miles.rollout import MilesRolloutPool, seek_executor_cursor
from yeto.rl.engine.ports import GroupMetadata, RolloutBatchHandle

from test_rl_engine_driver import _strict_config, _strict_syncer
from test_rl_reconfig_e1 import NAME, _setup


class DataSourceState:
    """What Miles' ``RolloutDataSource`` keeps in the rollout process."""

    def __init__(self) -> None:
        self.sample_offset = 0
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0

    def cursor(self) -> dict[str, int]:
        return {"sample_offset": self.sample_offset, "epoch_id": self.epoch_id,
                "sample_group_index": self.sample_group_index, "sample_index": self.sample_index}


class MilesLikePool:
    """Delegates to the elastic fake; group ids and cursor follow Miles' data source."""

    def __init__(self, inner, source: DataSourceState, *, report_cursor=True, seekable=True,
                 die_at=None) -> None:
        self._inner = inner
        self._source = source
        self._report_cursor = report_cursor
        self._die_at = die_at
        self.seeks: list[dict[str, int]] = []
        if not seekable:
            self.seek_data_cursor = None  # the protocol's optional verb is absent

    def generate(self, rollout_id, *, expected_policy_version=None) -> RolloutBatchHandle:
        if self._die_at is not None and rollout_id == self._die_at:
            raise RuntimeError("killed at QUIESCING (test)")  # before prepare, like the GPU kill
        batch = self._inner.generate(rollout_id, expected_policy_version=expected_policy_version)
        groups = []
        for g in batch.groups:
            gi = self._source.sample_group_index
            samples = tuple(f"s{self._source.sample_index + i}" for i in range(len(g.sample_ids)))
            groups.append(replace(g, group_id=f"g{gi}", sample_ids=samples))
            self._source.sample_group_index += 1
            self._source.sample_offset += 1
            self._source.sample_index += len(g.sample_ids)
        return replace(batch, groups=tuple(groups),
                       data_cursor=self._source.cursor() if self._report_cursor else None)

    def seek_data_cursor(self, cursor):
        self.seeks.append(dict(cursor))
        for k, v in cursor.items():
            setattr(self._source, k, int(v))
        return self._source.cursor()

    def __getattr__(self, name):  # membership verbs etc.
        return getattr(self._inner, name)


def _island(tmp_path, syncer, *, rounds, tape, source, ledger=True, **pool_kw):
    sync_factory = lambda e: StrictAvgSync(  # noqa: E731
        _strict_config(tmp_path, e, learner_id=0, rounds=rounds, tape=tape),
        client_factory=lambda _bridge: syncer.client(0))
    driver, ctl, *_rest = _setup(tmp_path, rounds=rounds, outer="strict-avg", sync_factory=sync_factory,
                                 ledger=ledger)
    pool = MilesLikePool(driver.rollout, source, **pool_kw)
    driver.rollout = pool
    return driver, ctl, pool


def _records(tmp_path):
    return read_journal(tmp_path / "state/ledger")


def _events(tmp_path):
    import json

    return [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]


def _close(driver, ctl):
    ctl.close()
    if driver.ledger is not None:
        driver.ledger.close()


def _run_until_killed(tmp_path, syncer, *, rounds, report_cursor):
    """Rollouts 0 and 1 train and are outer-recorded (syncer at v2); the learner dies at
    the safe point before rollout 2 draws anything (r6's QUIESCING kill)."""
    driver, ctl, _pool = _island(tmp_path, syncer, rounds=rounds, tape="a.jsonl",
                                 source=DataSourceState(), report_cursor=report_cursor, die_at=2)
    with pytest.raises(RuntimeError, match="killed at QUIESCING"):
        driver.run()
    assert syncer.version == 2
    assert [driver.ledger.state(i) for i in (0, 1)] == ["outer_recorded", "outer_recorded"]
    _close(driver, ctl)


def _syncer(rounds):
    return _strict_syncer(FakeEngine(tensors={NAME: torch.zeros(1, 2)}), learners=1, rounds=rounds)


def test_r6_without_cursor_restore_redraws_trained_groups(tmp_path):
    """The failure as observed on the GPU: a cursor-less ledger + a restarted rollout
    process whose counter is back at 0 -> rollout 2 re-draws g0,g1 -> LedgerError."""
    rounds = 5
    syncer = _syncer(rounds)
    _run_until_killed(tmp_path, syncer, rounds=rounds, report_cursor=False)
    driver, ctl, _pool = _island(tmp_path, syncer, rounds=rounds, tape="b.jsonl",
                                 source=DataSourceState(), report_cursor=False, seekable=False)
    with pytest.raises(LedgerError, match=r"rollout 2: groups \['g0', 'g1'\] were already trained in rollout 0"):
        driver.run()
    _close(driver, ctl)


def test_r6_restart_seeks_the_data_source_and_keeps_training(tmp_path):
    rounds = 5
    syncer = _syncer(rounds)
    _run_until_killed(tmp_path, syncer, rounds=rounds, report_cursor=True)
    recs = _records(tmp_path)
    prepared = {r["rollout_id"]: r for r in recs if r["kind"] == "prepared"}
    assert prepared[1]["data_cursor"] == {"sample_offset": 4, "epoch_id": 0,
                                          "sample_group_index": 4, "sample_index": 8}

    restarted = DataSourceState()  # a new rollout process: counter at 0
    driver, ctl, pool = _island(tmp_path, syncer, rounds=rounds, tape="b.jsonl", source=restarted)
    driver.run()
    assert pool.seeks == [prepared[1]["data_cursor"]]
    recs = _records(tmp_path)
    prepared = [(r["rollout_id"], r["attempt"], r["group_ids"]) for r in recs if r["kind"] == "prepared"]
    assert prepared == [(0, 0, ["g0", "g1"]), (1, 0, ["g2", "g3"]), (2, 0, ["g4", "g5"]),
                        (3, 0, ["g6", "g7"]), (4, 0, ["g8", "g9"])]
    assert all(driver.ledger.state(i) == "outer_recorded" for i in range(rounds))
    assert not [r for r in recs if r["kind"] in ("superseded", "discarded")]
    applied = [r["rollout_id"] for r in recs if r["kind"] == "optimizer_applied"]
    assert applied == [0, 1, 2, 3, 4]  # every rollout consumed exactly once, three after the restart
    assert syncer.version == rounds
    restored = [e for e in _events(tmp_path) if e["event"] == "rl_data_cursor_restored"]
    assert [(e["rollout_id"], e["data_cursor"]["sample_group_index"]) for e in restored] == [(2, 4)]
    _close(driver, ctl)


def test_restart_refuses_when_the_pool_cannot_seek_to_a_recorded_cursor(tmp_path):
    rounds = 5
    syncer = _syncer(rounds)
    _run_until_killed(tmp_path, syncer, rounds=rounds, report_cursor=True)
    driver, ctl, _pool = _island(tmp_path, syncer, rounds=rounds, tape="b.jsonl",
                                 source=DataSourceState(), seekable=False)
    with pytest.raises(DriverError, match="cannot seek its data source"):
        driver.run()
    assert not [r for r in _records(tmp_path) if r["kind"] == "prepared" and r["rollout_id"] == 2]
    _close(driver, ctl)


def test_restart_refuses_when_a_seekable_pool_has_no_recorded_cursor(tmp_path):
    rounds = 5
    syncer = _syncer(rounds)
    _run_until_killed(tmp_path, syncer, rounds=rounds, report_cursor=False)
    driver, ctl, pool = _island(tmp_path, syncer, rounds=rounds, tape="b.jsonl",
                                source=DataSourceState())
    with pytest.raises(DriverError, match="holds no data cursor for rollout 1"):
        driver.run()
    assert pool.seeks == []
    _close(driver, ctl)


def test_restart_refuses_when_the_seek_lands_elsewhere(tmp_path):
    rounds = 5
    syncer = _syncer(rounds)
    _run_until_killed(tmp_path, syncer, rounds=rounds, report_cursor=True)
    driver, ctl, pool = _island(tmp_path, syncer, rounds=rounds, tape="b.jsonl",
                                source=DataSourceState())
    pool.seek_data_cursor = lambda cursor: {**dict(cursor), "sample_group_index": 0}
    with pytest.raises(DriverError, match="landed on"):
        driver.run()
    _close(driver, ctl)


# --------------------------------------------------------------------------- units
def test_ledger_restart_cursor_follows_the_recorded_batch(tmp_path):
    led = BatchLedger(tmp_path)
    cursor = {"sample_offset": 4, "epoch_id": 0, "sample_group_index": 4, "sample_index": 8}
    g = GroupMetadata("g0", ("s0",), "yeto:0:h", 0.5, 0.1, 3)
    led.prepare(RolloutBatchHandle(0, 0, "h", (g,), 1, 0, data_cursor=cursor), policy_token="yeto:0:h")
    assert led.restart_cursor(0) is None
    assert led.restart_cursor(1) is None  # not outer-recorded yet
    led.optimizer_applied(0)
    led.outer_recorded(0)
    assert led.restart_cursor(1) == cursor
    led.close()
    reopened = BatchLedger(tmp_path)  # survives the journal round trip
    assert reopened.restart_cursor(1) == cursor
    assert reopened.restart_cursor(2) is None
    reopened.close()
    led2 = BatchLedger(tmp_path / "nocursor")
    led2.prepare(RolloutBatchHandle(0, 0, "h", (g,), 1, 0), policy_token="yeto:0:h")
    led2.optimizer_applied(0)
    led2.outer_recorded(0)
    assert led2.restart_cursor(1) is None
    assert "data_cursor" not in read_journal(tmp_path / "nocursor" / "ledger")[0]
    led2.close()


class _Dataset:
    def __init__(self):
        self.shuffled: list[int] = []

    def shuffle(self, epoch):
        self.shuffled.append(epoch)


def _miles_source(*, shuffle):
    return SimpleNamespace(sample_offset=0, epoch_id=0, sample_group_index=0, sample_index=0,
                           args=SimpleNamespace(rollout_shuffle=shuffle), dataset=_Dataset(),
                           buffer=[])


def test_seek_executor_cursor_mirrors_miles_load():
    source = _miles_source(shuffle=True)
    cursor = {"sample_offset": 8, "epoch_id": 0, "sample_group_index": 8, "sample_index": 64}
    assert seek_executor_cursor(SimpleNamespace(data_source=source), cursor) == cursor
    assert source.dataset.shuffled == []  # same epoch: the initial shuffle stands
    nxt = {"sample_offset": 1, "epoch_id": 1, "sample_group_index": 9, "sample_index": 72}
    assert seek_executor_cursor(SimpleNamespace(data_source=source), nxt) == nxt
    assert source.dataset.shuffled == [1]  # new epoch: re-shuffled like RolloutDataSource.load
    plain = _miles_source(shuffle=False)
    seek_executor_cursor(SimpleNamespace(data_source=plain), nxt)
    assert plain.dataset.shuffled == []
    with pytest.raises(RuntimeError, match="lacks"):
        seek_executor_cursor(SimpleNamespace(data_source=plain), {"sample_offset": 1})
    with pytest.raises(RuntimeError, match="no data_source"):
        seek_executor_cursor(SimpleNamespace(), cursor)


def test_miles_pool_seeks_a_local_executor():
    source = _miles_source(shuffle=False)
    pool = SimpleNamespace(_executor=SimpleNamespace(data_source=source), _last_cursor=None)
    cursor = {"sample_offset": 8, "epoch_id": 0, "sample_group_index": 8, "sample_index": 64}
    assert MilesRolloutPool.seek_data_cursor(pool, cursor) == cursor
    assert (source.sample_offset, source.sample_group_index, source.sample_index) == (8, 8, 64)
    with pytest.raises(RuntimeError, match="neither a local executor nor a Ray actor"):
        MilesRolloutPool.seek_data_cursor(SimpleNamespace(_executor=object()), cursor)


# --- S17 N16: strict relaunch with NO ledger (verl V2 island 1, new container) -------

def test_strict_relaunch_without_ledger_skips_trained_rounds(tmp_path):
    """V2: island 1 left after v2 and was relaunched (same id, new container, no
    ledger). Before the fix its data source restarted at 0 and round 2 re-drew the
    groups of round 0; now the driver moves it by 2 whole rounds."""
    rounds = 4
    syncer = _syncer(rounds)
    driver, ctl, _ = _island(tmp_path, syncer, rounds=rounds, tape="a.jsonl",
                             source=DataSourceState(), die_at=2, ledger=False)
    with pytest.raises(RuntimeError, match="killed at QUIESCING"):
        driver.run()
    _close(driver, ctl)
    assert syncer.version == 2
    source = DataSourceState()
    driver, ctl, pool = _island(tmp_path / "relaunch", syncer, rounds=rounds, tape="b.jsonl",
                                source=source, ledger=False)
    driver.run()
    g = driver.sync.config.groups_per_round
    assert pool.seeks == [{"sample_offset": 2 * g, "epoch_id": 0,
                           "sample_group_index": 2 * g, "sample_index": 0}]
    assert source.sample_group_index == rounds * g  # drew rounds 2, 3 only, from g4 on
    restored = [e for e in _events(tmp_path / "relaunch") if e["event"] == "cursor_restored"]
    assert [(e["rollout_id"], e["source"]) for e in restored] == [(2, "base_version")]
    _close(driver, ctl)


def test_strict_relaunch_without_ledger_offset_only_cursor(tmp_path):
    """A backend whose cursor is ``{"sample_offset": n}`` only (verl) is moved the same way."""
    from yeto.rl.engine.bridges import whole_round_restart_cursor

    assert whole_round_restart_cursor({"sample_offset": 0}, 2, 8) == {"sample_offset": 16}
    assert whole_round_restart_cursor({"sample_offset": 0}, 0, 8) is None
    assert whole_round_restart_cursor({"epoch_id": 0}, 2, 8) is None  # no offset: unknown
    assert whole_round_restart_cursor(None, 1, 2) == {"sample_offset": 2, "epoch_id": 0,
                                                     "sample_group_index": 2, "sample_index": 0}
    assert whole_round_restart_cursor(None, 1, None) is None


def test_strict_fresh_start_without_ledger_does_not_seek(tmp_path):
    rounds = 2
    syncer = _syncer(rounds)
    driver, ctl, pool = _island(tmp_path, syncer, rounds=rounds, tape="a.jsonl",
                                source=DataSourceState(), ledger=False)
    driver.run()
    assert pool.seeks == [] and syncer.version == rounds
    _close(driver, ctl)
