"""The trainer follows the pools' member tables (local store, fake providers; no cloud):
a shard a pool adds while the run trains is consumed from the next round on without a
trainer restart, the round in flight is not interrupted, a retired shard never holds
the data end, and only a table's own lane prefix is followed."""

from __future__ import annotations

import itertools
import json

import pytest

from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute, quiet_elastic
from yeto.launch.pool import store_state
from yeto.launch.pool.island_pool import IslandPool, PoolConfig
from yeto.launch.pool.provider import ProviderPoolOps
from yeto.launch.pool.status import EventLog
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine import member_table as mt
from yeto.rl.engine.buffer_payload import encode_group
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer, shard_index
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

_clock = itertools.count(1)


@pytest.fixture(autouse=True)
def _member_table_on(monkeypatch):
    monkeypatch.setenv(mt.MEMBER_TABLE_ENV, "1")


def _write(writer, shard, n, version, groups):
    for _ in range(groups):
        sid = f"{shard}-g{n[shard]}"
        n[shard] += 1
        ref = writer.write_payload(sid, encode_group([{"index": n[shard], "shard": shard}]))
        writer.append(SampleIndexEntry(sid, version, ref, 8, float(next(_clock))))


def _reader(objects, shards):
    buffer = FileSampleBuffer(objects, batch_size=1, merge_shards=True, expected_shards=shards)
    buffer._members.refresh_s = 0  # noqa: SLF001 - every refresh reads the tables (the default: 15 s)
    return br.BatchReader(buffer, max_policy_age=8, wait_s=0.5, poll_s=0.01, sleep=lambda _s: None,
                          prefetch=False, read_workers=1)


def test_a_shard_a_pool_adds_is_consumed_from_the_next_round_on(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    writers = {s: FileSampleBuffer(objects, batch_size=1, index=shard_index(s)) for s in ("p-0", "p-1")}
    n = {"p-0": 0, "p-1": 0}
    table = mt.MemberTableWriter(objects, "nc", "p")
    assert table.publish(["p-0"]) and not table.publish(["p-0"])  # unchanged: no new epoch
    reader = _reader(objects, ["p-0"])  # the trainer's --rl-buffer-shards: the initial set
    _write(writers["p-0"], "p-0", n, 1, 2)
    _write(writers["p-1"], "p-1", n, 1, 2)  # a new member of the pool: not in the list yet
    out, _ = reader.take(1, 2, current_version=1)
    assert [g[0]["shard"] for _, g in out] == ["p-0", "p-0"]  # this round: the shards it started with
    assert table.publish(["p-0", "p-1"]) and table.epoch == 2
    out, _ = reader.take(2, 2, current_version=2)  # the next round reads the table first
    assert [g[0]["shard"] for _, g in out] == ["p-1", "p-1"]  # its earlier groups too: nothing lost
    assert reader.buffer.member_epoch == 2
    assert set(reader.buffer.shard_positions(reader.buffer.read_cursor)) == {"p-0", "p-1"}


def test_a_retired_shard_never_holds_the_data_end(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    writer = FileSampleBuffer(objects, batch_size=1, index=shard_index("p-0"))
    n = {"p-0": 0}
    table = mt.MemberTableWriter(objects, "nc", "p")
    table.publish(["p-0", "p-1"])
    buffer = FileSampleBuffer(objects, batch_size=1, merge_shards=True, expected_shards=["p-0"])
    buffer._members.refresh_s = 0  # noqa: SLF001
    _write(writer, "p-0", n, 1, 1)
    writer.mark_data_end()
    buffer.refresh()
    assert not buffer.data_ended  # p-1 (assigned, never written) is expected now
    table.publish(["p-0", "p-1"], retired=["p-1"])  # the pool was removed: p-1 retired
    buffer.refresh()
    assert buffer.data_ended


def test_only_the_tables_own_shard_base_is_followed(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    objects.write(mt.table_key("stray"), json.dumps(
        {"schema": mt.SCHEMA, "pool": "stray", "shard_base": "q", "epoch": 5, "shards": ["other-0", "q-0"]}).encode())
    reader = mt.MemberTableReader(objects, refresh_s=0)
    added, _ = reader.poll()
    assert added == {"q-0"}
    assert reader.poll() == (set(), set())  # same epoch: nothing new


def test_without_the_env_the_trainer_keeps_its_fixed_list(tmp_path, monkeypatch):
    monkeypatch.delenv(mt.MEMBER_TABLE_ENV)
    buffer = FileSampleBuffer(DirectoryObjectStore(tmp_path), batch_size=1, merge_shards=True,
                              expected_shards=["p-0"])
    assert buffer._members is None and buffer.refresh_members() == []  # noqa: SLF001


def test_a_store_pool_publishes_its_shards_and_a_trainer_follows_them_without_a_restart(tmp_path):
    """End to end: a pool on the store control plane orders two members, a scale up adds a
    third; the trainer launched with the first shard only consumes all three."""
    objects = DirectoryObjectStore(tmp_path / "store")
    clock = Clock()
    control = store_state.open_pool_control(objects, "nc", "p", clock=clock)
    control.members = mt.MemberTableWriter(objects, "nc", "p")
    ops = ProviderPoolOps(FakeProvider("x", clock), FakeRoute(), FakeFencer())
    config = PoolConfig(target=2, accounts=[0, 1, 2], shard_base="p", elastic=quiet_elastic())
    pool = IslandPool.from_doc(None, config, ops, control=control, clock=clock, log=lambda _m: None,
                               events=EventLog(log=lambda _m: None, clock=clock))
    pool.tick()
    reader = _reader(objects, ["p-0"])  # launched with the first shard only
    pool.config.target = 3
    pool.tick()
    shards = sorted(m.shard for m in pool.live())
    assert shards == ["p-0", "p-1", "p-2"]
    writers = {s: FileSampleBuffer(objects, batch_size=1, index=shard_index(s)) for s in shards}
    n = dict.fromkeys(shards, 0)
    for s in shards:
        _write(writers[s], s, n, 1, 1)
    out, _ = reader.take(1, 3, current_version=1)
    assert sorted(g[0]["shard"] for _, g in out) == shards
    assert any(e["event"].endswith("member_table") for e in pool.events.events)


def test_run_up_store_mode_tells_the_trainer_to_follow_the_member_table(tmp_path, monkeypatch):
    from tests.test_rollout_entry import _two_pools  # noqa: PLC0415
    from yeto.rl import up  # noqa: PLC0415

    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    run_cfg, _, _ = _two_pools(tmp_path)
    store = up.Run(dict(run_cfg, controller="in_process", rollout={"control_plane": "store"}))
    argv = store.trainer_argv()
    assert argv[-4:] == ["--island-env", f"{mt.MEMBER_TABLE_ENV}=1", "--island-env", f"{mt.MEMBER_LANES_ENV}=*"]
    assert "--island-env" not in up.Run(run_cfg).trainer_argv()


def test_a_table_outside_the_trainers_lanes_is_ignored_and_reported_once(tmp_path, monkeypatch):
    """REV #756: the lane is the trainer's (its initial shard bases, or YETO_RL_MEMBER_LANES),
    not the table's own shard_base: another lane's table routes nothing here."""
    objects = DirectoryObjectStore(tmp_path)
    mine = mt.MemberTableWriter(objects, "mine", "i1-nc")
    mine.publish(["i1-nc-0"])
    events = []
    buffer = FileSampleBuffer(objects, batch_size=1, merge_shards=True, expected_shards=["i1-nc-0"])
    buffer.merge_events = lambda name, **kw: events.append((name, kw))
    buffer._members.refresh_s = 0  # noqa: SLF001
    mine.publish(["i1-nc-0", "i1-nc-1"])
    mt.MemberTableWriter(objects, "theirs", "i2-nc").publish(["i2-nc-0"])
    assert buffer.refresh_members() == ["i1-nc-1"]
    assert "i2-nc-0" not in buffer._expected_shards  # noqa: SLF001
    assert [e for e in events if e[0] == "rl_member_table_ignored"] == [
        ("rl_member_table_ignored", {"pool": "theirs", "shard_base": "i2-nc"})]
    buffer.refresh_members()
    assert len([e for e in events if e[0] == "rl_member_table_ignored"]) == 1  # reported once
    monkeypatch.setenv(mt.MEMBER_LANES_ENV, "i1")  # island 1's lane: a pool added to island 1 later
    assert mt.lanes_from(__import__("os").environ, ["r-nc-i1-0"]) == frozenset({"i1"})
    monkeypatch.setenv(mt.MEMBER_LANES_ENV, "*")
    assert mt.lanes_from(__import__("os").environ, ["i1-nc-0"]) is None


def test_an_island_lane_matches_by_suffix_any_other_lane_by_prefix():
    """A fleet compiles island k's pools to <base>-i<k>: lane i<k> (or -i<k>) is a suffix."""
    one = frozenset({"i1"})
    assert mt.in_lanes("r-sglang-i1", one) and mt.in_lanes("r-sglang-i1", frozenset({"-i1"}))
    assert not mt.in_lanes("r-sglang-i10", one) and not mt.in_lanes("r-sglang-i2", one)
    assert not mt.in_lanes("i1-modal", one)  # the pool's name form is not its shard_base
    assert mt.in_lanes("team-nc", frozenset({"team"})) and mt.in_lanes("team", frozenset({"team"}))
    assert not mt.in_lanes("teams-nc", frozenset({"team"}))
    assert mt.lane_matches("anything", "*") and mt.in_lanes("anything", None)


def test_a_table_under_another_pools_key_is_not_followed(tmp_path):
    objects = DirectoryObjectStore(tmp_path)
    objects.write(mt.table_key("nc"), json.dumps(
        {"schema": mt.SCHEMA, "pool": "modal", "shard_base": "p", "epoch": 1, "shards": ["p-0"]}).encode())
    assert mt.read_tables(objects) == []


def test_the_epoch_only_grows_past_a_table_another_holder_wrote():
    from tests.test_fleet_store_control import _objects  # noqa: PLC0415

    objects = _objects()
    old = mt.MemberTableWriter(objects, "nc", "p")
    old.publish(["p-0"])
    new = mt.MemberTableWriter(objects, "nc", "p")
    new.publish(["p-0", "p-1"])
    new.publish(["p-0", "p-1", "p-2"])
    assert old.publish(["p-0", "p-3"])  # a stale cache: the stored epoch is read again
    assert json.loads(objects.read(mt.table_key("nc")))["epoch"] == 4


def test_the_seal_covers_the_member_table_shards_past_the_initial_list(tmp_path, monkeypatch):
    """REV #756: on the store control plane the trainer consumes member-table shards past its
    --rl-buffer-shards; run up's seal ends them too (a fenced or lost sampler's never ends)."""
    from tests.test_rollout_entry import _two_pools  # noqa: PLC0415
    from yeto.rl import up  # noqa: PLC0415

    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    run_cfg, _, _ = _two_pools(tmp_path)
    objects = DirectoryObjectStore(tmp_path / "buffer")
    run = up.Run(dict(run_cfg, controller="in_process", rollout={"control_plane": "store"}))
    monkeypatch.setattr(run, "_objects", lambda: objects)
    initial = run.data_end_shards()
    mt.MemberTableWriter(objects, "nc", "late").publish(["late-0", "late-1"])
    assert run.data_end_shards() == initial + ["late-0", "late-1"]
    legacy = up.Run(run_cfg)
    monkeypatch.setattr(legacy, "_objects", lambda: objects)
    assert "late-0" not in legacy.data_end_shards()  # the default control plane: unchanged

    def down():
        raise ConnectionError("store down")

    monkeypatch.setattr(run, "_objects", down)
    with pytest.raises(ConnectionError):  # never a seal on a part of the shards
        run.data_end_shards()
