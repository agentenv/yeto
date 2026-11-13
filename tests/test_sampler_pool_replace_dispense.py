"""Sampler pool churn against the real syncer dispenser: a sampler node departs, the
pool fences its shard through :class:`yeto.launch.pool.data_path.SyncerDataPath`
(MEMBER_RETIRE), and a replacement joins either on the same shard name (same member
id, new incarnation) or on a fresh shard. CPU only, no GPU, no network beyond
localhost.

Checked at the seam of ``launch/pool/data_path.py`` + ``task_dispense`` +
``syncer/src/{members,dispense,member_retire}.rs``:

* no position is ever held by two live members (the replacement never gets a
  position another live sampler holds);
* the departed sampler's unreported positions go back first, at once on a fence,
  after the lease without one;
* a late report from the departed incarnation is refused, so nothing is counted
  trained twice; the dispense log passes ``dispense_check``;
* a fence retried before the replacement joins (a lost answer) is idempotent
  (``no_member``). ``SyncerDataPath`` fences the member's *current* incarnation,
  so it is the pool's start gating (``IslandPool._fence_blocker``: nothing
  continues a shard before its fence succeeded) that keeps a retry from reaching
  the replacement.

Sampler members are not learners of the fragment round (the dispenser needs no
learner session: the syncer here is started with ``--learners 4`` and no learner
ever connects), so pool churn cannot move the round's learner count or quorum."""

from __future__ import annotations

import time

import pytest

from tests.test_dispense_syncer import FP, KEY, Syncer, syncer_binary  # noqa: F401 - the fixture
from yeto.launch.checks.dispense_check import check
from yeto.launch.pool.data_path import SyncerDataPath
from yeto.rl.engine import task_dispense as td

ROWS = 64


def _conn(port: int, shard: str, incarnation: int) -> td.DispenseConnection:
    return td.DispenseConnection(("127.0.0.1", port), learner_id=td.sampler_holder_id(shard),
                                 incarnation=incarnation, rows=ROWS, fingerprint=FP, key=KEY, timeout_s=5.0)


def _positions(grant: td.Grant) -> list[int]:
    return [p for start, end in grant.ranges for p in range(start, end)]


def _trained(conn: td.DispenseConnection, seq: int, grant: td.Grant) -> td.Ack:
    return conn.report(seq, [(grant.block, td.TRAINED, list(grant.ranges))])


def test_a_fenced_shard_replaced_on_the_same_name_takes_its_rows_and_fences_the_old(
        syncer_binary, tmp_path):  # noqa: F811
    syncer = Syncer(syncer_binary, tmp_path, lease_s=600.0)
    path = SyncerDataPath(("127.0.0.1", syncer.port), key=KEY)
    try:
        a, b, gone = _conn(syncer.port, "nc-0", 10), _conn(syncer.port, "nc-2", 20), _conn(syncer.port, "nc-1", 11)
        ga, gb, gg = a.claim(1, 4), b.claim(1, 4), gone.claim(1, 4)
        held = {"nc-0": set(_positions(ga)), "nc-2": set(_positions(gb)), "nc-1": set(_positions(gg))}
        assert not (held["nc-0"] & held["nc-1"]) and not (held["nc-2"] & held["nc-1"])
        # nc-1's node goes away (no LEAVE, lease still running): the pool fences its shard
        answer = path.fence_shard("nc-1", by="ticket-1")
        assert answer["status"] == "retired" and answer["incarnation"] == 11 and answer["reclaimed"] == 4
        # the replacement on the same shard name: same member id, new incarnation
        # a fence retried before the replacement starts (the pool holds the start until
        # its fence succeeded; an answer lost on the way is sent again) is idempotent
        assert path.fence_shard("nc-1", by="ticket-1")["status"] == "no_member"
        new = _conn(syncer.port, "nc-1", 12)
        gn = new.claim(1, 4)
        assert set(_positions(gn)) == held["nc-1"], "the fenced rows go first"
        assert not set(_positions(gn)) & (held["nc-0"] | held["nc-2"])
        # the departed process comes back (a healed partition): refused on every request
        # (the syncer closes a retired member's connection; a redial of the old
        # incarnation is refused at its HELLO)
        with pytest.raises((td.DispenseFenced, ConnectionError)):
            _trained(gone, 2, gg)
        with pytest.raises(td.DispenseFenced):
            _conn(syncer.port, "nc-1", 11)
        for conn, grant in ((a, ga), (b, gb), (new, gn)):
            assert _trained(conn, 2, grant).settled == 4
        for conn in (a, b, gone, new):
            conn.close()
        summary = check(syncer.records())
        assert summary["ok"], summary["problems"]
        # nc-1's rows trained once at most (the old incarnation's report was refused)
        assert summary["islands"][str(td.sampler_holder_id("nc-1"))]["incarnations"] == 2
    finally:
        syncer.stop()


def test_a_replacement_on_a_fresh_shard_never_gets_the_departed_rows_before_the_lease(
        syncer_binary, tmp_path):  # noqa: F811
    """Without a fence (no pool fencer): the departed sampler's rows stay its own
    for the lease, a replacement on a new shard name gets other rows, then the
    departed rows go first to the next claim after the lease expires."""
    syncer = Syncer(syncer_binary, tmp_path, lease_s=1.5)
    try:
        a, gone = _conn(syncer.port, "nc-0", 10), _conn(syncer.port, "nc-1", 11)
        ga, gg = a.claim(1, 4), gone.claim(1, 4)
        fresh = _conn(syncer.port, "nc-3", 13)  # the pool's next shard name
        gf = fresh.claim(1, 4)
        assert not set(_positions(gf)) & (set(_positions(ga)) | set(_positions(gg)))
        assert _trained(a, 2, ga).settled == 4 and _trained(fresh, 2, gf).settled == 4
        deadline, got, seq = time.monotonic() + 10, [], 3
        while not set(_positions(gg)) <= set(got) and time.monotonic() < deadline:
            for conn in (a, fresh):
                conn.heartbeat()
            grant = fresh.claim(seq, 4)
            if grant.status == td.GRANT:
                got += _positions(grant)
                _trained(fresh, seq + 1, grant)
            else:
                time.sleep(0.1)
            seq += 2
        assert set(_positions(gg)) <= set(got), "the departed rows never came back"
        with pytest.raises((td.DispenseFenced, ConnectionError)):
            _trained(gone, 2, gg)
        for conn in (a, fresh, gone):
            conn.close()
        records = syncer.records()
        reclaims = [r for r in records if r["kind"] == "dispense_reclaim"]
        assert [r["reason"] for r in reclaims] == ["lease_expired"]
        assert reclaims[0]["island_id"] == td.sampler_holder_id("nc-1")
        summary = check(records)
        assert summary["ok"], summary["problems"]
    finally:
        syncer.stop()
