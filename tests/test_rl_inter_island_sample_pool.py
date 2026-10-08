"""Object-store sample-pool index (tasks 0.12). Pure CPU."""

from __future__ import annotations

import pytest

from yeto.rl.engine.island_ledger import ACCEPT, ACCEPT_IS, CrossIslandLedger, DeltaEntry
from yeto.rl.engine.sample_pool import SampleIndexEntry, SampleIndexError, SamplePoolIndex

SHA = "a" * 64


def _entry(island="b", version=0, group="g1", lp=True, uri="s3://bucket/x", **kw):
    h = kw.pop("policy_hash", f"h{version}")
    return SampleIndexEntry(island, version, 4, h, group, "p1", 8, uri, 1234, SHA, lp, **kw)


def _ledger():
    led = CrossIslandLedger(mode="elastic", theta=0.3)
    led.join("a", now=0.0); led.join("b", now=0.0)
    return led


def test_roundtrip_and_validation():
    e = _entry(uri="modal-volume://vol/x")
    assert SampleIndexEntry.from_json(e.to_json()) == e
    for bad in (dict(uri="http://x"), dict(advantage_included=False), dict(schema="v0")):
        with pytest.raises(SampleIndexError):
            _entry(**bad)
    with pytest.raises(SampleIndexError):
        SampleIndexEntry("b", 0, 4, "h0", "g", "p", 8, "nebius-os://b/x", 1, "short", True)


def test_group_never_spans_islands():
    idx = SamplePoolIndex()
    idx.add(_entry("b", group="g1"))
    idx.add(_entry("b", group="g1"))  # idempotent
    with pytest.raises(SampleIndexError):
        idx.add(_entry("a", group="g1"))
    with pytest.raises(SampleIndexError):
        idx.add(_entry("b", group="g1", uri="s3://bucket/other"))


def test_select_uses_ledger_verdicts():
    led = _ledger()
    idx = SamplePoolIndex()
    idx.add(_entry("b", 0, "old"))
    idx.add(_entry("b", 0, "old-nolp", lp=False))
    led.submit(DeltaEntry("a", 0, 4, "h0", 100, 4)); led.try_advance(timed_out=False)
    idx.add(_entry("b", 1, "new"))
    got = {s.entry.group_id: s.verdict for s in idx.select(led, consumer_island="a")}
    assert got["new"].verdict == ACCEPT
    assert got["old"].verdict == ACCEPT_IS and got["old"].correction == "tis"
    assert "old-nolp" not in got
    assert len(idx.select(led, consumer_island="a", limit=1)) == 1
    assert idx.prune_below(1) == 2 and len(idx) == 1


def test_legacy_ledger_selects_nothing_cross_island():
    led = CrossIslandLedger()
    led.join("a", now=0.0); led.join("b", now=0.0)
    idx = SamplePoolIndex()
    idx.add(_entry("b", 0, "g"))
    assert idx.select(led, consumer_island="a") == []
