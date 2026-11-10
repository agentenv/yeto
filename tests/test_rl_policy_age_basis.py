"""S3: the sample-buffer reader and the driver measure policy age from one basis.

The driver accepts round N against ``published_version`` (the version of the
policy token it hands the rollout side); the reader used Miles'
``RolloutFnInput.weight_version``, which lags by one. A group exactly
``max_policy_age`` behind the reader's basis was kept by the reader and then
refused by the driver with PolicyIdentityError (GPU-A: behavior v0, limit 2,
failed in round 3). CPU only, no Ray, no Miles."""

from __future__ import annotations

import dataclasses
import json

import pytest

from tests.buffer_rollout_procs import sample
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine.policy_age import age_floor, check_batch_ages, is_over_age, split_batch_ages
from yeto.rl.engine.ports import GroupMetadata
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer

LIMIT = 2
PUBLISHED = 3          # driver.published_version in round 3
MILES_WEIGHT = 2       # RolloutFnInput.weight_version seen by the reader in that round
TOKEN = f"yeto:{PUBLISHED}:h{PUBLISHED}"


def _group(g, version):
    return [sample(2 * g, g, version), sample(2 * g + 1, g, version)]


def _reader_with(tmp_path, versions):
    buffer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    writer = br.GroupWriter(buffer, "s0")
    for g, version in enumerate(versions):
        writer.write(_group(g, version), group_key=f"g{g}")
    return br.BatchReader(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1),
                          max_policy_age=LIMIT, wait_s=0.0, poll_s=0.0)


def _meta(gid, version, digest=None, versions=None):
    digest = f"h{version}" if digest is None else digest
    return GroupMetadata(gid, (f"{gid}-s0", f"{gid}-s1"), f"yeto:{version}:{digest}", 0.5, 0.5, 6,
                         policy_versions=versions)


def test_boundary_group_reader_kept_but_driver_refused_with_the_old_basis(tmp_path):
    """Reproduction: behavior v0 at the boundary of the Miles basis."""
    reader = _reader_with(tmp_path, (0,))
    groups, report = reader.take(PUBLISHED, 1, current_version=MILES_WEIGHT)  # old basis
    assert report.stale_groups == 0 and report.behavior_versions == {0: 1}   # reader keeps it
    # ... and the driver's guard at published_version refuses the same group
    known = {1: "h1", 2: "h2", 3: "h3"}  # v0 left the window (retain_from(3 - 2))
    assert check_batch_ages([_meta("g0", 0)], PUBLISHED, LIMIT, known) == ["g0=yeto:0:h0 (not a published policy)"]


def test_reader_gates_on_the_driver_token(tmp_path):
    basis = br.reader_basis_version(MILES_WEIGHT, TOKEN)
    assert basis == PUBLISHED
    reader = _reader_with(tmp_path, (0, 1))
    groups, report = reader.take(PUBLISHED, 1, current_version=basis)
    assert report.stale_groups == 1 and report.behavior_versions == {1: 1}
    # no token (unit fixtures, legacy): Miles' weight_version
    assert br.reader_basis_version(MILES_WEIGHT, None) == MILES_WEIGHT
    assert br.reader_basis_version(None, None) is None
    with pytest.raises(br.SampleBufferRolloutError, match="malformed"):
        br.reader_basis_version(MILES_WEIGHT, "mixed:1|2")


def test_reader_and_driver_agree_on_every_boundary():
    for basis in range(0, 6):
        for oldest in range(0, basis + 1):
            reader_skips = oldest < age_floor(basis, LIMIT)
            over, problems = split_batch_ages([_meta("g", oldest)], basis, LIMIT,
                                              {v: f"h{v}" for v in range(age_floor(basis, LIMIT), basis + 1)
                                               if v >= 0})
            assert reader_skips == is_over_age(oldest, basis, LIMIT) == bool(over)
            assert problems == []


def test_buffer_round_uses_the_driver_token(monkeypatch, tmp_path):
    from tests.test_rl_buffer_rollout import _import_plugin_with_stub_miles

    plugin = _import_plugin_with_stub_miles(monkeypatch, {})
    monkeypatch.setattr(plugin, "current_policy_token", lambda: TOKEN)
    input = type("I", (), {"weight_version": MILES_WEIGHT})()
    assert plugin._BufferRound._basis_version(input) == PUBLISHED

    def no_ray():
        raise ImportError("ray")

    monkeypatch.setattr(plugin, "current_policy_token", no_ray)
    assert plugin._BufferRound._basis_version(input) == MILES_WEIGHT


def test_split_batch_ages_keeps_identity_errors_fatal():
    known = {1: "h1", 2: "h2", 3: "h3"}
    over, problems = split_batch_ages(
        [_meta("old", 0),                       # over-age: discarded
         _meta("seg", 3, versions=(0, 3)),       # a v0 segment under a v3 token: discarded
         _meta("carried", 0, versions=(0, 3)),   # carried-over, oldest v0: discarded
         _meta("ok", 1), _meta("newer", 4, digest="h4"),        # newer than the basis: refused
         _meta("forged", 2, digest="x"),         # in-window version, other weights: refused
         GroupMetadata("mixed", ("m",), "mixed:1|2", 0.5, 0.5, 1)],
        PUBLISHED, LIMIT, known)
    assert [g.group_id for g in over] == ["old", "seg", "carried"]
    assert problems == ["newer=yeto:4:h4 (not a published policy)",
                        "forged=yeto:2:x (not a published policy)",
                        "mixed=mixed:1|2 (not a published policy)"]


def _stale_driver(tmp_path, *, discard=True, token=None):
    from tests.test_rl_policy_age import _driver

    driver = _driver(tmp_path, 1)
    rollout = driver.rollout
    original = rollout.generate

    def gen(rid, **kw):
        h = original(rid, **kw)
        if rid != 2:
            return h
        g0 = dataclasses.replace(h.groups[0], policy_token=token or policy_of(driver, 0))
        return dataclasses.replace(h, groups=(g0,) + h.groups[1:])

    rollout.generate = gen
    if not discard:
        type(rollout).discard_groups = None  # restored by the fixture below
    return driver


def policy_of(driver, version):
    return driver.published_policies.label(version) or f"yeto:{version}:gone"


@pytest.fixture
def restore_discard():
    from yeto.rl.engine.fake import FakeRolloutPool

    saved = FakeRolloutPool.discard_groups
    yield
    FakeRolloutPool.discard_groups = saved


def test_driver_discards_a_stale_token_group_and_reports_it(tmp_path):
    """Round 2, limit 1: a group sampled from v0 (hash already forgotten) is dropped."""
    driver = _stale_driver(tmp_path)
    driver.run()
    events = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    [dropped] = [e for e in events if e["event"] == "rl_policy_age_discarded"]
    assert dropped["groups"] == [["r2-g0", 0]] and dropped["basis_version"] == 2
    assert len([e for e in events if e["event"] == "rl_round_trained"]) == 3
    assert ("discard_groups", 2, ("r2-g0",)) in driver.rollout.engine.calls


def test_driver_refuses_when_the_rollout_cannot_drop(tmp_path, restore_discard):
    from yeto.rl.engine.driver import PolicyAgeExceeded

    with pytest.raises(PolicyAgeExceeded, match="r2-g0"):
        _stale_driver(tmp_path, discard=False).run()
    events = (tmp_path / "e.jsonl").read_text()
    assert '"rl_policy_age_discarded"' in events  # reported before refusing


def test_driver_still_refuses_a_version_newer_than_published(tmp_path):
    from yeto.rl.engine.driver import PolicyIdentityError

    with pytest.raises(PolicyIdentityError, match="not a published policy"):
        _stale_driver(tmp_path, token="yeto:5:h5").run()
