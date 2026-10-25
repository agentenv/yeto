"""S27 C5 (G-CT): ``--rl-verify-every N`` on a no-sync full-parameter island.

The island hashes every rank's shard (about 8.4 s a round on Nano 8xH200) only on
the first publication, every N-th version, the last round and rounds that keep a
round cut; the other rounds publish an unverified identity.  Default: unchanged.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from test_rl_full_parameter_local_state import _Driver, _ranks, _state  # noqa: E402
from yeto.rl.adapters.miles import state_plugin as sp  # noqa: E402
from yeto.rl.adapters.miles.state import MilesFullParameterLocalPolicyState  # noqa: E402
from yeto.rl.engine import run_flags  # noqa: E402
from yeto.rl.engine.run_flags import check_verify_every  # noqa: E402
from yeto.rl.engine.driver import IslandDriver, verify_publication  # noqa: E402
from yeto.rl.engine.policy_digest import WeightsChanged  # noqa: E402


def _driver(ps, *, every=None, published=1, cut_due=lambda v: False, last=10, cuts_every=None):
    sync = SimpleNamespace(round_cuts=None if cuts_every is None else SimpleNamespace(every=cuts_every),
                           cut_due=cut_due, is_final_round=lambda d, *, rollout_id: rollout_id + 1 >= last)
    d = SimpleNamespace(policy_state=ps, sync=sync, published_version=published, verify_every_round=False)
    if every is not None:
        d.verify_every = every
    return d


def test_default_full_parameter_island_verifies_every_round():
    _group, ps = _state(_ranks())
    assert MilesFullParameterLocalPolicyState.supports_unverified_rounds is False
    assert all(verify_publication(_driver(ps), v) for v in range(1, 12))
    assert all(verify_publication(_driver(ps, cuts_every=4), v) for v in range(1, 12))
    # the interval alone does nothing until the state allows unverified rounds
    assert all(verify_publication(_driver(ps, every=4), v) for v in range(1, 12))


def test_interval_verifies_first_every_nth_last_and_cut_rounds():
    _group, ps = _state(_ranks())
    ps.allow_unverified_rounds()
    assert MilesFullParameterLocalPolicyState.supports_unverified_rounds is False  # instance only
    assert verify_publication(_driver(ps, every=4, published=None), 3)  # first publication
    d = _driver(ps, every=4, cut_due=lambda v: v == 7, last=10)
    assert [v for v in range(1, 11) if verify_publication(d, v)] == [4, 7, 8, 10]
    assert all(verify_publication(_driver(ps, every=1), v) for v in range(1, 11))
    # the interval replaces the round-cut interval (a due cut is still verified)
    d = _driver(ps, every=5, cuts_every=2, cut_due=lambda v: v == 4, last=99)
    assert [v for v in range(1, 11) if verify_publication(d, v)] == [4, 5, 10]


def test_unverified_round_skips_the_shard_hash_and_keeps_the_holds_check(capsys):
    ranks = _ranks()
    group, ps = _state(ranks)
    ps.allow_unverified_rounds()
    assert ps.export_identity(policy_version=1) is None  # nothing verified yet
    verified = ps.export_digest(policy_version=1)
    group.calls.clear()
    unverified = ps.export_identity(policy_version=2)
    assert group.calls == [sp.WEIGHTS_VERSION]  # no EXPORT_FULL_DIGEST
    assert unverified.verified is False and unverified.policy_version == 2
    assert unverified.layout == verified.layout and unverified.layout_hash == verified.layout_hash
    assert unverified.payload_digest[1] == verified.payload_digest[1]
    assert unverified.policy_tensor_hash() != verified.policy_tensor_hash()
    assert "unverified (no shard hash" in capsys.readouterr().out
    unverified.check_holds()
    ranks[0].train_step()
    with pytest.raises(WeightsChanged):
        unverified.check_holds()


def test_driver_export_publishes_identities_between_verified_rounds(monkeypatch):
    ranks = _ranks()
    group, ps = _state(ranks)
    ps.allow_unverified_rounds()
    driver = _Driver(ps)
    driver.sync = SimpleNamespace(round_cuts=None, cut_due=lambda v: False,
                                  is_final_round=lambda d, *, rollout_id: rollout_id + 1 >= 6)
    driver.verify_every_round, driver.verify_every, driver.published_version = False, 3, None
    hashed = []
    for version in range(1, 7):
        group.calls.clear()
        state = IslandDriver.export_local_resident(driver, policy_version=version)
        hashed.append(sp.EXPORT_FULL_DIGEST in group.calls)
        assert state.verified is hashed[-1]
        driver.published_version = version
        for r in ranks:
            r.train_step()
    assert hashed == [True, False, True, False, False, True]  # first, 3rd, 6th (last)


def test_launch_check_and_flag():
    parser = argparse.ArgumentParser()
    run_flags.add_rl_flags(parser, run_flags.CLI)
    assert parser.parse_args(["--rl-verify-every", "8"]).rl_verify_every == 8
    assert parser.parse_args([]).rl_verify_every is None
    assert any("--rl-verify-every" in f.options for f in run_flags.flags_for(run_flags.ISLAND))
    from yeto.launch.rl_ports import _ports_infra_flags

    assert " --rl-verify-every 8" in _ports_infra_flags(SimpleNamespace(rl_verify_every=8))[1]
    assert "--rl-verify-every" not in _ports_infra_flags(SimpleNamespace())[1]
    check_verify_every(SimpleNamespace(rl_verify_every=None))
    check_verify_every(SimpleNamespace(rl_verify_every=8, rl_single_island_no_sync=True))
    with pytest.raises(ValueError, match="no-sync"):
        check_verify_every(SimpleNamespace(rl_verify_every=8, rl_single_island_no_sync=False))
    with pytest.raises(ValueError, match="N >= 1"):
        check_verify_every(SimpleNamespace(rl_verify_every=0, rl_single_island_no_sync=True))
    with pytest.raises(ValueError, match="exclude each other"):
        check_verify_every(SimpleNamespace(rl_verify_every=8, rl_single_island_no_sync=True,
                                           rl_verify_every_round=True))
