"""rl-algo-supplement 3.1 (design D6): user-code mechanisms are never declared."""

import pytest

from yeto.rl.adapters.miles import entry
from yeto.rl.engine.algorithm import AlgorithmSpecError, allowance_names, check_unverified_allowance


def test_never_declarable_list_is_the_d6_set():
    assert entry.NEVER_DECLARABLE == {
        "corrections:custom",
        "features:plugins",
        "losses:custom_loss",
        "features:custom_pg_loss_reducer",
    }


def test_never_declarable_names_are_real_mechanisms():
    # a typo would make the guard vacuous
    assert entry.NEVER_DECLARABLE <= allowance_names()


def test_never_declarable_is_disjoint_from_miles_declared():
    assert not entry.NEVER_DECLARABLE & set(entry.MILES_DECLARED)
    for pin in (None, *sorted(set().union(*entry.MILES_DECLARED_PINS.values()))):
        declared = {f"{d}:{n}" for d, names in entry.declared_by_dimension(pin).items() for n in names}
        assert not entry.NEVER_DECLARABLE & declared, pin


def test_vendored_drgrpo_reducer_is_not_in_the_list():
    # Dr.GRPO is declared as loss_aggregations:constant (named reducer), not as the generic reducer
    assert "loss_aggregations:constant" in entry.MILES_DECLARED
    assert "loss_aggregations:constant" not in entry.NEVER_DECLARABLE
    assert "reward_postprocessors:custom_reward_postprocess" not in entry.NEVER_DECLARABLE


@pytest.mark.parametrize("name", sorted(entry.NEVER_DECLARABLE))
def test_allowance_for_user_code_single_island_only(name):
    assert check_unverified_allowance([name], islands=1, outer_sync=False) == (name,)
    with pytest.raises(AlgorithmSpecError, match="single-island"):
        check_unverified_allowance([name], islands=2, outer_sync=False)
    with pytest.raises(AlgorithmSpecError, match="single-island"):
        check_unverified_allowance([name], islands=1, outer_sync=True)
    # the critic strict-avg exception does not cover user code
    with pytest.raises(AlgorithmSpecError, match="single-island"):
        check_unverified_allowance([name, "execution:critic"], islands=2, outer_sync=True,
                                   sync_preset="strict-avg")
