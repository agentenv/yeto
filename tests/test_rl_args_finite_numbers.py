"""RL launch refuses non-finite numeric knobs (nan/inf) up front. No Ray.

A NaN fails every ``<= 0`` comparison, so a bare sign check lets it through;
an infinite timeout or outer learning rate is never a usable setting either.
"""

from __future__ import annotations

import pytest

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401
from test_rl_sync_membership_options import _launcher_args


@pytest.fixture(autouse=True)
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")


THREE = "modal:2xh100,modal:2xh100,modal:2xh100"


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("--cybergym-timeout", "CyberGym timeout"),
        ("--inner-lr", "inner-lr"),
        ("--rl-sync-outer-lr", "outer-lr"),
    ],
)
def test_non_finite_values_are_refused(flag, message, value):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (f"{flag}={value}",), gpu=THREE)
    with pytest.raises(ValueError, match=message):
        _prepare_rl_args(args)


def test_finite_values_still_pass():
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", ("--cybergym-timeout", "30", "--rl-sync-outer-lr", "0.5"),
                          gpu=THREE)
    _prepare_rl_args(args)
