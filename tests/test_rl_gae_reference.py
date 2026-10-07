import math

import pytest
import torch

from tests.rl_gae_reference import (
    gae_cross_segment,
    gae_decoupled,
    gae_length_adaptive,
    gae_vanilla,
    length_adaptive_lambda,
)

D = torch.float64


def _t(x):
    return torch.tensor(x, dtype=D)


def test_vanilla_hand_computed():
    r, v = _t([0.0, 0.0, 1.0]), _t([0.5, 0.25, 0.5])
    g, l = 0.9, 0.5
    d0 = 0 + g * 0.25 - 0.5
    d1 = 0 + g * 0.5 - 0.25
    d2 = 1 - 0.5
    a2 = d2
    a1 = d1 + g * l * a2
    a0 = d0 + g * l * a1
    adv, ret = gae_vanilla(r, v, g, l)
    assert torch.allclose(adv, _t([a0, a1, a2]))
    assert torch.allclose(ret, adv + v)


def test_vanilla_gamma_lambda_one_is_monte_carlo():
    r, v = _t([0.1, -0.2, 0.3, 1.0]), _t([0.4, 0.3, 0.2, 0.1])
    adv, ret = gae_vanilla(r, v, 1.0, 1.0)
    mc = torch.flip(torch.cumsum(torch.flip(r, [0]), 0), [0])
    assert torch.allclose(ret, mc)


def test_length_adaptive_lambda_value():
    assert length_adaptive_lambda(100, 1.5) == pytest.approx(1 - 1 / 150)


def test_length_adaptive_alpha_to_infinity_is_lambda_one():
    r, v = _t([0.0, 0.2, 1.0]), _t([0.3, 0.1, 0.6])
    a_inf, _ = gae_length_adaptive(r, v, 1.0, 3, alpha=1e12)
    a1, _ = gae_vanilla(r, v, 1.0, 1.0)
    assert torch.allclose(a_inf, a1, atol=1e-9)


def test_decoupled_hand_computed():
    r, v = _t([0.0, 1.0]), _t([0.2, 0.4])
    g, lp, lc = 1.0, 0.5, 1.0
    d0, d1 = 0 + 0.4 - 0.2, 1 - 0.4
    adv, ret = gae_decoupled(r, v, g, lp, lc)
    assert torch.allclose(adv, _t([d0 + lp * d1, d1]))
    assert torch.allclose(ret, _t([d0 + d1 + 0.2, d1 + 0.4]))  # lambda_c=1 -> MC return
    a_same, r_same = gae_decoupled(r, v, g, lp, lp)
    assert torch.allclose(r_same, gae_vanilla(r, v, g, lp)[1])


def test_cross_segment_single_segment_equals_vanilla():
    r, v = _t([0.1, 0.0, -0.3, 1.0]), _t([0.5, 0.4, 0.3, 0.2])
    a, ret = gae_cross_segment(r, v, [0, 0, 0, 0], 0.95, 0.9)
    av, rv = gae_vanilla(r, v, 0.95, 0.9)
    assert torch.allclose(a, av) and torch.allclose(ret, rv)


def test_cross_segment_two_segments_hand_computed():
    # seg0 = tokens 0,1 ; seg1 = tokens 2,3,4 (n_2 = 3); terminal reward at last token
    r = _t([0.0, 0.0, 0.0, 0.0, 1.0])
    v = _t([0.1, 0.2, 0.3, 0.4, 0.5])
    g, l = 0.9, 0.8
    w = g * l
    # seg0 local: no bootstrap past token 1
    d0 = 0 + g * 0.2 - 0.1
    d1 = 0 + g * 0.0 - 0.2
    loc0 = [d0 + w * d1, d1]
    d2 = g * 0.4 - 0.3
    d3 = g * 0.5 - 0.4
    d4 = 1.0 - 0.5
    loc1 = [d2 + w * d3 + w * w * d4, d3 + w * d4, d4]
    adv, ret = gae_cross_segment(r, v, [0, 0, 1, 1, 1], g, l)
    exp = _t([x * w**3 for x in loc0] + loc1)
    assert torch.allclose(adv, exp)
    assert torch.allclose(ret, exp + v)


def test_cross_segment_three_segments_factor():
    r, v = torch.zeros(6, dtype=D), torch.ones(6, dtype=D)
    seg = [0, 1, 1, 2, 2, 2]
    adv, _ = gae_cross_segment(r, v, seg, 1.0, 0.5)
    # segment 0 single token: local = 0 + 0 - 1 = -1, N_{>0} = 5
    assert adv[0].item() == pytest.approx(-1 * 0.5**5)
    # segment 1 last token local = -1, N_{>1} = 3
    assert adv[2].item() == pytest.approx(-1 * 0.5**3)
    assert math.isfinite(adv.sum().item())
