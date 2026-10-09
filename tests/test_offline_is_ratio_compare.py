import math

from tools.offline_is_ratio_compare import compare, icepop, m2po, tis


def test_tis_clamps_and_icepop_zeros():
    r = [0.1, 1.0, 3.0]
    assert tis(r) == [0.1, 1.0, 2.0]
    assert icepop(r) == [0.0, 1.0, 0.0]


def test_m2po_masks_until_budget():
    r = [1.0] * 9 + [math.e]  # (log r)^2: nine 0s and one 1 -> mean 0.1 > 0.04
    w = m2po(r, tau=0.04)
    assert w[-1] == 0.0 and w[:9] == [1.0] * 9
    assert m2po([1.0, 1.05], tau=0.04) == [1.0, 1.05]


def test_compare_metrics():
    out = compare([1.0] * 4)
    for k in ("raw", "tis", "icepop", "m2po"):
        assert out[k]["altered_frac"] == 0.0
        assert out[k]["ess_over_n"] == 1.0
        assert out[k]["var_w"] == 0.0
    out = compare([1.0, 1.0, 1.0, 10.0])
    assert out["tis"]["altered_frac"] == 0.25
    assert out["raw"]["ess_over_n"] < out["tis"]["ess_over_n"]
