"""S26 E2: adaptive TIS (AIS) as a Miles custom TIS function."""

from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")

from yeto.rl.algos import adaptive_tis as ais  # noqa: E402

ARGS = Namespace(tis_clip=5.0, tis_clip_low=0.0)


def _call(train, rollout, pg=None, mask=None):
    train, rollout = torch.tensor(train), torch.tensor(rollout)
    pg = torch.ones_like(train) if pg is None else torch.tensor(pg)
    mask = torch.ones_like(train) if mask is None else torch.tensor(mask)
    return ais.ais_tis_function(ARGS, pg_loss=pg, train_log_probs=[train],
                                rollout_log_probs=[rollout], loss_masks=[mask])


def test_on_policy_batch_is_left_uncorrected():
    lp = [-1.0, -0.5, -2.0, -0.1]
    loss, masks, metrics = _call(lp, lp, pg=[0.3, -0.2, 0.5, 1.0])
    assert torch.equal(loss, torch.tensor([0.3, -0.2, 0.5, 1.0]))
    assert float(metrics["ais_alpha"][0]) == 0.0
    assert float(metrics["tis_clipfrac"].sum()) == 0.0


def test_large_uniform_mismatch_is_plain_tis():
    # every token has the same ratio e^0.1: CV 0 (a_ess 1), no variance blow-up, a_mis 1
    train = [-1.0, -0.5, -2.0, -0.1]
    rollout = [t - 0.1 for t in train]
    loss, _, metrics = _call(train, rollout, pg=[1.0, -1.0, 1.0, -1.0])
    expected = torch.tensor([1.0, -1.0, 1.0, -1.0]) * torch.exp(torch.tensor(0.1))
    assert torch.allclose(loss, expected, atol=1e-6)
    assert float(metrics["ais_alpha"][0]) == pytest.approx(1.0)


def test_small_mismatch_scales_the_correction_by_the_necessity_gate():
    train = [-1.0, -1.0]
    rollout = [-1.01, -1.01]  # mean |log diff| 0.01 = delta / 2
    loss, _, metrics = _call(train, rollout)
    assert float(metrics["ais_gate_mis"][0]) == pytest.approx(0.5, rel=1e-4)
    w = torch.exp(torch.tensor(0.01))
    assert torch.allclose(loss, torch.full((2,), 1 + 0.5 * (float(w) - 1)), atol=1e-6)


def test_degenerate_weights_reduce_alpha_and_truncation_is_reported():
    # one token far off (clamped at C=5), the rest on-policy: low ESS, high variance
    train = [0.0] * 7 + [3.0]
    rollout = [0.0] * 8
    _, _, metrics = _call(train, rollout, pg=[1.0, -1.0] * 4)
    g = metrics
    assert float(g["ais_gate_ess"][0]) == pytest.approx(0.75)  # mean 1.5, var 1.75
    assert float(g["ais_alpha"][0]) < float(g["ais_gate_ess"][0])
    assert float(g["tis_clipfrac"].sum()) == 1.0
    assert float(g["tis"][7]) == pytest.approx(float(torch.exp(torch.tensor(3.0))))


def test_masked_tokens_do_not_enter_the_gates():
    train = [-1.0, -1.0, 5.0]
    rollout = [-1.0, -1.0, 0.0]
    _, _, metrics = _call(train, rollout, mask=[1.0, 1.0, 0.0])
    assert float(metrics["ais_alpha"][0]) == 0.0


def test_spec_maps_to_the_miles_custom_tis_path():
    from yeto.rl.adapters.miles.offpolicy_correction import tis_argv
    from yeto.rl.engine.algorithm import AlgorithmSpec

    spec = AlgorithmSpec.from_dict({"schema": "yeto-rl-algorithm-spec-v2", "correction": {
        "method": "custom", "function": {"path": ais.PATH}, "upper": 5.0, "lower": 0.0}})
    argv = tis_argv(spec)
    assert argv[:1] == ["--use-tis"]
    assert argv[argv.index("--custom-tis-function-path") + 1] == ais.PATH
    assert argv[argv.index("--tis-clip") + 1] == "5.0"


def test_named_mechanism_is_not_declared_on_miles_until_gpu_evidence():
    from yeto.rl.adapters.miles.entry import MILES_DECLARED
    from yeto.rl.algos.mismatch_correction import ADAPTIVE_TIS_PATH, selected_corrections
    from yeto.rl.engine.algorithm import AlgorithmSpec

    assert ADAPTIVE_TIS_PATH == ais.PATH
    spec = AlgorithmSpec.from_dict({"schema": "yeto-rl-algorithm-spec-v2", "correction": {
        "method": "custom", "function": {"path": ais.PATH}, "upper": 5.0, "lower": 0.0}})
    assert selected_corrections(spec) == ("adaptive_tis",)
    assert "corrections:adaptive_tis" not in MILES_DECLARED  # default off: GPU not verified
