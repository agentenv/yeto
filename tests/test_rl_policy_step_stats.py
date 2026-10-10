"""rl-algo-supplement follow-up (S19 #13): per-step advantage and PPO-ratio statistics
recorded by the Miles state plugin (4.7 advantage variance, 5.2 clipfrac offline check).
CPU only; Miles is replaced by fake modules."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from yeto.rl.adapters.miles import state_plugin as sp  # noqa: E402


def _miles_compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None):
    # mirror of Miles 64b591a4b loss_hub.math_utils.compute_policy_loss (standard clip)
    ratio = torch.exp(-ppo_kl)
    pg1 = -ratio * advantages
    pg2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    return torch.maximum(pg1, pg2), torch.gt(pg2, pg1).float()


@pytest.fixture
def fake_miles(monkeypatch):
    losses_mod = types.ModuleType("miles.backends.training_utils.loss_hub.losses")
    model_mod = types.ModuleType("miles.backends.megatron_utils.model")
    losses_mod.compute_policy_loss = _miles_compute_policy_loss
    seen = {}

    def policy_loss_function(args, batch, logits, sum_of_sample_mean):
        adv = torch.cat(batch["advantages"])
        loss, clipfrac = losses_mod.compute_policy_loss(batch["ppo_kl"], adv, args.eps_clip, args.eps_clip_high)
        seen["clipfrac_tokens"] = clipfrac
        return loss.sum(), {"pg_loss": float(loss.mean()), "pg_clipfrac": float(clipfrac.mean())}

    losses_mod.policy_loss_function = policy_loss_function
    pkg = {name: types.ModuleType(name) for name in (
        "miles", "miles.backends", "miles.backends.training_utils",
        "miles.backends.training_utils.loss_hub", "miles.backends.megatron_utils")}
    pkg["miles.backends.training_utils.loss_hub"].losses = losses_mod
    pkg["miles.backends.megatron_utils"].model = model_mod
    for name, module in {**pkg, losses_mod.__name__: losses_mod, model_mod.__name__: model_mod}.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("_STEP_GRAD_NORMS", "_STEP_APPLIED_LRS", "_STEP_LOSSES", "_EV_STATS"):
        monkeypatch.setattr(sp, name, [])
    monkeypatch.setattr(sp, "_POLICY_STATS", {})
    monkeypatch.setattr(sp, "_RECORDER_INSTALLED", False)
    monkeypatch.setattr(sp, "_POLICY_METRICS_INSTALLED", False)
    monkeypatch.setattr(sp, "_record_applied_lr", lambda *a: None)
    monkeypatch.setattr(sp, "_arm_grad_audit", lambda *a: None)
    return losses_mod, model_mod, seen


def test_step_stats_recorded_and_loss_unchanged(fake_miles):
    losses_mod, model_mod, seen = fake_miles
    args = types.SimpleNamespace(eps_clip=0.2, eps_clip_high=0.28)
    advs = [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([-0.5, -0.5])]
    ppo_kl = torch.tensor([-0.5, 0.0, 0.1, 0.5, -0.1])  # ratios 1.65, 1, 0.90, 0.61, 1.11
    batch = {"advantages": advs, "loss_masks": [torch.ones(3), torch.ones(2)], "ppo_kl": ppo_kl}
    expected = _miles_compute_policy_loss(ppo_kl, torch.cat(advs), 0.2, 0.28)

    def train_one_step(*a, **k):
        return losses_mod.policy_loss_function(args, batch, None, None)[1], 0.5

    model_mod.train_one_step = train_one_step
    assert sp.install_grad_norm_recorder()
    reported = model_mod.train_one_step(optimizer=None)[0]
    assert reported["pg_loss"] == pytest.approx(float(expected[0].mean()))  # observation only
    (record,) = sp.step_losses(None)
    m = record["metrics"]
    # 4.7: token- and sample-level advantage variance
    tok = torch.cat(advs).double()
    assert m["yeto/adv_tokens"] == 5
    assert m["yeto/adv_token_mean"] == pytest.approx(float(tok.mean()))
    assert m["yeto/adv_token_var"] == pytest.approx(float(tok.var(unbiased=False)))
    assert m["yeto/adv_samples"] == 2
    assert m["yeto/adv_sample_mean"] == pytest.approx(0.25)
    assert m["yeto/adv_sample_var"] == pytest.approx(0.5625)
    # 5.2: counts reproduce Miles' per-token clip indicator
    assert m["yeto/ratio_above_high_pos_adv"] == 1  # 1.65 > 1.28 with A > 0
    assert m["yeto/ratio_below_low_neg_adv"] == 1  # 0.61 < 0.8 with A < 0
    assert m["yeto/clipfrac_recomputed"] == pytest.approx(float(seen["clipfrac_tokens"].mean()))
    assert m["yeto/clip_eps_low"] == 0.2 and m["yeto/clip_eps_high"] == 0.28
    assert m["yeto/ratio_max"] == pytest.approx(float(torch.exp(torch.tensor(0.5))))

    model_mod.train_one_step(optimizer=None)  # reset per optimizer step
    (again,) = sp.step_losses(None)
    assert again["metrics"]["yeto/adv_tokens"] == 5


def test_masked_tokens_excluded():
    stats: dict[str, float] = {}
    orig = sp._POLICY_STATS
    try:
        sp._POLICY_STATS = stats
        sp.accumulate_advantage_stats([torch.tensor([2.0, 9.0])], [torch.tensor([1, 0])])
        # Miles zeroes ppo_kl and A on masked tokens: they never count as clipped
        sp.accumulate_ratio_stats(torch.tensor([0.0, -3.0]), torch.tensor([0.0, 1.0]), 0.2, 0.28)
    finally:
        sp._POLICY_STATS = orig
    out = sp.step_policy_stats(stats)
    assert out["yeto/adv_tokens"] == 1 and out["yeto/adv_token_mean"] == 2.0
    assert out["yeto/ratio_tokens"] == 1 and out["yeto/clipfrac_recomputed"] == 1.0


def test_no_data_no_keys():
    assert sp.step_policy_stats({}) == {}
