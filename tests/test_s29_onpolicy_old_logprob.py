"""S29 MFU-C: on-policy reuse of the old log-probs (``onpolicy_old_logprob``).

Pins (1) the guard: reuse is refused unless the update pass runs on the recompute's
weights (one optimizer step, trigger_parameter_sync_step 1) and nothing but the PPO
ratio and the TIS weights reads old_log_probs; (2) the equivalence: with old ==
current the reused loss and its gradient equal verl's decoupled loss; (3) the verl
hook anchor and the fail-loud paths (NaN placeholder, missing hook).
"""

import json
from pathlib import Path

import pytest

from yeto.rl.adapters.verl import onpolicy_old_logprob as oplp
from yeto.rl.adapters.verl import patch_verl


def _config(**over):
    cfg = {
        "actor_rollout_ref": {"actor": {"ppo_epochs": 1, "ppo_mini_batch_size": 44,
                                        "policy_loss": {"loss_mode": "vanilla"}},
                              "rollout": {"n": 4}},
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False,
                      "rollout_correction": {"rollout_is": "token", "rollout_is_threshold": 2.0,
                                             "bypass_mode": False, "rollout_rs": None,
                                             "rollout_is_batch_normalize": False}},
        "async_training": {"trigger_parameter_sync_step": 1},
    }
    for path, value in over.items():
        node = cfg
        *head, last = path.split(".")
        for key in head:
            node = node[key]
        node[last] = value
    return cfg


def test_mode_parsing():
    assert oplp.mode({}) == "off"
    assert oplp.mode({oplp.ENV: "on"}) == "on"
    assert oplp.mode({oplp.ENV: "1"}) == "on"
    assert oplp.mode({oplp.ENV: "measure"}) == "measure"
    assert oplp.mode({oplp.ENV: "off"}) == "off"
    assert oplp.mode({oplp.ENV: "bogus"}) == "off"


def test_guard_accepts_one_step_round():
    tis = oplp.check_config(_config(), batch_rows=176)
    assert tis == {"rollout_is": "token", "rollout_is_threshold": 2.0}
    assert json.loads(oplp.meta("on", tis))["mode"] == "on"


@pytest.mark.parametrize("over,rows,needle", [
    ({"actor_rollout_ref.actor.ppo_epochs": 2}, 176, "ppo_epochs"),
    ({"actor_rollout_ref.actor.ppo_mini_batch_size": 22}, 176, "mini-batch"),
    ({}, 88, "mini-batch"),
    ({"async_training.trigger_parameter_sync_step": 2}, 176, "trigger_parameter_sync_step"),
    ({"actor_rollout_ref.actor.policy_loss.loss_mode": "gspo"}, 176, "loss_mode"),
    ({"algorithm.use_kl_in_reward": True}, 176, "use_kl_in_reward"),
    ({"algorithm.rollout_correction.bypass_mode": True}, 176, "bypass_mode"),
    ({"algorithm.rollout_correction.rollout_rs": "token_k1"}, 176, "rollout_rs"),
    ({"algorithm.rollout_correction.rollout_is_batch_normalize": True}, 176, "batch_normalize"),
])
def test_guard_refuses(over, rows, needle):
    with pytest.raises(oplp.OnPolicyReuseError, match=needle):
        oplp.check_config(_config(**over), batch_rows=rows)


def test_guard_refuses_later_trigger_step():
    with pytest.raises(oplp.OnPolicyReuseError, match="local_trigger_step"):
        oplp.check_config(_config(), batch_rows=176, local_trigger_step=2)


def test_missing_hook_is_refused():
    with pytest.raises(oplp.OnPolicyReuseError, match="hook did not run"):
        oplp.derived_metrics({"actor/pg_loss": 0.1}, "on")
    got = oplp.derived_metrics({"actor/yeto_oplp/active": 1.0, "actor/yeto_oplp/tok": 4.0,
                                "actor/yeto_oplp/is_w_sum": 3.0, "actor/yeto_oplp/is_high_sum": 1.0,
                                "actor/yeto_oplp/diff_sum": 0.2}, "measure")
    assert got == {"yeto_oplp/rollout_is_mean": 0.75, "yeto_oplp/rollout_is_ratio_fraction_high": 0.25,
                   "yeto_oplp/old_vs_cur_abs_diff_mean": 0.05}


def test_hook_anchor_and_profile():
    target, anchor, hook, mode = patch_verl.ONPOLICY_OLDLP
    assert target == "verl/workers/utils/losses.py" and mode == "replace"
    assert patch_verl.ONPOLICY_OLDLP in patch_verl.PROFILES["megatron"]
    assert anchor.rstrip("\n").splitlines()[-1] in hook  # the replaced lines are kept
    assert "yeto_onpolicy_old_logprob" in hook and oplp.META_KEY == "yeto_onpolicy_old_logprob"


# ---- numerics (need torch and verl) -------------------------------------------------
torch = pytest.importorskip("torch")
rch = pytest.importorskip("verl.trainer.ppo.rollout_corr_helper")
core = pytest.importorskip("verl.trainer.ppo.core_algos")


class _Data(dict):
    def keys(self):  # TensorDict-like
        return super().keys()


def _batch(seed=0, bs=3, width=7):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(bs, width)
    mask[0, 5:] = 0
    mask[2, 3:] = 0
    current = -torch.rand(bs, width, generator=g) * 3
    rollout = current + torch.randn(bs, width, generator=g) * 0.8  # some weights > 2
    adv = torch.randn(bs, 1, generator=g).expand(bs, width).contiguous()
    return current, rollout, mask, adv


def _vanilla(old, logp, adv, mask, weights):
    class Cfg(dict):
        clip_ratio = 0.2
        clip_ratio_low = 0.2
        clip_ratio_high = 0.2
        clip_ratio_c = 3.0
        loss_agg_mode = "token-mean"
        global_batch_info = {}

        def get(self, k, d=None):
            return getattr(self, k, d)

    cfg = Cfg()
    loss, _ = core.compute_policy_loss_vanilla(old_log_prob=old, log_prob=logp, advantages=adv,
                                               response_mask=mask.bool(), loss_agg_mode="token-mean",
                                               config=cfg, rollout_is_weights=weights)
    return loss


def test_on_mode_equals_decoupled_when_old_is_current():
    current, rollout, mask, adv = _batch()
    tis = {"rollout_is": "token", "rollout_is_threshold": 2.0}
    # decoupled (verl today): old recomputed == current, driver TIS weights from old
    w_ref, _ = rch.compute_rollout_correction_weights(current - rollout, mask, "token", 2.0, False)
    lp_ref = current.clone().requires_grad_(True)
    loss_ref = _vanilla(current.clone(), lp_ref, adv, mask, w_ref)
    loss_ref.backward()
    # reuse: NaN placeholder old, no driver weights; the hook fills both
    lp = current.clone().requires_grad_(True)
    data = _Data(response_mask=mask, rollout_log_probs=rollout)
    old, w, stats = oplp.in_loss(oplp.meta("on", tis), lp, torch.full_like(current, float("nan")), data, None)
    assert not old.requires_grad and torch.equal(old, current)
    assert torch.equal(w, w_ref)
    loss = _vanilla(old, lp, adv, mask, w)
    loss.backward()
    assert torch.equal(loss, loss_ref)
    assert torch.allclose(lp.grad, lp_ref.grad, rtol=0, atol=0)
    assert (w_ref * mask).max() == 2.0  # the batch exercises the truncation
    assert stats["is_high_sum"][0] == float((((current - rollout).exp() > 2.0) & mask.bool()).sum())
    assert stats["is_w_sum"][0] == pytest.approx(float((w_ref * mask).sum()))
    assert stats["tok"][0] == float(mask.sum())


def test_measure_mode_leaves_loss_inputs_and_reports_difference():
    current, rollout, mask, _ = _batch(1)
    noise = torch.zeros_like(current)
    noise[1, 2] = 3e-3
    recomputed = current + noise
    w_ref, _ = rch.compute_rollout_correction_weights(recomputed - rollout, mask, "token", 2.0, False)
    data = _Data(response_mask=mask, rollout_log_probs=rollout)
    old, w, stats = oplp.in_loss(oplp.meta("measure", {"rollout_is": "token", "rollout_is_threshold": 2.0}),
                                 current.clone(), recomputed, data, w_ref)
    assert old is recomputed and w is w_ref
    assert stats["diff_max"][0] == pytest.approx(3e-3, rel=1e-4)
    assert stats["diff_sum"][0] == pytest.approx(3e-3, rel=1e-4)


def test_small_recompute_noise_is_second_order():
    """The real difference between the two paths: verl's ratio exp(lp - old) is 1 + eps
    with eps the forward-to-forward noise; reuse makes it exactly 1."""
    current, rollout, mask, adv = _batch(2)
    old = current + torch.randn_like(current) * 1e-4
    w_ref, _ = rch.compute_rollout_correction_weights(old - rollout, mask, "token", 2.0, False)
    lp_ref = current.clone().requires_grad_(True)
    _vanilla(old, lp_ref, adv, mask, w_ref).backward()
    lp = current.clone().requires_grad_(True)
    o, w, _ = oplp.in_loss(oplp.meta("on", {"rollout_is": "token", "rollout_is_threshold": 2.0}), lp,
                           old, _Data(response_mask=mask, rollout_log_probs=rollout), None)
    _vanilla(o, lp, adv, mask, w).backward()
    assert torch.allclose(lp.grad, lp_ref.grad, atol=1e-4 * float(lp.grad.abs().max()) * 10)


def test_on_mode_needs_rollout_log_probs():
    current, _, mask, _ = _batch()
    with pytest.raises(oplp.OnPolicyReuseError, match="rollout_log_probs"):
        oplp.in_loss(oplp.meta("on", {"rollout_is": "token", "rollout_is_threshold": 2.0}),
                     current, current, _Data(response_mask=mask), None)


def test_installed_verl_has_the_anchor_once():
    import verl

    losses = Path(verl.__file__).parent / "workers" / "utils" / "losses.py"
    text = losses.read_text()
    _, anchor, hook, _ = patch_verl.ONPOLICY_OLDLP
    assert text.count(anchor) == 1 or patch_verl._applied(text, hook)
