"""S29 MFU-C: on-policy reuse of the old log-probs (``onpolicy_old_logprob``).

Pins (1) the guard: reuse is refused unless the update pass runs on the recompute's
weights (one optimizer step, trigger_parameter_sync_step 1) and nothing but the PPO
ratio and the TIS weights reads old_log_probs; (2) the equivalence: with old ==
current the reused loss and its gradient equal verl's decoupled loss; (3) the verl
hook anchor and the fail-loud paths (NaN placeholder, missing hook). The numerics
(2) live in test_s29_onpolicy_old_logprob_numerics.py, which needs torch and verl;
this file runs without them.
"""

import json

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


# ---- static checks at plan time ------------------------------------------------------
@pytest.mark.parametrize("over,needle", [
    ({"actor_rollout_ref.actor.ppo_epochs": 2}, "ppo_epochs"),
    ({"async_training.trigger_parameter_sync_step": 2}, "trigger_parameter_sync_step"),
    ({"actor_rollout_ref.actor.policy_loss.loss_mode": "gspo"}, "loss_mode"),
    ({"algorithm.use_kl_in_reward": True}, "use_kl_in_reward"),
    ({"algorithm.rollout_correction.bypass_mode": True}, "bypass_mode"),
    ({"algorithm.rollout_correction.rollout_rs": "token_k1"}, "rollout_rs"),
    ({"algorithm.rollout_correction.rollout_is_batch_normalize": True}, "batch_normalize"),
])
def test_static_check_refuses_without_a_batch(over, needle):
    with pytest.raises(oplp.OnPolicyReuseError, match=needle):
        oplp.check_static(_config(**over))


def test_static_check_ignores_the_batch_size():
    oplp.check_static(_config(**{"actor_rollout_ref.actor.ppo_mini_batch_size": 22}))


def test_config_from_overrides_parses_hydra_values():
    cfg = oplp.config_from_overrides([
        "actor_rollout_ref.actor.ppo_epochs=1", "++algorithm.rollout_correction.rollout_rs=null",
        "+algorithm.use_kl_in_reward=False", "algorithm.rollout_correction.rollout_is='token'",
        "algorithm.rollout_correction.rollout_is_threshold=2.0", "~trainer.logger",
        "async_training.trigger_parameter_sync_step=1", "async_training.trigger_parameter_sync_step=4"])
    assert cfg["actor_rollout_ref"]["actor"]["ppo_epochs"] == 1
    assert cfg["algorithm"]["rollout_correction"] == {"rollout_rs": None, "rollout_is": "token",
                                                      "rollout_is_threshold": 2.0}
    assert cfg["algorithm"]["use_kl_in_reward"] is False
    assert cfg["async_training"]["trigger_parameter_sync_step"] == 4  # the later write wins
    with pytest.raises(oplp.OnPolicyReuseError, match="trigger_parameter_sync_step"):
        oplp.check_static(cfg)


def test_static_check_accepts_the_written_defaults():
    oplp.check_static(oplp.config_from_overrides([
        "actor_rollout_ref.actor.ppo_epochs=1", "algorithm.use_kl_in_reward=False",
        "algorithm.rollout_correction.bypass_mode=False", "algorithm.rollout_correction.rollout_rs=null",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"]))


def test_island_static_check_fails_at_plan_time_only_when_asked():
    from types import SimpleNamespace

    from yeto.rl.adapters.verl import island_entry

    bad = ["actor_rollout_ref.actor.ppo_epochs=2"]
    island_entry.onpolicy_old_logprob_static_check(SimpleNamespace(rl_onpolicy_old_logprob=None), bad)
    island_entry.onpolicy_old_logprob_static_check(SimpleNamespace(rl_onpolicy_old_logprob="off"), bad)
    for mode in ("on", "measure"):
        with pytest.raises(SystemExit, match="ppo_epochs"):
            island_entry.onpolicy_old_logprob_static_check(SimpleNamespace(rl_onpolicy_old_logprob=mode), bad)
    island_entry.onpolicy_old_logprob_static_check(
        SimpleNamespace(rl_onpolicy_old_logprob="on"), ["actor_rollout_ref.actor.ppo_epochs=1"])
