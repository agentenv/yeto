"""critic_lr_warmup launch check (S13 G1 SAO: warmup 10 vs decay 3 -> Megatron assert)."""

from yeto.rl.algos.critic import critic_lr_warmup_problems
from yeto.rl.algos.sao import sao_algorithm_spec
from yeto.rl.engine.algorithm import AlgorithmSpec, launch_problems


def _vals(**kw):
    v = {"num_rollout": 3, "rollout_batch_size": 4, "n_samples_per_prompt": 8,
         "global_batch_size": 32, "lr_decay_iters": None, "extra_argv": ()}
    v.update(kw)
    return v


def test_g1_explicit_decay_is_refused():
    spec = sao_algorithm_spec("reasoning")
    probs = critic_lr_warmup_problems(spec, _vals(lr_decay_iters=3))
    assert probs and "critic_lr_warmup=10" in probs[0] and "(3)" in probs[0]
    assert any(p.startswith("[critic_lr_warmup]") for p in launch_problems(spec, _vals(lr_decay_iters=3)))


def test_default_decay_counts_critic_epochs():
    spec = sao_algorithm_spec("reasoning", num_critic_epochs=2)
    # 3*4*8*2 // 32 = 6 <= 10 -> refused; 20 rounds -> 40 > 10 -> fine
    assert critic_lr_warmup_problems(spec, _vals())
    assert critic_lr_warmup_problems(spec, _vals(num_rollout=20)) == []


def test_boundary_equal_is_refused_and_below_ok():
    spec = sao_algorithm_spec("reasoning")
    assert critic_lr_warmup_problems(spec, _vals(lr_decay_iters=10))
    assert critic_lr_warmup_problems(spec, _vals(lr_decay_iters=11)) == []


def test_skips_when_unknown_or_overridden_or_no_critic():
    spec = sao_algorithm_spec("reasoning")
    assert critic_lr_warmup_problems(spec, {"rollout_batch_size": 4}) == []
    assert critic_lr_warmup_problems(spec, _vals(num_rollout=0)) == []
    assert critic_lr_warmup_problems(spec, _vals(lr_decay_iters=3, extra_argv=("--lr-decay-iters", "50"))) == []
    assert critic_lr_warmup_problems(AlgorithmSpec(), _vals(lr_decay_iters=1)) == []
