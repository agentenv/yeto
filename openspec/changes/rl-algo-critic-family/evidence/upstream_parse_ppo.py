"""2.2: upstream Miles c35702e parse_args on the ports critic argv.

The venv has no megatron-core, so the argv is parsed with the FSDP parser and
``miles_validate_args`` runs with ``train_backend`` switched to megatron (the
shared PPO block, arguments.py:3591-3610, requires it). Usage:
PYTHONPATH=/tmp/miles-c35702e:<worktree> miles-next-venv/bin/python upstream_parse_ppo.py
"""
import sys

from miles.utils import arguments

from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.miles_adapter import algorithm_flags as af

original = arguments.miles_validate_args


def validate(args):
    args.train_backend = "megatron"
    return original(args)


arguments.miles_validate_args = validate
CASES = {
    "ppo_defaults": ["--advantage-estimator", "ppo"],
    "ppo_tuned": ["--advantage-estimator", "ppo", "--gamma", "0.99", "--lambd", "0.95",
                  "--value-clip", "0.3", "--critic-lr", "1e-6", "--critic-lr-warmup-iters", "5"],
    "ppo_load": ["--advantage-estimator", "ppo", "--critic-load", "/tmp/critic-tiny"],
}
for name, extra in CASES.items():
    spec, rest, _ = af.absorb_extra_argv(AlgorithmSpec(), extra)
    assert not spec.rejections(), spec.rejections()
    argv = ["train.py", "--train-backend", "fsdp", "--hf-checkpoint", "/tmp/critic-tiny",
            "--ref-load", "/tmp/critic-tiny", "--rollout-batch-size", "2",
            "--n-samples-per-prompt", "2", "--global-batch-size", "4", "--num-rollout", "1",
            "--colocate", "--actor-num-gpus-per-node", "1",
            "--advantage-estimator", spec.advantage_estimator, *af.algorithm_argv(spec)]
    sys.argv = argv
    args = arguments.parse_args()
    got = {k: getattr(args, k) for k in (
        "use_critic", "gamma", "lambd", "value_clip", "critic_lr", "critic_lr_warmup_iters",
        "num_critic_only_steps", "critic_load", "critic_num_gpus_per_node", "offload_train",
        "kl_coef")}
    print(name, af.algorithm_argv(spec), got)
    assert got["use_critic"] and got["num_critic_only_steps"] == 0
    assert got["gamma"] == spec.advantage.gamma and got["lambd"] == spec.advantage.lambd
    assert got["value_clip"] == spec.critic.value_clip
print("OK")
