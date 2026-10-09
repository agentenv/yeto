"""Neutral name -> verl binding and the three-way check (rl-verl-backend D11; decoupling 4.4a style).

What the verl island binds by name, and how each binding is checked before launch:

1. config -- every neutral algorithm/run field maps to verl hydra keys whose
   override strings equal, character for character, the table below
   (``CONFIG_GOLDEN``), built through ``config.build_overrides``;
2. identity -- the reward function verl loads (``reward_fn``) is recorded by
   module, qualname and module-source sha256, and so is the yeto function it
   wraps; the verl advantage estimator ``grpo`` is recorded by verl's registry
   function identity when verl is importable (skipped, and reported as
   skipped, on machines without verl);
3. call -- one fixed CPU input through the verl wrapper equals a direct call
   of the yeto reward function.

The result is a JSON-able dict for the runtime manifest / review documents.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect

from . import config as vconf
from . import reward_fn

BINDING_SCHEMA = "yeto-verl-binding-v1"

# neutral field -> exact verl override strings (fork acad9875 key names)
CONFIG_GOLDEN = {
    "advantage.estimator=grpo": ["algorithm.adv_estimator=grpo"],
    "advantage.std_normalization=true": ["algorithm.norm_adv_by_std_in_grpo=True"],
    "kl.placement=none": ["algorithm.use_kl_in_reward=False", "actor_rollout_ref.actor.use_kl_loss=False"],
    "entropy_coef=0": ["actor_rollout_ref.actor.entropy_coeff=0"],
    "correction.method=tis(upper 2.0, lower 0)": [
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=2.0"],
    "lora(rank 32, alpha=rank)": ["actor_rollout_ref.model.lora_rank=32",
                                  "actor_rollout_ref.model.lora_alpha=32"],
    "sampling(T=1, top_p=1, top_k=-1)": ["actor_rollout_ref.rollout.temperature=1.0",
                                         "actor_rollout_ref.rollout.top_p=1.0",
                                         "actor_rollout_ref.rollout.top_k=-1"],
    "old_logprob=recompute (no bypass)": ["algorithm.rollout_correction.bypass_mode=False"],
}

CALL_SAMPLE = ("The answer is \\boxed{72}.", "72")


class BindingError(RuntimeError):
    pass


def _identity(fn) -> dict:
    module = inspect.getmodule(fn)
    source = inspect.getsource(module)
    return {"module": module.__name__, "qualname": fn.__qualname__,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest()}


def _sample_config() -> vconf.VerlRunConfig:
    return vconf.VerlRunConfig(model_path="/m", train_file="/t", val_file="/v", out_dir="/o",
                               lora_rank=32, lora_alpha=32, correction="tis", tis_upper=2.0)


def check_bindings(*, verl_available: bool | None = None) -> dict:
    overrides = set(vconf.build_overrides(_sample_config()))
    config_rows = {}
    for neutral, wanted in CONFIG_GOLDEN.items():
        missing = [w for w in wanted if w not in overrides]
        config_rows[neutral] = {"verl": wanted, "ok": not missing, "missing": missing}
    bad = [k for k, v in config_rows.items() if not v["ok"]]
    if bad:
        raise BindingError(f"verl config binding differs from the golden table: {bad}")

    from yeto.rl.rewards.builtin import gsm8k_reward
    from yeto.rl.rewards.types import Trajectory

    wrapper = getattr(reward_fn, reward_fn.function_for("gsm8k_reward:score"))
    identity = {"verl_reward_wrapper": _identity(wrapper), "yeto_reward": _identity(gsm8k_reward)}
    response, label = CALL_SAMPLE
    via_verl = wrapper("yeto/zhuzilin/gsm8k", response, label)
    direct = float(gsm8k_reward(Trajectory(response=response, label=label)).value)
    if via_verl != direct:
        raise BindingError(f"reward via verl wrapper {via_verl} != yeto {direct}")
    call = {"input": list(CALL_SAMPLE), "via_verl": via_verl, "direct": direct, "equal": True}

    if verl_available is None:
        verl_available = importlib.util.find_spec("verl") is not None
    if verl_available:
        from verl.trainer.ppo import core_algos

        fn = core_algos.get_adv_estimator_fn("grpo")
        identity["verl_adv_estimator_grpo"] = _identity(fn)
    else:
        identity["verl_adv_estimator_grpo"] = {"skipped": "verl not importable on this machine"}
    return {"schema": BINDING_SCHEMA, "config": config_rows, "identity": identity, "call": call}
