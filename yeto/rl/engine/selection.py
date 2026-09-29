"""``--rl-engine`` selection and the R0 ports support matrix (task 5.1).

Import-light (no torch/ray/miles): the CLI, launcher, SSH harness, learner
and benchmark all call :func:`require_ports_supported` before any remote or
GPU work starts. The choice is always explicit; nothing here infers it.

R0 ports support (spec ``rl-engine-selection`` "不支持的组合明确拒绝"):
causal LM + LoRA, GRPO, serial colocated, strict-avg / decoupled presets.
Everything else is rejected with a pointer to ``--rl-engine legacy``.
"""

from __future__ import annotations

from collections.abc import Sequence

RL_ENGINES = ("legacy", "ports")
DEFAULT_RL_ENGINE = "legacy"
PORTS_SYNC_PRESETS = frozenset({"strict-avg", "decoupled"})


class UnsupportedPortsCombination(ValueError):
    """The requested run is only supported by the legacy engine path."""


def _flag_value(argv: Sequence[str], flag: str) -> str | None:
    tokens = list(argv)
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def ports_rejections(
    *,
    sync_preset: str = "strict-avg",
    parameter_mode: str = "lora",
    model_kind: str = "causal-lm",
    tuning: str = "lora",
    model_recipe: str = "generic",
    lora_targets: str | None = None,
    expert_full_count: int = 0,
    rollout_num_gpus: int | None = None,
    use_critic: bool = False,
    advantage_estimator: str = "grpo",
    extra_argv: Sequence[str] = (),
) -> list[str]:
    """Reasons the combination is outside the R0 ports matrix (empty = OK)."""

    reasons: list[str] = []
    if model_kind != "causal-lm":
        reasons.append(f"model kind {model_kind!r} (ports supports causal LMs only)")
    if tuning != "lora" or parameter_mode != "lora" or sync_preset == "dense-full":
        reasons.append("full-parameter / dense-full training")
    if "sao" in str(sync_preset) or any(str(t).startswith("--sao") for t in extra_argv):
        reasons.append("SAO (streaming compaction)")
    elif sync_preset not in PORTS_SYNC_PRESETS and sync_preset != "dense-full":
        reasons.append(f"sync preset {sync_preset!r}")
    if (
        model_recipe == "deepseek-v4-flash"
        or lora_targets == "attention-routed-experts"
        or int(expert_full_count or 0) > 0
    ):
        reasons.append("DeepSeek V4 recipe (clone/expert-full LoRA)")
    estimator = _flag_value(extra_argv, "--advantage-estimator") or advantage_estimator
    if use_critic or "--use-critic" in extra_argv or estimator != "grpo":
        reasons.append(f"critic / non-GRPO advantage estimator ({estimator})")
    if rollout_num_gpus is not None or _flag_value(extra_argv, "--rollout-num-gpus"):
        reasons.append("fixed partition (dedicated rollout GPUs)")
    return reasons


def require_ports_supported(**kwargs) -> None:
    reasons = ports_rejections(**kwargs)
    if reasons:
        raise UnsupportedPortsCombination(
            "--rl-engine ports does not support: "
            + "; ".join(reasons)
            + ". This combination is only supported by the legacy path "
            "(--rl-engine legacy)."
        )


def check_rl_engine(value: str) -> str:
    if value not in RL_ENGINES:
        raise ValueError(f"--rl-engine must be one of {RL_ENGINES}, got {value!r}")
    return value
