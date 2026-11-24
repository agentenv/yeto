"""ms-swift Megatron GRPO trained from the yeto sample buffer.

Load with ms-swift's plugin mechanism next to ``swift_plugin.py``::

    megatron rlhf --rlhf_type grpo --use_vllm false --reward_funcs yeto_buffer \\
        --external_plugins .../yeto/swift_megatron/rl_buffer.py \\
                           .../yeto/swift_megatron/swift_plugin.py

With ``YETO_SAMPLE_BUFFER_ROOT`` set, the import replaces the rollout of
``MegatronGRPOTrainer``: instead of generating with vLLM, global rank 0 takes
whole groups from the buffer (``BatchReader``, the reader the Miles trainer
island uses), broadcasts them, and every rank installs its slice as the
completions of its local samples (response token ids, rollout log-probs,
finish reason).  The reward is the sampler's reward: ``_score_completions``
returns it instead of calling reward functions.  Everything after the rollout
(reference / old log-probs, advantages, the GRPO loss, the optimizer step and
the yeto DiLoCo callback) is ms-swift's own code.

Environment:
  YETO_SAMPLE_BUFFER_ROOT       buffer store root (required to activate)
  YETO_SWIFT_RL_MAX_POLICY_AGE  BatchReader max_policy_age (default 1000000)
  YETO_SWIFT_RL_WAIT_S          seconds a take waits for new groups (default 600)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT_ENV = "YETO_SAMPLE_BUFFER_ROOT"
REWARD_KEY = "yeto_reward"


def _ensure_repo_on_path() -> None:
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)


def response_fields(sample: dict[str, Any]) -> dict[str, Any]:
    """The swift rollout fields of one buffer sample (``engine_sampler.sample_record``)."""
    n = int(sample["response_length"])
    tokens = list(sample["tokens"])
    if n < 1 or n > len(tokens):
        raise ValueError(f"sample {sample.get('index')}: response_length {n} of {len(tokens)} tokens")
    logprobs = [float(x) for x in (sample.get("rollout_log_probs") or [])]
    if logprobs and len(logprobs) != n:
        raise ValueError(f"sample {sample.get('index')}: {len(logprobs)} log-probs for {n} response tokens")
    return {
        "response_token_ids": [tokens[-n:]],
        "rollout_logprobs": [logprobs] if logprobs else [],
        "finish_reason": "length" if sample.get("status") == "truncated" else "stop",
        "text": str(sample.get("response") or ""),
        "reward": float(sample.get("reward", 0.0)),
    }


def prompt_messages(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [dict(m) for m in prompt if m.get("role") != "assistant"]


def flatten_groups(groups, group_size: int) -> list[dict[str, Any]]:
    out = []
    for _, group in groups:
        if len(group) != group_size:
            raise ValueError(f"buffer group of {len(group)} samples; num_generations is {group_size}")
        out.extend(group)
    return out


class _Reader:
    def __init__(self) -> None:
        from yeto.rl.engine.batch_reader import BatchReader, open_buffer

        self.buffer = open_buffer(batch_size=1, reader=True, create_index=False)
        self.reader = BatchReader(self.buffer,
                                  max_policy_age=int(os.environ.get("YETO_SWIFT_RL_MAX_POLICY_AGE", "1000000")),
                                  wait_s=float(os.environ.get("YETO_SWIFT_RL_WAIT_S", "600")))
        self.rollout_id = 0

    def take(self, groups: int):
        got, _report = self.reader.take(self.rollout_id, groups, current_version=None)
        self.rollout_id += 1
        return got


def install() -> bool:
    if not os.environ.get(ROOT_ENV):
        return False
    _ensure_repo_on_path()
    import torch
    import torch.distributed as dist
    from swift.megatron.trainers.grpo_trainer import MegatronGRPOTrainer

    state: dict[str, Any] = {}

    def _generate_completions(self, samples):
        samples = self._preprocess_inputs(samples)
        n_local = len(samples)
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        ng = int(self.num_generations)
        total = n_local * world
        if total % ng:
            raise ValueError(f"{total} samples per rollout is not a whole number of groups of {ng}")
        payload = [None]
        if rank == 0:
            reader = state.setdefault("reader", _Reader())
            payload = [flatten_groups(reader.take(total // ng), ng)]
        if world > 1:
            dist.broadcast_object_list(payload, src=0)
        mine = payload[0][rank * n_local:(rank + 1) * n_local]
        for s, b in zip(samples, mine):
            f = response_fields(b)
            s.messages = prompt_messages(b["prompt"]) + [{"role": "assistant", "content": f["text"]}]
            s.response_token_ids = f["response_token_ids"]
            s.rollout_logprobs = f["rollout_logprobs"]
            s.finish_reason = f["finish_reason"]
            s.add_eos = False
            s.rollout_infos = {**(s.rollout_infos or {}), REWARD_KEY: f["reward"]}
        return samples

    def _score_completions(self, samples):
        cols = max(1, len(self.reward_funcs or ()))
        r = torch.tensor([float(s.rollout_infos[REWARD_KEY]) for s in samples],
                         dtype=torch.float32, device=self.device)
        return r.unsqueeze(1).repeat(1, cols) / cols

    loss_func = MegatronGRPOTrainer.loss_func

    def counting_loss_func(self, output_tensor, data):
        # GRPO's loss metrics hold no token count; the DiLoCo callback weighs the
        # island by its loss tokens (swift_plugin.LOSS_TOKENS_ATTR), so count the
        # completion tokens of every micro-batch of the step here.
        mask = data["grpo_batch"].completion_mask
        attr = "_yeto_loss_tokens"
        setattr(self, attr, int(getattr(self, attr, None) or 0) + int(mask.sum().item()))
        return loss_func(self, output_tensor, data)

    MegatronGRPOTrainer.loss_func = counting_loss_func
    MegatronGRPOTrainer._generate_completions = _generate_completions
    MegatronGRPOTrainer._score_completions = _score_completions
    _allow_no_vllm()
    _register_reward_name()
    return True


def _allow_no_vllm() -> None:
    """Megatron GRPO asserts ``use_vllm`` at argument parsing
    (``MegatronArguments._init_grpo``); a buffer-fed trainer has no engine.  The
    check runs with ``use_vllm`` raised and the value is restored afterwards, so
    ``_init_rollout_engine`` skips the engine.  The buffer carries the sampler's
    log-probs, so rollout importance sampling stays available (ms-swift ties it
    to the installed vLLM version)."""
    from swift.megatron.arguments.megatron_args import MegatronArguments
    from swift.megatron.trainers.rollout_mixin import MegatronRolloutMixin

    init_grpo = MegatronArguments._init_grpo

    def _init_grpo(self):
        use_vllm, self.use_vllm = self.use_vllm, True
        try:
            return init_grpo(self)
        finally:
            self.use_vllm = use_vllm

    init_engine = MegatronRolloutMixin._init_rollout_engine

    def _init_rollout_engine(self):
        init_engine(self)
        if not self.args.use_vllm:
            self.vllm_version_ge_0_10_2 = True

    MegatronArguments._init_grpo = _init_grpo
    MegatronRolloutMixin._init_rollout_engine = _init_rollout_engine


def _register_reward_name() -> None:
    """``--reward_funcs yeto_buffer``: a name ms-swift's argument checks accept;
    the reward itself comes from the buffer (``_score_completions``)."""
    try:
        from swift.rewards import ORM, orms
    except ImportError:  # older layout
        from swift.plugin import ORM, orms

    class YetoBufferReward(ORM):
        def __call__(self, completions, **kwargs):
            raise RuntimeError("yeto_buffer reward is read from the sample buffer, not computed")

    orms["yeto_buffer"] = YetoBufferReward


install()
