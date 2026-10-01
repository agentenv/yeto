"""CPU check: do sample-mean and per-token aggregation give the same gradient
under Miles' loss scaling + Megatron's forward_step scaling (as on the LoRA
bridge path, where TransformerConfig.calculate_per_token_loss stays False)?

Uses Miles' own get_sum_of_sample_mean (cp_utils) and transcribes the two
scaling steps:
  Miles  loss.py:203-220  (per-token: return loss unscaled + num_tokens;
                            default: loss * num_mb / global_batch, num_tokens=1)
  Megatron schedules.forward_step (config.calculate_per_token_loss False):
                            output /= num_tokens; output /= num_microbatches
Run: PYTHONPATH=/home/michael/work/miles-next miles-next-venv/bin/python repro_token_grad.py
"""
from types import SimpleNamespace

import torch

from miles.backends.training_utils import cp_utils

cp_utils.get_parallel_state = lambda: SimpleNamespace(cp=SimpleNamespace(size=1))
torch.manual_seed(0)
lengths = [3, 7, 5, 9]            # varied response lengths (as in the G1 batch)
adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
masks = [torch.ones(n) for n in lengths]
N = int(sum(lengths)); n_mb = 1; global_batch = len(lengths)
theta0 = torch.randn(N, dtype=torch.float64)


def grad(per_token: bool, megatron_config_per_token: bool = False):
    theta = theta0.clone().requires_grad_(True)
    logp = torch.nn.functional.logsigmoid(theta)          # stand-in log-probs
    ratio = torch.exp(logp - logp.detach())
    x = -ratio * torch.cat([a.repeat(n) for a, n in zip(adv, lengths)])
    reducer = cp_utils.get_sum_of_sample_mean([0] * 4, lengths, masks, per_token)
    loss = reducer(x)
    if not per_token:                                     # Miles loss.py:203-210
        loss = loss * n_mb / global_batch
        num_tokens = 1
    else:                                                 # Miles loss.py:212-220
        num_tokens = N
    if not megatron_config_per_token:                     # Megatron forward_step
        loss = loss / num_tokens / n_mb
    loss.backward()
    return theta.grad


g_sample, g_token = grad(False), grad(True)
print("grad_norm sample-mean :", g_sample.norm().item())
print("grad_norm per-token   :", g_token.norm().item())
print("bitwise equal         :", torch.equal(g_sample, g_token))
print("equal-length control  :", end=" ")
lengths = [5, 5, 5, 5]; masks = [torch.ones(5)] * 4; N = 20; theta0 = torch.randn(N, dtype=torch.float64)
print(torch.allclose(grad(False), grad(True)))
