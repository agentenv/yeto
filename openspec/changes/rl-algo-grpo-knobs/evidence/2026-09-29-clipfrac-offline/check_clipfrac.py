"""Offline: does Miles' pg_clipfrac see the upper bound? compute_policy_loss
(math_utils.py:254-277) with eps_clip_high = eps_clip (A) vs 10 (B) on the
same ppo_kl/advantages. Run: PYTHONPATH=/home/michael/work/miles-next
miles-next-venv/bin/python check_clipfrac.py"""
import torch
from miles.backends.training_utils.loss_hub import math_utils as mu

f = getattr(mu.compute_policy_loss, "_torchdynamo_orig_callable", mu.compute_policy_loss)
torch.manual_seed(0)
r = torch.exp(torch.randn(4096) * 0.01)
ppo_kl = -torch.log(r)
adv = torch.randn(4096)
for name, hi in (("A sym", 0.001), ("B hi=10", 10.0)):
    loss, cf = f(ppo_kl, adv, 0.001, hi, None)
    print(name, "clipfrac", cf.mean().item(), "loss", loss.mean().item())
up = ((adv > 0) & (r > 1.001)).float().mean().item()
print("fraction of tokens with A>0 and ratio>1.001 (only these differ):", up)
