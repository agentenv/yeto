"""CPU stand-ins for DP-sharded (DistributedOptimizer-like) LoRA ranks used by the 4.6 reshard tests.

Protocol only -- NOT a GPU/Megatron substitute and not acceptance evidence.
Each DP rank owns the range ``[numel*dp//D, numel*(dp+1)//D)`` of every
adapter parameter (FP32 main slice, Adam moments, step), exports it in the
fork-M5 ``lora_named_optimizer_v2`` shape and merges/slices like fork-M5.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import torch

GBS = 4
MBS = 1
FORMAT = "lora_named_optimizer_v2"


class RangeAdam:
    """Adam over this rank's range of every parameter (elementwise, like a DistOpt shard)."""

    def __init__(self, named, dp, dp_size, lr=0.05, betas=(0.9, 0.999), eps=1e-8):
        self.param_groups = [{"lr": lr, "betas": betas, "eps": eps, "weight_decay": 0.0, "params": []}]
        self.slots = {}
        for name, p in named:
            n = p.numel()
            start, end = n * dp // dp_size, n * (dp + 1) // dp_size
            self.slots[name] = {"p": p, "start": start, "end": end,
                                "param": p.detach().reshape(-1)[start:end].clone().float(),
                                "exp_avg": None, "exp_avg_sq": None, "step": None}

    def init_state(self):
        for s in self.slots.values():
            if s["exp_avg"] is None:
                s["exp_avg"] = torch.zeros_like(s["param"])
                s["exp_avg_sq"] = torch.zeros_like(s["param"])
                s["step"] = torch.tensor(0.0)

    def step_with(self, grads):
        self.init_state()
        g0 = self.param_groups[0]
        b1, b2 = g0["betas"]
        for name, s in self.slots.items():
            g = grads[name].reshape(-1)[s["start"]:s["end"]]
            s["step"] = s["step"] + 1
            s["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)
            s["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1 - b2)
            t = float(s["step"])
            denom = (s["exp_avg_sq"] / (1 - b2 ** t)).sqrt().add_(g0["eps"])
            s["param"].addcdiv_(s["exp_avg"], denom, value=-g0["lr"] / (1 - b1 ** t))


class ShardedBackend:
    def __init__(self, coord):
        self._coord = coord

    def coord(self):
        return dict(self._coord)

    def named_parameters(self, model):
        return list(model[0].named_parameters())

    def is_adapter(self, name):
        return "lora" in name

    def export_optimizer(self, optimizer, named):
        optimizer.init_state()
        hyper = {k: v for k, v in optimizer.param_groups[0].items() if k != "params"}
        entries = {}
        for name, s in optimizer.slots.items():
            entries[name] = {
                "numel": s["p"].numel(), "shape": tuple(s["p"].shape), "start": s["start"], "end": s["end"],
                "tensors": {"param": s["param"].clone(), "exp_avg": s["exp_avg"].clone(),
                            "exp_avg_sq": s["exp_avg_sq"].clone()},
                "scalars": {"step": s["step"].clone()},
                "hyper": dict(hyper),
            }
        return {"format": FORMAT, "entries": entries}

    def check_optimizer(self, optimizer, named, states):
        merged, covered = {}, {}
        for shard in states:
            assert shard["format"] == FORMAT
            for name, e in shard["entries"].items():
                if name not in merged:
                    merged[name] = {"numel": e["numel"], "shape": tuple(e["shape"]),
                                    "tensors": {k: torch.zeros(e["numel"], dtype=v.dtype) for k, v in e["tensors"].items()},
                                    "scalars": dict(e["scalars"]), "hyper": e["hyper"]}
                    covered[name] = torch.zeros(e["numel"], dtype=torch.bool)
                for k, v in e["tensors"].items():
                    merged[name]["tensors"][k][e["start"]:e["end"]] = v
                if not all(torch.equal(merged[name]["scalars"][k], v) for k, v in e["scalars"].items()):
                    raise ValueError(f"DP shards disagree on the step of {name}")
                covered[name][e["start"]:e["end"]] = True
        missing = sorted(n for n, m in covered.items() if not bool(m.all()))
        if missing:
            raise ValueError(f"optimizer state of {missing} is not fully covered; a DP shard file is missing")
        names = {n for n, _ in named}
        if set(merged) != names:
            raise KeyError(f"named optimizer state differs: {sorted(names ^ set(merged))}")
        return merged

    def load_optimizer(self, optimizer, named, merged):
        for name, s in optimizer.slots.items():
            full = merged[name]
            s["param"] = full["tensors"]["param"][s["start"]:s["end"]].clone()
            s["exp_avg"] = full["tensors"]["exp_avg"][s["start"]:s["end"]].clone()
            s["exp_avg_sq"] = full["tensors"]["exp_avg_sq"][s["start"]:s["end"]].clone()
            s["step"] = full["scalars"]["step"].clone()
        optimizer.param_groups[0].update(merged[next(iter(merged))]["hyper"])

    def capture_rng(self):
        return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}

    def restore_rng(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])

    def megatron_counters(self):
        return {}

    def set_megatron_counters(self, counters):
        pass


class Scheduler:
    def __init__(self, optimizer, lr=0.05):
        self.optimizer, self.lr0, self.num_steps = optimizer, lr, 0

    def step(self, increment):
        self.num_steps += increment
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr0 / (1 + self.num_steps / GBS)

    def state_dict(self):
        return {"num_steps": self.num_steps, "lr0": self.lr0}

    def load_state_dict(self, state):
        assert self.lr0 == state["lr0"]
        self.step(state["num_steps"])


class Lora(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(1234)
        self.base = torch.nn.Linear(6, 6)
        self.base.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(torch.randn(3, 6, generator=g) * 0.1)
        self.lora_B = torch.nn.Parameter(torch.randn(6, 3, generator=g) * 0.1)

    def forward(self, x):
        return self.base(x) + x @ self.lora_A.T @ self.lora_B.T


def make_world(dp_size, *, seed=0, args=None):
    """``dp_size`` freshly built ranks (identical replicas, like a Megatron DP group after init)."""
    ranks = []
    for dp in range(dp_size):
        model = Lora(seed)
        named = [(n, p) for n, p in model.named_parameters() if "lora" in n]
        opt = RangeAdam(named, dp, dp_size)
        coord = {"global_rank": dp, "tp": 0, "pp": 0, "dp": dp, "dp_size": dp_size,
                 "tp_size": 1, "pp_size": 1, "cp_size": 1, "ep_size": 1}
        ranks.append(SimpleNamespace(
            args=args or default_args(dp_size), model=[model], optimizer=opt,
            opt_param_scheduler=Scheduler(opt), _yeto_cut_backend=ShardedBackend(coord),
        ))
    return ranks


def default_args(dp_size):
    return SimpleNamespace(fp16=False, bf16=False, global_batch_size=GBS, micro_batch_size=MBS,
                           actor_num_nodes=1, actor_num_gpus_per_node=dp_size, num_steps_per_rollout=1,
                           use_distributed_optimizer=True, seed=1234,
                           lora_dropout=0.0, hidden_dropout=0.0, attention_dropout=0.0)


def train_step(ranks, batch):
    """One DP step: round-robin partition, per-sample loss weight 1/GBS, sum-reduce, range Adam, all-gather."""
    dp = len(ranks)
    partial = []
    for r, rank in enumerate(ranks):
        model = rank.model[0]
        for p in model.parameters():
            p.grad = None
        idx = list(range(r, batch.shape[0], dp))
        loss = sum(model(batch[i:i + 1]).pow(2).mean() for i in idx) / GBS
        loss.backward()
        partial.append({n: p.grad.detach().clone() for n, p in model.named_parameters() if "lora" in n})
    reduced = {n: sum(g[n] for g in partial) for n in partial[0]}
    for rank in ranks:
        rank.optimizer.step_with(reduced)
        rank.opt_param_scheduler.step(GBS)
    # all-gather the updated main ranges into every replica
    for name in reduced:
        pieces = [rank.optimizer.slots[name]["param"] for rank in ranks]
        full = torch.cat(pieces)
        for rank in ranks:
            p = dict(rank.model[0].named_parameters())[name]
            with torch.no_grad():
                p.copy_(full.reshape(p.shape))
    random.random()


def params(ranks):
    return {n: p.detach().clone() for n, p in ranks[0].model[0].named_parameters() if "lora" in n}


def gathered(ranks):
    """Full optimizer state gathered over the DP ranks (for comparisons in tests)."""
    backend = ranks[0]._yeto_cut_backend
    named = [(n, p) for n, p in ranks[0].model[0].named_parameters() if "lora" in n]
    return backend.check_optimizer(ranks[0].optimizer, named,
                                   [r._yeto_cut_backend.export_optimizer(r.optimizer, named) for r in ranks])
