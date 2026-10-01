"""CPU stand-ins for a Megatron LoRA rank used by the cut tests (not a GPU substitute)."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import torch

GBS = 4


class TorchCutBackend:
    """Same interface as ``MilesCutBackend`` over a plain torch optimizer."""

    def __init__(self, coord=None):
        self._coord = coord or {"global_rank": 0, "tp": 0, "pp": 0, "dp": 0, "dp_size": 1, "cp_size": 1, "ep_size": 1,
                                "tp_size": 1, "pp_size": 1}

    def coord(self):
        return dict(self._coord)

    def named_parameters(self, model):
        return list(model[0].named_parameters())

    def is_adapter(self, name):
        return "lora" in name

    def export_optimizer(self, optimizer, named):
        by_id = {id(p): n for n, p in named}
        entries = {}
        for group in optimizer.param_groups:
            hyper = {k: v for k, v in group.items() if k != "params"}
            for p in group["params"]:
                state = optimizer.state.get(p, {})
                entries[by_id[id(p)]] = {
                    "param": p.detach().clone(),
                    "state": {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()},
                    "hyper": dict(hyper),
                }
        return {"format": "torch-test", "entries": entries}

    def check_optimizer(self, optimizer, named, states):
        assert len(states) == 1
        merged = states[0]["entries"]
        names = {n for n, p in named if p.requires_grad}
        if set(merged) != names:
            raise ValueError(f"optimizer names differ {sorted(set(merged) ^ names)}")
        return merged

    def load_optimizer(self, optimizer, named, merged):
        by_name = dict(named)
        for name, entry in merged.items():
            p = by_name[name]
            p.data.copy_(entry["param"])
            optimizer.state[p] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in entry["state"].items()}
            for group in optimizer.param_groups:
                if any(q is p for q in group["params"]):
                    group.update(entry["hyper"])

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
    """Megatron-like OptimizerParamScheduler: ``num_steps`` counts samples."""

    def __init__(self, optimizer, lr=0.1):
        self.optimizer, self.lr0, self.num_steps = optimizer, lr, 0
        self._apply()

    def _apply(self):
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr0 / (1 + self.num_steps / GBS)

    def step(self, increment):
        self.num_steps += increment
        self._apply()

    def state_dict(self):
        return {"num_steps": self.num_steps, "lr0": self.lr0}

    def load_state_dict(self, state):
        # Megatron semantics: hyper-parameters via _check_and_set, then
        # step(increment=num_steps) -- progress is ADDED to the current one.
        assert self.lr0 == state["lr0"], "_check_and_set mismatch"
        self.step(state["num_steps"])


class LoraModule(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.base = torch.nn.Linear(6, 6)
        self.base.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(torch.randn(2, 6, generator=g) * 0.1)
        self.lora_B = torch.nn.Parameter(torch.randn(6, 2, generator=g) * 0.1)
        self.drop = torch.nn.Dropout(0.3)

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.lora_A.T @ self.lora_B.T


def make_rank(seed=0, *, args=None, coord=None):
    torch.manual_seed(1234)  # frozen base identical in every build
    module = LoraModule(seed)
    opt = torch.optim.Adam([module.lora_A, module.lora_B], lr=0.1)
    return SimpleNamespace(
        args=args or SimpleNamespace(fp16=False, bf16=False, global_batch_size=GBS),
        model=[module],
        optimizer=opt,
        opt_param_scheduler=Scheduler(opt),
        _yeto_cut_backend=TorchCutBackend(coord),
    )


def train_step(rank, batch):
    """One optimizer step whose result depends on the dropout RNG and the moments."""
    rank.optimizer.zero_grad()
    out = rank.model[0](batch)
    out.pow(2).mean().backward()
    rank.optimizer.step()
    rank.opt_param_scheduler.step(GBS)
    random.random()
    np.random.rand()


def params(rank):
    return {n: p.detach().clone() for n, p in rank.model[0].named_parameters() if "lora" in n}


class LazyStateDistOptBackend(TorchCutBackend):
    """Reproduces the fork-M5 + Megatron DistOpt behaviour seen on GPU (C1 diagnostic 2):

    * reading state (``_get_main_param_and_optimizer_states``) indexes a defaultdict ->
      creates empty per-param entries on a fresh optimizer;
    * the loader initializes Adam state (dummy step) only when ``state`` is empty, and
    * its setter copies only keys the destination already has (extra keys silently dropped).
    ``side_effect_free`` wraps reads the way ``MilesCutBackend.export_optimizer`` does.
    """

    def __init__(self, coord=None, *, side_effect_free=True):
        super().__init__(coord)
        self.side_effect_free = side_effect_free

    def export_optimizer(self, optimizer, named):
        from collections import defaultdict

        from yeto.rl.engine.miles_adapter.cut_plugin import side_effect_free_state

        if not isinstance(optimizer.state, defaultdict):
            optimizer.state = defaultdict(dict, optimizer.state)
        by_id = {id(p): n for n, p in named}

        def read():
            entries = {}
            for group in optimizer.param_groups:
                hyper = {k: v for k, v in group.items() if k != "params"}
                for p in group["params"]:
                    state = optimizer.state[p]  # defaultdict: creates an empty entry
                    entries[by_id[id(p)]] = {"param": p.detach().clone(),
                                             "state": {k: (v.clone() if torch.is_tensor(v) else v)
                                                       for k, v in state.items()},
                                             "hyper": dict(hyper)}
            return {"format": "torch-test", "entries": entries}

        if self.side_effect_free:
            with side_effect_free_state(optimizer):
                return read()
        return read()

    def load_optimizer(self, optimizer, named, merged):
        by_name = dict(named)
        if not optimizer.state:  # M5: dummy init only when the whole state is empty
            for n, p in named:
                optimizer.state[p] = {"step": torch.tensor(0.0), "exp_avg": torch.zeros_like(p),
                                      "exp_avg_sq": torch.zeros_like(p)}
        for name, entry in merged.items():
            p = by_name[name]
            p.data.copy_(entry["param"])
            dst = optimizer.state[p]
            for key, value in entry["state"].items():  # Megatron setter: existing keys only
                if key in dst:
                    dst[key] = value.clone() if torch.is_tensor(value) else value
