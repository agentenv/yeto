"""S25 N10.3b: the ``yeto_grpo`` estimator keeps ABORTED placeholders out of GRPO groups (CPU).

verl is not installed on the test machines, so ``verl.trainer.ppo.core_algos``
is a stub whose ``compute_grpo_outcome_advantage`` is verl's own function,
compiled from the pinned fork's source (``verl-fork-acad9875``; skip if the
checkout is absent).
"""

from __future__ import annotations

import ast
import importlib
import sys
import types
from collections import defaultdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from yeto.rl.agentic.groups import ISOLATED, isolate_rows  # noqa: E402

VERL_CORE_ALGOS = Path("/home/michael/work/verl-fork-acad9875/verl/trainer/ppo/core_algos.py")


def _verl_grpo():
    if not VERL_CORE_ALGOS.exists():
        pytest.skip(f"verl fork checkout not found: {VERL_CORE_ALGOS}")
    tree = ast.parse(VERL_CORE_ALGOS.read_text())
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_grpo_outcome_advantage"]
    fn.decorator_list = []
    namespace = {"torch": torch, "np": np, "defaultdict": defaultdict, "Optional": __import__("typing").Optional,
                 "AlgoConfig": object}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(VERL_CORE_ALGOS), "exec"), namespace)
    return namespace["compute_grpo_outcome_advantage"]


@pytest.fixture
def advantage(monkeypatch):
    registry = {}
    core = types.ModuleType("verl.trainer.ppo.core_algos")
    core.compute_grpo_outcome_advantage = _verl_grpo()

    def register_adv_est(name):
        def deco(fn):
            registry[name] = fn
            return fn
        return deco

    core.register_adv_est, core.registry = register_adv_est, registry
    for name in ("verl", "verl.trainer", "verl.trainer.ppo"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["verl.trainer.ppo"].core_algos = core
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.core_algos", core)
    monkeypatch.delitem(sys.modules, "yeto.rl.adapters.verl.advantage", raising=False)
    module = importlib.import_module("yeto.rl.adapters.verl.advantage")
    yield module, core
    sys.modules.pop("yeto.rl.adapters.verl.advantage", None)


def test_isolate_rows_moves_only_the_excluded_rows():
    assert isolate_rows(["a", "a", "b"], [False, True, False]) == ["a", (ISOLATED, 1), "b"]
    with pytest.raises(ValueError):
        isolate_rows(["a"], [])


def test_aborted_placeholder_does_not_move_its_siblings_baseline(advantage):
    module, core = advantage
    assert core.registry["yeto_grpo"] is module.compute_yeto_grpo_advantage
    # group "p": rewards 1, 0, 1 plus an ABORTED placeholder (reward 0, mask all 0); group "q": 1, 0
    rewards = torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[1, 1], [1, 1], [1, 0], [0, 0], [1, 1], [1, 1]], dtype=torch.float32)
    index = np.array(["p", "p", "p", "p", "q", "q"], dtype=object)
    config = {"norm_adv_by_std_in_grpo": True}

    ours, _ = module.compute_yeto_grpo_advantage(rewards, mask, index, config=config)
    # reference: verl GRPO on the batch without the placeholder row
    keep = [0, 1, 2, 4, 5]
    ref, _ = core.compute_grpo_outcome_advantage(rewards[keep], mask[keep], index[keep])
    assert torch.allclose(ours[keep], ref)
    assert torch.equal(ours[3], torch.zeros(2))  # the placeholder trains nothing
    # plain verl GRPO would count the placeholder: group "p" mean 0.5 instead of 2/3
    plain, _ = core.compute_grpo_outcome_advantage(rewards, mask, index)
    assert not torch.allclose(plain[keep[:3]], ref[:3])


def test_std_normalization_follows_the_config(advantage):
    module, core = advantage
    rewards = torch.tensor([[1.0], [0.0], [0.0]])
    mask = torch.ones(3, 1)
    index = np.array(["p", "p", "p"], dtype=object)
    ours, _ = module.compute_yeto_grpo_advantage(rewards, mask, index, config={"norm_adv_by_std_in_grpo": False})
    assert torch.allclose(ours[:, 0], torch.tensor([2 / 3, -1 / 3, -1 / 3]))
