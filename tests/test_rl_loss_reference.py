"""decoupling 4.7: neutral loss reference vs the paper formulas and vs the
Miles fork functions at the pinned commit, bitwise on CPU.

The fork comparison does not import Miles (it is blocked on this host and the
module pulls torch.distributed / cp_utils): it takes the function *source*
from a local Miles checkout at ``MILES_NEXT_COMMIT`` (``git show``), drops
the ``@torch.compile`` decorators and executes the definitions with torch
only. Without a checkout (``YETO_MILES_REPO``, default
``/home/michael/work/miles-next``) those tests are skipped, not failed.
"""

from __future__ import annotations

import ast
import os
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import rl_loss_variant_reference as paper
from yeto.rl import MILES_NEXT_COMMIT
from yeto.rl.algos import loss_reference as ref

MATH = "miles/backends/training_utils/loss_hub/math_utils.py"
CORR = "miles/backends/training_utils/loss_hub/corrections.py"
REPO = Path(os.environ.get("YETO_MILES_REPO", "/home/michael/work/miles-next"))


def _fork(path: str, names: tuple[str, ...]) -> dict:
    if not (REPO / ".git").exists():
        pytest.skip(f"no Miles checkout at {REPO}")
    try:
        source = subprocess.run(["git", "-C", str(REPO), "show", f"{MILES_NEXT_COMMIT}:{path}"],
                                check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError:
        pytest.skip(f"{REPO} lacks {MILES_NEXT_COMMIT}")
    tree = ast.parse(source)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []  # @torch.compile: eager is the reference semantics
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id.startswith("_LOG_RATIO") for t in node.targets):
            keep.append(node)
    namespace = {"torch": torch, "Namespace": Namespace, "Any": object}
    exec(compile(ast.Module(body=keep, type_ignores=[]), f"{MILES_NEXT_COMMIT}:{path}", "exec"),
         namespace)
    return namespace


def _inputs(seed=0, n=257):
    g = torch.Generator().manual_seed(seed)
    logp = -torch.rand(n, generator=g) * 4
    old = logp + torch.randn(n, generator=g) * 0.3
    rollout = old + torch.randn(n, generator=g) * 0.2
    adv = torch.randn(n, generator=g)
    adv[::7] = 0.0
    return logp, old, rollout, adv


def _same(a, b):
    assert a.dtype == b.dtype and torch.equal(a, b)


def test_matches_paper_formulas():
    logp, old, _, adv = _inputs()
    torch.testing.assert_close(ref.cispo(logp, old, adv, 0.2, 0.28)[0],
                               paper.cispo(logp, old, adv, 0.2, 0.28))
    torch.testing.assert_close(ref.sapo(logp, old, adv)[0], paper.sapo(logp, old, adv))
    _, old2, rollout, _ = _inputs(1)
    torch.testing.assert_close(ref.tis_weight(old2, rollout, 0.0, 2.0)[0],
                               paper.tis_weight(old2, rollout, 0.0, 2.0))
    torch.testing.assert_close(ref.icepop_weight(old2, rollout, 0.5, 2.0)[0],
                               paper.icepop_weight(old2, rollout, 0.5, 2.0))


@pytest.mark.parametrize("dual", [None, 3.0])
def test_ppo_clip_is_bitwise_the_fork(dual):
    fork = _fork(MATH, ("_safe_clamp_log_ratio", "_safe_exp_neg_ppo_kl", "compute_policy_loss"))
    for seed in range(3):
        logp, old, _, adv = _inputs(seed)
        got = ref.ppo_clip(logp, old, adv, 0.2, 0.28, dual_clip_c=dual)
        want = fork["compute_policy_loss"](old - logp, adv, 0.2, 0.28, dual)
        _same(got[0], want[0])
        _same(got[1], want[1])


def test_cispo_sapo_are_bitwise_the_fork():
    fork = _fork(MATH, ("_safe_clamp_log_ratio", "_safe_exp_neg_ppo_kl",
                        "compute_cispo_loss", "compute_sapo_loss"))
    for seed in range(3):
        logp, old, _, adv = _inputs(seed)
        for got, want in (
            (ref.cispo(logp, old, adv, 0.2, 0.28), fork["compute_cispo_loss"](old - logp, logp, adv, 0.2, 0.28)),
            (ref.sapo(logp, old, adv, 1.0, 1.05), fork["compute_sapo_loss"](old - logp, adv, 1.0, 1.05)),
        ):
            _same(got[0], want[0])
            _same(got[1], want[1])


@pytest.mark.parametrize("estimator", ref.KL_ESTIMATORS)
@pytest.mark.parametrize("with_ratio", [False, True])
def test_kl_is_bitwise_the_fork(estimator, with_ratio):
    fork = _fork(MATH, ("_safe_clamp_log_ratio", "compute_approx_kl"))
    logp, old, _, _ = _inputs(2)
    is_ratio = torch.exp(logp - old) if with_ratio else None
    _same(ref.kl(logp, old, estimator, is_ratio),
          fork["compute_approx_kl"](logp, old, estimator, is_ratio))


@pytest.mark.parametrize("bounds", [(0.0, 2.0), (0.5, 1.5)])
def test_tis_icepop_are_bitwise_the_fork(bounds):
    fork = _fork(CORR, ("vanilla_tis_function", "icepop_function"))
    low, high = bounds
    logp, old, rollout, adv = _inputs(3)
    pg = ref.ppo_clip(logp, old, adv, 0.2, 0.2)[0]
    args = Namespace(tis_clip_low=low, tis_clip=high)
    masks = [torch.ones(len(pg))]
    for name, fn in (("vanilla_tis_function", ref.tis_weight), ("icepop_function", ref.icepop_weight)):
        loss, _, metrics = fork[name](args, pg_loss=pg, train_log_probs=[old],
                                      rollout_log_probs=[rollout], loss_masks=masks)
        weight, clipfrac = fn(old, rollout, low, high)
        _same(pg * weight, loss)
        _same(clipfrac, metrics["tis_clipfrac"])
