"""The trainer tp*cp node rule on multi-node islands (K3.1 TP8 x CP2 on 8-GPU nodes).

* the launcher (and so ``yeto launch --dry-run``) applies it to a colocated island
  too: before, only a fixed partition was checked and the learner refused the
  layout after the paid island booted (mfu16 PREFLIGHT, 2026-11-18);
* refusals and the opt-in warning say what actually spans nodes: with Megatron's
  tp-cp-ep-dp-pp order a TP8 group stays inside an 8-GPU node, only CP spans."""

from __future__ import annotations

import sys
import types

import pytest

from yeto.rl.adapters.miles.placement import PlacementRequest
from yeto.rl.engine.multinode import tp_cp_node_note


def test_note_says_whether_tp_or_only_cp_spans():
    assert "every TP group (tp=8) stays inside a node; only the CP groups (cp=2)" in tp_cp_node_note(8, 2, 8)
    assert "TP groups themselves (tp=16) span nodes" in tp_cp_node_note(16, 1, 8)
    assert "TP groups themselves (tp=3)" in tp_cp_node_note(3, 2, 8)


def _k31_formal(**kw):
    """Customer layout b: 6 x 8 B300, PP3 x TP8 x CP2 x EP16, trainer only."""
    return PlacementRequest("fixed-partition", trainer_gpus=48, rollout_gpus=0, gpus_per_engine=1,
                            gpus_per_node=8, model_parallel=8 * 2 * 3, node_parallel=16,
                            expert_parallel=16, **kw)


def test_k31_formal_layout_needs_the_opt_in():
    with pytest.raises(ValueError, match="--rl-allow-cross-node-tp lifts it"):
        _k31_formal()
    assert _k31_formal(allow_cross_node_tp=True).trainer_shape() == (6, 8)


def test_learner_refusal_names_cp_not_tp():
    from types import SimpleNamespace as NS

    from yeto.rl.adapters.miles.config import placement_request

    par = NS(actor_num_nodes=2, actor_num_gpus_per_node=8, colocated=True, rollout_num_gpus_per_engine=8,
             island_gpus_per_node=8, tensor_parallel=8, pipeline_parallel=1, context_parallel=2,
             expert_parallel=16, allow_cross_node_tp=False, allow_cross_node_engine_tp=False)
    with pytest.raises(ValueError, match=r"spans nodes .*; every TP group \(tp=8\) stays inside a node; "
                                         r"only the CP groups \(cp=2\) span nodes"):
        placement_request(NS(parallel=par))
    par.allow_cross_node_tp = True
    assert placement_request(NS(parallel=par)).trainer_shape() == (2, 8)


def _colocated_args(*extra):
    from test_rl_launcher import _args

    return _args(("--gpu", "nebius:2x4xl40s@eu-north1", "--rl-engine", "ports",
                  "--rollout-num-gpus-per-engine", "4", "--tensor-parallel", "4", "--context-parallel", "2",
                  "--rollout-batch-size", "4", "--n-samples-per-prompt", "8", *extra))


def test_launcher_refuses_a_colocated_tp_cp_group_wider_than_a_node(monkeypatch):
    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task, _prepare_rl_args

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    with pytest.raises(ValueError, match=r"TP\*CP \(4x2\) must fit and divide one node's 4 trainer GPUs.*"
                                         r"every TP group \(tp=4\) stays inside a node"):
        _prepare_rl_args(_colocated_args())


def test_launcher_warns_on_the_colocated_opt_in(monkeypatch, capsys):
    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task, _prepare_rl_args

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    _prepare_rl_args(_colocated_args("--rl-allow-cross-node-tp"))
    out = capsys.readouterr().err  # stdout of a dry run is its JSON plan
    assert "WARNING: --rl-allow-cross-node-tp: the trainer TP*CP group (8) spans nodes" in out
    assert "only the CP groups (cp=2) span nodes" in out


def test_launcher_in_node_tp_cp_needs_no_opt_in(monkeypatch, capsys):
    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task, _args, _prepare_rl_args

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    _prepare_rl_args(_args(("--gpu", "nebius:2x4xl40s@eu-north1", "--rl-engine", "ports",
                            "--rollout-num-gpus-per-engine", "4", "--tensor-parallel", "2",
                            "--context-parallel", "2", "--rollout-batch-size", "4",
                            "--n-samples-per-prompt", "8")))
    assert "--rl-allow-cross-node-tp" not in capsys.readouterr().err
