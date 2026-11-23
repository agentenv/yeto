"""Launch memory estimate of a fixed partition and of a generic partial-experts run.

A Qwen3-30B-A3B-shaped island (4 GPUs, 2 rollout, EP 2, 8 of 128 experts per layer
trainable) was estimated as a colocated 4-way data-parallel full fine-tune of every
parameter (grads + optimizer 460 GiB per GPU, "fits": false). The trainer has two
GPUs, the routed experts are split over EP ranks and only the selected experts train.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

import yeto.launch_preflight as lp
import yeto.memory_estimate as me
from yeto.rl.adapters.miles.models.partial_experts.selection import ExpertSelection

QWEN3_30B_A3B = {"model_type": "qwen3_moe", "num_hidden_layers": 48, "hidden_size": 2048,
                 "intermediate_size": 6144, "moe_intermediate_size": 768, "num_experts": 128,
                 "num_experts_per_tok": 8, "decoder_sparse_step": 1, "mlp_only_layers": [],
                 "vocab_size": 151936, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
                 "tie_word_embeddings": False}
H100_ISLAND = SimpleNamespace(gpu="H100", total_gpus=4)


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "qwen3-30b-a3b"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(QWEN3_30B_A3B))
    return path


@pytest.fixture
def selection(tmp_path):
    path = tmp_path / "selection.json"
    ExpertSelection.in_place("uniform", model="qwen3_moe", num_experts=128, num_trainable=8, ep_size=2,
                             layers=range(48)).write(path)
    return path


def _args(model, **kw):
    base = dict(training_mode="rl", model=str(model), tuning="full", seq_len=4096, rollout_max_response_len=4096,
                tensor_parallel=1, pipeline_parallel=1, rl_placement="fixed-partition", rollout_num_gpus=2,
                rl_standby_gpus=0, rl_sample_role=None, rl_buffer_shards=None, expert_parallel=2,
                rl_expert_selection=None, sglang_mem_fraction_static=0.7, rl_activation_recompute=True)
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_fixed_partition_trains_on_the_gpus_left_after_rollout_and_standby(model):
    arch = me.arch_from_hf_config(QWEN3_30B_A3B)
    req = me.request_from_args(_args(model, rollout_num_gpus=1, rl_standby_gpus=1), H100_ISLAND, arch)
    assert (req.dp, req.colocated) == (2, False)
    assert me.estimate(req)["rollout_phase_gib"] == 0.0
    colocated = me.request_from_args(_args(model, rl_placement="colocated"), H100_ISLAND, arch)
    assert (colocated.dp, colocated.colocated, colocated.ep, colocated.trainable_params) == (4, True, 1, None)


def test_only_the_selected_experts_carry_grads_and_optimizer_state(model, selection):
    rec = me.estimate_for_launch(_args(model, rl_expert_selection=str(selection)), H100_ISLAND, margin=0.9)
    breakdown = me.param_breakdown(QWEN3_30B_A3B)
    per_expert, routed = breakdown["per_expert"], breakdown["routed_experts"]
    trainable = 48 * 8 * per_expert
    assert (rec["ep"], rec["trainable_params"], rec["dp"], rec["colocated"]) == (2, trainable, 2, False)
    arch = me.arch_from_hf_config(QWEN3_30B_A3B)
    parts = rec["parts_gib"]
    assert parts["weights"] == pytest.approx(2 * (arch.params - routed / 2) / me.GIB, abs=1e-3)
    assert parts["grads_optimizer"] == pytest.approx(16 * trainable / 2 / me.GIB, abs=1e-3)
    assert parts["inference"] == 0.0 and rec["fits"], rec
    whole = me.estimate_for_launch(_args(model), H100_ISLAND, margin=0.9)
    assert not whole["fits"] and "ep" not in whole and whole["parts_gib"]["grads_optimizer"] > 400


def test_the_distributed_optimizer_shards_over_the_expert_data_parallel_group(model, selection):
    arch = me.arch_from_hf_config(QWEN3_30B_A3B)
    args = _args(model, rl_expert_selection=str(selection), rl_placement="colocated")
    req = me.request_from_args(args, H100_ISLAND, arch, QWEN3_30B_A3B)
    t = req.trainable_params / req.ep
    plain = me.estimate(req)["parts_gib"]["grads_optimizer"]
    sharded = me.estimate(me.replace(req, distributed_optimizer=True))["parts_gib"]["grads_optimizer"]
    assert (req.dp, req.ep) == (4, 2)
    assert plain == pytest.approx(16 * t / me.GIB, abs=1e-3)
    assert sharded == pytest.approx((4 * t + 12 * t / 2) / me.GIB, abs=1e-3)


def test_other_launches_keep_their_estimate(model, tmp_path):
    slot_order = tmp_path / "k31.json"
    ExpertSelection.synthetic("uniform", model="kimi_k31", num_experts=128, num_trainable=8, ep_size=2,
                              layers=range(48)).write(slot_order)
    assert me.partial_experts_shape(_args(model, rl_expert_selection=str(slot_order)), QWEN3_30B_A3B) is None
    assert me.partial_experts_shape(_args(model, tuning="lora", rl_expert_selection=str(slot_order)),
                                    QWEN3_30B_A3B) is None
    assert me.partial_experts_shape(_args(model, rl_expert_selection=str(tmp_path / "missing.json")),
                                    QWEN3_30B_A3B) is None
    arch = me.arch_from_hf_config(QWEN3_30B_A3B)
    req = me.request_from_args(_args(model), H100_ISLAND, arch)
    legacy = me.MemoryRequest(arch=arch, gpu="H100", gpu_mem_gib=req.gpu_mem_gib, tokens=4096, tuning="full",
                              dp=2, recompute=True, colocated=False, offload=True, mem_fraction_static=0.7)
    assert me.estimate(req) == me.estimate(legacy)


def _calibrated(**kw):
    base = dict(training_mode="rl", model="Qwen/Qwen3.5-4B", tuning="full", seq_len=4096,
                rollout_max_response_len=None, tensor_parallel=1, pipeline_parallel=1, rl_placement="fixed-partition",
                rollout_num_gpus=1, rl_standby_gpus=0, rl_sample_role=None, rl_buffer_shards=None,
                sglang_mem_fraction_static=0.4, preflight_memory=None, preflight_memory_margin=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_calibrated_fixed_partition_launch_still_passes_the_refusing_preflight(monkeypatch):
    """--preflight-memory error (the default) refuses a calibrated model over the margin; the
    smaller trainer data-parallel size of a fixed partition must not refuse what passed."""
    monkeypatch.setenv("HF_HUB_CACHE", "/nonexistent")  # builtin Qwen3.5-4B config
    assert lp.memory_config(_calibrated()) == (lp.DEFAULT_MEMORY_MODE, lp.DEFAULT_MEMORY_MARGIN) == ("error", 0.9)
    report = lp.memory_preflight(_calibrated(), [SimpleNamespace(gpu="H200", total_gpus=2)], out=io.StringIO())
    (rec,) = report["islands"]
    assert rec["calibrated"] and rec["fits"] and (rec["dp"], rec["colocated"]) == (1, False), rec


@pytest.mark.parametrize("gpu", ["H100", "H200"])
@pytest.mark.parametrize("tuning", ["full", "lora"])
@pytest.mark.parametrize("seq", [2048, 4096, 8192, 16384])
def test_a_fixed_partition_never_estimates_above_the_old_colocated_estimate(monkeypatch, gpu, tuning, seq):
    monkeypatch.setenv("HF_HUB_CACHE", "/nonexistent")
    spec = SimpleNamespace(gpu=gpu, total_gpus=2)
    new = me.estimate_for_launch(_calibrated(tuning=tuning, seq_len=seq), spec, margin=0.9)
    old = me.estimate_for_launch(_calibrated(tuning=tuning, seq_len=seq, rl_placement="colocated"), spec, margin=0.9)
    assert new["peak_gib"] <= old["peak_gib"] and (new["fits"] or not old["fits"])
