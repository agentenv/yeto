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
    parts = rec["parts_gib"]
    # the partial-experts estimate counts the parameters exactly (param_breakdown)
    assert parts["weights"] == pytest.approx(2 * (breakdown["total"] - routed / 2) / me.GIB, abs=1e-3)
    # DP 2 / EP 2: an expert data-parallel group of one rank, nothing to shard
    assert parts["grads_optimizer"] == pytest.approx(16 * trainable / 2 / me.GIB, abs=1e-3)
    assert parts["inference"] == 0.0 and rec["fits"], rec
    whole = me.estimate_for_launch(_args(model), H100_ISLAND, margin=0.9)
    assert not whole["fits"] and "ep" not in whole and whole["parts_gib"]["grads_optimizer"] > 400


def _partial_request(model, selection, **kw):
    spec = kw.pop("spec", H100_ISLAND)
    arch = me.arch_from_hf_config(QWEN3_30B_A3B)
    return me.request_from_args(_args(model, rl_expert_selection=str(selection), **kw), spec, arch, QWEN3_30B_A3B)


def test_measured_static_memory_after_init_matches(model, selection):
    """A measured 4 x H100 island (fixed partition, 2 trainer GPUs, EP 2, 8 of 128 experts
    trainable, seq 1024, no recompute): Miles logged ``allocated_GB 36.62`` on both trainer
    ranks before the first weight update, bf16 weights 29.87 + fp32 main grads 3.37 + fp32
    masters 3.37 GiB (the Adam moments are allocated by the first step)."""
    req = _partial_request(model, selection, seq_len=1024, rollout_max_response_len=1024,
                           rl_activation_recompute=False)
    assert (req.tp, req.pp, req.cp, req.dp, req.ep) == (1, 1, 1, 2, 2)
    state = me.model_state_bytes(req)
    assert state["weights"] / me.GIB == pytest.approx(29.87, abs=0.01)
    assert state["grads"] / me.GIB == pytest.approx(3.375, abs=0.01)
    assert state["master"] / me.GIB == pytest.approx(3.375, abs=0.01)
    assert me.static_after_init_gib(req) == pytest.approx(36.62, abs=0.01)


def test_tensor_parallel_splits_the_dense_part_not_the_experts(model, selection):
    """--expert-tensor-parallel-size 1 (Miles adapter): TP divides the non-expert
    parameters only; EP and PP divide the routed experts."""
    breakdown = me.param_breakdown(QWEN3_30B_A3B)
    dense, routed = breakdown["total"] - breakdown["routed_experts"], breakdown["routed_experts"]
    tp1 = _partial_request(model, selection)
    tp2 = me.replace(tp1, tp=2)
    pp2 = me.replace(tp1, pp=2)
    w = {name: me.estimate(r)["parts_gib"]["weights"] for name, r in (("tp1", tp1), ("tp2", tp2), ("pp2", pp2))}
    assert w["tp1"] == pytest.approx(2 * (dense + routed / 2) / me.GIB, abs=1e-3)
    assert w["tp2"] == pytest.approx(2 * (dense / 2 + routed / 2) / me.GIB, abs=1e-3)
    assert w["pp2"] == pytest.approx(2 * (dense / 2 + routed / 4) / me.GIB, abs=1e-3)
    grads = {name: me.model_state_bytes(r)["grads"] for name, r in (("tp1", tp1), ("tp2", tp2), ("pp2", pp2))}
    assert grads["tp2"] == grads["tp1"] == 2 * grads["pp2"]


def test_the_distributed_optimizer_shards_over_the_expert_data_parallel_group(model, selection):
    req = _partial_request(model, selection, rl_placement="colocated")
    t = req.trainable_params / req.ep
    plain = me.estimate(me.replace(req, distributed_optimizer=False))["parts_gib"]["grads_optimizer"]
    sharded = me.estimate(me.replace(req, distributed_optimizer=True))["parts_gib"]["grads_optimizer"]
    assert (req.dp, req.ep) == (4, 2)
    assert plain == pytest.approx(16 * t / me.GIB, abs=1e-3)
    assert sharded == pytest.approx((4 * t + 12 * t / 2) / me.GIB, abs=1e-3)
    # expert data-parallel group = TP x CP x DP / EP (expert tensor parallel 1)
    tp2 = me.replace(req, distributed_optimizer=True, tp=2, dp=2)
    assert me.model_state_bytes(tp2)["master"] == pytest.approx(4 * t / 2)


def test_the_distributed_optimizer_follows_the_miles_adapter(model, selection):
    """Miles turns the distributed optimizer on for Adam; the adapter keeps it on a no-sync
    island and at DP > 1, and turns it off for the full-parameter sync at DP 1."""
    eight = SimpleNamespace(gpu="H100", total_gpus=8)
    no_sync = _partial_request(model, selection, rl_placement="colocated", rl_single_island_no_sync=True,
                               spec=eight, expert_parallel=8)
    assert (no_sync.dp, no_sync.distributed_optimizer) == (8, True)
    synced_dp1 = _partial_request(model, selection, rl_placement="fixed-partition", rollout_num_gpus=3,
                                  expert_parallel=1)
    assert (synced_dp1.dp, synced_dp1.distributed_optimizer) == (1, False)
    replicated = _partial_request(model, selection, rl_placement="colocated", rl_dp_optimizer="replicated")
    assert (replicated.dp, replicated.distributed_optimizer) == (4, False)


def test_context_parallel_is_passed_through(model, selection):
    one = _partial_request(model, selection, rl_placement="colocated", seq_len=32768,
                           rollout_max_response_len=32768)
    two = _partial_request(model, selection, rl_placement="colocated", seq_len=32768,
                           rollout_max_response_len=32768, context_parallel=2)
    assert (one.cp, one.dp, two.cp, two.dp) == (1, 4, 2, 2)
    e1, e2 = me.estimate(one)["parts_gib"], me.estimate(two)["parts_gib"]
    assert e2["activations"] == pytest.approx(e1["activations"] / 2, abs=1e-3)
    assert e2["logits_block"] == pytest.approx(e1["logits_block"] / 2, abs=1e-3)


def test_dynamic_batch_tokens_follow_the_miles_bins(model, selection):
    """--rl-max-tokens-per-gpu: Miles packs rows into bins of max_tokens_per_gpu x CP
    tokens; a longer row gets its own bin."""
    longer_bins = _partial_request(model, selection, seq_len=8192, rollout_max_response_len=8192,
                                   rl_max_tokens_per_gpu=16384)
    shorter_bins = _partial_request(model, selection, seq_len=32768, rollout_max_response_len=32768,
                                    rl_max_tokens_per_gpu=16384)
    with_cp = _partial_request(model, selection, seq_len=8192, rollout_max_response_len=8192,
                               rl_max_tokens_per_gpu=8192, context_parallel=2, rl_placement="colocated")
    without = _partial_request(model, selection, seq_len=8192, rollout_max_response_len=8192)
    assert (longer_bins.tokens, shorter_bins.tokens, with_cp.tokens, without.tokens) == (16384, 32768, 16384, 8192)


def test_a_dense_estimate_does_not_move_with_bins_no_wider_than_a_row(tmp_path, monkeypatch):
    _builtin_configs_only(tmp_path, monkeypatch)
    spec = SimpleNamespace(gpu="H100", total_gpus=2)
    for tuning in ("full", "lora"):
        base = me.estimate_for_launch(_calibrated(tuning=tuning, seq_len=8192), spec, margin=0.9)
        binned = me.estimate_for_launch(_calibrated(tuning=tuning, seq_len=8192, rl_max_tokens_per_gpu=8192), spec,
                                        margin=0.9)
        assert binned == base


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


def _builtin_configs_only(tmp_path, monkeypatch) -> None:
    """No Hugging Face cache anywhere: the builtin Qwen3.5-4B config, whatever the machine has."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "no-hub"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "no-hf-home"))


def test_a_calibrated_fixed_partition_launch_still_passes_the_refusing_preflight(tmp_path, monkeypatch):
    """--preflight-memory error (the default) refuses a calibrated model over the margin; the
    smaller trainer data-parallel size of a fixed partition must not refuse what passed."""
    _builtin_configs_only(tmp_path, monkeypatch)
    assert lp.memory_config(_calibrated()) == (lp.DEFAULT_MEMORY_MODE, lp.DEFAULT_MEMORY_MARGIN) == ("error", 0.9)
    report = lp.memory_preflight(_calibrated(), [SimpleNamespace(gpu="H200", total_gpus=2)], out=io.StringIO())
    (rec,) = report["islands"]
    assert rec["arch_source"] == "builtin" and rec["calibrated"] and rec["fits"], rec
    assert (rec["dp"], rec["colocated"]) == (1, False)


@pytest.mark.parametrize("gpu", ["H100", "H200"])
@pytest.mark.parametrize("tuning", ["full", "lora"])
@pytest.mark.parametrize("seq", [2048, 4096, 8192, 16384])
def test_a_fixed_partition_never_estimates_above_the_old_colocated_estimate(tmp_path, monkeypatch, gpu, tuning,
                                                                            seq):
    _builtin_configs_only(tmp_path, monkeypatch)
    spec = SimpleNamespace(gpu=gpu, total_gpus=2)
    new = me.estimate_for_launch(_calibrated(tuning=tuning, seq_len=seq), spec, margin=0.9)
    old = me.estimate_for_launch(_calibrated(tuning=tuning, seq_len=seq, rl_placement="colocated"), spec, margin=0.9)
    assert new["peak_gib"] <= old["peak_gib"] and (new["fits"] or not old["fits"])
