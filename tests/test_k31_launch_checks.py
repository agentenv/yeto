"""Launch-time checks of the Kimi-K3.1 full form (yeto.launch.k31_checks)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yeto.launch import k31_checks
from yeto.rl.adapters.miles.models.partial_experts.selection import main as selection_main


def _selection(tmp_path, *, trainable=64, ep=8, layers=92, name="sel.json"):
    out = tmp_path / name
    argv = ["--model", "kimi_k31", "--num-experts", "896", "--num-trainable", str(trainable),
            "--ep-size", str(ep), "--moe-layer-offset", "1", "--mode", "uniform",
            "--num-moe-layers", str(layers), "--output", str(out)]
    if layers != 92:
        argv.append("--skip-layer-check")
    assert selection_main(argv) == 0
    return out


def _args(selection, **kw):
    base = dict(rl_expert_selection=str(selection), expert_parallel=8, tensor_parallel=8,
                pipeline_parallel=6, context_parallel=1, seq_len=16384,
                rollout_max_response_len=12288, model="moonshotai/Kimi-K3.1-Pro",
                rl_model_recipe="kimi_k31", tuning="full")
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_64_expert_selection_passes_and_reports_its_count(tmp_path):
    summary = k31_checks.check_expert_selection(_args(_selection(tmp_path)))
    assert summary["num_trainable"] == 64 and summary["moe_layers"] == 92
    assert summary["warnings"] == []


def test_a_forged_or_mismatched_selection_is_refused(tmp_path):
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps({"model": "kimi_k31", "experts": {}}))
    with pytest.raises(ValueError, match="not a valid selection"):
        k31_checks.check_expert_selection(_args(forged))
    tampered = json.loads(_selection(tmp_path).read_text())
    tampered["layers"]["1"]["trainable"][0], tampered["layers"]["1"]["trainable"][1] = (
        tampered["layers"]["1"]["trainable"][1], tampered["layers"]["1"]["trainable"][0])
    forged.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="not a valid selection"):
        k31_checks.check_expert_selection(_args(forged))
    with pytest.raises(ValueError, match="no such file"):
        k31_checks.check_expert_selection(_args(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="--expert-parallel 4"):
        k31_checks.check_expert_selection(_args(_selection(tmp_path), expert_parallel=4))


def test_selection_layers_are_checked_against_a_local_checkpoint(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"text_config": {
        "num_hidden_layers": 4, "first_k_dense_replace": 1, "num_experts": 896}}))
    cut = _selection(tmp_path, layers=3, name="cut.json")
    assert k31_checks.check_expert_selection(_args(cut, model=str(model)))["moe_layers"] == 3
    with pytest.raises(ValueError, match="does not match the checkpoint"):
        k31_checks.check_expert_selection(_args(_selection(tmp_path), model=str(model)))


def test_memory_estimate_warns_on_262k_and_on_overfull_gpus(tmp_path):
    args = _args(_selection(tmp_path), seq_len=262144, rollout_max_response_len=8192)
    short = k31_checks.memory_estimate(_args(_selection(tmp_path)), num_trainable=64,
                                       moe_layers=92, gpu="B300")
    assert short["warnings"] == [] and short["per_gpu_gib"] < short["budget_gib"]["no_expandable_segments"]
    fits = k31_checks.memory_estimate(args, num_trainable=64, moe_layers=92, gpu="B300")
    assert any("--rollout-max-response-len is 8192" in w for w in fits["warnings"])
    assert any("--context-parallel 1" in w for w in fits["warnings"])
    full = k31_checks.memory_estimate(args, num_trainable=128, moe_layers=92, gpu="B300")
    assert full["per_gpu_gib"] > fits["per_gpu_gib"]
    # PP6 TP8 CP1 at 262k: 32768 tokens/GPU, six microbatches of checkpoints on stage 0
    assert full["per_gpu_gib"] > full["budget_gib"]["expandable_segments"]
    assert any("even with expandable segments" in w for w in full["warnings"])


@pytest.mark.parametrize("point", k31_checks.MEMORY_CALIBRATION, ids=lambda p: p[0])
def test_stage_model_matches_every_measured_peak(point):
    """Allocated peaks of trainer-1m d1/a3/c1, mfu-16l a0b and the historical 524k PP3 run
    (the old estimate gave 52.1 GiB for a0b, measured 98.9)."""
    _name, layers, (a, b), in_flight, tokens, tp, ep, trainable, measured = point
    got = k31_checks.stage_memory(a, b, num_layers=layers, in_flight=in_flight, tokens_per_gpu=tokens,
                                  tp=tp, ep=ep, num_trainable=trainable)["peak_alloc"]
    assert abs(got - measured) / measured < 0.06


def _formal(**kw):
    base = dict(tensor_parallel=8, pipeline_parallel=3, expert_parallel=16, context_parallel=2,
                seq_len=262144, rollout_max_response_len=262144, rl_optimizer_offload_fraction=1.0,
                modal_env=["YETO_MILES_TRAIN_EXTRA_ARGV=--log-probs-chunk-size 16384 "
                           "--decoder-first-pipeline-num-layers 32 --decoder-last-pipeline-num-layers 32"])
    base.update(kw)
    return SimpleNamespace(**base)


def test_formal_262k_layout_uses_its_split_and_warns_on_the_thin_margin():
    est = k31_checks.memory_estimate(_formal(), num_trainable=128, moe_layers=92, gpu="B300")
    assert est["pipeline_split"] == [32, 29, 32] and est["tokens_per_gpu"] == 16384
    assert [s["in_flight"] for s in est["stages_gib"]] == [3, 2, 1]
    assert est["busiest_stage"] == 0 and 150 < est["per_gpu_gib"] < 170  # layout-mem: 160 central
    assert est["budget_gib"] == {"no_expandable_segments": 166.3, "expandable_segments": 213.4}
    assert any("watch nvidia-smi" in w and "without expandable segments" in w for w in est["warnings"])
    es = k31_checks.memory_estimate(
        _formal(modal_env=_formal().modal_env + ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]),
        num_trainable=128, moe_layers=92, gpu="B300")
    assert es["expandable_segments"] and es["warnings"] == []
    even = k31_checks.memory_estimate(_formal(modal_env=[]), num_trainable=128, moe_layers=92, gpu="B300")
    assert even["pipeline_split"] == [31, 31, 31]


def test_mfu16_two_node_layouts():
    """a0b: L16 TP8 x CP2 x EP16, 131072 tokens per CP rank, measured 89.8-98.9 GiB allocated."""
    a0b = k31_checks.memory_estimate(_formal(pipeline_parallel=1, modal_env=[], rl_max_tokens_per_gpu=131072),
                                     num_trainable=128, moe_layers=15, gpu="B300")
    assert 89.8 * 0.95 < a0b["per_gpu_gib"] < 98.9 * 1.05 and a0b["warnings"] == []
    s2 = k31_checks.memory_estimate(_formal(pipeline_parallel=1, modal_env=[], rl_max_tokens_per_gpu=131072),
                                    num_trainable=128, moe_layers=32, gpu="B300")
    assert 135 < s2["per_gpu_gib"] < 160  # s2-prep: about 170 GiB reserved


@pytest.mark.parametrize("layers, pp, first, last, split", [
    (93, 3, None, None, [31, 31, 31]), (93, 3, 32, 32, [32, 29, 32]), (93, 3, 33, None, [33, 30, 30]),
    (93, 6, 14, 15, [14, 16, 16, 16, 16, 15]), (16, 2, 7, None, [7, 9]), (16, 1, None, None, [16])])
def test_pipeline_split(layers, pp, first, last, split):
    assert k31_checks.pipeline_split(layers, pp, first, last) == split
    assert sum(split) == layers


def test_checkpoint_rows_count_the_attention_residual_bank():
    assert k31_checks.checkpoint_rows(0, 16) == 16 + 1 + 2   # bank generations of layers 0 and 12
    # stage from layer 32: the received bank (3 rows) + generations of 36, 48, 60, the packed entry 1 + 3
    assert k31_checks.checkpoint_rows(32, 61) == 29 + (3 + 4 + 5 + 6) + (1 + 3)


def test_a_cut_sizes_the_modal_disk_per_node_and_refuses_past_modals_limit(tmp_path, monkeypatch):
    from yeto.launch import modal_islands

    monkeypatch.setattr(k31_checks, "_model_config", lambda model: None)
    sel = _selection(tmp_path)
    args = _args(sel, rl_cut_every=2, modal_ephemeral_disk_gib=None)
    gib = modal_islands.modal_ephemeral_disk_gib(args, num_nodes=2)
    assert 1.3 * 1.5e12 / 2 ** 30 < gib <= modal_islands.MODAL_EPHEMERAL_DISK_MAX_GIB  # ~1.57 TB of shards a node
    with pytest.raises(ValueError, match="Modal's 3072 GiB"):
        modal_islands.modal_ephemeral_disk_gib(args, num_nodes=1)
    assert modal_islands.modal_ephemeral_disk_gib(_args(sel, rl_cut_every=2, modal_ephemeral_disk_gib=900),
                                                  num_nodes=1) == 900  # explicit wins
    with pytest.raises(ValueError, match="limit"):
        modal_islands.modal_ephemeral_disk_gib(_args(sel, modal_ephemeral_disk_gib=4000))


def test_full_form_record_reports_the_load_grace_and_warns_when_short(tmp_path, capsys):
    from types import SimpleNamespace as NS

    sel = _selection(tmp_path)
    spec = [NS(gpu="B300")]
    rec = k31_checks.check_full_form(_args(sel, load_grace_minutes=None), spec)
    assert rec["cost_guard"] == {"load_grace_minutes": 90.0, "warnings": []}
    rec = k31_checks.check_full_form(_args(sel, load_grace_minutes=45.0), spec)
    assert rec["cost_guard"]["warnings"] and "--load-grace-minutes 45" in capsys.readouterr().err
