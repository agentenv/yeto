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
    fits = k31_checks.memory_estimate(args, num_trainable=64, moe_layers=92, gpu="B300")
    assert fits["per_gpu_gib"] < fits["capacity_gib"] * k31_checks.WARN_FRACTION
    assert any("--rollout-max-response-len is 8192" in w for w in fits["warnings"])
    assert any("--context-parallel 1" in w for w in fits["warnings"])
    full = k31_checks.memory_estimate(args, num_trainable=128, moe_layers=92, gpu="B300")
    assert full["per_gpu_gib"] > fits["per_gpu_gib"]
    assert any("GiB per GPU" in w for w in full["warnings"])
    short = k31_checks.memory_estimate(_args(_selection(tmp_path)), num_trainable=64,
                                       moe_layers=92, gpu="B300")
    assert short["warnings"] == []


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
