"""Launch memory estimate of sampler islands, and the records it must leave alone.

A sampler island (``--rl-sample-role sampler``, or a ``sampler`` island of a
store-coupled pair or synced group) runs SGLang engines only; it was estimated
as a trainer. It is now estimated as an engine: weights per engine TP plus the
KV cache of one longest context inside ``--sglang-mem-fraction-static`` of the
card, peak = that static share + the fixed overhead. Advisory only: an over-limit
sampler island warns and never stops a launch.

Every other island keeps its record byte for byte (the dry-run ``memory_estimate``,
the preflight and the manifest): ``tests/data/memory_estimate_records.json`` was
written by the code before sampler islands were estimated (regenerate it only for
an intended trainer-side change: ``YETO_UPDATE_GOLDEN=1``).
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

import yeto.memory_estimate as me

DATA = Path(__file__).parent / "data"
GOLDEN = DATA / "memory_estimate_records.json"
CONFIGS = {
    "qwen3_8b": {"model_type": "qwen3", "max_position_embeddings": 40960, **me.BUILTIN_CONFIGS["Qwen/Qwen3-8B"]},
    "qwen3_0p6b": {"model_type": "qwen3", "max_position_embeddings": 40960,
                   **me.BUILTIN_CONFIGS["Qwen/Qwen3-0.6B"]},
    "kimi_k31": json.loads((DATA / "model_registry" / "kimi_k31.json").read_text()),
}
COMMON = ("--data", "org/data", "--training-mode", "rl", "--total-steps", "4", "--rollout-batch-size", "8",
          "--n-samples-per-prompt", "2", "--seq-len", "4096", "--rollout-max-response-len", "2048",
          "--reward-function", "pkg.reward:score")
# (name, model, --gpu, flags, the islands whose record must not change)
LAUNCHES = (
    ("k31_buffer_trainer", "kimi_k31", "modal:2x8xh100",
     ("--rl-sample-role", "trainer", "--rl-single-island-no-sync", "--rl-model-recipe", "kimi_k31",
      "--tuning", "full", "--expert-parallel", "8"), (0,)),
    ("no_sync_lora", "qwen3_8b", "modal:1x8xh100", ("--rl-single-island-no-sync", "--tuning", "lora"), (0,)),
    ("strict_avg_two_islands", "qwen3_8b", "modal:1x8xh100,modal:1x8xh100",
     ("--tuning", "lora", "--rl-sync-preset", "strict-avg"), (0, 1)),
    ("full_parameter_fixed_partition", "qwen3_0p6b", "modal:1x8xh100,modal:1x8xh100",
     ("--tuning", "full", "--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
      "--rollout-num-gpus", "4"), (0, 1)),
    ("pair_trainer", "qwen3_8b", "modal:1x8xh100,modal:1x8xh100",
     ("--tuning", "lora", "--rl-single-island-no-sync", "--rl-island-sample-role", "0:sampler",
      "--rl-island-sample-role", "1:trainer"), (1,)),
    ("synced_group_trainers", "qwen3_8b", "modal:1x8xh100,modal:1x8xh100,modal:1x8xh100",
     ("--tuning", "lora", "--rl-sample-role", "trainer", "--rl-island-sample-role", "2:sampler"), (0, 1)),
)


@pytest.fixture
def models(tmp_path):
    out = {}
    for name, config in CONFIGS.items():
        folder = tmp_path / name
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(config))
        out[name] = str(folder)
    return out


def _records(models, model: str, gpu: str, flags: tuple[str, ...]) -> list[dict]:
    from yeto.cli import parse_args
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch_preflight import dry_run_memory

    args = parse_args(["--gpu", gpu, "--model", models[model], *COMMON, *flags])
    records = dry_run_memory(args, parse_gpu_spec(args.gpu))
    text = json.dumps(records, sort_keys=True).replace(models[model], f"<{model}>")
    return json.loads(text)


def test_trainer_and_no_sync_records_are_unchanged(models, capsys):
    got = {}
    for name, model, gpu, flags, islands in LAUNCHES:
        records = _records(models, model, gpu, flags)
        got[name] = {str(m): records[m] for m in islands}
    if os.environ.get("YETO_UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
    want = json.loads(GOLDEN.read_text())
    assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True)


def _launch(models, model, gpu, *flags):
    from yeto.cli import parse_args

    return parse_args(["--gpu", gpu, "--model", models[model], *COMMON, *flags])


def test_a_pairs_sampler_island_is_estimated_as_its_engine(models):
    records = _records(models, "qwen3_8b", "modal:1x8xh100,modal:1x8xh100",
                       ("--tuning", "lora", "--rl-single-island-no-sync", "--rl-island-sample-role", "0:sampler",
                        "--rl-island-sample-role", "1:trainer", "--sglang-mem-fraction-static", "0.88"))
    sampler = records[0]
    arch = me.arch_from_hf_config(CONFIGS["qwen3_8b"])
    cap = me.gpu_capacity_gib("H100")
    assert (sampler["role"], sampler["calibrated"], sampler["engine_tp"], sampler["tokens"]) == (
        "sampler", False, 1, 4096)
    parts = sampler["parts_gib"]
    assert parts["weights"] == pytest.approx(2 * arch.params / me.GIB, abs=1e-3)
    kv = 2 * 2 * 36 * 8 * 128 * 4096 / me.GIB  # K and V, 36 layers, 8 KV heads x 128, bf16
    assert parts["kv_cache_one_context"] == pytest.approx(kv, abs=1e-3)
    assert sampler["peak_gib"] == pytest.approx(0.88 * cap + me.R0_GIB, abs=1e-3)
    assert sum(parts.values()) == pytest.approx(sampler["peak_gib"], abs=1e-2)
    assert sampler["engine_fits"] and not sampler["fits"]  # 0.88 x 79.18 + R0 > 90 % of the card
    assert "role" not in records[1] and records[1]["colocated"]  # the trainer island as before


def test_sampler_roles_come_from_the_launch(models):
    group = _records(models, "qwen3_8b", "modal:1x8xh100,modal:1x8xh100,modal:1x8xh100",
                     ("--tuning", "lora", "--rl-sample-role", "trainer", "--rl-island-sample-role", "2:sampler"))
    assert [r.get("role") for r in group] == [None, None, "sampler"]
    alone = _records(models, "qwen3_8b", "modal:1x8xh100", ("--rl-sample-role", "sampler",
                                                           "--rl-single-island-no-sync"))
    assert alone[0]["role"] == "sampler"
    both = _records(models, "qwen3_8b", "modal:1x8xh100", ("--rl-single-island-no-sync",))
    assert "role" not in both[0]


def test_kv_cache_per_gpu_follows_the_attention_layout():
    cfg = CONFIGS["qwen3_8b"]
    qwen = me.arch_from_hf_config(cfg)
    assert me.kv_bytes_per_token(qwen, cfg, 2) == me.kv_bytes_per_token(qwen, cfg, 1) / 2
    assert me.kv_bytes_per_token(qwen, cfg, 16) == me.kv_bytes_per_token(qwen, cfg, 8)  # 8 KV heads
    k31 = CONFIGS["kimi_k31"]
    arch = me.arch_from_hf_config(k31)
    mla = me.kv_bytes_per_token(arch, k31, 8)
    assert mla == me.kv_bytes_per_token(arch, k31, 1) == 2 * 93 * (512 + 64)


def test_an_engine_that_does_not_hold_the_model_warns_and_never_stops(models):
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch_preflight import memory_preflight

    args = _launch(models, "qwen3_8b", "modal:1x8xh100", "--rl-sample-role", "sampler",
                   "--rl-single-island-no-sync", "--sglang-mem-fraction-static", "0.15")
    out = io.StringIO()
    report = memory_preflight(args, parse_gpu_spec(args.gpu), out=out)  # mode error: no PreflightError
    rec = report["islands"][0]
    assert rec["role"] == "sampler" and not rec["engine_fits"] and not rec["fits"]
    assert [s["change"] for s in rec["suggestions"]] == ["--rollout-num-gpus-per-engine 2",
                                                        "--sglang-mem-fraction-static 0.6"]
    text = out.getvalue()
    assert "WARNING (sampler engine estimate, advisory only):" in text and "STOP" not in text
    assert "sampler engine: weights + one context" in text
