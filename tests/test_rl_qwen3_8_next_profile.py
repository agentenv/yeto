"""CPU contracts for the Qwen3.8-Flash-Next-4layer native LoRA profile (M4).

Snapshot-guards the rendered conversion / launch commands, cross-checks the
pinned Megatron model args against a Miles checkout when one is present, and
checks the shell launchers' dry-run output is byte-identical to the Python
renderers.  No Miles import, no network, no GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from yeto.rl.profiles import qwen3_8_next as q

ROOT = Path(__file__).resolve().parents[1]
CONVERT_SH = ROOT / "scripts" / "convert_qwen3_8_next.sh"
LAUNCH_SH = ROOT / "scripts" / "run_qwen3_8_next_4layer_lora.sh"
MILES_CHECKOUT = Path(os.environ.get("YETO_MILES_CHECKOUT", "/home/michael/work/miles-m3"))

# Regenerate only when a command change is intended:
#   YETO_Q38N_SNAPSHOT_PRINT=1 python -m pytest -q -s tests/test_rl_qwen3_8_next_profile.py
GOLDEN = {
    "convert_4layer": "d1f016cb7f7d596f65c3b35339fbac965fea9d6f0452f42e3e1be328ca9bd833",
    "launch_4layer": "2347fc20ffa56099a58cf707c5ab306bd116de9f5ee86f752cd90b41fdc0a101",
}


def _digest(tokens: list[str]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def _maybe_print(name: str, tokens: list[str]) -> None:
    if os.environ.get("YETO_Q38N_SNAPSHOT_PRINT"):
        print(f"\n{name}: {_digest(tokens)}\n{shlex.join(tokens)}")


def _run(cmd: list[str], env: dict[str, str] | None = None) -> str:
    merged = {**os.environ, "PYTHONPATH": str(ROOT), **(env or {})}
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env=merged).stdout


# ---------------------------------------------------------------- profile


def test_profile_defaults_match_miles_ci_4layer_shape():
    p = q.Qwen38NextLoraProfile()
    assert p.name == "qwen3_8_next_4layer_lora"
    assert p.model_name == "Qwen3.8-Flash-Next-4layer"
    assert p.megatron_model_type == "qwen3.8-flash-next-4layer"
    assert p.hf_repo == "CharyZeng/Qwen3.8-Flash-Next-4layer"
    assert len(p.hf_revision) == 40
    assert p.hf_checkpoint == "/root/models/Qwen3.8-Flash-Next-4layer"
    assert p.torch_dist == "/root/ckpt/qwen3.8-flash-next-4layer_torch_dist"
    # run_qwen3_8_next.py::_train for 1x8: TP2 PP2 EP4 ETP1, 4-GPU engines TP4/EP4
    assert p.parallel == {
        "tp": 2, "pp": 2, "cp": 1, "ep": 4, "etp": 1,
        "rollout_num_gpus_per_engine": 4, "sglang_tp": 4, "sglang_ep": 4,
    }
    assert (p.convert_tp, p.convert_pp) == (2, 1)
    assert p.lora_expert_rank < p.lora_rank  # exercises the zero-padded export path
    assert p.expected_lora_keys() == 54


def test_profile_rejects_invalid_shapes_and_ranks():
    with pytest.raises(ValueError, match="4 or 8 GPUs"):
        q.Qwen38NextLoraProfile(num_gpus_per_node=6)
    with pytest.raises(ValueError, match="32 GPUs"):
        q.Qwen38NextLoraProfile(variant="full")
    with pytest.raises(ValueError, match="r_e <= lora_rank"):
        q.Qwen38NextLoraProfile(lora_rank=8, lora_expert_rank=16)
    with pytest.raises(ValueError, match="positive"):
        q.Qwen38NextLoraProfile(lora_rank=0)
    with pytest.raises(ValueError, match="unknown variant"):
        q.Qwen38NextLoraProfile(variant="8layer")
    full = q.Qwen38NextLoraProfile(variant="full", num_nodes=8, num_gpus_per_node=4)
    assert full.parallel["pp"] == 8 and full.parallel["ep"] == 4
    assert q.Qwen38NextLoraProfile(lora_expert_rank=0).effective_expert_rank == 32


def test_profile_from_env_overrides_typed_fields():
    p = q.profile_from_env(
        {
            "YETO_Q38N_NUM_ROLLOUT": "20",
            "YETO_Q38N_LORA_EXPERT_RANK": "16",
            "YETO_Q38N_SKIP_SAVING": "true",
            "YETO_Q38N_LORA_DROPOUT": "0.05",
            "YETO_Q38N_MILES_ROOT": "/opt/miles",
        }
    )
    assert (p.num_rollout, p.lora_expert_rank, p.skip_saving, p.lora_dropout) == (20, 16, True, 0.05)
    assert p.convert_command()[3] == "/opt/miles/tools/convert_hf_to_torch_dist.py"


def test_lora_targets_are_the_m3_verified_set():
    assert len(q.LORA_TARGET_MODULES) == 14
    assert all(t.startswith("model.language_model.layers.*.") for t in q.LORA_TARGET_MODULES)
    suffixes = {t.removeprefix("model.language_model.layers.*.") for t in q.LORA_TARGET_MODULES}
    assert suffixes == {
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b",
        "linear_attn.in_proj_a", "linear_attn.out_proj",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.experts.gate_up_proj", "mlp.experts.down_proj",
        "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj",
    }
    assert q.LORA_DEFERRED_TARGET not in q.LORA_TARGET_MODULES
    # 3 GDN layers x 10 keys + 1 QSA layer x 8 + 4 MoE layers x 4 = 54 lora:* keys per engine
    assert 3 * 10 + 1 * 8 + 4 * 4 == q.EXPECTED_LORA_KEYS_4LAYER


def test_expected_trainable_params_formula():
    # M3 acceptance #1 formula at r=32, r_e=8
    assert q.expected_trainable_params_4layer(32, 8) == (
        35424 * 32 * 3 + 29696 * 32 + 7040 * 32 * 4 + 512 * 7040 * 8 * 4
    )


# ------------------------------------------------------------ model args


def test_model_args_pin_matches_miles_checkout_when_present():
    script = MILES_CHECKOUT / "miles" / "utils" / "external_utils" / "model_args_utils.py"
    if not script.is_file():
        pytest.skip(f"no Miles checkout at {MILES_CHECKOUT}")
    for variant, model_type in q.MEGATRON_MODEL_TYPES.items():
        out = subprocess.run(
            [sys.executable, str(script), model_type],
            check=True, capture_output=True, text=True, cwd=str(MILES_CHECKOUT),
        ).stdout
        assert out.split() == list(q.model_args(variant)), variant


def test_model_args_variants_differ_only_in_layer_count():
    four, full = q.model_args("4layer"), q.model_args("full")
    assert four[four.index("--num-layers") + 1] == "4"
    assert full[full.index("--num-layers") + 1] == "48"
    assert four[four.index("--moe-layer-freq") + 1] == "[1,1,1,1]"
    assert full[full.index("--moe-layer-freq") + 1] == "[" + ",".join(["1"] * 48) + "]"
    strip = {"--num-layers", "--moe-layer-freq"}
    def without(tokens):
        out, skip = [], False
        for t in tokens:
            if skip:
                skip = False
                continue
            if t in strip:
                skip = True
                continue
            out.append(t)
        return out
    assert without(four) == without(full)


# ---------------------------------------------------------------- commands


def test_convert_command_snapshot():
    p = q.Qwen38NextLoraProfile()
    cmd = p.convert_command()
    _maybe_print("convert_4layer", cmd)
    assert cmd[:3] == ["torchrun", "--nproc-per-node", "8", ][:3]
    assert cmd[3] == "/root/miles/tools/convert_hf_to_torch_dist.py"
    assert cmd[-8:] == [
        "--hf-checkpoint", "/root/models/Qwen3.8-Flash-Next-4layer",
        "--save", "/root/ckpt/qwen3.8-flash-next-4layer_torch_dist",
        "--tensor-model-parallel-size", "2",
        "--pipeline-model-parallel-size", "1",
    ]
    assert p.convert_env()["CONVERT_KEEP_PP1"] == "1"
    assert _digest(cmd) == GOLDEN["convert_4layer"]
    custom = p.convert_command(hf_checkpoint="/data/hf", output="/data/td", nproc=4)
    assert custom[2] == "4" and custom[-7] == "/data/hf" and custom[-5] == "/data/td"


def test_launch_command_snapshot_and_lora_flags():
    p = q.Qwen38NextLoraProfile()
    cmd = p.launcher_command()
    _maybe_print("launch_4layer", cmd)
    assert cmd[:3] == ["python3", "/root/miles/scripts/run_qwen3_8_next.py", "train"]
    assert "--no-check-weight-update-equal" in cmd
    assert "--no-skip-saving" in cmd and "--enable-r3" in cmd
    extra = shlex.split(cmd[cmd.index("--extra-args") + 1])
    assert extra == p.lora_extra_args()
    flags = dict(zip(extra, extra[1:]))
    assert flags["--megatron-to-hf-mode"] == "raw"
    assert flags["--lora-rank"] == "32" and flags["--lora-alpha"] == "64"
    assert flags["--lora-expert-rank"] == "8"
    assert flags["--sglang-max-lora-rank"] == "32"
    assert flags["--target-modules"].split(",") == list(q.LORA_TARGET_MODULES)
    assert "--lora-type" not in extra  # native plugin refuses canonical_lora
    assert "--experts-shared-outer-loras" not in extra  # per-expert layout
    assert "--check-lora-weight-equal" in extra and "--lora-base-cpu-backup" in extra
    assert "--sglang-lora-strict-loading" in extra
    assert p.launcher_env()["MILES_SCRIPT_EXTERNAL_RAY"] == "1"
    assert _digest(cmd) == GOLDEN["launch_4layer"]
    with_extra = p.launcher_command(extra_args=["--num-rollout", "2"])
    assert shlex.split(with_extra[-1])[-2:] == ["--num-rollout", "2"]


def test_download_commands_pin_revision():
    cmds = q.Qwen38NextLoraProfile().download_commands()
    assert cmds[1][:5] == ["hf", "download", q.HF_REPO_4LAYER, "--revision", q.HF_REVISION_4LAYER]
    assert cmds[2][-1] == "/root/datasets/dapo-math-17k"


# --------------------------------------------------------- shell launchers


def test_module_cli_renders_the_same_commands():
    p = q.Qwen38NextLoraProfile()
    out = _run([sys.executable, "-m", "yeto.rl.profiles.qwen3_8_next", "convert-command"])
    assert shlex.split(out) == p.convert_command()
    out = _run([sys.executable, "-m", "yeto.rl.profiles.qwen3_8_next", "launch-command", "--", "--num-rollout", "2"])
    assert shlex.split(out) == p.launcher_command(extra_args=["--num-rollout", "2"])
    out = _run([sys.executable, "-m", "yeto.rl.profiles.qwen3_8_next", "model-args"])
    assert shlex.split(out) == list(q.model_args("4layer"))
    manifest = json.loads(_run([sys.executable, "-m", "yeto.rl.profiles.qwen3_8_next", "manifest"]))
    assert manifest == p.manifest()
    env_out = _run(
        [sys.executable, "-m", "yeto.rl.profiles.qwen3_8_next", "convert-command"],
        env={"YETO_Q38N_VARIANT": "full", "YETO_Q38N_NUM_NODES": "8", "YETO_Q38N_NUM_GPUS_PER_NODE": "4"},
    )
    assert "--num-layers 48" in env_out and "qwen3.8-flash-next_torch_dist" in env_out


def test_convert_script_dry_run_prints_rendered_command():
    p = q.Qwen38NextLoraProfile()
    out = _run(["bash", str(CONVERT_SH), "--dry-run", "--yeto-root", str(ROOT)])
    lines = [l for l in out.splitlines() if not l.startswith("#")]
    assert len(lines) == 1 and shlex.split(lines[0]) == p.convert_command()
    assert "#   CONVERT_KEEP_PP1=1" in out
    out = _run(
        ["bash", str(CONVERT_SH), "--dry-run", "--yeto-root", str(ROOT), "--hf-checkpoint", "/x/hf",
         "--output", "/x/td", "--nproc", "4", "--miles-root", "/opt/miles"],
    )
    cmd = shlex.split([l for l in out.splitlines() if not l.startswith("#")][0])
    assert cmd == q.Qwen38NextLoraProfile(miles_root="/opt/miles").convert_command(
        hf_checkpoint="/x/hf", output="/x/td", nproc=4
    )


def test_launch_script_dry_run_prints_rendered_steps():
    p = q.Qwen38NextLoraProfile()
    out = _run(["bash", str(LAUNCH_SH), "--dry-run", "--timeout", "1234", "--yeto-root", str(ROOT), "--", "--num-rollout", "2"])
    launch = [l for l in out.splitlines() if l.lstrip().startswith("timeout ")]
    assert len(launch) == 1
    tokens = shlex.split(launch[0])
    assert tokens[:4] == ["timeout", "--signal=TERM", "--kill-after=120", "1234"]
    assert tokens[4:] == p.launcher_command(extra_args=["--num-rollout", "2"])
    assert "hard timeout 1234s, 8 GPUs" in out
    assert f"{ROOT}/scripts/convert_qwen3_8_next.sh --variant 4layer" in out
    assert f"hf download {q.HF_REPO_4LAYER} --revision {q.HF_REVISION_4LAYER}" in out
    for script in (CONVERT_SH, LAUNCH_SH):
        assert os.access(script, os.X_OK)
        assert "set -euo pipefail" in script.read_text()
