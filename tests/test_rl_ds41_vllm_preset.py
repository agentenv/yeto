"""The DeepSeek-V4.1-Flash MI355X preset of the vLLM sampler (``VllmOptions.model_preset``):
recipe flags and environment, parsers and tokenizer mode, none of the K3.1 overrides."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from yeto.rl.adapters.sampler_engines.vllm_server import MODEL_PRESETS, ROCM_ENV_PRESET
from yeto.rl.engine.sampler_engine import SamplerEngineError

from test_rl_k31_engine_sampler_r3 import _config
from test_rl_k31_vllm_sampler import _engine

ROOT = Path(__file__).resolve().parents[1]


def test_ds41_mi355x_preset_argv_and_env(tmp_path):
    engine, _ = _engine(tmp_path, "rocm", config=_config(tmp_path), model_preset="ds41-mi355x")
    argv, env = engine.argv(engine.config.base_dir), engine.env()
    assert argv[argv.index("--tensor-parallel-size") + 1] == "8"
    assert argv[argv.index("--block-size") + 1] == "128" and argv[argv.index("--moe-backend") + 1] == "aiter"
    assert json.loads(argv[argv.index("--attention-config") + 1]) == {"indexer_kv_dtype": "mxfp4",
                                                                      "indexer_sparse_logits": True}
    assert json.loads(argv[argv.index("--engram-config") + 1]) == {"cpu_offload": False}
    assert argv[argv.index("--tokenizer-mode") + 1] == "deepseek_v41"
    assert argv[argv.index("--tool-call-parser") + 1] == "deepseek_v41" and "--enable-auto-tool-choice" in argv
    assert argv[argv.index("--reasoning-parser") + 1] == "deepseek_v41"
    # no K3.1 overrides, no DSpark / mono decode
    for flag in ("--decode-context-parallel-size", "--hf-overrides", "--additional-config", "--kv-cache-dtype",
                 "--speculative-config", "--monokernel", "--mamba-ssm-cache-dtype", "--quantization-config"):
        assert flag not in argv, flag
    for key in ("VLLM_ROCM_USE_AITER_MOE", "VLLM_USE_BREAKABLE_CUDAGRAPH", "VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA"):
        assert env[key] == "1", key
    assert set(ROCM_ENV_PRESET) <= set(env)
    assert "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION" not in env and "AITER_FLYDSL_SPARSE_MLA" not in env


def test_ds41_preset_options_win_and_rules(tmp_path):
    engine, _ = _engine(tmp_path, "rocm", model_preset="ds41-mi355x", tool_call_parser="other",
                        reasoning_parser="other_r", tokenizer_mode="auto")
    argv = engine.argv(engine.config.base_dir)
    assert argv[argv.index("--tool-call-parser") + 1] == "other"
    assert argv[argv.index("--reasoning-parser") + 1] == "other_r"
    assert argv[argv.index("--tokenizer-mode") + 1] == "auto"
    given, _ = _engine(tmp_path, "rocm", config=_config(tmp_path, extra_args=("--tokenizer-mode", "slow")),
                       model_preset="ds41-mi355x")
    assert given.argv(given.config.base_dir).count("--tokenizer-mode") == 1  # extra_args' flag only
    r3, _ = _engine(tmp_path, "rocm", model_preset="ds41-mi355x", return_routed_experts=True)
    assert "--enable-return-routed-experts" in r3.argv(r3.config.base_dir) and r3.cp_overrides == []
    with pytest.raises(SamplerEngineError, match="not both"):
        _engine(tmp_path, "rocm", model_preset="ds41-mi355x", k31_preset="rl-precision")
    with pytest.raises(SamplerEngineError, match="model_preset must be one of"):
        _engine(tmp_path, "rocm", model_preset="ds41")
    with pytest.raises(SamplerEngineError, match="rocm preset"):
        _engine(tmp_path, "cuda", model_preset="ds41-mi355x")
    with pytest.raises(SamplerEngineError, match="need upstream vLLM"):
        _engine(tmp_path, "npu", tokenizer_mode="deepseek_v41")
    hot, _ = _engine(tmp_path, "rocm", config=_config(tmp_path, in_place_reload=True), model_preset="ds41-mi355x")
    assert hot._linear_reload()  # in place through vllm_linear_reload, not reload_weights
    assert {p["platform"] for p in MODEL_PRESETS.values()} == {"rocm"}


def test_ds41_example_spec_uses_the_preset():
    spec = json.loads((ROOT / "examples/ds41/sampler-spec.nc-mi355x-vllm.json").read_text())
    options = spec["engine"]["options"]
    assert options["model_preset"] == "ds41-mi355x" and "k31_preset" not in options and options["tp"] == 8
    assert spec["platform"] == "rocm" and spec["engine"]["server"]["max_model_len"] == 65536


def test_chat_renderer_uses_the_checkpoint_encoder_without_a_chat_template(tmp_path):
    from yeto.rl.engine.engine_sampler_entry import chat_renderer

    (tmp_path / "encoding").mkdir()
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"add_bos_token": False}))
    (tmp_path / "encoding" / "encoding.py").write_text(
        "def encode_messages(messages, thinking_mode, reasoning_effort=None):\n"
        "    body = ''.join(f'<{m[\"role\"]}>{m[\"content\"]}' for m in messages)\n"
        "    return f'<bos>{body}<assistant>' + ('<think>' if thinking_mode == 'thinking' else '</think>')\n")
    assert chat_renderer(tmp_path)("hi") == "<bos><user>hi<assistant><think>"
    chat = chat_renderer(tmp_path, template_kwargs={"thinking_mode": "chat"})
    assert chat([{"role": "user", "content": "a"}]) == "<bos><user>a<assistant></think>"


def test_nc_launcher_picks_the_interpreter_and_mounts_the_ds41_model():
    from yeto.launch.nc import ssh_sampler

    ds41 = ("vllm/vllm-openai-rocm:nightly-e68efc685ab1e6f1eda004ced7157e3ca411da2f@sha256:"
            "520e098f54ec96837228aaddea0a833cbf6374dbdffd480c1d65f26ffab7701b")
    assert ssh_sampler.python_for(ds41) == "/usr/bin/python3"
    assert ssh_sampler.python_for(ssh_sampler.IMAGE) == ssh_sampler.PYTHON == "/opt/runtime-venv/bin/python3"
    assert ssh_sampler.python_for(None) == ssh_sampler.PYTHON
    assert "exec /usr/bin/python3 -m yeto.rl.engine.engine_sampler" in ssh_sampler.sampler_command(
        "/work/sampler/s.json", python="/usr/bin/python3")[-1]
    assert ssh_sampler.MOUNTS["/mnt/nvme/models/DeepSeek-V4.1-Flash"] == "/models/deepseek-v4.1-flash"
    assert ssh_sampler.MOUNTS[ssh_sampler.MODEL_HOST] == "/models/kimi-k3.1-pro"  # K3.1 unchanged
