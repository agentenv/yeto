"""S25 T-G1: the pieces the stage-1 mini rehearsal added (CPU)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from yeto.rl.adapters.miles.models import glm5_next
from yeto.rl.adapters.sampler_engines import sglang_load_report as slr
from yeto.rl.adapters.sampler_engines.server import ServerConfig
from yeto.rl.adapters.sampler_engines.sglang_server import SGLangOptions, SGLangServerEngine
from yeto.rl.engine.sampler_checks import FixedPrompt, PostLoadChecks
from yeto.rl.engine.sampler_engine import Completion

REPO = Path(__file__).resolve().parents[1]


def test_recipe_takes_expert_and_dense_counts_from_the_checkpoint():
    shape = glm5_next.checkpoint_geometry({"text_config": {"num_hidden_layers": 4, "first_k_dense_replace": 1,
                                                           "n_routed_experts": 32}})
    assert shape == {"num_layers": 4, "first_k_dense_replace": 1, "num_experts": 32}
    argv = glm5_next.model_argv(**shape)
    assert argv[argv.index("--num-experts") + 1] == "32"
    assert argv[argv.index("--moe-layer-freq") + 1] == "[0,1,1,1]"
    full = glm5_next.model_argv()
    assert full[full.index("--num-experts") + 1] == "288" and full.count("--num-experts") == 1
    with pytest.raises(ValueError):
        glm5_next.model_argv(num_experts=0)


class _Param:
    def __init__(self, calls: list[str]) -> None:
        self.weight_loader = lambda *a, **k: calls.append(a[2] if len(a) > 2 else "?")


class _FakeModel:
    """Maps names like SGLang's GLM5.3 loader: an expert data tensor reaches a loader
    only when its fused name is a parameter."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.params = {"model.layers.1.mlp.experts.w13_weight": _Param(self.calls),
                       "model.layers.1.mlp.experts.w2_weight": _Param(self.calls),
                       "model.layers.1.mlp.experts.w13_weight_scale": _Param(self.calls)}

    def named_parameters(self):
        return iter(self.params.items())

    def load_weights(self, weights):
        for name, tensor in weights:
            fused = (name.replace("experts.0.gate_proj.", "experts.w13_").replace("experts.0.up_proj.", "experts.w13_")
                     .replace("experts.0.down_proj.", "experts.w2_").replace("language_model.", ""))
            param = self.params.get(fused)
            if param is not None:
                param.weight_loader(param, tensor, fused)


def _weights(suffix: str):
    base = "model.language_model.layers.1.mlp.experts.0."
    return [(f"{base}{m}_proj.{suffix}", 0) for m in ("gate", "up", "down")] + \
           [(f"{base}{m}_proj.weight_scale", 0) for m in ("gate", "up", "down")]


def test_load_report_counts_only_loaded_expert_matrices(tmp_path, monkeypatch):
    cls = type("M", (_FakeModel,), {})
    slr.install(cls)
    slr.install(cls)  # idempotent
    monkeypatch.setenv(slr.REPORT_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(slr.RENAME_ENV, raising=False)
    cls().load_weights(_weights("weight_packed"))  # no rename: SGLang skips the packed data silently
    assert slr.read_reports(tmp_path)["expert_matrices"] == 0
    slr.clear_reports(tmp_path)
    assert slr.read_reports(tmp_path) is None
    monkeypatch.setenv(slr.RENAME_ENV, "1")
    model = cls()
    model.load_weights(_weights("weight_packed"))
    report = slr.read_reports(tmp_path)
    assert report["expert_matrices"] == 3 and report["ranks"] == 1
    assert sorted(set(model.calls)) == ["model.layers.1.mlp.experts.w13_weight", "model.layers.1.mlp.experts.w13_weight_scale",
                                        "model.layers.1.mlp.experts.w2_weight"]
    assert list(slr.rename_packed([("a.mlp.experts.3.up_proj.weight_packed", 1), ("x.weight_packed", 2)])) == \
        [("a.mlp.experts.3.up_proj.weight", 1), ("x.weight_packed", 2)]


def test_patch_model_file_is_idempotent(tmp_path):
    f = tmp_path / "glm5_next.py"
    f.write_text("class A: ...\nEntryClass = [A]\n")
    assert slr.patch_model_file(f) is True and slr.patch_model_file(f) is False
    assert f.read_text().count(slr.MARKER) == 1
    compile(f.read_text(), str(f), "exec")
    (tmp_path / "other.py").write_text("x = 1\n")
    with pytest.raises(ValueError):
        slr.patch_model_file(tmp_path / "other.py")


def _engine(tmp_path: Path, tp: int = 1) -> SGLangServerEngine:
    config = ServerConfig(base_dir=tmp_path, card_model="H100", device_family="cuda", engine_version="x",
                          precision="mxfp4", quant_scheme="mxfp4-g32-rtn", topology=f"tp{tp}", max_model_len=64)
    return SGLangServerEngine(config, SGLangOptions(tp=tp, renames_weight_packed=True,
                                                    load_report_dir=str(tmp_path / "reports")))


def test_adapter_reads_the_report_and_exports_the_hook_env(tmp_path):
    engine = _engine(tmp_path, tp=2)
    env = engine.env()
    assert env[slr.RENAME_ENV] == "1" and env[slr.REPORT_DIR_ENV] == str(tmp_path / "reports")
    assert engine.loaded_expert_matrices() is None
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "rank0-1.json").write_text(json.dumps({"expert_matrices": 288}))
    assert engine.loaded_expert_matrices() == 0  # rank 1 did not report
    (reports / "rank1-2.json").write_text(json.dumps({"expert_matrices": 287}))
    assert engine.loaded_expert_matrices() == 287
    with pytest.raises(Exception):
        SGLangServerEngine(ServerConfig(base_dir=tmp_path, card_model="H100", device_family="cuda", engine_version="x",
                                        precision="mxfp4", quant_scheme="mxfp4-g32-rtn", topology="tp2",
                                        max_model_len=64), SGLangOptions(tp=1))


class _Greedy:
    def generate(self, prompts, sampling):
        return [[Completion(text="the the the", token_ids=(1, 2, 3), logprobs=(-1.0,) * 3, finish_reason="length")]
                for _ in prompts]


def test_fixed_answers_may_be_recorded_without_failing():
    prompts = (FixedPrompt("Q: 2+3?", "5"),)
    strict = PostLoadChecks(fixed_prompts=prompts).fixed_prompt_answers(_Greedy())
    assert strict["passed"] is False and strict["answers"][0]["problems"] == ["answer missing"]
    loose = PostLoadChecks(fixed_prompts=prompts, require_fixed_answers=False).fixed_prompt_answers(_Greedy())
    assert loose["passed"] is True and loose["answers"][0]["answered"] is False
    assert loose["answers_required"] is False


def test_mini_modal_plan_has_no_errors_and_uses_the_volume_store():
    from yeto.rl.stage1 import expand, load_plan

    out = expand(load_plan(REPO / "examples/stage1/glm5next-mini-modal.yaml"))
    assert out.errors == [] and out.gaps == []
    env = out.trainer["env"]
    assert env["YETO_RL_STORE_IMPL"] == "yeto.cloud.modal_ckpt_store:store_from_env"
    assert env["YETO_RL_STORE_MODAL_VOLUME"] == "yeto-s25-g1-store"
    argv = out.trainer["miles"]["island_argv"]
    assert argv[argv.index("--miles-commit") + 1] == "1618b5c38db80e17ec928cfa47fe832cb18565fc"
    assert out.trainer["miles"]["model_args"][out.trainer["miles"]["model_args"].index("--num-experts") + 1] == "32"
    spec = out.sampler["engines"][0]["spec"]
    assert spec["checks"] == {"require_fixed_answers": False, "fixed_max_tokens": 64,
                              "expected_expert_matrices": 288}
    assert spec["engine"]["options"]["load_report_dir"].endswith("/load-report")
    assert "--frozen-base" in out.requant["argv"]
    assert argv[argv.index("--megatron-ref-load") + 1] == out.trainer["torch_dist"]
    assert out.trainer["convert_argv"][out.trainer["convert_argv"].index("--save") + 1] == out.trainer["torch_dist"]
    # the PPU plan keeps OSS and passes no --miles-commit for its short commit
    ppu = expand(load_plan(REPO / "examples/stage1/glm53-flash-ppu.yaml"))
    assert ppu.errors == [] and "--miles-commit" not in ppu.trainer["miles"]["island_argv"]
    assert "--frozen-base" not in ppu.requant["argv"]


def test_island_argv_check_runs_the_island_combination_rules():
    from yeto.rl.stage1 import _check_island_argv, expand, load_plan

    argv = expand(load_plan(REPO / "examples/stage1/glm5next-mini-modal.yaml")).trainer["miles"]["island_argv"]
    assert "--rollout-num-gpus" not in argv and _check_island_argv(argv) is None
    problem = _check_island_argv([*argv, "--rollout-num-gpus", "0"])
    assert problem and "fixed-partition" in problem


_MINI_TEXT_CONFIG = {
    "model_type": "glm5_next_text", "num_hidden_layers": 4, "first_k_dense_replace": 1, "hidden_size": 4096,
    "intermediate_size": 12288, "num_attention_heads": 64, "num_key_value_heads": 64, "n_routed_experts": 32,
    "n_shared_experts": 1, "moe_intermediate_size": 2048, "num_experts_per_tok": 8, "vocab_size": 154880,
    "max_position_embeddings": 1048576, "rms_norm_eps": 1e-5, "q_lora_rank": 1536, "kv_lora_rank": 512,
    "qk_head_dim": 256, "qk_rope_head_dim": 0, "v_head_dim": 256, "hidden_act": "silu", "attention_bias": False,
}


def test_mini_island_argv_translates_to_a_miles_argv_on_cpu(tmp_path):
    """The island path up to the Miles argv (Modal runs 1-3 each failed one step of it)."""
    from yeto.rl.adapters.miles.island_entry import build_ports_launch, parse_args
    from yeto.rl.engine.run_config import resolve_rl_run_config
    from yeto.rl.stage1 import expand, load_plan

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "glm5_next", "text_config": _MINI_TEXT_CONFIG}))
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"messages": [{"role": "user", "content": "hi"}], "label": "1"}\n')
    argv = expand(load_plan(REPO / "examples/stage1/glm5next-mini-modal.yaml")).trainer["miles"]["island_argv"][3:]
    argv = [str(model) if a == "/data/models/glm5next-mini-bf16-7fb7883" else a for a in argv]
    argv = [str(REPO / a) if a.startswith("examples/rl_algorithms/") else a for a in argv]
    args = parse_args(argv)
    provider = glm5_next.glm5_next_provider(model)
    config = resolve_rl_run_config(args, model_path=model, prompt_path=prompts, provider=provider,
                                   target_modules=[], yeto_policy_sync=False, verify_ref_load=False)
    assert config.geometry.num_layers == 4 and config.geometry.moe.num_experts == 32
    miles = list(build_ports_launch(args, config, []).argv)
    for flag in ("--debug-train-only", "--use-distributed-optimizer", "--optimizer-cpu-offload",
                 "--use-precision-aware-optimizer", "--use-tis"):
        assert flag in miles
    assert miles[miles.index("--num-experts") + 1] == "32" and miles.count("--num-experts") == 1
    assert miles[miles.index("--moe-layer-freq") + 1] == "[0,1,1,1]"
    assert miles[miles.index("--megatron-to-hf-mode") + 1] == "raw"
