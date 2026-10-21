"""Rollout quantization tiers (S26 E6): launch refusals, tier 1 engine plans, tier 2
scoped FP8 layout, conversion, release scope check and the releaser."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tests.test_rl_quant_scope import H, dense_shapes, moe_shapes
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.quant import tiers
from yeto.rl.quant.awq import AwqSmoothing
from yeto.rl.quant.fp8 import quantize_fp8_per_channel
from yeto.rl.quant.release import scoped_releaser
from yeto.rl.quant.requant import (
    CheckpointLayout,
    LayoutError,
    requantize_checkpoint,
    save_scoped_template,
    scoped_layout,
)
from yeto.rl.quant.schemes import FP8_FFN_CHANNEL, release_dir_for
from yeto.rl.quant.scope import QuantScopeError
from yeto.rl.quant.tiers import QuantTierError


def write_checkpoint(directory: Path, shapes: dict[str, list[int]], config: dict, seed: int = 0) -> Path:
    """Two shards (layer 0 + embeddings, the rest), an index and ``config.json``."""
    g = torch.Generator().manual_seed(seed)
    directory.mkdir(parents=True, exist_ok=True)
    shards: dict[str, dict[str, torch.Tensor]] = {"model-00001-of-00002.safetensors": {},
                                                  "model-00002-of-00002.safetensors": {}}
    for name, shape in shapes.items():
        shard = "model-00001-of-00002.safetensors" if ".layers.1." not in name else "model-00002-of-00002.safetensors"
        shards[shard][name] = torch.randn(*shape, generator=g).to(torch.bfloat16)
    weight_map = {}
    for shard, tensors in shards.items():
        save_file(tensors, str(directory / shard), metadata={"format": "pt"})
        weight_map.update({n: shard for n in tensors})
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "tokenizer.json").write_text("{}")
    return directory


@pytest.fixture
def dense(tmp_path: Path) -> Path:
    return write_checkpoint(tmp_path / "dense", dense_shapes(), {"num_hidden_layers": 2, "tie_word_embeddings": True})


@pytest.fixture
def moe(tmp_path: Path) -> Path:
    return write_checkpoint(tmp_path / "moe", moe_shapes(), {"num_hidden_layers": 2, "num_experts": 2})


def _read(directory: Path) -> dict[str, torch.Tensor]:
    out = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            out.update({k: f.get_tensor(k) for k in f.keys()})
    return out


# ---------------------------------------------------------------- launch refusals
def _args(**kw):
    base = {"rl_rollout_quant_tier": 1, "gpu": "modal:1xh100", "tuning": "full", "rl_island_sample_role": None}
    return SimpleNamespace(**{**base, **kw})


def test_tier_0_and_1_pass_and_tier_3_is_refused():
    assert tiers.validate_launch(_args(rl_rollout_quant_tier=0, gpu="aws:1xa100")) == 0
    assert tiers.validate_launch(_args()) == 1
    assert tiers.validate_launch(_args(gpu="aws:1xl4@us-east-1,nebius:2x8xh200", tuning="lora")) == 1
    assert tiers.validate_launch(_args(gpu=None)) == 1  # auto fleet: the planner picks the card
    with pytest.raises(QuantTierError, match="interface only"):
        tiers.validate_launch(_args(rl_rollout_quant_tier=3))
    with pytest.raises(QuantTierError, match="expected one of"):
        tiers.validate_launch(_args(rl_rollout_quant_tier=4))


@pytest.mark.parametrize("gpu", ["aws:1xa100", "modal:1xa10g", "modal:1xh100,aws:2xa100-80gb", "ssh:1x8x910b4"])
def test_cards_without_fp8_are_refused(gpu):
    with pytest.raises(QuantTierError, match="FP8 tensor cores"):
        tiers.validate_launch(_args(gpu=gpu))


def test_tier_2_needs_sampler_islands_and_full_parameters():
    with pytest.raises(QuantTierError, match="--rl-island-sample-role"):
        tiers.validate_launch(_args(rl_rollout_quant_tier=2))
    with pytest.raises(QuantTierError, match="--tuning full"):
        tiers.validate_launch(_args(rl_rollout_quant_tier=2, rl_island_sample_role=["1:sampler"], tuning="lora"))
    assert tiers.validate_launch(_args(rl_rollout_quant_tier=2, rl_island_sample_role=["1:sampler"])) == 2
    assert tiers.release_scheme(2) == FP8_FFN_CHANNEL and tiers.release_scheme(1) == "bf16"


def test_launcher_runs_the_tier_check_before_anything_else():
    from yeto.launch.rl_args import _prepare_rl_args

    args = SimpleNamespace(training_mode="rl", rl_rollout_quant_tier=3, rl_image=None, gpu="modal:1xh100")
    with pytest.raises(QuantTierError, match="interface only"):
        _prepare_rl_args(args)


def test_flag_is_one_shared_definition_for_cli_and_island():
    from yeto.rl.engine.run_flags import CLI, ISLAND, rl_flags

    (flag,) = [f for f in rl_flags() if f.options == (tiers.FLAG,)]
    assert flag.layers == frozenset({CLI, ISLAND}) and not flag.island
    assert flag.kwargs["choices"] == tiers.TIERS and flag.kwargs["default"] == 0


# ---------------------------------------------------------------- tier 1 plans
def test_online_plan_on_a_dense_checkpoint(dense):
    plan = tiers.online_plan(dense)
    assert plan.kind == "dense"
    assert plan.quantized == tuple(f"model.layers.{i}.mlp.{p}" for i in range(2)
                                   for p in ("down_proj", "gate_proj", "up_proj"))
    assert "model.layers.0.self_attn.q_proj" in plan.explicit and "self_attn" in plan.segments
    env = tiers.sglang_online_env(plan)
    assert set(env) == {"SGLANG_FP8_IGNORED_LAYERS"}
    entries = env["SGLANG_FP8_IGNORED_LAYERS"].split(",")
    assert "model.layers.1.self_attn.o_proj" in entries and not any("mlp.up_proj" in e for e in entries)
    assert tiers.sglang_online_argv() == ["--sglang-quantization", "fp8"]


def test_vllm_overrides_carry_verl_fp8_config_with_our_ignore_list(dense):
    plan = tiers.online_plan(dense)
    overrides = tiers.vllm_online_overrides(plan, tiers.num_hidden_layers(dense))
    assert overrides[0] == "actor_rollout_ref.rollout.quantization=fp8"
    assert not any("kv_cache_dtype" in o for o in overrides)  # KV cache stays BF16 (gate, 10-21)
    key, value = overrides[1].split("=", 1)
    assert key == "++actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides"
    assert value.startswith("'") and value.endswith("'")
    config = json.loads(value[1:-1])["quantization_config"]
    assert {k: config[k] for k in tiers.VLLM_FP8_CONFIG} == tiers.VLLM_FP8_CONFIG
    ignored = set(config["ignored_layers"])
    assert {"model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.v_proj", "model.layers.0.mlp.gate"} <= ignored
    assert not {m for m in ignored if m.endswith(("gate_proj", "up_proj", "down_proj"))}


def test_online_plan_refuses_a_model_with_an_unknown_linear(tmp_path):
    shapes = {**dense_shapes(), "visual.blocks.0.mlp.fc1.weight": [16, 8]}
    ckpt = write_checkpoint(tmp_path / "vl", shapes, {"num_hidden_layers": 2})
    plan = tiers.online_plan(ckpt)  # the explicit list keeps the vision linear BF16
    assert "visual.blocks.0.mlp.fc1" in plan.explicit and "visual.blocks.0.mlp.fc1" not in plan.quantized


def test_miles_island_env_and_argv_for_tier_1(dense):
    from yeto.rl.adapters.miles.entry import rollout_quant_env

    assert rollout_quant_env(None) == {}
    assert rollout_quant_env(SimpleNamespace(yeto_rollout_quant_tier=0, hf_checkpoint=str(dense))) == {}
    env = rollout_quant_env(SimpleNamespace(yeto_rollout_quant_tier=1, hf_checkpoint=str(dense)))
    assert "model.layers.0.self_attn.k_proj" in env["SGLANG_FP8_IGNORED_LAYERS"]


def test_miles_connect_island_ray_forwards_the_ignore_list(dense):
    from yeto.rl.adapters.miles.entry import connect_island_ray

    seen = {}

    class FakeRay:
        @staticmethod
        def is_initialized():
            return False

        @staticmethod
        def init(address, runtime_env):
            seen.update(address=address, env=runtime_env["env_vars"])

    args = SimpleNamespace(yeto_rollout_quant_tier=1, hf_checkpoint=str(dense))
    connect_island_ray(environ={"RAY_ADDRESS": "10.0.0.1:6379"}, ray_module=FakeRay, miles_args=args)
    assert "model.layers.1.self_attn.q_proj" in seen["env"]["SGLANG_FP8_IGNORED_LAYERS"]
    environ: dict[str, str] = {}
    assert connect_island_ray(environ=environ, ray_module=FakeRay, miles_args=args) is None
    assert "SGLANG_FP8_IGNORED_LAYERS" in environ


def test_verl_tier_1_is_refused_on_a_colocated_island_and_at_launch(dense):
    from yeto.rl.adapters.verl.island_entry import rollout_quant_overrides

    assert rollout_quant_overrides(0, dense, colocated=True) == ([], {"tier": 0, "name": "bf16"})
    assert rollout_quant_overrides(1, dense, colocated=False)[0] == []
    with pytest.raises(SystemExit, match="not available on verl"):
        rollout_quant_overrides(1, dense, colocated=True)
    with pytest.raises(tiers.QuantTierError, match="not available on verl"):
        tiers.validate_launch(_args(rl_backend="verl"))
    assert tiers.validate_launch(_args(rl_backend="miles")) == 1


# ---------------------------------------------------------------- tier 2 layout and conversion
def test_scoped_layout_quantizes_only_the_dense_ffn(dense, tmp_path):
    layout = save_scoped_template(dense, tmp_path / "t", FP8_FFN_CHANNEL)
    assert layout.scheme.name == FP8_FFN_CHANNEL
    assert layout.quantized_modules == frozenset(f"model.layers.{i}.mlp.{p}" for i in range(2)
                                                 for p in ("gate_proj", "up_proj", "down_proj"))
    assert layout.tensors["model.layers.0.mlp.up_proj.weight"].dtype == "F8_E4M3"
    assert layout.tensors["model.layers.0.mlp.up_proj.weight_scale"].shape == (16, 1)
    assert layout.tensors["model.layers.1.mlp.up_proj.weight_scale"].shard == "model-00002-of-00002.safetensors"
    assert layout.tensors["model.layers.0.self_attn.q_proj.weight"].dtype == "BF16"
    config = json.loads((tmp_path / "t" / "config.json").read_text())
    q = config["quantization_config"]
    assert (q["quant_method"], q["format"]) == ("compressed-tensors", "float-quantized")
    assert {"model.layers.0.self_attn.o_proj", "model.embed_tokens", "lm_head"} <= set(q["ignore"])  # lm_head tied, absent
    # fused alias for engines without a packed_modules_mapping (SGLang Qwen3); FFN stays quantized
    assert "model.layers.0.self_attn.qkv_proj" in q["ignore"]
    assert not any(name.endswith("gate_up_proj") for name in q["ignore"])
    assert CheckpointLayout.load(tmp_path / "t").quantized_modules == layout.quantized_modules
    assert (tmp_path / "t" / "tokenizer.json").is_file()


def test_scoped_layout_quantizes_only_moe_routed_experts(moe):
    layout, kind = scoped_layout(moe, FP8_FFN_CHANNEL)
    assert kind == "moe"
    assert layout.quantized_modules == frozenset(f"model.layers.1.mlp.experts.{e}.{p}" for e in range(2)
                                                 for p in ("gate_proj", "up_proj", "down_proj"))
    ignore = set(layout.quantization_config["ignore"])
    assert {"model.layers.1.mlp.gate", "model.layers.1.mlp.shared_expert.up_proj", "lm_head",
            "model.layers.0.mlp.up_proj", "model.layers.0.mlp.gate_up_proj",
            "model.layers.1.mlp.shared_expert.gate_up_proj"} <= ignore
    assert not any(".experts." in name for name in ignore)


def test_scoped_layout_refuses_unknown_scheme_and_empty_scope(tmp_path):
    ckpt = write_checkpoint(tmp_path / "attn", {k: v for k, v in dense_shapes().items() if ".mlp." not in k},
                            {"num_hidden_layers": 2})
    with pytest.raises(LayoutError, match="not a scoped scheme"):
        scoped_layout(ckpt, "w4a8-int4-channel")
    with pytest.raises(LayoutError, match="no in-scope linear"):
        scoped_layout(ckpt, FP8_FFN_CHANNEL)


def test_conversion_matches_the_reference_and_keeps_the_rest_bit_equal(dense, tmp_path):
    save_scoped_template(dense, tmp_path / "t", FP8_FFN_CHANNEL)
    stats = requantize_checkpoint(dense, tmp_path / "t", tmp_path / "out", AwqSmoothing.none())
    assert stats["quantized"] == 6  # FP8 weight tensors
    src, out = _read(dense), _read(tmp_path / "out")
    for name, tensor in src.items():
        module = name[: -len(".weight")]
        if ".mlp." in name:
            q, scale = quantize_fp8_per_channel(tensor)
            assert torch.equal(out[name].view(torch.uint8), q.view(torch.uint8)), name
            assert torch.equal(out[module + ".weight_scale"], scale), name
        else:
            assert out[name].dtype == torch.bfloat16 and torch.equal(out[name], tensor), name
    assert tiers.check_release_scope(tmp_path / "out") == 6
    assert json.loads((tmp_path / "out" / "config.json").read_text())["quantization_config"]["format"] == "float-quantized"


def test_release_scope_check_refuses_fp8_attention_and_unlisted_bf16(dense, tmp_path):
    save_scoped_template(dense, tmp_path / "t", FP8_FFN_CHANNEL)
    requantize_checkpoint(dense, tmp_path / "t", tmp_path / "out", AwqSmoothing.none())
    bad = tmp_path / "out" / "model-00001-of-00002.safetensors"
    tensors = _read(tmp_path / "out")
    shard0 = {k: v for k, v in tensors.items() if ".layers.1." not in k}
    shard0["model.layers.0.self_attn.q_proj.weight"] = quantize_fp8_per_channel(shard0["model.layers.0.self_attn.q_proj.weight"])[0]
    save_file(shard0, str(bad), metadata={"format": "pt"})
    with pytest.raises(QuantScopeError, match=r"self_attn.q_proj \(attention\)"):
        tiers.check_release_scope(tmp_path / "out")

    config_path = tmp_path / "out" / "config.json"
    requantize_checkpoint(dense, tmp_path / "t", tmp_path / "out", AwqSmoothing.none())
    config = json.loads(config_path.read_text())
    config["quantization_config"]["ignore"].remove("model.layers.0.self_attn.q_proj")
    config_path.write_text(json.dumps(config))
    with pytest.raises(QuantScopeError, match="not in the config ignore list"):
        tiers.check_release_scope(tmp_path / "out")
    config["quantization_config"]["format"] = "int-quantized"
    config_path.write_text(json.dumps(config))
    with pytest.raises(QuantScopeError, match="expected compressed-tensors/float-quantized"):
        tiers.check_release_scope(tmp_path / "out")


def test_scoped_releaser_converts_each_bf16_release(dense, tmp_path):
    store = CheckpointStore(tmp_path / "store")
    trained = write_checkpoint(tmp_path / "trained", dense_shapes(), {"num_hidden_layers": 2}, seed=9)
    PolicyReleaseStore(store).release(3, "h3", lambda d: [os.link(p, d / p.name) for p in trained.iterdir()])
    releaser = scoped_releaser(store, dense, tmp_path / "template", FP8_FFN_CHANNEL, log=lambda m: None)
    record = releaser.sync()
    assert record is not None and record.version == 3 and record.policy_hash == f"h3+{FP8_FFN_CHANNEL}"
    assert record.derived_from["scheme"] == FP8_FFN_CHANNEL and record.derived_from["expert_matrices"] == 0  # dense: no routed expert
    target = PolicyReleaseStore(store, release_dir=release_dir_for(FP8_FFN_CHANNEL))
    out = target.resolve(target.latest())
    assert tiers.check_release_scope(out) == 6
    q, _ = quantize_fp8_per_channel(_read(trained)["model.layers.1.mlp.down_proj.weight"])
    assert torch.equal(_read(out)["model.layers.1.mlp.down_proj.weight"].view(torch.uint8), q.view(torch.uint8))
    assert releaser.sync() is None  # converted once
    assert release_dir_for(FP8_FFN_CHANNEL) == "policy-releases-fp8-w8a8-channel"


def test_port_loader_refuses_an_out_of_scope_fp8_release(dense, tmp_path):
    from yeto.rl.adapters.sampler_engines.fake import FAKE_IDENTITY, FakeEngine
    from yeto.rl.engine.policy_release import ReleaseNotLoaded
    from yeto.rl.engine.sampler_checks import PortReleaseLoader, PostLoadChecks

    save_scoped_template(dense, tmp_path / "t", FP8_FFN_CHANNEL)
    requantize_checkpoint(dense, tmp_path / "t", tmp_path / "out", AwqSmoothing.none())
    config_path = tmp_path / "out" / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["ignore"] = []
    config_path.write_text(json.dumps(config))
    from dataclasses import replace

    engine = FakeEngine(identity=replace(FAKE_IDENTITY, quant_scheme=FP8_FFN_CHANNEL), expert_matrices=None)
    events = []
    loader = PortReleaseLoader(engine, PostLoadChecks(), lambda e, **f: events.append((e, f)))
    release = SimpleNamespace(seq=1, version=1, derived_from={"scheme": FP8_FFN_CHANNEL, "expert_matrices": 0})
    with pytest.raises(ReleaseNotLoaded, match="quant_scope"):
        loader._serve(release, tmp_path / "out", 1, first=False)
    assert engine.loads == [] and events[-1][1]["step"] == "quant_scope"


def test_moe_release_expert_count_matches_the_sampler_index_check(moe, tmp_path):
    from yeto.rl.engine.sampler_checks import count_release_experts
    from yeto.rl.quant.release import routed_expert_matrices

    layout = save_scoped_template(moe, tmp_path / "t", FP8_FFN_CHANNEL)
    requantize_checkpoint(moe, tmp_path / "t", tmp_path / "out", AwqSmoothing.none())
    assert routed_expert_matrices(layout.quantized_modules) == count_release_experts(tmp_path / "out") == 6
    assert tiers.check_release_scope(tmp_path / "out") == 6
    out = _read(tmp_path / "out")
    assert out["model.layers.1.mlp.gate.weight"].dtype == torch.bfloat16
    assert out["model.layers.1.mlp.shared_expert.up_proj.weight"].dtype == torch.bfloat16
    assert out["model.layers.1.mlp.experts.1.up_proj.weight"].dtype == torch.float8_e4m3fn


def test_miles_translation_adds_sglang_flags_only_for_tier_1():
    from tests.test_rl_miles_adapter_config import make_config, sub
    from yeto.rl.adapters.miles import config as mc
    from yeto.rl.engine.algorithm import AlgorithmSpec

    base = mc.translate_run_config(make_config(), AlgorithmSpec())
    assert "--sglang-quantization" not in base.argv and "yeto_rollout_quant_tier" not in base.runtime_attrs
    tier1 = mc.translate_run_config(sub(make_config(), "serving", rollout_quant_tier=1), AlgorithmSpec())
    argv = list(tier1.argv)
    assert argv[argv.index("--sglang-quantization") + 1] == "fp8"
    assert "--sglang-kv-cache-dtype" not in argv
    assert tier1.runtime_attrs["yeto_rollout_quant_tier"] == 1
    tier2 = mc.translate_run_config(sub(make_config(), "serving", rollout_quant_tier=2), AlgorithmSpec())
    assert "--sglang-quantization" not in tier2.argv and tier2.runtime_attrs["yeto_rollout_quant_tier"] == 2


@pytest.mark.parametrize("tier", [0, 1])
def test_island_task_forwards_the_tier_only_when_set(monkeypatch, tier):
    import sys
    import types

    from tests import test_rl_launcher as tl
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.rl_args import _prepare_rl_args
    from yeto.launcher import make_miles_island_task

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=tl._Task, Resources=tl._Resources, Storage=tl._Storage, StorageMode=tl._StorageMode))
    args = tl._args(("--gpu", "aws:1xh100@us-east-1", "--rl-rollout-quant-tier", str(tier)))
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    _prepare_rl_args(args)
    task = make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:29400")
    assert ("--rl-rollout-quant-tier 1" in task.run) is (tier == 1)


MILES_NEXT = Path("/home/michael/work/miles-next")


@pytest.mark.skipif(not (MILES_NEXT / "miles/utils/workers/process_utils.py").is_file(), reason="no miles-next checkout")
def test_ignore_list_reaches_the_sglang_server_process(dense, tmp_path, monkeypatch):
    """Tier 1, item 4 of the GPU review: the last hop. Miles' CommandActor starts
    ``sglang.launch_server`` with ``process_utils.launch_bound_subprocess(["/bin/sh", "-c",
    "exec <cmd>"], envs={})`` (ray_worker_manager.py:754, command_actor.py:22), whose child env
    is ``{**os.environ, **envs}``. The actor's os.environ holds the job runtime_env entries
    (connect_island_ray, tested above). This runs Miles' own function and reads the
    variable from the started process."""
    import importlib.util
    import sys

    from yeto.rl.adapters.miles.entry import rollout_quant_env

    monkeypatch.setenv("PYTHONPATH", f"{MILES_NEXT}:{os.environ.get('PYTHONPATH', '')}")
    for key, value in rollout_quant_env(SimpleNamespace(yeto_rollout_quant_tier=1, hf_checkpoint=str(dense))).items():
        monkeypatch.setenv(key, value)  # what the Ray worker's environment holds
    # loaded from its file, so no real ``miles`` package lands in sys.modules (other tests fake it)
    spec = importlib.util.spec_from_file_location("miles_process_utils",
                                                  MILES_NEXT / "miles/utils/workers/process_utils.py")
    process_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(process_utils)
    out = tmp_path / "seen.txt"
    cmd = f"exec {sys.executable} -c \"import os; open('{out}', 'w').write(os.environ['SGLANG_FP8_IGNORED_LAYERS'])\""
    process = process_utils.launch_bound_subprocess(["/bin/sh", "-c", cmd], envs={})
    assert process.wait(timeout=60) == 0
    assert out.read_text() == os.environ["SGLANG_FP8_IGNORED_LAYERS"]
    assert "model.layers.1.self_attn.v_proj" in out.read_text().split(",")


QWEN3_4B = json.loads((Path(__file__).parent / "data" / "qwen3_4b_instruct_2507_weights.json").read_text())


@pytest.mark.skipif(not (MILES_NEXT / "miles/backends/megatron_utils/megatron_to_hf/qwen2.py").is_file(),
                    reason="no miles-next checkout")
def test_miles_raw_export_names_equal_the_hf_base_names():
    """Item 5 of the GPU review (raw path): every Megatron parameter of Qwen3-4B (TE layout,
    tied embeddings: no output_layer) through Miles' own ``convert_qwen2_to_hf`` gives exactly
    the tensor names of the HF base index. The Qwen3-4B launch uses Miles' bridge export
    (Megatron-Bridge, not in this checkout); that path is checked on the first release."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "miles_qwen2_to_hf", MILES_NEXT / "miles/backends/megatron_utils/megatron_to_hf/qwen2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = QWEN3_4B
    head_dim, groups, heads = 4, 2, 4
    args = SimpleNamespace(kv_channels=head_dim, hidden_size=8, num_attention_heads=heads, num_query_groups=groups)
    params = {"module.module.embedding.word_embeddings.weight": torch.zeros(4, 8),
              "module.module.decoder.final_layernorm.weight": torch.zeros(8)}
    for i in range(c["num_hidden_layers"]):
        p = f"module.module.decoder.layers.{i}."
        params.update({p + "self_attention.linear_qkv.weight": torch.zeros(groups * (heads // groups + 2) * head_dim, 8),
                       p + "self_attention.linear_qkv.layer_norm_weight": torch.zeros(8),
                       p + "self_attention.linear_proj.weight": torch.zeros(8, 8),
                       p + "self_attention.q_layernorm.weight": torch.zeros(head_dim),
                       p + "self_attention.k_layernorm.weight": torch.zeros(head_dim),
                       p + "mlp.linear_fc1.weight": torch.zeros(16, 8),
                       p + "mlp.linear_fc1.layer_norm_weight": torch.zeros(8),
                       p + "mlp.linear_fc2.weight": torch.zeros(8, 8)})
    names = {hf for name, param in params.items() for hf, _ in module.convert_qwen2_to_hf(args, name, param)}
    assert c["tie_word_embeddings"] and "lm_head.weight" not in c["weight_names"]
    assert names == set(c["weight_names"])


def test_tied_lm_head_in_an_export_is_dropped(tmp_path):
    """Item 5 fallback: an exporter that writes the tied head anyway still converts."""
    shapes = dense_shapes()
    base = write_checkpoint(tmp_path / "base", shapes, {"num_hidden_layers": 2, "tie_word_embeddings": True})
    export = write_checkpoint(tmp_path / "export", {**shapes, "lm_head.weight": [32, H]},
                              {"num_hidden_layers": 2, "tie_word_embeddings": True})
    save_scoped_template(base, tmp_path / "t", FP8_FFN_CHANNEL)
    stats = requantize_checkpoint(export, tmp_path / "t", tmp_path / "out", AwqSmoothing.none())
    assert stats["dropped_tied_lm_head"] == 1 and "lm_head.weight" not in _read(tmp_path / "out")
    untied = write_checkpoint(tmp_path / "base2", shapes, {"num_hidden_layers": 2})
    save_scoped_template(untied, tmp_path / "t2", FP8_FFN_CHANNEL)
    with pytest.raises(LayoutError, match="1 extra"):
        requantize_checkpoint(export, tmp_path / "t2", tmp_path / "out2", AwqSmoothing.none())
