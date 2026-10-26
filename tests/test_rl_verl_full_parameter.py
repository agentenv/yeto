"""S27 B10 phase 1: the full-parameter boundary of a verl FSDP2 island.

CPU only: the layout and name rules are pure, and the worker-side operations
run against plain ``torch`` tensors (no DTensor, no verl, no Ray).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yeto.rl.adapters.verl import config as vconf
from yeto.rl.adapters.verl import full_parameter as fp
from yeto.rl.local_learner import ComponentIdentity

COMPONENT = ComponentIdentity(role="actor", model_revision="a" * 40, config_hash="b" * 64)
SHAPES = {
    "model.embed_tokens.weight": (64, 8),
    "model.layers.0.mlp.down_proj.weight": (8, 16),
    "model.layers.0.self_attn.q_proj.weight": (8, 8),
    "model.norm.weight": (8,),
}


class _Module:
    """Stand-in for the unwrapped FSDP2 training module."""

    def __init__(self, shapes=SHAPES, *, dtype=torch.float32, prefix="", frozen=(), buffers=()):
        self._params = {
            f"{prefix}{name}": torch.nn.Parameter(
                torch.arange(_numel(shape), dtype=dtype).reshape(shape),
                requires_grad=name not in frozen,
            )
            for name, shape in shapes.items()
        }
        # persistent buffers, as a leaf nn.Module holds them (S27 G-EQ3: the HF
        # export reads them too, a MoE router's correction bias is one)
        self._buffers = {f"{prefix}{name}": torch.full(shape, 0.5, dtype=dtype)
                         for name, shape in dict(buffers).items()}
        self._non_persistent_buffers_set: set[str] = set()

    def named_parameters(self):
        return list(self._params.items())

    def named_modules(self):
        return [("", self)]


def _numel(shape) -> int:
    total = 1
    for dimension in shape:
        total *= dimension
    return total


def _worker(module) -> SimpleNamespace:
    return SimpleNamespace(actor=SimpleNamespace(engine=SimpleNamespace(module=module, optimizer=None)))


# ----------------------------------------------------------------------------- names
@pytest.mark.parametrize("name, want", [
    ("model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj.weight"),
    ("_fsdp_wrapped_module.model.layers.0.mlp.down_proj.weight",
     "model.layers.0.mlp.down_proj.weight"),
    ("model.layers._checkpoint_wrapped_module.0.mlp.down_proj.weight",
     "model.layers.0.mlp.down_proj.weight"),
    ("_orig_mod.model.norm.weight", "model.norm.weight"),
])
def test_wrapper_segments_do_not_reach_the_boundary(name, want):
    assert fp.base_to_canonical(name) == want


def test_an_adapter_parameter_is_refused():
    with pytest.raises(fp.VerlFullParameterError, match="lora_rank=0"):
        fp.base_to_canonical("base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight")


# ----------------------------------------------------------------------------- specs and layout
def test_specs_are_sorted_fp32_and_complete():
    specs = fp.island_specs(SHAPES)
    assert [spec.name for spec in specs] == sorted(SHAPES)
    assert {spec.dtype for spec in specs} == {"float32"}
    assert {spec.shard_id for spec in specs} == {"global"}  # card-count independent
    assert sum(spec.numel for spec in specs) == sum(_numel(shape) for shape in SHAPES.values())


def test_an_empty_boundary_is_refused():
    with pytest.raises(fp.VerlFullParameterError, match="没有任何参数"):
        fp.island_specs({})


def test_the_layout_hash_does_not_depend_on_the_card_count():
    """Every FSDP2 parameter is gathered whole, so the specs carry no shard
    identity: a 2-card and a 4-card island plan the same layout.  (A Miles
    layout names each rank's shard, so its hash changes with the shape.)"""
    specs = fp.island_specs(SHAPES)
    first = fp.plan_layout(specs, component=COMPONENT, minimum_fragments=2)
    again = fp.plan_layout(fp.island_specs(SHAPES), component=COMPONENT, minimum_fragments=2)
    assert first.layout_hash == again.layout_hash
    assert first.fragments.num_fragments == 2


def test_every_parameter_lands_in_exactly_one_fragment():
    layout = fp.plan_layout(fp.island_specs(SHAPES), component=COMPONENT, minimum_fragments=3)
    by_fragment = [fp.fragment_tensor_names(layout, index)
                   for index in range(layout.fragments.num_fragments)]
    seen = [name for names in by_fragment for name in names]
    assert sorted(seen) == sorted(SHAPES)
    assert all(len(set(names)) == len(names) for names in by_fragment)


def test_a_parameter_above_the_bound_is_split_into_row_slices():
    layout = fp.plan_layout(fp.island_specs(SHAPES), component=COMPONENT, minimum_fragments=1,
                            max_fragment_bytes=1024)  # 256 fp32 scalars per fragment
    assert layout.has_split_parameters
    names = [name for index in range(layout.fragments.num_fragments)
             for name in fp.fragment_tensor_names(layout, index)]
    # The embedding (512 scalars) spans fragments; each fragment names it once.
    assert names.count("model.embed_tokens.weight") >= 2


# ----------------------------------------------------------------------------- worker side
def test_the_manifest_reports_every_trainable_parameter():
    manifest = fp.remote_manifest(_worker(_Module()))
    assert {name: tuple(shape) for name, shape in manifest["shapes"].items()} == SHAPES
    assert manifest["param_dtypes"] == ["torch.float32"] and manifest["sharded"] is False


def test_a_frozen_parameter_stays_out_of_the_boundary():
    manifest = fp.remote_manifest(_worker(_Module(frozen=("model.norm.weight",))))
    assert "model.norm.weight" not in manifest["shapes"]


def test_bf16_parameters_are_refused_instead_of_exported_as_masters():
    with pytest.raises(fp.VerlFullParameterError, match="FP32 主权重"):
        fp.remote_manifest(_worker(_Module(dtype=torch.bfloat16)))


def test_the_fsdp_wrapper_prefix_is_stripped_in_the_manifest():
    manifest = fp.remote_manifest(_worker(_Module(prefix="_fsdp_wrapped_module.")))
    assert {name: tuple(shape) for name, shape in manifest["shapes"].items()} == SHAPES


def test_only_the_reading_rank_returns_the_fragments_bytes():
    module = _Module()
    names = ("model.norm.weight", "model.layers.0.mlp.down_proj.weight")
    owner = fp.remote_export_fragment(_worker(module), names, True)
    other = fp.remote_export_fragment(_worker(module), names, False)
    assert set(owner["tensors"]) == set(names) and "tensors" not in other
    assert owner["names"] == other["names"] == list(names)  # same collective order
    assert all(value.dtype == torch.float32 and value.device.type == "cpu"
               for value in owner["tensors"].values())


def test_an_export_of_an_unknown_parameter_is_refused():
    with pytest.raises(fp.VerlFullParameterError, match="模型里没有的参数"):
        fp.remote_export_fragment(_worker(_Module()), ("model.absent.weight",), True)


def test_apply_writes_the_fragment_and_export_reads_it_back():
    module = _Module()
    names = ("model.norm.weight",)
    before = fp.remote_export_fragment(_worker(module), names, True)["tensors"]
    changed = {name: value + 0.5 for name, value in before.items()}
    assert fp.remote_apply_fragment(_worker(module), changed, False) == {
        "applied": 1, "moments_zeroed": 0}
    after = fp.remote_export_fragment(_worker(module), names, True)["tensors"]
    assert torch.equal(after["model.norm.weight"], changed["model.norm.weight"])


def test_apply_refuses_a_shape_that_is_not_the_models():
    module = _Module()
    with pytest.raises(fp.VerlFullParameterError, match="形状不符"):
        fp.remote_apply_fragment(_worker(module), {"model.norm.weight": torch.zeros(9)}, False)


def test_reset_zeroes_the_adam_moments():
    module = _Module()
    worker = _worker(module)
    parameter = module.named_parameters()[0][1]
    worker.actor.engine.optimizer = SimpleNamespace(
        state={parameter: {"exp_avg": torch.ones(3), "exp_avg_sq": torch.ones(3)}})
    result = fp.remote_apply_fragment(worker, {"model.norm.weight": torch.zeros(8)}, True)
    assert result == {"applied": 1, "moments_zeroed": 2}


def test_shape_drift_between_workers_names_the_parameter():
    first = fp.remote_manifest(_worker(_Module()))
    second = fp.remote_manifest(_worker(_Module(frozen=("model.norm.weight",))))
    assert fp.agree_on_shapes([first, first]) == SHAPES
    with pytest.raises(fp.VerlFullParameterError, match="model.norm.weight"):
        fp.agree_on_shapes([first, second])


# ----------------------------------------------------------------------------- run config
def _run(**kw) -> vconf.VerlRunConfig:
    return vconf.VerlRunConfig(model_path="/m", train_file="/t", val_file="/v", out_dir="/o", **kw)


def test_a_full_parameter_island_needs_no_adapter_and_fp32_masters():
    cfg = _run(parameter_mode="full", lora_rank=0, n_gpus=4)
    vconf.validate(cfg)
    assert vconf.full_parameter(cfg)
    with pytest.raises(vconf.VerlConfigError, match="lora_rank=0"):
        vconf.validate(_run(parameter_mode="full", lora_rank=32))
    with pytest.raises(vconf.VerlConfigError, match="FP32 主权重"):
        vconf.validate(_run(parameter_mode="full", lora_rank=0, model_dtype="bf16"))
    with pytest.raises(vconf.VerlConfigError, match="parameter_mode"):
        vconf.validate(_run(parameter_mode="adapter"))


def test_full_mode_overrides_carry_no_lora_keys():
    overrides = vconf.build_overrides(_run(parameter_mode="full", lora_rank=0, n_gpus=4))
    assert "actor_rollout_ref.model.lora_rank=0" in overrides
    assert not [key for key in overrides if "lora_alpha" in key or "target_modules" in key
                or "lora.merge" in key]
    assert "trainer.n_gpus_per_node=4" in overrides


def test_the_lora_island_command_is_unchanged():
    """The default mode stays LoRA: the override list must not gain a key."""
    overrides = vconf.build_overrides(_run())
    assert "actor_rollout_ref.model.lora_rank=32" in overrides
    assert "actor_rollout_ref.model.lora_alpha=32" in overrides
    assert "actor_rollout_ref.model.target_modules=all-linear" in overrides
    assert "actor_rollout_ref.model.lora.merge=False" in overrides
    assert not [key for key in overrides if "parameter_mode" in key]


def test_the_manifest_and_the_export_include_persistent_buffers():
    """S27 G-EQ3: a checkpoint tensor can be a persistent buffer, so the export
    must read it; the sync layout stays parameters only."""
    module = _Module(buffers={"gate.bias": (4,)})
    manifest = fp.remote_manifest(_worker(module))
    assert "gate.bias" not in manifest["shapes"]  # not a parameter: not in the sync layout
    assert manifest["buffers"] == {"gate.bias": [4]} and manifest["buffer_dtypes"] == ["torch.float32"]
    assert fp.agree_on_shapes([manifest, manifest], key="buffers") == {"gate.bias": (4,)}
    exported = fp.remote_export_fragment(_worker(module), ["gate.bias"], True)["tensors"]
    assert torch.equal(exported["gate.bias"], torch.full((4,), 0.5))
    assert fp.agree_on_shapes([manifest]) == {name: tuple(shape)
                                              for name, shape in manifest["shapes"].items()}


def test_a_rank_whose_buffer_differs_stops_the_island():
    """S27 G-EQ3: persistent buffers are not sharded, so each rank's forward reads
    its own copy; a rank that did not get the checkpoint's values computes wrong
    log-probs while the parameters are fine."""
    good = fp.remote_manifest(_worker(_Module(buffers={"gate.bias": (4,)})))
    assert good["buffer_digests"] == {"gate.bias": [2.0, 2.0, 4.0]}  # 4 x 0.5
    assert fp.agree_on_buffers([good, good]) == good["buffer_digests"]
    other = _Module(buffers={"gate.bias": (4,)})
    other._buffers["gate.bias"] = torch.zeros(4)  # an uninitialised copy
    bad = fp.remote_manifest(_worker(other))
    with pytest.raises(fp.VerlFullParameterError, match="持久 buffer 不分片"):
        fp.agree_on_buffers([good, bad])
    with pytest.raises(fp.VerlFullParameterError, match="buffer 集合不一致"):
        fp.agree_on_buffers([good, fp.remote_manifest(_worker(_Module()))])
    assert fp.agree_on_buffers([{"shapes": {}}]) == {}  # an older worker: no field, no failure
