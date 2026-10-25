"""S27 G-EQ3: the HF export writes the base checkpoint's own files.

``transformers`` loads a checkpoint through a registered conversion table, so a
model's state names are not the checkpoint's (``nemotron_h``: ``backbone.`` ->
``model.``, per-expert tensors -> one stacked tensor).  An export that writes
the model's names cannot be delta-encoded against the base and a sampler cannot
load it (S27 G-EQ2, VERL-ASCEND-RL §17).  These tests build a tiny ``nemotron_h``
checkpoint whose layout is the real one (per-expert bf16 weights, an f32 router
bias, several shards), export from the model's tensors, and require every file
to come back byte-for-byte.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

from yeto.rl.adapters.verl.hf_checkpoint_keys import (  # noqa: E402
    INDEX_FILE,
    CheckpointKeyError,
    base_export,
    checkpoint_key_map,
    model_key_map,
)
from yeto.rl.adapters.verl.hf_export import VerlHfWriter, write_like  # noqa: E402

# Big enough that one shard is over release_delta.MIN_DELTA_BYTES (1 MiB), so a
# delta release of this checkpoint encodes words instead of copying files whole.
# The block pattern uses only M (mamba), * (attention) and E (MoE): transformers
# 5.3, which the Ascend image carries, does not know the "-" (dense MLP) block.
TINY = dict(vocab_size=4096, hidden_size=256, intermediate_size=256, num_hidden_layers=4,
            hybrid_override_pattern="M*EE", num_attention_heads=8, num_key_value_heads=4, head_dim=32,
            mamba_num_heads=8, mamba_head_dim=32, ssm_state_size=16, n_groups=1, conv_kernel=4,
            n_routed_experts=4, n_shared_experts=1, moe_intermediate_size=128,
            moe_shared_expert_intermediate_size=128, num_experts_per_tok=2, tie_word_embeddings=False,
            torch_dtype="bfloat16")


def _run(coroutine):
    return asyncio.new_event_loop().run_until_complete(coroutine)


class _Group:
    """The island's full-parameter group: FP32 masters of the model's tensors."""

    def __init__(self, parameters, buffers):
        self.parameters = parameters
        self.buffers = buffers
        self.asked: list[list[str]] = []

    async def manifests(self):
        return [{"shapes": {n: list(t.shape) for n, t in self.parameters.items()},
                 "param_dtypes": ["torch.float32"],
                 "buffers": {n: list(t.shape) for n, t in self.buffers.items()}}] * 2

    async def export_parameters(self, names):
        self.asked.append(list(names))
        whole = {**self.parameters, **self.buffers}
        return {name: whole[name] for name in names}


def _tiny_model():
    from transformers import NemotronHForCausalLM
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    torch.manual_seed(7)
    config = NemotronHConfig(**TINY)
    model = NemotronHForCausalLM(config)
    return model, config


def _base_checkpoint(directory: Path, *, files: int = 3, bias_dtype=torch.float32):
    """A checkpoint in the *loader's* layout (``backbone.``, per-expert tensors),
    written over ``files`` shards; returns the model's FP32 master tensors.

    The checkpoint's bf16 values upcast to FP32 are exactly what the island
    holds after it loaded this checkpoint, so an export of them must reproduce
    these bytes.
    """
    from safetensors.torch import save_file
    from transformers.core_model_loading import revert_weight_conversion

    model, config = _tiny_model()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(config.to_json_string())
    (directory / "tokenizer.json").write_text("{}")
    state = {name: value.detach() for name, value in model.state_dict().items()}
    checkpoint = revert_weight_conversion(model, state)  # upstream: the save path
    buffers = {name for name, _ in model.named_buffers()}
    cast = {name: value.to(bias_dtype if name.endswith("e_score_correction_bias") else torch.bfloat16)
            for name, value in checkpoint.items()}
    names = sorted(cast)
    shards = [names[i::files] for i in range(files)]
    weight_map = {}
    for index, group in enumerate(shards, start=1):
        file_name = f"model-{index:05d}-of-{files:05d}.safetensors"
        save_file({name: cast[name].contiguous() for name in group}, str(directory / file_name),
                  metadata={"format": "pt"})
        weight_map.update({name: file_name for name in group})
    (directory / INDEX_FILE).write_text(json.dumps(
        {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in cast.values())},
         "weight_map": dict(sorted(weight_map.items()))}, indent=2))
    masters = {}  # the model's tensors as the island holds them, from the written bytes
    reloaded = revert_weight_conversion(model, {name: value.to(torch.float32) for name, value in state.items()})
    del reloaded
    forward = base_export(directory, {name: tuple(value.shape) for name, value in state.items()})[0]
    assert forward is not None, "the tiny checkpoint must map back to the model"
    for model_name, sources in forward.key_map.targets().items():
        if forward.key_map.keys[sources[0]].index is None:
            masters[model_name] = cast[sources[0]].to(torch.float32)
        else:
            masters[model_name] = torch.stack([cast[name].to(torch.float32) for name in sources])
    parameters = {name: value for name, value in masters.items() if name not in buffers}
    return parameters, {name: value for name, value in masters.items() if name in buffers}


def _bytes(path: Path) -> bytes:
    return path.read_bytes()


def test_the_export_reproduces_the_base_checkpoints_files_byte_for_byte(tmp_path):
    base = tmp_path / "base"
    parameters, buffers = _base_checkpoint(base)
    group = _Group(parameters, buffers)
    writer = VerlHfWriter(group, base, run=_run)
    out = tmp_path / "round-1"
    writer(1, out)
    assert writer.last_layout["layout"] == "base", writer.last_layout
    shards = sorted(p.name for p in base.iterdir() if p.name.endswith(".safetensors"))
    assert len(shards) == 3
    for name in shards:
        assert _bytes(out / name) == _bytes(base / name), name
    assert _bytes(out / INDEX_FILE) == _bytes(base / INDEX_FILE)
    assert (out / "config.json").is_file() and (out / "tokenizer.json").is_file()
    assert (out / ".complete").is_file()


def test_the_export_asks_for_every_model_tensor_once_per_file_group(tmp_path):
    """The reads are the files' needs: a tensor two files share is read once
    (the writer keeps it until its last file), and nothing is read twice."""
    base = tmp_path / "base"
    parameters, buffers = _base_checkpoint(base)
    group = _Group(parameters, buffers)
    VerlHfWriter(group, base, run=_run)(1, tmp_path / "out")
    asked = [name for batch in group.asked for name in batch]
    assert sorted(asked) == sorted(set(asked))  # never fetched twice
    assert set(asked) == set(parameters) | set(buffers)  # parameters AND persistent buffers


def test_a_changed_weight_changes_only_its_own_file(tmp_path):
    """What makes the delta small: one updated tensor leaves the other shards
    identical to the base."""
    base = tmp_path / "base"
    parameters, buffers = _base_checkpoint(base)
    index = json.loads((base / INDEX_FILE).read_text())["weight_map"]
    name = "model.embeddings.weight"
    parameters[name] = parameters[name] + 1.0
    VerlHfWriter(_Group(parameters, buffers), base, run=_run)(1, tmp_path / "out")
    changed = index["backbone.embedding.weight"]  # two chained renames: backbone., embedding.weight
    for file_name in sorted(set(index.values())):
        same = _bytes(tmp_path / "out" / file_name) == _bytes(base / file_name)
        assert same is (file_name != changed), file_name


def test_a_stacked_expert_tensor_is_cut_back_into_per_expert_tensors(tmp_path):
    from safetensors.torch import load_file

    base = tmp_path / "base"
    parameters, buffers = _base_checkpoint(base)
    stacked = "model.layers.2.mixer.experts.up_proj"
    assert parameters[stacked].ndim == 3 and parameters[stacked].shape[0] == TINY["n_routed_experts"]
    parameters[stacked] = parameters[stacked] + torch.arange(
        parameters[stacked].shape[0], dtype=torch.float32).reshape(-1, 1, 1)
    out = tmp_path / "out"
    VerlHfWriter(_Group(parameters, buffers), base, run=_run)(1, out)
    index = json.loads((base / INDEX_FILE).read_text())["weight_map"]
    for expert in range(TINY["n_routed_experts"]):
        name = f"backbone.layers.2.mixer.experts.{expert}.up_proj.weight"
        written = load_file(str(out / index[name]))[name]
        assert torch.equal(written, parameters[stacked][expert].to(torch.bfloat16)), name


def test_the_router_bias_keeps_the_bases_f32_dtype(tmp_path):
    """The export casts per tensor, not per file: an f32 buffer stays f32, so the
    file keeps the base's size and a delta against the base stays possible."""
    from safetensors.torch import load_file

    base = tmp_path / "base"
    parameters, buffers = _base_checkpoint(base)
    out = tmp_path / "out"
    VerlHfWriter(_Group(parameters, buffers), base, run=_run)(1, out)
    index = json.loads((base / INDEX_FILE).read_text())["weight_map"]
    name = "backbone.layers.2.mixer.gate.e_score_correction_bias"
    assert load_file(str(out / index[name]))[name].dtype == torch.float32
    weight = "backbone.embedding.weight"
    assert load_file(str(out / index[weight]))[weight].dtype == torch.bfloat16


def test_a_base_whose_layout_cannot_be_reproduced_falls_back_and_says_why(tmp_path):
    base = tmp_path / "base"
    parameters, buffers = _base_checkpoint(base)
    extra = dict(parameters)
    extra["model.not_in_the_base"] = torch.zeros(4)
    plan, why = base_export(base, {n: tuple(t.shape) for n, t in {**extra, **buffers}.items()})
    assert plan is None and "not in the base" in why
    writer = VerlHfWriter(_Group(extra, buffers), base, run=_run)
    out = tmp_path / "out"
    writer(1, out)
    assert writer.last_layout["layout"] == "regrouped" and "not in the base" in writer.last_layout["why"]
    from safetensors.torch import load_file

    # the tiny model fits one shard, so the fallback writes model.safetensors under the model's names
    written = load_file(str(out / "model.safetensors"))
    assert "model.not_in_the_base" in written and "model.embeddings.weight" in written


def test_base_export_refusals(tmp_path):
    base = tmp_path / "no-weights"
    base.mkdir()
    assert base_export(base, {"a": (2,)})[0] is None
    (base / INDEX_FILE).write_text("{}")
    plan, why = base_export(base, {"a": (2,)})
    assert plan is None and "weight_map" in why
    (base / INDEX_FILE).write_text(json.dumps({"weight_map": {"a": "model.bin"}}))
    assert "not safetensors" in base_export(base, {"a": (2,)})[1]
    (base / INDEX_FILE).write_text(json.dumps({"weight_map": {"a": "model-00001.safetensors"}}))
    assert "not there" in base_export(base, {"a": (2,)})[1]


def test_a_checkpoint_without_a_conversion_table_maps_to_itself(tmp_path):
    """A plain checkpoint (the names are the state names) needs no model build:
    the identity map keeps the files, which is what every non-Nemotron island does."""
    from safetensors.torch import save_file

    base = tmp_path / "plain"
    base.mkdir()
    (base / "config.json").write_text('{"torch_dtype": "bfloat16"}')
    tensors = {"a.weight": torch.zeros(4, 2, dtype=torch.bfloat16), "b.weight": torch.ones(3, dtype=torch.bfloat16)}
    save_file(tensors, str(base / "model.safetensors"), metadata={"format": "pt"})

    def refuse(_path):
        raise AssertionError("an identity layout must not build the model")

    plan, _why = base_export(base, {name: tuple(t.shape) for name, t in tensors.items()}, conversion=refuse)
    assert plan is not None and plan.files == ("model.safetensors",) and plan.index is False
    masters = {name: value.to(torch.float32) for name, value in tensors.items()}
    out = tmp_path / "out"
    VerlHfWriter(_Group(masters, {}), base, run=_run)(1, out)
    assert _bytes(out / "model.safetensors") == _bytes(base / "model.safetensors")
    assert not (out / INDEX_FILE).exists()  # the base has none either


def test_write_like_refuses_a_tensor_that_is_not_the_headers(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "f.safetensors"
    save_file({"a": torch.zeros(4, dtype=torch.bfloat16)}, str(path), metadata={"format": "pt"})
    from yeto.rl.adapters.verl.hf_checkpoint_keys import read_safetensors_header

    header, _parsed = read_safetensors_header(path)
    with pytest.raises(ValueError, match="is torch.float32"):
        write_like(header, {"a": torch.zeros(4)}, tmp_path / "x")
    with pytest.raises(ValueError, match="shape"):
        write_like(header, {"a": torch.zeros(5, dtype=torch.bfloat16)}, tmp_path / "x")
    with pytest.raises(ValueError, match="the header names"):
        write_like(header, {"b": torch.zeros(4, dtype=torch.bfloat16)}, tmp_path / "x")
    assert write_like(header, {"a": torch.zeros(4, dtype=torch.bfloat16)}, tmp_path / "y") == (
        8 + len(header) + 8)


def test_the_key_map_refuses_a_conversion_it_cannot_reverse():
    from transformers.core_model_loading import Transpose, WeightConverter

    converter = WeightConverter(source_patterns=["q.weight"], target_patterns="q.weight",
                                operations=[Transpose()])
    with pytest.raises(CheckpointKeyError, match="unsupported checkpoint conversion"):
        checkpoint_key_map(["q.weight"], ["q.weight"], rules=[converter], prefix=None)


def test_model_key_map_names_the_stacked_sources_in_loading_order():
    model, _config = _tiny_model()
    checkpoint = ["backbone.layers.2.mixer.experts.%d.up_proj.weight" % e for e in (0, 2, 1, 3)]
    key_map = model_key_map(model, checkpoint)
    target = "model.layers.2.mixer.experts.up_proj"
    assert key_map.targets()[target] == tuple(sorted(checkpoint))  # natural order, not the input's
    assert [key_map.keys[name].index for name in sorted(checkpoint)] == [0, 1, 2, 3]
    assert key_map.unexpected == ()


# ----------------------------------------------------------------- delta releases of a verl trainer
def test_a_buffer_trainers_first_release_is_a_delta_against_the_stores_base(tmp_path):
    """S27 G-EQ3: the base checkpoint both sides hold is already a release in the
    store, so the first cross-cloud update sends the changed words, not 63 GB."""
    from yeto.rl.engine.policy_release import PolicyReleaseStore
    from yeto.rl.engine.resume import CheckpointStore

    from yeto.rl.adapters.verl.buffer_ports import delta_chain_for
    from yeto.rl.adapters.verl.merged_release import write_full_checkpoint

    base = tmp_path / "base"
    parameters, buffers = _base_checkpoint(base, files=2)
    store = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))

    def write_v0(directory: Path) -> None:
        for path in sorted(base.iterdir()):
            if path.is_file():
                (directory / path.name).write_bytes(path.read_bytes())

    v0 = store.release(0, "base-hash", write_v0, buffer_selected=0, buffer_shards={"modal-0": 0})
    assert v0 is not None and v0.path == "r1-v0"
    events: list[dict] = []
    chain = delta_chain_for(store, 2, emit=lambda name, **fields: events.append({"event": name, **fields}))
    assert events and events[-1]["adopted"] is True and events[-1]["base_version"] == 0
    assert chain.base() is not None and chain.base().release.path == "r1-v0"

    # a step moved part of one tensor: at lr 1e-6 a real step moves about 1 % of
    # the bf16 words (release_delta module doc), so most of the file is unchanged
    name = "model.embeddings.weight"
    moved = parameters[name].clone()
    moved[:64] += 0.5
    parameters[name] = moved
    staged = tmp_path / "staging" / "v1"
    staged.parent.mkdir(parents=True)
    # the publisher's own writer path: it leaves no .complete marker in the release
    write_full_checkpoint(VerlHfWriter(_Group(parameters, buffers), base, run=_run), staged)
    release = store.release_staged(1, "v1-hash", staged, position=lambda: (1, {"modal-0": 1}),
                                   delta_base=chain.base(), digest_full=True)
    assert release is not None and release.delta is not None
    assert release.delta["base_path"] == "r1-v0" and 0.0 < release.delta["ratio"] < 1.0
    # the delta is smaller than the weights it stands for
    assert sum(release.files.values()) < sum(v0.files.values())
    # and a sampler rebuilds the release from its own copy of the base
    from yeto.rl.engine.release_delta import FileEntry, apply_directory

    entries = {n: FileEntry(**e) for n, e in release.delta["files"].items()}
    rebuilt = tmp_path / "rebuilt"
    apply_directory(tmp_path / "store" / "policy-releases" / "r1-v0",
                    [(tmp_path / "store" / "policy-releases" / release.path, entries)], rebuilt)
    for file_name in sorted(p.name for p in staged.iterdir() if p.name.endswith(".safetensors")):
        assert _bytes(rebuilt / file_name) == _bytes(staged / file_name), file_name


def test_delta_releases_need_a_base_whose_files_are_readable(tmp_path):
    from yeto.rl.engine.policy_release import PolicyReleaseStore
    from yeto.rl.engine.resume import CheckpointStore

    from yeto.rl.adapters.verl.buffer_ports import delta_chain_for

    store = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    events: list[dict] = []
    emit = lambda name, **fields: events.append(fields)  # noqa: E731
    assert delta_chain_for(store, 0, emit=emit) is None  # the flag is off
    chain = delta_chain_for(store, 2, emit=emit)  # an empty store: the first release is full
    assert chain is not None and chain.base() is None
    assert events[-1]["adopted"] is False and "no full release" in events[-1]["reason"]


def test_the_export_gives_the_workers_cached_device_blocks_back(tmp_path):
    """S27 G-EQ3: the gathers leave large blocks in the device allocator, and the
    step after a release needs that room (npu1: the step after the first release
    died asking for 9.00 GiB)."""
    base = tmp_path / "base"
    parameters, buffers = _base_checkpoint(base, files=2)
    group = _Group(parameters, buffers)
    calls = []

    async def release_memory():
        calls.append(True)
        return [{"before": {"allocated": 1, "reserved": 40}, "after": {"allocated": 1, "reserved": 8}},
                {"before": {"allocated": 1, "reserved": 36}, "after": {"allocated": 1, "reserved": 4}}]

    group.release_memory = release_memory
    writer = VerlHfWriter(group, base, run=_run)
    writer(1, tmp_path / "out")
    assert calls == [True]
    assert writer.last_layout["freed"] == {"reserved_before": 40, "reserved_after": 8, "workers": 2}


def test_an_export_without_a_release_memory_port_still_works(tmp_path):
    base = tmp_path / "base"
    parameters, buffers = _base_checkpoint(base, files=2)
    writer = VerlHfWriter(_Group(parameters, buffers), base, run=_run)
    writer(1, tmp_path / "out")
    assert writer.last_layout["freed"] is None
