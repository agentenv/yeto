"""K3.1 MXFP4 releases on sglang-server: native weight_packed names pass the
preflight, the file-level expert update plan, the shuffle probe and the fallback
to a full reload."""
from __future__ import annotations

import json

import numpy as np
import pytest

from yeto.rl.adapters.sampler_engines import k31_release as kr
from yeto.rl.adapters.sampler_engines.sglang_server import SGLangOptions, SGLangServerEngine
from yeto.rl.engine.release_delta import write_changes
from yeto.rl.engine.sampler_engine import SamplerEngineError, SamplingParams

from test_rl_k31_engine_sampler_r3 import _Http, _Process, _config

LAYERS, EXPERTS = 2, 4


def _expert_names(layer, expert):
    return [f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{p}.{k}" for p in ("w1", "w2", "w3")
            for k in kr.EXPERT_KINDS]


def _release(root, *, dense_in=None, missing_index=False):
    """A K3.1-shaped release: one safetensors file per (layer, pair of experts), the
    dense weights in model-dense.safetensors, config.json and the index."""
    root.mkdir(parents=True, exist_ok=True)
    index = {}
    for layer in range(LAYERS):
        for half in range(EXPERTS // 2):
            file = f"model-l{layer}-e{half}.safetensors"
            (root / file).write_bytes(b"x")
            for expert in (2 * half, 2 * half + 1):
                for name in _expert_names(layer, expert):
                    index[name] = file
    (root / "model-dense.safetensors").write_bytes(b"d")
    index["model.embed_tokens.weight"] = "model-dense.safetensors"
    index["model.layers.0.block_sparse_moe.gate.weight"] = dense_in or "model-dense.safetensors"
    if not missing_index:
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    (root / "config.json").write_text(json.dumps({"architectures": ["KimiK31ForConditionalGeneration"],
                                                  "num_hidden_layers": LAYERS, "first_k_dense_replace": 0,
                                                  "n_routed_experts": EXPERTS}))
    return index


class _Tensor:
    def __init__(self, name, nbytes=8):
        self.name, self.nbytes = name, nbytes

    def numel(self):
        return self.nbytes

    def element_size(self):
        return 1


def _read(path, names):
    return [(n, _Tensor(n)) for n in names]


def test_expert_tensor_names_and_native_packed_architectures(tmp_path):
    assert kr.parse_expert_tensor("model.layers.12.block_sparse_moe.experts.895.w3.weight_scale") == (12, 895, "w3", "weight_scale")
    assert kr.parse_expert_tensor("model.layers.1.block_sparse_moe.gate.weight") is None
    # the Kimi-K3 checkpoint (and so the trainer's release) prefixes language_model.
    assert kr.parse_expert_tensor("language_model.model.layers.3.block_sparse_moe.experts.7.w1.weight_packed") == (
        3, 7, "w1", "weight_packed")
    assert kr.parse_expert_tensor("model.layers.1.mlp.experts.0.w1.weight_packed") is None  # the SGLang-side name
    _release(tmp_path / "r")
    assert kr.native_packed_names(tmp_path / "r")
    (tmp_path / "r" / "config.json").write_text(json.dumps({"architectures": ["DeepseekV3ForCausalLM"]}))
    assert not kr.native_packed_names(tmp_path / "r") and kr.native_packed_names(tmp_path / "none") is False


def test_preflight_accepts_weight_packed_of_a_k31_release_without_the_rename_patch(tmp_path):
    _release(tmp_path / "base")
    engine = SGLangServerEngine(_config(tmp_path), spawn=lambda a, e: _Process(), http=_Http({("GET", "/health"): {}}),
                                sleep=lambda s: None)
    engine.start(None, 0)  # no "weight_packed" refusal
    (tmp_path / "base" / "config.json").write_text(json.dumps({"architectures": ["DeepseekV3ForCausalLM"]}))
    with pytest.raises(SamplerEngineError, match="weight_packed"):
        engine.start(None, 1)


def test_expert_update_plan_takes_only_expert_files(tmp_path):
    index = _release(tmp_path / "r")
    plan = kr.expert_update_plan(tmp_path / "r", ["model-l1-e0.safetensors"], index=index)
    assert plan.files == ("model-l1-e0.safetensors",) and plan.experts == {1: (0, 1)}
    assert plan.tensor_count == 12 and plan.expert_count == 2
    with pytest.raises(kr.FileSparseUnavailable, match="non_expert_tensor_changed"):
        kr.expert_update_plan(tmp_path / "r", ["model-dense.safetensors"], index=index)
    with pytest.raises(kr.FileSparseUnavailable, match="file_not_in_index"):
        kr.expert_update_plan(tmp_path / "r", ["other.safetensors"], index=index)
    with pytest.raises(kr.FileSparseUnavailable, match="non_weight_file_changed"):
        kr.expert_update_plan(tmp_path / "r", ["config.json"], index=index)
    with pytest.raises(kr.FileSparseUnavailable, match="no_changes"):
        kr.expert_update_plan(tmp_path / "r", [], index=index)
    with pytest.raises(kr.FileSparseUnavailable, match="no_index"):
        kr.expert_update_plan(tmp_path / "r", ["model-l1-e0.safetensors"], index={})
    (tmp_path / "r" / "model-l1-e0.safetensors").unlink()
    with pytest.raises(kr.FileSparseUnavailable, match="file_missing"):
        kr.expert_update_plan(tmp_path / "r", ["model-l1-e0.safetensors"], index=index)
    # the probe plan: the lowest expert, both kinds of every projection
    probe = kr.shuffle_probe_plan(tmp_path / "r", index=index)
    assert probe.experts == {0: (0,)} and probe.tensor_count == 6 and probe.files == ("model-l0-e0.safetensors",)
    with pytest.raises(kr.FileSparseUnavailable, match="no_expert_tensors"):
        kr.shuffle_probe_plan(tmp_path / "r", index={"a": "b"})


def test_batches_split_on_bytes_but_never_a_tensor(tmp_path):
    index = _release(tmp_path / "r")
    plan = kr.expert_update_plan(tmp_path / "r", ["model-l0-e0.safetensors", "model-l1-e1.safetensors"], index=index)
    chunks = list(kr.batches(plan, _read, max_bytes=40))  # 5 tensors of 8 bytes per batch
    assert [len(c) for c in chunks] == [5, 5, 5, 5, 4] and sum(len(c) for c in chunks) == 24
    assert chunks[0][0][0].startswith("model.layers.0.") and chunks[-1][-1][0].startswith("model.layers.1.")
    assert len(list(kr.batches(plan, _read, max_bytes=1))) == 24  # one oversized tensor per batch


class _Server:
    """The /generate and weight endpoints of a fake SGLang whose logits depend on a
    'transform' flag: when it transforms weights after loading, a raw tensor write
    changes the scores (the aiter shuffle / Marlin repack risk)."""

    def __init__(self, *, transforms: bool):
        self.transforms = transforms
        self.shift = 0.0
        self.tensor_calls: list[dict] = []
        self.disk_calls: list[dict] = []
        self.pauses: list[str] = []

    def routes(self):
        return {("GET", "/health"): {}, ("POST", "/generate"): self.generate,
                ("POST", "/update_weights_from_tensor"): self.tensor, ("POST", "/update_weights_from_disk"): self.disk,
                ("POST", "/pause_generation"): lambda b: self.pauses.append("pause") or {},
                ("POST", "/continue_generation"): lambda b: self.pauses.append("continue") or {}}

    def generate(self, body):
        texts = body["text"] if isinstance(body["text"], list) else [body["text"]]
        if body["sampling_params"]["max_new_tokens"] == 0:
            return [{"meta_info": {"input_token_logprobs": [[None, 1, None]] + [[-0.5 - self.shift, 2, None]] * (len(t) - 1)}}
                    for t in texts]
        return [{"text": "x", "meta_info": {"output_token_logprobs": [[-0.1, 7, None]], "finish_reason": {"type": "stop"}}}
                for _ in texts]

    def tensor(self, body):
        self.tensor_calls.append(body)
        if self.transforms:
            self.shift += 0.1
        return {"success": True}

    def disk(self, body):
        self.disk_calls.append(body)
        self.shift = 0.0
        return {"success": True}


def _engine(tmp_path, server, **options):
    _release(tmp_path / "base")
    return SGLangServerEngine(_config(tmp_path, in_place_reload=True),
                              SGLangOptions(expert_file_update="auto", **options), read_tensors=_read,
                              serialize=lambda named: f"{len(named)} tensors", spawn=lambda a, e: _Process(),
                              http=_Http(server.routes()), sleep=lambda s: None)


def _delta_release(tmp_path, name, files, *, base="base"):
    release = tmp_path / name
    _release(release)
    write_changes(release, base_dir=tmp_path / base, files={f: 0 for f in files})
    return release


def test_probe_passes_then_changed_expert_files_go_as_tensors_and_other_changes_reload(tmp_path):
    server = _Server(transforms=False)
    engine = _engine(tmp_path, server)
    report = engine.start(None, 0)
    prepare = report.detail["expert_update_prepare"]
    assert prepare["enabled"] and prepare["max_delta_logprob"] == 0.0 and prepare["tensors"] == 6
    assert len(server.tensor_calls) == 1 and server.disk_calls == []  # the probe wrote one expert, no repair
    assert "weight_version" not in server.tensor_calls[0]
    release = _delta_release(tmp_path, "r1", ["model-l1-e0.safetensors", "model-l0-e1.safetensors"])
    load = engine.load(release, 1)
    assert load.reload_mode == "in_place" and load.detail["expert_update"] is True
    assert load.detail["experts"] == 4 and load.detail["files"] == 2 and load.detail["batches"] >= 1
    assert server.tensor_calls[-1]["weight_version"] == "1" and server.tensor_calls[-1]["flush_cache"] is False
    assert len(server.tensor_calls[-1]["serialized_named_tensors"]) == 8  # one payload per TP rank
    assert server.disk_calls == [] and engine.served_version() == 1
    # a release that also changes a dense file: nothing sent, full reload
    dense = _delta_release(tmp_path, "r2", ["model-dense.safetensors"], base="r1")
    sent = len(server.tensor_calls)
    load = engine.load(dense, 2)
    assert load.detail["expert_update_fallback"] == "non_expert_tensor_changed"
    assert len(server.tensor_calls) == sent and server.disk_calls[-1]["model_path"] == str(dense)
    # a full release (no change manifest) reloads from disk too
    full = tmp_path / "r3"
    _release(full)
    assert engine.load(full, 3).detail["expert_update_fallback"] == "no_change_manifest"
    # a release rebuilt from another base
    other = _delta_release(tmp_path, "r4", ["model-l1-e0.safetensors"], base="r1")
    assert engine.load(other, 4).detail["expert_update_fallback"] == "base_mismatch"


def test_probe_detects_a_post_load_transform_and_turns_the_path_off(tmp_path):
    server = _Server(transforms=True)
    engine = _engine(tmp_path, server)
    report = engine.start(None, 0)
    prepare = report.detail["expert_update_prepare"]
    assert prepare["enabled"] is False and "transforms expert weights" in prepare["error"]
    assert prepare["max_delta_logprob"] == pytest.approx(0.1)
    assert server.disk_calls and server.disk_calls[0]["model_path"] == str(tmp_path / "base")  # repaired
    assert engine.expert_update_off == "probe_failed"
    release = _delta_release(tmp_path, "r1", ["model-l1-e0.safetensors"])
    load = engine.load(release, 1)
    assert load.detail["expert_update_fallback"] == "probe_failed" and server.disk_calls[-1]["model_path"] == str(release)
    assert len(server.tensor_calls) == 1  # only the probe's write


def test_probe_can_be_skipped_and_a_failed_batch_repairs_from_disk(tmp_path):
    server = _Server(transforms=False)
    engine = _engine(tmp_path, server, expert_update_probe=False, live_update=True)
    report = engine.start(None, 0)
    assert report.detail["expert_update_prepare"] == {"probe": False, "enabled": True, "prepare_s": 0.0}
    assert server.tensor_calls == []
    release = _delta_release(tmp_path, "r1", ["model-l1-e0.safetensors"])
    def failing(body):
        server.tensor_calls.append(body)
        return {"success": False, "message": "shape mismatch"}
    engine._http.routes[("POST", "/update_weights_from_tensor")] = failing
    load = engine.load(release, 1)
    assert load.detail["expert_update"] is False and "shape mismatch" in load.detail["expert_update_repair"]
    assert load.detail["live"] is True and server.pauses == ["pause", "continue", "pause", "continue"]
    assert server.disk_calls[-1]["model_path"] == str(release) and server.disk_calls[-1]["flush_cache"] is False


def test_expert_file_update_options_are_checked(tmp_path):
    with pytest.raises(SamplerEngineError, match="in_place_reload"):
        SGLangServerEngine(_config(tmp_path), SGLangOptions(expert_file_update="auto"))
    with pytest.raises(SamplerEngineError, match="expert_file_update"):
        SGLangServerEngine(_config(tmp_path, in_place_reload=True), SGLangOptions(expert_file_update="yes"))


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_during_the_expert_probe_is_not_swallowed(tmp_path, interrupt):
    """``except BaseException`` around the probe write used to turn Ctrl-C / SystemExit
    into a SamplerEngineError, which the probe then recorded as ``probe_failed``: the
    start went on as if nothing happened."""
    server = _Server(transforms=False)

    def tensor(body):
        raise interrupt()

    server.tensor = tensor
    engine = _engine(tmp_path, server)
    with pytest.raises(interrupt):
        engine.start(None, 0)
