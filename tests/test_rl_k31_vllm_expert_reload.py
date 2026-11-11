"""The vLLM expert reload of a partial-experts overlay: the CPU plan, the byte
fingerprint, and the adapter path (pause -> worker-extension RPC with the
overlay's release files -> resume; a non-overlay release restarts)."""
from __future__ import annotations

import json

import pytest

from yeto.rl.adapters.sampler_engines import vllm_expert_reload as ver
from yeto.rl.adapters.sampler_engines.vllm_server import VllmOptions, VllmServerEngine
from yeto.rl.engine.release_overlay import overlay_release

from test_rl_k31_engine_sampler_r3 import _Http, _Process, _config

ROCM_CARDS = json.dumps([{"gpu": i} for i in range(8)])
PIECES = [(p, k) for p in ("w1", "w2", "w3") for k in ("weight_packed", "weight_scale")]


def _name(layer, expert, proj, kind):
    return f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.{kind}"


def _expert_names(layer, expert):
    return [_name(layer, expert, p, k) for p, k in PIECES]


def _write_index(directory, weight_map):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ver.INDEX_FILE).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    for file in set(weight_map.values()):
        (directory / file).write_bytes(b"")


def _base_and_release(tmp_path, base_dir):
    base_map = {n: "model-00001.safetensors" for e in range(4) for n in _expert_names(1, e)}
    base_map["language_model.model.embed_tokens.weight"] = "model-00002.safetensors"
    _write_index(base_dir, base_map)
    release = tmp_path / "rel"
    _write_index(release, {n: "model-00001.safetensors" for e in (0, 2) for n in _expert_names(1, e)})
    (release / "config.json").write_text((base_dir / "config.json").read_text())
    return release


# ------------------------------------------------------------------ plan
def test_plan_takes_the_overlay_release_files_and_groups_whole_experts(tmp_path):
    base = tmp_path / "base"
    (base / "config.json").parent.mkdir(parents=True)
    (base / "config.json").write_text("{}")
    release = _base_and_release(tmp_path, base)
    overlay = overlay_release(base, release, tmp_path / "ov")
    tensors = ver.released_tensors(overlay)
    assert set(tensors.values()) == {"release-model-00001.safetensors"}
    plan = ver.group_by_layer(tensors)
    assert list(plan) == [1] and plan[1]["experts"] == [0, 2] and len(plan[1]["tensors"]) == 12
    only = ver.released_tensors(overlay, ["model-00002.safetensors"])
    assert list(only) == ["language_model.model.embed_tokens.weight"]


def test_plan_refuses_non_expert_tensors_and_partial_experts():
    with pytest.raises(ver.ExpertReloadError, match="not a routed-expert"):
        ver.group_by_layer({"language_model.model.embed_tokens.weight": "f"})
    with pytest.raises(ver.ExpertReloadError, match="partly"):
        ver.group_by_layer({n: "f" for n in _expert_names(3, 7)[:5]})


# ------------------------------------------------------------------ fingerprint
def test_fingerprint_sees_every_byte_and_its_position():
    torch = pytest.importorskip("torch")
    x = torch.arange(1000, dtype=torch.uint8)
    same = ver.fingerprint(x.clone())
    assert ver.fingerprint(x) == same
    y = x.clone()
    y[517] ^= 1
    assert ver.fingerprint(y) != same
    swapped = x.clone()
    swapped[[3, 900]] = swapped[[900, 3]]
    assert ver.fingerprint(swapped) != same
    assert ver.fingerprint(x.view(10, 100)) == same  # bytes, not shape (numel kept)


def test_rows_view_writes_expert_slots_of_an_fp4_like_byte_tensor():
    torch = pytest.importorskip("torch")
    served = torch.zeros(6, 4, 3, dtype=torch.uint8)
    rows = ver._rows_u8(served, 6)
    rows.index_copy_(0, torch.tensor([1, 4]), torch.full((2, 12), 9, dtype=torch.uint8))
    assert served[1].eq(9).all() and served[4].eq(9).all() and served[[0, 2, 3, 5]].eq(0).all()
    with pytest.raises(ver.ExpertReloadError, match="split"):
        ver._rows_u8(torch.zeros(7, dtype=torch.uint8), 2)


# ------------------------------------------------------------------ adapter
def _engine(tmp_path, routes):
    http = _Http({("GET", "/health"): {}, **routes})
    config = _config(tmp_path, card_model="MI355X", device_family="amd", in_place_reload=True)
    engine = VllmServerEngine(config, VllmOptions(platform="rocm", tp=8, expert_reload=True),
                              spawn=lambda a, e: _Process(), http=http, sleep=lambda s: None,
                              run=lambda argv: ROCM_CARDS)
    return engine, http


def test_overlay_reloads_only_the_release_files_through_the_extension(tmp_path):
    calls = []

    def rpc(body):
        calls.append(body)
        return {"results": [{"seconds": 3.5, "experts": 2}] * 8}

    routes = {("POST", "/pause?mode=wait"): {}, ("POST", "/collective_rpc"): rpc, ("POST", "/resume"): {}}
    engine, http = _engine(tmp_path, routes)
    assert ver.EXTENSION in engine.argv(engine.config.base_dir)
    release = _base_and_release(tmp_path, engine.config.base_dir)
    engine.start(None, 0)
    overlay = overlay_release(engine.config.base_dir, release, tmp_path / "ov")
    report = engine.load(overlay, 1)
    assert report.reload_mode == "in_place" and report.detail["reload"].endswith(ver.RELOAD_METHOD)
    assert report.detail["experts"] == 2 and report.detail["rank_seconds"] == 3.5
    assert [p for m, p, b in http.calls if m == "POST"] == ["/pause?mode=wait", "/collective_rpc", "/resume"]
    assert calls[0]["method"] == ver.RELOAD_METHOD
    assert json.loads(calls[0]["kwargs"]["files"]) == ["release-model-00001.safetensors"]
    # a second overlay of the same base and selection reloads in place again
    overlay2 = overlay_release(engine.config.base_dir, release, tmp_path / "ov2")
    assert engine.load(overlay2, 2).detail["reload"].endswith(ver.RELOAD_METHOD)


def test_a_release_that_is_not_an_overlay_restarts_and_resumes_after_a_failed_rpc(tmp_path):
    routes = {("POST", "/pause?mode=wait"): {}, ("POST", "/resume"): {}}
    engine, http = _engine(tmp_path, routes)
    release = _base_and_release(tmp_path, engine.config.base_dir)
    engine.start(None, 0)
    report = engine.load(release, 1)  # a plain release directory: no release-* files
    assert report.detail["reload"] == "restart" and report.detail["expert_reload_off"] == "not_an_overlay"
    overlay = overlay_release(engine.config.base_dir, release, tmp_path / "ov")
    engine._served_dir = engine.config.base_dir
    with pytest.raises(Exception, match="collective_rpc"):
        engine.load(overlay, 2)
    assert [p for m, p, b in http.calls if m == "POST"][-1] == "/resume"


def test_reload_hands_freed_staging_blocks_back_to_the_device():
    calls = []

    class _Cuda:
        def synchronize(self):
            calls.append("synchronize")

        def empty_cache(self):
            calls.append("empty_cache")

    class _Torch:
        cuda = _Cuda()

    ver.release_cached_memory(_Torch())
    assert calls == ["synchronize", "empty_cache"]
