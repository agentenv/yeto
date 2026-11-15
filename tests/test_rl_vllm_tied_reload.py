"""In-place reload of a model with tied word embeddings (vllm_tied_reload) on
every vllm-server line: release classification, the path plan, the row check,
the worker extension's sync on CPU tensors, and the engine's call sequence."""

from __future__ import annotations

import json
import sys
import types

import pytest
import torch
from safetensors.torch import save_file

from yeto.rl.adapters.sampler_engines import vllm_tied_reload as tvr
from yeto.rl.adapters.sampler_engines.vllm_server import VllmOptions, VllmServerEngine
from yeto.rl.engine.sampler_engine import SamplerEngineError

from test_rl_k31_engine_sampler_r3 import _Http, _Process, _config

VOCAB, HIDDEN = 40, 6


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB, HIDDEN, generator=g).to(torch.bfloat16)


def _write(path, *, tied=True, head="equal", seed=0, sharded=False):
    """A tiny release: head in {"equal", "absent", "distinct"}."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"],
                                                  "tie_word_embeddings": tied}))
    embed = _weights(seed)
    sd = {"model.embed_tokens.weight": embed, "model.norm.weight": torch.ones(HIDDEN, dtype=torch.bfloat16)}
    if head == "equal":
        sd["lm_head.weight"] = embed.clone()
    elif head == "distinct":
        sd["lm_head.weight"] = (embed.float() * 1.02).to(torch.bfloat16)
    if sharded:
        save_file({"model.embed_tokens.weight": sd.pop("model.embed_tokens.weight")}, str(path / "a.safetensors"))
        save_file(sd, str(path / "b.safetensors"))
        names = {"model.embed_tokens.weight": "a.safetensors", **{k: "b.safetensors" for k in sd}}
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": names}))
    else:
        save_file(sd, str(path / "model.safetensors"))
    return path


# ------------------------------------------------------------------ classification and plan
@pytest.mark.parametrize("sharded", [False, True])
def test_release_tie_classifies_the_head(tmp_path, sharded):
    for head, layout in (("equal", tvr.SHARED), ("absent", tvr.SHARED), ("distinct", tvr.SPLIT)):
        rel = tvr.release_tie(_write(tmp_path / f"{head}-{sharded}", head=head, sharded=sharded))
        assert rel.config_tied and rel.lm_head == head and rel.cold_layout == layout
    untied = tvr.release_tie(_write(tmp_path / "untied", tied=False, head="distinct"))
    assert not untied.config_tied and untied.cold_layout == tvr.SPLIT


def test_plan_covers_both_served_layouts():
    rel = lambda head: tvr.ReleaseTie(True, head, "e", None if head == tvr.ABSENT else "h")  # noqa: E731
    assert tvr.plan(tvr.SHARED, rel(tvr.EQUAL))[0] == tvr.IN_PLACE
    assert tvr.plan(tvr.SHARED, rel(tvr.ABSENT))[0] == tvr.IN_PLACE
    assert tvr.plan(tvr.SHARED, rel(tvr.DISTINCT))[0] == tvr.RESTART  # one storage, two tensors
    assert tvr.plan(tvr.SPLIT, rel(tvr.ABSENT))[0] == tvr.IN_PLACE_SYNC  # the head would stay stale
    assert tvr.plan(tvr.SPLIT, rel(tvr.EQUAL))[0] == tvr.IN_PLACE
    assert tvr.plan(tvr.SPLIT, rel(tvr.DISTINCT))[0] == tvr.IN_PLACE
    assert tvr.plan(None, rel(tvr.EQUAL))[0] == tvr.RESTART
    assert tvr.plan(tvr.SHARED, tvr.ReleaseTie(True, None))[0] == tvr.RESTART


def test_sample_rows_are_distinct_in_range_and_salted():
    a, b = tvr.sample_rows(151936, 1), tvr.sample_rows(151936, 2)
    assert len(set(a)) == tvr.SAMPLE_ROWS and 0 in a and 151935 in a and a != b
    assert all(0 <= r < 151936 for r in a + b)
    assert tvr.sample_rows(5, 0) == [0, 1, 2, 3, 4]


def test_rows_are_read_bit_exact_from_bf16(tmp_path):
    rel = _write(tmp_path / "r", head="distinct")
    files = tvr._tensor_files(rel)
    want = (_weights(0).float() * 1.02).to(torch.bfloat16).float()
    got = tvr._Tensor.find(files, "lm_head.weight").rows([0, 7, VOCAB - 1])
    assert got == [want[0].tolist(), want[7].tolist(), want[VOCAB - 1].tolist()]


# ------------------------------------------------------------------ verify (the safety net)
def _state(embed, head, rows, *, shared):
    return {"shared": shared, "embed": {str(r): embed[r].float().tolist() for r in rows},
            "lm_head": {str(r): head[r].float().tolist() for r in rows}}


def test_verify_accepts_what_a_cold_start_serves_and_rejects_a_stale_head(tmp_path):
    rows = tvr.sample_rows(VOCAB, 3)
    absent = _write(tmp_path / "absent", head="absent", seed=1)
    new = _weights(1)
    tvr.verify(absent, tvr.release_tie(absent), [_state(new, new, rows, shared=True)], rows)
    with pytest.raises(tvr.TiedReloadError, match="lm_head row"):  # split server, head not synced
        tvr.verify(absent, tvr.release_tie(absent), [_state(new, _weights(0), rows, shared=False)], rows)
    distinct = _write(tmp_path / "distinct", head="distinct", seed=1)
    head = (new.float() * 1.02).to(torch.bfloat16)
    tvr.verify(distinct, tvr.release_tie(distinct), [_state(new, head, rows, shared=False)], rows)
    with pytest.raises(tvr.TiedReloadError):  # a shared server serves the embeddings as the head
        tvr.verify(distinct, tvr.release_tie(distinct), [_state(new, new, rows, shared=True)], rows)


def test_verify_merges_tp_ranks_and_needs_every_row(tmp_path):
    rel = _write(tmp_path / "r", head="equal", seed=2)
    w, rows = _weights(2), tvr.sample_rows(VOCAB, 0)
    lo, hi = [r for r in rows if r < VOCAB // 2], [r for r in rows if r >= VOCAB // 2]
    tvr.verify(rel, tvr.release_tie(rel), [_state(w, w, lo, shared=True), _state(w, w, hi, shared=True)], rows)
    with pytest.raises(tvr.TiedReloadError, match="no rank reported"):
        tvr.verify(rel, tvr.release_tie(rel), [_state(w, w, lo, shared=True)], rows)


# ------------------------------------------------------------------ the worker extension on CPU tensors
class _Embedding(torch.nn.Module):
    def __init__(self, weight, start=0):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.shard_indices = types.SimpleNamespace(org_vocab_start_index=start,
                                                   org_vocab_end_index=start + weight.shape[0])


class _Head(_Embedding):
    pass


@pytest.fixture
def _fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm.model_executor.layers.vocab_parallel_embedding")
    mod.VocabParallelEmbedding, mod.ParallelLMHead = _Embedding, _Head
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)


def _worker(embed, head, *, shared):
    model = torch.nn.Module()
    model.embed = _Embedding(embed)
    model.head = _Head(head)
    if shared:
        model.head.weight = model.embed.weight
    ext = tvr.TiedEmbeddingExtension()
    ext.model_runner = types.SimpleNamespace(get_model=lambda: model)
    return ext, model


def test_extension_syncs_a_split_head_in_its_own_storage(_fake_vllm):
    ext, model = _worker(_weights(1), _weights(0), shared=False)
    ptr = model.head.weight.data_ptr()
    assert ext.yeto_tied_state("[]")["shared"] is False
    assert ext.yeto_tied_sync() == {"synced": True, "shared": False}
    assert torch.equal(model.head.weight, model.embed.weight) and model.head.weight.data_ptr() == ptr
    state = ext.yeto_tied_state(json.dumps([0, 5]))
    assert state["lm_head"]["5"] == _weights(1)[5].float().tolist()


def test_extension_leaves_a_shared_head_alone(_fake_vllm):
    ext, _ = _worker(_weights(1), _weights(1), shared=True)
    assert ext.yeto_tied_state("[]")["shared"] is True
    assert ext.yeto_tied_sync() == {"synced": False, "shared": True}


# ------------------------------------------------------------------ the engine on both lines
class _Server:
    """A fake vLLM server with the tied extension: embeddings shared or split,
    reload_weights as vLLM does it (module doc of vllm_tied_reload)."""

    def __init__(self, release, shared):
        self.shared, self.sync_broken = shared, False
        self.load(release, cold=True)

    def load(self, release, cold=False):
        sd = {}
        for f in release.glob("*.safetensors"):
            from safetensors.torch import load_file  # noqa: PLC0415

            sd.update(load_file(str(f)))
        embed, head = sd["model.embed_tokens.weight"], sd.get("lm_head.weight")
        if cold:
            self.embed, self.head = embed, (embed if head is None else head)
            return
        if self.shared:
            self.embed = self.head = embed  # one storage: embed_tokens is written last (measured)
        else:
            self.embed = embed
            if head is not None:
                self.head = head

    def rpc(self, body, release_of):
        method, kwargs = body["method"], body.get("kwargs") or {}
        if method == "reload_weights":
            self.load(release_of(kwargs["weights_path"]))
            return {"results": [None]}
        if method == tvr.SYNC_METHOD:
            if not self.shared and not self.sync_broken:
                self.head = self.embed.clone()
            return {"results": [{"synced": not self.shared}]}
        if method == tvr.STATE_METHOD:
            rows = json.loads(kwargs["rows"])
            return {"results": [_state(self.embed, self.head, rows, shared=self.shared)]}
        raise AssertionError(method)


def _tied_engine(tmp_path, platform, cold, *, shared):
    server = _Server(cold, shared)
    from pathlib import Path  # noqa: PLC0415

    http = _Http({("GET", "/health"): {}, ("POST", "/pause?mode=wait"): {}, ("POST", "/resume"): {},
                  ("GET", "/version"): {"version": "x"},
                  ("POST", "/collective_rpc"): lambda body: server.rpc(body, Path)})
    if platform == "npu":
        config = _config(tmp_path, base_dir=cold, card_model="910B4", device_family="npu", precision="bf16",
                         quant_scheme="bf16", topology="tp1", max_model_len=2048, in_place_reload=True)
        options = VllmOptions(platform="npu", tp=1, npu_in_place_reload=True)
    else:
        config = _config(tmp_path, base_dir=cold, in_place_reload=True)
        options = VllmOptions(platform=platform, tp=1)
    engine = VllmServerEngine(config, options, spawn=lambda a, e: _Process(), http=http, sleep=lambda s: None)
    return engine, server, http


@pytest.mark.parametrize("platform", ["npu", "cuda", "rocm"])
def test_engine_reloads_tied_releases_in_place_and_verifies(tmp_path, platform, capsys, monkeypatch):
    monkeypatch.setattr(VllmServerEngine, "_check_rocm_cards", lambda self: None)
    cold = _write(tmp_path / "cold", head="equal", seed=0)
    engine, server, http = _tied_engine(tmp_path, platform, cold, shared=True)
    assert "--worker-extension-cls" in engine.argv(cold)
    assert tvr.EXTENSION in engine.argv(cold)
    engine.start(None, 0)
    out = capsys.readouterr().out
    assert "tie_word_embeddings=true" in out and "served embeddings=shared" in out
    for i, head in enumerate(("equal", "absent"), start=1):
        rel = _write(tmp_path / f"r{i}", head=head, seed=i)
        report = engine.load(rel, i)
        assert report.reload_mode == "in_place" and report.detail["tied_check"]["lm_head_rows_equal_release"]
        assert torch.equal(server.head, _weights(i))
    # a head that differs from the embeddings cannot live in one storage: restart
    rel = _write(tmp_path / "r3", head="distinct", seed=3)
    report = engine.load(rel, 3)
    assert report.detail["reload"] == "restart"
    assert report.detail["in_place_off"] == "lm_head_differs_from_embeddings_on_a_shared_server"


def test_engine_syncs_the_head_of_a_split_server_and_fails_loudly_when_stale(tmp_path):
    cold = _write(tmp_path / "cold", head="distinct", seed=0)
    engine, server, http = _tied_engine(tmp_path, "npu", cold, shared=False)
    engine.start(None, 0)
    rel = _write(tmp_path / "r1", head="absent", seed=1)
    report = engine.load(rel, 1)
    assert report.detail["reload"] == "collective_rpc reload_weights + " + tvr.SYNC_METHOD
    assert torch.equal(server.head, _weights(1))
    posts = [p for m, p, b in http.calls if m == "POST"][-5:]
    assert posts == ["/pause?mode=wait", "/collective_rpc", "/collective_rpc", "/collective_rpc", "/resume"]
    server.sync_broken = True  # the safety net: a head that did not follow the embeddings
    with pytest.raises(SamplerEngineError, match="tied-embeddings check failed"):
        engine.load(_write(tmp_path / "r2", head="absent", seed=2), 2)
    assert engine._process is None  # killed, not resumed with a wrong head


def test_untied_model_argv_and_path_are_unchanged(tmp_path, capsys):
    cold = _write(tmp_path / "cold", tied=False, head="distinct", seed=0)
    engine, server, http = _tied_engine(tmp_path, "npu", cold, shared=False)
    assert "--worker-extension-cls" not in engine.argv(cold)
    engine.start(None, 0)
    assert "tie_word_embeddings=false" in capsys.readouterr().out
    report = engine.load(_write(tmp_path / "r1", tied=False, head="distinct", seed=1), 1)
    assert report.detail["reload"] == "collective_rpc reload_weights"
    assert [p for m, p, b in http.calls if m == "POST"] == ["/pause?mode=wait", "/collective_rpc", "/resume"]


def test_tied_model_without_in_place_reload_gets_no_extension(tmp_path):
    cold = _write(tmp_path / "cold", head="equal")
    config = _config(tmp_path, base_dir=cold, card_model="910B4", device_family="npu", precision="bf16",
                     quant_scheme="bf16", topology="tp1", max_model_len=2048)
    engine = VllmServerEngine(config, VllmOptions(platform="npu", tp=1))
    assert "--worker-extension-cls" not in engine.argv(cold)
