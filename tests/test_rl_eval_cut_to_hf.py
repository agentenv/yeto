"""S27 R-5: Miles full-parameter cut -> HF directory, round trip on a tiny Qwen3 (CPU)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from yeto.rl.engine.cut import (AlgorithmIdentity, CutFile, CutManifest, CutProgress,  # noqa: E402
                                commit_manifest, cut_dir, sha256_file)
from yeto.rl.eval import cut_to_hf  # noqa: E402
from yeto.rl.eval.cut_to_hf import MaterializeError, materialize_cut  # noqa: E402

# Miles' Qwen2/Qwen3 converter imports only re + torch; load it straight from a Miles
# checkout so the test exercises the real name/QKV mapping without importing Miles.
_MILES_QWEN2 = [Path(os.environ.get("YETO_MILES_SRC", "/home/michael/work/miles-next")),
                Path.home() / "miles"]


def _miles_qwen2():
    for root in _MILES_QWEN2:
        path = root / "miles/backends/megatron_utils/megatron_to_hf/qwen2.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("_miles_qwen2", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.convert_qwen2_to_hf
    return None


def _to_megatron(hf_sd, cfg):
    """Inverse of Miles' qwen2 mapping (the trainer's view), names as DDP(Float16Module) gives them."""
    heads, groups, hd, h = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size
    out = {"module.module.embedding.word_embeddings.weight": hf_sd["model.embed_tokens.weight"],
           "module.module.decoder.final_layernorm.weight": hf_sd["model.norm.weight"]}
    if "lm_head.weight" in hf_sd and not cfg.tie_word_embeddings:
        out["module.module.output_layer.weight"] = hf_sd["lm_head.weight"]
    for i in range(cfg.num_hidden_layers):
        p, m = f"model.layers.{i}.", f"module.module.decoder.layers.{i}."
        q = hf_sd[p + "self_attn.q_proj.weight"].view(groups, heads // groups, hd, h)
        k = hf_sd[p + "self_attn.k_proj.weight"].view(groups, 1, hd, h)
        v = hf_sd[p + "self_attn.v_proj.weight"].view(groups, 1, hd, h)
        out[m + "self_attention.linear_qkv.weight"] = torch.cat([q, k, v], dim=1).reshape(-1, h)
        out[m + "self_attention.linear_proj.weight"] = hf_sd[p + "self_attn.o_proj.weight"]
        out[m + "self_attention.linear_qkv.layer_norm_weight"] = hf_sd[p + "input_layernorm.weight"]
        out[m + "self_attention.q_layernorm.weight"] = hf_sd[p + "self_attn.q_norm.weight"]
        out[m + "self_attention.k_layernorm.weight"] = hf_sd[p + "self_attn.k_norm.weight"]
        out[m + "mlp.linear_fc1.weight"] = torch.cat([hf_sd[p + "mlp.gate_proj.weight"],
                                                      hf_sd[p + "mlp.up_proj.weight"]], dim=0)
        out[m + "mlp.linear_fc1.layer_norm_weight"] = hf_sd[p + "post_attention_layernorm.weight"]
        out[m + "mlp.linear_fc2.weight"] = hf_sd[p + "mlp.down_proj.weight"]
    return {k[len("module."):]: v.clone() for k, v in out.items()}  # DDP-only prefix as stored


@pytest.fixture
def tiny(tmp_path):
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                   tie_word_embeddings=False, dtype="float32")
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    base = tmp_path / "base"
    model.save_pretrained(base)
    (base / "tokenizer.json").write_text("{}")
    return cfg, model, base


def _cut(tmp_path, model_copy, *, layout=None, cut_id="r000003-a", extra_tp_shard=False):
    root = tmp_path / "store" / "round-cuts"
    d = cut_dir(root, cut_id)
    d.mkdir(parents=True)
    shard = d / "trainer_tp0_pp0_dp0.pt"
    torch.save({"schema": "yeto.cut_shard/v1", "cut_id": cut_id, "coord": {"tp": 0, "pp": 0, "dp": 0},
                "adapter": model_copy}, shard)
    files = [CutFile(shard.name, sha256_file(shard), shard.stat().st_size, {"tp": 0, "pp": 0, "dp": 0})]
    if extra_tp_shard:
        other = d / "trainer_tp1_pp0_dp0.pt"
        torch.save({"adapter": model_copy}, other)
        files.append(CutFile(other.name, sha256_file(other), other.stat().st_size, {"tp": 1, "pp": 0, "dp": 0}))
    commit_manifest(root, CutManifest(
        cut_id=cut_id, epoch=1,
        runtime={"backend_fingerprint": "f", "layout": layout or {"world": 1, "tp": 1, "pp": 1, "ep": 1, "cp": 1},
                 "rng_policy": "exact"},
        progress=CutProgress(local_step=3, scheduler_samples=12, global_batch_size=4, next_rollout_id=3,
                             policy_version=3, policy_hash="h" * 64),
        algorithm=AlgorithmIdentity("a" * 64),
        data={"sample_offset": 0, "epoch_id": 0, "sample_group_index": 0, "sample_index": 0},
        ledger={"carried_over": 0, "ready_unconsumed": 0}, outer={"settled": True},
        files=tuple(files),
        rank_summaries=tuple({"path": f.path, "scheduler_samples": 12, "has_optimizer_state": True,
                              "has_rng": True, "state_digest": "s", "rng_digest": "r"} for f in files),
        created_unix=1234.5))
    return d


def test_round_trip_with_miles_qwen_converter(tiny, tmp_path):
    convert_qwen2 = _miles_qwen2()
    if convert_qwen2 is None:
        pytest.skip("no Miles checkout with megatron_to_hf/qwen2.py")
    cfg, model, base = tiny
    trained = {k: v + 0.01 for k, v in model.state_dict().items()}  # the "trained" weights in the cut
    cut = _cut(tmp_path, _to_megatron(trained, cfg))
    out = materialize_cut(cut, base_model=base, out_root=tmp_path / "hf",
                          convert=lambda args, name, t: convert_qwen2(args, name, t))
    assert (Path(out) / "tokenizer.json").is_file()
    loaded = transformers.Qwen3ForCausalLM.from_pretrained(out).eval()
    for k, v in loaded.state_dict().items():
        assert torch.equal(v, trained[k]), k
    ids = torch.tensor([[1, 5, 9, 2]])
    ref = transformers.Qwen3ForCausalLM(cfg).eval()
    ref.load_state_dict(trained)
    assert torch.equal(loaded(ids).logits, ref(ids).logits)
    marker = json.loads((Path(out) / cut_to_hf.DONE).read_text())
    assert marker["cut_id"] == "r000003-a"
    # second call reuses the directory (same manifest)
    stamp = (Path(out) / "model.safetensors").stat().st_mtime_ns
    assert materialize_cut(cut, base_model=base, out_root=tmp_path / "hf", convert=lambda *a: 1 / 0) == out
    assert (Path(out) / "model.safetensors").stat().st_mtime_ns == stamp


def _identity(args, name, tensor):
    return [(name.replace("module.module.", ""), tensor)]


def test_refuses_tensor_parallel_cut(tiny, tmp_path):
    _, _, base = tiny
    cut = _cut(tmp_path, {"module.x": torch.zeros(1)}, layout={"world": 2, "tp": 2, "pp": 1}, extra_tp_shard=True)
    with pytest.raises(MaterializeError, match="without param_layout"):
        materialize_cut(cut, base_model=base, out_root=tmp_path / "hf", convert=_identity)


def test_refuses_lora_cut_and_name_mismatch(tiny, tmp_path):
    _, _, base = tiny
    lora = _cut(tmp_path, {"module.decoder.layers.0.linear_qkv.lora_A.weight": torch.zeros(2)}, cut_id="r1-a")
    with pytest.raises(MaterializeError, match="LoRA"):
        materialize_cut(lora, base_model=base, out_root=tmp_path / "hf", convert=_identity)
    odd = _cut(tmp_path, {"module.decoder.weird.weight": torch.zeros(2)}, cut_id="r2-a")
    with pytest.raises(MaterializeError, match="differ from the base model"):
        materialize_cut(odd, base_model=base, out_root=tmp_path / "hf", convert=_identity)


def test_megatron_args_and_env_entry(tiny, monkeypatch):
    cfg, _, base = tiny
    args = cut_to_hf.megatron_args(json.loads((base / "config.json").read_text()))
    assert (args.num_query_groups, args.kv_channels, args.vocab_size) == (2, 8, 64)
    monkeypatch.delenv("YETO_EVAL_BASE_MODEL", raising=False)
    with pytest.raises(MaterializeError, match="YETO_EVAL_BASE_MODEL"):
        cut_to_hf.materialize(SimpleNamespace(path="x"))
    assert cut_to_hf.canonical_name("module.decoder.x") == "module.module.decoder.x"


# --- TP/PP/EP merge on cuts written by real Megatron sharding (subprocess worker) -----------

_REPO = Path(__file__).resolve().parents[1]
_MEGATRON_PY = Path(os.environ.get("YETO_MEGATRON_PYTHON", "/home/michael/work/megatron-cpu-venv/bin/python"))
_MEGATRON_SRC = Path(os.environ.get("YETO_MEGATRON_SRC", "/home/michael/work/third_party/megatron-lm-miles"))


def _miles_src():
    for root in _MILES_QWEN2:
        if (root / "miles/backends/megatron_utils/named_weights.py").is_file():
            return root
    return None


def _run_worker(out, world, *flags):
    import socket
    import subprocess

    miles = _miles_src()
    if not (_MEGATRON_PY.is_file() and (_MEGATRON_SRC / "megatron/core").is_dir() and miles):
        pytest.skip("needs a megatron-core Python (YETO_MEGATRON_PYTHON/YETO_MEGATRON_SRC) and a Miles checkout")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
           "PYTHONPATH": os.pathsep.join([str(_MEGATRON_SRC), str(miles), str(_REPO)])}
    procs = [subprocess.Popen([str(_MEGATRON_PY), "-W", "ignore", str(_REPO / "tests/megatron_cut_worker.py"),
                               str(out), "--world", str(world), "--rank", str(r), *flags],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for r in range(world)]
    logs = [p.communicate(timeout=600)[0] for p in procs]
    for p, log in zip(procs, logs):
        assert p.returncode == 0, log[-3000:]


def _commit_worker_cut(root, worker_out, cut_id):
    d = cut_dir(root, cut_id)
    d.mkdir(parents=True)
    files = []
    for coord_file in sorted(worker_out.glob("coord-*.json")):
        rec = json.loads(coord_file.read_text())
        src = worker_out / rec["path"]
        os.replace(src, d / rec["path"])
        files.append(CutFile(rec["path"], sha256_file(d / rec["path"]), (d / rec["path"]).stat().st_size,
                             rec["coord"]))
    commit_manifest(root, CutManifest(
        cut_id=cut_id, epoch=1,
        runtime={"backend_fingerprint": "f", "layout": {"world": len(files)}, "rng_policy": "exact"},
        progress=CutProgress(local_step=1, scheduler_samples=4, global_batch_size=4, next_rollout_id=1,
                             policy_version=1, policy_hash="h" * 64),
        algorithm=AlgorithmIdentity("a" * 64),
        data={"sample_offset": 0, "epoch_id": 0, "sample_group_index": 0, "sample_index": 0},
        ledger={"carried_over": 0, "ready_unconsumed": 0}, outer={"settled": True}, files=tuple(files),
        rank_summaries=tuple({"path": f.path, "scheduler_samples": 4, "has_optimizer_state": True,
                              "has_rng": True, "state_digest": "s", "rng_digest": "r"} for f in files)))
    return d


def _miles_converter(family):
    path = _miles_src() / f"miles/backends/megatron_utils/megatron_to_hf/{family}.py"
    spec = importlib.util.spec_from_file_location(f"_miles_{family}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    convert = getattr(module, f"convert_{family}_to_hf")

    def wrapped(args, name, tensor):
        # The CPU worker builds Megatron's local (non-TE) layer spec, whose input norm is a
        # separate module; the TE spec used in training fuses it into linear_qkv, and Megatron
        # checkpoints the local one under that fused name too (the reference keys show it).
        name = name.replace(".input_layernorm.weight", ".self_attention.linear_qkv.layer_norm_weight")
        return convert(args, name, tensor)

    return wrapped


@pytest.mark.parametrize("case", [
    pytest.param(dict(world=4, flags=("--tp", "2", "--pp", "2"), family="qwen2", moe=False), id="dense-tp2-pp2"),
    pytest.param(dict(world=2, flags=("--ep", "2", "--moe"), family="qwen3moe", moe=True), id="moe-ep2"),
])
def test_merge_real_megatron_shards_matches_sharded_state_dict(tmp_path, case):
    worker_out = tmp_path / "worker"
    worker_out.mkdir()
    _run_worker(worker_out, case["world"], *case["flags"])
    cut = _commit_worker_cut(tmp_path / "store" / "round-cuts", worker_out, "r000001-a")

    common = dict(vocab_size=64, hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=8, tie_word_embeddings=False, dtype="float32")
    if case["moe"]:
        cfg = transformers.Qwen3MoeConfig(intermediate_size=48, moe_intermediate_size=16, num_experts=4,
                                          num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
                                          norm_topk_prob=True, **common)
        model_cls = transformers.Qwen3MoeForCausalLM
    else:
        cfg = transformers.Qwen3Config(intermediate_size=48, **common)
        model_cls = transformers.Qwen3ForCausalLM
    base = tmp_path / "base"
    model_cls(cfg).save_pretrained(base)

    out = materialize_cut(cut, base_model=base, out_root=tmp_path / "hf", convert=_miles_converter(case["family"]))
    from safetensors.torch import load_file

    ours = load_file(str(Path(out) / "model.safetensors"))
    reference = torch.load(worker_out / "reference_hf.pt")
    assert set(ours) == set(reference)
    assert [k for k in reference if not torch.equal(ours[k], reference[k])] == []

    # logits: our directory vs the reference tensors loaded through the same HF loader
    from safetensors.torch import save_file

    ref_dir = tmp_path / "reference"
    model_cls(cfg).save_pretrained(ref_dir)
    save_file(reference, str(ref_dir / "model.safetensors"), metadata={"format": "pt"})
    loaded = model_cls.from_pretrained(out).eval()
    ref_model = model_cls.from_pretrained(ref_dir).eval()
    ids = torch.tensor([[3, 17, 42, 5, 9]])
    with torch.no_grad():
        logits = loaded(ids).logits
        assert torch.equal(logits, ref_model(ids).logits)
    assert torch.isfinite(logits).all()
