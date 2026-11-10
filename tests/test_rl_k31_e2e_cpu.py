"""Kimi-K3.1 async GRPO on CPU, end to end over the fake engine and a fake object
store: engine sampler -> sample buffer -> expert selection -> trainer argv ->
release with the selection manifest -> sampler hot update, policy-age gate and the
near-region peer fan-out. Nothing here starts Miles, SGLang or a GPU.

The Kimi-K3-4layer smoke shape is used throughout: 4 layers (layer 0 dense,
MoE layers 1..3), 896 routed experts, top-16, 128 trainable per layer, EP 8.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yeto.cloud import s3_store
from yeto.rl.adapters.miles import config as mc
from yeto.rl.adapters.miles.buffer_rollout import BatchReader
from yeto.rl.adapters.miles.models.kimi_k31 import model as k31_model
from yeto.rl.adapters.miles.models.kimi_k31 import ports_recipe as k31_recipe
from yeto.rl.adapters.miles.models.kimi_k31 import provider as k31_provider
from yeto.rl.adapters.miles.models.kimi_k31 import release as k31_release
from yeto.rl.adapters.miles.models.partial_experts import selection as sel
from yeto.rl.adapters.sampler_engines import from_spec
from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.adapters.sampler_engines.sglang_server import moe_layer_count as sglang_moe_layer_count
from yeto.rl.algos import score_centering as sc
from yeto.rl.engine import buffer_payload as bp
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine import run_config as rc
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.policy_release import PolicyRelease, PolicyReleaseStore
from yeto.rl.engine.release_delta import write_changes
from yeto.rl.engine.sample_buffer import INDEX, FileSampleBuffer, needs_shards, shard_index
from yeto.rl.engine.sampler_engine import Provenance

from test_rl_miles_adapter_config import make_config, sub
from test_rl_object_store import FakeS3

NUM_LAYERS, DENSE, NUM_EXPERTS, TOPK, NUM_TRAINABLE, EP = 4, 1, 896, 16, 128, 8
L_MOE = NUM_LAYERS - DENSE  # 3
MOE_LAYERS = list(range(DENSE, NUM_LAYERS))  # HF layers 1..3
BUCKET_URI = "s3://fake-bucket/_e2e/k31"
SAMPLER_ID = "nc-mi355x-0"


def reward_fn(item, completion) -> float:
    """The spec's reward (``module:function``): one when the answer text appears."""
    return 1.0 if str(item.label or "") in (completion.text or "") else 0.0


# ------------------------------------------------------------------ F2 validator
# Copied from the Miles fork (k31/integ) miles/utils/args/configs/partial_experts.py,
# minus the pydantic trait: the pure-Python checks ExpertSelection.from_json runs.
# A selection yeto writes must pass this copy unchanged and give the same sha256.
F2_SELECTION_SCHEMA = 1


class F2ExpertSelectionError(ValueError):
    pass


def _f2_require(condition: bool, message: str) -> None:
    if not condition:
        raise F2ExpertSelectionError(message)


def f2_canonical_layers_sha256(layers) -> str:
    canonical = {
        str(k): {"trainable": [int(x) for x in v["trainable"]], "slot_order": [int(x) for x in v["slot_order"]]}
        for k, v in layers.items()
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def f2_expert_selection_from_json(obj) -> dict:
    """``ExpertSelection.from_json`` + ``validate`` of the Miles fork, as plain dicts."""
    _f2_require(isinstance(obj, dict), "selection must be a JSON object")
    _f2_require(obj.get("schema") == F2_SELECTION_SCHEMA, f"schema {obj.get('schema')!r}, expected 1")
    for key in ("num_experts", "num_trainable", "ep_size", "layers"):
        _f2_require(key in obj, f"missing key {key!r}")
    layers_obj = obj["layers"]
    _f2_require(isinstance(layers_obj, dict) and layers_obj, "layers must be a non-empty object")
    layers = {}
    for key, layer in layers_obj.items():
        _f2_require(str(key).isdigit(), f"layer key {key!r} is not an HF layer index")
        _f2_require(isinstance(layer, dict), f"layer {key}: must be an object")
        _f2_require("trainable" in layer and "slot_order" in layer, f"layer {key}: needs trainable and slot_order")
        layers[int(key)] = (tuple(int(x) for x in layer["trainable"]), tuple(int(x) for x in layer["slot_order"]))
    n, k, ep = int(obj["num_experts"]), int(obj["num_trainable"]), int(obj["ep_size"])
    _f2_require(n > 0 and k > 0 and ep > 0, "num_experts, num_trainable, ep_size must be positive")
    _f2_require(k < n, f"num_trainable {k} must be below num_experts {n}")
    _f2_require(n % ep == 0, f"ep_size {ep} does not divide num_experts {n}")
    _f2_require(k % ep == 0, f"ep_size {ep} does not divide num_trainable {k}: unequal trainable slots per rank")
    n_local, k_local = n // ep, k // ep
    full = list(range(n))
    for layer, (trainable, slot_order) in sorted(layers.items()):
        where = f"layer {layer}"
        _f2_require(len(trainable) == k, f"{where}: {len(trainable)} trainable ids, expected {k}")
        _f2_require(list(trainable) == sorted(set(trainable)), f"{where}: trainable ids must be ascending and unique")
        _f2_require(all(0 <= e < n for e in trainable), f"{where}: trainable id out of range")
        _f2_require(sorted(slot_order) == full, f"{where}: slot_order is not a permutation of range({n})")
        selected = set(trainable)
        for rank in range(ep):
            local = slot_order[rank * n_local:(rank + 1) * n_local]
            flags = [e in selected for e in local]
            _f2_require(flags == [True] * k_local + [False] * (n_local - k_local),
                        f"{where}: EP rank {rank} must hold exactly its first {k_local} local slots trainable")
    sha = f2_canonical_layers_sha256(
        {str(k_): {"trainable": list(v[0]), "slot_order": list(v[1])} for k_, v in layers.items()})
    if obj.get("sha256") is not None:
        _f2_require(obj["sha256"] == sha, f"sha256 {obj['sha256']} does not match layers ({sha})")
    return {"model": str(obj.get("model", "")), "num_experts": n, "num_trainable": k, "ep_size": ep,
            "layers": layers, "sha256": sha}


# ------------------------------------------------------------------ fixtures
def _hf_config(num_layers: int = NUM_LAYERS) -> dict:
    """``config.json`` of Kimi-K3.1 cut down to ``num_layers`` (the field names of
    the public Kimi-K3 text_config; K3.1 adds the DSA indexer tables only)."""
    text = {
        "model_type": "kimi_linear", "architectures": ["KimiLinearForCausalLM"],
        "num_hidden_layers": num_layers, "first_k_dense_replace": DENSE,
        "hidden_size": 7168, "intermediate_size": 33792, "num_attention_heads": 96, "num_key_value_heads": 96,
        "num_experts": NUM_EXPERTS, "num_experts_per_token": TOPK, "moe_intermediate_size": 3072,
        "routed_expert_hidden_size": 3584, "num_shared_experts": 2, "num_expert_group": 1, "topk_group": 1,
        "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
        "v_head_dim": 128, "vocab_size": 163840, "max_position_embeddings": 1048576, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    }
    return {"architectures": ["KimiK31ForConditionalGeneration"], "model_type": "kimi_k31",
            "tie_word_embeddings": False, "text_config": text}


class Bucket:
    """One fake S3 bucket shared by the trainer and the sampler 'nodes'; each node
    sees it through its own ``S3Store`` stage directory (the object API path a Modal
    trainer and an NC sampler use, no mount)."""

    def __init__(self, tmp_path: Path, client=None, uri: str = BUCKET_URI) -> None:
        self.client = client or FakeS3()
        self.uri = uri
        self.tmp = tmp_path
        self.nodes = 0

    def store(self, name: str) -> s3_store.S3Store:
        self.nodes += 1
        return s3_store.S3Store(self.tmp / f"stage-{name}", self.uri, environ={}, client=self.client,
                                stage=self.tmp / f"stage-{name}")

    def stores(self, name: str):
        """``(release store, sample buffer)`` as ``engine_sampler_entry.build_stores`` builds them."""
        store = self.store(name)
        index = shard_index(SAMPLER_ID) if needs_shards(store) else INDEX
        # each node downloads releases into its own cache (the default cache is keyed by
        # the store URI, which every test here shares: never reuse another test's download)
        return (PolicyReleaseStore(store, cache_dir=self.tmp / f"release-cache-{name}"),
                FileSampleBuffer(store, batch_size=1, index=index))


def _prompts_file(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.jsonl"
    rows = [{"prompt": f"question {i}: what is {i}+{i}?", "answer": str(2 * i)} for i in range(6)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _spec(tmp_path: Path, *, num_rollouts: int, selection_sha256: str | None = None, **more) -> dict:
    """The engine sampler spec of a K3.1 sampler node, on the fake engine."""
    spec = {
        "sampler_id": SAMPLER_ID,
        "engine": {"factory": from_spec.FACTORY, "adapter": "fake",
                   "kwargs": {"routed_experts": [NUM_EXPERTS, L_MOE, TOPK], "routed_experts_pad_rows": 1,
                              "expert_file_update": True, "max_model_len": 4096}},
        "rollout_fields": {"routed_experts": True, "top_logprobs_num": 128, "fp32_lm_head": True},
        "platform": "rocm",
        "sampling": {"max_tokens": 6, "n": 2, "temperature": 1.0, "top_p": 1.0, "top_k": -1},
        "num_rollouts": num_rollouts, "max_backlog_groups": 100, "max_prompt_tokens": 64,
        "store": str(tmp_path / "unused-store-path"),  # the stores callback replaces it
        "prompts": {"path": str(_prompts_file(tmp_path)), "input_key": "prompt", "label_key": "answer",
                    "groups_per_round": 2, "format": "text"},
        "reward": f"{__name__}:reward_fn",
        "event_tape": str(tmp_path / "tape" / "events.jsonl"),
        "incarnation_lease": False,
        "max_consecutive_load_failures": 10,
        # a cut-down rehearsal model cannot answer the fixed prompts (the stage-1 K3-4layer spec)
        "checks": {"require_fixed_answers": False},
    }
    if selection_sha256 is not None:
        spec["expert_selection_sha256"] = selection_sha256
    spec.update(more)
    return spec


def _run_sampler(spec: dict, bucket: Bucket, name: str, environ: dict | None = None) -> tuple[int, list]:
    events: list[tuple[str, dict]] = []
    rounds = entry.run(spec, environ=dict(environ or {}), resource_sampler=lambda emit: None,
                       stores=lambda incarnation, emit: bucket.stores(name),
                       on_event=lambda event, **fields: events.append((event, fields)))
    return rounds, events


def _read_groups(bucket: Bucket, name: str) -> list[tuple[int, list[dict]]]:
    """Every group in the buffer as ``(behavior_version, samples)``, trainer side."""
    _, buffer = bucket.stores(name)
    buffer.refresh()
    out = []
    for entry_ in buffer.entries():
        out.append((entry_.behavior_version, bp.decode_group(buffer.read_payload(entry_.payload_ref))))
    return out


def _release_dir(directory: Path, *, changed: list[str] | None = None, base_dir: Path | None = None) -> None:
    """A K3.1-shaped release: one safetensors file per (layer, pair of experts) under
    the checkpoint's names, config.json and the index; ``changed`` adds the change
    manifest of a release rebuilt from ``base_dir`` (release_delta.write_changes)."""
    directory.mkdir(parents=True, exist_ok=True)
    index = {}
    for layer in MOE_LAYERS:
        for half in range(2):
            file = f"model-l{layer}-e{half}.safetensors"
            (directory / file).write_bytes(f"{directory.name}:{file}".encode())
            for expert in (2 * half, 2 * half + 1):
                for proj in ("w1", "w2", "w3"):
                    for kind in ("weight_packed", "weight_scale"):
                        index[f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.{kind}"] = file
    (directory / "model-dense.safetensors").write_bytes(f"{directory.name}:dense".encode())
    index["model.embed_tokens.weight"] = "model-dense.safetensors"
    index["model.layers.1.block_sparse_moe.gate.weight"] = "model-dense.safetensors"
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    (directory / "config.json").write_text(json.dumps(_hf_config()))
    if changed is not None:
        write_changes(directory, base_dir=base_dir or directory, files={f: 0 for f in changed})


def _release(releases: PolicyReleaseStore, version: int, *, manifest: dict | None, changed=None,
             buffer_selected: int | None = None) -> PolicyRelease:
    record = releases.release(version, f"policy-hash-{version}",
                              lambda d: _release_dir(d, changed=changed), buffer_selected=buffer_selected,
                              manifest=manifest)
    assert record is not None
    return record


# ------------------------------------------------------------------ 1. sampler
def test_sampler_writes_k31_rollout_fields_to_the_object_store_buffer(tmp_path):
    bucket = Bucket(tmp_path)
    spec = _spec(tmp_path, num_rollouts=2)
    rounds, events = _run_sampler(spec, bucket, "sampler-a")
    assert rounds == 2
    names = [n for n, _ in events]
    assert names[0] == "rl_sampler_start" and names.count("rl_sampler_engine_round") == 2
    start = events[0][1]
    assert start["provenance"]["routed_experts"] == "true"
    assert start["sampling"] == {"max_tokens": 6, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "n": 2,
                                 "seed": None, "stop": [], "repetition_penalty": 1.0, "frequency_penalty": 0.0,
                                 "presence_penalty": 0.0, "allow_penalties": False}
    groups = _read_groups(bucket, "trainer-reader")
    assert len(groups) == 4 and {v for v, _ in groups} == {0}  # 2 rounds x 2 prompts, base policy v0
    for _, samples in groups:
        assert len(samples) == 2
        for sample in samples:
            routed = bp.rollout_routed_experts(sample)
            response = int(sample["response_length"])
            assert routed.dtype == np.uint16 and routed.shape == (1 + response - 1, L_MOE, TOPK)
            assert (routed[-1] == bp.ROUTED_EXPERTS_PAD).all()  # the fake pads the last position
            live = routed[:-1]
            assert live.size and live.max() < NUM_EXPERTS
            ids, vals = bp.rollout_top_logprobs(sample)
            assert ids.shape == vals.shape == (response, 128) and ids.dtype == np.int32 and vals.dtype == np.float32
            assert list(ids[:, 0]) == list(sample["tokens"][-response:])  # the sampled token leads its row
            assert len(sample["rollout_log_probs"]) == response
            engine = sample["metadata"]["engine"]
            assert engine == {"version_fields": {"fp32_lm_head": True, "temperature": 1.0, "top_p": 1.0,
                                                 "top_k": -1}, "routed_experts": True, "top_logprobs_num": 128}
            assert sample["metadata"]["provenance"]["routed_experts"] == "true"
    # the same bytes landed in the bucket, not on a mount: a second node reads them by object key
    keys = sorted(k for _, k in bucket.client.objects)
    assert any(k.endswith(f"sample-buffer/{SAMPLER_ID}.jsonl") or "sample-buffer" in k for k in keys), keys[:5]



def test_spec_platform_and_rollout_fields_reach_the_sglang_adapter_options(tmp_path):
    """The same spec sections on the real NC node adapter: ``platform: rocm`` and the
    K3.1 rollout fields become SGLangOptions, the ROCm preset and the server flags,
    before any process starts (the adapter is built over a fake process / HTTP)."""
    from yeto.rl.adapters.sampler_engines.sglang_server import SGLangOptions, SGLangServerEngine

    from test_rl_k31_engine_sampler_r3 import _Http, _Process, _config

    fields = {"routed_experts": True, "top_logprobs_num": 128, "fp32_lm_head": True}
    options = from_spec.apply_rollout_fields("sglang-server", fields, from_spec.apply_platform(
        "sglang-server", "rocm", {"tp": 8, "dp_attention_flag": "attn-dp-size"}))
    assert options == {"tp": 8, "dp_attention_flag": "attn-dp-size", "platform": "rocm",
                       "return_routed_experts": True, "top_logprobs_num": 128, "fp32_lm_head": True}
    config = _config(tmp_path)
    engine = SGLangServerEngine(config, SGLangOptions(**options), spawn=lambda a, e: _Process(),
                                http=_Http({("GET", "/health"): {}}), sleep=lambda s: None)
    have = entry.check_rollout_fields({"rollout_fields": fields, "sampling": {"max_tokens": 1, "temperature": 1.0}},
                                      engine)
    assert have["routed_experts"] and have["top_logprobs_num"] == 128 and have["version_fields"]["fp32_lm_head"]
    argv = engine.argv(config.base_dir)
    assert "--enable-return-routed-experts" in argv and "--enable-fp32-lm-head" in argv
    env = engine.env()
    for name, value in (("ROCM_QUICK_REDUCE_QUANTIZATION", "NONE"), ("SGLANG_ROCM_RUNTIME_RESERVE_GB", "10"),
                        ("AITER_SITUV2_A8W4", "1"), ("HSA_NO_SCRATCH_RECLAIM", "1")):
        assert env.get(name) == value, name
    assert "MX_PATCH" not in env  # the PPU W4A16 patch, not ROCm
    # the fake engine refuses nothing of the same section (the CPU rehearsal)
    fake = from_spec.build({"adapter": "fake", "platform": "rocm", "rollout_fields": fields,
                            "kwargs": {"routed_experts": [NUM_EXPERTS, L_MOE, TOPK]}})
    assert fake.returns_routed_experts and fake.top_logprobs_num == 128 and fake.fp32_lm_head
    with pytest.raises(ValueError, match="vllm-server returns routed experts \\(R3\\) only on upstream"):
        from_spec.build({"adapter": "vllm-server", "rollout_fields": fields,
                         "server": {"base_dir": str(tmp_path), "port": 1}})

def test_sglang_moe_layer_count_of_kimi_k31_pro_is_92(tmp_path):
    """The sampler reads L_moe = num_hidden_layers (93) - first_k_dense_replace (1) = 92, read
    from the nested text_config; the two derivations agree on the full and the 4-layer model."""
    for layers, want in ((93, 92), (NUM_LAYERS, L_MOE)):
        (tmp_path / "config.json").write_text(json.dumps(_hf_config(layers)))
        assert sglang_moe_layer_count(tmp_path) == want == k31_model.moe_layer_count(_hf_config(layers))
    assert k31_model.NUM_MOE_LAYERS == 92 and len(k31_model.moe_layers()) == 92


# ------------------------------------------------------------------ 2. selection
def _selection_from_bucket(bucket: Bucket, name: str, out: Path) -> sel.ExpertSelection:
    counts: dict = {}
    tokens = 0
    for _, samples in _read_groups(bucket, name):
        for sample in samples:
            routed = bp.rollout_routed_experts(sample)
            sel.counts_from_routed_experts(routed, num_experts=NUM_EXPERTS, moe_layer_offset=DENSE, into=counts)
            tokens += int(routed.shape[0])
    selection = sel.ExpertSelection.from_counts(counts, model=k31_model.MODEL_TYPE, num_trainable=NUM_TRAINABLE,
                                                ep_size=EP, source={"prompts_sha256": "0" * 64, "tokens": tokens})
    selection.write(out)
    return selection


def test_selection_from_the_buffer_passes_the_miles_f2_validator_with_the_same_sha(tmp_path):
    bucket = Bucket(tmp_path)
    _run_sampler(_spec(tmp_path, num_rollouts=3), bucket, "sampler-a")
    out = tmp_path / "expert_selection.json"
    selection = _selection_from_bucket(bucket, "trainer", out)
    doc = json.loads(out.read_text())
    # INTERFACES 2.3 shape, K3-4layer: layers 1..3, 128 of 896, EP 8
    assert doc["schema"] == 1 and doc["model"] == "kimi_k31"
    assert (doc["num_experts"], doc["num_trainable"], doc["ep_size"]) == (NUM_EXPERTS, NUM_TRAINABLE, EP)
    assert sorted(int(k) for k in doc["layers"]) == MOE_LAYERS
    assert doc["source"]["method"] == "routing_frequency" and doc["source"]["tokens"] > 0
    assert all(len(layer["trainable"]) == 128 and len(layer["slot_order"]) == 896 for layer in doc["layers"].values())
    # the pads never count: every counted id is a real expert
    assert all(0 <= e < NUM_EXPERTS for layer in doc["layers"].values() for e in layer["trainable"])
    # Miles (F2) accepts the file and computes the same identity
    f2 = f2_expert_selection_from_json(doc)
    assert f2["sha256"] == doc["sha256"] == selection.sha256
    assert f2["layers"][1][0] == tuple(doc["layers"]["1"]["trainable"])
    # yeto's own validator and the CLI agree
    sel.ExpertSelection.load_file(out, expected_sha256=doc["sha256"]).check_layers(MOE_LAYERS)
    assert subprocess.run([sys.executable, "-m", "yeto.rl.adapters.miles.models.partial_experts.selection",
                           "--validate", str(out)], capture_output=True, cwd=Path(__file__).resolve().parents[1],
                          env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}).returncode == 0
    # re-derived for EP 4 (same trainable set): F2 still accepts it
    four = selection.with_ep_size(4)
    f2_four = f2_expert_selection_from_json(four.to_json())
    assert f2_four["ep_size"] == 4 and f2_four["layers"][2][0] == f2["layers"][2][0]
    # F2 refuses what yeto refuses: a trainable id in a frozen slot
    broken = json.loads(out.read_text())
    order = broken["layers"]["1"]["slot_order"]
    order[0], order[-1] = order[-1], order[0]
    broken.pop("sha256")
    with pytest.raises(F2ExpertSelectionError, match="EP rank 0"):
        f2_expert_selection_from_json(broken)
    with pytest.raises(sel.SelectionError):
        sel.validate_document({**broken, "sha256": doc["sha256"]})
    # the release manifest of this (4-layer) run; the full-model check refuses it
    manifest = k31_release.release_manifest_fields(selection, layers=MOE_LAYERS)
    assert manifest["expert_selection_sha256"] == doc["sha256"]
    assert manifest["partial_experts"]["layers"] == L_MOE and manifest["partial_experts"]["ep_size"] == EP
    with pytest.raises(sel.SelectionError, match="model MoE layers"):
        k31_release.release_manifest_fields(selection)


# ------------------------------------------------------------------ 3. trainer argv
def _k31_config(selection_path: str, *, ep: int = EP, cp: int = 1, seq_len: int = 8192, role: str | None = None,
                **top):
    cfg = make_config(colocated=False)
    geometry = dataclasses.replace(
        cfg.geometry, num_layers=NUM_LAYERS, hidden_size=7168, num_attention_heads=96, num_query_groups=96,
        kv_channels=192, ffn_hidden_size=33792, vocab_size=163840, multi_latent_attention=True,
        mla_dims=(("q_lora_rank", 1536), ("kv_lora_rank", 512), ("qk_head_dim", 128),
                  ("qk_pos_emb_head_dim", 64), ("v_head_dim", 128)),
        moe=rc.MoeGeometry(NUM_EXPERTS, 3072, TOPK, (0, 1, 1, 1), 6144))
    cfg = dataclasses.replace(
        cfg, geometry=geometry,
        model_recipe=rc.ModelRecipe(name=rc.RECIPE_KIMI_K31, provider_class="KimiK31ModelProvider",
                                    training_attention_backend="auto", gdn=rc.GdnRecipe(gated_delta_net=False),
                                    expert_selection=selection_path),
        trainable=rc.TrainableConfig(parameter_mode="full", lora_rank=0, lora_targets="", target_modules=(),
                                     expert_full_count=0),
        parallel=dataclasses.replace(cfg.parallel, actor_num_gpus_per_node=8, tensor_parallel=1,
                                     expert_parallel=ep, data_parallel=8 // cp, context_parallel=cp,
                                     dedicated_rollout_gpus=8, visible_gpus_per_node=16),
        batch=dataclasses.replace(cfg.batch, seq_len=seq_len),
        serving=dataclasses.replace(cfg.serving, offload_train=False),
        **top)
    if role is not None:
        from yeto.rl.engine.sample_roles import SampleBufferConfig

        cfg = dataclasses.replace(cfg, sample_buffer=SampleBufferConfig(role=role, store="/store", wait_s=60.0))
        if role == "trainer":  # a trainer island: a fixed partition without rollout GPUs
            cfg = sub(cfg, "parallel", dedicated_rollout_gpus=0, visible_gpus_per_node=8)
    return cfg


def _argv(config, spec: AlgorithmSpec, **kw) -> list[str]:
    return list(mc.translate_run_config(config, spec, **kw).argv)


def _k31_argv(config, spec: AlgorithmSpec, **kw) -> list[str]:
    """Stage 1 runs one island without an outer sync (no full-parameter sync preset)."""
    return _argv(config, spec, full_parameter_sync=False, **kw)


def _value(argv, flag):
    return argv[argv.index(flag) + 1]


def test_k31_trainer_argv_carries_the_recipe_and_the_score_centering_fragment(tmp_path):
    cfg = _k31_config("/run/expert_selection.json")
    argv = _k31_argv(cfg, sc.score_centering_spec())
    # INTERFACES 2.5: the score-centering fragment
    for flag, value in (("--loss-type", "score_centering"), ("--advantage-estimator", "grpo"),
                        ("--rollout-top-logprobs-num", "128"), ("--score-centering-is", "none"),
                        ("--rollout-temperature", "1.0"), ("--rollout-top-p", "1.0"), ("--rollout-top-k", "-1")):
        assert argv.count(flag) == 1 and _value(argv, flag) == value, flag
    for flag in ("--use-rollout-logprobs", "--disable-grpo-std-normalization", "--calculate-per-token-loss"):
        assert argv.count(flag) == 1, flag
    # Y3 recipe + F2 flags: raw bridge, the plugin spec, frozen router bias, the selection file
    assert _value(argv, "--megatron-to-hf-mode") == "raw" and argv.count("--megatron-to-hf-mode") == 1
    assert _value(argv, "--model-name") == "kimi_k31"
    assert argv[argv.index("--spec") + 1: argv.index("--spec") + 3] == ["miles_plugins.models.kimi_k31",
                                                                         "get_kimi_k31_spec"]
    assert _value(argv, "--moe-router-bias-update-rate") == "0" and argv.count("--moe-router-bias-update-rate") == 1
    assert _value(argv, "--partial-experts-selection") == "/run/expert_selection.json"
    assert "--use-rollout-routing-replay" in argv and "--no-enable-sample-ownership-checker" in argv
    # the layer count follows the checkpoint (4-layer smoke model), one of each flag
    assert _value(argv, "--num-layers") == "4" and argv.count("--num-layers") == 1
    assert _value(argv, "--moe-layer-freq") == "[0]+[1]*3" and argv.count("--moe-layer-freq") == 1
    assert _value(argv, "--num-experts") == "896" and argv.count("--num-experts") == 1
    assert _value(argv, "--expert-model-parallel-size") == "8"
    # no virtual pipeline, zigzag CP accepted when the sequence length allows
    assert not any(f in argv for f in k31_recipe.VPP_FLAGS)
    zigzag = _k31_argv(_k31_config("/run/s.json", cp=2, seq_len=8192), sc.score_centering_spec())
    assert _value(zigzag, "--context-parallel-size") == "2"
    with pytest.raises(mc.MilesConfigError, match="zigzag"):
        _k31_argv(_k31_config("/run/s.json", cp=2, seq_len=8190), sc.score_centering_spec())
    with pytest.raises(mc.MilesConfigError, match="VPP"):
        _k31_argv(cfg, sc.score_centering_spec(), extra_argv=("--num-layers-per-virtual-pipeline-stage", "1"))
    with pytest.raises(mc.MilesConfigError, match="EP 3"):
        _k31_argv(_k31_config("/run/s.json", ep=3), sc.score_centering_spec())
    with pytest.raises(k31_recipe.KimiK31RecipeError, match="trainable experts"):
        k31_recipe.check_expert_parallel(7)  # divides 896, not the 128 trainable experts
    with pytest.raises(mc.MilesConfigError, match="rl-expert-selection"):
        _k31_argv(_k31_config(None), sc.score_centering_spec())
    # --rl-sample-role trainer (async: samples come from the store, max_policy_age >= 1): the
    # trainer island starts no engine; the spec must tolerate the staleness and correct it.
    # Found here: the staleness rule accepted only correction.method tis/custom, which
    # score_centering refuses; its own score_centering_is tis / mis now counts as the correction.
    def stale(importance_sampling: str, staleness: int = 2) -> AlgorithmSpec:
        payload = sc.score_centering_spec(importance_sampling=importance_sampling).structured_dict()
        payload["execution"] = {**(payload.get("execution") or {}), "max_policy_staleness": staleness}
        return AlgorithmSpec.from_dict(payload)

    trainer = _k31_argv(_k31_config("/run/s.json", role="trainer"), stale("tis"), max_policy_age=2)
    assert "--debug-train-only" in trainer and "--rollout-num-gpus" not in trainer
    assert _value(trainer, "--partial-experts-selection") == "/run/s.json"
    assert _value(trainer, "--score-centering-is") == "tis" and "--use-tis" not in trainer
    assert any("[policy_staleness]" in r for r in stale("none").rejections()) and not stale("tis").rejections()
    with pytest.raises(mc.MilesConfigError, match="max_policy_staleness"):
        _k31_argv(_k31_config("/run/s.json", role="trainer"), sc.score_centering_spec(), max_policy_age=2)


def test_default_grpo_argv_is_unchanged_by_the_k31_wiring():
    """The default GRPO path (generic recipe, no --rl-loss) emits the same argv it did
    before this wiring: no K3.1 flag leaks in (the golden test pins the bytes)."""
    argv = _argv(make_config(), AlgorithmSpec())
    for flag in ("--partial-experts-selection", "--loss-type", "--rollout-top-logprobs-num", "--score-centering-is",
                 "--use-rollout-routing-replay", "--moe-router-bias-update-rate"):
        assert flag not in argv, flag
    assert _value(argv, "--megatron-to-hf-mode") == "bridge"


def test_rl_loss_cli_flags_reach_the_island_and_the_algorithm_spec(tmp_path):
    """Y3 -> Y2 request: ``--rl-loss`` / ``--rl-score-centering-*`` / ``--rl-rollout-top-logprobs-num``
    / ``--rl-expert-selection`` / ``--rl-model-recipe kimi_k31`` parse on both layers, the
    launcher forwards them to the island, and the island applies them to the spec."""
    import argparse

    from yeto.launch.rl_island_task import _k31_island_flags
    from yeto.rl.adapters.miles.island_entry import resolve_algorithm_with_loss
    from yeto.rl.engine.run_flags import CLI, ISLAND, add_rl_flags

    given = ["--rl-model-recipe", "kimi_k31", "--rl-expert-selection", "/run/expert_selection.json",
             "--rl-loss", "score_centering", "--rl-score-centering-is", "tis", "--rl-score-centering-tis-clip", "3",
             "--rl-rollout-top-logprobs-num", "64"]
    parsed = {}
    for layer in (CLI, ISLAND):
        parser = argparse.ArgumentParser()
        add_rl_flags(parser, layer)
        for action in parser._actions:  # only the new flags are under test here
            action.required = False
        args, _ = parser.parse_known_args(given)
        parsed[layer] = args
        assert args.rl_model_recipe == "kimi_k31" and args.rl_expert_selection == "/run/expert_selection.json"
        assert args.rl_loss == "score_centering" and args.rl_score_centering_is == "tis"
        assert args.rl_score_centering_tis_clip == 3.0 and args.rl_rollout_top_logprobs_num == 64
    forwarded = _k31_island_flags(parsed[CLI]).split()
    assert forwarded == ["--rl-expert-selection", "/run/expert_selection.json", "--rl-loss", "score_centering",
                         "--rl-score-centering-is", "tis", "--rl-score-centering-tis-clip", "3.0",
                         "--rl-rollout-top-logprobs-num", "64"]
    assert _k31_island_flags(SimpleNamespace()) == ""  # nothing given: the default island argv is unchanged
    spec = resolve_algorithm_with_loss(SimpleNamespace(**{**vars(parsed[ISLAND]), "rl_algorithm_spec": None}))
    assert spec.loss.variant == "score_centering" and spec.loss.score_centering_is == "tis"
    assert spec.loss.score_centering_tis_clip == 3.0 and spec.loss.rollout_top_logprobs_num == 64
    assert spec.correction.use_rollout_logprobs and not spec.advantage.std_normalization
    plain = resolve_algorithm_with_loss(SimpleNamespace(rl_algorithm_spec=None))
    assert plain == AlgorithmSpec()
    with pytest.raises(Exception, match="needs --rl-loss score_centering"):
        resolve_algorithm_with_loss(SimpleNamespace(rl_algorithm_spec=None, rl_score_centering_is="mis"))


def test_run_config_resolves_the_k31_recipe_from_the_provider_view(tmp_path):
    """The island's provider view of the checkpoint (AutoBridge cannot read kimi_k31)
    and ``resolve_rl_run_config`` agree on the geometry the recipe argv is built from."""
    model = tmp_path / "Kimi-K3-4layer"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(_hf_config()))
    assert k31_provider.is_kimi_k31(model)
    provider = k31_provider.kimi_k31_provider(model)
    assert provider.num_layers == 4 and provider.moe_layer_freq == (0, 1, 1, 1) and provider.num_moe_layers == 3
    assert provider.num_moe_experts == 896 and provider.moe_router_topk == 16 and provider.moe_latent_size == 3584
    assert provider.kv_channels == 192 and provider.q_lora_rank == 1536
    with pytest.raises(ValueError, match="grouped top-k"):
        k31_provider.provider_view({**_hf_config(), "text_config": {**_hf_config()["text_config"], "topk_group": 4}})
    with pytest.raises(ValueError, match="not a Kimi"):
        k31_provider.provider_view({"model_type": "deepseek_v4", "text_config": {}})
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"problem": "q", "answer": "a"}\n')
    selection = sel.ExpertSelection.synthetic("uniform", model="kimi_k31", num_experts=NUM_EXPERTS,
                                              num_trainable=NUM_TRAINABLE, ep_size=EP, layers=MOE_LAYERS)
    selection_path = tmp_path / "expert_selection.json"
    selection.write(selection_path)
    # an async run tolerates staleness through the spec file (no legacy flag sets execution.*)
    base_spec = AlgorithmSpec().structured_dict()
    base_spec["execution"] = {**(base_spec.get("execution") or {}), "max_policy_staleness": 2}
    spec_path = tmp_path / "algorithm_spec.json"
    spec_path.write_text(AlgorithmSpec.from_dict(base_spec).canonical_json())
    # the stage-1 trainer island argv: 8 GPUs, TP1 EP8 (DP 8), no engine (the NC node samples), age 2
    from yeto.rl.adapters.miles import island_entry

    argv = ["--model", str(model), "--model-revision", "rev", "--data", str(prompts),
            "--rl-prompt-column", "problem", "--rl-label-column", "answer",
            "--reward-function", "yeto.rl.rewards:math", "--reward-sha256", "0" * 64, "--source-sha256", "0" * 64,
            "--learner-id", "0", "--global-rounds", "2", "--groups-per-round", "8", "--samples-per-group", "8",
            "--optimizer-steps", "1", "--over-sampling-batch-size", "8", "--rollout-max-response-len", "1024",
            "--completed-groups-path", str(tmp_path / "groups.jsonl"), "--event-tape", str(tmp_path / "tape.jsonl"),
            "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "8", "--inner-lr", "1e-5", "--seq-len", "8192",
            "--seed", "1", "--miles-root", str(tmp_path), "--parameter-mode", "full",
            "--rl-model-recipe", "kimi_k31", "--rl-expert-selection", str(selection_path),
            "--rl-sample-role", "trainer", "--rl-resume-store", "/store",  # the pair launch's trainer island
            "--rl-max-policy-age", "2", "--expert-parallel", "8",
            "--rl-single-island-no-sync", "--rl-algorithm-spec", str(spec_path),
            "--rl-loss", "score_centering", "--rl-score-centering-is", "tis"]
    args = island_entry.parse_args(argv)
    config = rc.resolve_rl_run_config(args, model_path=model, prompt_path=prompts, provider=provider,
                                      target_modules=[], yeto_policy_sync=False, verify_ref_load=False)
    assert config.model_recipe.name == "kimi_k31" and config.model_recipe.expert_selection == str(selection_path)
    assert config.sample_buffer is not None and config.sample_buffer.role == "trainer"
    assert config.geometry.num_layers == 4 and config.geometry.moe.num_experts == 896
    assert config.parallel.expert_parallel == 8 and config.parallel.data_parallel == 8
    # the island's own spec -> argv step (--rl-loss applied to the spec file, then the translation)
    launch = island_entry.build_ports_launch(args, config)
    argv_out = list(launch.argv)
    assert _value(argv_out, "--partial-experts-selection") == str(selection_path)
    assert _value(argv_out, "--num-layers") == "4" and _value(argv_out, "--moe-layer-freq") == "[0]+[1]*3"
    assert "--debug-train-only" in argv_out and _value(argv_out, "--expert-model-parallel-size") == "8"
    assert _value(argv_out, "--loss-type") == "score_centering" and _value(argv_out, "--score-centering-is") == "tis"
    assert _value(argv_out, "--megatron-to-hf-mode") == "raw" and "--use-tis" not in argv_out
    assert "--rollout-num-gpus" not in argv_out
    # the release manifest the island hands its publisher (yeto_rl_release_manifest)
    manifest = island_entry._kimi_k31_release_manifest(args, provider)
    assert manifest["expert_selection_sha256"] == selection.sha256
    # the selection flag belongs to the recipe's full form; a LoRA run refuses it
    # (tests/test_rl_k31_lora.py covers --tuning lora)
    with pytest.raises(ValueError, match="belongs to"):
        rc.resolve_rl_run_config(island_entry.parse_args([*argv[:-4], "--rl-model-recipe", "generic"]),
                                 model_path=model, prompt_path=prompts, provider=provider, target_modules=[],
                                 yeto_policy_sync=False, verify_ref_load=False)
    with pytest.raises(ValueError, match="rl-expert-selection belongs to the full-parameter"):
        rc.resolve_rl_run_config(
            island_entry.parse_args([*argv, "--parameter-mode", "lora", "--lora-r", "8", "--lora-targets", "attention"]),
            model_path=model, prompt_path=prompts, provider=provider, target_modules=["linear_qkv"],
            yeto_policy_sync=False, verify_ref_load=False)


# ------------------------------------------------------------------ 4. trainer island assembly
def test_trainer_island_assembles_upstream_components_and_carries_every_wire_key(monkeypatch):
    from yeto.rl.adapters.miles.ext_args import WIRE_KEYS, ext_attr, install_ext
    from yeto.rl.adapters.miles.island_args import ALL_KEYS, DRIVER_KEYS

    from test_rl_infra_switches import drive_run_ports_island

    calls, composed = drive_run_ports_island(monkeypatch)
    seen = composed["_upstream"]
    orch = seen["startup_info"].args
    # upstream train.py order and arguments
    assert calls.index("preflight_stage") < calls.index("connect_island_ray") < calls.index("compose_island")
    assert seen["handles"] == (orch, ["actor-config"], "CAPABILITY")
    assert seen["rollout"] == (orch, "CAPABILITY", {"actor": "HANDLE"})  # --rl-sample-role trainer: real handles
    assert seen["training"] == (orch, {"actor": "HANDLE"})
    assert composed["orchestrator_args"] is orch and composed["rollout_executor"] == "EXECUTOR"
    # a sampler island (no trainer) hands create_rollout_components an empty handle map
    import inspect

    from yeto.rl.adapters.miles import sampler_island

    assert "trainer_handles={}" in inspect.getsource(sampler_island.run_sampler_island)
    # yeto_ext: all 24 wire keys install and read back through ext_attr (new-pin carrier)
    assert len(WIRE_KEYS) == 26
    values = {key: f"value:{key}" for key in sorted(WIRE_KEYS)}
    values["rl_max_policy_age"] = 2
    config = install_ext(SimpleNamespace(yeto_ext={}), values)
    assert set(config.yeto_ext) == WIRE_KEYS
    for key in WIRE_KEYS:
        assert ext_attr(config, key) == values[key] and ext_attr(config, f"yeto_{key}") == values[key]
    assert ext_attr(config, "rl_max_policy_age") == 2
    # the K3.1 driver keys this wiring added are registered (Y1's static gate)
    assert {"rl_expert_selection", "rl_release_manifest"} <= DRIVER_KEYS <= ALL_KEYS
    assert not ({"rl_expert_selection", "rl_release_manifest"} & WIRE_KEYS)  # driver-only: never cross


def test_island_entry_installs_the_k31_runtime_and_builds_the_release_manifest(tmp_path, monkeypatch):
    from yeto.rl.adapters.miles import island_entry
    from yeto.rl.adapters.miles.models.kimi_k31 import ports_runtime

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(_hf_config()))
    selection = sel.ExpertSelection.synthetic("uniform", model="kimi_k31", num_experts=NUM_EXPERTS,
                                              num_trainable=NUM_TRAINABLE, ep_size=EP, layers=MOE_LAYERS)
    path = tmp_path / "expert_selection.json"
    selection.write(path)
    provider = k31_provider.kimi_k31_provider(model)
    args = SimpleNamespace(rl_model_recipe="kimi_k31", rl_expert_selection=str(path))
    manifest = island_entry._kimi_k31_release_manifest(args, provider)
    assert manifest["expert_selection_sha256"] == selection.sha256
    assert manifest["partial_experts"]["layers"] == L_MOE
    assert island_entry._kimi_k31_release_manifest(SimpleNamespace(rl_model_recipe="generic"), provider) is None
    # the expert count comes from the checkpoint provider, not the Pro constant
    assert int(provider.num_moe_experts) == manifest["partial_experts"]["num_experts"]
    cut = sel.ExpertSelection.synthetic("uniform", model="kimi_k31", num_experts=64, num_trainable=16,
                                        ep_size=2, layers=MOE_LAYERS)
    cut.write(path)
    with pytest.raises(ValueError, match="checkpoint has"):
        island_entry._kimi_k31_release_manifest(args, provider)
    small = SimpleNamespace(first_k_dense_replace=provider.first_k_dense_replace,
                            num_layers=provider.num_layers, num_moe_experts=64)
    assert island_entry._kimi_k31_release_manifest(args, small)["partial_experts"]["num_experts"] == 64
    # a selection of another layer set is refused before any release is written
    full = sel.ExpertSelection.synthetic("uniform", model="kimi_k31", num_experts=NUM_EXPERTS,
                                         num_trainable=NUM_TRAINABLE, ep_size=EP, layers=range(1, 93))
    full.write(path)
    with pytest.raises(sel.SelectionError, match="model MoE layers"):
        island_entry._kimi_k31_release_manifest(args, provider)
    # the runtime patch arms its import hook under the env flag the Ray workers inherit
    monkeypatch.delenv(ports_runtime.ENV_FLAG, raising=False)
    armed = []
    monkeypatch.setattr(ports_runtime, "install_or_defer", lambda name, fn: armed.append(name))
    island_entry._require_env(ports_runtime.ENV_FLAG, "1")
    ports_runtime.install()
    assert armed == ["miles.backends.megatron_utils.model"] and os.environ[ports_runtime.ENV_FLAG] == "1"


# ------------------------------------------------------------------ 5. release -> sampler

def test_release_hot_update_selection_gate_and_policy_age(tmp_path):
    """One sampler incarnation over six rounds while the trainer node acts between
    rounds (``on_event`` of the round record, as a trainer releasing during a round):
    r1 (v1, the run's selection, two expert files changed) is loaded in place through
    the file-level path; r2 (another selection) and r3 (no manifest) are refused and
    the engines keep v1; then the trainer's policy-age gate over the buffer."""
    bucket = Bucket(tmp_path)
    trainer_releases, trainer_buffer = bucket.stores("trainer")
    state: dict = {}
    changed = ["model-l1-e0.safetensors", "model-l3-e1.safetensors"]

    def trainer_acts(event, **fields):
        if event != "rl_sampler_engine_round":
            return
        if fields["rollout_id"] == 1:  # rounds 0..1 on v0: select the experts, release v1
            selection = _selection_from_bucket(bucket, "trainer", tmp_path / "expert_selection.json")
            state["selection"] = selection
            trainer_buffer.refresh()
            state["r1"] = _release(trainer_releases, 1, changed=changed, buffer_selected=len(trainer_buffer),
                                   manifest=k31_release.release_manifest_fields(selection, layers=MOE_LAYERS))
            island.loader.expected_selection_sha256 = selection.sha256  # the spec's expert_selection_sha256
        elif fields["rollout_id"] == 2:  # another slot order = another selection identity
            other = state["selection"].with_ep_size(4)
            assert other.sha256 != state["selection"].sha256
            state["other"] = other
            state["r2"] = _release(trainer_releases, 2, changed=changed,
                                   manifest=k31_release.release_manifest_fields(other, layers=MOE_LAYERS))
        elif fields["rollout_id"] == 3:
            state["r3"] = _release(trainer_releases, 3, manifest=None)

    spec = _spec(tmp_path, num_rollouts=6)
    events: list = []
    island = None

    def make_island(*a, **kw):  # the spec's expert_selection_sha256 is set once the selection exists
        nonlocal island
        island = original_build(*a, **kw)
        return island

    original_build = entry.build_engine_sampler
    entry.build_engine_sampler = make_island
    try:
        rounds = entry.run(spec, environ={}, resource_sampler=lambda emit: None,
                           stores=lambda incarnation, emit: bucket.stores("sampler"),
                           on_event=lambda event, **fields: (events.append((event, fields)), trainer_acts(event, **fields)))
    finally:
        entry.build_engine_sampler = original_build
    assert rounds == 6
    selection = state["selection"]
    latest = json.loads(bucket.client.objects[("fake-bucket", "_e2e/k31/policy-releases/LATEST.json")])
    assert latest["seq"] == 3 and [r.seq for r in (state["r1"], state["r2"], state["r3"])] == [1, 2, 3]
    assert state["r1"].manifest["expert_selection_sha256"] == selection.sha256
    r2_record = json.loads(bucket.client.objects[("fake-bucket", "_e2e/k31/policy-releases/r2-v2/RELEASE.json")])
    assert r2_record["manifest"]["expert_selection_sha256"] == state["other"].sha256  # on the wire, in the record
    assert latest["manifest"] is None  # r3 declared nothing (r1 was pruned: the store keeps two releases)
    # v1 loaded once, in place, through the k31_release plan (two files, four experts, layers 1 and 3)
    loaded = [f for n, f in events if n == "rl_sampler_policy_loaded"]
    assert [(f["seq"], f["policy_version"], f["reload_mode"]) for f in loaded] == [(1, 1, "in_place")]
    checked = [f for n, f in events if n == "rl_sampler_load_checked" and f["policy_version"] == 1]
    assert checked[0]["expert_update"] is True and checked[0]["files"] == 2
    assert checked[0]["experts"] == 4 and checked[0]["layers"] == [1, 3]
    # r2 and r3 refused before the engine was touched; v1 kept
    refused = [f for n, f in events if n == "rl_sampler_release_not_loaded"]
    assert [(f["seq"], f["reason"], f["policy_version"]) for f in refused] == [
        (2, "release_refused:expert_selection", 1), (3, "release_refused:expert_selection", 1)]
    # the loader re-checks LATEST at every round's admission: r2 once (round 3), r3 twice (rounds 4, 5);
    # each refusal counts toward max_consecutive_load_failures although the engine keeps serving v1
    failed = [f for n, f in events if n == "sampler_load_check_failed"]
    assert [f["step"] for f in failed] == ["expert_selection"] * 3
    assert all(f["engine_touched"] is False and f["kept_version"] == 1 for f in failed)
    assert failed[0]["detail"] == {"release": state["other"].sha256, "expected": selection.sha256}
    assert failed[1]["detail"] == failed[2]["detail"] == {"release": None, "expected": selection.sha256}
    assert [f["failures"] for f in failed] == [1, 2, 3]
    versions = sorted(v for v, _ in _read_groups(bucket, "trainer-2"))
    assert versions == [0] * 4 + [1] * 8  # rounds 0..1 on v0, rounds 2..5 on v1
    # the trainer's policy-age gate over everything in the buffer
    _, buffer = bucket.stores("trainer-3")
    buffer.refresh()
    reader = BatchReader(buffer, max_policy_age=1, wait_s=0.5, poll_s=0.01, sleep=lambda s: None, prefetch=False,
                         read_workers=1)
    taken, report = reader.take(0, 8, current_version=2)  # floor v1: the four v0 groups are stale
    assert report.groups == 8 and report.stale_groups == 4 and set(report.behavior_versions) == {1}
    assert all(bp.rollout_routed_experts(s) is not None for _, group in taken for s in group)
    with pytest.raises(bp.SampleBufferRolloutError):
        reader.take(1, 1, current_version=5)  # floor v4: nothing fresh is left


def test_release_download_cache_is_keyed_by_the_store(tmp_path):
    """Found by this rehearsal: the sampler's release cache defaulted to one shared
    ``~/.cache/yeto/policy-releases/<r<seq>-v<version>>`` for every store, so a node that
    serves a second run would hand its engine the first run's ``r1-v1`` download. The
    default is now keyed by the store identity; an explicit cache_dir is used as given."""
    from yeto.rl.engine.policy_release import DEFAULT_CACHE_DIR, store_cache_key
    from yeto.rl.engine.resume import CheckpointStore

    a = s3_store.S3Store(tmp_path / "a", "s3://bucket/run-a", environ={}, client=FakeS3(), stage=tmp_path / "a")
    b = s3_store.S3Store(tmp_path / "b", "s3://bucket/run-b", environ={}, client=FakeS3(), stage=tmp_path / "b")
    assert store_cache_key(a) != store_cache_key(b) and len(store_cache_key(a)) == 16
    assert store_cache_key(a) == store_cache_key(
        s3_store.S3Store(tmp_path / "c", "s3://bucket/run-a", environ={}, client=FakeS3(), stage=tmp_path / "c"))
    assert PolicyReleaseStore(a).cache_dir == Path(DEFAULT_CACHE_DIR).expanduser() / store_cache_key(a)
    assert PolicyReleaseStore(a).cache_dir != PolicyReleaseStore(b).cache_dir
    local = CheckpointStore(tmp_path / "local")
    assert PolicyReleaseStore(local).cache_dir.name == store_cache_key(local)
    assert PolicyReleaseStore(a, cache_dir=tmp_path / "given").cache_dir == tmp_path / "given"


def test_full_release_without_a_change_manifest_reloads_in_full(tmp_path):
    bucket = Bucket(tmp_path)
    releases, _ = bucket.stores("trainer")
    events: list = []

    def trainer_acts(event, **fields):
        events.append((event, fields))
        if event == "rl_sampler_engine_round" and fields["rollout_id"] == 0:
            _release(releases, 1, manifest=None)  # a full release: no .yeto-changes.json

    rounds = entry.run(_spec(tmp_path, num_rollouts=2), environ={}, resource_sampler=lambda emit: None,
                       stores=lambda incarnation, emit: bucket.stores("sampler"), on_event=trainer_acts)
    assert rounds == 2
    checked = [f for n, f in events if n == "rl_sampler_load_checked" and f["policy_version"] == 1]
    assert checked and checked[0]["expert_update_fallback"] == "no_change_manifest"
    assert sorted(v for v, _ in _read_groups(bucket, "trainer-2")) == [0, 0, 1, 1]

def test_version_fields_of_the_buffer_match_the_score_centering_argv(tmp_path):
    """The trainer's score_centering check (Y3): the sampler stamped ``engine.version_fields``
    equal to the argv's --rollout-temperature / --rollout-top-p / --rollout-top-k, fp32 head on."""
    bucket = Bucket(tmp_path)
    _run_sampler(_spec(tmp_path, num_rollouts=1), bucket, "sampler-a")
    argv = _k31_argv(_k31_config("/run/s.json"), sc.score_centering_spec())
    want = {"temperature": float(_value(argv, "--rollout-temperature")), "top_p": float(_value(argv, "--rollout-top-p")),
            "top_k": int(_value(argv, "--rollout-top-k")), "fp32_lm_head": True}
    for _, samples in _read_groups(bucket, "trainer"):
        for sample in samples:
            assert sample["metadata"]["engine"]["version_fields"] == want
            assert sample["metadata"]["engine"]["top_logprobs_num"] == int(_value(argv, "--rollout-top-logprobs-num"))


# ------------------------------------------------------------------ 6. near-region peers
def test_near_region_peer_pull_then_object_store_fallback(tmp_path, monkeypatch):
    """Peer fan-out around the object store: a sampler node pulls a release from a
    peer that already placed it (digest-guarded), a dead peer is skipped, and a release
    no peer holds comes from the object store."""
    from yeto.rl.spot import peer_fanout as pf
    from yeto.rl.spot import s3_delivery as sd
    from yeto.rl.spot_rollout import cut_manifest as cm

    from test_spot_s3_delivery import FakeS3 as FakeDeliveryS3
    from test_spot_s3_delivery import _release as delivery_release

    for key, value in (("S3_BUCKET", "b"), ("S3_PREFIX", "run"), ("S3_REGION", "auto"), ("S3_ASSUME_PRESENT", ""),
                       ("S3_METRICS", str(tmp_path / "m.jsonl"))):
        monkeypatch.setenv(key, value)
    trainer, bucket = tmp_path / "trainer", tmp_path / "bucket"
    delivery_release(trainer, "r1-v1", 1, None)
    s3 = FakeDeliveryS3(bucket)
    sd.publish(s3, trainer, "r1-v1")
    seed = tmp_path / "node-seed"
    sd.pull(s3, seed, "r1-v1")  # the first NC node pulled from the object store
    assert pf.servable(seed, "r1-v1")
    server = pf.ReleaseFileServer(seed, host="127.0.0.1", port=0).start()
    try:
        peers = ["127.0.0.1:1", f"127.0.0.1:{server.port}"]
        island = tmp_path / "node-2"
        pointer = json.loads((bucket / cm.LATEST).read_text())
        downloads = []

        class Counting(FakeDeliveryS3):
            def copy_down(self, key, local):
                downloads.append(key)
                super().copy_down(key, local)

        row = sd.pull(Counting(bucket), island, "r1-v1", digest=pointer["digest"], peers=peers)
        assert row["source"] == f"peer:127.0.0.1:{server.port}" and downloads == []
        assert (island / cm.RELEASES / "r1-v1" / "w.bin").read_bytes() == b"r1-v1" * 1000
        delivery_release(trainer, "r2-v2", 2, "r1-v1")
        sd.publish(s3, trainer, "r2-v2")
        pointer = json.loads((bucket / cm.LATEST).read_text())
        row = sd.pull(Counting(bucket), island, "r2-v2", digest=pointer["digest"], peers=peers)
        assert row["source"] == "s3" and downloads == ["r2-v2"]  # no peer has it yet
        assert pf.servable(island, "r2-v2")  # and this node now serves it to the next one
    finally:
        server.stop()


# ------------------------------------------------------------------ 7. the real R2 bucket (opt-in)
R2_ENV = "YETO_E2E_R2"


@pytest.mark.skipif(not os.environ.get(R2_ENV), reason=f"set {R2_ENV}=1 with an aws profile for the R2 bucket")
def test_sampler_and_trainer_hand_off_through_the_real_r2_bucket(tmp_path):
    """The same chain over the real bucket (``s3://<bucket>/_e2e/<run>/``): small
    objects only, every key removed at the end. Credentials come from the aws
    profile ``YETO_E2E_R2_PROFILE`` and the bucket ``YETO_E2E_R2_BUCKET`` (both required) and are never printed."""
    import uuid

    import boto3

    profile = os.environ.get("YETO_E2E_R2_PROFILE")
    bucket_name = os.environ.get("YETO_E2E_R2_BUCKET")
    if not profile or not bucket_name:
        pytest.skip("set YETO_E2E_R2_PROFILE and YETO_E2E_R2_BUCKET for the real-bucket check")
    session = boto3.Session(profile_name=profile)
    client = session.client("s3")
    run = f"_e2e/e1-{uuid.uuid4().hex[:8]}"
    uri = f"s3://{bucket_name}/{run}"
    bucket = Bucket(tmp_path, client=client, uri=uri)
    try:
        releases, buffer = bucket.stores("trainer")
        state: dict = {}
        events: list = []

        def trainer_acts(event, **fields):  # the trainer node releases v1 after round 0
            events.append((event, fields))
            if event == "rl_sampler_engine_round" and fields["rollout_id"] == 0:
                selection = _selection_from_bucket(bucket, "trainer", tmp_path / "expert_selection.json")
                state["selection"] = selection
                buffer.refresh()
                _release(releases, 1, manifest=k31_release.release_manifest_fields(selection, layers=MOE_LAYERS),
                         changed=["model-l1-e0.safetensors"], buffer_selected=len(buffer))
                island.loader.expected_selection_sha256 = selection.sha256

        island = None
        original_build = entry.build_engine_sampler

        def make_island(*a, **kw):
            nonlocal island
            island = original_build(*a, **kw)
            return island

        entry.build_engine_sampler = make_island
        try:
            rounds = entry.run(_spec(tmp_path, num_rollouts=2), environ={}, resource_sampler=lambda emit: None,
                               stores=lambda incarnation, emit: bucket.stores("sampler"), on_event=trainer_acts)
        finally:
            entry.build_engine_sampler = original_build
        assert rounds == 2
        loaded = [f for n, f in events if n == "rl_sampler_policy_loaded"]
        assert loaded and loaded[0]["policy_version"] == 1 and loaded[0]["reload_mode"] == "in_place"
        checked = [f for n, f in events if n == "rl_sampler_load_checked" and f["policy_version"] == 1]
        assert checked[0]["expert_update"] is True and checked[0]["files"] == 1
        assert sorted(v for v, _ in _read_groups(bucket, "trainer-2")) == [0, 0, 1, 1]
        record = json.loads(client.get_object(Bucket=bucket_name, Key=f"{run}/policy-releases/LATEST.json")["Body"].read())
        assert record["manifest"]["expert_selection_sha256"] == state["selection"].sha256
        keys = [o["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket_name, Prefix=run)
                for o in page.get("Contents", [])]
        assert any(k.endswith("policy-releases/LATEST.json") for k in keys)
    finally:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=run):
            for obj in page.get("Contents", []):
                client.delete_object(Bucket=bucket_name, Key=obj["Key"])
        left = [o for page in paginator.paginate(Bucket=bucket_name, Prefix=run) for o in page.get("Contents", [])]
        assert left == []


# ------------------------------------------------------------------ 8. the examples/k31 drafts
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "k31"


def _filled(doc, tmp_path: Path):
    """Angle-bracket placeholders of an example spec replaced by scratch paths."""
    if isinstance(doc, dict):
        return {k: _filled(v, tmp_path) for k, v in doc.items() if k != "_comment"}
    if isinstance(doc, list):
        return [_filled(v, tmp_path) for v in doc]
    if isinstance(doc, str) and "<" in doc:
        return str(tmp_path / doc.replace("<", "").replace(">", "").replace("/", "_").replace(" ", "-"))
    return doc


def test_example_specs_and_launch_drafts_are_loadable(tmp_path):
    for script in ("launch-trainer-modal-b300.sh", "sampler-node-nc-mi355x.sh"):
        subprocess.run(["bash", "-n", str(EXAMPLES / script)], check=True)
        text = (EXAMPLES / script).read_text()
        assert "r2.cloudflarestorage.com/" not in text  # placeholders only
    # the CPU rehearsal spec runs the fake sampler end to end
    spec = _filled(json.loads((EXAMPLES / "sampler-spec.cpu-fake.json").read_text()), tmp_path)
    Path(spec["prompts"]["path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(spec["prompts"]["path"]).write_text("".join(json.dumps({"prompt": f"{i}+{i}", "answer": str(2 * i)}) + "\n"
                                                     for i in range(4)))
    spec.update(num_rollouts=1, reward=f"{__name__}:reward_fn", reward_kind="neutral")
    # found here: a prompts file shorter than groups_per_round made one round draw a row twice
    # and the second group's payload collided with the first (same key); refused at the spec now
    short = {**spec, "prompts": {**spec["prompts"], "groups_per_round": 5}}
    with pytest.raises(entry.SamplerSpecError, match="exceeds the 4 rows"):
        entry.build_prompts(short)
    bucket = Bucket(tmp_path)
    rounds, events = _run_sampler(spec, bucket, "example")
    assert rounds == 1 and events[0][1]["provenance"]["routed_experts"] == "true"
    # the NC node spec: every section the entry reads parses; the engine section becomes
    # SGLang options with the ROCm preset and the K3.1 fields (the server is not started)
    nc = _filled(json.loads((EXAMPLES / "sampler-spec.nc-mi355x.json").read_text()), tmp_path)
    nc["peers"] = ["10.0.0.11:8431"]  # the one placeholder that is not a path
    config = entry.island_config(nc)
    assert config.sampling.n == 8 and config.expert_selection_sha256 is not None
    options = from_spec.apply_rollout_fields("sglang-server", nc["rollout_fields"], from_spec.apply_platform(
        "sglang-server", nc["platform"], dict(nc["engine"]["options"])))
    assert options["platform"] == "rocm" and options["return_routed_experts"] and options["fp32_lm_head"]
    assert options["top_logprobs_num"] == 128 and options["expert_file_update"] == "auto"
    from yeto.rl.adapters.sampler_engines.sglang_server import SGLangOptions
    from yeto.rl.adapters.sampler_engines.server import ServerConfig
    from yeto.rl.spot.peer_fanout import peers_from

    SGLangOptions(**options)
    ServerConfig(**{**nc["engine"]["server"], "base_dir": Path(nc["engine"]["server"]["base_dir"]), "extra_args": ()})
    assert peers_from(nc) == [nc["peers"][0]]
    # the async algorithm spec: staleness 2 is accepted once --rl-loss score_centering tis is applied
    base = AlgorithmSpec.from_json_file(EXAMPLES / "algorithm-spec.async-age2.json")
    assert base.execution.max_policy_staleness == 2 and base.rejections()
    applied = sc.spec_from_rl_loss_args(SimpleNamespace(rl_loss="score_centering", rl_score_centering_is="tis"), base)
    assert applied.rejections() == [] and applied.execution.max_policy_staleness == 2
    argv = _k31_argv(_k31_config("/run/s.json", role="trainer"), applied, max_policy_age=2)
    assert _value(argv, "--score-centering-is") == "tis" and "--debug-train-only" in argv


def test_a_cut_down_expert_count_flows_from_the_checkpoint_to_argv_and_release(tmp_path):
    """SC2's Kimi-K3-4layer-64experts: the routed-expert count of the checkpoint (64,
    16 trainable over EP 2) reaches Miles' --num-experts and the release manifest."""
    from yeto.rl.adapters.miles import island_entry

    model = tmp_path / "Kimi-K3-4layer-64experts"
    model.mkdir()
    hf = _hf_config()
    hf["text_config"]["num_experts"] = 64
    (model / "config.json").write_text(json.dumps(hf))
    provider = k31_provider.kimi_k31_provider(model)
    assert provider.num_moe_experts == 64
    prompts = tmp_path / "p.jsonl"
    prompts.write_text('{"problem": "q", "answer": "a"}\n')
    selection = sel.ExpertSelection.synthetic("uniform", model="kimi_k31", num_experts=64, num_trainable=16,
                                              ep_size=2, layers=MOE_LAYERS)
    selection_path = tmp_path / "expert_selection.json"
    selection.write(selection_path)
    argv = ["--model", str(model), "--model-revision", "rev", "--data", str(prompts),
            "--rl-prompt-column", "problem", "--rl-label-column", "answer",
            "--reward-function", "yeto.rl.rewards:math", "--reward-sha256", "0" * 64, "--source-sha256", "0" * 64,
            "--learner-id", "0", "--global-rounds", "2", "--groups-per-round", "8", "--samples-per-group", "4",
            "--optimizer-steps", "1", "--over-sampling-batch-size", "8", "--rollout-max-response-len", "64",
            "--completed-groups-path", str(tmp_path / "groups.jsonl"), "--event-tape", str(tmp_path / "tape.jsonl"),
            "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "2", "--inner-lr", "1e-5", "--seq-len", "2048",
            "--seed", "1", "--miles-root", str(tmp_path), "--parameter-mode", "full",
            "--rl-model-recipe", "kimi_k31", "--rl-expert-selection", str(selection_path),
            "--expert-parallel", "2", "--rl-single-island-no-sync"]
    args = island_entry.parse_args(argv)
    config = rc.resolve_rl_run_config(args, model_path=model, prompt_path=prompts, provider=provider,
                                      target_modules=[], yeto_policy_sync=False, verify_ref_load=False)
    assert config.geometry.moe.num_experts == 64
    argv_out = list(island_entry.build_ports_launch(args, config).argv)
    assert argv_out.count("--num-experts") == 1 and _value(argv_out, "--num-experts") == "64"
    assert _value(argv_out, "--partial-experts-selection") == str(selection_path)
    manifest = island_entry._kimi_k31_release_manifest(args, provider)
    assert manifest["partial_experts"]["num_experts"] == 64 and manifest["partial_experts"]["num_trainable"] == 16
