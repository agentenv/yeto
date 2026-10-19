"""Stacked native source (N8.5b), per-rank re-quantization (N8.2) and the trainer adapters.

Both schemes use the tiny artefacts of tests/test_rl_quant_requant.py (W4A8, R2
AWQ) and tests/test_rl_quant_mxfp4.py (MXFP4 RTN). The reference output is the
single-process converter on the per-expert BF16 source; every other path must
write the same bytes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests import test_rl_quant_mxfp4 as mx
from tests import test_rl_quant_requant as w4
from yeto.rl.adapters.miles.quant_shards import hf_slices, megatron_expert_slices
from yeto.rl.adapters.verl.quant_shards import fsdp2_slices, shard0_rows
from yeto.rl.engine.policy_release import (
    RELEASE_DIR,
    W4A8_RELEASE_DIR,
    PolicyFollower,
    PolicyReleaseStore,
    SchemeMismatch,
    release_scheme,
)
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.quant.awq import AwqSmoothing
from yeto.rl.quant.release import RequantReleaser
from yeto.rl.quant.requant import (
    INDEX_FILE,
    MXFP4_G32_RTN,
    W4A8_INT4_CHANNEL,
    CheckpointLayout,
    LayoutError,
    finalize_checkpoint,
    requantize_checkpoint,
)
from yeto.rl.quant.schemes import BF16, release_dir_for
from yeto.rl.quant.shards import (
    PART_DIR,
    SourceSlice,
    merge_parts,
    requantize_distributed,
    write_part,
)
from yeto.rl.quant.stacked import StackedSourceIndex, unstack_experts

_PER_EXPERT = re.compile(r"^(?P<prefix>.+\.mlp\.experts)\.(?P<e>\d+)\.(?P<proj>gate|up|down)_proj\.weight$")


def _build(tmp: Path, scheme: str) -> dict:
    """Template, smoothing and the per-expert BF16 source of one scheme's tiny artefact."""
    tmp.mkdir(parents=True, exist_ok=True)
    original = tmp / "original"
    if scheme == "w4a8":
        s = (0.8 + 0.4 * torch.rand(w4.H, generator=torch.Generator().manual_seed(7))).double()
        scales = tmp / "awq-scales"
        scales.mkdir()
        torch.save(s, scales / f"{w4.NORM}.pt")
        src = w4._source_tensors()
        w4._write_ckpt(original, w4._reference_artefact(src, s), w4._shard_of)
        (original / "config.json").write_text(json.dumps({"model_type": "glm5_next", "quantization_config": w4.QCONFIG}))
        (original / "recipe.yaml").write_text(w4.RECIPE)
        smoothing = AwqSmoothing.from_artefact(original / "recipe.yaml", scales)
    else:
        src = mx._source_tensors()
        mx._write_ckpt(original, mx._reference_artefact(src), mx._shard_of)
        (original / "config.json").write_text(json.dumps({"model_type": "glm5_next", "quantization_config": mx.QCONFIG}))
        smoothing = AwqSmoothing.none()
    template = tmp / "template"
    CheckpointLayout.from_checkpoint(original).save_template(original, template)
    source = tmp / "bf16"
    w4._write_ckpt(source, src, lambda n: "model-00001-of-00001.safetensors")
    reference = tmp / "reference"
    requantize_checkpoint(source, template, reference, smoothing)
    return {"src": src, "template": template, "smoothing": smoothing, "reference": reference, "tmp": tmp}


@pytest.fixture(params=["w4a8", "mxfp4"])
def world(request, tmp_path):
    return _build(tmp_path / request.param, request.param)


def _stack(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The trainer-native layout: experts.gate_up_proj [E, 2I, H] and experts.down_proj [E, H, I]."""
    groups: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    out = {}
    for name, t in src.items():
        m = _PER_EXPERT.match(name)
        if m is None:
            out[name] = t
        else:
            groups.setdefault(m["prefix"], {}).setdefault(int(m["e"]), {})[m["proj"]] = t
    for prefix, experts in groups.items():
        order = sorted(experts)
        assert order == list(range(len(order)))
        out[prefix + ".gate_up_proj"] = torch.stack([torch.cat([experts[e]["gate"], experts[e]["up"]]) for e in order])
        out[prefix + ".down_proj"] = torch.stack([experts[e]["down"] for e in order])
    return out


def _assert_same_files(a: Path, b: Path) -> None:
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    for name in names:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


# -- N8.5b: stacked native source --------------------------------------------------

def test_unstack_round_trips_the_per_expert_tensors(world):
    stacked = _stack(world["src"])
    got = {}
    for name, t in stacked.items():
        if name.endswith((".gate_up_proj", ".down_proj")):
            got.update(unstack_experts(name, t))
    expected = {n: t for n, t in world["src"].items() if _PER_EXPERT.match(n)}
    assert got.keys() == expected.keys() and all(torch.equal(got[n], expected[n]) for n in got)


def test_stacked_native_source_converts_to_the_same_bytes(world):
    native = world["tmp"] / "native"
    w4._write_ckpt(native, _stack(world["src"]), lambda n: "model-00001-of-00002.safetensors"
                   if "gate_up" in n else "model-00002-of-00002.safetensors")
    out = world["tmp"] / "from-native"
    requantize_checkpoint(native, world["template"], out, world["smoothing"])
    _assert_same_files(out, world["reference"])


def test_source_with_stacked_and_per_expert_experts_is_rejected():
    names = ["a.mlp.experts.down_proj", "a.mlp.experts.0.down_proj.weight"]
    with pytest.raises(ValueError, match="both stacked and per-expert"):
        StackedSourceIndex.from_names(names, lambda n: (1, 4, 4))


# -- N8.2: parts and merge -----------------------------------------------------------

def _row_split(src: dict[str, torch.Tensor], ranks: int) -> list[list[SourceSlice]]:
    """Per-expert matrices round robin; other tensors of dim >= 2 split by rows; small ones on rank 0."""
    out: list[list[SourceSlice]] = [[] for _ in range(ranks)]
    for i, (name, t) in enumerate(sorted(src.items())):
        if _PER_EXPERT.match(name):
            out[i % ranks].append(SourceSlice(name, t))
        elif t.dim() >= 2 and t.shape[0] >= ranks:
            for r in range(ranks):
                a, b = shard0_rows(t.shape[0], r, ranks)
                out[r].append(SourceSlice(name, t[a:b], a))
        else:
            out[0].append(SourceSlice(name, t))
    return out


@pytest.mark.parametrize("ranks", [2, 3])
def test_parts_merge_to_the_single_process_bytes(world, ranks):
    out = world["tmp"] / f"merged-{ranks}"
    parts = out / PART_DIR
    for r, slices in enumerate(_row_split(world["src"], ranks)):
        write_part(parts, r, slices, world["template"], world["smoothing"])
    shards = list(CheckpointLayout.load(world["template"]).shards())
    for r in range(ranks):  # each "rank" merges its own subset of shards
        merge_parts(parts, world["template"], out, shards[r::ranks])
    finalize_checkpoint(world["template"], out)
    shutil.rmtree(parts)
    _assert_same_files(out, world["reference"])


def test_merge_rejects_gaps_overlaps_and_missing_tensors(world):
    src, tpl, sm = world["src"], world["template"], world["smoothing"]
    big = next(n for n, t in sorted(src.items()) if t.dim() == 2 and t.shape[0] >= 4 and not _PER_EXPERT.match(n))
    rest = [SourceSlice(n, t) for n, t in src.items() if n != big]
    rows = src[big].shape[0]
    cases = {
        "gap": [SourceSlice(big, src[big][: rows // 2])],
        "overlap": [SourceSlice(big, src[big]), SourceSlice(big, src[big][1:], 1)],
        "no rank gave": [],
    }
    for i, (msg, extra) in enumerate(cases.items()):
        parts = world["tmp"] / f"bad-{i}"
        write_part(parts, 0, rest, tpl, sm)
        if extra:
            write_part(parts, 1, extra, tpl, sm)
        with pytest.raises(LayoutError, match=msg):
            merge_parts(parts, tpl, world["tmp"] / f"bad-out-{i}")


def test_a_slice_that_does_not_fit_is_rejected_on_the_rank(world):
    name, t = next((n, t) for n, t in world["src"].items() if t.dim() == 2 and not _PER_EXPERT.match(n))
    with pytest.raises(LayoutError, match="does not fit"):
        write_part(world["tmp"] / "p", 0, [SourceSlice(name, t, row_start=1)], world["template"], world["smoothing"])


# -- Megatron adapter ----------------------------------------------------------------

def _megatron_rank(src: dict[str, torch.Tensor], ep_rank: int, num_local: int, sequential: bool):
    """Megatron names for the experts of one EP rank (layers keep their global index; offset 0)."""
    params = []
    for name, t in src.items():
        m = _PER_EXPERT.match(name)
        if m is None or m["proj"] == "up":
            continue
        e = int(m["e"])
        if e // num_local != ep_rank:
            continue
        layer = re.search(r"layers\.(\d+)\.", name).group(1)
        local = e % num_local
        fc = "1" if m["proj"] == "gate" else "2"
        weight = torch.cat([t, src[name.replace("gate_proj", "up_proj")]]) if fc == "1" else t
        key = (f"module.decoder.layers.{layer}.mlp.experts.local_experts.{local}.linear_fc{fc}.weight" if sequential
               else f"module.decoder.layers.{layer}.mlp.experts.linear_fc{fc}.weight{local}")
        params.append((key, weight))
    params.append(("module.decoder.layers.0.self_attention.linear_qkv.weight", torch.zeros(2, 2)))  # skipped
    return params


def test_megatron_expert_parallel_ranks_merge_to_the_same_bytes(world):
    src, out = world["src"], world["tmp"] / "megatron"
    parts = out / PART_DIR
    non_expert = [(n, t) for n, t in src.items() if not _PER_EXPERT.match(n)]
    for ep_rank in (0, 1):  # E = 2: one local expert per rank; rank 1 uses the sequential-MLP names
        slices = list(megatron_expert_slices(_megatron_rank(src, ep_rank, 1, sequential=ep_rank == 1),
                                             ep_rank=ep_rank, num_local_experts=1))
        if ep_rank == 0:
            slices += list(hf_slices(non_expert))
        write_part(parts, ep_rank, slices, world["template"], world["smoothing"])
    merge_parts(parts, world["template"], out)
    finalize_checkpoint(world["template"], out)
    shutil.rmtree(parts)
    _assert_same_files(out, world["reference"])


def test_megatron_adapter_refuses_expert_tensor_parallelism_and_bad_indices():
    with pytest.raises(ValueError, match="expert tensor parallel"):
        list(megatron_expert_slices([], ep_rank=0, num_local_experts=1, expert_tp_size=2))
    with pytest.raises(ValueError, match="local expert 3"):
        list(megatron_expert_slices([("decoder.layers.0.mlp.experts.linear_fc2.weight3", torch.zeros(2, 2))],
                                    ep_rank=0, num_local_experts=2))


def test_megatron_names_map_to_global_layer_and_expert():
    fc1 = torch.arange(8.0).reshape(4, 2)
    got = list(megatron_expert_slices([("decoder.layers.1.mlp.experts.linear_fc1.weight1", fc1)],
                                      ep_rank=2, num_local_experts=4, layer_offset=10))
    assert [s.name for s in got] == ["model.language_model.layers.11.mlp.experts.9.gate_proj.weight",
                                     "model.language_model.layers.11.mlp.experts.9.up_proj.weight"]
    assert torch.equal(got[0].tensor, fc1[:2]) and torch.equal(got[1].tensor, fc1[2:])


def test_shard0_rows_follows_torch_chunk():
    for rows in (0, 1, 5, 288):
        for world_size in (1, 2, 3, 7):
            chunks = torch.arange(rows).chunk(world_size) if rows else ()
            got = [shard0_rows(rows, r, world_size) for r in range(world_size)]
            starts = [(int(c[0]), int(c[-1]) + 1) for c in chunks]
            assert [g for g in got if g[1] > g[0]] == starts


# -- 2 gloo ranks, FSDP2 DTensors ------------------------------------------------------

def _gloo_worker(rank: int, world_size: int, init: str, case: str, out: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard

    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world_size)
    try:
        w = _build(Path(case), Path(case).name) if rank == 0 else None
        dist.barrier()
        if rank != 0:  # rank 0 built the files; the other ranks load the same inputs
            w = _load(Path(case))
        mesh = init_device_mesh("cpu", (world_size,))
        params = []
        for name, t in sorted(_stack(w["src"]).items()):
            if t.dim() >= 2:
                a, b = shard0_rows(t.shape[0], rank, world_size)
                local = t[a:b].clone()
                params.append(("_fsdp." + name, DTensor.from_local(local, mesh, [Shard(0)], run_check=False,
                                                                   shape=t.shape, stride=t.stride())))
            else:
                params.append(("_fsdp." + name, t))  # replicated: rank 0 gives it
        slices = fsdp2_slices(params, rank=rank, rename=lambda n: n.removeprefix("_fsdp."))
        requantize_distributed(slices, w["template"], Path(out), w["smoothing"], rank=rank,
                               world_size=world_size, barrier=dist.barrier)
    finally:
        dist.destroy_process_group()


def _load(case: Path) -> dict:
    from safetensors.torch import load_file

    scheme = case.name
    original = case / "original"
    smoothing = (AwqSmoothing.from_artefact(original / "recipe.yaml", case / "awq-scales") if scheme == "w4a8"
                 else AwqSmoothing.none())
    return {"src": load_file(str(case / "bf16" / "model-00001-of-00001.safetensors")),
            "template": case / "template", "smoothing": smoothing}


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo is not available")
@pytest.mark.parametrize("scheme", ["w4a8", "mxfp4"])
def test_two_gloo_ranks_write_the_single_process_bytes(tmp_path, scheme):
    case, out = tmp_path / scheme, tmp_path / "release"
    init = f"file://{tmp_path / 'rendezvous'}"
    mp.start_processes(_gloo_worker, args=(2, init, str(case), str(out)), nprocs=2, join=True, start_method="spawn")
    assert not (out / PART_DIR).exists()
    assert json.loads((out / INDEX_FILE).read_text())["weight_map"]
    _assert_same_files(out, case / "reference")


# -- N8.3: release directories, records and the reader check ---------------------------

def test_release_directories_are_named_by_scheme_with_the_old_w4a8_name():
    assert release_dir_for(BF16) == RELEASE_DIR == "policy-releases"
    assert release_dir_for(W4A8_INT4_CHANNEL.name) == W4A8_RELEASE_DIR == "policy-releases-w4a8"
    assert release_dir_for(MXFP4_G32_RTN.name) == "policy-releases-mxfp4-g32-rtn"
    with pytest.raises(ValueError, match="unknown weight scheme"):
        release_dir_for("fp8")


def _bf16_store(tmp: Path, world: dict):
    store = CheckpointStore(tmp / "store")
    source = world["tmp"] / "bf16"
    PolicyReleaseStore(store).release(1, "h1", lambda d: [os.link(p, d / p.name) for p in source.iterdir()])
    return store


def test_mxfp4_release_cli_writes_its_own_directory_and_record(tmp_path):
    w = _build(tmp_path / "mxfp4", "mxfp4")
    store = _bf16_store(tmp_path, w)
    done = subprocess.run([sys.executable, "-m", "yeto.rl.quant.release", "--store", str(store.root),
                           "--template", str(w["template"]), "--no-awq"], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])})
    assert done.returncode == 0, done.stderr
    assert not (store.root / "policy-releases-w4a8").exists()
    rec = PolicyReleaseStore(store, release_dir="policy-releases-mxfp4-g32-rtn").latest()
    assert rec.version == 1 and rec.policy_hash == "h1+mxfp4-g32-rtn"
    assert {k: rec.derived_from[k] for k in ("release_dir", "version", "scheme", "awq", "awq_source")} == {
        "release_dir": "policy-releases", "version": 1, "scheme": "mxfp4-g32-rtn", "awq": "none", "awq_source": None}
    assert rec.derived_from["expert_matrices"] == len(CheckpointLayout.load(w["template"]).quantized_modules)
    out = store.root / "policy-releases-mxfp4-g32-rtn" / rec.path
    for f in w["reference"].glob("*.safetensors"):
        assert (out / f.name).read_bytes() == f.read_bytes()


def test_releaser_refuses_a_directory_of_another_scheme_and_bad_awq_records(tmp_path):
    store = CheckpointStore(tmp_path / "s")
    bf16 = PolicyReleaseStore(store)
    w4a8 = PolicyReleaseStore(store, release_dir="policy-releases-w4a8")
    noop = lambda src, out: None  # noqa: E731
    with pytest.raises(ValueError, match="mxfp4-g32-rtn releases go to"):
        RequantReleaser(bf16, w4a8, noop, scheme="mxfp4-g32-rtn")
    with pytest.raises(ValueError, match="not a re-quantized scheme"):
        RequantReleaser(bf16, w4a8, noop, scheme="bf16")
    with pytest.raises(ValueError, match="give a source exactly"):
        RequantReleaser(bf16, w4a8, noop, awq="fixed-artefact")


def test_follower_refuses_a_release_of_another_scheme_before_reading_it(tmp_path):
    store = CheckpointStore(tmp_path / "s")
    # A W4A8 release written into the MXFP4 directory (a misconfigured releaser or copy).
    wrong = PolicyReleaseStore(store, release_dir="policy-releases-mxfp4-g32-rtn")
    rec = wrong.release(3, "h+w4a8-int4-channel", lambda d: (d / "model.safetensors").write_text("x"),
                        derived_from={"scheme": "w4a8-int4-channel", "version": 3})
    assert release_scheme(rec) == "w4a8-int4-channel"
    loaded = []
    follower = PolicyFollower(wrong, lambda r, d: loaded.append(r), scheme="mxfp4-g32-rtn")
    with pytest.raises(SchemeMismatch, match="w4a8-int4-channel"):
        follower.refresh()
    assert loaded == [] and follower.release is None
    assert PolicyFollower(wrong, lambda r, d: loaded.append(r), scheme="w4a8-int4-channel").refresh().version == 3
    bf16 = PolicyReleaseStore(store)
    bf16.release(4, "h4", lambda d: (d / "model.safetensors").write_text("y"))
    with pytest.raises(SchemeMismatch, match="'bf16'"):
        PolicyFollower(bf16, lambda r, d: None, scheme="w4a8-int4-channel").refresh()
