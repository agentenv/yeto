"""One parameter larger than the fragment byte bound spans several fragments.

Qwen3.8-27B at TP2 holds a 248320x5120 vocabulary shard per rank (~2.54 GB
in f32), above the 2 GiB fragment bound. The layout splits such a parameter
into dimension-0 row slices; each slice is its own fragment tensor. These
tests use a small bound in place of 2 GiB:

* layouts that split nothing keep their exact hash and Miles plan hash;
* the Miles owner plan of a split layout installs on the real Miles check;
* two learners merged by the real Rust syncer give the numpy result, for
  averaged and RDA slices (RDA is per row slice), f32 and int8 pushes.
"""

from __future__ import annotations

# ruff: noqa: I001 -- install the optional bridge stub before Miles import.

import shutil
import socket
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("miles")

bridge_stub = types.ModuleType("yeto.rl.adapters.miles.models.deepseek_v4.bridge")
bridge_stub.ensure_deepseek_v4_bridge = lambda: None
sys.modules.setdefault("yeto.rl.adapters.miles.models.deepseek_v4.bridge", bridge_stub)

from miles.backends.megatron_utils.full_parameter_state import (
    FullParameterShardManifest,
    FullParameterShardSpec,
    FullParameterTopology,
    _layout_hash,
    _validate_fragment_plan_against_manifest,
)

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.fragments import MERGE_AVG, MERGE_RDA
from yeto.protocol import DTYPE_F32, DTYPE_F32_PUSH_I8, DTYPE_FLAG_TENSOR_BLOCKS, SyncerClient
from yeto.rl.adapters.miles.models.chunked_full_parameter import (
    MilesChunkedFullParameterAdapter,
)
from yeto.rl.local_learner import (
    ComponentIdentity,
    ParameterLayout,
    ParameterSpec,
    split_parameter_specs,
)

EMBEDDING = "embedding.word_embeddings.weight"
MATRIX = "decoder.layers.0.mlp.linear_fc1.weight"
NORM = "decoder.final_layernorm.weight"
# Shapes per TP rank: the vocabulary shard is the only parameter above the
# 16-scalar (64-byte) test bound that stands in for 2 GiB.
SHAPES = {EMBEDDING: (10, 4), MATRIX: (3, 4), NORM: (4,)}
BOUND_BYTES = 64


def _topology(rank: int) -> FullParameterTopology:
    return FullParameterTopology(
        tp_rank=rank, tp_size=2, pp_rank=0, pp_size=1, ep_rank=0, ep_size=1,
        cp_rank=0, cp_size=1, dp_rank=0, dp_size=1,
    )


def _manifest(rank: int, shapes=SHAPES) -> FullParameterShardManifest:
    owner = _topology(rank)
    specs = tuple(sorted(
        FullParameterShardSpec(
            role="actor", shard_id=owner.shard_id, name=name, shape=shape,
            dtype="float32", numel=int(np.prod(shape)),
        )
        for name, shape in shapes.items()
    ))
    return FullParameterShardManifest(owner, "actor", _layout_hash(owner, specs), specs)


def _component() -> ComponentIdentity:
    return ComponentIdentity("actor", "a" * 40, "b" * 64)


def _adapter(*, bound: int = BOUND_BYTES, matrix_merge: str = "avg", shapes=SHAPES):
    return MilesChunkedFullParameterAdapter.create(
        (_manifest(0, shapes), _manifest(1, shapes)),
        algorithm="grpo",
        components=(_component(),),
        minimum_fragments=2,
        max_fragment_bytes=bound,
        max_chunk_bytes=32,
        matrix_merge=matrix_merge,
    )


def _spec(name: str, shape: tuple[int, ...]) -> ParameterSpec:
    return ParameterSpec("actor", name, shape, "float32", int(np.prod(shape)), "tp0")


def test_split_rows_are_balanced_and_each_fits_the_bound():
    (whole,) = split_parameter_specs([_spec(MATRIX, (3, 4))], 16)
    assert whole.is_whole and whole.tensor_name == "actor::tp0::" + MATRIX
    slices = split_parameter_specs([_spec(EMBEDDING, (10, 4))], 16)
    assert [(unit.row_start, unit.row_stop) for unit in slices] == [(0, 3), (3, 6), (6, 10)]
    assert [unit.flat_range for unit in slices] == [(0, 12), (12, 24), (24, 40)]
    assert all(unit.numel <= 16 for unit in slices)
    assert slices[1].tensor_name == f"actor::tp0::{EMBEDDING}#rows=3:6"
    assert slices[1].shape == (3, 4)
    # Qwen3.8-27B at TP2: 124160 vocabulary rows of 5120 per rank, 2 GiB bound.
    big = split_parameter_specs([_spec(EMBEDDING, (124160, 5120))], (2 << 30) // 4)
    assert len(big) == 2 and all(unit.numel * 4 <= 2 << 30 for unit in big)
    with pytest.raises(ValueError, match="one dimension-0 row"):
        split_parameter_specs([_spec(EMBEDDING, (2, 20))], 16)


def test_layouts_that_split_nothing_keep_their_hash():
    specs = [_spec(MATRIX, (3, 4)), _spec(NORM, (4,)), _spec(EMBEDDING, (10, 4))]
    for merge in ("avg", "rda"):
        plain = ParameterLayout.create(
            algorithm="grpo", components=(_component(),), specs=specs, num_fragments=3,
            fragment_strategy="owner_affine", matrix_merge=merge,
        )
        bounded = ParameterLayout.create(
            algorithm="grpo", components=(_component(),), specs=specs, num_fragments=3,
            fragment_strategy="owner_affine", matrix_merge=merge, max_fragment_numel=40,
        )
        assert bounded.layout_hash == plain.layout_hash
        assert not bounded.has_split_parameters
    large = _adapter(bound=4096)
    assert not large.layout.has_split_parameters
    assert all(fragment.flat_ranges == () for plan in large.plans for fragment in plan.fragments)


def test_split_layout_changes_the_hash_and_installs_on_the_miles_check():
    adapter = _adapter()
    layout = adapter.layout
    assert layout.has_split_parameters
    assert layout.layout_hash != _adapter(bound=4096).layout.layout_hash
    assert all(fragment.numel * 4 <= BOUND_BYTES for fragment in layout.fragments.fragments)
    # Every rank's vocabulary shard spans three fragments, each one averaged.
    split = [
        fragment_id for fragment_id in range(layout.fragments.num_fragments)
        if any(not unit.is_whole for unit in layout.fragment_slices(fragment_id))
    ]
    assert len(split) == 6
    assert all(layout.fragments.fragments[f].merge_mode == MERGE_AVG for f in split)
    for plan in adapter.plans:
        owner = plan.topology.tp_rank
        _validate_fragment_plan_against_manifest(plan, _manifest(owner))
        assert sum(fragment.numel for fragment in plan.fragments) == sum(
            int(np.prod(shape)) for shape in SHAPES.values()
        )
    # Plans and layout are deterministic.
    assert _adapter().layout.layout_hash == layout.layout_hash
    assert [plan.plan_hash for plan in _adapter().plans] == [plan.plan_hash for plan in adapter.plans]


def test_rda_split_matrix_keeps_rda_per_row_slice():
    shapes = {EMBEDDING: (2, 4), MATRIX: (10, 4), NORM: (4,)}
    layout = _adapter(matrix_merge="rda", shapes=shapes).layout
    modes = {
        unit.tensor_name: layout.fragments.fragments[fragment_id].merge_mode
        for fragment_id in range(layout.fragments.num_fragments)
        for unit in layout.fragment_slices(fragment_id)
    }
    matrix_slices = [name for name in modes if MATRIX in name]
    assert len(matrix_slices) == 6 and all("#rows=" in name for name in matrix_slices)
    assert {modes[name] for name in matrix_slices} == {MERGE_RDA}
    assert {mode for name, mode in modes.items() if EMBEDDING in name} == {MERGE_AVG}


# -- real Rust syncer -------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CARGO = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
LEARNERS = 2
needs_cargo = pytest.mark.skipif(not Path(CARGO).exists(), reason="needs cargo to build the syncer")


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    subprocess.run([CARGO, "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


@pytest.fixture(autouse=True)
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "split-parameter-test-key")


def _start_syncer(binary: Path, tmp_path: Path, fragments: int, *extra: str):
    from yeto import launcher

    args = _prepared_args(fragments, *extra)
    args.quorum = LEARNERS  # a strict sweep waits for both learners
    command = launcher.role_syncer_commands(args, LEARNERS, binary=str(binary))["actor"]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = (command
               .replace(f"--port {launcher.SYNCER_PORT}", f"--port {port}")
               .replace(launcher.RL_SYNCER_CHECKPOINT, str(tmp_path / "state.ckpt"))
               .replace(launcher.RL_SYNCER_EVENT_TAPE, str(tmp_path / "tape.jsonl")))
    process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    return process, port, launcher.rl_syncer_profile(args, LEARNERS).sha256


def _prepared_args(fragments: int, *extra: str):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (
        "--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
        "--rl-rollout-gpus", "1", "--rl-parameter-layout-sha256", "e" * 64,
        "--fragments", str(fragments), "--total-steps", "1", *extra),
        gpu="modal:2xh100")
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def test_fragment_max_bytes_flag_reaches_the_island_and_is_checked():
    from yeto.launch.rl_island_task import _rl_trainable_island_flags

    default = _rl_trainable_island_flags(_prepared_args(2), 0, 1)
    assert "--sweep-fragment-max-bytes" not in default
    island = _rl_trainable_island_flags(
        _prepared_args(2, "--rl-sync-fragment-max-bytes", "1073741824"), 0, 1)
    assert "--sweep-fragment-max-bytes 1073741824" in island
    for bad in ("6", "4294967296"):
        with pytest.raises(ValueError, match="--rl-sync-fragment-max-bytes a multiple of 4"):
            _prepared_args(2, "--rl-sync-fragment-max-bytes", bad)


def _wait(predicate, what: str, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise TimeoutError(what)


def _rda(deltas: list[np.ndarray]) -> np.ndarray:
    norms = [float(np.linalg.norm(d.astype(np.float64))) for d in deltas]
    direction = sum(d.astype(np.float64) / n for d, n in zip(deltas, norms)) / len(deltas)
    return direction * (sum(norms) / len(norms)) / np.linalg.norm(direction)


def _sweep(binary, tmp_path, layout, *, dtype=DTYPE_F32, extra=(), push_i8_block=128):
    """One strict sweep of two learners; returns (initial, deltas, finals)."""

    fragments = layout.fragments
    process, port, profile = _start_syncer(binary, tmp_path, fragments.num_fragments, *extra)
    rng = np.random.default_rng(7)
    initial = {f: rng.standard_normal(frag.numel).astype("<f4")
               for f, frag in enumerate(fragments.fragments)}
    deltas = {(learner, f): rng.standard_normal(frag.numel).astype("<f4")
              for learner in range(LEARNERS) for f, frag in enumerate(fragments.fragments)}
    clients = [
        SyncerClient(("127.0.0.1", port), learner, fragments, dtype=dtype, num_streams=1,
                     connect_timeout=10, max_reconnects=0, finalization_timeout=30,
                     syncer_profile_hash=bytes.fromhex(profile), push_i8_block=push_i8_block)
        for learner in range(LEARNERS)
    ]
    try:
        for client in clients:
            client.start()
        for f in range(fragments.num_fragments):
            clients[0].send_init(f, initial[f].tobytes())
        pushed = {learner: set() for learner in range(LEARNERS)}

        def drive():
            for learner, client in enumerate(clients):
                client.check_health()
                for permit in client.drain_pulls():
                    client.push_fragment(permit.fragment_id, permit.global_step,
                                         permit.round_attempt, 0, 1, 1, 7,
                                         deltas[(learner, permit.fragment_id)].tobytes())
                    pushed[learner].add(permit.fragment_id)
            return all(len(done) == fragments.num_fragments for done in pushed.values())

        _wait(drive, "the syncer did not permit every fragment")
        results = [client.wait_for_final_fragments(timeout=30) for client in clients]
        _final_manifest, finals = results[0]
        merged = {fragment.fragment_id: np.frombuffer(bytes(fragment.data), dtype="<f4")
                  for fragment in finals}
        for client, (each, _finals) in zip(clients, results):
            client.acknowledge_finalization(each)
        assert process.wait(timeout=15) == 0
    finally:
        for client in clients:
            client.close()
        if process.poll() is None:
            process.terminate()
        log, _ = process.communicate(timeout=10)
    assert "syncer semantic profile" not in log, log
    return initial, deltas, merged


def _expected(layout, initial, deltas, f):
    """numpy merge of fragment ``f``: one merge per fragment tensor (slice)."""

    fragment = layout.fragments.fragments[f]
    out, offset = [], 0
    for _name, numel in fragment.tensors:
        parts = [deltas[(learner, f)][offset:offset + numel] for learner in range(LEARNERS)]
        merged = (sum(p.astype(np.float64) for p in parts) / LEARNERS
                  if fragment.merge_mode == MERGE_AVG else _rda(parts))
        out.append(merged)
        offset += numel
    return initial[f].astype(np.float64) + np.concatenate(out)


@needs_cargo
@pytest.mark.parametrize("matrix_merge", ["avg", "rda"])
def test_real_syncer_merges_split_slices_like_numpy(syncer_binary, tmp_path, matrix_merge):
    shapes = {EMBEDDING: (10, 4), MATRIX: (10, 4), NORM: (4,)}
    layout = _adapter(matrix_merge=matrix_merge, shapes=shapes).layout
    assert layout.has_split_parameters
    initial, deltas, merged = _sweep(syncer_binary, tmp_path, layout)
    assert set(merged) == set(range(layout.fragments.num_fragments))
    for f, value in merged.items():
        want = _expected(layout, initial, deltas, f)
        np.testing.assert_allclose(value[-want.size:], want, rtol=0, atol=1e-5, err_msg=str(f))


@needs_cargo
def test_real_syncer_int8_pushes_quantize_each_slice_as_its_own_segment(syncer_binary, tmp_path):
    layout = _adapter().layout
    initial, deltas, merged = _sweep(
        syncer_binary, tmp_path, layout, dtype=DTYPE_F32_PUSH_I8 | DTYPE_FLAG_TENSOR_BLOCKS,
        extra=("--rl-sync-push-dtype", "int8", "--rl-sync-push-int8-block", "32"),
        push_i8_block=32,
    )
    for f, value in merged.items():
        want = _expected(layout, initial, deltas, f)
        # Each delta rounds within half its block scale (absmax/127).
        bound = max(np.abs(deltas[(learner, f)]).max() for learner in range(LEARNERS)) / 127
        np.testing.assert_allclose(value[-want.size:], want, rtol=0, atol=bound + 1e-6)
    assert any(not np.allclose(value[-want.size:], _expected(layout, initial, deltas, f),
                               rtol=0, atol=1e-7)
               for f, value in merged.items()), "int8 pushes must actually quantize"


def test_quantized_segments_follow_the_slices():
    layout = _adapter().layout
    client = SyncerClient(("127.0.0.1", 9), 0, layout.fragments,
                          dtype=DTYPE_F32_PUSH_I8 | DTYPE_FLAG_TENSOR_BLOCKS,
                          num_streams=1, push_i8_block=32)
    for f, fragment in enumerate(layout.fragments.fragments):
        assert client.push_segments(f) == [numel for _name, numel in fragment.tensors]
