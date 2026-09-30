"""rl-infra-spec 4.2: ReconfigurationCut manifest (CPU)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from yeto.rl.engine.cut import (
    MANIFEST_NAME,
    AlgorithmIdentity,
    CutError,
    CutFile,
    CutManifest,
    CutProgress,
    RestoreExpectation,
    commit_manifest,
    cut_dir,
    sha256_file,
    verify_cut,
)

ALGO = AlgorithmIdentity("a" * 64, ("yeto.rl.algos.x@" + "b" * 64,), "c" * 64,
                         {"ref_load": "/m/ref", "base_model_revision": "rev1"})
LAYOUT = {"world": 1, "tp": 1, "pp": 1, "cp": 1, "ep": 1, "dp": 1}


def _manifest(tmp_path, cut_id="cut-1", **overrides):
    d = cut_dir(tmp_path, cut_id)
    d.mkdir(parents=True, exist_ok=True)
    shard = d / "trainer_tp0_pp0_dp0.pt"
    shard.write_bytes(b"x" * 100)
    base = dict(
        cut_id=cut_id,
        epoch=3,
        runtime={"backend_fingerprint": "miles@0af62f4d", "layout": LAYOUT, "rng_policy": "exact"},
        progress=CutProgress(local_step=2, scheduler_samples=8, global_batch_size=4, next_rollout_id=2,
                             policy_version=2, policy_hash="h" * 64),
        algorithm=ALGO,
        data={"sample_offset": 16, "epoch_id": 0, "sample_group_index": 16, "sample_index": 128},
        ledger={"carried_over": 0, "ready_unconsumed": 0, "optimizer_applied": 2},
        outer={"settled": True, "round": 1},
        files=(CutFile(shard.name, sha256_file(shard), 100, {"tp": 0, "pp": 0, "dp": 0}),),
        rank_summaries=({"path": shard.name, "scheduler_samples": 8, "has_optimizer_state": True,
                         "has_rng": True, "state_digest": "s", "rng_digest": "r"},),
    )
    base.update(overrides)
    return CutManifest(**base)


def _expect(**overrides):
    base = dict(algorithm=ALGO, layout=LAYOUT, backend_fingerprint="miles@0af62f4d", local_step=2,
                policy_version=2, epoch=3)
    base.update(overrides)
    return RestoreExpectation(**base)


def test_commit_then_verify_roundtrip(tmp_path):
    m = _manifest(tmp_path)
    commit_manifest(tmp_path, m)
    assert verify_cut(tmp_path, "cut-1", _expect()) == m
    with pytest.raises(CutError, match="immutable"):
        commit_manifest(tmp_path, m)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"data": {"sample_offset": 1}}, "cursor field"),
        ({"ledger": {"carried_over": 1, "ready_unconsumed": 0}}, "carried_over"),
        ({"ledger": {"carried_over": 0, "ready_unconsumed": 2}}, "ready-unconsumed"),
        ({"outer": {"settled": False}}, "not settled"),
        ({"rank_summaries": ()}, "rank_summaries"),
        ({"progress": CutProgress(2, 12, 4, 2, 2, "h")}, "scheduler progress"),
        ({"runtime": {"layout": LAYOUT}}, "backend_fingerprint"),
    ],
)
def test_incomplete_cut_is_refused(tmp_path, overrides, match):
    with pytest.raises(CutError, match=match):
        commit_manifest(tmp_path, _manifest(tmp_path, **overrides))
    assert not (cut_dir(tmp_path, "cut-1") / MANIFEST_NAME).exists()


def test_missing_optimizer_or_rng_on_a_rank_is_refused(tmp_path):
    m = _manifest(tmp_path)
    bad = ({**m.rank_summaries[0], "has_rng": False},)
    with pytest.raises(CutError, match="optimizer state or RNG"):
        commit_manifest(tmp_path, replace(m, rank_summaries=bad))


def test_bad_checksum_and_truncation_are_refused(tmp_path):
    commit_manifest(tmp_path, _manifest(tmp_path))
    shard = cut_dir(tmp_path, "cut-1") / "trainer_tp0_pp0_dp0.pt"
    shard.write_bytes(b"y" * 100)
    with pytest.raises(CutError, match="checksum"):
        verify_cut(tmp_path, "cut-1", _expect())
    shard.write_bytes(b"x" * 50)
    with pytest.raises(CutError, match="truncated"):
        verify_cut(tmp_path, "cut-1", _expect())
    shard.unlink()
    with pytest.raises(CutError, match="missing shard"):
        verify_cut(tmp_path, "cut-1", _expect())


def test_edited_or_truncated_manifest_is_refused(tmp_path):
    path = commit_manifest(tmp_path, _manifest(tmp_path))
    raw = json.loads(path.read_text())
    raw["progress"]["local_step"] = 3
    path.write_text(json.dumps(raw))
    with pytest.raises(CutError, match="hash mismatch"):
        verify_cut(tmp_path, "cut-1", _expect())
    path.write_text(path.read_text()[:40])
    with pytest.raises(CutError, match="not JSON"):
        verify_cut(tmp_path, "cut-1", _expect())


def test_uncommitted_cut_does_not_exist(tmp_path):
    _manifest(tmp_path)  # shards only, no manifest
    with pytest.raises(CutError, match="no committed manifest"):
        verify_cut(tmp_path, "cut-1", _expect())


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"local_step": 3}, "local_step"),
        ({"policy_version": 1}, "policy_version"),
        ({"algorithm": replace(ALGO, algorithm_spec_sha256="d" * 64)}, "algorithm identity"),
        ({"algorithm": replace(ALGO, plugin_sha256=())}, "algorithm identity"),
        ({"algorithm": replace(ALGO, runtime_attrs_sha256=None)}, "algorithm identity"),
        ({"algorithm": replace(ALGO, ref_model={"ref_load": "/m/other", "base_model_revision": "rev1"})},
         "algorithm identity"),
        ({"layout": {**LAYOUT, "dp": 2}}, "parallel layout"),
        ({"backend_fingerprint": "miles@other"}, "fingerprint"),
        ({"epoch": 2}, "newer"),
    ],
)
def test_restore_mismatch_is_refused(tmp_path, overrides, match):
    commit_manifest(tmp_path, _manifest(tmp_path))
    with pytest.raises(CutError, match=match):
        verify_cut(tmp_path, "cut-1", _expect(**overrides))


def test_path_escape_and_bad_id_are_refused(tmp_path):
    with pytest.raises(CutError):
        CutFile.from_dict({"path": "../x", "sha256": "0", "bytes": 1})
    with pytest.raises(CutError):
        cut_dir(tmp_path, "../evil")


def test_identity_from_spec_collects_plugin_refs():
    from yeto.rl.engine.algorithm import AlgorithmSpec

    spec = AlgorithmSpec()
    ident = AlgorithmIdentity.from_spec(spec, runtime_attrs={"yeto_algo_plugins": {"k": 1}})
    assert ident.algorithm_spec_sha256 == spec.sha256()
    assert ident.plugin_sha256 == ()
    assert ident.runtime_attrs_sha256 is not None
    assert AlgorithmIdentity.from_spec(spec).runtime_attrs_sha256 is None


def test_iter_plugin_refs_walks_nested_dataclasses():
    from dataclasses import dataclass

    from yeto.rl.engine.algorithm import PluginRef
    from yeto.rl.engine.cut import iter_plugin_refs

    ref = PluginRef("yeto.rl.algos.reducers.x", "e" * 64)

    @dataclass(frozen=True)
    class Loss:
        reducer: object = ref

    @dataclass(frozen=True)
    class Spec:
        loss: object = Loss()
        plugins: tuple = (ref,)

        def sha256(self):
            return "f" * 64

    assert list(iter_plugin_refs(Spec())) == [ref]
    assert AlgorithmIdentity.from_spec(Spec()).plugin_sha256 == (f"yeto.rl.algos.reducers.x@{'e' * 64}",)
