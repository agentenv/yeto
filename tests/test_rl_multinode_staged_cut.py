"""A round cut on a multi-node island whose staged s3 store has one stage per node
(Modal: no bucket mount, yeto.launch.storage). The driver's commit only sees its own
node's disk, so each rank uploads its own shard (cut.push_shard_from_rank), the
manifest carries the object keys and sizes, and the driver checks the shards it
cannot see by HEAD before it commits the manifest and the pointer. A fresh node
restores both shards; a prune deletes the shards every node uploaded. CPU nodes are
separate stage directories over one fake bucket."""
from __future__ import annotations

import json

import pytest
from test_rl_cut import LAYOUT, _manifest
from test_rl_object_store import URI, FakeS3

from yeto.cloud import s3_store
from yeto.rl.engine.cut import (
    MANIFEST_NAME,
    CutError,
    CutFile,
    commit_manifest,
    cut_dir,
    push_shard_from_rank,
    sha256_file,
    uploaded_shard_sizes,
)
from yeto.rl.engine.object_store import PUSHED_STATE
from yeto.rl.engine.resume import ROUND_CUTS, STORE_IMPL_ENV, store_for

CUT = "r000002-abc"


def _env(tmp_path, node):
    return {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: URI,
            s3_store.STAGE_ENV: str(tmp_path / node), "YETO_RL_CUT_SHARD_EVICT": "0"}  # kept: checked below


def _store(tmp_path, node, s3):
    return store_for(tmp_path / "unused-mount", environ=_env(tmp_path, node), client=s3)


def _write_shard(store, dp, size=100):
    directory = cut_dir(store.root / ROUND_CUTS, CUT)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"trainer_tp0_pp0_dp{dp}.pt"
    final.write_bytes(bytes([dp]) * size)
    return final


def _summary(final, pushed):
    return {"path": final.name, "sha256": sha256_file(final), "bytes": final.stat().st_size,
            "coord": {"tp": 0, "pp": 0, "dp": int(final.stem[-1])}, **(pushed or {})}


def _two_node_cut(tmp_path, s3):
    head, other = _store(tmp_path, "node-a", s3), _store(tmp_path, "node-b", s3)
    local = _write_shard(head, 0)
    remote = _write_shard(other, 1)
    pushed_local = push_shard_from_rank(local, environ=_env(tmp_path, "node-a"), store_kwargs={"client": s3})
    pushed_remote = push_shard_from_rank(remote, environ=_env(tmp_path, "node-b"), store_kwargs={"client": s3})
    return head, other, [_summary(local, pushed_local), _summary(remote, pushed_remote)]


def _cut_manifest(head, summaries):
    files = tuple(CutFile(s["path"], s["sha256"], s["bytes"], s["coord"]) for s in summaries)
    shard = cut_dir(head.root / ROUND_CUTS, CUT) / "trainer_tp0_pp0_dp0.pt"
    data = shard.read_bytes()  # the fixture below writes a shard of that name: keep the rank's
    manifest = _manifest(head.root / ROUND_CUTS, cut_id=CUT, files=files,
                     runtime={"backend_fingerprint": "miles@0af62f4d", "layout": {**LAYOUT, "dp": 2,
                                                                                 "world": 2},
                              "rng_policy": "exact"},
                     rank_summaries=tuple({"path": s["path"], "scheduler_samples": 8, "has_optimizer_state": True,
                                           "has_rng": True, "state_digest": "s", "rng_digest": "r",
                                           "object_key": s.get("object_key")} for s in summaries))
    shard.write_bytes(data)
    return manifest


def test_each_rank_uploads_its_shard_and_records_it_on_its_node(tmp_path):
    s3 = FakeS3()
    _, other, summaries = _two_node_cut(tmp_path, s3)
    key = f"{ROUND_CUTS}/{CUT}/trainer_tp0_pp0_dp1.pt"
    assert summaries[1]["object_key"] == key and summaries[1]["object_bytes"] == 100
    assert s3.objects[("bkt", f"runs/a/{key}")] == bytes([1]) * 100
    assert key in json.loads((other.root / PUSHED_STATE).read_text())  # node B's push skips it


def test_the_driver_commits_a_cut_whose_shards_are_on_other_nodes_after_a_head_check(tmp_path):
    s3 = FakeS3()
    head, _, summaries = _two_node_cut(tmp_path, s3)
    manifest = _cut_manifest(head, summaries)
    with pytest.raises(CutError, match="missing shard trainer_tp0_pp0_dp1.pt"):
        commit_manifest(head.root / ROUND_CUTS, manifest)  # the old check: only this disk
    sizes = uploaded_shard_sizes(head, summaries)
    assert sizes == {"trainer_tp0_pp0_dp0.pt": 100, "trainer_tp0_pp0_dp1.pt": 100}
    commit_manifest(head.root / ROUND_CUTS, manifest, remote_sizes=sizes)
    assert (cut_dir(head.root / ROUND_CUTS, CUT) / MANIFEST_NAME).is_file()

    # the driver's commit uploads the manifest and the pointer (node A's shard is pushed already)
    (head.root / "LATEST").write_text(json.dumps({"seq": 0}))
    head.commit()
    assert ("bkt", f"runs/a/{ROUND_CUTS}/{CUT}/{MANIFEST_NAME}") in s3.objects
    assert ("bkt", "runs/a/LATEST") in s3.objects

    # a fresh node restores every shard, whichever node wrote it
    fresh = _store(tmp_path, "node-c", s3)
    fresh.reload()
    for dp in (0, 1):
        assert (cut_dir(fresh.root / ROUND_CUTS, CUT) / f"trainer_tp0_pp0_dp{dp}.pt").read_bytes() == bytes([dp]) * 100


@pytest.mark.parametrize("damage, match", [
    ("missing", "neither on this disk nor in the object store"),
    ("short", "bytes in the object store, manifest says 100"),
    ("failed", "neither on this disk nor in the object store"),
])
def test_a_shard_from_another_node_that_is_not_whole_in_the_store_refuses_the_cut(tmp_path, damage, match):
    s3 = FakeS3()
    head, _, summaries = _two_node_cut(tmp_path, s3)
    key = ("bkt", f"runs/a/{ROUND_CUTS}/{CUT}/trainer_tp0_pp0_dp1.pt")
    if damage == "missing":
        del s3.objects[key]
    elif damage == "short":
        s3.objects[key] = b"x" * 10
    else:
        summaries[1] = {k: v for k, v in summaries[1].items() if not k.startswith("object_")}
        summaries[1]["object_push_error"] = "ClientError: boom"
    manifest = _cut_manifest(head, summaries)
    with pytest.raises(CutError, match=match):
        commit_manifest(head.root / ROUND_CUTS, manifest, remote_sizes=uploaded_shard_sizes(head, summaries))
    assert not (cut_dir(head.root / ROUND_CUTS, CUT) / MANIFEST_NAME).exists()


def test_a_failed_upload_is_reported_not_raised_and_other_stores_push_nothing(tmp_path):
    class Broken(FakeS3):
        def put_object(self, *a, **kw):
            raise RuntimeError("no network")

    store = _store(tmp_path, "node-b", Broken())
    final = _write_shard(store, 1)
    got = push_shard_from_rank(final, environ=_env(tmp_path, "node-b"), store_kwargs={"client": Broken()})
    assert got is not None and "no network" in got["object_push_error"]
    # no store implementation (a shared mount / plain directory): nothing to do
    assert push_shard_from_rank(final, environ={}) is None
    # switched off
    off = {**_env(tmp_path, "node-b"), "YETO_RL_CUT_SHARD_PUSH": "0"}
    assert push_shard_from_rank(final, environ=off) is None
    # a file outside this node's stage
    outside = tmp_path / "elsewhere" / "trainer_tp0_pp0_dp1.pt"
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    assert push_shard_from_rank(outside, environ=_env(tmp_path, "node-b"), store_kwargs={"client": FakeS3()}) is None
    # a single-node island needs nothing remote: no rank reported an upload
    assert uploaded_shard_sizes(store, [{"path": "a.pt"}]) is None


def test_a_prune_deletes_the_shards_every_node_uploaded(tmp_path):
    s3 = FakeS3()
    head, _, _ = _two_node_cut(tmp_path, s3)
    deleted = head.delete_remote_tree(f"{ROUND_CUTS}/{CUT}")
    assert sorted(deleted) == [f"{ROUND_CUTS}/{CUT}/trainer_tp0_pp0_dp0.pt", f"{ROUND_CUTS}/{CUT}/trainer_tp0_pp0_dp1.pt"]
    assert not [k for k in s3.objects if "/round-cuts/" in k[1]]
    assert not [k for k in json.loads((head.root / PUSHED_STATE).read_text()) if k.startswith(ROUND_CUTS)]


def test_a_rank_drops_its_local_shard_after_the_upload_and_no_push_prunes_it(tmp_path):
    s3 = FakeS3()
    env = {**_env(tmp_path, "node-b"), "YETO_RL_CUT_SHARD_EVICT": "1"}
    other = _store(tmp_path, "node-b", s3)
    final = _write_shard(other, 1)
    pushed = push_shard_from_rank(final, environ=env, store_kwargs={"client": s3})
    key = f"{ROUND_CUTS}/{CUT}/trainer_tp0_pp0_dp1.pt"
    assert pushed["object_key"] == key and pushed["object_bytes"] == 100
    assert not final.exists()  # a K3.1 cut is ~1.35 TB per node: it must not stay on the disk
    other.commit()  # the node's next push does not take the dropped file for a prune
    assert s3.objects[("bkt", f"runs/a/{key}")] == bytes([1]) * 100
