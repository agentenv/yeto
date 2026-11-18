"""Restoring a round cut on a multi-node island whose staged store has one stage per
node (SSH / Modal): the reload leaves the rank shards in the object store
(``s3_store.LAZY_CUT_SHARDS_ENV``), the driver checks them by HEAD, and each rank
fetches only the shard it reads (``cut.fetch_shard_to_rank``) and drops it once
loaded. CPU nodes are separate stage directories over one fake bucket."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_rl_cut import _expect
from test_rl_multinode_staged_cut import CUT, _cut_manifest, _env, _two_node_cut
from test_rl_object_store import URI, FakeS3

from yeto.cloud import s3_store
from yeto.rl.engine.cut import (
    MANIFEST_NAME,
    CutError,
    commit_manifest,
    cut_dir,
    drop_fetched_shard,
    fetch_shard_to_rank,
    manifest_shard_keys,
    uploaded_shard_sizes,
    verify_cut,
)
from yeto.rl.engine.object_store import PUSHED_STATE
from yeto.rl.engine.resume import ROUND_CUTS, store_for

LAYOUT2 = {"world": 2, "tp": 1, "pp": 1, "cp": 1, "ep": 1, "dp": 2}


def _lazy_env(tmp_path, node, **extra):
    return {**_env(tmp_path, node), s3_store.LAZY_CUT_SHARDS_ENV: "1", **extra}


def _committed(tmp_path, s3):
    head, _, summaries = _two_node_cut(tmp_path, s3)
    manifest = _cut_manifest(head, summaries)
    commit_manifest(head.root / ROUND_CUTS, manifest, remote_sizes=uploaded_shard_sizes(head, summaries))
    (head.root / "LATEST").write_text(json.dumps({"seq": 0}))
    head.commit()
    return manifest


def _fresh(tmp_path, node, s3, **extra):
    store = store_for(tmp_path / "unused-mount", environ=_lazy_env(tmp_path, node, **extra), client=s3)
    store.reload()
    return store


def test_a_lazy_reload_takes_the_manifest_and_leaves_the_shards_to_their_ranks(tmp_path, monkeypatch):
    s3 = FakeS3()
    manifest = _committed(tmp_path, s3)
    fresh = _fresh(tmp_path, "node-c", s3)
    directory = cut_dir(fresh.root / ROUND_CUTS, CUT)
    assert (directory / MANIFEST_NAME).is_file() and (fresh.root / "LATEST").is_file()
    assert not list(directory.glob("*.pt"))  # node 0 no longer downloads every node's shards
    # the driver's check: the shards it cannot see pass by their size in the object store
    for key, value in _lazy_env(tmp_path, "node-c").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(s3_store, "_client", lambda env, require_credentials: s3)
    assert verify_cut(fresh.root / ROUND_CUTS, CUT, _expect(layout=LAYOUT2)) == manifest


def test_each_rank_fetches_its_own_shard_and_drops_it_after_the_load(tmp_path):
    s3 = FakeS3()
    _committed(tmp_path, s3)
    for node, dp in (("node-c", 0), ("node-d", 1)):
        env = _lazy_env(tmp_path, node, YETO_RL_CUT_SHARD_EVICT="1")
        store = _fresh(tmp_path, node, s3, YETO_RL_CUT_SHARD_EVICT="1")
        path = cut_dir(store.root / ROUND_CUTS, CUT) / f"trainer_tp0_pp0_dp{dp}.pt"
        assert fetch_shard_to_rank(path, environ=env, store_kwargs={"client": s3}) == 100
        assert path.read_bytes() == bytes([dp]) * 100
        other = cut_dir(store.root / ROUND_CUTS, CUT) / f"trainer_tp0_pp0_dp{1 - dp}.pt"
        assert not other.exists()  # only the shard this rank reads
        key = f"{ROUND_CUTS}/{CUT}/{path.name}"
        assert key in json.loads((store.root / PUSHED_STATE).read_text())  # no push sends it back
        assert fetch_shard_to_rank(path, environ=env, store_kwargs={"client": s3}) is None  # here already
        assert drop_fetched_shard(path, environ=env, store_kwargs={"client": s3}) == 100
        assert not path.exists() and key not in json.loads((store.root / PUSHED_STATE).read_text())
        store.commit()  # the dropped file is no prune: the object stays
        assert ("bkt", f"runs/a/{key}") in s3.objects


def test_a_shard_missing_from_the_store_refuses_the_restore(tmp_path, monkeypatch):
    s3 = FakeS3()
    _committed(tmp_path, s3)
    del s3.objects[("bkt", f"runs/a/{ROUND_CUTS}/{CUT}/trainer_tp0_pp0_dp1.pt")]
    fresh = _fresh(tmp_path, "node-c", s3)
    for key, value in _lazy_env(tmp_path, "node-c").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(s3_store, "_client", lambda env, require_credentials: s3)
    with pytest.raises(CutError, match="trainer_tp0_pp0_dp1.pt \\(neither on this disk nor in the object store\\)"):
        verify_cut(fresh.root / ROUND_CUTS, CUT, _expect(layout=LAYOUT2))
    path = cut_dir(fresh.root / ROUND_CUTS, CUT) / "trainer_tp0_pp0_dp1.pt"
    with pytest.raises(Exception, match="not in the object store"):
        fetch_shard_to_rank(path)


def test_without_the_lazy_flag_or_a_staged_store_nothing_changes(tmp_path):
    s3 = FakeS3()
    _committed(tmp_path, s3)
    full = store_for(tmp_path / "unused-mount", environ=_env(tmp_path, "node-e"), client=s3)
    full.reload()  # one node: every shard comes with the reload, as before
    assert len(list(cut_dir(full.root / ROUND_CUTS, CUT).glob("*.pt"))) == 2
    plain = tmp_path / "shared" / ROUND_CUTS / CUT / "trainer_tp0_pp0_dp0.pt"
    assert fetch_shard_to_rank(plain, environ={}) is None
    assert drop_fetched_shard(plain, environ={}) == 0


def test_manifest_shard_keys_lists_only_committed_shards():
    data = {f"{ROUND_CUTS}/a/{MANIFEST_NAME}": json.dumps({"files": [{"path": "s0.pt"}, {"path": "s1.pt"}]}),
            f"{ROUND_CUTS}/c/{MANIFEST_NAME}": "not json"}

    class Objects:
        def read(self, key):
            value = data.get(key)
            return None if value is None else value.encode()

    keys = [f"{ROUND_CUTS}/a/{MANIFEST_NAME}", f"{ROUND_CUTS}/a/s0.pt", f"{ROUND_CUTS}/a/s1.pt",
            f"{ROUND_CUTS}/a/critic/x.pt", f"{ROUND_CUTS}/b/s0.pt", f"{ROUND_CUTS}/c/{MANIFEST_NAME}",
            f"{ROUND_CUTS}/c/s0.pt"]
    assert manifest_shard_keys(Objects(), keys) == {f"{ROUND_CUTS}/a/s0.pt", f"{ROUND_CUTS}/a/s1.pt"}


def test_the_launch_makes_a_multi_node_staged_store_lazy(monkeypatch):
    from yeto.launch.storage import store_object_envs

    monkeypatch.setenv(s3_store.KEY_ID_ENV, "k")
    monkeypatch.setenv(s3_store.SECRET_ENV, "s")
    trainer = SimpleNamespace(rl_checkpoint_store=URI, rl_sample_role="trainer")
    assert store_object_envs(trainer, "ssh", num_nodes=6)[s3_store.LAZY_CUT_SHARDS_ENV] == "1"
    assert store_object_envs(trainer, "modal", num_nodes=2)[s3_store.LAZY_CUT_SHARDS_ENV] == "1"
    assert s3_store.LAZY_CUT_SHARDS_ENV not in store_object_envs(trainer, "ssh", num_nodes=1)
    assert s3_store.LAZY_CUT_SHARDS_ENV not in store_object_envs(trainer, "modal", num_nodes=1)
    assert s3_store.LAZY_CUT_SHARDS_ENV not in store_object_envs(trainer, "nebius", num_nodes=2)  # bucket mount


def test_the_miles_rank_restore_fetches_a_shard_another_node_wrote(tmp_path, monkeypatch):
    """cut_plugin._load_verified: the shard is fetched, verified and loaded, then dropped."""
    torch = pytest.importorskip("torch")
    from yeto.rl.adapters.miles import cut_plugin
    from yeto.rl.engine.cut import CutFile, sha256_file

    s3 = FakeS3()
    writer = store_for(tmp_path / "unused-mount", environ=_env(tmp_path, "node-a"), client=s3)
    directory = cut_dir(writer.root / ROUND_CUTS, CUT)
    directory.mkdir(parents=True)
    shard = directory / "trainer_tp0_pp0_dp1.pt"
    torch.save({"schema": cut_plugin.SHARD_SCHEMA, "cut_id": CUT, "coord": {"tp": 0, "pp": 0, "dp": 1}}, shard)
    entry = CutFile(shard.name, sha256_file(shard), shard.stat().st_size, {"tp": 0, "pp": 0, "dp": 1}).to_dict()
    writer.push_local_file(f"{ROUND_CUTS}/{CUT}/{shard.name}")
    env = _lazy_env(tmp_path, "node-b", YETO_RL_CUT_SHARD_EVICT="1")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(s3_store, "_client", lambda env, require_credentials: s3)
    reader = store_for(tmp_path / "unused-mount", environ=env, client=s3)
    loaded = cut_plugin._load_verified(str(cut_dir(reader.root / ROUND_CUTS, CUT)), entry, CUT)
    assert loaded["coord"] == {"tp": 0, "pp": 0, "dp": 1}
    assert not (cut_dir(reader.root / ROUND_CUTS, CUT) / shard.name).exists()  # dropped after the load
