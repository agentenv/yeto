"""YETO_CACHE_DIR moves the default release cache root (a process with another HOME shares the
node's cache); unset, or an explicit cache_dir, behaves as before."""
from __future__ import annotations

from pathlib import Path

from yeto.rl.engine import policy_release as pr
from yeto.rl.engine.resume import CheckpointStore


def test_unset_keeps_the_home_default(monkeypatch, tmp_path):
    monkeypatch.delenv(pr.CACHE_DIR_ENV, raising=False)
    store = CheckpointStore(tmp_path / "store")
    assert pr.default_cache_dir() == Path(pr.DEFAULT_CACHE_DIR).expanduser()
    assert pr.PolicyReleaseStore(store).cache_dir == Path(pr.DEFAULT_CACHE_DIR).expanduser() / pr.store_cache_key(store)
    assert pr.default_cache_dir({pr.CACHE_DIR_ENV: "  "}) == Path(pr.DEFAULT_CACHE_DIR).expanduser()


def test_env_moves_the_default_root_and_keeps_the_per_store_key(monkeypatch, tmp_path):
    monkeypatch.setenv(pr.CACHE_DIR_ENV, str(tmp_path / "shared-cache"))
    store = CheckpointStore(tmp_path / "store")
    expected = tmp_path / "shared-cache" / "policy-releases" / pr.store_cache_key(store)
    assert pr.PolicyReleaseStore(store).cache_dir == expected
    # a scheme's own release dir stays a sub-directory of it
    assert pr.PolicyReleaseStore(store, release_dir="policy-releases-w4a8").cache_dir == expected / "policy-releases-w4a8"


def test_an_explicit_cache_dir_wins_over_the_env(monkeypatch, tmp_path):
    monkeypatch.setenv(pr.CACHE_DIR_ENV, str(tmp_path / "shared-cache"))
    store = CheckpointStore(tmp_path / "store")
    assert pr.PolicyReleaseStore(store, cache_dir=tmp_path / "given").cache_dir == tmp_path / "given"
