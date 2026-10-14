"""rl-identity-tiering L9.4-L9.6: run code packs (no cloud; directory store)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from yeto import codepack
from yeto.codepack import CodePackError, DirPackStore


def _git(repo: Path, *argv: str) -> str:
    return subprocess.run(["git", *argv], cwd=repo, check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "yeto").mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / ".gitignore").write_text("ignored.txt\n")
    (root / "yeto" / "a.py").write_text("A = 1\n")
    (root / "run.sh").write_text("#!/bin/sh\n")
    (root / "run.sh").chmod(0o755)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def test_clean_pack_is_deterministic(repo: Path) -> None:
    first, data1 = codepack.build_pack(repo)
    (repo / "yeto" / "a.py").touch()  # mtime only
    second, data2 = codepack.build_pack(repo)
    assert data1 == data2 and first.digest == second.digest
    assert not first.dirty and first.commit == _git(repo, "rev-parse", "HEAD").strip()


def test_dirty_patch_includes_untracked_and_skips_ignored(repo: Path) -> None:
    clean, _ = codepack.build_pack(repo)
    (repo / "ignored.txt").write_text("x")
    assert codepack.build_pack(repo)[0].digest == clean.digest
    (repo / "yeto" / "new.py").write_text("NEW = 2\n")
    index_before = (repo / ".git" / "index").read_bytes()
    dirty, data = codepack.build_pack(repo)
    assert dirty.dirty and dirty.digest != clean.digest
    assert b"yeto/new.py" in codepack.dirty_patch(repo)
    assert (repo / ".git" / "index").read_bytes() == index_before  # real index untouched
    tree = codepack.unpack(data, dirty.digest, repo.parent / "cache")
    assert (tree / "yeto" / "new.py").read_text() == "NEW = 2\n"
    assert not (tree / "ignored.txt").exists()
    assert (tree / "run.sh").stat().st_mode & 0o111


def test_deleted_tracked_file_is_not_packed(repo: Path) -> None:
    (repo / "run.sh").unlink()
    pack, data = codepack.build_pack(repo)
    tree = codepack.unpack(data, pack.digest, repo.parent / "cache")
    assert pack.dirty and not (tree / "run.sh").exists()


def test_verify_tree_rejects_changed_tree(repo: Path) -> None:
    pack, data = codepack.build_pack(repo)
    tree = codepack.unpack(data, pack.digest, repo.parent / "cache")
    assert codepack.verify_tree(tree, pack.digest) == pack.digest
    assert codepack.main(["verify", "--dir", str(tree), "--digest", pack.digest]) == 0
    (tree / "yeto" / "a.py").write_text("A = 2\n")
    with pytest.raises(CodePackError, match="digest mismatch"):
        codepack.verify_tree(tree, pack.digest)
    assert codepack.main(["verify", "--dir", str(tree), "--digest", pack.digest]) == 1
    (tree / "yeto" / "a.py").unlink()
    with pytest.raises(CodePackError, match="missing"):
        codepack.verify_tree(tree, pack.digest)


def test_unpack_rejects_wrong_bytes(repo: Path) -> None:
    pack, data = codepack.build_pack(repo)
    with pytest.raises(CodePackError, match="does not have digest"):
        codepack.unpack(data + b"\0", pack.digest, repo.parent / "cache")


def test_resume_takes_the_manifest_pack_after_local_change(repo: Path, tmp_path: Path) -> None:
    store = DirPackStore(tmp_path / "store", uri_base="file-store://run")
    first = codepack.resolve_run_pack(previous=None, repo=repo, store=store,
                                      cache_root=tmp_path / "c1", semantic_sha256="s",
                                      log=lambda _m: None)
    assert not first.reused
    assert first.pack.pack_uri == f"file-store://run/code_packs/{first.pack.digest}.tar"
    (repo / "yeto" / "a.py").write_text("A = 99\n")  # local tree drifts after start
    manifest = {"code_pack": first.pack.record(), "semantic_sha256": "s"}
    again = codepack.resolve_run_pack(previous=manifest, repo=repo, store=store,
                                      cache_root=tmp_path / "c2", semantic_sha256="s",
                                      log=lambda _m: None)
    assert again.reused and again.pack == first.pack
    assert (again.tree / "yeto" / "a.py").read_text() == "A = 1\n"


def test_upgrade_needs_equal_semantic_hash(repo: Path, tmp_path: Path) -> None:
    store = DirPackStore(tmp_path / "store")
    first = codepack.resolve_run_pack(previous=None, repo=repo, store=store,
                                      cache_root=tmp_path / "c", semantic_sha256="s1",
                                      log=lambda _m: None)
    (repo / "yeto" / "a.py").write_text("A = 3\n")
    manifest = {"code_pack": first.pack.record(), "semantic_sha256": "s1"}
    with pytest.raises(CodePackError, match="semantic hash changes"):
        codepack.resolve_run_pack(previous=manifest, repo=repo, store=store,
                                  cache_root=tmp_path / "c", semantic_sha256="s2",
                                  upgrade_reason="fix", log=lambda _m: None)
    with pytest.raises(CodePackError, match="unknown"):
        codepack.resolve_run_pack(previous={"code_pack": first.pack.record()}, repo=repo,
                                  store=store, cache_root=tmp_path / "c",
                                  semantic_sha256="s1", upgrade_reason="fix",
                                  log=lambda _m: None)
    up = codepack.resolve_run_pack(previous=manifest, repo=repo, store=store,
                                   cache_root=tmp_path / "c", semantic_sha256="s1",
                                   upgrade_reason="logging fix", log=lambda _m: None)
    assert not up.reused and up.pack.digest != first.pack.digest
    (step,) = up.pack.upgrades
    assert (step["from"], step["to"], step["reason"]) == (first.pack.digest, up.pack.digest,
                                                          "logging fix")
    assert codepack.CodePack.from_record(up.pack.record()) == up.pack
    with pytest.raises(CodePackError, match="reason"):
        codepack.upgrade_record(first.pack, up.pack, old_semantic="s", new_semantic="s",
                                reason=" ")


def test_store_path_must_be_visible(tmp_path: Path) -> None:
    with pytest.raises(CodePackError, match="not visible"):
        codepack.pack_store_for(str(tmp_path / "absent"))
    with pytest.raises(CodePackError, match="no code pack store"):
        codepack.pack_store_for("gs://bucket/x")
    assert isinstance(codepack.pack_store_for(str(tmp_path)), DirPackStore)


def test_modal_volume_pack_store_creates_the_volume_on_put(monkeypatch):
    import sys
    import types

    from yeto import codepack

    calls = []

    class _Batch:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def put_file(self, fileobj, path):
            calls.append(("put", path))

    class _Volume:
        @staticmethod
        def from_name(name, create_if_missing=False):
            calls.append(("from_name", name, create_if_missing))
            return types.SimpleNamespace(batch_upload=lambda force: _Batch())

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(Volume=_Volume))
    store = codepack._ModalVolumePackStore("modal-volume://vol/run")
    assert store.put("abc", b"x") == "modal-volume://vol/run/code_packs/abc.tar"
    assert calls == [("from_name", "vol", True), ("put", "/run/code_packs/abc.tar")]
