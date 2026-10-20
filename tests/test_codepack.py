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


def test_unpack_keeps_a_tree_another_launch_placed_first(repo: Path, monkeypatch) -> None:
    # S25 T-EVAL: two launches from one checkout unpack the same digest at once; the
    # later rename finds the earlier tree in place (ENOTEMPTY) and must reuse it
    pack, data = codepack.build_pack(repo)
    cache = repo.parent / "cache"
    first = codepack.unpack(data, pack.digest, cache)
    real_is_dir = Path.is_dir
    calls = {"n": 0}

    def racing_is_dir(self: Path) -> bool:  # the first existence check misses the tree
        if self == first and calls["n"] == 0:
            calls["n"] += 1
            return False
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", racing_is_dir)
    assert codepack.unpack(data, pack.digest, cache) == first
    assert codepack.verify_tree(first, pack.digest) == pack.digest
    assert [p.name for p in cache.iterdir()] == [first.name]  # no staging dir left behind


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


def test_modal_volume_store_creates_the_volume_on_put(monkeypatch: pytest.MonkeyPatch) -> None:
    # the first start stores the pack before the island creates the checkpoint volume
    import io
    import sys
    import types

    calls: list[tuple[str, bool]] = []
    stored: dict[str, bytes] = {}

    class _Batch:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def put_file(self, f: io.BytesIO, path: str) -> None:
            stored[path] = f.read()

    class _Volume:
        def batch_upload(self, force: bool = False) -> _Batch:
            return _Batch()

        def read_file(self, path: str):
            yield stored[path]

    def from_name(name: str, create_if_missing: bool = False) -> _Volume:
        calls.append((name, create_if_missing))
        return _Volume()

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(Volume=types.SimpleNamespace(from_name=from_name)))
    store = codepack._ModalVolumePackStore("modal-volume://vol/run")
    uri = store.put("abc", b"pack")
    assert uri == "modal-volume://vol/run/code_packs/abc.tar"
    assert store.get(uri) == b"pack"
    assert calls == [("vol", True), ("vol", False)]


# ---------------------------------------------------------------------------
# release mode (L9.12)
# ---------------------------------------------------------------------------
IMAGE = "ghcr.io/x/yeto-release@sha256:" + "a" * 64
OTHER_IMAGE = "ghcr.io/x/yeto-release@sha256:" + "b" * 64


def test_release_pack_is_the_image_digest() -> None:
    pack = codepack.release_pack("docker:" + IMAGE)
    assert (pack.mode, pack.digest, pack.pack_uri, pack.commit) == ("release", "a" * 64, IMAGE, None)
    assert codepack.CodePack.from_record(pack.record()) == pack
    for bad in ("ghcr.io/x/yeto-release:tag", "ghcr.io/x@sha256:abc", "@sha256:" + "a" * 64):
        with pytest.raises(CodePackError, match="pinned by digest"):
            codepack.release_pack(bad)


def _release(previous, image=IMAGE, **kw):
    return codepack.resolve_run_pack(previous=previous, repo=".", store=None, cache_root=".",
                                     release_image=image, log=lambda _m: None, **kw)


def test_release_run_reuses_its_image_and_refuses_another(repo: Path, tmp_path: Path) -> None:
    first = _release(None, semantic_sha256="s")
    assert not first.reused and first.tree is None
    manifest = {"code_pack": first.pack.record(), "semantic_sha256": "s"}
    assert _release(manifest, semantic_sha256="s").reused
    with pytest.raises(CodePackError, match="--rl-code-pack-upgrade"):
        _release(manifest, OTHER_IMAGE, semantic_sha256="s")
    with pytest.raises(CodePackError, match="semantic hash changes"):
        _release(manifest, OTHER_IMAGE, semantic_sha256="t", upgrade_reason="new release")
    up = _release(manifest, OTHER_IMAGE, semantic_sha256="s", upgrade_reason="new release")
    (step,) = up.pack.upgrades
    assert (step["from"], step["to"], up.pack.mode) == ("a" * 64, "b" * 64, "release")
    # a git-patch start of a release run is refused (it would fetch the image as a tar)
    with pytest.raises(CodePackError, match="release image"):
        codepack.resolve_run_pack(previous=manifest, repo=repo, store=DirPackStore(tmp_path / "s"),
                                  cache_root=tmp_path / "c", semantic_sha256="s",
                                  log=lambda _m: None)


def test_git_patch_run_moves_to_release_only_by_upgrade(repo: Path, tmp_path: Path) -> None:
    store = DirPackStore(tmp_path / "store")
    dev = codepack.resolve_run_pack(previous=None, repo=repo, store=store, cache_root=tmp_path / "c",
                                    semantic_sha256="s", log=lambda _m: None)
    manifest = {"code_pack": dev.pack.record(), "semantic_sha256": "s"}
    with pytest.raises(CodePackError, match="--rl-code-pack-upgrade"):
        _release(manifest, semantic_sha256="s")
    up = _release(manifest, semantic_sha256="s", upgrade_reason="ship release")
    (step,) = up.pack.upgrades
    assert (step["from_mode"], step["to_mode"]) == ("git-patch", "release")


def _release_tree(tmp_path: Path) -> tuple[Path, Path]:
    import json

    from yeto.provenance import source_tree_sha256

    src = tmp_path / "opt" / "yeto" / "src"
    (src / "yeto").mkdir(parents=True)
    (src / "yeto" / "a.py").write_text("A = 1\n")
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(json.dumps({"schema": codepack.RELEASE_SCHEMA, "yeto_commit": "c" * 40,
                                    "source_root": str(src),
                                    "source_tree_sha256": source_tree_sha256(src / "yeto")}))
    return src, manifest


def test_verify_release_tree(tmp_path: Path) -> None:
    src, manifest = _release_tree(tmp_path)
    link = tmp_path / "sky_workdir"
    link.symlink_to(src)  # the image links ~/sky_workdir to the source root
    assert codepack.verify_release_tree(link, manifest)["yeto_commit"] == "c" * 40
    with pytest.raises(CodePackError, match="not the image source root"):
        codepack.verify_release_tree(tmp_path, manifest)  # a shipped local tree
    (src / "yeto" / "a.py").write_text("A = 2\n")
    with pytest.raises(CodePackError, match="hashes to"):
        codepack.verify_release_tree(src, manifest)
    with pytest.raises(CodePackError, match="not a yeto release image"):
        codepack.verify_release_tree(src, tmp_path / "missing.json")
