"""rl-identity-tiering L9.4-L9.6: the launcher side of the run code pack (fake sky/Modal,
directory checkpoint store; no cloud)."""

from __future__ import annotations

import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest

import yeto.launcher as launcher
from yeto import codepack
from yeto.codepack import CodePackError
from yeto.launch import code_pack
from test_rl_algorithm_provenance import _fake_sky, _launcher_args  # noqa: F401  (fixture)
from test_rl_launcher_island_failure import Clock, _setup

_REAL_RUN = subprocess.run


def _git(repo: Path, *argv: str) -> None:
    _REAL_RUN(["git", *argv], cwd=repo, check=True, capture_output=True)


def _repo(root: Path) -> Path:
    (root / "yeto").mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "yeto" / "a.py").write_text("A = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture
def launch(monkeypatch, tmp_path):
    """``launch(*extra)`` runs one fake no-sync Modal launch with a code pack and
    returns ``(args, manifest)``."""

    repo = _repo(tmp_path / "repo")
    store = tmp_path / "store"
    store.mkdir()
    record = []
    _setup(monkeypatch, tmp_path, failing=set(), succeeding={0}, clock=Clock(), record=record)
    # _setup stubs subprocess.run for the launcher; the pack needs real git.
    monkeypatch.setattr(codepack, "subprocess", types.SimpleNamespace(
        run=_REAL_RUN, CalledProcessError=subprocess.CalledProcessError))
    # a fake modal-volume:// pack store backed by a local directory
    monkeypatch.setitem(codepack._PACK_STORES, "modal-volume",
                        lambda uri: codepack.DirPackStore(store, uri_base=uri.rstrip("/")))
    monkeypatch.setattr(launcher, "_verify_modal_app_stopped", lambda *a, **k: True)
    monkeypatch.setattr(code_pack, "REPO_ROOT", repo)
    monkeypatch.setattr(code_pack, "CODE_PACK_CACHE", tmp_path / "cache")

    def run(*extra):
        from yeto import runs

        args = _launcher_args("ports", ("--controller", "local", "--rl-single-island-no-sync",
                                        "--rl-checkpoint-store", "modal-volume://vol/run",
                                        "--rl-code-pack", "git-patch", *extra),
                              gpu="modal:1xa100")
        args.keep, args.recover_timeout = False, 0.0
        # a no-sync run refuses old tapes; a later start here is about code only
        shutil.rmtree(runs.run_dir(args.cluster_prefix) / "events", ignore_errors=True)
        launcher.run(args)
        path = runs.run_dir(args.cluster_prefix) / "run_manifest.json"
        return args, json.loads(path.read_text())

    run.repo, run.store = repo, store
    return run


def test_first_start_writes_pack_and_resume_reuses_it(launch):
    args, manifest = launch()
    pack = manifest["code_pack"]
    assert manifest["code_pack_mode"] == "git-patch" == pack["mode"]
    assert manifest["code_pack_digest"] == pack["digest"] == args.code_pack.digest
    assert pack["pack_uri"] == f"modal-volume://vol/run/code_packs/{pack['digest']}.tar"
    assert (launch.store / "code_packs" / f"{pack['digest']}.tar").is_file()
    assert pack["upgrades"] == []
    assert code_pack.code_workdir(args) == args.code_pack_tree
    assert code_pack.code_pack_island_flag(args) == f" --code-pack-digest {pack['digest']}"
    # the local tree changes; the next start of the same run keeps the run pack
    (launch.repo / "yeto" / "a.py").write_text("A = 2\n")
    args2, manifest2 = launch()
    assert manifest2["code_pack"] == pack
    assert (Path(args2.code_pack_tree) / "yeto" / "a.py").read_text() == "A = 1\n"


def test_upgrade_records_old_to_new(launch):
    _, first = launch()
    (launch.repo / "yeto" / "a.py").write_text("A = 3\n")
    _, manifest = launch("--rl-code-pack-upgrade", "hotfix")
    (step,) = manifest["code_pack"]["upgrades"]
    assert step["from"] == first["code_pack_digest"] != step["to"]
    assert step["to"] == manifest["code_pack_digest"] and step["reason"] == "hotfix"


def test_upgrade_refused_when_semantic_hash_changes(launch):
    from yeto import runs

    args, manifest = launch()
    manifest["semantic_sha256"] = "0" * 64  # the run started with another algorithm
    path = runs.run_dir(args.cluster_prefix) / "run_manifest.json"
    path.write_text(json.dumps(manifest))
    (launch.repo / "yeto" / "a.py").write_text("A = 4\n")
    with pytest.raises(CodePackError, match="semantic hash changes"):
        launch("--rl-code-pack-upgrade", "hotfix")


@pytest.mark.parametrize("mode,store,expected", [
    ("auto", "s3://bucket/run", "git-patch"),
    ("auto", "modal-volume://vol/run", "git-patch"),
    ("auto", "/mnt/shared/run", "off"),  # a path may be visible on the islands only
    ("auto", None, "off"),
    ("off", "s3://bucket/run", "off"),
    ("git-patch", "/mnt/shared/run", "git-patch"),
])
def test_auto_packs_a_run_with_a_uri_store(mode, store, expected):
    args = types.SimpleNamespace(rl_code_pack=mode, rl_checkpoint_store=store)
    assert code_pack.code_pack_mode(args) == expected


def test_launcher_default_is_auto():
    import argparse

    from yeto.rl.engine.run_flags import CLI, add_rl_flags

    parser = argparse.ArgumentParser()
    add_rl_flags(parser, CLI)
    assert parser.parse_args([]).rl_code_pack == "auto"


def test_off_and_upgrade_needs_pack():
    args = types.SimpleNamespace(training_mode="rl", rl_code_pack="off",
                                 rl_code_pack_upgrade=None)
    code_pack.prepare_run_code_pack(args)
    assert not hasattr(args, "code_pack") and code_pack.code_pack_island_flag(args) == ""
    args.rl_code_pack_upgrade = "x"
    with pytest.raises(ValueError, match="requires --rl-code-pack"):
        code_pack.prepare_run_code_pack(args)
    args = types.SimpleNamespace(training_mode="rl", rl_code_pack="git-patch",
                                 rl_checkpoint_store=None)
    with pytest.raises(ValueError, match="requires --rl-checkpoint-store"):
        code_pack.prepare_run_code_pack(args)


def test_island_refuses_a_tree_that_is_not_the_pack(tmp_path):
    from yeto.rl.adapters.miles.island_entry import build_parser, verify_island_code_pack

    repo = _repo(tmp_path / "repo")
    pack, data = codepack.build_pack(repo)
    tree = codepack.unpack(data, pack.digest, tmp_path / "cache")
    verify_island_code_pack(None)  # no pack: the transition source-tree check only
    verify_island_code_pack(pack.digest, tree)
    with pytest.raises(CodePackError, match="digest mismatch"):
        verify_island_code_pack("f" * 64, tree)
    assert any("--code-pack-digest" in a.option_strings for a in build_parser()._actions)
