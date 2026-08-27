from __future__ import annotations

import hashlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from yeto.rl.secrlenv_taskpack_subset import (
    TaskPackSubsetError,
    derive_task_pack,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json(value) + b"\n")


@dataclass(frozen=True)
class _FakeTaskPack:
    root: Path
    manifest: dict

    @classmethod
    def load(cls, root: str | Path) -> _FakeTaskPack:
        path = Path(root).resolve()
        envelope = json.loads((path / "manifest.json").read_text())
        manifest = envelope["manifest"]
        digest = hashlib.sha256(_canonical_json(manifest)).hexdigest()
        if envelope["sha256"] != digest:
            raise RuntimeError("manifest mismatch")
        for entry in manifest["tasks"].values():
            if not (path / entry["path"]).is_dir():
                raise RuntimeError("missing task")
        return cls(path, manifest)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.manifest)).hexdigest()


@pytest.fixture(autouse=True)
def _fake_secrlenv_taskpack(monkeypatch):
    package = types.ModuleType("secrlenv_rl")
    module = types.ModuleType("secrlenv_rl.taskpack")
    module.TaskPack = _FakeTaskPack
    module._atomic_json = _atomic_json
    module._canonical_json = _canonical_json
    package.taskpack = module
    monkeypatch.setitem(sys.modules, "secrlenv_rl", package)
    monkeypatch.setitem(sys.modules, "secrlenv_rl.taskpack", module)


def _parent_pack(tmp_path: Path) -> tuple[Path, str, list[str]]:
    parent = tmp_path / "parent"
    tasks: dict[str, dict] = {}
    task_ids = ["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"]
    for index, task_id in enumerate(task_ids):
        relative = Path("tasks") / task_id
        task_root = parent / relative
        task_root.mkdir(parents=True)
        (task_root / "compose.yml").write_text(f"task={index}\n")
        tasks[task_id] = {
            "path": relative.as_posix(),
            "tree_sha256": f"tree-{index}",
            "services": {
                "target": {
                    "immutable": f"example.invalid/task-{index}@sha256:{index:064x}",
                    "image_id": f"sha256:{index + 1:064x}",
                }
            },
            "target_service": "target",
            "exposed_services": ["target"],
            "prompt_tiers": ["l2"],
        }
    manifest = {
        "schema": "secrlenv-task-pack/v1",
        "source_repository": "https://example.invalid/source",
        "source_revision": "a" * 40,
        "tasks": tasks,
    }
    digest = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    _atomic_json(parent / "manifest.json", {"manifest": manifest, "sha256": digest})
    return parent, digest, task_ids


def _selection(tmp_path: Path, task_ids: list[str]) -> Path:
    path = tmp_path / "ordered-task-ids.txt"
    path.write_text("".join(f"{task_id}\n" for task_id in task_ids))
    return path


def test_derives_exact_parent_subset_and_seals_contract(tmp_path):
    parent, parent_digest, task_ids = _parent_pack(tmp_path)
    selected = [task_ids[2], task_ids[0]]
    ordered = _selection(tmp_path, selected)
    parent_before = {
        path.relative_to(parent): path.read_bytes()
        for path in parent.rglob("*")
        if path.is_file()
    }

    child, contract = derive_task_pack(
        parent,
        ordered,
        tmp_path / "output",
        expected_parent_sha256=parent_digest,
        expected_count=2,
    )

    derived = _FakeTaskPack.load(child)
    source = _FakeTaskPack.load(parent)
    assert set(derived.manifest["tasks"]) == set(selected)
    assert derived.manifest["source_repository"] == source.manifest["source_repository"]
    assert derived.manifest["source_revision"] == source.manifest["source_revision"]
    assert all(
        derived.manifest["tasks"][task_id] == source.manifest["tasks"][task_id]
        for task_id in selected
    )
    envelope = json.loads(contract.read_text())
    assert envelope["contract"]["parent_task_pack_sha256"] == parent_digest
    assert envelope["contract"]["ordered_task_count"] == 2
    assert envelope["contract"]["child_task_pack_sha256"] == derived.sha256
    assert oct(contract.stat().st_mode & 0o777) == "0o600"
    assert oct(child.parent.stat().st_mode & 0o777) == "0o700"
    assert not list(child.parent.glob(".*.derivation.lock"))
    assert parent_before == {
        path.relative_to(parent): path.read_bytes()
        for path in parent.rglob("*")
        if path.is_file()
    }


def test_rejects_existing_or_parent_overlapping_output(tmp_path):
    parent, parent_digest, task_ids = _parent_pack(tmp_path)
    ordered = _selection(tmp_path, task_ids[:1])
    existing = tmp_path / "existing"
    existing.mkdir(mode=0o755)

    with pytest.raises(TaskPackSubsetError, match="must be fresh"):
        derive_task_pack(
            parent,
            ordered,
            existing,
            expected_parent_sha256=parent_digest,
            expected_count=1,
        )
    assert oct(existing.stat().st_mode & 0o777) == "0o755"

    with pytest.raises(TaskPackSubsetError, match="overlaps"):
        derive_task_pack(
            parent,
            ordered,
            parent / "derived",
            expected_parent_sha256=parent_digest,
            expected_count=1,
        )
    assert not (parent / "derived").exists()


def test_rejects_symlinked_selection_without_creating_output(tmp_path):
    parent, parent_digest, task_ids = _parent_pack(tmp_path)
    ordered = _selection(tmp_path, task_ids[:1])
    alias = tmp_path / "ordered-alias.txt"
    alias.symlink_to(ordered)
    output = tmp_path / "output"

    with pytest.raises(TaskPackSubsetError, match="may not be a symlink"):
        derive_task_pack(
            parent,
            alias,
            output,
            expected_parent_sha256=parent_digest,
            expected_count=1,
        )
    assert not output.exists()


def test_invalid_selection_cleans_fresh_output(tmp_path):
    parent, parent_digest, _task_ids = _parent_pack(tmp_path)
    ordered = _selection(tmp_path, ["CVE-2099-9999"])
    output = tmp_path / "output"

    with pytest.raises(TaskPackSubsetError, match="absent from the parent"):
        derive_task_pack(
            parent,
            ordered,
            output,
            expected_parent_sha256=parent_digest,
            expected_count=1,
        )
    assert not output.exists() or not any(output.iterdir())
