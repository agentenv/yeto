"""Derive a smaller immutable SecRLEnv task pack from an attested parent pack."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class TaskPackSubsetError(RuntimeError):
    """Raised when a parent-pack derivation cannot be proven exact."""


def _read_ordered_task_ids(path: Path, expected_count: int) -> tuple[bytes, list[str]]:
    if path.is_symlink() or not path.is_file():
        raise TaskPackSubsetError("ordered task-ID input is not a regular file")
    raw = path.read_bytes()
    try:
        task_ids = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise TaskPackSubsetError("ordered task-ID input is not UTF-8") from exc
    if not raw.endswith(b"\n"):
        raise TaskPackSubsetError("ordered task-ID input must end with a newline")
    if len(task_ids) != expected_count or len(task_ids) != len(set(task_ids)):
        raise TaskPackSubsetError("ordered task-ID count or uniqueness mismatch")
    if any(not task_id.strip() or task_id != task_id.strip() for task_id in task_ids):
        raise TaskPackSubsetError("ordered task-ID input contains an invalid entry")
    return raw, task_ids


def derive_task_pack(
    parent_path: Path,
    ordered_task_ids_path: Path,
    output_root: Path,
    *,
    expected_parent_sha256: str,
    expected_count: int,
) -> tuple[Path, Path]:
    """Copy an exact selected subset of an already-attested parent task pack."""

    from secrlenv_rl.taskpack import TaskPack, _atomic_json, _canonical_json

    parent_input = parent_path.expanduser()
    ordered_input = ordered_task_ids_path.expanduser()
    output_input = output_root.expanduser()
    tool_path = Path(__file__)
    if tool_path.is_symlink() or not tool_path.is_file():
        raise TaskPackSubsetError("derivation tool is not a regular file")
    tool_raw = tool_path.read_bytes()
    tool_sha256 = hashlib.sha256(tool_raw).hexdigest()
    if parent_input.is_symlink():
        raise TaskPackSubsetError("parent task pack may not be a symlink")
    if ordered_input.is_symlink():
        raise TaskPackSubsetError("ordered task-ID input may not be a symlink")
    if output_input.is_symlink():
        raise TaskPackSubsetError("output root may not be a symlink")
    if os.path.lexists(output_input):
        raise TaskPackSubsetError("output root must be fresh")
    if output_input.parent.is_symlink() or not output_input.parent.is_dir():
        raise TaskPackSubsetError("output parent must be an existing real directory")
    parent_path = parent_input.resolve()
    ordered_task_ids_path = ordered_input.resolve()
    output_root = output_input.resolve()
    if parent_path == ordered_task_ids_path:
        raise TaskPackSubsetError("parent and ordered task-ID inputs must differ")
    if (
        output_root == parent_path
        or output_root.is_relative_to(parent_path)
        or parent_path.is_relative_to(output_root)
        or output_root == ordered_task_ids_path
        or output_root.is_relative_to(ordered_task_ids_path)
        or ordered_task_ids_path.is_relative_to(output_root)
    ):
        raise TaskPackSubsetError("output root overlaps a protected input")
    if not _SHA256.fullmatch(expected_parent_sha256):
        raise TaskPackSubsetError("invalid expected parent task-pack SHA-256")
    if isinstance(expected_count, bool) or expected_count <= 0:
        raise TaskPackSubsetError("expected count must be positive")
    parent = TaskPack.load(parent_path)
    if parent.sha256 != expected_parent_sha256:
        raise TaskPackSubsetError("parent task-pack identity mismatch")
    ordered_raw, task_ids = _read_ordered_task_ids(
        ordered_task_ids_path, expected_count
    )
    parent_tasks = parent.manifest["tasks"]
    if any(task_id not in parent_tasks for task_id in task_ids):
        raise TaskPackSubsetError("selection contains a task absent from the parent")
    stage: Path | None = None
    child_path: Path | None = None
    contract_path: Path | None = None
    lock_path: Path | None = None
    root_created = False
    success = False
    try:
        output_root.mkdir(parents=False, exist_ok=False, mode=0o700)
        root_created = True
        stage = Path(
            tempfile.mkdtemp(prefix=".taskpack-subset-stage-", dir=output_root)
        )
        os.chmod(stage, 0o700)
        selected_tasks: dict[str, Any] = {}
        for task_id in task_ids:
            entry = copy.deepcopy(parent_tasks[task_id])
            source = parent.root / entry["path"]
            target = stage / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target, symlinks=True, copy_function=shutil.copy2)
            selected_tasks[task_id] = entry
        manifest = {
            "schema": parent.manifest["schema"],
            "source_repository": parent.manifest["source_repository"],
            "source_revision": parent.manifest["source_revision"],
            "tasks": selected_tasks,
        }
        child_sha256 = hashlib.sha256(_canonical_json(manifest)).hexdigest()
        envelope = {"manifest": manifest, "sha256": child_sha256}
        _atomic_json(stage / "manifest.json", envelope)
        child = TaskPack.load(stage)
        if child.sha256 != child_sha256:
            raise TaskPackSubsetError("derived task-pack identity mismatch")
        if set(child.manifest["tasks"]) != set(task_ids):
            raise TaskPackSubsetError("derived task-pack task set mismatch")
        for task_id in task_ids:
            if child.manifest["tasks"][task_id] != parent_tasks[task_id]:
                raise TaskPackSubsetError("derived task manifest entry mismatch")
        parent_after_copy = TaskPack.load(parent_path)
        if parent_after_copy.sha256 != expected_parent_sha256:
            raise TaskPackSubsetError("parent task pack changed during derivation")
        child_path = output_root / child_sha256
        contract_path = output_root / f"{child_sha256}.derivation.envelope.json"
        lock_path = output_root / f".{child_sha256}.derivation.lock"
        lock_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            lock_flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, lock_flags, 0o600)
        except FileExistsError as exc:
            raise TaskPackSubsetError("derived task-pack output is locked") from exc
        else:
            os.close(descriptor)
        if child_path.exists() or contract_path.exists():
            raise TaskPackSubsetError("derived task-pack output already exists")
        os.replace(stage, child_path)
        final_child = TaskPack.load(child_path)
        if final_child.sha256 != child_sha256:
            raise TaskPackSubsetError("final task-pack identity mismatch")
        if ordered_task_ids_path.read_bytes() != ordered_raw:
            raise TaskPackSubsetError("ordered task-ID input changed during derivation")
        if tool_path.read_bytes() != tool_raw:
            raise TaskPackSubsetError("derivation tool changed during derivation")
        contract = {
            "schema": "secrlenv-taskpack-parent-derivation/v1",
            "parent_task_pack_sha256": parent.sha256,
            "ordered_task_ids_sha256": hashlib.sha256(ordered_raw).hexdigest(),
            "ordered_task_count": len(task_ids),
            "child_task_pack_sha256": child_sha256,
            "child_task_count": len(child.manifest["tasks"]),
            "exact_task_set": True,
            "exact_manifest_entries": True,
            "source_repository_unchanged": True,
            "source_revision_unchanged": True,
            "derivation_tool_sha256": tool_sha256,
        }
        contract_envelope = {
            "contract": contract,
            "sha256": hashlib.sha256(_canonical_json(contract)).hexdigest(),
        }
        _atomic_json(contract_path, contract_envelope)
        os.chmod(contract_path, 0o600)
        success = True
        return child_path, contract_path
    finally:
        if lock_path is not None:
            lock_path.unlink(missing_ok=True)
        if not success and root_created:
            shutil.rmtree(output_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--ordered-task-ids", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    child, _contract = derive_task_pack(
        args.parent,
        args.ordered_task_ids,
        args.output_root,
        expected_parent_sha256=args.expected_parent_sha256,
        expected_count=args.expected_count,
    )
    from secrlenv_rl.taskpack import TaskPack

    pack = TaskPack.load(child)
    print(f"secrlenv_taskpack_subset=ready tasks={len(pack.manifest['tasks'])}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TaskPackSubsetError as exc:
        print(f"secrlenv task-pack subset failed: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1) from None
