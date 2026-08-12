"""Fail-closed preflight for immutable SecrlEnv task images."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

MIN_DATA_FREE_BYTES = 2 * 1024**4
_NAMED_DIGEST = re.compile(r".+@sha256:[0-9a-f]{64}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")


class TaskImagePreflightError(RuntimeError):
    """Raised when task images cannot be attested before GPU work starts."""


class TaskPackLike(Protocol):
    manifest: Mapping[str, Any]
    sha256: str


def task_image_pins(pack: TaskPackLike) -> tuple[tuple[str, str], ...]:
    """Return the unique, deterministic immutable-ref to image-ID contract."""

    tasks = pack.manifest.get("tasks")
    if not isinstance(tasks, Mapping) or not tasks:
        raise TaskImagePreflightError("task pack contains no tasks")
    pins: dict[str, str] = {}
    for task in tasks.values():
        services = task.get("services") if isinstance(task, Mapping) else None
        if not isinstance(services, Mapping) or not services:
            raise TaskImagePreflightError("task pack contains an invalid service set")
        for service in services.values():
            if not isinstance(service, Mapping):
                raise TaskImagePreflightError("task pack contains an invalid service")
            immutable = service.get("immutable")
            image_id = service.get("image_id")
            if not isinstance(immutable, str) or not (
                _NAMED_DIGEST.fullmatch(immutable) or _IMAGE_ID.fullmatch(immutable)
            ):
                raise TaskImagePreflightError(
                    "task images must use immutable digest references"
                )
            if not isinstance(image_id, str) or not _IMAGE_ID.fullmatch(image_id):
                raise TaskImagePreflightError("task image has an invalid image ID")
            previous = pins.setdefault(immutable, image_id)
            if previous != image_id:
                raise TaskImagePreflightError(
                    "one immutable task image reference has conflicting image IDs"
                )
    return tuple(sorted(pins.items()))


def _load_task_pack(path: Path) -> TaskPackLike:
    from secrlenv_rl.taskpack import TaskPack, TaskPackError

    try:
        return TaskPack.load(path)
    except TaskPackError:
        raise TaskImagePreflightError("task-pack validation failed") from None


def _inspect_image(reference: str) -> str | None:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", reference],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        return None
    lines = result.stdout.splitlines()
    if len(lines) != 1:
        raise TaskImagePreflightError("task image inspection returned invalid output")
    return lines[0].strip()


def _pull_image(reference: str) -> None:
    result = subprocess.run(
        ["docker", "pull", reference],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode:
        raise TaskImagePreflightError("task image pull failed")


def _data_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _can_registry_pull(reference: str) -> bool:
    if not _NAMED_DIGEST.fullmatch(reference):
        return False
    registry = reference.split("/", 1)[0].split(":", 1)[0]
    return registry not in {"localhost", "127.0.0.1", "[::1]"}


def preflight_task_images(
    task_pack: Path,
    expected_task_pack_sha256: str,
    *,
    data_root: Path = Path("/data"),
    minimum_free_bytes: int = MIN_DATA_FREE_BYTES,
    load_pack: Callable[[Path], TaskPackLike] = _load_task_pack,
    inspect_image: Callable[[str], str | None] = _inspect_image,
    pull_image: Callable[[str], None] = _pull_image,
    data_free_bytes: Callable[[Path], int] = _data_free_bytes,
) -> tuple[int, int]:
    """Attest and, when safe, pull every image in an attested task pack.

    Returns ``(present, pulled)``. No task, service, or image identity is printed.
    """

    pack = load_pack(task_pack)
    if pack.sha256 != expected_task_pack_sha256:
        raise TaskImagePreflightError("task-pack identity mismatch")
    pins = task_image_pins(pack)
    if data_free_bytes(data_root) < minimum_free_bytes:
        raise TaskImagePreflightError(
            "insufficient /data space before task image preflight"
        )
    present = 0
    pulled = 0
    for reference, expected_id in pins:
        actual_id = inspect_image(reference)
        if actual_id is None:
            if not _can_registry_pull(reference):
                raise TaskImagePreflightError(
                    "missing task image must be provisioned locally"
                )
            if data_free_bytes(data_root) < minimum_free_bytes:
                raise TaskImagePreflightError(
                    "insufficient /data space before task image pull"
                )
            pull_image(reference)
            pulled += 1
            if data_free_bytes(data_root) < minimum_free_bytes:
                raise TaskImagePreflightError(
                    "insufficient /data space after task image pull"
                )
            actual_id = inspect_image(reference)
            if actual_id is None:
                raise TaskImagePreflightError(
                    "task image remained unavailable after pull"
                )
        else:
            present += 1
        if actual_id != expected_id:
            raise TaskImagePreflightError("task image identity mismatch")
    return present, pulled


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-pack", type=Path, required=True)
    parser.add_argument("--expected-task-pack-sha256", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_task_pack_sha256):
        raise TaskImagePreflightError("invalid expected task-pack SHA-256")
    present, pulled = preflight_task_images(
        args.task_pack,
        args.expected_task_pack_sha256,
        data_root=args.data_root,
    )
    print(
        "secrlenv_task_images=ready "
        f"unique={present + pulled} present={present} pulled={pulled}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TaskImagePreflightError as exc:
        print(f"secrlenv task image preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
