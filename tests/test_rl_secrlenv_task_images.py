from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import yeto.rl.secrlenv_task_images as task_images
from yeto.rl.secrlenv_task_images import (
    MIN_DATA_FREE_BYTES,
    TaskImagePreflightError,
    preflight_task_images,
    task_image_pins,
)


@dataclass(frozen=True)
class _Pack:
    manifest: dict[str, Any]
    sha256: str = "a" * 64


def _pack(*services: dict[str, str], sha256: str = "a" * 64) -> _Pack:
    return _Pack(
        {
            "tasks": {
                f"task-{index}": {"services": {"target": service}}
                for index, service in enumerate(services)
            }
        },
        sha256,
    )


def _service(index: int, *, image_id: int | None = None) -> dict[str, str]:
    return {
        "immutable": f"registry.example/image-{index}@sha256:{index:064x}",
        "image_id": f"sha256:{(index if image_id is None else image_id):064x}",
    }


def test_task_image_pins_deduplicates_and_sorts():
    service_1 = _service(1)
    service_2 = _service(2)

    assert task_image_pins(_pack(service_2, service_1, service_1)) == (
        (service_1["immutable"], service_1["image_id"]),
        (service_2["immutable"], service_2["image_id"]),
    )


def test_task_image_pins_rejects_conflicts_and_mutable_references():
    service = _service(1)
    conflict = {**service, "image_id": "sha256:" + "2" * 64}
    with pytest.raises(TaskImagePreflightError, match="conflicting"):
        task_image_pins(_pack(service, conflict))

    mutable = {"immutable": "registry.example/image:latest", "image_id": "sha256:" + "1" * 64}
    with pytest.raises(TaskImagePreflightError, match="immutable digest"):
        task_image_pins(_pack(mutable))


def test_task_pack_identity_is_checked_before_any_image_operation(tmp_path):
    calls: list[str] = []
    with pytest.raises(TaskImagePreflightError, match="task-pack identity"):
        preflight_task_images(
            tmp_path,
            "b" * 64,
            load_pack=lambda _path: _pack(_service(1)),
            inspect_image=lambda _reference: calls.append("inspect") or None,
            pull_image=lambda _reference: calls.append("pull"),
        )
    assert calls == []


def test_present_task_image_is_attested_without_pull(tmp_path):
    service = _service(1)
    pulled: list[str] = []

    assert preflight_task_images(
        tmp_path,
        "a" * 64,
        load_pack=lambda _path: _pack(service),
        inspect_image=lambda reference: service["image_id"],
        pull_image=pulled.append,
        data_free_bytes=lambda _path: MIN_DATA_FREE_BYTES,
    ) == (1, 0)
    assert pulled == []


def test_missing_task_image_is_pulled_once_and_reattested(tmp_path):
    service = _service(1)
    inspections = iter((None, service["image_id"]))
    pulled: list[str] = []

    assert preflight_task_images(
        tmp_path,
        "a" * 64,
        load_pack=lambda _path: _pack(service, service),
        inspect_image=lambda _reference: next(inspections),
        pull_image=pulled.append,
        data_free_bytes=lambda _path: MIN_DATA_FREE_BYTES,
    ) == (0, 1)
    assert pulled == [service["immutable"]]


def test_task_image_pull_requires_safe_disk_before_and_after(tmp_path):
    service = _service(1)
    pulled: list[str] = []
    with pytest.raises(TaskImagePreflightError, match="before"):
        preflight_task_images(
            tmp_path,
            "a" * 64,
            load_pack=lambda _path: _pack(service),
            inspect_image=lambda _reference: None,
            pull_image=pulled.append,
            data_free_bytes=lambda _path: MIN_DATA_FREE_BYTES - 1,
        )
    assert pulled == []

    inspections = iter((None, service["image_id"]))
    free = iter(
        (MIN_DATA_FREE_BYTES, MIN_DATA_FREE_BYTES, MIN_DATA_FREE_BYTES - 1)
    )
    with pytest.raises(TaskImagePreflightError, match="after"):
        preflight_task_images(
            tmp_path,
            "a" * 64,
            load_pack=lambda _path: _pack(service),
            inspect_image=lambda _reference: next(inspections),
            pull_image=pulled.append,
            data_free_bytes=lambda _path: next(free),
        )
    assert pulled == [service["immutable"]]


def test_task_image_identity_mismatch_fails_closed(tmp_path):
    service = _service(1)
    with pytest.raises(TaskImagePreflightError, match="identity mismatch"):
        preflight_task_images(
            tmp_path,
            "a" * 64,
            load_pack=lambda _path: _pack(service),
            inspect_image=lambda _reference: "sha256:" + "f" * 64,
            data_free_bytes=lambda _path: MIN_DATA_FREE_BYTES,
        )


def test_present_raw_id_task_image_is_attested_without_pull(tmp_path):
    image_id = "sha256:" + "1" * 64
    service = {"immutable": image_id, "image_id": image_id}
    pulled: list[str] = []

    assert preflight_task_images(
        tmp_path,
        "a" * 64,
        load_pack=lambda _path: _pack(service),
        inspect_image=lambda _reference: image_id,
        pull_image=pulled.append,
        data_free_bytes=lambda _path: MIN_DATA_FREE_BYTES,
    ) == (1, 0)
    assert pulled == []


def test_missing_raw_id_task_image_fails_without_pull(tmp_path):
    image_id = "sha256:" + "1" * 64
    service = {"immutable": image_id, "image_id": image_id}
    pulled: list[str] = []

    with pytest.raises(TaskImagePreflightError, match="provisioned locally"):
        preflight_task_images(
            tmp_path,
            "a" * 64,
            load_pack=lambda _path: _pack(service),
            inspect_image=lambda _reference: None,
            pull_image=pulled.append,
            data_free_bytes=lambda _path: MIN_DATA_FREE_BYTES,
        )
    assert pulled == []


def test_missing_local_registry_task_image_fails_without_pull(tmp_path):
    image_id = "sha256:" + "1" * 64
    service = {
        "immutable": "localhost/task@sha256:" + "2" * 64,
        "image_id": image_id,
    }
    pulled: list[str] = []

    with pytest.raises(TaskImagePreflightError, match="provisioned locally"):
        preflight_task_images(
            tmp_path,
            "a" * 64,
            load_pack=lambda _path: _pack(service),
            inspect_image=lambda _reference: None,
            pull_image=pulled.append,
            data_free_bytes=lambda _path: MIN_DATA_FREE_BYTES,
        )
    assert pulled == []


def test_main_emits_one_aggregate_marker_without_image_details(monkeypatch, capsys):
    monkeypatch.setattr(task_images, "preflight_task_images", lambda *_args, **_kwargs: (2, 1))

    assert task_images.main(
        [
            "--task-pack",
            "/task-pack",
            "--expected-task-pack-sha256",
            "a" * 64,
        ]
    ) == 0
    assert capsys.readouterr().out == (
        "secrlenv_task_images=ready unique=3 present=2 pulled=1\n"
    )
