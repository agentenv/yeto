from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import yeto.capture_v2_store as store_mod
from yeto.capture_v2_store import (
    SCHEMA,
    SCHEMA_VERSION,
    CaptureObjectStore,
    CaptureStoreError,
    ManifestEntry,
    ObjectRef,
)


def _manifest_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _install_manifest(store: CaptureObjectStore, value: dict) -> str:
    raw = _manifest_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    store.manifest_path(digest).write_bytes(raw)
    return digest


def test_atomic_content_addressing_deduplicates_and_accounts_exactly(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    first = store.put_bytes(b"shared")
    duplicate = store.put_bytes(b"shared")
    second = store.put_bytes(b"unique payload")

    assert first.inserted is True
    assert first.physical_bytes_added == len(b"shared")
    assert duplicate.ref == first.ref
    assert duplicate.inserted is False
    assert duplicate.physical_bytes_added == 0

    manifest = store.publish_manifest(
        "boundary-0001",
        [
            ManifestEntry("worker/0/restore", first.ref),
            ManifestEntry("worker/1/restore", duplicate.ref),
            ManifestEntry("syncer/post-fragment", second.ref),
        ],
        metadata={"fragment": 0, "step": 1},
    )
    loaded = store.load_manifest(manifest)
    accounting = loaded["accounting"]
    assert accounting == {
        "references": 3,
        "unique_objects": 2,
        "logical_bytes": 2 * len(b"shared") + len(b"unique payload"),
        "physical_bytes": len(b"shared") + len(b"unique payload"),
        "deduplicated_bytes": len(b"shared"),
    }

    repeated = store.publish_manifest(
        "boundary-0001",
        [
            ManifestEntry("worker/0/restore", first.ref),
            ManifestEntry("worker/1/restore", duplicate.ref),
            ManifestEntry("syncer/post-fragment", second.ref),
        ],
        metadata={"fragment": 0, "step": 1},
    )
    assert repeated.sha256 == manifest.sha256
    assert repeated.inserted is False

    audit = store.audit()
    assert audit.as_json() == {
        "manifests": 1,
        "references": 3,
        "unique_objects": 2,
        "logical_bytes": 2 * len(b"shared") + len(b"unique payload"),
        "physical_bytes": len(b"shared") + len(b"unique payload"),
        "deduplicated_bytes": len(b"shared"),
    }


def test_concurrent_atomic_insert_publishes_exactly_once(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    payload = b"one immutable value from racing producers"
    producers = 8
    barrier = threading.Barrier(producers)

    def insert_after_barrier(_index):
        barrier.wait()
        return store.put_bytes(payload)

    with ThreadPoolExecutor(max_workers=producers) as pool:
        results = list(pool.map(insert_after_barrier, range(producers)))

    assert sum(result.inserted for result in results) == 1
    assert len({result.ref for result in results}) == 1
    assert store.verify_object(results[0].ref).read_bytes() == payload
    assert [path.name for path in store.objects_dir.iterdir()] == [
        results[0].ref.sha256
    ]


def test_store_accounting_deduplicates_shared_objects_across_manifests(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    shared = store.put_bytes(b"one physical object")
    store.publish_manifest(
        "boundary-a", [ManifestEntry("worker/0/restore", shared.ref)]
    )
    store.publish_manifest(
        "boundary-b", [ManifestEntry("worker/1/restore", shared.ref)]
    )

    audit = store.audit()
    assert audit.manifests == 2
    assert audit.references == 2
    assert audit.unique_objects == 1
    assert audit.logical_bytes == 2 * shared.ref.bytes
    assert audit.physical_bytes == shared.ref.bytes
    assert audit.deduplicated_bytes == shared.ref.bytes


def test_put_file_streams_exact_bytes_and_rejects_symlink_sources(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    source = tmp_path / "source.bin"
    source.write_bytes((b"large enough to stream\0" * 100_000) + b"tail")

    result = store.put_file(source)
    assert result.ref.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result.ref.bytes == source.stat().st_size
    assert store.verify_object(result.ref).read_bytes() == source.read_bytes()

    symlink = tmp_path / "source-link.bin"
    symlink.symlink_to(source)
    with pytest.raises(CaptureStoreError, match="regular non-symlink"):
        store.put_file(symlink)


def test_existing_corrupt_digest_path_is_never_overwritten(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    raw = b"authoritative bytes"
    digest = hashlib.sha256(raw).hexdigest()
    target = store.object_path(digest)
    target.write_bytes(b"x" * len(raw))
    before = target.read_bytes()

    with pytest.raises(CaptureStoreError, match="CAS SHA-256 mismatch"):
        store.put_bytes(raw)

    assert target.read_bytes() == before


def test_object_corruption_is_detected_by_manifest_load_and_store_audit(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    result = store.put_bytes(b"capture tensor payload")
    manifest = store.publish_manifest(
        "boundary-corruption",
        [ManifestEntry("worker/0/fragment/0", result.ref)],
    )
    object_path = store.object_path(result.ref.sha256)
    corrupted = bytearray(object_path.read_bytes())
    corrupted[-1] ^= 1
    object_path.write_bytes(corrupted)

    with pytest.raises(CaptureStoreError, match="CAS SHA-256 mismatch"):
        store.load_manifest(manifest)
    with pytest.raises(CaptureStoreError, match="CAS SHA-256 mismatch"):
        store.audit()


def test_manifest_corruption_is_detected_from_its_content_address(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    result = store.put_bytes(b"payload")
    manifest = store.publish_manifest(
        "boundary-manifest-corruption",
        [ManifestEntry("worker/0/restore", result.ref)],
    )
    path = store.manifest_path(manifest.sha256)
    corrupted = bytearray(path.read_bytes())
    corrupted[len(corrupted) // 2] ^= 1
    path.write_bytes(corrupted)

    with pytest.raises(CaptureStoreError, match="manifest SHA-256 mismatch"):
        store.load_manifest(manifest)
    with pytest.raises(CaptureStoreError, match="manifest SHA-256 mismatch"):
        store.audit()


def test_orphan_missing_and_temporary_objects_fail_closed(tmp_path):
    store = CaptureObjectStore(tmp_path / "orphan-cas")
    referenced = store.put_bytes(b"referenced")
    orphan = store.put_bytes(b"not referenced")
    store.publish_manifest(
        "boundary-with-orphan",
        [ManifestEntry("worker/0/restore", referenced.ref)],
    )
    with pytest.raises(CaptureStoreError, match="orphan objects"):
        store.audit()
    assert store.audit(require_no_orphans=False).unique_objects == 1

    store.object_path(orphan.ref.sha256).unlink()
    store.object_path(referenced.ref.sha256).unlink()
    with pytest.raises(CaptureStoreError, match="regular non-symlink"):
        store.audit()

    temporary_store = CaptureObjectStore(tmp_path / "temporary-cas")
    temporary = temporary_store.objects_dir / ".interrupted.tmp-123"
    temporary.write_bytes(b"partial")
    with pytest.raises(CaptureStoreError, match="unexpected file"):
        temporary_store.audit()


def test_resealed_manifest_with_false_accounting_fails_semantic_validation(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    result = store.put_bytes(b"12345")
    value = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "manifest_id": "false-accounting",
        "metadata": {},
        "objects": [
            {
                "role": "worker/0/restore",
                "sha256": result.ref.sha256,
                "bytes": result.ref.bytes,
            }
        ],
        "accounting": {
            "references": 1,
            "unique_objects": 1,
            "logical_bytes": 4,
            "physical_bytes": 4,
            "deduplicated_bytes": 0,
        },
    }
    digest = _install_manifest(store, value)

    with pytest.raises(CaptureStoreError, match="accounting mismatch"):
        store.load_manifest(digest)
    with pytest.raises(CaptureStoreError, match="accounting mismatch"):
        store.audit()


def test_boolean_schema_or_accounting_values_do_not_alias_integers(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    result = store.put_bytes(b"x")
    base = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "manifest_id": "strict-integers",
        "metadata": {},
        "objects": [{"role": "worker/0/restore", **result.ref.as_json()}],
        "accounting": {
            "references": 1,
            "unique_objects": 1,
            "logical_bytes": 1,
            "physical_bytes": 1,
            "deduplicated_bytes": 0,
        },
    }
    schema_bool = {**base, "schema_version": True}
    schema_digest = _install_manifest(store, schema_bool)
    with pytest.raises(CaptureStoreError, match="unsupported schema"):
        store.load_manifest(schema_digest)

    accounting_bool = json.loads(json.dumps(base))
    accounting_bool["accounting"]["references"] = True
    accounting_digest = _install_manifest(store, accounting_bool)
    with pytest.raises(CaptureStoreError, match="accounting mismatch"):
        store.load_manifest(accounting_digest)


def test_resealed_manifest_cannot_reference_a_missing_object(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    absent = ObjectRef(hashlib.sha256(b"absent").hexdigest(), len(b"absent"))
    value = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "manifest_id": "missing-object",
        "metadata": {},
        "objects": [{"role": "worker/0/restore", **absent.as_json()}],
        "accounting": {
            "references": 1,
            "unique_objects": 1,
            "logical_bytes": absent.bytes,
            "physical_bytes": absent.bytes,
            "deduplicated_bytes": 0,
        },
    }
    digest = _install_manifest(store, value)

    with pytest.raises(CaptureStoreError, match="regular non-symlink"):
        store.load_manifest(digest)


def test_atomic_publish_failure_leaves_no_partial_or_temporary_entry(
    tmp_path, monkeypatch
):
    store = CaptureObjectStore(tmp_path / "cas")
    original_link = os.link

    def fail_link(source, destination):
        if Path(destination).parent == store.objects_dir:
            raise OSError("injected atomic-link failure")
        return original_link(source, destination)

    monkeypatch.setattr(store_mod.os, "link", fail_link)
    with pytest.raises(CaptureStoreError, match="injected atomic-link failure"):
        store.put_bytes(b"never visible")

    assert list(store.objects_dir.iterdir()) == []


def test_atomic_manifest_failure_publishes_no_partial_manifest(tmp_path, monkeypatch):
    store = CaptureObjectStore(tmp_path / "cas")
    result = store.put_bytes(b"object survives for retry")
    original_link = os.link

    def fail_manifest_link(source, destination):
        if Path(destination).parent == store.manifests_dir:
            raise OSError("injected manifest-link failure")
        return original_link(source, destination)

    monkeypatch.setattr(store_mod.os, "link", fail_manifest_link)
    with pytest.raises(CaptureStoreError, match="injected manifest-link failure"):
        store.publish_manifest(
            "never-published",
            [ManifestEntry("worker/0/restore", result.ref)],
        )

    assert list(store.manifests_dir.iterdir()) == []
    assert all(".tmp-" not in path.name for path in store.root.rglob("*"))
    with pytest.raises(CaptureStoreError, match="orphan objects"):
        store.audit()


def test_store_tree_rejects_symlinks_nested_directories_and_unknown_files(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    target = tmp_path / "outside"
    target.write_bytes(b"outside")
    symlink = store.objects_dir / hashlib.sha256(b"link").hexdigest()
    symlink.symlink_to(target)
    with pytest.raises(CaptureStoreError, match="symlink or non-file"):
        store.audit()

    symlink.unlink()
    nested = store.manifests_dir / "nested"
    nested.mkdir()
    with pytest.raises(CaptureStoreError, match="symlink or non-file"):
        store.audit()

    nested.rmdir()
    (store.root / "unexpected").write_bytes(b"not part of the CAS")
    with pytest.raises(CaptureStoreError, match="CAS layout mismatch"):
        store.audit()


def test_invalid_roles_duplicate_roles_and_nonfinite_metadata_are_rejected(tmp_path):
    store = CaptureObjectStore(tmp_path / "cas")
    result = store.put_bytes(b"payload")

    with pytest.raises(CaptureStoreError, match="unsafe path component"):
        ManifestEntry("worker/../restore", result.ref)
    with pytest.raises(CaptureStoreError, match="duplicate manifest role"):
        store.publish_manifest(
            "duplicate-role",
            [
                ManifestEntry("worker/0/restore", result.ref),
                ManifestEntry("worker/0/restore", result.ref),
            ],
        )
    with pytest.raises(CaptureStoreError, match="not canonical JSON data"):
        store.publish_manifest(
            "nonfinite-metadata",
            [ManifestEntry("worker/0/restore", result.ref)],
            metadata={"loss": float("nan")},
        )
