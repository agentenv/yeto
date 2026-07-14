"""Immutable content-addressed storage for capture-v2 inputs.

This module deliberately knows nothing about learners, syncers, optimizers, or
PyTorch serialization.  Producers hand it already-serialized bytes or regular
files.  It gives them two small guarantees on which the later capture-v2
materializer can build:

* an object becomes visible only at its SHA-256 path after its complete bytes
  have been written and fsynced; and
* a manifest becomes visible only after every referenced object has been
  independently verified.

Manifests are content-addressed too.  There is no mutable "latest" pointer in
this foundation: a campaign index can add that separately while retaining an
immutable manifest digest as its authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "yeto.capture-v2-object-manifest"
SCHEMA_VERSION = 1
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ROLE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}\Z")
_COPY_CHUNK_BYTES = 1024 * 1024


class CaptureStoreError(ValueError):
    """The object store is malformed or an immutable operation is unsafe."""


@dataclass(frozen=True)
class ObjectRef:
    """Identity of one immutable object."""

    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        _validate_digest(self.sha256, "object sha256")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int):
            raise CaptureStoreError("object bytes must be an integer")
        if self.bytes < 0:
            raise CaptureStoreError("object bytes must be non-negative")

    def as_json(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True)
class ManifestEntry:
    """One logical role referencing an immutable object."""

    role: str
    object: ObjectRef

    def __post_init__(self) -> None:
        _validate_role(self.role)

    def as_json(self) -> dict[str, Any]:
        return {"role": self.role, **self.object.as_json()}


@dataclass(frozen=True)
class PutResult:
    """Result of inserting an object or manifest into the CAS."""

    ref: ObjectRef
    inserted: bool

    @property
    def physical_bytes_added(self) -> int:
        return self.ref.bytes if self.inserted else 0


@dataclass(frozen=True)
class ManifestRef:
    """Content identity and logical name of a published manifest."""

    manifest_id: str
    sha256: str
    bytes: int
    inserted: bool

    def __post_init__(self) -> None:
        _validate_manifest_id(self.manifest_id)
        _validate_digest(self.sha256, "manifest sha256")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int):
            raise CaptureStoreError("manifest bytes must be an integer")
        if self.bytes <= 0:
            raise CaptureStoreError("manifest bytes must be positive")


@dataclass(frozen=True)
class StoreAudit:
    """Verified logical and physical accounting across all manifests."""

    manifests: int
    references: int
    unique_objects: int
    logical_bytes: int
    physical_bytes: int

    @property
    def deduplicated_bytes(self) -> int:
        return self.logical_bytes - self.physical_bytes

    def as_json(self) -> dict[str, int]:
        return {
            "manifests": self.manifests,
            "references": self.references,
            "unique_objects": self.unique_objects,
            "logical_bytes": self.logical_bytes,
            "physical_bytes": self.physical_bytes,
            "deduplicated_bytes": self.deduplicated_bytes,
        }


def _validate_digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise CaptureStoreError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _validate_manifest_id(value: Any) -> str:
    if not isinstance(value, str) or _MANIFEST_ID_RE.fullmatch(value) is None:
        raise CaptureStoreError(
            "manifest_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )
    return value


def _validate_role(value: Any) -> str:
    if not isinstance(value, str) or _ROLE_RE.fullmatch(value) is None:
        raise CaptureStoreError("manifest role is malformed")
    if value.startswith("/") or any(
        part in ("", ".", "..") for part in value.split("/")
    ):
        raise CaptureStoreError("manifest role contains an unsafe path component")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_COPY_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CaptureStoreError(f"manifest is not canonical JSON data: {exc}") from exc


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CaptureStoreError(f"manifest contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise CaptureStoreError(f"manifest contains non-finite JSON number {value}")


def _load_json_strict(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_no_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureStoreError(f"cannot decode {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptureStoreError(f"{context} must be a JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class CaptureObjectStore:
    """A flat SHA-256 object store with immutable content-addressed manifests."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        if self.root.is_symlink():
            raise CaptureStoreError(f"object-store root must not be a symlink: {root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects_dir = self.root / "objects" / "sha256"
        self.manifests_dir = self.root / "manifests" / "sha256"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.root,
            self.root / "objects",
            self.objects_dir,
            self.root / "manifests",
            self.manifests_dir,
        ):
            if path.is_symlink() or not path.is_dir():
                raise CaptureStoreError(
                    f"object-store component must be a regular directory: {path}"
                )

    def object_path(self, digest: str) -> Path:
        """Return the only permitted path for an object digest."""

        _validate_digest(digest, "object sha256")
        return self.objects_dir / digest

    def manifest_path(self, digest: str) -> Path:
        """Return the only permitted path for a manifest digest."""

        _validate_digest(digest, "manifest sha256")
        return self.manifests_dir / f"{digest}.json"

    def put_bytes(self, raw: bytes) -> PutResult:
        """Insert complete bytes, deduplicating only after exact verification."""

        if not isinstance(raw, bytes):
            raise TypeError("put_bytes requires bytes")
        digest = hashlib.sha256(raw).hexdigest()
        ref = ObjectRef(digest, len(raw))
        inserted = self._insert_bytes(self.object_path(digest), raw, ref)
        return PutResult(ref, inserted)

    def put_file(self, source: str | os.PathLike[str]) -> PutResult:
        """Copy a regular non-symlink file into the CAS without loading it whole."""

        source_path = Path(source)
        if source_path.is_symlink() or not source_path.is_file():
            raise CaptureStoreError(
                f"source must be a regular non-symlink file: {source}"
            )
        temporary = self._temporary_path(self.objects_dir, "incoming")
        digest = hashlib.sha256()
        byte_count = 0
        try:
            with source_path.open("rb") as reader, temporary.open("xb") as writer:
                while block := reader.read(_COPY_CHUNK_BYTES):
                    writer.write(block)
                    digest.update(block)
                    byte_count += len(block)
                writer.flush()
                os.fsync(writer.fileno())
            ref = ObjectRef(digest.hexdigest(), byte_count)
            inserted = self._publish_temporary(
                temporary, self.object_path(ref.sha256), ref
            )
            return PutResult(ref, inserted)
        finally:
            temporary.unlink(missing_ok=True)

    def verify_object(self, ref: ObjectRef) -> Path:
        """Verify object kind, size, path identity, and full SHA-256."""

        path = self.object_path(ref.sha256)
        self._verify_regular_file(path, ref)
        return path

    def publish_manifest(
        self,
        manifest_id: str,
        entries: Iterable[ManifestEntry],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ManifestRef:
        """Verify every object, then atomically publish an immutable manifest."""

        manifest_id = _validate_manifest_id(manifest_id)
        entry_list = list(entries)
        roles: set[str] = set()
        unique: dict[str, ObjectRef] = {}
        for entry in entry_list:
            if not isinstance(entry, ManifestEntry):
                raise TypeError("manifest entries must be ManifestEntry values")
            if entry.role in roles:
                raise CaptureStoreError(f"duplicate manifest role {entry.role!r}")
            roles.add(entry.role)
            previous = unique.setdefault(entry.object.sha256, entry.object)
            if previous.bytes != entry.object.bytes:
                raise CaptureStoreError(
                    f"object {entry.object.sha256} has inconsistent byte counts"
                )
            self.verify_object(entry.object)

        logical_bytes = sum(entry.object.bytes for entry in entry_list)
        physical_bytes = sum(ref.bytes for ref in unique.values())
        manifest = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "manifest_id": manifest_id,
            "metadata": dict(metadata or {}),
            "objects": [entry.as_json() for entry in entry_list],
            "accounting": {
                "references": len(entry_list),
                "unique_objects": len(unique),
                "logical_bytes": logical_bytes,
                "physical_bytes": physical_bytes,
                "deduplicated_bytes": logical_bytes - physical_bytes,
            },
        }
        raw = _canonical_json(manifest)
        digest = hashlib.sha256(raw).hexdigest()
        ref = ObjectRef(digest, len(raw))
        inserted = self._insert_bytes(self.manifest_path(digest), raw, ref)
        return ManifestRef(manifest_id, digest, len(raw), inserted)

    def load_manifest(self, ref: ManifestRef | str) -> dict[str, Any]:
        """Verify a content-addressed manifest and all objects it references."""

        digest = ref.sha256 if isinstance(ref, ManifestRef) else ref
        _validate_digest(digest, "manifest sha256")
        path = self.manifest_path(digest)
        if path.is_symlink() or not path.is_file():
            raise CaptureStoreError(f"missing regular manifest file: {path}")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise CaptureStoreError(f"cannot read manifest {path}: {exc}") from exc
        actual_size = len(raw)
        if isinstance(ref, ManifestRef) and actual_size != ref.bytes:
            raise CaptureStoreError(
                f"manifest {digest} size mismatch: expected {ref.bytes}, got {actual_size}"
            )
        actual_digest = hashlib.sha256(raw).hexdigest()
        if actual_digest != digest:
            raise CaptureStoreError(
                f"manifest SHA-256 mismatch: expected {digest}, got {actual_digest}"
            )
        manifest = _load_json_strict(raw, context=f"manifest {path}")
        if raw != _canonical_json(manifest):
            raise CaptureStoreError(f"manifest {digest} is not canonical JSON")
        self._validate_manifest_value(manifest, context=f"manifest {digest}")
        if isinstance(ref, ManifestRef) and manifest["manifest_id"] != ref.manifest_id:
            raise CaptureStoreError(
                f"manifest id mismatch: expected {ref.manifest_id!r}, "
                f"got {manifest['manifest_id']!r}"
            )
        return manifest

    def audit(self, *, require_no_orphans: bool = True) -> StoreAudit:
        """Verify the exact store tree, every manifest, and global accounting."""

        self._verify_store_layout()
        object_files = self._exact_object_files()
        manifest_files = self._exact_manifest_files()
        references = 0
        logical_bytes = 0
        referenced: dict[str, ObjectRef] = {}
        for path in manifest_files:
            digest = path.name.removesuffix(".json")
            manifest = self.load_manifest(digest)
            references += manifest["accounting"]["references"]
            logical_bytes += manifest["accounting"]["logical_bytes"]
            for row in manifest["objects"]:
                ref = ObjectRef(row["sha256"], row["bytes"])
                previous = referenced.setdefault(ref.sha256, ref)
                if previous.bytes != ref.bytes:
                    raise CaptureStoreError(
                        f"object {ref.sha256} has inconsistent sizes across manifests"
                    )

        actual_digests = {path.name for path in object_files}
        referenced_digests = set(referenced)
        missing = sorted(referenced_digests - actual_digests)
        orphaned = sorted(actual_digests - referenced_digests)
        if missing:
            raise CaptureStoreError(f"store is missing referenced objects: {missing}")
        if require_no_orphans and orphaned:
            raise CaptureStoreError(f"store contains orphan objects: {orphaned}")

        for path in object_files:
            digest = _validate_digest(path.name, "object filename")
            if digest in referenced:
                self.verify_object(referenced[digest])
            else:
                actual = _sha256_file(path)
                if actual != digest:
                    raise CaptureStoreError(
                        f"orphan object SHA-256 mismatch: expected {digest}, got {actual}"
                    )

        physical_bytes = sum(ref.bytes for ref in referenced.values())
        return StoreAudit(
            manifests=len(manifest_files),
            references=references,
            unique_objects=len(referenced),
            logical_bytes=logical_bytes,
            physical_bytes=physical_bytes,
        )

    def _insert_bytes(self, target: Path, raw: bytes, ref: ObjectRef) -> bool:
        if target.exists() or target.is_symlink():
            self._verify_regular_file(target, ref)
            return False
        temporary = self._temporary_path(target.parent, target.name)
        try:
            with temporary.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            return self._publish_temporary(temporary, target, ref)
        finally:
            temporary.unlink(missing_ok=True)

    def _publish_temporary(self, temporary: Path, target: Path, ref: ObjectRef) -> bool:
        try:
            # Linking a fully written inode is an atomic no-replace insertion.
            # Unlike os.replace(), it cannot overwrite an existing corrupt CAS
            # object during a producer race.
            os.link(temporary, target)
        except FileExistsError:
            self._verify_regular_file(target, ref)
            return False
        except OSError as exc:
            raise CaptureStoreError(
                f"cannot atomically publish {target}: {exc}"
            ) from exc
        else:
            temporary.unlink()
            _fsync_directory(target.parent)
            self._verify_regular_file(target, ref)
            return True

    def _verify_regular_file(self, path: Path, ref: ObjectRef) -> None:
        if path.is_symlink() or not path.is_file():
            raise CaptureStoreError(
                f"CAS entry is not a regular non-symlink file: {path}"
            )
        actual_size = path.stat().st_size
        if actual_size != ref.bytes:
            raise CaptureStoreError(
                f"CAS size mismatch for {ref.sha256}: expected {ref.bytes}, "
                f"got {actual_size}"
            )
        actual_digest = _sha256_file(path)
        if actual_digest != ref.sha256:
            raise CaptureStoreError(
                f"CAS SHA-256 mismatch: expected {ref.sha256}, got {actual_digest}"
            )

    def _validate_manifest_value(self, value: dict[str, Any], *, context: str) -> None:
        expected_keys = {
            "schema",
            "schema_version",
            "manifest_id",
            "metadata",
            "objects",
            "accounting",
        }
        if set(value) != expected_keys:
            raise CaptureStoreError(
                f"{context} fields differ: expected={sorted(expected_keys)}, "
                f"actual={sorted(value)}"
            )
        if value["schema"] != SCHEMA or (
            type(value["schema_version"]) is not int
            or value["schema_version"] != SCHEMA_VERSION
        ):
            raise CaptureStoreError(f"{context} uses an unsupported schema")
        _validate_manifest_id(value["manifest_id"])
        if not isinstance(value["metadata"], dict):
            raise CaptureStoreError(f"{context} metadata must be an object")
        rows = value["objects"]
        if not isinstance(rows, list):
            raise CaptureStoreError(f"{context} objects must be an array")
        roles: set[str] = set()
        unique: dict[str, ObjectRef] = {}
        entries: list[ManifestEntry] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {"role", "sha256", "bytes"}:
                raise CaptureStoreError(f"{context} object row {index} is malformed")
            entry = ManifestEntry(
                _validate_role(row["role"]), ObjectRef(row["sha256"], row["bytes"])
            )
            if entry.role in roles:
                raise CaptureStoreError(f"{context} has duplicate role {entry.role!r}")
            roles.add(entry.role)
            previous = unique.setdefault(entry.object.sha256, entry.object)
            if previous.bytes != entry.object.bytes:
                raise CaptureStoreError(
                    f"{context} gives object {entry.object.sha256} inconsistent sizes"
                )
            self.verify_object(entry.object)
            entries.append(entry)

        expected_accounting = {
            "references": len(entries),
            "unique_objects": len(unique),
            "logical_bytes": sum(entry.object.bytes for entry in entries),
            "physical_bytes": sum(ref.bytes for ref in unique.values()),
        }
        expected_accounting["deduplicated_bytes"] = (
            expected_accounting["logical_bytes"] - expected_accounting["physical_bytes"]
        )
        accounting = value["accounting"]
        if (
            not isinstance(accounting, dict)
            or set(accounting) != set(expected_accounting)
            or any(type(item) is not int or item < 0 for item in accounting.values())
            or accounting != expected_accounting
        ):
            raise CaptureStoreError(
                f"{context} accounting mismatch: expected={expected_accounting}, "
                f"actual={accounting}"
            )

    def _exact_object_files(self) -> list[Path]:
        return self._exact_files(self.objects_dir, manifest=False)

    def _exact_manifest_files(self) -> list[Path]:
        return self._exact_files(self.manifests_dir, manifest=True)

    def _verify_store_layout(self) -> None:
        expected = {
            self.root: {"objects", "manifests"},
            self.root / "objects": {"sha256"},
            self.root / "manifests": {"sha256"},
        }
        for directory, expected_names in expected.items():
            actual_names = {child.name for child in directory.iterdir()}
            if actual_names != expected_names:
                raise CaptureStoreError(
                    f"CAS layout mismatch below {directory}: "
                    f"expected={sorted(expected_names)}, actual={sorted(actual_names)}"
                )
            for child in directory.iterdir():
                if child.is_symlink() or not child.is_dir():
                    raise CaptureStoreError(
                        f"CAS layout entry is not a regular directory: {child}"
                    )

    @staticmethod
    def _exact_files(directory: Path, *, manifest: bool) -> list[Path]:
        result: list[Path] = []
        for child in directory.iterdir():
            if child.is_symlink() or not child.is_file():
                raise CaptureStoreError(
                    f"CAS directory contains a symlink or non-file entry: {child}"
                )
            name = child.name.removesuffix(".json") if manifest else child.name
            if _DIGEST_RE.fullmatch(name) is None or (
                manifest and not child.name.endswith(".json")
            ):
                raise CaptureStoreError(
                    f"CAS directory contains an unexpected file: {child}"
                )
            result.append(child)
        return sorted(result)

    @staticmethod
    def _temporary_path(directory: Path, label: str) -> Path:
        return directory / f".{label}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
