"""Build or verify the closed provenance manifest for the Qwen3.5 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

_SCHEMA = "yeto-qwen35-megatron-conversion-v1"
_MANIFEST_NAME = "conversion-manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _closed_files(
    root: Path,
    *,
    omit_manifest: bool = False,
    allow_empty: bool = False,
) -> list[dict[str, object]]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"artifact root is not a real directory: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if ".cache" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise RuntimeError(f"artifact contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if omit_manifest and relative == _MANIFEST_NAME:
            continue
        size = path.stat().st_size
        if size < 1 and not allow_empty:
            raise RuntimeError(f"artifact contains an empty file: {path}")
        records.append({"path": relative, "size": size, "sha256": _sha256(path)})
    if not records:
        raise RuntimeError(f"artifact contains no files: {root}")
    return records


def _source_identity(root: Path) -> tuple[int, int, str]:
    records = _closed_files(root, allow_empty=True)
    digest = hashlib.sha256(b"yeto-conversion-source-v1\0")
    total = 0
    for record in records:
        digest.update(str(record["path"]).encode())
        digest.update(b"\0")
        digest.update(int(record["size"]).to_bytes(8, "little"))
        digest.update(bytes.fromhex(str(record["sha256"])))
        total += int(record["size"])
    return len(records), total, digest.hexdigest()


def _metadata_revisions(model_root: Path) -> list[str]:
    metadata_root = model_root / ".cache" / "huggingface" / "download"
    if not metadata_root.is_dir() or metadata_root.is_symlink():
        raise RuntimeError("Hugging Face metadata root is absent")
    revisions = []
    for path in sorted(metadata_root.rglob("*.metadata")):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("Hugging Face metadata is not a regular file")
        revisions.append(path.read_text().splitlines()[0])
    return revisions


def _canonical(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_exclusive(path: Path, payload: dict[str, object]) -> None:
    if (
        path.exists()
        or path.is_symlink()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise RuntimeError("manifest output path is not fresh and safe")
    temporary: Path | None = None
    try:
        descriptor, raw = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(raw)
        os.fchmod(descriptor, 0o600)
        encoded = _canonical(payload)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            if stream.write(encoded) != len(encoded):
                raise RuntimeError("short manifest write")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build(args) -> None:
    model_root = args.model.resolve(strict=True)
    checkpoint_root = args.checkpoint.resolve(strict=True)
    source_root = args.conversion_source.resolve(strict=True)
    output = args.output
    if output != checkpoint_root / _MANIFEST_NAME:
        raise RuntimeError("manifest must be the checkpoint-root sibling")
    model_files = _closed_files(model_root)
    checkpoint_files = _closed_files(checkpoint_root, omit_manifest=True)
    revisions = _metadata_revisions(model_root)
    if len(revisions) != len(model_files) or set(revisions) != {args.revision}:
        raise RuntimeError("Hugging Face files do not bind the expected revision")
    config = next(
        (record for record in model_files if record["path"] == "config.json"), None
    )
    if config is None:
        raise RuntimeError("model config is absent")
    source_count, source_bytes, source_hash = _source_identity(source_root)
    payload = {
        "schema": _SCHEMA,
        "image_digest": args.image_digest,
        "model_repo": "Qwen/Qwen3.5-4B",
        "model_revision": args.revision,
        "model_config_sha256": config["sha256"],
        "model_files": model_files,
        "checkpoint_files": checkpoint_files,
        "conversion_source_file_count": source_count,
        "conversion_source_bytes": source_bytes,
        "conversion_source_aggregate_sha256": source_hash,
    }
    _write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "manifest_sha256": _sha256(output),
                "model_files": len(model_files),
                "checkpoint_files": len(checkpoint_files),
            },
            sort_keys=True,
        )
    )


def verify(args) -> None:
    manifest = args.manifest.resolve(strict=True)
    if (
        manifest.name != _MANIFEST_NAME
        or manifest.is_symlink()
        or manifest.stat().st_mode & 0o077
    ):
        raise RuntimeError("conversion manifest path or mode is invalid")
    if _sha256(manifest) != args.expected_sha256:
        raise RuntimeError("conversion manifest identity changed")
    payload = json.loads(manifest.read_text())
    if _canonical(payload) != manifest.read_bytes() or payload.get("schema") != _SCHEMA:
        raise RuntimeError(
            "conversion manifest is noncanonical or has the wrong schema"
        )
    if (
        payload.get("model_revision") != args.expected_revision
        or payload.get("model_config_sha256") != args.expected_config_sha256
        or payload.get("image_digest") != args.expected_image_digest
    ):
        raise RuntimeError("conversion manifest provenance differs from the probe")
    checkpoint_root = manifest.parent
    model_root = args.model.resolve(strict=True)
    if payload.get("model_files") != _closed_files(model_root):
        raise RuntimeError("model files changed after conversion")
    if payload.get("checkpoint_files") != _closed_files(
        checkpoint_root, omit_manifest=True
    ):
        raise RuntimeError("Megatron checkpoint changed after conversion")
    revisions = _metadata_revisions(model_root)
    if set(revisions) != {payload.get("model_revision")} or len(revisions) != len(
        payload["model_files"]
    ):
        raise RuntimeError("Hugging Face revision metadata changed")
    print(
        json.dumps(
            {
                "checkpoint_files": len(payload["checkpoint_files"]),
                "model_files": len(payload["model_files"]),
                "verified": True,
            },
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--model", type=Path, required=True)
    build_parser.add_argument("--checkpoint", type=Path, required=True)
    build_parser.add_argument("--conversion-source", type=Path, required=True)
    build_parser.add_argument("--revision", required=True)
    build_parser.add_argument("--image-digest", required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.set_defaults(function=build)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--model", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--expected-sha256", required=True)
    verify_parser.add_argument("--expected-revision", required=True)
    verify_parser.add_argument("--expected-config-sha256", required=True)
    verify_parser.add_argument("--expected-image-digest", required=True)
    verify_parser.set_defaults(function=verify)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
