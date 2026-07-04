"""NAVA integration helpers used by Yeto learners and data preparation."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse


def is_s3_uri(uri: str) -> bool:
    return isinstance(uri, str) and uri.startswith("s3://")


def split_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc or not p.path:
        raise ValueError(f"bad s3 uri: {uri!r}")
    return p.netloc, p.path.lstrip("/")


def stable_cache_path(uri: str, cache_dir: str | Path) -> Path:
    parsed = urlparse(uri)
    suffix = Path(parsed.path).suffix
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir) / f"{digest}{suffix}"


def download_s3(uri: str, dest: str | Path) -> Path:
    import boto3

    bucket, key = split_s3_uri(uri)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
    boto3.client("s3").download_file(bucket, key, str(tmp))
    tmp.replace(dest)
    return dest


def materialize_uri(uri: str, cache_dir: str | Path | None = None) -> str:
    """Return a local path for local/S3 URIs; HTTP(S) is returned unchanged."""
    if not isinstance(uri, str):
        return uri
    if uri.startswith(("http://", "https://")):
        return uri
    if is_s3_uri(uri):
        if cache_dir is None:
            return presign_s3(uri)
        dest = stable_cache_path(uri, cache_dir)
        if not dest.exists():
            download_s3(uri, dest)
        return str(dest)
    return os.path.expanduser(uri)


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_uri(uri: str, cache_dir: str | Path | None = None) -> str | None:
    """Best-effort sha256 for local/S3 URIs; HTTP(S) is intentionally skipped."""
    if not isinstance(uri, str) or uri.startswith(("http://", "https://")):
        return None
    if is_s3_uri(uri) and cache_dir is None:
        cache_dir = os.environ.get("YETO_NAVA_ASSET_CACHE") or Path(tempfile.gettempdir()) / "yeto-nava-assets"
    path = materialize_uri(uri, cache_dir)
    if isinstance(path, str) and path.startswith(("http://", "https://")):
        return None
    return sha256_file(path)


def presign_s3(uri: str, expires: int = 12 * 3600) -> str:
    import boto3

    bucket, key = split_s3_uri(uri)
    region = os.environ.get("NAVA_S3_REGION")
    client = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=int(os.environ.get("NAVA_S3_PRESIGN_EXPIRES", expires)),
    )


def install_nava_uri_resolver(cache_dir: str | Path | None = None) -> None:
    """Monkey-patch NAVA's BOS resolver so VAE adapters also accept s3:// URIs."""
    import importlib

    try:
        mod = importlib.import_module("nava_src.vae._bos_signer")
    except ModuleNotFoundError:
        return
    old_resolve = mod.resolve

    def resolve(uri):
        if isinstance(uri, str) and uri.startswith("s3://"):
            return materialize_uri(uri, cache_dir)
        return old_resolve(uri)

    mod.resolve = resolve
