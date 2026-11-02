#!/usr/bin/env python3
"""S30 P0-b: policy releases npu1 -> AWS S3 -> GPU island, no dev machine in the data path.

npu1 (trainer store) runs ``publish-loop``: every new release the trainer names in
``policy-releases/LATEST.json`` gets a cut manifest (``yeto.rl.spot_rollout.cut_manifest``),
its files go to ``s3://$S3_BUCKET/$S3_PREFIX/<release>/`` with rclone (many files in
parallel through the npu1 HTTP proxy), then the manifest (``manifests/<release>.json``),
the publish metrics (``pub/<release>.json``) and last the pointer ``LATEST.json``
(release, manifest digest). A release whose delta base was never published (the loop
fell behind) has its base published first, so a delta chain on S3 never has a hole.

The VM runs ``pull-loop`` (or one ``pull <release>``): it polls the pointer, downloads a
new release (and any missing bases) into ``staged-<release>``, and ``cut_manifest.place``
checks every size and sha256 against the manifest whose digest the pointer names before
LATEST.json in the local store moves. Both sides are resumable: rclone skips files already
copied with the same size, a manifest is written only after all its files, and a half
staged download is reused on the next try.

Every published/placed release appends one JSON line of metrics (bytes, seconds, MB/s,
lag behind the trainer's ``released_at``) to ``$S3_METRICS`` for the patrol script.

    # npu1
    S3_BUCKET=<bucket> S3_PROXY=<proxy> python3 tools/probes/s29_spot/s3_delivery.py publish-loop --store <store>
    # VM (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or ~/.aws profile yeto-rl)
    S3_BUCKET=<bucket> python3 tools/probes/s29_spot/s3_delivery.py pull-loop --store <store>

Environment: S3_BUCKET (required), S3_PREFIX (s29r120), S3_REGION
(eu-north-1), S3_PROXY (empty: direct), S3_TRANSFERS (4), RCLONE (rclone), AWS_PROFILE
(yeto-rl when ~/.aws/credentials has it), S3_METRICS (<store>/s3-delivery.jsonl),
S3_ASSUME_PRESENT (r1-v0: full releases the island gets elsewhere, never uploaded).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from yeto.rl.spot_rollout import cut_manifest as cm  # noqa: E402

REMOTE = "s3w"
DEFAULT_TRANSFERS = 4


def env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def log(message: str) -> None:
    print(f"[s3-delivery {time.strftime('%H:%M:%S')}] {message}", flush=True)


class S3:
    """rclone against one bucket/prefix (an env-only remote: no rclone.conf)."""

    def __init__(self) -> None:
        self.bucket = os.environ.get("S3_BUCKET", "")
        if not self.bucket:
            raise SystemExit("S3_BUCKET is required")
        self.prefix = env("S3_PREFIX", "s29r120").strip("/")
        self.rclone = env("RCLONE", "rclone")
        self.transfers = int(env("S3_TRANSFERS", str(DEFAULT_TRANSFERS)))
        e = dict(os.environ)
        tag = f"RCLONE_CONFIG_{REMOTE.upper()}_"
        e.update({tag + "TYPE": "s3", tag + "PROVIDER": "AWS", tag + "ENV_AUTH": "true",
                  tag + "REGION": env("S3_REGION", "eu-north-1"), tag + "NO_CHECK_BUCKET": "true",
                  "RCLONE_CONFIG": os.devnull})
        creds = Path.home() / ".aws" / "credentials"
        if "AWS_ACCESS_KEY_ID" not in e and "AWS_PROFILE" not in e and creds.is_file() \
                and "[yeto-rl]" in creds.read_text():
            e["AWS_PROFILE"] = "yeto-rl"
        proxy = os.environ.get("S3_PROXY", "")
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            if proxy:
                e[key] = proxy
            else:
                e.pop(key, None)
        self.environ = e

    def url(self, key: str = "") -> str:
        return f"{REMOTE}:{self.bucket}/{self.prefix}/{key}".rstrip("/")

    def run(self, *args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        cmd = [self.rclone, *args, "--timeout", "2m", "--retries", "10", "--low-level-retries", "20", "--stats-log-level", "DEBUG"]
        return subprocess.run(cmd, env=self.environ, check=check, text=True,
                              stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.PIPE if capture else None)

    def copy_up(self, local: Path, key: str) -> None:
        # files <= 200 MB go up in one PUT each, many at a time; larger ones as 16 MB parts
        self.run("copy", str(local), self.url(key), "--transfers", str(self.transfers),
                 "--checkers", str(2 * self.transfers), "--s3-upload-cutoff", "200M",
                 "--s3-chunk-size", "16M", "--s3-upload-concurrency", "4", "-q")

    def copy_down(self, key: str, local: Path) -> None:
        self.run("sync", self.url(key), str(local), "--transfers", str(self.transfers),
                 "--checkers", str(2 * self.transfers), "--multi-thread-streams", "4", "-q")

    def put_bytes(self, key: str, data: bytes) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(data)
        try:
            self.run("copyto", handle.name, self.url(key), "-q")
        finally:
            os.unlink(handle.name)

    def get_bytes(self, key: str) -> bytes | None:
        out = self.run("cat", self.url(key), check=False, capture=True)
        if out.returncode != 0:
            if "not found" in out.stderr.lower() or "doesn't exist" in out.stderr.lower() \
                    or "nosuchkey" in out.stderr.lower():
                return None
            raise RuntimeError(f"rclone cat {key}: {out.stderr.strip()[-300:]}")
        # rclone cat of a missing key exits 0 with no output; every object we write is non-empty
        return out.stdout.encode() or None

    def get_json(self, key: str) -> Any:
        raw = self.get_bytes(key)
        return None if raw is None else json.loads(raw)


def metrics(store: Path, row: dict[str, Any]) -> None:
    path = Path(os.environ.get("S3_METRICS") or store / "s3-delivery.jsonl")
    with open(path, "a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def assume_present() -> set[str]:
    return {n for n in env("S3_ASSUME_PRESENT", "r1-v0").split(",") if n}


# ---------------------------------------------------------------- npu1 side
def release_record(store: Path, release: str) -> dict[str, Any]:
    root = store / cm.RELEASES
    latest = json.loads((root / cm.LATEST).read_text())
    if latest.get("path") == release:
        return latest
    own = root / release / "RELEASE.json"
    if own.is_file():
        return json.loads(own.read_text())
    raise cm.ManifestError(f"no record for {release}: LATEST names {latest.get('path')!r}, no RELEASE.json")


def publish(s3: S3, store: Path, release: str, *, depth: int = 0) -> dict[str, Any] | None:
    """Publish ``release`` (and its unpublished bases first). Returns the metrics row,
    None when it was already on S3."""
    cm.check_release_name(release)
    if release in assume_present() or s3.get_bytes(f"manifests/{release}.json") is not None:
        return None
    record = release_record(store, release)
    base = (record.get("delta") or {}).get("base_path")
    if base and depth < 64:
        publish(s3, store, str(base), depth=depth + 1)
    detected = time.time()
    manifest = cm.make(store, release, record=record)
    digest = cm.digest(manifest)
    t0 = time.time()
    s3.copy_up(store / cm.RELEASES / release, release)
    seconds = time.time() - t0
    s3.put_bytes(f"manifests/{release}.json", cm.encode(manifest))
    published = time.time()
    row = {"side": "publish", "release": release, "digest": digest, "bytes": manifest["bytes"],
           "files": len(manifest["files"]), "upload_s": round(seconds, 1),
           "MBps": round(manifest["bytes"] / 1e6 / max(seconds, 1e-6), 2), "transfers": s3.transfers,
           "released_at": record.get("released_at"), "detected_at": round(detected, 1),
           "published_at": round(published, 1),
           "lag_s": round(published - float(record["released_at"]), 1) if record.get("released_at") else None}
    s3.put_bytes(f"pub/{release}.json", json.dumps(row, sort_keys=True).encode())
    order = list(cm._order(record) or (-1, -1))
    pointer = s3.get_json(cm.LATEST)
    if pointer is None or list(pointer.get("record_order") or (-1, -1)) < order:  # never roll back
        s3.put_bytes(cm.LATEST, json.dumps({"release": release, "digest": digest, "record_order": order,
                                            "published_at": row["published_at"]}).encode())
    metrics(store, row)
    log(f"published {release}: {manifest['bytes'] / 1e9:.2f} GB in {seconds:.0f} s "
        f"({row['MBps']} MB/s), lag {row['lag_s']} s")
    return row


def publish_loop(s3: S3, store: Path, interval: float) -> None:
    failures = 0
    while True:
        try:
            latest = json.loads((store / cm.RELEASES / cm.LATEST).read_text())
            release = latest.get("path")
            if release:
                publish(s3, store, str(release))
            failures = 0
        except Exception as error:  # keep the loop alive; the next poll resumes the copy
            failures += 1
            log(f"publish failed ({failures} in a row): {error}")
            time.sleep(min(600.0, interval * 2 ** min(failures, 5)))
            continue
        time.sleep(interval)


# ---------------------------------------------------------------- VM side
def pull(s3: S3, store: Path, release: str, *, digest: str | None = None, depth: int = 0) -> dict[str, Any] | None:
    """Download and place ``release`` (and missing bases first) into ``store``."""
    cm.check_release_name(release)
    root = store / cm.RELEASES
    if (root / release).is_dir():
        return None
    raw = s3.get_bytes(f"manifests/{release}.json")
    if raw is None:
        raise cm.ManifestError(f"{release} is not on S3 (no manifest)")
    manifest = json.loads(raw)
    expected = digest or cm.digest(manifest)  # S3 is authenticated; the pointer's digest when we have it
    cm.check(manifest, expected)
    base = manifest.get("base")
    if base and not (root / base).is_dir():
        if depth >= 64:
            raise cm.ManifestError(f"delta chain of {release} too deep")
        pull(s3, store, str(base), depth=depth + 1)
    staged = store / f"staged-{release}"
    staged.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s3.copy_down(release, staged)
    seconds = time.time() - t0
    cm.place(manifest, staged, store, expected_digest=expected)
    shutil.rmtree(staged, ignore_errors=True)
    placed = time.time()
    pub = s3.get_json(f"pub/{release}.json") or {}
    released = manifest["record"].get("released_at")
    row = {"side": "pull", "release": release, "bytes": manifest["bytes"], "download_s": round(seconds, 1),
           "MBps": round(manifest["bytes"] / 1e6 / max(seconds, 1e-6), 2), "placed_at": round(placed, 1),
           "published_at": pub.get("published_at"),
           "lag_s": round(placed - float(released), 1) if released else None}
    metrics(store, row)
    log(f"placed {release}: {manifest['bytes'] / 1e9:.2f} GB in {seconds:.0f} s ({row['MBps']} MB/s), "
        f"trainer->island lag {row['lag_s']} s")
    return row


def pull_loop(s3: S3, store: Path, interval: float) -> None:
    failures = 0
    while True:
        try:
            pointer = s3.get_json(cm.LATEST)
            if pointer:
                pull(s3, store, str(pointer["release"]), digest=pointer.get("digest"))
            failures = 0
        except Exception as error:
            failures += 1
            log(f"pull failed ({failures} in a row): {error}")
            time.sleep(min(600.0, interval * 2 ** min(failures, 5)))
            continue
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="s3_delivery")
    p.add_argument("cmd", choices=["publish", "publish-loop", "pull", "pull-loop", "pointer"])
    p.add_argument("release", nargs="?")
    p.add_argument("--store", required=True, type=Path)
    p.add_argument("--interval", type=float, default=30.0)
    a = p.parse_args(argv)
    s3 = S3()
    if a.cmd == "publish":
        publish(s3, a.store, a.release or json.loads((a.store / cm.RELEASES / cm.LATEST).read_text())["path"])
    elif a.cmd == "publish-loop":
        publish_loop(s3, a.store, a.interval)
    elif a.cmd == "pull":
        pointer = s3.get_json(cm.LATEST) or {}
        release = a.release or pointer["release"]
        pull(s3, a.store, release, digest=pointer.get("digest") if pointer.get("release") == release else None)
    elif a.cmd == "pull-loop":
        pull_loop(s3, a.store, a.interval)
    else:
        print(json.dumps(s3.get_json(cm.LATEST)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
