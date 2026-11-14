"""Policy releases trainer host -> AWS S3 -> GPU island, no dev machine in the data path.

The trainer host (trainer store) runs ``publish-loop``: every new release the trainer names in
``policy-releases/LATEST.json`` gets a cut manifest (``yeto.rl.spot_rollout.cut_manifest``),
its files go to ``s3://$S3_BUCKET/$S3_PREFIX/<release>/`` with rclone (many files in
parallel through the trainer host's HTTP proxy), then the manifest (``manifests/<release>.json``),
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

    # trainer host
    S3_BUCKET=<bucket> S3_PREFIX=<run> S3_REGION=<region> S3_PROXY=<proxy> \
        python3 -m yeto.rl.spot.s3_delivery publish-loop --store <store>
    # VM (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, or AWS_PROFILE)
    S3_BUCKET=<bucket> S3_PREFIX=<run> S3_REGION=<region> \
        python3 -m yeto.rl.spot.s3_delivery pull-loop --store <store>

Near-region fan-out (:mod:`.peer_fanout`): with ``--peers host:port,...`` (or
``$YETO_RELEASE_PEERS``) the VM asks those peers for the release first and takes the object
store only when none delivers; with ``--serve-port`` it serves its own placed releases to
the other VMs of the cluster. The pointer's digest guards both paths the same way.

Environment: S3_BUCKET, S3_PREFIX and S3_REGION (required), S3_PROXY (empty: direct),
S3_TRANSFERS (4), RCLONE (rclone), AWS_PROFILE (unset: the default credential chain),
S3_METRICS (<store>/s3-delivery.jsonl),
S3_ASSUME_PRESENT (comma list of full releases the island gets elsewhere, never uploaded;
default ``r1-v0``, the trainer's first full release, i.e. the base model; set it to an empty
string to upload every release).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from yeto.rl.spot import peer_fanout
from yeto.rl.spot_rollout import cut_manifest as cm

REMOTE = "s3w"
DEFAULT_TRANSFERS = 4


def env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required")
    return value


def log(message: str) -> None:
    print(f"[s3-delivery {time.strftime('%H:%M:%S')}] {message}", flush=True)


class S3:
    """rclone against one bucket/prefix (an env-only remote: no rclone.conf)."""

    def __init__(self) -> None:
        self.bucket = required("S3_BUCKET")
        self.prefix = required("S3_PREFIX").strip("/")
        region = required("S3_REGION")
        self.rclone = env("RCLONE", "rclone")
        self.transfers = int(env("S3_TRANSFERS", str(DEFAULT_TRANSFERS)))
        e = dict(os.environ)
        tag = f"RCLONE_CONFIG_{REMOTE.upper()}_"
        e.update({tag + "TYPE": "s3", tag + "PROVIDER": "AWS", tag + "ENV_AUTH": "true",
                  tag + "REGION": region, tag + "NO_CHECK_BUCKET": "true",
                  "RCLONE_CONFIG": os.devnull})
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
    # Unset: skip the base model (r1-v0), so a delta chain never walks into a full-model upload.
    return {n.strip() for n in os.environ.get("S3_ASSUME_PRESENT", "r1-v0").split(",") if n.strip()}


# ---------------------------------------------------------------- trainer side
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
def pull(s3: S3, store: Path, release: str, *, digest: str | None = None, depth: int = 0,
         peers: Sequence[str] = (), fetch: peer_fanout.Fetch = peer_fanout.urllib_fetch) -> dict[str, Any] | None:
    """Download and place ``release`` (and missing bases first) into ``store``:
    from the first of ``peers`` that holds it, else from S3 (module doc)."""
    cm.check_release_name(release)
    root = store / cm.RELEASES
    if (root / release).is_dir():
        try:
            backfill_manifest(s3, store, release)
        except Exception as error:  # noqa: BLE001 - placed already; only peer serving lacks it
            log(f"manifest backfill of {release} failed: {error}")
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
        pull(s3, store, str(base), depth=depth + 1, peers=peers, fetch=fetch)
    pub = s3.get_json(f"pub/{release}.json") or {}
    released = manifest["record"].get("released_at")
    row = peer_fanout.pull_from_peers(peers, store, release, digest=expected, fetch=fetch, log=log) if peers else None
    if row is None:
        staged = store / f"staged-{release}"
        staged.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        s3.copy_down(release, staged)
        seconds = time.time() - t0
        cm.place(manifest, staged, store, expected_digest=expected)
        shutil.rmtree(staged, ignore_errors=True)
        peer_fanout.save_manifest(store, manifest)  # this node can now serve the release
        row = {"side": "pull", "source": "s3", "release": release, "bytes": manifest["bytes"],
               "download_s": round(seconds, 1),
               "MBps": round(manifest["bytes"] / 1e6 / max(seconds, 1e-6), 2), "placed_at": round(time.time(), 1)}
    row.update(published_at=pub.get("published_at"),
               lag_s=round(row["placed_at"] - float(released), 1) if released else None)
    metrics(store, row)
    log(f"placed {release} from {row['source']}: {manifest['bytes'] / 1e9:.2f} GB in {row['download_s']:.0f} s "
        f"({row['MBps']} MB/s), trainer->island lag {row['lag_s']} s")
    return row


def backfill_manifest(s3: S3, store: Path, release: str) -> bool:
    """Write the local manifest of a release that is placed but has none (a crash
    between ``cm.place`` and ``save_manifest``), so this node serves it to peers
    again. The S3 manifest is checked against the placed files (sizes and sha256)
    first; True when one was written."""
    cm.check_release_name(release)
    if peer_fanout.manifest_path(store, release).is_file() or not (store / cm.RELEASES / release).is_dir():
        return False
    raw = s3.get_bytes(f"manifests/{release}.json")
    if raw is None:  # never uploaded (S3_ASSUME_PRESENT): nothing to serve it with
        return False
    manifest = json.loads(raw)
    cm.check(manifest, None)
    if manifest.get("release") != release:
        raise cm.ManifestError(f"manifests/{release}.json names {manifest.get('release')!r}")
    cm.verify(manifest, store / cm.RELEASES / release)
    peer_fanout.save_manifest(store, manifest)
    log(f"backfilled the local manifest of {release} (placed before an interrupted save)")
    return True


def backfill_manifests(s3: S3, store: Path) -> list[str]:
    """:func:`backfill_manifest` for every placed release of ``store``; a release
    that cannot be backfilled is logged and left unserved. Returns the backfilled."""
    root = store / cm.RELEASES
    done: list[str] = []
    if not root.is_dir():
        return done
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        try:
            if backfill_manifest(s3, store, path.name):
                done.append(path.name)
        except Exception as error:  # noqa: BLE001 - one bad release must not stop the sweep
            log(f"manifest backfill of {path.name} failed: {error}")
    return done


def pull_loop(s3: S3, store: Path, interval: float, *, peers: Sequence[str] = ()) -> None:
    try:
        backfill_manifests(s3, store)  # releases an earlier run placed but crashed before recording
    except Exception as error:  # noqa: BLE001 - the pull loop still has to start
        log(f"manifest backfill at start failed: {error}")
    failures = 0
    while True:
        try:
            pointer = s3.get_json(cm.LATEST)
            if pointer:
                pull(s3, store, str(pointer["release"]), digest=pointer.get("digest"), peers=peers)
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
    p.add_argument("--peers", default=None, help="host:port,... of peer file servers to pull from first "
                                                 f"(default: ${peer_fanout.PEERS_ENV})")
    p.add_argument("--serve-port", type=int, default=None,
                   help="serve this store's placed releases to peers on this port (pull / pull-loop)")
    a = p.parse_args(argv)
    s3 = S3()
    peers = peer_fanout.parse_peers(a.peers) if a.peers is not None else peer_fanout.peers_from(environ=os.environ)
    server = None
    if a.serve_port is not None and a.cmd in ("pull", "pull-loop"):
        server = peer_fanout.ReleaseFileServer(a.store, port=a.serve_port).start()
        log(f"serving placed releases to peers on port {server.port}")
    if a.cmd == "publish":
        publish(s3, a.store, a.release or json.loads((a.store / cm.RELEASES / cm.LATEST).read_text())["path"])
    elif a.cmd == "publish-loop":
        publish_loop(s3, a.store, a.interval)
    elif a.cmd == "pull":
        backfill_manifests(s3, a.store)
        pointer = s3.get_json(cm.LATEST) or {}
        release = a.release or pointer["release"]
        pull(s3, a.store, release, digest=pointer.get("digest") if pointer.get("release") == release else None,
             peers=peers)
        if server is not None:
            server.stop()
    elif a.cmd == "pull-loop":
        pull_loop(s3, a.store, a.interval, peers=peers)
    else:
        print(json.dumps(s3.get_json(cm.LATEST)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
