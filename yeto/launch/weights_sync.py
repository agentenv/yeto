"""Copy a model's weights from an object store (or a shared directory) onto a
node's local disk, in the Hugging Face hub cache layout, verified by sha256.

This file runs on a bare-metal training node before the trainer starts
(:mod:`yeto.launch.ssh_weights` sends it over SSH and runs it with the node's
``python3 -I``), so it uses the standard library only and imports nothing of
yeto. It also runs locally (``python -m yeto.launch.weights_sync``).

Layout under ``--root`` (the same one ``storage.model_store_env`` reads):

* ``hub/models--ORG--NAME/snapshots/REV/<file>``: the snapshot; with
  ``HF_HUB_CACHE=<root>/hub``, ``snapshot_download(..., local_files_only=True)``
  returns that directory and never calls the Hub;
* ``yeto-complete/ORG--NAME@REV.json``: written last, when every selected file
  is in place and verified (removed when a sync starts);
* ``yeto-weights/ORG--NAME@REV/``: work state, outside the snapshot so no
  loader or release copier ever sees it: ``parts/<file>`` (a download in
  progress), ``parts/<file>.done`` (its journal: one finished chunk index per
  line) and ``verified.json`` (size, mtime and sha256 of each file verified, so
  a later sync skips a file without hashing it again).

Integrity. Every file has a size and a sha256 from the source's manifest: the
``yeto-weights-manifest.json`` object next to the files (``manifest`` below
writes one), or else the ``sha256`` user metadata of each object (what the
model mirror to the bucket stores). A file without a sha256 is refused. A file
is downloaded in chunks (``--chunk-mb``) by ``--workers`` concurrent range
requests, written in place into a sparse part file, hashed once complete and
renamed into the snapshot only on a match. A mismatch discards the part and
downloads the file again (``--attempts``); a second mismatch is an error.

Resume. A restarted sync keeps every chunk its journal lists and fetches the
rest. A file already in the snapshot is skipped when ``verified.json`` has it
with the same size and mtime and the manifest's sha256; otherwise it is hashed
(skipped on a match, downloaded again on a mismatch).

Pipeline stages. ``--stages S[,S...]`` with ``--pp`` (and Megatron's
``--pp-first`` / ``--pp-last`` uneven split) copies only the safetensors files
that hold a tensor of those stages' decoder layers, plus every file holding a
tensor outside the decoder layers (embedding, final norm, head, vision tower:
every stage gets those) and every non-safetensors file (config, tokenizer,
the full ``model.safetensors.index.json``). mbridge's ``SafeTensorIO`` opens
only the files of the tensors a stage asks for, so the index stays whole.
Layers at or past ``num_hidden_layers`` (MTP blocks) belong to the last stage.

Disk space: refused before any byte is written when the file system's free
space is below ``--headroom`` (1.2) times the bytes still to download.

Progress: one line every ``--progress-s`` seconds (bytes, rate, ETA); a last
``YETO_WEIGHTS_SUMMARY {json}`` line.

Credentials of an ``s3://`` (or ``r2://``) source: ``YETO_WEIGHTS_S3_ACCESS_KEY_ID``,
``YETO_WEIGHTS_S3_SECRET_ACCESS_KEY`` (``_SESSION_TOKEN``), endpoint
``YETO_WEIGHTS_S3_ENDPOINT_URL`` and region ``YETO_WEIGHTS_S3_REGION``, each
falling back to the ``AWS_*`` name. Requests are signed with SigV4, path style.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import hmac
import http.client
import json
import multiprocessing
import os
import re
import shutil
import sys
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent import futures

MANIFEST_NAME = "yeto-weights-manifest.json"
MANIFEST_FORMAT = "yeto-weights-manifest/1"
INDEX_NAME = "model.safetensors.index.json"
SUMMARY_TAG = "YETO_WEIGHTS_SUMMARY"
SHA_META = "sha256"
ENV_PREFIX = "YETO_WEIGHTS_S3_"
DEFAULT_CHUNK_MB = 64
DEFAULT_WORKERS = 64
DEFAULT_HEADROOM = 1.2
READ_BLOCK = 8 * 1024 * 1024
REQUEST_ATTEMPTS = 6
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
# a decoder layer of the language model: "model.layers.N." (Llama/DeepSeek/Kimi
# naming, also under "language_model."), "language_model.layers.N."
LAYER_RE = re.compile(r"(?:^|\.)(?:model|language_model)\.layers\.(\d+)\.")
HEX40 = re.compile(r"[0-9a-f]{40}")


class SyncError(RuntimeError):
    """The weights cannot be copied or verified as asked."""


def log(message: str) -> None:
    print(f"[yeto-weights] {message}", flush=True)


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


# ---------------------------------------------------------------- sources
class DirSource:
    """A directory (local disk or a shared file system mount)."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        if not os.path.isdir(self.path):
            raise SyncError(f"source {path} is not a directory")

    def describe(self) -> str:
        return self.path

    def list(self) -> dict:
        out = {}
        for base, _dirs, files in os.walk(self.path):
            for name in files:
                full = os.path.join(base, name)
                out[os.path.relpath(full, self.path).replace(os.sep, "/")] = os.path.getsize(full)
        return out

    def read(self, rel: str) -> bytes | None:
        try:
            with open(os.path.join(self.path, rel), "rb") as fh:
                return fh.read()
        except FileNotFoundError:
            return None

    def meta_sha256(self, rel: str) -> str | None:
        return None

    def read_into(self, rel: str, lo: int, hi: int, fd: int) -> int:
        """Copy bytes [lo, hi) of ``rel`` to ``fd`` at offset ``lo``; the count."""
        done = 0
        with open(os.path.join(self.path, rel), "rb") as src:
            src.seek(lo)
            while lo + done < hi:
                data = src.read(min(READ_BLOCK, hi - lo - done))
                if not data:
                    break
                os.pwrite(fd, data, lo + done)
                done += len(data)
        return done


def _quote(value: str, safe: str = "-_.~") -> str:
    return urllib.parse.quote(value, safe=safe)


class S3Source:
    """An S3-compatible bucket prefix (AWS S3, Cloudflare R2, MinIO), path style."""

    def __init__(self, uri: str, environ: dict | None = None) -> None:
        env = os.environ if environ is None else environ
        scheme, _, rest = uri.partition("://")
        bucket, _, prefix = rest.partition("/")
        if scheme not in ("s3", "r2") or not bucket:
            raise SyncError(f"source {uri!r}: expected s3://BUCKET/PREFIX")
        self.uri, self.bucket, self.prefix = uri, bucket, prefix.strip("/")

        def get(name: str) -> str | None:
            return env.get(ENV_PREFIX + name) or env.get("AWS_" + name)

        self.key_id, self.secret, self.token = get("ACCESS_KEY_ID"), get("SECRET_ACCESS_KEY"), get("SESSION_TOKEN")
        if not (self.key_id and self.secret):
            raise SyncError(f"source {uri}: no credentials ({ENV_PREFIX}ACCESS_KEY_ID / "
                            f"{ENV_PREFIX}SECRET_ACCESS_KEY or the AWS_ names)")
        self.region = get("REGION") or get("DEFAULT_REGION")
        endpoint = get("ENDPOINT_URL")
        if not endpoint:
            self.region = self.region or "us-east-1"
            endpoint = f"https://s3.{self.region}.amazonaws.com"
        self.region = self.region or "auto"  # an S3-compatible endpoint without a region (Cloudflare R2)
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise SyncError(f"endpoint {endpoint!r} is not an http(s) URL")
        self.https = parsed.scheme == "https"
        self.netloc = parsed.netloc
        self.base_path = parsed.path.rstrip("/")
        self._local = threading.local()

    def describe(self) -> str:
        return self.uri

    def _key(self, rel: str) -> str:
        return f"{self.prefix}/{rel}" if self.prefix else rel

    def _conn(self, fresh: bool = False):
        conn = getattr(self._local, "conn", None)
        if conn is None or fresh:
            if conn is not None:
                conn.close()
            cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
            conn = cls(self.netloc, timeout=120)
            self._local.conn = conn
        return conn

    def _signed(self, method: str, path: str, query: dict) -> dict:
        now = datetime.datetime.now(datetime.timezone.utc)  # noqa: UP017 - python 3.8 hosts
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        day = amz_date[:8]
        headers = {"host": self.netloc, "x-amz-content-sha256": "UNSIGNED-PAYLOAD", "x-amz-date": amz_date}
        if self.token:
            headers["x-amz-security-token"] = self.token
        names = sorted(headers)
        canonical = "\n".join([
            method, path, "&".join(f"{_quote(k)}={_quote(str(v))}" for k, v in sorted(query.items())),
            "".join(f"{k}:{headers[k].strip()}\n" for k in names), ";".join(names), "UNSIGNED-PAYLOAD"])
        scope = f"{day}/{self.region}/s3/aws4_request"
        to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
        key = ("AWS4" + self.secret).encode()
        for part in (day, self.region, "s3", "aws4_request"):
            key = hmac.new(key, part.encode(), hashlib.sha256).digest()
        signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
        headers["Authorization"] = (f"AWS4-HMAC-SHA256 Credential={self.key_id}/{scope}, "
                                    f"SignedHeaders={';'.join(names)}, Signature={signature}")
        return headers

    def _request(self, method: str, key: str | None, query: dict | None = None, extra: dict | None = None,
                 sink=None):
        """One signed request, retried on connection errors and 5xx/429. Returns
        ``(status, headers, body)``; with ``sink`` (a callable taking the response)
        the body is whatever ``sink`` returns."""
        query = query or {}
        path = f"{self.base_path}/{_quote(self.bucket)}" + (f"/{_quote(key, safe='/-_.~')}" if key else "/")
        target = path + ("?" + "&".join(f"{_quote(k)}={_quote(str(v))}" for k, v in sorted(query.items()))
                         if query else "")
        error: Exception | None = None
        for attempt in range(REQUEST_ATTEMPTS):
            try:
                headers = {**self._signed(method, path, query), **(extra or {})}
                conn = self._conn(fresh=attempt > 0)
                conn.request(method, target, headers=headers)
                resp = conn.getresponse()
                status = resp.status
                if status >= 500 or status == 429:
                    resp.read()
                    raise OSError(f"HTTP {status}")
                if sink is not None and status in (200, 206):
                    return status, resp.headers, sink(resp)
                body = resp.read()
                return status, resp.headers, body
            except (OSError, http.client.HTTPException) as exc:
                error = exc
                self._conn(fresh=True)
                time.sleep(min(30.0, 0.5 * 2 ** attempt))
        raise SyncError(f"{method} {self.uri}/{key or ''}: {error!r} after {REQUEST_ATTEMPTS} attempts")

    def list(self) -> dict:
        out, token = {}, None
        prefix = self.prefix + "/" if self.prefix else ""
        while True:
            query = {"list-type": "2", "prefix": prefix}
            if token:
                query["continuation-token"] = token
            status, _h, body = self._request("GET", None, query)
            if status != 200:
                raise SyncError(f"LIST {self.uri}: HTTP {status} {body[:300]!r}")
            root = ET.fromstring(body)
            ns = root.tag.partition("}")[0] + "}" if root.tag.startswith("{") else ""
            for item in root.iter(f"{ns}Contents"):
                key = item.findtext(f"{ns}Key") or ""
                if key.endswith("/"):
                    continue
                out[key[len(prefix):]] = int(item.findtext(f"{ns}Size") or 0)
            if (root.findtext(f"{ns}IsTruncated") or "").lower() != "true":
                return out
            token = root.findtext(f"{ns}NextContinuationToken")

    def read(self, rel: str) -> bytes | None:
        status, _h, body = self._request("GET", self._key(rel))
        if status == 404:
            return None
        if status != 200:
            raise SyncError(f"GET {self.uri}/{rel}: HTTP {status}")
        return body

    def meta_sha256(self, rel: str) -> str | None:
        status, headers, _b = self._request("HEAD", self._key(rel))
        if status != 200:
            raise SyncError(f"HEAD {self.uri}/{rel}: HTTP {status}")
        value = headers.get("x-amz-meta-" + SHA_META)
        return value.strip().lower() if value else None

    def read_into(self, rel: str, lo: int, hi: int, fd: int) -> int:
        def sink(resp) -> int:
            buf = bytearray(min(READ_BLOCK, hi - lo))
            view, done = memoryview(buf), 0
            while lo + done < hi:
                n = resp.readinto(view[: min(len(buf), hi - lo - done)])
                if not n:
                    break
                os.pwrite(fd, view[:n], lo + done)
                done += n
            return done

        if hi <= lo:
            return 0
        for attempt in range(REQUEST_ATTEMPTS):
            status, _h, done = self._request("GET", self._key(rel), extra={"Range": f"bytes={lo}-{hi - 1}"},
                                             sink=sink)
            if status not in (200, 206):
                raise SyncError(f"GET {self.uri}/{rel} bytes {lo}-{hi - 1}: HTTP {status}")
            if status == 200 and lo != 0:
                raise SyncError(f"GET {self.uri}/{rel}: the endpoint ignored the Range header")
            if done == hi - lo:
                return done
            self._conn(fresh=True)  # a short body: read the chunk again
            time.sleep(min(30.0, 0.5 * 2 ** attempt))
        raise SyncError(f"GET {self.uri}/{rel} bytes {lo}-{hi - 1}: short read after {REQUEST_ATTEMPTS} attempts")


def open_source(uri: str, environ: dict | None = None):
    if uri.startswith(("s3://", "r2://")):
        return S3Source(uri, environ)
    if uri.startswith("file://"):
        return DirSource(uri[len("file://"):])
    if "://" in uri:
        raise SyncError(f"source {uri!r}: expected s3://, r2://, file:// or a directory")
    return DirSource(uri)


# ---------------------------------------------------------------- manifest
def build_manifest(source, *, hash_missing: bool = False, workers: int = 16) -> dict:
    """``{"format", "source", "files": {rel: {"size", "sha256"}}}`` from the
    source's listing and each object's sha256 metadata (``hash_missing``: hash the
    files that have none by reading them; a directory source is always hashed)."""
    sizes = {rel: size for rel, size in source.list().items() if rel != MANIFEST_NAME}
    if not sizes:
        raise SyncError(f"{source.describe()} holds no files")

    def entry(rel: str) -> tuple:
        sha = source.meta_sha256(rel)
        if sha is None and (hash_missing or isinstance(source, DirSource)):
            sha = _hash_source(source, rel, sizes[rel])
        return rel, sha

    with futures.ThreadPoolExecutor(max(1, workers)) as pool:
        shas = dict(pool.map(entry, sorted(sizes)))
    return {"format": MANIFEST_FORMAT, "source": source.describe(),
            "files": {rel: {"size": sizes[rel], "sha256": shas[rel]} for rel in sorted(sizes)}}


def _hash_source(source, rel: str, size: int) -> str:
    if isinstance(source, DirSource):
        return hash_file(os.path.join(source.path, rel))
    digest = hashlib.sha256()
    status, _h, body = source._request("GET", source._key(rel))
    if status != 200 or len(body) != size:
        raise SyncError(f"GET {source.describe()}/{rel}: HTTP {status}, {len(body)} of {size} bytes")
    digest.update(body)
    return digest.hexdigest()


def load_manifest(source, *, workers: int = 16) -> dict:
    """The manifest object of the source when it has one (checked against the
    listing), else one built from the objects' sha256 metadata."""
    listing = source.list()
    raw = source.read(MANIFEST_NAME) if MANIFEST_NAME in listing else None
    if raw is None:
        manifest = build_manifest(source, workers=workers)
        origin = "object metadata"
    else:
        manifest = json.loads(raw)
        if manifest.get("format") != MANIFEST_FORMAT or not isinstance(manifest.get("files"), dict):
            raise SyncError(f"{source.describe()}/{MANIFEST_NAME} is not a {MANIFEST_FORMAT} manifest")
        origin = MANIFEST_NAME
    files = manifest["files"]
    for rel, item in files.items():
        if rel not in listing:
            raise SyncError(f"the manifest lists {rel}, which {source.describe()} does not hold")
        if int(listing[rel]) != int(item["size"]):
            raise SyncError(f"{rel}: the manifest says {item['size']} bytes, the source holds {listing[rel]}")
        sha = str(item.get("sha256") or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise SyncError(f"{rel} has no sha256 in the {origin} of {source.describe()}; write a manifest "
                            f"(python -m yeto.launch.weights_sync manifest --source ... --hash) next to the files")
        item["sha256"] = sha
    log(f"manifest: {len(files)} files, {_gb(sum(int(i['size']) for i in files.values()))} ({origin})")
    return manifest


# ---------------------------------------------------------------- pipeline stages
def stage_layer_ranges(num_layers: int, pp: int, first: int | None = None, last: int | None = None) -> list:
    """Megatron's decoder layers of each pipeline stage, as ``range`` objects: the
    first / last stage take ``first`` / ``last`` layers when given, the others an
    equal share. Without either and ``num_layers % pp``, yeto's run-config default
    (the remainder split over the first and last stage, the first one larger)."""
    pp = int(pp)
    if pp < 1:
        raise SyncError(f"pipeline parallel {pp} < 1")
    if pp == 1:
        return [range(num_layers)]
    if first is None and last is None and num_layers % pp:
        middle = num_layers // pp
        rem = num_layers - middle * pp
        first, last = middle + (rem + 1) // 2, middle + rem // 2
    fixed = [n for n in (first, last) if n is not None]
    rest_stages = pp - len(fixed)
    rest_layers = num_layers - sum(fixed)
    if any(n < 1 for n in fixed) or rest_layers < rest_stages or (rest_stages and rest_layers % rest_stages) \
            or (not rest_stages and rest_layers):
        raise SyncError(f"{num_layers} layers do not split over PP {pp} with first {first} / last {last}")
    share = rest_layers // rest_stages if rest_stages else 0
    sizes = [share] * pp
    if first is not None:
        sizes[0] = first
    if last is not None:
        sizes[-1] = last
    out, start = [], 0
    for size in sizes:
        out.append(range(start, start + size))
        start += size
    return out


def num_decoder_layers(config: dict) -> int:
    for doc in (config.get("text_config") or {}, config.get("language_config") or {}, config):
        if isinstance(doc, dict) and doc.get("num_hidden_layers"):
            return int(doc["num_hidden_layers"])
    raise SyncError("config.json has no num_hidden_layers")


def select_shards(weight_map: dict, num_layers: int, pp: int, stages, first: int | None = None,
                  last: int | None = None) -> set:
    """The safetensors files that pipeline ``stages`` read (module doc)."""
    ranges = stage_layer_ranges(num_layers, pp, first, last)
    wanted = set()
    for stage in stages:
        if not 0 <= int(stage) < pp:
            raise SyncError(f"stage {stage} is not one of PP {pp}'s stages")
        wanted.update(ranges[int(stage)])
    last_stage = pp - 1 in {int(s) for s in stages}
    out = set()
    for name, filename in weight_map.items():
        if filename in out:
            continue
        match = LAYER_RE.search(name)
        if match is None:
            out.add(filename)  # embedding, final norm, head, vision tower: every stage
            continue
        layer = int(match.group(1))
        if layer in wanted or (layer >= num_layers and last_stage):
            out.add(filename)
    return out


# ---------------------------------------------------------------- worker tasks
_SOURCES: dict = {}


def _worker_source(spec: tuple):
    source = _SOURCES.get(spec)
    if source is None:
        uri, env = spec[0], dict(spec[1])
        source = _SOURCES[spec] = open_source(uri, env)
    return source


def task_fetch(spec: tuple, rel: str, lo: int, hi: int, part: str) -> int:
    """Download bytes [lo, hi) of ``rel`` into ``part`` (a worker task)."""
    source = _worker_source(spec)
    fd = os.open(part, os.O_WRONLY)
    try:
        done = source.read_into(rel, lo, hi, fd)
    finally:
        os.close(fd)
    if done != hi - lo:
        raise SyncError(f"{rel}: {done} of {hi - lo} bytes at offset {lo}")
    return done


def hash_file(path: str) -> str:
    digest = hashlib.sha256()
    buf = bytearray(READ_BLOCK)
    view = memoryview(buf)
    with open(path, "rb", buffering=0) as fh:
        while True:
            n = fh.readinto(view)
            if not n:
                return digest.hexdigest()
            digest.update(view[:n])


# ---------------------------------------------------------------- the sync
class _Pool:
    """Worker threads, or (``procs``) worker processes forked from this one."""

    def __init__(self, workers: int, procs: bool) -> None:
        if procs:
            ctx = multiprocessing.get_context("fork")
            self.pool = futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        else:
            self.pool = futures.ThreadPoolExecutor(max_workers=workers)

    def submit(self, fn, *args):
        return self.pool.submit(fn, *args)

    def close(self, cancel: bool = False) -> None:
        if sys.version_info >= (3, 9):  # noqa: UP036 - this file runs on the node's python3
            self.pool.shutdown(wait=True, cancel_futures=cancel)
        else:  # pragma: no cover - python 3.8 hosts
            self.pool.shutdown(wait=True)


class Sync:
    """One sync of ``files`` (``{rel: {"size", "sha256"}}``) from ``source`` into
    ``snapshot`` with work state in ``work`` (module doc)."""

    def __init__(self, source, spec: tuple, snapshot: str, work: str, *, chunk: int, workers: int, procs: bool,
                 headroom: float, attempts: int, progress_s: float, rehash: bool) -> None:
        self.source, self.spec = source, spec
        self.snapshot, self.work = snapshot, work
        self.parts = os.path.join(work, "parts")
        self.chunk, self.workers, self.procs = max(1, int(chunk)), max(1, int(workers)), procs
        self.headroom, self.attempts, self.progress_s, self.rehash = headroom, max(1, attempts), progress_s, rehash
        self.verified_path = os.path.join(work, "verified.json")
        try:
            with open(self.verified_path) as fh:
                self.verified = json.load(fh)
        except (FileNotFoundError, ValueError):
            self.verified = {}
        self.stats = {"files": 0, "skipped": 0, "hashed": 0, "downloaded": 0, "redownloaded": 0,
                      "bytes_total": 0, "bytes_fetched": 0, "bytes_resumed": 0}

    # -- state files
    def _final(self, rel: str) -> str:
        return os.path.join(self.snapshot, rel)

    def _part(self, rel: str) -> str:
        return os.path.join(self.parts, rel)

    def _journal(self, rel: str) -> str:
        return self._part(rel) + ".done"

    def _save_verified(self) -> None:
        tmp = self.verified_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.verified, fh, indent=0, sort_keys=True)
        os.replace(tmp, self.verified_path)

    def _stamp(self, rel: str, sha: str) -> None:
        st = os.stat(self._final(rel))
        self.verified[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha}
        self._save_verified()

    def _stamped(self, rel: str, item: dict) -> bool:
        stamp = self.verified.get(rel)
        if not stamp or self.rehash:
            return False
        try:
            st = os.stat(self._final(rel))
        except FileNotFoundError:
            return False
        return (stamp.get("sha256") == item["sha256"] and stamp.get("size") == st.st_size == int(item["size"])
                and stamp.get("mtime_ns") == st.st_mtime_ns)

    def _chunks(self, size: int) -> list:
        return [(i, i * self.chunk, min(size, (i + 1) * self.chunk)) for i in range((size + self.chunk - 1) // self.chunk)]

    def _journal_done(self, rel: str, item: dict) -> set:
        """Finished chunk indices of a part that matches ``item`` (else the part is dropped)."""
        part, journal = self._part(rel), self._journal(rel)
        header = f"yeto-part {item['size']} {item['sha256']} {self.chunk}"
        try:
            with open(journal) as fh:
                lines = fh.read().split("\n")
        except FileNotFoundError:
            lines = []
        if lines and lines[0] == header and os.path.exists(part) and os.path.getsize(part) == int(item["size"]):
            return {int(x) for x in lines[1:-1] if x.isdigit()}  # the last element: after the final newline
        for path in (part, journal):
            if os.path.exists(path):
                os.unlink(path)
        return set()

    def _start_part(self, rel: str, item: dict) -> None:
        part = self._part(rel)
        os.makedirs(os.path.dirname(part), exist_ok=True)
        fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            os.ftruncate(fd, int(item["size"]))  # sparse: blocks are allocated as chunks land
        finally:
            os.close(fd)
        with open(self._journal(rel), "w") as fh:
            fh.write(f"yeto-part {item['size']} {item['sha256']} {self.chunk}\n")

    # -- plan
    def plan(self, files: dict) -> tuple:
        """``(to_hash, to_fetch)``: files in place to verify by hashing, and
        ``{rel: (item, done_chunks)}`` to download (fresh or resumed)."""
        to_hash, to_fetch = [], {}
        for rel, item in sorted(files.items()):
            final = self._final(rel)
            if os.path.exists(final):
                if os.path.getsize(final) != int(item["size"]):
                    os.unlink(final)
                elif self._stamped(rel, item):
                    self.stats["skipped"] += 1
                    continue
                else:
                    to_hash.append(rel)
                    continue
            to_fetch[rel] = (item, self._journal_done(rel, item))
        return to_hash, to_fetch

    def need_bytes(self, to_fetch: dict) -> int:
        need = 0
        for item, done in to_fetch.values():
            need += int(item["size"]) - sum(hi - lo for i, lo, hi in self._chunks(int(item["size"])) if i in done)
        return need

    def check_space(self, need: int, root: str) -> None:
        free = shutil.disk_usage(root).free
        if free < self.headroom * need:
            raise SyncError(f"not enough disk space on {root}: {_gb(need)} to download needs "
                            f"{_gb(self.headroom * need)} free ({self.headroom:g}x), the file system has {_gb(free)}")

    # -- run
    def run(self, files: dict, root: str) -> None:
        self.stats["files"] += len(files)
        self.stats["bytes_total"] += sum(int(i["size"]) for i in files.values())
        skipped = self.stats["skipped"]
        to_hash, to_fetch = self.plan(files)
        skipped = self.stats["skipped"] - skipped
        need = self.need_bytes(to_fetch)
        self.check_space(need, root)
        resumed = sum(int(item["size"]) for item, _ in to_fetch.values()) - need
        self.stats["bytes_resumed"] += resumed
        if to_fetch or to_hash:
            log(f"{len(files)} files: {skipped} verified earlier, {len(to_hash)} in place to verify, "
                f"{len(to_fetch)} to download ({_gb(need)}" + (f", {_gb(resumed)} already in parts" if resumed else "")
                + f"), {self.workers} {'processes' if self.procs else 'threads'}, {self.chunk >> 20} MiB chunks")
        pool = _Pool(self.workers, self.procs)
        try:
            self._drive(pool, files, to_hash, to_fetch, need)
        except BaseException:
            pool.close(cancel=True)
            raise
        pool.close()

    def _drive(self, pool: _Pool, files: dict, to_hash: list, to_fetch: dict, need: int) -> None:
        pending = {}  # future -> (kind, rel, chunk index)
        remaining = {}  # rel -> chunk indices not finished yet
        attempts = {rel: 0 for rel in files}
        journals = {}

        def fetch(rel: str, item: dict, done: set) -> None:
            attempts[rel] += 1
            if not done:
                self._start_part(rel, item)
            todo = [c for c in self._chunks(int(item["size"])) if c[0] not in done]
            remaining[rel] = {c[0] for c in todo}
            if not todo:
                verify_part(rel)
            for index, lo, hi in todo:
                pending[pool.submit(task_fetch, self.spec, rel, lo, hi, self._part(rel))] = ("chunk", rel, index)

        def verify_part(rel: str) -> None:
            journal = journals.pop(rel, None)
            if journal is not None:
                journal.close()
            pending[pool.submit(hash_file, self._part(rel))] = ("part", rel, None)

        for rel in to_hash:
            pending[pool.submit(hash_file, self._final(rel))] = ("final", rel, None)
        for rel, (item, done) in to_fetch.items():
            fetch(rel, item, done)

        t0 = last_t = time.monotonic()
        fetched = last_fetched = 0
        while pending:
            ready, _ = futures.wait(list(pending), timeout=self.progress_s, return_when=futures.FIRST_COMPLETED)
            for fut in ready:
                kind, rel, index = pending.pop(fut)
                item = files[rel]
                try:
                    result = fut.result()
                except Exception as exc:  # one failed task ends the sync
                    raise SyncError(f"{rel}: {exc}") from exc
                if kind == "chunk":
                    fetched += result
                    remaining[rel].discard(index)
                    journal = journals.get(rel)
                    if journal is None:
                        journal = journals[rel] = open(self._journal(rel), "a")  # noqa: SIM115 - closed in verify_part
                    journal.write(f"{index}\n")
                    journal.flush()
                    if not remaining[rel]:
                        verify_part(rel)
                elif kind == "final":
                    self.stats["hashed"] += 1
                    if result == item["sha256"]:
                        self._stamp(rel, result)
                    else:
                        log(f"{rel}: sha256 {result[:12]} != manifest {item['sha256'][:12]}; downloading it again")
                        os.unlink(self._final(rel))
                        self.stats["redownloaded"] += 1
                        need += int(item["size"])
                        fetch(rel, item, self._journal_done(rel, item))
                else:  # a complete part
                    if result == item["sha256"]:
                        final = self._final(rel)
                        os.makedirs(os.path.dirname(final), exist_ok=True)
                        os.replace(self._part(rel), final)
                        os.unlink(self._journal(rel))
                        self._stamp(rel, result)
                        self.stats["downloaded"] += 1
                    else:
                        for path in (self._part(rel), self._journal(rel)):
                            if os.path.exists(path):
                                os.unlink(path)
                        if attempts[rel] >= self.attempts:
                            raise SyncError(f"{rel}: sha256 {result} does not match the manifest's {item['sha256']} "
                                            f"after {attempts[rel]} downloads")
                        log(f"{rel}: downloaded sha256 {result[:12]} != manifest {item['sha256'][:12]}; "
                            "downloading it again")
                        self.stats["redownloaded"] += 1
                        need += int(item["size"])
                        fetch(rel, item, set())
            now = time.monotonic()
            if pending and now - last_t >= self.progress_s:
                rate = (fetched - last_fetched) / max(1e-9, now - last_t)
                avg = fetched / max(1e-9, now - t0)
                eta = (need - fetched) / avg if avg > 0 else float("inf")
                log(f"{_gb(fetched)} / {_gb(need)} ({100.0 * fetched / max(1, need):.0f}%), "
                    f"{rate / 1e9:.2f} GB/s now, {avg / 1e9:.2f} GB/s avg, ETA {eta:.0f} s, "
                    f"{len([p for p in pending.values() if p[0] != 'chunk'])} hashing")
                last_t, last_fetched = now, fetched
        self.stats["bytes_fetched"] += fetched


def repo_dirname(model: str) -> str:
    return "models--" + model.replace("/", "--")


def store_paths(root: str, model: str, revision: str) -> dict:
    flat = model.replace("/", "--")
    return {"snapshot": os.path.join(root, "hub", repo_dirname(model), "snapshots", revision),
            "work": os.path.join(root, "yeto-weights", f"{flat}@{revision}"),
            "marker": os.path.join(root, "yeto-complete", f"{flat}@{revision}.json")}


def sync(source_uri: str, root: str, model: str, revision: str, *, stages=None, pp: int = 1,
         pp_first: int | None = None, pp_last: int | None = None, chunk_mb: int = DEFAULT_CHUNK_MB,
         workers: int = DEFAULT_WORKERS, procs: bool = False, headroom: float = DEFAULT_HEADROOM,
         attempts: int = 2, progress_s: float = 10.0, rehash: bool = False, environ: dict | None = None,
         node: str | None = None, chunk_bytes: int | None = None) -> dict:
    """Copy ``model@revision`` from ``source_uri`` under ``root`` (module doc);
    returns the summary the marker holds."""
    if not HEX40.fullmatch(revision or ""):
        raise SyncError(f"revision {revision!r}: a 40-hex commit (the hub cache's snapshot name) is required")
    env = dict(os.environ if environ is None else environ)
    source = open_source(source_uri, env)
    spec = (source_uri, tuple(sorted((k, v) for k, v in env.items() if k.startswith((ENV_PREFIX, "AWS_")))))
    paths = store_paths(root, model, revision)
    for key in ("snapshot", "work"):
        os.makedirs(paths[key], exist_ok=True)
    os.makedirs(os.path.dirname(paths["marker"]), exist_ok=True)
    if os.path.exists(paths["marker"]):
        os.unlink(paths["marker"])  # rewritten when this sync completes
    t0 = time.time()
    manifest = load_manifest(source)
    files = manifest["files"]
    job = Sync(source, spec, paths["snapshot"], paths["work"], chunk=chunk_bytes or int(chunk_mb) << 20, workers=workers,
               procs=procs, headroom=headroom, attempts=attempts, progress_s=progress_s, rehash=rehash)
    meta = {rel: item for rel, item in files.items() if not rel.endswith(".safetensors")}
    job.run(meta, root)
    shards = {rel: item for rel, item in files.items() if rel.endswith(".safetensors")}
    selected = sorted(shards)
    if stages is not None:
        index_path = os.path.join(paths["snapshot"], INDEX_NAME)
        if not os.path.exists(index_path):
            raise SyncError(f"stage selection needs {INDEX_NAME}, which {source.describe()} does not hold")
        with open(index_path) as fh:
            weight_map = json.load(fh)["weight_map"]
        with open(os.path.join(paths["snapshot"], "config.json")) as fh:
            layers = num_decoder_layers(json.load(fh))
        picked = select_shards(weight_map, layers, pp, stages, pp_first, pp_last)
        missing = sorted(picked - set(shards))
        if missing:
            raise SyncError(f"{INDEX_NAME} names {len(missing)} files the source does not hold, e.g. {missing[:3]}")
        unindexed = sorted(set(shards) - set(weight_map.values()))
        selected = sorted(picked | set(unindexed))
        log(f"stages {','.join(str(s) for s in stages)} of PP {pp} ({layers} layers, "
            + ", ".join(f"stage {i}: {r.start}-{r.stop - 1}" for i, r in
                        enumerate(stage_layer_ranges(layers, pp, pp_first, pp_last)))
            + f"): {len(selected)} of {len(shards)} safetensors files, "
            f"{_gb(sum(int(shards[r]['size']) for r in selected))} of {_gb(sum(int(i['size']) for i in shards.values()))}")
    job.run({rel: shards[rel] for rel in selected}, root)
    took = time.time() - t0
    summary = {"model": model, "revision": revision, "source": source.describe(), "node": node,
               "partial": len(selected) < len(shards), "stages": list(stages) if stages is not None else None,
               "pp": pp, "pp_first": pp_first, "pp_last": pp_last,
               "files": sorted(meta) + selected, "seconds": round(took, 1),
               "fetch_gbps": round(job.stats["bytes_fetched"] / max(took, 1e-9) / 1e9, 3), **job.stats}
    tmp = paths["marker"] + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(summary, fh, indent=1)
    os.replace(tmp, paths["marker"])
    log(f"done: {summary['downloaded']} downloaded ({_gb(job.stats['bytes_fetched'])}), {summary['skipped']} "
        f"verified earlier, {summary['hashed']} hashed in place, {_gb(job.stats['bytes_total'])} in "
        f"{paths['snapshot']} after {took:.0f} s ({summary['fetch_gbps']:.2f} GB/s)")
    return summary


# ---------------------------------------------------------------- CLI
def _stages(text: str | None):
    if text is None or text == "":
        return None
    return sorted({int(x) for x in text.split(",") if x.strip()})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="weights_sync", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("sync", help="copy model@revision from the source under --root")
    run.add_argument("--source", required=True)
    run.add_argument("--root", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--stages", default=None, help="pipeline stages to copy (comma separated); unset: all files")
    run.add_argument("--pp", type=int, default=1)
    run.add_argument("--pp-first", type=int, default=None)
    run.add_argument("--pp-last", type=int, default=None)
    run.add_argument("--chunk-mb", type=int, default=DEFAULT_CHUNK_MB)
    run.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    run.add_argument("--procs", action="store_true", help="workers are forked processes (Linux), not threads")
    run.add_argument("--headroom", type=float, default=DEFAULT_HEADROOM)
    run.add_argument("--attempts", type=int, default=2)
    run.add_argument("--progress-s", type=float, default=10.0)
    run.add_argument("--rehash", action="store_true", help="hash files in place even when verified earlier")
    run.add_argument("--node", default=None, help="a name for the log and the marker")
    man = sub.add_parser("manifest", help=f"write the source's {MANIFEST_NAME}")
    man.add_argument("--source", required=True)
    man.add_argument("--out", required=True)
    man.add_argument("--hash", action="store_true", help="hash objects without sha256 metadata")
    plan = sub.add_parser("plan", help="the files and bytes of each pipeline stage (reads config and index)")
    plan.add_argument("--source", required=True)
    plan.add_argument("--pp", type=int, required=True)
    plan.add_argument("--pp-first", type=int, default=None)
    plan.add_argument("--pp-last", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "sync":
            summary = sync(args.source, args.root, args.model, args.revision, stages=_stages(args.stages),
                           pp=args.pp, pp_first=args.pp_first, pp_last=args.pp_last, chunk_mb=args.chunk_mb,
                           workers=args.workers, procs=args.procs, headroom=args.headroom, attempts=args.attempts,
                           progress_s=args.progress_s, rehash=args.rehash, node=args.node)
            print(f"{SUMMARY_TAG} {json.dumps({k: v for k, v in summary.items() if k != 'files'})}", flush=True)
        elif args.cmd == "manifest":
            doc = build_manifest(open_source(args.source), hash_missing=args.hash)
            with open(args.out, "w") as fh:
                json.dump(doc, fh, indent=1, sort_keys=True)
            log(f"wrote {args.out}: {len(doc['files'])} files")
        else:
            source = open_source(args.source)
            sizes = source.list()
            weight_map = json.loads(source.read(INDEX_NAME) or b"{}").get("weight_map") or {}
            layers = num_decoder_layers(json.loads(source.read("config.json") or b"{}"))
            ranges = stage_layer_ranges(layers, args.pp, args.pp_first, args.pp_last)
            small = sum(size for rel, size in sizes.items() if not rel.endswith(".safetensors"))
            for stage, layer_range in enumerate(ranges):
                picked = select_shards(weight_map, layers, args.pp, [stage], args.pp_first, args.pp_last)
                print(json.dumps({"stage": stage, "layers": [layer_range.start, layer_range.stop - 1],
                                  "files": len(picked), "bytes": sum(sizes.get(f, 0) for f in picked) + small}))
    except SyncError as exc:
        log(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
