"""Node-side weights copy (yeto.launch.weights_sync) against a fake S3 endpoint:
full copy, resume, sha256 mismatch, pipeline-stage selection, disk-space refusal."""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import sys
import threading
import time
import urllib.parse
from collections import Counter
from pathlib import Path

import pytest

from yeto.launch import weights_sync as ws

REV = "c" * 40
MODEL = "org/tiny"
CHUNK = 64 * 1024
LAYERS = 6
# layer files: 2 layers each; the shared file holds the embedding, final norm and head
LAYER_FILES = {f"model-0000{i + 1}-of-00004.safetensors": (2 * i, 2 * i + 1) for i in range(3)}
SHARED_FILE = "model-00004-of-00004.safetensors"


def tiny_model(seed: int = 0) -> dict[str, bytes]:
    weight_map = {}
    for name, layers in LAYER_FILES.items():
        for layer in layers:
            for t in ("self_attn.q_proj.weight", "mlp.experts.0.w1.weight_packed"):
                weight_map[f"language_model.model.layers.{layer}.{t}"] = name
    for t in ("language_model.model.embed_tokens.weight", "language_model.model.norm.weight",
              "language_model.lm_head.weight"):
        weight_map[t] = SHARED_FILE
    files = {"config.json": json.dumps({"text_config": {"num_hidden_layers": LAYERS}}).encode(),
             ws.INDEX_NAME: json.dumps({"metadata": {}, "weight_map": weight_map}).encode(),
             "tokenizer_config.json": b"{}", "assets/logo.png": b"\x89PNG"}
    for i, name in enumerate([*LAYER_FILES, SHARED_FILE]):
        size = 3 * CHUNK + 1000 * (i + 1) + seed  # 4 chunks, the last one short
        files[name] = hashlib.sha256(f"{name}{seed}".encode()).digest() * (size // 32) + b"x" * (size % 32)
    return files


class FakeS3:
    """Path-style S3 over HTTP: ListObjectsV2 (paged), HEAD with ``x-amz-meta-sha256``,
    GET with Range. ``fail_after``: range GETs answered before every later one gets
    403; ``corrupt``: key -> how many range GETs at offset 0 return flipped bytes;
    ``deny``: (key, offset) range GETs answered 403; ``stall``: (key, offset) -> seconds
    a range GET waits before it is answered."""

    def __init__(self, objects: dict[str, bytes], *, bucket: str = "bkt", meta_sha: bool = True,
                 page: int = 3) -> None:
        self.bucket, self.objects, self.meta_sha, self.page = bucket, dict(objects), meta_sha, page
        self.fail_after: int | None = None
        self.corrupt: Counter = Counter()
        self.ranges: Counter = Counter()  # key -> range GETs answered
        self.deny: set = set()
        self.stall: dict = {}
        self.unsigned = 0
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # quiet
                pass

            def _reply(self, status, body=b"", headers=None):
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _key(self):
                url = urllib.parse.urlsplit(self.path)
                parts = urllib.parse.unquote(url.path).lstrip("/").split("/", 1)
                if not self.headers.get("Authorization", "").startswith("AWS4-HMAC-SHA256 Credential=kid/"):
                    fake.unsigned += 1
                return parts[0], (parts[1] if len(parts) > 1 else ""), urllib.parse.parse_qs(url.query)

            def do_HEAD(self):
                bucket, key, _q = self._key()
                data = fake.objects.get(key) if bucket == fake.bucket else None
                if data is None:
                    return self._reply(404)
                headers = {"x-amz-meta-sha256": hashlib.sha256(data).hexdigest()} if fake.meta_sha else {}
                self.send_response(200)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()

            def do_GET(self):
                bucket, key, query = self._key()
                if bucket != fake.bucket:
                    return self._reply(404)
                if not key and query.get("list-type") == ["2"]:
                    return self._list(query)
                data = fake.objects.get(key)
                if data is None:
                    return self._reply(404)
                rng = self.headers.get("Range")
                if not rng:
                    return self._reply(200, data)
                lo, hi = (int(x) for x in rng.removeprefix("bytes=").split("-"))
                if (key, lo) in fake.deny:
                    return self._reply(403, b"<Error>denied</Error>")
                if (key, lo) in fake.stall:
                    time.sleep(fake.stall[key, lo])
                if fake.fail_after is not None and sum(fake.ranges.values()) >= fake.fail_after:
                    return self._reply(403, b"<Error>denied</Error>")
                fake.ranges[key] += 1
                body = data[lo:hi + 1]
                if lo == 0 and fake.corrupt[key] > 0:
                    fake.corrupt[key] -= 1
                    body = bytes([body[0] ^ 0xFF]) + body[1:]
                self._reply(206, body, {"Content-Range": f"bytes {lo}-{hi}/{len(data)}"})

            def _list(self, query):
                prefix = query.get("prefix", [""])[0]
                keys = sorted(k for k in fake.objects if k.startswith(prefix))
                start = int(query.get("continuation-token", ["0"])[0])
                page = keys[start:start + fake.page]
                more = start + fake.page < len(keys)
                items = "".join(f"<Contents><Key>{k}</Key><Size>{len(fake.objects[k])}</Size></Contents>"
                                for k in page)
                token = f"<NextContinuationToken>{start + fake.page}</NextContinuationToken>" if more else ""
                body = ('<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                        f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>{items}{token}"
                        "</ListBucketResult>").encode()
                self._reply(200, body, {"Content-Type": "application/xml"})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def env(self) -> dict[str, str]:
        return {"YETO_WEIGHTS_S3_ACCESS_KEY_ID": "kid", "YETO_WEIGHTS_S3_SECRET_ACCESS_KEY": "secret",
                "YETO_WEIGHTS_S3_ENDPOINT_URL": self.endpoint, "YETO_WEIGHTS_S3_REGION": "auto"}

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def model():
    return tiny_model()


@pytest.fixture
def s3(model):
    fake = FakeS3({f"models/tiny/{k}": v for k, v in model.items()})
    yield fake
    fake.close()


def run_sync(s3, root, **kw):
    kw = {"chunk_bytes": CHUNK, "workers": 4, "progress_s": 0.05, "environ": s3.env(), **kw}
    return ws.sync("s3://bkt/models/tiny", str(root), MODEL, REV, **kw)


def snapshot(root: Path) -> Path:
    return root / "hub" / "models--org--tiny" / "snapshots" / REV


def test_full_copy_lands_in_the_hub_layout_with_marker_and_no_work_files_in_the_snapshot(s3, model, tmp_path):
    summary = run_sync(s3, tmp_path)
    snap = snapshot(tmp_path)
    got = {p.relative_to(snap).as_posix(): p.read_bytes() for p in snap.rglob("*") if p.is_file()}
    assert got == model  # every file, byte-equal; nothing else in the snapshot (no parts, no journals)
    marker = json.loads((tmp_path / "yeto-complete" / f"org--tiny@{REV}.json").read_text())
    assert marker["partial"] is False and marker["downloaded"] == len(model) and marker["skipped"] == 0
    assert summary["bytes_fetched"] == summary["bytes_total"] == sum(map(len, model.values()))
    assert s3.unsigned == 0
    # chunked: a 4-chunk file took 4 range requests
    assert s3.ranges[f"models/tiny/{SHARED_FILE}"] == 4
    assert not list((tmp_path / "yeto-weights" / f"org--tiny@{REV}" / "parts").rglob("*.safetensors"))


def test_a_second_sync_skips_every_verified_file_without_downloading_or_hashing(s3, tmp_path, monkeypatch):
    run_sync(s3, tmp_path)
    before = sum(s3.ranges.values())
    hashed = []
    monkeypatch.setattr(ws, "hash_file", lambda p: hashed.append(p) or "0" * 64)
    summary = run_sync(s3, tmp_path)
    assert sum(s3.ranges.values()) == before and hashed == []
    assert summary["skipped"] == summary["files"] and summary["downloaded"] == 0


def test_an_interrupted_copy_resumes_from_its_journal(s3, model, tmp_path):
    s3.fail_after = 9  # 4 metadata files (1 chunk each) + 5 chunks of the shards, then 403
    with pytest.raises(ws.SyncError, match="HTTP 403"):
        run_sync(s3, tmp_path, workers=1)
    assert not (tmp_path / "yeto-complete" / f"org--tiny@{REV}.json").exists()
    done_before = sum(s3.ranges.values())
    assert done_before == 9
    s3.fail_after = None
    s3.ranges.clear()
    summary = run_sync(s3, tmp_path)
    total_chunks = sum((len(v) + CHUNK - 1) // CHUNK for k, v in model.items() if k.endswith(".safetensors"))
    assert sum(s3.ranges.values()) == total_chunks - 5  # the 5 finished shard chunks were kept
    # file 1 finished (its hash was still queued) and chunk 0 of file 2; the metadata files were verified
    assert summary["bytes_resumed"] == 4 * CHUNK + 1000 and summary["skipped"] == 4
    snap = snapshot(tmp_path)
    assert all((snap / k).read_bytes() == v for k, v in model.items())


def _one_batch_failures_first(monkeypatch):
    """The shard run's first wait (its 16 chunk tasks; the metadata run has 4) returns
    every task at once, the failed ones first: a failure that comes before the chunks
    that landed with it in one batch."""
    real, first = ws.futures.wait, []

    def wait(fs, timeout=None, return_when=ws.futures.ALL_COMPLETED):
        if return_when == ws.futures.FIRST_COMPLETED and not first and len(fs) >= 16:
            first.append(True)
            done, rest = real(fs, return_when=ws.futures.ALL_COMPLETED)
            return sorted(done, key=lambda f: f.exception() is None), rest
        return real(fs, timeout=timeout, return_when=return_when)

    monkeypatch.setattr(ws.futures, "wait", wait)


def test_chunks_that_landed_with_a_failing_one_stay_in_the_journal(s3, model, tmp_path, monkeypatch):
    shard = "model-00002-of-00004.safetensors"
    s3.deny.add((f"models/tiny/{shard}", 0))
    with monkeypatch.context() as m:
        _one_batch_failures_first(m)
        with pytest.raises(ws.SyncError, match=rf"^{shard}: .*HTTP 403") as raised:
            run_sync(s3, tmp_path)
    assert type(raised.value) is ws.SyncError and isinstance(raised.value.__cause__, Exception)
    assert sum(s3.ranges.values()) == 4 + 16 - 1  # the 4 metadata files, then every other shard chunk
    s3.deny.clear()
    s3.ranges.clear()
    summary = run_sync(s3, tmp_path)
    assert dict(s3.ranges) == {f"models/tiny/{shard}": 1}, "only the chunk that failed is fetched again"
    shards = sum(len(v) for k, v in model.items() if k.endswith(".safetensors"))
    assert summary["bytes_resumed"] == shards - CHUNK and summary["skipped"] == 4  # the metadata run had finished
    snap = snapshot(tmp_path)
    assert all((snap / k).read_bytes() == v for k, v in model.items())


def test_a_failing_sync_waits_for_its_running_chunks_up_to_a_bound(s3, tmp_path, monkeypatch):
    """Chunks 1 and 2 of file 1 are running when chunk 0 of file 2 fails: chunk 2 lands
    within FAIL_WAIT_S and is journaled; chunk 1 does not, is not journaled, and the
    error is raised without waiting for it."""
    slow = "model-00001-of-00004.safetensors"
    monkeypatch.setattr(ws, "FAIL_WAIT_S", 1.0)
    s3.stall[f"models/tiny/{slow}", CHUNK] = 4.0
    s3.stall[f"models/tiny/{slow}", 2 * CHUNK] = 0.3
    s3.deny.add(("models/tiny/model-00002-of-00004.safetensors", 0))
    t0 = time.monotonic()
    with pytest.raises(ws.SyncError, match="HTTP 403"):
        run_sync(s3, tmp_path)
    assert time.monotonic() - t0 < 3.0, "the error is raised without waiting for the slow chunk"
    journal = tmp_path / "yeto-weights" / f"org--tiny@{REV}" / "parts" / f"{slow}.done"
    done = journal.read_text().split("\n")[1:]
    assert "2" in done, "a chunk that landed while the sync was failing is journaled"
    assert "1" not in done, "a chunk that did not land in time is not journaled"


def test_a_sha256_mismatch_downloads_the_file_again_and_fails_when_it_persists(s3, model, tmp_path):
    key = f"models/tiny/{SHARED_FILE}"
    s3.corrupt[key] = 1
    summary = run_sync(s3, tmp_path)
    assert summary["redownloaded"] == 1 and s3.ranges[key] == 8
    assert (snapshot(tmp_path) / SHARED_FILE).read_bytes() == model[SHARED_FILE]
    other = tmp_path / "other"
    s3.corrupt[key] = 5
    with pytest.raises(ws.SyncError, match=f"{SHARED_FILE}: sha256 .* after 2 downloads"):
        run_sync(s3, other)
    assert not (snapshot(other) / SHARED_FILE).exists()  # a bad file never enters the snapshot


def test_a_file_in_place_with_other_bytes_is_hashed_and_replaced(s3, model, tmp_path):
    snap = snapshot(tmp_path)
    (snap / "assets").mkdir(parents=True)
    bad = bytearray(model[SHARED_FILE])
    bad[100] ^= 1
    (snap / SHARED_FILE).write_bytes(bytes(bad))  # same size, never verified
    (snap / "config.json").write_bytes(model["config.json"])  # right bytes, never verified
    summary = run_sync(s3, tmp_path)
    assert (snap / SHARED_FILE).read_bytes() == model[SHARED_FILE]
    assert summary["hashed"] == 2 and summary["redownloaded"] == 1
    assert s3.ranges["models/tiny/config.json"] == 0  # verified in place by its hash


def test_stage_selection_copies_only_that_stages_layer_files_plus_shared_and_small_files(s3, model, tmp_path):
    # PP 3 over 6 layers, first 1 / last 1: stage 1 runs layers 1-4 (files 1, 2 and 3)
    summary = run_sync(s3, tmp_path / "mid", stages=[1], pp=3, pp_first=1, pp_last=1)
    names = {p.name for p in snapshot(tmp_path / "mid").rglob("*") if p.is_file()}
    assert names >= {"config.json", ws.INDEX_NAME, "logo.png", SHARED_FILE, *LAYER_FILES}
    # even split: stage 0 = layers 0-1 (file 1), stage 2 = layers 4-5 (file 3)
    run_sync(s3, tmp_path / "s0", stages=[0], pp=3)
    shards = {p.name for p in snapshot(tmp_path / "s0").glob("*.safetensors")}
    assert shards == {"model-00001-of-00004.safetensors", SHARED_FILE}
    s2 = run_sync(s3, tmp_path / "s2", stages=[2], pp=3)
    assert {p.name for p in snapshot(tmp_path / "s2").glob("*.safetensors")} == \
        {"model-00003-of-00004.safetensors", SHARED_FILE}
    assert s2["partial"] is True and s2["stages"] == [2] and summary["partial"] is False
    # the index stays whole: mbridge reads only the files of the tensors a stage loads
    assert (snapshot(tmp_path / "s0") / ws.INDEX_NAME).read_bytes() == model[ws.INDEX_NAME]


def test_select_shards_rules():
    weight_map = {
        "model.layers.0.mlp.w": "a", "model.layers.1.mlp.w": "b", "model.layers.2.mlp.w": "c",
        "model.layers.3.eh_proj.weight": "mtp",  # past num_hidden_layers: an MTP block, last stage
        "model.embed_tokens.weight": "emb", "lm_head.weight": "head",
        "vision_tower.encoder.layers.0.attn.w": "vis",  # a vision layer is not a decoder layer
        "model.language_model.layers.1.x": "b2",  # transformers >= 4.52 VL naming
    }
    shared = {"emb", "head", "vis"}
    assert ws.select_shards(weight_map, 3, 3, [0]) == {"a"} | shared
    assert ws.select_shards(weight_map, 3, 3, [1]) == {"b", "b2"} | shared
    assert ws.select_shards(weight_map, 3, 3, [2]) == {"c", "mtp"} | shared
    with pytest.raises(ws.SyncError, match="stage 3"):
        ws.select_shards(weight_map, 3, 3, [3])


def test_stage_layer_ranges_follow_megatron_and_the_run_config_default():
    def spans(*a):
        return [(r.start, r.stop - 1) for r in ws.stage_layer_ranges(*a)]

    assert spans(93, 3) == [(0, 30), (31, 61), (62, 92)]
    assert spans(93, 3, 32, 32) == [(0, 31), (32, 60), (61, 92)]
    assert spans(93, 3, 32, 33) == [(0, 31), (32, 59), (60, 92)]
    assert spans(94, 3) == [(0, 31), (32, 62), (63, 93)]  # run_config: 31 + remainder 1 on the first stage
    assert spans(9, 4, 3, None) == [(0, 2), (3, 4), (5, 6), (7, 8)]
    assert spans(5, 1) == [(0, 4)]
    assert spans(10, 4) == [(0, 2), (3, 4), (5, 6), (7, 9)]
    for bad in ((93, 3, 36, 60), (10, 4, 5, None), (93, 3, 0, 32)):
        with pytest.raises(ws.SyncError):
            ws.stage_layer_ranges(*bad)


def test_not_enough_disk_space_is_refused_before_any_shard_byte(s3, tmp_path, monkeypatch):
    real = shutil.disk_usage

    def tight(path):
        usage = real(path)
        return type(usage)(usage.total, usage.used, 10 * CHUNK)  # less than 1.2 x the shards' bytes

    monkeypatch.setattr(ws.shutil, "disk_usage", tight)
    with pytest.raises(ws.SyncError, match=r"not enough disk space .* needs .* free \(1.2x\)"):
        run_sync(s3, tmp_path)
    assert not any(k.endswith(".safetensors") for k in s3.ranges)
    assert not (tmp_path / "yeto-complete" / f"org--tiny@{REV}.json").exists()


def test_a_manifest_object_is_the_authority_and_is_checked_against_the_listing(model, tmp_path):
    manifest = {"format": ws.MANIFEST_FORMAT, "files": {
        k: {"size": len(v), "sha256": hashlib.sha256(v).hexdigest()} for k, v in model.items()}}
    objects = {f"m/{k}": v for k, v in model.items()}
    fake = FakeS3({**objects, f"m/{ws.MANIFEST_NAME}": json.dumps(manifest).encode()}, meta_sha=False)
    try:
        ws.sync("s3://bkt/m", str(tmp_path / "a"), MODEL, REV, chunk_bytes=CHUNK, workers=2, environ=fake.env())
        assert not (snapshot(tmp_path / "a") / ws.MANIFEST_NAME).exists()
        manifest["files"]["config.json"]["size"] += 1
        fake.objects[f"m/{ws.MANIFEST_NAME}"] = json.dumps(manifest).encode()
        with pytest.raises(ws.SyncError, match="config.json: the manifest says"):
            ws.sync("s3://bkt/m", str(tmp_path / "b"), MODEL, REV, environ=fake.env())
        del fake.objects[f"m/{ws.MANIFEST_NAME}"]  # no manifest and no sha256 metadata: refused
        with pytest.raises(ws.SyncError, match="has no sha256"):
            ws.sync("s3://bkt/m", str(tmp_path / "c"), MODEL, REV, environ=fake.env())
    finally:
        fake.close()


def test_a_directory_source_needs_no_credentials(model, tmp_path):
    src = tmp_path / "nfs" / "tiny"
    for rel, data in model.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(data)
    out = tmp_path / "out"
    assert ws.main(["manifest", "--source", str(src), "--out", str(src / ws.MANIFEST_NAME)]) == 0
    summary = ws.sync(str(src), str(out), MODEL, REV, chunk_bytes=CHUNK, workers=3, environ={})
    assert summary["downloaded"] == len(model)
    assert all((snapshot(out) / k).read_bytes() == v for k, v in model.items())


@pytest.mark.skipif(sys.platform == "win32" or not hasattr(os, "fork"), reason="forked workers")
@pytest.mark.filterwarnings("ignore:.*multi-threaded.*:DeprecationWarning")
def test_forked_worker_processes(s3, model, tmp_path):
    summary = run_sync(s3, tmp_path, procs=True, workers=3)
    assert summary["downloaded"] == len(model)
    assert all((snapshot(tmp_path) / k).read_bytes() == v for k, v in model.items())


def test_revision_must_be_a_commit_and_credentials_are_required(s3, tmp_path):
    with pytest.raises(ws.SyncError, match="40-hex"):
        ws.sync("s3://bkt/models/tiny", str(tmp_path), MODEL, "main", environ=s3.env())
    with pytest.raises(ws.SyncError, match="no credentials"):
        ws.sync("s3://bkt/models/tiny", str(tmp_path), MODEL, REV, environ={})


def test_cli_sync_prints_one_summary_line(s3, tmp_path, monkeypatch, capsys):
    for k, v in s3.env().items():
        monkeypatch.setenv(k, v)
    rc = ws.main(["sync", "--source", "s3://bkt/models/tiny", "--root", str(tmp_path), "--model", MODEL,
                  "--revision", REV, "--stages", "0", "--pp", "3", "--workers", "2", "--node", "n0"])
    out = capsys.readouterr().out
    assert rc == 0
    tagged = [line for line in out.splitlines() if line.startswith(ws.SUMMARY_TAG + " ")]
    assert len(tagged) == 1
    summary = json.loads(tagged[0].split(" ", 1)[1])
    assert summary["node"] == "n0" and summary["stages"] == [0] and "files" not in summary
    assert "stages 0 of PP 3 (6 layers, stage 0: 0-1, stage 1: 2-3, stage 2: 4-5)" in out
