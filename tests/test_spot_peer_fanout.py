"""Near-region release fan-out: a node pulls a release from a peer's file server when
one holds it (digest-guarded), falls back to the object store otherwise, and serves what
it placed."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from yeto.rl.spot import peer_fanout as pf
from yeto.rl.spot import s3_delivery as sd
from yeto.rl.spot_rollout import cut_manifest as cm

from test_spot_s3_delivery import FakeS3, _release


@pytest.fixture
def s3env(tmp_path, monkeypatch):
    for key, value in (("S3_BUCKET", "b"), ("S3_PREFIX", "run"), ("S3_REGION", "r"), ("S3_ASSUME_PRESENT", ""),
                       ("S3_METRICS", str(tmp_path / "m.jsonl"))):
        monkeypatch.setenv(key, value)
    return tmp_path


def _placed_store(tmp_path, name, bucket) -> Path:
    """A node that already pulled ``name`` from the bucket (so it can serve it)."""
    store = tmp_path / f"node-{name}"
    s3 = FakeS3(bucket)
    sd.pull(s3, store, name)
    return store


def test_parse_peers_and_sources():
    assert pf.parse_peers("a:1, b:2,,a:1") == ["a:1", "b:2"]
    assert pf.parse_peers(["10.0.0.5"]) == [f"10.0.0.5:{pf.DEFAULT_PORT}"]
    assert pf.parse_peers(None) == [] and pf.parse_peers("") == []
    with pytest.raises(ValueError, match="host:port"):
        pf.parse_peers("a:b:c")
    assert pf.peers_from({"peers": ["x:1"]}, environ={pf.PEERS_ENV: "y:2"}) == ["x:1"]
    assert pf.peers_from({}, environ={pf.PEERS_ENV: "y:2"}) == ["y:2"]
    assert pf.peers_from(None, environ={}) == []


def test_file_server_serves_only_placed_releases(s3env):
    tmp_path = s3env
    trainer, bucket = tmp_path / "trainer", tmp_path / "bucket"
    _release(trainer, "r1-v0", 1, None)
    s3 = FakeS3(bucket)
    sd.publish(s3, trainer, "r1-v0")
    store = _placed_store(tmp_path, "r1-v0", bucket)
    assert pf.servable(store, "r1-v0") and not pf.servable(store, "r9-v9") and not pf.servable(store, "../x")
    # a release whose directory exists but was never placed through the manifest is not served
    (store / cm.RELEASES / "r2-v2").mkdir()
    (store / cm.RELEASES / "r2-v2" / "w.bin").write_bytes(b"half")
    server = pf.ReleaseFileServer(store, host="127.0.0.1", port=0).start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        manifest = json.loads(pf.urllib_fetch(f"{base}/manifests/r1-v0.json", 5))
        assert manifest["release"] == "r1-v0" and cm.digest(manifest) == json.loads((bucket / cm.LATEST).read_text())["digest"]
        assert pf.urllib_fetch(f"{base}/r1-v0/w.bin", 5) == b"r1-v0" * 1000
        for bad in ("/manifests/r2-v2.json", "/r2-v2/w.bin", "/r1-v0/../LATEST.json", "/r1-v0/missing", "/manifests/x",
                    "/r1-v0/.hidden", "/"):
            with pytest.raises(pf.PeerError, match="HTTP 404"):
                pf.urllib_fetch(f"{base}{bad}", 5)
        head = urllib.request.Request(f"{base}/r1-v0/w.bin", method="HEAD")
        with urllib.request.urlopen(head, timeout=5) as response:  # noqa: S310
            assert response.headers["Content-Length"] == str(5000) and response.read() == b""
    finally:
        server.stop()


def test_pull_takes_a_peer_first_and_falls_back_to_s3(s3env):
    tmp_path = s3env
    trainer, bucket = tmp_path / "trainer", tmp_path / "bucket"
    _release(trainer, "r1-v0", 1, None)
    _release(trainer, "r2-v2", 2, "r1-v0")
    s3 = FakeS3(bucket)
    sd.publish(s3, trainer, "r2-v2")
    seed = _placed_store(tmp_path, "r2-v2", bucket)  # pulled both from S3
    assert pf.servable(seed, "r1-v0") and pf.servable(seed, "r2-v2")
    server = pf.ReleaseFileServer(seed, host="127.0.0.1", port=0).start()
    try:
        peers = [f"127.0.0.1:1", f"127.0.0.1:{server.port}"]  # the first peer is dead
        island = tmp_path / "island"
        pointer = json.loads((bucket / cm.LATEST).read_text())
        downloads = []
        class CountingS3(FakeS3):
            def copy_down(self, key, local):
                downloads.append(key)
                super().copy_down(key, local)
        row = sd.pull(CountingS3(bucket), island, pointer["release"], digest=pointer["digest"], peers=peers)
        assert row["source"] == f"peer:127.0.0.1:{server.port}" and downloads == []  # base and release from the peer
        assert (island / cm.RELEASES / "r2-v2" / "w.bin").read_bytes() == b"r2-v2" * 1000
        assert (island / cm.RELEASES / "r1-v0" / "w.bin").read_bytes() == b"r1-v0" * 1000
        assert json.loads((island / cm.RELEASES / cm.LATEST).read_text())["path"] == "r2-v2"
        assert pf.servable(island, "r2-v2") and not (island / "staged-peer-r2-v2").exists()
        rows = [json.loads(l) for l in (tmp_path / "m.jsonl").read_text().splitlines()]
        assert [r.get("source") for r in rows if r["side"] == "pull"] == ["s3", "s3", "peer:127.0.0.1:%d" % server.port] * 1 + ["peer:127.0.0.1:%d" % server.port]
        # a release no peer has comes from S3
        _release(trainer, "r3-v3", 3, "r2-v2")
        sd.publish(s3, trainer, "r3-v3")
        pointer = json.loads((bucket / cm.LATEST).read_text())
        row = sd.pull(CountingS3(bucket), island, "r3-v3", digest=pointer["digest"], peers=peers)
        assert row["source"] == "s3" and downloads == ["r3-v3"]
    finally:
        server.stop()


def test_peer_with_a_wrong_manifest_or_corrupt_file_never_places(s3env):
    tmp_path = s3env
    trainer, bucket = tmp_path / "trainer", tmp_path / "bucket"
    _release(trainer, "r1-v0", 1, None)
    s3 = FakeS3(bucket)
    sd.publish(s3, trainer, "r1-v0")
    good = json.loads((bucket / "manifests" / "r1-v0.json").read_text())
    island = tmp_path / "island"
    objects = {"/manifests/r1-v0.json": cm.encode(good), "/r1-v0/w.bin": b"r1-v0" * 1000,
               "/r1-v0/RELEASE.json": (trainer / cm.RELEASES / "r1-v0" / "RELEASE.json").read_bytes()}
    def fetch(url, timeout):
        path = url.split("peer", 1)[1]
        if path not in objects:
            raise pf.PeerError(f"{url}: HTTP 404")
        return objects[path]
    digest = cm.digest(good)
    # wrong digest (a peer serving another build of the same release name)
    tampered = {**good, "bytes": 1}
    objects["/manifests/r1-v0.json"] = cm.encode(tampered)
    with pytest.raises(pf.PeerError, match="digest"):
        pf.pull_from_peer("peer", island, "r1-v0", digest=digest, fetch=fetch)
    objects["/manifests/r1-v0.json"] = cm.encode(good)
    # right manifest, corrupt bytes of the right size
    objects["/r1-v0/w.bin"] = b"X" * 5000
    with pytest.raises(pf.PeerError, match="sha256"):
        pf.pull_from_peer("peer", island, "r1-v0", digest=digest, fetch=fetch)
    objects["/r1-v0/w.bin"] = b"short"
    with pytest.raises(pf.PeerError, match="bytes"):
        pf.pull_from_peer("peer", island, "r1-v0", digest=digest, fetch=fetch)
    assert not (island / cm.RELEASES / "r1-v0").exists() and not (island / "staged-peer-r1-v0").exists()
    assert pf.pull_from_peers(["peer"], island, "r1-v0", digest=digest, fetch=fetch) is None
    # then the bytes are right: placed and servable
    objects["/r1-v0/w.bin"] = b"r1-v0" * 1000
    row = pf.pull_from_peer("peer", island, "r1-v0", digest=digest, fetch=fetch)
    assert row["source"] == "peer:peer" and pf.servable(island, "r1-v0")
    # a release whose delta base is not local yet is refused before any file download
    base_first = {**good, "release": "r2-v2", "base": "r9-v9"}
    objects["/manifests/r2-v2.json"] = cm.encode(base_first)
    with pytest.raises(pf.PeerError, match="delta base"):
        pf.pull_from_peer("peer", island, "r2-v2", digest=None, fetch=fetch)


def test_release_file_server_death_is_detected_and_restarted(tmp_path, monkeypatch):
    """The serve thread used to die silently (an exception out of serve_forever):
    peers then fell back to the object store forever with no log line."""
    import time as _time

    store = tmp_path / "store"
    (store / cm.RELEASES).mkdir(parents=True)
    server = pf.ReleaseFileServer(store, host="127.0.0.1", port=0, poll_interval=0.01)
    boom = {"n": 1}
    real = server._server.service_actions

    def flaky():
        if boom["n"]:
            boom["n"] -= 1
            raise OSError("select failed")
        real()

    monkeypatch.setattr(server._server, "service_actions", flaky)
    server.start()
    for _ in range(200):
        if not server.alive():
            break
        _time.sleep(0.01)
    assert not server.alive()
    assert "select failed" in (server.error or "")
    assert server.ensure_alive() is False  # was dead: restarted
    assert server.alive() and server.restarts == 1
    assert server.ensure_alive() is True
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"http://127.0.0.1:{server.port}/manifests/nope.json", timeout=5)
    server.stop()
    assert server.ensure_alive() is True  # a stopped server is not restarted
    assert not server.alive()


def test_release_file_server_rebinds_then_reports_unhealthy(tmp_path, monkeypatch):
    """A serve thread that dies again on every restart (a broken socket) is rebound after
    ``rebind_after`` deaths in a row, and declared unhealthy after ``max_restarts``."""
    store = tmp_path / "store"
    (store / cm.RELEASES).mkdir(parents=True)
    server = pf.ReleaseFileServer(store, host="127.0.0.1", port=0, rebind_after=2, max_restarts=3)
    binds = []
    real_bind = server._bind

    def bind(host, port):
        binds.append((host, port))
        new = real_bind(host, port)
        new.serve_forever = lambda poll_interval=0.5: (_ for _ in ()).throw(OSError("bad fd"))
        return new

    server._server.serve_forever = lambda poll_interval=0.5: (_ for _ in ()).throw(OSError("bad fd"))
    monkeypatch.setattr(server, "_bind", bind)
    port, logs = server.port, []
    server.start()
    results = []
    for _ in range(5):
        server._thread.join(timeout=5)
        results.append(server.ensure_alive(logs.append))
    assert results == [False] * 5
    assert binds == [("127.0.0.1", port), ("127.0.0.1", port)]  # deaths 2 and 3 rebind on the same port
    assert server.healthy is False and server.restarts == 3
    assert any("unhealthy" in line for line in logs)
    server.stop()


def test_release_file_server_failed_rebind_is_unhealthy(tmp_path, monkeypatch):
    store = tmp_path / "store"
    (store / cm.RELEASES).mkdir(parents=True)
    server = pf.ReleaseFileServer(store, host="127.0.0.1", port=0, rebind_after=1)
    server._server.serve_forever = lambda poll_interval=0.5: (_ for _ in ()).throw(OSError("bad fd"))

    def no_bind(host, port):
        raise OSError(98, "Address already in use")

    monkeypatch.setattr(server, "_bind", no_bind)
    server.start()
    server._thread.join(timeout=5)
    logs = []
    assert server.ensure_alive(logs.append) is False
    assert server.healthy is False and "rebind" in logs[-1]
    assert server.ensure_alive(logs.append) is False  # stays unhealthy, no restart loop
    server.stop()
