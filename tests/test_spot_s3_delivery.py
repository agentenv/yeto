"""``yeto.rl.spot.s3_delivery`` publish/pull against a local fake bucket."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from yeto.rl.spot import s3_delivery as sd
from yeto.rl.spot_rollout import cut_manifest as cm


class FakeS3(sd.S3):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.uploads: list[str] = []

    def _path(self, key: str) -> Path:
        return self.root / key

    def copy_up(self, local: Path, key: str) -> None:
        self.uploads.append(key)
        shutil.copytree(local, self._path(key), dirs_exist_ok=True)

    def copy_down(self, key: str, local: Path) -> None:
        shutil.copytree(self._path(key), local, dirs_exist_ok=True)

    def put_bytes(self, key: str, data: bytes) -> None:
        self._path(key).parent.mkdir(parents=True, exist_ok=True)
        self._path(key).write_bytes(data)

    def get_bytes(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.is_file() else None


def _release(store: Path, name: str, seq: int, base: str | None) -> dict:
    d = store / cm.RELEASES / name
    d.mkdir(parents=True)
    (d / "w.bin").write_bytes(name.encode() * 1000)
    record = {"path": name, "seq": seq, "version": seq, "released_at": 1000.0 + seq,
              "files": {"w.bin": 0}, "delta": {"base_path": base} if base else None}
    (d / "RELEASE.json").write_text(json.dumps(record))
    (store / cm.RELEASES / cm.LATEST).write_text(json.dumps(record))
    return record


def test_publish_fills_chain_holes_and_pull_places_the_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("S3_ASSUME_PRESENT", "r1-v0")
    monkeypatch.setenv("S3_PREFIX", "test-run")
    monkeypatch.setenv("S3_REGION", "test-region")
    monkeypatch.setenv("S3_METRICS", str(tmp_path / "m.jsonl"))
    trainer, island, bucket = tmp_path / "trainer", tmp_path / "vm", tmp_path / "bucket"
    _release(trainer, "r1-v0", 1, None)
    (island / cm.RELEASES / "r1-v0").mkdir(parents=True)  # the island's base came from elsewhere
    s3 = FakeS3(bucket)
    _release(trainer, "r2-v2", 2, "r1-v0")
    _release(trainer, "r3-v3", 3, "r2-v2")  # the loop missed r2-v2
    row = sd.publish(s3, trainer, "r3-v3")
    assert s3.uploads == ["r2-v2", "r3-v3"] and row["release"] == "r3-v3"
    assert sd.publish(s3, trainer, "r3-v3") is None  # already published
    pointer = json.loads((bucket / cm.LATEST).read_text())
    assert pointer["release"] == "r3-v3" and pointer["record_order"] == [3, 3]
    sd.publish(s3, trainer, "r2-v2")
    assert json.loads((bucket / cm.LATEST).read_text())["release"] == "r3-v3"  # never rolled back

    out = sd.pull(s3, island, pointer["release"], digest=pointer["digest"])
    assert out["release"] == "r3-v3"
    assert (island / cm.RELEASES / "r2-v2" / "w.bin").read_bytes() == b"r2-v2" * 1000
    assert json.loads((island / cm.RELEASES / cm.LATEST).read_text())["path"] == "r3-v3"
    assert not (island / "staged-r3-v3").exists()
    sides = [json.loads(line)["side"] for line in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert sides == ["publish", "publish", "pull", "pull"]


def test_pull_refuses_a_tampered_release(tmp_path, monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("S3_ASSUME_PRESENT", "r1-v0")
    monkeypatch.setenv("S3_PREFIX", "test-run")
    monkeypatch.setenv("S3_REGION", "test-region")
    monkeypatch.setenv("S3_METRICS", str(tmp_path / "m.jsonl"))
    trainer, island, bucket = tmp_path / "trainer", tmp_path / "vm", tmp_path / "bucket"
    (island / cm.RELEASES / "r1-v0").mkdir(parents=True)
    s3 = FakeS3(bucket)
    _release(trainer, "r2-v2", 2, "r1-v0")
    sd.publish(s3, trainer, "r2-v2")
    (bucket / "r2-v2" / "w.bin").write_bytes(b"evil")
    with pytest.raises(cm.ManifestError):
        sd.pull(s3, island, "r2-v2")
    assert not (island / cm.RELEASES / "r2-v2").exists()
    with pytest.raises(cm.ManifestError, match="digest"):
        sd.pull(s3, island, "r2-v2", digest="0" * 64)


def test_rclone_environment_uses_the_proxy_only_when_set(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("S3_ASSUME_PRESENT", "r1-v0")
    monkeypatch.setenv("S3_PREFIX", "test-run")
    monkeypatch.setenv("S3_REGION", "test-region")
    monkeypatch.setenv("S3_PROXY", "http://proxy:1")
    assert sd.S3().environ["HTTPS_PROXY"] == "http://proxy:1"
    monkeypatch.setenv("S3_PROXY", "")
    monkeypatch.setenv("HTTPS_PROXY", "http://other:2")
    e = sd.S3().environ
    assert "HTTPS_PROXY" not in e and e["RCLONE_CONFIG_S3W_PROVIDER"] == "AWS"


@pytest.mark.parametrize("missing", ["S3_BUCKET", "S3_PREFIX", "S3_REGION"])
def test_bucket_prefix_and_region_are_required(monkeypatch, missing):
    for name, value in (("S3_BUCKET", "b"), ("S3_PREFIX", "p"), ("S3_REGION", "r")):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(missing)
    with pytest.raises(SystemExit, match=missing):
        sd.S3()


def test_no_aws_profile_unless_one_is_set(tmp_path, monkeypatch):
    for name, value in (("S3_BUCKET", "b"), ("S3_PREFIX", "p"), ("S3_REGION", "r")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".aws").mkdir()
    (tmp_path / ".aws" / "credentials").write_text("[some-profile]\naws_access_key_id = x\n")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    s3 = sd.S3()
    assert "AWS_PROFILE" not in s3.environ and s3.environ["RCLONE_CONFIG_S3W_REGION"] == "r"
    assert s3.url("x") == "s3w:b/p/x"
    monkeypatch.setenv("AWS_PROFILE", "chosen")
    assert sd.S3().environ["AWS_PROFILE"] == "chosen"


def test_assume_present_defaults_to_the_base_release(monkeypatch):
    monkeypatch.delenv("S3_ASSUME_PRESENT", raising=False)
    assert sd.assume_present() == {"r1-v0"}  # the base model is never uploaded by default
    monkeypatch.setenv("S3_ASSUME_PRESENT", "")
    assert sd.assume_present() == set()
    monkeypatch.setenv("S3_ASSUME_PRESENT", "r1-v0,r5-v0")
    assert sd.assume_present() == {"r1-v0", "r5-v0"}


def _crash_between_place_and_manifest(tmp_path, monkeypatch):
    for name, value in (("S3_BUCKET", "test-bucket"), ("S3_ASSUME_PRESENT", "r1-v0"), ("S3_PREFIX", "test-run"),
                        ("S3_REGION", "test-region"), ("S3_METRICS", str(tmp_path / "m.jsonl"))):
        monkeypatch.setenv(name, value)
    trainer, island, bucket = tmp_path / "trainer", tmp_path / "vm", tmp_path / "bucket"
    (island / cm.RELEASES / "r1-v0").mkdir(parents=True)
    s3 = FakeS3(bucket)
    _release(trainer, "r2-v2", 2, "r1-v0")
    sd.publish(s3, trainer, "r2-v2")
    real = sd.peer_fanout.save_manifest

    def crash(store, manifest):
        raise KeyboardInterrupt("killed after place, before save_manifest")

    monkeypatch.setattr(sd.peer_fanout, "save_manifest", crash)
    with pytest.raises(KeyboardInterrupt):
        sd.pull(s3, island, "r2-v2")
    monkeypatch.setattr(sd.peer_fanout, "save_manifest", real)
    assert (island / cm.RELEASES / "r2-v2").is_dir()
    assert not sd.peer_fanout.servable(island, "r2-v2")
    return s3, island


def test_next_pull_backfills_the_manifest_a_crash_after_place_lost(tmp_path, monkeypatch):
    s3, island = _crash_between_place_and_manifest(tmp_path, monkeypatch)
    assert sd.pull(s3, island, "r2-v2") is None  # already placed: nothing downloaded again
    assert sd.peer_fanout.servable(island, "r2-v2")
    saved = json.loads(sd.peer_fanout.manifest_path(island, "r2-v2").read_text())
    assert saved["release"] == "r2-v2" and saved == json.loads((tmp_path / "bucket/manifests/r2-v2.json").read_text())


def test_startup_backfills_every_placed_release_without_a_manifest(tmp_path, monkeypatch):
    s3, island = _crash_between_place_and_manifest(tmp_path, monkeypatch)
    assert sd.backfill_manifests(s3, island) == ["r2-v2"]  # r1-v0 has no manifest on S3: left alone
    assert sd.peer_fanout.servable(island, "r2-v2") and not sd.peer_fanout.servable(island, "r1-v0")
    assert sd.backfill_manifests(s3, island) == []


def test_backfill_refuses_a_placed_release_whose_files_do_not_match(tmp_path, monkeypatch):
    s3, island = _crash_between_place_and_manifest(tmp_path, monkeypatch)
    (island / cm.RELEASES / "r2-v2" / "w.bin").write_bytes(b"x" * 5000)
    assert sd.backfill_manifests(s3, island) == []
    assert not sd.peer_fanout.servable(island, "r2-v2")  # never advertised to peers


def _count_verify(monkeypatch) -> list[str]:
    calls: list[str] = []
    real = sd.cm.verify

    def counted(manifest, staged):
        calls.append(manifest["release"])
        return real(manifest, staged)

    monkeypatch.setattr(sd.cm, "verify", counted)
    return calls


def test_pull_does_not_rehash_a_placed_release_that_keeps_failing_verification(tmp_path, monkeypatch):
    s3, island = _crash_between_place_and_manifest(tmp_path, monkeypatch)
    weights = island / cm.RELEASES / "r2-v2" / "w.bin"
    good = weights.read_bytes()
    weights.write_bytes(b"x" * len(good))  # same size: only the sha256 catches it
    calls = _count_verify(monkeypatch)
    for _ in range(3):
        assert sd.pull(s3, island, "r2-v2") is None
    assert calls == ["r2-v2"]  # hashed once, then remembered while nothing changed
    assert not sd.peer_fanout.servable(island, "r2-v2")
    weights.write_bytes(good)  # the placed files change: checked again, and now they match
    os.utime(weights, ns=(weights.stat().st_atime_ns, weights.stat().st_mtime_ns + 1_000_000))
    assert sd.pull(s3, island, "r2-v2") is None
    assert calls == ["r2-v2", "r2-v2"]
    assert sd.peer_fanout.servable(island, "r2-v2")


def test_a_changed_s3_manifest_retries_a_remembered_backfill_failure(tmp_path, monkeypatch):
    s3, island = _crash_between_place_and_manifest(tmp_path, monkeypatch)
    weights = island / cm.RELEASES / "r2-v2" / "w.bin"
    weights.write_bytes(b"y" * len(weights.read_bytes()))
    calls = _count_verify(monkeypatch)
    assert sd.backfill_manifests(s3, island) == [] and sd.backfill_manifests(s3, island) == []
    assert calls == ["r2-v2"]
    key = tmp_path / "bucket" / "manifests" / "r2-v2.json"
    manifest = json.loads(key.read_text())
    manifest["files"]["w.bin"]["sha256"] = sd.cm.file_sha256(weights)
    key.write_text(json.dumps(manifest))
    assert sd.backfill_manifests(s3, island) == ["r2-v2"]
    assert calls == ["r2-v2", "r2-v2"]


def test_an_in_place_rewrite_that_keeps_the_mtime_is_verified_again(tmp_path, monkeypatch):
    """``rsync --inplace -t`` rewrites the bytes in place and restores the mtime:
    the inode and mtime stay, the ctime moves, so the release is hashed again."""
    s3, island = _crash_between_place_and_manifest(tmp_path, monkeypatch)
    weights = island / cm.RELEASES / "r2-v2" / "w.bin"
    good = weights.read_bytes()
    weights.write_bytes(b"x" * len(good))
    calls = _count_verify(monkeypatch)
    assert sd.pull(s3, island, "r2-v2") is None and sd.pull(s3, island, "r2-v2") is None
    assert calls == ["r2-v2"]
    before = weights.stat()
    with open(weights, "r+b") as handle:  # in place: same inode
        handle.write(good)
    os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = weights.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (before.st_ino, before.st_size, before.st_mtime_ns)
    assert sd.pull(s3, island, "r2-v2") is None
    assert calls == ["r2-v2", "r2-v2"]
    assert sd.peer_fanout.servable(island, "r2-v2")


@pytest.mark.parametrize("gone", ["manifest_saved", "release_removed"])
def test_a_remembered_backfill_failure_is_dropped_when_nothing_is_left_to_backfill(tmp_path, monkeypatch, gone):
    s3, island = _crash_between_place_and_manifest(tmp_path, monkeypatch)
    placed = island / cm.RELEASES / "r2-v2"
    (placed / "w.bin").write_bytes(b"z" * len((placed / "w.bin").read_bytes()))
    assert sd.backfill_manifests(s3, island) == []
    assert (str(island), "r2-v2") in sd._BACKFILL_REFUSED
    if gone == "manifest_saved":
        sd.peer_fanout.save_manifest(island, json.loads((tmp_path / "bucket/manifests/r2-v2.json").read_text()))
    else:
        shutil.rmtree(placed)
    assert sd.backfill_manifest(s3, island, "r2-v2") is False
    assert (str(island), "r2-v2") not in sd._BACKFILL_REFUSED


class _Stop(Exception):
    pass


class _DyingServer:
    """ensure_alive answers from a script: a restart, then given up (healthy False)."""

    def __init__(self, script):
        self.script, self.healthy, self.restarts, self.deaths_in_a_row = list(script), True, 0, 0
        self.error, self.port = "OSError('bad fd')", 18555

    def ensure_alive(self, log):
        step = self.script.pop(0) if self.script else "ok"
        if step == "restart":
            self.restarts += 1
            self.deaths_in_a_row += 1
            return False
        if step == "back":
            self.healthy = True
            return True
        if step == "dead":
            self.deaths_in_a_row += 1
            self.healthy = False
            return False
        return self.healthy


def _run_loop(tmp_path, monkeypatch, server, rounds):
    store = tmp_path / "store"
    (store / cm.RELEASES).mkdir(parents=True)
    monkeypatch.setenv("S3_METRICS", str(tmp_path / "metrics.jsonl"))
    monkeypatch.setattr(sd, "backfill_manifests", lambda s3, store: None)
    left = [rounds]

    def sleep(_s):
        left[0] -= 1
        if left[0] <= 0:
            raise _Stop

    monkeypatch.setattr(sd.time, "sleep", sleep)
    s3 = FakeS3.__new__(FakeS3)
    s3.get_json = lambda key: None
    with pytest.raises(_Stop):
        sd.pull_loop(s3, store, 1.0, server=server)
    path = tmp_path / "metrics.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_pull_loop_records_a_peer_server_that_gave_up(tmp_path, monkeypatch):
    """healthy=False must reach the metrics tape (patrol / alerts), once, not only a log line."""
    rows = _run_loop(tmp_path, monkeypatch, _DyingServer(["ok", "restart", "dead", "dead", "dead"]), 6)
    events = [r["event"] for r in rows]
    assert events == ["peer_server_restarted", "peer_server_unhealthy"]
    bad = rows[-1]
    assert bad["side"] == "serve" and bad["healthy"] is False and bad["port"] == 18555
    assert bad["restarts"] == 1 and "bad fd" in bad["error"] and bad["time_unix"] > 0


def test_pull_loop_survives_a_metrics_tape_it_cannot_write(tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "metrics", lambda store, row: (_ for _ in ()).throw(OSError(28, "No space left")))
    assert _run_loop(tmp_path, monkeypatch, _DyingServer(["dead"]), 3) == []


def test_pull_loop_records_a_peer_server_serving_again(tmp_path, monkeypatch):
    """After a restart (or a give-up) the server serves again: a healthy row clears the alert."""
    rows = _run_loop(tmp_path, monkeypatch, _DyingServer(["restart", "ok", "ok", "dead", "back", "ok"]), 7)
    assert [r["event"] for r in rows] == ["peer_server_restarted", "peer_server_healthy",
                                          "peer_server_unhealthy", "peer_server_healthy"]
    assert rows[-1]["healthy"] is True
