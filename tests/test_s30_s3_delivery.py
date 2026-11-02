"""S30 P0-b: tools/probes/s29_spot/s3_delivery.py publish/pull against a local fake bucket."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from yeto.rl.spot_rollout import cut_manifest as cm

_SPEC = importlib.util.spec_from_file_location(
    "s3_delivery", Path(__file__).resolve().parents[1] / "tools/probes/s29_spot/s3_delivery.py")
sd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sd)


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
