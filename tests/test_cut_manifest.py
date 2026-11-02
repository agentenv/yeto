"""A release handed to a spot island through a near store (cut_manifest)."""
import json
import shutil

import pytest

from yeto.rl.spot_rollout import cut_manifest as cm


def _store(tmp_path):
    root = tmp_path / "store" / "policy-releases"
    (root / "r1-v0").mkdir(parents=True)
    (root / "r1-v0" / "model.safetensors").write_bytes(b"base")
    rel = root / "r2-v2"
    rel.mkdir()
    (rel / "model.safetensors.ydelta").write_bytes(b"delta" * 1000)
    (rel / "config.json").write_text("{}")
    record = {"path": "r2-v2", "version": 2, "seq": 2, "delta": {"base_path": "r1-v0"},
              "files": {"model.safetensors.ydelta": 5000, "config.json": 2}}
    (root / "LATEST.json").write_text(json.dumps(record))
    return tmp_path / "store", record


def _far(tmp_path):
    far = tmp_path / "far" / "policy-releases" / "r1-v0"
    far.mkdir(parents=True)
    (far / "model.safetensors").write_bytes(b"base")
    return tmp_path / "far"


def test_make_place_round_trip(tmp_path):
    store, record = _store(tmp_path)
    manifest = cm.make(store, "r2-v2")
    assert manifest["version"] == 2 and manifest["base"] == "r1-v0" and manifest["bytes"] == 5002
    sha = cm.digest(json.loads(cm.encode(manifest)))
    staged = tmp_path / "staged"
    shutil.copytree(store / "policy-releases" / "r2-v2", staged)
    far = _far(tmp_path)
    target = cm.place(manifest, staged, far, expected_digest=sha)
    assert (target / "model.safetensors.ydelta").read_bytes() == b"delta" * 1000
    assert json.loads((far / "policy-releases" / "LATEST.json").read_text()) == record
    # idempotent: placing again keeps the release (LATEST already names it)
    assert cm.place(manifest, tmp_path / "gone", far, expected_digest=sha) == target


def test_a_corrupt_file_or_a_wrong_digest_is_refused(tmp_path):
    store, _ = _store(tmp_path)
    manifest = cm.make(store, "r2-v2")
    staged = tmp_path / "staged"
    shutil.copytree(store / "policy-releases" / "r2-v2", staged)
    (staged / "model.safetensors.ydelta").write_bytes(b"x" * 5000)  # same size, other bytes
    far = _far(tmp_path)
    with pytest.raises(cm.ManifestError, match="sha256"):
        cm.place(manifest, staged, far, expected_digest=cm.digest(manifest))
    with pytest.raises(cm.ManifestError, match="digest"):
        cm.place(manifest, staged, far, expected_digest="0" * 64)
    assert not (far / "policy-releases" / "LATEST.json").exists()


def test_missing_base_or_wrong_latest_is_refused(tmp_path):
    store, _ = _store(tmp_path)
    manifest = cm.make(store, "r2-v2")
    with pytest.raises(cm.ManifestError, match="delta base"):
        cm.place(manifest, tmp_path, tmp_path / "empty", expected_digest=None)
    with pytest.raises(cm.ManifestError, match="names"):
        cm.make(store, "r1-v0")


def test_cli(tmp_path, capsys):
    store, _ = _store(tmp_path)
    out = tmp_path / "m.json"
    assert cm.main(["make", "--store", str(store), "--release", "r2-v2", "--out", str(out)]) == 0
    sha = capsys.readouterr().out.split()[0]
    staged = tmp_path / "staged"
    shutil.copytree(store / "policy-releases" / "r2-v2", staged)
    far = _far(tmp_path)
    assert cm.main(["place", "--manifest", str(out), "--digest", sha, "--staged", str(staged),
                    "--store", str(far)]) == 0
    assert cm.main(["place", "--manifest", str(out), "--digest", "1" * 64, "--staged", str(staged),
                    "--store", str(far)]) == 2


def test_placing_an_older_manifest_again_keeps_the_newer_latest(tmp_path):
    store, record = _store(tmp_path)
    old = cm.make(store, "r2-v2")
    staged = tmp_path / "staged"
    shutil.copytree(store / "policy-releases" / "r2-v2", staged)
    far = _far(tmp_path)
    cm.place(old, staged, far, expected_digest=None)
    root = store / "policy-releases"
    (root / "r3-v3").mkdir()
    (root / "r3-v3" / "config.json").write_text("{}")
    newer = {"path": "r3-v3", "version": 3, "seq": 3, "files": {"config.json": 2}}
    (root / "LATEST.json").write_text(json.dumps(newer))
    staged3 = tmp_path / "staged3"
    shutil.copytree(root / "r3-v3", staged3)
    cm.place(cm.make(store, "r3-v3"), staged3, far, expected_digest=None)
    latest = far / "policy-releases" / "LATEST.json"
    assert json.loads(latest.read_text()) == newer
    # the idempotent branch with the older manifest: the release is kept, LATEST is not rolled back
    assert cm.place(old, tmp_path / "gone", far, expected_digest=None) == far / "policy-releases" / "r2-v2"
    assert json.loads(latest.read_text()) == newer
    assert record["seq"] == 2


@pytest.mark.parametrize("name", ["../evil", "a/b", "..", ".", "", "x\\y", "r2..v2"])
def test_a_release_name_with_a_path_is_refused(tmp_path, name):
    store, _ = _store(tmp_path)
    manifest = dict(cm.make(store, "r2-v2"), release=name)
    with pytest.raises(cm.ManifestError, match="plain release directory name"):
        cm.place(manifest, tmp_path / "staged", _far(tmp_path), expected_digest=None)
    assert not (tmp_path / "far" / "policy-releases" / "LATEST.json").exists()
