"""spot_rollout.materialize._prune: a staging directory a killed materialize left
behind (no done marker: the process died inside save_file) is dropped once it is
older than every kept output, so a restarted sampler does not leak a model-sized
directory per crash on its persistent work dir."""

import json

from yeto.rl.spot_rollout.materialize import DONE, _prune


def _done(root, name):
    d = root / name
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x" * 16)
    (d / DONE).write_text(json.dumps({"merged_version": 0}))
    return d


def _torn(root, name):
    d = root / f".{name}.tmp"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x" * 8)  # partial, no done marker
    return d


def test_prune_drops_a_torn_staging_dir_older_than_the_kept_outputs(tmp_path):
    torn = _torn(tmp_path, "v00000003-aaaaaaaaaaaa")
    _done(tmp_path, "v00000004-bbbbbbbbbbbb")
    current = _done(tmp_path, "v00000005-cccccccccccc")
    _prune(tmp_path, keep=2, current=current)
    assert not torn.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v00000004-bbbbbbbbbbbb", "v00000005-cccccccccccc"]


def test_prune_keeps_a_staging_dir_newer_than_the_oldest_kept_output(tmp_path):
    _done(tmp_path, "v00000004-bbbbbbbbbbbb")
    current = _done(tmp_path, "v00000005-cccccccccccc")
    newer = _torn(tmp_path, "v00000006-dddddddddddd")  # may be another process writing now
    _prune(tmp_path, keep=2, current=current)
    assert newer.exists()


def test_prune_leaves_foreign_entries(tmp_path):
    other = tmp_path / ".cache"
    other.mkdir()
    _done(tmp_path, "v00000004-bbbbbbbbbbbb")
    current = _done(tmp_path, "v00000005-cccccccccccc")
    _prune(tmp_path, keep=1, current=current)
    assert other.exists()
    assert not (tmp_path / "v00000004-bbbbbbbbbbbb").exists()
