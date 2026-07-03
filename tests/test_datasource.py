"""--data source classification and sky mount plumbing."""

from __future__ import annotations

from yeto.datasource import (
    HEAD_DATA_PATH,
    LEARNER_DATA_PATH,
    head_stage,
    kind,
    learner_data_arg,
    learner_file_mounts,
)


def test_hf_ids_pass_through():
    for d in ("org/dataset", "armand0e/claude-fable-5-claude-code"):
        assert kind(d) == "hf"
        assert learner_data_arg(d) == d
        assert learner_file_mounts(d) == {}
        assert head_stage(d) == (d, {})


def test_cloud_uris_mount_directly():
    for d in ("s3://bucket/traces", "gs://bucket/x", "r2://b/p"):
        assert kind(d) == "cloud"
        assert learner_data_arg(d) == LEARNER_DATA_PATH
        assert learner_file_mounts(d) == {LEARNER_DATA_PATH: d}
        # Cloud sources are reachable from the head directly: no staging.
        assert head_stage(d) == (d, {})


def test_local_paths_by_shape_and_existence(tmp_path):
    existing = tmp_path / "rows.jsonl"
    existing.write_text("{}\n")
    for d in ("/abs/path", "./rel", "../up", "~/home-ish", str(existing)):
        assert kind(d) == "local", d
    assert learner_data_arg(str(existing)) == LEARNER_DATA_PATH
    assert learner_file_mounts(str(existing)) == {LEARNER_DATA_PATH: str(existing)}


def test_head_stage_rewrites_local_paths(tmp_path):
    src = tmp_path / "data"
    src.mkdir()
    rewritten, mounts = head_stage(str(src))
    assert rewritten == HEAD_DATA_PATH
    assert mounts == {HEAD_DATA_PATH: str(src)}
    # The rewritten path is itself classified local, so the head's launcher
    # performs the second hop onto learners.
    assert kind(rewritten) == "local"


def test_cloud_detection_covers_all_sky_stores():
    # Detection delegates to sky's registry, so any store sky supports is a
    # valid --data source, not just the hand-rolled fallback tuple.
    for uri in ("s3://b/x", "gs://b/x", "r2://b/x", "oci://b/x"):
        assert kind(uri) == "cloud", uri
