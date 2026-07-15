"""Tests for immutable benchmark resume manifests."""

import json

import pytest

from yeto.benchmark_resume import (
    build_data_manifest,
    load_resume_config,
    validate_data_manifest,
    validate_record_keys,
    write_json_atomic,
)


def test_data_manifest_reuses_splits_and_detects_input_changes(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    train = work / "train.jsonl"
    evaluation = work / "eval.jsonl"
    source = work / "source"
    source.mkdir()
    train.write_text('{"row": 1}\n', encoding="utf-8")
    evaluation.write_text('{"row": 2}\n', encoding="utf-8")
    (source / "media.bin").write_bytes(b"media")

    manifest = build_data_manifest(
        work,
        train,
        evaluation,
        train_rows=1,
        eval_rows=1,
        source=source,
    )

    assert validate_data_manifest(work, manifest) == (train, evaluation, 1)
    (source / "media.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source benchmark data changed"):
        validate_data_manifest(work, manifest)
    (source / "media.bin").write_bytes(b"media")
    train.write_text('{"row": 3}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="train benchmark data changed"):
        validate_data_manifest(work, manifest)


def test_resume_config_requires_exact_identity_and_manifest(tmp_path):
    config_path = tmp_path / "config.json"
    identity = {"benchmark": "lm", "arguments": {"model": "org/model"}}
    manifest = {"format_version": 1, "train": {}, "eval": {}}
    write_json_atomic(
        config_path,
        {"resume_identity": identity, "data_manifest": manifest},
    )

    assert load_resume_config(config_path, identity) == manifest
    changed = {"benchmark": "lm", "arguments": {"model": "org/other"}}
    with pytest.raises(ValueError, match="arguments.model"):
        load_resume_config(config_path, changed)

    config_path.write_text(json.dumps({"format_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="predates immutable resume"):
        load_resume_config(config_path, identity)


def test_resume_records_reject_duplicates_and_unexpected_runs():
    expected = {
        ("base", "base", None),
        ("baseline", "baseline-m2", 17),
        ("diloco", "m2", 17),
    }
    records = [
        {"kind": "base", "arm": "base", "seed": None},
        {"kind": "baseline", "arm": "baseline-m2", "seed": 17},
        {"kind": "diloco", "arm": "m2", "seed": 17, "learners": 2},
    ]
    validate_record_keys(records, expected)
    with pytest.raises(ValueError, match="duplicate"):
        validate_record_keys(records + [records[-1]], expected)
    with pytest.raises(ValueError, match="unexpected"):
        validate_record_keys(
            records + [{"kind": "diloco", "arm": "m4", "seed": 17}],
            expected,
        )


def test_resume_records_require_base_and_matching_baseline():
    expected = {
        ("base", "base", None),
        ("baseline", "baseline-m2", 17),
        ("diloco", "m2", 17),
    }
    with pytest.raises(ValueError, match="base evaluation"):
        validate_record_keys(
            [{"kind": "baseline", "arm": "baseline-m2", "seed": 17}],
            expected,
        )
    with pytest.raises(ValueError, match="missing baseline-m2"):
        validate_record_keys(
            [
                {"kind": "base", "arm": "base", "seed": None},
                {
                    "kind": "diloco",
                    "arm": "m2",
                    "seed": 17,
                    "learners": 2,
                },
            ],
            expected,
        )
