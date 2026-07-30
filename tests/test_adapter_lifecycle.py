import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.adapter_lifecycle import (
    directory_sha256,
    inspect_parent_adapter,
    prepare_parent_source,
    training_artifact_metadata,
    training_recipe,
)


MODEL_RECORD = {
    "source": "huggingface",
    "requested_identifier": "org/model",
    "resolved_identifier": "org/model",
    "requested_revision": "main",
    "resolved_revision": "a" * 40,
}

DATASET_RECORD = {
    "source": "huggingface",
    "requested_identifier": "org/data",
    "resolved_identifier": "org/data",
    "requested_revision": "main",
    "resolved_revision": "b" * 40,
}


def args(**overrides):
    values = dict(
        resume_from=None,
        branch_from=None,
        adapter_sha256=None,
        tuning="lora",
        base_quantization="none",
        shard="ddp",
        lora_r=16,
        lora_alpha=32,
        lora_targets="auto",
        loss_function="cross_entropy",
        train_on="assistant",
        assistant_mask_mode="native",
        data_format="openai",
        seq_len=2048,
        seed=0,
        fragments=8,
        fragment_pattern="binpack",
        matrix_merge="rda",
        _provenance={
            "model": MODEL_RECORD,
            "dataset": DATASET_RECORD,
            "trust_remote_code": False,
        },
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def write_adapter(path: Path, *, manifest=True):
    path.mkdir()
    (path / "adapter_model.safetensors").write_bytes(b"safe adapter weights")
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": "org/model",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
        encoding="utf-8",
    )
    if manifest:
        source_args = args()
        (path / "yeto_provenance.json").write_text(
            json.dumps(
                {
                    "artifact_kind": "causal-lm-training-output",
                    "model": MODEL_RECORD,
                    "dataset": DATASET_RECORD,
                    "trust_remote_code": False,
                    "artifact": {"training_recipe": training_recipe(source_args)},
                }
            ),
            encoding="utf-8",
        )
    return path


def test_directory_hash_is_stable_and_content_sensitive(tmp_path):
    adapter = write_adapter(tmp_path / "adapter")
    first = directory_sha256(adapter)
    assert first == directory_sha256(adapter)
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    assert directory_sha256(adapter) != first


def test_directory_hash_rejects_symbolic_links(tmp_path):
    adapter = write_adapter(tmp_path / "adapter")
    (adapter / "linked-config.json").symlink_to(adapter / "adapter_config.json")
    with pytest.raises(ValueError, match="symbolic link"):
        directory_sha256(adapter)


def test_branch_accepts_legacy_safe_adapter_and_records_lineage(tmp_path):
    adapter = write_adapter(tmp_path / "adapter", manifest=False)
    run_args = args(branch_from=str(adapter))

    lineage = inspect_parent_adapter(run_args)

    assert lineage["mode"] == "branch"
    assert lineage["legacy_source"] is True
    assert lineage["sha256"] == directory_sha256(adapter)
    assert run_args.branch_from == str(adapter.resolve())
    run_args._adapter_lineage = lineage
    assert training_artifact_metadata(run_args)["parent_adapter"] == lineage


def test_resume_requires_exact_recorded_recipe(tmp_path):
    adapter = write_adapter(tmp_path / "adapter")
    run_args = args(resume_from=str(adapter))
    lineage = inspect_parent_adapter(run_args)
    assert lineage["mode"] == "resume"
    assert lineage["legacy_source"] is False

    run_args.seq_len = 4096
    with pytest.raises(ValueError, match="seq_len: 2048 -> 4096"):
        inspect_parent_adapter(run_args)


def test_resume_rejects_dataset_drift_but_branch_allows_it(tmp_path):
    adapter = write_adapter(tmp_path / "adapter")
    changed = dict(DATASET_RECORD, resolved_revision="c" * 40)
    run_args = args(
        resume_from=str(adapter),
        _provenance={
            "model": MODEL_RECORD,
            "dataset": changed,
            "trust_remote_code": False,
        },
    )
    with pytest.raises(ValueError, match="resume dataset differs"):
        inspect_parent_adapter(run_args)

    run_args.resume_from = None
    run_args.branch_from = str(adapter)
    assert inspect_parent_adapter(run_args)["mode"] == "branch"


def test_resume_rejects_legacy_artifact_but_branch_is_actionable(tmp_path):
    adapter = write_adapter(tmp_path / "adapter", manifest=False)
    with pytest.raises(ValueError, match="use --branch-from"):
        inspect_parent_adapter(args(resume_from=str(adapter)))


def test_parent_adapter_rejects_rank_and_base_drift(tmp_path):
    adapter = write_adapter(tmp_path / "adapter")
    with pytest.raises(ValueError, match="lora_r=16"):
        inspect_parent_adapter(args(branch_from=str(adapter), lora_r=8))
    different_model = dict(MODEL_RECORD, resolved_identifier="org/other")
    with pytest.raises(ValueError, match="base model resolved_identifier"):
        inspect_parent_adapter(
            args(
                branch_from=str(adapter),
                _provenance={"model": different_model},
            )
        )


def test_parent_adapter_requires_causal_lora_config(tmp_path):
    adapter = write_adapter(tmp_path / "adapter", manifest=False)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["peft_type"] = "IA3"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="peft_type=LORA"):
        inspect_parent_adapter(args(branch_from=str(adapter)))


def test_local_launch_source_is_attested_and_cloud_requires_digest(tmp_path):
    adapter = write_adapter(tmp_path / "adapter")
    local_args = args(branch_from=str(adapter))
    prepare_parent_source(local_args)
    assert local_args.adapter_sha256 == directory_sha256(adapter)

    with pytest.raises(ValueError, match="require --adapter-sha256"):
        prepare_parent_source(args(branch_from="s3://bucket/adapter"))
    with pytest.raises(ValueError, match="64 hexadecimal"):
        prepare_parent_source(
            args(
                branch_from="s3://bucket/adapter",
                adapter_sha256="not-a-digest",
            )
        )
