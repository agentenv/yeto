from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = _load("phase_map_contract_runner", "scripts/run_phase_map.py")
finalizer = _load(
    "phase_map_contract_finalizer", "scripts/finalize_p0_lifecycle.py"
)


def _command_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        python_executable="python3",
        command_repo_root=ROOT,
        run_dir=tmp_path / "run",
        model_path=tmp_path / "model",
        data=tmp_path / "data.parquet",
        token_budget=65536,
        seq_len=128,
        micro_batch_size=1,
        inner_lr=0.001,
        eval_rows=1024,
        confirmation_audit_rows=1024,
        train_rows=5000,
        seed=337,
        eval_split_seed=331,
        training_seed=337337,
        device="cuda",
        gpu_slots=1,
        learner_max_steps=512,
        syncer_checkpoint_every=1,
        arm_timeout_min=30,
        capture_every_step=True,
        require_distinct_learner_gpu_uuids=False,
    )


def test_pre_p3_surfaces_cannot_name_or_acquire_audit_artifacts(
    tmp_path: Path,
) -> None:
    command = runner.compare_command(
        _command_args(tmp_path), h=16, mu=0.5, eta=0.0875
    )
    assert all("audit" not in token.lower() for token in command), command

    manifest = {
        "lineage": {"descendant_kind": "p0a_single_gpu_bound"},
        "results": [
            {
                "cell_id": "p0a-cell",
                "attempt": 1,
                "status": "COMPLETED",
            }
        ],
    }
    audit_file = (
        tmp_path
        / "frozen-eval"
        / "seed-337"
        / "materialized"
        / "confirmation-audit.jsonl"
    )
    audit_file.parent.mkdir(parents=True)
    audit_file.write_text('{"must_not_be_acquired":true}\n')
    acquisition = runner.acquisition_paths(tmp_path, manifest)
    assert all("audit" not in path.as_posix().lower() for path in acquisition)

    # Production result rows are part of the sealed manifest.  A quarantine
    # implementation may validate absence internally, but it may not expose an
    # audit path/hash field through the pre-P3 result schema.
    result_source = inspect.getsource(runner.result_attempt)
    assert '"audit_access_log_uri"' not in result_source
    assert '"audit_access_log_sha256"' not in result_source


@pytest.mark.parametrize(
    ("token_budget", "expected_steps"),
    [(65_536, 128), (655_360, 1_280)],
)
def test_frozen_command_derives_exact_learner_step_cap(
    tmp_path: Path, token_budget: int, expected_steps: int
) -> None:
    args = _command_args(tmp_path)
    args.token_budget = token_budget
    args.learner_max_steps = 9_999

    command = runner.compare_command(args, h=16, mu=0.5, eta=0.0875)

    assert command.count("--learner-max-steps") == 1
    assert int(command[command.index("--learner-max-steps") + 1]) == expected_steps


def test_every_canary_result_constructor_emits_normalized_workload_hash() -> None:
    for constructor in (
        runner.result_attempt,
        runner.infra_failure_attempt,
        runner.scientific_failure_attempt,
    ):
        assert "normalized_workload_command_hash" in inspect.getsource(constructor)


def _load_replay_fixture_module():
    return _load(
        "phase_map_contract_replay_fixtures",
        "tests/test_validate_p0_replay.py",
    )


def test_finalizer_preserves_raw_deletion_evidence_bytes(tmp_path: Path) -> None:
    fixtures = _load_replay_fixture_module()
    _root, copied = fixtures._fixture(tmp_path)
    original = tmp_path / "deleted.json"
    assert copied.read_bytes() == original.read_bytes()
    assert finalizer.sha256_file(copied) == finalizer.sha256_file(original)


def _refresh_pending_seals(root: Path, deletion_path: Path) -> None:
    acquisition_manifest = root / "phase-map-acquisition-manifest.json"
    live_manifest = root / "phase-map-manifest.json"
    manifest = json.loads(acquisition_manifest.read_text())
    manifest["lineage"]["descendant_kind"] = "p0b_four_gpu_bound"
    finalizer.write_object(acquisition_manifest, manifest)
    live_manifest.write_bytes(acquisition_manifest.read_bytes())

    acquisition_seal = root / "acquisition-seal.json"
    seal = json.loads(acquisition_seal.read_text())
    seal["phase_map_manifest_sha256"] = finalizer.sha256_file(
        acquisition_manifest
    )
    seal["phase_map_manifest_canonical_sha256"] = finalizer.sha256_bytes(
        finalizer.canonical_json(manifest)
    )
    finalizer.write_object(acquisition_seal, seal)

    acquisition_checksum = root / "acquisition.sha256"
    relatives = [
        line.partition("  ")[2]
        for line in acquisition_checksum.read_text().splitlines()
    ]
    acquisition_checksum.write_text(
        "".join(
            f"{finalizer.sha256_file(root / relative)}  {relative}\n"
            for relative in relatives
        )
    )

    deletion = json.loads(deletion_path.read_text())
    by_role = {
        item["role"]: item
        for item in deletion["artifact_object_seal"]["objects"]
    }
    by_role["phase_map_manifest"]["sha256"] = finalizer.sha256_file(
        acquisition_manifest
    )
    by_role["phase_map_manifest"]["size"] = acquisition_manifest.stat().st_size
    by_role["acquisition_checksum"]["sha256"] = finalizer.sha256_file(
        acquisition_checksum
    )
    by_role["acquisition_checksum"]["size"] = acquisition_checksum.stat().st_size
    deletion_path.write_text(json.dumps(deletion))


def test_launched_p0b_without_full_gpu_barrier_evidence_cannot_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = _load_replay_fixture_module()
    monkeypatch.setattr(fixtures.finalizer, "finalize", lambda *_args, **_kwargs: {})
    root, _unused_copy = fixtures._fixture(tmp_path)
    deletion = tmp_path / "deleted.json"
    _refresh_pending_seals(root, deletion)

    with pytest.raises(finalizer.FinalizationError, match="lacks attempt evidence"):
        finalizer.finalize(root, deletion, expected_instance_id="123")
