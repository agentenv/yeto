from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts import run_cplg_online_acquisition as runner


RUN_ID = "exp2-cplg-active-e1-m1-r1"
SOURCE_COMMIT = "a" * 40


def _json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def _write_checksummed(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(value)
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _copy_frozen_config(tmp_path: Path) -> tuple[Path, dict]:
    source = (
        runner.REPO_ROOT
        / "experiments"
        / "optimizer"
        / "cplg-sgd-active-e1-r1-config.json"
    )
    path = tmp_path / "frozen-config.json"
    path.write_bytes(source.read_bytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return path, json.loads(path.read_bytes())


def _compare_argv(config: dict, root: Path) -> list[str]:
    values = runner._expected_compare_values(
        config,
        run_id=RUN_ID,
        run_config_sha256=runner.CONFIG_SHA256,
        source_commit=SOURCE_COMMIT,
    )
    values["--work-dir"] = str(root / "work")
    values["--report-dir"] = str(root / "report")
    order = [
        "--model",
        "--data",
        "--seq-len",
        "--micro-batch-size",
        "--inner-lr",
        "--tuning",
        "--shard",
        "--lora-r",
        "--lora-alpha",
        "--eval-rows",
        "--max-rows",
        "--shuffle-rows-seed",
        "--training-seed",
        "--device",
        "--gpu-slots",
        "--delta-correction",
        "--matrix-merge",
        "--outer-momentum",
        "--outer-lr",
        "--token-budget",
        "--syncer-total-steps",
        "--learner-max-steps",
        "--fixed-window-microsteps",
        "--delta-norm-ref",
        "--syncer-checkpoint-every",
        "--strict-quorum",
        "--barrier-sync",
        "--deterministic-commit-order",
        "--settings",
        "--skip-baseline",
        "--cplg-online-run-id",
        "--cplg-online-run-config-sha256",
        "--cplg-online-source-commit",
        "--arm-timeout-min",
        "--work-dir",
        "--report-dir",
    ]
    argv: list[str] = []
    for name in order:
        argv.append(name)
        if name not in runner.COMPARE_FLAGS:
            argv.append(values[name])
    return argv


def _replace_option(argv: list[str], name: str, value: str) -> list[str]:
    changed = list(argv)
    changed[changed.index(name) + 1] = value
    return changed


def _write_input_provenance(
    root: Path, monkeypatch: pytest.MonkeyPatch, config: dict
) -> Path:
    model = root.parent / "inputs" / "Qwen3.5-9B"
    data = root.parent / "inputs" / "Capybara-local" / "train.parquet"
    model.mkdir(parents=True)
    data.parent.mkdir(parents=True)
    (model / "config.json").write_text("model-config\n")
    (model / "weights.safetensors").write_bytes(b"weights")
    data.write_bytes(b"parquet-input")
    monkeypatch.setattr(runner, "MODEL_PATH", str(model))
    monkeypatch.setattr(runner, "DATA_PATH", str(data))

    provenance = root / "input-provenance"
    provenance.mkdir(parents=True)
    model_manifest = provenance / "yeto-model-files.sha256"
    model_manifest.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n"
            for path in sorted(model.iterdir())
        )
    )
    data_manifest = provenance / "yeto-data.sha256"
    data_manifest.write_text(
        f"{hashlib.sha256(data.read_bytes()).hexdigest()}  {data}\n"
    )
    (provenance / "yeto-runtime.txt").write_text(
        "torch=2.test\ntransformers=4.test\ncuda=12.test\n"
        "rustc 1.test\ncargo 1.test\nNVIDIA A100\n"
    )
    (provenance / "yeto-optimizer-image.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_run": "image-builder",
                "repo_url": "https://github.com/agentenv/yeto.git",
                "repo_commit": "b" * 40,
                "model_files_included": True,
                "huggingface_cache_included": False,
                "credentials_included": False,
                "run_artifacts_included": False,
                "model_checksum_manifest": "/etc/yeto-model-files.sha256",
                "model_symlink_manifest": "/etc/yeto-model-symlinks.txt",
                "data_checksum_manifest": "/etc/yeto-data.sha256",
                "runtime_manifest": "/etc/yeto-runtime.txt",
            }
        )
        + "\n"
    )
    (provenance / "verification.log").write_text("model: OK\ndata: OK\n")
    manifest = root / "input-provenance.sha256"
    manifest.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n"
            for path in sorted(provenance.iterdir())
        )
    )

    resource = config["resource_envelope"]["gpu_acquisition"]
    (root / "spec.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "repo_commit": SOURCE_COMMIT,
                "scientific_bindings": {
                    "scientific_config_sha256": runner.CONFIG_SHA256
                },
                "cloud": {
                    "image": resource["image"],
                    "expected_source_image_id": resource["expected_source_image_id"],
                    "machine_type": resource["machine_type"],
                    "accelerator_count": resource["accelerator_count"],
                },
            }
        )
        + "\n"
    )
    (root / "command.sh").write_text("python3 acquisition-only\n")
    (root / "git-status.txt").write_text(f"clean {SOURCE_COMMIT}\n")
    (root / "git-diff.patch").write_text("# clean\n")
    return manifest


def _event_rows() -> bytes:
    return b"".join(
        (
            json.dumps(
                {"commit_seq": commit, "fragment": (commit - 1) % 4},
                sort_keys=True,
            )
            + "\n"
        ).encode()
        for commit in range(1, 33)
    )


def _write_complete_outputs(root: Path) -> None:
    work = root / "work"
    report = root / "report"
    work.mkdir(parents=True)
    report.mkdir(parents=True)
    (work / "train.jsonl").write_text(json.dumps({"messages": []}) + "\n")
    (work / "eval.jsonl").write_text(
        "".join(json.dumps({"messages": [], "row": row}) + "\n" for row in range(8))
    )
    (report / "results.jsonl").write_text(
        "".join(
            json.dumps({"arm": arm, "eval_loss": loss}) + "\n"
            for arm, loss in (
                (runner.BASE_ARM, 2.0),
                (runner.STOCK_ARM, 1.5),
                (runner.CANDIDATE_ARM, 1.51),
            )
        )
    )
    (report / "report.md").write_text("# GPU evaluation\n")
    shared_layout = "1" * 64
    shared_initial = "2" * 64
    candidate_head = "3" * 64
    for arm, optimizer in (
        (runner.STOCK_ARM, "nesterov"),
        (runner.CANDIDATE_ARM, "cplg-sgd"),
    ):
        arm_root = work / arm
        arm_root.mkdir()
        tape = arm_root / "tape.jsonl"
        checkpoint = arm_root / "state.ckpt"
        tape.write_bytes(_event_rows())
        checkpoint.write_bytes(f"checkpoint:{arm}".encode())
        (arm_root / "syncer.log").write_text("syncer closed\n")
        (arm_root / "learner-0.log").write_text(
            "inner loop done at local_step=34 global_step=34\n"
        )
        (arm_root / "export.log").write_text("adapter exported\n")
        export = arm_root / "export"
        export.mkdir()
        (export / "adapter_config.json").write_text("{}\n")
        (export / "adapter_model.safetensors").write_bytes(b"adapter")
        _write_checksummed(
            arm_root / "cplg_online_initial_state.json",
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "run_config_sha256": runner.CONFIG_SHA256,
                "source_commit": SOURCE_COMMIT,
                "arm": arm,
                "outer_optimizer": optimizer,
                "layout_sha256": shared_layout,
                "initial_state_sha256": shared_initial,
                "fragments": 4,
                "expected_commits": 32,
            },
        )
        _write_checksummed(
            arm_root / "cplg_online_completion.json",
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "arm": arm,
                "terminal_local_steps": 34,
                "raw_training_tokens": 4352,
                "final_global_step": 34,
                "commits_observed": 32,
                "commits_per_fragment": [8, 8, 8, 8],
                "interval_start_ns": 100,
                "interval_end_ns": 200,
                "interval_ns": 100,
                "event_tape_sha256": hashlib.sha256(tape.read_bytes()).hexdigest(),
                "final_checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "ledger_head": candidate_head if arm == runner.CANDIDATE_ARM else None,
                "ledger_rows": 32 if arm == runner.CANDIDATE_ARM else None,
                "writer_dropped": 0,
                "writer_abandoned": 0,
                "writer_pending": 0,
                "writer_errors": 0,
            },
        )
        _write_checksummed(
            arm_root / "learner-0" / "learner_completion.json",
            {
                "schema": "yeto.learner_completion.v1",
                "learner_id": 0,
                "local_step": 34,
                "raw_tokens": 4352,
                "global_step": 34,
                "reconnect_count": 0,
                "terminal_status": "syncer_shutdown",
            },
        )
    candidate = work / runner.CANDIDATE_ARM
    (candidate / "cplg_action_ledger.jsonl").write_text(
        "".join(json.dumps({"row_index": row}) + "\n" for row in range(32))
    )
    _write_checksummed(
        candidate / "cplg_action_ledger_manifest.json",
        {"ledger_head": candidate_head, "ledger_rows": 32},
    )


def _run_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path, config = _copy_frozen_config(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    input_manifest = _write_input_provenance(root, monkeypatch, config)
    argv = _compare_argv(config, root)
    return config_path, config, root, input_manifest, argv


def test_exact_compare_argv_construction_and_contract(tmp_path, monkeypatch):
    _config_path, config, root, _input, argv = _run_fixture(tmp_path, monkeypatch)

    work, report = runner._validate_frozen_config(
        config,
        argv,
        run_id=RUN_ID,
        run_config_sha256=runner.CONFIG_SHA256,
        source_commit=SOURCE_COMMIT,
    )

    assert work == root / "work"
    assert report == root / "report"
    assert argv == _compare_argv(config, root)
    values, flags = runner._parse_compare_options(argv)
    assert values["--settings"] == "cplg_m1_stock,cplg_m1_candidate"
    assert values["--syncer-total-steps"] == "32"
    assert values["--learner-max-steps"] == "96"
    assert values["--token-budget"] == "4352"
    assert values["--eval-rows"] == "8"
    assert flags == runner.COMPARE_FLAGS
    assert argv == runner._build_compare_argv(
        config,
        run_id=RUN_ID,
        run_config_sha256=runner.CONFIG_SHA256,
        source_commit=SOURCE_COMMIT,
        work_dir=root / "work",
        report_dir=root / "report",
    )


@pytest.mark.parametrize(
    ("option", "wrong"),
    [
        ("--model", "/different/model"),
        ("--data", "/different/data.parquet"),
        ("--shuffle-rows-seed", "272"),
        ("--training-seed", "271272"),
        ("--lora-r", "4"),
        ("--lora-alpha", "8"),
        ("--seq-len", "256"),
        ("--micro-batch-size", "2"),
        ("--inner-lr", "0.002"),
        ("--token-budget", "32768"),
        ("--syncer-total-steps", "256"),
        ("--learner-max-steps", "256"),
        ("--fixed-window-microsteps", "8"),
        ("--eval-rows", "16"),
        ("--outer-lr", "0.29"),
        ("--outer-momentum", "0.1"),
    ],
)
def test_config_digest_cannot_mask_mismatched_compare_argv(
    tmp_path, monkeypatch, option, wrong
):
    _config_path, config, _root, _input, argv = _run_fixture(tmp_path, monkeypatch)
    with pytest.raises(runner.RunnerError, match=option):
        runner._validate_frozen_config(
            config,
            _replace_option(argv, option, wrong),
            run_id=RUN_ID,
            run_config_sha256=runner.CONFIG_SHA256,
            source_commit=SOURCE_COMMIT,
        )


def test_compare_argv_rejects_duplicate_unknown_and_noncanonical_options(
    tmp_path, monkeypatch
):
    _config_path, config, _root, _input, argv = _run_fixture(tmp_path, monkeypatch)
    cases = [
        [*argv, "--model", runner.MODEL_PATH],
        [*argv, "--dry-run", "true"],
        [*argv, "--model=wrong"],
        [*argv, "stray-positional"],
    ]
    for changed in cases:
        with pytest.raises(runner.RunnerError):
            runner._validate_frozen_config(
                config,
                changed,
                run_id=RUN_ID,
                run_config_sha256=runner.CONFIG_SHA256,
                source_commit=SOURCE_COMMIT,
            )


def test_frozen_config_requires_basename_bound_sidecar(tmp_path):
    config_path, _config = _copy_frozen_config(tmp_path)
    sidecar = config_path.with_name(config_path.name + ".sha256")
    sidecar.write_text(f"{runner.CONFIG_SHA256}  other.json\n")

    with pytest.raises(runner.RunnerError, match="basename-bound"):
        runner._checksummed_json(config_path, "frozen run configuration")


def test_runner_rejects_a_different_correctly_checksummed_config(tmp_path, monkeypatch):
    config_path, config, root, input_manifest, argv = _run_fixture(
        tmp_path, monkeypatch
    )
    config["workload"]["training_seed"] += 1
    _write_checksummed(config_path, config)

    with pytest.raises(runner.RunnerError, match="run-config SHA-256"):
        runner.run_acquisition(
            run_id=RUN_ID,
            run_config=config_path,
            input_manifest=input_manifest,
            expected_source_commit=SOURCE_COMMIT,
            manifest_output=root / "report" / "acquisition_manifest.json",
            terminal_output=root / "report" / "acquisition_terminal.json",
            compare_argv=argv,
            python="python3",
        )


def test_clean_source_binding_checks_commit_and_all_untracked_files(monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[1:3] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, SOURCE_COMMIT + "\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner._require_clean_checkout(SOURCE_COMMIT)
    assert ["git", "status", "--porcelain=v1", "--untracked-files=all"] in commands

    monkeypatch.setattr(runner, "_git_head", lambda: "b" * 40)
    with pytest.raises(runner.RunnerError, match="differs"):
        runner._require_clean_checkout(SOURCE_COMMIT)


def test_input_provenance_verifies_model_data_runtime_image_and_cloud_spec(
    tmp_path, monkeypatch
):
    _config_path, config, root, input_manifest, _argv = _run_fixture(
        tmp_path, monkeypatch
    )
    result = runner._verify_input_provenance(
        input_manifest,
        root=root,
        config=config,
        source_commit=SOURCE_COMMIT,
    )
    assert result["source_image_id"] == "7290368630472593484"
    assert result["image"].endswith("yeto-optimizer-a100-20260714")
    assert runner.SHA256_RE.fullmatch(result["model_manifest_sha256"])
    assert runner.SHA256_RE.fullmatch(result["data_manifest_sha256"])


def test_freshness_rejects_existing_work_before_launch(tmp_path, monkeypatch):
    config_path, _config, root, input_manifest, argv = _run_fixture(
        tmp_path, monkeypatch
    )
    (root / "work").mkdir()
    monkeypatch.setattr(runner, "_require_clean_checkout", lambda _commit: None)

    with pytest.raises(runner.RunnerError, match="must be fresh"):
        runner.run_acquisition(
            run_id=RUN_ID,
            run_config=config_path,
            input_manifest=input_manifest,
            expected_source_commit=SOURCE_COMMIT,
            manifest_output=root / "report" / "acquisition_manifest.json",
            terminal_output=root / "report" / "acquisition_terminal.json",
            compare_argv=argv,
            python="python3",
        )


def test_required_output_presence_and_receipts(tmp_path):
    root = tmp_path / "run"
    _write_complete_outputs(root)
    evidence = runner._verify_required_outputs(
        root / "work",
        root / "report",
        run_id=RUN_ID,
        config_sha256=runner.CONFIG_SHA256,
        source_commit=SOURCE_COMMIT,
    )
    assert evidence == {
        "layout_sha256": "1" * 64,
        "initial_state_sha256": "2" * 64,
        "terminal_local_steps": 34,
        "raw_training_tokens": 4352,
        "commits_per_arm": 32,
        "evaluation_rows": 8,
    }

    (root / "work" / runner.STOCK_ARM / "export.log").unlink()
    with pytest.raises(runner.RunnerError, match="export log"):
        runner._verify_required_outputs(
            root / "work",
            root / "report",
            run_id=RUN_ID,
            config_sha256=runner.CONFIG_SHA256,
            source_commit=SOURCE_COMMIT,
        )


def test_required_receipt_rejects_bad_sidecar(tmp_path):
    root = tmp_path / "run"
    _write_complete_outputs(root)
    receipt = root / "work" / runner.STOCK_ARM / "cplg_online_completion.json"
    receipt.with_name(receipt.name + ".sha256").write_text(
        f"{'0' * 64}  {receipt.name}\n"
    )
    with pytest.raises(runner.RunnerError, match="basename-bound"):
        runner._verify_required_outputs(
            root / "work",
            root / "report",
            run_id=RUN_ID,
            config_sha256=runner.CONFIG_SHA256,
            source_commit=SOURCE_COMMIT,
        )


def test_manifest_hashing_is_sorted_exact_and_excludes_lifecycle_files(tmp_path):
    root = tmp_path / "tree"
    (root / "b").mkdir(parents=True)
    (root / "a.txt").write_bytes(b"aaa")
    (root / "b" / "z.bin").write_bytes(b"zz")
    excluded = root / "b" / "excluded.json"
    excluded.write_bytes(b"not sealed")

    files, total = runner._manifest_files(root, excluded={excluded})

    assert [entry["path"] for entry in files] == ["a.txt", "b/z.bin"]
    assert total == 5
    assert files[0]["sha256"] == hashlib.sha256(b"aaa").hexdigest()
    assert files[1]["sha256"] == hashlib.sha256(b"zz").hexdigest()


def test_manifest_hashing_rejects_symlink_file_and_directory(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    target = root / "target"
    target.write_text("target")
    link = root / "file-link"
    link.symlink_to(target)
    with pytest.raises(runner.RunnerError, match="symlink"):
        runner._manifest_files(root, excluded=set())
    link.unlink()

    real = root / "real-dir"
    real.mkdir()
    (real / "file").write_text("payload")
    (root / "dir-link").symlink_to(real, target_is_directory=True)
    with pytest.raises(runner.RunnerError, match="symlink"):
        runner._manifest_files(root, excluded=set())


def test_manifest_hashing_rejects_special_file(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(runner.RunnerError, match="special file"):
        runner._manifest_files(root, excluded=set())


def test_atomic_checksummed_publish_is_fresh_and_basename_bound(tmp_path):
    output = tmp_path / "receipt.json"
    digest = runner._publish_json(output, {"status": "ACQUIRED"})
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.with_name(output.name + ".sha256").read_text() == (
        f"{digest}  {output.name}\n"
    )
    assert not list(tmp_path.glob("*.tmp"))
    original = output.read_bytes()
    with pytest.raises(runner.RunnerError, match="not fresh"):
        runner._publish_json(output, {"status": "REPLACED"})
    assert output.read_bytes() == original


def test_terminal_success_path_is_acquisition_only(tmp_path, monkeypatch):
    config_path, _config, root, input_manifest, argv = _run_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(runner, "_require_clean_checkout", lambda _commit: None)
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        _write_complete_outputs(root)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.run_acquisition(
        run_id=RUN_ID,
        run_config=config_path,
        input_manifest=input_manifest,
        expected_source_commit=SOURCE_COMMIT,
        manifest_output=root / "report" / "acquisition_manifest.json",
        terminal_output=root / "report" / "acquisition_terminal.json",
        compare_argv=argv,
        python="/pinned/python3",
    )

    assert commands == [["/pinned/python3", str(runner.COMPARE), *argv]]
    assert result["status"] == "GPU_ACQUISITION_COMPLETE"
    assert result["scientific_verdict"] is None
    assert result["gpu_analysis_performed"] is False
    manifest = json.loads((root / "report" / "acquisition_manifest.json").read_bytes())
    terminal = json.loads((root / "report" / "acquisition_terminal.json").read_bytes())
    assert manifest["status"] == "ACQUIRED"
    assert manifest["scientific_verdict"] is None
    assert (
        terminal["acquisition_manifest_sha256"]
        == hashlib.sha256(
            (root / "report" / "acquisition_manifest.json").read_bytes()
        ).hexdigest()
    )
    paths = {entry["path"] for entry in manifest["files"]}
    assert "report/acquisition_manifest.json" not in paths
    assert "report/acquisition_terminal.json" not in paths
    assert "work/cplg_m1_stock/state.ckpt" in paths
    assert "work/cplg_m1_candidate/cplg_action_ledger.jsonl" in paths


def test_terminal_failure_publishes_incomplete_run_receipt_without_manifest(
    tmp_path, monkeypatch
):
    config_path, _config, root, input_manifest, argv = _run_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(runner, "_require_clean_checkout", lambda _commit: None)

    def fail(command, **_kwargs):
        raise subprocess.CalledProcessError(17, command)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_acquisition(
            run_id=RUN_ID,
            run_config=config_path,
            input_manifest=input_manifest,
            expected_source_commit=SOURCE_COMMIT,
            manifest_output=root / "report" / "acquisition_manifest.json",
            terminal_output=root / "report" / "acquisition_terminal.json",
            compare_argv=argv,
            python="python3",
        )

    assert not (root / "report" / "acquisition_manifest.json").exists()
    terminal = json.loads((root / "report" / "acquisition_terminal.json").read_bytes())
    assert terminal["status"] == "INFRA_FAILURE"
    assert terminal["acquisition_manifest"] is None
    assert terminal["acquisition_manifest_sha256"] is None
    assert terminal["scientific_verdict"] is None
    assert terminal["next_action"].startswith("incomplete_run_")


def test_runner_never_invokes_validator_or_any_analysis_program(tmp_path, monkeypatch):
    config_path, _config, root, input_manifest, argv = _run_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(runner, "_require_clean_checkout", lambda _commit: None)
    invoked: list[list[str]] = []

    def fake_run(command, **_kwargs):
        invoked.append(command)
        _write_complete_outputs(root)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert (
        runner.main(
            [
                "--run-id",
                RUN_ID,
                "--run-config",
                str(config_path),
                "--input-manifest",
                str(input_manifest),
                "--expected-source-commit",
                SOURCE_COMMIT,
                "--manifest-output",
                str(root / "report" / "acquisition_manifest.json"),
                "--terminal-output",
                str(root / "report" / "acquisition_terminal.json"),
                "--python",
                "python3",
                "--",
                *argv,
            ]
        )
        == 0
    )
    assert len(invoked) == 1
    assert invoked[0][1] == str(runner.COMPARE)
    flattened = " ".join(argument for command in invoked for argument in command)
    assert "validate_cplg_online_pair.py" not in flattened
    assert "replay" not in flattened
    assert "bootstrap" not in flattened
    assert "analysis" not in flattened
