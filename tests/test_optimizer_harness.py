from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yeto.optimizer_harness as harness

from yeto.optimizer_harness import (
    HarnessError,
    abandon,
    abandonment_script,
    adopt,
    build_parser,
    canary_launch_command,
    image_create_command,
    launch,
    launch_command,
    load_state,
    load_spec,
    matched_sgd_command,
    matched_start_script,
    sanitize_script,
    save_state,
    start_script,
    verify_candidate_image_description,
    verify_description,
    verify_disk_description,
)


COMMIT = "f08563a9bf944062a51e1b85dc987cbc071ca7bd"


def _raw() -> dict:
    return {
        "schema_version": 1,
        "run_id": "exp-test",
        "repo_url": "https://github.com/agentenv/yeto.git",
        "repo_commit": COMMIT,
        "cloud": {
            "provider": "gcp",
            "project": "test-project",
            "zone": "us-central1-b",
            "instance_name": "exp-test-vm",
            "machine_type": "a2-highgpu-1g",
            "provisioning_model": "SPOT",
            "termination_action": "DELETE",
            "boot_disk_size_gb": 250,
            "boot_disk_type": "pd-ssd",
            "image": "optimizer-image-v1",
            "scopes": ["storage-rw"],
            "labels": {
                "managed-by": "yeto-optimizer-harness",
                "run-id": "exp-test",
            },
        },
        "execution": {
            "remote_repo_dir": "/home/test/yeto",
            "remote_run_dir": "/home/test/runs/exp-test",
            "required_paths": ["/home/test/models/model"],
            "completion_paths": ["/home/test/runs/exp-test/report/results.jsonl"],
            "env": {"HF_HUB_OFFLINE": "1"},
            "command": [
                "python",
                "scripts/compare_diloco.py",
                "--settings",
                "candidate,control",
                "--outer-lr",
                "0.28",
                "--baseline-loss",
                "0.0",
                "--fixed-window-microsteps",
                "16",
                "--syncer-probe-capture",
                "--syncer-probe-capture-every",
                "1",
                "--work-dir",
                "/home/test/runs/exp-test/work",
                "--report-dir",
                "/home/test/runs/exp-test/report",
            ],
        },
        "artifacts": {
            "uri": "gs://optimizer-tests/runs/exp-test",
            "sync_interval_seconds": 60,
        },
        "checks": {
            "require_injected_baseline": True,
            "injected_baseline_report_only": True,
            "expected_arms": ["candidate", "control"],
            "expected_flags": {
                "--outer-lr": "0.28",
                "--fixed-window-microsteps": "16",
            },
        },
        "analysis": [
            {"name": "scale-fit", "command": ["python", "scripts/analyze.py"]}
        ],
        "image": {
            "name": "optimizer-image-v2",
            "family": "optimizer-image",
            "storage_location": "us-central1",
            "sanitize_paths": [
                "/home/test/runs/exp-test",
                "/home/shou/.cache/huggingface/token",
            ],
        },
    }


def _spec(tmp_path: Path, mutate=None):
    raw = _raw()
    if mutate:
        mutate(raw)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(raw))
    return load_spec(path)


def _description(spec, *, instance_id="123", nonce="0123456789abcdef"):
    description = {
        "name": spec.instance_name,
        "id": instance_id,
        "selfLink": (
            f"https://www.googleapis.com/compute/v1/projects/{spec.project}/"
            f"zones/{spec.zone}/instances/{spec.instance_name}"
        ),
        "labels": {
            **spec.cloud["labels"],
            "ownership-nonce": nonce,
        },
        "scheduling": {
            "provisioningModel": "SPOT",
            "instanceTerminationAction": "DELETE",
        },
        "disks": [
            {
                "autoDelete": True,
                "boot": True,
                "source": (
                    f"https://www.googleapis.com/compute/v1/projects/{spec.project}/"
                    f"zones/{spec.zone}/disks/{spec.instance_name}"
                ),
            }
        ],
    }
    if "max_run_duration_seconds" in spec.cloud:
        description["scheduling"]["maxRunDuration"] = {
            "seconds": str(spec.cloud["max_run_duration_seconds"]),
            "nanos": 0,
        }
    return description


def test_strict_quorum_budget_requires_explicit_async_headroom(tmp_path):
    def mutate(raw):
        command = raw["execution"]["command"]
        baseline_index = command.index("--baseline-loss")
        command[baseline_index:baseline_index] = [
            "--strict-quorum",
            "--syncer-total-steps",
            "340",
            "--learner-max-steps",
            "1487",
        ]
        raw["checks"]["strict_quorum_step_budget"] = {
            "fragments": 4,
            "min_headroom_steps": 128,
        }

    with pytest.raises(HarnessError, match="required>=1488"):
        _spec(tmp_path, mutate)

    spec = _spec(
        tmp_path,
        lambda raw: (
            mutate(raw),
            raw["execution"]["command"].__setitem__(
                raw["execution"]["command"].index("1487"), "1600"
            ),
        ),
    )
    assert spec.checks["strict_quorum_step_budget"] == {
        "fragments": 4,
        "min_headroom_steps": 128,
        "ideal_learner_steps": 1360,
        "required_learner_steps": 1488,
    }


def test_strict_quorum_budget_enforces_empirical_shutdown_headroom(tmp_path):
    def mutate(raw, learner_cap="1599"):
        command = raw["execution"]["command"]
        baseline_index = command.index("--baseline-loss")
        command[baseline_index:baseline_index] = [
            "--strict-quorum",
            "--syncer-total-steps",
            "340",
            "--learner-max-steps",
            learner_cap,
        ]
        raw["checks"]["strict_quorum_step_budget"] = {
            "fragments": 4,
            "min_headroom_steps": 128,
            "empirical_shutdown_upper_bound_steps": 1471,
            "min_post_empirical_headroom_steps": 129,
        }

    with pytest.raises(HarnessError, match="empirical_required>=1600.*required>=1600"):
        _spec(tmp_path, mutate)

    spec = _spec(tmp_path, lambda raw: mutate(raw, "1600"))
    assert spec.checks["strict_quorum_step_budget"] == {
        "fragments": 4,
        "min_headroom_steps": 128,
        "ideal_learner_steps": 1360,
        "required_learner_steps": 1600,
        "empirical_shutdown_upper_bound_steps": 1471,
        "min_post_empirical_headroom_steps": 129,
        "empirical_required_learner_steps": 1600,
    }


@pytest.mark.parametrize(
    "lone_key,lone_value",
    [
        ("empirical_shutdown_upper_bound_steps", 1471),
        ("min_post_empirical_headroom_steps", 129),
    ],
)
def test_strict_quorum_budget_requires_complete_empirical_pair(
    tmp_path, lone_key, lone_value
):
    def mutate(raw):
        command = raw["execution"]["command"]
        baseline_index = command.index("--baseline-loss")
        command[baseline_index:baseline_index] = [
            "--strict-quorum",
            "--syncer-total-steps",
            "340",
            "--learner-max-steps",
            "1600",
        ]
        raw["checks"]["strict_quorum_step_budget"] = {
            "fragments": 4,
            "min_headroom_steps": 128,
            lone_key: lone_value,
        }

    with pytest.raises(HarnessError, match="must be set together"):
        _spec(tmp_path, mutate)


def _boot_disk_description(
    spec,
    *,
    disk_id="456",
    source_image_id="789",
    source_image=None,
):
    instance = _description(spec)
    return {
        "id": disk_id,
        "selfLink": instance["disks"][0]["source"],
        "users": [instance["selfLink"]],
        "sourceImageId": source_image_id,
        "sourceImage": source_image
        or (
            "https://www.googleapis.com/compute/v1/projects/test-project/"
            "global/images/optimizer-image-v1"
        ),
    }


class SourceImageRunner:
    dry_run = False

    def __init__(self, spec, disk=None):
        self.spec = spec
        self.disk = disk or _boot_disk_description(spec)
        self.nonce = "0000000000000000"
        self.commands = []

    def run(self, command, *, check=True, capture=True):
        self.commands.append(command)
        if command[:3] == ["gcloud", "storage", "ls"]:
            return subprocess.CompletedProcess(command, 1, "", "not found")
        if command[:4] == ["gcloud", "compute", "instances", "list"]:
            return subprocess.CompletedProcess(command, 0, "[]", "")
        if command[:4] == ["gcloud", "compute", "instances", "create"]:
            labels = next(item for item in command if item.startswith("--labels="))
            self.nonce = re.search(r"ownership-nonce=([a-f0-9]{16})", labels).group(1)
            return subprocess.CompletedProcess(command, 0, "[]", "")
        if command[:4] == ["gcloud", "compute", "instances", "add-labels"]:
            labels = next(item for item in command if item.startswith("--labels="))
            self.nonce = re.search(r"ownership-nonce=([a-f0-9]{16})", labels).group(1)
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:4] == ["gcloud", "compute", "instances", "describe"]:
            description = _description(self.spec, nonce=self.nonce)
            return subprocess.CompletedProcess(command, 0, json.dumps(description), "")
        if command[:4] == ["gcloud", "compute", "disks", "describe"]:
            return subprocess.CompletedProcess(command, 0, json.dumps(self.disk), "")
        raise AssertionError(f"unexpected command: {command}")


def test_valid_spec_and_safe_launch_command(tmp_path):
    spec = _spec(tmp_path)
    command = launch_command(spec, "0123456789abcdef")
    assert "--provisioning-model=SPOT" in command
    assert "--instance-termination-action=DELETE" in command
    assert "--no-restart-on-failure" in command
    assert "--metadata=block-project-ssh-keys=true" in command
    assert "--boot-disk-type=pd-ssd" in command
    labels = next(item for item in command if item.startswith("--labels="))
    assert "managed-by=yeto-optimizer-harness" in labels
    assert "ownership-nonce=0123456789abcdef" in labels
    assert "--image=optimizer-image-v1" in command


def test_launch_command_and_live_identity_pin_max_run_duration(tmp_path):
    def cap_runtime(raw):
        raw["cloud"]["max_run_duration_seconds"] = 7200

    spec = _spec(tmp_path, cap_runtime)
    command = launch_command(spec, "0123456789abcdef")
    assert "--max-run-duration=7200s" in command
    verify_description(spec, _description(spec))

    wrong = _description(spec)
    wrong["scheduling"]["maxRunDuration"]["seconds"] = "7199"
    with pytest.raises(HarnessError, match="max run duration"):
        verify_description(spec, wrong)


@pytest.mark.parametrize("value", [59, 86401, True, "7200"])
def test_rejects_invalid_max_run_duration(tmp_path, value):
    def mutate(raw):
        raw["cloud"]["max_run_duration_seconds"] = value

    with pytest.raises(HarnessError, match="max_run_duration_seconds"):
        _spec(tmp_path, mutate)


def test_rejects_unknown_boot_disk_type(tmp_path):
    def mutate(raw):
        raw["cloud"]["boot_disk_type"] = "local-ssd"

    with pytest.raises(HarnessError, match="boot_disk_type"):
        _spec(tmp_path, mutate)


@pytest.mark.parametrize("value", [123, "", "12x34"])
def test_expected_source_image_id_must_be_digit_string(tmp_path, value):
    def mutate(raw):
        raw["cloud"]["expected_source_image_id"] = value

    with pytest.raises(HarnessError, match="expected_source_image_id"):
        _spec(tmp_path, mutate)


def test_expected_source_image_id_requires_image_backed_launch(tmp_path):
    def mutate(raw):
        raw["cloud"]["machine_image"] = raw["cloud"].pop("image")
        raw["cloud"]["expected_source_image_id"] = "789"

    with pytest.raises(HarnessError, match="only with cloud.image"):
        _spec(tmp_path, mutate)


def test_rejects_unpinned_commit(tmp_path):
    with pytest.raises(HarnessError, match="full 40-character"):
        _spec(tmp_path, lambda raw: raw.__setitem__("repo_commit", "main"))


def test_rejects_duplicate_json_keys(tmp_path):
    raw = _raw()
    path = tmp_path / "spec.json"
    payload = json.dumps(raw).replace(
        '"schema_version": 1,',
        '"schema_version": 1, "schema_version": 1,',
        1,
    )
    path.write_text(payload)
    with pytest.raises(HarnessError, match="duplicate JSON object key"):
        load_spec(path)


def test_rejects_missing_injected_baseline_guard(tmp_path):
    def mutate(raw):
        command = raw["execution"]["command"]
        index = command.index("--baseline-loss")
        del command[index : index + 2]

    with pytest.raises(HarnessError, match="requires --baseline-loss"):
        _spec(tmp_path, mutate)


def test_requires_one_honest_skip_baseline_and_rejects_duplicate_causal_flags(tmp_path):
    def honest(raw):
        command = raw["execution"]["command"]
        index = command.index("--baseline-loss")
        del command[index : index + 2]
        command.append("--skip-baseline")
        raw["checks"].pop("require_injected_baseline")
        raw["checks"].pop("injected_baseline_report_only")
        raw["checks"]["require_skip_baseline"] = True

    spec = _spec(tmp_path, honest)
    assert "--skip-baseline" in spec.command

    def missing(raw):
        honest(raw)
        raw["execution"]["command"].remove("--skip-baseline")

    with pytest.raises(HarnessError, match="exactly one --skip-baseline"):
        _spec(tmp_path, missing)

    def duplicate_lr(raw):
        raw["execution"]["command"].extend(["--outer-lr", "0.28"])

    with pytest.raises(HarnessError, match="exactly one --outer-lr"):
        _spec(tmp_path, duplicate_lr)


def test_rejects_expected_lr_or_arm_mismatch(tmp_path):
    def mutate_lr(raw):
        raw["checks"]["expected_flags"]["--outer-lr"] = "0.7"

    with pytest.raises(HarnessError, match="--outer-lr"):
        _spec(tmp_path, mutate_lr)

    def mutate_arms(raw):
        raw["checks"]["expected_arms"] = ["candidate", "wrong"]

    with pytest.raises(HarnessError, match="expected_arms"):
        _spec(tmp_path, mutate_arms)


def test_rejects_broad_or_unlisted_sanitize_path(tmp_path):
    def broad(raw):
        raw["image"]["sanitize_paths"].append("/home")

    with pytest.raises(HarnessError, match="too broad"):
        _spec(tmp_path, broad)

    def unlisted(raw):
        raw["image"]["sanitize_paths"].append("/home/test/important")

    with pytest.raises(HarnessError, match="predeclared"):
        _spec(tmp_path, unlisted)


def test_exact_id_nonce_and_labels_are_fail_closed(tmp_path):
    spec = _spec(tmp_path)
    description = _description(spec)
    assert (
        verify_description(
            spec,
            description,
            expected_id="123",
            expected_nonce="0123456789abcdef",
        )
        == "123"
    )
    with pytest.raises(HarnessError, match="exact id"):
        verify_description(spec, description, expected_id="999")
    with pytest.raises(HarnessError, match="nonce"):
        verify_description(spec, description, expected_nonce="fedcba9876543210")
    description["labels"]["managed-by"] = "someone-else"
    with pytest.raises(HarnessError, match="label"):
        verify_description(spec, description)


def test_start_script_pins_commit_and_sets_backup(tmp_path):
    spec = _spec(tmp_path)
    script = start_script(spec)
    assert COMMIT in script
    assert "git -C" in script and "rev-parse HEAD" in script
    assert "gcloud storage rsync --recursive" in script
    assert "runner.pid" in script and "backup.pid" in script
    assert "--baseline-loss 0.0" in script


def test_checkout_source_mode_fetches_exact_detached_commit(tmp_path):
    def checkout(raw):
        raw["execution"]["source_mode"] = "checkout"

    spec = _spec(tmp_path, checkout)
    script = start_script(spec)
    assert "git clone --filter=blob:none --no-checkout" in script
    assert f"fetch --depth=1 origin {COMMIT}" in script
    assert f"checkout --detach {COMMIT}" in script


def test_image_script_refuses_live_runner_and_removes_credentials(tmp_path):
    spec = _spec(tmp_path)
    script = sanitize_script(spec)
    assert "REFUSING: experiment runner is still live" in script
    assert "/home/shou/.cache/huggingface/token" in script
    assert "cloud-init clean --logs" in script
    command = image_create_command(
        spec,
        "exact-boot-disk",
        source_instance_id="123",
        source_disk_id="456",
        image_nonce="0123456789abcdef",
    )
    assert "--source-disk=exact-boot-disk" in command
    assert not any(item.startswith("--family=") for item in command)
    assert any("image-nonce=0123456789abcdef" in item for item in command)
    assert "--force" not in command


def test_candidate_image_is_unpromoted_and_canary_uses_exact_image(tmp_path):
    spec = _spec(tmp_path)
    image = {
        "id": "789",
        "name": "optimizer-image-v2",
        "selfLink": (
            "https://www.googleapis.com/compute/v1/projects/test-project/"
            "global/images/optimizer-image-v2"
        ),
        "status": "READY",
        "sourceDisk": (
            "https://www.googleapis.com/compute/v1/projects/test-project/"
            "zones/us-central1-b/disks/exp-test-vm"
        ),
        "sourceDiskId": "456",
        "labels": {
            "managed-by": "yeto-optimizer-harness",
            "image-nonce": "0123456789abcdef",
            "source-disk-id": "456",
            "source-instance-id": "123",
            "source-run": "exp-test",
        },
    }
    assert (
        verify_candidate_image_description(
            spec,
            image,
            source_instance_id="123",
            source_disk_id="456",
            source_disk_self_link=image["sourceDisk"],
            image_nonce="0123456789abcdef",
        )
        == "789"
    )
    image["family"] = "optimizer-image"
    with pytest.raises(HarnessError, match="prematurely"):
        verify_candidate_image_description(
            spec,
            image,
            source_instance_id="123",
            source_disk_id="456",
            source_disk_self_link=image["sourceDisk"],
            image_nonce="0123456789abcdef",
        )

    command = canary_launch_command(spec, "0123456789abcdef", "789")
    assert "--image=projects/test-project/global/images/optimizer-image-v2" in command
    assert "--boot-disk-auto-delete" in command
    assert "--metadata=block-project-ssh-keys=true" in command
    assert any("source-image-id=789" in item for item in command)


def test_launch_requires_confirmation_and_enforces_aggregate_gpu_cap(tmp_path):
    spec = _spec(tmp_path)

    class Runner:
        dry_run = False

        def __init__(self, inventory):
            self.inventory = inventory
            self.commands = []

        def run(self, command, *, check=True):
            import subprocess

            self.commands.append(command)
            if command[:3] == ["gcloud", "storage", "ls"]:
                return subprocess.CompletedProcess(command, 1, "", "not found")
            if command[:4] == ["gcloud", "compute", "instances", "list"]:
                return subprocess.CompletedProcess(
                    command, 0, json.dumps(self.inventory), ""
                )
            raise AssertionError(f"unexpected command: {command}")

    with pytest.raises(HarnessError, match="launch requires --yes"):
        launch(spec, Runner([]), tmp_path / "state", confirmed=False)

    four_active = [
        {
            "status": "RUNNING",
            "guestAccelerators": [{"acceleratorCount": 8}],
        }
    ]
    with pytest.raises(HarnessError, match="campaign cap"):
        launch(spec, Runner(four_active), tmp_path / "state", confirmed=True)

    args = build_parser().parse_args(["launch", "spec.json"])
    assert args.yes is False
    args = build_parser().parse_args(["launch", "spec.json", "--yes"])
    assert args.yes is True


def test_adopt_only_spec_cannot_launch(tmp_path):
    def adopt_only(raw):
        raw["cloud"]["adopt_only"] = True

    spec = _spec(tmp_path, adopt_only)

    class NoCalls:
        dry_run = False

        def run(self, command, *, check=True):
            raise AssertionError(f"unexpected command: {command}")

    with pytest.raises(HarnessError, match="adopt_only"):
        launch(spec, NoCalls(), tmp_path / "state", confirmed=True)


def test_declared_tree_checksum_manifest_is_verified_before_success(tmp_path):
    manifest = "/home/test/runs/exp-test/work/m4/capture-artifacts.sha256"

    def add_manifest(raw):
        raw["execution"]["completion_paths"].append(manifest)
        raw["execution"]["checksum_manifests"] = [manifest]

    spec = _spec(tmp_path, add_manifest)
    script = start_script(spec)
    assert "manifest=" + manifest in script
    assert 'cd "$(dirname "$manifest")"' in script
    assert 'sha256sum -c "$(basename "$manifest")"' in script
    assert script.index(
        "declared checksum manifest verification failed"
    ) < script.index('printf "%s\\n" "$code" > "$run/runner.exit.tmp"')

    completion_check = harness.remote_completion_check_script(spec)
    assert 'cd "$(dirname "$manifest")"' in completion_check
    assert 'sha256sum -c "$(basename "$manifest")"' in completion_check


def test_rendered_runner_preserves_apostrophes_and_records_checksum_failure(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        harness,
        "SAFE_IMAGE_PATH_PREFIXES",
        (*harness.SAFE_IMAGE_PATH_PREFIXES, "/private/"),
    )
    run = tmp_path / "runs" / "exp-test"
    run.mkdir(parents=True)
    manifest = run / "missing-artifacts.sha256"

    def add_missing_manifest(raw):
        raw["execution"]["remote_run_dir"] = str(run)
        raw["execution"]["completion_paths"] = [str(manifest)]
        raw["execution"]["checksum_manifests"] = [str(manifest)]
        raw["image"]["sanitize_paths"][0] = str(run)

    spec = _spec(tmp_path, add_missing_manifest)
    body = harness._runner_body(spec)
    rendered = f"{harness._render_bash_c(body)} _ {shlex.quote(str(run))} true"
    result = subprocess.run(
        ["bash", "-c", rendered],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 14
    assert "declared checksum manifest is missing" in result.stderr
    assert (run / "runner.exit").read_text() == "14\n"
    assert f'nohup {harness._render_bash_c(body)} _ "$run"' in start_script(spec)


def test_declared_tree_checksum_manifest_must_be_scoped_and_completed(tmp_path):
    def outside(raw):
        path = "/home/test/other/capture-artifacts.sha256"
        raw["execution"]["completion_paths"].append(path)
        raw["execution"]["checksum_manifests"] = [path]

    with pytest.raises(HarnessError, match="below execution.remote_run_dir"):
        _spec(tmp_path, outside)

    def not_completed(raw):
        raw["execution"]["checksum_manifests"] = [
            "/home/test/runs/exp-test/work/m4/capture-artifacts.sha256"
        ]

    with pytest.raises(HarnessError, match="must also be a completion path"):
        _spec(tmp_path, not_completed)


def test_input_manifests_are_verified_and_copied_before_runner(tmp_path):
    model_manifest = "/etc/yeto-model-files.sha256"
    runtime_manifest = "/etc/yeto-runtime.txt"

    def add_inputs(raw):
        raw["execution"]["required_paths"].extend([model_manifest, runtime_manifest])
        raw["execution"]["input_checksum_manifests"] = [model_manifest]
        raw["execution"]["input_provenance_paths"] = [
            model_manifest,
            runtime_manifest,
        ]

    spec = _spec(tmp_path, add_inputs)
    script = start_script(spec)
    assert f"sha256sum -c {model_manifest}" in script
    assert f'cp -- {model_manifest} "$run/input-provenance/"' in script
    assert f'cp -- {runtime_manifest} "$run/input-provenance/"' in script
    assert 'xargs -0 sha256sum > "$run/input-provenance.sha256"' in script
    assert script.index(f"sha256sum -c {model_manifest}") < script.index(
        "nohup bash -c"
    )


def test_input_manifests_must_be_required_and_preserved(tmp_path):
    def not_preserved(raw):
        raw["execution"]["required_paths"].append("/etc/yeto-model-files.sha256")
        raw["execution"]["input_checksum_manifests"] = ["/etc/yeto-model-files.sha256"]

    with pytest.raises(HarnessError, match="must also be an execution.input"):
        _spec(tmp_path, not_preserved)

    def not_required(raw):
        raw["execution"]["input_provenance_paths"] = ["/etc/yeto-runtime.txt"]

    with pytest.raises(HarnessError, match="must also be an execution.required"):
        _spec(tmp_path, not_required)


def test_cloud_doctor_rejects_insufficient_a2_cpu_quota(tmp_path):
    def four_gpu(raw):
        raw["cloud"]["machine_type"] = "a2-highgpu-4g"
        raw["cloud"]["accelerator_count"] = 4
        raw["cloud"]["max_total_accelerators"] = 4

    spec = _spec(tmp_path, four_gpu)

    class QuotaRunner:
        dry_run = False

        def run(self, command, *, check=True):
            if "machine-types" in command:
                payload = {
                    "guestCpus": 48,
                    "accelerators": [{"guestAcceleratorCount": 4}],
                }
            elif "regions" in command:
                payload = {
                    "quotas": [
                        {"metric": "A2_CPUS", "limit": 12, "usage": 0},
                        {
                            "metric": "PREEMPTIBLE_NVIDIA_A100_GPUS",
                            "limit": 16,
                            "usage": 0,
                        },
                    ]
                }
            else:
                raise AssertionError(command)
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    with pytest.raises(HarnessError, match=r"A2_CPUS.*requested=48"):
        harness.validate_compute_quota(spec, QuotaRunner())


def test_cloud_doctor_accepts_exact_available_a2_and_gpu_quota(tmp_path):
    def four_gpu(raw):
        raw["cloud"]["machine_type"] = "a2-highgpu-4g"
        raw["cloud"]["accelerator_count"] = 4
        raw["cloud"]["max_total_accelerators"] = 4

    spec = _spec(tmp_path, four_gpu)

    class QuotaRunner:
        dry_run = False

        def run(self, command, *, check=True):
            payload = (
                {
                    "guestCpus": 48,
                    "accelerators": [{"guestAcceleratorCount": 4}],
                }
                if "machine-types" in command
                else {
                    "quotas": [
                        {"metric": "A2_CPUS", "limit": 60, "usage": 12},
                        {
                            "metric": "PREEMPTIBLE_NVIDIA_A100_GPUS",
                            "limit": 16,
                            "usage": 12,
                        },
                    ]
                }
            )
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    harness.validate_compute_quota(spec, QuotaRunner())


def test_exp254_checked_in_drafts_are_nonlaunchable_and_sequential():
    root = Path(__file__).resolve().parents[1] / "experiments" / "optimizer"
    smoke = load_spec(root / "exp2-54-smoke-capture-draft.json")
    development = load_spec(root / "exp2-54a-seed223-development-draft.json")
    confirmation = load_spec(root / "exp2-54b-seed239-confirmation-locked.json")

    assert all(spec.cloud["adopt_only"] for spec in (smoke, development, confirmation))
    assert smoke.cloud["max_total_accelerators"] == 4
    assert development.cloud["max_total_accelerators"] == 4
    assert confirmation.cloud["max_total_accelerators"] == 4
    assert smoke.cloud["max_run_duration_seconds"] == 3600
    assert development.cloud["max_run_duration_seconds"] == 7200
    assert confirmation.cloud["max_run_duration_seconds"] == 7200
    assert len({spec.artifact_uri for spec in (smoke, development, confirmation)}) == 3

    assert smoke.cloud["labels"]["evidence"] == "none"
    assert smoke.checks["strict_quorum_step_budget"]["required_learner_steps"] == 80
    assert (
        development.checks["strict_quorum_step_budget"]["required_learner_steps"] == 512
    )
    assert (
        confirmation.checks["strict_quorum_step_budget"]["required_learner_steps"]
        == 512
    )

    dev_seed = development.command[development.command.index("--training-seed") + 1]
    locked_seed = confirmation.command[
        confirmation.command.index("--training-seed") + 1
    ]
    assert dev_seed == "223223"
    assert locked_seed == "__OPEN_SEED239_ONLY_AFTER_FROZEN_SELECTION__"
    assert smoke.command[smoke.command.index("--settings") + 1] == (
        "capture_m4_off,capture_m4_on"
    )
    assert "--syncer-probe-capture" in smoke.command
    assert "--optimizer-state-capture-parity" in smoke.command
    assert smoke.command[smoke.command.index("--syncer-probe-capture-every") + 1] == "1"
    assert (
        sum(
            path.endswith("/syncer_probe/index.jsonl")
            for path in smoke.execution["completion_paths"]
        )
        == 2
    )
    assert any(
        path.endswith("/optimizer_state_capture_parity.json")
        for path in smoke.execution["completion_paths"]
    )
    assert any(
        path.endswith("/optimizer_state_capture_parity.inputs.sha256")
        for path in smoke.execution["checksum_manifests"]
    )
    assert smoke.execution["input_checksum_manifests"] == [
        "/etc/yeto-model-files.sha256",
        "/etc/yeto-data.sha256",
    ]
    assert any(
        path.endswith("/scripts/validate_optimizer_capture_parity.py")
        for path in smoke.execution["required_paths"]
    )


@pytest.mark.parametrize(
    "name",
    [
        "exp2-54a-seed223-development-draft.json",
        "exp2-54b-seed239-confirmation-locked.json",
    ],
)
def test_exp254_full_capture_checksums_every_learner_and_transcript(name):
    root = Path(__file__).resolve().parents[1] / "experiments" / "optimizer"
    spec = load_spec(root / name)
    completion = spec.execution["completion_paths"]

    assert not any("bcmp_shadow" in path for path in completion)
    assert sum(path.endswith("/manifest.json") for path in completion) == 4
    assert sum(path.endswith("/manifest.json.sha256") for path in completion) == 4
    assert any(
        path.endswith("/syncer_response_transcript.json")
        or path.endswith("/syncer_response_transcript.jsonl")
        for path in completion
    )
    assert any(
        path.endswith("/optimizer_state_capture_validation.json") for path in completion
    )
    assert any(
        path.endswith("/optimizer_state_capture_validation.json.sha256")
        for path in completion
    )
    assert any(
        path.endswith("/optimizer_state_capture_artifacts.sha256")
        for path in completion
    )
    manifests = spec.execution["checksum_manifests"]
    assert sum(path.endswith("/manifest.json.sha256") for path in manifests) == 4
    assert any(
        path.endswith("/optimizer_state_capture_validation.json.sha256")
        for path in manifests
    )
    assert any(
        path.endswith("/optimizer_state_capture_artifacts.sha256") for path in manifests
    )
    assert "--optimizer-state-capture" in spec.command
    assert spec.command[spec.command.index("--syncer-total-steps") + 1] == "32"


def test_exp254_r4_draft_requires_observed_barrier_and_strict_writer():
    root = Path(__file__).resolve().parents[1] / "experiments" / "optimizer"
    spec = load_spec(root / "exp2-54-smoke-r4-barrier-draft.json")

    assert spec.cloud["labels"]["draft"] == "true"
    assert spec.cloud["adopt_only"] is True
    assert spec.repo_commit == "8cbff7650440e87a321ad525d485eef4c295a7d4"
    for flag in (
        "--strict-quorum",
        "--barrier-sync",
        "--optimizer-state-capture",
        "--optimizer-state-capture-parity",
        "--optimizer-state-capture-parity-require-barrier",
        "--optimizer-state-capture-strict-writer",
        "--syncer-probe-capture",
    ):
        assert spec.command.count(flag) == 1
        assert spec.checks["expected_flags"][flag] == ""
    assert (
        spec.command[
            spec.command.index("--optimizer-state-capture-min-joined-boundaries") + 1
        ]
        == "16"
    )
    assert (
        spec.command[
            spec.command.index("--optimizer-state-capture-min-joined-per-fragment") + 1
        ]
        == "4"
    )


def test_exp254_r5_async_canary_is_exactly_pinned_launchable_and_one_gpu():
    root = Path(__file__).resolve().parents[1] / "experiments" / "optimizer"
    spec = load_spec(root / "exp2-54-smoke-r5-async-canary.json")

    run_id = "exp2-54-smoke-r5-async-canary"
    assert spec.run_id == spec.instance_name == run_id
    assert spec.repo_commit == "4ed893d195f260ba7560680d1cf3e5030f1e7bed"
    assert spec.artifact_uri == ("gs://yeto-exp2-52-model-training-497007/" + run_id)
    assert spec.execution["source_mode"] == "checkout"
    assert spec.remote_repo_dir == f"/home/shou/experiments/{run_id}/repo"
    assert spec.remote_run_dir == f"/home/shou/runs/{run_id}"

    assert spec.cloud["adopt_only"] is False
    assert spec.cloud["labels"]["draft"] == "false"
    assert spec.cloud["machine_type"] == "a2-highgpu-1g"
    assert spec.cloud["accelerator_count"] == 1
    assert spec.cloud["max_total_accelerators"] == 1
    assert spec.cloud["provisioning_model"] == "SPOT"
    assert spec.cloud["boot_disk_type"] == "pd-ssd"
    assert spec.cloud["image"] == (
        "projects/model-training-497007/global/images/yeto-optimizer-a100-20260714"
    )
    assert spec.cloud["expected_source_image_id"] == "7290368630472593484"

    expected_values = {
        "--gpu-slots": "1",
        "--settings": "capture_m1_off,capture_m1_on",
        "--syncer-total-steps": "16",
        "--fixed-window-microsteps": "4",
        "--optimizer-state-capture-parity-overhead-limit": "0.02",
        "--optimizer-state-capture-writer-max-items": "4",
        "--optimizer-state-capture-writer-max-bytes": "4294967296",
        "--optimizer-state-capture-min-joined-boundaries": "16",
        "--optimizer-state-capture-min-joined-per-fragment": "4",
    }
    for flag, expected in expected_values.items():
        assert spec.command.count(flag) == 1
        assert spec.command[spec.command.index(flag) + 1] == expected
    for flag in (
        "--strict-quorum",
        "--barrier-sync",
        "--optimizer-state-capture",
        "--optimizer-state-capture-parity",
        "--optimizer-state-capture-parity-require-barrier",
        "--optimizer-state-capture-strict-writer",
        "--optimizer-state-capture-background-writer",
        "--syncer-probe-capture",
    ):
        assert spec.command.count(flag) == 1
        assert spec.checks["expected_flags"][flag] == ""
    assert "--learner-gpus" not in spec.command
    assert spec.checks["strict_quorum_step_budget"]["fragments"] == 4

    completion = spec.execution["completion_paths"]
    learner_completion = [
        path for path in completion if "optimizer_state_capture_learner_" in path
    ]
    assert len(learner_completion) == 2
    assert all(
        "optimizer_state_capture_learner_0/" in path for path in learner_completion
    )
    assert sum(path.endswith("/syncer_probe/index.jsonl") for path in completion) == 2
    assert {
        path for path in completion if path.endswith("/syncer_probe/index.jsonl")
    } == {
        f"/home/shou/runs/{run_id}/work/capture_m1_off/syncer_probe/index.jsonl",
        f"/home/shou/runs/{run_id}/work/capture_m1_on/syncer_probe/index.jsonl",
    }
    for suffix in (
        "/optimizer_state_capture_parity.json",
        "/optimizer_state_capture_parity.json.sha256",
        "/optimizer_state_capture_parity.inputs.sha256",
        "/optimizer_state_capture_committed_boundaries.json",
        "/optimizer_state_capture_committed_boundaries.json.sha256",
    ):
        assert sum(path.endswith(suffix) for path in completion) == 1

    checksums = spec.execution["checksum_manifests"]
    learner_checksums = [
        path for path in checksums if "optimizer_state_capture_learner_" in path
    ]
    assert len(learner_checksums) == 1
    assert "optimizer_state_capture_learner_0/" in learner_checksums[0]
    assert any(
        path.endswith("/optimizer_state_capture_parity.json.sha256")
        for path in checksums
    )
    assert any(
        path.endswith("/optimizer_state_capture_parity.inputs.sha256")
        for path in checksums
    )
    assert any(
        path.endswith("/optimizer_state_capture_committed_boundaries.json.sha256")
        for path in checksums
    )
    assert not any(
        re.search(r"optimizer_state_capture_learner_[1-9]", path)
        for path in (*completion, *checksums)
    )
    assert not any("capture_m4_" in path for path in (*completion, *checksums))

    rendered_launch = launch_command(spec)
    assert rendered_launch[:4] == ["gcloud", "compute", "instances", "create"]
    assert "--machine-type=a2-highgpu-1g" in rendered_launch
    assert "--provisioning-model=SPOT" in rendered_launch
    assert "--boot-disk-type=pd-ssd" in rendered_launch


def test_launch_verifies_boot_disk_image_provenance_before_recording_state(
    tmp_path, monkeypatch
):
    def pin(raw):
        raw["cloud"]["expected_source_image_id"] = "789"

    spec = _spec(tmp_path, pin)
    runner = SourceImageRunner(spec)
    monkeypatch.setattr(harness.secrets, "token_hex", lambda _: "0123456789abcdef")
    state = launch(spec, runner, tmp_path / "state", confirmed=True)

    assert state["boot_disk"] == {
        "name": "exp-test-vm",
        "id": "456",
        "self_link": state["boot_disk_self_link"],
        "user": state["instance_self_link"],
        "source_image_id": "789",
        "source_image": (
            "https://www.googleapis.com/compute/v1/projects/test-project/"
            "global/images/optimizer-image-v1"
        ),
        "source_image_path": ("projects/test-project/global/images/optimizer-image-v1"),
    }
    assert state["boot_disk_id"] == "456"
    assert state["source_image_id"] == "789"
    disk_commands = [
        command
        for command in runner.commands
        if command[:4] == ["gcloud", "compute", "disks", "describe"]
    ]
    assert len(disk_commands) == 1
    assert disk_commands[0][4] == "exp-test-vm"
    assert load_state(spec, tmp_path / "state")["boot_disk"]["id"] == "456"


@pytest.mark.parametrize(
    ("disk_changes", "message"),
    [
        ({"id": "not-numeric"}, "boot disk id"),
        ({"selfLink": "different-disk"}, "self-link differs"),
        ({"users": []}, "not bound only"),
        ({"sourceImageId": "790"}, "sourceImageId differs"),
        (
            {
                "sourceImage": (
                    "https://www.googleapis.com/compute/v1/projects/test-project/"
                    "global/images/different-image"
                )
            },
            "sourceImage path differs",
        ),
    ],
)
def test_launch_rejects_unpinned_boot_disk_provenance(
    tmp_path, monkeypatch, disk_changes, message
):
    def pin(raw):
        raw["cloud"]["expected_source_image_id"] = "789"

    spec = _spec(tmp_path, pin)
    disk = _boot_disk_description(spec)
    disk.update(disk_changes)
    runner = SourceImageRunner(spec, disk)
    monkeypatch.setattr(harness.secrets, "token_hex", lambda _: "0123456789abcdef")

    with pytest.raises(HarnessError, match=message):
        launch(spec, runner, tmp_path / "state", confirmed=True)
    quarantined = load_state(spec, tmp_path / "state")
    assert quarantined["status"] == "PROVENANCE_FAILED"
    assert quarantined["instance_id"] == "123"
    assert message.split()[0] in quarantined["provenance_error"]


def test_adopt_verifies_and_records_pinned_boot_disk_provenance(tmp_path, monkeypatch):
    def pin(raw):
        raw["cloud"]["expected_source_image_id"] = "789"

    spec = _spec(tmp_path, pin)
    runner = SourceImageRunner(spec)
    monkeypatch.setattr(harness.secrets, "token_hex", lambda _: "0123456789abcdef")
    state = adopt(
        spec,
        runner,
        tmp_path / "state",
        exact_instance_id="123",
        confirmed=True,
    )

    assert state["boot_disk"]["id"] == "456"
    assert state["boot_disk"]["source_image_id"] == "789"
    assert state["boot_disk_id"] == "456"
    assert state["source_image_id"] == "789"
    assert [command[2:4] for command in runner.commands][-2:] == [
        ["instances", "describe"],
        ["disks", "describe"],
    ]


def test_adopt_rejects_source_image_mismatch_without_accepting_state(
    tmp_path, monkeypatch
):
    def pin(raw):
        raw["cloud"]["expected_source_image_id"] = "789"

    spec = _spec(tmp_path, pin)
    disk = _boot_disk_description(spec, source_image_id="999")
    runner = SourceImageRunner(spec, disk)
    monkeypatch.setattr(harness.secrets, "token_hex", lambda _: "0123456789abcdef")

    with pytest.raises(HarnessError, match="sourceImageId differs"):
        adopt(
            spec,
            runner,
            tmp_path / "state",
            exact_instance_id="123",
            confirmed=True,
        )
    quarantined = load_state(spec, tmp_path / "state")
    assert quarantined["status"] == "PROVENANCE_FAILED"
    assert quarantined["instance_id"] == "123"
    assert "sourceImageId differs" in quarantined["provenance_error"]


def test_image_confirmation_and_exact_disk_id_are_required(tmp_path):
    args = build_parser().parse_args(
        ["create-image", "spec.json", "--instance-id", "123"]
    )
    assert args.yes is False
    args = build_parser().parse_args(
        ["create-image", "spec.json", "--instance-id", "123", "--yes"]
    )
    assert args.yes is True

    disk = {
        "id": "456",
        "selfLink": "disk-link",
        "users": ["instance-link"],
    }
    assert (
        verify_disk_description(
            disk,
            expected_id="456",
            expected_self_link="disk-link",
            expected_user="instance-link",
        )
        == "456"
    )
    with pytest.raises(HarnessError, match="boot disk id"):
        verify_disk_description(
            disk,
            expected_id="999",
            expected_self_link="disk-link",
            expected_user="instance-link",
        )


def test_matched_command_freezes_fitted_eta_and_drops_capture(tmp_path):
    def scaffold(raw):
        command = raw["execution"]["command"]
        command[command.index("--settings") + 1] = "scaffold_lite,scaffold_sgd"
        raw["checks"]["expected_arms"] = ["scaffold_lite", "scaffold_sgd"]

    spec = _spec(tmp_path, scaffold)
    command = matched_sgd_command(
        spec,
        {
            "source_outer_lr": 0.28,
            "eta_match": 0.35,
            "same_state_causal_fit": False,
        },
    )
    assert command[command.index("--settings") + 1] == "scaffold_sgd"
    assert command[command.index("--outer-lr") + 1] == "0.35"
    assert command[command.index("--work-dir") + 1].endswith("/matched-sgd/work")
    assert "--syncer-probe-capture" not in command

    with pytest.raises(HarnessError, match="source_outer_lr"):
        matched_sgd_command(
            spec,
            {
                "source_outer_lr": 0.175,
                "eta_match": 0.35,
                "same_state_causal_fit": False,
            },
        )

    script, rendered, result_path = matched_start_script(
        spec,
        {
            "source_outer_lr": 0.28,
            "eta_match": 0.35,
            "same_state_causal_fit": False,
        },
        subdirectory="matched-sgd",
    )
    assert rendered == command
    assert result_path.endswith("/matched-sgd/report/results.jsonl")
    assert "final-manifest.sha256" in script
    assert "gcloud storage rsync --recursive" in script


class AbandonRunner:
    dry_run = False

    def __init__(
        self,
        spec,
        state,
        *,
        completion_returncode=9,
        preservation_returncode=0,
        preservation_sha256=None,
        disk_survives=False,
    ):
        self.spec = spec
        self.state = state
        self.completion_returncode = completion_returncode
        self.preservation_returncode = preservation_returncode
        self.preservation_sha256 = preservation_sha256
        self.disk_survives = disk_survives
        self.deleted = False
        self.commands = []

    def run(self, command, *, check=True, capture=True):
        self.commands.append(command)
        if command[:4] == ["gcloud", "compute", "instances", "describe"]:
            if self.deleted:
                return subprocess.CompletedProcess(
                    command, 1, "", "The resource was not found"
                )
            return subprocess.CompletedProcess(
                command, 0, json.dumps(_description(self.spec)), ""
            )
        if command[:4] == ["gcloud", "compute", "disks", "describe"]:
            if self.deleted and not self.disk_survives:
                return subprocess.CompletedProcess(
                    command, 1, "", "HTTPError 404: disk was not found"
                )
            disk = {
                "id": "456",
                "selfLink": self.state["boot_disk_self_link"],
                "users": [self.state["instance_self_link"]],
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(disk), "")
        if command[:3] == ["gcloud", "compute", "ssh"]:
            remote_script = next(
                item.removeprefix("--command=")
                for item in command
                if item.startswith("--command=")
            )
            if "abandonment.json" not in remote_script:
                return subprocess.CompletedProcess(
                    command,
                    self.completion_returncode,
                    "",
                    "missing completion artifact" if self.completion_returncode else "",
                )
            stdout = (
                f"ABANDONMENT_RECORD_SHA256={self.preservation_sha256}\n"
                if self.preservation_sha256
                else ""
            )
            result = subprocess.CompletedProcess(
                command,
                self.preservation_returncode,
                stdout,
                "remote preservation failed" if self.preservation_returncode else "",
            )
            if check and result.returncode:
                raise AssertionError("abandon must inspect preservation return codes")
            return result
        if command[:4] == ["gcloud", "compute", "instances", "delete"]:
            self.deleted = True
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected command: {command}")


def _saved_owned_state(spec, tmp_path):
    description = _description(spec)
    state = harness._base_state(spec, description, "0123456789abcdef")
    state["status"] = "RUNNING_EXPERIMENT"
    save_state(spec, state, tmp_path)
    return state


def test_abandon_cli_and_remote_script_are_explicit_and_fail_closed(tmp_path):
    spec = _spec(tmp_path)
    state = harness._base_state(spec, _description(spec), "0123456789abcdef")
    script, record, digest = abandonment_script(
        spec,
        state,
        boot_disk_id="456",
        reason="operator stopped failed smoke",
        requested_at_utc="2026-07-14T09:00:00Z",
    )
    assert record["reason"] == "operator stopped failed smoke"
    assert record["boot_disk_id"] == "456"
    assert record["labels"]["ownership-nonce"] == "0123456789abcdef"
    assert len(digest) == 64
    assert "run is complete; use delete" in script
    assert "gcloud storage rsync --recursive" in script
    assert "gcloud storage cat" in script
    assert 'stop_pid_file "$run/runner.pid" runner' in script
    assert "\x00" not in script
    assert "tr '\\000' ' '" in script
    assert "conflicting remote abandonment record" in script

    args = build_parser().parse_args(
        [
            "abandon",
            "spec.json",
            "--instance-id",
            "123",
            "--reason",
            "failed smoke",
        ]
    )
    assert args.yes is False
    assert args.reason == "failed smoke"
    args = build_parser().parse_args(
        [
            "abandon",
            "spec.json",
            "--instance-id",
            "123",
            "--reason",
            "failed smoke",
            "--yes",
        ]
    )
    assert args.yes is True


def test_abandon_preserves_deletes_and_retains_audit_state(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    state_dir = tmp_path / "state"
    state = _saved_owned_state(spec, state_dir)
    timestamp = "2026-07-14T09:00:00Z"
    monkeypatch.setattr(harness, "_utc_now", lambda: timestamp)
    _, _, digest = abandonment_script(
        spec,
        state,
        boot_disk_id="456",
        reason="runner failed before producing results",
        requested_at_utc=timestamp,
    )
    runner = AbandonRunner(spec, state, preservation_sha256=digest)

    result = abandon(
        spec,
        runner,
        state_dir,
        exact_instance_id="123",
        reason="runner failed before producing results",
        confirmed=True,
    )

    assert result["status"] == "ABANDONED"
    assert result["abandon_reason"] == "runner failed before producing results"
    assert result["abandoned_at_utc"] == timestamp
    assert result["abandonment"]["boot_disk_id"] == "456"
    assert result["abandonment"]["record_sha256"] == digest
    assert load_state(spec, state_dir) == result
    assert runner.deleted is True
    delete_commands = [
        command
        for command in runner.commands
        if command[:4] == ["gcloud", "compute", "instances", "delete"]
    ]
    assert len(delete_commands) == 1
    assert spec.instance_name in delete_commands[0]
    assert any(
        command[:4] == ["gcloud", "compute", "disks", "describe"]
        for command in runner.commands
    )


@pytest.mark.parametrize(
    ("completion_returncode", "preservation_returncode", "sha256", "error", "status"),
    [
        (0, 0, None, "run is complete; use delete", "RUNNING_EXPERIMENT"),
        (9, 21, None, "run is complete; use delete", "RUNNING_EXPERIMENT"),
        (9, 23, None, "preservation failed", "ABANDONING_PRESERVATION"),
        (9, 0, "0" * 64, "exact remote record", "ABANDONING_PRESERVATION"),
    ],
)
def test_abandon_refuses_completed_or_unpreserved_run(
    tmp_path,
    monkeypatch,
    completion_returncode,
    preservation_returncode,
    sha256,
    error,
    status,
):
    spec = _spec(tmp_path)
    state_dir = tmp_path / "state"
    state = _saved_owned_state(spec, state_dir)
    monkeypatch.setattr(harness, "_utc_now", lambda: "2026-07-14T09:00:00Z")
    runner = AbandonRunner(
        spec,
        state,
        completion_returncode=completion_returncode,
        preservation_returncode=preservation_returncode,
        preservation_sha256=sha256,
    )
    with pytest.raises(HarnessError, match=error):
        abandon(
            spec,
            runner,
            state_dir,
            exact_instance_id="123",
            reason="failed run",
            confirmed=True,
        )
    assert runner.deleted is False
    assert load_state(spec, state_dir)["status"] == status


def test_abandon_preservation_failure_is_idempotently_retryable(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    state_dir = tmp_path / "state"
    initial = _saved_owned_state(spec, state_dir)
    timestamps = iter(("2026-07-14T09:00:00Z", "2026-07-14T09:05:00Z"))
    monkeypatch.setattr(harness, "_utc_now", lambda: next(timestamps))
    failed_runner = AbandonRunner(
        spec,
        initial,
        preservation_returncode=23,
    )
    with pytest.raises(HarnessError, match="preservation failed"):
        abandon(
            spec,
            failed_runner,
            state_dir,
            exact_instance_id="123",
            reason="failed run",
            confirmed=True,
        )

    retry_state = load_state(spec, state_dir)
    requested_at = retry_state["abandonment"]["requested_at_utc"]
    assert retry_state["status"] == "ABANDONING_PRESERVATION"
    assert requested_at == "2026-07-14T09:00:00Z"
    _, _, digest = abandonment_script(
        spec,
        retry_state,
        boot_disk_id="456",
        reason="failed run",
        requested_at_utc=requested_at,
    )
    retry_runner = AbandonRunner(spec, retry_state, preservation_sha256=digest)
    result = abandon(
        spec,
        retry_runner,
        state_dir,
        exact_instance_id="123",
        reason="failed run",
        confirmed=True,
    )
    assert result["status"] == "ABANDONED"
    assert result["abandoned_at_utc"] == "2026-07-14T09:05:00Z"
    assert result["abandonment"]["requested_at_utc"] == requested_at


def test_abandon_requires_confirmation_reason_and_exact_id(tmp_path):
    spec = _spec(tmp_path)
    state_dir = tmp_path / "state"
    state = _saved_owned_state(spec, state_dir)
    runner = AbandonRunner(spec, state)

    with pytest.raises(HarnessError, match="requires --yes"):
        abandon(
            spec,
            runner,
            state_dir,
            exact_instance_id="123",
            reason="failed",
            confirmed=False,
        )
    with pytest.raises(HarnessError, match="--reason"):
        abandon(
            spec,
            runner,
            state_dir,
            exact_instance_id="123",
            reason="   ",
            confirmed=True,
        )
    with pytest.raises(HarnessError, match="recorded exact id"):
        abandon(
            spec,
            runner,
            state_dir,
            exact_instance_id="999",
            reason="failed",
            confirmed=True,
        )
    assert runner.deleted is False


def test_abandon_refuses_recorded_boot_disk_id_mismatch(tmp_path):
    spec = _spec(tmp_path)
    state_dir = tmp_path / "state"
    state = _saved_owned_state(spec, state_dir)
    state["boot_disk"] = {"id": "999"}
    save_state(spec, state, state_dir)
    runner = AbandonRunner(spec, state)

    with pytest.raises(HarnessError, match="boot disk id"):
        abandon(
            spec,
            runner,
            state_dir,
            exact_instance_id="123",
            reason="failed run",
            confirmed=True,
        )
    assert runner.deleted is False
    assert not any(
        command[:3] == ["gcloud", "compute", "ssh"] for command in runner.commands
    )


def test_abandon_refuses_recorded_label_mismatch(tmp_path):
    spec = _spec(tmp_path)
    state_dir = tmp_path / "state"
    state = _saved_owned_state(spec, state_dir)
    state["labels"]["unexpected"] = "replacement"
    save_state(spec, state, state_dir)
    runner = AbandonRunner(spec, state)

    with pytest.raises(HarnessError, match="recorded ownership labels"):
        abandon(
            spec,
            runner,
            state_dir,
            exact_instance_id="123",
            reason="failed run",
            confirmed=True,
        )
    assert runner.deleted is False


def test_abandon_fails_if_auto_delete_disk_survives(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    state_dir = tmp_path / "state"
    state = _saved_owned_state(spec, state_dir)
    timestamp = "2026-07-14T09:00:00Z"
    monkeypatch.setattr(harness, "_utc_now", lambda: timestamp)
    monkeypatch.setattr(harness.time, "sleep", lambda _seconds: None)
    _, _, digest = abandonment_script(
        spec,
        state,
        boot_disk_id="456",
        reason="failed run",
        requested_at_utc=timestamp,
    )
    runner = AbandonRunner(
        spec,
        state,
        preservation_sha256=digest,
        disk_survives=True,
    )

    with pytest.raises(HarnessError, match="boot disk still exists"):
        abandon(
            spec,
            runner,
            state_dir,
            exact_instance_id="123",
            reason="failed run",
            confirmed=True,
        )
    assert load_state(spec, state_dir)["status"] == "ABANDONING"
