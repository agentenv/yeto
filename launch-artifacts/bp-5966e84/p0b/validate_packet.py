#!/usr/bin/env python3
"""Fail-closed validation for the non-launching P0b packet."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[2]
RUN_ID = "bp-p0b-5966e84-20260715a"
SOURCE_COMMIT = "8d58208cacafef12cb95f2642b4fa700531151b4"
PROTECTED_INSTANCE_ID = "3908640733128066700"
STATE_PATH = Path("/tmp/yeto-p0b-state") / f"{RUN_ID}.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None):
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    spec_path = PACKET / "optimizer-harness-p0b.json"
    bootstrap_path = PACKET / "bootstrap.sh"
    execute_path = PACKET / "execute-argv.json"
    materialization_path = PACKET / "materialized" / "materialization.json"
    bound_path = PACKET / "materialized" / "bound-manifest.json"
    replay_path = PACKET.parent / "p0a-parent" / "p0a-replay-report.json"
    parent_path = PACKET.parent / "p0a-parent" / "phase-map-manifest.json"

    spec = json.loads(spec_path.read_text())
    bootstrap = bootstrap_path.read_bytes()
    execute = json.loads(execute_path.read_text())
    materialization = json.loads(materialization_path.read_text())
    bound = json.loads(bound_path.read_text())
    replay = json.loads(replay_path.read_text())
    parent = json.loads(parent_path.read_text())

    require(spec["run_id"] == RUN_ID, "spec run ID mismatch")
    require(spec["repo_commit"] == SOURCE_COMMIT, "spec source commit mismatch")
    require(spec["cloud"]["instance_name"] == RUN_ID, "instance name mismatch")
    require(spec["cloud"]["machine_type"] == "a2-highgpu-4g", "machine mismatch")
    require(spec["cloud"]["accelerator_count"] == 4, "accelerator count mismatch")
    require(spec["cloud"]["max_total_accelerators"] == 16, "capacity mismatch")
    require(spec["cloud"]["provisioning_model"] == "SPOT", "not Spot")
    require(spec["cloud"]["termination_action"] == "DELETE", "not DELETE")
    require(
        spec["artifacts"]["uri"]
        == f"gs://yeto-exp2-52-model-training-497007/{RUN_ID}",
        "artifact prefix mismatch",
    )
    embedded_command = spec["execution"]["command"][2]
    encoded = shlex.split(embedded_command)[2]
    require(base64.b64decode(encoded) == bootstrap, "embedded bootstrap mismatch")
    require(PROTECTED_INSTANCE_ID.encode() not in bootstrap, "protected ID in bootstrap")
    require(
        PROTECTED_INSTANCE_ID not in json.dumps(spec, sort_keys=True),
        "protected ID in spec",
    )

    syntax = run(["/bin/bash", "-n", str(bootstrap_path)])
    require(syntax.returncode == 0, f"bootstrap syntax failed: {syntax.stderr}")
    harness_validate = run(
        [sys.executable, "-m", "yeto.optimizer_harness", "validate", str(spec_path)]
    )
    require(
        harness_validate.returncode == 0,
        f"harness validation failed: {harness_validate.stderr}",
    )
    harness_render = run(
        [sys.executable, "-m", "yeto.optimizer_harness", "render", str(spec_path)]
    )
    require(
        harness_render.returncode == 0,
        f"harness render failed: {harness_render.stderr}",
    )
    render = harness_render.stdout.rstrip() + "\n"
    for token in (
        "--machine-type=a2-highgpu-4g",
        "--provisioning-model=SPOT",
        "--instance-termination-action=DELETE",
        "--maintenance-policy=TERMINATE",
        "--no-restart-on-failure",
        "--max-run-duration=14400s",
        SOURCE_COMMIT,
    ):
        require(token in render, f"render lacks {token}")
    require(PROTECTED_INSTANCE_ID not in render, "protected ID in render")

    require(replay["status"] == "PASS", "parent replay is not PASS")
    require(replay["all_steps_replayed"] is True, "parent replay is incomplete")
    require(replay["cell_count"] == 3, "parent replay cell count mismatch")
    require(
        replay["replay_validator_git_commit"] == SOURCE_COMMIT,
        "replay/source commit mismatch",
    )
    require(
        sha256_file(replay_path)
        == "4c1616de16708590d6a30aaf3af805adc4bad47b827087b49251d678e200c276",
        "parent replay raw hash mismatch",
    )
    require(
        sha256_bytes(canonical_json(parent))
        == "02b1f99537d2611e3462ebe1b4ccedd11fdc07588b7c01d3abdeabbdb5b9d8f8",
        "parent canonical hash mismatch",
    )
    require(bound["status"] == "bound_launch_authority", "bound status mismatch")
    require(bound["frozen"]["git_commit"] == SOURCE_COMMIT, "bound source mismatch")
    require(len(bound["expected_cells"]) == 3, "P0b must contain three cells")
    require(
        {cell["mu"] for cell in bound["expected_cells"]} == {0, 0.5, 0.9},
        "P0b mu block mismatch",
    )
    require(
        materialization["bound_manifest_hash"]
        == sha256_bytes(canonical_json(bound)),
        "bound canonical hash mismatch",
    )
    require(
        materialization["randomization_plan_hash"]
        == bound["frozen"]["randomization_plan_hash"],
        "plan hash mismatch",
    )
    require(
        execute[-4:] == [
            "--expected-randomization-plan-hash",
            materialization["randomization_plan_hash"],
            "--expected-bound-manifest-hash",
            materialization["bound_manifest_hash"],
        ],
        "execute argv does not bind materialization",
    )
    require(shlex.join(execute) in bootstrap.decode(), "bootstrap execute argv drift")

    branch = run(
        [
            "git",
            "ls-remote",
            "git@github.com:agentenv/yeto.git",
            "experiment/best-paper-phase-map",
        ]
    )
    require(branch.returncode == 0, f"remote verification failed: {branch.stderr}")
    remote_head = branch.stdout.split()[0]
    exact_source = run(["git", "cat-file", "-e", f"{SOURCE_COMMIT}^{{commit}}"])
    source_ancestry = run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, remote_head]
    )
    require(exact_source.returncode == 0, "packet source commit object is absent")
    require(
        source_ancestry.returncode == 0,
        "packet source commit is not an ancestor of the remote branch head",
    )
    bundle = PACKET / "source" / "yeto-8d58208ca.bundle"
    bundle_check = run(["git", "bundle", "verify", str(bundle)])
    require(bundle_check.returncode == 0, f"source bundle invalid: {bundle_check.stderr}")

    cloud_env = dict(os.environ)
    cloud_env["CLOUDSDK_CONFIG"] = "/private/tmp/yeto-gcloud-admin-codex"
    prefix = spec["artifacts"]["uri"]
    storage = run(
        ["gcloud", "storage", "ls", "--all-versions", f"{prefix}/**"],
        env=cloud_env,
    )
    storage_text = (storage.stdout + storage.stderr).strip()
    prefix_empty = storage.returncode == 0 and storage.stdout.strip() == ""
    if storage.returncode != 0:
        prefix_empty = bool(
            re.search(r"matched no objects|matched no URLs|not found", storage_text, re.I)
        )
    require(prefix_empty, f"artifact prefix is not proven empty: {storage_text}")

    describe = run(
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            RUN_ID,
            "--project=model-training-497007",
            "--zone=us-central1-c",
            "--format=json",
        ],
        env=cloud_env,
    )
    describe_text = (describe.stdout + describe.stderr).strip()
    instance_absent = describe.returncode != 0 and bool(
        re.search(r"not found|was not found|could not fetch resource", describe_text, re.I)
    )
    require(instance_absent, "instance name is not proven absent")
    require(not STATE_PATH.exists(), f"controller state already exists: {STATE_PATH}")

    gates = PACKET / "gates"
    gates.mkdir(parents=True, exist_ok=True)
    (gates / "harness-validate.txt").write_text(harness_validate.stdout)
    (PACKET / "optimizer-harness-p0b.rendered.txt").write_text(render)
    cloud_report = {
        "status": "PASS",
        "read_only": True,
        "artifact_prefix": prefix,
        "artifact_prefix_empty": True,
        "instance_name": RUN_ID,
        "instance_absent": True,
        "controller_state_path": str(STATE_PATH),
        "controller_state_absent": True,
        "protected_instance_id": PROTECTED_INSTANCE_ID,
        "protected_instance_targeted": False,
        "storage_command_returncode": storage.returncode,
        "storage_command_output": storage_text,
        "describe_command_returncode": describe.returncode,
        "describe_command_output": describe_text,
    }
    (gates / "cloud-readonly-preflight.json").write_text(
        json.dumps(cloud_report, indent=2, sort_keys=True) + "\n"
    )
    report = {
        "status": "PASS",
        "run_id": RUN_ID,
        "source_commit": SOURCE_COMMIT,
        "bootstrap_bash_syntax": "PASS",
        "embedded_bootstrap_byte_identity": "PASS",
        "harness_validate": "PASS",
        "harness_render": "PASS",
        "bound_manifest_validation": "PASS",
        "source_bundle_validation": "PASS",
        "remote_source_validation": "PASS",
        "remote_branch_head": remote_head,
        "fresh_namespace_readonly_preflight": "PASS",
        "parent_replay_status": replay["status"],
        "parent_replay_raw_sha256": sha256_file(replay_path),
        "parent_manifest_canonical_sha256": sha256_bytes(canonical_json(parent)),
        "bound_manifest_canonical_sha256": materialization["bound_manifest_hash"],
        "randomization_plan_hash": materialization["randomization_plan_hash"],
        "campaign_command_hash": materialization["campaign_command_hash"],
        "protected_instance_targeted": False,
        "launch_performed": False,
    }
    (gates / "packet-validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
