#!/usr/bin/env python3
"""Preflight or drain one EXPLORATORY Lane B per-GPU queue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


LABEL = "EXPLORATORY"
COMMIT = "a886a3996905913d37ec56cc14914878f636283d"
RESULT_ROOT = Path("/root/yeto-results-explore")
CONTROL_ROOT = RESULT_ROOT / "_controller"
REPO = Path("/root/yeto")
TELEMETRY_SCHEMA = "yeto_rho_telemetry_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected JSON object")
            rows.append(value)
    return rows


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def gpu_inventory() -> dict[int, dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory: dict[int, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        index, gpu_uuid, name, memory_used, utilization = [
            part.strip() for part in line.split(",", 4)
        ]
        inventory[int(index)] = {
            "uuid": gpu_uuid,
            "name": name,
            "memory_used_mib": memory_used,
            "utilization_percent": utilization,
        }
    return inventory


def option(command: list[str], flag: str) -> str:
    try:
        index = command.index(flag)
    except ValueError as exc:
        raise ValueError(f"missing command flag {flag}") from exc
    if index + 1 >= len(command):
        raise ValueError(f"missing value for {flag}")
    return command[index + 1]


def preflight(queue_path: Path, manifest_path: Path) -> Path:
    queue = json.loads(queue_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    if queue.get("label") != LABEL or manifest.get("label") != LABEL:
        errors.append("queue/manifest is not labeled EXPLORATORY")
    if queue.get("schema") != "yeto_e1x_lane_b_gpu_queue_v1":
        errors.append("queue schema mismatch")
    if manifest.get("schema") != "yeto_e1x_lane_b_finite_t_manifest_v1":
        errors.append("manifest schema mismatch")
    if queue.get("manifest_sha256") != sha256_file(manifest_path):
        errors.append("queue does not bind this manifest")
    if manifest.get("cell_count") != 72:
        errors.append("manifest does not contain exactly 72 cells")
    if manifest.get("source_git_commit") != COMMIT:
        errors.append("manifest Git commit mismatch")
    try:
        if git("rev-parse", "HEAD") != COMMIT:
            errors.append("node repo is not at the requested commit")
        if git("status", "--porcelain=v1", "--untracked-files=all"):
            errors.append("node repo is dirty")
    except Exception as exc:
        errors.append(f"Git verification failed: {exc}")
    if not Path("/root/yeto-venv/bin/python").is_file():
        errors.append("requested virtualenv Python is missing")
    inventory = gpu_inventory()
    if set(inventory) != set(range(8)):
        errors.append(f"expected GPU indices 0..7, found {sorted(inventory)}")
    if any("H200" not in item["name"].upper() for item in inventory.values()):
        errors.append("a node GPU is not an H200")
    for index, item in inventory.items():
        if int(item["memory_used_mib"]) > 100 or int(item["utilization_percent"]) != 0:
            errors.append(
                f"GPU {index} is not idle: memory={item['memory_used_mib']} MiB, "
                f"utilization={item['utilization_percent']}%"
            )
    node = str(queue.get("node"))
    gpu = int(queue.get("gpu", -1))
    if gpu not in inventory:
        errors.append(f"queue GPU {gpu} is unavailable")

    for seed in (401, 409):
        seed_root = Path(f"/root/yeto-data/outer-mup-explore/seed-{seed}")
        input_manifest = seed_root / "input-manifest.json"
        try:
            value = json.loads(input_manifest.read_text())
            if value.get("label") != LABEL:
                errors.append(f"seed {seed} manifest lacks EXPLORATORY label")
            for name in ("train.jsonl", "eval.jsonl", "confirmation-audit.jsonl"):
                if not (seed_root / name).is_file():
                    errors.append(f"seed {seed} input is missing: {name}")
        except Exception as exc:
            errors.append(f"seed {seed} manifest error: {exc}")

    for cell in queue.get("cells", []):
        cell_id = str(cell.get("cell_id", ""))
        command = list(cell.get("command", []))
        if cell.get("label") != LABEL or not cell_id.startswith("e1x-"):
            errors.append(f"non-exploratory cell identity: {cell_id}")
        if cell.get("assignment") != {"node": node, "gpu": gpu}:
            errors.append(f"assignment mismatch: {cell_id}")
        if int(cell.get("seed", -1)) not in (401, 409):
            errors.append(f"invalid seed: {cell_id}")
        if int(cell.get("seed", -1)) == 307:
            errors.append(f"reserved seed entered queue: {cell_id}")
        if canonical_sha256(command) != cell.get("command_hash"):
            errors.append(f"command hash mismatch: {cell_id}")
        if "--rho-telemetry" not in command:
            errors.append(f"rho telemetry disabled: {cell_id}")
        try:
            if option(command, "--gpu-offset") != str(gpu):
                errors.append(f"GPU offset mismatch: {cell_id}")
            if option(command, "--work-dir") != str(
                RESULT_ROOT / cell_id / "attempt-1" / "work"
            ):
                errors.append(f"work path escapes exploratory root: {cell_id}")
            if option(command, "--report-dir") != str(
                RESULT_ROOT / cell_id / "attempt-1" / "report"
            ):
                errors.append(f"report path escapes exploratory root: {cell_id}")
            s = int(cell["s"])
            if int(option(command, "--token-budget")) != s * 512:
                errors.append(f"token/step scaling mismatch: {cell_id}")
            if int(option(command, "--learner-max-steps")) != s:
                errors.append(f"learner step cap mismatch: {cell_id}")
            if int(option(command, "--syncer-total-steps")) != 4 * s // 512:
                errors.append(f"outer step count mismatch: {cell_id}")
        except Exception as exc:
            errors.append(f"command validation error for {cell_id}: {exc}")

    proof = {
        "label": LABEL,
        "schema": "yeto_e1x_lane_b_preflight_v1",
        "checked_at_utc": utc_now(),
        "node": node,
        "gpu": gpu,
        "gpu_inventory": inventory,
        "queue": str(queue_path),
        "queue_sha256": sha256_file(queue_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "git_commit": COMMIT,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    proof_path = CONTROL_ROOT / "preflight" / f"{node}-gpu{gpu}.json"
    write_json_atomic(proof_path, proof)
    if errors:
        raise SystemExit("; ".join(errors))
    print(json.dumps({"label": LABEL, "node": node, "gpu": gpu, "status": "PASS"}))
    return proof_path


def prepend_label(path: Path, header: str) -> None:
    if not path.is_file():
        return
    text = path.read_text(errors="replace")
    if not text.startswith(header):
        path.write_text(header + text)


def require_file(
    failures: list[str], artifacts: dict[str, dict[str, object]], label: str, path: Path
) -> bool:
    if not path.is_file():
        failures.append(f"missing {label}: {path}")
        return False
    artifacts[label] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    return True


def validate_and_bank(
    cell: dict[str, object], attempt: Path, node: str, gpu: int, return_code: int
) -> dict[str, object]:
    failures: list[str] = []
    artifacts: dict[str, dict[str, object]] = {}
    report = attempt / "report"
    work = attempt / "work" / "m4"
    results_path = report / "results.jsonl"
    telemetry_path = work / "rho-telemetry.jsonl"
    tape_path = work / "tape.jsonl"
    result_row: dict[str, object] | None = None

    if return_code != 0:
        failures.append(f"compare_diloco exited {return_code}")
    if require_file(failures, artifacts, "results", results_path):
        try:
            rows = read_jsonl(results_path)
            if len(rows) != 1:
                failures.append(f"expected one endpoint result, found {len(rows)}")
            else:
                result_row = rows[0]
                loss = result_row.get("eval_loss")
                if not isinstance(loss, (int, float)) or not math.isfinite(loss):
                    failures.append("endpoint eval loss is nonfinite")
                if result_row.get("eval_rows") != 1024:
                    failures.append("endpoint eval row count mismatch")
                if result_row.get("learner_exit_codes") != [0, 0, 0, 0]:
                    failures.append("learner exit codes are not all zero")
                if result_row.get("syncer_exit_code") != 0:
                    failures.append("syncer exit code is nonzero")
        except Exception as exc:
            failures.append(f"result validation failed: {exc}")

    if require_file(failures, artifacts, "rho_telemetry_raw", telemetry_path):
        try:
            rows = read_jsonl(telemetry_path)
            expected = int(cell["expected_telemetry_rows"])
            if len(rows) != expected:
                failures.append(f"rho telemetry rows {len(rows)} != {expected}")
            for index, row in enumerate(rows, 1):
                if row.get("schema") != TELEMETRY_SCHEMA:
                    failures.append(f"rho telemetry schema mismatch at row {index}")
                    break
                if row.get("outer_step") != index:
                    failures.append(f"rho telemetry step mismatch at row {index}")
                    break
                norm = row.get("pseudo_gradient", {}).get("l2_norm")
                if not isinstance(norm, (int, float)) or not math.isfinite(norm):
                    failures.append(f"rho telemetry nonfinite norm at row {index}")
                    break
        except Exception as exc:
            failures.append(f"rho telemetry validation failed: {exc}")

    if require_file(failures, artifacts, "event_tape_raw", tape_path):
        try:
            rows = read_jsonl(tape_path)
            expected = int(cell["expected_outer_steps"])
            if len(rows) != expected:
                failures.append(f"event tape rows {len(rows)} != {expected}")
            for index, row in enumerate(rows, 1):
                responders = row.get("responders", [])
                if row.get("step") != index or len(responders) != 4:
                    failures.append(f"strict quorum/step mismatch at row {index}")
                    break
                if any(item.get("c_steps") != 512 for item in responders):
                    failures.append(f"fixed H mismatch at row {index}")
                    break
        except Exception as exc:
            failures.append(f"event tape validation failed: {exc}")

    acquisition_path = report / "acquisition-state.json"
    if require_file(failures, artifacts, "acquisition_state", acquisition_path):
        try:
            state = json.loads(acquisition_path.read_text())
            if state.get("phase") != "endpoint_recorded":
                failures.append("acquisition state lacks endpoint_recorded phase")
        except Exception as exc:
            failures.append(f"acquisition state validation failed: {exc}")
    for label, path in (
        ("barrier_registry", report / "barrier-version-trace.json"),
        ("recorded_command", attempt / "command.sh"),
        ("recorded_git_commit", attempt / "git_commit.txt"),
        ("recorded_git_diff", attempt / "git_diff.patch"),
    ):
        require_file(failures, artifacts, label, path)
    commit_path = attempt / "git_commit.txt"
    if commit_path.is_file() and commit_path.read_text().strip() != COMMIT:
        failures.append("recorded Git commit mismatch")
    diff_path = attempt / "git_diff.patch"
    if diff_path.is_file() and diff_path.read_text():
        failures.append("recorded Git diff is nonempty")
    for learner in range(4):
        log_path = work / f"learner-{learner}.log"
        if require_file(failures, artifacts, f"learner_{learner}_log_raw", log_path):
            text = log_path.read_text(errors="replace")
            match = re.search(r"inner loop done at local_step=(\d+) global_step=(\d+)", text)
            if not match:
                failures.append(f"learner {learner} lacks completion line")
            elif int(match.group(1)) != int(cell["s"]):
                failures.append(f"learner {learner} local step mismatch")
            elif int(match.group(2)) != int(cell["expected_outer_steps"]):
                failures.append(f"learner {learner} outer step mismatch")

    evidence = {
        "label": LABEL,
        "schema": "yeto_e1x_lane_b_cell_bank_v1",
        "cell_id": cell["cell_id"],
        "node": node,
        "gpu": gpu,
        "validated_at_utc": utc_now(),
        "status": "COMPLETED" if not failures else "INVALID",
        "failures": failures,
        "config": {
            key: cell[key]
            for key in (
                "grid",
                "h",
                "s",
                "t",
                "mu",
                "d_pred",
                "eta0_reference",
                "eta_center",
                "eta_index",
                "eta_ladder_exponent",
                "eta",
                "seed",
                "training_seed",
                "token_budget",
                "expected_outer_steps",
                "expected_telemetry_rows",
            )
        },
        "command": cell["command"],
        "command_hash": cell["command_hash"],
        "result": result_row,
        "raw_artifacts_before_pruning": artifacts,
        "evidence_mode": "compact exploratory bank; checkpoint/export omitted",
    }
    return evidence


def compact_successful_attempt(attempt: Path, evidence: dict[str, object]) -> None:
    cell_id = str(evidence["cell_id"])
    root = RESULT_ROOT.resolve()
    resolved_attempt = attempt.resolve()
    if not cell_id.startswith("e1x-") or not resolved_attempt.is_relative_to(root):
        raise RuntimeError(f"refusing to prune outside e1x exploratory root: {attempt}")
    work = attempt / "work" / "m4"
    bank = attempt / "bank"
    bank.mkdir(parents=True, exist_ok=False)
    (bank / "EXPLORATORY.md").write_text(
        "# EXPLORATORY — compact cell bank\n\n"
        "The multi-GB training checkpoint/export was intentionally pruned after validation.\n"
    )
    for source_name, destination_name in (
        ("rho-telemetry.jsonl", "rho-telemetry.jsonl"),
        ("tape.jsonl", "event-tape.jsonl"),
    ):
        shutil.copy2(work / source_name, bank / destination_name)
    for learner in range(4):
        source = work / f"learner-{learner}.log"
        destination = bank / f"learner-{learner}.log"
        shutil.copy2(source, destination)
        prepend_label(destination, "EXPLORATORY\n")
    metadata = {
        "label": LABEL,
        "schema": "yeto_e1x_compact_bank_artifact_index_v1",
        "cell_id": cell_id,
        "created_at_utc": utc_now(),
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(bank.iterdir())
            if path.is_file()
        },
        "pruned": [
            "work/ (checkpoint, exported model, learner model copies)",
            "report/eval-provenance/",
            "report/per-example-loss/",
        ],
    }
    write_json_atomic(bank / "artifact-index.json", metadata)
    work_root = attempt / "work"
    if work_root.is_dir():
        shutil.rmtree(work_root)
    for directory in (
        attempt / "report" / "eval-provenance",
        attempt / "report" / "per-example-loss",
    ):
        if directory.is_dir():
            shutil.rmtree(directory)
    prepend_label(attempt / "command.sh", "# EXPLORATORY\n")
    prepend_label(attempt / "report" / "report.md", "# EXPLORATORY\n\n")


def emit(log_path: Path, message: str) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("EXPLORATORY — Lane B slot log\n")
    with log_path.open("a") as log:
        log.write(line + "\n")


def run_queue(queue_path: Path, manifest_path: Path) -> int:
    queue = json.loads(queue_path.read_text())
    node = str(queue["node"])
    gpu = int(queue["gpu"])
    proof_path = CONTROL_ROOT / "preflight" / f"{node}-gpu{gpu}.json"
    if not proof_path.is_file():
        raise SystemExit(f"missing preflight proof: {proof_path}")
    proof = json.loads(proof_path.read_text())
    if (
        proof.get("label") != LABEL
        or proof.get("status") != "PASS"
        or proof.get("queue_sha256") != sha256_file(queue_path)
        or proof.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise SystemExit("preflight proof does not bind this queue/manifest")

    lock_path = CONTROL_ROOT / "locks" / f"{node}-gpu{gpu}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"slot already has a worker: {node} gpu{gpu}") from exc
    lock_handle.write(f"{LABEL} pid={os.getpid()}\n")
    lock_handle.flush()

    status_path = CONTROL_ROOT / "slots" / f"{node}-gpu{gpu}.json"
    worker_log = CONTROL_ROOT / "logs" / f"{node}-gpu{gpu}.log"
    inventory = gpu_inventory()
    completed = 0
    failures = 0
    for queue_index, cell in enumerate(queue["cells"]):
        cell_id = str(cell["cell_id"])
        attempt = RESULT_ROOT / cell_id / "attempt-1"
        evidence_path = attempt / "bank-result.json"
        if evidence_path.is_file():
            existing = json.loads(evidence_path.read_text())
            if existing.get("label") == LABEL and existing.get("status") == "COMPLETED":
                completed += 1
                emit(worker_log, f"SKIP completed {cell_id}")
                continue
            raise SystemExit(f"existing noncompleted bank requires review: {cell_id}")
        if attempt.exists():
            raise SystemExit(f"refusing to overwrite partial attempt: {attempt}")
        attempt.mkdir(parents=True)
        command = list(cell["command"])
        if canonical_sha256(command) != cell["command_hash"]:
            raise SystemExit(f"command hash mismatch immediately before launch: {cell_id}")
        attempt_id = f"{cell_id}-a1-{uuid.uuid4()}"
        start_record = {
            "label": LABEL,
            "schema": "yeto_e1x_lane_b_attempt_start_v1",
            "attempt_id": attempt_id,
            "cell_id": cell_id,
            "attempt_number": 1,
            "node": node,
            "gpu": gpu,
            "gpu_uuid": inventory[gpu]["uuid"],
            "gpu_name": inventory[gpu]["name"],
            "start_utc": utc_now(),
            "command": command,
            "command_hash": cell["command_hash"],
            "config": cell,
        }
        write_json_atomic(attempt / "attempt-start.json", start_record)
        (attempt / "command-exploratory.sh").write_text(
            "#!/bin/sh\n# EXPLORATORY — Lane B cell\nexec " + shlex.join(command) + "\n"
        )
        write_json_atomic(
            status_path,
            {
                "label": LABEL,
                "schema": "yeto_e1x_lane_b_slot_status_v1",
                "node": node,
                "gpu": gpu,
                "state": "RUNNING",
                "cell_id": cell_id,
                "queue_index": queue_index,
                "queue_total": len(queue["cells"]),
                "completed": completed,
                "failures": failures,
                "updated_at_utc": utc_now(),
            },
        )
        emit(worker_log, f"START {cell_id} queue={queue_index + 1}/{len(queue['cells'])}")
        controller_log = attempt / "controller.stdout.log"
        started = time.monotonic()
        with controller_log.open("w") as output:
            output.write("EXPLORATORY — compare_diloco stdout/stderr\n")
            output.flush()
            process = subprocess.Popen(
                command,
                cwd=str(REPO),
                stdout=output,
                stderr=subprocess.STDOUT,
                env=dict(os.environ),
            )
            return_code = process.wait()
        end_record = {
            "label": LABEL,
            "schema": "yeto_e1x_lane_b_attempt_end_v1",
            "attempt_id": attempt_id,
            "cell_id": cell_id,
            "attempt_number": 1,
            "node": node,
            "gpu": gpu,
            "end_utc": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "process_return_code": return_code,
        }
        write_json_atomic(attempt / "attempt-end.json", end_record)
        evidence = validate_and_bank(cell, attempt, node, gpu, return_code)
        evidence["attempt_id"] = attempt_id
        evidence["attempt_start_sha256"] = sha256_file(attempt / "attempt-start.json")
        evidence["attempt_end_sha256"] = sha256_file(attempt / "attempt-end.json")
        if evidence["status"] == "COMPLETED":
            compact_successful_attempt(attempt, evidence)
            completed += 1
        else:
            failures += 1
        write_json_atomic(evidence_path, evidence)
        emit(
            worker_log,
            f"END {cell_id} status={evidence['status']} wall={end_record['wall_seconds']:.1f}s",
        )
        write_json_atomic(
            status_path,
            {
                "label": LABEL,
                "schema": "yeto_e1x_lane_b_slot_status_v1",
                "node": node,
                "gpu": gpu,
                "state": "BETWEEN_CELLS",
                "last_cell_id": cell_id,
                "last_status": evidence["status"],
                "queue_index": queue_index + 1,
                "queue_total": len(queue["cells"]),
                "completed": completed,
                "failures": failures,
                "updated_at_utc": utc_now(),
            },
        )

    write_json_atomic(
        status_path,
        {
            "label": LABEL,
            "schema": "yeto_e1x_lane_b_slot_status_v1",
            "node": node,
            "gpu": gpu,
            "state": "DRAINED",
            "queue_total": len(queue["cells"]),
            "completed": completed,
            "failures": failures,
            "updated_at_utc": utc_now(),
        },
    )
    emit(worker_log, f"DRAINED completed={completed} failures={failures}")
    return 0 if failures == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight(args.queue, args.manifest)
        return 0
    return run_queue(args.queue, args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
