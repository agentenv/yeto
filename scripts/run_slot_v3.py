#!/usr/bin/env python3
"""Preflight or drain one preregistered outer-muP v3 (finite-horizon) GPU work queue.

Identical to run_slot_v2.py except for the registered v3 adaptations:
  1. v3 executes AT the registration commit (the --outer-bias-correction build
     requires it), so instead of a hardcoded execution-commit constant this
     controller binds the raw sha256 of both v3 contract files as constants and
     requires manifest.source.git_commit == manifest.registration.git_commit ==
     the node repo HEAD (clean).
  2. Per-cell recorded-git-commit evidence is checked against the cell's own
     source_git_commit field, hash-bound inside the manifest.
  3. All results live under /root/yeto-results-v3/ and slot status files under
     slots-v3/ so no v1/v2/pilot controller record is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


TELEMETRY_SCHEMA = "yeto_rho_telemetry_v1"
CONTRACT_JSON_SHA256 = "103b2035d048f4744f9a8aa8cdc00f7478498a46cbcf53f4b2af7d425bf948d5"
CONTRACT_MD_SHA256 = "9494833b5d0d330e3bb77a4023fbcf2e1e75e2d3452420dfae94c9e3b8698347"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: malformed JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: row is not an object")
            rows.append(row)
    return rows


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def gpu_inventory() -> dict[int, dict[str, str]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    inventory = {}
    for line in result.stdout.splitlines():
        index, gpu_uuid, name = [part.strip() for part in line.split(",", 2)]
        inventory[int(index)] = {"uuid": gpu_uuid, "name": name}
    return inventory


def verify_preflight(manifest_path: Path, node_label: str, proof_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    errors = []
    if manifest.get("schema") != "yeto_outer_mup_day1_launch_manifest_v1":
        errors.append("launch manifest schema mismatch")
    if manifest.get("manifest_variant") != "v3_finitehorizon":
        errors.append("launch manifest is not the v3 finite-horizon variant")
    if manifest.get("status") != "AUTHORIZED":
        errors.append("launch manifest is not authorized")
    expected_manifest_sha = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if expected_manifest_sha.is_file():
        expected = expected_manifest_sha.read_text().split()[0]
        if sha256_file(manifest_path) != expected:
            errors.append("launch manifest sidecar hash mismatch")
    repo = Path("/root/yeto")
    try:
        if git(repo, "rev-parse", "HEAD") != manifest["source"]["git_commit"]:
            errors.append("node Git commit mismatch")
        if git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
            errors.append("node Git worktree is dirty")
        for ancestor in ("a49b528", "3f6c5c1"):
            if subprocess.run(
                ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, "HEAD"]
            ).returncode:
                errors.append(f"required E0 commit is not an ancestor: {ancestor}")
    except Exception as exc:
        errors.append(f"Git preflight failed: {exc}")
    if manifest["source"].get("git_commit") != manifest.get("registration", {}).get(
        "git_commit"
    ):
        errors.append("v3 must execute at the registration commit")
    contract_constants = {"json": CONTRACT_JSON_SHA256, "md": CONTRACT_MD_SHA256}
    for kind in ("json", "md"):
        item = manifest["contract"]
        if item.get(f"{kind}_sha256") != contract_constants[kind]:
            errors.append(f"manifest contract {kind} hash differs from controller constant")
        path = manifest_path.parent / Path(item[f"{kind}_path"]).name
        if not path.is_file() or sha256_file(path) != contract_constants[kind]:
            errors.append(f"contract {kind} hash mismatch")
    models = [manifest["inputs"]["model"]]
    models.extend(manifest["inputs"].get("probe_models", {}).values())
    seen_models = set()
    for model in models:
        model_root = Path(model["path"])
        if str(model_root) in seen_models:
            continue
        seen_models.add(str(model_root))
        for name, record in model["files"].items():
            path = model_root / name
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                errors.append(f"model file mismatch: {path}")
    for seed, record in manifest["inputs"]["seeds"].items():
        for label in ("train", "eval", "audit", "split_provenance"):
            item = record["files"][label]
            path = Path(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                errors.append(f"seed {seed} input mismatch: {path}")
    for cell in manifest["cells"]:
        if canonical_sha256(cell["command"]) != cell["command_hash"]:
            errors.append(f"initial command hash mismatch: {cell['cell_id']}")
        if "--require-distinct-learner-gpu-uuids" in cell["command"]:
            errors.append(f"forbidden H200 flag in {cell['cell_id']}")
        if cell.get("cell_kind", "training") == "rho_probe":
            if not any(str(part).endswith("scripts/rho_probe.py") for part in cell["command"]):
                errors.append(f"rho probe script missing from {cell['cell_id']}")
            if "--outer-rounds" not in cell["command"] or "--gpu-offset" not in cell["command"]:
                errors.append(f"rho probe work/GPU binding missing from {cell['cell_id']}")
        elif "--rho-telemetry" not in cell["command"]:
            errors.append(f"rho telemetry missing from {cell['cell_id']}")
        for retry in cell["registered_retry_commands"]:
            if canonical_sha256(retry["command"]) != retry["command_hash"]:
                errors.append(f"retry command hash mismatch: {cell['cell_id']}")
    inventory = gpu_inventory()
    if set(inventory) != set(range(8)):
        errors.append(f"expected GPU indices 0..7, got {sorted(inventory)}")
    if any("H200" not in item["name"].upper() for item in inventory.values()):
        errors.append("non-H200 device in node inventory")
    proof = {
        "schema": "yeto_outer_mup_node_authority_v1",
        "node": node_label,
        "checked_at_utc": utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "git_commit": manifest["source"]["git_commit"],
        "gpu_inventory": inventory,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    write_json_atomic(proof_path, proof)
    if errors:
        raise SystemExit("; ".join(errors))
    return proof


def validate_cell(cell: dict, attempt_root: Path, command: list[str]) -> dict:
    expected = cell["expected"]
    arm_name = cell.get("arm_name", f"m{cell.get('m', 4)}")
    work = attempt_root / "work" / arm_name
    report = attempt_root / "report"
    failures: list[str] = []
    artifacts: dict[str, dict[str, object]] = {}

    def require_file(label: str, path: Path, *, hash_file: bool = True) -> bool:
        if not path.is_file():
            failures.append(f"missing {label}: {path}")
            return False
        item: dict[str, object] = {"path": str(path), "bytes": path.stat().st_size}
        if hash_file:
            item["sha256"] = sha256_file(path)
        artifacts[label] = item
        return True

    telemetry_path = work / "rho-telemetry.jsonl"
    if require_file("rho_telemetry", telemetry_path):
        try:
            rows = read_jsonl(telemetry_path)
            if len(rows) != expected["telemetry_rows"]:
                failures.append(f"telemetry rows {len(rows)} != {expected['telemetry_rows']}")
            for index, row in enumerate(rows, 1):
                if row.get("schema") != TELEMETRY_SCHEMA:
                    failures.append(f"telemetry schema mismatch at row {index}")
                    break
                if row.get("outer_step") != index:
                    failures.append(f"telemetry outer_step mismatch at row {index}")
                    break
                if row.get("fragment") != (index - 1) % 4:
                    failures.append(f"telemetry fragment mismatch at row {index}")
                    break
                norm = row.get("pseudo_gradient", {}).get("l2_norm")
                if not isinstance(norm, (int, float)) or not math.isfinite(norm):
                    failures.append(f"nonfinite telemetry norm at row {index}")
                    break
                cross = row.get("cross_worker", {})
                if cross.get("worker_count") != expected["learner_count"]:
                    failures.append(f"telemetry worker count mismatch at row {index}")
                    break
                fragment_round = row.get("fragment_round")
                autocorrelation = row.get("autocorrelation", {})
                for lag in range(1, 5):
                    value = autocorrelation.get(f"lag_{lag}")
                    should_exist = isinstance(fragment_round, int) and fragment_round > lag
                    if should_exist and (
                        not isinstance(value, (int, float)) or not math.isfinite(value)
                    ):
                        failures.append(
                            f"lag {lag} undefined after sufficient history at row {index}"
                        )
                        break
        except Exception as exc:
            failures.append(f"telemetry validation error: {exc}")

    tape_path = work / "tape.jsonl"
    if require_file("event_tape", tape_path):
        try:
            tape = read_jsonl(tape_path)
            if len(tape) != expected["outer_steps"]:
                failures.append(f"outer commits {len(tape)} != {expected['outer_steps']}")
            for index, row in enumerate(tape, 1):
                if row.get("step") != index:
                    failures.append(f"event tape step mismatch at {index}")
                    break
                responders = row.get("responders")
                if not isinstance(responders, list) or sorted(
                    item.get("id") for item in responders
                ) != list(range(expected["learner_count"])):
                    failures.append(f"strict quorum responder mismatch at {index}")
                    break
                if any(item.get("c_steps") != cell["h"] for item in responders):
                    failures.append(f"fixed H step mismatch at outer step {index}")
                    break
                if any(item.get("c_tokens") != cell["h"] * 128 for item in responders):
                    failures.append(f"fixed H token mismatch at outer step {index}")
                    break
        except Exception as exc:
            failures.append(f"event tape validation error: {exc}")

    require_file("checkpoint", work / "state.ckpt")
    require_file("per_example_loss", report / "per-example-loss" / f"{arm_name}.jsonl")
    require_file("eval_rows", report / "eval-provenance" / "eval_rows.jsonl")
    require_file("eval_provenance", report / "eval-provenance" / "eval_provenance.json")
    require_file("barrier_registry", report / "barrier-version-trace.json")
    require_file("results", report / "results.jsonl")
    require_file("acquisition_state", report / "acquisition-state.json")
    require_file("recorded_command", attempt_root / "command.sh")
    require_file("recorded_git_commit", attempt_root / "git_commit.txt")
    require_file("recorded_git_diff", attempt_root / "git_diff.patch")

    command_path = attempt_root / "command.sh"
    if command_path.is_file() and command_path.read_text() != shlex.join(command[1:]) + "\n":
        failures.append("recorded command differs from registered argv")
    commit_path = attempt_root / "git_commit.txt"
    if commit_path.is_file() and commit_path.read_text().strip() != cell["source_git_commit"]:
        failures.append("recorded Git commit mismatch")
    diff_path = attempt_root / "git_diff.patch"
    if diff_path.is_file() and diff_path.read_text():
        failures.append("recorded Git diff is nonempty")

    results_path = report / "results.jsonl"
    scientific_divergence = False
    if results_path.is_file():
        try:
            results = read_jsonl(results_path)
            if len(results) != 1:
                failures.append(f"expected one result row, found {len(results)}")
            else:
                result = results[0]
                loss = result.get("eval_loss")
                if not isinstance(loss, (int, float)) or not math.isfinite(loss):
                    scientific_divergence = True
                    failures.append("endpoint evaluation loss is nonfinite")
                if result.get("eval_rows") != expected["eval_rows"]:
                    failures.append("evaluation row count mismatch")
                if result.get("learner_exit_codes") != [0] * expected["learner_count"]:
                    failures.append("learner exit code evidence mismatch")
                if result.get("syncer_exit_code") != 0:
                    failures.append("syncer exit code evidence mismatch")
        except Exception as exc:
            failures.append(f"result validation error: {exc}")
    eval_rows_path = report / "eval-provenance" / "eval_rows.jsonl"
    if eval_rows_path.is_file():
        try:
            if len(read_jsonl(eval_rows_path)) != expected["eval_rows"]:
                failures.append("row-aligned evaluation rows mismatch")
        except Exception as exc:
            failures.append(f"evaluation row validation error: {exc}")

    for learner in range(expected["learner_count"]):
        log_path = work / f"learner-{learner}.log"
        if require_file(f"learner_{learner}_log", log_path):
            text = log_path.read_text(errors="replace")
            match = re.search(r"inner loop done at local_step=(\d+) global_step=(\d+)", text)
            if not match:
                failures.append(f"learner {learner} lacks completion line")
            elif int(match.group(1)) != expected["learner_steps_per_learner"]:
                failures.append(f"learner {learner} step count mismatch")
            elif int(match.group(2)) != expected["outer_steps"]:
                failures.append(f"learner {learner} outer step count mismatch")

    state_path = report / "acquisition-state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
            if state.get("phase") != "endpoint_recorded" or state.get("scientific_status") != "COMPLETED":
                failures.append("acquisition state is not completed endpoint evidence")
        except Exception as exc:
            failures.append(f"acquisition state validation error: {exc}")

    status = "COMPLETED"
    if scientific_divergence:
        status = "SCIENTIFIC_DIVERGENCE"
    elif failures:
        status = "INVALID_WORK"
    return {
        "schema": "yeto_outer_mup_cell_evidence_v1",
        "cell_id": cell["cell_id"],
        "validated_at_utc": utc_now(),
        "status": status,
        "failures": failures,
        "expected": expected,
        "observed_artifacts": artifacts,
        "seed": cell["seed"],
        "training_seed": cell["training_seed"],
        "command_hash": canonical_sha256(command),
    }


def validate_probe(cell: dict, attempt_root: Path, command: list[str]) -> dict:
    expected = cell["expected"]
    arm_name = f"m{cell['m']}"
    report_path = attempt_root / "probe-report.json"
    run_root = attempt_root / "run"
    work = run_root / "work" / arm_name
    compare_report = run_root / "compare-report"
    failures: list[str] = []
    artifacts: dict[str, dict[str, object]] = {}

    def require_file(label: str, path: Path) -> bool:
        if not path.is_file():
            failures.append(f"missing {label}: {path}")
            return False
        artifacts[label] = {
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        return True

    report = {}
    if require_file("probe_report", report_path):
        try:
            report = json.loads(report_path.read_text())
            if report.get("schema") != "yeto_rho_probe_report_v1" or report.get("status") != "complete":
                failures.append("probe report schema/status mismatch")
            config = report.get("config", {})
            for key in ("scale", "h", "m", "mu", "seed"):
                if config.get(key) != cell[key]:
                    failures.append(f"probe config {key} mismatch")
            if config.get("eta") != cell["eta"]:
                failures.append("probe config eta mismatch")
            if config.get("outer_rounds") != expected["outer_steps"]:
                failures.append("probe outer-round count mismatch")
            if config.get("gpu_offset") != cell["initial_assignment"]["gpu"]:
                failures.append("probe GPU offset mismatch")
            work_evidence = report.get("work_evidence", {})
            if work_evidence != {
                "full_registered_outer_rounds": True,
                "telemetry_present": True,
                "all_lags_defined": True,
            }:
                failures.append("probe report work evidence is incomplete")
        except Exception as exc:
            failures.append(f"probe report validation error: {exc}")

    telemetry_path = work / "rho-telemetry.jsonl"
    if require_file("rho_telemetry", telemetry_path):
        try:
            rows = read_jsonl(telemetry_path)
            if len(rows) != expected["telemetry_rows"]:
                failures.append("probe telemetry row count mismatch")
            if [row.get("outer_step") for row in rows] != list(
                range(1, expected["outer_steps"] + 1)
            ):
                failures.append("probe telemetry outer steps are not exact")
            if any(row.get("schema") != TELEMETRY_SCHEMA for row in rows):
                failures.append("probe telemetry schema mismatch")
            if report and report.get("telemetry", {}).get("sha256") != sha256_file(telemetry_path):
                failures.append("probe report telemetry hash mismatch")
        except Exception as exc:
            failures.append(f"probe telemetry validation error: {exc}")

    tape_path = work / "tape.jsonl"
    if require_file("event_tape", tape_path):
        try:
            tape = read_jsonl(tape_path)
            if len(tape) != expected["outer_steps"]:
                failures.append("probe event tape count mismatch")
            for index, row in enumerate(tape, 1):
                responders = row.get("responders", [])
                if row.get("step") != index or len(responders) != cell["m"]:
                    failures.append(f"probe quorum/step mismatch at {index}")
                    break
                if any(item.get("c_steps") != cell["h"] for item in responders):
                    failures.append(f"probe fixed H mismatch at {index}")
                    break
        except Exception as exc:
            failures.append(f"probe event tape validation error: {exc}")
    require_file("checkpoint", work / "state.ckpt")
    require_file("compare_results", compare_report / "results.jsonl")
    require_file("acquisition_state", compare_report / "acquisition-state.json")
    require_file("recorded_compare_command", run_root / "command.sh")
    require_file("recorded_git_commit", run_root / "git_commit.txt")
    require_file("recorded_git_diff", run_root / "git_diff.patch")

    results_path = compare_report / "results.jsonl"
    if results_path.is_file():
        try:
            results = read_jsonl(results_path)
            if len(results) != 1:
                failures.append("probe compare result count mismatch")
            else:
                result = results[0]
                if result.get("checkpoint_only") is not True:
                    failures.append("probe compare was not checkpoint-only")
                if result.get("eval_loss") is not None or result.get("evaluation_role") != "none":
                    failures.append("probe exposed endpoint evaluation loss")
                if result.get("learner_exit_codes") != [0] * cell["m"]:
                    failures.append("probe learner exit codes mismatch")
                if result.get("syncer_exit_code") != 0:
                    failures.append("probe syncer exit code mismatch")
        except Exception as exc:
            failures.append(f"probe result validation error: {exc}")
    if (compare_report / "per-example-loss").exists():
        failures.append("probe emitted forbidden endpoint per-example loss")
    state_path = compare_report / "acquisition-state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text())
            if state.get("phase") != "checkpoint_recorded":
                failures.append("probe acquisition state is not checkpoint-only")
        except Exception as exc:
            failures.append(f"probe acquisition state error: {exc}")
    for learner in range(cell["m"]):
        log_path = work / f"learner-{learner}.log"
        if require_file(f"learner_{learner}_log", log_path):
            text = log_path.read_text(errors="replace")
            match = re.search(r"inner loop done at local_step=(\d+) global_step=(\d+)", text)
            if not match or int(match.group(1)) != expected["learner_steps_per_learner"]:
                failures.append(f"probe learner {learner} step count mismatch")
            elif int(match.group(2)) != expected["outer_steps"]:
                failures.append(f"probe learner {learner} outer count mismatch")
    commit_path = run_root / "git_commit.txt"
    if commit_path.is_file() and commit_path.read_text().strip() != cell["source_git_commit"]:
        failures.append("probe recorded Git commit mismatch")
    diff_path = run_root / "git_diff.patch"
    if diff_path.is_file() and diff_path.read_text():
        failures.append("probe recorded Git diff is nonempty")
    return {
        "schema": "yeto_outer_mup_probe_evidence_v1",
        "cell_id": cell["cell_id"],
        "validated_at_utc": utc_now(),
        "status": "COMPLETED" if not failures else "INVALID_WORK",
        "failures": failures,
        "expected": expected,
        "observed_artifacts": artifacts,
        "seed": cell["seed"],
        "training_seed": cell["training_seed"],
        "command_hash": canonical_sha256(command),
        "verification_loss_seen": False,
    }


def run_queue(manifest_path: Path, node_label: str, gpu: int, proof_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text())
    if manifest["contract"].get("json_sha256") != CONTRACT_JSON_SHA256:
        raise SystemExit("manifest does not bind the registered v3 contract")
    proof = json.loads(proof_path.read_text())
    if proof.get("status") != "PASS" or proof.get("node") != node_label:
        raise SystemExit("node authority proof is missing or invalid")
    if proof.get("manifest_sha256") != sha256_file(manifest_path):
        raise SystemExit("node authority proof binds a different manifest")
    queue = [
        cell for cell in manifest["cells"]
        if cell["initial_assignment"] == {"node": node_label, "gpu": gpu}
    ]
    queue.sort(key=lambda cell: cell["randomization"]["slot_queue_index"])
    status_path = Path(f"/root/yeto-results-v3/_controller/slots-v3/{node_label}-gpu{gpu}.json")
    inventory = gpu_inventory()
    completed = 0
    failures = 0
    for queue_index, cell in enumerate(queue):
        attempt_root = Path(f"/root/yeto-results-v3/{cell['cell_id']}/attempt-1")
        evidence_path = attempt_root / "evidence.json"
        if evidence_path.is_file():
            evidence = json.loads(evidence_path.read_text())
            if evidence.get("status") == "COMPLETED":
                completed += 1
                continue
            raise SystemExit(f"existing noncompleted attempt requires controller review: {cell['cell_id']}")
        if attempt_root.exists():
            raise SystemExit(f"refusing to overwrite existing attempt: {attempt_root}")
        attempt_root.mkdir(parents=True)
        command = cell["command"]
        if canonical_sha256(command) != cell["command_hash"]:
            raise SystemExit(f"command hash changed for {cell['cell_id']}")
        attempt_id = f"{cell['cell_id']}-a1-{uuid.uuid4()}"
        start = {
            "schema": "yeto_outer_mup_attempt_start_v1",
            "attempt_id": attempt_id,
            "attempt_number": 1,
            "cell_id": cell["cell_id"],
            "node": node_label,
            "gpu_index": gpu,
            "gpu_uuid": inventory[gpu]["uuid"],
            "gpu_name": inventory[gpu]["name"],
            "start_utc": utc_now(),
            "command": command,
            "command_hash": cell["command_hash"],
            "seed": cell["seed"],
            "training_seed": cell["training_seed"],
            "git_commit": manifest["source"]["git_commit"],
        }
        write_json_atomic(attempt_root / "attempt-start.json", start)
        write_json_atomic(status_path, {
            "schema": "yeto_outer_mup_slot_status_v1", "node": node_label,
            "gpu": gpu, "gpu_uuid": inventory[gpu]["uuid"], "state": "RUNNING",
            "cell_id": cell["cell_id"], "queue_index": queue_index,
            "queue_total": len(queue), "updated_at_utc": utc_now(),
        })
        log_path = attempt_root / "controller.stdout.log"
        start_monotonic = time.monotonic()
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd="/root/yeto",
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(os.environ),
            )
            return_code = process.wait()
        end = {
            "schema": "yeto_outer_mup_attempt_end_v1",
            "attempt_id": attempt_id,
            "attempt_number": 1,
            "cell_id": cell["cell_id"],
            "node": node_label,
            "gpu_index": gpu,
            "end_utc": utc_now(),
            "wall_seconds": time.monotonic() - start_monotonic,
            "process_return_code": return_code,
        }
        write_json_atomic(attempt_root / "attempt-end.json", end)
        if return_code == 0:
            if cell.get("cell_kind", "training") == "rho_probe":
                evidence = validate_probe(cell, attempt_root, command)
            else:
                evidence = validate_cell(cell, attempt_root, command)
        else:
            evidence = {
                "schema": "yeto_outer_mup_cell_evidence_v1",
                "cell_id": cell["cell_id"],
                "validated_at_utc": utc_now(),
                "status": "INFRA_FAILURE",
                "failures": [f"scientific process exited {return_code} before valid completion"],
                "seed": cell["seed"],
                "training_seed": cell["training_seed"],
                "command_hash": cell["command_hash"],
            }
        evidence["attempt_id"] = attempt_id
        evidence["attempt_start_sha256"] = sha256_file(attempt_root / "attempt-start.json")
        evidence["attempt_end_sha256"] = sha256_file(attempt_root / "attempt-end.json")
        write_json_atomic(evidence_path, evidence)
        if evidence["status"] == "COMPLETED":
            completed += 1
        else:
            failures += 1
        write_json_atomic(status_path, {
            "schema": "yeto_outer_mup_slot_status_v1", "node": node_label,
            "gpu": gpu, "gpu_uuid": inventory[gpu]["uuid"], "state": "BETWEEN_CELLS",
            "last_cell_id": cell["cell_id"], "last_status": evidence["status"],
            "completed": completed, "failures": failures,
            "queue_index": queue_index + 1, "queue_total": len(queue),
            "updated_at_utc": utc_now(),
        })
    write_json_atomic(status_path, {
        "schema": "yeto_outer_mup_slot_status_v1", "node": node_label,
        "gpu": gpu, "gpu_uuid": inventory[gpu]["uuid"], "state": "DRAINED",
        "completed": completed, "failures": failures, "queue_total": len(queue),
        "updated_at_utc": utc_now(),
    })
    return 0 if failures == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--node-label", required=True, choices=("h200-n1", "h200-n2"))
    parser.add_argument("--gpu", type=int, choices=range(8))
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        verify_preflight(args.manifest, args.node_label, args.proof)
        print(json.dumps({"node": args.node_label, "status": "PASS"}, sort_keys=True))
        return 0
    if args.gpu is None:
        parser.error("--gpu is required unless --preflight is used")
    return run_queue(args.manifest, args.node_label, args.gpu, args.proof)


if __name__ == "__main__":
    raise SystemExit(main())
