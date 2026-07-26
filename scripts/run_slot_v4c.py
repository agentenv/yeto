#!/usr/bin/env python3
"""Preflight or drain one hash-bound v4c seed-power GPU queue."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

try:
    import run_slot_v4 as base
except ModuleNotFoundError:  # package import in tests
    from scripts import run_slot_v4 as base


CONTRACT_JSON_SHA256 = "76ed8e7c8b72a77a7d09b77a5cb661d7a0c3be074afdb2f5db4beb86c6b0d899"
CONTRACT_MD_SHA256 = "3cfacc81a4efa7093a753b868142226093a1792f4f94d2de899608a5329d5a37"
ANALYZER_SHA256 = "b8e5470d2b512f487948413104a1783fe45adbbe68871f61873d3a9bae73cf27"
ANALYZER_DEPENDENCY_SHA256 = (
    "9a6bd4110b55a5487501ab4b32eef205854400dd8735c83c88dd7580951cbab5"
)
V4_MANIFEST_SHA256 = "150dab251f29ab191aca4bfa8297950f3f22167f5949d1c8795e467706d2fb1e"
V4B_MANIFEST_SHA256 = "f2abf80d975572dde33ee2c750c1fb91598df8bfea5a78696bdc2c5d3608b55b"
G4B_READOUT_SHA256 = "d58a05c46396d94786c6bcdcffa4f9c72abcc47036652a5260ee87256446be97"
EXECUTION_REPO = Path("/root/yeto-v4c")
RESULT_LINK = Path("/root/yeto-results-v4c")
RESULT_TARGET = Path("/data/yeto-results-v4c")
MIN_FREE_BYTES = 1_000_000_000_000
MAX_PROOF_AGE_SECONDS = 300


def command_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def matching_processes(pattern: str) -> list[dict]:
    result = subprocess.run(
        ["pgrep", "-af", pattern], capture_output=True, text=True
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or f"pgrep failed for {pattern}")
    rows = []
    for line in result.stdout.splitlines():
        pid, _, command = line.partition(" ")
        if pid.isdigit():
            rows.append({"pid": int(pid), "command": command})
    return rows


def gpu_compute_processes(
    gpu: int, inventory: dict[int, dict[str, str]] | None = None
) -> list[dict]:
    if inventory is None:
        inventory = base.gpu_inventory()
    gpu_uuid = inventory[gpu]["uuid"]
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi process query failed")
    processes = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) == 4 and parts[0] == gpu_uuid:
            processes.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": int(parts[1]),
                    "process_name": parts[2],
                    "used_memory_mib": int(parts[3]),
                }
            )
    return processes


def gpu_sample(gpu: int) -> dict:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    return {"raw": result.stdout.strip(), "return_code": result.returncode}


def storage_proof() -> tuple[dict, list[str]]:
    errors = []
    proof = {
        "result_link": str(RESULT_LINK),
        "result_target_expected": str(RESULT_TARGET),
        "is_symlink": RESULT_LINK.is_symlink(),
    }
    try:
        resolved = RESULT_LINK.resolve(strict=True)
        proof["result_target_resolved"] = str(resolved)
        proof["same_device_as_data"] = resolved.stat().st_dev == Path("/data").stat().st_dev
        usage = shutil.disk_usage(resolved)
        proof["filesystem_total_bytes"] = usage.total
        proof["filesystem_used_bytes"] = usage.used
        proof["filesystem_free_bytes"] = usage.free
        findmnt = subprocess.run(
            ["findmnt", "-n", "-o", "SOURCE,FSTYPE,TARGET", "--target", str(resolved)],
            capture_output=True,
            text=True,
        )
        proof["findmnt"] = findmnt.stdout.strip()
        if not RESULT_LINK.is_symlink():
            errors.append("v4c results path is not a symlink")
        if resolved != RESULT_TARGET:
            errors.append(f"v4c results resolve to {resolved}, not {RESULT_TARGET}")
        if not proof["same_device_as_data"]:
            errors.append("v4c results are not on the /data filesystem")
        if usage.free < MIN_FREE_BYTES:
            errors.append(f"v4c results filesystem has only {usage.free} bytes free")
    except Exception as exc:
        errors.append(f"storage preflight failed: {exc}")
    return proof, errors


def verify_registered_files(manifest: dict) -> list[str]:
    errors = []
    constants = {
        "json_sha256": CONTRACT_JSON_SHA256,
        "md_sha256": CONTRACT_MD_SHA256,
        "analyzer_sha256": ANALYZER_SHA256,
        "analyzer_dependency_sha256": ANALYZER_DEPENDENCY_SHA256,
    }
    paths = {
        "json_sha256": EXECUTION_REPO
        / "experiment-specs/outer-mup-v4c-seedpower-prereg.json",
        "md_sha256": EXECUTION_REPO
        / "experiment-specs/outer-mup-v4c-seedpower-prereg.md",
        "analyzer_sha256": EXECUTION_REPO / "scripts/analyze_v4c.py",
        "analyzer_dependency_sha256": EXECUTION_REPO / "scripts/analyze_v4b.py",
    }
    contract = manifest.get("contract", {})
    for field, expected in constants.items():
        if contract.get(field) != expected:
            errors.append(f"manifest contract {field} differs from controller constant")
        path = paths[field]
        if not path.is_file() or base.sha256_file(path) != expected:
            errors.append(f"registered file hash mismatch: {path}")
    return errors


def verify_base_evidence(manifest: dict) -> list[str]:
    errors = []
    expected = {
        "v4_manifest_sha256": V4_MANIFEST_SHA256,
        "v4b_manifest_sha256": V4B_MANIFEST_SHA256,
        "g4b_readout_sha256": G4B_READOUT_SHA256,
    }
    bindings = manifest.get("base_evidence", {})
    path_fields = {
        "v4_manifest_sha256": "v4_manifest_path",
        "v4b_manifest_sha256": "v4b_manifest_path",
        "g4b_readout_sha256": "g4b_readout_path",
    }
    for hash_field, expected_hash in expected.items():
        if bindings.get(hash_field) != expected_hash:
            errors.append(f"base binding {hash_field} mismatch")
        path = Path(bindings.get(path_fields[hash_field], ""))
        if not path.is_file() or base.sha256_file(path) != expected_hash:
            errors.append(f"base evidence is absent or mismatched: {path}")
    return errors


def verify_cells(manifest: dict, node_label: str) -> list[str]:
    errors = []
    cells = manifest.get("cells", [])
    seen = set()
    coordinate_counts = {}
    for cell in cells:
        cell_id = cell.get("cell_id", "<missing>")
        if cell_id in seen:
            errors.append(f"duplicate cell id: {cell_id}")
        seen.add(cell_id)
        coordinate = (cell.get("s"), cell.get("mu"), cell.get("eta"))
        coordinate_counts[coordinate] = coordinate_counts.get(coordinate, 0) + 1
        command = cell.get("command", [])
        if base.canonical_sha256(command) != cell.get("command_hash"):
            errors.append(f"initial command hash mismatch: {cell_id}")
        retries = cell.get("registered_retry_commands", [])
        if (
            len(retries) != 1
            or base.canonical_sha256(retries[0].get("command", []))
            != retries[0].get("command_hash")
        ):
            errors.append(f"retry command hash mismatch: {cell_id}")
        assignment = cell.get("assignment", {})
        if assignment.get("node") not in ("h200-n1", "h200-n2"):
            errors.append(f"invalid node assignment: {cell_id}")
        if assignment.get("gpu") not in range(8):
            errors.append(f"invalid GPU assignment: {cell_id}")
        if cell.get("seed") not in (541, 547):
            errors.append(f"unregistered seed: {cell_id}")
        if cell.get("training_seed") != int(f"{cell.get('seed')}{cell.get('seed')}"):
            errors.append(f"training seed mismatch: {cell_id}")
        if "--rho-telemetry" not in command or "--outer-bias-correction" in command:
            errors.append(f"telemetry/bias-correction mismatch: {cell_id}")
        if command_value(command, "--gpu-offset") != str(assignment.get("gpu")):
            errors.append(f"GPU command binding mismatch: {cell_id}")
        if command_value(command, "--outer-momentum") != str(cell.get("mu")):
            errors.append(f"momentum binding mismatch: {cell_id}")
        if not command or command[1] != str(EXECUTION_REPO / "scripts/compare_diloco.py"):
            errors.append(f"execution checkout mismatch: {cell_id}")
        for flag in ("--work-dir", "--report-dir"):
            value = command_value(command, flag)
            if value is None or not value.startswith(str(RESULT_LINK) + "/"):
                errors.append(f"{flag} escapes v4c results: {cell_id}")
    if len(cells) != 44 or len(seen) != 44:
        errors.append("manifest is not the complete 44-cell stage")
    if len(coordinate_counts) != 22 or set(coordinate_counts.values()) != {2}:
        errors.append(f"eta/additional-seed balance failure: {coordinate_counts}")
    if sum(cell.get("s") == 10240 for cell in cells) != 24:
        errors.append("long-cell count is not 24")
    for gpu in range(8):
        queue = sorted(
            [
                cell
                for cell in cells
                if cell.get("assignment") == {"node": node_label, "gpu": gpu}
            ],
            key=lambda cell: cell.get("slot_queue_index", -1),
        )
        if not queue or queue[0].get("s") != 10240:
            errors.append(f"S10240 is not first on {node_label}/gpu{gpu}")
        costs = [cell.get("estimated_cost_units") for cell in queue]
        if costs != sorted(costs, reverse=True):
            errors.append(f"queue is not longest-first on {node_label}/gpu{gpu}")
    return errors


def verify_preflight(
    manifest_path: Path,
    node_label: str,
    target_gpus: list[int],
    proof_path: Path,
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    errors = []
    if (
        manifest.get("schema")
        != "yeto_outer_mup_v4c_seedpower_launch_manifest_v1"
        or manifest.get("manifest_variant") != "v4c_five_seed_combined_grids"
        or manifest.get("stage") != "V4C_SEED_POWER"
        or manifest.get("status") != "AUTHORIZED"
        or len(manifest.get("cells", [])) != 44
    ):
        errors.append("launch manifest identity mismatch")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != base.sha256_file(
        manifest_path
    ):
        errors.append("manifest sidecar is absent or mismatched")
    expected_target = list(range(6)) if max(target_gpus, default=-1) < 6 else [6, 7]
    if sorted(set(target_gpus)) != expected_target:
        errors.append(
            f"target GPUs must be exactly {expected_target}, got {target_gpus}"
        )
    try:
        head = base.git(EXECUTION_REPO, "rev-parse", "HEAD")
        if head != manifest.get("source", {}).get("git_commit"):
            errors.append("isolated checkout commit mismatch")
        if base.git(EXECUTION_REPO, "status", "--porcelain=v1", "--untracked-files=all"):
            errors.append("isolated execution checkout is dirty")
        if manifest.get("registration", {}).get("git_commit") != head:
            errors.append("v4c is not executing at its registration commit")
    except Exception as exc:
        errors.append(f"Git preflight failed: {exc}")
    errors.extend(verify_registered_files(manifest))
    errors.extend(verify_base_evidence(manifest))
    errors.extend(verify_cells(manifest, node_label))
    for record in manifest.get("inputs", {}).get("files", []):
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            errors.append(f"input size mismatch: {path}")
        elif base.sha256_file(path) != record["sha256"]:
            errors.append(f"input hash mismatch: {path}")

    inventory = base.gpu_inventory()
    if set(inventory) != set(range(8)):
        errors.append(f"expected GPU indices 0..7, got {sorted(inventory)}")
    if any("H200" not in item["name"].upper() for item in inventory.values()):
        errors.append("non-H200 device in node inventory")
    target_state = {}
    for gpu in target_gpus:
        processes = gpu_compute_processes(gpu, inventory)
        sample = gpu_sample(gpu)
        target_state[str(gpu)] = {
            "compute_processes": processes,
            "sample": sample,
        }
        if processes:
            errors.append(f"target GPU {gpu} has compute processes")
        try:
            memory_used_mib = float(sample["raw"].split(",", 1)[0].strip())
            target_state[str(gpu)]["memory_used_mib"] = memory_used_mib
            if memory_used_mib > 16:
                errors.append(
                    f"target GPU {gpu} reports {memory_used_mib} MiB used"
                )
        except (KeyError, TypeError, ValueError):
            errors.append(f"target GPU {gpu} memory sample is invalid")
    v5b = matching_processes(
        "[r]un_slot_v5b.py|/root/yeto-v5b/scripts/compare_diloco.py|"
        "/root/yeto-v5b/syncer/|yeto-results-v5b"
    )
    v6 = matching_processes("[r]un_slot_v6.py|/root/yeto-v6/")
    if target_gpus == [6, 7] and v5b:
        errors.append("deferred claim requires every v5b process to drain")
    if v6:
        errors.append("v6 process exists before V4C DONE")
    storage, storage_errors = storage_proof()
    errors.extend(storage_errors)
    checked = time.time()
    proof = {
        "schema": "yeto_outer_mup_v4c_node_claim_v1",
        "node": node_label,
        "checked_at_utc": base.utc_now(),
        "checked_at_unix_s": checked,
        "manifest_path": str(manifest_path),
        "manifest_sha256": base.sha256_file(manifest_path),
        "git_commit": manifest.get("source", {}).get("git_commit"),
        "target_gpus": target_gpus,
        "target_gpu_state": target_state,
        "v5b_processes": v5b,
        "v6_processes": v6,
        "gpu_inventory": inventory,
        "storage": storage,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    base.write_json_atomic(proof_path, proof)
    if errors:
        raise SystemExit("; ".join(errors))
    return proof


def load_launch_authority(path: Path, manifest_path: Path) -> dict:
    authority = json.loads(path.read_text())
    errors = []
    if authority.get("schema") != "yeto_outer_mup_v4c_launch_authority_v1":
        errors.append("launch authority schema mismatch")
    if authority.get("status") != "AUTHORIZED":
        errors.append("launch authority is not authorized")
    if authority.get("manifest_sha256") != base.sha256_file(manifest_path):
        errors.append("launch authority binds another manifest")
    started = authority.get("wall_clock_start_unix_s")
    deadline = authority.get("hard_deadline_unix_s")
    if not isinstance(started, (int, float)) or not isinstance(deadline, (int, float)):
        errors.append("launch authority lacks numeric wall times")
    elif deadline - started != 43_200:
        errors.append("launch authority does not encode the registered 12h ceiling")
    if errors:
        raise SystemExit("; ".join(errors))
    return authority


def load_retry_authority(path: Path, manifest_path: Path, manifest: dict) -> set[str]:
    authority = json.loads(path.read_text())
    errors = []
    if authority.get("schema") != "yeto_outer_mup_v4c_retry_authority_v1":
        errors.append("retry authority schema mismatch")
    if authority.get("status") != "AUTHORIZED":
        errors.append("retry authority is not authorized")
    if authority.get("manifest_sha256") != base.sha256_file(manifest_path):
        errors.append("retry authority binds another manifest")
    if authority.get("reason") not in set(
        manifest.get("retry_contract", {}).get("allowed_reasons", [])
    ):
        errors.append("retry reason is not registered")
    groups = authority.get("retry_group_ids")
    if not isinstance(groups, list) or not groups or len(groups) != len(set(groups)):
        errors.append("retry groups are empty, duplicated, or malformed")
        groups = []
    known = {cell["retry_group_id"] for cell in manifest.get("cells", [])}
    if set(groups) - known:
        errors.append("retry authority contains an unknown group")
    if errors:
        raise SystemExit("; ".join(errors))
    return set(groups)


def run_queue(
    manifest_path: Path,
    node_label: str,
    gpu: int,
    proof_path: Path,
    launch_authority_path: Path,
    attempt_number: int,
    retry_authority_path: Path | None,
) -> int:
    proof = json.loads(proof_path.read_text())
    if (
        proof.get("schema") != "yeto_outer_mup_v4c_node_claim_v1"
        or proof.get("status") != "PASS"
        or proof.get("node") != node_label
        or proof.get("manifest_sha256") != base.sha256_file(manifest_path)
        or gpu not in proof.get("target_gpus", [])
    ):
        raise SystemExit("fresh target-GPU claim proof is missing or invalid")
    checked = proof.get("checked_at_unix_s")
    if not isinstance(checked, (int, float)) or not 0 <= time.time() - checked <= MAX_PROOF_AGE_SECONDS:
        raise SystemExit("target-GPU claim proof is stale or future-dated")
    if gpu_compute_processes(gpu):
        raise SystemExit(f"target GPU {gpu} became occupied after its proof")
    if gpu >= 6 and matching_processes(
        "[r]un_slot_v5b.py|/root/yeto-v5b/scripts/compare_diloco.py|"
        "/root/yeto-v5b/syncer/|yeto-results-v5b"
    ):
        raise SystemExit("v5b has not fully drained; deferred v4c claim forbidden")
    if matching_processes("[r]un_slot_v6.py|/root/yeto-v6/"):
        raise SystemExit("v6 process exists before V4C DONE")
    base.RESULT_LINK = RESULT_LINK
    base.load_launch_authority = load_launch_authority
    base.load_retry_authority = load_retry_authority
    return base.run_queue(
        manifest_path,
        node_label,
        gpu,
        proof_path,
        launch_authority_path,
        attempt_number,
        retry_authority_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--node-label", choices=("h200-n1", "h200-n2"), required=True)
    parser.add_argument("--target-gpu", type=int, choices=range(8), action="append")
    parser.add_argument("--gpu", type=int, choices=range(8))
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--launch-authority", type=Path)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--retry-authority", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        if not args.target_gpu:
            parser.error("--target-gpu is required for preflight")
        verify_preflight(
            args.manifest,
            args.node_label,
            sorted(set(args.target_gpu)),
            args.proof,
        )
        print(json.dumps({"node": args.node_label, "status": "PASS"}, sort_keys=True))
        return 0
    if args.gpu is None or args.launch_authority is None:
        parser.error("--gpu and --launch-authority are required for queue execution")
    return run_queue(
        args.manifest,
        args.node_label,
        args.gpu,
        args.proof,
        args.launch_authority,
        args.attempt,
        args.retry_authority,
    )


if __name__ == "__main__":
    raise SystemExit(main())
