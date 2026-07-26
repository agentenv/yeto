#!/usr/bin/env python3
"""Preflight or drain one hash-bound G4b extension GPU queue."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

try:
    import run_slot_v4 as base
except ModuleNotFoundError:  # package import in tests
    from scripts import run_slot_v4 as base


# Filled from the frozen registration artifacts before their registration commit.
CONTRACT_JSON_SHA256 = "df18b72125b08e0594b8e2f33b0460bffff2de68222e5ad453168e67d72bd026"
CONTRACT_MD_SHA256 = "18ccb8b1cbed5896cf893f84ba6f62f7a2b53fdc1f7db8f53810e2468cd556d4"
ANALYZER_SHA256 = "9a6bd4110b55a5487501ab4b32eef205854400dd8735c83c88dd7580951cbab5"
RESULT_LINK = Path("/root/yeto-results-v4b")
RESULT_TARGET = Path("/data/yeto-results-v4b")
MIN_FREE_BYTES = 1_000_000_000_000


def command_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


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
            errors.append("v4b results path is not a symlink")
        if resolved != RESULT_TARGET:
            errors.append(f"v4b results resolve to {resolved}, not {RESULT_TARGET}")
        if not proof["same_device_as_data"]:
            errors.append("v4b results are not on the /data filesystem")
        if usage.free < MIN_FREE_BYTES:
            errors.append(f"v4b results filesystem has only {usage.free} bytes free")
    except Exception as exc:
        errors.append(f"storage preflight failed: {exc}")
    return proof, errors


def verify_preflight(manifest_path: Path, node_label: str, proof_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    errors = []
    if manifest.get("schema") != "yeto_outer_mup_v4b_extension_launch_manifest_v1":
        errors.append("launch manifest schema mismatch")
    if manifest.get("manifest_variant") != "v4b_preoutcome_downward_extension":
        errors.append("launch manifest variant mismatch")
    if manifest.get("stage") != "V4B_EXTENSION" or len(manifest.get("cells", [])) != 18:
        errors.append("launch manifest is not the complete 18-cell V4B_EXTENSION stage")
    if manifest.get("status") != "AUTHORIZED":
        errors.append("launch manifest is not authorized")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != base.sha256_file(manifest_path):
        errors.append("launch manifest sidecar is absent or mismatched")

    repo = Path("/root/yeto")
    try:
        head = base.git(repo, "rev-parse", "HEAD")
        if head != manifest["source"]["git_commit"]:
            errors.append("node Git commit mismatch")
        if base.git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
            errors.append("node Git worktree is dirty")
        if manifest["registration"]["git_commit"] != head:
            errors.append("v4b must execute at its registration commit")
    except Exception as exc:
        errors.append(f"Git preflight failed: {exc}")

    constants = {
        "json_sha256": CONTRACT_JSON_SHA256,
        "md_sha256": CONTRACT_MD_SHA256,
        "analyzer_sha256": ANALYZER_SHA256,
    }
    contract = manifest.get("contract", {})
    for field, expected in constants.items():
        if contract.get(field) != expected:
            errors.append(f"manifest contract {field} differs from controller constant")
    contract_paths = {
        "json_sha256": repo / "experiment-specs/outer-mup-v4b-extension-prereg.json",
        "md_sha256": repo / "experiment-specs/outer-mup-v4b-extension-prereg.md",
        "analyzer_sha256": repo / "scripts/analyze_v4b.py",
    }
    for field, path in contract_paths.items():
        if not path.is_file() or base.sha256_file(path) != constants[field]:
            errors.append(f"registered file hash mismatch: {path}")

    base_v4 = manifest.get("base_v4", {})
    base_v4_path = Path(base_v4.get("manifest_path", ""))
    if not base_v4_path.is_file() or base.sha256_file(base_v4_path) != base_v4.get(
        "manifest_sha256"
    ):
        errors.append("base v4 manifest is absent or hash-mismatched")

    for record in manifest.get("inputs", {}).get("files", []):
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            errors.append(f"input size mismatch: {path}")
        elif base.sha256_file(path) != record["sha256"]:
            errors.append(f"input hash mismatch: {path}")

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
            errors.append(f"registered retry command mismatch: {cell_id}")
        assignment = cell.get("assignment", {})
        if assignment.get("gpu") not in range(6):
            errors.append(f"v4b assignment escapes GPUs 0..5: {cell_id}")
        if "--rho-telemetry" not in command or "--outer-bias-correction" in command:
            errors.append(f"v4b telemetry/bias-correction mismatch: {cell_id}")
        if command_value(command, "--gpu-offset") != str(assignment.get("gpu")):
            errors.append(f"GPU binding mismatch: {cell_id}")
        if command_value(command, "--outer-momentum") != str(cell.get("mu")):
            errors.append(f"momentum binding mismatch: {cell_id}")
        for flag in ("--work-dir", "--report-dir"):
            value = command_value(command, flag)
            if value is None or not value.startswith(str(RESULT_LINK) + "/"):
                errors.append(f"{flag} escapes new v4b results: {cell_id}")
    if len(coordinate_counts) != 6 or any(count != 3 for count in coordinate_counts.values()):
        errors.append(f"extension curve-cell balance failure: {coordinate_counts}")
    if sum(cell.get("s") == 10240 for cell in cells) != 12:
        errors.append("v4b does not contain exactly 12 long cells")
    for gpu in range(6):
        queue = sorted(
            [cell for cell in cells if cell.get("assignment") == {"node": node_label, "gpu": gpu}],
            key=lambda cell: cell.get("slot_queue_index", -1),
        )
        if not queue or queue[0].get("s") != 10240:
            errors.append(f"S10240 is not first on {node_label}/gpu{gpu}")

    inventory = base.gpu_inventory()
    if set(inventory) != set(range(8)):
        errors.append(f"expected GPU indices 0..7, got {sorted(inventory)}")
    if any("H200" not in item["name"].upper() for item in inventory.values()):
        errors.append("non-H200 device in node inventory")
    storage, storage_errors = storage_proof()
    errors.extend(storage_errors)
    proof = {
        "schema": "yeto_outer_mup_v4b_node_authority_v1",
        "node": node_label,
        "checked_at_utc": base.utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": base.sha256_file(manifest_path),
        "git_commit": manifest.get("source", {}).get("git_commit"),
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
    if authority.get("schema") != "yeto_outer_mup_v4b_launch_authority_v1":
        errors.append("launch authority schema mismatch")
    if authority.get("status") != "AUTHORIZED":
        errors.append("launch authority is not authorized")
    if authority.get("manifest_sha256") != base.sha256_file(manifest_path):
        errors.append("launch authority binds another manifest")
    started = authority.get("wall_clock_start_unix_s")
    deadline = authority.get("hard_deadline_unix_s")
    if not isinstance(started, (int, float)) or not isinstance(deadline, (int, float)):
        errors.append("launch authority lacks numeric wall times")
    elif abs(deadline - started - 43_200) > 1e-6:
        errors.append("launch authority does not encode the registered 12h ceiling")
    if errors:
        raise SystemExit("; ".join(errors))
    return authority


def load_retry_authority(path: Path, manifest_path: Path, manifest: dict) -> set[str]:
    authority = json.loads(path.read_text())
    errors = []
    if authority.get("schema") != "yeto_outer_mup_v4b_retry_authority_v1":
        errors.append("retry authority schema mismatch")
    if authority.get("status") != "AUTHORIZED":
        errors.append("retry authority is not authorized")
    if authority.get("manifest_sha256") != base.sha256_file(manifest_path):
        errors.append("retry authority binds another manifest")
    allowed = set(manifest.get("retry_contract", {}).get("allowed_reasons", []))
    if authority.get("reason") not in allowed:
        errors.append("retry authority reason is not registered")
    groups = authority.get("retry_group_ids")
    if not isinstance(groups, list) or not groups or len(set(groups)) != len(groups):
        errors.append("retry authority groups are empty, duplicated, or malformed")
        groups = []
    known = {cell["retry_group_id"] for cell in manifest.get("cells", [])}
    if set(groups) - known:
        errors.append("retry authority contains an unknown group")
    if errors:
        raise SystemExit("; ".join(errors))
    return set(groups)


def gpu_is_idle(gpu: int) -> tuple[bool, str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and not result.stdout.strip(), result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--node-label", choices=("h200-n1", "h200-n2"), required=True)
    parser.add_argument("--gpu", type=int, choices=range(6))
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--launch-authority", type=Path)
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--retry-authority", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        verify_preflight(args.manifest, args.node_label, args.proof)
        print(json.dumps({"node": args.node_label, "status": "PASS"}, sort_keys=True))
        return 0
    if args.gpu is None or args.launch_authority is None:
        parser.error("--gpu and --launch-authority are required for queue execution")
    idle, processes = gpu_is_idle(args.gpu)
    if not idle:
        raise SystemExit(f"refusing occupied GPU {args.gpu}: {processes}")
    # Reuse the already-audited v4 queue/evidence engine while rebinding only its
    # result root and v4b authority readers. The scientific command is separately
    # frozen in the v4b manifest and never targets the prior v4 tree.
    base.RESULT_LINK = RESULT_LINK
    base.RESULT_TARGET = RESULT_TARGET
    base.load_launch_authority = load_launch_authority
    base.load_retry_authority = load_retry_authority
    return base.run_queue(
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
