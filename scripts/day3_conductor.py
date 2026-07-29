#!/usr/bin/env python3
"""Persistent loss-blind conductor for V19 + V18 -> V16; V14 stays staged."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import day3_common as common


BRANCH = "experiment/day3-fleet"
CONTROL = Path("/private/tmp/day3-control")
NOTE = Path("/private/tmp/h200-day3-note.md")
POLL_SECONDS = 60
MILESTONE_SECONDS = 15 * 60
PROGRAM_TOTALS = {"v19": 54, "v18": 96, "v16": 612, "v14": 160}
PROGRAM_QUEUE_TOTALS = {"v19": 9, "v18": 7, "v16": 7, "v14": 7}
TERMINAL = {"COMPLETED", "SCIENTIFIC_DIVERGENCE", "INFRA_FAILURE", "INVALID_WORK"}


class ConductorError(RuntimeError):
    pass


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def note(line: str) -> None:
    with NOTE.open("a", encoding="utf-8") as destination:
        destination.write(f"{timestamp()} {line.rstrip()}\n")


def run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode:
        raise ConductorError(
            f"command failed ({result.returncode}): {shlex.join(command)}\n"
            f"stdout={result.stdout[-4000:]}\nstderr={result.stderr[-4000:]}"
        )
    return result


def remote(node: str, script: str, *, check: bool = True) -> subprocess.CompletedProcess:
    return run(["ssh", "-o", "BatchMode=yes", f"root@{node}", script], check=check)


def public_head() -> str:
    head = common.git("rev-parse", "HEAD")
    remote_tip = run(
        ["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"], cwd=common.REPO
    ).stdout.split()
    if len(remote_tip) != 2 or remote_tip[0] != head:
        raise ConductorError("local day-3 HEAD is not the public branch tip")
    if common.git("status", "--porcelain=v1", "--untracked-files=no"):
        raise ConductorError("tracked day-3 worktree changes forbid conductor start")
    return head


def gpu_snapshot(node: str) -> list[dict[str, int]]:
    result = remote(
        node,
        "nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu "
        "--format=csv,noheader,nounits",
    )
    rows = []
    for line in result.stdout.splitlines():
        index, memory, free, utilization = [
            int(value.strip()) for value in line.split(",")
        ]
        rows.append(
            {
                "index": index,
                "memory_mib": memory,
                "free_mib": free,
                "utilization": utilization,
            }
        )
    if [row["index"] for row in rows] != list(range(8)):
        raise ConductorError(f"{node}: incomplete H200 inventory")
    return rows


def fleet_clear() -> tuple[bool, dict[str, object]]:
    snapshot = {}
    clear = True
    for node in common.NODES:
        try:
            rows = gpu_snapshot(node)
            snapshot[node] = rows
            if any(row["memory_mib"] > 1024 for row in rows):
                clear = False
        except Exception as exc:  # transient SSH is a wait state, not launch authority
            snapshot[node] = {"error": str(exc)}
            clear = False
    return clear, snapshot


def wait_for_fleet() -> None:
    last_note = 0.0
    consecutive = 0
    while consecutive < 2:
        clear, snapshot = fleet_clear()
        consecutive = consecutive + 1 if clear else 0
        now = time.monotonic()
        if now - last_note >= MILESTONE_SECONDS or last_note == 0.0:
            compact = {
                node: (
                    value
                    if isinstance(value, dict)
                    else {
                        "max_memory_mib": max(row["memory_mib"] for row in value),
                        "max_utilization": max(row["utilization"] for row in value),
                    }
                )
                for node, value in snapshot.items()
            }
            note(f"FLEET WAIT: live occupancy guard {json.dumps(compact, sort_keys=True)}")
            last_note = now
        if consecutive < 2:
            time.sleep(30 if clear else POLL_SECONDS)


def setup_node(node: str, commit: str) -> tuple[str, str]:
    repo = str(common.REMOTE_REPO)
    script = f"""
set -eu
test -d /root/yeto/.git
if test ! -d {shlex.quote(repo)}/.git; then
  test ! -e {shlex.quote(repo)}
  origin_url=$(git -C /root/yeto remote get-url origin)
  git clone --no-checkout "$origin_url" {shlex.quote(repo)}
else
  test -z "$(git -C {shlex.quote(repo)} status --porcelain=v1 --untracked-files=no)"
fi
git -C {shlex.quote(repo)} fetch origin {shlex.quote(BRANCH)}
test "$(git -C {shlex.quote(repo)} rev-parse FETCH_HEAD)" = {shlex.quote(commit)}
git -C {shlex.quote(repo)} checkout -B {shlex.quote(BRANCH)} {shlex.quote(commit)}
test -z "$(git -C {shlex.quote(repo)} status --porcelain=v1 --untracked-files=no)"
mkdir -p /data/tmp /data/hf-datasets-cache /data/yeto-results-day3 /root/day3-control
if test -e /root/yeto-results-day3 || test -L /root/yeto-results-day3; then
  test -L /root/yeto-results-day3
  test "$(readlink -f /root/yeto-results-day3)" = /data/yeto-results-day3
else
  ln -s /data/yeto-results-day3 /root/yeto-results-day3
fi
case "$(findmnt -T /data/yeto-results-day3 -n -o SOURCE)" in
  /dev/mapper/*) ;;
  *) exit 42 ;;
esac
. /root/.cargo/env
env HF_DATASETS_CACHE=/data/hf-datasets-cache TMPDIR=/data/tmp \
  cargo build --release --manifest-path {shlex.quote(repo)}/syncer/Cargo.toml >/root/day3-control/build.log 2>&1
/root/yeto-venv/bin/python {shlex.quote(repo)}/scripts/build_v14_model_proof.py \
  --output /root/day3-control/v14-model-proof.json >/root/day3-control/v14-model-proof.log
sha256sum {shlex.quote(repo)}/syncer/target/release/yeto-syncer
sha256sum /root/day3-control/v14-model-proof.json
"""
    result = remote(node, script)
    lines = [line.split()[0] for line in result.stdout.splitlines() if len(line.split()) >= 2]
    if len(lines) < 2:
        raise ConductorError(f"{node}: missing deployment hashes: {result.stdout}")
    return lines[-2], lines[-1]


def build_and_copy_manifests(commit: str) -> dict[str, Path]:
    deployment = {node: setup_node(node, commit) for node in common.NODES}
    syncer_hashes = {value[0] for value in deployment.values()}
    model_proof_hashes = {value[1] for value in deployment.values()}
    if len(syncer_hashes) != 1 or len(model_proof_hashes) != 1:
        raise ConductorError(f"node deployment hashes differ: {deployment}")
    syncer_hash = next(iter(syncer_hashes))
    run(
        [
            "scp",
            "-q",
            "root@h200-n1:/root/day3-control/v14-model-proof.json",
            str(CONTROL / "v14-model-proof.json"),
        ]
    )
    if common.sha256_file(CONTROL / "v14-model-proof.json") != next(iter(model_proof_hashes)):
        raise ConductorError("collected V14 model proof hash mismatch")
    v19_inputs = CONTROL / "v19-input-manifest.json"
    v16_inputs = CONTROL / "v16-input-manifest.json"
    if common.sha256_file(v19_inputs) != (
        "2617bd8ee756d793bc7d0e43971ed085bfbb4aa15e564d3cc63096db4b0ed675"
    ):
        raise ConductorError("local V19 input proof hash mismatch")
    if common.sha256_file(v16_inputs) != (
        "b77b353dcc8a548c8e4ff31a252ed74919f325482795a15732b4ae9261b08164"
    ):
        raise ConductorError("local V16 input proof hash mismatch")
    builders = {
        "v19": [
            "scripts/build_v19_manifest.py",
            "--input-manifest",
            str(v19_inputs),
        ],
        "v18": [
            "scripts/build_v18_manifest.py",
            "--model-input-manifest",
            str(v19_inputs),
        ],
        "v16": [
            "scripts/build_v16_manifest.py",
            "--input-manifest",
            str(v16_inputs),
        ],
        "v14": [
            "scripts/build_v14_manifest.py",
            "--model-proof",
            str(CONTROL / "v14-model-proof.json"),
        ],
    }
    manifests = {}
    for program, builder in builders.items():
        output = CONTROL / f"{program}-manifest.json"
        run(
            [
                sys.executable,
                *builder,
                "--syncer-sha256",
                syncer_hash,
                "--output",
                str(output),
            ],
            cwd=common.REPO,
        )
        manifests[program] = output
    for node in common.NODES:
        for program, manifest in manifests.items():
            for path in (manifest, manifest.with_suffix(manifest.suffix + ".sha256")):
                run(["scp", "-q", str(path), f"root@{node}:/root/day3-control/{path.name}"])
        run(
            [
                "scp",
                "-q",
                str(CONTROL / "v14-model-proof.json"),
                f"root@{node}:/root/day3-control/v14-model-proof.json",
            ]
        )
    note(
        "DEPLOYED day-3 commit "
        f"{commit}; syncer={syncer_hash}; manifests="
        + json.dumps(
            {program: common.sha256_file(path) for program, path in manifests.items()},
            sort_keys=True,
        )
    )
    deployment_state = {
        "schema": "yeto_day3_deployment_v1",
        "created_at": timestamp(),
        "source_git_commit": commit,
        "syncer_sha256": syncer_hash,
        "v14_model_proof_sha256": next(iter(model_proof_hashes)),
        "manifests": {
            program: {
                "path": str(path),
                "sha256": common.sha256_file(path),
            }
            for program, path in manifests.items()
        },
    }
    common.write_json_atomic(CONTROL / "deployment.json", deployment_state)
    return manifests


def load_or_build_manifests(commit: str) -> dict[str, Path]:
    state_path = CONTROL / "deployment.json"
    if not state_path.is_file():
        return build_and_copy_manifests(commit)
    state = common.read_json(state_path)
    if (
        state.get("schema") != "yeto_day3_deployment_v1"
        or state.get("source_git_commit") != commit
    ):
        raise ConductorError("existing frozen day-3 deployment differs from public HEAD")
    manifests = {}
    for program in ("v19", "v18", "v16", "v14"):
        record = state.get("manifests", {}).get(program, {})
        path = Path(str(record.get("path", "")))
        if (
            not path.is_file()
            or not path.with_suffix(path.suffix + ".sha256").is_file()
            or common.sha256_file(path) != record.get("sha256")
        ):
            raise ConductorError(f"frozen {program} manifest/state mismatch")
        manifests[program] = path
    for node in common.NODES:
        script_lines = [
            "set -eu",
            f'test "$(git -C {common.REMOTE_REPO} rev-parse HEAD)" = {shlex.quote(commit)}',
        ]
        for program, path in manifests.items():
            expected = common.sha256_file(path)
            script_lines.append(
                f'test "$(sha256sum /root/day3-control/{program}-manifest.json | cut -d" " -f1)" = {expected}'
            )
        remote(node, "\n".join(script_lines))
    note(f"DAY-3 RESUME: reused frozen deployment state {common.sha256_file(state_path)}")
    return manifests


def validate_launch_partition(manifests: dict[str, Path]) -> None:
    schedules = {}
    for program in ("v19", "v18", "v16"):
        manifest = common.read_json(manifests[program])
        slots = {(queue["node"], int(queue["gpu"])) for queue in manifest["queues"]}
        if len(slots) != PROGRAM_QUEUE_TOTALS[program]:
            raise ConductorError(f"{program}: queue slots are duplicated or miscounted")
        if slots & set(common.PROTECTED_SLOTS):
            raise ConductorError(f"{program}: protected GPU appears in launch partition")
        if manifest.get("capacity", {}).get("class") != "135M-class":
            raise ConductorError(f"{program}: shared-host class is not authorized")
        schedules[program] = slots
    if schedules["v19"] & schedules["v18"]:
        raise ConductorError("V19 and V18 primary GPU partitions overlap")
    if schedules["v18"] != schedules["v16"]:
        raise ConductorError("V16 does not reuse exactly the post-V18 capacity partition")
    note(
        "FLEET PARTITION: V19 9 clear GPUs + V18/V16 7 clear GPUs; "
        "all 16 passed two post-foreign-job clear samples"
    )


def controller_pattern(program: str, queue_id: str | None = None) -> str:
    suffix = f".*--queue-id {queue_id}" if queue_id else ""
    return f"[r]un_day3_queue.py.*{program}-manifest.json{suffix}"


def program_active(program: str) -> bool:
    for node in common.NODES:
        result = remote(node, f"pgrep -af {shlex.quote(controller_pattern(program))} || true")
        if result.stdout.strip():
            return True
    return False


def queue_active(program: str, queue_id: str, node: str) -> bool:
    result = remote(
        node,
        f"pgrep -af {shlex.quote(controller_pattern(program, queue_id))} || true",
    )
    return bool(result.stdout.strip())


def launch_queue(
    program: str,
    queue: dict,
    *,
    attempt: int = 1,
    authority: Path | None = None,
) -> None:
    node = queue["node"] if authority is None else common.read_json(authority)["node"]
    queue_id = queue["queue_id"]
    pattern = controller_pattern(program, queue_id)
    authority_arg = (
        f" --retry-authority {shlex.quote('/root/day3-control/' + authority.name)}"
        if authority is not None
        else ""
    )
    launch_id = uuid.uuid4().hex[:12]
    log_suffix = (
        f"a{attempt}" + (f"-{authority.stem}" if authority else "") + f"-{launch_id}"
    )
    command = (
        f"/root/yeto-venv/bin/python {common.REMOTE_REPO}/scripts/run_day3_queue.py "
        f"--manifest /root/day3-control/{program}-manifest.json "
        f"--node-label {shlex.quote(node)} --queue-id {shlex.quote(queue_id)} "
        f"--attempt {attempt}{authority_arg}"
    )
    script = f"""
set -eu
test -z "$(pgrep -af {shlex.quote(pattern)} || true)"
mkdir -p /data/yeto-results-day3/_controller/logs /data/yeto-results-day3/_controller/pids
nohup env HF_DATASETS_CACHE=/data/hf-datasets-cache TMPDIR=/data/tmp \
  {command} \
  >/data/yeto-results-day3/_controller/logs/{shlex.quote(queue_id)}-{log_suffix}.log 2>&1 </dev/null &
pid=$!
echo "$pid" >/data/yeto-results-day3/_controller/pids/{shlex.quote(queue_id)}-{log_suffix}.pid
echo "$pid"
"""
    result = remote(node, script)
    note(f"{program.upper()} QUEUE START: {node}/{queue_id} attempt={attempt} pid={result.stdout.strip()}")


def launch_program(program: str, manifest_path: Path) -> None:
    manifest = common.read_json(manifest_path)
    for queue in manifest["queues"]:
        launch_queue(program, queue)
    note(
        f"{program.upper()} STARTED: cells={len(manifest['cells'])}; "
        f"queues={len(manifest['queues'])}; manifest={common.sha256_file(manifest_path)}"
    )


def ensure_initial_controllers(
    program: str, manifest_path: Path, status: dict[str, object]
) -> int:
    manifest = common.read_json(manifest_path)
    records_by_queue: dict[str, list[dict]] = {}
    for record in status["cells"]:
        records_by_queue.setdefault(record["queue_id"], []).append(record)
    launched = 0
    for queue in manifest["queues"]:
        queue_id = queue["queue_id"]
        records = records_by_queue.get(queue_id, [])
        attempt1_terminal = len(records) == int(queue["scientific_cells"]) and all(
            record.get("attempt_statuses", {}).get("1") in TERMINAL
            for record in records
        )
        if attempt1_terminal or queue_active(program, queue_id, queue["node"]):
            continue
        launch_queue(program, queue)
        launched += 1
    if launched:
        note(f"{program.upper()} ATTEMPT-1 RESUME: relaunched {launched} incomplete queues")
    return launched


def node_status(program: str, node: str) -> dict:
    result = remote(
        node,
        f"/root/yeto-venv/bin/python {common.REMOTE_REPO}/scripts/day3_status.py "
        f"--manifest /root/day3-control/{program}-manifest.json --node-label {node}",
    )
    return json.loads(result.stdout)


def combined_status(program: str, manifest_path: Path) -> dict[str, object]:
    nodes = {node: node_status(program, node) for node in common.NODES}
    counts: dict[str, int] = {}
    groups = {}
    cells = []
    slots = []
    for payload in nodes.values():
        for status, count in payload["status_counts"].items():
            counts[status] = counts.get(status, 0) + int(count)
        groups.update(payload["groups"])
        cells.extend(payload["cell_records"])
        slots.extend(payload["slots"])
    terminal_cells = sum(count for status, count in counts.items() if status in TERMINAL)
    terminal_groups = sum(bool(record["terminal"]) for record in groups.values())
    return {
        "nodes": nodes,
        "status_counts": counts,
        "terminal_cells": terminal_cells,
        "completed_cells": counts.get("COMPLETED", 0),
        "terminal_groups": terminal_groups,
        "groups": groups,
        "cells": cells,
        "slots": slots,
        "total": len(common.read_json(manifest_path)["cells"]),
    }


def retry_candidates(status: dict) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    records_by_group: dict[str, list[dict]] = {}
    for record in status["cells"]:
        records_by_group.setdefault(record["retry_group_id"], []).append(record)
    for group_id, records in records_by_group.items():
        attempt1 = [record.get("attempt_statuses", {}).get("1") for record in records]
        attempt2 = [record.get("attempt_statuses", {}).get("2") for record in records]
        if not all(value in TERMINAL for value in attempt1):
            continue
        if "INFRA_FAILURE" not in attempt1:
            continue
        if all(value in TERMINAL for value in attempt2):
            continue
        candidates.append(
            {
                "retry_group_id": group_id,
                "queue_id": records[0]["queue_id"],
                "resume_attempt2": any(value is not None for value in attempt2),
            }
        )
    return candidates


def choose_retry_gpu(
    manifest: dict, group_id: str, *, reserved: set[tuple[str, int]]
) -> tuple[str, int] | None:
    group_cells = [
        cell for cell in manifest["cells"] if cell["retry_group_id"] == group_id
    ]
    if not group_cells:
        raise ConductorError(f"retry group disappeared from manifest: {group_id}")
    first = group_cells[0]
    options = [
        (str(record["node"]), int(record["gpu"]))
        for record in first["registered_retry_commands"]
        if int(record["attempt_number"]) == 2
    ]
    if any(
        {
            (str(record["node"]), int(record["gpu"]))
            for record in cell["registered_retry_commands"]
            if int(record["attempt_number"]) == 2
        }
        != set(options)
        for cell in group_cells
    ):
        raise ConductorError(f"retry relocation set differs inside group: {group_id}")
    snapshots = {node: gpu_snapshot(node) for node in {node for node, _ in options}}
    capacity = manifest["capacity"]
    required_free = int(capacity["minimum_free_before_launch_mib"])
    maximum_used = int(capacity["exclusive_max_prelaunch_used_mib"])
    for node, gpu in options:
        if (node, gpu) in reserved:
            continue
        row = snapshots[node][gpu]
        if row["free_mib"] >= required_free and row["memory_mib"] <= maximum_used:
            return node, gpu
    return None


def maybe_launch_retries(program: str, manifest_path: Path, status: dict) -> int:
    if program_active(program):
        return 0
    manifest = common.read_json(manifest_path)
    queue_by_id = {queue["queue_id"]: queue for queue in manifest["queues"]}
    launched_queues = set()
    reserved: set[tuple[str, int]] = set()
    launched = 0
    for candidate in retry_candidates(status):
        group_id = str(candidate["retry_group_id"])
        queue_id = str(candidate["queue_id"])
        if queue_id in launched_queues:
            continue
        authority = CONTROL / f"{program}-retry-{group_id}.json"
        if candidate["resume_attempt2"] and not authority.is_file():
            raise ConductorError(
                f"partial attempt-2 group lacks its frozen retry authority: {group_id}"
            )
        if authority.is_file():
            payload = common.read_json(authority)
            if (
                payload.get("schema") != "yeto_day3_retry_authority_v1"
                or payload.get("manifest_sha256") != common.sha256_file(manifest_path)
                or payload.get("program") != program
                or payload.get("retry_group_id") != group_id
                or payload.get("queue_id") != queue_id
                or payload.get("attempt") != 2
                or payload.get("finite_endpoint_seen") is not False
            ):
                raise ConductorError(f"existing retry authority mismatch: {authority}")
            selected = (str(payload["node"]), int(payload["gpu"]))
            if selected in reserved:
                continue
        else:
            selected = choose_retry_gpu(manifest, group_id, reserved=reserved)
            if selected is None:
                continue
            node, gpu = selected
            payload = {
                "schema": "yeto_day3_retry_authority_v1",
                "created_at": timestamp(),
                "manifest_sha256": common.sha256_file(manifest_path),
                "program": program,
                "queue_id": queue_id,
                "retry_group_id": group_id,
                "attempt": 2,
                "node": node,
                "gpu": gpu,
                "attempt1_node": queue_by_id[queue_id]["node"],
                "attempt1_gpu": queue_by_id[queue_id]["gpu"],
                "reason": "frozen runner classified attempt 1 as INFRA_FAILURE before a valid endpoint",
                "finite_endpoint_seen": False,
                "whole_group_relocation": True,
            }
            common.write_json_atomic(authority, payload)
        node, gpu = selected
        run(["scp", "-q", str(authority), f"root@{node}:/root/day3-control/{authority.name}"])
        launch_queue(program, queue_by_id[queue_id], attempt=2, authority=authority)
        launched_queues.add(queue_id)
        reserved.add((node, gpu))
        launched += 1
    return launched


def collect_and_analyze(program: str, manifest_path: Path) -> None:
    result_root = CONTROL / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    manifest = common.read_json(manifest_path)
    for node in common.NODES:
        node_cells = [
            cell["cell_id"]
            for cell in manifest["cells"]
            if cell["assignment"]["node"] == node
        ]
        rules = []
        for cell_id in node_cells:
            rules.extend(
                [
                    f"+ /{cell_id}/",
                    f"+ /{cell_id}/attempt-*/",
                    f"+ /{cell_id}/attempt-*/evidence.json",
                    f"+ /{cell_id}/attempt-*/report/",
                    f"+ /{cell_id}/attempt-*/report/results.jsonl",
                    f"+ /{cell_id}/attempt-*/work/",
                    f"+ /{cell_id}/attempt-*/work/m4/",
                    f"+ /{cell_id}/attempt-*/work/m4/tape.jsonl",
                ]
            )
        rules.append("- *")
        filter_path = CONTROL / f"{program}-{node}-collection.rules"
        filter_path.write_text("\n".join(rules) + "\n")
        run(
            [
                "rsync",
                "-a",
                "--prune-empty-dirs",
                f"--filter=merge {filter_path}",
                f"root@{node}:/data/yeto-results-day3/",
                str(result_root) + "/",
            ]
        )
    output = CONTROL / f"{program}-readout.json"
    analyzer = common.REPO / "scripts" / f"analyze_{program}.py"
    result = run(
        [
            sys.executable,
            str(analyzer),
            "--manifest",
            str(manifest_path),
            "--result-root",
            str(result_root),
            "--output",
            str(output),
            "--note",
            str(NOTE),
        ],
        cwd=common.REPO,
        check=False,
    )
    if result.returncode:
        note(
            f"{program.upper()} ANALYZER ERROR: rc={result.returncode}; "
            f"stdout={result.stdout[-1000:]!r}; stderr={result.stderr[-1000:]!r}"
        )
        raise ConductorError(f"frozen {program} analyzer errored")
    note(f"{program.upper()} ANALYZED: output={output}; sha256={common.sha256_file(output)}")


def milestone(program: str, status: dict) -> str:
    retry_pending = len(retry_candidates(status))
    resolved_groups = int(status["terminal_groups"]) - retry_pending
    if program == "v19":
        return (
            f"V19 {resolved_groups}/9; cells={status['terminal_cells']}/54; "
            f"retry_pending={retry_pending}; statuses={status['status_counts']}"
        )
    return (
        f"{program.upper()} {status['terminal_cells']}/{PROGRAM_TOTALS[program]}; "
        f"groups={resolved_groups}; retry_pending={retry_pending}; "
        f"statuses={status['status_counts']}"
    )


def existing_readout(program: str, manifest_path: Path) -> bool:
    path = CONTROL / f"{program}-readout.json"
    if not path.is_file():
        return False
    payload = common.read_json(path)
    if payload.get("manifest_sha256") != common.sha256_file(manifest_path):
        raise ConductorError(f"existing {program} readout targets another manifest")
    return True


def record_v14_capacity_deferral(manifest_path: Path) -> None:
    marker = CONTROL / "v14-capacity-deferred.json"
    if marker.is_file():
        return
    payload = {
        "schema": "yeto_day3_capacity_deferral_v1",
        "created_at": timestamp(),
        "program": "v14",
        "manifest_sha256": common.sha256_file(manifest_path),
        "reason": (
            "V14 is SmolLM2-1.7B; the binding shared-host directive permits "
            "135M-class work only and places 1.7B+ off queue until user clearance"
        ),
    }
    common.write_json_atomic(marker, payload)
    note(
        "V14 CAPACITY-DEFERRED: registered model is SmolLM2-1.7B; "
        "1.7B+ remains off queue under the binding shared-host directive"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    CONTROL.mkdir(parents=True, exist_ok=True)
    commit = public_head()
    note(f"DAY-3 CONDUCTOR START: branch={BRANCH}; commit={commit}; pid={os.getpid()}")
    manifests = load_or_build_manifests(commit)
    validate_launch_partition(manifests)
    launched = {"v19", "v18"}
    analyzed = {
        program
        for program in ("v19", "v18", "v16")
        if existing_readout(program, manifests[program])
    }
    if "v18" in analyzed:
        launched.add("v16")
    initial_statuses = {
        program: combined_status(program, manifests[program]) for program in launched
    }
    for program in sorted(launched):
        ensure_initial_controllers(program, manifests[program], initial_statuses[program])
    note(
        "DAY-3 ACTIVE: V19 nine queues and V18 seven queues are armed on disjoint "
        "capacity-checked GPUs"
    )
    last_milestone = 0.0
    while True:
        statuses = {
            program: combined_status(program, manifests[program]) for program in launched
        }
        now = time.monotonic()
        if now - last_milestone >= MILESTONE_SECONDS or last_milestone == 0.0:
            for program in sorted(launched):
                note(milestone(program, statuses[program]))
            last_milestone = now
        for program in tuple(sorted(launched)):
            status = statuses[program]
            ensure_initial_controllers(program, manifests[program], status)
            maybe_launch_retries(program, manifests[program], status)
            active = program_active(program)
            candidates = retry_candidates(status)
            if (
                program not in analyzed
                and status["terminal_cells"] == PROGRAM_TOTALS[program]
                and not active
                and not candidates
            ):
                collect_and_analyze(program, manifests[program])
                analyzed.add(program)
                if program == "v18" and "v16" not in launched:
                    launched.add("v16")
                    v16_status = combined_status("v16", manifests["v16"])
                    ensure_initial_controllers("v16", manifests["v16"], v16_status)
                    note("V16 STARTED: P2 drained; seven safe GPUs transferred to P3")
                elif program == "v16":
                    record_v14_capacity_deferral(manifests["v14"])
        if analyzed >= {"v19", "v18", "v16"}:
            record_v14_capacity_deferral(manifests["v14"])
            note(
                "DAY-3 135M-CLASS COMPLETE: V19, V18, and V16 drained/analyzed; "
                "V14 remains capacity-deferred"
            )
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        note(f"DAY-3 CONDUCTOR ERROR: {type(exc).__name__}: {exc}")
        raise
