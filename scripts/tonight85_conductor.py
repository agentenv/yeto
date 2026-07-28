#!/usr/bin/env python3
"""Mac-side, loss-blind conductor for the TONIGHT-8.5 extended lease."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tonight85_common as common  # noqa: E402


BRANCH = "experiment/tonight-8.5-lean"
LOCAL_TZ = ZoneInfo("America/Los_Angeles")
NOT_BEFORE = datetime(2026, 7, 27, 20, 30, tzinfo=LOCAL_TZ).timestamp()
ANALYSIS_CUTOFF = datetime(2026, 7, 28, 8, 30, tzinfo=LOCAL_TZ).timestamp()
NOTE = Path("/private/tmp/h200-tonight85-note.md")
CONTROL = Path("/private/tmp/tonight85-control")
LOCAL_RESULTS = CONTROL / "results"
NODES = common.NODES
POLL_SECONDS = 30
PRIORITY_REMOTE = r"""
set -u
echo PRIORITY
pgrep -af '[r]un_slot_v9.py|[l]aunch_v9_stage.py|[r]un_slot_v10.py|[v]10-transfer|[7][bB]-wave' 2>/dev/null || true
echo COMPUTE
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
echo TONIGHT
pgrep -af '[r]un_slot_tonight85.py' 2>/dev/null || true
echo END
"""


class ConductorError(RuntimeError):
    """A hard safety or integrity condition failed."""


def now_local() -> str:
    return datetime.now(LOCAL_TZ).isoformat()


def note(line: str) -> None:
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    with NOTE.open("a", encoding="utf-8") as destination:
        destination.write(f"{now_local()} {line.rstrip()}\n")


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        capture_output=True,
        text=True,
    )
    if check and result.returncode:
        raise ConductorError(
            f"command failed ({result.returncode}): {shlex.join(command)}\n"
            f"stdout={result.stdout[-4000:]}\nstderr={result.stderr[-4000:]}"
        )
    return result


def remote(
    node: str, script: str, *, check: bool = True
) -> subprocess.CompletedProcess:
    return run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            f"root@{node}",
            "bash -s",
        ],
        check=check,
        input_text=script,
    )


def remote_sections(node: str) -> dict[str, list[str]]:
    result = remote(node, PRIORITY_REMOTE)
    sections: dict[str, list[str]] = {}
    current = None
    for line in result.stdout.splitlines():
        if line in {"PRIORITY", "COMPUTE", "TONIGHT", "END"}:
            current = line
            sections.setdefault(line, [])
        elif current:
            sections[current].append(line)
    return sections


def fleet_clear_for_tonight() -> tuple[bool, dict]:
    records = {}
    clear = time.time() >= NOT_BEFORE
    for node in NODES:
        try:
            sections = remote_sections(node)
        except Exception as exc:
            records[node] = {"error": str(exc)}
            clear = False
            continue
        records[node] = sections
        if (
            sections.get("PRIORITY")
            or sections.get("COMPUTE")
            or sections.get("TONIGHT")
        ):
            clear = False
    return clear, records


def wait_for_priority_drain() -> None:
    last_milestone = 0.0
    while True:
        clear, records = fleet_clear_for_tonight()
        if clear:
            note(
                "FLEET GATE PASS: after 20:30, no v9/v10 priority controllers, no tonight controllers, and no GPU compute PIDs on either node"
            )
            return
        if time.time() - last_milestone >= 900:
            summary = {
                node: {
                    "priority": len(record.get("PRIORITY", [])),
                    "compute": len(record.get("COMPUTE", [])),
                    "tonight": len(record.get("TONIGHT", [])),
                    "error": record.get("error"),
                }
                for node, record in records.items()
            }
            note(f"FLEET WAIT: {json.dumps(summary, sort_keys=True)}")
            last_milestone = time.time()
        mark_cutoff_deferrals()
        time.sleep(POLL_SECONDS)


def pushed_head() -> str:
    head = common.git("rev-parse", "HEAD")
    result = run(
        ["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"], cwd=common.REPO
    )
    fields = result.stdout.split()
    if len(fields) != 2 or fields[0] != head:
        raise ConductorError(f"origin/{BRANCH} is not exact local HEAD {head}")
    return head


def wait_compute_clear() -> None:
    while True:
        occupied = {}
        for node in NODES:
            sections = remote_sections(node)
            if (
                sections.get("PRIORITY")
                or sections.get("COMPUTE")
                or sections.get("TONIGHT")
            ):
                occupied[node] = {
                    "priority": sections.get("PRIORITY", []),
                    "compute": sections.get("COMPUTE", []),
                    "tonight": sections.get("TONIGHT", []),
                }
        if not occupied:
            return
        time.sleep(POLL_SECONDS)


def deploy_commit(commit: str, *, initial: bool) -> None:
    wait_compute_clear()
    for node in NODES:
        script = f"""
set -eu
test -d /root/yeto/.git
test -z "$(git -C /root/yeto status --porcelain=v1 --untracked-files=all)"
git -C /root/yeto fetch origin {shlex.quote(BRANCH)}
test "$(git -C /root/yeto rev-parse FETCH_HEAD)" = {shlex.quote(commit)}
git -C /root/yeto checkout --detach {shlex.quote(commit)}
test -z "$(git -C /root/yeto status --porcelain=v1 --untracked-files=all)"
mkdir -p /data/yeto-results-tonight85 /root/tonight85-control
if [ -e /root/yeto-results-tonight85 ] || [ -L /root/yeto-results-tonight85 ]; then
  test -L /root/yeto-results-tonight85
  test "$(readlink -f /root/yeto-results-tonight85)" = /data/yeto-results-tonight85
else
  ln -s /data/yeto-results-tonight85 /root/yeto-results-tonight85
fi
test "$(sha256sum /root/yeto-data/tonight85-v13/manifest.json | awk '{{print $1}}')" = b77b353dcc8a548c8e4ff31a252ed74919f325482795a15732b4ae9261b08164
test "$(sha256sum /root/yeto-data/tonight85-v13/train.jsonl | awk '{{print $1}}')" = f9bcb68e84370667cfe2418450c9fcd112cd0cc9236936e48e3aa35f9dd27ace
test "$(sha256sum /root/yeto-data/tonight85-v13/eval.jsonl | awk '{{print $1}}')" = 08ac254552fbd6529c68577c94920e4f33f6e75b5c94c518350c18948ec48df4
. /root/.cargo/env
env HF_DATASETS_CACHE=/data/hf-datasets-cache cargo build --release --manifest-path /root/yeto/syncer/Cargo.toml
"""
        remote(node, script)
    note(
        f"DEPLOYED pushed commit {commit} to both nodes"
        + ("; release syncer rebuilt" if initial else "")
    )


def copy_manifest(local_path: Path) -> dict[str, Path]:
    if not local_path.with_suffix(local_path.suffix + ".sha256").is_file():
        raise ConductorError(f"manifest sidecar missing: {local_path}")
    remote_paths = {}
    for node in NODES:
        destination = f"/root/tonight85-control/{local_path.name}"
        run(["scp", "-q", str(local_path), f"root@{node}:{destination}"])
        run(
            [
                "scp",
                "-q",
                str(local_path.with_suffix(local_path.suffix + ".sha256")),
                f"root@{node}:{destination}.sha256",
            ]
        )
        remote_paths[node] = Path(destination)
    return remote_paths


def cells_for_stage(manifest: dict, stage: str) -> list[dict]:
    return [cell for cell in manifest["cells"] if cell["stage"] == stage]


def stage_slots(
    manifest: dict, stage: str, retry_groups: set[str] | None = None
) -> list[tuple[str, str]]:
    cells = cells_for_stage(manifest, stage)
    if retry_groups is not None:
        cells = [cell for cell in cells if cell["retry_group_id"] in retry_groups]
    return sorted({(cell["assignment"]["node"], cell["slot_id"]) for cell in cells})


def launch_stage(
    manifest_path: Path,
    manifest: dict,
    stage: str,
    *,
    attempt: int = 1,
    retry_groups: set[str] | None = None,
) -> list[tuple[str, str]]:
    wait_compute_clear()
    remote_paths = copy_manifest(manifest_path)
    slots = stage_slots(manifest, stage, retry_groups)
    if not slots:
        raise ConductorError(f"stage {stage} has no launch slots")
    groups_arg = (
        ["--retry-groups", ",".join(sorted(retry_groups))] if retry_groups else []
    )
    for node, slot_id in slots:
        argv = [
            "/root/yeto-venv/bin/python",
            "/root/yeto/scripts/run_slot_tonight85.py",
            "--manifest",
            str(remote_paths[node]),
            "--node-label",
            node,
            "--stage",
            stage,
            "--slot-id",
            slot_id,
            "--attempt",
            str(attempt),
            *groups_arg,
        ]
        label = f"{stage}-{slot_id}-a{attempt}"
        script = f"""
set -eu
mkdir -p /root/yeto-results-tonight85/_controller/logs /root/yeto-results-tonight85/_controller/pids
nohup env HF_DATASETS_CACHE=/data/hf-datasets-cache HF_HUB_CACHE=/root/yeto-hf-cache/hub PYTHONUNBUFFERED=1 {shlex.join(argv)} > /root/yeto-results-tonight85/_controller/logs/{shlex.quote(label)}.log 2>&1 < /dev/null &
pid=$!
echo "$pid" > /root/yeto-results-tonight85/_controller/pids/{shlex.quote(label)}.pid
"""
        remote(node, script)
    note(
        f"STAGE STARTED: {stage} attempt {attempt}; {len(slots)} queues; "
        f"manifest={common.sha256_file(manifest_path)}"
    )
    return slots


def read_remote_status(
    node: str, stage: str, slot_id: str, attempt: int
) -> dict | None:
    path = f"/root/yeto-results-tonight85/_controller/slots/{stage}/{slot_id}-a{attempt}.json"
    try:
        result = remote(
            node, f"test -f {shlex.quote(path)} && cat {shlex.quote(path)} || true\n"
        )
    except Exception as exc:
        return {"state": "TRANSIENT_SSH_ERROR", "error": str(exc)}
    if not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ConductorError(f"{node}:{path}: malformed status: {exc}") from exc


def wait_stage(stage: str, slots: list[tuple[str, str]], attempt: int) -> dict:
    last_note = 0.0
    while True:
        statuses = {
            f"{node}/{slot}": read_remote_status(node, stage, slot, attempt)
            for node, slot in slots
        }
        if all(
            status is not None and status.get("state") == "DRAINED"
            for status in statuses.values()
        ):
            completed = sum(
                int(status.get("completed", 0)) for status in statuses.values()
            )
            failures = sum(
                int(status.get("failures", 0)) for status in statuses.values()
            )
            note(
                f"STAGE DRAINED: {stage} attempt {attempt}; completed={completed}; failures={failures}"
            )
            return {"completed": completed, "failures": failures, "statuses": statuses}
        if time.time() - last_note >= 900:
            summary = {
                key: None
                if status is None
                else {
                    "state": status.get("state"),
                    "cell": status.get("cell_id"),
                    "completed": status.get("completed"),
                    "failures": status.get("failures"),
                }
                for key, status in statuses.items()
            }
            note(
                f"STAGE ACTIVE: {stage} a{attempt} {json.dumps(summary, sort_keys=True)}"
            )
            last_note = time.time()
        mark_cutoff_deferrals()
        time.sleep(POLL_SECONDS)


def sync_results() -> None:
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    for node in NODES:
        command = [
            "rsync",
            "-az",
            "--prune-empty-dirs",
            "--include=*/",
            "--include=evidence.json",
            "--include=results.jsonl",
            "--exclude=*",
            f"root@{node}:/root/yeto-results-tonight85/",
            f"{LOCAL_RESULTS}/",
        ]
        while True:
            result = run(command, check=False)
            if result.returncode == 0:
                break
            note(
                f"RESULT SWEEPER WAIT: rsync from {node} failed rc={result.returncode}; retrying"
            )
            mark_cutoff_deferrals()
            time.sleep(POLL_SECONDS)


def evidence_status(cell: dict, attempt: int) -> str | None:
    path = LOCAL_RESULTS / cell["cell_id"] / f"attempt-{attempt}" / "evidence.json"
    if not path.is_file():
        return None
    return common.read_json(path).get("status")


def retry_groups_for(manifest: dict, stage: str) -> set[str]:
    cells = cells_for_stage(manifest, stage)
    by_group: dict[str, list[dict]] = {}
    for cell in cells:
        by_group.setdefault(cell["retry_group_id"], []).append(cell)
    groups = set()
    for group, records in by_group.items():
        statuses = [evidence_status(cell, 1) for cell in records]
        if "SCIENTIFIC_DIVERGENCE" in statuses:
            continue
        if any(
            status not in {"COMPLETED", "SCIENTIFIC_DIVERGENCE"} for status in statuses
        ):
            groups.add(group)
    return groups


def maybe_retry(manifest_path: Path, manifest: dict, stage: str) -> None:
    sync_results()
    groups = retry_groups_for(manifest, stage)
    if not groups:
        return
    authority = {
        "schema": "yeto_tonight85_retry_authority_v1",
        "status": "AUTHORIZED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "manifest_sha256": common.sha256_file(manifest_path),
        "retry_group_ids": sorted(groups),
        "reason": "loss-blind missing/invalid/infrastructure endpoint audit",
        "scientific_losses_inspected": False,
    }
    authority_path = CONTROL / f"retry-authority-{stage}.json"
    common.write_json_atomic(authority_path, authority)
    note(
        f"RETRY AUTHORIZED: {stage}; groups={','.join(sorted(groups))}; authority={common.sha256_file(authority_path)}"
    )
    slots = launch_stage(manifest_path, manifest, stage, attempt=2, retry_groups=groups)
    wait_stage(stage, slots, 2)
    sync_results()


DEFERRED: set[str] = set()
ANALYZED: set[str] = set()


def mark_cutoff_deferrals() -> None:
    if time.time() < ANALYSIS_CUTOFF:
        return
    for gate in ("G11", "G12", "G13"):
        if gate not in ANALYZED and gate not in DEFERRED:
            note(
                f"{gate} VERDICT: DEFERRED_ARXIV_V2 (not fully drained and analyzed by 08:30 PDT; no partial gate)"
            )
            DEFERRED.add(gate)


def run_analyzer(
    gate: str,
    script: str,
    argv: list[str],
    output: Path,
) -> None:
    sync_results()
    before_cutoff = time.time() < ANALYSIS_CUTOFF and gate not in DEFERRED
    analyzer_note = NOTE if before_cutoff else CONTROL / "arxiv-v2-readouts.md"
    command = [
        sys.executable,
        str(common.REPO / "scripts" / script),
        *argv,
        "--result-root",
        str(LOCAL_RESULTS),
        "--output",
        str(output),
        "--note",
        str(analyzer_note),
    ]
    result = run(command, cwd=common.REPO, check=False)
    if result.returncode:
        if before_cutoff:
            note(
                f"{gate} VERDICT: NOT_EVALUABLE (frozen analyzer failed: {result.stderr[-1000:]})"
            )
            ANALYZED.add(gate)
        else:
            note(f"{gate} ARXIV-V2 ANALYZER: NOT_EVALUABLE ({result.stderr[-500:]})")
        return
    readout = common.read_json(output)
    verdict = readout["gate"]["verdict"]
    if before_cutoff:
        ANALYZED.add(gate)
    else:
        note(
            f"{gate} ARXIV-V2 READOUT: {verdict}; cutoff disposition remains DEFERRED_ARXIV_V2"
        )


def commit_prediction(path: Path, message: str) -> str:
    status = common.git("status", "--porcelain=v1", "--untracked-files=all")
    lines = [line for line in status.splitlines() if line]
    expected_suffix = str(path.relative_to(common.REPO))
    if any(line[3:] != expected_suffix for line in lines) or len(lines) != 1:
        raise ConductorError(
            f"prediction commit worktree has unexpected changes: {lines}"
        )
    run(["git", "add", expected_suffix], cwd=common.REPO)
    run(["git", "commit", "-m", message], cwd=common.REPO)
    commit = common.git("rev-parse", "HEAD")
    run(["git", "push", "origin", f"HEAD:refs/heads/{BRANCH}"], cwd=common.REPO)
    if pushed_head() != commit:
        raise ConductorError("prediction push verification failed")
    note(
        f"PREDICTION SEALED+PUSHED: {expected_suffix}; commit={commit}; sha256={common.sha256_file(path)}"
    )
    return commit


def build_dynamic_manifest(kind: str, prediction: Path, output: Path) -> dict:
    run(
        [
            sys.executable,
            str(common.REPO / "scripts/build_tonight85_manifest.py"),
            "--kind",
            kind,
            "--dynamic",
            str(prediction),
            "--output",
            str(output),
        ],
        cwd=common.REPO,
    )
    return common.read_json(output)


def build_refreshed_static_manifest(output: Path) -> dict:
    run(
        [
            sys.executable,
            str(common.REPO / "scripts/build_tonight85_manifest.py"),
            "--kind",
            "static",
            "--output",
            str(output),
        ],
        cwd=common.REPO,
    )
    return common.read_json(output)


def run_stage_with_retry(manifest_path: Path, manifest: dict, stage: str) -> None:
    slots = launch_stage(manifest_path, manifest, stage)
    wait_stage(stage, slots, 1)
    maybe_retry(manifest_path, manifest, stage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-manifest", type=Path, required=True)
    args = parser.parse_args()
    CONTROL.mkdir(parents=True, exist_ok=True)
    static_path = args.static_manifest.resolve()
    static = common.read_json(static_path)
    registration_commit = pushed_head()
    if static["source"]["git_commit"] != registration_commit:
        raise ConductorError(
            "static manifest is not bound to the pushed registration commit"
        )
    note(
        f"CONDUCTOR STARTED: registration={registration_commit}; pid={os.getpid()}; waiting for priority drain"
    )
    wait_for_priority_drain()
    deploy_commit(registration_commit, initial=True)

    # S2+S3 first, interleaved nine cells per one-GPU queue.
    run_stage_with_retry(static_path, static, "short_scans")
    run_analyzer(
        "G12",
        "analyze_v12.py",
        ["--manifest", str(static_path)],
        CONTROL / "v12-readout.json",
    )
    run_analyzer(
        "G13",
        "analyze_v13.py",
        ["--manifest", str(static_path)],
        CONTROL / "v13-readout.json",
    )

    # V11 registered probes, then Mac-side prediction seal/push, then truth.
    run_stage_with_retry(static_path, static, "v11_anchor")
    sync_results()
    v11_prediction = (
        common.REPO / "experiment-specs/outer-mup-v11-sealed-predictions.json"
    )
    run(
        [
            sys.executable,
            str(common.REPO / "scripts/build_v11_predictions.py"),
            "--manifest",
            str(static_path),
            "--result-root",
            str(LOCAL_RESULTS),
            "--output",
            str(v11_prediction),
        ],
        cwd=common.REPO,
    )
    v11_commit = commit_prediction(
        v11_prediction, "Seal v11 ratio-transport predictions"
    )
    deploy_commit(v11_commit, initial=False)
    v11_manifest_path = CONTROL / "v11-truth-manifest.json"
    v11_manifest = build_dynamic_manifest(
        "v11-truth", v11_prediction, v11_manifest_path
    )
    run_stage_with_retry(v11_manifest_path, v11_manifest, "v11_truth")
    run_analyzer(
        "G11",
        "analyze_v11.py",
        [
            "--manifest",
            str(v11_manifest_path),
            "--predictions",
            str(v11_prediction),
        ],
        CONTROL / "v11-readout.json",
    )

    # Reduced v7 is last. Refresh its prep commands at the already-pushed v11
    # descendant so node HEAD and recorded command provenance remain exact.
    v7_prep_path = CONTROL / "v7-prep-manifest.json"
    v7_prep = build_refreshed_static_manifest(v7_prep_path)
    run_stage_with_retry(v7_prep_path, v7_prep, "v7_smoke")
    run_stage_with_retry(v7_prep_path, v7_prep, "v7_pilot")
    sync_results()
    v7_prediction = (
        common.REPO / "experiment-specs/outer-mup-v7-lean-sealed-prediction.json"
    )
    run(
        [
            sys.executable,
            str(common.REPO / "scripts/build_v7_lean_prediction.py"),
            "--manifest",
            str(v7_prep_path),
            "--result-root",
            str(LOCAL_RESULTS),
            "--output",
            str(v7_prediction),
        ],
        cwd=common.REPO,
    )
    v7_commit = commit_prediction(v7_prediction, "Seal reduced v7 raw prediction")
    deploy_commit(v7_commit, initial=False)
    v7_manifest_path = CONTROL / "v7-raw-manifest.json"
    v7_manifest = build_dynamic_manifest("v7-raw", v7_prediction, v7_manifest_path)
    run_stage_with_retry(v7_manifest_path, v7_manifest, "v7_raw")
    sync_results()
    run(
        [
            sys.executable,
            str(common.REPO / "scripts/analyze_v7_lean.py"),
            "--manifest",
            str(v7_manifest_path),
            "--prediction",
            str(v7_prediction),
            "--result-root",
            str(LOCAL_RESULTS),
            "--output",
            str(CONTROL / "v7-lean-readout.json"),
            "--note",
            str(NOTE),
        ],
        cwd=common.REPO,
    )
    mark_cutoff_deferrals()
    note("TONIGHT-8.5 CONDUCTOR COMPLETE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        note(f"CONDUCTOR BLOCKED: {type(exc).__name__}: {exc}")
        raise
