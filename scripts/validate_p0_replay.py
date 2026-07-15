#!/usr/bin/env python3
"""Replay every sealed P0 RDA/Nesterov commit after exact GPU deletion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CKPT_MAGIC = 0xD1705A7E
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_RELATIVE_PATH = Path("scripts/validate_p0_replay.py")
AUTHORITATIVE_PREREG_RELATIVE_PATH = Path(
    "experiment-specs/best-paper-phase-map-p0-p1-prereg.json"
)
AUTHORITATIVE_PREREG_COMMIT = "16d27bc60deb6d8910bf0111c7fb57c9d0eb5b80"
AUTHORITATIVE_PREREG_SHA256 = (
    "7cba3c62328b4bfe15fffbc523979274e834e8e720e16f70d79621eaf6ebdb7b"
)
ADOPTED_PARALLEL_AMENDMENT_PATH = Path("docs/AMENDMENT-parallel-cells.md")
ADOPTED_PARALLEL_AMENDMENT_SHA256 = (
    "e2c87fd6c2ec0e4b91f488b5771334e0befd175560a3e2ccfcf349be1ee8b3dd"
)
P0A_SOURCE_REBIND_FROM_COMMIT = "0af7f4a80426babc14896c7c1f7885abcb331d46"
P0B_REPLAY_SOURCE_REBIND_FROM_COMMIT = (
    "8d58208cacafef12cb95f2642b4fa700531151b4"
)
P0B_REPLAY_ERRATUM_PATH = Path("docs/ERRATUM-p0b-replay-validator.md")
P0B_REPLAY_ERRATUM_SHA256 = (
    "241c151a7ef6b3ff18221618000ed772c331fd691d69d882c192d3bb8a169aa4"
)
PARAM_ATOL = 2e-6
PARAM_RTOL = 2e-6
TAPE_NORM_RTOL = 2e-4
CANDIDATE_RE = re.compile(
    r"candidate_step_(\d{8})_fragment_(\d{4})_learner_(\d{4})\.f32$"
)


class ReplayError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ReplayError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ReplayError(f"{path}:{number}: expected a JSON object")
        rows.append(value)
    return rows


def validate_barrier_version_trace(
    root: Path,
    attempt: Path,
    result: dict[str, Any],
    tape: list[dict[str, Any]],
    acquisition_registry: dict[str, str],
    *,
    h: int,
    seq_len: int,
) -> dict[str, Any]:
    """Independently replay each learner's barrier causal state machine."""
    registry_path = attempt / "report" / "barrier-version-trace.json"
    registry_relative = registry_path.relative_to(root).as_posix()
    if acquisition_registry.get(registry_relative) != sha256_file(registry_path):
        raise ReplayError("barrier trace registry is not acquisition-sealed")
    registry = read_json(registry_path)
    if (
        registry.get("schema") != "yeto_barrier_version_trace_v1"
        or registry.get("learner_count") != 4
    ):
        raise ReplayError("barrier trace registry schema/count mismatch")

    hardware = result.get("hardware")
    if not isinstance(hardware, dict):
        raise ReplayError("P0 result lacks barrier hardware attestations")
    registry_sha = sha256_file(registry_path)
    registry_canonical_sha = sha256_bytes(canonical_json(registry))
    if (
        hardware.get("barrier_version_trace_sha256") != registry_sha
        or hardware.get("barrier_version_trace_canonical_sha256")
        != registry_canonical_sha
        or not str(hardware.get("barrier_version_trace_uri", "")).endswith(
            "/" + registry_relative
        )
    ):
        raise ReplayError("result hardware does not bind the sealed barrier registry")

    def verify_entry(entry: Any, expected_relative: str, label: str) -> Path:
        if not isinstance(entry, dict) or entry.get("path") != expected_relative:
            raise ReplayError(f"{label} registry path mismatch")
        path = attempt / expected_relative
        root_relative = path.relative_to(root).as_posix()
        if not path.is_file() or path.is_symlink():
            raise ReplayError(f"{label} artifact is missing or unsafe")
        digest = sha256_file(path)
        if (
            entry.get("sha256") != digest
            or acquisition_registry.get(root_relative) != digest
        ):
            raise ReplayError(f"{label} artifact is not hash- and acquisition-bound")
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size != path.stat().st_size:
            raise ReplayError(f"{label} artifact size mismatch")
        return path

    tape_path = verify_entry(
        registry.get("syncer_tape"), "work/m4/tape.jsonl", "syncer tape"
    )
    if read_jsonl(tape_path) != tape:
        raise ReplayError("barrier registry tape differs from replay tape")
    trace_entries = registry.get("learner_traces")
    if not isinstance(trace_entries, list) or len(trace_entries) != 4:
        raise ReplayError("barrier registry lacks exactly four learner traces")
    trace_paths: dict[int, Path] = {}
    for entry in trace_entries:
        if not isinstance(entry, dict):
            raise ReplayError("malformed learner trace registry entry")
        learner_id = entry.get("learner_id")
        if (
            isinstance(learner_id, bool)
            or not isinstance(learner_id, int)
            or learner_id not in range(4)
            or learner_id in trace_paths
        ):
            raise ReplayError("barrier registry learner IDs are not exactly 0..3")
        trace_paths[learner_id] = verify_entry(
            entry,
            f"work/m4/learner-{learner_id}/barrier-version-trace.jsonl",
            f"learner {learner_id} trace",
        )
    if set(trace_paths) != {0, 1, 2, 3}:
        raise ReplayError("barrier registry learner IDs are not exactly 0..3")

    if [row.get("step") for row in tape] != list(range(1, len(tape) + 1)):
        raise ReplayError("barrier tape steps are not exact contiguous commit IDs")
    prior_fragment_commit = {fragment: 0 for fragment in range(4)}
    fragment_counts: dict[int, int] = defaultdict(int)
    expected_pushes: dict[tuple[int, int], dict[str, int]] = {}
    for step, tape_row in enumerate(tape, 1):
        fragment = tape_row.get("fragment")
        responders = tape_row.get("responders")
        if (
            isinstance(fragment, bool)
            or not isinstance(fragment, int)
            or fragment not in range(4)
            or not isinstance(responders, list)
            or sorted(responder.get("id") for responder in responders) != [0, 1, 2, 3]
        ):
            raise ReplayError(f"barrier tape step {step} lacks exact fragment/quorum")
        expected_base = prior_fragment_commit[fragment]
        if {responder.get("base_version") for responder in responders} != {expected_base}:
            raise ReplayError(
                f"barrier tape step {step} bases differ from prior fragment commit"
            )
        for responder in responders:
            if (
                responder.get("c_steps") != h
                or responder.get("c_tokens") != h * seq_len
            ):
                raise ReplayError(f"barrier tape step {step} has wrong fixed work")
            learner_id = int(responder["id"])
            expected_pushes[(learner_id, step)] = {
                "fragment": fragment,
                "base_version": expected_base,
                "c_steps": h,
                "c_tokens": h * seq_len,
            }
        prior_fragment_commit[fragment] = step
        fragment_counts[fragment] += 1
    if len(tape) % 4 or fragment_counts != {
        fragment: len(tape) // 4 for fragment in range(4)
    }:
        raise ReplayError("barrier tape does not balance exact updates per fragment")

    expected_inner_steps = len(tape) // 4 * h
    for learner_id in range(4):
        awaiting: dict[int, tuple[int, int, int]] = {}
        reset_local_step = {fragment: 0 for fragment in range(4)}
        initial_fragments: set[int] = set()
        pushes: set[int] = set()
        broadcasts: set[int] = set()
        next_inner_step = 1
        previous_local_step = 0
        for event_seq, event in enumerate(read_jsonl(trace_paths[learner_id]), 1):
            if (
                event.get("schema") != "yeto_barrier_trace_v1"
                or event.get("event_seq") != event_seq
                or event.get("learner_id") != learner_id
            ):
                raise ReplayError(
                    f"learner {learner_id} barrier event sequence/identity mismatch"
                )
            local_step = event.get("local_step")
            if (
                isinstance(local_step, bool)
                or not isinstance(local_step, int)
                or local_step < previous_local_step
            ):
                raise ReplayError(f"learner {learner_id} local_step is not monotone")
            previous_local_step = local_step
            declared_awaiting = event.get("awaiting_fragments")
            event_kind = event.get("event")
            if event_kind == "initial_broadcast_applied":
                fragment = event.get("fragment")
                initial_version = event.get("broadcast_version")
                if (
                    event_seq not in range(1, 5)
                    or isinstance(fragment, bool)
                    or not isinstance(fragment, int)
                    or fragment != event_seq - 1
                    or local_step != 0
                    or isinstance(initial_version, bool)
                    or not isinstance(initial_version, int)
                    or initial_version != 0
                    or declared_awaiting != []
                    or awaiting
                    or fragment in initial_fragments
                ):
                    raise ReplayError(
                        f"learner {learner_id} initial broadcast prefix is not exact"
                    )
                initial_fragments.add(fragment)
                continue
            if event_kind == "inner_step_started":
                if (
                    initial_fragments != {0, 1, 2, 3}
                    or awaiting
                    or declared_awaiting != []
                    or local_step != next_inner_step
                ):
                    raise ReplayError(
                        f"learner {learner_id} starts an inner step while blocked/out of order"
                    )
                next_inner_step += 1
                continue
            if event_kind == "push_sent":
                fragment = event.get("fragment")
                pull_step = event.get("pull_step")
                expected = expected_pushes.get((learner_id, pull_step))
                if (
                    isinstance(fragment, bool)
                    or not isinstance(fragment, int)
                    or fragment not in range(4)
                    or isinstance(pull_step, bool)
                    or not isinstance(pull_step, int)
                    or pull_step in pushes
                    or fragment in awaiting
                    or initial_fragments != {0, 1, 2, 3}
                    or expected is None
                    or expected
                    != {
                        "fragment": fragment,
                        "base_version": event.get("base_version"),
                        "c_steps": event.get("c_steps"),
                        "c_tokens": event.get("c_tokens"),
                    }
                    or local_step != ((pull_step - 1) // 4 + 1) * h
                    or local_step != reset_local_step[fragment] + h
                ):
                    raise ReplayError(
                        f"learner {learner_id} push does not biject to barrier tape"
                    )
                awaiting[fragment] = (
                    pull_step,
                    int(event["base_version"]),
                    local_step,
                )
                pushes.add(pull_step)
            elif event_kind == "broadcast_applied":
                fragment = event.get("fragment")
                pending = awaiting.get(fragment)
                if pending is None:
                    raise ReplayError(
                        f"learner {learner_id} broadcast lacks an outstanding push"
                    )
                pull_step, base_version, push_local_step = pending
                if (
                    event.get("pushed_base_version") != base_version
                    or event.get("broadcast_version") != pull_step
                    or pull_step <= base_version
                    or local_step != push_local_step
                    or pull_step in broadcasts
                ):
                    raise ReplayError(
                        f"learner {learner_id} broadcast does not release exact push"
                    )
                del awaiting[fragment]
                reset_local_step[fragment] = local_step
                broadcasts.add(pull_step)
            else:
                raise ReplayError(
                    f"learner {learner_id} has unknown barrier event {event_kind!r}"
                )
            if declared_awaiting != sorted(awaiting):
                raise ReplayError(
                    f"learner {learner_id} declared awaiting state is false"
                )
        expected_steps = {
            step for expected_learner, step in expected_pushes if expected_learner == learner_id
        }
        if awaiting or pushes != expected_steps or broadcasts != expected_steps:
            raise ReplayError(
                f"learner {learner_id} lacks exact push/broadcast coverage"
            )
        if initial_fragments != {0, 1, 2, 3}:
            raise ReplayError(
                f"learner {learner_id} lacks exact initial broadcast coverage"
            )
        if next_inner_step - 1 != expected_inner_steps:
            raise ReplayError(
                f"learner {learner_id} inner-step count differs from frozen work"
            )
        if set(reset_local_step.values()) != {expected_inner_steps}:
            raise ReplayError(
                f"learner {learner_id} fragment windows do not end at the frozen step"
            )

    expected_attestations = {
        "barrier_trace_validated": True,
        "base_versions_match": True,
        "no_inner_step_while_blocked": True,
        "barrier_trace_learner_count": 4,
        "barrier_trace_commit_count": len(tape),
        "barrier_trace_inner_steps_per_learner": expected_inner_steps,
    }
    if any(hardware.get(key) != value for key, value in expected_attestations.items()):
        raise ReplayError("result barrier summary attestations differ from raw traces")
    return {
        **expected_attestations,
        "barrier_version_trace_sha256": registry_sha,
        "barrier_version_trace_canonical_sha256": registry_canonical_sha,
    }


def git_output(*args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).decode(errors="replace")[-2000:]
        raise ReplayError(f"git source attestation failed: {detail}")
    return result.stdout


def replay_source_rebind_authority(
    frozen_commit: str,
) -> tuple[str, Path, str] | None:
    if frozen_commit == P0A_SOURCE_REBIND_FROM_COMMIT:
        return (
            "amendment",
            ADOPTED_PARALLEL_AMENDMENT_PATH,
            ADOPTED_PARALLEL_AMENDMENT_SHA256,
        )
    if frozen_commit == P0B_REPLAY_SOURCE_REBIND_FROM_COMMIT:
        return (
            "erratum",
            P0B_REPLAY_ERRATUM_PATH,
            P0B_REPLAY_ERRATUM_SHA256,
        )
    return None


def verify_replay_source(manifest: dict[str, Any]) -> dict[str, Any]:
    """Prove replay code comes from the exact P0 or adopted fixed commit."""
    frozen_commit = str((manifest.get("frozen") or {}).get("git_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", frozen_commit):
        raise ReplayError("P0 manifest lacks a frozen 40-hex Git commit")
    head = git_output("rev-parse", "HEAD").decode().strip()
    source_rebind = head != frozen_commit
    rebind_authority: tuple[str, Path, str] | None = None
    if source_rebind:
        rebind_authority = replay_source_rebind_authority(frozen_commit)
        if rebind_authority is None:
            raise ReplayError("replay source rebind lacks an adopted transition")
        git_output("cat-file", "-e", f"{frozen_commit}^{{commit}}")
        git_output("cat-file", "-e", f"{head}^{{commit}}")
        git_output("merge-base", "--is-ancestor", frozen_commit, head)
        _kind, authority_path, authority_sha256 = rebind_authority
        authority_blob = git_output("show", f"{head}:{authority_path.as_posix()}")
        if sha256_bytes(authority_blob) != authority_sha256:
            raise ReplayError("replay source rebind lacks its exact authority document")
    if git_output("status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ReplayError("CPU replay requires a completely clean checkout")
    script_blob = git_output("show", f"{head}:{SCRIPT_RELATIVE_PATH.as_posix()}")
    script_bytes = (REPO_ROOT / SCRIPT_RELATIVE_PATH).read_bytes()
    if script_bytes != script_blob:
        raise ReplayError("replay validator bytes differ from replay Git blob")
    prereg_blob = git_output(
        "show",
        f"{AUTHORITATIVE_PREREG_COMMIT}:{AUTHORITATIVE_PREREG_RELATIVE_PATH.as_posix()}",
    )
    if sha256_bytes(prereg_blob) != AUTHORITATIVE_PREREG_SHA256:
        raise ReplayError("authoritative preregistration blob hash mismatch")
    lineage = manifest.get("lineage") or {}
    if (
        lineage.get("authoritative_prereg_path")
        != AUTHORITATIVE_PREREG_RELATIVE_PATH.as_posix()
        or lineage.get("authoritative_prereg_source_commit")
        != AUTHORITATIVE_PREREG_COMMIT
        or lineage.get("authoritative_prereg_template_sha256")
        != AUTHORITATIVE_PREREG_SHA256
    ):
        raise ReplayError("P0 manifest does not bind the authoritative preregistration")
    attestation = {
        "replay_validator_git_commit": head,
        "replay_validator_script_path": SCRIPT_RELATIVE_PATH.as_posix(),
        "replay_validator_script_sha256": sha256_bytes(script_bytes),
        "replay_validator_git_blob_sha256": sha256_bytes(script_blob),
        "authoritative_prereg_path": AUTHORITATIVE_PREREG_RELATIVE_PATH.as_posix(),
        "authoritative_prereg_source_commit": AUTHORITATIVE_PREREG_COMMIT,
        "authoritative_prereg_template_sha256": AUTHORITATIVE_PREREG_SHA256,
    }
    if source_rebind:
        assert rebind_authority is not None
        kind, authority_path, authority_sha256 = rebind_authority
        attestation["replay_source_rebind_from_git_commit"] = frozen_commit
        attestation[f"replay_source_rebind_{kind}_path"] = authority_path.as_posix()
        attestation[f"replay_source_rebind_{kind}_sha256"] = authority_sha256
    return attestation


def validate_phase_manifest(
    manifest: dict[str, Any],
    *,
    parent_manifest: dict[str, Any] | None = None,
    parent_replay_report: dict[str, Any] | None = None,
    parent_replay_report_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the independently authored authority/integrity validator first."""
    from scripts.validate_phase_map import validate_and_summarize

    report = validate_and_summarize(
        manifest,
        claim_level="integrity",
        parent_manifest=parent_manifest,
        parent_replay_report=parent_replay_report,
        parent_replay_report_sha256=parent_replay_report_sha256,
    )
    if report.get("valid") is not True or report.get("integrity_status") != "VALIDATED":
        raise ReplayError("P0 phase-map integrity validator did not return VALIDATED")
    return {
        "phase_map_integrity_status": report["integrity_status"],
        "phase_map_validator_report_sha256": sha256_bytes(canonical_json(report)),
    }


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ReplayError("timestamps must be timezone-aware")
    return parsed


def verify_acquisition(root: Path, checksum_path: Path) -> dict[str, str]:
    registry: dict[str, str] = {}
    for line in checksum_path.read_text().splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ReplayError(f"malformed acquisition checksum line: {line!r}")
        path = root / relative
        if relative in registry or not path.is_file() or path.is_symlink():
            raise ReplayError(f"missing, duplicate, or unsafe artifact: {relative}")
        if sha256_file(path) != digest:
            raise ReplayError(f"acquisition checksum mismatch: {relative}")
        registry[relative] = digest
    if not registry:
        raise ReplayError("acquisition checksum manifest is empty")
    return registry


def checkpoint_fragments(path: Path) -> tuple[int, list[dict[str, int]]]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(16)
        if len(raw) != 16:
            raise ReplayError(f"truncated checkpoint header: {path}")
        magic, global_step, count = struct.unpack("<IQI", raw)
        if magic != CKPT_MAGIC:
            raise ReplayError(f"bad checkpoint magic: {path}")
        fragments = []
        offset = 16
        for fragment in range(count):
            handle.seek(offset)
            header = handle.read(16)
            if len(header) != 16:
                raise ReplayError(f"truncated fragment header: {path}")
            version, numel = struct.unpack("<QQ", header)
            params_offset = offset + 16
            momentum_offset = params_offset + 4 * numel
            offset = momentum_offset + 4 * numel
            if offset > size:
                raise ReplayError(f"truncated fragment payload: {path}")
            fragments.append(
                {
                    "id": fragment,
                    "version": int(version),
                    "numel": int(numel),
                    "params_offset": params_offset,
                    "momentum_offset": momentum_offset,
                }
            )
    return int(global_step), fragments


def read_fragment(path: Path, meta: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    count = meta["numel"]
    params = np.fromfile(
        path, dtype="<f4", count=count, offset=meta["params_offset"]
    )
    momentum = np.fromfile(
        path, dtype="<f4", count=count, offset=meta["momentum_offset"]
    )
    if params.size != count or momentum.size != count:
        raise ReplayError(f"truncated fragment arrays: {path}")
    return params, momentum


def l2(values: np.ndarray) -> float:
    if not np.all(np.isfinite(values)):
        raise ReplayError("replay vector contains nonfinite values")
    total = 0.0
    chunk = 1_000_000
    for start in range(0, values.size, chunk):
        current = values[start : start + chunk].astype(np.float64)
        total += float(np.dot(current, current))
    return math.sqrt(total)


def merge_avg(deltas: list[np.ndarray], weights: list[float]) -> np.ndarray:
    total = float(sum(weights))
    if total <= 0:
        return np.zeros_like(deltas[0])
    out = np.zeros_like(deltas[0])
    for delta, weight in zip(deltas, weights, strict=True):
        out += np.float32(weight / total) * delta
    return out


def merge_rda(deltas: list[np.ndarray], weights: list[float]) -> np.ndarray:
    total = float(sum(weights))
    if total <= 0:
        return np.zeros_like(deltas[0])
    norms = [l2(delta) for delta in deltas]
    radial = sum(norm * weight for norm, weight in zip(norms, weights)) / total
    out = np.zeros_like(deltas[0])
    for delta, weight, norm in zip(deltas, weights, norms, strict=True):
        if norm:
            out += np.float32(weight / total / norm) * delta
    direction_norm = l2(out)
    if direction_norm < 1e-12:
        return merge_avg(deltas, weights)
    out *= np.float32(radial / direction_norm)
    return out


def replay_merge(
    layout_fragment: dict[str, Any],
    learner_deltas: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    numel = sum(int(tensor["numel"]) for tensor in layout_fragment["tensors"])
    if any(delta.size != numel for delta in learner_deltas):
        raise ReplayError("candidate length differs from resolved layout")
    merged = np.empty(numel, dtype="<f4")
    offset = 0
    mode = layout_fragment["merge_mode"]
    if mode not in ("avg", "rda"):
        raise ReplayError(f"P0 replay does not permit merge mode {mode!r}")
    for tensor in layout_fragment["tensors"]:
        end = offset + int(tensor["numel"])
        slices = [delta[offset:end] for delta in learner_deltas]
        merged[offset:end] = (
            merge_avg(slices, weights) if mode == "avg" else merge_rda(slices, weights)
        )
        offset = end
    return merged


def option(command: list[str], flag: str) -> str:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise ReplayError(f"frozen command lacks {flag}") from exc


def validate_exact_learner_max_steps(
    command: list[str], tape: list[dict[str, Any]], *, h: int
) -> int:
    """Recompute the exact physical learner ceiling from sealed raw work."""
    if command.count("--learner-max-steps") != 1:
        raise ReplayError("P0 command must have exactly one learner step cap")
    if len(tape) % 4:
        raise ReplayError("P0 tape cannot define exact per-learner work")
    expected = (len(tape) // 4) * h
    try:
        observed = int(option(command, "--learner-max-steps"))
    except ValueError as exc:
        raise ReplayError("P0 command has a malformed learner step cap") from exc
    if observed != expected:
        raise ReplayError(
            "P0 command learner step cap differs from exact sealed tape work"
        )
    return expected


def check_close(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    atol: float,
    rtol: float,
    label: str,
) -> tuple[float, float]:
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(observed)):
        raise ReplayError(f"{label}: compared arrays contain nonfinite values")
    difference = np.abs(predicted.astype(np.float64) - observed.astype(np.float64))
    maximum = float(difference.max(initial=0.0))
    scale = float(np.abs(observed.astype(np.float64)).max(initial=0.0))
    limit = atol + rtol * scale
    if not all(math.isfinite(value) for value in (maximum, scale, limit)):
        raise ReplayError(f"{label}: comparison produced a nonfinite diagnostic")
    if maximum > limit:
        raise ReplayError(f"{label}: max_abs={maximum} exceeds frozen limit={limit}")
    return maximum, limit


def replay_cell(
    root: Path,
    result: dict[str, Any],
    expected_cell: dict[str, Any],
    frozen_command_hash: str,
    registry: dict[str, str],
    *,
    atol: float,
    rtol: float,
    tape_rtol: float,
) -> dict[str, Any]:
    attempt = (
        root
        / "cells"
        / result["cell_id"]
        / f"attempt-{result['attempt']}"
    )
    command = json.loads((attempt / "command.json").read_text())
    if not isinstance(command, list):
        raise ReplayError("command.json must be an argv array")
    if any("audit" in str(token).casefold() for token in command):
        raise ReplayError("sealed P0 command names an audit surface")
    frozen_root = (root / "frozen-eval").resolve()
    for flag in ("--data", "--prebound-development-eval"):
        path = Path(option(command, flag)).resolve()
        try:
            relative_frozen = path.relative_to(frozen_root)
            relative_root = path.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise ReplayError(f"{flag} is outside the sealed frozen bundle") from exc
        if (
            not path.is_file()
            or path.is_symlink()
            or registry.get(relative_root) != sha256_file(path)
            or "audit" in relative_frozen.as_posix().casefold()
        ):
            raise ReplayError(f"{flag} is not a sealed sanitized input")
    command_hash = sha256_bytes(canonical_json(command))
    if (
        result.get("command_hash") != command_hash
        or frozen_command_hash != command_hash
    ):
        raise ReplayError("replayed command hash differs from result/frozen registry")
    coordinate_fields = ("h", "mu", "eta", "seed", "training_seed")
    for field in coordinate_fields:
        if result.get(field) != expected_cell.get(field):
            raise ReplayError(f"result coordinate {field} differs from expected cell")
    command_coordinates = {
        "h": int(option(command, "--fixed-window-microsteps")),
        "mu": float(option(command, "--outer-momentum")),
        "eta": float(option(command, "--outer-lr")),
        "seed": int(option(command, "--shuffle-rows-seed")),
        "training_seed": int(option(command, "--training-seed")),
    }
    if command_coordinates != {field: expected_cell[field] for field in coordinate_fields}:
        raise ReplayError("replayed command coordinates differ from frozen cell")
    command_seq_len = int(option(command, "--seq-len"))
    for flag in (
        "--syncer-probe-capture",
        "--strict-quorum",
        "--barrier-sync",
        "--version-matched-anchor",
    ):
        if flag not in command:
            raise ReplayError(f"P0 command lacks {flag}")
    if (
        command.count("--pipeline-depth") != 1
        or option(command, "--pipeline-depth") != "4"
    ):
        raise ReplayError("P0 barrier command must use one pipeline slot per fragment")
    if command.count("--wan-streams") != 1 or option(command, "--wan-streams") != "0":
        raise ReplayError("P0 barrier command must serialize learner push streams")
    if option(command, "--syncer-probe-capture-every") != "1":
        raise ReplayError("P0 must capture every applied step")
    eta = np.float32(float(option(command, "--outer-lr")))
    mu = np.float32(float(option(command, "--outer-momentum")))
    capture = attempt / "work" / "m4" / "syncer_probe"
    tape_path = attempt / "work" / "m4" / "tape.jsonl"
    final_state = attempt / "work" / "m4" / "state.ckpt"
    layout_paths = [
        attempt / "work" / "m4" / f"learner-{learner}" / "resolved-layout.json"
        for learner in range(4)
    ]
    layout_path = layout_paths[0]
    for path in (tape_path, final_state, *layout_paths, capture / "index.jsonl"):
        relative = path.relative_to(root).as_posix()
        if registry.get(relative) != sha256_file(path):
            raise ReplayError(f"replay input is not sealed: {relative}")
    if len({path.read_bytes() for path in layout_paths}) != 1:
        raise ReplayError("four learner layout artifacts are not byte-identical")
    capture_files = [path for path in capture.rglob("*") if path.is_file()]
    for path in capture_files:
        relative = path.relative_to(root).as_posix()
        if registry.get(relative) != sha256_file(path):
            raise ReplayError(f"capture is not sealed: {relative}")

    tape = read_jsonl(tape_path)
    index = read_jsonl(capture / "index.jsonl")
    if [int(row.get("step", -1)) for row in tape] != list(
        range(1, len(tape) + 1)
    ):
        raise ReplayError("tape steps are not one contiguous ordered sequence")
    validate_exact_learner_max_steps(
        command, tape, h=command_coordinates["h"]
    )
    barrier_attestation = validate_barrier_version_trace(
        root,
        attempt,
        result,
        tape,
        registry,
        h=command_coordinates["h"],
        seq_len=command_seq_len,
    )
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    candidate_names: set[str] = set()
    for row in index:
        try:
            step = int(row["step"])
            fragment = int(row["fragment"])
            learner = int(row["learner_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayError("capture index has malformed coordinates") from exc
        candidate_name = str(row.get("candidate_f32", ""))
        if candidate_name in candidate_names:
            raise ReplayError("capture candidate filenames are not unique")
        candidate_names.add(candidate_name)
        match = CANDIDATE_RE.fullmatch(Path(candidate_name).name)
        if (
            match is None
            or Path(candidate_name).parent.as_posix() != "candidates"
            or tuple(map(int, match.groups())) != (step, fragment, learner)
        ):
            raise ReplayError("capture candidate filename/index coordinates differ")
        by_step[step].append(row)
    if set(by_step) != {int(row["step"]) for row in tape}:
        raise ReplayError("capture steps do not exactly equal tape steps")
    if any(len(rows) != 4 for rows in by_step.values()):
        raise ReplayError("every captured step must contain four candidates")

    state_paths = sorted((capture / "states").glob("state_before_step_*.ckpt"))
    if len(state_paths) != len(tape):
        raise ReplayError("capture must contain one pre-state per applied step")
    expected_state_paths = [
        capture / "states" / f"state_before_step_{step:08d}.ckpt"
        for step in range(1, len(tape) + 1)
    ]
    if state_paths != expected_state_paths:
        raise ReplayError("pre-state filenames do not exactly match tape steps")
    state_meta: dict[Path, tuple[int, list[dict[str, int]]]] = {}
    version_source: dict[tuple[int, int], tuple[Path, dict[str, int]]] = {}
    for state_index, path in enumerate([*state_paths, final_state]):
        global_step, fragments = checkpoint_fragments(path)
        expected_global_step = state_index if path != final_state else len(tape)
        if global_step != expected_global_step:
            raise ReplayError(
                f"checkpoint global step {global_step} != {expected_global_step}: {path}"
            )
        if len(fragments) != 4:
            raise ReplayError("every replay checkpoint must contain four fragments")
        state_meta[path] = (global_step, fragments)
        for fragment in fragments:
            version_source.setdefault(
                (fragment["id"], fragment["version"]), (path, fragment)
            )

    layout = read_json(layout_path)
    fragments = layout.get("fragments")
    if not isinstance(fragments, list) or len(fragments) != 4:
        raise ReplayError("resolved P0 layout must contain four fragments")
    first_global_step, first_fragments = state_meta[state_paths[0]]
    if first_global_step != 0:
        raise ReplayError("first P0 capture does not begin at global step zero")
    for fragment in first_fragments:
        _initial_params, initial_momentum = read_fragment(state_paths[0], fragment)
        if np.any(initial_momentum != np.float32(0.0)):
            raise ReplayError("P0 first captured momentum buffer is not exactly zero")
    maximum_param_error = 0.0
    maximum_momentum_error = 0.0
    maximum_gnorm_relative_error = 0.0
    maximum_step_norm_relative_error = 0.0
    steps = []
    prior_target_versions = [-1, -1, -1, -1]
    for tape_row in tape:
        step = int(tape_row["step"])
        fragment_id = int(tape_row["fragment"])
        if fragment_id not in range(4):
            raise ReplayError(f"step {step}: tape fragment is outside [0,3]")
        rows = sorted(by_step[step], key=lambda row: int(row["learner_id"]))
        if {int(row["learner_id"]) for row in rows} != {0, 1, 2, 3}:
            raise ReplayError(f"step {step}: candidate learner set is not full quorum")
        state_names = {row["state_checkpoint"] for row in rows}
        if len(state_names) != 1:
            raise ReplayError(f"step {step}: candidates do not share one pre-state")
        pre_state = capture / next(iter(state_names))
        expected_pre_state = state_paths[step - 1]
        if pre_state != expected_pre_state or pre_state not in state_meta:
            raise ReplayError(f"step {step}: index points at the wrong pre-state")
        pre_global_step, pre_fragments = state_meta[pre_state]
        if pre_global_step != step - 1:
            raise ReplayError(f"step {step}: pre-state global-step mismatch")
        post_state = state_paths[step] if step < len(tape) else final_state
        post_global_step, post_fragments = state_meta[post_state]
        if post_global_step != step:
            raise ReplayError(f"step {step}: post-state global-step mismatch")
        pre_meta = pre_fragments[fragment_id]
        post_meta = post_fragments[fragment_id]
        if pre_meta["version"] <= prior_target_versions[fragment_id]:
            raise ReplayError(f"step {step}: target fragment version is not monotone")
        prior_target_versions[fragment_id] = pre_meta["version"]
        if post_meta["version"] != step:
            raise ReplayError(f"step {step}: committed fragment version is not the step")
        current, momentum = read_fragment(pre_state, pre_meta)
        tape_responders = tape_row.get("responders")
        if not isinstance(tape_responders, list) or len(tape_responders) != 4:
            raise ReplayError(f"step {step}: tape does not contain exact full quorum")
        responder_by_id = {int(row.get("id", -1)): row for row in tape_responders}
        if set(responder_by_id) != {0, 1, 2, 3}:
            raise ReplayError(f"step {step}: tape responder IDs are not full quorum")
        learner_deltas = []
        weights = []
        for row in rows:
            learner_id = int(row["learner_id"])
            if int(row["fragment"]) != fragment_id:
                raise ReplayError(f"step {step}: index/tape fragment mismatch")
            if int(row.get("syncer_global_step", -1)) != pre_global_step:
                raise ReplayError(f"step {step}: captured syncer global step mismatch")
            if int(row.get("current_fragment_version", -1)) != pre_meta["version"]:
                raise ReplayError(f"step {step}: captured current fragment version mismatch")
            responder = responder_by_id[learner_id]
            for index_field, tape_field in (
                ("base_version", "base_version"),
                ("c_steps", "c_steps"),
                ("c_tokens", "c_tokens"),
            ):
                if int(row.get(index_field, -1)) != int(
                    responder.get(tape_field, -2)
                ):
                    raise ReplayError(
                        f"step {step}: capture/tape responder {index_field} mismatch"
                    )
            if responder.get("anchor_base_resolved") is not True:
                raise ReplayError(f"step {step}: responder base anchor is unresolved")
            c_steps = int(row.get("c_steps", 0))
            c_tokens = int(row.get("c_tokens", 0))
            if c_steps <= 0 or c_tokens <= 0:
                raise ReplayError(f"step {step}: responder work is non-positive")
            if c_steps != command_coordinates["h"]:
                raise ReplayError(f"step {step}: responder c_steps differs from H")
            if c_tokens != c_steps * command_seq_len:
                raise ReplayError(f"step {step}: responder c_tokens differs from H*seq_len")
            recomputed_weight = float(c_tokens) * float(c_tokens) / float(c_steps)
            index_weight = float(row.get("weight", float("nan")))
            tape_weight = float(responder.get("weight", float("nan")))
            if (
                not math.isfinite(index_weight)
                or not math.isclose(index_weight, recomputed_weight, rel_tol=1e-12)
                or not math.isclose(tape_weight, recomputed_weight, rel_tol=1e-12)
            ):
                raise ReplayError(f"step {step}: responder weight is not c_tokens^2/c_steps")
            candidate_path = capture / row["candidate_f32"]
            candidate = np.fromfile(candidate_path, dtype="<f4")
            if not np.all(np.isfinite(candidate)):
                raise ReplayError(f"step {step}: candidate contains nonfinite values")
            base_key = (fragment_id, int(row["base_version"]))
            if base_key not in version_source:
                raise ReplayError(f"step {step}: base version is absent from captures")
            base_path, base_meta = version_source[base_key]
            base, _unused = read_fragment(base_path, base_meta)
            if candidate.size != base.size:
                raise ReplayError(f"step {step}: candidate size mismatch")
            learner_deltas.append(base - candidate)
            weights.append(recomputed_weight)
        merged = replay_merge(fragments[fragment_id], learner_deltas, weights)
        momentum_after = mu * momentum + merged
        direction = merged + mu * momentum_after
        predicted = current - eta * direction
        observed_params, observed_momentum = read_fragment(post_state, post_meta)
        param_error, _ = check_close(
            predicted,
            observed_params,
            atol=atol,
            rtol=rtol,
            label=f"step {step} params",
        )
        momentum_error, _ = check_close(
            momentum_after,
            observed_momentum,
            atol=atol,
            rtol=rtol,
            label=f"step {step} momentum",
        )
        gnorm = l2(merged)
        step_norm = l2(eta * direction)
        tape_gnorm = float(tape_row["gnorm"])
        tape_step_norm = float(tape_row["outer_step_norm"])
        if (
            not math.isfinite(tape_gnorm)
            or tape_gnorm < 0
            or not math.isfinite(tape_step_norm)
            or tape_step_norm < 0
        ):
            raise ReplayError(f"step {step}: tape norm is nonfinite or negative")
        gnorm_relative = abs(gnorm - tape_gnorm) / max(gnorm, tape_gnorm, 1e-12)
        step_relative = abs(step_norm - tape_step_norm) / max(
            step_norm, tape_step_norm, 1e-12
        )
        if gnorm_relative > tape_rtol or step_relative > tape_rtol:
            raise ReplayError(f"step {step}: tape norm differs from replay")
        for other_fragment in range(4):
            if other_fragment == fragment_id:
                continue
            before_meta = pre_fragments[other_fragment]
            after_meta = post_fragments[other_fragment]
            if before_meta["version"] != after_meta["version"]:
                raise ReplayError(f"step {step}: a non-target fragment version changed")
            before_params, before_momentum = read_fragment(pre_state, before_meta)
            after_params, after_momentum = read_fragment(post_state, after_meta)
            if not np.array_equal(before_params, after_params) or not np.array_equal(
                before_momentum, after_momentum
            ):
                raise ReplayError(f"step {step}: a non-target fragment payload changed")
        maximum_param_error = max(maximum_param_error, param_error)
        maximum_momentum_error = max(maximum_momentum_error, momentum_error)
        maximum_gnorm_relative_error = max(maximum_gnorm_relative_error, gnorm_relative)
        maximum_step_norm_relative_error = max(
            maximum_step_norm_relative_error, step_relative
        )
        steps.append(
            {
                "step": step,
                "fragment": fragment_id,
                "param_max_abs_error": param_error,
                "momentum_max_abs_error": momentum_error,
                "gnorm_relative_error": gnorm_relative,
                "outer_step_norm_relative_error": step_relative,
            }
        )
    capture_registry = {
        path.relative_to(root).as_posix(): registry[path.relative_to(root).as_posix()]
        for path in sorted(capture_files)
    }
    return {
        "cell_id": result["cell_id"],
        "attempt": result["attempt"],
        "commit_count": len(steps),
        "all_steps_replayed": True,
        "capture_file_count": len(capture_registry),
        "capture_registry_sha256": hashlib.sha256(
            canonical_json(capture_registry)
        ).hexdigest(),
        "first_momentum_buffer_exact_zero": True,
        "state_chain_contiguous": True,
        "non_target_fragments_unchanged": True,
        "capture_tape_responder_join_exact": True,
        **barrier_attestation,
        "max_param_abs_error": maximum_param_error,
        "max_momentum_abs_error": maximum_momentum_error,
        "max_gnorm_relative_error": maximum_gnorm_relative_error,
        "max_outer_step_norm_relative_error": maximum_step_norm_relative_error,
        "steps": steps,
    }


def require_numeric_id(value: Any, label: str) -> str:
    rendered = str(value or "")
    if not re.fullmatch(r"[0-9]+", rendered):
        raise ReplayError(f"{label} must be an exact numeric provider ID")
    return rendered


def validate_lifecycle_finalization(
    root: Path,
    final_manifest: dict[str, Any],
    deletion_evidence: Path,
) -> tuple[dict[str, Any], Path]:
    """Bind immutable acquisition bytes to the post-delete final manifest."""
    acquisition_path = root / "phase-map-acquisition-manifest.json"
    acquisition = read_json(acquisition_path)
    if acquisition.get("status") != "sealed_acquisition_pending_teardown":
        raise ReplayError("acquisition manifest is not the immutable pending seal")
    kind = (final_manifest.get("lineage") or {}).get("descendant_kind")
    if kind not in ("p0a_single_gpu_bound", "p0b_four_gpu_bound"):
        raise ReplayError("legacy or non-canary P0 descendant kind is forbidden")
    if (acquisition.get("lineage") or {}).get("descendant_kind") != kind:
        raise ReplayError("acquisition/final descendant kinds differ")
    if final_manifest.get("status") != "sealed_results":
        raise ReplayError("replay requires the post-deletion sealed_results manifest")

    envelope_path = root / "phase-map-lifecycle-seal.json"
    envelope = read_json(envelope_path)
    acquisition_sha = sha256_file(acquisition_path)
    acquisition_canonical = sha256_bytes(canonical_json(acquisition))
    final_path = root / "phase-map-manifest.json"
    if (
        envelope.get("schema") != "yeto_p0_lifecycle_finalization_v1"
        or envelope.get("status") != "SEALED"
        or envelope.get("descendant_kind") != kind
        or envelope.get("acquisition_manifest_sha256") != acquisition_sha
        or envelope.get("acquisition_manifest_canonical_sha256")
        != acquisition_canonical
        or envelope.get("acquisition_checksum_sha256")
        != sha256_file(root / "acquisition.sha256")
        or envelope.get("acquisition_seal_sha256")
        != sha256_file(root / "acquisition-seal.json")
        or envelope.get("deletion_evidence_sha256")
        != sha256_file(deletion_evidence)
        or envelope.get("final_manifest_sha256") != sha256_file(final_path)
        or envelope.get("final_manifest_canonical_sha256")
        != sha256_bytes(canonical_json(final_manifest))
    ):
        raise ReplayError("lifecycle envelope does not bind acquisition/deletion/final")

    transition = {
        "acquisition_status": "sealed_acquisition_pending_teardown",
        "acquisition_manifest_sha256": acquisition_sha,
        "acquisition_manifest_canonical_sha256": acquisition_canonical,
        "acquisition_checksum_sha256": sha256_file(root / "acquisition.sha256"),
        "acquisition_seal_sha256": sha256_file(root / "acquisition-seal.json"),
        "final_manifest_status": "sealed_results",
        "deletion_evidence_sha256": sha256_file(deletion_evidence),
        "finalized_at": envelope.get("finalized_at"),
    }
    teardown_fields = {
        *transition,
        "artifact_sealed_at",
        "deletion_requested_at",
        "deletion_completed_at",
        "instance_not_found_evidence_uri",
        "instance_not_found_evidence_sha256",
        "disk_not_found_evidence_uri",
        "disk_not_found_evidence_sha256",
        "zero_accelerator_evidence_uri",
        "zero_accelerator_evidence_sha256",
    }
    reconstructed = json.loads(json.dumps(final_manifest))
    reconstructed["status"] = "sealed_acquisition_pending_teardown"
    final_rows = reconstructed.get("results")
    acquisition_rows = acquisition.get("results")
    if not isinstance(final_rows, list) or not isinstance(acquisition_rows, list):
        raise ReplayError("P0 lifecycle manifests lack result rows")
    for index, row in enumerate(final_rows):
        hardware = row.get("hardware") if isinstance(row, dict) else None
        if not isinstance(hardware, dict):
            raise ReplayError(f"final results[{index}] lacks hardware")
        if any(hardware.get(key) != value for key, value in transition.items()):
            raise ReplayError(f"final results[{index}] transition hashes differ")
        for uri_field, hash_field in (
            (
                "instance_not_found_evidence_uri",
                "instance_not_found_evidence_sha256",
            ),
            ("disk_not_found_evidence_uri", "disk_not_found_evidence_sha256"),
            (
                "zero_accelerator_evidence_uri",
                "zero_accelerator_evidence_sha256",
            ),
        ):
            relative = hardware.get(uri_field)
            if not isinstance(relative, str) or relative.startswith(("/", "../")):
                raise ReplayError(f"final results[{index}] has unsafe lifecycle URI")
            path = (root / relative).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError as exc:
                raise ReplayError("lifecycle evidence URI escapes artifact root") from exc
            if (
                not path.is_file()
                or path.is_symlink()
                or hardware.get(hash_field) != sha256_file(path)
            ):
                raise ReplayError(
                    f"final results[{index}] lifecycle evidence hash differs"
                )
        for key in teardown_fields:
            hardware.pop(key, None)
    if canonical_json(reconstructed) != canonical_json(acquisition):
        raise ReplayError("final manifest mutates pre-delete scientific acquisition")
    times = [
        parse_time(str(envelope.get(field)))
        for field in (
            "artifact_sealed_at",
            "deletion_requested_at",
            "deletion_completed_at",
            "finalized_at",
        )
    ]
    if not (times[0] <= times[1] < times[2] < times[3]):
        raise ReplayError("lifecycle envelope timestamp order is invalid")
    return acquisition, acquisition_path


def validate_deletion_and_provider_binding(
    root: Path,
    manifest: dict[str, Any],
    acquisition_manifest: dict[str, Any],
    deletion: dict[str, Any],
    registry: dict[str, str],
    *,
    acquisition_path: Path,
    acquisition_manifest_path: Path,
    replay_started: datetime,
) -> dict[str, Any]:
    seal_path = root / "acquisition-seal.json"
    relative_seal = seal_path.relative_to(root).as_posix()
    if registry.get(relative_seal) != sha256_file(seal_path):
        raise ReplayError("acquisition seal metadata is not checksum-sealed")
    seal = read_json(seal_path)
    if (
        seal.get("schema") != "yeto_phase_map_acquisition_seal_v1"
        or seal.get("loss_blind_mechanical_seal") is not True
        or seal.get("phase_map_manifest_sha256")
        != sha256_file(acquisition_manifest_path)
        or seal.get("phase_map_manifest_canonical_sha256")
        != sha256_bytes(canonical_json(acquisition_manifest))
    ):
        raise ReplayError("acquisition seal does not bind the P0 manifest")
    sealed_at = parse_time(str(seal.get("sealed_at_utc")))
    requested_at = parse_time(str(deletion.get("deletion_requested_at_utc")))
    deleted_at = parse_time(str(deletion.get("deleted_at_utc")))
    if not sealed_at <= requested_at < deleted_at < replay_started:
        raise ReplayError(
            "required order is acquisition seal <= deletion request < completion < replay"
        )

    project = str(deletion.get("project", ""))
    zone = str(deletion.get("zone", ""))
    instance_name = str(deletion.get("instance_name", ""))
    boot_disk_name = str(deletion.get("deleted_boot_disk_name", ""))
    artifact_uri = str(deletion.get("artifact_uri", "")).rstrip("/")
    repo_commit = str(deletion.get("repo_commit", ""))
    if not all((project, zone, instance_name, boot_disk_name, artifact_uri, repo_commit)):
        raise ReplayError("deletion evidence lacks exact resource/source coordinates")
    instance_id = require_numeric_id(deletion.get("deleted_instance_id"), "instance ID")
    boot_disk_id = require_numeric_id(
        deletion.get("deleted_boot_disk_id"), "boot disk ID"
    )
    source_image_id = require_numeric_id(
        deletion.get("source_image_id"), "source image ID"
    )
    if repo_commit != (manifest.get("frozen") or {}).get("git_commit"):
        raise ReplayError("deletion evidence repo commit differs from P0 manifest")
    if source_image_id != str((manifest.get("frozen") or {}).get("image_id")):
        raise ReplayError("deleted VM source image differs from P0 frozen image")

    not_found = deletion.get("provider_not_found_verification")
    if not isinstance(not_found, dict):
        raise ReplayError("deletion evidence lacks provider not-found proofs")
    for key, expected_name, expected_id in (
        ("instance", instance_name, instance_id),
        ("boot_disk", boot_disk_name, boot_disk_id),
    ):
        proof = not_found.get(key)
        if (
            not isinstance(proof, dict)
            or proof.get("result") != "NOT_FOUND"
            or str(proof.get("name")) != expected_name
            or str(proof.get("provider_id")) != expected_id
            or parse_time(str(proof.get("verified_at_utc"))) > deleted_at
        ):
            raise ReplayError(f"deletion evidence has invalid exact {key} absence proof")
    accelerator = deletion.get("post_delete_accelerator_proof")
    if (
        not isinstance(accelerator, dict)
        or accelerator.get("project") != project
        or accelerator.get("campaign_owned_accelerators") != 0
        or not isinstance(accelerator.get("total_active_accelerators"), int)
        or accelerator.get("total_active_accelerators", -1) < 0
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(accelerator.get("inventory_sha256", ""))
        )
        or parse_time(str(accelerator.get("queried_at_utc"))) > deleted_at
    ):
        raise ReplayError(
            "deletion evidence lacks a zero campaign-owned accelerator proof"
        )

    artifact_seal = deletion.get("artifact_object_seal")
    objects = artifact_seal.get("objects") if isinstance(artifact_seal, dict) else None
    if (
        not isinstance(artifact_seal, dict)
        or artifact_seal.get("schema")
        != "optimizer_harness_artifact_object_seal_v1"
        or not isinstance(objects, list)
    ):
        raise ReplayError("deletion evidence lacks immutable GCS object generations")
    if not (
        sealed_at
        <= parse_time(str(artifact_seal.get("sealed_at_utc")))
        <= requested_at
    ):
        raise ReplayError("GCS object generation seal has an invalid lifecycle time")
    by_role = {
        str(item.get("role")): item for item in objects if isinstance(item, dict)
    }
    for role, filename, expected_hash in (
        (
            "phase_map_manifest",
            "phase-map-manifest.json",
            sha256_file(acquisition_manifest_path),
        ),
        ("acquisition_checksum", "acquisition.sha256", sha256_file(acquisition_path)),
    ):
        item = by_role.get(role)
        if (
            not isinstance(item, dict)
            or item.get("uri") != f"{artifact_uri}/{filename}"
            or item.get("sha256") != expected_hash
            or not re.fullmatch(r"[0-9]+", str(item.get("generation", "")))
            or not re.fullmatch(r"[0-9]+", str(item.get("metageneration", "")))
            or not isinstance(item.get("size"), int)
            or item["size"] <= 0
        ):
            raise ReplayError(f"GCS object seal does not bind exact {role}")

    evidence_files: dict[str, Path] = {}
    evidence_root = root / "provider-evidence"
    for path in evidence_root.rglob("*.json"):
        digest = sha256_file(path)
        relative = path.relative_to(root).as_posix()
        if registry.get(relative) == digest:
            evidence_files[digest] = path
    bound_evidence: dict[str, dict[str, Any]] = {}
    prior_lifecycle_records = deletion.get("prior_provider_lifecycle_proofs", [])
    if not isinstance(prior_lifecycle_records, list):
        raise ReplayError("prior_provider_lifecycle_proofs must be an array")
    prior_lifecycle = {
        str(record.get("provisioning_evidence_sha256")): record
        for record in prior_lifecycle_records
        if isinstance(record, dict)
    }
    for result in manifest.get("results", []):
        hardware = result.get("hardware") or {}
        digest = str(hardware.get("provisioning_evidence_sha256", ""))
        path = evidence_files.get(digest)
        if path is None:
            raise ReplayError("result provider evidence is absent from acquisition seal")
        evidence = read_json(path)
        immutable_expected = {
            "provider": "gcp",
            "market": "spot",
            "project": project,
            "zone": zone,
            "source_image_id": source_image_id,
        }
        if any(
            str(evidence.get(key)) != str(value)
            for key, value in immutable_expected.items()
        ):
            raise ReplayError("per-attempt provider evidence differs from frozen campaign")
        for key in (
            "project",
            "zone",
            "instance_name",
            "instance_id",
            "boot_disk_id",
            "source_image_id",
        ):
            if str(hardware.get(key)) != str(evidence.get(key)):
                raise ReplayError(f"result hardware {key} differs from its provider snapshot")
        is_final_deleted_resource = (
            str(evidence.get("instance_name")) == instance_name
            and str(evidence.get("instance_id")) == instance_id
            and str(evidence.get("boot_disk_id")) == boot_disk_id
        )
        lifecycle = "final_deletion_evidence"
        if not is_final_deleted_resource:
            record = prior_lifecycle.get(digest)
            if not isinstance(record, dict):
                raise ReplayError("historical attempt lacks exact provider lifecycle proof")
            lifecycle_path = (root / str(record.get("lifecycle_evidence_path", ""))).resolve()
            try:
                lifecycle_relative = lifecycle_path.relative_to(root.resolve()).as_posix()
            except ValueError as exc:
                raise ReplayError("historical lifecycle proof escapes acquisition root") from exc
            if (
                not lifecycle_path.is_file()
                or lifecycle_path.is_symlink()
                or registry.get(lifecycle_relative) != sha256_file(lifecycle_path)
                or record.get("lifecycle_evidence_sha256")
                != sha256_file(lifecycle_path)
            ):
                raise ReplayError("historical provider lifecycle proof is not sealed")
            lifecycle_evidence = read_json(lifecycle_path)
            if (
                lifecycle_evidence.get("verified_instance_absent") is not True
                or lifecycle_evidence.get("verified_boot_disk_absent") is not True
                or str(lifecycle_evidence.get("deleted_instance_id"))
                != str(evidence.get("instance_id"))
                or str(lifecycle_evidence.get("deleted_boot_disk_id"))
                != str(evidence.get("boot_disk_id"))
            ):
                raise ReplayError("historical provider lifecycle IDs/absence differ")
            lifecycle = lifecycle_relative
        bound_evidence[digest] = {
            "sha256": digest,
            "sealed_path": path.relative_to(root).as_posix(),
            "instance_id": str(evidence["instance_id"]),
            "boot_disk_id": str(evidence["boot_disk_id"]),
            "lifecycle_proof": lifecycle,
        }
    return {
        "acquisition_sealed_at_utc": seal["sealed_at_utc"],
        "deletion_requested_at_utc": deletion["deletion_requested_at_utc"],
        "deleted_at_utc": deletion["deleted_at_utc"],
        "project": project,
        "zone": zone,
        "instance_name": instance_name,
        "instance_id": instance_id,
        "boot_disk_name": boot_disk_name,
        "boot_disk_id": boot_disk_id,
        "source_image_id": source_image_id,
        "artifact_uri": artifact_uri,
        "provider_evidence": [bound_evidence[key] for key in sorted(bound_evidence)],
        "artifact_object_seal": artifact_seal,
    }


def validate(
    root: Path,
    deletion_evidence: Path,
    output: Path,
    *,
    atol: float,
    rtol: float,
    tape_rtol: float,
    parent_manifest_path: Path | None = None,
    parent_replay_report_path: Path | None = None,
    expected_parent_replay_report_sha256: str | None = None,
    source_verifier: Any = verify_replay_source,
    manifest_validator: Any = validate_phase_manifest,
) -> dict[str, Any]:
    if (atol, rtol, tape_rtol) != (PARAM_ATOL, PARAM_RTOL, TAPE_NORM_RTOL):
        raise ReplayError(
            "P0 replay tolerances are frozen in source and may not be overridden"
        )
    started = datetime.now(timezone.utc)
    deletion = read_json(deletion_evidence)
    if (
        deletion.get("status") != "DELETED"
        or deletion.get("verified_instance_absent") is not True
        or deletion.get("verified_boot_disk_absent") is not True
    ):
        raise ReplayError("CPU replay requires verified exact VM and disk deletion")
    acquisition = root / "acquisition.sha256"
    registry = verify_acquisition(root, acquisition)
    manifest_path = root / "phase-map-manifest.json"
    manifest = read_json(manifest_path)
    kind = manifest.get("lineage", {}).get("descendant_kind")
    if kind not in ("p0a_single_gpu_bound", "p0b_four_gpu_bound"):
        raise ReplayError("legacy or non-canary P0 descendant kind is forbidden")
    acquisition_manifest, acquisition_manifest_path = (
        validate_lifecycle_finalization(root, manifest, deletion_evidence)
    )
    source_attestation = source_verifier(manifest)
    if not isinstance(source_attestation, dict):
        raise ReplayError("replay source verifier did not return an attestation")
    if kind == "p0b_four_gpu_bound":
        if (
            parent_manifest_path is None
            or parent_replay_report_path is None
            or expected_parent_replay_report_sha256 is None
        ):
            raise ReplayError("P0b replay requires exact P0a parent and replay report")
        parent_manifest = read_json(parent_manifest_path)
        parent_replay = read_json(parent_replay_report_path)
        report_sha = sha256_file(parent_replay_report_path)
        if report_sha != expected_parent_replay_report_sha256:
            raise ReplayError("P0a replay raw hash differs from explicit authority")
        if manifest_validator is validate_phase_manifest:
            integrity_attestation = manifest_validator(
                manifest,
                parent_manifest=parent_manifest,
                parent_replay_report=parent_replay,
                parent_replay_report_sha256=report_sha,
            )
        else:
            integrity_attestation = manifest_validator(manifest)
    else:
        if any(
            value is not None
            for value in (
                parent_manifest_path,
                parent_replay_report_path,
                expected_parent_replay_report_sha256,
            )
        ):
            raise ReplayError("P0a is parentless and rejects parent replay inputs")
        integrity_attestation = manifest_validator(manifest)
    if not isinstance(integrity_attestation, dict):
        raise ReplayError("phase-map validator did not return an attestation")
    deletion_binding = validate_deletion_and_provider_binding(
        root,
        manifest,
        acquisition_manifest,
        deletion,
        registry,
        acquisition_path=acquisition,
        acquisition_manifest_path=acquisition_manifest_path,
        replay_started=started,
    )
    expected_cells_raw = manifest.get("expected_cells")
    command_hashes = (manifest.get("frozen") or {}).get("cell_command_hashes")
    if not isinstance(expected_cells_raw, list) or not expected_cells_raw:
        raise ReplayError("P0 manifest lacks explicit expected cells")
    expected_cells = {
        str(cell.get("cell_id")): cell
        for cell in expected_cells_raw
        if isinstance(cell, dict)
    }
    if len(expected_cells) != len(expected_cells_raw) or not isinstance(
        command_hashes, dict
    ) or set(command_hashes) != set(expected_cells):
        raise ReplayError("P0 expected-cell/command registry is malformed")
    results_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    replayed_attempts = []
    for result in manifest.get("results", []):
        cell_id = str(result.get("cell_id", ""))
        if cell_id not in expected_cells:
            raise ReplayError("P0 result refers to an unregistered cell")
        results_by_cell[cell_id].append(result)
        status = result.get("status")
        if status in ("COMPLETED", "DIVERGED"):
            replayed_attempts.append(
                replay_cell(
                    root,
                    result,
                    expected_cells[cell_id],
                    str(command_hashes[cell_id]),
                    registry,
                    atol=atol,
                    rtol=rtol,
                    tape_rtol=tape_rtol,
                )
            )
        elif status != "INFRA_FAILURE":
            raise ReplayError("P0 contains an unresolved/non-scientific result")
    if not replayed_attempts:
        raise ReplayError("P0 manifest has no replayable cells")
    cells = []
    for cell_id in sorted(expected_cells):
        attempts = sorted(
            results_by_cell.get(cell_id, []), key=lambda row: int(row.get("attempt", 0))
        )
        if not attempts or attempts[-1].get("status") not in ("COMPLETED", "DIVERGED"):
            raise ReplayError("every P0 expected cell needs a final scientific attempt")
        replay_rows = [row for row in replayed_attempts if row["cell_id"] == cell_id]
        if not replay_rows:
            raise ReplayError("final P0 cell has no replayed scientific attempt")
        cells.append(
            {
                "cell_id": cell_id,
                "all_steps_replayed": True,
                "final_attempt": attempts[-1]["attempt"],
                "replayed_attempt_count": len(replay_rows),
                "replayed_attempts": replay_rows,
            }
        )
    report = {
        "schema": "yeto_p0_cpu_replay_v1",
        "status": "PASS",
        "gpu_deleted_before_replay": True,
        "replay_started_at_utc": started.isoformat().replace("+00:00", "Z"),
        "replay_completed_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "acquisition_sealed_at_utc": deletion_binding[
            "acquisition_sealed_at_utc"
        ],
        "deletion_requested_at_utc": deletion["deletion_requested_at_utc"],
        "deleted_at_utc": deletion["deleted_at_utc"],
        "deleted_instance_id": deletion["deleted_instance_id"],
        "deleted_boot_disk_id": deletion["deleted_boot_disk_id"],
        "deletion_evidence_sha256": sha256_file(deletion_evidence),
        "acquisition_manifest_sha256": sha256_file(acquisition),
        "acquisition_phase_map_manifest_sha256": sha256_file(
            acquisition_manifest_path
        ),
        "acquisition_phase_map_manifest_canonical_sha256": sha256_bytes(
            canonical_json(acquisition_manifest)
        ),
        "lifecycle_finalization_sha256": sha256_file(
            root / "phase-map-lifecycle-seal.json"
        ),
        "phase_map_manifest_sha256": sha256_file(manifest_path),
        "phase_map_manifest_canonical_sha256": sha256_bytes(
            canonical_json(manifest)
        ),
        **source_attestation,
        **integrity_attestation,
        "deletion_and_provider_binding": deletion_binding,
        "frozen_tolerance": {
            "param_atol": atol,
            "param_rtol": rtol,
            "tape_norm_rtol": tape_rtol,
            "replay_dtype": "numpy_little_endian_f32_with_f64_norm_accumulation",
        },
        "cell_count": len(cells),
        "replayed_scientific_attempt_count": len(replayed_attempts),
        "all_steps_replayed": True,
        "cells": cells,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{sha256_file(output)}  {output.name}\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--deletion-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--parent-replay-report", type=Path)
    parser.add_argument("--expected-parent-replay-report-sha256")
    parser.add_argument("--param-atol", type=float, default=PARAM_ATOL)
    parser.add_argument("--param-rtol", type=float, default=PARAM_RTOL)
    parser.add_argument("--tape-norm-rtol", type=float, default=TAPE_NORM_RTOL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate(
            args.run_root.resolve(),
            args.deletion_evidence.resolve(),
            args.output.resolve(),
            atol=args.param_atol,
            rtol=args.param_rtol,
            tape_rtol=args.tape_norm_rtol,
            parent_manifest_path=(
                None if args.parent_manifest is None else args.parent_manifest.resolve()
            ),
            parent_replay_report_path=(
                None
                if args.parent_replay_report is None
                else args.parent_replay_report.resolve()
            ),
            expected_parent_replay_report_sha256=(
                args.expected_parent_replay_report_sha256
            ),
        )
    except (OSError, ValueError, ReplayError) as exc:
        print(f"P0 replay error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
