"""Strict terminal reconciler for the two-island Milestone-1 dense run.

The launcher deliberately treats a zero container exit as insufficient.  This
module joins the independently persisted learner, rollout, optimizer,
publication, metric-history, syncer, placement, and daemon-cleanup evidence
before publishing one privacy-safe terminal report.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

FINAL_REPORT_SCHEMA = "yeto-m1-dense-final-report-v1"
EVAL_SUMMARY_SCHEMA = "yeto-m1-dense-heldout-summary-v1"
_STARTED_SCHEMA = "yeto-m1-dense-full-island-started-v1"
_PLACEMENT_SCHEMA = "yeto-m1-dense-full-ray-placement-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TRAIN_METRICS = ("train/grad_norm", "train/train_rollout_kl")


class FinalReportError(RuntimeError):
    """Terminal evidence does not prove one complete M1 run."""


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FinalReportError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FinalReportError(f"{name} must be a lowercase SHA256")
    return value


def _expect_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FinalReportError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str, *, unit: bool = False) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise FinalReportError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (unit and not 0.0 <= result <= 1.0):
        raise FinalReportError(f"{name} is outside its finite range")
    return result


def _safe_path(path: Path, root: Path, name: str, *, directory: bool = False) -> None:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise FinalReportError(f"{name} is missing") from exc
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise FinalReportError(f"{name} escapes the run root")
    current = path
    while True:
        try:
            current.lstat()
        except OSError as exc:
            raise FinalReportError(f"{name} is unreadable") from exc
        if current.is_symlink():
            raise FinalReportError(f"{name} traverses a symlink")
        if current == root:
            break
        parent = current.parent
        if parent == current:
            raise FinalReportError(f"{name} is outside the run root")
        current = parent
    if directory:
        if not path.is_dir():
            raise FinalReportError(f"{name} is not a directory")
    elif not path.is_file():
        raise FinalReportError(f"{name} is not a regular file")


def _load_json(
    path: Path,
    root: Path,
    name: str,
    *,
    canonical: bool = False,
    private: bool = True,
) -> dict[str, Any]:
    _safe_path(path, root, name)
    if private and path.stat().st_mode & 0o077:
        raise FinalReportError(f"{name} is not private")
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalReportError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise FinalReportError(f"{name} must contain an object")
    if canonical and raw != _canonical(value):
        raise FinalReportError(f"{name} is not canonical JSON")
    return value


def _json_lines(
    path: Path,
    root: Path,
    name: str,
    *,
    tolerate_invalid: bool = False,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    _safe_path(path, root, name)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FinalReportError(f"{name} is unreadable") from exc
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.endswith(b"\n"):
            if tolerate_invalid:
                continue
            raise FinalReportError(f"{name} has a torn row {line_number}")
        try:
            value = json.loads(line, object_pairs_hook=_unique_object)
        except (UnicodeError, json.JSONDecodeError, FinalReportError) as exc:
            if tolerate_invalid:
                continue
            raise FinalReportError(f"{name} has invalid row {line_number}") from exc
        if not isinstance(value, dict):
            if tolerate_invalid:
                continue
            raise FinalReportError(f"{name} row {line_number} is not an object")
        values.append(value)
    if not values and not allow_empty:
        raise FinalReportError(f"{name} contains no complete records")
    return values


def _tree_sha256(root: Path, run_root: Path, name: str) -> str:
    _safe_path(root, run_root, name, directory=True)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise FinalReportError(f"{name} is empty")
    digest = hashlib.sha256(b"yeto-m1-evidence-tree-v1\0")
    for path in files:
        _safe_path(path, run_root, name)
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _metric_series(
    directory: Path,
    run_root: Path,
    metric: str,
    rounds: int,
) -> list[list[int | float]]:
    _safe_path(directory, run_root, "Miles metric history", directory=True)
    candidates: list[list[list[int | float]]] = []
    for path in sorted(directory.glob("*.jsonl")):
        for row in _json_lines(
            path, run_root, "Miles metric history", allow_empty=True
        ):
            if set(row) != {"metric", "series"} or not isinstance(row["series"], list):
                raise FinalReportError("Miles metric history row is malformed")
            if row["metric"] != metric:
                continue
            points: list[list[int | float]] = []
            for point in row["series"]:
                if not isinstance(point, list) or len(point) != 2:
                    raise FinalReportError(f"Miles {metric} series is malformed")
                step = _expect_int(point[0], f"Miles {metric} step")
                points.append([step, _finite(point[1], f"Miles {metric} value")])
            candidates.append(points)
    if not candidates:
        raise FinalReportError(f"Miles did not persist required metric {metric}")
    unique = {_canonical(points) for points in candidates}
    if len(unique) != 1:
        raise FinalReportError(f"Miles metric writers disagree for {metric}")
    points = candidates[0]
    if [point[0] for point in points] != list(range(rounds)):
        raise FinalReportError(f"Miles {metric} does not cover every training round")
    return points


def _validate_island_shell_evidence(
    manifest: dict[str, Any], manifest_sha256: str, island_id: int, run_root: Path
) -> dict[str, str]:
    island = manifest["topology"]["islands"][island_id]
    island_root = Path(island["host_run_dir"])
    _safe_path(island_root, run_root, f"island {island_id} evidence", directory=True)
    exit_path = island_root / "learner.exit"
    _safe_path(exit_path, run_root, f"island {island_id} exit evidence")
    if exit_path.read_bytes() != b"0\n":
        raise FinalReportError(f"island {island_id} learner did not exit successfully")
    started = _load_json(
        island_root / "container-started.json",
        run_root,
        f"island {island_id} startup marker",
        canonical=True,
    )
    if started != {
        "schema": _STARTED_SCHEMA,
        "island_id": island_id,
        "manifest_sha256": manifest_sha256,
    }:
        raise FinalReportError(f"island {island_id} startup identity differs")
    placement = _load_json(
        island_root / "ray-placement.json",
        run_root,
        f"island {island_id} placement evidence",
        canonical=True,
    )
    if placement != {
        "schema": _PLACEMENT_SCHEMA,
        "island_id": island_id,
        "manifest_sha256": manifest_sha256,
        "actor_bundle_indices": [0, 1],
        "actor_local_gpu_ids": [0, 1],
        "actor_gpu_uuids": island["trainer_gpu_uuids"],
        "inference_bundle_indices": [2, 3],
        "inference_local_gpu_ids": [2, 3],
        "inference_gpu_uuids": island["inference_gpu_uuids"],
        "inference_engine_count": 2,
        "inference_tp": 1,
    }:
        raise FinalReportError(f"island {island_id} placement identity differs")
    return {
        "startup": _sha256(island_root / "container-started.json"),
        "ray_placement": _sha256(island_root / "ray-placement.json"),
    }


def _eval_summary(
    manifest: dict[str, Any], island_id: int, run_root: Path
) -> tuple[dict[int, dict[str, Any]], str]:
    evaluation = manifest["evaluation"]
    path = Path(evaluation["island_summary_paths"][island_id])
    value = _load_json(
        path,
        run_root,
        f"island {island_id} heldout summary",
        canonical=True,
    )
    expected_versions = [0, manifest["rounds"]]
    if (
        set(value)
        != {
            "schema",
            "island_id",
            "dataset_name",
            "prompt_count",
            "samples_per_prompt",
            "expected_policy_versions",
            "complete",
            "results",
        }
        or value["schema"] != EVAL_SUMMARY_SCHEMA
        or value["island_id"] != island_id
        or value["dataset_name"] != evaluation["dataset_name"]
        or value["prompt_count"] != evaluation["prompt_count"]
        or value["samples_per_prompt"] != evaluation["samples_per_prompt"]
        or value["expected_policy_versions"] != expected_versions
        or value["complete"] is not True
        or not isinstance(value["results"], list)
        or len(value["results"]) != 2
    ):
        raise FinalReportError(f"island {island_id} heldout summary differs")
    by_version: dict[int, dict[str, Any]] = {}
    expected_samples = evaluation["prompt_count"] * evaluation["samples_per_prompt"]
    for expected_version, row in zip(expected_versions, value["results"], strict=True):
        if not isinstance(row, dict) or set(row) != {
            "policy_version",
            "policy_hash",
            "sample_count",
            "result",
            "pass_at_1",
        }:
            raise FinalReportError("heldout result row is malformed")
        if (
            row["policy_version"] != expected_version
            or row["sample_count"] != expected_samples
        ):
            raise FinalReportError("heldout result identity differs")
        _expect_sha(row["policy_hash"], "heldout policy hash")
        normalized = {
            **row,
            "result": _finite(row["result"], "heldout result", unit=True),
            "pass_at_1": _finite(row["pass_at_1"], "heldout pass@1", unit=True),
        }
        by_version[expected_version] = normalized
    return by_version, _sha256(path)


def _learner_events(
    manifest: dict[str, Any], island_id: int, run_root: Path
) -> tuple[
    list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]], str
]:
    rounds = manifest["rounds"]
    path = (
        Path(manifest["topology"]["islands"][island_id]["host_run_dir"])
        / "learner-events.jsonl"
    )
    events = _json_lines(path, run_root, f"island {island_id} learner event tape")
    if any(event.get("event") == "rl_strict_failure" for event in events):
        raise FinalReportError(f"island {island_id} recorded a strict RL failure")
    publications = [
        row for row in events if row.get("event") == "rl_dense_policy_publication"
    ]
    local_steps = [row for row in events if row.get("event") == "rl_dense_local_step"]
    if len(publications) != rounds + 1 or len(local_steps) != rounds:
        raise FinalReportError(f"island {island_id} has an incomplete policy lifecycle")
    pubs: dict[int, dict[str, Any]] = {}
    for expected_version, row in enumerate(publications):
        if (
            row.get("island_id") != island_id
            or row.get("policy_version") != expected_version
            or row.get("terminal") is not (expected_version == rounds)
        ):
            raise FinalReportError(f"island {island_id} publication order differs")
        policy_hash = _expect_sha(
            row.get("sync/global_policy_hash"), "published policy hash"
        )
        pubs[expected_version] = {
            "policy_version": expected_version,
            "policy_hash": policy_hash,
        }
    steps: dict[int, dict[str, Any]] = {}
    expected_trajectories = (
        manifest["profile"]["groups_per_round"]
        * manifest["profile"]["samples_per_group"]
    )
    for expected_base, row in enumerate(local_steps):
        if (
            row.get("island_id") != island_id
            or row.get("base_policy_version") != expected_base
            or row.get("target_policy_version") != expected_base + 1
            or row.get("base_policy_hash") != pubs[expected_base]["policy_hash"]
            or row.get("target_policy_hash") != pubs[expected_base + 1]["policy_hash"]
            or row.get("trajectory_count") != expected_trajectories
            or row.get("optimizer_steps") != 1
        ):
            raise FinalReportError(f"island {island_id} local-step receipt differs")
        _expect_sha(row.get("input_batch_hash"), "local input batch hash")
        _expect_sha(row.get("sweep_update_id"), "local sweep update identity")
        _expect_int(row.get("trained_tokens"), "local trained tokens", minimum=1)
        steps[expected_base] = row
    return events, pubs, steps, _sha256(path)


def _validate_eval_events(
    manifest: dict[str, Any],
    island_id: int,
    events: list[dict[str, Any]],
    evaluation: dict[int, dict[str, Any]],
) -> None:
    expected_versions = [0, manifest["rounds"]]
    rows = [row for row in events if row.get("event") == "rl_eval_result"]
    if len(rows) != len(expected_versions):
        raise FinalReportError(f"island {island_id} has incomplete heldout events")
    for version, row in zip(expected_versions, rows, strict=True):
        expected = evaluation[version]
        if (
            row.get("island_id") != island_id
            or row.get("rollout_id") != version
            or row.get("policy_version") != version
            or row.get("sync/global_policy_hash") != expected["policy_hash"]
            or row.get("dataset_name") != manifest["evaluation"]["dataset_name"]
            or row.get("sample_count") != expected["sample_count"]
            or _finite(row.get("rl/eval/result"), "heldout event result", unit=True)
            != expected["result"]
            or _finite(row.get("rl/eval/pass_at_1"), "heldout event pass@1", unit=True)
            != expected["pass_at_1"]
        ):
            raise FinalReportError(f"island {island_id} heldout event differs")


def _trajectory_rounds(
    manifest: dict[str, Any],
    island_id: int,
    publications: dict[int, dict[str, Any]],
    local_steps: dict[int, dict[str, Any]],
    run_root: Path,
) -> tuple[list[dict[str, Any]], str]:
    from yeto.rl.trajectory_evidence import read_trajectory_batch_evidence

    matches = [
        Path(value)
        for value in sorted(
            glob.glob(manifest["evaluation"]["trajectory_evidence_globs"][island_id])
        )
    ]
    if len(matches) != 1:
        raise FinalReportError(
            f"island {island_id} has no unique trajectory evidence root"
        )
    root = matches[0]
    _safe_path(root, run_root, f"island {island_id} trajectory root", directory=True)
    rounds = manifest["rounds"]
    paths = sorted(root.glob("rollout-*.json"))
    expected_paths = [root / f"rollout-{version:08d}.json" for version in range(rounds)]
    if paths != expected_paths:
        raise FinalReportError(f"island {island_id} trajectory schedule differs")
    expected_groups = manifest["profile"]["groups_per_round"]
    expected_samples = manifest["profile"]["samples_per_group"]
    result: list[dict[str, Any]] = []
    for version, path in enumerate(paths):
        _safe_path(path, run_root, f"island {island_id} trajectory evidence")
        try:
            evidence = read_trajectory_batch_evidence(path)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise FinalReportError(
                "trajectory evidence failed semantic verification"
            ) from exc
        step = local_steps[version]
        if (
            evidence.rollout_id != version
            or evidence.behavior_policy_hash != publications[version]["policy_hash"]
            or evidence.input_batch_hash != step["input_batch_hash"]
            or evidence.trained_tokens != step["trained_tokens"]
            or len(evidence.envelopes) != expected_groups * expected_samples
        ):
            raise FinalReportError(f"island {island_id} trajectory batch differs")
        groups: dict[str, int] = {}
        rewards: list[float] = []
        trajectory_ids: set[str] = set()
        sample_indices: set[int] = set()
        for envelope in evidence.envelopes:
            if (
                envelope.behavior_policy_version != version
                or envelope.behavior_policy_hash != publications[version]["policy_hash"]
                or re.fullmatch(rf"r{version}:g[0-9]+", envelope.prompt_group_id)
                is None
                or envelope.reward_contract_hash != manifest["profile"]["reward_sha256"]
                or envelope.trajectory_id in trajectory_ids
                or envelope.sample_index in sample_indices
            ):
                raise FinalReportError("trajectory policy/reward identity differs")
            _expect_sha(envelope.cleanup_evidence_hash, "trajectory cleanup evidence")
            trajectory_ids.add(envelope.trajectory_id)
            sample_indices.add(envelope.sample_index)
            groups[envelope.prompt_group_id] = (
                groups.get(envelope.prompt_group_id, 0) + 1
            )
            rewards.append(_finite(envelope.reward, "trajectory reward", unit=True))
        if len(groups) != expected_groups or set(groups.values()) != {expected_samples}:
            raise FinalReportError("trajectory GRPO grouping differs")
        result.append(
            {
                "round": version + 1,
                "base_policy_version": version,
                "target_policy_version": version + 1,
                "target_policy_hash": publications[version + 1]["policy_hash"],
                "groups": expected_groups,
                "accepted_trajectories": len(rewards),
                "trained_tokens": evidence.trained_tokens,
                "optimizer_steps": step["optimizer_steps"],
                "reward_mean": sum(rewards) / len(rewards),
                "reward_min": min(rewards),
                "reward_max": max(rewards),
                "reward_success_count": sum(value == 1.0 for value in rewards),
            }
        )
    return result, _tree_sha256(root, run_root, "trajectory evidence")


def _syncer_summary(
    manifest: dict[str, Any],
    island_steps: list[dict[int, dict[str, Any]]],
    run_root: Path,
) -> tuple[dict[str, Any], str]:
    path = run_root / "syncer" / "events.jsonl"
    rows = _json_lines(path, run_root, "syncer event tape", tolerate_invalid=True)
    total_steps = manifest["syncer"]["total_steps"]
    fragments = manifest["model"]["fragment_count"]
    rounds = manifest["rounds"]
    roster = list(range(manifest["topology"]["island_count"]))
    merges = sorted(
        (
            row
            for row in rows
            if "step" in row and row.get("event") != "policy_sweep_ledger"
        ),
        key=lambda row: row.get("step", -1),
    )
    if [row.get("step") for row in merges] != list(range(1, total_steps + 1)):
        raise FinalReportError("syncer tape does not contain every unique merge")
    norms = []
    # The sync protocol fingerprint covers its FragmentLayout wire encoding;
    # it is deliberately distinct from ParameterLayout.layout_hash (which
    # also binds algorithm/components/shard metadata).  The syncer makes the
    # former a fixed-roster handshake invariant, while the metadata probe and
    # manifest bind the latter.
    layout_hash = _expect_sha(
        merges[0].get("sync/layout_hash"), "syncer fragment-layout hash"
    )
    expected_members = [{"id": learner_id, "generation": 0} for learner_id in roster]
    for step, row in enumerate(merges, 1):
        policy_round = (step - 1) // fragments + 1
        fragment = (step - 1) % fragments
        expected_base = (
            0 if policy_round == 1 else (policy_round - 2) * fragments + fragment + 1
        )
        responders = sorted(
            row.get("responders", []), key=lambda item: item.get("id", -1)
        )
        if (
            row.get("fragment") != fragment
            or row.get("protocol_version") != 4
            or row.get("delta_semantics") != "local_minus_raw_anchor"
            or row.get("attempt") != 1
            or row.get("policy_round") != policy_round
            or row.get("sweep_fragment") != fragment
            or row.get("sweep_fragments") != fragments
            or row.get("sweep_complete") is not (fragment == fragments - 1)
            or row.get("launch_base_version") != expected_base
            or row.get("sync/base_version") != expected_base
            or row.get("sync/layout_hash") != layout_hash
            or row.get("sync/quorum") != len(roster)
            or row.get("quorum") != len(roster)
            or row.get("sync/responders") != len(roster)
            or row.get("expected") != roster
            or row.get("responded") != roster
            or row.get("expected_members") != expected_members
            or row.get("responded_members") != expected_members
            or row.get("missed_grace") != []
            or row.get("missed_members") != []
            or row.get("sync/rejected_stale_updates") != 0
            or len(responders) != len(roster)
        ):
            raise FinalReportError(f"syncer merge {step} violates the fixed roster")
        global_norm = _finite(
            row.get("sync/global_delta_norm"), "syncer global delta norm"
        )
        if (
            _finite(row.get("gnorm"), "syncer duplicate global delta norm")
            != global_norm
        ):
            raise FinalReportError(f"syncer merge {step} norm fields disagree")
        for learner_id, responder in enumerate(responders):
            tokens = island_steps[learner_id][policy_round - 1]["trained_tokens"]
            if (
                responder.get("id") != learner_id
                or responder.get("generation") != 0
                or responder.get("base_version") != expected_base
                or responder.get("staleness") != 0
                or responder.get("c_steps") != 1
                or responder.get("c_tokens") != tokens
                or responder.get("accounted_c_steps")
                != (1 if fragment == fragments - 1 else 0)
                or responder.get("accounted_c_tokens")
                != (tokens if fragment == fragments - 1 else 0)
                or _finite(responder.get("weight"), "syncer learner weight") != 1.0
                or _finite(responder.get("contribution"), "syncer contribution")
                != 1.0 / len(roster)
            ):
                raise FinalReportError(f"syncer merge {step} progress differs")
        norms.append(
            {
                "round": policy_round,
                "fragment": fragment,
                "global_delta_norm": _finite(global_norm, "syncer global delta norm"),
            }
        )
        if norms[-1]["global_delta_norm"] < 0:
            raise FinalReportError("syncer global delta norm is negative")
    ledgers = [row for row in rows if row.get("event") == "policy_sweep_ledger"]
    complete = [
        row
        for row in ledgers
        if row.get("phase") == "complete" and row.get("global_step") == total_steps
    ]
    if len(complete) != 1:
        raise FinalReportError("syncer has no unique terminal policy ledger")
    ledger = complete[0]
    expected_versions = list(range(total_steps - fragments + 1, total_steps + 1))
    if (
        ledger.get("event_id")
        != f"policy-sweep-ledger:{layout_hash}:complete:{total_steps}"
        or ledger.get("protocol_version") != 4
        or ledger.get("sync/layout_hash") != layout_hash
        or ledger.get("policy_round") != rounds
        or ledger.get("sweep_fragments") != fragments
        or ledger.get("sweep_complete") is not True
        or ledger.get("versions") != expected_versions
        or not isinstance(ledger.get("ledger"), list)
    ):
        raise FinalReportError("syncer terminal policy ledger differs")
    expected_ledger = [
        {
            "id": learner_id,
            "merges": total_steps,
            "steps": rounds,
            "tokens": sum(
                row["trained_tokens"] for row in island_steps[learner_id].values()
            ),
        }
        for learner_id in roster
    ]
    if sorted(ledger["ledger"], key=lambda row: row.get("id", -1)) != expected_ledger:
        raise FinalReportError("syncer terminal token/step ledger differs")
    return (
        {
            "layout_hash": layout_hash,
            "total_steps": total_steps,
            "policy_sweep_fragments": fragments,
            "full_roster": roster,
            "merge_count": len(merges),
            "global_delta_norms": norms,
            "terminal_fragment_versions": expected_versions,
            "ledger": expected_ledger,
        },
        _sha256(path),
    )


def _ordered_hash(value: Any, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical(value)).hexdigest()


def build_final_report(
    manifest: dict[str, Any],
    manifest_sha256: str,
    *,
    daemon_health: dict[str, Any],
) -> dict[str, Any]:
    """Reconcile final-mode artifacts without copying task or token identities."""

    _expect_sha(manifest_sha256, "manifest")
    if manifest.get("launch_mode") != "two-island-final":
        raise FinalReportError("terminal report is only defined for two-island-final")
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("enabled") is not True:
        raise FinalReportError("terminal report requires heldout evaluation")
    if (
        daemon_health.get("ok") is not True
        or daemon_health.get("active_episodes") != 0
        or daemon_health.get("task_pack_sha256")
        != manifest["secrlenv"]["task_pack_sha256"]
    ):
        raise FinalReportError("SecRLEnv has not reached authenticated cleanup zero")
    run_root = Path(manifest["launch"]["host_run_root"])
    _safe_path(run_root, run_root, "run root", directory=True)
    island_rows = []
    island_steps: list[dict[int, dict[str, Any]]] = []
    all_publications: list[dict[int, dict[str, Any]]] = []
    input_hash_rows = []
    for island_id in range(manifest["topology"]["island_count"]):
        shell_evidence = _validate_island_shell_evidence(
            manifest, manifest_sha256, island_id, run_root
        )
        eval_by_version, eval_sha = _eval_summary(manifest, island_id, run_root)
        events, publications, local_steps, events_sha = _learner_events(
            manifest, island_id, run_root
        )
        _validate_eval_events(manifest, island_id, events, eval_by_version)
        round_rows, trajectory_sha = _trajectory_rounds(
            manifest, island_id, publications, local_steps, run_root
        )
        metric_dir = Path(evaluation["metric_history_dirs"][island_id])
        metrics = {
            metric: _metric_series(metric_dir, run_root, metric, manifest["rounds"])
            for metric in _TRAIN_METRICS
        }
        metric_by_step = {
            metric: {int(step): float(value) for step, value in points}
            for metric, points in metrics.items()
        }
        for index, row in enumerate(round_rows):
            row["grad_norm"] = metric_by_step["train/grad_norm"][index]
            row["train_rollout_kl"] = metric_by_step["train/train_rollout_kl"][index]
        totals = {
            "rounds": manifest["rounds"],
            "groups": sum(row["groups"] for row in round_rows),
            "accepted_trajectories": sum(
                row["accepted_trajectories"] for row in round_rows
            ),
            "trained_tokens": sum(row["trained_tokens"] for row in round_rows),
            "local_steps": len(round_rows),
            "optimizer_steps": sum(row["optimizer_steps"] for row in round_rows),
            "publications": len(publications),
        }
        island_rows.append(
            {
                "island_id": island_id,
                "final_policy_version": manifest["rounds"],
                "final_policy_hash": publications[manifest["rounds"]]["policy_hash"],
                "publications": [
                    publications[version] for version in range(manifest["rounds"] + 1)
                ],
                "rounds": round_rows,
                "totals": totals,
                "evaluation": {
                    "initial": eval_by_version[0],
                    "final": eval_by_version[manifest["rounds"]],
                },
                "evidence_sha256": {
                    **shell_evidence,
                    "learner_events": events_sha,
                    "heldout_summary": eval_sha,
                    "trajectory_tree": trajectory_sha,
                    "metric_history_tree": _tree_sha256(
                        metric_dir, run_root, "Miles metric history"
                    ),
                },
            }
        )
        island_steps.append(local_steps)
        all_publications.append(publications)
        input_hash_rows.append(
            [
                local_steps[version]["input_batch_hash"]
                for version in range(manifest["rounds"])
            ]
        )
    reference_hashes = [
        all_publications[0][version]["policy_hash"]
        for version in range(manifest["rounds"] + 1)
    ]
    for publications in all_publications[1:]:
        if [
            publications[version]["policy_hash"]
            for version in range(manifest["rounds"] + 1)
        ] != reference_hashes:
            raise FinalReportError("islands published different global policies")
    syncer, syncer_sha = _syncer_summary(manifest, island_steps, run_root)
    for norm in syncer["global_delta_norms"]:
        round_index = norm["round"] - 1
        for island in island_rows:
            island["rounds"][round_index].setdefault("global_delta_norms", []).append(
                {
                    "fragment": norm["fragment"],
                    "value": norm["global_delta_norm"],
                }
            )
    rounds = manifest["rounds"]
    initial_results = [row["evaluation"]["initial"] for row in island_rows]
    final_results = [row["evaluation"]["final"] for row in island_rows]
    initial_result = sum(row["result"] for row in initial_results) / len(
        initial_results
    )
    final_result = sum(row["result"] for row in final_results) / len(final_results)
    initial_pass = sum(row["pass_at_1"] for row in initial_results) / len(
        initial_results
    )
    final_pass = sum(row["pass_at_1"] for row in final_results) / len(final_results)
    totals = {
        "islands": len(island_rows),
        "rounds_per_island": rounds,
        "groups": sum(row["totals"]["groups"] for row in island_rows),
        "accepted_trajectories": sum(
            row["totals"]["accepted_trajectories"] for row in island_rows
        ),
        "trained_tokens": sum(row["totals"]["trained_tokens"] for row in island_rows),
        "local_steps": sum(row["totals"]["local_steps"] for row in island_rows),
        "optimizer_steps": sum(row["totals"]["optimizer_steps"] for row in island_rows),
        "publications": sum(row["totals"]["publications"] for row in island_rows),
        "sync_merges": syncer["merge_count"],
    }
    return {
        "schema": FINAL_REPORT_SCHEMA,
        "status": "passed",
        "manifest_sha256": manifest_sha256,
        "run_id": manifest["run_id"],
        "rounds": rounds,
        "fragment_count": manifest["model"]["fragment_count"],
        "topology": {
            "island_count": manifest["topology"]["island_count"],
            "gpus_per_island": manifest["topology"]["gpus_per_island"],
            "trainer_tp": manifest["topology"]["trainer_tp"],
            "inference_engines": manifest["topology"]["inference_engines"],
            "inference_tp": manifest["topology"]["inference_tp"],
            "cross_island_collective": manifest["topology"]["cross_island_collective"],
        },
        "bindings": {
            "image_digest": manifest["image"]["digest"],
            "launch_bundle_sha256": manifest["launch_bundle"]["aggregate_sha256"],
            "yeto_source_sha256": manifest["provenance"]["yeto_source_sha256"],
            "miles_source_sha256": manifest["provenance"]["miles_source_sha256"],
            "model_revision": manifest["model"]["revision"],
            "model_config_sha256": manifest["model"]["config_sha256"],
            "parameter_layout_hash": manifest["model"]["parameter_layout_hash"],
            "conversion_manifest_sha256": manifest["model"][
                "conversion_manifest_sha256"
            ],
            "train_data_sha256": manifest["data"]["train_sha256"],
            "heldout_data_sha256": manifest["data"]["heldout_sha256"],
            "reward_sha256": manifest["profile"]["reward_sha256"],
            "task_pack_sha256": manifest["secrlenv"]["task_pack_sha256"],
            "syncer_binary_sha256": manifest["syncer"]["binary_sha256"],
            # This is deliberately named for what it proves.  Existing
            # evidence does not carry a prompt-group or generation-seed
            # manifest, so the report never mislabels this aggregate.
            "ordered_island_input_batch_hashes_sha256": _ordered_hash(
                input_hash_rows, b"yeto-m1-ordered-island-input-batches-v1\0"
            ),
        },
        "final_policy": {"version": rounds, "policy_hash": reference_hashes[-1]},
        "totals": totals,
        "heldout": {
            "dataset_name": evaluation["dataset_name"],
            "prompt_count_per_island": evaluation["prompt_count"],
            "samples_per_prompt": evaluation["samples_per_prompt"],
            "initial": {
                "policy_version": 0,
                "policy_hash": reference_hashes[0],
                "mean_result": initial_result,
                "mean_pass_at_1": initial_pass,
            },
            "final": {
                "policy_version": rounds,
                "policy_hash": reference_hashes[-1],
                "mean_result": final_result,
                "mean_pass_at_1": final_pass,
            },
            "result_delta": final_result - initial_result,
            "pass_at_1_delta": final_pass - initial_pass,
        },
        "islands": island_rows,
        "syncer": {**syncer, "evidence_sha256": syncer_sha},
        "secrlenv_cleanup": {
            "ok": True,
            "active_episodes": 0,
            "task_pack_sha256": daemon_health["task_pack_sha256"],
        },
    }


def write_report_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if (
        path.exists()
        or path.is_symlink()
        or not path.parent.is_dir()
        or path.parent.is_symlink()
    ):
        raise FinalReportError("final report path must be fresh")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        raw = _canonical(payload)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            if handle.write(raw) != len(raw):
                raise OSError("short final report write")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
