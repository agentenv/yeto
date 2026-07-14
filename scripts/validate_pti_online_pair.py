#!/usr/bin/env python3
"""Fail-closed validator for an exploratory online PTI-SGD/SGD-0.28 pair.

This validator binds the two terminal losses to complete 32-commit event tapes
and final syncer checkpoints.  It validates the causal PTI interlock from the
durable tape fields, including exact stock fallback hashes.  Passing this gate
is *exploratory online, non-CRN evidence*: the two arms are not restored from a
common boundary and the result cannot satisfy the frozen capture-v2 CRN gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
from typing import Any


SCHEMA = "yeto.pti-sgd.online-exploratory-validation.v1"
CLAIM_SCOPE = "exploratory_online_non_crn"
EXPECTED_COMMITS = 32
EXPECTED_FRAGMENTS = 4
EXPECTED_PER_FRAGMENT = EXPECTED_COMMITS // EXPECTED_FRAGMENTS
CKPT_MAGIC = 0xD170_5A7E
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
PTI_FIELDS = (
    "pti_shadow_score",
    "pti_score_count",
    "pti_interlock_open",
    "pti_used_nonstock",
    "pti_state_cleared",
    "pti_reason",
    "pti_stock_sha256",
    "pti_previous_stock_sha256",
    "pti_candidate_sha256",
    "pti_action_sha256",
)
PTI_REASONS = {
    "warmup",
    "interlock_closed",
    "candidate_selected",
    "degenerate_stock",
    "invalid_shadow_score",
    "degenerate_transverse",
}


class ValidationError(RuntimeError):
    """The pair is incomplete, malformed, or inconsistent."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant {value!r}")


def _validate_finite_tree(value: Any, *, source: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"{source}: non-finite JSON number")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_tree(item, source=f"{source}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_tree(item, source=f"{source}[{index}]")


def _regular_file(path: Path, *, label: str, nonempty: bool = True) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValidationError(f"missing {label}: {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValidationError(f"{label} is not a regular non-symlink file: {path}")
    if nonempty and metadata.st_size == 0:
        raise ValidationError(f"{label} is empty: {path}")
    return metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _regular_file(path, label=label)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read {label} {path}: {exc}") from exc
    if not lines:
        raise ValidationError(f"{label} is empty: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValidationError(f"{path}:{line_number}: blank JSONL record")
        try:
            row = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, UnicodeError, ValidationError) as exc:
            raise ValidationError(
                f"{path}:{line_number}: malformed JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValidationError(f"{path}:{line_number}: record is not an object")
        _validate_finite_tree(row, source=f"{path}:{line_number}")
        rows.append(row)
    return rows


def _exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def _exact_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValidationError(f"{label} must be a boolean, got {value!r}")
    return value


def _finite_number(value: Any, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be a finite number, got {value!r}")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValidationError(f"{label} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValidationError(f"{label} must be {qualifier}, got {value!r}")
    return number


def _digest(value: Any, *, label: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        qualifier = "a lowercase SHA-256 digest"
        if optional:
            qualifier += " or null"
        raise ValidationError(f"{label} must be {qualifier}, got {value!r}")
    return value


def _load_results(
    path: Path, *, stock_arm: str, pti_arm: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = _read_jsonl(path, label="results JSONL")
    by_arm: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        arm = row.get("arm")
        if not isinstance(arm, str) or not arm:
            raise ValidationError(f"{path}:{index}: invalid result arm {arm!r}")
        if arm in by_arm:
            raise ValidationError(f"{path}: duplicate result arm {arm!r}")
        _finite_number(row.get("eval_loss"), label=f"result {arm!r} eval_loss")
        by_arm[arm] = row
    missing = [arm for arm in (stock_arm, pti_arm) if arm not in by_arm]
    if missing:
        raise ValidationError(f"{path}: missing result arms {missing}")
    stock = by_arm[stock_arm]
    pti = by_arm[pti_arm]
    stock_m = _exact_int(stock.get("m"), label=f"{stock_arm}.m", minimum=1)
    pti_m = _exact_int(pti.get("m"), label=f"{pti_arm}.m", minimum=1)
    if stock_m != pti_m:
        raise ValidationError(
            f"result arms have different learner counts: {stock_m} != {pti_m}"
        )
    stock_loss = _finite_number(stock["eval_loss"], label=f"{stock_arm}.eval_loss")
    pti_loss = _finite_number(pti["eval_loss"], label=f"{pti_arm}.eval_loss")
    return (
        stock,
        pti,
        {
            "stock": stock_loss,
            "pti": pti_loss,
            "pti_minus_stock": pti_loss - stock_loss,
            "stock_minus_pti_gain": stock_loss - pti_loss,
            "pti_relative_to_stock": pti_loss / stock_loss if stock_loss != 0 else None,
        },
    )


def _validate_responder(
    responder: Any, *, row_label: str, position: int
) -> tuple[int, int, int, int, float]:
    label = f"{row_label}.responders[{position}]"
    if not isinstance(responder, dict):
        raise ValidationError(f"{label} must be an object")
    learner = _exact_int(responder.get("id"), label=f"{label}.id")
    base_version = _exact_int(
        responder.get("base_version"), label=f"{label}.base_version"
    )
    c_steps = _exact_int(responder.get("c_steps"), label=f"{label}.c_steps", minimum=1)
    c_tokens = _exact_int(
        responder.get("c_tokens"), label=f"{label}.c_tokens", minimum=1
    )
    weight = _finite_number(
        responder.get("weight"), label=f"{label}.weight", positive=True
    )
    try:
        expected_weight = float(c_tokens) ** 2 / c_steps
    except OverflowError as exc:
        raise ValidationError(f"{label}: weight calculation overflow") from exc
    if not math.isfinite(expected_weight) or not math.isclose(
        weight, expected_weight, rel_tol=1e-12, abs_tol=0.0
    ):
        raise ValidationError(
            f"{label}.weight {weight!r} does not equal c_tokens^2/c_steps "
            f"({expected_weight!r})"
        )
    return learner, base_version, c_steps, c_tokens, weight


def _validate_schedule(
    rows: list[dict[str, Any]], *, arm: str, learners: int
) -> tuple[list[tuple[Any, ...]], Counter[int]]:
    if len(rows) != EXPECTED_COMMITS:
        raise ValidationError(
            f"{arm}: expected exactly {EXPECTED_COMMITS} commits, got {len(rows)}"
        )
    schedule: list[tuple[Any, ...]] = []
    fragments: Counter[int] = Counter()
    seen_steps: set[int] = set()
    previous_elapsed = -1
    for offset, row in enumerate(rows, 1):
        label = f"{arm} tape row {offset}"
        commit_seq = _exact_int(
            row.get("commit_seq"), label=f"{label}.commit_seq", minimum=1
        )
        if commit_seq != offset:
            raise ValidationError(
                f"{label}: commit_seq must be exact file order {offset}, got {commit_seq}"
            )
        step = _exact_int(row.get("step"), label=f"{label}.step", minimum=1)
        if step in seen_steps:
            raise ValidationError(f"{label}: duplicate step {step}")
        seen_steps.add(step)
        fragment = _exact_int(row.get("fragment"), label=f"{label}.fragment")
        if fragment >= EXPECTED_FRAGMENTS:
            raise ValidationError(
                f"{label}.fragment must be in [0, {EXPECTED_FRAGMENTS}), got {fragment}"
            )
        fragments[fragment] += 1
        elapsed = _exact_int(
            row.get("commit_elapsed_ns"), label=f"{label}.commit_elapsed_ns", minimum=1
        )
        if elapsed <= previous_elapsed:
            raise ValidationError(
                f"{label}: commit_elapsed_ns is not strictly increasing"
            )
        previous_elapsed = elapsed
        responders = row.get("responders")
        if not isinstance(responders, list):
            raise ValidationError(f"{label}.responders must be a list")
        canonical_responders = tuple(
            _validate_responder(responder, row_label=label, position=position)
            for position, responder in enumerate(responders)
        )
        responder_ids = [responder[0] for responder in canonical_responders]
        if responder_ids != list(range(learners)):
            raise ValidationError(
                f"{label}: expected responder IDs {list(range(learners))}, "
                f"got {responder_ids}"
            )
        schedule.append((commit_seq, step, fragment, canonical_responders))
    if seen_steps != set(range(1, EXPECTED_COMMITS + 1)):
        raise ValidationError(
            f"{arm}: step set must be exactly 1..{EXPECTED_COMMITS}, "
            f"got {sorted(seen_steps)}"
        )
    expected_balance = {fragment: EXPECTED_PER_FRAGMENT for fragment in range(4)}
    if dict(sorted(fragments.items())) != expected_balance:
        raise ValidationError(
            f"{arm}: fragments are not exactly balanced: "
            f"expected {expected_balance}, got {dict(sorted(fragments.items()))}"
        )
    return schedule, fragments


def _validate_stock_tape(rows: list[dict[str, Any]], *, arm: str) -> None:
    inactive = {
        "pti_shadow_score": None,
        "pti_score_count": 0,
        "pti_interlock_open": False,
        "pti_used_nonstock": False,
        "pti_state_cleared": False,
        "pti_reason": None,
        "pti_stock_sha256": None,
        "pti_previous_stock_sha256": None,
        "pti_candidate_sha256": None,
        "pti_action_sha256": None,
    }
    for offset, row in enumerate(rows, 1):
        missing = [field for field in PTI_FIELDS if field not in row]
        if missing:
            raise ValidationError(
                f"{arm} tape row {offset}: missing PTI fields {missing}"
            )
        for field, expected in inactive.items():
            if row[field] != expected or type(row[field]) is not type(expected):
                raise ValidationError(
                    f"{arm} tape row {offset}: stock field {field} must be "
                    f"{expected!r}, got {row[field]!r}"
                )


def _validate_pti_tape(
    rows: list[dict[str, Any]], *, arm: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    states = {
        fragment: {"previous": None, "pending": False, "scores": []}
        for fragment in range(EXPECTED_FRAGMENTS)
    }
    summaries = {
        fragment: {
            "commits": 0,
            "resolved_scores": 0,
            "positive_scores": 0,
            "candidate_actions": 0,
            "stock_fallbacks": 0,
            "state_clears": 0,
            "score_sum": 0.0,
            "score_min": None,
            "score_max": None,
        }
        for fragment in range(EXPECTED_FRAGMENTS)
    }
    selected = 0
    fallbacks = 0
    resolved = 0
    positive = 0
    clears = 0

    for offset, row in enumerate(rows, 1):
        label = f"{arm} tape row {offset}"
        missing = [field for field in PTI_FIELDS if field not in row]
        if missing:
            raise ValidationError(f"{label}: missing PTI fields {missing}")
        fragment = row["fragment"]
        state = states[fragment]
        summary = summaries[fragment]
        summary["commits"] += 1

        score_value = row["pti_shadow_score"]
        score = None
        if score_value is not None:
            score = _finite_number(score_value, label=f"{label}.pti_shadow_score")
        score_count = _exact_int(
            row["pti_score_count"], label=f"{label}.pti_score_count"
        )
        if score_count > 3:
            raise ValidationError(f"{label}.pti_score_count exceeds three")
        interlock = _exact_bool(
            row["pti_interlock_open"], label=f"{label}.pti_interlock_open"
        )
        used = _exact_bool(row["pti_used_nonstock"], label=f"{label}.pti_used_nonstock")
        state_cleared = _exact_bool(
            row["pti_state_cleared"], label=f"{label}.pti_state_cleared"
        )
        reason = row["pti_reason"]
        if reason not in PTI_REASONS:
            raise ValidationError(f"{label}: invalid PTI reason {reason!r}")
        stock_hash = _digest(row["pti_stock_sha256"], label=f"{label}.pti_stock_sha256")
        previous_hash = _digest(
            row["pti_previous_stock_sha256"],
            label=f"{label}.pti_previous_stock_sha256",
            optional=True,
        )
        candidate_hash = _digest(
            row["pti_candidate_sha256"],
            label=f"{label}.pti_candidate_sha256",
            optional=True,
        )
        action_hash = _digest(
            row["pti_action_sha256"], label=f"{label}.pti_action_sha256"
        )

        if state["previous"] is None:
            if previous_hash is not None:
                raise ValidationError(
                    f"{label}: previous stock hash exists after reset"
                )
            if reason not in {"warmup", "degenerate_stock"}:
                raise ValidationError(
                    f"{label}: expected warmup or degenerate stock after reset"
                )
        elif previous_hash != state["previous"]:
            raise ValidationError(
                f"{label}: previous stock hash does not match preceding same-fragment stock"
            )

        expected_score = bool(state["pending"])
        resolves_normally = reason not in {"degenerate_stock", "invalid_shadow_score"}
        if score is not None:
            if not expected_score or not resolves_normally:
                raise ValidationError(
                    f"{label}: shadow score has no eligible pending shadow"
                )
            scores = [*state["scores"], score][-3:]
            summary["resolved_scores"] += 1
            summary["score_sum"] += score
            summary["score_min"] = (
                score
                if summary["score_min"] is None
                else min(summary["score_min"], score)
            )
            summary["score_max"] = (
                score
                if summary["score_max"] is None
                else max(summary["score_max"], score)
            )
            resolved += 1
            if score > 0.0:
                summary["positive_scores"] += 1
                positive += 1
        else:
            scores = list(state["scores"])
            if expected_score and reason not in {
                "degenerate_stock",
                "invalid_shadow_score",
            }:
                raise ValidationError(f"{label}: pending shadow was not resolved")

        expected_open = len(scores) == 3 and all(value > 0.0 for value in scores)
        ordinary = reason in {"interlock_closed", "candidate_selected"}
        if ordinary:
            if state_cleared:
                raise ValidationError(
                    f"{label}: ordinary PTI decision cannot clear state"
                )
            if candidate_hash is None:
                raise ValidationError(
                    f"{label}: ordinary PTI decision lacks candidate hash"
                )
            if score_count != len(scores):
                raise ValidationError(
                    f"{label}: score_count {score_count} != causal window {len(scores)}"
                )
            if interlock != expected_open:
                raise ValidationError(
                    f"{label}: interlock {interlock} != three-positive rule {expected_open}"
                )
            if used != expected_open:
                raise ValidationError(
                    f"{label}: nonstock action disagrees with interlock"
                )
            expected_reason = (
                "candidate_selected" if expected_open else "interlock_closed"
            )
            if reason != expected_reason:
                raise ValidationError(
                    f"{label}: expected reason {expected_reason!r}, got {reason!r}"
                )
            state["previous"] = stock_hash
            state["pending"] = True
            state["scores"] = scores
        else:
            if interlock or used:
                raise ValidationError(
                    f"{label}: fallback/error reason cannot select PTI"
                )
            if candidate_hash is not None:
                raise ValidationError(
                    f"{label}: fallback/error reason has candidate hash"
                )
            if reason == "warmup":
                if state["previous"] is not None:
                    raise ValidationError(f"{label}: warmup is only valid after reset")
                if score is not None or score_count != 0 or state_cleared:
                    raise ValidationError(f"{label}: malformed warmup record")
                state["previous"] = stock_hash
                state["pending"] = False
                state["scores"] = []
            else:
                if reason == "invalid_shadow_score" and not state["pending"]:
                    raise ValidationError(
                        f"{label}: invalid-shadow fallback has no pending shadow"
                    )
                if reason == "degenerate_transverse":
                    if score_count != len(scores):
                        raise ValidationError(
                            f"{label}: degenerate score count mismatch"
                        )
                elif score is not None or score_count != 0:
                    raise ValidationError(
                        f"{label}: clearing reason must expose zero scores"
                    )
                if not state_cleared:
                    raise ValidationError(
                        f"{label}: error fallback must clear fragment state"
                    )
                state["previous"] = None
                state["pending"] = False
                state["scores"] = []

        if used:
            if action_hash != candidate_hash:
                raise ValidationError(
                    f"{label}: selected action hash is not candidate hash"
                )
            selected += 1
            summary["candidate_actions"] += 1
        else:
            if action_hash != stock_hash:
                raise ValidationError(
                    f"{label}: fallback action hash is not exact stock hash"
                )
            fallbacks += 1
            summary["stock_fallbacks"] += 1
        if state_cleared:
            clears += 1
            summary["state_clears"] += 1

    if selected == 0:
        raise ValidationError("PTI tape selected no nonstock actions")
    fragment_output: dict[str, dict[str, Any]] = {}
    for fragment, summary in sorted(summaries.items()):
        score_total = summary.pop("score_sum")
        fragment_output[str(fragment)] = {
            **summary,
            "action_fraction": summary["candidate_actions"] / summary["commits"],
            "mean_shadow_score": (
                score_total / summary["resolved_scores"]
                if summary["resolved_scores"]
                else None
            ),
        }
    return (
        {
            "commits": len(rows),
            "candidate_actions": selected,
            "stock_fallbacks": fallbacks,
            "action_fraction": selected / len(rows),
            "resolved_scores": resolved,
            "positive_scores": positive,
            "state_clears": clears,
        },
        fragment_output,
    )


def _checkpoint_summary(path: Path, *, label: str) -> dict[str, Any]:
    metadata = _regular_file(path, label=label)
    try:
        with path.open("rb") as handle:
            head = handle.read(16)
    except OSError as exc:
        raise ValidationError(f"cannot read {label} {path}: {exc}") from exc
    if len(head) != 16:
        raise ValidationError(f"{label} is truncated before its header")
    magic, global_step, fragments = struct.unpack("<IQI", head)
    if magic != CKPT_MAGIC:
        raise ValidationError(f"{label} has invalid syncer checkpoint magic")
    if global_step != EXPECTED_COMMITS:
        raise ValidationError(
            f"{label} global_step must be {EXPECTED_COMMITS}, got {global_step}"
        )
    if fragments != EXPECTED_FRAGMENTS:
        raise ValidationError(
            f"{label} fragment count must be {EXPECTED_FRAGMENTS}, got {fragments}"
        )
    return {
        "bytes": metadata.st_size,
        "sha256": _sha256_file(path),
        "global_step": global_step,
        "fragments": fragments,
    }


def _write_evidence(path: Path, evidence: dict[str, Any]) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(
            evidence,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    def publish(destination: Path, payload: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    publish(path, raw)
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    publish(sidecar, f"{digest}  {path.name}\n".encode("ascii"))
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # The files themselves were fsynced. Some filesystems do not permit
        # directory fsync; atomic replace semantics still hold there.
        pass
    return digest


def _validate_pair(
    *,
    results: Path,
    stock_arm_dir: Path,
    pti_arm_dir: Path,
    stock_arm: str,
    pti_arm: str,
) -> dict[str, Any]:
    if stock_arm == pti_arm:
        raise ValidationError("stock and PTI arm names must differ")
    stock_row, pti_row, loss = _load_results(
        results, stock_arm=stock_arm, pti_arm=pti_arm
    )
    learners = stock_row["m"]
    stock_tape_path = stock_arm_dir / "tape.jsonl"
    pti_tape_path = pti_arm_dir / "tape.jsonl"
    stock_rows = _read_jsonl(stock_tape_path, label="stock event tape")
    pti_rows = _read_jsonl(pti_tape_path, label="PTI event tape")
    stock_schedule, _ = _validate_schedule(stock_rows, arm=stock_arm, learners=learners)
    pti_schedule, _ = _validate_schedule(pti_rows, arm=pti_arm, learners=learners)
    if stock_schedule != pti_schedule:
        for offset, (stock_item, pti_item) in enumerate(
            zip(stock_schedule, pti_schedule), 1
        ):
            if stock_item != pti_item:
                raise ValidationError(
                    f"arm commit/responder schedules first differ at commit {offset}: "
                    f"stock={stock_item!r}, pti={pti_item!r}"
                )
        raise ValidationError("arm commit/responder schedules differ")
    _validate_stock_tape(stock_rows, arm=stock_arm)
    actions, per_fragment = _validate_pti_tape(pti_rows, arm=pti_arm)
    stock_checkpoint = _checkpoint_summary(
        stock_arm_dir / "state.ckpt", label="stock final checkpoint"
    )
    pti_checkpoint = _checkpoint_summary(
        pti_arm_dir / "state.ckpt", label="PTI final checkpoint"
    )
    schedule_json = [
        [
            commit_seq,
            step,
            fragment,
            [list(responder) for responder in responders],
        ]
        for commit_seq, step, fragment, responders in stock_schedule
    ]
    return {
        "results": {
            "sha256": _sha256_file(results),
            "rows": len(_read_jsonl(results, label="results JSONL")),
            "stock_arm": stock_arm,
            "pti_arm": pti_arm,
            "learners": learners,
            "stock_wall_s": (
                _finite_number(stock_row["wall_s"], label=f"{stock_arm}.wall_s")
                if "wall_s" in stock_row
                else None
            ),
            "pti_wall_s": (
                _finite_number(pti_row["wall_s"], label=f"{pti_arm}.wall_s")
                if "wall_s" in pti_row
                else None
            ),
        },
        "loss": loss,
        "actions": actions,
        "per_fragment": per_fragment,
        "schedule": {
            "commits": EXPECTED_COMMITS,
            "fragments": EXPECTED_FRAGMENTS,
            "commits_per_fragment": EXPECTED_PER_FRAGMENT,
            "canonical_sha256": _canonical_sha256(schedule_json),
        },
        "artifacts": {
            "stock_tape": {
                "sha256": _sha256_file(stock_tape_path),
                "records": len(stock_rows),
            },
            "pti_tape": {
                "sha256": _sha256_file(pti_tape_path),
                "records": len(pti_rows),
            },
            "stock_checkpoint": stock_checkpoint,
            "pti_checkpoint": pti_checkpoint,
        },
    }


def run_gate(
    *,
    results: Path,
    stock_arm_dir: Path,
    pti_arm_dir: Path,
    stock_arm: str,
    pti_arm: str,
    output: Path,
) -> dict[str, Any]:
    """Validate and always atomically publish a PASS or FAIL evidence file."""
    results = results.resolve()
    stock_arm_dir = stock_arm_dir.resolve()
    pti_arm_dir = pti_arm_dir.resolve()
    error: str | None = None
    detail: dict[str, Any] | None = None
    try:
        detail = _validate_pair(
            results=results,
            stock_arm_dir=stock_arm_dir,
            pti_arm_dir=pti_arm_dir,
            stock_arm=stock_arm,
            pti_arm=pti_arm,
        )
    except Exception as exc:  # malformed evidence must still leave FAIL evidence
        error = str(exc)
    evidence: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS" if error is None else "FAIL",
        "claim_scope": CLAIM_SCOPE,
        "capture_v2_crn_gate_satisfied": False,
        "inputs": {
            "results": str(results),
            "stock_arm_dir": str(stock_arm_dir),
            "pti_arm_dir": str(pti_arm_dir),
            "stock_arm": stock_arm,
            "pti_arm": pti_arm,
        },
        "fixed_protocol": {
            "commits": EXPECTED_COMMITS,
            "fragments": EXPECTED_FRAGMENTS,
            "commits_per_fragment": EXPECTED_PER_FRAGMENT,
            "pti_interlock": "three_most_recent_resolved_scores_strictly_positive",
            "stock_fallback": "exact_action_sha256_equals_stock_sha256",
        },
        "validation": detail,
        "errors": [] if error is None else [error],
        "limitations": [
            "The arms evolve online from different states after the first nonstock PTI action.",
            "This result is paired by schedule and evaluation input, not a same-state common-random-numbers boundary comparison.",
            "A PASS does not satisfy or replace the frozen capture-v2 PTI CRN confirmation gate and does not prove optimizer superiority.",
        ],
    }
    artifact_sha256 = _write_evidence(output, evidence)
    result = dict(evidence)
    result["artifact_sha256"] = artifact_sha256
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--stock-arm-dir", type=Path, required=True)
    parser.add_argument("--pti-arm-dir", type=Path, required=True)
    parser.add_argument("--stock-arm", required=True)
    parser.add_argument("--pti-arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evidence = run_gate(
        results=args.results,
        stock_arm_dir=args.stock_arm_dir,
        pti_arm_dir=args.pti_arm_dir,
        stock_arm=args.stock_arm,
        pti_arm=args.pti_arm,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "claim_scope": evidence["claim_scope"],
                "output": str(args.output),
                "artifact_sha256": evidence["artifact_sha256"],
                "errors": evidence["errors"],
            },
            sort_keys=True,
        )
    )
    return 0 if evidence["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
