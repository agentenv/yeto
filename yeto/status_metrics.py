"""Summaries for syncer event tapes used by `yeto status`."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_tape(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(Path(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # The syncer deliberately places a durable ledger snapshot
                # after a torn diagnostic tail. Ignore that damaged line so
                # status can still consume the authoritative reconciliation.
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def summarize_tape(records: list[dict]) -> dict:
    totals: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "responses": 0.0,
            "missed": 0.0,
            "tokens": 0.0,
            "steps": 0.0,
            "weight": 0.0,
        }
    )
    total_weight = 0.0
    total_missed = 0
    rounds_with_misses = 0
    recent_misses: list[str] = []
    merge_records = [
        record
        for record in records
        if "step" in record and record.get("event") != "policy_sweep_ledger"
    ]
    latest_snapshot_index = next(
        (
            index
            for index in range(len(records) - 1, -1, -1)
            if records[index].get("event") == "policy_sweep_ledger"
        ),
        None,
    )
    latest_snapshot = (
        records[latest_snapshot_index]
        if latest_snapshot_index is not None
        else None
    )
    snapshot_step = (
        latest_snapshot.get("global_step")
        if latest_snapshot is not None
        else None
    )

    for rec in merge_records:
        responders = rec.get("responders") or []
        responded = {int(r["id"]) for r in responders if "id" in r}
        expected = {int(x) for x in rec.get("expected", [])}
        if rec.get("missed_grace") is not None:
            missed = {int(x) for x in rec.get("missed_grace", [])}
        elif expected:
            missed = expected - responded
        else:
            missed = set()

        if missed:
            rounds_with_misses += 1
            total_missed += len(missed)
            step = rec.get("step", "?")
            fragment = rec.get("fragment", "?")
            ids = ",".join(str(x) for x in sorted(missed))
            recent_misses.append(f"step={step}/frag={fragment}: [{ids}]")
            for learner_id in missed:
                totals[learner_id]["missed"] += 1

        for responder in responders:
            learner_id = int(responder["id"])
            weight = float(responder.get("weight", 0.0))
            # Dense policy sweeps repeat the same raw local progress on every
            # fragment for merge evidence, but charge it to the durable
            # ledger only when the complete policy sweep closes. Legacy tape
            # records do not carry the accounted fields and retain their
            # original behavior through these fallbacks.
            accounted_tokens = responder.get(
                "accounted_c_tokens", responder.get("c_tokens", 0.0)
            )
            accounted_steps = responder.get(
                "accounted_c_steps", responder.get("c_steps", 0.0)
            )
            totals[learner_id]["responses"] += 1
            totals[learner_id]["tokens"] += float(accounted_tokens)
            totals[learner_id]["steps"] += float(accounted_steps)
            totals[learner_id]["weight"] += weight
            total_weight += weight

    # A sweep snapshot is an fsynced view of the checkpoint ledger. It can
    # cover one committed merge whose ordinary diagnostic record was lost in
    # a crash, so it replaces accounting through its global step. Merge
    # records appended after a resume snapshot still count while the resumed
    # run is live and before its final snapshot lands.
    if latest_snapshot is not None:
        for row in totals.values():
            row["responses"] = 0.0
            row["steps"] = 0.0
            row["tokens"] = 0.0
        for entry in latest_snapshot.get("ledger") or []:
            if not isinstance(entry, dict) or "id" not in entry:
                continue
            row = totals[int(entry["id"])]
            row["responses"] = float(entry.get("merges", 0))
            row["steps"] = float(entry.get("steps", 0))
            row["tokens"] = float(entry.get("tokens", 0))
        for record in records[latest_snapshot_index + 1 :]:
            if not (
                "step" in record
                and record.get("event") != "policy_sweep_ledger"
                and (
                    not isinstance(snapshot_step, int)
                    or record.get("step", 0) > snapshot_step
                )
            ):
                continue
            for responder in record.get("responders") or []:
                if "id" not in responder:
                    continue
                row = totals[int(responder["id"])]
                row["responses"] += 1
                row["steps"] += float(
                    responder.get(
                        "accounted_c_steps", responder.get("c_steps", 0.0)
                    )
                )
                row["tokens"] += float(
                    responder.get(
                        "accounted_c_tokens", responder.get("c_tokens", 0.0)
                    )
                )

    contributions = []
    for learner_id in sorted(totals):
        row = totals[learner_id]
        contribution = row["weight"] / total_weight if total_weight > 0 else 0.0
        contributions.append(
            {
                "id": learner_id,
                "responses": int(row["responses"]),
                "missed": int(row["missed"]),
                "tokens": int(row["tokens"]),
                "steps": int(row["steps"]),
                "weight": row["weight"],
                "contribution": contribution,
            }
        )

    last_merge = merge_records[-1] if merge_records else {}
    latest_step = last_merge.get("step")
    latest_fragment = last_merge.get("fragment")
    if latest_snapshot is not None:
        if isinstance(snapshot_step, int) and (
            not isinstance(latest_step, int) or snapshot_step >= latest_step
        ):
            latest_step = snapshot_step
            fragments = latest_snapshot.get("sweep_fragments")
            latest_fragment = (
                (snapshot_step - 1) % fragments
                if snapshot_step > 0 and isinstance(fragments, int) and fragments > 0
                else None
            )
    return {
        "rounds": len(merge_records),
        "latest_step": latest_step,
        "latest_fragment": latest_fragment,
        "total_missed": total_missed,
        "rounds_with_misses": rounds_with_misses,
        "recent_misses": recent_misses[-5:],
        "contributions": contributions,
    }


def render_tape_summary(path: str | Path) -> list[str]:
    records = load_tape(path)
    summary = summarize_tape(records)
    lines = [
        "",
        f"TAPE {Path(path)}",
        (
            "ROUNDS "
            f"{summary['rounds']}  "
            f"LATEST {summary['latest_step']}/{summary['latest_fragment']}  "
            f"MISSED {summary['total_missed']} across {summary['rounds_with_misses']} rounds"
        ),
    ]
    rows = summary["contributions"]
    if rows:
        lines.append("NODE  RESPONSES  MISSED  TOKENS  STEPS  CONTRIBUTION")
        for row in rows:
            lines.append(
                f"{row['id']:<4}  "
                f"{row['responses']:<9}  "
                f"{row['missed']:<6}  "
                f"{row['tokens']:<6}  "
                f"{row['steps']:<5}  "
                f"{row['contribution'] * 100:>6.2f}%"
            )
    if summary["recent_misses"]:
        lines.append("RECENT MISSES " + "; ".join(summary["recent_misses"]))
    return lines
