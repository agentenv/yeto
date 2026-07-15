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
            records.append(json.loads(line))
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

    for rec in records:
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
            totals[learner_id]["responses"] += 1
            totals[learner_id]["tokens"] += float(responder.get("c_tokens", 0.0))
            totals[learner_id]["steps"] += float(responder.get("c_steps", 0.0))
            totals[learner_id]["weight"] += weight
            total_weight += weight

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

    last = records[-1] if records else {}
    return {
        "rounds": len(records),
        "latest_step": last.get("step"),
        "latest_fragment": last.get("fragment"),
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
