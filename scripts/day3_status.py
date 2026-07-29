#!/usr/bin/env python3
"""Report day-3 progress from slot-state and evidence status metadata only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import day3_common as common


TERMINAL = {"COMPLETED", "SCIENTIFIC_DIVERGENCE", "INFRA_FAILURE", "INVALID_WORK"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--node-label", choices=common.NODES, required=True)
    args = parser.parse_args()
    manifest = common.read_json(args.manifest)
    program = manifest["program"]
    cells = [
        cell for cell in manifest["cells"] if cell["assignment"]["node"] == args.node_label
    ]
    statuses = Counter()
    groups: dict[str, dict[str, object]] = {}
    cell_records = []
    for cell in cells:
        selected_attempt = None
        status = "MISSING_EVIDENCE"
        attempt_statuses = {}
        for attempt in (1, 2):
            evidence_path = (
                common.RESULT_ROOT / cell["cell_id"] / f"attempt-{attempt}" / "evidence.json"
            )
            if evidence_path.is_file():
                evidence = common.read_json(evidence_path)
                selected_attempt = attempt
                status = str(evidence.get("status", "MALFORMED_EVIDENCE"))
                attempt_statuses[str(attempt)] = status
        statuses[status] += 1
        group = groups.setdefault(
            cell["retry_group_id"],
            {"statuses": Counter(), "cells": 0, "queue_id": cell["queue_id"]},
        )
        group["statuses"][status] += 1
        group["cells"] += 1
        cell_records.append(
            {
                "cell_id": cell["cell_id"],
                "retry_group_id": cell["retry_group_id"],
                "queue_id": cell["queue_id"],
                "attempt": selected_attempt,
                "status": status,
                "attempt_statuses": attempt_statuses,
            }
        )
    slot_records = []
    slot_root = common.RESULT_ROOT / "_controller" / "slots" / program
    for queue in manifest["queues"]:
        if queue["node"] != args.node_label:
            continue
        selected = None
        for attempt in (1, 2):
            path = slot_root / f"{queue['queue_id']}-a{attempt}.json"
            if path.is_file():
                state = common.read_json(path)
                selected = {
                    "queue_id": queue["queue_id"],
                    "attempt": attempt,
                    "state": state.get("state"),
                    "completed": state.get("completed"),
                    "failures": state.get("failures"),
                    "queue_total": state.get("queue_total"),
                    "cell_id": state.get("cell_id"),
                    "updated_at_utc": state.get("updated_at_utc"),
                }
        slot_records.append(
            selected
            or {"queue_id": queue["queue_id"], "attempt": None, "state": "NOT_STARTED"}
        )
    group_records = {
        group_id: {
            "cells": record["cells"],
            "queue_id": record["queue_id"],
            "statuses": dict(record["statuses"]),
            "terminal": all(status in TERMINAL for status in record["statuses"]),
        }
        for group_id, record in groups.items()
    }
    payload = {
        "schema": "yeto_day3_status_v1",
        "program": program,
        "node": args.node_label,
        "cells": len(cells),
        "status_counts": dict(statuses),
        "terminal_cells": sum(count for status, count in statuses.items() if status in TERMINAL),
        "completed_cells": statuses["COMPLETED"],
        "groups": group_records,
        "slots": slot_records,
        "cell_records": cell_records,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
