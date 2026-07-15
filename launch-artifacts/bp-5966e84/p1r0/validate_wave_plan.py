#!/usr/bin/env python3
"""Independent deterministic validation for the P1-R0 prebinding schedule."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


PACKET = Path(__file__).resolve().parent
MASTER_SEED_HEX = "0728fa50c14f4e52113407ab12e173b7ef4eb3b3b36f192ec7b814dd411223c5"
STUDY_ID = "bp-phase-map-p1-r0"
SLOTS = ("v0", "v1", "v2", "v3")
PROTECTED_INSTANCE_ID = "3908640733128066700"
EXPECTED_RUN_IDS = {
    slot: f"bp-p1r0-w{index + 1}-5966e84-20260715a"
    for index, slot in enumerate(SLOTS)
}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def payload(domain: str, token: str) -> bytes:
    return (
        bytes.fromhex(MASTER_SEED_HEX)
        + b"\x00"
        + domain.encode("utf-8")
        + b"\x00"
        + STUDY_ID.encode("utf-8")
        + b"\x00"
        + token.encode("utf-8")
    )


def rank_python(domain: str, token: str) -> str:
    return sha256_bytes(payload(domain, token))


def rank_openssl(domain: str, token: str) -> str:
    result = subprocess.run(
        ["openssl", "dgst", "-sha256", "-binary"],
        input=payload(domain, token),
        capture_output=True,
        check=True,
    )
    return result.stdout.hex()


def ranked(tokens: list[str], domain: str) -> list[str]:
    return sorted(
        tokens,
        key=lambda token: (bytes.fromhex(rank_openssl(domain, token)), token.encode()),
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def cloud_read_only(run_id: str, prefix: str) -> dict[str, object]:
    env = dict(os.environ)
    env["CLOUDSDK_CONFIG"] = "/private/tmp/yeto-gcloud-admin-codex"
    storage = subprocess.run(
        ["gcloud", "storage", "ls", "--all-versions", f"{prefix}/**"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    storage_text = (storage.stdout + storage.stderr).strip()
    empty = storage.returncode == 0 and not storage.stdout.strip()
    if storage.returncode != 0:
        empty = bool(
            re.search(r"matched no objects|matched no URLs|not found", storage_text, re.I)
        )
    describe = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            run_id,
            "--project=model-training-497007",
            "--zone=us-central1-c",
            "--format=json",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    describe_text = (describe.stdout + describe.stderr).strip()
    absent = describe.returncode != 0 and bool(
        re.search(r"not found|was not found|could not fetch resource", describe_text, re.I)
    )
    state = Path("/tmp/yeto-p1r0-state") / f"{run_id}.json"
    return {
        "run_id": run_id,
        "artifact_prefix": prefix,
        "artifact_prefix_empty": empty,
        "instance_absent": absent,
        "controller_state_path": str(state),
        "controller_state_absent": not state.exists(),
        "storage_returncode": storage.returncode,
        "storage_output": storage_text,
        "describe_returncode": describe.returncode,
        "describe_output": describe_text,
    }


def main() -> None:
    schedule_path = PACKET / "wave-plan.prebinding.json"
    schedule = json.loads(schedule_path.read_text())
    launch_cells = json.loads((PACKET / "launch-cells.json").read_text())
    scientific = json.loads(
        (PACKET / "scientific-randomization-plan.json").read_text()
    )
    waves = [
        json.loads(path.read_text()) for path in sorted((PACKET / "waves").glob("*.json"))
    ]
    require(schedule["status"] == "PREBOUND_SCHEDULE_ONLY_NOT_LAUNCH_AUTHORITY", "status mismatch")
    require(schedule["launch_eligible"] is False, "prebinding plan must not launch")
    require(schedule["requested_vm_run_ids"] == EXPECTED_RUN_IDS, "run IDs mismatch")
    require(len(launch_cells) == 36, "P1-R0 must have 36 launch cells")
    require(len(waves) == 12, "P1-R0 must have 12 waves")
    require(scientific["randomization_plan_hash"] == schedule["scientific_randomization_plan_hash"], "scientific plan hash mismatch")
    require(
        schedule["prebinding_schedule_canonical_sha256"]
        == sha256_bytes(
            canonical_json(
                {
                    key: value
                    for key, value in schedule.items()
                    if key != "prebinding_schedule_canonical_sha256"
                }
            )
        ),
        "prebinding schedule canonical hash mismatch",
    )

    by_group: dict[str, list[dict[str, object]]] = {}
    for cell in launch_cells:
        by_group.setdefault(str(cell["block_id"]), []).append(cell)
        require(cell["seed"] == 347 and cell["training_seed"] == 347347, "seed mismatch")
        require(cell["h"] in (16, 64, 256), "H mismatch")
        require(cell["eta"] in (0.021875, 0.04375, 0.0875, 0.175), "eta mismatch")
        require(cell["mu"] in (0, 0.5, 0.9), "mu mismatch")
    require(len(by_group) == 12, "group count mismatch")
    for group_id, cells in by_group.items():
        require(len(cells) == 3, f"{group_id} is not a three-cell group")
        require({cell["mu"] for cell in cells} == {0, 0.5, 0.9}, f"{group_id} mu block mismatch")
        require(len({cell["h"] for cell in cells}) == 1, f"{group_id} H mismatch")
        require(len({cell["eta"] for cell in cells}) == 1, f"{group_id} eta mismatch")

    expected_group_order = ranked(list(by_group), "wave")
    require(
        [wave["group_id"] for wave in waves] == expected_group_order,
        "wave/time-block order mismatch",
    )
    seen_cells: set[str] = set()
    rank_vectors: list[dict[str, str]] = []
    for index, wave in enumerate(waves):
        group_id = wave["group_id"]
        cells = by_group[group_id]
        cell_ids = [str(cell["cell_id"]) for cell in cells]
        arm_order = ranked(cell_ids, f"arm-order|{group_id}|1")
        slot_order = ranked(list(SLOTS), f"slot-order|{group_id}|1")
        assignment = dict(zip(arm_order, slot_order))
        launch_order = ranked(cell_ids, f"launch-order|{group_id}|1")
        rows = wave["assigned_cells_in_dispatch_order"]
        require(wave["wave_index"] == index, "wave index mismatch")
        require(wave["time_block_index"] == index, "time block mismatch")
        require(len(rows) == 3, "wave must dispatch exactly three cells")
        require([row["cell_id"] for row in rows] == launch_order, "dispatch order mismatch")
        require(
            all(row["logical_slot"] == assignment[row["cell_id"]] for row in rows),
            "arm-to-slot assignment mismatch",
        )
        active_slots = {row["logical_slot"] for row in rows}
        require(len(active_slots) == 3, "wave does not use three distinct slots")
        require(wave["idle_slot"] == next(slot for slot in SLOTS if slot not in active_slots), "idle slot mismatch")
        require(not (seen_cells & set(launch_order)), "cell appears in multiple waves")
        seen_cells.update(launch_order)
        tokens = [("wave", group_id)]
        tokens.extend((f"arm-order|{group_id}|1", cell_id) for cell_id in cell_ids)
        tokens.extend((f"launch-order|{group_id}|1", cell_id) for cell_id in cell_ids)
        tokens.extend((f"slot-order|{group_id}|1", slot) for slot in SLOTS)
        for domain, token in tokens:
            python_rank = rank_python(domain, token)
            openssl_rank = rank_openssl(domain, token)
            require(python_rank == openssl_rank, "independent rank implementations disagree")
            if len(rank_vectors) < 24:
                rank_vectors.append(
                    {"domain": domain, "token": token, "sha256": python_rank}
                )
    require(seen_cells == {str(cell["cell_id"]) for cell in launch_cells}, "cell coverage mismatch")

    vm_total = 0
    for slot in SLOTS:
        vm = json.loads((PACKET / "vms" / f"{slot}.json").read_text())
        require(vm["logical_slot"] == slot, "VM slot mismatch")
        require(vm["requested_run_id"] == EXPECTED_RUN_IDS[slot], "VM run ID mismatch")
        require(vm["launch_eligible"] is False, "VM prebinding plan must not launch")
        vm_total += vm["assigned_cell_count"]
    require(vm_total == 36, "per-VM assignments do not cover 36 cells")

    cloud_rows = []
    for slot in SLOTS:
        run_id = EXPECTED_RUN_IDS[slot]
        prefix = f"gs://yeto-exp2-52-model-training-497007/{run_id}"
        row = cloud_read_only(run_id, prefix)
        require(row["artifact_prefix_empty"] is True, f"{run_id} prefix is not empty")
        require(row["instance_absent"] is True, f"{run_id} instance exists")
        require(row["controller_state_absent"] is True, f"{run_id} state exists")
        cloud_rows.append(row)

    requested_grammar = re.compile(
        r"^bp-(p1r0|p1ad|p2|p3t)-[0-9a-f]{16}-c[1-9][0-9]*-v[0-3]-g[1-9][0-9]*$"
    )
    run_id_grammar_pass = all(
        requested_grammar.fullmatch(run_id) is not None
        for run_id in EXPECTED_RUN_IDS.values()
    )
    require(run_id_grammar_pass is False, "requested aliases unexpectedly match normative grammar")
    require(
        PROTECTED_INSTANCE_ID not in json.dumps(schedule, sort_keys=True),
        "protected instance appears in schedule",
    )

    gates = PACKET / "gates"
    gates.mkdir(parents=True, exist_ok=True)
    (gates / "rank-golden-vectors.json").write_text(
        json.dumps(rank_vectors, indent=2, sort_keys=True) + "\n"
    )
    (gates / "cloud-readonly-preflight.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "read_only": True,
                "protected_instance_targeted": False,
                "vms": cloud_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    report = {
        "status": "PASS",
        "scope": "deterministic_prebinding_schedule_only",
        "source_commit": schedule["source_commit"],
        "scientific_cell_count": 36,
        "atomic_group_count": 12,
        "wave_count": 12,
        "cells_per_wave": 3,
        "maximum_campaign_a100s": 16,
        "independent_rank_implementations": "PASS",
        "exact_cell_coverage": "PASS",
        "atomic_three_mu_groups": "PASS",
        "wave_order": "PASS",
        "arm_to_slot_assignment": "PASS",
        "dispatch_order": "PASS",
        "fresh_requested_namespaces": "PASS",
        "requested_run_id_aliases": EXPECTED_RUN_IDS,
        "requested_run_ids_match_amendment_generation_grammar": False,
        "parallel_parent_binding_complete": False,
        "parallel_executor_aggregator_implemented": False,
        "launch_eligible": False,
        "launch_performed": False,
        "protected_instance_targeted": False,
    }
    (gates / "wave-plan-validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
