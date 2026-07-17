#!/usr/bin/env python3
"""Freeze a one-slot launch packet for the triggered audit serial fallback.

This builder keeps the frozen audit materialization and compatibility roster
used by the reviewed work-evidence validators, but authorizes only one Spot
``a2-highgpu-1g`` generation and binds the serial ratchet contract.  It performs
no provider mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from scripts import audit_135m_serial as serial
from scripts import build_audit_135m_launch_packet as legacy
from scripts import run_parallel_phase_map as compatibility


P1_CONTROLLER = Path(
    "/private/tmp/yeto-p1r0-launcher/p1-adaptive-session/p1ad_campaign_controller.py"
)
R0_CONTROLLER = Path(
    "/private/tmp/yeto-p1r0-launcher/p1r0-session/p1r0_controller.py"
)


class SerialPacketError(RuntimeError):
    pass


def _load(path: Path, label: str) -> dict[str, Any]:
    return serial.load_object(path, label)


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _ensure_compatibility_binding(
    *, packet: Path, parent_manifest: Path, stage_code: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    materialized = packet / "materialized"
    binding_dir = packet / "binding"
    required = (
        binding_dir / "parallel-binding.json",
        binding_dir / "parallel-roster.json",
        binding_dir / "parallel-plan.json",
    )
    if not any(path.exists() for path in required):
        compatibility.bind_campaign_inputs(
            stage_code=stage_code,
            parent_manifest_path=parent_manifest,
            bound_manifest_path=materialized / "bound-manifest.json",
            scientific_plan_path=materialized / "scientific-randomization-plan.json",
            output_dir=binding_dir,
        )
    if not all(path.is_file() for path in required):
        raise SerialPacketError("compatibility binding is incomplete")
    binding = _load(required[0], "compatibility binding")
    roster = _load(required[1], "compatibility roster")
    plan = _load(required[2], "compatibility plan")
    # The full compatibility plan remains byte-exact for evidence validation,
    # but only v0 is physical launch authority in the serial packet.
    binding["logical_slots"] = ["v0"]
    binding["physical_generation_run_ids"] = {
        "v0": compatibility.physical_run_id(
            stage_code, compatibility.roster_hash(roster), 1, "v0", 1
        )
    }
    binding["launch_authorized"] = False
    binding["compatibility_only_parallel_executor_authorized"] = False
    _write_atomic(required[0], binding)
    return binding, roster, plan


def build(args: argparse.Namespace) -> dict[str, Any]:
    packet = args.packet_root.resolve()
    materialized = packet / "materialized"
    bound = _load(materialized / "bound-manifest.json", "bound manifest")
    scientific = _load(
        materialized / "scientific-randomization-plan.json", "scientific plan"
    )
    parent = _load(args.parent_manifest, "parent manifest")
    stage_code = str(scientific.get("stage_code"))
    if stage_code != args.stage_code or bound.get("study_id") != scientific.get(
        "study_id"
    ):
        raise SerialPacketError("serial packet stage/study identity differs")
    _binding, roster, plan = _ensure_compatibility_binding(
        packet=packet, parent_manifest=args.parent_manifest, stage_code=stage_code
    )
    serial_binding = serial.build_serial_binding(
        stage_code=stage_code,
        parent=parent,
        bound=bound,
        scientific=scientific,
        compatibility_roster=roster,
        compatibility_plan=plan,
    )
    serial_binding_path = packet / "binding" / "serial-binding.json"
    serial.write_json_create_only(serial_binding_path, serial_binding)
    reviewed_helper_sha256 = {
        "p1_capacity_controller": serial.sha256_file(P1_CONTROLLER),
        "gcp_backend_controller": serial.sha256_file(R0_CONTROLLER),
    }
    authorization = serial.runtime_authorization(
        serial_binding,
        hard_ceiling_usd=float(bound["audit_135m_contract"]["hard_ceiling_usd"]),
        reviewed_helper_sha256=reviewed_helper_sha256,
    )
    serial_authorization_path = packet / "serial-runtime-authorization.json"
    serial.write_json_create_only(serial_authorization_path, authorization)

    # Reuse the reviewed one-GPU harness/bootstrap construction while reducing
    # its physical identity universe to the sole authorized serial slot.
    prior_globals = (
        legacy.AUDIT_BLOCK_WIDTH,
        legacy.MAX_CONCURRENT_BLOCKS,
        legacy.SLOTS,
    )
    try:
        legacy.AUDIT_BLOCK_WIDTH = 1
        legacy.MAX_CONCURRENT_BLOCKS = 1
        legacy.SLOTS = ("v0",)
        result = legacy.build(
            argparse.Namespace(
                packet_root=packet,
                parent_manifest=args.parent_manifest,
                seed_bundle_registry=args.seed_bundle_registry,
                worker_wrapper=args.worker_wrapper,
                science_root=args.science_root,
                initial_zone=args.initial_zone,
            )
        )
    finally:
        (
            legacy.AUDIT_BLOCK_WIDTH,
            legacy.MAX_CONCURRENT_BLOCKS,
            legacy.SLOTS,
        ) = prior_globals

    identity_path = packet / "identity-plan.json"
    identity = _load(identity_path, "identity plan")
    identity.update(
        {
            "execution_mode": serial_binding["execution_mode"],
            "parallel_executor_authorized": False,
            "serial_plan_hash": serial_binding["serial_plan_hash"],
            "serial_runtime_authorization_hash": authorization[
                "authorization_canonical_sha256"
            ],
            "serial_binding_path": str(serial_binding_path),
            "serial_runtime_authorization_path": str(
                serial_authorization_path
            ),
            "target_width": 1,
            "maximum_concurrent_blocks": 1,
            "target_1g_slot_count": 1,
            "logical_slots": ["v0"],
            "reviewed_helper_sha256": reviewed_helper_sha256,
        }
    )
    _write_atomic(identity_path, identity)

    review_path = packet / "review-packet.json"
    review = _load(review_path, "review packet")
    review.update(
        {
            "status": "SEALED_SERIAL_LAUNCH_AUTHORIZED",
            "execution_mode": serial_binding["execution_mode"],
            "parallel_executor_authorized": False,
            "serial_amendment_raw_sha256": serial_binding[
                "serial_amendment_raw_sha256"
            ],
            "serial_plan_hash": serial_binding["serial_plan_hash"],
            "serial_binding_raw_sha256": serial.sha256_file(
                serial_binding_path
            ),
            "serial_runtime_authorization_hash": authorization[
                "authorization_canonical_sha256"
            ],
            "serial_runtime_authorization_raw_sha256": serial.sha256_file(
                serial_authorization_path
            ),
            "maximum_concurrent_blocks": 1,
            "maximum_active_vms": 1,
            "maximum_active_a100_equivalent": 1,
            "target_1g_slot_count": 1,
            "logical_slots": ["v0"],
            "completed_cell_ratchet": True,
            "mid_cell_preemption_retries_only_that_cell": True,
            "compatibility_parallel_binding_is_not_launch_authority": True,
            "reviewed_helper_sha256": reviewed_helper_sha256,
        }
    )
    _write_atomic(review_path, review)
    review_digest = serial.sha256_file(review_path)
    (packet / "review-packet.json.sha256").write_text(
        f"{review_digest}  review-packet.json\n", encoding="utf-8"
    )
    return {
        **result,
        "status": "SEALED_SERIAL_LAUNCH_AUTHORIZED",
        "execution_mode": serial_binding["execution_mode"],
        "parallel_executor_authorized": False,
        "serial_plan_hash": serial_binding["serial_plan_hash"],
        "serial_runtime_authorization_hash": authorization[
            "authorization_canonical_sha256"
        ],
        "review_packet_sha256": review_digest,
        "cloud_mutation_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-code", choices=sorted(serial.audit.STAGE_CODES), required=True)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--seed-bundle-registry", type=Path, required=True)
    parser.add_argument(
        "--worker-wrapper",
        type=Path,
        default=serial.REPO_ROOT / "scripts" / "audit_135m_vm_worker.py",
    )
    parser.add_argument("--science-root", type=Path, required=True)
    parser.add_argument(
        "--initial-zone", choices=legacy.ZONE_ROTATION, default="us-east1-b"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build(args)
    except (
        OSError,
        KeyError,
        ValueError,
        SerialPacketError,
        serial.SerialAuditError,
        compatibility.ParallelPhaseMapError,
        legacy.PacketError,
    ) as exc:
        print(f"audit serial packet error: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
