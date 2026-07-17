#!/usr/bin/env python3
"""Build and validate a portable, full-evidence audit-135M replay archive.

The build side consumes a local source index naming the sealed campaign roots,
their promoted cumulative manifests, selection inputs, final analysis, and cost
ledger.  It copies the complete campaign roots, including checkpoint bytes and
hidden private artifacts, into a create-only portable directory and hashes
every file.

The validate side is intended for a clean detached checkout on the isolated
replay host.  It verifies the Git/source authority, every archive file, fully
re-aggregates every parallel or serial campaign from VM evidence, re-promotes every
cumulative suffix (including hidden-batch verification), reproduces selection,
precision decisions, and the final registered gate, then writes one sealed
JSON replay report.  It performs no GPU work and no cloud mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCE_SCHEMA = "audit_135m_replay_source_index_v1"
PORTABLE_SCHEMA = "audit_135m_replay_input_v1"
REPORT_SCHEMA = "audit_135m_replay_report_v1"
SOURCE_FILES = (
    "experiment-specs/tuned-baseline-audit-prereg.json",
    "experiment-specs/tuned-baseline-audit-prereg.md",
    "scripts/audit_135m_contract.py",
    "scripts/audit_135m_analysis.py",
    "scripts/audit_135m_hidden_evaluator.py",
    "scripts/audit_135m_kernel_capture.py",
    "scripts/audit_135m_kernel_law.py",
    "scripts/audit_135m_phase_manifest.py",
    "scripts/audit_135m_serial.py",
    "scripts/run_parallel_phase_map.py",
    "scripts/run_phase_map.py",
    "docs/AMENDMENT-audit-135m-serial-fallback.md",
)


class ReplayError(RuntimeError):
    """The archive or a deterministic replay product differs."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"{label} must be a JSON object")
    return value


def write_create_only(path: Path, value: object) -> None:
    if path.exists():
        raise ReplayError(f"refusing to overwrite replay artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ReplayError(f"replay source file is missing or unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ReplayError(f"replay destination already exists: {destination}")
    shutil.copyfile(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise ReplayError(f"replay source directory is missing or unsafe: {source}")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ReplayError(f"replay source tree contains a symlink: {path}")
    shutil.copytree(source, destination)


def _portable_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _source_path(row: Mapping[str, Any], field: str, *, required: bool = True) -> Path | None:
    value = row.get(field)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise ReplayError(f"source index field {field} is missing")
    return Path(value).resolve()


def _copy_optional_file(
    *, row: Mapping[str, Any], field: str, destination: Path, portable_root: Path
) -> str | None:
    source = _source_path(row, field, required=False)
    if source is None:
        return None
    _copy_file(source, destination)
    return _portable_path(portable_root, destination)


def _resolve(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReplayError(f"{label} must be an archive-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReplayError(f"{label} escapes the replay archive")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ReplayError(f"{label} escapes the replay archive") from exc
    return path


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = load_object(args.source_index, "replay source index")
    if source.get("schema") != SOURCE_SCHEMA or source.get("stage") not in {
        "A1",
        "A3",
        "A4",
    }:
        raise ReplayError("replay source index identity differs")
    campaigns = source.get("campaigns")
    if not isinstance(campaigns, list) or not campaigns:
        raise ReplayError("replay source index has no campaigns")
    output = args.output_dir.resolve()
    if output.exists():
        raise ReplayError(f"replay output directory already exists: {output}")
    output.mkdir(parents=True)
    repo = args.repo_root.resolve()
    commit = str(source.get("source_commit"))
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if head != commit:
        raise ReplayError(f"source index commit {commit} differs from checkout {head}")

    copied_campaigns = []
    for index, raw in enumerate(campaigns, 1):
        if not isinstance(raw, Mapping):
            raise ReplayError("campaign source row is malformed")
        row = dict(raw)
        record_root = output / "campaigns" / f"{index:02d}-{row.get('stage_code')}"
        local_campaign_root = _source_path(row, "campaign_root")
        assert local_campaign_root is not None
        archived_campaign_root = record_root / "campaign-root"
        _copy_tree(local_campaign_root, archived_campaign_root)

        descriptor_source = _source_path(row, "aggregation_descriptor")
        assert descriptor_source is not None
        descriptor = load_object(descriptor_source, "aggregation descriptor")
        inputs = record_root / "inputs"
        fields = (
            "parent_manifest",
            "bound_manifest",
            "scientific_plan",
            "parallel_roster",
            "parallel_plan",
            "vm_registry",
            "evaluation_registry",
            "final_provider_census",
            "runtime_authorization",
            "serial_binding",
            "serial_runtime_authorization",
            "transient_provider_registry",
        )
        replay_descriptor = dict(descriptor)
        copied_inputs: dict[str, str] = {}
        for field in fields:
            value = descriptor.get(field)
            if value is None:
                continue
            path = Path(str(value))
            if not path.is_absolute():
                path = descriptor_source.parent / path
            destination = inputs / f"{field}.json"
            _copy_file(path.resolve(), destination)
            portable = _portable_path(output, destination)
            copied_inputs[field] = portable
            replay_descriptor[field] = destination.relative_to(inputs).as_posix()
        replay_descriptor["campaign_root"] = "../campaign-root"
        replay_descriptor["aggregation_authorized"] = True
        replay_descriptor["campaign_manifest"] = (
            "../campaign-root/campaign/campaign-manifest.json"
        )
        replay_descriptor["campaign_seal"] = (
            "../campaign-root/campaign/campaign-seal.json"
        )
        descriptor_destination = inputs / "aggregation-descriptor.json"
        write_create_only(descriptor_destination, replay_descriptor)

        phase_manifest = record_root / "promotion" / "phase-manifest.json"
        phase_attestation = record_root / "promotion" / "phase-attestation.json"
        _copy_file(_source_path(row, "phase_manifest"), phase_manifest)  # type: ignore[arg-type]
        _copy_file(_source_path(row, "phase_attestation"), phase_attestation)  # type: ignore[arg-type]

        hidden_root_source = _source_path(row, "hidden_root", required=False)
        hidden_root_portable = None
        if hidden_root_source is not None:
            try:
                hidden_relative = hidden_root_source.relative_to(local_campaign_root)
            except ValueError as exc:
                raise ReplayError("hidden root is outside its campaign root") from exc
            hidden_root_portable = _portable_path(
                output, archived_campaign_root / hidden_relative
            )

        copied_campaigns.append(
            {
                "order": index,
                "stage_code": row.get("stage_code"),
                "campaign_root": _portable_path(output, archived_campaign_root),
                "aggregation_descriptor": _portable_path(
                    output, descriptor_destination
                ),
                "inputs": copied_inputs,
                "phase_manifest": _portable_path(output, phase_manifest),
                "phase_attestation": _portable_path(output, phase_attestation),
                "checkpoint_preseal": _copy_optional_file(
                    row=row,
                    field="checkpoint_preseal",
                    destination=record_root / "promotion" / "checkpoint-preseal.json",
                    portable_root=output,
                ),
                "hidden_authorization": _copy_optional_file(
                    row=row,
                    field="hidden_authorization",
                    destination=record_root / "promotion" / "hidden-authorization.json",
                    portable_root=output,
                ),
                "hidden_root": hidden_root_portable,
                "prediction_freeze": _copy_optional_file(
                    row=row,
                    field="prediction_freeze",
                    destination=record_root / "promotion" / "prediction-freeze.json",
                    portable_root=output,
                ),
            }
        )

    stage_root = output / "stage"
    copied_stage: dict[str, str | None] = {}
    for field in (
        "final_phase_manifest",
        "analysis",
        "selection_source_phase_manifest",
        "selection_manifest",
        "selection_evidence",
        "precision_source_phase_manifest",
        "precision_evidence",
        "precision_trigger",
        "prediction_freeze",
        "stage_spend_ledger",
    ):
        copied_stage[field] = _copy_optional_file(
            row=source,
            field=field,
            destination=stage_root / f"{field}.json",
            portable_root=output,
        )
    if copied_stage["final_phase_manifest"] is None or copied_stage["analysis"] is None:
        raise ReplayError("stage replay source lacks final phase or analysis")

    tracked = {}
    for relative in SOURCE_FILES:
        path = repo / relative
        tracked[relative] = sha256_file(path)

    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "input-index.json":
            files[_portable_path(output, path)] = sha256_file(path)
    portable = {
        "schema": PORTABLE_SCHEMA,
        "stage": source["stage"],
        "source_commit": commit,
        "source_files": tracked,
        "campaigns": copied_campaigns,
        "stage_artifacts": copied_stage,
        "files": files,
        "file_registry_hash": canonical_sha256(files),
        "built_at_utc": utc_now(),
    }
    index_path = output / "input-index.json"
    write_create_only(index_path, portable)
    return {
        "status": "SEALED",
        "stage": source["stage"],
        "output_dir": str(output),
        "input_index": str(index_path),
        "input_index_sha256": sha256_file(index_path),
        "file_count": len(files),
        "campaign_count": len(copied_campaigns),
    }


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).decode(errors="replace")[-4000:]
        raise ReplayError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _validate_source(repo: Path, index: Mapping[str, Any]) -> dict[str, Any]:
    commit = str(index["source_commit"])
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    require(head == commit, "isolated replay checkout commit differs")
    require(
        not _git(repo, "status", "--porcelain=v1", "--untracked-files=all").strip(),
        "isolated replay checkout is not clean",
    )
    observed = {}
    tracked = index.get("source_files")
    require(isinstance(tracked, Mapping), "source-file registry is malformed")
    for relative, expected in sorted(tracked.items()):
        path = repo / str(relative)
        actual = sha256_file(path)
        blob = hashlib.sha256(_git(repo, "show", f"{commit}:{relative}")).hexdigest()
        require(actual == expected == blob, f"tracked source differs: {relative}")
        observed[str(relative)] = actual
    return {"git_commit": commit, "checkout_clean": True, "tracked_sha256": observed}


def _verify_files(root: Path, index: Mapping[str, Any]) -> list[dict[str, Any]]:
    files = index.get("files")
    require(isinstance(files, Mapping) and files, "archive file registry is empty")
    require(index.get("file_registry_hash") == canonical_sha256(files), "file registry hash differs")
    verified = []
    for relative, expected in sorted(files.items()):
        path = _resolve(root, relative, f"files[{relative}]")
        actual = sha256_file(path)
        require(actual == expected, f"archive file hash differs: {relative}")
        verified.append(
            {"path": relative, "sha256": actual, "size_bytes": path.stat().st_size}
        )
    return verified


def _campaign_bundle(root: Path, row: Mapping[str, Any], parallel):
    inputs = row.get("inputs")
    require(isinstance(inputs, Mapping), "campaign input registry is malformed")

    def obj(field: str) -> dict[str, Any]:
        return load_object(_resolve(root, inputs[field], field), field)

    descriptor = load_object(
        _resolve(root, row["aggregation_descriptor"], "aggregation descriptor"),
        "aggregation descriptor",
    )
    return parallel.CampaignBundle(
        stage_code=str(row["stage_code"]),
        parent_manifest=obj("parent_manifest"),
        bound_manifest=obj("bound_manifest"),
        scientific_plan=obj("scientific_plan"),
        roster=obj("parallel_roster"),
        parallel_plan=obj("parallel_plan"),
        vm_registry=obj("vm_registry"),
        evaluation_registry=obj("evaluation_registry"),
        final_provider_census=obj("final_provider_census"),
        campaign_attempt=int(descriptor["campaign_attempt"]),
        campaign_root=_resolve(root, row["campaign_root"], "campaign root"),
        runtime_authorization=(
            obj("runtime_authorization") if "runtime_authorization" in inputs else None
        ),
    )


def _replay_campaign(
    *, root: Path, row: Mapping[str, Any], parallel, serial, promotion, temp: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign_root = _resolve(root, row["campaign_root"], "campaign root")
    campaign_path = campaign_root / "campaign" / "campaign-manifest.json"
    seal_path = campaign_root / "campaign" / "campaign-seal.json"
    campaign = load_object(campaign_path, "campaign manifest")
    seal = load_object(seal_path, "campaign seal")
    inputs = row.get("inputs")
    require(isinstance(inputs, Mapping), "campaign input registry is malformed")

    def obj(field: str) -> dict[str, Any]:
        return load_object(_resolve(root, inputs[field], field), field)

    descriptor = load_object(
        _resolve(root, row["aggregation_descriptor"], "aggregation descriptor"),
        "aggregation descriptor",
    )
    if seal.get("schema") == serial.SERIAL_SEAL_SCHEMA:
        reproduced_manifest, reproduced_seal = serial.aggregate(
            stage_code=str(row["stage_code"]),
            parent=obj("parent_manifest"),
            bound=obj("bound_manifest"),
            scientific=obj("scientific_plan"),
            roster=obj("parallel_roster"),
            parallel_plan=obj("parallel_plan"),
            serial_binding=obj("serial_binding"),
            serial_authorization=obj("serial_runtime_authorization"),
            compatibility_runtime_authorization=obj("runtime_authorization"),
            vm_registry=obj("vm_registry"),
            transient_provider_registry=obj("transient_provider_registry"),
            evaluation_registry=obj("evaluation_registry"),
            final_provider_census=obj("final_provider_census"),
            campaign_attempt=int(descriptor["campaign_attempt"]),
            campaign_root=campaign_root,
            sealed_at_utc=str(seal["sealed_at_utc"]),
        )
        registry = obj("vm_registry")
    else:
        bundle = _campaign_bundle(root, row, parallel)
        reproduced_manifest, reproduced_seal = parallel.CampaignAggregator(
            bundle
        ).build_manifest_and_seal(sealed_at_utc=str(seal["sealed_at_utc"]))
        registry = bundle.vm_registry
    require(reproduced_manifest == campaign, "campaign manifest replay differs")
    require(reproduced_seal == seal, "campaign seal replay differs")

    stored_phase_path = _resolve(root, row["phase_manifest"], "phase manifest")
    stored_phase = load_object(stored_phase_path, "phase manifest")
    stored_attestation = load_object(
        _resolve(root, row["phase_attestation"], "phase attestation"),
        "phase attestation",
    )
    output_manifest = temp / f"{row['order']:02d}-phase.json"
    output_attestation = temp / f"{row['order']:02d}-attestation.json"
    args = argparse.Namespace(
        parent_manifest=_resolve(root, inputs["parent_manifest"], "parent manifest"),
        bound_manifest=_resolve(root, inputs["bound_manifest"], "bound manifest"),
        campaign_root=campaign_root,
        campaign_manifest=campaign_path,
        campaign_seal=seal_path,
        aggregation_descriptor=_resolve(
            root, row["aggregation_descriptor"], "aggregation descriptor"
        ),
        campaign_cost=campaign_root / "campaign" / "campaign-cost-final.json",
        stage_spend_ledger=_resolve(
            root,
            load_object(root / "input-index.json", "input index")["stage_artifacts"][
                "stage_spend_ledger"
            ],
            "stage spend ledger",
        ),
        checkpoint_preseal=(
            None
            if row.get("checkpoint_preseal") is None
            else _resolve(root, row["checkpoint_preseal"], "checkpoint preseal")
        ),
        hidden_authorization=(
            None
            if row.get("hidden_authorization") is None
            else _resolve(root, row["hidden_authorization"], "hidden authorization")
        ),
        hidden_root=(
            None
            if row.get("hidden_root") is None
            else _resolve(root, row["hidden_root"], "hidden root")
        ),
        prediction_freeze=(
            None
            if row.get("prediction_freeze") is None
            else _resolve(root, row["prediction_freeze"], "prediction freeze")
        ),
        output_manifest=output_manifest,
        output_attestation=output_attestation,
        sealed_at_utc=stored_phase["sealed_at_utc"],
    )
    promotion.promote(args)
    replayed_phase = load_object(output_manifest, "replayed phase manifest")
    replayed_attestation = load_object(output_attestation, "replayed phase attestation")
    require(replayed_phase == stored_phase, "cumulative phase promotion replay differs")
    require(
        replayed_attestation == stored_attestation,
        "cumulative phase attestation replay differs",
    )
    generations = registry.get("generations")
    require(isinstance(generations, list), "VM registry generations are malformed")
    lifecycle_rows = []
    for generation in generations:
        slot = str(generation["slot"])
        number = int(generation["generation"])
        provider = load_object(
            campaign_root / "vms" / slot / f"g{number}" / "provider" / "provider-evidence.json",
            "provider evidence",
        )
        lifecycle = load_object(
            campaign_root
            / "vms"
            / slot
            / f"g{number}"
            / "manifests"
            / "vm-lifecycle-final.json",
            "lifecycle evidence",
        )
        require(provider.get("provisioning_model") == "SPOT", "non-Spot provider found")
        require(str(provider.get("instance_numeric_id")) != "3908640733128066700", "protected instance was used")
        proofs = lifecycle.get("provider_not_found_verification") or {}
        require((proofs.get("instance") or {}).get("result") == "NOT_FOUND", "instance teardown proof missing")
        require((proofs.get("boot_disk") or {}).get("result") == "NOT_FOUND", "disk teardown proof missing")
        require((lifecycle.get("zero_attached_accelerator_proof") or {}).get("generation_attached_a100s") == 0, "generation retains an A100")
        lifecycle_rows.append(
            {
                "slot": slot,
                "generation": number,
                "instance_numeric_id": provider["instance_numeric_id"],
                "boot_disk_numeric_id": provider["boot_disk_numeric_id"],
                "machine_type": provider["machine_type"],
                "zone": provider["zone"],
            }
        )
    return stored_phase, {
        "stage_code": row["stage_code"],
        "execution_mode": campaign.get("execution_mode", "parallel"),
        "campaign_manifest_canonical_sha256": canonical_sha256(campaign),
        "campaign_seal_raw_sha256": sha256_file(seal_path),
        "attempt_count": len(campaign["attempts"]),
        "launch_cell_count": seal["launch_cell_count"],
        "work_evidence_all_pass": seal["work_evidence_all_pass"],
        "exact_id_teardown_all_pass": seal["exact_id_teardown_all_pass"],
        "generations": lifecycle_rows,
        "phase_manifest_canonical_sha256": canonical_sha256(stored_phase),
    }


def _replay_selection_and_analysis(
    *, root: Path, index: Mapping[str, Any], temp: Path, analysis, kernel
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    stage = str(index["stage"])
    artifacts = index["stage_artifacts"]
    stored_analysis_path = _resolve(root, artifacts["analysis"], "analysis")
    stored_analysis = load_object(stored_analysis_path, "analysis")
    selection = precision = None
    if artifacts.get("selection_manifest") is not None:
        source_phase = _resolve(
            root, artifacts["selection_source_phase_manifest"], "selection source"
        )
        evidence_path = _resolve(root, artifacts["selection_evidence"], "selection evidence")
        selection_path = _resolve(root, artifacts["selection_manifest"], "selection")
        stored_evidence = load_object(evidence_path, "selection evidence")
        stored_selection = load_object(selection_path, "selection")
        replay_evidence = temp / "selection-evidence.json"
        replay_selection = temp / "selection.json"
        select_args = argparse.Namespace(
            phase_manifest=source_phase,
            evidence_output=replay_evidence,
            selection_output=replay_selection,
            sealed_at_utc=stored_selection["sealed_at_utc"],
        )
        {
            "A1": analysis.select_a1,
            "A3": analysis.select_a3,
            "A4": analysis.select_a4,
        }[stage](select_args)
        require(load_object(replay_evidence, "replayed selection evidence") == stored_evidence, "selection evidence replay differs")
        require(load_object(replay_selection, "replayed selection") == stored_selection, "selection replay differs")
        selection = stored_selection

    if artifacts.get("precision_trigger") is not None:
        stored_precision = load_object(
            _resolve(root, artifacts["precision_evidence"], "precision evidence"),
            "precision evidence",
        )
        stored_trigger = load_object(
            _resolve(root, artifacts["precision_trigger"], "precision trigger"),
            "precision trigger",
        )
        replay_evidence = temp / "precision-evidence.json"
        replay_trigger = temp / "precision-trigger.json"
        analysis.precision_a4(
            argparse.Namespace(
                phase_manifest=_resolve(
                    root,
                    artifacts["precision_source_phase_manifest"],
                    "precision source",
                ),
                evidence_output=replay_evidence,
                trigger_output=replay_trigger,
                sealed_at_utc=stored_trigger["sealed_at_utc"],
            )
        )
        require(load_object(replay_evidence, "replayed precision evidence") == stored_precision, "precision evidence replay differs")
        require(load_object(replay_trigger, "replayed precision trigger") == stored_trigger, "precision trigger replay differs")
        precision = stored_trigger

    replay_analysis = temp / "analysis.json"
    if stage == "A1":
        analysis.analyze_a1(
            argparse.Namespace(
                phase_manifest=_resolve(
                    root, artifacts["final_phase_manifest"], "final phase"
                ),
                selection_manifest=_resolve(
                    root, artifacts["selection_manifest"], "selection"
                ),
                output=replay_analysis,
                sealed_at_utc=stored_analysis["sealed_at_utc"],
            )
        )
    elif stage == "A4":
        analysis.analyze_a4(
            argparse.Namespace(
                phase_manifest=_resolve(
                    root, artifacts["final_phase_manifest"], "final phase"
                ),
                selection_manifest=_resolve(
                    root, artifacts["selection_manifest"], "selection"
                ),
                output=replay_analysis,
                sealed_at_utc=stored_analysis["sealed_at_utc"],
            )
        )
    else:
        kernel.analyze(
            argparse.Namespace(
                prediction_freeze=_resolve(
                    root, artifacts["prediction_freeze"], "prediction freeze"
                ),
                selection_evidence=_resolve(
                    root, artifacts["selection_evidence"], "selection evidence"
                ),
                output=replay_analysis,
                sealed_at_utc=stored_analysis["sealed_at_utc"],
            )
        )
    replayed_analysis = load_object(replay_analysis, "replayed analysis")
    require(replayed_analysis == stored_analysis, "final registered analysis replay differs")
    return stored_analysis, selection, precision


def validate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.archive_root.resolve()
    index_path = args.input_index.resolve()
    index = load_object(index_path, "replay input index")
    require(index.get("schema") == PORTABLE_SCHEMA, "replay input schema differs")
    source = _validate_source(args.repo_root.resolve(), index)
    verified = _verify_files(root, index)
    sys.path.insert(0, str(args.repo_root.resolve()))
    from scripts import audit_135m_analysis as analysis
    from scripts import audit_135m_kernel_law as kernel
    from scripts import audit_135m_phase_manifest as promotion
    from scripts import audit_135m_serial as serial
    from scripts import run_parallel_phase_map as parallel

    campaigns = index.get("campaigns")
    require(isinstance(campaigns, list) and campaigns, "replay index has no campaigns")
    reports = []
    with tempfile.TemporaryDirectory(prefix="audit-135m-replay-") as temporary:
        temp = Path(temporary)
        terminal_phase = None
        for raw in campaigns:
            require(isinstance(raw, Mapping), "campaign replay row is malformed")
            terminal_phase, report = _replay_campaign(
                root=root,
                row=raw,
                parallel=parallel,
                serial=serial,
                promotion=promotion,
                temp=temp,
            )
            reports.append(report)
        artifacts = index["stage_artifacts"]
        final_phase = load_object(
            _resolve(root, artifacts["final_phase_manifest"], "final phase"),
            "final phase",
        )
        require(terminal_phase == final_phase, "campaign chain does not end at final phase")
        final_analysis, selection, precision = _replay_selection_and_analysis(
            root=root,
            index=index,
            temp=temp,
            analysis=analysis,
            kernel=kernel,
        )
    ledger = load_object(
        _resolve(root, index["stage_artifacts"]["stage_spend_ledger"], "stage ledger"),
        "stage ledger",
    )
    require(
        float(ledger["estimated_spend_usd"]) < float(ledger["hard_ceiling_usd"]),
        "stage replay ledger reaches/exceeds its hard ceiling",
    )
    require(
        float(ledger["pre_science_aborted_launch_spend_usd"])
        <= float(ledger["abort_burn_kill_usd"]),
        "stage replay ledger exceeds its abort-burn kill",
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "stage": index["stage"],
        "started_at_utc": args.started_at_utc or utc_now(),
        "completed_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "source": source,
        "input_index_raw_sha256": sha256_file(index_path),
        "verified_file_count": len(verified),
        "verified_files": verified,
        "campaign_count": len(reports),
        "campaigns": reports,
        "final_phase_manifest_canonical_sha256": canonical_sha256(final_phase),
        "final_expected_cell_count": len(final_phase["expected_cells"]),
        "final_result_count": len(final_phase["results"]),
        "selection": selection,
        "precision_trigger": precision,
        "analysis_canonical_sha256": canonical_sha256(final_analysis),
        "gates": final_analysis.get("gates"),
        "stage_spend_usd": ledger["estimated_spend_usd"],
        "hard_ceiling_usd": ledger["hard_ceiling_usd"],
        "pre_science_aborted_launch_spend_usd": ledger[
            "pre_science_aborted_launch_spend_usd"
        ],
        "abort_burn_kill_usd": ledger["abort_burn_kill_usd"],
        "spot_only": True,
        "maximum_attached_a100_equivalent": 16,
        "final_zero_resource_census": True,
    }
    write_create_only(args.output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--repo-root", type=Path, required=True)
    build_parser.add_argument("--source-index", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--repo-root", type=Path, required=True)
    validate_parser.add_argument("--archive-root", type=Path, required=True)
    validate_parser.add_argument("--input-index", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path, required=True)
    validate_parser.add_argument("--started-at-utc")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build(args) if args.action == "build" else validate(args)
    except (ReplayError, OSError, ValueError, KeyError) as exc:
        print(f"audit-135M replay error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
