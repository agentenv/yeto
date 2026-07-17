#!/usr/bin/env python3
"""Audit-135M wrapper around the sealed P1 per-generation worker.

The packet carries the reviewed P1 worker as ``p1r0_vm_worker_base.py``.  This
wrapper replaces only its static binding and seed-selection layer so one VM can
execute the cumulative A1/A3/A4 plans, validates the exact current-stage seed
bundle, and adds checkpoint/export files to the formal evidence registry.  It
never emits a loss through public status.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load packet module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("audit_135m_vm_worker_base", HERE / "p1r0_vm_worker_base.py")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(base.canonical_json(value)).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a JSON object")
    return dict(value)


def audit_init(self, config_path: Path) -> None:
    self.config_path = config_path.resolve()
    self.config = base.load_json(self.config_path)
    self.repo = Path(self.config["repo_root"])
    sys.path.insert(0, str(self.repo))
    from scripts import run_phase_map as phase

    packet = Path(self.config["packet_root"])
    pexec_path = packet / "run_parallel_phase_map.py"
    pexec_spec = importlib.util.spec_from_file_location(
        "audit_135m_runtime_parallel_phase_map", pexec_path
    )
    if pexec_spec is None or pexec_spec.loader is None:
        raise RuntimeError("cannot load packet-bound parallel executor")
    pexec = importlib.util.module_from_spec(pexec_spec)
    sys.modules[pexec_spec.name] = pexec
    pexec_spec.loader.exec_module(pexec)

    self.pexec = pexec
    self.phase = phase
    self.run_root = Path(self.config["remote_run_dir"])
    self.control = self.run_root / "control"
    self.inbox = self.control / "inbox"
    self.outcomes = self.control / "outcomes"
    self.needs_fix = self.control / "needs-fix"
    self.public_status = self.control / "public-status.json"
    self.gen_relative = self.config["generation_campaign_relative"]
    self.provider_path = self.run_root / "provider" / "provider-evidence.json"
    self.provider_hash = base.sha256_file(self.provider_path)
    self.provider = base.load_json(self.provider_path)
    self.machine_type = str(self.provider["machine_type"])
    self.plan = base.load_json(packet / "scientific-randomization-plan.json")
    self.bound = base.load_json(packet / "bound-manifest.json")
    self.parent = base.load_json(packet / "parent-manifest.json")
    self.roster = base.load_json(packet / "parallel-roster.json")
    self.parallel_plan = base.load_json(packet / "parallel-plan.json")
    self.revision_binding = base.load_json(
        packet / "controller-amendment-revision.json"
    )
    self.runtime_authorization = base.load_json(
        packet / "runtime-authorization.json"
    )
    self.seed_registry = base.load_json(packet / "seed-bundle-registry.json")
    self.cells = {str(row["cell_id"]): row for row in self.plan["cells"]}
    self.phase_args = SimpleNamespace(
        seq_len=128,
        require_distinct_learner_gpu_uuids=False,
    )
    self.expected_eval = None
    self.parallel_eval_path = None
    self.parallel_eval_entry = None


def _select_eval(self, cell: Mapping[str, Any]) -> None:
    seed = int(cell["seed"])
    root = (
        Path(self.config["science_root"])
        / "phase-map"
        / "frozen-eval"
        / f"seed-{seed}"
    )
    expected = base.load_json(root / "eval-freeze.json")
    expected["_eval_sequences_path"] = str(
        root / "provenance" / "eval_sequences.jsonl"
    )
    parallel_path = root / "parallel-eval-freeze.json"
    self.expected_eval = expected
    self.parallel_eval_path = parallel_path
    self.parallel_eval_entry = {
        "path": f"common/evaluation/seed-{seed}.json",
        "sha256": base.sha256_file(parallel_path),
        **{
            field: self.bound["frozen"][field]
            for field in self.pexec.EVAL_BOUND_FIELDS
        },
    }


def audit_verify_static_state(self) -> None:
    head = subprocess.run(
        ["git", "-C", str(self.repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(self.repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != self.config["source_commit"] or dirty:
        raise RuntimeError("worker source checkout is not exact and clean")
    if self.pexec.canonical_sha256(self.bound) != self.config[
        "bound_manifest_canonical_sha256"
    ]:
        raise RuntimeError("worker bound manifest hash differs")
    rebuilt_roster = self.pexec.build_parallel_roster(
        stage_code=self.config["stage_code"],
        bound_manifest=self.bound,
        parent_manifest=self.parent,
        scientific_plan=self.plan,
    )
    if self.pexec.canonical_json(rebuilt_roster) != self.pexec.canonical_json(
        self.roster
    ):
        raise RuntimeError("worker roster does not reconstruct exactly")
    if self.pexec.roster_hash(self.roster) != self.config["roster_hash"]:
        raise RuntimeError("worker roster hash differs")
    rebuilt_plan = self.pexec.build_parallel_plan(
        self.roster, expected_roster_hash=self.config["roster_hash"]
    )
    if self.pexec.canonical_json(rebuilt_plan) != self.pexec.canonical_json(
        self.parallel_plan
    ):
        raise RuntimeError("worker parallel plan does not reconstruct exactly")
    if self.pexec.parallel_plan_hash(self.parallel_plan) != self.config[
        "parallel_plan_hash"
    ]:
        raise RuntimeError("worker parallel plan hash differs")
    if self.plan["randomization_plan_hash"] != self.config[
        "scientific_randomization_plan_hash"
    ]:
        raise RuntimeError("worker scientific plan hash differs")
    authorization_hash = self.pexec.multiseed_runtime_authorization_hash(
        stage_code=self.config["stage_code"],
        design_contract_hash=self.roster["audit_135m_design_contract_hash"],
        roster_digest=self.config["roster_hash"],
        parallel_digest=self.config["parallel_plan_hash"],
        bound_digest=self.config["bound_manifest_canonical_sha256"],
        scientific_digest=self.config["scientific_randomization_plan_hash"],
        hard_ceiling_usd=self.roster["hard_ceiling_usd"],
        authorization=self.runtime_authorization,
    )
    if authorization_hash != self.config["runtime_authorization_hash"]:
        raise RuntimeError("worker runtime authorization hash differs")
    if (
        self.revision_binding.get("controller_commit")
        != self.config["controller_commit"]
        or self.revision_binding.get("parallel_plan_hash")
        != self.config["parallel_plan_hash"]
        or self.revision_binding.get("science_commands_unchanged") is not True
    ):
        raise RuntimeError("worker controller binding differs")
    dataset = Path(self.config["science_root"]) / "inputs" / "train.parquet"
    if base.sha256_file(dataset) != self.config["data_sha256"]:
        raise RuntimeError("worker dataset hash differs")
    model_root = Path(self.config["science_root"]) / "inputs" / "model"
    checked = subprocess.run(
        ["sha256sum", "-c", "model-files.sha256"],
        cwd=model_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if checked.returncode:
        raise RuntimeError("worker model manifest verification failed")
    seeds = self.seed_registry.get("seeds")
    expected_seeds = {str(cell["seed"]) for cell in self.plan["cells"]}
    if (
        self.seed_registry.get("schema")
        != "audit_135m_seed_bundle_registry_v1"
        or set(seeds or {}) != expected_seeds
        or self.seed_registry.get("audit_objects_included") is not False
        or self.seed_registry.get("audit_model_evaluation_accesses") != []
    ):
        raise RuntimeError("worker seed registry differs from the exact suffix")
    frozen_root = Path(self.config["science_root"]) / "phase-map" / "frozen-eval"
    for seed_text in sorted(expected_seeds, key=int):
        entry = _mapping(seeds[seed_text], f"seed {seed_text} registry")
        seed_root = frozen_root / f"seed-{seed_text}"
        train = seed_root / "materialized" / "train.jsonl"
        split = base.load_json(seed_root / "materialized" / "split_provenance.json")
        parallel_eval = seed_root / "parallel-eval-freeze.json"
        if (
            base.sha256_file(train) != entry["train_rows_sha256"]
            or canonical_sha256(split["train_source_indices"])
            != entry["train_source_indices_sha256"]
            or base.sha256_file(parallel_eval)
            != entry["parallel_eval_freeze_sha256"]
            or int(split["train_shuffle_seed"]) != int(seed_text)
        ):
            raise RuntimeError(f"worker seed {seed_text} materialization differs")
    self.pexec.validate_provider_record(self.provider, self.config["identity"])


def audit_common_inventory(self, attempt_dir: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "command": attempt_dir / "command.json",
        "attempt_start": attempt_dir / "attempt-start.json",
        "compare_log": attempt_dir / "compare.log",
        "gpu_monitor": attempt_dir / "gpu-monitor.csv",
    }
    work_root = attempt_dir / "work"
    arm_dirs = [path for path in work_root.iterdir() if path.is_dir()] if work_root.is_dir() else []
    if len(arm_dirs) == 1:
        for log in sorted(arm_dirs[0].glob("learner-*.log")):
            learner_id = log.stem.removeprefix("learner-")
            paths[f"learner_{learner_id}_log"] = log
        for name, role in (
            ("finite-kernel-capture.log", "finite_kernel_capture_log"),
            ("finite-kernel-status.json", "finite_kernel_capture_status"),
        ):
            paths[role] = arm_dirs[0] / name
    return {
        role: base.inventory_entry(self._relative(path), path)
        for role, path in paths.items()
        if path.is_file()
    }


def audit_run_command(
    self, request: Mapping[str, Any], attempt_dir: Path
) -> tuple[int, str, str, dict[str, int]]:
    cell_id = str(request["cell_id"])
    cell = self.cells[cell_id]
    arm_name = self.phase.cell_arm_name(cell)
    learner_count = int(cell["target_work"]["learner_count"])
    command = self.pexec.project_scientific_command_for_machine_type(
        request["command"], self.machine_type
    )
    executed_command_hash = self.pexec.canonical_sha256(command)
    base.write_json_create_only(attempt_dir / "command.json", command)
    started = base.utc_now()
    base.write_json_create_only(
        attempt_dir / "attempt-start.json",
        {
            "attempt_id": f"{cell_id}-attempt-{request['retry_round']}",
            "cell_id": cell_id,
            "attempt": int(request["retry_round"]),
            "started_at_utc": started,
            "command_hash": executed_command_hash,
            "frozen_command_hash": request["command_hash"],
            "machine_type": self.machine_type,
            "gpu_slots": self.pexec.machine_shape_contract(self.machine_type)[
                "gpu_slots"
            ],
            "provider_evidence_sha256": self.provider_hash,
            "fresh_initial_state": True,
            "resumed_from_attempt": None,
            "optimizer_state_input": None,
            "checkpoint_input": None,
            "prior_attempt_artifacts_used": False,
        },
    )
    log_path = attempt_dir / "compare.log"
    gpu_log = attempt_dir / "gpu-monitor.csv"
    env = dict(os.environ)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": str(self.repo),
        }
    )
    maxima = {"gpu_utilization_percent": 0, "gpu_memory_mib": 0, "samples": 0}
    with log_path.open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=attempt_dir,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            utilization, memory = self._gpu_sample(gpu_log)
            maxima["gpu_utilization_percent"] = max(
                maxima["gpu_utilization_percent"], utilization
            )
            maxima["gpu_memory_mib"] = max(maxima["gpu_memory_mib"], memory)
            maxima["samples"] += 1
            tape = attempt_dir / "work" / arm_name / "tape.jsonl"
            learner_lines = {
                str(learner): self._line_count(
                    attempt_dir
                    / "work"
                    / arm_name
                    / f"learner-{learner}.log"
                )
                for learner in range(learner_count)
            }
            self.public(
                phase="TRAINING",
                current_cell_id=cell_id,
                retry_round=int(request["retry_round"]),
                observed_outer_commits=self._line_count(tape),
                learner_log_lines=learner_lines,
                gpu_samples=maxima["samples"],
                maximum_gpu_utilization_percent=maxima[
                    "gpu_utilization_percent"
                ],
                maximum_gpu_memory_mib=maxima["gpu_memory_mib"],
            )
            time.sleep(15)
        exit_code = int(process.returncode)
    utilization, memory = self._gpu_sample(gpu_log)
    maxima["gpu_utilization_percent"] = max(
        maxima["gpu_utilization_percent"], utilization
    )
    maxima["gpu_memory_mib"] = max(maxima["gpu_memory_mib"], memory)
    maxima["samples"] += 1
    return exit_code, started, base.utc_now(), maxima


def audit_diverged_outcome(
    self,
    request: Mapping[str, Any],
    attempt_dir: Path,
    started: str,
    ended: str,
    detail: str,
) -> dict[str, Any]:
    cell = self.cells[str(request["cell_id"])]
    arm_name = self.phase.cell_arm_name(cell)
    report = attempt_dir / "report" / "parallel-evidence"
    report.mkdir(parents=True, exist_ok=True)
    tape_source = attempt_dir / "work" / arm_name / "tape.jsonl"
    tape_rows = base.read_jsonl(tape_source) if tape_source.is_file() else []
    prefix_path = report / "tape-prefix.jsonl"
    base.write_jsonl(
        prefix_path,
        [
            {"outer_step": int(row["step"]), "fragment": int(row["fragment"])}
            for row in tape_rows
        ],
    )
    divergence_path = report / "scientific-divergence.json"
    base.write_json_create_only(
        divergence_path,
        {
            "schema": "yeto_parallel_scientific_divergence_v1",
            "cell_id": request["cell_id"],
            "attempt_id": f"{request['cell_id']}-attempt-{request['retry_round']}",
            "last_finite_step": len(tape_rows),
            "first_nonfinite_event": {"classification": detail},
            "recorded_at_utc": base.utc_now(),
        },
    )
    inventory = self._common_inventory(attempt_dir)
    inventory["tape_prefix"] = base.inventory_entry(
        self._relative(prefix_path), prefix_path
    )
    inventory["scientific_divergence"] = base.inventory_entry(
        self._relative(divergence_path), divergence_path
    )
    return {
        "status": "DIVERGED",
        "failure_reason": None,
        "loss": None,
        "resumed": False,
        "resume_source": None,
        "scientific_started_at": started,
        "scientific_ended_at": ended,
        "artifact_inventory": inventory,
    }


def _checkpoint_registry(self, request: Mapping[str, Any], attempt_dir: Path) -> Path:
    cell = self.cells[str(request["cell_id"])]
    arm_name = self.phase.cell_arm_name(cell)
    arm_root = attempt_dir / "work" / arm_name
    checkpoint = arm_root / "state.ckpt"
    export = arm_root / "export"
    if not checkpoint.is_file() or checkpoint.is_symlink() or not export.is_dir():
        raise base.NeedsFix("completed audit cell lacks checkpoint/export evidence")
    files = []
    for path in [checkpoint, *sorted(export.rglob("*"))]:
        if not path.is_file() or path.is_symlink():
            continue
        files.append(
            {
                "path": path.relative_to(attempt_dir).as_posix(),
                "sha256": base.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if len(files) < 2:
        raise base.NeedsFix("checkpoint/export registry is incomplete")
    value = {
        "schema": "audit_135m_checkpoint_inventory_v1",
        "cell_id": request["cell_id"],
        "attempt": int(request["retry_round"]),
        "loss_exposed": False,
        "files": files,
    }
    value["inventory_canonical_sha256"] = canonical_sha256(value)
    path = attempt_dir / "report" / "parallel-evidence" / "checkpoint-inventory.json"
    base.write_json_create_only(path, value)
    return path


def _add_checkpoint_inventory_files(
    self,
    *,
    attempt_dir: Path,
    inventory_path: Path,
    inventory: dict[str, Any],
) -> None:
    value = base.load_json(inventory_path)
    for index, row in enumerate(value["files"]):
        path = attempt_dir / row["path"]
        inventory[f"checkpoint_file_{index:04d}"] = base.inventory_entry(
            self._relative(path), path
        )


def _finite_kernel_capture(
    self, *, cell: Mapping[str, Any], attempt_dir: Path
) -> Path | None:
    arm_name = self.phase.cell_arm_name(dict(cell))
    path = attempt_dir / "work" / arm_name / "finite-kernel.json"
    required = cell.get("finite_kernel_capture_required") is True
    if not required:
        if path.exists():
            raise base.NeedsFix("unregistered cell emitted a finite-kernel capture")
        return None
    if not path.is_file() or path.is_symlink():
        raise base.NeedsFix("registered finite-kernel cell lacks its compact seal")
    value = base.load_json(path)
    preimage = dict(value)
    digest = preimage.pop("capture_canonical_sha256", None)
    target = cell["target_work"]
    if (
        value.get("schema") != "audit_135m_finite_kernel_capture_v1"
        or value.get("status") != "SEALED"
        or value.get("loss_exposed") is not False
        or digest != canonical_sha256(preimage)
        or value.get("expected_outer_steps") != target["outer_steps"]
        or value.get("observed_outer_steps") != target["outer_steps"]
        or value.get("learner_count") != target["learner_count"]
        or value.get("K_H") != target["outer_steps"]
        or value.get("state_transition_replay_exact") is not True
        or value.get("all_registered_updates_covered") is not True
        or value.get("large_capture_cleanup_complete") is not True
        or not isinstance(value.get("V_H_psd"), (int, float))
        or float(value["V_H_psd"]) <= 0.0
    ):
        raise base.NeedsFix("finite-kernel capture identity/coverage/hash differs")
    if (attempt_dir / "work" / arm_name / "syncer_probe").exists() or (
        attempt_dir / "work" / arm_name / "finite-kernel-scratch"
    ).exists():
        raise base.NeedsFix("finite-kernel cell retained a large raw capture")
    return path


def audit_completed_outcome(
    self,
    request: Mapping[str, Any],
    attempt_dir: Path,
    started: str,
    ended: str,
    maxima: Mapping[str, int],
) -> dict[str, Any]:
    cell = self.cells[str(request["cell_id"])]
    checkpoint_only = cell.get("evaluation_mode") != "development_endpoint"
    if checkpoint_only:
        results_path = attempt_dir / "report" / "results.jsonl"
        rows = base.read_jsonl(results_path)
        arm_name = self.phase.cell_arm_name(cell)
        matching = [row for row in rows if row.get("arm") == arm_name]
        learner_count = int(cell["target_work"]["learner_count"])
        expected_steps = int(cell["target_work"]["learner_steps_per_learner"])
        if len(rows) != 1 or len(matching) != 1:
            raise base.NeedsFix("checkpoint-only compare output is not one exact arm")
        result = matching[0]
        if (
            result.get("checkpoint_only") is not True
            or result.get("evaluation_role") != "none"
            or result.get("eval_loss") is not None
            or result.get("syncer_exit_code") != 0
            or result.get("learner_exit_codes") != [0] * learner_count
        ):
            raise base.NeedsFix("checkpoint-only result/exit contract differs")
        tape_path = attempt_dir / "work" / arm_name / "tape.jsonl"
        tape = base.read_jsonl(tape_path)
        observed_work = self.phase.validate_tape(
            tape_path, cell, self.phase_args
        )
        barrier = self.phase.validate_barrier_version_trace(
            attempt_dir, tape, cell, self.phase_args
        )
        if barrier["inner_step_counts"] != {
            learner: expected_steps for learner in range(learner_count)
        }:
            raise base.NeedsFix("checkpoint-only barrier proof lacks every learner step")
        if maxima["gpu_utilization_percent"] <= 0 or maxima["gpu_memory_mib"] <= 0:
            raise base.NeedsFix("nvidia-smi did not observe checkpoint-only GPU work")
        self.phase.validate_layout(attempt_dir, cell)
        report = attempt_dir / "report" / "parallel-evidence"
        report.mkdir(parents=True, exist_ok=False)
        learner_steps_path = report / "learner-steps.json"
        base.write_json_create_only(
            learner_steps_path,
            {
                "schema": "yeto_parallel_learner_steps_v1",
                "learners": {
                    str(learner): list(range(1, expected_steps + 1))
                    for learner in range(learner_count)
                },
            },
        )
        updates = [
            {
                "outer_step": int(row["step"]),
                "fragment": int(row["fragment"]),
                "responders": [
                    {
                        "learner_id": int(responder["id"]),
                        "base_version": int(responder["base_version"]),
                        "microsteps": int(responder["c_steps"]),
                        "tokens": int(responder["c_tokens"]),
                        "version_matched_anchor": bool(
                            responder["anchor_base_resolved"]
                        ),
                    }
                    for responder in row["responders"]
                ],
            }
            for row in tape
        ]
        work_events_path = report / "work-events.json"
        base.write_json_create_only(
            work_events_path,
            {"schema": "yeto_parallel_work_events_v1", "updates": updates},
        )
        learners = {}
        for learner in range(learner_count):
            pushes = []
            broadcasts = []
            for update in updates:
                responder = next(
                    row
                    for row in update["responders"]
                    if row["learner_id"] == learner
                )
                pushes.append(
                    {
                        "outer_step": update["outer_step"],
                        "fragment": update["fragment"],
                        "base_version": responder["base_version"],
                    }
                )
                broadcasts.append(
                    {
                        "outer_step": update["outer_step"],
                        "fragment": update["fragment"],
                        "pushed_base_version": responder["base_version"],
                        "broadcast_version": update["outer_step"],
                    }
                )
            learners[str(learner)] = {
                "initial_fragments": [0, 1, 2, 3],
                "pushes": pushes,
                "broadcasts": broadcasts,
                "inner_steps_while_blocked": [],
            }
        barrier_events_path = report / "barrier-events.json"
        base.write_json_create_only(
            barrier_events_path,
            {"schema": "yeto_parallel_barrier_events_v1", "learners": learners},
        )
        result_evidence_path = report / "results.json"
        base.write_json_create_only(
            result_evidence_path,
            {
                "schema": "audit_135m_checkpoint_only_result_v1",
                "arm": arm_name,
                "runner_exit_code": 0,
                "syncer_exit_code": result["syncer_exit_code"],
                "learner_exit_codes": result["learner_exit_codes"],
                "loss": None,
                "evaluation_role": "none",
            },
        )
        inventory_path = _checkpoint_registry(self, request, attempt_dir)
        kernel_capture = _finite_kernel_capture(
            self, cell=cell, attempt_dir=attempt_dir
        )
        inventory = self._common_inventory(attempt_dir)
        for role, path in {
            "learner_steps": learner_steps_path,
            "work_events": work_events_path,
            "barrier_events": barrier_events_path,
            "results": result_evidence_path,
            "raw_tape": tape_path,
            "barrier_registry": attempt_dir
            / "report"
            / "barrier-version-trace.json",
            "checkpoint_inventory": inventory_path,
        }.items():
            inventory[role] = base.inventory_entry(self._relative(path), path)
        if kernel_capture is not None:
            inventory["finite_kernel_capture"] = base.inventory_entry(
                self._relative(kernel_capture), kernel_capture
            )
        _add_checkpoint_inventory_files(
            self,
            attempt_dir=attempt_dir,
            inventory_path=inventory_path,
            inventory=inventory,
        )
        return {
            "status": "COMPLETED",
            "failure_reason": None,
            "loss": None,
            "resumed": False,
            "resume_source": None,
            "scientific_started_at": started,
            "scientific_ended_at": ended,
            "artifact_inventory": inventory,
            "gpu_work_evidence": dict(maxima),
            "observed_work": observed_work,
        }
    _select_eval(self, cell)
    arm_name = self.phase.cell_arm_name(cell)
    learner_count = int(cell["target_work"]["learner_count"])
    expected_steps = int(cell["target_work"]["learner_steps_per_learner"])
    results_path, raw_loss, observed_work, work_evidence = (
        self.phase.validate_cell_work_evidence(
            self.phase_args, cell, attempt_dir, runner_exit_code=0
        )
    )
    tape_path = attempt_dir / "work" / arm_name / "tape.jsonl"
    tape = base.read_jsonl(tape_path)
    barrier = work_evidence["barrier"]
    if barrier["inner_step_counts"] != {
        learner: expected_steps for learner in range(learner_count)
    }:
        raise base.NeedsFix("development barrier proof lacks every learner step")
    if maxima["gpu_utilization_percent"] <= 0 or maxima["gpu_memory_mib"] <= 0:
        raise base.NeedsFix("nvidia-smi did not observe development GPU work")
    self.phase.validate_layout(attempt_dir, cell)
    _summary, raw_losses_path = self.phase.validate_eval(
        attempt_dir / "report", raw_loss, self.expected_eval
    )
    report = attempt_dir / "report" / "parallel-evidence"
    report.mkdir(parents=True, exist_ok=False)
    learner_steps_path = report / "learner-steps.json"
    base.write_json_create_only(
        learner_steps_path,
        {
            "schema": "yeto_parallel_learner_steps_v1",
            "learners": {
                str(learner): list(range(1, expected_steps + 1))
                for learner in range(learner_count)
            },
        },
    )
    updates = [
        {
            "outer_step": int(row["step"]),
            "fragment": int(row["fragment"]),
            "responders": [
                {
                    "learner_id": int(responder["id"]),
                    "base_version": int(responder["base_version"]),
                    "microsteps": int(responder["c_steps"]),
                    "tokens": int(responder["c_tokens"]),
                    "version_matched_anchor": bool(
                        responder["anchor_base_resolved"]
                    ),
                }
                for responder in row["responders"]
            ],
        }
        for row in tape
    ]
    work_events_path = report / "work-events.json"
    base.write_json_create_only(
        work_events_path,
        {"schema": "yeto_parallel_work_events_v1", "updates": updates},
    )
    learners = {}
    for learner in range(learner_count):
        pushes = []
        broadcasts = []
        for update in updates:
            responder = next(
                row
                for row in update["responders"]
                if row["learner_id"] == learner
            )
            pushes.append(
                {
                    "outer_step": update["outer_step"],
                    "fragment": update["fragment"],
                    "base_version": responder["base_version"],
                }
            )
            broadcasts.append(
                {
                    "outer_step": update["outer_step"],
                    "fragment": update["fragment"],
                    "pushed_base_version": responder["base_version"],
                    "broadcast_version": update["outer_step"],
                }
            )
        learners[str(learner)] = {
            "initial_fragments": [0, 1, 2, 3],
            "pushes": pushes,
            "broadcasts": broadcasts,
            "inner_steps_while_blocked": [],
        }
    barrier_events_path = report / "barrier-events.json"
    base.write_json_create_only(
        barrier_events_path,
        {"schema": "yeto_parallel_barrier_events_v1", "learners": learners},
    )
    result_row = base.read_jsonl(results_path)[0]
    result_evidence_path = report / "results.json"
    base.write_json_create_only(
        result_evidence_path,
        {
            "schema": "yeto_parallel_cell_result_v1",
            "arm": arm_name,
            "runner_exit_code": 0,
            "syncer_exit_code": result_row["syncer_exit_code"],
            "learner_exit_codes": result_row["learner_exit_codes"],
            "eval_loss": raw_loss,
        },
    )
    raw_losses = base.read_jsonl(raw_losses_path)
    positive_losses = [
        row for row in raw_losses if int(row.get("token_count", 0)) > 0
    ]
    eval_losses_path = report / "eval-losses.jsonl"
    base.write_jsonl(eval_losses_path, positive_losses)
    _finite_kernel_capture(self, cell=cell, attempt_dir=attempt_dir)
    inventory = self._common_inventory(attempt_dir)
    for role, path in {
        "learner_steps": learner_steps_path,
        "work_events": work_events_path,
        "barrier_events": barrier_events_path,
        "results": result_evidence_path,
        "eval_losses": eval_losses_path,
        "raw_tape": tape_path,
        "barrier_registry": attempt_dir
        / "report"
        / "barrier-version-trace.json",
    }.items():
        inventory[role] = base.inventory_entry(self._relative(path), path)
    inventory["eval_freeze"] = {
        "path": self.parallel_eval_entry["path"],
        "sha256": self.parallel_eval_entry["sha256"],
        "size_bytes": self.parallel_eval_path.stat().st_size,
    }
    return {
        "status": "COMPLETED",
        "failure_reason": None,
        "loss": raw_loss,
        "resumed": False,
        "resume_source": None,
        "scientific_started_at": started,
        "scientific_ended_at": ended,
        "artifact_inventory": inventory,
        "gpu_work_evidence": dict(maxima),
        "observed_work": observed_work,
    }


base.Worker.__init__ = audit_init
base.Worker.verify_static_state = audit_verify_static_state
base.Worker._run_command = audit_run_command
base.Worker._common_inventory = audit_common_inventory
base.Worker._completed_outcome = audit_completed_outcome
base.Worker._diverged_outcome = audit_diverged_outcome


if __name__ == "__main__":
    base.main()
