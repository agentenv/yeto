#!/usr/bin/env python3
"""Build the two-pass SAO/streaming-DiLoCo contracts for TB 2.1.

The real Miles layout probe has to discover the common actor/critic fragment
count before the syncer profile can be frozen.  This tool makes that boundary
explicit:

1. ``probe-context`` writes the deterministic, plan-bound SAO context used
   only by the provisional layout probe.
2. ``profile --total-steps 1`` writes a provisional, fully valid profile.
3. Run the layout probe once with that context/profile to discover ``P``.
4. ``prepare`` verifies that exact context-to-attestation binding, freezes the
   final profile (``total_steps=P``), and creates all
   benchmark-neutral SAO contexts, and emits the island-0 inputs for the final
   layout probe.
5. Run the layout probe again with the final profile and island-0 context.
6. ``finalize`` verifies that probe and creates one context-bound attestation
   plus one streaming runtime contract for every island.

No process is started and no existing output is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from yeto.rl.sao_streaming_runtime import (
    streaming_layout_attestation_schema,
    streaming_runtime_schema,
)
from yeto.syncer_profile import (
    SYNCER_SEMANTIC_PROFILE_SCHEMA,
    SyncerSemanticProfile,
)

PLAN_SCHEMA = "yeto.tbench21-sao-diloco-plan.v1"
BASE_CONTEXT_SCHEMA = "miles.sao-runtime.v2"
BUNDLE_SCHEMA = "yeto.tbench21-sao-streaming-contracts.v1"
BENCHMARK = "terminal-bench-2.1"
MODEL = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
ISLANDS = 8
ACTOR_SYNCER_PORT = 29400
CRITIC_SYNCER_PORT = 29401
MAX_FRAGMENT_BYTES = 2 << 30
MAX_CHUNK_BYTES = 256 << 20
PROVISIONAL_LAYOUT_HASH = hashlib.sha256(
    b"yeto.tbench21-sao-provisional-layout.v1\n"
).hexdigest()


class ContractError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"contract input is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"contract input is malformed: {path}") from error
    if not isinstance(value, dict):
        raise ContractError(f"contract input is not an object: {path}")
    return value


def _write_private(path: Path, value: object) -> str:
    if path.exists() or path.is_symlink():
        raise ContractError(f"refusing to overwrite contract output: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = _canonical(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ContractError(f"contract output is not private: {path}")
    return _sha256_bytes(encoded)


def _profile(total_steps: int, *, learners: int = ISLANDS) -> dict[str, object]:
    if total_steps < 1:
        raise ContractError("syncer total_steps must be positive")
    raw: dict[str, object] = {
        "schema": SYNCER_SEMANTIC_PROFILE_SCHEMA,
        "learners": learners,
        "quorum": learners,
        "grace_ms": 0,
        "grace_gamma": 0.8,
        "grace_tau": 2.0,
        "pipeline": 2,
        "min_round_interval_ms": 0,
        "sync_interval_steps": 1.0,
        "delta_correction": "none",
        "quorum_timeout_s": 7200,
        "final_ack_timeout_s": 7200,
        "total_steps": total_steps,
        "policy_sweep_fragments": None,
        "outer_lr": 0.7,
        "outer_momentum": 0.9,
        "checkpoint_enabled": True,
        "checkpoint_every": 1,
        "resume": True,
        "mark_final_checkpoint": True,
        "learner_budget_steps": None,
        "max_base_lag": 0,
        "learner_weight": "equal",
        "require_profile_binding": True,
    }
    SyncerSemanticProfile.from_mapping(raw).validate_sao()
    return raw


def write_profile(output: Path, *, total_steps: int) -> None:
    profile = _profile(total_steps)
    digest = _write_private(output.resolve(), profile)
    semantic = SyncerSemanticProfile.from_mapping(profile)
    print(
        json.dumps(
            {
                "path": str(output.resolve()),
                "file_sha256": digest,
                "semantic_sha256": semantic.sha256,
                "total_steps": total_steps,
            },
            sort_keys=True,
        )
    )


def _validate_plan(path: Path) -> tuple[dict[str, Any], str]:
    plan = _load_json(path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ContractError("wrong Terminal-Bench plan schema")
    topology = plan.get("topology")
    rollouts = plan.get("rollouts")
    compaction = plan.get("compaction")
    if (
        not isinstance(topology, dict)
        or topology.get("islands") != ISLANDS
        or topology.get("physical_gpus") != ISLANDS
        or topology.get("one_physical_gpu_per_island") is not True
        or topology.get("model") != MODEL
        or topology.get("model_revision") != MODEL_REVISION
        or not isinstance(rollouts, dict)
        or rollouts.get("train") != 176
        or rollouts.get("per_task") != 4
        or rollouts.get("episode_timeout_seconds") != 1800
        or not isinstance(compaction, dict)
        or compaction.get("enabled") is not True
        or compaction.get("trainer_objective") != "sao"
    ):
        raise ContractError("Terminal-Bench plan differs from the frozen run contract")
    return plan, _sha256_file(path)


def _attested_layout(path: Path) -> tuple[dict[str, Any], str, int]:
    payload = _load_json(path)
    if payload.get("schema") != streaming_layout_attestation_schema():
        raise ContractError("layout probe did not emit the benchmark-neutral v2 schema")
    settings = payload.get("settings")
    actor = payload.get("actor")
    critic = payload.get("critic")
    if not all(isinstance(value, dict) for value in (settings, actor, critic)):
        raise ContractError("layout probe is incomplete")
    actor_fragments = actor.get("expected_fragments")
    critic_fragments = critic.get("expected_fragments")
    if (
        type(actor_fragments) is not int
        or actor_fragments < 1
        or critic_fragments != actor_fragments
        or settings.get("minimum_fragments") != actor_fragments
        or settings.get("max_fragment_bytes") != MAX_FRAGMENT_BYTES
        or settings.get("max_chunk_bytes") != MAX_CHUNK_BYTES
        or settings.get("algorithm") != "sao"
        or settings.get("fragment_strategy") != "owner_affine"
        or settings.get("wire_dtype") != "fp32"
    ):
        raise ContractError(
            "actor/critic layout probe did not derive one exact fragment cut"
        )
    for name, role in (("actor", actor), ("critic", critic)):
        component = role.get("component")
        layout = role.get("parameter_layout_sha256")
        fragments = role.get("fragments")
        if (
            role.get("role") != name
            or not isinstance(component, dict)
            or set(component) != {"model_revision", "config_sha256"}
            or not isinstance(layout, str)
            or len(layout) != 64
            or not isinstance(fragments, list)
            or len(fragments) != actor_fragments
        ):
            raise ContractError(f"{name} layout evidence is malformed")
    return payload, _sha256_file(path), actor_fragments


def _plan_file(
    plan_path: Path, plan: dict[str, Any], relative: str
) -> tuple[Path, str]:
    files = plan.get("files")
    if not isinstance(files, dict) or not isinstance(files.get(relative), str):
        raise ContractError(f"plan does not bind {relative}")
    source = plan_path.parent / relative
    digest = _sha256_file(source)
    if digest != files[relative]:
        raise ContractError(f"plan file changed after freezing: {relative}")
    return source, digest


def _safe_container_dir(path: Path, name: str) -> Path:
    if not path.is_absolute() or ".." in path.parts or path == Path("/"):
        raise ContractError(f"{name} must be an absolute safe non-root path")
    return path


def _base_context(
    *,
    plan_sha256: str,
    data_sha256: str,
    reward_sha256: str,
    layout_hash: str,
    island_id: int,
    container_plan_dir: Path,
    container_run_dir: Path,
) -> dict[str, object]:
    container_plan_dir = _safe_container_dir(
        container_plan_dir, "container plan directory"
    )
    container_run_dir = _safe_container_dir(
        container_run_dir, "container run directory"
    )
    return {
        "schema": BASE_CONTEXT_SCHEMA,
        "benchmark": BENCHMARK,
        "model": MODEL,
        "data": str(container_plan_dir / f"train/island-{island_id}.jsonl"),
        "data_sha256": data_sha256,
        "base_model_revision": MODEL_REVISION,
        "rollout_model_revision": MODEL_REVISION,
        "data_revision": plan_sha256,
        "layout_hash": layout_hash,
        "lora_config_hash": "full-parameter:qwen35-0.8b:tp1",
        "reward_sha256": reward_sha256,
        "dynamic_sampling_max_replacements": 0,
        "completed_groups_path": str(container_run_dir / "completed-groups.pt"),
        "event_tape": str(container_run_dir / "events.jsonl"),
        "learner_id": island_id,
    }


def write_probe_context(
    *,
    plan_path: Path,
    reward_source: Path,
    output: Path,
    container_plan_dir: Path,
    container_run_dir: Path,
) -> str:
    """Write the real, deterministic v2 context for the first layout probe."""

    plan_path = plan_path.resolve()
    plan, plan_sha256 = _validate_plan(plan_path)
    reward_source = reward_source.resolve()
    if not reward_source.is_file() or reward_source.is_symlink():
        raise ContractError("reward source is missing")
    _, data_sha256 = _plan_file(plan_path, plan, "train/island-0.jsonl")
    reward_sha256 = _sha256_file(reward_source)
    context = _base_context(
        plan_sha256=plan_sha256,
        data_sha256=data_sha256,
        reward_sha256=reward_sha256,
        layout_hash=PROVISIONAL_LAYOUT_HASH,
        island_id=0,
        container_plan_dir=container_plan_dir,
        container_run_dir=container_run_dir,
    )
    output = output.resolve()
    digest = _write_private(output, context)
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": digest,
                "plan_sha256": plan_sha256,
                "reward_sha256": reward_sha256,
                "provisional_layout_hash": PROVISIONAL_LAYOUT_HASH,
            },
            sort_keys=True,
        )
    )
    return digest


def _validate_probe_context(
    path: Path,
    *,
    plan_path: Path,
    plan: dict[str, Any],
    plan_sha256: str,
    reward_sha256: str,
) -> str:
    path = path.expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ContractError(
            "provisional SAO context must be an absolute regular non-symlink"
        )
    path = path.resolve(strict=True)
    payload = _load_json(path)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ContractError("provisional SAO context must have mode 0600")
    try:
        data_path = Path(payload["data"])
        completed_path = Path(payload["completed_groups_path"])
        event_path = Path(payload["event_tape"])
    except (KeyError, TypeError) as error:
        raise ContractError("provisional SAO context paths are malformed") from error
    if (
        data_path.name != "island-0.jsonl"
        or data_path.parent.name != "train"
        or completed_path.name != "completed-groups.pt"
        or event_path.name != "events.jsonl"
        or completed_path.parent != event_path.parent
    ):
        raise ContractError("provisional SAO context paths changed")
    container_plan_dir = _safe_container_dir(
        data_path.parent.parent, "provisional container plan directory"
    )
    container_run_dir = _safe_container_dir(
        completed_path.parent, "provisional container run directory"
    )
    _, data_sha256 = _plan_file(plan_path, plan, "train/island-0.jsonl")
    expected = _base_context(
        plan_sha256=plan_sha256,
        data_sha256=data_sha256,
        reward_sha256=reward_sha256,
        layout_hash=PROVISIONAL_LAYOUT_HASH,
        island_id=0,
        container_plan_dir=container_plan_dir,
        container_run_dir=container_run_dir,
    )
    if payload != expected:
        raise ContractError(
            "provisional SAO context differs from its plan/reward binding"
        )
    return _sha256_file(path)


def prepare(
    *,
    plan_path: Path,
    provisional_context_path: Path,
    provisional_attestation_path: Path,
    reward_source: Path,
    critic_contract: Path,
    output_dir: Path,
    container_plan_dir: Path,
    container_contract_dir: Path,
    container_run_root: Path,
) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise ContractError("prepare output directory must be fresh")
    for value, name in (
        (container_plan_dir, "container plan directory"),
        (container_contract_dir, "container contract directory"),
        (container_run_root, "container run root"),
    ):
        _safe_container_dir(value, name)

    plan_path = plan_path.resolve()
    plan, plan_sha256 = _validate_plan(plan_path)
    reward_source = reward_source.resolve()
    critic_contract = critic_contract.resolve()
    if not reward_source.is_file() or reward_source.is_symlink():
        raise ContractError("reward source is missing")
    if not critic_contract.is_file() or critic_contract.is_symlink():
        raise ContractError("critic value-pretraining contract is missing")
    reward_sha256 = _sha256_file(reward_source)
    provisional_context_sha256 = _validate_probe_context(
        provisional_context_path,
        plan_path=plan_path,
        plan=plan,
        plan_sha256=plan_sha256,
        reward_sha256=reward_sha256,
    )
    provisional, provisional_sha256, fragments = _attested_layout(
        provisional_attestation_path.resolve()
    )
    if provisional.get("sao_context_sha256") != provisional_context_sha256:
        raise ContractError(
            "provisional layout attestation is not bound to the probe context"
        )
    expected_provisional_profile_sha256 = SyncerSemanticProfile.from_mapping(
        _profile(1)
    ).sha256
    if (
        provisional["settings"].get("syncer_profile_sha256")
        != expected_provisional_profile_sha256
    ):
        raise ContractError(
            "provisional layout attestation is not bound to the one-step profile"
        )
    critic_contract_sha256 = _sha256_file(critic_contract)
    profile = _profile(fragments)
    semantic_profile = SyncerSemanticProfile.from_mapping(profile)
    training_contract = {
        "schema": "yeto.tbench21-sao-training-contract.v1",
        "plan_sha256": plan_sha256,
        "critic_value_pretraining_contract_sha256": critic_contract_sha256,
        "reward_source_sha256": reward_sha256,
        "actor_component": provisional["actor"]["component"],
        "critic_component": provisional["critic"]["component"],
        "actor_layout_sha256": provisional["actor"]["parameter_layout_sha256"],
        "critic_layout_sha256": provisional["critic"]["parameter_layout_sha256"],
        "expected_fragments": fragments,
        "syncer_profile_sha256": semantic_profile.sha256,
        "optimizer_steps_per_round": {"actor": 1, "critic": 2},
    }
    training_contract_sha256 = _sha256_bytes(_canonical(training_contract))

    output_dir.mkdir(mode=0o700, parents=True)
    profile_path = output_dir / "syncer-profile.json"
    profile_file_sha256 = _write_private(profile_path, profile)
    training_path = output_dir / "training-contract.json"
    _write_private(training_path, training_contract)
    contexts: list[dict[str, object]] = []
    for island_id in range(ISLANDS):
        relative = f"train/island-{island_id}.jsonl"
        _, data_sha256 = _plan_file(plan_path.resolve(), plan, relative)
        island_dir = output_dir / f"island-{island_id}"
        context_path = island_dir / "sao-context.json"
        context = _base_context(
            plan_sha256=plan_sha256,
            data_sha256=data_sha256,
            reward_sha256=reward_sha256,
            layout_hash=provisional["actor"]["parameter_layout_sha256"],
            island_id=island_id,
            container_plan_dir=container_plan_dir,
            container_run_dir=container_run_root / f"island-{island_id}",
        )
        context_sha256 = _write_private(context_path, context)
        contexts.append(
            {
                "island_id": island_id,
                "host_path": str(context_path),
                "container_path": str(
                    container_contract_dir / f"island-{island_id}" / "sao-context.json"
                ),
                "sha256": context_sha256,
            }
        )

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "phase": "prepared",
        "plan": {"path": str(plan_path.resolve()), "sha256": plan_sha256},
        "provisional_attestation": {
            "path": str(provisional_attestation_path.resolve()),
            "sha256": provisional_sha256,
        },
        "provisional_context": {
            "path": str(provisional_context_path.resolve()),
            "sha256": provisional_context_sha256,
        },
        "container_paths": {
            "plan_dir": str(container_plan_dir),
            "contract_dir": str(container_contract_dir),
            "run_root": str(container_run_root),
        },
        "expected_fragments": fragments,
        "syncer_profile": {
            "host_path": str(profile_path),
            "container_path": str(container_contract_dir / "syncer-profile.json"),
            "file_sha256": profile_file_sha256,
            "semantic_sha256": semantic_profile.sha256,
            "profile": profile,
        },
        "training_contract": {
            "host_path": str(training_path),
            "sha256": training_contract_sha256,
            "contract": training_contract,
        },
        "reward_source_sha256": reward_sha256,
        "critic_value_pretraining_contract_sha256": critic_contract_sha256,
        "contexts": contexts,
    }
    manifest_path = output_dir / "prepare-manifest.json"
    _write_private(manifest_path, manifest)
    print(
        json.dumps(
            {
                "prepare_manifest": str(manifest_path),
                "expected_fragments": fragments,
                "final_profile_path": str(profile_path),
                "final_profile_sha256": semantic_profile.sha256,
                "final_probe_sao_context": contexts[0],
            },
            sort_keys=True,
        )
    )


def finalize(
    *,
    prepare_manifest_path: Path,
    final_attestation_path: Path,
    syncer_host: str,
) -> None:
    prepared = _load_json(prepare_manifest_path.resolve())
    if prepared.get("schema") != BUNDLE_SCHEMA or prepared.get("phase") != "prepared":
        raise ContractError("prepare manifest has the wrong schema or phase")
    provisional_context = prepared.get("provisional_context")
    if not isinstance(provisional_context, dict):
        raise ContractError("prepare manifest has no provisional context binding")
    provisional_context_path = Path(provisional_context.get("path", ""))
    provisional_context_sha256 = provisional_context.get("sha256")
    if (
        not provisional_context_path.is_absolute()
        or provisional_context_path.is_symlink()
        or not provisional_context_path.is_file()
        or not isinstance(provisional_context_sha256, str)
        or len(provisional_context_sha256) != 64
        or _sha256_file(provisional_context_path) != provisional_context_sha256
    ):
        raise ContractError("provisional SAO context changed after prepare")
    if (
        not syncer_host
        or len(syncer_host) > 253
        or any(c.isspace() for c in syncer_host)
    ):
        raise ContractError("syncer host is invalid")
    output_dir = prepare_manifest_path.resolve().parent
    final_template, final_template_sha256, fragments = _attested_layout(
        final_attestation_path.resolve()
    )
    if fragments != prepared.get("expected_fragments"):
        raise ContractError("final layout probe changed the fragment count")
    profile_record = prepared.get("syncer_profile")
    if not isinstance(profile_record, dict):
        raise ContractError("prepare manifest has no syncer profile")
    profile = profile_record.get("profile")
    semantic_profile = SyncerSemanticProfile.from_mapping(profile)
    semantic_profile.validate_sao()
    if (
        semantic_profile.total_steps != fragments
        or semantic_profile.sha256 != profile_record.get("semantic_sha256")
        or final_template["settings"].get("syncer_profile_sha256")
        != semantic_profile.sha256
    ):
        raise ContractError(
            "final layout probe is not bound to the frozen syncer profile"
        )
    contexts = prepared.get("contexts")
    container_paths = prepared.get("container_paths")
    training = prepared.get("training_contract")
    if (
        not isinstance(contexts, list)
        or len(contexts) != ISLANDS
        or not isinstance(container_paths, dict)
        or not isinstance(training, dict)
        or not isinstance(training.get("sha256"), str)
    ):
        raise ContractError("prepare manifest is incomplete")
    first_context_sha = contexts[0].get("sha256")
    if final_template.get("sao_context_sha256") != first_context_sha:
        raise ContractError("final layout probe is not bound to island 0")
    contract_dir = Path(container_paths["contract_dir"])
    run_root = Path(container_paths["run_root"])
    role_records = {name: final_template[name] for name in ("actor", "critic")}
    generated: list[dict[str, object]] = []
    for island_id, context_record in enumerate(contexts):
        if context_record.get("island_id") != island_id:
            raise ContractError("prepare manifest island order changed")
        context_sha = context_record.get("sha256")
        if not isinstance(context_sha, str) or len(context_sha) != 64:
            raise ContractError("prepare manifest has an invalid context hash")
        island_dir = output_dir / f"island-{island_id}"
        attestation = dict(final_template)
        attestation["sao_context_sha256"] = context_sha
        attestation_path = island_dir / "layout-attestation.json"
        attestation_sha256 = _write_private(attestation_path, attestation)

        def role_payload(
            role: str,
            port: int,
            optimizer_steps: int,
            *,
            learner_id: int = island_id,
        ) -> dict[str, object]:
            record = role_records[role]
            return {
                "role": role,
                "component": record["component"],
                "syncer": {"host": syncer_host, "port": port},
                "learner_id": learner_id,
                "learner_generation": 0,
                "learner_generations": [0] * ISLANDS,
                "total_fragment_steps": fragments,
                "expected_fragments": fragments,
                "parameter_layout_sha256": record["parameter_layout_sha256"],
                "local_horizon": 1,
                "optimizer_steps_per_round": optimizer_steps,
                "training_contract_sha256": training["sha256"],
                "wan_streams": 4,
                "wait_timeout_seconds": 7200.0,
                "poll_seconds": 0.01,
                "max_fragment_bytes": MAX_FRAGMENT_BYTES,
                "max_chunk_bytes": MAX_CHUNK_BYTES,
            }

        streaming = {
            "schema": streaming_runtime_schema(),
            "sao_context_sha256": context_sha,
            "trajectory_evidence": {
                "directory": str(
                    run_root / f"island-{island_id}" / "trajectory-evidence"
                ),
                "kind": BENCHMARK,
                "schema_version": 2,
            },
            "layout_attestation": {
                "path": str(
                    contract_dir / f"island-{island_id}" / "layout-attestation.json"
                ),
                "sha256": attestation_sha256,
            },
            "syncer_profile": profile,
            "actor": role_payload("actor", ACTOR_SYNCER_PORT, 1),
            "critic": role_payload("critic", CRITIC_SYNCER_PORT, 2),
        }
        streaming_path = island_dir / "sao-streaming-context.json"
        streaming_sha256 = _write_private(streaming_path, streaming)
        generated.append(
            {
                "island_id": island_id,
                "sao_context": context_record,
                "layout_attestation": {
                    "host_path": str(attestation_path),
                    "container_path": str(
                        contract_dir / f"island-{island_id}" / "layout-attestation.json"
                    ),
                    "sha256": attestation_sha256,
                },
                "streaming_context": {
                    "host_path": str(streaming_path),
                    "container_path": str(
                        contract_dir
                        / f"island-{island_id}"
                        / "sao-streaming-context.json"
                    ),
                    "sha256": streaming_sha256,
                },
            }
        )

    manifest = {
        **prepared,
        "phase": "final",
        "final_attestation_template": {
            "path": str(final_attestation_path.resolve()),
            "sha256": final_template_sha256,
        },
        "islands": generated,
    }
    final_manifest = output_dir / "manifest.json"
    _write_private(final_manifest, manifest)
    print(
        json.dumps(
            {
                "manifest": str(final_manifest),
                "islands": ISLANDS,
                "expected_fragments": fragments,
                "syncer_profile_sha256": semantic_profile.sha256,
            },
            sort_keys=True,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    probe_context = commands.add_parser("probe-context")
    probe_context.add_argument("--plan-manifest", type=Path, required=True)
    probe_context.add_argument("--reward-source", type=Path, required=True)
    probe_context.add_argument("--output", type=Path, required=True)
    probe_context.add_argument("--container-plan-dir", type=Path, required=True)
    probe_context.add_argument("--container-run-dir", type=Path, required=True)

    profile = commands.add_parser("profile")
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--total-steps", type=int, required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--plan-manifest", type=Path, required=True)
    prepare_parser.add_argument("--provisional-context", type=Path, required=True)
    prepare_parser.add_argument("--provisional-attestation", type=Path, required=True)
    prepare_parser.add_argument("--reward-source", type=Path, required=True)
    prepare_parser.add_argument("--critic-contract", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--container-plan-dir", type=Path, required=True)
    prepare_parser.add_argument("--container-contract-dir", type=Path, required=True)
    prepare_parser.add_argument("--container-run-root", type=Path, required=True)

    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--prepare-manifest", type=Path, required=True)
    finalize_parser.add_argument("--final-attestation", type=Path, required=True)
    finalize_parser.add_argument("--syncer-host", default="host.docker.internal")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "probe-context":
        write_probe_context(
            plan_path=args.plan_manifest,
            reward_source=args.reward_source,
            output=args.output,
            container_plan_dir=args.container_plan_dir,
            container_run_dir=args.container_run_dir,
        )
    elif args.command == "profile":
        write_profile(args.output, total_steps=args.total_steps)
    elif args.command == "prepare":
        prepare(
            plan_path=args.plan_manifest,
            provisional_context_path=args.provisional_context,
            provisional_attestation_path=args.provisional_attestation,
            reward_source=args.reward_source,
            critic_contract=args.critic_contract,
            output_dir=args.output_dir,
            container_plan_dir=args.container_plan_dir,
            container_contract_dir=args.container_contract_dir,
            container_run_root=args.container_run_root,
        )
    elif args.command == "finalize":
        finalize(
            prepare_manifest_path=args.prepare_manifest,
            final_attestation_path=args.final_attestation,
            syncer_host=args.syncer_host,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
