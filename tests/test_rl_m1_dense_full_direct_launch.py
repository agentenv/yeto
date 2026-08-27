from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "probes"
    / "m1_dense_full_direct_launch.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "m1_dense_full_direct_launch", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
launch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(launch)

_REPORT_PATH = _MODULE_PATH.with_name("m1_dense_full_final_report.py")
_REPORT_SPEC = importlib.util.spec_from_file_location(
    "m1_dense_full_final_report", _REPORT_PATH
)
assert _REPORT_SPEC is not None and _REPORT_SPEC.loader is not None
reporter = importlib.util.module_from_spec(_REPORT_SPEC)
_REPORT_SPEC.loader.exec_module(reporter)


def _sha(character: str) -> str:
    return character * 64


def _topology(rank: int) -> dict[str, int]:
    return {
        "tp_rank": rank,
        "tp_size": 2,
        "pp_rank": 0,
        "pp_size": 1,
        "ep_rank": 0,
        "ep_size": 1,
        "cp_rank": 0,
        "cp_size": 1,
        "dp_rank": 0,
        "dp_size": 1,
    }


def _probe() -> dict:
    fragments = [
        {
            "fragment_id": index,
            "role": "actor",
            "shard_id": f"owner-{index // 2}",
            "plan_hash": _sha(str(index // 2 + 1)),
            "tensor_count": 1,
            "numel": 10 + index,
            "fp32_bytes": (10 + index) * 4,
        }
        for index in range(4)
    ]
    owners = []
    for rank in range(2):
        rows = fragments[rank * 2 : rank * 2 + 2]
        owners.append(
            {
                "role": "actor",
                "shard_id": f"owner-{rank}",
                "topology": _topology(rank),
                "manifest_layout_hash": _sha(str(rank + 3)),
                "plan_hash": _sha(str(rank + 1)),
                "parameter_tensor_count": 2,
                "parameter_scalar_count": sum(row["numel"] for row in rows),
                "fragment_ids": [row["fragment_id"] for row in rows],
                "fragment_count": 2,
                "max_fragment_bytes": max(row["fp32_bytes"] for row in rows),
            }
        )
    return {
        "schema": launch.PROBE_SCHEMA,
        "observed_utc": "2026-08-24T00:00:00Z",
        "algorithm": "grpo",
        "probe_mode": "full_parameter_manifest_only",
        "role": "actor",
        "fragment_strategy": "owner_affine",
        "model_repo": launch.MODEL_REPO,
        "model_revision": launch.MODEL_REVISION,
        "model_config_sha256": _sha("b"),
        "model_path": "/models/hf",
        "checkpoint_path": "/models/torch_dist",
        "conversion_manifest": {
            "path": "/models/torch_dist/conversion-manifest.json",
            "sha256": _sha("c"),
            "schema": "yeto-qwen35-megatron-conversion-v1",
            "model_file_count": 14,
            "checkpoint_file_count": 8,
            "conversion_source_aggregate_sha256": _sha("d"),
        },
        "miles_image_digest": f"sha256:{_sha('e')}",
        "yeto_source": {"path": "/root/yeto", "source_tree_sha256": _sha("f")},
        "miles_source": {
            "path": "/root/miles",
            "execution_source_sha256": _sha("0"),
        },
        "hardware": {"gpu_count": 2},
        "megatron_bridge": {
            "distribution_name": "megatron-bridge",
            "distribution_version": "0.1.0",
            "direct_url": {
                "url": "https://example.invalid/bridge.git",
                "vcs_info": {"vcs": "git", "commit_id": "1" * 40},
            },
            "direct_url_commit": "1" * 40,
        },
        "actor_topology": {
            "actor_num_nodes": 1,
            "actor_num_gpus_per_node": 2,
            "tp_size": 2,
            "pp_size": 1,
            "ep_size": 1,
            "cp_size": 1,
            "dp_size": 1,
        },
        "sequence_length": launch.PRODUCTION_SEQUENCE_LENGTH,
        "parameter_layout_hash": _sha("2"),
        "owner_count": 2,
        "minimum_fragment_count": 2,
        "derived_fragment_count": 4,
        "parameter_tensor_count": 4,
        "parameter_scalar_count": sum(row["numel"] for row in fragments),
        "max_fragment_bytes_limit": launch.MAX_FRAGMENT_BYTES,
        "max_chunk_bytes": 256 << 20,
        "observed_max_fragment_bytes": max(row["fp32_bytes"] for row in fragments),
        "owner_plans": owners,
        "fragments": fragments,
    }


def test_probe_verifier_binds_exact_runtime_sources_bridge_and_derived_p():
    probe = _probe()

    assert (
        launch._validate_probe(
            probe,
            image_digest=probe["miles_image_digest"],
            yeto_source_sha256=_sha("f"),
            miles_source_sha256=_sha("0"),
        )["derived_fragment_count"]
        == 4
    )

    wrong_source = copy.deepcopy(probe)
    wrong_source["yeto_source"] = {
        "path": "/root/yeto",
        "execution_source_sha256": _sha("f"),
    }
    with pytest.raises(launch.LaunchContractError, match="keys differ"):
        launch._validate_probe(
            wrong_source,
            image_digest=probe["miles_image_digest"],
            yeto_source_sha256=_sha("f"),
            miles_source_sha256=_sha("0"),
        )

    oversized = copy.deepcopy(probe)
    oversized["fragments"][3]["numel"] = launch.MAX_FRAGMENT_BYTES // 4 + 1
    oversized["fragments"][3]["fp32_bytes"] = launch.MAX_FRAGMENT_BYTES + 4
    with pytest.raises(launch.LaunchContractError, match="FP32 bound"):
        launch._validate_probe(
            oversized,
            image_digest=probe["miles_image_digest"],
            yeto_source_sha256=_sha("f"),
            miles_source_sha256=_sha("0"),
        )


def test_launch_bundle_attestation_rejects_tool_drift(tmp_path):
    source_root = Path(__file__).resolve().parents[1]
    for relative in launch.LAUNCH_BUNDLE_FILES:
        source = source_root / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    expected = launch._launch_bundle(tmp_path)

    assert launch._attest_launch_bundle(expected, tmp_path) == expected
    changed = tmp_path / launch.LAUNCH_BUNDLE_FILES[1]
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(launch.LaunchContractError, match="changed after manifest"):
        launch._attest_launch_bundle(expected, tmp_path)


def _ports(offset: int) -> dict[str, int]:
    return {
        "ray_gcs": offset,
        "ray_dashboard": offset + 1,
        "ray_client": offset + 2,
        "rollout_engine_base": offset + 3,
        "sglang_router": offset + 4,
        "sglang_router_prometheus": offset + 5,
        "train_master_base": offset + 6,
        "session_server": offset + 7,
    }


def _manifest() -> dict:
    islands = []
    for island_id in range(2):
        uuids = [f"GPU-{island_id}{index:032d}" for index in range(4)]
        islands.append(
            {
                "island_id": island_id,
                "container_name": f"m1-island-{island_id}",
                "host_gpu_indices": list(range(island_id * 4, island_id * 4 + 4)),
                "gpu_uuids": uuids,
                "trainer_gpu_uuids": uuids[:2],
                "inference_gpu_uuids": uuids[2:],
                "ports": _ports(20000 + island_id * 100),
                "host_run_dir": f"/runs/m1/island-{island_id}",
                "container_run_dir": "/evidence",
            }
        )
    manifest = {
        "schema": launch.MANIFEST_SCHEMA,
        "launch_mode": "two-island-final",
        "run_id": "m1-test",
        "rounds": 3,
        "image": {
            "repository": launch.MILES_IMAGE_REPOSITORY,
            "digest": launch.MILES_IMAGE_DIGEST,
            "reference": (
                f"{launch.MILES_IMAGE_REPOSITORY}@{launch.MILES_IMAGE_DIGEST}"
            ),
        },
        "launch_bundle": launch._launch_bundle(Path(__file__).resolve().parents[1]),
        "provenance": {
            "probe_evidence_path": "/inputs/probe.json",
            "probe_evidence_sha256": _sha("b"),
            "yeto_root": "/src/yeto",
            "yeto_source_sha256": _sha("c"),
            "miles_root": "/src/miles",
            "miles_source_sha256": _sha("d"),
            "megatron_bridge": _probe()["megatron_bridge"],
        },
        "model": {
            "repo": launch.MODEL_REPO,
            "revision": launch.MODEL_REVISION,
            "host_model_path": "/models/hf",
            "container_model_path": "/models/hf",
            "container_hf_cache_snapshot_path": launch.MODEL_CACHE_SNAPSHOT,
            "config_sha256": _sha("f"),
            "host_checkpoint_path": "/models/torch_dist",
            "container_checkpoint_path": "/models/torch_dist",
            "conversion_manifest_sha256": _sha("0"),
            "conversion_manifest": {"schema": "test-fixture"},
            "parameter_layout_hash": _sha("1"),
            "fragment_count": 4,
            "observed_max_fragment_bytes": 1024,
        },
        "data": {
            "host_train_path": "/inputs/train.jsonl",
            "container_train_path": "/workspace/data/train.jsonl",
            "train_sha256": launch.TRAIN_DATA_SHA256,
            "host_heldout_path": "/inputs/heldout.jsonl",
            "container_heldout_path": "/workspace/data/heldout.jsonl",
            "heldout_sha256": launch.HELDOUT_DATA_SHA256,
            "manifest_path": "/inputs/manifest.json",
            "manifest_sha256": launch.DATA_MANIFEST_SHA256,
            "docker_image_inventory_path": "/inputs/images.json",
            "docker_image_inventory_sha256": launch.DOCKER_IMAGE_INVENTORY_SHA256,
            "docker_image_inventory": {"images": []},
        },
        "profile": {
            "seq_len": 4096,
            "rollout_max_response_len": 256,
            "groups_per_round": 1,
            "samples_per_group": 3,
            "over_sampling_batch_size": 2,
            "inner_lr": 1e-6,
            "seed": 7,
            "sglang_mem_fraction_static": 0.4,
            "reward_function": "yeto_miles_secrlenv.reward:reward_func",
            "reward_source_path": "/src/yeto/yeto_miles_secrlenv/reward.py",
            "reward_sha256": _sha("6"),
            "custom_generate_function_path": "yeto_miles_secrlenv.generate.generate",
            "custom_agent_function_path": "yeto_miles_secrlenv.codex_harness_agent.run",
            "dynamic_sampling_filter_path": "yeto_miles_secrlenv.reward.check_group",
            "dynamic_sampling_max_replacements": 0,
            "secrlenv_max_infrastructure_replacements": 1,
            "tito_model": "qwen35",
        },
        "harness": {
            "contract": {
                "agent_function_path": launch.CUSTOM_AGENT_FUNCTION,
                "agent_source_sha256": _sha("0"),
                "controller_binary_path": "/inputs/codex",
                "controller_package_manifest_path": "/inputs/package.json",
                "controller_app_server_schema_path": "/inputs/schema.json",
                "bundle_binary_path": "/bundle/codex",
                "bundle_package_manifest_path": "/bundle/package.json",
                "bundle_app_server_schema_path": "/bundle/schema.json",
                "container_binary_path": "/opt/yeto/codex/codex",
                "container_app_server_schema_path": "/opt/yeto/codex/schema.json",
                "binary_sha256": _sha("a"),
                "binary_size_bytes": 123,
                "cli_version": "codex-cli 1.0",
                "npm_package": "@openai/codex",
                "target": "x86_64-unknown-linux-musl",
                "npm_tarball_sha256": _sha("1"),
                "package_manifest_sha256": _sha("2"),
                "app_server_protocol_revision": "v2",
                "app_server_schema_sha256": _sha("b"),
                "base_instructions_sha256": _sha("c"),
                "terminal_exec_tool_schema_sha256": _sha("d"),
                "submit_tool_schema_sha256": _sha("e"),
                "dynamic_tools_schema_sha256": _sha("f"),
                "reasoning_effort": "xhigh",
                "backend": {
                    "model": "qwen35",
                    "max_tokens": 256,
                    "reasoning_effort": "xhigh",
                    "thinking": {"type": "enabled"},
                    "chat_template": "qwen35",
                    "chat_template_kwargs": {"clear_thinking": False},
                    "tito_allowed_append_roles": ["tool", "user"],
                },
            },
            "contract_path": "/inputs/codex.json",
            "contract_sha256": _sha("7"),
            "artifacts": [
                {
                    "host_path": "/inputs/codex",
                    "container_path": "/opt/yeto/codex/codex",
                    "sha256": _sha("a"),
                    "size_bytes": 123,
                    "executable": True,
                },
                {
                    "host_path": "/inputs/package.json",
                    "container_path": "/opt/yeto/codex/codex-package.json",
                    "sha256": _sha("2"),
                    "size_bytes": 1,
                    "executable": False,
                },
                {
                    "host_path": "/inputs/schema.json",
                    "container_path": "/opt/yeto/codex/schema.json",
                    "sha256": _sha("b"),
                    "size_bytes": 1,
                    "executable": False,
                },
            ],
            "environment": {},
        },
        "secrlenv": {
            "host_health_url": "http://127.0.0.1:28869/healthz",
            "container_url": "http://127.0.0.1:28869",
            "task_pack_sha256": launch.TASK_PACK_SHA256,
            "bearer_token_host_path": "/inputs/token",
            "bearer_token_sha256": _sha("9"),
            "daemon_contract_path": "/inputs/daemon.json",
            "daemon_contract_sha256": _sha("a"),
            "daemon_contract": {"port": 28869},
        },
        "topology": {
            "host_gpu_count": 8,
            "island_count": 2,
            "gpus_per_island": 4,
            "trainer_tp": 2,
            "inference_engines": 2,
            "inference_tp": 1,
            "cross_island_collective": False,
            "islands": islands,
        },
        "syncer": {
            "binary_path": "/inputs/yeto-syncer",
            "binary_sha256": launch.SYNCER_BINARY_SHA256,
            "build_manifest_path": "/inputs/syncer-build.json",
            "build_manifest_sha256": launch.SYNCER_BUILD_MANIFEST_SHA256,
            "build_manifest": {},
            "platform": "linux-x86_64",
            "port": 29400,
            "total_steps": 12,
            "policy_sweep_fragments": 4,
        },
        "launch": {"host_run_root": "/runs/m1", "uses_ssh_harness": False},
        "learners": [],
    }
    manifest["syncer"]["argv"] = launch._syncer_argv(
        manifest["syncer"]["binary_path"],
        port=29400,
        learners=2,
        total_steps=12,
        fragments=4,
        run="/runs/m1",
    )
    manifest["harness"]["environment"] = launch._harness_environment(
        manifest["harness"]["contract"]
    )
    manifest["evaluation"] = launch._evaluation_contract(
        launch_mode=manifest["launch_mode"],
        rounds=manifest["rounds"],
        profile=manifest["profile"],
        data=manifest["data"],
        run_root=Path(manifest["launch"]["host_run_root"]),
        yeto_root=Path(manifest["provenance"]["yeto_root"]),
        island_count=manifest["topology"]["island_count"],
    )
    manifest["learners"] = [
        {
            "island_id": island_id,
            "argv": launch._learner_argv(
                manifest=manifest, island_id=island_id, run_dir="/evidence"
            ),
        }
        for island_id in range(2)
    ]
    return manifest


def test_manifest_closes_direct_two_island_topology_ports_and_exact_argv():
    manifest = _manifest()

    assert launch._validate_manifest(manifest) is manifest
    assert manifest["syncer"]["total_steps"] == 12
    assert (
        manifest["topology"]["islands"][0]["trainer_gpu_uuids"]
        == manifest["topology"]["islands"][0]["gpu_uuids"][:2]
    )
    assert "--policy-sweep-fragments" in manifest["syncer"]["argv"]
    assert "cargo" not in manifest["syncer"]["argv"]
    container_argv = launch._container_command(
        manifest, Path("/runs/m1/manifest.json"), _sha("f"), 0
    )
    assert container_argv[container_argv.index("--gpus") + 1].startswith('"device=GPU-')
    assert "HF_HUB_OFFLINE=1" in container_argv
    assert "MILES_CI_GATE_RECORD_DIR=/evidence/miles-metrics" in container_argv
    assert "/inputs/heldout.jsonl:/workspace/data/heldout.jsonl:ro" in container_argv
    assert container_argv[container_argv.index("--network") + 1] == "host"
    assert "host.docker.internal:host-gateway" not in container_argv
    assert launch._flag_value(manifest["learners"][0]["argv"], "--syncer") == (
        "127.0.0.1:29400"
    )
    assert f"/models/hf:{launch.MODEL_CACHE_SNAPSHOT}:ro" in container_argv

    overlapping = copy.deepcopy(manifest)
    overlapping["topology"]["islands"][1]["ports"]["session_server"] = manifest[
        "topology"
    ]["islands"][0]["ports"]["session_server"]
    overlapping["learners"][1]["argv"] = launch._learner_argv(
        manifest=overlapping, island_id=1, run_dir="/evidence"
    )
    with pytest.raises(launch.LaunchContractError, match="ports overlap"):
        launch._validate_manifest(overlapping)

    changed_argv = copy.deepcopy(manifest)
    changed_argv["learners"][0]["argv"].append("--colocate")
    with pytest.raises(launch.LaunchContractError, match="direct argv differs"):
        launch._validate_manifest(changed_argv)

    unsafe_environment = copy.deepcopy(manifest)
    unsafe_environment["harness"]["environment"]["HF_HUB_OFFLINE"] = "0"
    with pytest.raises(launch.LaunchContractError, match="environment differs"):
        launch._validate_manifest(unsafe_environment)

    single = copy.deepcopy(manifest)
    single["launch_mode"] = "single-island-gate"
    single["rounds"] = 1
    single["topology"]["host_gpu_count"] = 4
    single["topology"]["island_count"] = 1
    single["topology"]["islands"] = single["topology"]["islands"][:1]
    single["learners"] = single["learners"][:1]
    single["syncer"]["total_steps"] = 4
    single["evaluation"] = launch._evaluation_contract(
        launch_mode=single["launch_mode"],
        rounds=single["rounds"],
        profile=single["profile"],
        data=single["data"],
        run_root=Path(single["launch"]["host_run_root"]),
        yeto_root=Path(single["provenance"]["yeto_root"]),
        island_count=single["topology"]["island_count"],
    )
    single["syncer"]["argv"] = launch._syncer_argv(
        single["syncer"]["binary_path"],
        port=29400,
        learners=1,
        total_steps=4,
        fragments=4,
        run="/runs/m1",
    )
    single["learners"][0]["argv"] = launch._learner_argv(
        manifest=single, island_id=0, run_dir="/evidence"
    )
    assert launch._validate_manifest(single)["topology"]["host_gpu_count"] == 4

    too_short = copy.deepcopy(manifest)
    too_short["rounds"] = 2
    with pytest.raises(launch.LaunchContractError, match="requires 3..8 rounds"):
        launch._validate_manifest(too_short)


def test_linux_syncer_build_manifest_is_bound_without_launch_time_cargo(tmp_path):
    binary = tmp_path / "yeto-syncer"
    binary.write_bytes(
        b"\x7fELF" + bytes((2, 1)) + b"\0" * 12 + (62).to_bytes(2, "little")
    )
    binary.chmod(0o700)
    binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    payload = {
        "schema": "yeto-m1-syncer-linux-musl-build/v1",
        "binary": {
            "linkage": "static-pie",
            "mode": "0700",
            "sha256": binary_sha,
            "size_bytes": binary.stat().st_size,
        },
        "build": {
            "alpine_version": "3.22.1",
            "cargo_locked": True,
            "cargo_version": "1.88.0 (873a06493 2025-05-10)",
            "image_index_digest": f"sha256:{_sha('d')}",
            "image_linux_amd64_digest": f"sha256:{_sha('e')}",
            "musl_dev_version": "1.2.5-r12",
            "platform": "linux/amd64",
            "rust_commit": "6b00bc3880198600130e1cf62b8f8a93494488cc",
            "rust_version": "1.88.0",
        },
        "source": {
            "cargo_lock_sha256": _sha("f"),
            "cargo_toml_sha256": _sha("0"),
            "files": [{"path": "src/main.rs", "sha256": _sha("1")}],
        },
    }
    build = tmp_path / "build-manifest.json"
    build.write_bytes(launch._canonical(payload))
    build.chmod(0o600)
    build_sha = hashlib.sha256(build.read_bytes()).hexdigest()

    assert launch._is_linux_x86_64_elf(binary)
    assert (
        launch._validate_syncer_build_manifest(
            build,
            expected_sha256=build_sha,
            binary_path=binary,
            binary_sha256=binary_sha,
        )
        == payload
    )

    host_script = Path("tools/probes/run_m1_dense_full_two_island.sh").read_text()
    assert 'm1_dense_full_direct_launch.py" launch' in host_script
    assert "ssh_harness" not in host_script
    assert "cargo build" not in host_script


def _write_private(path: Path, value) -> None:
    path.write_bytes(launch._canonical(value))
    path.chmod(0o600)


def _write_terminal_fixture(tmp_path: Path) -> tuple[dict, str]:
    from yeto.rl.contracts import TrajectoryEnvelope
    from yeto.rl.trajectory_evidence import (
        TrajectoryBatchEvidence,
        _envelope_batch_hash,
        write_trajectory_batch_evidence,
    )

    manifest = _manifest()
    manifest_sha = _sha("f")
    run_root = tmp_path / "run"
    run_root.mkdir(mode=0o700)
    manifest["launch"]["host_run_root"] = str(run_root)
    for island_id, island in enumerate(manifest["topology"]["islands"]):
        island["host_run_dir"] = str(run_root / f"island-{island_id}")
    manifest["evaluation"] = launch._evaluation_contract(
        launch_mode=manifest["launch_mode"],
        rounds=manifest["rounds"],
        profile=manifest["profile"],
        data=manifest["data"],
        run_root=run_root,
        yeto_root=Path(manifest["provenance"]["yeto_root"]),
        island_count=manifest["topology"]["island_count"],
    )
    policy_hashes = [_sha(character) for character in "abcd"]
    trained_tokens = 6
    for island_id, island in enumerate(manifest["topology"]["islands"]):
        island_root = Path(island["host_run_dir"])
        island_root.mkdir(mode=0o700)
        (island_root / "audit").mkdir(mode=0o700)
        (island_root / "miles-metrics").mkdir(mode=0o700)
        (island_root / "learner.exit").write_text("0\n")
        (island_root / "learner.exit").chmod(0o600)
        _write_private(
            island_root / "container-started.json",
            {
                "schema": "yeto-m1-dense-full-island-started-v1",
                "island_id": island_id,
                "manifest_sha256": manifest_sha,
            },
        )
        _write_private(
            island_root / "ray-placement.json",
            {
                "schema": "yeto-m1-dense-full-ray-placement-v1",
                "island_id": island_id,
                "manifest_sha256": manifest_sha,
                "actor_bundle_indices": [0, 1],
                "actor_local_gpu_ids": [0, 1],
                "actor_gpu_uuids": island["trainer_gpu_uuids"],
                "inference_bundle_indices": [2, 3],
                "inference_local_gpu_ids": [2, 3],
                "inference_gpu_uuids": island["inference_gpu_uuids"],
                "inference_engine_count": 2,
                "inference_tp": 1,
            },
        )
        events = [
            {
                "event": "rl_dense_policy_publication",
                "island_id": island_id,
                "policy_version": 0,
                "sync/global_policy_hash": policy_hashes[0],
                "terminal": False,
            },
            {
                "event": "rl_eval_result",
                "island_id": island_id,
                "rollout_id": 0,
                "policy_version": 0,
                "sync/global_policy_hash": policy_hashes[0],
                "dataset_name": launch.EVAL_DATASET_NAME,
                "sample_count": 2,
                "rl/eval/result": 0.25,
                "rl/eval/pass_at_1": 0.0,
            },
        ]
        trajectory_root = (
            island_root / "audit" / f"trajectory-evidence-{island_id}-fixture"
        )
        trajectory_root.mkdir(mode=0o700)
        for version in range(manifest["rounds"]):
            envelopes = tuple(
                TrajectoryEnvelope(
                    trajectory_id=_sha(str(sample + 1)),
                    task_id=f"task-{island_id}-{version}-{sample}",
                    prompt_group_id=f"r{version}:g0",
                    sample_index=sample,
                    behavior_policy_version=version,
                    behavior_policy_hash=policy_hashes[version],
                    token_ids=(1, 2),
                    response_token_count=2,
                    behavior_logprobs_hash=_sha("e"),
                    reward=float(sample == 0),
                    reward_contract_hash=manifest["profile"]["reward_sha256"],
                    cleanup_evidence_hash=_sha("9"),
                )
                for sample in range(3)
            )
            input_hash = _envelope_batch_hash(envelopes, trained_tokens)
            write_trajectory_batch_evidence(
                trajectory_root,
                TrajectoryBatchEvidence(
                    version,
                    policy_hashes[version],
                    input_hash,
                    trained_tokens,
                    envelopes,
                ),
            )
            events.extend(
                (
                    {
                        "event": "rl_dense_local_step",
                        "island_id": island_id,
                        "base_policy_version": version,
                        "base_policy_hash": policy_hashes[version],
                        "target_policy_version": version + 1,
                        "target_policy_hash": policy_hashes[version + 1],
                        "input_batch_hash": input_hash,
                        "trained_tokens": trained_tokens,
                        "trajectory_count": 3,
                        "optimizer_steps": 1,
                        "sweep_update_id": _sha(str(version + 4)),
                    },
                    {
                        "event": "rl_dense_policy_publication",
                        "island_id": island_id,
                        "policy_version": version + 1,
                        "sync/global_policy_hash": policy_hashes[version + 1],
                        "terminal": version + 1 == manifest["rounds"],
                    },
                )
            )
        events.append(
            {
                "event": "rl_eval_result",
                "island_id": island_id,
                "rollout_id": manifest["rounds"],
                "policy_version": manifest["rounds"],
                "sync/global_policy_hash": policy_hashes[-1],
                "dataset_name": launch.EVAL_DATASET_NAME,
                "sample_count": 2,
                "rl/eval/result": 0.75,
                "rl/eval/pass_at_1": 0.5,
            }
        )
        event_path = island_root / "learner-events.jsonl"
        event_path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in events
            )
        )
        event_path.chmod(0o600)
        _write_private(
            island_root / "eval-summary.json",
            {
                "schema": launch.EVAL_SUMMARY_SCHEMA,
                "island_id": island_id,
                "dataset_name": launch.EVAL_DATASET_NAME,
                "prompt_count": 2,
                "samples_per_prompt": 1,
                "expected_policy_versions": [0, manifest["rounds"]],
                "complete": True,
                "results": [
                    {
                        "policy_version": 0,
                        "policy_hash": policy_hashes[0],
                        "sample_count": 2,
                        "result": 0.25,
                        "pass_at_1": 0.0,
                    },
                    {
                        "policy_version": manifest["rounds"],
                        "policy_hash": policy_hashes[-1],
                        "sample_count": 2,
                        "result": 0.75,
                        "pass_at_1": 0.5,
                    },
                ],
            },
        )
        metric_path = island_root / "miles-metrics" / "actor.jsonl"
        metric_path.write_text(
            json.dumps(
                {"metric": "train/grad_norm", "series": [[0, 1.0], [1, 1.1], [2, 1.2]]}
            )
            + "\n"
            + json.dumps(
                {
                    "metric": "train/train_rollout_kl",
                    "series": [[0, 0.01], [1, 0.02], [2, 0.03]],
                }
            )
            + "\n"
        )
        metric_path.chmod(0o600)

    syncer_root = run_root / "syncer"
    syncer_root.mkdir(mode=0o700)
    fragment_layout_hash = _sha("8")
    syncer_rows = []
    fragments = manifest["model"]["fragment_count"]
    for step in range(1, manifest["syncer"]["total_steps"] + 1):
        policy_round = (step - 1) // fragments + 1
        fragment = (step - 1) % fragments
        base = 0 if policy_round == 1 else (policy_round - 2) * fragments + fragment + 1
        syncer_rows.append(
            {
                "step": step,
                "protocol_version": 4,
                "delta_semantics": "local_minus_raw_anchor",
                "attempt": 1,
                "fragment": fragment,
                "policy_round": policy_round,
                "sweep_fragment": fragment,
                "sweep_fragments": fragments,
                "sweep_complete": fragment == fragments - 1,
                "launch_base_version": base,
                "sync/base_version": base,
                "sync/layout_hash": fragment_layout_hash,
                "sync/quorum": 2,
                "quorum": 2,
                "sync/responders": 2,
                "sync/rejected_stale_updates": 0,
                "sync/global_delta_norm": 0.1 * step,
                "gnorm": 0.1 * step,
                "expected": [0, 1],
                "responded": [0, 1],
                "expected_members": [
                    {"id": 0, "generation": 0},
                    {"id": 1, "generation": 0},
                ],
                "responded_members": [
                    {"id": 0, "generation": 0},
                    {"id": 1, "generation": 0},
                ],
                "missed_grace": [],
                "missed_members": [],
                "responders": [
                    {
                        "id": learner_id,
                        "generation": 0,
                        "base_version": base,
                        "staleness": 0,
                        "c_steps": 1,
                        "c_tokens": trained_tokens,
                        "accounted_c_steps": 1 if fragment == fragments - 1 else 0,
                        "accounted_c_tokens": trained_tokens
                        if fragment == fragments - 1
                        else 0,
                        "weight": 1.0,
                        "contribution": 0.5,
                    }
                    for learner_id in range(2)
                ],
            }
        )
    total_steps = manifest["syncer"]["total_steps"]
    syncer_rows.append(
        {
            "event": "policy_sweep_ledger",
            "event_id": f"policy-sweep-ledger:{fragment_layout_hash}:complete:{total_steps}",
            "phase": "complete",
            "protocol_version": 4,
            "sync/layout_hash": fragment_layout_hash,
            "global_step": total_steps,
            "policy_round": manifest["rounds"],
            "sweep_fragments": fragments,
            "sweep_complete": True,
            "versions": list(range(total_steps - fragments + 1, total_steps + 1)),
            "ledger": [
                {"id": learner_id, "merges": total_steps, "steps": 3, "tokens": 18}
                for learner_id in range(2)
            ],
        }
    )
    syncer_path = syncer_root / "events.jsonl"
    syncer_path.write_text("".join(json.dumps(row) + "\n" for row in syncer_rows))
    syncer_path.chmod(0o600)
    return manifest, manifest_sha


def test_terminal_reconciler_requires_complete_cross_island_policy_and_accounting(
    tmp_path,
):
    manifest, manifest_sha = _write_terminal_fixture(tmp_path)
    result = reporter.build_final_report(
        manifest,
        manifest_sha,
        daemon_health={
            "ok": True,
            "active_episodes": 0,
            "task_pack_sha256": launch.TASK_PACK_SHA256,
        },
    )

    assert result["schema"] == launch.FINAL_REPORT_SCHEMA
    assert result["status"] == "passed"
    assert result["final_policy"] == {"version": 3, "policy_hash": _sha("d")}
    assert result["totals"]["accepted_trajectories"] == 18
    assert result["totals"]["trained_tokens"] == 36
    assert result["totals"]["sync_merges"] == 12
    assert result["heldout"]["result_delta"] == 0.5
    assert "task-" not in json.dumps(result)

    broken = (
        Path(manifest["topology"]["islands"][1]["host_run_dir"])
        / "learner-events.jsonl"
    )
    text = broken.read_text().replace(_sha("d"), _sha("e"))
    broken.write_text(text)
    with pytest.raises(reporter.FinalReportError, match="heldout|different global"):
        reporter.build_final_report(
            manifest,
            manifest_sha,
            daemon_health={
                "ok": True,
                "active_episodes": 0,
                "task_pack_sha256": launch.TASK_PACK_SHA256,
            },
        )
