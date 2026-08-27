from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
from pathlib import Path

import pytest

from yeto.rl.sao_streaming_runtime import load_sao_streaming_runtime

TOOL = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "probes"
    / "build_tbench21_sao_streaming_contracts.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_tbench21_sao_streaming_contracts", TOOL
)
assert SPEC is not None and SPEC.loader is not None
contracts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contracts)


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _attestation(context_sha256: str, profile_sha256: str) -> dict[str, object]:
    def role(name: str, marker: str) -> dict[str, object]:
        return {
            "role": name,
            "component": {
                "model_revision": marker * 40,
                "config_sha256": marker * 64,
            },
            "parameter_layout_sha256": ("3" if name == "actor" else "4") * 64,
            "expected_fragments": 2,
            "parameter_tensor_count": 2,
            "parameter_scalar_count": 20,
            "fragments": [
                {
                    "fragment_id": index,
                    "role": name,
                    "shard_id": "tp0-of-1.pp0-of-1.ep0-of-1.cp0-of-1.dp0-of-1",
                    "parameter_count": 1,
                    "scalar_count": 10,
                    "payload_bytes": 40,
                }
                for index in range(2)
            ],
            "manifests": [{"topology": {}, "source_layout_sha256": "7" * 64}],
        }

    return {
        "schema": "miles.sao-streaming-layouts.v2",
        "sao_context_sha256": context_sha256,
        "settings": {
            "algorithm": "sao",
            "fragment_strategy": "owner_affine",
            "minimum_fragments": 2,
            "max_fragment_bytes": 2 << 30,
            "max_chunk_bytes": 256 << 20,
            "wire_dtype": "fp32",
            "syncer_profile_sha256": profile_sha256,
        },
        "actor": role("actor", "a"),
        "critic": role("critic", "b"),
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    plan_dir = tmp_path / "plan"
    files = {}
    for island_id in range(8):
        relative = f"train/island-{island_id}.jsonl"
        files[relative] = _write(
            plan_dir / relative,
            {
                "messages": [{"role": "system", "content": "solve"}],
                "metadata": {"island_id": island_id},
            },
        )
    plan = {
        "schema": "yeto.tbench21-sao-diloco-plan.v1",
        "topology": {
            "islands": 8,
            "physical_gpus": 8,
            "one_physical_gpu_per_island": True,
            "model": "Qwen/Qwen3.5-0.8B",
            "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        },
        "rollouts": {
            "train": 176,
            "per_task": 4,
            "episode_timeout_seconds": 1800,
        },
        "compaction": {"enabled": True, "trainer_objective": "sao"},
        "files": files,
    }
    plan_path = plan_dir / "manifest.json"
    _write(plan_path, plan)
    reward = tmp_path / "reward.py"
    reward.write_text("def reward(): return 1\n")
    critic = tmp_path / "value_pretrain_contract.json"
    _write(critic, {"schema": "test"})
    return plan_dir, plan_path, reward, critic


def test_two_pass_builds_eight_loadable_runtime_contracts(tmp_path):
    plan_dir, plan_path, reward, critic = _inputs(tmp_path)

    run_root = tmp_path / "run"
    run_root.mkdir()
    provisional_context = tmp_path / "provisional-sao-context.json"
    provisional_context_sha = contracts.write_probe_context(
        plan_path=plan_path,
        reward_source=reward,
        output=provisional_context,
        container_plan_dir=plan_dir,
        container_run_dir=run_root / "provisional-layout",
    )
    provisional_profile = contracts._profile(1)
    provisional_semantic = contracts.SyncerSemanticProfile.from_mapping(
        provisional_profile
    ).sha256
    provisional_path = tmp_path / "provisional-attestation.json"
    _write(
        provisional_path,
        _attestation(provisional_context_sha, provisional_semantic),
    )
    output = tmp_path / "contracts"
    contracts.prepare(
        plan_path=plan_path,
        provisional_context_path=provisional_context,
        provisional_attestation_path=provisional_path,
        reward_source=reward,
        critic_contract=critic,
        output_dir=output,
        container_plan_dir=plan_dir,
        container_contract_dir=output,
        container_run_root=run_root,
    )
    prepared = json.loads((output / "prepare-manifest.json").read_text())
    assert prepared["expected_fragments"] == 2
    assert prepared["syncer_profile"]["profile"]["total_steps"] == 2
    assert prepared["provisional_context"] == {
        "path": str(provisional_context),
        "sha256": provisional_context_sha,
    }

    final_profile_sha = prepared["syncer_profile"]["semantic_sha256"]
    final_probe = tmp_path / "final-attestation.json"
    _write(
        final_probe,
        _attestation(prepared["contexts"][0]["sha256"], final_profile_sha),
    )
    contracts.finalize(
        prepare_manifest_path=output / "prepare-manifest.json",
        final_attestation_path=final_probe,
        syncer_host="host.docker.internal",
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["phase"] == "final"
    assert len(manifest["islands"]) == 8
    for island_id, record in enumerate(manifest["islands"]):
        runtime = load_sao_streaming_runtime(
            record["streaming_context"]["host_path"],
            expected_sha256=record["streaming_context"]["sha256"],
            expected_sao_context_sha256=record["sao_context"]["sha256"],
        )
        assert runtime.streams.actor.learner_id == island_id
        assert runtime.streams.actor.total_fragment_steps == 2
        assert runtime.streams.critic.optimizer_steps_per_round == 2
        assert runtime.trajectory_evidence_kind == "terminal-bench-2.1"


def test_probe_context_is_deterministic_private_and_plan_bound(tmp_path):
    plan_dir, plan_path, reward, _ = _inputs(tmp_path)
    run_dir = tmp_path / "probe-run"
    first = tmp_path / "first-context.json"
    second = tmp_path / "second-context.json"

    contracts.main(
        [
            "probe-context",
            "--plan-manifest",
            str(plan_path),
            "--reward-source",
            str(reward),
            "--output",
            str(first),
            "--container-plan-dir",
            str(plan_dir),
            "--container-run-dir",
            str(run_dir),
        ]
    )
    first_sha = hashlib.sha256(first.read_bytes()).hexdigest()
    second_sha = contracts.write_probe_context(
        plan_path=plan_path,
        reward_source=reward,
        output=second,
        container_plan_dir=plan_dir,
        container_run_dir=run_dir,
    )

    assert first_sha == second_sha
    assert first.read_bytes() == second.read_bytes()
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    context = json.loads(first.read_text())
    assert set(context) == {
        "schema",
        "benchmark",
        "model",
        "data",
        "data_sha256",
        "base_model_revision",
        "rollout_model_revision",
        "data_revision",
        "layout_hash",
        "lora_config_hash",
        "reward_sha256",
        "dynamic_sampling_max_replacements",
        "completed_groups_path",
        "event_tape",
        "learner_id",
    }
    assert context["schema"] == "miles.sao-runtime.v2"
    assert context["benchmark"] == "terminal-bench-2.1"
    assert context["layout_hash"] == contracts.PROVISIONAL_LAYOUT_HASH
    assert (
        context["data_revision"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    )
    assert (
        context["data_sha256"]
        == hashlib.sha256((plan_dir / "train/island-0.jsonl").read_bytes()).hexdigest()
    )
    assert context["reward_sha256"] == hashlib.sha256(reward.read_bytes()).hexdigest()


def test_prepare_rejects_layout_attestation_for_another_probe_context(tmp_path):
    plan_dir, plan_path, reward, critic = _inputs(tmp_path)
    context = tmp_path / "provisional-sao-context.json"
    contracts.write_probe_context(
        plan_path=plan_path,
        reward_source=reward,
        output=context,
        container_plan_dir=plan_dir,
        container_run_dir=tmp_path / "probe-run",
    )
    semantic_sha = contracts.SyncerSemanticProfile.from_mapping(
        contracts._profile(1)
    ).sha256
    attestation = tmp_path / "provisional-attestation.json"
    _write(attestation, _attestation("f" * 64, semantic_sha))

    with pytest.raises(
        contracts.ContractError,
        match="not bound to the probe context",
    ):
        contracts.prepare(
            plan_path=plan_path,
            provisional_context_path=context,
            provisional_attestation_path=attestation,
            reward_source=reward,
            critic_contract=critic,
            output_dir=tmp_path / "contracts",
            container_plan_dir=plan_dir,
            container_contract_dir=tmp_path / "container-contracts",
            container_run_root=tmp_path / "online-run",
        )


def test_prepare_rejects_modified_probe_context_even_when_rehashed(tmp_path):
    plan_dir, plan_path, reward, critic = _inputs(tmp_path)
    context = tmp_path / "provisional-sao-context.json"
    contracts.write_probe_context(
        plan_path=plan_path,
        reward_source=reward,
        output=context,
        container_plan_dir=plan_dir,
        container_run_dir=tmp_path / "probe-run",
    )
    payload = json.loads(context.read_text())
    payload["layout_hash"] = "f" * 64
    context_sha = _write(context, payload)
    context.chmod(0o600)
    semantic_sha = contracts.SyncerSemanticProfile.from_mapping(
        contracts._profile(1)
    ).sha256
    attestation = tmp_path / "provisional-attestation.json"
    _write(attestation, _attestation(context_sha, semantic_sha))

    with pytest.raises(
        contracts.ContractError,
        match="differs from its plan/reward binding",
    ):
        contracts.prepare(
            plan_path=plan_path,
            provisional_context_path=context,
            provisional_attestation_path=attestation,
            reward_source=reward,
            critic_contract=critic,
            output_dir=tmp_path / "contracts",
            container_plan_dir=plan_dir,
            container_contract_dir=tmp_path / "container-contracts",
            container_run_root=tmp_path / "online-run",
        )
