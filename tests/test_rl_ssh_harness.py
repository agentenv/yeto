import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.core import CanonicalTensorSpec, canonical_state, tensors_from_flat
from yeto.rl.export import specs_manifest
from yeto.rl.manifest import build_run_manifest, canonical_json, manifest_sha256
from yeto.rl.ssh_harness import (
    HarnessError,
    _container_name,
    _docker_ref,
    _learner_argv,
    _target_host,
    _verify_oracle,
    load_plan,
)


def _bytes(value):
    return value.numpy().astype("<f4", copy=False).tobytes()


def _write_jsonl(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def _oracle_fixture(tmp_path):
    specs = (CanonicalTensorSpec("base_model.model.x.lora_A.weight", (2,), 2),)
    layout = specs_manifest(specs)
    manifest_hash = "ab" * 32
    manifest = {
        "workload": {
            "learners": 2,
            "global_rounds": 2,
            "groups_per_island_round": 1,
            "samples_per_group": 2,
            "local_optimizer_steps": 1,
        },
        "canonical_lora": layout,
    }
    plan = {"manifest_sha256": manifest_hash}
    bases = [torch.tensor([0.0, 0.0]), torch.tensor([2.0, 2.0])]
    deltas = [
        [torch.tensor([1.0, 3.0]), torch.tensor([3.0, 1.0])],
        [torch.tensor([2.0, 0.0]), torch.tensor([0.0, 2.0])],
    ]
    commits = [torch.tensor([2.0, 2.0]), torch.tensor([3.0, 3.0])]
    syncer_events = []
    learner_events = [[], []]
    for version in (1, 2):
        base_state = canonical_state(
            version - 1,
            tensors_from_flat(bases[version - 1], specs),
            expected_specs=specs,
            expected_layout_fingerprint=layout["layout_fingerprint"],
        )
        commit_state = canonical_state(
            version,
            tensors_from_flat(commits[version - 1], specs),
            expected_specs=specs,
            expected_layout_fingerprint=layout["layout_fingerprint"],
        )
        digests = []
        for learner_id in (0, 1):
            audit = tmp_path / f"learner-{learner_id}" / "audit"
            audit.mkdir(parents=True, exist_ok=True)
            stem = f"round-{version:08d}"
            base_data = _bytes(bases[version - 1])
            delta_data = _bytes(deltas[version - 1][learner_id])
            delta_sha = hashlib.sha256(delta_data).hexdigest()
            (audit / f"{stem}.base.f32").write_bytes(base_data)
            (audit / f"{stem}.delta.f32").write_bytes(delta_data)
            (audit / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "run_manifest_sha256": manifest_hash,
                        "learner_id": learner_id,
                        "base_version": version - 1,
                        "base_policy_hash": base_state.policy_hash,
                        "target_step": version,
                        "layout_fingerprint": layout["layout_fingerprint"],
                        "numel": 2,
                        "base_f32_sha256": hashlib.sha256(base_data).hexdigest(),
                        "delta_sha256": delta_sha,
                    }
                )
            )
            digests.append({"id": learner_id, "delta_sha256": delta_sha})
            learner_events[learner_id].append(
                {
                    "run_manifest_sha256": manifest_hash,
                    "learner_id": learner_id,
                    "base_version": version - 1,
                    "base_policy_hash": base_state.policy_hash,
                    "committed_version": version,
                    "committed_policy_hash": commit_state.policy_hash,
                    "delta_sha256": delta_sha,
                    "groups": 1,
                    "samples_per_group": 2,
                    "trajectories": 2,
                    "optimizer_steps": 1,
                    "rollout_identity_set": [
                        {"version": version - 1, "policy_hash": base_state.policy_hash}
                    ],
                    "trainer_applied_identity": {
                        "version": version,
                        "policy_hash": commit_state.policy_hash,
                    },
                    "rollout_applied_identity": {
                        "version": version,
                        "policy_hash": commit_state.policy_hash,
                    },
                }
            )
        syncer_events.append(
            {
                "committed_version": version,
                "run_manifest_sha256": manifest_hash,
                "fixed_roster": 2,
                "responded": [0, 1],
                "checkpoint_committed": True,
                "delta_digests": digests,
                "policy_sha256": commit_state.policy_hash,
            }
        )
    _write_jsonl(tmp_path / "syncer/syncer-events.jsonl", syncer_events)
    for learner_id in (0, 1):
        _write_jsonl(
            tmp_path / f"learner-{learner_id}/cache/events.jsonl",
            learner_events[learner_id],
        )
    checkpoint = SimpleNamespace(
        layout_fingerprint=layout["layout_fingerprint"],
        fragments=((3.0, 3.0),),
    )
    return plan, manifest, checkpoint


def test_two_host_oracle_matches_ordered_f32_average(tmp_path):
    plan, manifest, checkpoint = _oracle_fixture(tmp_path)
    _verify_oracle(plan, manifest, checkpoint, tmp_path)

    delta = tmp_path / "learner-1/audit/round-00000002.delta.f32"
    delta.write_bytes(_bytes(torch.tensor([9.0, 9.0])))
    with pytest.raises(HarnessError, match="audit identity mismatch"):
        _verify_oracle(plan, manifest, checkpoint, tmp_path)


def test_harness_requires_one_digest_pinned_docker_image():
    digest = "radixark/miles@sha256:" + "d" * 64
    assert _docker_ref("docker:" + digest) == digest
    with pytest.raises(HarnessError, match="sha256:DIGEST"):
        _docker_ref("docker:radixark/miles:latest")


def test_harness_uses_first_ssh_host_as_default_syncer_name():
    assert _target_host("alice@h200-a.tailnet.ts.net") == "h200-a.tailnet.ts.net"
    assert _target_host("100.64.1.2") == "100.64.1.2"


def test_learner_command_keeps_logical_id_cache_and_audit_paths():
    plan = {
        "run_id": "acceptance",
        "manifest_sha256": "ab" * 32,
        "source_sha256": "cd" * 32,
        "syncer_address": "h200-a:29400",
        "data": {"learner_arg": "org/data"},
        "learner": {
            "model": "org/model",
            "model_revision": "a" * 40,
            "data_revision": "b" * 40,
            "reward_function": "pkg.reward:score",
            "reward_sha256": "ef" * 32,
            "generate_function": None,
            "generate_sha256": None,
            "global_rounds": 2,
            "groups_per_round": 1,
            "samples_per_group": 2,
            "optimizer_steps": 1,
            "round_timeout_s": 0,
            "lora_r": 8,
            "lora_targets": "attention",
            "inner_lr": 1e-5,
            "seq_len": 128,
            "seed": 7,
            "wan_streams": 4,
            "trust_remote_code": True,
        },
    }
    argv = _learner_argv(plan, 1)
    assert argv[argv.index("--learner-id") + 1] == "1"
    assert argv[argv.index("--cache-dir") + 1] == "/workspace/cache"
    assert argv[argv.index("--audit-dir") + 1] == "/workspace/audit"
    assert "--trust-remote-code" in argv
    assert _container_name(plan, 1) == "yeto-rl-acceptance-l1"


def test_persisted_plan_is_bound_to_the_canonical_manifest(tmp_path):
    layout = specs_manifest(
        (CanonicalTensorSpec("base_model.model.x.lora_A.weight", (2,), 2),)
    )
    image = "docker:radixark/miles@sha256:" + "d" * 64
    args = SimpleNamespace(
        cluster_prefix="acceptance",
        model="org/model",
        model_revision="a" * 40,
        model_requested_identifier="org/model",
        model_requested_revision="main",
        data="org/data",
        data_revision="b" * 40,
        data_requested_identifier="org/data",
        data_requested_revision="main",
        trust_remote_code=True,
        lora_r=8,
        lora_targets="attention",
        rl_groups_per_island_round=1,
        rl_samples_per_group=2,
        rl_local_optimizer_steps=1,
        rl_global_rounds=2,
        inner_lr=1e-5,
        seq_len=128,
        seed=7,
        reward_function="pkg.reward:score",
        rl_generate_function=None,
        rl_generate_sha256=None,
        source_sha256="cd" * 32,
        learner_image=image,
        rl_canonical_layout=layout,
        rl_data_sha256=None,
        _provenance={
            "model": {
                "resolved_identifier": "org/model",
                "resolved_revision": "a" * 40,
            },
            "dataset": {
                "resolved_identifier": "org/data",
                "resolved_revision": "b" * 40,
            },
        },
    )
    manifest = build_run_manifest(args, learners=2, reward_sha256="ef" * 32)
    text = canonical_json(manifest)
    digest = manifest_sha256(text)
    learner = {
        "model": "org/model",
        "model_revision": "a" * 40,
        "data_revision": "b" * 40,
        "reward_function": None,
        "reward_sha256": "ef" * 32,
        "generate_function": None,
        "generate_sha256": None,
        "global_rounds": 2,
        "groups_per_round": 1,
        "samples_per_group": 2,
        "optimizer_steps": 1,
        "lora_r": 8,
        "lora_targets": "attention",
        "inner_lr": 1e-5,
        "seq_len": 128,
        "seed": 7,
        "trust_remote_code": True,
    }
    learner["reward_function"] = manifest["reward"]["callable"]
    plan = {
        "schema": 1,
        "run_id": "acceptance",
        "remote_run": ".cache/yeto/acceptance",
        "hosts": ["alice@h200-a", "alice@h200-b"],
        "docker_image": image.removeprefix("docker:"),
        "syncer_address": "h200-a:29400",
        "manifest_sha256": digest,
        "source_sha256": "cd" * 32,
        "syncer_source_sha256": "12" * 32,
        "learner": learner,
        "syncer": {
            "pipeline": 1,
            "sync_interval_steps": 0.0,
            "delta_correction": "none",
            "outer_lr": 1.0,
            "outer_momentum": 0.0,
            "total_steps": 2,
        },
    }
    (tmp_path / "rl-manifest.json").write_text(text)
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    _, loaded, loaded_manifest = load_plan(tmp_path / "plan.json")
    assert loaded == plan
    assert loaded_manifest == text

    plan["docker_image"] = "radixark/other@sha256:" + "e" * 64
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(HarnessError, match="canonical manifest"):
        load_plan(tmp_path / "plan.json")
