from __future__ import annotations

# ruff: noqa: I001 -- install the optional Bridge stub before Miles import.

import asyncio
import json
import sys
import types
from types import SimpleNamespace

import pytest

bridge_stub = types.ModuleType("yeto.rl.deepseek_v4_bridge")
bridge_stub.ensure_deepseek_v4_bridge = lambda: None
sys.modules.setdefault("yeto.rl.deepseek_v4_bridge", bridge_stub)

from miles.backends.megatron_utils.full_parameter_state import (
    FullParameterShardManifest,
    FullParameterShardSpec,
    FullParameterTopology,
    _layout_hash,
)
from yeto.rl import miles_full_parameter_manifest_probe as module
from yeto.rl.miles_full_parameter_manifest_probe import (
    MilesFullParameterManifestProbeSync,
)


def _topology(rank: int) -> FullParameterTopology:
    return FullParameterTopology(
        tp_rank=rank,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        ep_rank=0,
        ep_size=1,
        cp_rank=0,
        cp_size=1,
        dp_rank=0,
        dp_size=1,
    )


def _manifest(rank: int) -> FullParameterShardManifest:
    topology = _topology(rank)
    specs = tuple(
        sorted(
            FullParameterShardSpec(
                role="actor",
                shard_id=topology.shard_id,
                name=f"model.weight{index}",
                shape=(300_000_000,),
                dtype="float32",
                numel=300_000_000,
            )
            for index in range(2)
        )
    )
    return FullParameterShardManifest(
        topology=topology,
        role="actor",
        layout_hash=_layout_hash(topology, specs),
        specs=specs,
    )


class ManifestOnlyGroup:
    async def full_parameter_shard_manifests(self):
        return (_manifest(1), _manifest(0))


def _args():
    return SimpleNamespace(
        debug_train_only=True,
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        seq_length=4096,
        start_rollout_id=0,
        num_rollout=1,
        use_critic=False,
        lora_rank=0,
        external_policy_sync_run_until_stop=False,
        eval_interval=None,
    )


def test_manifest_probe_derives_bounded_owner_plan_without_parameter_payloads(
    monkeypatch,
    tmp_path,
):
    evidence_path = tmp_path / "manifest-probe.json"
    monkeypatch.setenv(
        "YETO_FULL_PARAMETER_MANIFEST_PROBE_EVIDENCE",
        str(evidence_path),
    )
    monkeypatch.setenv("YETO_FULL_PARAMETER_MODEL_REVISION", "a" * 40)
    monkeypatch.setenv("YETO_FULL_PARAMETER_CONFIG_HASH", "b" * 64)
    monkeypatch.setenv(
        "YETO_FULL_PARAMETER_CONVERSION_MANIFEST_SHA256",
        "c" * 64,
    )
    monkeypatch.setenv("YETO_MILES_IMAGE_DIGEST", f"sha256:{'d' * 64}")
    monkeypatch.setattr(
        module,
        "_source_provenance",
        lambda: (
            {"path": "/root/yeto", "source_tree_sha256": "e" * 64},
            {"path": "/root/miles", "execution_source_sha256": "f" * 64},
        ),
    )
    monkeypatch.setattr(module, "_hardware_identity", lambda: {"gpu_count": 2})
    monkeypatch.setattr(
        module,
        "_megatron_bridge_identity",
        lambda: {"direct_url_commit": "1" * 40},
    )
    monkeypatch.setattr(
        module,
        "_conversion_manifest_provenance",
        lambda **_kwargs: {
            "path": "/models/torch_dist/conversion-manifest.json",
            "sha256": "c" * 64,
        },
    )
    args = _args()
    probe = MilesFullParameterManifestProbeSync(args)

    asyncio.run(
        probe.initialize(actor_model=ManifestOnlyGroup(), rollout_manager=object())
    )
    assert args.num_rollout == args.start_rollout_id == 0
    with pytest.raises(RuntimeError, match="entered the train loop"):
        asyncio.run(probe.after_local_train())
    asyncio.run(probe.finalize())

    encoded = evidence_path.read_bytes()
    payload = json.loads(encoded)
    assert (
        encoded
        == (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    assert evidence_path.stat().st_mode & 0o777 == 0o600
    assert payload["probe_mode"] == "full_parameter_manifest_only"
    assert payload["algorithm"] == "grpo"
    assert payload["fragment_strategy"] == "owner_affine"
    assert payload["sequence_length"] == 4096
    assert payload["owner_count"] == payload["minimum_fragment_count"] == 2
    assert payload["derived_fragment_count"] == 4
    assert payload["parameter_tensor_count"] == 4
    assert payload["parameter_scalar_count"] == 1_200_000_000
    assert payload["observed_max_fragment_bytes"] == 1_200_000_000
    assert payload["observed_max_fragment_bytes"] <= 2 << 30
    assert [row["fragment_id"] for row in payload["fragments"]] == list(range(4))
    assert {row["shard_id"] for row in payload["fragments"]} == {
        _topology(0).shard_id,
        _topology(1).shard_id,
    }
    assert all(row["plan_hash"] for row in payload["owner_plans"])
