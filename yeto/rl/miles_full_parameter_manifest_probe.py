"""Metadata-only fragment-plan probe for a TP2 Qwen3.5-4B actor."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from ..provenance import source_tree_sha256
from .local_learner import ComponentIdentity
from .miles import miles_execution_source_sha256
from .miles_chunked_full_parameter import MilesChunkedFullParameterAdapter
from .miles_full_parameter_probe import _hardware_identity, _write_private_json

_EVIDENCE_ENV = "YETO_FULL_PARAMETER_MANIFEST_PROBE_EVIDENCE"
_MODEL_REVISION_ENV = "YETO_FULL_PARAMETER_MODEL_REVISION"
_CONFIG_HASH_ENV = "YETO_FULL_PARAMETER_CONFIG_HASH"
_CONVERSION_MANIFEST_ENV = "YETO_FULL_PARAMETER_CONVERSION_MANIFEST_SHA256"
_CONVERSION_MANIFEST_PATH_ENV = "YETO_FULL_PARAMETER_CONVERSION_MANIFEST_PATH"
_IMAGE_DIGEST_ENV = "YETO_MILES_IMAGE_DIGEST"
_YETO_SOURCE_ROOT_ENV = "YETO_FULL_PARAMETER_YETO_SOURCE_ROOT"
_YETO_SOURCE_SHA256_ENV = "YETO_FULL_PARAMETER_YETO_SOURCE_SHA256"
_MILES_SOURCE_ROOT_ENV = "YETO_FULL_PARAMETER_MILES_SOURCE_ROOT"
_MILES_SOURCE_SHA256_ENV = "YETO_FULL_PARAMETER_MILES_SOURCE_SHA256"

_MODEL_REPO = "Qwen/Qwen3.5-4B"
_MODEL_PATH = "/models/hf"
_MAX_FRAGMENT_BYTES = 2 << 30
_MAX_CHUNK_BYTES = 256 << 20
_PRODUCTION_SEQUENCE_LENGTH = 4096
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _required_environment(name: str, pattern: re.Pattern[str]) -> str:
    value = os.environ.get(name, "")
    if not pattern.fullmatch(value):
        raise RuntimeError(f"{name} is malformed")
    return value


def _real_directory_environment(name: str) -> Path:
    raw = os.environ.get(name)
    if raw is None:
        raise RuntimeError(f"{name} is required")
    unresolved = Path(raw)
    if (
        not unresolved.is_absolute()
        or unresolved.is_symlink()
        or not unresolved.is_dir()
    ):
        raise RuntimeError(f"{name} is not a real absolute directory")
    return unresolved.resolve()


def _source_provenance() -> tuple[dict[str, str], dict[str, str]]:
    yeto_root = _real_directory_environment(_YETO_SOURCE_ROOT_ENV)
    miles_root = _real_directory_environment(_MILES_SOURCE_ROOT_ENV)
    expected_yeto = _required_environment(_YETO_SOURCE_SHA256_ENV, _SHA256)
    expected_miles = _required_environment(_MILES_SOURCE_SHA256_ENV, _SHA256)
    actual_yeto = source_tree_sha256(yeto_root / "yeto")
    actual_miles = miles_execution_source_sha256(miles_root)
    if actual_yeto != expected_yeto:
        raise RuntimeError("Yeto execution source identity changed")
    if actual_miles != expected_miles:
        raise RuntimeError("Miles execution source identity changed")
    return (
        {"path": str(yeto_root), "source_tree_sha256": actual_yeto},
        {"path": str(miles_root), "execution_source_sha256": actual_miles},
    )


def _conversion_manifest_provenance(
    *,
    model_revision: str,
    model_config_sha256: str,
    image_digest: str,
    expected_sha256: str,
) -> dict[str, object]:
    raw_path = os.environ.get(_CONVERSION_MANIFEST_PATH_ENV)
    if raw_path is None:
        raise RuntimeError(f"{_CONVERSION_MANIFEST_PATH_ENV} is required")
    unresolved = Path(raw_path)
    if (
        not unresolved.is_absolute()
        or unresolved.is_symlink()
        or not unresolved.is_file()
        or unresolved.stat().st_mode & 0o077
    ):
        raise RuntimeError("conversion manifest path or mode is invalid")
    path = unresolved.resolve()
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise RuntimeError("conversion manifest identity changed")
    payload = json.loads(encoded)
    canonical = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if (
        encoded != canonical
        or payload.get("schema") != "yeto-qwen35-megatron-conversion-v1"
    ):
        raise RuntimeError("conversion manifest is not canonical")
    if (
        payload.get("model_repo") != _MODEL_REPO
        or payload.get("model_revision") != model_revision
        or payload.get("model_config_sha256") != model_config_sha256
        or payload.get("image_digest") != image_digest
    ):
        raise RuntimeError("conversion manifest provenance changed")
    model_files = payload.get("model_files")
    checkpoint_files = payload.get("checkpoint_files")
    source_hash = payload.get("conversion_source_aggregate_sha256")
    if (
        not isinstance(model_files, list)
        or not model_files
        or not isinstance(checkpoint_files, list)
        or not checkpoint_files
        or not isinstance(source_hash, str)
        or not _SHA256.fullmatch(source_hash)
    ):
        raise RuntimeError("conversion manifest inventory is incomplete")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "schema": payload["schema"],
        "model_file_count": len(model_files),
        "checkpoint_file_count": len(checkpoint_files),
        "conversion_source_aggregate_sha256": source_hash,
    }


def _megatron_bridge_identity() -> dict[str, object]:
    try:
        distribution = metadata.distribution("megatron-bridge")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("Megatron-Bridge distribution is not installed") from exc
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("Megatron-Bridge has no installed direct_url provenance")
    direct_url = json.loads(direct_url_text)
    if not isinstance(direct_url, dict):
        raise TypeError("Megatron-Bridge direct_url provenance is malformed")
    vcs = direct_url.get("vcs_info")
    if not isinstance(vcs, dict):
        raise TypeError("Megatron-Bridge direct_url has no VCS identity")
    commit = vcs.get("commit_id")
    if (
        not isinstance(direct_url.get("url"), str)
        or not direct_url["url"]
        or vcs.get("vcs") != "git"
        or not isinstance(commit, str)
        or not _GIT_COMMIT.fullmatch(commit)
    ):
        raise RuntimeError("Megatron-Bridge direct_url commit is unavailable")
    distribution_name = distribution.metadata.get("Name")
    if not isinstance(distribution_name, str) or not distribution_name:
        raise RuntimeError("Megatron-Bridge distribution name is unavailable")
    return {
        "distribution_name": distribution_name,
        "distribution_version": distribution.version,
        "direct_url": direct_url,
        "direct_url_commit": commit,
    }


def _require_probe_profile(args) -> int:
    expected = {
        "debug_train_only": True,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 2,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "seq_length": _PRODUCTION_SEQUENCE_LENGTH,
    }
    for name, value in expected.items():
        if getattr(args, name, None) != value:
            raise RuntimeError(f"manifest probe requires {name}={value!r}")
    start = getattr(args, "start_rollout_id", None)
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise RuntimeError("manifest probe received an invalid start rollout ID")
    if (
        getattr(args, "num_rollout", None) != start + 1
        or getattr(args, "use_critic", False)
        or getattr(args, "lora_rank", 0) > 0
        or getattr(args, "external_policy_sync_run_until_stop", False)
        or getattr(args, "eval_interval", None) is not None
    ):
        raise RuntimeError("manifest probe requires the exact metadata-only schedule")
    return start


def _plan_evidence(
    adapter: MilesChunkedFullParameterAdapter,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    plans_by_topology = {plan.topology: plan for plan in adapter.plans}
    owner_rows = []
    fragment_rows = []
    for manifest in adapter.manifests:
        plan = plans_by_topology.get(manifest.topology)
        if plan is None:
            raise RuntimeError("fragment plan omitted a topology owner")
        fragments = tuple(plan.fragments)
        owner_fragment_bytes = [fragment.numel * 4 for fragment in fragments]
        owner_rows.append(
            {
                "role": manifest.role,
                "shard_id": manifest.topology.shard_id,
                "topology": asdict(manifest.topology),
                "manifest_layout_hash": manifest.layout_hash,
                "plan_hash": plan.plan_hash,
                "parameter_tensor_count": len(manifest.specs),
                "parameter_scalar_count": sum(spec.numel for spec in manifest.specs),
                "fragment_ids": [fragment.fragment_id for fragment in fragments],
                "fragment_count": len(fragments),
                "max_fragment_bytes": max(owner_fragment_bytes),
            }
        )
        for fragment in fragments:
            fragment_rows.append(
                {
                    "fragment_id": fragment.fragment_id,
                    "role": manifest.role,
                    "shard_id": manifest.topology.shard_id,
                    "plan_hash": plan.plan_hash,
                    "tensor_count": len(fragment.wire_names),
                    "numel": fragment.numel,
                    "fp32_bytes": fragment.numel * 4,
                }
            )
    fragment_rows.sort(key=lambda row: int(row["fragment_id"]))
    expected_ids = list(range(adapter.layout.fragments.num_fragments))
    if [row["fragment_id"] for row in fragment_rows] != expected_ids:
        raise RuntimeError("fragment plans do not exactly cover the derived layout")
    observed_max = max(int(row["fp32_bytes"]) for row in fragment_rows)
    if observed_max > adapter.max_fragment_bytes:
        raise RuntimeError("derived fragment plan exceeds its semantic byte bound")
    return owner_rows, fragment_rows, observed_max


class MilesFullParameterManifestProbeSync:
    """Derive the real owner-affine plan without exporting parameter payloads."""

    def __init__(self, args) -> None:
        self.args = args
        evidence = os.environ.get(_EVIDENCE_ENV)
        if evidence is None:
            raise RuntimeError(f"{_EVIDENCE_ENV} is required")
        self.evidence_path = Path(evidence)
        self.component = ComponentIdentity(
            "actor",
            os.environ.get(_MODEL_REVISION_ENV, ""),
            os.environ.get(_CONFIG_HASH_ENV, ""),
        )
        self.conversion_manifest_sha256 = _required_environment(
            _CONVERSION_MANIFEST_ENV,
            _SHA256,
        )
        self.image_digest = _required_environment(_IMAGE_DIGEST_ENV, _IMAGE_DIGEST)
        self.evidence: dict[str, object] | None = None

    async def initialize(self, *, actor_model, rollout_manager) -> None:
        del rollout_manager
        start = _require_probe_profile(self.args)
        manifests = tuple(await actor_model.full_parameter_shard_manifests())
        owner_count = len(manifests)
        adapter = MilesChunkedFullParameterAdapter.create(
            manifests,
            algorithm="grpo",
            components=(self.component,),
            minimum_fragments=owner_count,
            max_fragment_bytes=_MAX_FRAGMENT_BYTES,
            max_chunk_bytes=_MAX_CHUNK_BYTES,
        )
        topology_sizes = {
            (
                manifest.topology.tp_size,
                manifest.topology.pp_size,
                manifest.topology.ep_size,
                manifest.topology.cp_size,
                manifest.topology.dp_size,
            )
            for manifest in adapter.manifests
        }
        if topology_sizes != {(2, 1, 1, 1, 1)} or owner_count != 2:
            raise RuntimeError("manifest probe did not observe the exact TP2 topology")
        owner_rows, fragment_rows, observed_max = _plan_evidence(adapter)
        yeto_source, miles_source = _source_provenance()
        conversion = _conversion_manifest_provenance(
            model_revision=self.component.model_revision,
            model_config_sha256=self.component.config_hash,
            image_digest=self.image_digest,
            expected_sha256=self.conversion_manifest_sha256,
        )
        self.evidence = {
            "schema": "yeto-miles-full-parameter-manifest-probe-v1",
            "observed_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "probe_mode": "full_parameter_manifest_only",
            "algorithm": adapter.layout.algorithm,
            "role": self.component.role,
            "fragment_strategy": adapter.layout.fragment_strategy,
            "model_repo": _MODEL_REPO,
            "model_revision": self.component.model_revision,
            "model_config_sha256": self.component.config_hash,
            "model_path": _MODEL_PATH,
            "checkpoint_path": str(Path(conversion["path"]).parent),
            "conversion_manifest": conversion,
            "miles_image_digest": self.image_digest,
            "yeto_source": yeto_source,
            "miles_source": miles_source,
            "hardware": _hardware_identity(),
            "megatron_bridge": _megatron_bridge_identity(),
            "actor_topology": {
                "actor_num_nodes": 1,
                "actor_num_gpus_per_node": 2,
                "tp_size": 2,
                "pp_size": 1,
                "ep_size": 1,
                "cp_size": 1,
                "dp_size": 1,
            },
            "sequence_length": _PRODUCTION_SEQUENCE_LENGTH,
            "parameter_layout_hash": adapter.layout.layout_hash,
            "owner_count": owner_count,
            "minimum_fragment_count": owner_count,
            "derived_fragment_count": adapter.layout.fragments.num_fragments,
            "parameter_tensor_count": adapter.expected_parameter_tensor_count,
            "parameter_scalar_count": adapter.expected_parameter_scalar_count,
            "max_fragment_bytes_limit": adapter.max_fragment_bytes,
            "max_chunk_bytes": adapter.max_chunk_bytes,
            "observed_max_fragment_bytes": observed_max,
            "owner_plans": owner_rows,
            "fragments": fragment_rows,
        }
        # ``train.py`` already constructed the actor.  Closing this half-open
        # range skips generation and training; debug-train-only also makes its
        # mandatory initial SGLang publication a no-op.
        self.args.num_rollout = start

    async def after_local_train(self, **_kwargs) -> bool:
        raise RuntimeError("metadata-only manifest probe entered the train loop")

    async def finalize(self) -> None:
        if self.evidence is None:
            raise RuntimeError("metadata-only manifest probe did not complete")
        _write_private_json(self.evidence_path, self.evidence)


def create_full_parameter_manifest_probe(args) -> MilesFullParameterManifestProbeSync:
    """Miles ``--external-policy-sync-path`` factory for the plan probe."""

    return MilesFullParameterManifestProbeSync(args)
