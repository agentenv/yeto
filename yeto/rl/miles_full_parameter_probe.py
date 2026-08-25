"""Non-destructive hardware probe for the Miles full-parameter boundary."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import torch

from .local_learner import ComponentIdentity, make_parameter_cut, parameter_values
from .miles_full_parameter import MilesFullParameterAdapter

_EVIDENCE_ENV = "YETO_FULL_PARAMETER_PROBE_EVIDENCE"
_MODEL_REVISION_ENV = "YETO_FULL_PARAMETER_MODEL_REVISION"
_CONFIG_HASH_ENV = "YETO_FULL_PARAMETER_CONFIG_HASH"
_FRAGMENT_COUNT_ENV = "YETO_FULL_PARAMETER_FRAGMENT_COUNT"
_CONVERSION_MANIFEST_ENV = "YETO_FULL_PARAMETER_CONVERSION_MANIFEST_SHA256"
_IMAGE_DIGEST_ENV = "YETO_MILES_IMAGE_DIGEST"
_YETO_SOURCE_ENV = "YETO_FULL_PARAMETER_YETO_SOURCE_ROOT"
_MILES_SOURCE_ENV = "YETO_FULL_PARAMETER_MILES_SOURCE_ROOT"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _positive_int_environment(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.isascii() or not raw.isdecimal():
        raise RuntimeError(f"{name} must be a positive decimal integer")
    value = int(raw)
    if value < 1:
        raise RuntimeError(f"{name} must be a positive decimal integer")
    return value


def _source_identity(root: Path) -> dict[str, object]:
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise RuntimeError("probe source root is not a real absolute directory")
    digest = sha256(b"yeto-probe-source-v1\0")
    count = 0
    total = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise RuntimeError("probe source tree contains a symlink")
        if not path.is_file():
            continue
        content_hash = sha256(path.read_bytes()).digest()
        size = path.stat().st_size
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(size.to_bytes(8, "little"))
        digest.update(content_hash)
        count += 1
        total += size
    if count < 1:
        raise RuntimeError("probe source tree is empty")
    return {"file_count": count, "bytes": total, "aggregate_sha256": digest.hexdigest()}


def _hardware_identity() -> dict[str, object]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [
        tuple(part.strip() for part in line.split(","))
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if len(rows) != 2 or any(len(row) != 3 for row in rows) or len(set(rows)) != 1:
        raise RuntimeError("probe container does not expose exactly two identical GPUs")
    name, driver, memory_mib = rows[0]
    if not memory_mib.isdecimal():
        raise RuntimeError("probe GPU memory field is malformed")
    return {
        "gpu_count": 2,
        "gpu_name": name,
        "driver_version": driver,
        "memory_mib_per_gpu": int(memory_mib),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _probe_target(adapter: MilesFullParameterAdapter, initial):
    """Advance one master scalar by one BF16-visible representable value."""

    values = parameter_values(adapter.layout, initial)
    first = adapter.layout.specs[0]
    value = values[first.wire_name]
    flat = value.reshape(-1)
    original = flat[0].clone()
    visible = original.to(dtype=torch.bfloat16)
    changed = torch.nextafter(
        visible,
        torch.full((), float("inf"), dtype=torch.bfloat16),
    ).to(dtype=torch.float32)
    if not torch.isfinite(changed).item():
        changed = torch.nextafter(
            visible,
            torch.full((), float("-inf"), dtype=torch.bfloat16),
        ).to(dtype=torch.float32)
    if not torch.isfinite(changed).item() or changed.item() == original.item():
        raise RuntimeError("cannot construct a finite full-parameter probe target")
    flat[0].copy_(changed)
    return make_parameter_cut(
        adapter.layout,
        policy_version=initial.policy_version + 1,
        values=values,
    )


def _write_private_json(path: Path, payload: dict[str, object]) -> None:
    parent = path.parent
    if (
        not path.is_absolute()
        or path.exists()
        or path.is_symlink()
        or not parent.is_dir()
        or parent.is_symlink()
    ):
        raise RuntimeError("full-parameter probe evidence path is not fresh and safe")
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary = Path(raw_path)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            written = stream.write(encoded)
            if written != len(encoded):
                raise RuntimeError("short write while recording probe evidence")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class MilesFullParameterProbeSync:
    """Exercise a real Miles export/apply round trip without starting rollout."""

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
        self.num_fragments = _positive_int_environment(_FRAGMENT_COUNT_ENV)
        self.conversion_manifest_sha256 = os.environ.get(_CONVERSION_MANIFEST_ENV, "")
        self.image_digest = os.environ.get(_IMAGE_DIGEST_ENV, "")
        if not _SHA256.fullmatch(self.conversion_manifest_sha256):
            raise RuntimeError("conversion manifest hash is malformed")
        if not _IMAGE_DIGEST.fullmatch(self.image_digest):
            raise RuntimeError("Miles image digest is malformed")
        self.yeto_source_root = Path(os.environ.get(_YETO_SOURCE_ENV, ""))
        self.miles_source_root = Path(os.environ.get(_MILES_SOURCE_ENV, ""))
        self.evidence: dict[str, object] | None = None

    async def initialize(self, *, actor_model, rollout_manager) -> None:
        del rollout_manager
        start = self.args.start_rollout_id
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise RuntimeError("Miles did not initialize a valid policy version")
        if self.args.num_rollout != start + 1:
            raise RuntimeError(
                "probe requires exactly one configured scheduler rollout"
            )
        yeto_source = _source_identity(self.yeto_source_root)
        miles_source = _source_identity(self.miles_source_root)
        hardware = _hardware_identity()
        adapter, initial = await MilesFullParameterAdapter.capture_initial(
            actor_model,
            policy_version=start,
            algorithm="grpo",
            components=(self.component,),
            num_fragments=self.num_fragments,
        )
        target = _probe_target(adapter, initial)
        changed_fragments = sum(
            left.payload_hash != right.payload_hash
            for left, right in zip(initial.fragments, target.fragments, strict=True)
        )
        if target.policy_hash == initial.policy_hash or changed_fragments != 1:
            raise RuntimeError("full-parameter probe target did not change")
        applied_parameters = await adapter.apply(actor_model, target)
        observed = await adapter.capture(
            actor_model,
            policy_version=target.policy_version,
        )
        if observed.policy_hash != target.policy_hash:
            raise RuntimeError("full-parameter probe round trip changed the target")
        self.args.start_rollout_id = target.policy_version
        self.evidence = {
            "schema": "yeto-miles-full-parameter-probe-v1",
            "algorithm": adapter.layout.algorithm,
            "role": self.component.role,
            "model_revision": self.component.model_revision,
            "model_config_sha256": self.component.config_hash,
            "conversion_manifest_sha256": self.conversion_manifest_sha256,
            "miles_image_digest": self.image_digest,
            "observed_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "yeto_source": yeto_source,
            "miles_source": miles_source,
            "hardware": hardware,
            "initial_policy_version": initial.policy_version,
            "target_policy_version": target.policy_version,
            "initial_policy_hash": initial.policy_hash,
            "target_policy_hash": target.policy_hash,
            "observed_policy_hash": observed.policy_hash,
            "parameter_layout_hash": adapter.layout.layout_hash,
            "parameter_tensor_count": adapter.expected_parameter_tensor_count,
            "applied_parameter_tensor_count": applied_parameters,
            "parameter_scalar_count": adapter.expected_parameter_scalar_count,
            "fragment_count": adapter.layout.fragments.num_fragments,
            "changed_fragment_count": changed_fragments,
            "changed_scalar_count": 1,
            "master_round_trip_exact": True,
            "model_master_round_trip_verified": True,
        }

    async def after_local_train(self, **_kwargs) -> bool:
        raise RuntimeError("full-parameter probe unexpectedly entered the train loop")

    async def finalize(self) -> None:
        if self.evidence is None:
            raise RuntimeError("full-parameter probe did not complete")
        _write_private_json(self.evidence_path, self.evidence)


def create_full_parameter_probe(args) -> MilesFullParameterProbeSync:
    """Miles ``--external-policy-sync-path`` factory for the hardware gate."""

    return MilesFullParameterProbeSync(args)
