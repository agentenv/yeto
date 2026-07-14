"""Passive, bounded optimizer-state capture for causal offline replay.

The learner owns all timing decisions.  This module only copies tensors at
explicit hook boundaries and writes integrity-checked artifacts.  In
particular it never calls an optimizer method, changes a parameter/gradient,
or consumes random numbers.

Schema v1 supports two complementary records:

``adamw_first_gradient``
    The raw AdamW moments and step clock for every tensor in one fragment,
    plus that fragment's first post-broadcast, post-clip gradient.  The hook
    must run immediately before ``optimizer.step()``.

``richardson_window``
    Full-f32 fragment states at the window anchor, exactly H/2 completed
    optimizer steps, and exactly H completed optimizer steps.  The hook must
    run immediately after the learner advances its local step counter and
    before it applies broadcasts.

Capture is intentionally not wired to environment variables.  Callers must
construct :class:`OptimizerStateCapture` explicitly, which keeps the normal
learner path inert when the corresponding CLI option is absent.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from .bounded_background_writer import (
    BackgroundWriterFailed,
    BoundedBackgroundWriter,
    WriteItem,
)
from .fragments import FragmentLayout


SCHEMA_NAME = "yeto.optimizer-state-capture"
SCHEMA_VERSION = 1
BACKGROUND_PUBLICATION_SCHEMA = "yeto.optimizer-state-capture-publication"
BACKGROUND_PUBLICATION_SCHEMA_VERSION = 1
BACKGROUND_TRAILER_BYTES = 8


class CaptureIntegrityError(ValueError):
    """An artifact or its checksum does not match the capture schema."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _digest_value(digest: "hashlib._Hash", value: Any) -> None:
    """Hash nested capture data without serializing or changing tensor dtype."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(_json_bytes(list(tensor.shape)))
        # Viewing bytes works for every torch dtype, including bfloat16.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            _digest_value(digest, str(key))
            _digest_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _digest_value(digest, item)
    elif value is None:
        digest.update(b"none\0")
    elif isinstance(value, bool):
        digest.update(b"bool\0" + (b"1" if value else b"0"))
    elif isinstance(value, int):
        digest.update(b"int\0" + str(value).encode("ascii") + b"\0")
    elif isinstance(value, float):
        digest.update(b"float\0" + value.hex().encode("ascii") + b"\0")
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        digest.update(b"str\0" + str(len(raw)).encode("ascii") + b"\0" + raw)
    else:
        raise TypeError(f"unsupported capture value type: {type(value).__name__}")


def capture_value_sha256(value: Any) -> str:
    """Return the schema's stable structural digest for nested capture data."""

    digest = hashlib.sha256()
    _digest_value(digest, value)
    return digest.hexdigest()


def layout_sha256(layout: FragmentLayout) -> str:
    description = [
        {
            "fragment_id": fid,
            "merge_mode": fragment.merge_mode,
            "tensors": [[name, numel] for name, numel in fragment.tensors],
        }
        for fid, fragment in enumerate(layout.fragments)
    ]
    return hashlib.sha256(_json_bytes(description)).hexdigest()


def _cpu_f32(tensor: torch.Tensor) -> torch.Tensor:
    # A real clone is essential: capture must not alias live parameters,
    # gradients, or optimizer state.
    return tensor.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()


def _cpu_exact(tensor: torch.Tensor) -> torch.Tensor:
    """Clone without a dtype conversion (used for exact optimizer restore)."""

    return tensor.detach().to(device="cpu", copy=True).contiguous()


def _clone_state_value(value: Any) -> Any:
    """Clone the tensor/plain-data subset accepted by torch state dicts."""

    if isinstance(value, torch.Tensor):
        return _cpu_exact(value)
    if isinstance(value, Mapping):
        return {str(key): _clone_state_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clone_state_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"capture cannot clone state value of type {type(value).__name__}")


def _tensor_storage_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_storage_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_storage_bytes(item) for item in value)
    return 0


def _f32_wire_sha256(flat: torch.Tensor) -> str:
    """Hash the exact bytes emitted by ``pack_flat(..., DTYPE_F32)``."""

    if flat.dtype != torch.float32:
        raise TypeError("f32 wire identity requires a float32 endpoint")
    wire = flat.detach().contiguous().cpu()
    return hashlib.sha256(wire.view(torch.uint8).numpy().tobytes()).hexdigest()


def _atomic_replace_bytes(path: Path, raw: bytes, serial: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{serial}")
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fsync_directory(path: Path) -> None:
    # Directory fsync is supported on the Linux training hosts.  Some local
    # filesystems reject it; the file-level atomicity remains useful there.
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


class OptimizerStateCapture:
    """Capture exact replay inputs at caller-specified learner boundaries.

    ``max_bytes`` includes finalized ``.pt`` files and their checksum
    sidecars.  The small run manifest and its checksum are excluded.  Pending
    midpoint tensors are independently bounded to at most ``max_bytes`` raw
    bytes and ``max_midpoint_windows`` windows.  One temporary serialized
    artifact can transiently exist before the exact on-disk size is known.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        params: Mapping[str, torch.Tensor],
        layout: FragmentLayout,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        learner_id: int,
        rank: int,
        every: int = 1,
        max_hmc_events: int = 32,
        max_midpoint_windows: int = 32,
        max_bytes: int = 4 * 1024**3,
        max_manifest_drops: int = 256,
        background_writer: bool = False,
        background_writer_max_items: int = 4,
        background_writer_max_bytes: int = 4 * 1024**3,
    ) -> None:
        if every < 1:
            raise ValueError("capture cadence 'every' must be >= 1")
        if max_hmc_events < 0 or max_midpoint_windows < 0:
            raise ValueError("capture event limits must be >= 0")
        if max_bytes < 0:
            raise ValueError("capture max_bytes must be >= 0")
        if background_writer and background_writer_max_items < 1:
            raise ValueError("background writer max_items must be positive")
        if background_writer and background_writer_max_bytes < 1:
            raise ValueError("background writer max_bytes must be positive")
        if type(optimizer) is not torch.optim.AdamW:
            raise TypeError("optimizer-state capture v1 requires torch.optim.AdamW")
        if scaler is not None:
            raise TypeError(
                "optimizer-state capture v1 requires native no-scaler AdamW"
            )

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError(
                f"capture directory must be empty to avoid artifact/manifest overwrite: "
                f"{self.directory}"
            )
        self.params = dict(params)
        self.layout = layout
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.learner_id = int(learner_id)
        self.rank = int(rank)
        self.every = int(every)
        self.max_hmc_events = int(max_hmc_events)
        self.max_midpoint_windows = int(max_midpoint_windows)
        self.max_bytes = int(max_bytes)
        self.max_manifest_drops = int(max_manifest_drops)
        self.background_writer_enabled = bool(background_writer)
        self.background_writer_max_items = int(background_writer_max_items)
        self.background_writer_max_bytes = int(background_writer_max_bytes)
        self.layout_digest = layout_sha256(layout)

        layout_names = set(layout.tensor_names())
        missing = layout_names.difference(self.params)
        if missing:
            raise ValueError(
                f"capture layout names missing from params: {sorted(missing)!r}"
            )
        non_f32 = {
            name: str(self.params[name].dtype)
            for name in layout_names
            if self.params[name].dtype != torch.float32
        }
        if non_f32:
            raise TypeError(
                "capture v1 requires fp32 LoRA optimizer parameters; refusing a silent cast: "
                f"{non_f32!r}"
            )

        self._param_group_by_id: dict[int, int] = {}
        for group_index, group in enumerate(optimizer.param_groups):
            for param in group["params"]:
                self._param_group_by_id[id(param)] = group_index
        ungrouped = [
            name
            for name in layout_names
            if id(self.params[name]) not in self._param_group_by_id
        ]
        if ungrouped:
            raise ValueError(
                f"capture params absent from optimizer groups: {sorted(ungrouped)!r}"
            )

        self._serial = 0
        self._broadcast_seen = 0
        self._window_reset_seen = 0
        self._hmc_admitted = 0
        self._midpoint_admitted = 0
        self._artifact_bytes = 0
        self._artifact_reserved_bytes = 0
        self._publication_lock = threading.Lock()
        self._pending_raw_bytes = 0
        self._pending_hmc: dict[int, dict[str, Any]] = {}
        self._pending_midpoint: dict[int, dict[str, Any]] = {}
        self._active_window_uuid_by_fragment: dict[int, str] = {}
        self._window_lifecycles: dict[str, dict[str, Any]] = {}
        self._push_retries: dict[str, dict[str, Any]] = {}
        self._push_attempts_by_serial: dict[int, dict[str, Any]] = {}
        self._push_attempt_serial = 0
        self._artifacts: list[dict[str, Any]] = []
        self._drop_counts: Counter[str] = Counter()
        self._recent_drops: list[dict[str, Any]] = []
        self._closed = False
        self._background_writer: BoundedBackgroundWriter | None = None
        self._background_writer_stats: dict[str, Any] | None = None
        if self.background_writer_enabled:
            self._background_writer = BoundedBackgroundWriter(
                self._publish_background_item,
                max_items=self.background_writer_max_items,
                max_bytes=self.background_writer_max_bytes,
                thread_name=(
                    f"optimizer-capture-writer-l{self.learner_id}-r{self.rank}"
                ),
            )
        try:
            self._write_manifest()
        except BaseException:
            if self._background_writer is not None:
                self._background_writer.close()
            raise

    @staticmethod
    def _group_config(group: Mapping[str, Any]) -> dict[str, Any]:
        """Only parameters that determine AdamW's mathematical update."""

        config: dict[str, Any] = {}
        for key in (
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
        ):
            if key not in group:
                continue
            value = group[key]
            if isinstance(value, tuple):
                value = list(value)
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(
                        f"unsupported non-scalar optimizer group field {key}"
                    )
                value = value.detach().cpu().item()
            config[key] = value
        return config

    def _next_serial(self) -> int:
        self._serial += 1
        return self._serial

    def _selected(self, ordinal: int) -> bool:
        return (ordinal - 1) % self.every == 0

    def _observe_background_writer(self) -> None:
        """Surface asynchronous publication failure at a capture boundary."""

        writer = self._background_writer
        if writer is None:
            return
        try:
            stats = writer.check()
        except BackgroundWriterFailed:
            # FAILED already stops the worker. close() joins it and re-raises
            # the same first cause, giving the persisted diagnostic a stable
            # worker_alive=false terminal snapshot.
            try:
                writer.close()
            except BackgroundWriterFailed as failure:
                self._background_writer_stats = writer.snapshot().as_json()
                with self._publication_lock:
                    self._artifact_reserved_bytes = 0
                try:
                    self._write_manifest_raw()
                except BaseException:
                    # Publication failure is the first error. A secondary
                    # diagnostic-manifest failure must never replace it.
                    pass
                raise failure
            raise AssertionError("failed writer close unexpectedly succeeded")
        else:
            self._background_writer_stats = stats.as_json()

    def _record_drop(self, reason: str, **identity: Any) -> None:
        self._drop_counts[reason] += 1
        if len(self._recent_drops) < self.max_manifest_drops:
            self._recent_drops.append({"reason": reason, **identity})
        self._write_manifest()

    def _common_identity(
        self,
        fragment_id: int,
        version: int,
        reset_local_step: int,
        reset_tokens: int,
        reset_reason: str,
    ) -> dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "rank": self.rank,
            "fragment_id": int(fragment_id),
            "fragment_version": int(version),
            "reset_local_step": int(reset_local_step),
            "reset_tokens": int(reset_tokens),
            "reset_reason": reset_reason,
            "layout_sha256": self.layout_digest,
        }

    def _make_window_uuid(
        self, identity: Mapping[str, Any], window_ordinal: int, window_steps: int
    ) -> str:
        immutable = {
            **identity,
            "window_ordinal": int(window_ordinal),
            "window_steps": int(window_steps),
            "capture_directory": str(self.directory.resolve()),
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
        }
        digest = hashlib.sha256(_json_bytes(immutable)).digest()
        return str(uuid.UUID(bytes=digest[:16], version=5))

    def _supersede_unpushed_window(
        self, fragment_id: int, *, next_version: int, next_local_step: int
    ) -> None:
        window_uuid = self._active_window_uuid_by_fragment.pop(fragment_id, None)
        if window_uuid is None:
            return
        lifecycle = self._window_lifecycles[window_uuid]
        if lifecycle["status"] == "pushed":
            return
        lifecycle["status"] = "superseded_unpushed"
        lifecycle["superseded_by_version"] = int(next_version)
        lifecycle["superseded_at_local_step"] = int(next_local_step)
        self._record_drop(
            "window_superseded_unpushed",
            window_uuid=window_uuid,
            fragment_id=fragment_id,
            fragment_version=lifecycle["fragment_version"],
            superseded_by_version=int(next_version),
            superseded_at_local_step=int(next_local_step),
        )

    def _fragment_state(self, fragment_id: int) -> torch.Tensor:
        for name, _ in self.layout.fragments[fragment_id].tensors:
            if self.params[name].dtype != torch.float32:
                raise TypeError(f"capture parameter {name!r} changed away from fp32")
        fragment = self.layout.fragments[fragment_id]
        return torch.cat(
            [_cpu_f32(self.params[name]).reshape(-1) for name, _ in fragment.tensors]
        )

    def _fragment_optimizer_state(self, fragment_id: int) -> dict[str, Any]:
        """Exact raw AdamW state plus explicit mathematical zero-state flags."""

        result: dict[str, Any] = {}
        for name, numel in self.layout.fragments[fragment_id].tensors:
            param = self.params[name]
            raw = self.optimizer.state.get(param)
            group_index = self._param_group_by_id[id(param)]
            if raw:
                raw_state = _clone_state_value(raw)
                raw_step = raw.get("step", 0)
                if isinstance(raw_step, torch.Tensor):
                    raw_step = raw_step.detach().cpu().item()
                initialized = "exp_avg" in raw and "exp_avg_sq" in raw
            else:
                raw_state = {}
                raw_step = 0
                initialized = False
            result[name] = {
                "shape": list(param.shape),
                "numel": int(numel),
                "parameter_dtype": str(param.dtype),
                "optimizer_group": group_index,
                "optimizer_group_config": self._group_config(
                    self.optimizer.param_groups[group_index]
                ),
                "optimizer_state_initialized": initialized,
                "optimizer_step": int(raw_step),
                "raw_optimizer_state": raw_state,
            }
        return result

    def _auxiliary_optimizer_state(self) -> dict[str, Any]:
        """Exact available scheduler/scaler state, explicitly marking absence."""

        return {
            "scheduler_present": self.scheduler is not None,
            "scheduler_state": (
                _clone_state_value(self.scheduler.state_dict())
                if self.scheduler is not None
                else None
            ),
            "scaler_present": self.scaler is not None,
            "scaler_state": (
                _clone_state_value(self.scaler.state_dict())
                if self.scaler is not None
                else None
            ),
        }

    def _fragment_snapshot(self, fragment_id: int) -> dict[str, Any]:
        return {
            "tensor_order": [
                name for name, _ in self.layout.fragments[fragment_id].tensors
            ],
            "parameters_f32": self._fragment_state(fragment_id),
            "optimizer": self._fragment_optimizer_state(fragment_id),
            "auxiliary_optimizer": self._auxiliary_optimizer_state(),
        }

    def note_broadcast(
        self,
        fragment_id: int,
        version: int,
        *,
        local_step: int,
        tokens_total: int,
        window_steps: int,
    ) -> str | None:
        """Arm first-gradient capture and reset the Richardson window.

        Call after the received fragment has been fully applied and its
        learner counters have been reset.
        """

        self._observe_background_writer()
        if self._closed:
            raise RuntimeError("capture is closed")
        self._validate_fragment_id(fragment_id)
        self._broadcast_seen += 1
        identity = self._common_identity(
            fragment_id, version, local_step, tokens_total, "broadcast"
        )

        old = self._pending_hmc.pop(fragment_id, None)
        if old is not None:
            self._record_drop("hmc_superseded_by_broadcast", **old["identity"])
        if self._selected(self._broadcast_seen):
            if self._hmc_admitted >= self.max_hmc_events:
                self._record_drop("hmc_event_limit", **identity)
            else:
                self._hmc_admitted += 1
                self._pending_hmc[fragment_id] = {
                    "identity": identity,
                    "broadcast_ordinal": self._broadcast_seen,
                }

        window_uuid = self.note_window_reset(
            fragment_id,
            version,
            local_step=local_step,
            tokens_total=tokens_total,
            window_steps=window_steps,
            reason="broadcast",
        )
        if window_uuid is not None and fragment_id in self._pending_hmc:
            self._pending_hmc[fragment_id]["identity"]["window_uuid"] = window_uuid
        return window_uuid

    def note_window_reset(
        self,
        fragment_id: int,
        version: int,
        *,
        local_step: int,
        tokens_total: int,
        window_steps: int,
        reason: str,
    ) -> str | None:
        """Start a candidate H/2,H window from the current fragment state."""

        self._observe_background_writer()
        if self._closed:
            raise RuntimeError("capture is closed")
        self._validate_fragment_id(fragment_id)
        self._window_reset_seen += 1
        identity = self._common_identity(
            fragment_id, version, local_step, tokens_total, reason
        )

        self._supersede_unpushed_window(
            fragment_id, next_version=version, next_local_step=local_step
        )

        old = self._pending_midpoint.pop(fragment_id, None)
        if old is not None:
            self._pending_raw_bytes -= old["raw_bytes"]
            self._record_drop("midpoint_superseded_by_reset", **old["identity"])

        if not self._selected(self._window_reset_seen):
            return None
        if self._midpoint_admitted >= self.max_midpoint_windows:
            self._record_drop("midpoint_window_limit", **identity)
            return None
        if window_steps < 2 or window_steps % 2:
            self._record_drop(
                "midpoint_requires_even_h_ge_2", window_steps=window_steps, **identity
            )
            return None

        window_uuid = self._make_window_uuid(
            identity, self._window_reset_seen, window_steps
        )
        identity["window_uuid"] = window_uuid

        anchor = self._fragment_snapshot(fragment_id)
        zero_decay = torch.zeros_like(anchor["parameters_f32"])
        raw_bytes = _tensor_storage_bytes(anchor) + 2 * _tensor_storage_bytes(
            zero_decay
        )
        if self._pending_raw_bytes + raw_bytes > self.max_bytes:
            self._record_drop("midpoint_pending_memory_limit", **identity)
            return None
        self._midpoint_admitted += 1
        self._pending_raw_bytes += raw_bytes
        self._pending_midpoint[fragment_id] = {
            "identity": identity,
            "window_ordinal": self._window_reset_seen,
            "window_steps": int(window_steps),
            "anchor": anchor,
            "midpoint": None,
            "step_history": [],
            "decoupled_decay_first_f32": zero_decay,
            "decoupled_decay_second_f32": zero_decay.clone(),
            "raw_bytes": raw_bytes,
        }
        self._active_window_uuid_by_fragment[fragment_id] = window_uuid
        self._window_lifecycles[window_uuid] = {
            "window_uuid": window_uuid,
            "status": "collecting",
            "fragment_id": int(fragment_id),
            "fragment_version": int(version),
            "reset_local_step": int(local_step),
            "reset_tokens": int(tokens_total),
            "window_steps": int(window_steps),
            "push_attempts": 0,
            "enqueued_pushes": 0,
        }
        self._write_manifest()
        return window_uuid

    def capture_first_post_broadcast_gradients(
        self,
        *,
        local_step_before_update: int,
        tokens_total: int,
        clip_total_norm: torch.Tensor | float | None = None,
        clip_max_norm: float | None = 1.0,
    ) -> None:
        """Record step context and save pending first gradients before AdamW."""

        self._observe_background_writer()
        if self._closed:
            raise RuntimeError("capture is closed")
        runtime_groups = [
            self._group_config(group) for group in self.optimizer.param_groups
        ]
        clip_applied = clip_max_norm is not None and clip_total_norm is not None
        if clip_applied and isinstance(clip_total_norm, torch.Tensor):
            clip_coefficient: torch.Tensor | float | None = _cpu_exact(
                torch.clamp(
                    float(clip_max_norm) / (clip_total_norm.detach() + 1e-6), max=1.0
                )
            )
        elif clip_applied:
            clip_coefficient = min(
                1.0, float(clip_max_norm) / (float(clip_total_norm) + 1e-6)
            )
        else:
            clip_coefficient = None
        clip_record = {
            "applied": clip_applied,
            "max_norm": clip_max_norm,
            "total_norm_before_clip": (
                _cpu_exact(clip_total_norm)
                if isinstance(clip_total_norm, torch.Tensor)
                else clip_total_norm
            ),
            "coefficient": clip_coefficient,
            "coefficient_formula": "clamp(max_norm / (total_norm + 1e-6), max=1)",
        }
        for fragment_id in list(self._pending_midpoint):
            pending = self._pending_midpoint[fragment_id]
            elapsed_after_step = (
                local_step_before_update - pending["identity"]["reset_local_step"] + 1
            )
            decay_parts = []
            for name, _ in self.layout.fragments[fragment_id].tensors:
                param = self.params[name]
                group_index = self._param_group_by_id[id(param)]
                group = runtime_groups[group_index]
                coefficient = float(group["lr"]) * float(group["weight_decay"])
                decay_parts.append(_cpu_f32(param.detach() * coefficient).reshape(-1))
            decay_vector = torch.cat(decay_parts)
            decay_half = (
                "decoupled_decay_first_f32"
                if elapsed_after_step <= pending["window_steps"] // 2
                else "decoupled_decay_second_f32"
            )
            # Capture-owned CPU tensors only; factual params/optimizer remain untouched.
            pending[decay_half].add_(decay_vector)
            step_record = {
                "local_step_before_update": int(local_step_before_update),
                "tokens_total_before_update": int(tokens_total),
                "optimizer_groups": _clone_state_value(runtime_groups),
                "clip": _clone_state_value(clip_record),
                "gradient_scaler_present": self.scaler is not None,
                "accepted_optimizer_step": True,
            }
            added = _tensor_storage_bytes(step_record)
            if self._pending_raw_bytes + added > self.max_bytes:
                self._drop_pending_midpoint(
                    fragment_id, "midpoint_pending_memory_limit"
                )
                continue
            pending["step_history"].append(step_record)
            pending["raw_bytes"] += added
            self._pending_raw_bytes += added
        for fragment_id in list(self._pending_hmc):
            pending = self._pending_hmc.pop(fragment_id)
            fragment = self.layout.fragments[fragment_id]
            tensors: dict[str, dict[str, Any]] = {}
            for name, numel in fragment.tensors:
                param = self.params[name]
                state = self.optimizer.state.get(param, {})
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                initialized = exp_avg is not None and exp_avg_sq is not None
                if initialized:
                    moment1 = _cpu_exact(exp_avg)
                    moment2 = _cpu_exact(exp_avg_sq)
                else:
                    moment1 = torch.zeros_like(param, device="cpu", dtype=torch.float32)
                    moment2 = torch.zeros_like(param, device="cpu", dtype=torch.float32)
                raw_step = state.get("step", 0)
                if isinstance(raw_step, torch.Tensor):
                    raw_step = raw_step.detach().cpu().item()
                gradient_present = param.grad is not None
                gradient = (
                    _cpu_exact(param.grad)
                    if gradient_present
                    else torch.zeros_like(param, device="cpu", dtype=torch.float32)
                )
                max_exp_avg_sq = state.get("max_exp_avg_sq")
                group_index = self._param_group_by_id[id(param)]
                group_config = self._group_config(
                    self.optimizer.param_groups[group_index]
                )
                tensors[name] = {
                    "shape": list(param.shape),
                    "numel": int(numel),
                    "parameter_dtype": str(param.dtype),
                    "optimizer_state_initialized": bool(initialized),
                    "optimizer_step": int(raw_step),
                    "optimizer_group": group_index,
                    "optimizer_group_config": dict(group_config),
                    "exp_avg": moment1,
                    "exp_avg_sq": moment2,
                    "max_exp_avg_sq": (
                        _cpu_exact(max_exp_avg_sq)
                        if max_exp_avg_sq is not None
                        else None
                    ),
                    "gradient_present": gradient_present,
                    "first_clipped_gradient": gradient,
                }
            metadata = {
                **pending["identity"],
                "broadcast_ordinal": pending["broadcast_ordinal"],
                "capture_local_step_before_update": int(local_step_before_update),
                "capture_tokens_total": int(tokens_total),
                "steps_since_reset_before_update": int(
                    local_step_before_update - pending["identity"]["reset_local_step"]
                ),
                "gradient_boundary": "post_allreduce_post_clip_pre_optimizer_step",
            }
            self._write_artifact(
                "adamw_first_gradient",
                metadata,
                {"clip": _clone_state_value(clip_record), "tensors": tensors},
            )

    def after_optimizer_step(
        self,
        *,
        local_step: int,
        tokens_total: int,
        current_window_steps: int,
    ) -> None:
        """Capture exact H/2 and H states before the step-boundary sync."""

        self._observe_background_writer()
        if self._closed:
            raise RuntimeError("capture is closed")
        for fragment_id in list(self._pending_midpoint):
            pending = self._pending_midpoint[fragment_id]
            if pending["window_steps"] != current_window_steps:
                self._drop_pending_midpoint(
                    fragment_id, "midpoint_window_schedule_changed"
                )
                continue
            elapsed = local_step - pending["identity"]["reset_local_step"]
            half = pending["window_steps"] // 2
            if elapsed == half and pending["midpoint"] is None:
                midpoint = self._fragment_snapshot(fragment_id)
                added = _tensor_storage_bytes(midpoint)
                if self._pending_raw_bytes + added > self.max_bytes:
                    self._drop_pending_midpoint(
                        fragment_id, "midpoint_pending_memory_limit"
                    )
                    continue
                pending["midpoint"] = midpoint
                pending["midpoint_local_step"] = int(local_step)
                pending["midpoint_tokens_total"] = int(tokens_total)
                pending["raw_bytes"] += added
                self._pending_raw_bytes += added
            if elapsed == pending["window_steps"]:
                pending = self._pending_midpoint.pop(fragment_id)
                self._pending_raw_bytes -= pending["raw_bytes"]
                if pending["midpoint"] is None:
                    self._record_drop("midpoint_boundary_missed", **pending["identity"])
                    continue
                endpoint = self._fragment_snapshot(fragment_id)
                if len(pending["step_history"]) != pending["window_steps"]:
                    self._record_drop(
                        "midpoint_step_history_incomplete",
                        recorded_steps=len(pending["step_history"]),
                        **pending["identity"],
                    )
                    continue
                half = pending["window_steps"] // 2
                group_count = len(self.optimizer.param_groups)
                first_lr_mass = [
                    sum(
                        float(row["optimizer_groups"][group]["lr"])
                        for row in pending["step_history"][:half]
                    )
                    for group in range(group_count)
                ]
                second_lr_mass = [
                    sum(
                        float(row["optimizer_groups"][group]["lr"])
                        for row in pending["step_history"][half:]
                    )
                    for group in range(group_count)
                ]
                metadata = {
                    **pending["identity"],
                    "window_ordinal": pending["window_ordinal"],
                    "window_steps": pending["window_steps"],
                    "midpoint_local_step": pending["midpoint_local_step"],
                    "midpoint_tokens_total": pending["midpoint_tokens_total"],
                    "endpoint_local_step": int(local_step),
                    "endpoint_tokens_total": int(tokens_total),
                    "accepted_midpoint_steps": half,
                    "accepted_endpoint_steps": pending["window_steps"],
                    "state_boundary": "post_optimizer_step_pre_broadcast",
                }
                wrote = self._write_artifact(
                    "richardson_window",
                    metadata,
                    {
                        "anchor": pending["anchor"],
                        "midpoint": pending["midpoint"],
                        "endpoint": endpoint,
                        "step_history": pending["step_history"],
                        "lr_mass_first_by_group": first_lr_mass,
                        "lr_mass_second_by_group": second_lr_mass,
                        "decoupled_decay_first_f32": pending[
                            "decoupled_decay_first_f32"
                        ],
                        "decoupled_decay_second_f32": pending[
                            "decoupled_decay_second_f32"
                        ],
                    },
                )
                lifecycle = self._window_lifecycles[pending["identity"]["window_uuid"]]
                lifecycle.update(
                    {
                        "status": "ready" if wrote else "artifact_write_failed",
                        "endpoint_local_step": int(local_step),
                        "endpoint_tokens_total": int(tokens_total),
                        "c_steps": int(pending["window_steps"]),
                        "c_tokens": int(
                            tokens_total - pending["identity"]["reset_tokens"]
                        ),
                        "expected_f32_payload_sha256": _f32_wire_sha256(
                            endpoint["parameters_f32"]
                        ),
                    }
                )
                self._write_manifest()
            elif elapsed > pending["window_steps"]:
                self._drop_pending_midpoint(fragment_id, "midpoint_boundary_missed")

    def note_push(
        self,
        *,
        window_uuid: str,
        fragment_id: int,
        pull_global_step: int,
        base_version: int,
        local_step: int,
        c_steps: int,
        c_tokens: int,
        wire_codec: str,
        payload: bytes,
    ) -> dict[str, Any]:
        """Join one fixed endpoint to an attempted syncer push.

        Repeated calls with the same window/pull/base identity are immutable
        retries: they share ``retry_identity`` and receive increasing
        ``retry_ordinal`` values. A changed endpoint or payload fails closed.
        """

        self._observe_background_writer()
        if self._closed:
            raise RuntimeError("capture is closed")
        lifecycle = self._window_lifecycles.get(window_uuid)
        if lifecycle is None:
            raise CaptureIntegrityError(f"unknown capture window_uuid {window_uuid}")
        if lifecycle["status"] not in ("ready", "pushed"):
            raise CaptureIntegrityError(
                f"capture window {window_uuid} is not push-ready: {lifecycle['status']}"
            )
        if wire_codec != "f32":
            raise CaptureIntegrityError("capture push join requires f32 wire codec")
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        expected = {
            "fragment_id": lifecycle["fragment_id"],
            "base_version": lifecycle["fragment_version"],
            "local_step": lifecycle["endpoint_local_step"],
            "c_steps": lifecycle["c_steps"],
            "c_tokens": lifecycle["c_tokens"],
            "payload_sha256": lifecycle["expected_f32_payload_sha256"],
        }
        actual = {
            "fragment_id": int(fragment_id),
            "base_version": int(base_version),
            "local_step": int(local_step),
            "c_steps": int(c_steps),
            "c_tokens": int(c_tokens),
            "payload_sha256": payload_sha256,
        }
        if actual != expected:
            raise CaptureIntegrityError(
                f"push does not match immutable capture endpoint: expected={expected!r} "
                f"actual={actual!r}"
            )

        retry_key = {
            "window_uuid": window_uuid,
            "pull_global_step": int(pull_global_step),
            "base_version": int(base_version),
            "learner_id": self.learner_id,
            "rank": self.rank,
            "fragment_id": int(fragment_id),
        }
        retry_identity = hashlib.sha256(_json_bytes(retry_key)).hexdigest()
        immutable_candidate = {
            **retry_key,
            "local_step": int(local_step),
            "c_steps": int(c_steps),
            "c_tokens": int(c_tokens),
            "wire_codec": wire_codec,
            "payload_sha256": payload_sha256,
        }
        retry = self._push_retries.get(retry_identity)
        if retry is None:
            retry = {"candidate": immutable_candidate, "attempts": 0}
            self._push_retries[retry_identity] = retry
        elif retry["candidate"] != immutable_candidate:
            raise CaptureIntegrityError(
                f"retry identity {retry_identity} changed immutable candidate fields"
            )
        retry["attempts"] += 1
        self._push_attempt_serial += 1
        record = {
            **immutable_candidate,
            "fragment_version": int(base_version),
            "retry_identity": retry_identity,
            "retry_ordinal": int(retry["attempts"]),
            "attempt_serial": int(self._push_attempt_serial),
            "push_site": "immediately_before_client.push_fragment",
        }
        if not self._write_artifact("push_candidate", record, {}):
            raise RuntimeError(
                "capture byte budget rejected an exact push-candidate record"
            )
        lifecycle["push_attempts"] += 1
        lifecycle["last_retry_identity"] = retry_identity
        lifecycle["last_pull_global_step"] = int(pull_global_step)
        self._push_attempts_by_serial[self._push_attempt_serial] = {
            "window_uuid": window_uuid,
            "retry_identity": retry_identity,
            "retry_ordinal": int(retry["attempts"]),
            "enqueued": False,
        }
        self._write_manifest()
        return record

    def note_push_enqueued(self, attempt_serial: int) -> None:
        """Finalize a prepared candidate only after the client accepted it."""

        self._observe_background_writer()
        attempt = self._push_attempts_by_serial.get(int(attempt_serial))
        if attempt is None:
            raise CaptureIntegrityError(f"unknown push attempt serial {attempt_serial}")
        if attempt["enqueued"]:
            raise CaptureIntegrityError(
                f"push attempt serial {attempt_serial} finalized twice"
            )
        attempt["enqueued"] = True
        lifecycle = self._window_lifecycles[attempt["window_uuid"]]
        lifecycle["status"] = "pushed"
        lifecycle["enqueued_pushes"] += 1
        lifecycle["last_enqueued_attempt_serial"] = int(attempt_serial)
        self._write_manifest()

    def close(self) -> None:
        """Record incomplete candidates and finalize the manifest."""

        self._observe_background_writer()
        if self._closed:
            return
        for fragment_id, pending in list(self._pending_hmc.items()):
            self._record_drop("hmc_incomplete_at_close", **pending["identity"])
            del self._pending_hmc[fragment_id]
        for fragment_id in list(self._pending_midpoint):
            self._drop_pending_midpoint(fragment_id, "midpoint_incomplete_at_close")
        for window_uuid, lifecycle in self._window_lifecycles.items():
            if lifecycle["status"] in (
                "pushed",
                "superseded_unpushed",
                "closed_unpushed",
            ):
                continue
            lifecycle["status"] = "closed_unpushed"
            self._record_drop(
                "window_unpushed_at_close",
                window_uuid=window_uuid,
                fragment_id=lifecycle["fragment_id"],
                fragment_version=lifecycle["fragment_version"],
            )
        if self._background_writer is not None:
            try:
                stats = self._background_writer.close()
            except BackgroundWriterFailed as failure:
                self._background_writer_stats = (
                    self._background_writer.snapshot().as_json()
                )
                with self._publication_lock:
                    self._artifact_reserved_bytes = 0
                try:
                    self._write_manifest_raw()
                except BaseException:
                    pass
                raise failure
            self._background_writer_stats = stats.as_json()
            with self._publication_lock:
                if self._artifact_reserved_bytes != 0:
                    raise CaptureIntegrityError(
                        "background writer drained with outstanding artifact bytes"
                    )
        self._closed = True
        self._write_manifest()

    def _drop_pending_midpoint(self, fragment_id: int, reason: str) -> None:
        pending = self._pending_midpoint.pop(fragment_id)
        self._pending_raw_bytes -= pending["raw_bytes"]
        self._record_drop(reason, **pending["identity"])

    def _validate_fragment_id(self, fragment_id: int) -> None:
        if not 0 <= fragment_id < self.layout.num_fragments:
            raise IndexError(f"fragment_id {fragment_id} outside capture layout")

    def _write_artifact(
        self, kind: str, metadata: dict[str, Any], payload: dict[str, Any]
    ) -> bool:
        self._observe_background_writer()
        serial = self._next_serial()
        envelope = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "metadata": metadata,
            "payload_sha256": capture_value_sha256(payload),
            "payload": payload,
        }
        name = (
            f"{serial:06d}-{kind}-l{self.learner_id}-r{self.rank}-"
            f"f{metadata['fragment_id']}-v{metadata['fragment_version']}.pt"
        )
        if self._background_writer is not None:
            return self._enqueue_background_artifact(
                serial=serial,
                name=name,
                kind=kind,
                metadata=metadata,
                envelope=envelope,
            )

        path = self.directory / name
        temporary = path.with_name(f".{name}.tmp-{os.getpid()}-{serial}")
        with temporary.open("wb") as handle:
            torch.save(envelope, handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _file_sha256(temporary)
        sidecar_raw = f"{digest}  {name}\n".encode("ascii")
        required = temporary.stat().st_size + len(sidecar_raw)
        if self._artifact_bytes + required > self.max_bytes:
            temporary.unlink(missing_ok=True)
            self._record_drop(f"{kind}_disk_byte_limit", **metadata)
            return False

        os.replace(temporary, path)
        sidecar = path.with_suffix(path.suffix + ".sha256")
        _atomic_replace_bytes(sidecar, sidecar_raw, self._next_serial())
        _fsync_directory(self.directory)
        self._artifact_bytes += required
        self._artifacts.append(
            {
                "path": name,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "sidecar_bytes": len(sidecar_raw),
                "kind": kind,
                "fragment_id": metadata["fragment_id"],
                "fragment_version": metadata["fragment_version"],
            }
        )
        self._write_manifest()
        return True

    def _enqueue_background_artifact(
        self,
        *,
        serial: int,
        name: str,
        kind: str,
        metadata: Mapping[str, Any],
        envelope: Mapping[str, Any],
    ) -> bool:
        """Serialize to immutable bytes, reserve disk budget, then enqueue."""

        writer = self._background_writer
        if writer is None:
            raise AssertionError("background artifact enqueue without a writer")
        try:
            buffer = io.BytesIO()
            torch.save(dict(envelope), buffer)
            artifact_bytes = buffer.tell()
            view = buffer.getbuffer()
            try:
                digest = hashlib.sha256(view[:artifact_bytes]).hexdigest()
            finally:
                view.release()
            sidecar_raw = f"{digest}  {name}\n".encode("ascii")
            required = artifact_bytes + len(sidecar_raw)
            header = {
                "schema": BACKGROUND_PUBLICATION_SCHEMA,
                "schema_version": BACKGROUND_PUBLICATION_SCHEMA_VERSION,
                "serial": int(serial),
                "name": name,
                "sha256": digest,
                "artifact_bytes": artifact_bytes,
                "sidecar_bytes": len(sidecar_raw),
                "required_bytes": required,
                "kind": kind,
                "fragment_id": int(metadata["fragment_id"]),
                "fragment_version": int(metadata["fragment_version"]),
            }
            header_raw = _json_bytes(header)
            buffer.write(header_raw)
            buffer.write(len(header_raw).to_bytes(BACKGROUND_TRAILER_BYTES, "big"))
            immutable_job = buffer.getvalue()
        except BaseException:
            # A local serialization/ownership failure occurs before admission.
            # Stop the otherwise-idleable non-daemon worker before propagating.
            self._stop_background_writer_after_local_error()
            raise

        with self._publication_lock:
            fits_disk_budget = (
                self._artifact_bytes + self._artifact_reserved_bytes + required
                <= self.max_bytes
            )
            if fits_disk_budget:
                self._artifact_reserved_bytes += required
        if not fits_disk_budget:
            self._record_drop(f"{kind}_disk_byte_limit", **metadata)
            return False

        try:
            writer.submit(immutable_job, reservation_bytes=len(immutable_job))
        except BaseException:
            with self._publication_lock:
                self._artifact_reserved_bytes -= required
            self._stop_background_writer_after_local_error()
            raise
        self._write_manifest()
        return True

    def _stop_background_writer_after_local_error(self) -> None:
        """Join the worker after a pre-admission failure without marking close."""

        writer = self._background_writer
        if writer is None:
            return
        try:
            stats = writer.close()
        except BackgroundWriterFailed as failure:
            self._background_writer_stats = writer.snapshot().as_json()
            with self._publication_lock:
                self._artifact_reserved_bytes = 0
            try:
                self._write_manifest_raw()
            except BaseException:
                pass
            raise failure
        self._background_writer_stats = stats.as_json()
        try:
            self._write_manifest_raw()
        except BaseException:
            pass

    def _publish_background_item(self, item: WriteItem) -> None:
        """Worker-only atomic file and sidecar publication."""

        raw = item.payload
        if len(raw) <= BACKGROUND_TRAILER_BYTES:
            raise CaptureIntegrityError("truncated background publication job")
        header_bytes = int.from_bytes(raw[-BACKGROUND_TRAILER_BYTES:], "big")
        header_start = len(raw) - BACKGROUND_TRAILER_BYTES - header_bytes
        if header_start <= 0:
            raise CaptureIntegrityError("invalid background publication header size")
        try:
            header = json.loads(raw[header_start:-BACKGROUND_TRAILER_BYTES])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CaptureIntegrityError(
                "malformed background publication header"
            ) from exc
        if (
            not isinstance(header, dict)
            or header.get("schema") != BACKGROUND_PUBLICATION_SCHEMA
            or header.get("schema_version") != BACKGROUND_PUBLICATION_SCHEMA_VERSION
        ):
            raise CaptureIntegrityError("unsupported background publication header")
        if header.get("artifact_bytes") != header_start:
            raise CaptureIntegrityError("background artifact length/header mismatch")
        name = header.get("name")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".pt")
        ):
            raise CaptureIntegrityError("unsafe background artifact name")
        digest = header.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise CaptureIntegrityError("invalid background artifact digest")
        sidecar_raw = f"{digest}  {name}\n".encode("ascii")
        required = header_start + len(sidecar_raw)
        if (
            header.get("sidecar_bytes") != len(sidecar_raw)
            or header.get("required_bytes") != required
        ):
            raise CaptureIntegrityError("background publication budget mismatch")

        path = self.directory / name
        serial = int(header["serial"])
        temporary = path.with_name(f".{name}.tmp-{os.getpid()}-{serial}")
        artifact_view = memoryview(raw)[:header_start]
        try:
            with temporary.open("wb") as handle:
                handle.write(artifact_view)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            artifact_view.release()
        os.replace(temporary, path)
        sidecar = path.with_suffix(path.suffix + ".sha256")
        _atomic_replace_bytes(sidecar, sidecar_raw, serial)
        _fsync_directory(self.directory)

        entry = {
            "path": name,
            "sha256": digest,
            "bytes": header_start,
            "sidecar_bytes": len(sidecar_raw),
            "kind": str(header["kind"]),
            "fragment_id": int(header["fragment_id"]),
            "fragment_version": int(header["fragment_version"]),
        }
        with self._publication_lock:
            self._artifact_reserved_bytes -= required
            self._artifact_bytes += required
            self._artifacts.append(entry)

    def _manifest(self) -> dict[str, Any]:
        with self._publication_lock:
            artifact_bytes = self._artifact_bytes
            artifact_reserved_bytes = self._artifact_reserved_bytes
            artifacts = list(self._artifacts)
        config = {
            "learner_id": self.learner_id,
            "rank": self.rank,
            "optimizer_class": (
                f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}"
            ),
            "torch_version": torch.__version__,
            "adamw_semantics": {
                "bias_correction": True,
                "epsilon_placement": "outside_sqrt",
                "weight_decay": "decoupled",
                "step_clock": "per_parameter_raw_optimizer_state",
            },
            "every": self.every,
            "max_hmc_events": self.max_hmc_events,
            "max_midpoint_windows": self.max_midpoint_windows,
            "max_bytes": self.max_bytes,
            "layout_sha256": self.layout_digest,
        }
        if self.background_writer_enabled:
            config.update(
                {
                    "background_writer": True,
                    "background_writer_max_items": self.background_writer_max_items,
                    "background_writer_max_bytes": self.background_writer_max_bytes,
                    "background_writer_payload_ownership": "exact_immutable_bytes",
                    "background_writer_publication": "fifo_single_worker",
                }
            )
        manifest = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "kind": "run_manifest",
            "config": config,
            "counters": {
                "broadcasts_seen": self._broadcast_seen,
                "window_resets_seen": self._window_reset_seen,
                "hmc_events_admitted": self._hmc_admitted,
                "midpoint_windows_admitted": self._midpoint_admitted,
                "artifact_bytes": artifact_bytes,
                "pending_raw_bytes": self._pending_raw_bytes,
                "push_attempt_serial": self._push_attempt_serial,
                "drop_counts": dict(sorted(self._drop_counts.items())),
                "closed": self._closed,
            },
            "artifacts": artifacts,
            "window_lifecycles": list(self._window_lifecycles.values()),
            "push_retries": {
                retry_identity: {
                    "attempts": retry["attempts"],
                    "candidate": retry["candidate"],
                }
                for retry_identity, retry in sorted(self._push_retries.items())
            },
            "push_attempts": {
                str(serial): dict(attempt)
                for serial, attempt in sorted(self._push_attempts_by_serial.items())
            },
            "recent_drops": list(self._recent_drops),
        }
        if self.background_writer_enabled:
            manifest["counters"]["background_artifact_reserved_bytes"] = (
                artifact_reserved_bytes
            )
            manifest["background_writer"] = dict(self._background_writer_stats or {})
        return manifest

    def _write_manifest(self) -> None:
        self._observe_background_writer()
        self._write_manifest_raw()

    def _write_manifest_raw(self) -> None:
        serial = self._next_serial()
        raw = _json_bytes(self._manifest())
        path = self.directory / "manifest.json"
        _atomic_replace_bytes(path, raw, serial)
        digest = hashlib.sha256(raw).hexdigest()
        sidecar_raw = f"{digest}  {path.name}\n".encode("ascii")
        _atomic_replace_bytes(
            path.with_suffix(".json.sha256"), sidecar_raw, self._next_serial()
        )
        _fsync_directory(self.directory)


def load_capture(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify an artifact's sidecar and payload digest, then return it."""

    artifact = Path(path)
    sidecar = artifact.with_suffix(artifact.suffix + ".sha256")
    try:
        fields = sidecar.read_text(encoding="ascii").strip().split()
    except FileNotFoundError as exc:
        raise CaptureIntegrityError(
            f"missing checksum sidecar for {artifact.name}"
        ) from exc
    if len(fields) != 2 or fields[1] != artifact.name:
        raise CaptureIntegrityError(f"malformed checksum sidecar for {artifact.name}")
    actual_file_digest = _file_sha256(artifact)
    if actual_file_digest != fields[0]:
        raise CaptureIntegrityError(f"file checksum mismatch for {artifact.name}")
    try:
        envelope = torch.load(artifact, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CaptureIntegrityError(f"could not decode {artifact.name}") from exc
    if (
        envelope.get("schema") != SCHEMA_NAME
        or envelope.get("schema_version") != SCHEMA_VERSION
    ):
        raise CaptureIntegrityError(f"unsupported capture schema in {artifact.name}")
    if capture_value_sha256(envelope.get("payload")) != envelope.get("payload_sha256"):
        raise CaptureIntegrityError(f"payload checksum mismatch for {artifact.name}")
    return envelope
