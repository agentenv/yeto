"""Non-authoritative, atomic local cache for one expensive RL round result."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .core import PolicyIdentity

_SCHEMA = 2


@dataclass(frozen=True)
class CachedResult:
    base_identity: PolicyIdentity
    target_step: int
    delta: torch.Tensor
    delta_sha256: str
    stats: dict[str, Any]
    push_attempts: int
    first_push_unix_ns: int | None


class ResultCache:
    def __init__(
        self,
        directory: str | Path,
        *,
        run_manifest_sha256: str,
        learner_id: int,
        layout_fingerprint: str,
    ) -> None:
        self.directory = Path(directory).expanduser()
        self.metadata_path = self.directory / "result.json"
        self.delta_path = self.directory / "delta.f32"
        self.identity = {
            "run_manifest_sha256": run_manifest_sha256,
            "learner_id": learner_id,
            "layout_fingerprint": layout_fingerprint,
        }

    @staticmethod
    def _delta_bytes(delta: torch.Tensor) -> bytes:
        value = delta.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(value).all().item():
            raise ValueError("cached delta contains NaN or Inf")
        return value.numpy().astype("<f4", copy=False).tobytes()

    def save(
        self,
        *,
        base_identity: PolicyIdentity,
        target_step: int,
        delta: torch.Tensor,
        stats: dict[str, Any],
    ) -> CachedResult:
        if target_step != base_identity.version + 1:
            raise ValueError("cached target step must immediately follow its base version")
        self.directory.mkdir(parents=True, exist_ok=True)
        data = self._delta_bytes(delta)
        digest = hashlib.sha256(data).hexdigest()
        metadata = {
            "schema": _SCHEMA,
            **self.identity,
            "base_version": base_identity.version,
            "base_policy_hash": base_identity.policy_hash,
            "target_step": target_step,
            "numel": delta.numel(),
            "delta_sha256": digest,
            "stats": stats,
            "push_attempts": 0,
            "first_push_unix_ns": None,
        }
        self._replace(self.delta_path, data)
        self._replace(
            self.metadata_path,
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
        )
        self._sync_directory()
        return CachedResult(
            base_identity,
            target_step,
            delta.cpu().clone(),
            digest,
            stats,
            0,
            None,
        )

    def load(
        self,
        *,
        base_identity: PolicyIdentity,
        target_step: int,
        expected_numel: int,
    ) -> CachedResult | None:
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            data = self.delta_path.read_bytes()
            matches = (
                metadata.get("schema") == _SCHEMA
                and all(metadata.get(key) == value for key, value in self.identity.items())
                and metadata.get("base_version") == base_identity.version
                and metadata.get("base_policy_hash") == base_identity.policy_hash
                and metadata.get("target_step") == target_step
                and metadata.get("numel") == expected_numel
                and len(data) == expected_numel * 4
                and hashlib.sha256(data).hexdigest() == metadata.get("delta_sha256")
                and type(metadata.get("push_attempts")) is int
                and metadata["push_attempts"] >= 0
                and (
                    (metadata["push_attempts"] == 0)
                    == (metadata.get("first_push_unix_ns") is None)
                )
                and (
                    metadata.get("first_push_unix_ns") is None
                    or (
                        type(metadata["first_push_unix_ns"]) is int
                        and metadata["first_push_unix_ns"] > 0
                    )
                )
            )
            if not matches:
                raise ValueError("cache identity mismatch")
            delta = torch.frombuffer(bytearray(data), dtype=torch.float32).clone()
            if not torch.isfinite(delta).all().item():
                raise ValueError("cache contains NaN or Inf")
            return CachedResult(
                base_identity,
                target_step,
                delta,
                metadata["delta_sha256"],
                dict(metadata.get("stats") or {}),
                metadata["push_attempts"],
                metadata.get("first_push_unix_ns"),
            )
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.clear()
            return None

    def record_push(self, cached: CachedResult) -> CachedResult:
        """Persist one send attempt so cache resends remain observable after restart."""

        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("schema") != _SCHEMA
                or any(metadata.get(key) != value for key, value in self.identity.items())
                or metadata.get("base_version") != cached.base_identity.version
                or metadata.get("base_policy_hash") != cached.base_identity.policy_hash
                or metadata.get("target_step") != cached.target_step
                or metadata.get("delta_sha256") != cached.delta_sha256
                or metadata.get("push_attempts") != cached.push_attempts
                or metadata.get("first_push_unix_ns") != cached.first_push_unix_ns
            ):
                raise RuntimeError("RL result cache changed before push")
        except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("cannot record RL result-cache push") from exc

        first_push = cached.first_push_unix_ns or time.time_ns()
        attempts = cached.push_attempts + 1
        metadata["first_push_unix_ns"] = first_push
        metadata["push_attempts"] = attempts
        self._replace(
            self.metadata_path,
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
        )
        self._sync_directory()
        return CachedResult(
            cached.base_identity,
            cached.target_step,
            cached.delta,
            cached.delta_sha256,
            cached.stats,
            attempts,
            first_push,
        )

    def clear(self) -> None:
        for path in (self.metadata_path, self.delta_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def clear_if_committed(self, policy_version: int) -> None:
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if int(metadata["base_version"]) < policy_version:
                self.clear()
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.clear()

    @staticmethod
    def _replace(path: Path, data: bytes) -> None:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _sync_directory(self) -> None:
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
