"""LoRA publication on the verl backend (rl-verl-backend 1.10 / 2.2, design D5).

One publisher interface, two transports chosen by configuration:

* ``memory`` (default): verl's own in-memory hot update (``TensorLoRARequest``
  via the checkpoint engine), driven by the trainer's ``update_weights``;
* ``disk`` (fallback for engines without in-memory LoRA updates, e.g. if
  vLLM-Ascend lacks it): the adapter is written to a directory named by the
  version; once the new version has loaded, older version directories are
  deleted.

After every publication the inference side reports a checksum of the adapter
it *registered* (``vllm_readback``), keyed by canonical names; it is compared
with the checksum of what the trainer sent.  This proves "the inference side
received this version" -- not "the slot in use holds it" (design D5 level).
When no readback can be obtained the result is ``LORA_UNVERIFIABLE``, never a
silent pass.

Checksums are over bf16 values (vLLM stores adapters in the model dtype; the
trainer side is cast the same way), canonical names sorted.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

TRANSPORTS = ("memory", "disk")
CHECK_LEVEL = "inference-received"  # D5: registered adapter, not the active slot
VERIFIED = "VERIFIED"
MISMATCH = "LORA_MISMATCH"
UNVERIFIABLE = "LORA_UNVERIFIABLE"
CHECKSUM_SCHEMA = "yeto-lora-bf16-checksum-v1"


def tensor_bf16_digest(tensor) -> str:
    import torch

    value = tensor.detach().to("cpu").contiguous().to(torch.bfloat16)
    return hashlib.sha256(value.view(torch.int16).numpy().tobytes()).hexdigest()


def lora_checksum(tensors: Mapping[str, object]) -> dict:
    """{"total": sha256, "per_name": {name: sha256}, "shapes": {...}} over canonical names."""
    per_name, shapes = {}, {}
    total = hashlib.sha256(CHECKSUM_SCHEMA.encode() + b"\0")
    for name in sorted(tensors):
        digest = tensor_bf16_digest(tensors[name])
        per_name[name] = digest
        shapes[name] = list(tensors[name].shape)
        total.update(name.encode() + b"\0" + digest.encode())
    return {"schema": CHECKSUM_SCHEMA, "total": total.hexdigest(), "count": len(per_name),
            "per_name": per_name, "shapes": shapes}


@dataclass(frozen=True)
class PublishCheck:
    version: int
    status: str  # VERIFIED / LORA_MISMATCH / LORA_UNVERIFIABLE
    sent_checksum: str
    readback_checksum: str | None
    level: str = CHECK_LEVEL
    detail: str = ""
    mismatched: tuple[str, ...] = ()

    def to_event(self) -> dict:
        return {"event": "rl_publication_check", "policy_version": self.version,
                "status": self.status, "sent_checksum": self.sent_checksum,
                "readback_checksum": self.readback_checksum, "level": self.level,
                "detail": self.detail, "mismatched": list(self.mismatched[:8]),
                "mismatched_count": len(self.mismatched)}


def compare(version: int, sent: dict, readback: dict | None, *, detail: str = "") -> PublishCheck:
    if not readback or not readback.get("per_name"):
        reason = (readback or {}).get("error") or detail or "no readback"
        return PublishCheck(version, UNVERIFIABLE, sent["total"], None, detail=reason)
    names = set(sent["per_name"]) | set(readback["per_name"])
    bad = tuple(sorted(n for n in names
                       if sent["per_name"].get(n) != readback["per_name"].get(n)))
    status = VERIFIED if not bad and sent["total"] == readback["total"] else MISMATCH
    return PublishCheck(version, status, sent["total"], readback["total"],
                        detail=detail or readback.get("note", ""), mismatched=bad)


class DiskTransport:
    """Versioned adapter directories; old versions are removed after a successful load."""

    def __init__(self, root: str | os.PathLike, save: Callable[[Mapping, Path], None]):
        self.root = Path(root)
        self.save = save
        self.loaded: int | None = None

    def path_for(self, version: int) -> Path:
        return self.root / f"v{version:08d}"

    def stage(self, version: int, tensors: Mapping) -> Path:
        target = self.path_for(version)
        tmp = target.with_name(target.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        self.save(tensors, tmp)
        (tmp / "yeto_checksum.json").write_text(json.dumps(lora_checksum(tensors)["total"]))
        os.replace(tmp, target)
        return target

    def commit(self, version: int, load: Callable[[Path], bool]) -> bool:
        """Load ``version``; on success delete every other version directory."""
        path = self.path_for(version)
        if not path.is_dir():
            raise FileNotFoundError(path)
        if not load(path):
            return False
        self.loaded = version
        for other in self.root.iterdir():
            if other.is_dir() and other != path and other.name.startswith("v"):
                shutil.rmtree(other)
        return True


def transport_for(name: str):
    if name not in TRANSPORTS:
        raise ValueError(f"publication transport must be one of {TRANSPORTS}, got {name!r}")
    return name
