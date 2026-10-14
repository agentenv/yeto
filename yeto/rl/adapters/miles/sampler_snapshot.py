"""Opt-in SGLang snapshot mode for sampler engines (rl-trainer-cold-start-cost L8.2, design D5).

Snapshot mode starts the rollout (sampler) SGLang engines with
``enable_memory_saver`` and ``enable_weights_cpu_backup``, the two flags a GPU
memory snapshot needs. The trainer is never affected.

The mode is gated: it is refused at launch unless ``sampler_snapshot_compat.json``
(shipped with the package, so with the image) holds a ``pass`` record for the
pinned SGLang commit AND the weight-update path the island uses. A record is
written only after a GPU check compared text and weight checksums with the flags
on and off across RL weight updates (L8.1: ``tensor``; L8.2: ``distributed``).

Miles does not expose ``--sglang-enable-memory-saver`` (it derives the field from
``--offload-rollout``), so the flags reach the engines as per-group overrides in
an inline ``--sglang-config`` that mirrors Miles' own default single-group layout.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from importlib import resources
from typing import Any, Mapping, Sequence

from .pins import SGLANG_NEXT_COMMIT

COMPAT_FILE = "sampler_snapshot_compat.json"
SNAPSHOT_SERVER_ARGS = {"enable_memory_saver": True, "enable_weights_cpu_backup": True}
# Weight-update path -> how the sampler engines receive trainer weights.
UPDATE_PATH_TENSOR = "tensor"  # colocated: update_weights_from_tensor (CUDA IPC)
UPDATE_PATH_DISTRIBUTED = "distributed"  # fixed partition: NCCL update_weights_from_distributed


class SamplerSnapshotRefused(ValueError):
    """Snapshot mode was requested but cannot be enabled; the message names why."""


@dataclass(frozen=True)
class CompatRecord:
    sglang_commit: str
    update_path: str
    result: str
    check: str
    evidence: str


def load_compat_records(text: str | None = None) -> tuple[CompatRecord, ...]:
    """Parse the shipped compatibility file (or ``text``, for tests)."""

    if text is None:
        text = resources.files(__package__).joinpath(COMPAT_FILE).read_text(encoding="utf-8")
    raw = json.loads(text)
    if raw.get("schema") != 1:
        raise ValueError(f"{COMPAT_FILE}: unsupported schema {raw.get('schema')!r}")
    return tuple(
        CompatRecord(
            sglang_commit=str(r["sglang_commit"]),
            update_path=str(r["update_path"]),
            result=str(r["result"]),
            check=str(r["check"]),
            evidence=str(r.get("evidence", "")),
        )
        for r in raw.get("records", ())
    )


def require_recorded_pass(
    update_path: str,
    *,
    sglang_commit: str = SGLANG_NEXT_COMMIT,
    records: Sequence[CompatRecord] | None = None,
) -> CompatRecord:
    """Return the pass record for (commit, path) or refuse with the missing check named."""

    if records is None:
        records = load_compat_records()
    for record in records:
        if record.sglang_commit == sglang_commit and record.update_path == update_path:
            if record.result == "pass":
                return record
            raise SamplerSnapshotRefused(
                f"sampler snapshot mode refused: the snapshot compatibility check "
                f"({record.check}) recorded {record.result!r}, not 'pass', for SGLang "
                f"{sglang_commit[:9]} on the {update_path!r} weight-update path"
            )
    raise SamplerSnapshotRefused(
        "sampler snapshot mode refused: no recorded pass of the SGLang snapshot "
        "compatibility check (flags on vs off across RL weight updates, design D5) for "
        f"SGLang {sglang_commit[:9]} on the {update_path!r} weight-update path "
        f"(see yeto/rl/adapters/miles/{COMPAT_FILE})"
    )


def sglang_config_arg(rollout_gpus: int) -> str:
    """Inline ``--sglang-config`` value: Miles' default single group plus the snapshot flags.

    Same layout Miles builds without a config (one ``default`` model, one
    ``regular`` group over all rollout GPUs), so only the overrides differ.
    """

    config = {
        "sglang": [
            {
                "name": "default",
                "server_groups": [
                    {
                        "worker_type": "regular",
                        "num_gpus": int(rollout_gpus),
                        "overrides": dict(SNAPSHOT_SERVER_ARGS),
                    }
                ],
            }
        ]
    }
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))  # JSON is valid YAML
    return "base64:" + base64.b64encode(payload.encode()).decode()


def snapshot_argv(
    request: Any,
    *,
    extra_argv: Sequence[str] = (),
    records: Sequence[CompatRecord] | None = None,
    sglang_commit: str = SGLANG_NEXT_COMMIT,
) -> list[str]:
    """Miles argv enabling snapshot mode for a placement ``request``; refuses unsupported shapes."""

    if request.kind == "colocated":
        raise SamplerSnapshotRefused(
            "sampler snapshot mode refused: it applies to dedicated sampler engines "
            "(fixed partition); colocated engines share GPUs with the trainer"
        )
    if getattr(request, "placement_map_arg", None) is not None:
        raise SamplerSnapshotRefused(
            "sampler snapshot mode refused: not supported with an explicit role -> bundle "
            "placement map (standby / named rollout cells)"
        )
    if "--sglang-config" in extra_argv:
        raise SamplerSnapshotRefused(
            "sampler snapshot mode refused: it writes --sglang-config itself; "
            "drop --sglang-config from the extra Miles argv"
        )
    require_recorded_pass(UPDATE_PATH_DISTRIBUTED, sglang_commit=sglang_commit, records=records)
    return ["--sglang-config", sglang_config_arg(request.rollout_gpus)]


def decode_sglang_config_arg(value: str) -> Mapping[str, Any]:
    """Inverse of :func:`sglang_config_arg` (tests and provenance readers)."""

    assert value.startswith("base64:"), value
    return json.loads(base64.b64decode(value[len("base64:"):]).decode())
