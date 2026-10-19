"""The sample-buffer payload: one group of samples as ``Sample.to_dict()``-shaped
mappings, and the writer that stores a group (payload first, then its index entry).

Engine-neutral (moved from ``adapters.miles.buffer_rollout``, which re-exports
these names): the Miles sampler and the engine sampler island
(:mod:`.engine_sampler`) write the same payloads, so one trainer reader serves
both. A sample carries ``response_length``, per-token ``rollout_log_probs`` and
its engine weight version (``weight_versions`` spans or
``metadata.weight_version``); the entry's ``behavior_version`` is the oldest.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .policy_versions import engine_version_number
from .trainer_lifecycle import SampleIndexEntry

PAYLOAD_SCHEMA = "yeto.rl.sample-buffer-payload/v1"
_NDARRAY = "__ndarray__"


class SampleBufferRolloutError(RuntimeError):
    pass


def _encode_value(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and hasattr(value, "dtype") and hasattr(value, "shape"):
        import numpy

        array = numpy.ascontiguousarray(value)
        return {_NDARRAY: base64.b64encode(array.tobytes()).decode("ascii"),
                "dtype": str(array.dtype), "shape": list(array.shape)}
    if hasattr(value, "value") and type(value).__module__ != "builtins":  # enum
        return value.value
    raise TypeError(f"sample field of type {type(value).__name__} is not serializable")


def _decode_hook(obj: dict[str, Any]) -> Any:
    if _NDARRAY in obj:
        import numpy

        data = base64.b64decode(obj[_NDARRAY])
        return numpy.frombuffer(data, dtype=obj["dtype"]).reshape(obj["shape"]).copy()
    return obj


def encode_group(samples: Sequence[Mapping[str, Any]]) -> bytes:
    """One payload: the group's samples as ``Sample.to_dict()`` mappings."""
    record = {"schema": PAYLOAD_SCHEMA, "samples": [dict(s) for s in samples]}
    return json.dumps(record, default=_encode_value, sort_keys=True, allow_nan=True).encode()


def decode_group(data: bytes) -> list[dict[str, Any]]:
    record = json.loads(data, object_hook=_decode_hook)
    if record.get("schema") != PAYLOAD_SCHEMA:
        raise SampleBufferRolloutError(f"payload schema {record.get('schema')!r} != {PAYLOAD_SCHEMA}")
    return list(record["samples"])


def _sample_versions(sample: Mapping[str, Any]) -> list[int]:
    versions = []
    for call in sample.get("weight_versions") or ():
        for span in call or ():
            versions.append(span["version"])
    if not versions and (sample.get("metadata") or {}).get("weight_version") is not None:
        versions.append(sample["metadata"]["weight_version"])
    numbers = [engine_version_number(v) for v in versions]
    if not numbers or None in numbers:
        raise SampleBufferRolloutError(
            f"sample {sample.get('index')}: no numeric engine weight version ({versions!r}); "
            "the buffer needs the behavior version of every sample")
    return [int(n) for n in numbers]


def group_entry_fields(samples: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """``(behavior_version, tokens)`` of a group: the oldest engine version of any
    token, and the response tokens. Every sample must carry per-token behavior
    log-probs (decoupled PPO needs them)."""
    if not samples:
        raise SampleBufferRolloutError("empty group")
    oldest, tokens = None, 0
    for sample in samples:
        length = int(sample.get("response_length") or 0)
        logprobs = sample.get("rollout_log_probs")
        if logprobs is None or len(logprobs) != length:
            raise SampleBufferRolloutError(
                f"sample {sample.get('index')}: rollout_log_probs "
                f"{'missing' if logprobs is None else len(logprobs)} != response_length {length}")
        version = min(_sample_versions(sample))
        oldest = version if oldest is None else min(oldest, version)
        tokens += length
    return int(oldest), tokens


class GroupWriter:
    """Writes finished groups to the buffer: payload first, then the index entry."""

    def __init__(self, buffer: Any, sampler_id: str) -> None:
        if not sampler_id:
            raise ValueError("sampler_id must be non-empty")
        self.buffer = buffer
        self.sampler_id = str(sampler_id)

    def write(self, samples: Sequence[Mapping[str, Any]], *, group_key: str,
              provenance: Mapping[str, str] | None = None) -> SampleIndexEntry:
        """``provenance``: the sampler's source record for the index entry (S25 N2.4)."""
        behavior_version, tokens = group_entry_fields(samples)
        sample_id = f"{self.sampler_id}-{group_key}"
        ref = self.buffer.write_payload(sample_id, encode_group(samples))
        entry = SampleIndexEntry(sample_id=sample_id, behavior_version=behavior_version,
                                 payload_ref=ref, tokens=tokens,
                                 provenance=None if provenance is None else dict(provenance))
        self.buffer.append(entry)
        return entry


