"""Object-store sample pool: index format and driver interface draft (tasks 0.12).

Only the index and the interface. Samples themselves live in an object store
(S3, Modal Volume or Nebius Object Storage); the coordinator (syncer) keeps
only these index entries (design D-S4 SAMPLE_INDEX). No transfer is
implemented here: :class:`CrossIslandSampleSource` is the shape the driver
will call once the transfer exists.

Group rule (user ruling 2026-10-07): one prompt group never spans islands and
has one policy_hash; the group's advantage is computed on the producing
island and travels with the samples; the consumer only applies the
importance-sampling correction the ledger names.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .island_ledger import ACCEPT, ACCEPT_IS, CrossIslandLedger, SampleGroup, Verdict

SCHEMA = "yeto.rl.sample-index/v1"
URI_SCHEMES = ("s3://", "modal-volume://", "nebius-os://")


class SampleIndexError(ValueError):
    pass


@dataclass(frozen=True)
class SampleIndexEntry:
    island_id: str
    outer_version: int
    inner_step: int
    policy_hash: str
    group_id: str  # one prompt group; never spans islands
    prompt_id: str
    n: int  # samples in the group
    uri: str  # object-store location of the serialized group
    size_bytes: int
    sha256: str
    has_behavior_logprob: bool
    advantage_included: bool = True  # computed by the producer (group rule)
    created_at: float = 0.0
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise SampleIndexError(f"schema {self.schema!r} != {SCHEMA}")
        if not self.uri.startswith(URI_SCHEMES):
            raise SampleIndexError(f"uri {self.uri!r} is not one of {URI_SCHEMES}")
        if self.n < 1 or self.size_bytes < 0 or self.outer_version < 0:
            raise SampleIndexError("n >= 1, size_bytes >= 0, outer_version >= 0 required")
        if len(self.sha256) != 64:
            raise SampleIndexError("sha256 must be 64 hex chars")
        if not self.advantage_included:
            raise SampleIndexError("group rule: the producer must include the group advantage")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_json(raw: str | Mapping[str, Any]) -> "SampleIndexEntry":
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
        return SampleIndexEntry(**data)

    def as_group(self) -> SampleGroup:
        # The ledger only needs to know whether behavior logprobs exist.
        return SampleGroup(self.island_id, self.outer_version, self.inner_step, self.policy_hash,
                           n=self.n, behavior_logprob=(0.0,) if self.has_behavior_logprob else None,
                           uri=self.uri)


@dataclass(frozen=True)
class Selected:
    entry: SampleIndexEntry
    verdict: Verdict


class SamplePoolIndex:
    """Coordinator-side index; one entry per (island, group)."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], SampleIndexEntry] = {}

    def add(self, entry: SampleIndexEntry) -> None:
        key = (entry.island_id, entry.group_id)
        other = next((e for (i, g), e in self._entries.items()
                      if g == entry.group_id and i != entry.island_id), None)
        if other is not None:
            raise SampleIndexError(f"group {entry.group_id} already indexed from island "
                                   f"{other.island_id}: a group never spans islands")
        if key in self._entries and self._entries[key] != entry:
            raise SampleIndexError(f"group {entry.group_id} re-indexed with different content")
        self._entries[key] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def select(self, ledger: CrossIslandLedger, *, consumer_island: str,
               limit: int | None = None) -> list[Selected]:
        """Entries the ledger accepts (as-is or with correction), oldest version first."""
        out: list[Selected] = []
        for e in sorted(self._entries.values(), key=lambda e: (e.outer_version, e.island_id,
                                                              e.group_id)):
            v = ledger.judge(e.as_group(), consumer_island=consumer_island)
            if v.verdict in (ACCEPT, ACCEPT_IS):
                out.append(Selected(e, v))
                if limit is not None and len(out) >= limit:
                    break
        return out

    def prune_below(self, outer_version: int) -> int:
        old = [k for k, e in self._entries.items() if e.outer_version < outer_version]
        for k in old:
            del self._entries[k]
        return len(old)


class CrossIslandSampleSource(Protocol):
    """Driver interface draft: where the driver's batch would read cross-island groups.

    ``fetch`` downloads the groups behind ``selected`` from the object store,
    verifies size and sha256, and returns rollout-batch payloads carrying the
    producer's advantage, behavior logprobs and the correction name from the
    verdict. Not implemented in this change.
    """

    def fetch(self, selected: Iterable[Selected]) -> list[Mapping[str, Any]]: ...
