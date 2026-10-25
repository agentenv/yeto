"""Behavior policies that are external merged cuts (S27 G-EM, relay mode).

A training island that reads its groups from the head's sample relay
(``--rl-sample-relay`` with ``--rl-sample-role trainer``) did not generate
them with its own publications. Spot sampler islands generated them with the
merged weights the syncer broadcast to its receive-only subscribers
(:mod:`yeto.rl.spot_rollout`). The island's own policy identity (local
version ``L``, the weight hash of each local publication) does not name
those weights, so relay mode names them explicitly:

* kind :data:`EXTERNAL_MERGED_CUT`;
* the *merged version* ``M`` of the cut
  (:func:`yeto.rl.engine.full_parameter_sync.merged_version`: the merges
  every fragment received), the version the spot samplers stamp on their
  engines and the unit the island's buffer gate uses;
* the cut digest (:func:`cut_digest`): SHA-256 over the layout fingerprint
  and, per fragment, the broadcast version and the SHA-256 of the bf16 bytes
  the subscriber received. Merged version 0 is the initial weights, which a
  spot sampler serves from its base checkpoint before the first merge (no
  broadcast exists yet; the trainers' first rounds need these groups): its
  digest is :func:`base_cut_digest` of that checkpoint's weight files.

The policy age is counted in merged versions on both sides: the trainer's
merged version at the publication a batch is checked against, minus the
oldest merged version of a group. A group from a cut newer than the trainer's
own merged version (the subscriber received a merge the trainer applies at
its next boundary) has a negative age and is accepted, as the buffer gate
accepts it.

Engine-neutral: the driver (:class:`ExternalMergedCuts`), the trajectory
evidence (``behavior_cut_of``, schema 5 of
:mod:`yeto.rl.trajectory_evidence`) and the relay pump (the sample metadata
key) share these rules; each backend only routes the values. Off relay mode
nothing here is used.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from yeto.rl.contracts import EXTERNAL_MERGED_CUT

from .policy_versions import engine_version_number

__all__ = [
    "CUT_METADATA_KEY",
    "EXTERNAL_MERGED_CUT",
    "ExternalCutError",
    "ExternalMergedCuts",
    "base_cut_digest",
    "behavior_cut_of",
    "cut_digest",
    "cut_record",
    "fragment_digest",
]

# Sample metadata key the trainer's relay pump writes on every relay sample:
# {"kind": EXTERNAL_MERGED_CUT, "merged_version": M, "digest": <sha256>}.
CUT_METADATA_KEY = "yeto_behavior_cut"
CUT_DIGEST_DOMAIN = b"yeto-merged-cut-v1\0"
BASE_CUT_DIGEST_DOMAIN = b"yeto-base-cut-v1\0"
BASE_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")
_SHA256_HEX = frozenset("0123456789abcdef")


class ExternalCutError(ValueError):
    """A sample or group does not name a valid external merged cut."""


def fragment_digest(data: bytes | bytearray | memoryview) -> bytes:
    """SHA-256 of one fragment's broadcast bytes (bf16 values, layout order)."""
    return hashlib.sha256(data).digest()


def cut_digest(layout_fingerprint: str, versions: Sequence[int], fragment_digests: Sequence[bytes]) -> str:
    """The digest of one merged cut: the layout fingerprint (hex) and, for every
    fragment in order, its broadcast version (u64) and :func:`fragment_digest`."""
    if len(versions) != len(fragment_digests) or not versions:
        raise ExternalCutError("a cut digest needs one version and one digest per fragment")
    body = hashlib.sha256(CUT_DIGEST_DOMAIN)
    body.update(bytes.fromhex(layout_fingerprint))
    for version, digest in zip(versions, fragment_digests):
        if type(version) is not int or version < 0 or len(digest) != 32:
            raise ExternalCutError("a cut digest needs non-negative versions and 32-byte digests")
        body.update(struct.pack("<Q", version))
        body.update(digest)
    return body.hexdigest()


def base_cut_digest(base_dir: str | Path) -> str:
    """The digest of merged version 0 (module doc): SHA-256 over the relative path
    and the SHA-256 of every weight file of the base checkpoint, in path order."""
    root = Path(base_dir)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name.endswith(BASE_WEIGHT_SUFFIXES))
    if not files:
        raise ExternalCutError(f"base checkpoint {root} has no weight file")
    body = hashlib.sha256(BASE_CUT_DIGEST_DOMAIN)
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                digest.update(chunk)
        name = path.relative_to(root).as_posix().encode()
        body.update(struct.pack("<Q", len(name)) + name + digest.digest())
    return body.hexdigest()


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def cut_record(merged_version: int, digest: str) -> dict[str, Any]:
    """The sample metadata value (:data:`CUT_METADATA_KEY`) of one cut."""
    if type(merged_version) is not int or merged_version < 0:
        raise ExternalCutError(f"merged version must be a non-negative int, got {merged_version!r}")
    if not _sha256(digest):
        raise ExternalCutError("cut digest must be a lowercase SHA-256")
    return {"kind": EXTERNAL_MERGED_CUT, "merged_version": merged_version, "digest": digest}


def parse_cut(value: Any) -> tuple[int, str]:
    """(merged version, digest) of a :func:`cut_record` value."""
    if not isinstance(value, Mapping) or value.get("kind") != EXTERNAL_MERGED_CUT:
        raise ExternalCutError(f"not an external merged cut record: {value!r:.200}")
    record = cut_record(value.get("merged_version"), value.get("digest"))
    if set(value) != set(record):
        raise ExternalCutError(f"external merged cut record has other fields: {sorted(value)}")
    return record["merged_version"], record["digest"]


def _metadata(sample: Any) -> Mapping[str, Any]:
    meta = sample.get("metadata") if isinstance(sample, Mapping) else getattr(sample, "metadata", None)
    return meta if isinstance(meta, Mapping) else {}


def behavior_cut_of(segments: Sequence[Any]) -> tuple[int, str]:
    """(merged version, digest) of the cut that generated one logical trajectory
    (one sample, or the segments of a compaction): the oldest cut any segment
    names. Every segment must name one (written by the relay pump)."""
    cuts = set()
    for sample in segments:
        value = _metadata(sample).get(CUT_METADATA_KEY)
        if value is None:
            raise ExternalCutError("a relay trajectory segment names no external merged cut")
        cuts.add(parse_cut(value))
    if not cuts:
        raise ExternalCutError("a trajectory without segments names no cut")
    oldest = min(version for version, _ in cuts)
    digests = {digest for version, digest in cuts if version == oldest}
    if len(digests) != 1:
        raise ExternalCutError(f"merged version {oldest} names {len(digests)} different cuts in one trajectory")
    return oldest, digests.pop()


def group_merged_version(group: Any) -> int | None:
    """Oldest merged version of a driver group: its version segments when listed,
    else its engine version (the plain number the spot sampler served)."""
    versions = getattr(group, "policy_versions", None)
    if versions:
        return min(int(v) for v in versions)
    return engine_version_number(getattr(group, "policy_token", ""))


class ExternalMergedCuts:
    """Driver side of relay mode (module doc).

    :meth:`published` records, for each local publication, the trainer's merged
    version at that point (the version its buffer gate compares against);
    :meth:`check_groups` checks a batch against the merged version of the
    publication it was generated under."""

    def __init__(self, keep: int = 64) -> None:
        self.keep = int(keep)
        self._merged: dict[int, int] = {}

    def published(self, local_version: int, merged_version: int) -> None:
        if type(merged_version) is not int or merged_version < 0:
            raise ExternalCutError(f"merged version must be a non-negative int, got {merged_version!r}")
        self._merged[int(local_version)] = merged_version
        for old in sorted(self._merged)[:-self.keep]:
            del self._merged[old]

    def merged_at(self, local_version: int | None) -> int | None:
        return None if local_version is None else self._merged.get(int(local_version))

    @staticmethod
    def check_groups(groups: Iterable[Any], merged_now: int, limit: int) -> tuple[list[str], dict[int, int]]:
        """(problems, {age: groups}); empty problems = accepted. A group must name a
        plain merged version whose age (``merged_now`` minus it) is at most ``limit``."""
        problems: list[str] = []
        ages: dict[int, int] = {}
        for group in groups:
            oldest = group_merged_version(group)
            if oldest is None:
                problems.append(f"{group.group_id}={getattr(group, 'policy_token', '')!r} "
                                "(no merged version)")
                continue
            age = int(merged_now) - oldest
            if age > int(limit):
                problems.append(f"{group.group_id}: merged version {oldest} is {age} behind {merged_now} "
                                f"(max_policy_age={limit})")
                continue
            ages[age] = ages.get(age, 0) + 1
        return problems, dict(sorted(ages.items()))
