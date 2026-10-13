"""Inference-engine weight versions and the yeto policy labels they stand for.

The publisher writes the plain decimal policy version (``"7"``) as the
inference engine's weight version, so the engine's own staleness accounting
(a bounded version lag between generation and training) works on numbers.
The policy label that names the exact weights, ``yeto:<version>:<hash>``,
stays on the yeto side: :class:`PublishedPolicies` maps every published
version to its label, refuses a version that is published again with
different weights, and translates the versions reported on generated tokens
back to labels. An engine version that no publication recorded translates to
nothing, so a sample from unknown weights is never trained.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .policy_age import token_version


class PolicyVersionConflict(ValueError):
    """A policy version was published again with different weights."""


def engine_weight_version(version: int) -> str:
    """The weight version the publisher writes to every inference engine."""
    if type(version) is not int or version < 0:
        raise ValueError(f"policy version must be a non-negative int, got {version!r}")
    return str(version)


def _is_plain_version(text: str) -> bool:
    return text.isascii() and text.isdigit()


def engine_version_number(raw: object) -> int | None:
    """Policy version of an engine-reported weight version (None if unknown).

    Accepts the plain decimal version and, for engines stamped by older
    publishers, the full ``yeto:<version>:<hash>`` label.
    """
    text = str(raw)
    return int(text) if _is_plain_version(text) else token_version(text)


def policy_label(version: int, policy_hash: str) -> str:
    return f"yeto:{version}:{policy_hash}"


@dataclass
class PublishedPolicies:
    """Published policy version -> weight hash, for the versions still in use."""

    _hashes: dict[int, str] = field(default_factory=dict)

    def check(self, version: int, policy_hash: str) -> None:
        """Refuse a publication whose version already names other weights."""
        engine_weight_version(version)  # validates the version
        if not policy_hash:
            raise ValueError(f"policy v{version} has no weight hash")
        known = self._hashes.get(version)
        if known is not None and known != policy_hash:
            raise PolicyVersionConflict(
                f"policy v{version} was published as {known} and again as {policy_hash}")

    def record(self, version: int, policy_hash: str) -> None:
        """Record a publication; the same version must always carry the same hash."""
        self.check(version, policy_hash)
        self._hashes[version] = policy_hash

    def retain_from(self, oldest: int) -> None:
        """Forget versions older than ``oldest`` (outside the policy-age window)."""
        for version in [v for v in self._hashes if v < oldest]:
            del self._hashes[version]

    def hash_of(self, version: int) -> str | None:
        return self._hashes.get(version)

    def label(self, version: int) -> str | None:
        policy_hash = self._hashes.get(version)
        return None if policy_hash is None else policy_label(version, policy_hash)

    def label_of_engine_version(self, raw: object) -> str | None:
        """Label of an engine-reported weight version, or None when unpublished.

        A full label is accepted only when it names the recorded hash.
        """
        version = engine_version_number(raw)
        label = None if version is None else self.label(version)
        if label is None or (not _is_plain_version(str(raw)) and str(raw) != label):
            return None
        return label

    def hashes(self) -> dict[int, str]:
        return dict(sorted(self._hashes.items()))

    def to_dict(self) -> dict[str, str]:
        return {str(v): h for v, h in sorted(self._hashes.items())}

    @classmethod
    def from_dict(cls, raw: Mapping[str, str] | None) -> "PublishedPolicies":
        published = cls()
        for version, policy_hash in (raw or {}).items():
            published.record(int(version), str(policy_hash))
        return published


def engine_version_matches(raw: object, label: str) -> bool:
    """Whether an engine-reported weight version is the published ``label``.

    A plain version number matches by version: the publication check bound
    that number to the label's weights. A full label must match exactly.
    """
    text = str(raw)
    return int(text) == token_version(label) if _is_plain_version(text) else text == label
