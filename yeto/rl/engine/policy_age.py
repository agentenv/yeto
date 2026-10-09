"""agentic-rollout-utilization stage 1: the policy-age limit (backend-neutral).

One switch, ``--rl-max-policy-age N`` (default 0), says how many published
policy versions a trained sample may lag behind the version being trained:

* 0 (default) is the deterministic mode: every sample comes from the policy
  just published; nothing here changes any default output;
* N > 0 lets an unfinished trajectory continue into later rounds; each token
  records the version that generated it (version segments) and a sample whose
  oldest version is more than N behind is discarded and reported.

The limit is part of the island's contract: :func:`bind_policy_age` folds a
non-zero limit into the identity the syncer compares (HELLO session contract
and elastic JOIN), so islands with different limits are refused per connection
while limit 0 keeps the identity byte-identical. Each backend declares how far
it supports the limit (:class:`PolicyAgeSupport`, adapter role ``policy_age``);
the launcher refuses an unsupported limit before any machine starts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_POLICY_AGE = 0
POLICY_AGE_DOMAIN = b"yeto-rl-policy-age-v1\0"
# ExecutionProfile.algorithm_contract for a non-zero limit (token-level version
# segments + cross-version importance-sampling correction, design decision 3).
BOUNDED_STALENESS_CONTRACT = "bounded-staleness"

# Stages of the change (proposal): 0 over-sample + cut-off (no carry-over),
# 1 the switch and its contract (limit 0 only), 2 single-turn carry-over,
# 3 agentic multi-turn carry-over.
STAGE_NAMES = {
    0: "over-sample and cut off, discard unfinished",
    1: "policy-age switch and contract (limit 0 only)",
    2: "single-turn carry-over",
    3: "agentic multi-turn carry-over",
}


class PolicyAgeError(ValueError):
    """The requested policy-age limit is malformed or unsupported."""


def validate_limit(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise PolicyAgeError(f"--rl-max-policy-age must be a non-negative integer, got {value!r}")
    return value


@dataclass(frozen=True)
class PolicyAgeSupport:
    """What a backend adapter supports: the highest stage it implements and
    the largest limit it can run (0 until stage 2 lands)."""

    backend: str
    stage: int
    max_policy_age: int

    def check(self, limit: int) -> None:
        """Raise before launch when ``limit`` exceeds what this backend runs."""
        validate_limit(limit)
        if limit > self.max_policy_age:
            raise PolicyAgeError(
                f"--rl-max-policy-age {limit}: backend {self.backend!r} supports up to stage "
                f"{self.stage} ({STAGE_NAMES.get(self.stage, '?')}), i.e. a policy-age limit "
                f"of at most {self.max_policy_age}; it is refused before any machine starts "
                "rather than silently run at 0"
            )


def bind_policy_age(identity_sha256: str | None, limit: int) -> str | None:
    """Fold a non-zero limit into the island identity the syncer compares.

    Limit 0 returns ``identity_sha256`` unchanged (default contracts are
    byte-identical); None stays None (no identity declared)."""
    validate_limit(limit)
    if identity_sha256 is None or limit == 0:
        return identity_sha256
    if len(identity_sha256) != 64 or any(c not in "0123456789abcdef" for c in identity_sha256):
        raise PolicyAgeError("identity must be a lowercase sha256 hex digest")
    body = POLICY_AGE_DOMAIN + bytes.fromhex(identity_sha256) + limit.to_bytes(4, "big")
    return hashlib.sha256(body).hexdigest()


def token_version(token: str) -> int | None:
    """Version of a ``yeto:<version>:<hash>`` policy token (None if malformed)."""
    parts = str(token).split(":")
    if len(parts) != 3 or parts[0] != "yeto":
        return None
    try:
        value = int(parts[1])
    except ValueError:
        return None
    return value if value >= 0 else None


def oldest_version(group: Any) -> int | None:
    """Oldest generating version of a group: its recorded version segments
    (``policy_versions``) when present, else its policy token's version."""
    versions = getattr(group, "policy_versions", None)
    if versions:
        return min(int(v) for v in versions)
    return token_version(getattr(group, "policy_token", ""))


def group_age(group: Any, current_version: int) -> int | None:
    oldest = oldest_version(group)
    return None if oldest is None else current_version - oldest


def split_by_age(groups: Iterable[Any], current_version: int, limit: int) -> tuple[list[Any], list[Any]]:
    """(kept, discarded): a group is kept iff its age is known, >= 0 and <= limit."""
    validate_limit(limit)
    kept: list[Any] = []
    discarded: list[Any] = []
    for g in groups:
        age = group_age(g, current_version)
        (kept if age is not None and 0 <= age <= limit else discarded).append(g)
    return kept, discarded


def check_batch_ages(groups: Sequence[Any], current_version: int, limit: int,
                     known_hashes: dict[int, str]) -> list[str]:
    """Driver guard: problems with a batch under ``limit`` (empty = accepted).

    Every group's token must name a published version within the limit whose
    hash the driver published (``known_hashes``: version -> policy hash)."""
    problems = []
    for g in groups:
        token = getattr(g, "policy_token", "")
        version = token_version(token)
        if version is None or known_hashes.get(version) is None \
                or token != f"yeto:{version}:{known_hashes[version]}":
            problems.append(f"{g.group_id}={token} (not a published policy)")
            continue
        age = group_age(g, current_version)
        if age is None or age < 0 or age > limit:
            problems.append(f"{g.group_id}: policy age {age} exceeds max_policy_age={limit}")
    return problems
