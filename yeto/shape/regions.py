"""`--regions` parsing for `yeto shape`: one filter, many clouds.

Each cloud calls its geography something different — AWS and Nebius have
regions, Verda has location codes, Modal has price-multiplied placement
hints — but the planner needs one question answered per offering: "may
this (cloud, region) be planned?". `parse_regions` turns the CLI string
into a `RegionFilter` that answers it.

Grammar (comma-separated entries):

    region              -> aws:region      (the pre-multi-cloud spelling)
    cloud:region        -> that cloud, that region/location
    cloud:all           -> that cloud, unrestricted
    all                 -> alone: every cloud unrestricted; in a list: aws:all

Clouds with no entry at all are unrestricted — except AWS, which falls back
to `default_aws` (a handful of US regions) because its per-region
placement-score asks are a scarce daily resource and "all AWS regions" is
never a sensible accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

ALL = "all"


@dataclass(frozen=True)
class RegionFilter:
    """Per-cloud allowlists. A cloud absent from `per_cloud` (or mapped to
    None) is unrestricted; `explicit` names the clouds whose allowlist came
    from the user rather than a default, which is what error reporting on
    unknown regions keys off."""

    per_cloud: Mapping[str, frozenset[str] | None]
    explicit: frozenset[str]

    def for_cloud(self, cloud: str) -> frozenset[str] | None:
        return self.per_cloud.get(cloud)

    def allows(self, cloud: str, region: str) -> bool:
        wanted = self.per_cloud.get(cloud)
        return wanted is None or region in wanted

    def is_explicit(self, cloud: str) -> bool:
        return cloud in self.explicit


def parse_regions(
    entries: Iterable[str] | None,
    default_aws: Iterable[str],
    known_clouds: Iterable[str],
) -> RegionFilter:
    """Build the filter; unknown cloud names are a ValueError naming the
    known ones. Region strings keep their case (Verda's FIN-03 is
    uppercase); cloud names are lowercased."""
    known = {c.lower() for c in known_clouds}
    if entries is None:
        return RegionFilter({"aws": frozenset(default_aws)}, frozenset())
    items = [e.strip() for e in entries if e and e.strip()]
    if items == [ALL]:
        return RegionFilter({}, frozenset())

    per_cloud: dict[str, set[str] | None] = {}
    explicit: set[str] = set()
    for item in items:
        if ":" in item:
            cloud, _, region = item.partition(":")
        else:
            cloud, region = "aws", item
        cloud = cloud.strip().lower()
        region = region.strip()
        if cloud not in known:
            raise ValueError(
                f"unknown cloud {cloud!r} in --regions entry {item!r}; "
                f"known: {', '.join(sorted(known))}"
            )
        if not region:
            raise ValueError(f"empty region in --regions entry {item!r}")
        if region.lower() == ALL:
            per_cloud[cloud] = None
            explicit.discard(cloud)
            continue
        if per_cloud.get(cloud, set()) is None:
            continue  # already unrestricted via cloud:all
        per_cloud.setdefault(cloud, set()).add(region)
        explicit.add(cloud)
    if "aws" not in per_cloud:
        per_cloud["aws"] = set(default_aws)
    return RegionFilter(
        {c: (None if r is None else frozenset(r)) for c, r in per_cloud.items()},
        frozenset(explicit),
    )
