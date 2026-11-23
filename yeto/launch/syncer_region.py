"""Where the syncer runs relative to its islands.

The syncer (or the head cluster that hosts it) runs in ``--syncer-region``:
``region`` (AWS) or ``cloud/region``, default AWS ``us-west-2``. Islands
broadcast-download every merged fragment from it, so a syncer in another region
than its islands throttles every fragment version. Measured on Modal
(2026-11-23, one TCP stream): syncer eu-west, islands us-west: 15.9 MB/s;
both ends in us-east: 132 MB/s.

* :func:`resolve_syncer_region`: when ``--syncer-region`` is left at its default
  and every island of a synced launch is a Modal or AWS island pinned
  (``@region``) to one geographic area with an AWS region, the syncer follows
  them there and the launch prints the inherited region. An explicit
  ``--syncer-region`` (even ``us-west-2``) is the opt-out. No-sync launches
  (one trainer, no syncer) keep the default untouched.
* :func:`region_findings`: warnings for synced Modal islands without
  ``@region`` (Modal places them in any region) or in another area than the
  syncer.

Import-light: the launcher's argparse default is built here.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence

DEFAULT_SYNCER_REGION = "us-west-2"
FOLLOW_CLOUDS = frozenset({"modal", "aws"})
MEASURED_CROSS_REGION_MB_S = 15.9
MEASURED_SAME_REGION_MB_S = 132.0
# Geographic area -> the AWS region a syncer VM there runs in. Modal's broad
# regions name the area; AWS / GCP / OCI regions reduce to one (see area_of).
AWS_REGION_OF_AREA = {
    "us-east": "us-east-1",
    "us-west": "us-west-2",
    "eu-west": "eu-west-1",
    "eu-central": "eu-central-1",
    "ap-northeast": "ap-northeast-1",
    "ap-southeast": "ap-southeast-1",
}
_OCI_CITY_AREA = {
    "ashburn": "us-east", "phoenix": "us-west", "sanjose": "us-west", "chicago": "us-central",
    "frankfurt": "eu-central", "london": "eu-west", "amsterdam": "eu-west", "tokyo": "ap-northeast",
    "singapore": "ap-southeast",
}
_AREA = re.compile(r"^([a-z]{2})-([a-z]+?)(?:-?\d+)?$")


class _DefaultSyncerRegion(str):
    """``--syncer-region`` left at its default (the launch may resolve it)."""

    __slots__ = ()


def default_syncer_region() -> str:
    """The argparse default of ``--syncer-region``, marked as unset."""
    return _DefaultSyncerRegion(DEFAULT_SYNCER_REGION)


def is_default_syncer_region(value: object) -> bool:
    return isinstance(value, _DefaultSyncerRegion)


def area_of(region: str | None) -> str | None:
    """Geographic area of a region string: ``us-east`` for Modal ``us-east``, AWS
    ``us-east-1``, GCP ``us-east4``, OCI ``us-ashburn-1``; None when unknown."""
    if not region:
        return None
    name = str(region).split("/", 1)[-1].lower()
    m = _AREA.match(name)
    if not m:
        return None
    country, part = m.groups()
    if part in _OCI_CITY_AREA:
        return _OCI_CITY_AREA[part]
    return f"{country}-{part}"


def _synced(args) -> bool:
    return (getattr(args, "training_mode", "sft") == "rl"
            and not getattr(args, "rl_single_island_no_sync", False))


def followed_region(args, specs: Sequence) -> str | None:
    """The AWS region the syncer follows the islands to, or None."""
    if not _synced(args) or not specs or int(getattr(args, "external_learners", 0) or 0):
        return None
    if any(spec.cloud not in FOLLOW_CLOUDS or not spec.region for spec in specs):
        return None
    areas = {area_of(spec.region) for spec in specs}
    if len(areas) != 1:
        return None
    exact = {spec.region for spec in specs if spec.cloud == "aws"}
    if len(exact) == 1 and all(spec.cloud == "aws" for spec in specs):
        return next(iter(exact))  # AWS islands in one AWS region: that region
    return AWS_REGION_OF_AREA.get(next(iter(areas)) or "")


def resolve_syncer_region(args, specs: Sequence, *, out=None) -> None:
    """Replace the default ``--syncer-region`` with the islands' region when
    :func:`followed_region` finds one (printed); drop the mark either way."""
    value = getattr(args, "syncer_region", None)
    if not is_default_syncer_region(value):
        return
    region = followed_region(args, specs)
    if region and region != str(value):
        print(f"[auto] syncer region {region}: follows the islands' @region (default {DEFAULT_SYNCER_REGION}; "
              "--syncer-region sets it explicitly)", file=out or sys.stdout, flush=True)
    args.syncer_region = region or str(value)


def region_findings(args, specs: Sequence) -> list[str]:
    """Warnings for synced Modal islands placed apart from the syncer."""
    if not _synced(args):
        return []
    modal = [(i, spec) for i, spec in enumerate(specs) if spec.cloud == "modal"]
    if not modal:
        return []
    syncer = str(getattr(args, "syncer_region", "") or DEFAULT_SYNCER_REGION)
    syncer_area = area_of(syncer)
    unpinned = [i for i, spec in modal if not spec.region]
    apart = [(i, spec.region) for i, spec in modal if spec.region and area_of(spec.region) != syncer_area]
    measured = (f"measured on Modal: syncer and islands in different regions {MEASURED_CROSS_REGION_MB_S:g} MB/s "
                f"per stream, the same region {MEASURED_SAME_REGION_MB_S:g} MB/s")
    out = []
    if unpinned:
        out.append(f"Modal island(s) {unpinned} have no @region, so Modal may place them far from the syncer "
                   f"({syncer}); {measured}. Pin every island to the syncer's area, e.g. modal:2xh100@"
                   f"{syncer_area or 'us-west'}, or pin them all to one region and leave --syncer-region "
                   "unset so the syncer follows.")
    if apart:
        listed = ", ".join(f"{i}@{r}" for i, r in apart)
        out.append(f"Modal island(s) {listed} run in another region than the syncer ({syncer}); {measured}. "
                   "Pin the islands and the syncer to one region (@region and --syncer-region).")
    return out
