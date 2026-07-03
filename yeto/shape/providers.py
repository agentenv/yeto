"""AWS quota and spot-placement-score fetchers for `yeto shape`.

The shaper needs two per-region signals to decide where learner islands can
actually launch: the vCPU quota that gates each instance family, and AWS's
spot placement score (a 1-10 likelihood that a spot ask of a given size will
be fulfilled). Both are slow per-call APIs, so everything fans out over a
thread pool and lands in a `TTLCache`; failures degrade to None rather than
aborting the shape — a missing signal just makes that region ineligible or
unranked, which the planner handles.

boto3 and sky are imported lazily inside methods so this module stays
importable (and testable) on machines without cloud credentials or the AWS
SDK installed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from yeto.shape.cache import TTLCache


@dataclass(frozen=True)
class QuotaKey:
    """One (region, quota-code) service-quota lookup."""

    region: str
    code: str  # AWS service-quotas quota code, e.g. "L-7212CCBC"


class AwsProviders:
    """Cached, parallel access to the AWS quota and placement-score APIs."""

    def __init__(self, cache: TTLCache, max_workers: int = 16) -> None:
        self.cache = cache
        self.max_workers = max_workers

    def quota_code(self, instance_type: str, use_spot: bool) -> str | None:
        """Map an instance type to its EC2 vCPU quota code via sky's catalog.

        SkyPilot already ships the instance-family -> quota-code table, so we
        reuse it instead of duplicating the mapping. Any failure (sky missing,
        unknown instance type) returns None: the caller then simply cannot
        quota-check that type, which is not fatal.
        """
        try:
            from sky.catalog import aws_catalog

            return aws_catalog.get_quota_code(instance_type, use_spot)
        except Exception:
            return None

    def quotas(self, keys: list[QuotaKey]) -> dict[QuotaKey, float | None]:
        """Fetch quota values for all keys in parallel; failures map to None.

        The try/except wraps `cache.get_or` (not just the boto3 call) so a
        failed fetch is never cached — the next run retries it.
        """

        def one(key: QuotaKey) -> float | None:
            try:
                return self.cache.get_or(
                    f"aws-quota:{key.region}:{key.code}",
                    lambda: self._fetch_quota(key),
                )
            except Exception:
                return None

        if not keys:
            return {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            values = list(pool.map(one, keys))
        return dict(zip(keys, values))

    def placement_scores(
        self, asks: list[tuple[str, int]], regions: list[str]
    ) -> dict[tuple[str, int, str], int | None]:
        """Fetch spot placement scores for each (instance_type, count) ask.

        The API scores a whole region list in one call, so we batch per ask
        rather than per (ask, region) — one cached call covers every region.
        Regions absent from the response (AWS omits regions it will not
        score) and failed calls both flatten to None.
        """

        def one(ask: tuple[str, int]) -> dict[str, int]:
            itype, count = ask
            key = f"aws-score:{itype}:{count}:{','.join(sorted(regions))}"
            try:
                return self.cache.get_or(
                    key, lambda: self._fetch_scores(itype, count, regions)
                )
            except Exception:
                return {}

        results: dict[tuple[str, int, str], int | None] = {}
        if asks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                per_ask = list(pool.map(one, asks))
        else:
            per_ask = []
        for (itype, count), scores in zip(asks, per_ask):
            for region in regions:
                results[(itype, count, region)] = scores.get(region)
        return results

    def _fetch_quota(self, key: QuotaKey) -> float:
        """One service-quotas API call. Raises on any failure (never cached)."""
        import boto3

        client = boto3.client("service-quotas", region_name=key.region)
        resp = client.get_service_quota(ServiceCode="ec2", QuotaCode=key.code)
        return float(resp["Quota"]["Value"])

    def _fetch_scores(
        self, itype: str, count: int, regions: list[str]
    ) -> dict[str, int]:
        """One spot-placement-scores API call covering all regions."""
        import boto3

        client = boto3.client("ec2", region_name="us-east-1")
        resp = client.get_spot_placement_scores(
            InstanceTypes=[itype],
            TargetCapacity=count,
            TargetCapacityUnitType="units",
            RegionNames=regions,
        )
        return {e["Region"]: e["Score"] for e in resp["SpotPlacementScores"]}
