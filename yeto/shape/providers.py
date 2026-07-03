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

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from yeto.shape.cache import TTLCache

# ClientError codes AWS uses for rate limiting; these mean "back off and
# retry", not "misconfigured", so they get a friendlier warning.
_THROTTLE_CODES = {
    "RequestLimitExceeded",
    "Throttling",
    "ThrottlingException",
    "TooManyRequestsException",
}


def credentials_available() -> bool:
    """True iff boto3 imports and can resolve AWS credentials locally.

    Only walks the local credential chain (env vars, config files, cached
    tokens) — no network call — so callers can cheaply skip the AWS signal
    fan-out entirely when it would just produce a wall of auth failures.
    """
    try:
        import boto3

        return boto3.session.Session().get_credentials() is not None
    except Exception:
        return False


def _client_error_code(exc: Exception) -> str | None:
    """Extract the AWS error code from a botocore ClientError, else None."""
    try:
        from botocore.exceptions import ClientError
    except ImportError:
        return None
    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code", "")
    return None


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
        # Human-readable notes about degraded signals (throttling, auth
        # failures, missing boto3); the CLI surfaces these after shaping so
        # a silent None never masks "AWS rate-limited us".
        self.warnings: list[str] = []
        self._warn_lock = threading.Lock()

    def _warn(self, message: str, once: bool = False) -> None:
        """Append a warning; fetches run on a thread pool, hence the lock."""
        with self._warn_lock:
            if once and message in self.warnings:
                return
            self.warnings.append(message)

    def _note_failure(
        self, exc: Exception, what: str, target: str, hint: str = ""
    ) -> None:
        """Record why a fetch failed, distinguishing throttling from errors.

        Throttling is transient and common when shaping large fleets, so it
        gets an actionable retry message; other AWS errors surface their
        code; a missing boto3 collapses to one warning for the whole run.
        """
        if isinstance(exc, ImportError):
            self._warn("boto3 unavailable; AWS signals disabled", once=True)
            return
        code = _client_error_code(exc)
        if code in _THROTTLE_CODES:
            self._warn(
                f"{what} throttled for {target}; results may be incomplete"
                f" — retry in a few minutes{hint}"
            )
        elif code is not None:
            self._warn(f"{what} failed for {target}: {code}")
        else:
            self._warn(f"{what} failed for {target}: {exc}")

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
            except Exception as exc:
                self._note_failure(exc, "quotas", f"{key.code} in {key.region}")
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
            except Exception as exc:
                self._note_failure(
                    exc,
                    "placement scores",
                    f"{itype} x{count}",
                    hint=" or pass --min-score 0",
                )
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

    def quota_usage(self, keys: list[QuotaKey]) -> dict[QuotaKey, float]:
        """Fetch currently-consumed spot vCPUs per quota bucket.

        Quota *limits* only bound what could run; planners need limit minus
        what is already running to know the true headroom. One fetch per
        distinct region covers every bucket in it, and usage moves much
        faster than limits, so it is cached under a short 5-minute TTL. A
        failed region degrades to 0.0 for its keys (plus a warning) — the
        planner then just trusts the raw limit, as before.
        """
        if not keys:
            return {}
        regions = sorted({key.region for key in keys})

        def one(region: str) -> dict[str, float]:
            try:
                return self.cache.get_or(
                    f"aws-usage:{region}",
                    lambda: self._fetch_usage(region),
                    ttl=300.0,
                )
            except Exception as exc:
                self._note_failure(exc, "spot usage", region)
                return {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            per_region = dict(zip(regions, pool.map(one, regions)))
        return {
            key: per_region[key.region].get(key.code, 0.0) for key in keys
        }

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

    def _fetch_usage(self, region: str) -> dict[str, float]:
        """Sum running spot vCPUs per quota code in one region.

        Paginated describe-instances over pending/running instances; only
        spot instances count against spot quota buckets. Raises on any
        failure (never cached).
        """
        import boto3

        client = boto3.client("ec2", region_name=region)
        pages = client.get_paginator("describe_instances").paginate(
            Filters=[
                {"Name": "instance-state-name", "Values": ["pending", "running"]}
            ]
        )
        usage: dict[str, float] = {}
        for page in pages:
            for reservation in page.get("Reservations", []):
                for inst in reservation.get("Instances", []):
                    if inst.get("InstanceLifecycle") != "spot":
                        continue
                    cpu = inst.get("CpuOptions") or {}
                    vcpus = float(
                        cpu.get("CoreCount", 0) * cpu.get("ThreadsPerCore", 0)
                    )
                    code = self.quota_code(inst["InstanceType"], use_spot=True)
                    if code is not None:
                        usage[code] = usage.get(code, 0.0) + vcpus
        return usage
