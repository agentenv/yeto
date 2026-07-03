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


# EC2 vCPU quota codes by instance-family prefix (spot verified against
# `aws service-quotas list-service-quotas` 2026-07; longest prefix wins so
# "dl1" resolves to DL, not the standard D bucket).
_SPOT_FAMILY_CODES = {
    "dl": "L-85EED4F7",  # All DL Spot
    "inf": "L-B5D1601B",  # All Inf Spot
    "trn": "L-6B0D517C",  # All Trn Spot
    "vt": "L-3819A6DF",  # All G and VT Spot
    "p": "L-7212CCBC",  # All P Spot (p2..p5en share one bucket)
    "g": "L-3819A6DF",  # All G and VT Spot
    "f": "L-88CF9481",  # All F Spot
    "x": "L-E3A00192",  # All X Spot
    **{k: "L-34B43A08" for k in ("a", "c", "d", "h", "i", "m", "r", "t", "z")},
}
_ONDEMAND_FAMILY_CODES = {
    "p": "L-417A185B",  # Running On-Demand P instances
    "g": "L-DB2E81BA",  # Running On-Demand G and VT instances
}


def _family_quota_code(instance_type: str, use_spot: bool) -> str | None:
    family = instance_type.split(".")[0].lower()
    table = _SPOT_FAMILY_CODES if use_spot else _ONDEMAND_FAMILY_CODES
    for prefix in sorted(table, key=len, reverse=True):
        if family.startswith(prefix):
            return table[prefix]
    return None


# sky accelerator name -> RunPod GPU type id (their GraphQL identifiers).
_RUNPOD_GPU_IDS = {
    "A100-80GB": "NVIDIA A100 80GB PCIe",
    "H100": "NVIDIA H100 80GB HBM3",
    "H200": "NVIDIA H200",
    "B200": "NVIDIA B200",
    "L40S": "NVIDIA L40S",
    "L4": "NVIDIA L4",
    "A40": "NVIDIA A40",
}
# RunPod has no quotas — the binding constraint is machine stock. Their API
# reports a coarse stockStatus per (GPU type, count); map it onto the same
# 1-10 pseudo-score scale the AWS placement score uses so one gate serves
# both clouds. Null stock (sold out / not offered at that count) is a
# *measured* zero, not an unknown.
STOCK_SCORE = {"High": 9, "Medium": 6, "Low": 3}


def runpod_available() -> bool:
    """True when RunPod credentials exist (config file or env var)."""
    import os

    return bool(os.environ.get("RUNPOD_API_KEY")) or os.path.exists(
        os.path.expanduser("~/.runpod/config.toml")
    )


class RunPodProviders:
    """Stock signals for RunPod secure-cloud pods (parallel + cached).

    Stock moves faster than quota ceilings, so entries cache for 15 minutes
    rather than the default hour.
    """

    def __init__(self, cache, max_workers: int = 8):
        self._cache = cache
        self._max_workers = max_workers
        self.warnings: list[str] = []
        self._warn_lock = threading.Lock()

    def _warn(self, msg: str) -> None:
        with self._warn_lock:
            if msg not in self.warnings:
                self.warnings.append(msg)

    def stock_scores(self, asks: list[tuple[str, int]]) -> dict[tuple[str, int], int | None]:
        """(sky gpu name, gpus per pod) -> pseudo-score.

        0 = measured out-of-stock; None = signal unavailable (API failure /
        unmapped GPU) — callers apply the same assumed/strict policy as for
        AWS placement scores.
        """
        out: dict[tuple[str, int], int | None] = {}

        def one(ask: tuple[str, int]) -> None:
            gpu, count = ask
            gpu_id = _RUNPOD_GPU_IDS.get(gpu)
            if gpu_id is None:
                self._warn(f"no RunPod GPU id mapping for {gpu}; stock unknown")
                out[ask] = None
                return
            try:
                status = self._cache.get_or(
                    f"runpod-stock:{gpu_id}:{count}",
                    lambda: self._fetch_stock(gpu_id, count),
                    ttl=900,
                )
            except Exception as exc:  # noqa: BLE001 - degrade, don't crash planning
                self._warn(f"runpod stock check failed for {gpu} x{count}: {exc}")
                out[ask] = None
                return
            out[ask] = STOCK_SCORE.get(status, 0) if status is not None else 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            list(pool.map(one, asks))
        return out

    def _fetch_stock(self, gpu_id: str, count: int) -> str | None:
        """One GraphQL lookup: secure-cloud stockStatus for `count` GPUs.

        Returns RunPod's literal status string ("High"/"Medium"/"Low") or
        None when the type/count combination is not stocked at all.
        """
        import json
        import urllib.request

        query = {
            "query": (
                'query { gpuTypes(input: {id: "%s"}) { lowestPrice(input: '
                "{gpuCount: %d, secureCloud: true}) { stockStatus } } }"
            )
            % (gpu_id, count)
        }
        req = urllib.request.Request(
            "https://api.runpod.io/graphql",
            data=json.dumps(query).encode(),
            headers={
                "Content-Type": "application/json",
                # Query-param auth 403s on current RunPod; Bearer works.
                "Authorization": f"Bearer {_runpod_api_key()}",
                # Cloudflare rejects the default Python-urllib user agent.
                "User-Agent": "yeto-shape/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.load(resp)
        types = payload.get("data", {}).get("gpuTypes") or []
        if not types:
            return None
        price = types[0].get("lowestPrice") or {}
        return price.get("stockStatus")


def _runpod_api_key() -> str:
    """API key from the env or ~/.runpod/config.toml (sky's auth source)."""
    import os
    import re

    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    path = os.path.expanduser("~/.runpod/config.toml")
    with open(path, encoding="utf-8") as f:
        m = re.search(r'api_key\s*=\s*"([^"]+)"', f.read())
    if not m:
        raise RuntimeError(f"no api_key found in {path}")
    return m.group(1)


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
        """Map an instance type to its EC2 vCPU quota code.

        SkyPilot ships an instance-type -> quota-code table which we try
        first, but it has gaps (no p5 rows as of sky 0.12 — that gap once
        made a planner treat P5 spot as unlimited). The fallback derives the
        code from the instance family letter(s), matching AWS's actual quota
        buckets ("All P Spot Instance Requests" covers p2..p5en alike).
        None means genuinely unmappable; callers must treat that as
        un-launchable, not unlimited.
        """
        try:
            from sky.catalog import aws_catalog

            code = aws_catalog.get_quota_code(instance_type, use_spot)
            if code:
                return code
        except Exception:
            pass
        return _family_quota_code(instance_type, use_spot)

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
