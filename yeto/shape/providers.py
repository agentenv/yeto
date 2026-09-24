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
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from yeto.shape.cache import TTLCache

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (catalog is light)
    from yeto.shape.catalog import Offering

# One capacity question: (sky GPU name, GPUs per node, region/location).
SignalAsk = tuple[str, int, str]


@runtime_checkable
class CloudSignals(Protocol):
    """What the shaper needs from every non-AWS cloud.

    AWS is deliberately NOT behind this protocol: its signals are
    three-dimensional (quota limit, quota usage, placement score) and the
    planner treats quota as a hard cap while treating the pseudo-score as a
    probability. Everything else — RunPod, Nebius, Verda, Modal — fits one
    shape: "do I have credentials", "what can I buy", "can I get it now".

    `scores` returns, per ask, an integer 0-10 on the same scale as AWS
    placement scores, or None. The two MUST NOT be conflated: 0 is a
    *measurement* (sold out) and rejects the shape without a warning; None
    means "could not find out" and follows the assumed/strict policy.
    """

    name: str
    warnings: list[str]

    def available(self) -> bool: ...

    def credential_hint(self) -> str: ...

    def known_gpus(self) -> frozenset[str]: ...

    def offerings(
        self, regions: set[str] | None, gpus: list[str] | None, cache: Any
    ) -> list[Offering] | None: ...

    def scores(self, asks: list[SignalAsk]) -> dict[SignalAsk, int | None]: ...

    def island_cap(self, score: int) -> int | None: ...


# Pseudo-score -> max islands of one shape. Stock-style signals are coarse,
# so the caps are deliberately blunt: plenty (9) is uncapped, medium (6)
# supports a few, low (3) means grab one if it wins.
_STOCK_CAPS: dict[int, int | None] = {9: None, 6: 4, 3: 1}


class BaseCloudSignals:
    """Shared plumbing for CloudSignals implementations: a cache handle, a
    thread-safe warning list, and the default answers (no own catalog, the
    blunt stock-cap table). Subclasses set `name` and implement
    `available`, `credential_hint`, and `scores`."""

    name = ""

    def __init__(self, cache: Any, max_workers: int = 8) -> None:
        self._cache = cache
        self._max_workers = max_workers
        self.warnings: list[str] = []
        self._warn_lock = threading.Lock()

    def _warn(self, msg: str) -> None:
        with self._warn_lock:
            if msg not in self.warnings:
                self.warnings.append(msg)

    def available(self) -> bool:
        raise NotImplementedError

    def credential_hint(self) -> str:
        raise NotImplementedError

    def known_gpus(self) -> frozenset[str]:
        return frozenset()

    def offerings(
        self, regions: set[str] | None, gpus: list[str] | None, cache: Any
    ) -> list[Offering] | None:
        return None  # ride SkyPilot's catalog

    def scores(self, asks: list[SignalAsk]) -> dict[SignalAsk, int | None]:
        raise NotImplementedError

    def island_cap(self, score: int) -> int | None:
        return _STOCK_CAPS.get(score, 1)

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


class RunPodProviders(BaseCloudSignals):
    """Stock signals for RunPod secure-cloud pods (parallel + cached).

    Stock moves faster than quota ceilings, so entries cache for 15 minutes
    rather than the default hour. RunPod is one global pool: the region in
    an ask is ignored and every region of a (GPU, count) gets the same
    score. The catalog still comes from sky (`offerings` -> None).
    """

    name = "runpod"

    def available(self) -> bool:
        return runpod_available()

    def credential_hint(self) -> str:
        return "RUNPOD_API_KEY or ~/.runpod/config.toml (run `runpod config`)"

    def known_gpus(self) -> frozenset[str]:
        return frozenset(_RUNPOD_GPU_IDS)

    def scores(self, asks: list[SignalAsk]) -> dict[SignalAsk, int | None]:
        by_shape = self.stock_scores(sorted({(gpu, count) for gpu, count, _ in asks}))
        return {ask: by_shape.get((ask[0], ask[1])) for ask in asks}

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


# --- Modal -------------------------------------------------------------------
#
# Modal is serverless: no catalog API, no stock API, no spot. Prices are a
# table copied from modal.com/pricing (dated; the planner warns when it is
# older than 90 days), a container is priced as GPU + the CPU and memory
# the runner reserves for it, and a region hint multiplies the whole
# thing. Availability is a constant "will run, may queue" pseudo-score.

MODAL_PRICES_RECORDED = "2026-09-23"
# $/second, as printed on the pricing page (per GPU; per physical core;
# per GiB). Multiply by 3600 for $/hr.
MODAL_GPU_USD_PER_SECOND: dict[str, float] = {
    "B200": 0.001736,
    "H200": 0.001261,
    "H100": 0.001097,
    "A100-80GB": 0.000694,
    "L40S": 0.000542,
    "A10G": 0.000306,
    "L4": 0.000222,
    "T4": 0.000164,
}
MODAL_CPU_USD_PER_CORE_SECOND = 0.0000131
MODAL_MEMORY_USD_PER_GIB_SECOND = 0.00000222
MODAL_BROAD_REGIONS = frozenset({"us", "eu", "ap"})
MODAL_NARROW_REGIONS = frozenset({
    "us-east", "us-central", "us-south", "us-west",
    "eu-west", "eu-north", "eu-south",
    "ap-northeast", "ap-southeast", "ap-south", "ap-melbourne", "jp", "au",
    "uk", "ca", "me", "sa", "af", "mx",
})
MODAL_REGION_MULTIPLIER = {"broad": 1.15, "narrow": 1.75}
# Modal has no stock signal; it autoscales and may queue. Above the
# default --min-score gate (7) so it competes on price, never "sold out".
MODAL_ASSUMED_SCORE = 8
MODAL_PRICE_TABLE_MAX_AGE_DAYS = 90


def modal_region_multiplier(region: str | None) -> float:
    if not region:
        return 1.0
    if region in MODAL_BROAD_REGIONS:
        return MODAL_REGION_MULTIPLIER["broad"]
    if region in MODAL_NARROW_REGIONS:
        return MODAL_REGION_MULTIPLIER["narrow"]
    raise ValueError(
        f"unknown Modal region {region!r}; broad: {', '.join(sorted(MODAL_BROAD_REGIONS))}; "
        f"narrow: {', '.join(sorted(MODAL_NARROW_REGIONS))}"
    )


def modal_node_price_per_hour(gpu: str, gpus_per_node: int, region: str | None = None) -> float:
    """GPU + reserved CPU + reserved memory for one container, per hour,
    times the region multiplier."""
    from yeto.modal_runner import MODAL_CPU_CORES_PER_GPU, MODAL_MEMORY_GIB_PER_GPU

    per_second = (
        gpus_per_node * MODAL_GPU_USD_PER_SECOND[gpu]
        + gpus_per_node * MODAL_CPU_CORES_PER_GPU * MODAL_CPU_USD_PER_CORE_SECOND
        + gpus_per_node * MODAL_MEMORY_GIB_PER_GPU * MODAL_MEMORY_USD_PER_GIB_SECOND
    )
    return round(per_second * 3600 * modal_region_multiplier(region), 4)


def modal_price_table_age_days(today: Any = None) -> int:
    import datetime as dt

    recorded = dt.date.fromisoformat(MODAL_PRICES_RECORDED)
    today = today or dt.date.today()
    return (today - recorded).days


class ModalSignals(BaseCloudSignals):
    """Modal: static priced catalog, constant availability, no network."""

    name = "modal"

    def available(self) -> bool:
        from yeto.modal_runner import modal_available

        return modal_available()

    def credential_hint(self) -> str:
        from yeto.modal_runner import modal_credential_hint

        return modal_credential_hint()

    def known_gpus(self) -> frozenset[str]:
        return frozenset(MODAL_GPU_USD_PER_SECOND)

    def offerings(
        self, regions: set[str] | None, gpus: list[str] | None, cache: Any
    ) -> list[Offering] | None:
        from yeto import launcher
        from yeto.modal_runner import MODAL_CPU_CORES_PER_GPU, MODAL_FULL_NODE, MODAL_GPUS
        from yeto.shape.catalog import PEAK_TFLOPS_BF16, Offering

        age = modal_price_table_age_days()
        if age > MODAL_PRICE_TABLE_MAX_AGE_DAYS:
            self._warn(
                f"modal: the static price table was recorded {MODAL_PRICES_RECORDED} "
                f"({age} days ago); check modal.com/pricing before trusting Modal costs"
            )
        # An unpinned island has region "" (no surcharge, no `@region` in
        # the launch key); a pinned one carries its hint and multiplier.
        region_list: list[str] = sorted(regions) if regions else [""]
        for region in region_list:
            mult = modal_region_multiplier(region)  # unknown region -> ValueError before planning
            if region:
                kind = "broad" if region in MODAL_BROAD_REGIONS else "narrow"
                self._warn(
                    f"modal: region {region} pinned at {mult}x ({kind} region surcharge, "
                    f"modal.com/pricing as of {MODAL_PRICES_RECORDED})"
                )
        want = set(gpus) if gpus else None
        rows: list[Offering] = []
        for gpu in sorted(MODAL_GPU_USD_PER_SECOND):
            if gpu not in MODAL_GPUS or gpu not in PEAK_TFLOPS_BF16 or (want and gpu not in want):
                continue
            counts = (1, 2, 4, 8) if gpu in MODAL_FULL_NODE else (1, 2, 4)
            for count in counts:
                for region in region_list:
                    price = modal_node_price_per_hour(gpu, count, region or None)
                    rows.append(
                        Offering(
                            gpu=gpu,
                            instance_type=f"{MODAL_GPUS[gpu]}:{count}",
                            gpus_per_node=count,
                            vcpus=MODAL_CPU_CORES_PER_GPU * count,
                            region=region,
                            spot_price=price,  # no spot on Modal: one price
                            on_demand_price=price,
                            gpu_mem_gb=launcher.GPU_MEM_GB[gpu],
                            cloud="modal",
                        )
                    )
        return rows

    def scores(self, asks: list[SignalAsk]) -> dict[SignalAsk, int | None]:
        return {ask: MODAL_ASSUMED_SCORE for ask in asks}

    def island_cap(self, score: int) -> int | None:
        return None  # autoscaling: no per-shape stock ceiling


# Cloud name -> factory taking the shared TTLCache. `yeto shape` consults
# this for `--clouds` (unknown names are an error) and for the default
# fleet (every cloud whose `available()` is true). Adding a cloud means
# adding one CloudSignals implementation here — nothing in plan.py
# should need to know the cloud's name.
CLOUD_SIGNALS: dict[str, Callable[[TTLCache], CloudSignals]] = {
    "runpod": RunPodProviders,
    "modal": ModalSignals,
}
