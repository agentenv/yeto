"""Tests for the AWS quota/score/usage providers (no network calls)."""

from __future__ import annotations

import pytest

from yeto.shape.cache import TTLCache
from yeto.shape.providers import AwsProviders, QuotaKey, credentials_available


def _client_error(code: str):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code}}, "op")


@pytest.fixture
def providers(tmp_path):
    return AwsProviders(cache=TTLCache(path=tmp_path / "c.json"), max_workers=4)


def test_quotas_fan_out_failure_and_caching(providers, monkeypatch):
    keys = [
        QuotaKey("us-east-1", "L-7212CCBC"),
        QuotaKey("us-west-2", "L-7212CCBC"),
        QuotaKey("eu-west-1", "L-BAD"),
    ]
    calls = {"n": 0}

    def fake_fetch(key: QuotaKey) -> float:
        calls["n"] += 1
        if key.code == "L-BAD":
            raise RuntimeError("access denied")
        return {"us-east-1": 96.0, "us-west-2": 0.0}[key.region]

    monkeypatch.setattr(providers, "_fetch_quota", fake_fetch)

    result = providers.quotas(keys)
    assert result == {keys[0]: 96.0, keys[1]: 0.0, keys[2]: None}
    assert calls["n"] == 3

    # Successes are served from cache; the failure was not cached, so it is
    # retried (and fails again -> None).
    result2 = providers.quotas(keys)
    assert result2 == result
    assert calls["n"] == 4  # only the failing key re-fetched


def test_quotas_empty(providers):
    assert providers.quotas([]) == {}


def test_placement_scores_flatten_missing_and_cache(providers, monkeypatch):
    regions = ["us-east-1", "us-west-2", "eu-west-1"]
    asks = [("p4d.24xlarge", 4), ("g6.48xlarge", 2)]
    calls = {"n": 0}

    def fake_fetch(itype, count, regs):
        calls["n"] += 1
        assert regs == regions
        # eu-west-1 deliberately absent from the response.
        return {"us-east-1": 7, "us-west-2": 3}

    monkeypatch.setattr(providers, "_fetch_scores", fake_fetch)

    result = providers.placement_scores(asks, regions)
    assert result[("p4d.24xlarge", 4, "us-east-1")] == 7
    assert result[("g6.48xlarge", 2, "us-west-2")] == 3
    assert result[("p4d.24xlarge", 4, "eu-west-1")] is None
    assert len(result) == len(asks) * len(regions)
    assert calls["n"] == 2  # one batched call per ask, not per region

    result2 = providers.placement_scores(asks, regions)
    assert result2 == result
    assert calls["n"] == 2  # served from cache


def test_placement_scores_failed_call_yields_none(providers, monkeypatch):
    regions = ["us-east-1", "us-west-2"]
    calls = {"n": 0}

    def fake_fetch(itype, count, regs):
        calls["n"] += 1
        raise RuntimeError("throttled")

    monkeypatch.setattr(providers, "_fetch_scores", fake_fetch)

    result = providers.placement_scores([("p4d.24xlarge", 1)], regions)
    assert result == {
        ("p4d.24xlarge", 1, "us-east-1"): None,
        ("p4d.24xlarge", 1, "us-west-2"): None,
    }
    # Failure is not cached: a second call retries the fetch.
    providers.placement_scores([("p4d.24xlarge", 1)], regions)
    assert calls["n"] == 2


def test_quota_usage_fan_out_and_bucket_mapping(providers, monkeypatch):
    keys = [
        QuotaKey("us-east-1", "L-7212CCBC"),
        QuotaKey("us-east-1", "L-3819A6DF"),
        QuotaKey("us-west-2", "L-7212CCBC"),
        QuotaKey("us-west-2", "L-UNSEEN"),
    ]
    fetched: list[str] = []

    def fake_fetch(region: str) -> dict[str, float]:
        fetched.append(region)
        return {
            "us-east-1": {"L-7212CCBC": 96.0, "L-3819A6DF": 8.0},
            "us-west-2": {"L-7212CCBC": 32.0},
        }[region]

    monkeypatch.setattr(providers, "_fetch_usage", fake_fetch)

    result = providers.quota_usage(keys)
    assert result == {
        keys[0]: 96.0,
        keys[1]: 8.0,
        keys[2]: 32.0,
        keys[3]: 0.0,  # bucket absent from the region dict
    }
    assert sorted(fetched) == ["us-east-1", "us-west-2"]  # one per region
    assert providers.warnings == []


def test_quota_usage_empty(providers):
    assert providers.quota_usage([]) == {}


def test_quota_usage_failed_region_zero_and_warning(providers, monkeypatch):
    keys = [
        QuotaKey("us-east-1", "L-7212CCBC"),
        QuotaKey("eu-west-1", "L-7212CCBC"),
    ]

    def fake_fetch(region: str) -> dict[str, float]:
        if region == "eu-west-1":
            raise _client_error("AccessDenied")
        return {"L-7212CCBC": 96.0}

    monkeypatch.setattr(providers, "_fetch_usage", fake_fetch)

    result = providers.quota_usage(keys)
    assert result == {keys[0]: 96.0, keys[1]: 0.0}
    assert len(providers.warnings) == 1
    assert "eu-west-1" in providers.warnings[0]
    assert "AccessDenied" in providers.warnings[0]


def test_quota_usage_cached_with_short_ttl(providers, monkeypatch):
    class RecordingCache:
        def __init__(self):
            self.ttls: list[float | None] = []

        def get_or(self, key, fetch, ttl=None):
            self.ttls.append(ttl)
            return fetch()

    cache = RecordingCache()
    providers.cache = cache
    monkeypatch.setattr(providers, "_fetch_usage", lambda region: {})
    providers.quota_usage([QuotaKey("us-east-1", "L-7212CCBC")])
    assert cache.ttls == [300.0]


def test_throttled_scores_warn_and_yield_none(providers, monkeypatch):
    def fake_fetch(itype, count, regs):
        raise _client_error("RequestLimitExceeded")

    monkeypatch.setattr(providers, "_fetch_scores", fake_fetch)

    result = providers.placement_scores([("p5.48xlarge", 2)], ["us-east-1"])
    assert result == {("p5.48xlarge", 2, "us-east-1"): None}
    assert len(providers.warnings) == 1
    assert "throttled" in providers.warnings[0]
    assert "p5.48xlarge x2" in providers.warnings[0]
    assert "--min-score 0" in providers.warnings[0]


def test_throttled_quota_warns(providers, monkeypatch):
    def fake_fetch(key):
        raise _client_error("ThrottlingException")

    monkeypatch.setattr(providers, "_fetch_quota", fake_fetch)

    result = providers.quotas([QuotaKey("us-east-1", "L-7212CCBC")])
    assert result == {QuotaKey("us-east-1", "L-7212CCBC"): None}
    assert len(providers.warnings) == 1
    assert "throttled" in providers.warnings[0]
    assert "us-east-1" in providers.warnings[0]


def test_non_throttle_client_error_warns_with_code(providers, monkeypatch):
    def fake_fetch(key):
        raise _client_error("UnauthorizedOperation")

    monkeypatch.setattr(providers, "_fetch_quota", fake_fetch)

    providers.quotas([QuotaKey("us-east-1", "L-7212CCBC")])
    assert len(providers.warnings) == 1
    assert "UnauthorizedOperation" in providers.warnings[0]
    assert "throttled" not in providers.warnings[0]


def test_boto3_missing_single_warning(providers, monkeypatch):
    def fake_fetch(key):
        raise ImportError("No module named 'boto3'")

    monkeypatch.setattr(providers, "_fetch_quota", fake_fetch)

    keys = [
        QuotaKey("us-east-1", "L-7212CCBC"),
        QuotaKey("us-west-2", "L-7212CCBC"),
    ]
    result = providers.quotas(keys)
    assert result == {keys[0]: None, keys[1]: None}
    assert providers.warnings == ["boto3 unavailable; AWS signals disabled"]


def test_credentials_available_true(monkeypatch):
    class FakeSession:
        def get_credentials(self):
            return object()

    import boto3

    monkeypatch.setattr(boto3.session, "Session", FakeSession)
    assert credentials_available() is True


def test_credentials_available_false_without_credentials(monkeypatch):
    class FakeSession:
        def get_credentials(self):
            return None

    import boto3

    monkeypatch.setattr(boto3.session, "Session", FakeSession)
    assert credentials_available() is False


def test_credentials_available_false_on_error(monkeypatch):
    class FakeSession:
        def get_credentials(self):
            raise RuntimeError("no config")

    import boto3

    monkeypatch.setattr(boto3.session, "Session", FakeSession)
    assert credentials_available() is False


def test_quota_code(providers):
    try:
        from sky.catalog import aws_catalog  # noqa: F401

        sky_available = True
    except Exception:
        sky_available = False

    if sky_available:
        assert providers.quota_code("p4d.24xlarge", True) == "L-7212CCBC"
    assert providers.quota_code("not-a-real-instance-type", True) is None


def test_quota_code_survives_import_failure(providers, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name.startswith("sky"):
            raise ImportError("sky not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    # Without sky, the family fallback still answers for known families...
    assert providers.quota_code("p4d.24xlarge", True) == "L-7212CCBC"
    # ...and unknown families stay None (callers fail closed).
    assert providers.quota_code("uq99.8xlarge", True) is None


def test_family_quota_code_fallback_covers_sky_gaps():
    from yeto.shape.providers import _family_quota_code

    # sky 0.12 has no p5 rows; the family fallback must map all P variants
    # to the shared "All P Spot" bucket (there is no separate P5 spot quota).
    assert _family_quota_code("p5.48xlarge", True) == "L-7212CCBC"
    assert _family_quota_code("p5en.48xlarge", True) == "L-7212CCBC"
    assert _family_quota_code("p4d.24xlarge", True) == "L-7212CCBC"
    # Longest-prefix wins: dl is DL, not the standard D bucket.
    assert _family_quota_code("dl1.24xlarge", True) == "L-85EED4F7"
    assert _family_quota_code("d3.xlarge", True) == "L-34B43A08"
    assert _family_quota_code("g6e.48xlarge", True) == "L-3819A6DF"
    assert _family_quota_code("vt1.24xlarge", True) == "L-3819A6DF"
    # Unknown family -> None (callers fail closed).
    assert _family_quota_code("uq99.8xlarge", True) is None
    # On-demand mappings for the families we can name.
    assert _family_quota_code("p5.48xlarge", False) == "L-417A185B"
    assert _family_quota_code("m7i.large", False) is None


# --- CloudSignals protocol / registry ---------------------------------------


def test_registry_lists_runpod_and_instances_satisfy_the_protocol(tmp_path):
    from yeto.shape.providers import CLOUD_SIGNALS, CloudSignals

    assert "runpod" in CLOUD_SIGNALS
    sig = CLOUD_SIGNALS["runpod"](TTLCache(path=tmp_path / "c.json"))
    assert isinstance(sig, CloudSignals) and sig.name == "runpod"
    assert sig.offerings(None, None, None) is None  # rides sky's catalog
    assert "H200" in sig.known_gpus()
    assert "RUNPOD_API_KEY" in sig.credential_hint()


def test_runpod_available_from_env_or_config(monkeypatch, tmp_path):
    from yeto.shape.providers import RunPodProviders

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    sig = RunPodProviders(TTLCache(path=tmp_path / "c.json"))
    assert not sig.available()
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    assert sig.available()
    monkeypatch.delenv("RUNPOD_API_KEY")
    (tmp_path / ".runpod").mkdir()
    (tmp_path / ".runpod" / "config.toml").write_text('api_key = "k"\n')
    assert sig.available()


def test_runpod_scores_fan_out_regions_and_dedupe_shapes(monkeypatch, tmp_path):
    from yeto.shape.providers import RunPodProviders

    sig = RunPodProviders(TTLCache(path=tmp_path / "c.json"))
    calls = []

    def fake_fetch(gpu_id, count):
        calls.append((gpu_id, count))
        return {"NVIDIA H100 80GB HBM3": "High", "NVIDIA B200": None}[gpu_id]

    monkeypatch.setattr(sig, "_fetch_stock", fake_fetch)
    asks = [("H100", 8, "CA"), ("H100", 8, "US"), ("B200", 8, "CA"), ("T4", 1, "CA")]
    out = sig.scores(asks)
    # Same (gpu, count) -> one fetch, every region gets that score; null
    # stock is a measured 0; an unmapped GPU is None plus a warning.
    assert out == {
        ("H100", 8, "CA"): 9,
        ("H100", 8, "US"): 9,
        ("B200", 8, "CA"): 0,
        ("T4", 1, "CA"): None,
    }
    assert sorted(calls) == [("NVIDIA B200", 8), ("NVIDIA H100 80GB HBM3", 8)]
    assert any("no RunPod GPU id mapping for T4" in w for w in sig.warnings)


def test_island_cap_table():
    from yeto.shape.providers import RunPodProviders

    sig = RunPodProviders(cache=None)
    assert sig.island_cap(9) is None
    assert sig.island_cap(6) == 4
    assert sig.island_cap(3) == 1
    assert sig.island_cap(8) == 1  # unknown levels are treated conservatively


# --- Modal -------------------------------------------------------------------


def test_modal_registered_and_credentials(monkeypatch, tmp_path):
    from yeto.shape.providers import CLOUD_SIGNALS, ModalSignals

    assert CLOUD_SIGNALS["modal"] is ModalSignals
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        monkeypatch.delenv(var, raising=False)
    sig = ModalSignals(cache=None)
    assert not sig.available() and "modal token new" in sig.credential_hint()
    (tmp_path / ".modal.toml").write_text("[default]\ntoken_id='x'\n")
    assert sig.available()


def test_modal_price_composes_gpu_cpu_memory_and_region(monkeypatch):
    from yeto.shape import providers as p

    base = p.modal_node_price_per_hour("H100", 8)
    gpu_only = 8 * p.MODAL_GPU_USD_PER_SECOND["H100"] * 3600
    cpu = 8 * 4 * p.MODAL_CPU_USD_PER_CORE_SECOND * 3600
    mem = 8 * 32 * p.MODAL_MEMORY_USD_PER_GIB_SECOND * 3600
    assert base == pytest.approx(gpu_only + cpu + mem, abs=1e-3)
    assert base > gpu_only  # CPU and memory are in the price
    assert p.modal_node_price_per_hour("H100", 8, "us") == pytest.approx(base * 1.15, abs=1e-3)
    assert p.modal_node_price_per_hour("H100", 8, "us-west") == pytest.approx(base * 1.75, abs=1e-3)
    with pytest.raises(ValueError, match="unknown Modal region 'mars'"):
        p.modal_region_multiplier("mars")


def test_modal_offerings_unpinned_pinned_and_shape_rules(monkeypatch):
    from yeto.shape.providers import ModalSignals

    sig = ModalSignals(cache=None)
    rows = sig.offerings(None, None, None)
    assert {o.region for o in rows} == {""}  # unpinned: no region, no surcharge
    h100 = [o for o in rows if o.gpu == "H100"]
    assert sorted(o.gpus_per_node for o in h100) == [1, 2, 4, 8]
    assert all(o.spot_price == o.on_demand_price for o in rows)  # no spot on Modal
    eight = next(o for o in h100 if o.gpus_per_node == 8)
    assert eight.instance_type == "H100:8" and eight.vcpus == 32 and eight.gpu_mem_gb == 80
    l4 = [o for o in rows if o.gpu == "L4"]
    assert sorted(o.gpus_per_node for o in l4) == [1, 2, 4]  # not a whole-node GPU
    assert not sig.warnings  # unpinned: nothing to disclose
    pinned = sig.offerings({"us", "eu-north"}, ["H100"], None)
    assert {o.region for o in pinned} == {"us", "eu-north"}
    by_region = {o.region: o.spot_price for o in pinned if o.gpus_per_node == 8}
    assert by_region["eu-north"] == pytest.approx(eight.spot_price * 1.75, abs=1e-3)
    assert by_region["us"] == pytest.approx(eight.spot_price * 1.15, abs=1e-3)
    # Pinning is disclosed with the multiplier and its source.
    assert sorted(sig.warnings) == [
        "modal: region eu-north pinned at 1.75x (narrow region surcharge, modal.com/pricing as of 2026-09-23)",
        "modal: region us pinned at 1.15x (broad region surcharge, modal.com/pricing as of 2026-09-23)",
    ]
    with pytest.raises(ValueError, match="unknown Modal region"):
        sig.offerings({"mars"}, None, None)
    assert sig.scores([("H100", 8, "")]) == {("H100", 8, ""): 8}
    assert sig.island_cap(8) is None


def test_modal_price_table_staleness_warning(monkeypatch):
    import datetime as dt

    from yeto.shape import providers as p

    assert p.modal_price_table_age_days(dt.date(2026, 9, 23)) == 0
    monkeypatch.setattr(p, "modal_price_table_age_days", lambda today=None: 120)
    sig = p.ModalSignals(cache=None)
    sig.offerings(None, ["H100"], None)
    assert any("recorded 2026-09-23 (120 days ago)" in w for w in sig.warnings)


# --- Nebius ------------------------------------------------------------------


def test_nebius_registered_and_gpu_table_is_plannable():
    from yeto.shape.catalog import PEAK_TFLOPS_BF16
    from yeto.shape.providers import _NEBIUS_PLATFORMS, CLOUD_SIGNALS, NebiusSignals

    assert CLOUD_SIGNALS["nebius"] is NebiusSignals
    assert set(_NEBIUS_PLATFORMS) <= set(PEAK_TFLOPS_BF16)
    assert NebiusSignals(cache=None).known_gpus() == frozenset(_NEBIUS_PLATFORMS)


def test_nebius_available_needs_token_and_tenant(monkeypatch, tmp_path):
    from yeto.shape.providers import NebiusSignals

    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("NEBIUS_IAM_TOKEN", "NEBIUS_TENANT_ID"):
        monkeypatch.delenv(var, raising=False)
    sig = NebiusSignals(cache=None)
    assert not sig.available()
    assert "NEBIUS_IAM_TOKEN.txt" in sig.credential_hint() and "NEBIUS_TENANT_ID.txt" in sig.credential_hint()
    (tmp_path / ".nebius").mkdir()
    (tmp_path / ".nebius" / "NEBIUS_IAM_TOKEN.txt").write_text("tok\n")
    assert not sig.available()  # token alone is not enough
    (tmp_path / ".nebius" / "NEBIUS_TENANT_ID.txt").write_text("tenant-1\n")
    assert sig.available()
    # Env vars work without the files.
    monkeypatch.setenv("HOME", str(tmp_path / "elsewhere"))
    assert not sig.available()
    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "t")
    monkeypatch.setenv("NEBIUS_TENANT_ID", "x")
    assert sig.available()


def test_nebius_project_ids_from_sky_config(tmp_path):
    from yeto.shape.providers import nebius_project_ids

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "nebius:\n  region_configs:\n    eu-north1:\n      project_id: project-e00aaa\n"
        "      fabric: fabric-3\n    us-central1:\n      subnet_id: only-a-subnet\n"
    )
    assert nebius_project_ids(str(cfg)) == {"eu-north1": "project-e00aaa"}
    assert nebius_project_ids(str(tmp_path / "missing.yaml")) == {}


def _advice(region, platform, preset, gpus, level, available, kind="preemptible"):
    return {
        "spec": {
            "region": region,
            "computeInstance": {
                "platform": platform,
                "preset": {"name": preset, "resources": {"gpuCount": gpus}},
            },
        },
        "status": {kind: {"availabilityLevel": f"AVAILABILITY_LEVEL_{level}", "available": available}},
    }


def test_nebius_scores_from_capacity_advice(monkeypatch, tmp_path):
    from yeto.shape.catalog import Offering
    from yeto.shape.providers import NebiusSignals

    sig = NebiusSignals(TTLCache(path=tmp_path / "c.json"))
    monkeypatch.setenv("NEBIUS_TENANT_ID", "tenant-1")
    rows = [
        Offering("H100", "gpu-h100-sxm_8gpu-128vcpu-1600gb", 8, 128, "eu-north1", 17.2, 30.8, 80, cloud="nebius"),
        Offering("H200", "gpu-h200-sxm_8gpu-128vcpu-1600gb", 8, 128, "eu-north1", 19.6, 36.0, 141, cloud="nebius"),
        Offering("H200", "gpu-h200-sxm_8gpu-128vcpu-1600gb", 8, 128, "us-central1", 19.6, 36.0, 141, cloud="nebius"),
        Offering("B200", "gpu-b200-sxm_8gpu-160vcpu-1792gb", 8, 160, "us-central1", 31.6, 57.2, 180, cloud="nebius"),
        Offering("L40S", "gpu-l40s-a_1gpu-8vcpu-32gb", 1, 8, "eu-north1", 0.75, 1.55, 48, cloud="nebius"),
    ]
    monkeypatch.setattr("yeto.shape.catalog.list_offerings", lambda regions, gpus, cache, clouds: rows)
    got = sig.offerings({"eu-north1", "us-central1"}, None, None)
    assert got == rows
    calls = {"n": 0}

    def fake_fetch():
        calls["n"] += 1
        return [
            _advice("eu-north1-a", "gpu-h100-sxm", "8gpu-128vcpu-1600gb", 8, "HIGH", 12),  # zone suffix
            _advice("eu-north1-a", "gpu-h200-sxm", "8gpu-128vcpu-1600gb", 8, "MEDIUM", 2),
            _advice("us-central1-b", "gpu-h200-sxm", "8gpu-128vcpu-1600gb", 8, "HIGH", 0),  # 0 units wins
            _advice("us-central1-b", "gpu-b200-sxm", "other-preset-name", 8, "UNKNOWN", 5),  # matched by gpuCount
        ]

    monkeypatch.setattr(sig, "_fetch_advice", fake_fetch)
    asks = [
        ("H100", 8, "eu-north1"),
        ("H200", 8, "eu-north1"),
        ("H200", 8, "us-central1"),
        ("B200", 8, "us-central1"),
        ("L40S", 1, "eu-north1"),  # no advice entry -> unknown, not sold out
        ("H100", 8, "eu-west1"),  # no catalog row -> unknown
    ]
    assert sig.scores(asks) == {
        ("H100", 8, "eu-north1"): 9,
        ("H200", 8, "eu-north1"): 6,
        ("H200", 8, "us-central1"): 0,
        ("B200", 8, "us-central1"): 9,
        ("L40S", 1, "eu-north1"): None,
        ("H100", 8, "eu-west1"): None,
    }
    assert calls["n"] == 1  # one advice fetch serves every ask...
    sig.scores(asks[:1])
    assert calls["n"] == 1  # ...and the cache serves the next call
    assert any("no catalog row" in w for w in sig.warnings)
    assert any("lists no preemptible entry" in w for w in sig.warnings)


def test_nebius_advice_request_respects_the_page_size_limit_and_pages(monkeypatch):
    """The live API rejects pageSize > 200 with HTTP 400 (2026-09-23: every
    Nebius shape fell back to the assumed score). Ask for 200 and follow
    nextPageToken."""
    import yeto.shape.providers as prov

    monkeypatch.setattr(prov, "nebius_iam_token", lambda: "tok")
    monkeypatch.setattr(prov, "nebius_tenant_id", lambda: "tenant-x")
    seen = []

    def fake_request(method, path, token, body=None, params=None):
        seen.append(dict(params))
        if int(params["pageSize"]) > 200:
            raise RuntimeError("HTTP 400: page_size must be <= 200")
        if "pageToken" not in params:
            return {"items": [{"n": 1}], "nextPageToken": "p2"}
        return {"items": [{"n": 2}]}

    monkeypatch.setattr(prov, "_nebius_request", fake_request)
    items = prov.NebiusSignals(cache=None)._fetch_advice()
    assert items == [{"n": 1}, {"n": 2}]
    assert [p.get("pageToken") for p in seen] == [None, "p2"]
    assert all(p["parentId"] == "tenant-x" and int(p["pageSize"]) <= 200 for p in seen)


def test_nebius_capacity_api_failure_degrades_to_none(monkeypatch, tmp_path):
    from yeto.shape.providers import NebiusSignals

    sig = NebiusSignals(TTLCache(path=tmp_path / "c.json"))
    monkeypatch.setenv("NEBIUS_TENANT_ID", "tenant-1")
    monkeypatch.setattr(sig, "_fetch_advice", lambda: (_ for _ in ()).throw(RuntimeError("HTTP 503")))
    assert sig.scores([("H100", 8, "eu-north1")]) == {("H100", 8, "eu-north1"): None}
    assert any("capacity check failed" in w and "503" in w for w in sig.warnings)


def test_nebius_live_prices_need_a_project_per_region(monkeypatch, tmp_path):
    from yeto.shape.catalog import Offering
    from yeto.shape.providers import NebiusSignals

    sig = NebiusSignals(TTLCache(path=tmp_path / "c.json"))
    monkeypatch.setattr(
        "yeto.shape.providers.nebius_project_ids", lambda config_path=None: {"eu-north1": "project-1"}
    )
    calls = []

    def fake_estimate(project_id, instance_type):
        calls.append((project_id, instance_type))
        return 15.5

    monkeypatch.setattr(sig, "_fetch_estimate", fake_estimate)
    rows = [
        Offering("H100", "gpu-h100-sxm_8gpu-128vcpu-1600gb", 8, 128, "eu-north1", 17.2, 30.8, 80, cloud="nebius"),
        Offering("H100", "gpu-h100-sxm_1gpu-16vcpu-200gb", 1, 16, "eu-north1", 2.15, 3.85, 80, cloud="nebius"),
        Offering("H200", "gpu-h200-sxm_8gpu-128vcpu-1600gb", 8, 128, "us-central1", 19.6, 36.0, 141, cloud="nebius"),
    ]
    out = sig.live_spot_prices(rows)
    assert out == {
        ("gpu-h100-sxm_8gpu-128vcpu-1600gb", "eu-north1"): 15.5,
        ("gpu-h100-sxm_1gpu-16vcpu-200gb", "eu-north1"): 15.5,
    }
    assert [c[0] for c in calls] == ["project-1", "project-1"]
    assert any("no project_id for region us-central1" in w for w in sig.warnings)
    # On-demand planning never overrides (the catalog on-demand price holds).
    sig.use_spot = False
    assert sig.live_spot_prices(rows) == {}


def test_nebius_estimate_parses_hourly_total(monkeypatch, tmp_path):
    from yeto.shape.providers import NebiusSignals

    sig = NebiusSignals(TTLCache(path=tmp_path / "c.json"))
    seen = {}

    def fake_request(method, path, token, body=None, params=None):
        seen.update(method=method, path=path, body=body)
        return {
            "currency": "usd",
            "resourceCosts": [
                {"aggregationUnit": {"unit": "hour"}, "general": {"total": {"cost": "18.250000", "costRounded": "18.25"}}}
            ],
        }

    monkeypatch.setenv("NEBIUS_IAM_TOKEN", "tok")
    monkeypatch.setattr("yeto.shape.providers._nebius_request", fake_request)
    assert sig._fetch_estimate("project-1", "gpu-h100-sxm_8gpu-128vcpu-1600gb") == 18.25
    assert seen["method"] == "POST" and seen["path"].endswith("/calculator/estimate-batch")
    spec = seen["body"]["resourceSpecs"][0]["spec"]
    assert spec["metadata"]["parentId"] == "project-1"
    assert spec["spec"]["resources"] == {"platform": "gpu-h100-sxm", "preset": "8gpu-128vcpu-1600gb"}
    assert "followsSpotPrice" in spec["spec"]


def test_availability_score_mapping():
    from yeto.shape.providers import _availability_score as s

    assert s(None) == 0
    assert s({"availabilityLevel": "AVAILABILITY_LEVEL_HIGH", "available": 3}) == 9
    assert s({"availabilityLevel": "AVAILABILITY_LEVEL_MEDIUM", "available": 3}) == 6
    assert s({"availabilityLevel": "AVAILABILITY_LEVEL_LOW", "available": 1}) == 3
    assert s({"availabilityLevel": "AVAILABILITY_LEVEL_HIGH", "available": 0}) == 0
    assert s({"availabilityLevel": "AVAILABILITY_LEVEL_LIMIT_REACHED", "available": 9}) == 0
    assert s({"availabilityLevel": "AVAILABILITY_LEVEL_UNKNOWN", "available": 4}) == 9
    assert s({"availabilityLevel": "AVAILABILITY_LEVEL_UNKNOWN", "available": 1}) == 6


# --- Verda -------------------------------------------------------------------


def _verda_fixture():
    import json
    from pathlib import Path

    return json.loads((Path(__file__).parent / "fixtures" / "verda_instance_types.json").read_text())


def test_verda_registered_and_credentials(monkeypatch, tmp_path):
    from yeto.shape.providers import CLOUD_SIGNALS, VerdaSignals

    assert CLOUD_SIGNALS["verda"] is VerdaSignals
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("VERDA_CLIENT_ID", "VERDA_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)
    sig = VerdaSignals(cache=None)
    assert not sig.available() and "~/.verda/config.json" in sig.credential_hint()
    (tmp_path / ".verda").mkdir()
    (tmp_path / ".verda" / "config.json").write_text('{"client_id": "id", "client_secret": "s"}')
    assert sig.available()
    monkeypatch.setenv("HOME", str(tmp_path / "nowhere"))
    monkeypatch.setenv("VERDA_CLIENT_ID", "id")
    monkeypatch.setenv("VERDA_CLIENT_SECRET", "s")
    assert sig.available()


def test_verda_offerings_from_live_fixture(monkeypatch, tmp_path):
    # The fixture is Verda's real /instance-types response (2026-09-23,
    # 70 types). sky's catalog had 4 GPU rows and no 8-GPU machine.
    from yeto.shape.providers import VerdaSignals

    sig = VerdaSignals(TTLCache(path=tmp_path / "c.json"))
    monkeypatch.setattr(sig, "_fetch_types", _verda_fixture)
    monkeypatch.setattr(sig, "_fetch_locations", lambda: [{"code": "FIN-01"}, {"code": "FIN-03"}])
    rows = sig.offerings(None, None, None)
    by_type = {(o.instance_type, o.region): o for o in rows}
    h100 = by_type[("8H100.80S.176V", "FIN-03")]
    assert (h100.gpu, h100.gpus_per_node, h100.vcpus, h100.gpu_mem_gb, h100.cloud) == ("H100", 8, 176, 80, "verda")
    assert (h100.on_demand_price, h100.spot_price) == (27.06, 13.53)
    assert by_type[("8H200.141S.176V", "FIN-01")].gpu == "H200"
    assert by_type[("8B200.240V", "FIN-01")].gpu == "B200"
    assert {o.region for o in rows} == {"FIN-01", "FIN-03"}
    # Confidential-computing, pre-bf16 and CPU types are not training targets.
    assert not any(t.endswith(".CC") or t.startswith(("1V100", "CPU")) for t, _ in by_type)
    assert not sig.warnings
    # Allowlists narrow it; the region set passed in narrows locations.
    only = sig.offerings({"FIN-03"}, ["B200"], None)
    assert {(o.gpu, o.region) for o in only} == {("B200", "FIN-03")}


def test_verda_locations_fall_back_to_the_known_list(monkeypatch, tmp_path):
    from yeto.shape.providers import VERDA_LOCATIONS_FALLBACK, VerdaSignals

    sig = VerdaSignals(TTLCache(path=tmp_path / "c.json"))
    monkeypatch.setattr(sig, "_fetch_types", _verda_fixture)
    monkeypatch.setattr(sig, "_fetch_locations", lambda: (_ for _ in ()).throw(RuntimeError("HTTP 401")))
    rows = sig.offerings(None, ["H100"], None)
    assert {o.region for o in rows} == set(VERDA_LOCATIONS_FALLBACK)
    assert any("could not read /locations" in w and "401" in w for w in sig.warnings)


def test_verda_scores_from_bulk_availability(monkeypatch, tmp_path):
    from yeto.shape.providers import VerdaSignals

    sig = VerdaSignals(TTLCache(path=tmp_path / "c.json"))
    monkeypatch.setattr(sig, "_fetch_types", _verda_fixture)
    monkeypatch.setattr(sig, "_fetch_locations", lambda: [{"code": "FIN-01"}, {"code": "FIN-03"}, {"code": "ICL-01"}])
    sig.offerings(None, None, None)
    calls = {"n": 0}

    def fake_availability():
        calls["n"] += 1
        return {"FIN-01": ["8H100.80S.176V", "1H100.80S.30V"], "FIN-03": []}

    monkeypatch.setattr(sig, "_fetch_availability", fake_availability)
    asks = [("H100", 8, "FIN-01"), ("H100", 8, "FIN-03"), ("H100", 8, "ICL-01"), ("H200", 8, "FIN-01")]
    assert sig.scores(asks) == {
        ("H100", 8, "FIN-01"): 9,  # in stock
        ("H100", 8, "FIN-03"): 0,  # location answered, type absent -> measured sold out
        ("H100", 8, "ICL-01"): None,  # location missing from the answer -> unknown
        ("H200", 8, "FIN-01"): 0,
    }
    assert calls["n"] == 1  # one bulk call serves every ask
    sig.scores(asks[:1])
    assert calls["n"] == 1  # cached
    assert any("no entry for location ICL-01" in w for w in sig.warnings)
    # API down -> every ask None with a warning.
    broken = VerdaSignals(TTLCache(path=tmp_path / "d.json"))
    monkeypatch.setattr(broken, "_fetch_availability", lambda: (_ for _ in ()).throw(RuntimeError("HTTP 503")))
    assert broken.scores(asks[:1]) == {asks[0]: None}
    assert any("availability check failed" in w for w in broken.warnings)


def test_verda_token_and_availability_requests(monkeypatch, tmp_path):
    from yeto.shape.providers import VerdaSignals

    sig = VerdaSignals(TTLCache(path=tmp_path / "c.json"))
    monkeypatch.setenv("VERDA_CLIENT_ID", "cid")
    monkeypatch.setenv("VERDA_CLIENT_SECRET", "sec")
    seen = []

    def fake_request(method, path, token=None, body=None, params=None):
        seen.append((method, path, token, body, params))
        if path == "/oauth2/token":
            assert body == {"grant_type": "client_credentials", "client_id": "cid", "client_secret": "sec"}
            return {"access_token": "T"}
        if path == "/instance-availability":
            return [{"location_code": "FIN-01", "availabilities": ["8H100.80S.176V"]}]
        raise AssertionError(path)

    monkeypatch.setattr("yeto.shape.providers._verda_request", fake_request)
    assert sig._fetch_availability() == {"FIN-01": ["8H100.80S.176V"]}
    assert sig._fetch_availability() == {"FIN-01": ["8H100.80S.176V"]}
    # One token exchange, then bearer-authenticated spot availability calls.
    assert [s[1] for s in seen] == ["/oauth2/token", "/instance-availability", "/instance-availability"]
    assert seen[1][2] == "T" and seen[1][4] == {"is_spot": "true"}
