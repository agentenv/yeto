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
