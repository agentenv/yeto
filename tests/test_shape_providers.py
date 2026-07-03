"""Tests for the AWS quota/score providers (no network, no boto3 needed)."""

from __future__ import annotations

import pytest

from yeto.shape.cache import TTLCache
from yeto.shape.providers import AwsProviders, QuotaKey


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
    assert providers.quota_code("p4d.24xlarge", True) is None
