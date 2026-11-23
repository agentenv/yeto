"""--syncer-region follows pinned islands (default only) and warns on Modal islands placed apart.

Measured on Modal: syncer eu-west / islands us-west 15.9 MB/s per stream; both in
us-east 132 MB/s. The follow rule touches synced launches only: a no-sync
launch (one trainer, no syncer) keeps the default. CPU only.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from yeto.gpu_spec import parse_gpu_spec
from yeto.launch import syncer_region as sr


def _args(gpu: str, **kw) -> SimpleNamespace:
    base = {"training_mode": "rl", "syncer_region": sr.default_syncer_region(), "gpu": gpu}
    return SimpleNamespace(**{**base, **kw})


def _resolve(args) -> str:
    out = io.StringIO()
    sr.resolve_syncer_region(args, parse_gpu_spec(args.gpu), out=out)
    assert not sr.is_default_syncer_region(args.syncer_region)
    return out.getvalue()


@pytest.mark.parametrize("region,area", [
    ("us-east", "us-east"), ("us-east-1", "us-east"), ("us-east4", "us-east"), ("us-ashburn-1", "us-east"),
    ("us-west-2", "us-west"), ("eu-west", "eu-west"), ("eu-north1", "eu-north"), ("aws/us-west-2", "us-west"),
    ("us-central1", "us-central"), (None, None), ("europe-west4", None),
])
def test_area_of(region, area):
    assert sr.area_of(region) == area


@pytest.mark.parametrize("gpu,region", [
    ("modal:2xh100@us-east,modal:2xh100@us-east", "us-east-1"),
    ("modal:2xh100@us-east-1,modal:2xh100@us-east", "us-east-1"),
    ("modal:2xh100@eu-west,modal:2xh100@eu-west", "eu-west-1"),
    ("aws:8xh100@us-east-2,aws:8xh100@us-east-2", "us-east-2"),
    ("modal:8xh100@us-east", "us-east-1"),
])
def test_the_syncer_follows_islands_pinned_to_one_area(gpu, region):
    args = _args(gpu)
    out = _resolve(args)
    assert args.syncer_region == region
    assert f"[auto] syncer region {region}" in out and "--syncer-region" in out


@pytest.mark.parametrize("gpu", [
    "modal:2xh100,modal:2xh100",  # unpinned
    "modal:2xh100@us-east,modal:2xh100",  # partly pinned
    "modal:2xh100@us-east,modal:2xh100@us-west",  # two areas
    "aws:8xa100@us-east-1,aws:8xa100@us-west-2",
    "gcp:8xh100@us-east4,gcp:8xh100@us-east4",  # not a follow cloud
    "modal:2xh100@us-central,modal:2xh100@us-central",  # no AWS region for the area
    "modal:2xh100@us-west,modal:2xh100@us-west",  # already the default: nothing printed
])
def test_the_default_stays_otherwise_and_prints_nothing(gpu):
    args = _args(gpu)
    assert _resolve(args) == ""
    assert args.syncer_region == "us-west-2"


@pytest.mark.parametrize("kw", [{"rl_single_island_no_sync": True}, {"training_mode": "sft"},
                                {"external_learners": 1}])
def test_no_sync_sft_and_external_learners_keep_the_default(kw):
    args = _args("modal:8xh100@us-east", **kw)
    assert _resolve(args) == ""
    assert args.syncer_region == "us-west-2"


def test_an_explicit_region_is_the_opt_out():
    args = _args("modal:2xh100@us-east,modal:2xh100@us-east", syncer_region="us-west-2")
    out = io.StringIO()
    sr.resolve_syncer_region(args, parse_gpu_spec(args.gpu), out=out)
    assert args.syncer_region == "us-west-2" and out.getvalue() == ""


def test_the_cli_default_is_marked_and_an_explicit_value_is_not():
    from yeto.cli import build_parser

    parser = build_parser()
    base = ["launch", "--gpu", "modal:1xh100", "--model", "m", "--data", "d"]
    assert sr.is_default_syncer_region(parser.parse_args(base).syncer_region)
    explicit = parser.parse_args(base + ["--syncer-region", "us-west-2"]).syncer_region
    assert explicit == "us-west-2" and not sr.is_default_syncer_region(explicit)


def test_unpinned_modal_islands_warn_with_the_measured_numbers():
    findings = sr.region_findings(_args("modal:2xh100,modal:2xh100", syncer_region="us-west-2"),
                                  parse_gpu_spec("modal:2xh100,modal:2xh100"))
    assert len(findings) == 1
    assert "no @region" in findings[0] and "15.9 MB/s" in findings[0] and "132 MB/s" in findings[0]


def test_modal_islands_apart_from_the_syncer_warn():
    gpu = "modal:2xh100@us-east,modal:2xh100@eu-west"
    findings = sr.region_findings(_args(gpu, syncer_region="us-east-1"), parse_gpu_spec(gpu))
    assert len(findings) == 1 and "1@eu-west" in findings[0] and "0@" not in findings[0]


@pytest.mark.parametrize("gpu,kw", [
    ("modal:2xh100@us-east,modal:2xh100@us-east", {"syncer_region": "us-east-1"}),
    ("aws:8xh100@us-east-1,aws:8xh100@us-west-2", {"syncer_region": "us-west-2"}),  # no Modal island
    ("modal:2xh100,modal:2xh100", {"syncer_region": "us-west-2", "rl_single_island_no_sync": True}),
])
def test_no_warning_when_together_without_modal_or_without_a_syncer(gpu, kw):
    assert sr.region_findings(_args(gpu, **kw), parse_gpu_spec(gpu)) == []
