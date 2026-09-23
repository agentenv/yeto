"""`--regions` grammar: cloud-prefixed entries, the legacy AWS-only
spelling, `all`, and error reporting."""

from __future__ import annotations

import pytest

from yeto.shape.regions import RegionFilter, parse_regions

DEFAULT = ["us-east-1", "us-east-2"]
KNOWN = ["aws", "runpod", "nebius", "verda", "modal"]


def test_mixed_entries_restrict_each_named_cloud_only():
    f = parse_regions(["us-east-1", "nebius:eu-north1", "verda:FIN-03"], DEFAULT, KNOWN)
    assert f.for_cloud("aws") == {"us-east-1"}
    assert f.for_cloud("nebius") == {"eu-north1"}
    assert f.for_cloud("verda") == {"FIN-03"}  # case preserved
    assert f.for_cloud("runpod") is None  # unnamed clouds stay unrestricted
    assert f.allows("runpod", "CA") and not f.allows("aws", "us-west-2")
    assert f.explicit == {"aws", "nebius", "verda"}


def test_legacy_bare_regions_mean_aws_and_nothing_else():
    f = parse_regions(["us-east-1", "us-east-2"], DEFAULT, KNOWN)
    assert f.for_cloud("aws") == {"us-east-1", "us-east-2"}
    assert all(f.for_cloud(c) is None for c in KNOWN if c != "aws")
    # Omitting --regions altogether limits AWS to the default list and is
    # not "explicit" (so no error reporting on empty default regions).
    d = parse_regions(None, DEFAULT, KNOWN)
    assert d.for_cloud("aws") == set(DEFAULT) and d.explicit == frozenset()


def test_all_forms():
    everything = parse_regions(["all"], DEFAULT, KNOWN)
    assert everything.for_cloud("aws") is None and everything.explicit == frozenset()
    one_cloud = parse_regions(["nebius:all", "us-east-1"], DEFAULT, KNOWN)
    assert one_cloud.for_cloud("nebius") is None
    assert one_cloud.for_cloud("aws") == {"us-east-1"}
    # `all` inside a list is aws:all; a later aws region does not re-restrict it.
    in_list = parse_regions(["all", "nebius:eu-north1"], DEFAULT, KNOWN)
    assert in_list.for_cloud("aws") is None
    assert in_list.for_cloud("nebius") == {"eu-north1"}
    # Only a non-aws entry: AWS keeps its default list rather than going wide.
    only_neb = parse_regions(["nebius:eu-north1"], DEFAULT, KNOWN)
    assert only_neb.for_cloud("aws") == set(DEFAULT) and not only_neb.is_explicit("aws")


def test_unknown_cloud_and_empty_region_are_errors():
    with pytest.raises(ValueError, match="unknown cloud 'azure'.*known: aws, modal, nebius, runpod, verda"):
        parse_regions(["azure:westus"], DEFAULT, KNOWN)
    with pytest.raises(ValueError, match="empty region"):
        parse_regions(["nebius:"], DEFAULT, KNOWN)


def test_filter_is_a_plain_value():
    f = RegionFilter({"aws": frozenset({"us-east-1"})}, frozenset({"aws"}))
    assert f == RegionFilter({"aws": frozenset({"us-east-1"})}, frozenset({"aws"}))
