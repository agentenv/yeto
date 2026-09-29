"""rl-infra-spec 1.1: runtime manifest consistency and capability refusal (pure)."""

from __future__ import annotations

import pytest

from yeto.rl.engine import runtime_manifest as rm

PINS = {"miles": "a" * 40, "sglang": "b" * 40, "image": "docker:ghcr.io/x/miles@sha256:" + "c" * 64}


def _manifest(**over):
    m = {
        "schema": rm.MANIFEST_SCHEMA,
        "image": PINS["image"],
        "commits": {"miles": PINS["miles"], "sglang": PINS["sglang"]},
        "versions": {"torch": "2.9", "megatron.core": "0.15", "cuda": "12.9", "nccl": "2.27",
                     "torch_memory_saver": "0.0.9", "peft": "0.17"},
        "interfaces": {name: True for name in rm.INTERFACES},
    }
    m.update(over)
    return m


def test_consistent_manifest_certifies_declared_capabilities():
    out = rm.certify(_manifest(), PINS, ["ports-partitioned-serial", "fixed-partition-standby"])
    assert out["certified"] == ["fixed-partition-standby", "ports-partitioned-serial"]


def test_pin_mismatch_or_missing_version_refuses():
    for bad in (
        _manifest(commits={"miles": "d" * 40, "sglang": PINS["sglang"]}),
        _manifest(image="docker:ghcr.io/x/miles@sha256:" + "e" * 64),
        _manifest(versions={"torch": "2.9"}),
        _manifest(schema="v0"),
    ):
        with pytest.raises(rm.ManifestMismatch):
            rm.certify(bad, PINS, [])


def test_missing_interface_combination_refuses_certification():
    no_m1 = _manifest(interfaces={**{n: True for n in rm.INTERFACES}, "fork_m1_placement_map": False})
    rm.certify(no_m1, PINS, ["ports-partitioned-serial"])  # no standby -> M1 not needed
    with pytest.raises(rm.ManifestMismatch, match="fork_m1_placement_map"):
        rm.certify(no_m1, PINS, ["fixed-partition-standby"])
    with pytest.raises(rm.ManifestMismatch, match="unknown capability"):
        rm.certify(_manifest(), PINS, ["teleport"])


def test_collect_runs_without_miles_and_reports_absent_interfaces():
    m = rm.collect(image=None)
    assert m["schema"] == rm.MANIFEST_SCHEMA
    assert set(m["interfaces"]) == set(rm.INTERFACES)
    assert rm.check_manifest(m, PINS)  # this CPU venv is not the pinned image
