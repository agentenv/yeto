"""rl-infra-spec 1.1: runtime manifest consistency and capability refusal (pure)."""

from __future__ import annotations

import pytest

from yeto.rl.engine import runtime_manifest as rm
from yeto.rl.engine.miles_adapter.runtime_manifest import MILES_RUNTIME

PINS = {"miles": "a" * 40, "sglang": "b" * 40, "image": "docker:ghcr.io/x/miles@sha256:" + "c" * 64}


def _manifest(**over):
    m = {
        "schema": rm.MANIFEST_SCHEMA,
        "image": PINS["image"],
        "commits": {"miles": PINS["miles"], "sglang": PINS["sglang"]},
        "versions": {"torch": "2.9", "megatron.core": "0.15", "cuda": "12.9", "nccl": "2.27",
                     "torch_memory_saver": "0.0.9", "peft": "0.17"},
        "interfaces": {name: True for name in MILES_RUNTIME.interfaces},
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
    no_m1 = _manifest(interfaces={**{n: True for n in MILES_RUNTIME.interfaces}, "fork_m1_placement_map": False})
    rm.certify(no_m1, PINS, ["ports-partitioned-serial"])  # no standby -> M1 not needed
    with pytest.raises(rm.ManifestMismatch, match="fork_m1_placement_map"):
        rm.certify(no_m1, PINS, ["fixed-partition-standby"])
    with pytest.raises(rm.ManifestMismatch, match="unknown capability"):
        rm.certify(_manifest(), PINS, ["teleport"])


def test_collect_runs_without_miles_and_reports_absent_interfaces():
    m = rm.collect(image=None)
    assert m["schema"] == rm.MANIFEST_SCHEMA
    assert set(m["interfaces"]) == set(MILES_RUNTIME.interfaces)
    assert rm.check_manifest(m, PINS)  # this CPU venv is not the pinned image


def test_miles_description_keeps_the_pre_decoupling_tables():
    """decoupling 2.5: the values moved out of the core unchanged."""
    assert MILES_RUNTIME.interfaces["run_plugin"] == ("miles.ray.train.group", "TrainerController.run_plugin")
    assert set(MILES_RUNTIME.interfaces) == {
        "run_plugin", "fork_m1_placement_map", "fork_m2_start_stop_cells", "fork_m3_router_cordon",
        "fork_m4_member_publish", "fork_m6_rebuild_trainer"}
    assert MILES_RUNTIME.capability_requirements["Placement.reconfigure"] == (
        "fork_m1_placement_map", "fork_m6_rebuild_trainer")
    assert MILES_RUNTIME.version_modules == ("torch", "megatron.core", "torch_memory_saver", "peft", "sglang", "ray")
    assert MILES_RUNTIME.checkouts == ("miles", "sglang") and MILES_RUNTIME.overlay_field == "miles_overlay"
    assert rm.runtime_description() is MILES_RUNTIME
    assert set(rm.expected_pins()) == {"miles", "sglang", "image"}


def test_core_collects_any_backend_description():
    """A backend other than Miles declares its own probes; the core names none of them."""
    other = rm.RuntimeDescription(
        interfaces={"json_dumps": ("json", "dumps"), "absent": ("no_such_module_xyz", "f")},
        capability_requirements={"serve": ("json_dumps",), "train": ("absent",)},
        version_modules=("json",), required_versions=(), checkouts=("json",),
        overlay_field="other_overlay", overlay_record=lambda: {"patch": 1},
        expected_pins=lambda: {"json": "x"})
    m = rm.collect(image=None, runtime=other)
    assert m["interfaces"] == {"json_dumps": True, "absent": False}
    assert m["other_overlay"] == {"patch": 1} and "miles_overlay" not in m
    assert set(m["commits"]) == {"json"}
    assert rm.missing_interfaces(m, "serve", other) == []
    assert rm.missing_interfaces(m, "train", other) == ["absent"]
    with pytest.raises(ValueError, match="unknown backend"):
        rm.runtime_description("nope")
