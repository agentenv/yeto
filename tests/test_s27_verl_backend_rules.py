"""S27 G-DB: run-config rules follow --rl-backend (V7); verl single-node layout flags (V6)."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from yeto.gpu_spec import parse_gpu_spec
from yeto.launch import topology
from yeto.rl.engine import backends, run_config


def test_run_config_rules_follow_the_run_backend():
    assert run_config._rules() is backends.module("run_config_rules", "miles")
    assert run_config._rules("verl") is backends.module("run_config_rules", "verl")
    assert run_config._rules(None) is backends.module("run_config_rules", "miles")


def test_ref_load_rule_comes_from_the_selected_backend():
    # Miles takes a Megatron reference checkpoint; the verl rule refuses it.
    verl = argparse.Namespace(rl_backend="verl", megatron_ref_load="/ref")
    with pytest.raises(ValueError, match="Miles/Megatron option"):
        run_config._resolve_ref_load(verl, "/m")
    plain = argparse.Namespace(rl_backend="verl", megatron_ref_load=None)
    assert run_config._resolve_ref_load(plain, "/m") == "/m"


def test_run_config_rules_ignore_the_environment(monkeypatch):
    # A Miles island never picks up another backend's rules from the environment.
    monkeypatch.setenv(backends.BACKEND_ENV, "verl")
    assert run_config._rules() is backends.module("run_config_rules", "miles")
    assert run_config._rules(run_config._backend_of(argparse.Namespace())) is backends.module(
        "run_config_rules", "miles")


@pytest.mark.parametrize("gpu", ["modal:1xh200", "modal:2xh200", "modal:4xh200"])
def test_single_node_verl_island_needs_no_bundle_map(gpu):
    (spec,) = parse_gpu_spec(gpu)
    args = SimpleNamespace(rl_backend="verl", rl_placement="colocated",
                           rl_elastic_initial_placement_slots=None)
    assert topology.rl_island_bundle_map_flag(args, spec) == ""


def test_verl_refuses_a_role_to_card_map_with_its_reason():
    flags = backends.module("launch_flags", "verl")
    with pytest.raises(ValueError, match="同卡共置") as error:
        flags.bundle_map_flag({"trainer": (2, 3), "rollout": (0, 1), "standby": ()})
    assert "trainer=[2, 3]" in str(error.value)
    assert flags.cross_node_flags(False, False) == ""
    with pytest.raises(ValueError, match="跨节点"):
        flags.cross_node_flags(True, False)
