"""Miles k31/b3-b7 (B3/B5/B6/B7 batches): pins, features, runtime probes, flag mapping.

The fork-era switches become Miles argv flags gated on the pin's features; the
default GRPO argv (test_decoupling_golden) does not move.
"""

from __future__ import annotations

import dataclasses

import pytest

from yeto.rl.adapters.miles import config as mc
from yeto.rl.adapters.miles import pins
from yeto.rl.adapters.miles import runtime_manifest as rm
from yeto.rl.engine.algorithm import AlgorithmSpec

from test_rl_miles_adapter_config import make_config

B3B7_FEATURES = (
    "engine_failure_isolation", "abort_tally", "agentic_suspend", "session_responses_api",
    "lora_per_token_loss", "bridge_moe_runtime_flags", "host_sync_free_flag",
    "nemotron_h_vl_bridge", "replay_hybrid_layers",
)


def test_k31_lineage_registers_the_b3b7_features():
    assert pins.MILES_K31_B3B7_COMMIT == "a07553f0a4f361b1618a707b416b690c5666e03d"
    k31 = pins.MILES_LINEAGES["k31"].commits
    assert k31[:3] == (pins.MILES_K31_F1_COMMIT, pins.MILES_K31_B2_COMMIT, pins.MILES_K31_B3B7_COMMIT)
    for feature in B3B7_FEATURES:
        assert pins.K31_FEATURES[feature] == pins.MILES_K31_B3B7_COMMIT
        assert pins.pin_has(feature) and pins.pin_has(feature, pins.MILES_K31_B3B7_COMMIT)
        assert not pins.pin_has(feature, pins.MILES_K31_B2_COMMIT)
        assert not pins.pin_has(feature, pins.MILES_K31_F1_COMMIT)
    # earlier k31 features carry forward to the new head
    for feature in ("run_plugin", "yeto_ext", "full_parameter_transport", "full_parameter_rank_state"):
        assert pins.pin_has(feature, pins.MILES_K31_B3B7_COMMIT)
    # the frozen ports-v1 entries of the same name keep their values
    assert pins.PORTS_V1_FEATURES["agentic_suspend"] == "ddce20992c9ee82270a22e259e9e2bf55f3053d1"
    assert pins.PORTS_V1_FEATURES["lora_per_token_loss"] == "0af62f4d48ed6a5b185c257578d8f7e22312aa87"
    assert pins.PORTS_V1_FEATURES["host_sync_free"] == "17841205b19fde01279fd9585711cb610bad67a8"
    assert pins.PORTS_V1_FEATURES["session_responses"] == "212da67914cf02f2fc2e0b7ee8f3455b79359ed8"
    from yeto import rl

    assert rl.MILES_K31_B3B7_COMMIT == pins.MILES_K31_B3B7_COMMIT


B3B7_PROBES = {
    "fork_b3_engine_failure_isolation": (
        "miles.backends.training_utils.weight_update.protocols.broadcast", "RolloutEngineJoinError"),
    "fork_b5_abort_tally": ("miles.rollout.inference_rollout.inference_rollout_train", "_abort_engines"),
    "fork_b5_agentic_suspend": ("miles.rollout.inference_rollout.inference_rollout_train", "suspend"),
    "fork_b6_session_responses_api": ("miles.rollout.session.responses_adapter", "to_chat_request"),
    "fork_b6_lora_per_token_loss": ("miles.backends.megatron_utils.lora.bridge", "_setup_lora_model_via_bridge"),
    "fork_b7_bridge_moe_runtime_flags": ("miles.utils.megatron_bridge_utils", "megatron_cli_defaults"),
    "fork_b7_bridge_moe_runtime_overrides": ("miles.utils.megatron_bridge_utils", "bridge_moe_runtime_overrides"),
    "fork_b7_host_sync_free_flag": ("miles.utils.sync_free", "configure"),
    "fork_b7_nemotron_h_vl_bridge": ("miles_plugins.megatron_bridge.nemotron_h", "NEMOTRON_H_OMNI_ARCHITECTURE"),
    "fork_b7_replay_hybrid_layers": ("miles.backends.megatron_utils.replay_utils", "_layer_ids_from_routers"),
}


def test_runtime_manifest_probes_every_b3b7_feature():
    for name, target in B3B7_PROBES.items():
        assert rm.INTERFACES[name] == target
    # no capability is refused for them yet (the probes are evidence, not gates)
    needed = {i for interfaces in rm.CAPABILITY_REQUIREMENTS.values() for i in interfaces}
    assert not needed & set(B3B7_PROBES)


def test_probes_resolve_in_a_b3b7_checkout():
    """Where the k31/b3-b7 Miles checkout and its dependencies are importable,
    every probe finds its attribute (the import-light ones at least need Miles)."""
    pytest.importorskip("miles.utils.sync_free")
    from yeto.rl.engine.runtime_manifest import _probe

    light = ("fork_b6_session_responses_api", "fork_b7_bridge_moe_runtime_flags",
             "fork_b7_bridge_moe_runtime_overrides", "fork_b7_host_sync_free_flag", "fork_b7_nemotron_h_vl_bridge")
    assert [n for n in light if not _probe(*B3B7_PROBES[n])] == []
    pytest.importorskip("megatron.core")
    pytest.importorskip("ray")
    assert [n for n, (m, a) in B3B7_PROBES.items() if not _probe(m, a)] == []


def _argv(cfg, **kw):
    return list(mc.translate_run_config(cfg, AlgorithmSpec(), **kw).argv)


def _pin(monkeypatch, commit):
    monkeypatch.setattr(pins, "MILES_NEXT_COMMIT", commit)
    monkeypatch.setattr(mc, "MILES_NEXT_COMMIT", commit)


def test_host_sync_free_default_emits_nothing_and_off_emits_the_typed_flag(monkeypatch):
    cfg = make_config()
    assert cfg.host_sync_free is True
    default = _argv(cfg)
    assert "--host-sync-free" not in default and "--no-host-sync-free" not in default
    off = _argv(dataclasses.replace(cfg, host_sync_free=False))
    assert off.count("--no-host-sync-free") == 1
    assert [a for a in off if a != "--no-host-sync-free"] == default
    # the switch is a flag, never the fork's environment variable
    launch = mc.translate_run_config(dataclasses.replace(cfg, host_sync_free=False), AlgorithmSpec())
    assert "MILES_HOST_SYNC_FREE" not in dict(launch.runtime_attrs)


def test_agentic_carry_over_needs_a_pin_with_the_suspend_flags(tmp_path, monkeypatch):
    from test_agentic_stage3_island_argv import _island_argv, _miles_argv, island_entry

    args = island_entry.parse_args(_island_argv(tmp_path))
    argv = _miles_argv(args, tmp_path, monkeypatch)
    assert "--agentic-suspend-between-turns" in argv  # the k31/b3-b7 head carries B5
    _pin(monkeypatch, pins.MILES_K31_B2_COMMIT)
    with pytest.raises(Exception, match="--agentic-suspend-between-turns"):
        _miles_argv(args, tmp_path, monkeypatch)
    _pin(monkeypatch, pins.MILES_PORTS_V1_COMMIT)  # ports-v1 ddce20992+ has the flags
    assert "--agentic-suspend-between-turns" in _miles_argv(args, tmp_path, monkeypatch)


def test_host_sync_free_off_refuses_a_pin_without_the_flag(monkeypatch):
    cfg = dataclasses.replace(make_config(), host_sync_free=False)
    _pin(monkeypatch, pins.MILES_K31_B2_COMMIT)
    with pytest.raises(mc.MilesConfigError, match="--no-host-sync-free needs Miles k31/b3-b7"):
        _argv(cfg)
    # ports-v1 has the sync-free paths but only the environment switch: refused too
    _pin(monkeypatch, pins.MILES_PORTS_V1_COMMIT)
    with pytest.raises(mc.MilesConfigError, match="--no-host-sync-free"):
        _argv(cfg)


def test_run_config_reads_the_host_sync_free_switch_from_the_island_args():
    from types import SimpleNamespace

    from yeto.rl.engine import run_config as rc

    assert rc.RLRunConfig.__dataclass_fields__["host_sync_free"].default is True
    assert bool(getattr(SimpleNamespace(), "rl_host_sync_free", True)) is True
    assert bool(getattr(SimpleNamespace(rl_host_sync_free=False), "rl_host_sync_free", True)) is False


def test_a27_timeout_maps_to_the_upstream_engine_request_timeout_flag():
    cfg = make_config(update_weight_group_timeout_s=120.0)
    argv = _argv(cfg)
    assert argv[argv.index("--update-weight-engine-request-timeout") + 1] == "120"
    assert "--update-weight-group-timeout-s" not in argv
    assert "--update-weight-engine-request-timeout" not in _argv(make_config())
