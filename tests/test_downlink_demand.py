"""Launch preflight downlink demand: per-fragment BCAST bytes and the slow-link WARNING.

A WARNING only: no argument changes and nothing added to the dry-run plan (the
plans are byte-locked by test_rl_sync_mode and the customer plan golden). CPU only.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto import memory_estimate as me
from yeto.launch import downlink_demand as dd

ARCH_8B = me.ModelArch(params=8e9, layers=32, hidden=4096, vocab=128256, intermediate=14336,
                       heads=32, kv_heads=8, head_dim=128)
TWO = "modal:2xh100,modal:2xh100"
FULL = ("--tuning", "full", "--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
        "--rl-rollout-gpus", "1", "--fragments", "4", "--rl-sync-full-parameter-rounds", "async",
        "--local-rl-rounds-per-sync", "2", "--rl-parameter-layout-sha256", "e" * 64)


def _args(**kw) -> SimpleNamespace:
    base = {"training_mode": "rl", "rl_sync_preset": "full-parameter", "tuning": "full", "fragments": 4,
            "lora_r": 16, "lora_targets": "attention", "model": "org/model"}
    return SimpleNamespace(**{**base, **kw})


def test_full_parameter_f32_bytes_per_fragment():
    rec = dd.downlink_demand(_args(), arch=ARCH_8B)
    assert rec["estimated"] and rec["synced"] == "full-parameter" and rec["bcast"] == "f32"
    assert rec["version_bytes"] == 4 * 8_000_000_000
    assert rec["fragment_bytes"] == 8_000_000_000
    assert rec["warn"] and rec["slow_link_s"] == pytest.approx(8e9 / (10 << 20), abs=0.1)


def test_bcast_delta_lowers_the_mean_bytes():
    rec = dd.downlink_demand(_args(rl_sync_bcast_delta_full_every=4), arch=ARCH_8B)
    assert rec["version_bytes"] == int(8e9 * (4 + 2 * 3) / 4)
    assert rec["bcast"] == "bf16 delta, f32 every 4"


def test_lora_fragments_are_small_and_quiet():
    rec = dd.downlink_demand(_args(rl_sync_preset="decoupled", tuning="lora"), arch=ARCH_8B)
    assert rec["synced"] == "lora"
    assert rec["synced_params"] == int(me.lora_params(ARCH_8B, 16, "attention"))
    assert not rec["warn"]


@pytest.mark.parametrize("kw", [
    {"training_mode": "sft"},
    {"rl_single_island_no_sync": True},
    {"rl_sync_preset": "sao-streaming"},
    {"rl_sync_preset": None},
])
def test_no_syncer_no_record(kw):
    assert dd.downlink_demand(_args(**kw), arch=ARCH_8B) is None


@pytest.mark.parametrize("kw", [{"rl_expert_selection": "sel.json"}, {"expert_full_count": 4}])
def test_partial_experts_are_not_estimated(kw):
    rec = dd.downlink_demand(_args(**kw), arch=ARCH_8B)
    assert rec == {"estimated": False, "reason": rec["reason"]} and "partial-expert" in rec["reason"]


def test_unknown_model_is_not_estimated(monkeypatch):
    monkeypatch.setattr(me, "resolve_arch", lambda model: (None, model, "none"))
    rec = dd.downlink_demand(_args())
    assert not rec["estimated"] and "org/model" in rec["reason"]


def test_preflight_prints_the_warning_only_above_the_threshold():
    out = io.StringIO()
    dd.downlink_preflight(_args(), out=out, arch=ARCH_8B)
    text = out.getvalue()
    assert "WARNING: downlink demand" in text and "--rl-sync-bcast-delta-full-every" in text
    quiet = io.StringIO()
    dd.downlink_preflight(_args(fragments=1024), out=quiet, arch=ARCH_8B)
    assert quiet.getvalue() == ""


def test_dry_run_warns_and_keeps_the_plan(monkeypatch, capsys):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "downlink-test-key")
    args = _launcher_args("ports", FULL, gpu=TWO)
    args.controller = "local"
    _prepare_rl_args(args)
    monkeypatch.setattr(me, "resolve_arch", lambda model: (ARCH_8B, model, "test"))
    after = dry_run_plan(args)
    assert "WARNING: downlink demand" in capsys.readouterr().out
    monkeypatch.setattr(dd, "downlink_preflight", lambda *a, **kw: None)
    before = dry_run_plan(args)  # the same plan without the preflight
    assert after == before
    assert "downlink" not in after


def test_the_customer_no_sync_trainer_prints_nothing(monkeypatch, capsys):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    monkeypatch.setattr(me, "resolve_arch", lambda model: (ARCH_8B, model, "test"))
    args = _launcher_args("ports", ("--tuning", "full", "--rl-single-island-no-sync"), gpu="modal:1xh100")
    args.controller = "local"
    _prepare_rl_args(args)
    dry_run_plan(args)
    assert "downlink demand" not in capsys.readouterr().out


def test_dry_run_warns_about_unpinned_modal_islands(monkeypatch, capsys):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "downlink-test-key")
    args = _launcher_args("ports", FULL, gpu=TWO)
    args.controller = "local"
    _prepare_rl_args(args)
    dry_run_plan(args)
    out = capsys.readouterr().out
    assert "WARNING: syncer region" in out and "no @region" in out
