"""``--rl-sync-preset decoupled`` routes by trainable form (yeto.rl.engine.sync_route)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.rl.engine.selection import sync_mode_default
from yeto.rl.engine.sync_route import (
    FULL_PARAMETER,
    LORA_FRAGMENTS,
    LORA_STRICT,
    NO_SYNC,
    SAO_STREAMING,
    SyncRouteError,
    apply_sync_route,
    resolve_sync_route,
)


@pytest.mark.parametrize(
    ("preset", "tuning", "expert_full", "transport"),
    [
        ("strict-avg", "lora", 0, LORA_STRICT),
        ("strict-avg", "full", 0, LORA_STRICT),  # the launcher keeps refusing it
        ("decoupled", "lora", 0, LORA_FRAGMENTS),
        ("full-parameter", "full", 0, FULL_PARAMETER),
        ("full-parameter", "lora", 3, FULL_PARAMETER),
        ("dense-full", "full", 0, FULL_PARAMETER),
        ("sao-streaming-full", "full", 0, SAO_STREAMING),
    ],
)
def test_presets_accepted_before_routing_map_to_themselves(preset, tuning, expert_full, transport):
    route = resolve_sync_route(preset=preset, tuning=tuning, expert_full_count=expert_full)
    canonical = "full-parameter" if preset == "dense-full" else preset
    assert route.preset == canonical
    assert route.requested == canonical
    assert not route.rerouted
    assert route.transport == transport


@pytest.mark.parametrize(("tuning", "expert_full"), [("full", 0), ("lora", 2)])
def test_decoupled_full_parameter_forms_route_to_async_rounds(tuning, expert_full):
    route = resolve_sync_route(preset="decoupled", tuning=tuning, expert_full_count=expert_full)
    assert route.rerouted
    assert (route.preset, route.transport, route.full_parameter_rounds) == (
        "full-parameter", FULL_PARAMETER, "async")
    assert route.line().startswith("[sync-route] decoupled -> full-parameter (async rounds): ")


def test_decoupled_refuses_an_explicit_sweep():
    with pytest.raises(SyncRouteError, match="strict sweep"):
        resolve_sync_route(preset="decoupled", tuning="full", full_parameter_rounds="sweep",
                           rounds_explicit=True)
    # The unset (sync-mode default) sweep is not a request.
    assert resolve_sync_route(preset="decoupled", tuning="full", full_parameter_rounds="sweep",
                              rounds_explicit=False).full_parameter_rounds == "async"


def test_no_sync_islands_are_never_rerouted():
    route = resolve_sync_route(preset="decoupled", tuning="full", single_island_no_sync=True)
    assert (route.preset, route.transport) == ("decoupled", NO_SYNC)


def _args(**overrides):
    base = dict(training_mode="rl", rl_sync_preset="decoupled", tuning="full", expert_full_count=0,
                rl_sync_full_parameter_rounds=sync_mode_default("sweep"),
                rl_sync_fragment_pipeline=sync_mode_default(1), fragments=8, pipeline=2,
                rl_single_island_no_sync=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_apply_rewrites_a_rerouted_run():
    args = _args()
    route = apply_sync_route(args)
    assert route is args.rl_sync_route and route.rerouted
    assert args.rl_sync_preset == "full-parameter"
    assert args.rl_sync_preset_requested == "decoupled"
    assert args.rl_sync_full_parameter_rounds == "async"
    assert type(args.rl_sync_full_parameter_rounds) is str  # explicit, no sync-mode mark
    assert args.rl_sync_fragment_pipeline == 2


def test_apply_bounds_the_pipeline_by_the_fragments_and_keeps_an_explicit_one():
    args = _args(fragments=1)
    apply_sync_route(args)
    assert args.rl_sync_fragment_pipeline == 1
    args = _args(rl_sync_fragment_pipeline=5)
    apply_sync_route(args)
    assert args.rl_sync_fragment_pipeline == 5


def test_apply_leaves_lora_decoupled_untouched():
    args = _args(tuning="lora")
    before = dict(vars(args))
    apply_sync_route(args)
    after = dict(vars(args))
    assert after.pop("rl_sync_preset_requested") == "decoupled"
    assert after.pop("rl_sync_route").transport == LORA_FRAGMENTS
    assert after == before


def test_apply_ignores_sft():
    args = _args(training_mode="sft")
    assert apply_sync_route(args) is None
    assert not hasattr(args, "rl_sync_route")
