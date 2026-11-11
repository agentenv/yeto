"""--rl-routed-expert-qat: the run config flag reaches the Miles trainer argv as --routed-expert-qat."""

from __future__ import annotations

import argparse
import dataclasses
from types import SimpleNamespace

import pytest
from test_rl_k31_e2e_cpu import _k31_argv, _k31_config, _value

from yeto.launch.rl_island_task import _k31_island_flags
from yeto.rl.adapters.miles import config as mc
from yeto.rl.adapters.miles.models.kimi_k31 import ports_recipe as k31_recipe
from yeto.rl.algos import score_centering as sc
from yeto.rl.engine.run_flags import CLI, ISLAND, add_rl_flags


def _with_qat(cfg, value):
    return dataclasses.replace(cfg, model_recipe=dataclasses.replace(cfg.model_recipe, routed_expert_qat=value))


def test_recipe_argv_appends_the_flag_only_when_set():
    off = k31_recipe.recipe_argv(selection_path="s.json", seq_length=4096, include_model_argv=False)
    on = k31_recipe.recipe_argv(selection_path="s.json", seq_length=4096, include_model_argv=False,
                                routed_expert_qat="mxfp4")
    assert "--routed-expert-qat" not in off
    assert on == [*off, "--routed-expert-qat", "mxfp4"]
    with pytest.raises(k31_recipe.KimiK31RecipeError, match="nvfp4"):
        k31_recipe.recipe_argv(selection_path="s.json", seq_length=4096, routed_expert_qat="nvfp4")


def test_run_config_translates_to_the_miles_argv():
    cfg = _k31_config("/run/expert_selection.json")
    argv = _k31_argv(_with_qat(cfg, "mxfp4"), sc.score_centering_spec())
    assert argv.count("--routed-expert-qat") == 1 and _value(argv, "--routed-expert-qat") == "mxfp4"
    # Miles installs the hook on the raw GPTModel path only
    assert _value(argv, "--megatron-to-hf-mode") == "raw"
    assert "--routed-expert-qat" not in _k31_argv(cfg, sc.score_centering_spec())
    with pytest.raises(mc.MilesConfigError, match="routed-expert QAT"):
        _k31_argv(_with_qat(cfg, "int4"), sc.score_centering_spec())


def test_cli_flag_parses_on_both_layers_and_is_forwarded_to_the_island():
    given = ["--rl-model-recipe", "kimi_k31", "--rl-expert-selection", "/run/s.json", "--rl-routed-expert-qat", "mxfp4"]
    parsed = {}
    for layer in (CLI, ISLAND):
        parser = argparse.ArgumentParser()
        add_rl_flags(parser, layer)
        for action in parser._actions:
            action.required = False
        parsed[layer], _ = parser.parse_known_args(given)
        assert parsed[layer].rl_routed_expert_qat == "mxfp4"
    assert _k31_island_flags(parsed[CLI]).split()[-2:] == ["--rl-routed-expert-qat", "mxfp4"]
    assert "--rl-routed-expert-qat" not in _k31_island_flags(SimpleNamespace(rl_expert_selection="/run/s.json"))
