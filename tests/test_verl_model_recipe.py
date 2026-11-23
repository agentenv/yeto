"""The verl backend refuses model recipes its island has no code for.

No verl code reads ``--rl-model-recipe``, ``--rl-expert-selection`` or
``--expert-full-count``: the launcher accepted e.g. ``--rl-backend verl
--rl-model-recipe kimi_k31 --tuning full --rl-expert-selection X`` and the island
dropped the flags (``parse_known_args``) and trained the generic recipe. Now the
launcher (``run_config_rules.check_model_recipe``) and the island refuse them. CPU only.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.adapters.verl import island_entry
from yeto.rl.adapters.verl import run_config_rules as rules
from yeto.rl.engine import backends

ISLAND = ("--model", "m", "--model-revision", "a" * 40, "--data", "d", "--learner-id", "0",
          "--reward-function", "yeto.rl.gsm8k_reward:score", "--global-rounds", "2", "--groups-per-round", "1",
          "--samples-per-group", "2", "--rollout-max-response-len", "8", "--event-tape", "t",
          "--inner-lr", "1", "--seq-len", "64", "--seed", "1", "--rl-single-island-no-sync")


def _prepare(*extra, backend="verl"):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", ("--rl-backend", backend, "--reward-function", "gsm8k_reward:score", *extra))
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def _k31_selection(tmp_path) -> str:
    from test_rl_k31_synced_cpu import _selection

    path = tmp_path / "expert_selection.json"
    _selection().write(path)
    return str(path)


def test_the_launcher_refuses_the_k31_partial_expert_recipe_on_verl(tmp_path):
    with pytest.raises(ValueError, match="generic recipe only.*--rl-model-recipe kimi_k31.*--rl-backend miles"):
        _prepare("--rl-model-recipe", "kimi_k31", "--tuning", "full", "--rl-expert-selection",
                 _k31_selection(tmp_path), "--expert-parallel", "8", "--rl-single-island-no-sync",
                 "--model", "moonshotai/Kimi-K3.1")


def test_the_launcher_refuses_a_named_recipe_on_verl():
    with pytest.raises(ValueError, match="--rl-model-recipe deepseek-v4-flash"):
        _prepare("--rl-model-recipe", "deepseek-v4-flash")


def test_the_launcher_still_accepts_the_generic_recipe_on_verl():
    assert _prepare("--rl-model-recipe", "generic").rl_backend == "verl"
    assert _prepare().rl_backend == "verl"


def test_miles_has_no_such_rule():
    # The hook is the backend's own: a Miles launch is unchanged
    assert getattr(backends.module("run_config_rules", "miles"), "check_model_recipe", None) is None


@pytest.mark.parametrize("fields, needle", [
    ({"rl_model_recipe": "kimi_k31"}, "--rl-model-recipe kimi_k31"),
    ({"rl_expert_selection": "sel.json"}, "--rl-expert-selection"),
    ({"expert_full_count": 4}, "--expert-full-count"),
])
def test_the_rule_names_each_ignored_flag(fields, needle):
    args = SimpleNamespace(**{"rl_model_recipe": "generic", "rl_expert_selection": None, "expert_full_count": 0,
                              **fields})
    with pytest.raises(ValueError, match=needle):
        rules.check_model_recipe(args)


@pytest.mark.parametrize("argv, needle", [
    (("--rl-model-recipe", "kimi_k31"), "--rl-model-recipe kimi_k31"),
    (("--rl-expert-selection", "/sel.json"), "--rl-expert-selection"),
    (("--expert-full-count", "2"), "--expert-full-count"),
])
def test_the_island_refuses_them_before_any_work(argv, needle):
    with pytest.raises(SystemExit, match=needle):
        island_entry.main([*ISLAND, *argv])


def test_the_island_parses_the_generic_recipe_instead_of_dropping_it():
    args, ignored = island_entry.parse_args([*ISLAND, "--rl-model-recipe", "generic"])
    assert args.rl_model_recipe == "generic" and "--rl-model-recipe" not in ignored
    assert rules.model_recipe_problems(recipe=args.rl_model_recipe, expert_selection=args.rl_expert_selection,
                                       expert_full_count=args.expert_full_count) == []
