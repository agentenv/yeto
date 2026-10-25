"""CPU tests of the S27 overlong presets (tools/eval/efficiency/overlong_specs.py).

The presets select existing yeto mechanisms (overlong_filter, overlong_penalty);
these tests check that the committed files are current, differ from the E1 spec
only in the overlong fields, pass the spec and launch checks for their cap, and
change the Miles argv only as expected. No Miles, no Ray, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.eval.efficiency import overlong_specs as ov
from yeto.rl.algos import grpo_knobs as gk
from yeto.rl.algos import reward_pipeline as rp
from yeto.rl.engine.algorithm import AlgorithmSpec

REPO = Path(__file__).resolve().parents[1]
PEN = {"pen2048": 2048, "pen3072": 3072, "pen4096": 4096}


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    monkeypatch.chdir(REPO)


def _load(name: str) -> AlgorithmSpec:
    return AlgorithmSpec.from_json_file(str(ov.preset_path(name)))


def test_committed_presets_are_current():
    assert ov.main(["--check"]) == 0
    for name in ov.PRESETS:
        assert _load(name).sha256() == ov.build(name).sha256()


@pytest.mark.parametrize("name", sorted(ov.PRESETS))
def test_preset_differs_from_e1_only_in_overlong_fields(name):
    base = json.loads(ov.BASE.read_text())
    preset = json.loads(ov.preset_path(name).read_text())
    changed = {
        (group, key)
        for group in preset
        if isinstance(preset[group], dict)
        for key in preset[group]
        if (base.get(group) or {}).get(key) != preset[group][key]
    }
    changed |= {(group, None) for group in preset if not isinstance(preset[group], dict)
                and base.get(group) != preset[group]}
    if name == "filter":
        assert changed == {("sampling", "overlong_filter"), ("plugins", None)}
    else:
        assert changed == {("advantage", "reward_postprocess"), ("advantage", "reward_shapers"),
                           ("plugins", None)}


@pytest.mark.parametrize("name", sorted(ov.PRESETS))
def test_preset_passes_spec_checks_and_is_declared_for_miles(name):
    from yeto.rl.adapters.miles.entry import declared_by_dimension

    spec = _load(name)
    assert spec.rejections() == []
    assert spec.verify_plugins() == ()
    declared = declared_by_dimension()
    missing = [(d, n) for d, n in spec.required_mechanisms() if n not in declared.get(d, set())]
    assert missing == []


@pytest.mark.parametrize("name", sorted(PEN))
def test_penalty_preset_needs_its_cap(name):
    spec, cap = _load(name), PEN[name]
    assert gk.launch_problems(spec, rollout_batch_size=32, rollout_max_response_len=cap) == []
    problems = gk.launch_problems(spec, rollout_batch_size=32, rollout_max_response_len=cap - 1)
    assert len(problems) == 1 and "exceeds the generation limit" in problems[0]


def test_filter_preset_has_no_cap_condition():
    assert gk.launch_problems(_load("filter"), rollout_batch_size=32, rollout_max_response_len=2048) == []


def test_penalty_values_keep_short_responses():
    # pen3072 / pen4096 start at 2048: a response of <= 2048 tokens keeps its reward
    for name in ("pen3072", "pen4096"):
        shaper = dict(_load(name).advantage.reward_shapers[0])
        lmax, lcache = shaper["max_length"], shaper["cache_length"]
        assert lmax - lcache == 2048
        assert rp.overlong_penalty_value(2048, lmax, lcache) == 0.0
        assert rp.overlong_penalty_value(lmax, lmax, lcache) == -1.0
        assert rp.overlong_penalty_value(2048 + lcache // 2, lmax, lcache) == -0.5
    assert [rp.overlong_penalty_value(n, 2048, 512) for n in (1536, 1792, 2048)] == [0.0, -0.5, -1.0]


def _miles_launch(spec):
    from test_rl_miles_adapter_config import make_config, sub
    from yeto.rl.adapters.miles import config as mc

    cfg = sub(make_config(), "batch", seq_len=2560, rollout_max_response_len=2048)
    return mc.translate_run_config(cfg, spec)


def test_miles_argv_changes_only_as_selected():
    base = _miles_launch(AlgorithmSpec.from_json_file(str(ov.BASE)))
    assert "--custom-reward-post-process-path" not in base.argv
    assert rp.PIPELINE_ATTR not in base.runtime_attrs

    filt = _miles_launch(_load("filter"))
    assert filt.argv == base.argv
    assert rp.read_plugins(_attrs(filt))["overlong_filter"] is True

    pen = _miles_launch(_load("pen2048"))
    i = pen.argv.index("--custom-reward-post-process-path")
    assert pen.argv[i + 1] == rp.DISPATCHER_PATH
    assert pen.argv[:i] + pen.argv[i + 2:] == base.argv
    config = rp.read_plugins(_attrs(pen))
    assert config["reward_pipeline"]["reward_shapers"] == [
        {"name": "overlong_penalty", "max_length": 2048, "cache_length": 512}]


def _attrs(launch):
    from types import SimpleNamespace

    return SimpleNamespace(**launch.runtime_attrs)


def _launch_plan(spec_path: str, cap: int):
    from test_rl_full_parameter_async import ASYNC, TWO_ISLANDS, _launcher_args
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", ASYNC + (
        "--rl-algorithm-spec", spec_path, "--rollout-max-response-len", str(cap),
        "--rl-max-policy-age", "1", "--rl-rollout-ahead", "1"), gpu=TWO_ISLANDS)
    args.controller = "local"
    _prepare_rl_args(args)
    return dry_run_plan(args)


@pytest.mark.parametrize(("name", "cap"), [("filter", 2048), ("pen2048", 2048), ("pen3072", 3072),
                                           ("pen4096", 4096)])
def test_two_island_full_parameter_rollout_ahead_plan_accepts_preset(monkeypatch, name, cap):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    plan = _launch_plan(str(ov.preset_path(name)), cap)
    assert len(plan["island_requests"]) == 2
    for island in plan["island_requests"]:
        assert f"--rollout-max-response-len {cap}" in island["learner_command"]


def test_two_island_plan_refuses_penalty_above_cap(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    with pytest.raises(Exception, match="exceeds the generation limit"):
        _launch_plan(str(ov.preset_path("pen3072")), 2048)
