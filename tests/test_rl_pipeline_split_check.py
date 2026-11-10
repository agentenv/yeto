"""Pipeline split check (translate_run_config): a split Megatron cannot build is refused
before any GPU starts, with workable splits in the message.

* PP>1 without an uneven split needs num_layers % PP == 0 (93 layers / PP6 refused);
* with --decoder-first/last-pipeline-num-layers (run config or extra argv) the layers left
  must split evenly over the other stages (15 / 18 over PP6 accepted, 15 / 17 refused);
* the run config's derived split (resolve_rl_run_config) always passes;
* the full Kimi-K3.1-Pro synced layout PP6 x EP8 x TP8 (DP 1) translates.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from yeto.rl.adapters.miles import config as mc
from yeto.rl.engine.algorithm import AlgorithmSpec

from test_rl_k31_e2e_cpu import _k31_config
from test_rl_miles_adapter_config import sub

FIRST, LAST = mc.PIPELINE_SPLIT_FLAGS


def _pro(pp=6, uneven=(15, 18), tp=8, ep=8, nodes=6):
    """Kimi-K3.1-Pro (93 layers, 896 experts, top-16) as one synced trainer island."""
    cfg = _k31_config("/run/expert_selection.json", ep=ep, role="trainer")
    g = cfg.geometry
    cfg = dataclasses.replace(cfg, geometry=dataclasses.replace(
        g, num_layers=93, moe=dataclasses.replace(g.moe, num_experts=896, router_topk=16,
                                                  layer_freq=(0,) + (1,) * 92)))
    return sub(cfg, "parallel", actor_num_nodes=nodes, actor_num_gpus_per_node=8, tensor_parallel=tp,
               pipeline_parallel=pp, expert_parallel=ep, context_parallel=1, data_parallel=1,
               uneven_pipeline_layers=uneven, visible_gpus_per_node=8,
               island_gpus_per_node=8 if nodes > 1 else None)


def _spec():
    return AlgorithmSpec.from_json_file("examples/k31/algorithm-spec.async-age2-tis.json")


def _value(argv, flag):
    return argv[argv.index(flag) + 1]


def test_93_layers_over_pp6_without_an_uneven_split_is_refused_with_suggestions():
    with pytest.raises(mc.MilesConfigError, match="93 layers do not split evenly over pipeline parallel 6") as exc:
        mc.translate_run_config(_pro(uneven=None), _spec(), max_policy_age=2, full_parameter_sync=True)
    msg = str(exc.value)
    assert FIRST in msg and LAST in msg
    # every suggestion adds up: first + last + 4 x middle == 93
    suggestions = re.findall(r"first (\d+) / last (\d+) \((\d+) middle stages x (\d+)\)", msg)
    assert suggestions and all(int(f) + int(l) + int(n) * int(k) == 93 for f, l, n, k in suggestions)
    assert ("17", "16", "4", "15") in suggestions  # the split resolve_rl_run_config derives


def test_the_15_18_split_translates_for_the_full_k31_pro_synced_layout():
    argv = list(mc.translate_run_config(_pro(uneven=(15, 18)), _spec(), max_policy_age=2,
                                        full_parameter_sync=True).argv)
    assert (_value(argv, "--tensor-model-parallel-size"), _value(argv, "--pipeline-model-parallel-size"),
            _value(argv, "--expert-model-parallel-size"), _value(argv, "--context-parallel-size")) == ("8", "6", "8", "1")
    assert (_value(argv, FIRST), _value(argv, LAST), _value(argv, "--num-layers")) == ("15", "18", "93")
    assert (_value(argv, "--actor-num-nodes"), _value(argv, "--actor-num-gpus-per-node")) == ("6", "8")


@pytest.mark.parametrize("uneven", [(15, 17), (20, 20), (0, 33)])
def test_a_split_that_does_not_add_up_is_refused(uneven):
    with pytest.raises(mc.MilesConfigError, match="pipeline split|>= 1"):
        mc.translate_run_config(_pro(uneven=uneven), _spec(), max_policy_age=2, full_parameter_sync=True)


def test_the_extra_argv_split_is_checked_and_must_agree_with_the_run_config():
    spec = _spec()
    ok = mc.translate_run_config(_pro(uneven=None), spec, extra_argv=(FIRST, "15", f"{LAST}=18"),
                                 max_policy_age=2, full_parameter_sync=True)
    assert _value(list(ok.argv), FIRST) == "15"
    with pytest.raises(mc.MilesConfigError, match="pipeline split does not add up"):
        mc.translate_run_config(_pro(uneven=None), spec, extra_argv=(FIRST, "16", LAST, "18"),
                                max_policy_age=2, full_parameter_sync=True)
    with pytest.raises(mc.MilesConfigError, match="pipeline split conflict"):
        mc.translate_run_config(_pro(uneven=(15, 18)), spec, extra_argv=(FIRST, "17"),
                                max_policy_age=2, full_parameter_sync=True)


@pytest.mark.parametrize("layers,pp,uneven,ok", [
    (93, 1, None, True), (96, 6, None, True), (93, 6, None, False),
    (93, 6, (17, 16), True), (93, 6, (15, 18), True), (93, 6, (15, 17), False),
    (93, 2, (47, 46), True), (93, 2, (47, 47), False), (4, 2, None, True), (5, 2, None, False),
    (93, 1, (15, 18), False),
])
def test_rule_table(layers, pp, uneven, ok):
    if ok:
        mc.check_pipeline_split(layers, pp, uneven)
    else:
        with pytest.raises(mc.MilesConfigError):
            mc.check_pipeline_split(layers, pp, uneven)


def test_one_sided_split_from_extra_argv():
    mc.check_pipeline_split(93, 6, None, (LAST, "18"))       # 75 over 5 stages = 15
    with pytest.raises(mc.MilesConfigError):
        mc.check_pipeline_split(93, 6, None, (FIRST, "16"))  # 77 over 5 stages


@pytest.mark.parametrize("layers,pp", [(93, 6), (61, 4), (95, 8), (7, 3), (93, 2), (48, 6)])
def test_the_split_the_run_config_derives_always_passes(layers, pp):
    """resolve_rl_run_config's own uneven split (engine.run_config) satisfies the check."""
    middle, remainder = layers // pp, layers % pp
    uneven = None if remainder == 0 else (middle + (remainder + 1) // 2, middle + remainder // 2)
    mc.check_pipeline_split(layers, pp, uneven)
