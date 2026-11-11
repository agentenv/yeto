"""Synced islands on different cards are refused before any cloud resource."""

from types import SimpleNamespace

import pytest

from yeto.gpu_spec import parse_gpu_spec
from yeto.launch_preflight import PreflightError, check_compat_groups


def _args(**kw):
    base = dict(training_mode="rl", rl_compat_group=None, rl_single_island_no_sync=False,
                rl_sample_role="trainer", rl_island_sample_role=None, rl_node_class=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_mixed_card_trainers_are_refused():
    specs = parse_gpu_spec("modal:1x1xh100,modal:1x1xl40s")
    with pytest.raises(PreflightError, match="nvidia-h100: islands \\[0\\], nvidia-l40s: islands \\[1\\]"):
        check_compat_groups(_args(), specs)


def test_declared_group_or_one_card_passes():
    check_compat_groups(_args(rl_compat_group="nvidia-h100"),
                        parse_gpu_spec("modal:1x1xh100,modal:1x1xl40s"))
    check_compat_groups(_args(), parse_gpu_spec("modal:1x1xl40s,modal:1x1xl40s"))
    check_compat_groups(_args(rl_single_island_no_sync=True), parse_gpu_spec("modal:1x1xh100"))


def test_sampler_island_of_a_synced_group_is_not_a_syncer_member():
    specs = parse_gpu_spec("modal:1x1xl40s,modal:1x1xl40s,modal:1x1xh100")
    check_compat_groups(_args(rl_island_sample_role=["2:sampler"]), specs)
    with pytest.raises(PreflightError):
        check_compat_groups(_args(rl_island_sample_role=["1:sampler"]), specs)
