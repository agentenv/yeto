"""rl-infra-spec 2.1: launcher shapes a LoRA fixed partition (trainer = node - rollout - standby)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto import launcher
from yeto.gpu_spec import ClusterSpec

SPEC = ClusterSpec(cloud="modal", region=None, num_nodes=1, gpus_per_node=4, gpu="H100")


def test_actor_gpus_per_node():
    colo = SimpleNamespace(rl_placement="colocated", rollout_num_gpus=0)
    assert launcher.rl_actor_gpus_per_node(colo, SPEC) == 4
    part = SimpleNamespace(rl_placement="fixed-partition", rollout_num_gpus=2, rl_standby_gpus=0)
    assert launcher.rl_actor_gpus_per_node(part, SPEC) == 2
    part.rl_standby_gpus = 1
    assert launcher.rl_actor_gpus_per_node(part, SPEC) == 1
    with pytest.raises(ValueError, match="rollout-num-gpus"):
        launcher.rl_actor_gpus_per_node(
            SimpleNamespace(rl_placement="fixed-partition", rollout_num_gpus=0), SPEC)


def test_partition_flags_forward_rollout_gpus_only_when_partitioned():
    base = dict(rl_expected_algorithm_sha256="a" * 64, rl_algorithm_spec=None,
                rl_allow_unverified_mechanism=None, rl_engine="ports")
    _, colo = launcher._ports_algorithm_flags(SimpleNamespace(**base, rl_placement="colocated",
                                                              rl_standby_gpus=0, rollout_num_gpus=0))
    assert "--rollout-num-gpus" not in colo and "--rl-placement" not in colo
    _, part = launcher._ports_algorithm_flags(SimpleNamespace(
        **base, rl_placement="fixed-partition", rl_standby_gpus=0, rollout_num_gpus=2))
    assert "--rl-placement fixed-partition" in part and "--rollout-num-gpus 2" in part
