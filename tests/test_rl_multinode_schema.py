"""rl-multinode-island tasks 1.1/1.2: --gpu min nodes, cfg topology, placement spellings."""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from yeto.gpu_spec import parse_gpu_spec, require_min_nodes
from yeto.rl.elastic_benchmark import capabilities as caps
from yeto.rl.elastic_benchmark.manifest import ManifestError
from yeto.rl.engine import multinode as mn

LEGACY = json.loads(
    '{"configs":{"T4R2S2":{"trainer":4,"rollout":2,"standby":2,"rollout_engine_gpus":1},'
    '"T4R4S0":{"trainer":4,"rollout":4,"standby":0,"rollout_engine_gpus":1}},'
    '"edges":[{"source":"T4R2S2","target":"T4R4S0","kind":"rollout-only"},'
    '{"source":"T4R4S0","target":"T4R2S2","kind":"rollout-only"}]}'
)


def _pool(nodes=2, g=8):
    return [{"uuid": f"GPU-n{n}-{i}", "model": "H100", "node": f"n{n}", "index": i}
            for n in range(nodes) for i in range(g)]


def _cfg(placement=None, **extra):
    block = {"trainer": 8, "rollout": 8, "standby": 0, "rollout_engine_gpus": 8,
             "parallel": {"tp": 2, "pp": 1}}
    if placement is not None:
        block["placement"] = placement
    return {"nodes": 2, "gpus_per_node": 8, "gpus": _pool(), "configs": {"T8R8S0": block},
            "edges": [], **extra}


# ---- 1.1 --gpu / min nodes

def test_gpu_spec_two_nodes_and_min_nodes():
    (spec,) = parse_gpu_spec("nebius:2x8xh100")
    assert (spec.num_nodes, spec.gpus_per_node, spec.total_gpus) == (2, 8, 16)
    assert require_min_nodes(spec, 2) is spec
    with pytest.raises(ValueError, match="at least 3 node"):
        require_min_nodes(spec, 3)
    (one,) = parse_gpu_spec("nebius:8xh100")
    assert str(one) == "nebius:1x8xH100" and require_min_nodes(one, 1) is one
    with pytest.raises(ValueError):
        require_min_nodes(one, 0)


def test_min_nodes_derivation():
    assert mn.min_nodes(trainer_min_gpus=2, rollout_min_gpus=8, standby_gpus=0, gpus_per_node=8) == 2
    assert mn.min_nodes(trainer_min_gpus=1, rollout_min_gpus=1, standby_gpus=0, gpus_per_node=8) == 1
    assert mn.min_nodes(trainer_min_gpus=16, rollout_min_gpus=8, standby_gpus=8, gpus_per_node=8) == 4
    with pytest.raises(mn.TopologyError, match="does not fit"):
        mn.min_nodes(trainer_min_gpus=2, rollout_min_gpus=16, standby_gpus=0, gpus_per_node=8)
    with pytest.raises(mn.TopologyError, match="whole number"):
        mn.min_nodes(trainer_min_gpus=12, rollout_min_gpus=1, standby_gpus=0, gpus_per_node=8)


# ---- 1.2 schema

def test_legacy_cfg_unchanged_snapshot():
    configs = caps.parse_configs(copy.deepcopy(LEGACY))
    assert {n: dataclasses.asdict(c) for n, c in configs.items()} == {
        "T4R2S2": {"name": "T4R2S2", "trainer": 4, "rollout": 2, "standby": 2, "parallel": (),
                   "rollout_engine_gpus": 1, "placement": None, "placement_slots": None,
                   "gradient_accumulation_declared": None, "capacity": None},
        "T4R4S0": {"name": "T4R4S0", "trainer": 4, "rollout": 4, "standby": 0, "parallel": (),
                   "rollout_engine_gpus": 1, "placement": None, "placement_slots": None,
                   "gradient_accumulation_declared": None, "capacity": None},
    }
    assert mn.topology_of(LEGACY) is None
    assert caps.config_rejection(configs["T4R2S2"], profile={"parameter_mode": "lora", "global_batch": 8,
                                                               "micro_batch": 1}, pool_size=None) is None


@pytest.mark.parametrize("placement", [
    {"trainer": [f"n0:{i}" for i in range(8)], "rollout": [[f"n1:{i}" for i in range(8)]], "standby": []},
    {"trainer": list(range(8)), "rollout": [list(range(8, 16))], "standby": []},
    {"trainer": [f"GPU-n0-{i}" for i in range(8)], "rollout": [[f"GPU-n1-{i}" for i in range(8)]], "standby": []},
])
def test_three_spellings_normalize_identically(placement):
    cfg = caps.parse_configs(_cfg(placement))["T8R8S0"]
    assert cfg.placement_slots == {"trainer": [(0, i) for i in range(8)],
                                   "rollout": [[(1, i) for i in range(8)]], "standby": []}
    assert caps.placement_rejection(cfg, caps.pool_gpus(_cfg(placement))) is None


def test_mixed_spellings_rejected():
    placement = {"trainer": ["n0:0", 1, 2, 3, 4, 5, 6, 7], "rollout": [list(range(8, 16))], "standby": []}
    with pytest.raises(ManifestError, match="mixes spellings"):
        caps.parse_configs(_cfg(placement))


def test_pool_size_vs_topology_rejected():
    bad = _cfg()
    bad["gpus"] = _pool()[:12]
    with pytest.raises(ManifestError, match="12 GPUs but nodes x gpus_per_node = 2 x 8 = 16"):
        caps.parse_configs(bad)
    bad = _cfg()
    bad["gpus"][3]["index"] = 9
    with pytest.raises(ManifestError, match="not 0..7"):
        caps.parse_configs(bad)
    small = _cfg()
    small["configs"]["T8R8S0"]["rollout"] = 4
    with pytest.raises(ManifestError, match="uses 12 GPUs but the island is 2 x 8 = 16"):
        caps.parse_configs(small)


def test_unresolved_pool_still_accepts_slot_spellings():
    cfg = _cfg({"trainer": [f"n0:{i}" for i in range(8)], "rollout": [[f"n1:{i}" for i in range(8)]],
                "standby": []})
    cfg["gpus"] = []
    assert caps.parse_configs(cfg)["T8R8S0"].placement_slots["rollout"] == [[(1, i) for i in range(8)]]
    cfg["configs"]["T8R8S0"]["placement"] = {
        "trainer": [f"GPU-n0-{i}" for i in range(8)], "rollout": [[f"GPU-n1-{i}" for i in range(8)]],
        "standby": []}
    with pytest.raises(ManifestError, match="neither n<k>:<g> nor a pool GPU uuid"):
        caps.parse_configs(cfg)


def test_node_rules_in_cfg():
    engine_spans = {"trainer": list(range(8)), "rollout": [list(range(4, 12))], "standby": []}
    engine_spans["trainer"] = list(range(4)) + list(range(12, 16))
    with pytest.raises(ManifestError, match="rollout engine .* spans nodes"):
        caps.parse_configs(_cfg(engine_spans))
    tp_spans = {"trainer": [7, 8] + list(range(6)), "rollout": [list(range(9, 16)) + [6]], "standby": []}
    with pytest.raises(ManifestError, match="(model-parallel group .* spans nodes|rollout engine .* spans nodes)"):
        caps.parse_configs(_cfg(tp_spans))
    counts = {"trainer": list(range(8)), "rollout": [list(range(8, 12))], "standby": list(range(12, 16))}
    with pytest.raises(ManifestError, match="maps T8 R4 S4 but declares T8 R8 S0"):
        caps.parse_configs(_cfg(counts))


def test_ep_whole_node_alignment():
    good = _cfg({"trainer": list(range(8)), "rollout": [list(range(8, 16))], "standby": []})
    good["configs"]["T8R8S0"]["parallel"] = {"tp": 2, "pp": 1, "ep": 4}
    assert caps.parse_configs(good)["T8R8S0"].dims["ep"] == 4
    bad = copy.deepcopy(good)
    bad["configs"]["T8R8S0"]["parallel"] = {"tp": 2, "pp": 1, "ep": 8}
    with pytest.raises(ManifestError, match="expert parallel 8 must divide the 4"):
        caps.parse_configs(bad)
