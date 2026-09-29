"""rl-infra-spec task 1.6: config / edge schema pure rejections (no GPU)."""

from __future__ import annotations

import copy

import pytest

from yeto.rl.elastic_benchmark import capabilities as caps
from yeto.rl.elastic_benchmark.manifest import ManifestError, example_manifest, manifest_hash
from yeto.rl.elastic_benchmark.plan import build_plan

G = [f"GPU-{i:08d}" for i in range(8)]


def _formal():
    m = example_manifest("formal")
    for gpu in m["resources"]["gpus"]:
        gpu["memory_gib"] = 80
    return m


def _reject(m, name):
    configs = caps.parse_configs(m["resources"])
    return caps.config_rejection(
        configs[name], profile=m["profile"], pool_size=8, pool=caps.pool_gpus(m["resources"])
    )


def test_explicit_placement_accepts_a_legal_map():
    m = _formal()
    m["resources"]["configs"]["P62"]["placement"] = {
        "trainer": G[:6], "rollout": [[G[6]], [G[7]]], "standby": []
    }
    assert _reject(m, "P62") is None


@pytest.mark.parametrize(
    "placement, match",
    [
        ({"trainer": G[:6], "rollout": [[G[6]], [G[6]]]}, "more than one slot"),
        ({"trainer": G[:5] + ["GPU-x"], "rollout": [[G[6]], [G[7]]]}, "not in pool"),
        ({"trainer": G[:5], "rollout": [[G[6]], [G[7]]], "standby": [G[5]]}, "config declares"),
        ({"trainer": G[:6], "rollout": [G[6], G[7]]}, "per-engine GPU lists"),
    ],
)
def test_illegal_gpu_mapping_is_rejected(placement, match):
    m = _formal()
    m["resources"]["configs"]["P62"]["placement"] = placement
    assert match in _reject(m, "P62")


def test_engine_across_nodes_is_rejected():
    m = _formal()
    for gpu in m["resources"]["gpus"][4:]:
        gpu["node"] = "n1"
    m["resources"]["configs"]["P62"]["rollout_engine_gpus"] = 2
    m["resources"]["configs"]["P62"]["placement"] = {"trainer": G[:6], "rollout": [[G[6], G[7]]]}
    assert _reject(m, "P62") is None
    m["resources"]["configs"]["P62"]["placement"] = {
        "trainer": G[:2] + G[3:7], "rollout": [[G[2], G[7]]]
    }
    assert "spans nodes" in _reject(m, "P62")


def test_unknown_runtime_fingerprint_blocks_every_target_arm():
    m = _formal()
    pinned = m["identity"]["fingerprints"]["runtime"]
    attest = caps.attestation_from_dict(
        {"runtime_fingerprint": "sha256:" + "1" * 64, "execution_modes": ["partitioned-serial"],
         "partitioned_driver": True}
    )
    plan = build_plan(m, attest, study_hash=manifest_hash(m))
    target = [i for i in plan.items if i.kind != "legacy-fixed"]
    assert target and all(i.status == caps.STATUS_BLOCKED for i in target)
    assert all("unknown runtime fingerprint" in i.reason for i in target)
    ok = caps.attestation_from_dict(
        {"runtime_fingerprint": pinned, "execution_modes": ["partitioned-serial"],
         "partitioned_driver": True}
    )
    plan = build_plan(m, ok, study_hash=manifest_hash(m))
    assert any(i.status == caps.STATUS_SUPPORTED and i.kind == "target-fixed-sweep" for i in plan.items)


def test_dense_full_dp_above_one_is_rejected():
    m = _formal()
    m["profile"]["parameter_mode"] = "full"
    assert "DP=1" in _reject(m, "P62")


def test_batch_semantics_mismatches_are_rejected():
    m = _formal()
    m["resources"]["configs"]["P62"]["gradient_accumulation"] = 7  # 48 / 6 = 8
    assert "declared gradient_accumulation" in _reject(m, "P62")
    m["resources"]["configs"]["P62"]["gradient_accumulation"] = 8
    assert _reject(m, "P62") is None
    m2 = _formal()
    m2["profile"]["optimizer_steps_per_round"] = 5
    assert "not divisible by 5 optimizer steps" in _reject(m2, "P44")
    m3 = _formal()
    m3["profile"]["optimizer_steps_per_round"] = 2  # per-step 24; DP6 -> 4 ok, DP4 -> 6 ok
    configs = caps.parse_configs(m3["resources"])
    rows = {r["config"]: r for r in caps.config_table(configs, profile=m3["profile"], pool_size=8)}
    assert rows["P62"]["gradient_accumulation"] == 4 and rows["P44"]["gradient_accumulation"] == 6
    m4 = _formal()
    m4["resources"]["configs"]["P62"]["parallel"] = {"tp": 4}
    assert "not divisible by TP*PP*CP" in _reject(m4, "P62")
    m5 = _formal()
    m5["resources"]["configs"]["P44"]["capacity"] = {"gpu_mem_peak_gib": 95}
    assert "exceeds pool GPU memory" in _reject(m5, "P44")


def test_edges_keep_complex_parallel_dims_and_declare_recovery():
    m = _formal()
    r = copy.deepcopy(m["resources"])
    r["configs"]["P62"]["parallel"] = {"tp": 2}
    with pytest.raises(ManifestError, match="only DP may change"):
        caps.validate_edges(r, caps.parse_configs(r))
    r = copy.deepcopy(m["resources"])
    r["edges"][0]["recovery"] = "magic"
    with pytest.raises(ManifestError, match="recovery"):
        caps.validate_edges(r, caps.parse_configs(r))
    r["edges"][0]["recovery"] = "reinit-rollout"
    caps.validate_edges(r, caps.parse_configs(r))
