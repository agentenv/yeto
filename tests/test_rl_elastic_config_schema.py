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


@pytest.mark.parametrize("pinned", [None, "unresolved"])
def test_unpinned_study_fingerprint_fails_closed(pinned):
    attest = caps.attestation_from_dict(
        {"runtime_fingerprint": "sha256:" + "1" * 64, "execution_modes": ["partitioned-serial"],
         "partitioned_driver": True}
    )
    assert "unresolved" in caps.fingerprint_rejection(pinned, attest)
    m = _formal()
    m["identity"]["fingerprints"]["runtime"] = pinned
    plan = build_plan(m, attest, study_hash="h")
    target = [i for i in plan.items if i.kind != "legacy-fixed"]
    assert target and all(i.status == caps.STATUS_BLOCKED for i in target)


def test_spec_mode_name_serial_colocated_is_an_alias():
    from yeto.rl.elastic_benchmark.manifest import canonical_execution_mode, validate_manifest
    from yeto.rl.engine.capabilities import EngineCapabilities
    from yeto.rl.engine.execution_profile import ExecutionProfile

    assert canonical_execution_mode("serial-colocated") == "colocated-serial"
    p = ExecutionProfile(name="p", execution_mode="serial-colocated", outer_protocol="none")
    assert p.execution_mode == "colocated-serial"
    attest = caps.attestation_from_dict({"execution_modes": ["serial-colocated"]})
    assert attest.execution_modes == frozenset({"colocated-serial"})
    c = EngineCapabilities(
        engine="e", runtime_fingerprint="sha256:" + "0" * 64, parameter_layouts=[],
        placements=["colocated"], advantage_estimators=[], dynamic_sampling_filters=[],
        execution_modes=["serial-colocated"],
    )
    assert c.execution_modes == frozenset({"colocated-serial"})
    m = example_manifest()
    m["profile"]["execution_mode"] = "serial-colocated"
    validate_manifest(m)
    with pytest.raises(ManifestError):
        m["profile"]["execution_mode"] = "serial-overlap"
        validate_manifest(m)


# -- alignment A1/A4: algorithm hash in the execution contract ----------------


def test_trainer_edges_are_certified_per_algorithm_hash():
    from yeto.rl.elastic_benchmark import capabilities as caps
    from yeto.rl.elastic_benchmark.manifest import ManifestError, example_manifest, validate_manifest
    from yeto.rl.elastic_benchmark.plan import build_plan
    from yeto.rl.engine.algorithm import AlgorithmSpec

    grpo = AlgorithmSpec().sha256()
    other = AlgorithmSpec(kl_coef=0.1).sha256()
    manifest = example_manifest()
    manifest["identity"]["fingerprints"]["runtime"] = "sha256:runtime"
    bad = __import__("copy").deepcopy(manifest)
    bad["profile"]["algorithm_spec_sha256"] = "GRPO"
    with pytest.raises(ManifestError, match="algorithm_spec_sha256"):
        validate_manifest(bad)

    def attested(hashes):
        edges = [
            {"source": "P62", "target": "P44", "kind": "role-transfer", "algorithm_spec_sha256": hashes},
            {"source": "P44", "target": "P62", "kind": "role-transfer", "algorithm_spec_sha256": hashes},
        ]
        return caps.attestation_from_dict(
            {
                "runtime_fingerprint": "sha256:runtime",
                "execution_modes": ["partitioned-serial"],
                "partitioned_driver": True,
                "certified_edges": edges,
            }
        )

    def rebuild_status(profile_hash, attestation):
        m = __import__("copy").deepcopy(manifest)
        if profile_hash is not None:
            m["profile"]["algorithm_spec_sha256"] = profile_hash
        plan = build_plan(m, attestation, study_hash="h")
        return {i.key.arm: (i.status, i.reason) for i in plan.items}["rebuild"]

    assert rebuild_status(grpo, attested([grpo])) == ("supported", None)
    status, reason = rebuild_status(other, attested([grpo]))
    assert status == "blocked_dependency" and "not certified for algorithm" in reason
    status, reason = rebuild_status(None, attested([grpo]))
    assert status == "blocked_dependency" and "algorithm_spec_sha256" in reason
    status, _ = rebuild_status(grpo, attested([]))
    assert status == "blocked_dependency"
    with pytest.raises(ManifestError):
        attested(["nothex"])
