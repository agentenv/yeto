import json

import pytest

from yeto.rl.engine.algorithm import BOUNDED_NONZERO_STD_FILTER, AlgorithmSpec
from yeto.rl.engine.capabilities import CapabilityMismatch, EngineCapabilities

from yeto.rl.elastic_benchmark.capabilities import attestation_from_dict


def _caps(**kw):
    base = dict(
        engine="miles-ports",
        runtime_fingerprint="sha256:" + "d" * 64,
        parameter_layouts=["lora"],
        placements=["colocated", "fixed-partition"],
        advantage_estimators=["grpo"],
        dynamic_sampling_filters=[BOUNDED_NONZERO_STD_FILTER],
        execution_modes=["colocated-serial"],
    )
    base.update(kw)
    return EngineCapabilities(**base)


def test_pr66_reader_parses_declaration():
    caps = _caps(certified_edges=[("P422", "P44", "rollout-only")], partitioned_driver=True)
    att = attestation_from_dict(json.loads(caps.to_json()))
    assert att.runtime_fingerprint == caps.runtime_fingerprint
    assert att.execution_modes == frozenset({"colocated-serial"})
    assert att.certified_edges == frozenset({("P422", "P44", "rollout-only")})
    assert att.partitioned_driver is True and att.auto_controller is False


def test_roundtrip():
    caps = _caps(port_verbs=["RolloutPool.add_engines"])
    assert EngineCapabilities.from_json(caps.to_json()) == caps


def test_handshake_rejects_with_options():
    caps = _caps()
    caps.check(layout="lora", placement="colocated", execution_mode="colocated-serial",
               algorithm=AlgorithmSpec())
    with pytest.raises(CapabilityMismatch, match=r"layout 'full'.*\['lora'\]"):
        caps.check(layout="full", placement="colocated",
                   execution_mode="colocated-serial", algorithm=AlgorithmSpec())
    ppo = type("A", (), {"advantage_estimator": "ppo", "dynamic_sampling_filter": None})()
    with pytest.raises(CapabilityMismatch, match=r"estimator 'ppo'.*\['grpo'\]"):
        caps.check(layout="lora", placement="colocated",
                   execution_mode="colocated-serial", algorithm=ppo)


def test_invalid_declarations():
    with pytest.raises(ValueError):
        _caps(runtime_fingerprint="abc")
    with pytest.raises(ValueError):
        _caps(parameter_layouts=["sparse"])
    with pytest.raises(ValueError):
        _caps(port_verbs=["RolloutPool.scale"])


# --- decoupling 4.5: backend-declared algorithm fields ----------------------


def test_algorithm_fields_undeclared_keeps_attestation_and_accepts_everything():
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.fake import fake_capabilities

    caps = fake_capabilities("ports-only", features={"eps_clip"})
    assert caps.algorithm_fields is None
    assert "algorithm_fields" not in caps.to_attestation_dict()
    caps.check(layout="lora", placement="colocated", execution_mode="colocated-serial",
               algorithm=AlgorithmSpec(loss={"eps_clip": 0.3}))


def test_algorithm_fields_declared_rejects_unsupported_field_before_launch():
    import pytest

    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.capabilities import CapabilityMismatch, EngineCapabilities
    from yeto.rl.engine.fake import fake_capabilities

    caps = fake_capabilities("ports-only", features={"eps_clip"},
                             algorithm_fields={"kl.coef"})
    kwargs = dict(layout="lora", placement="colocated", execution_mode="colocated-serial")
    assert AlgorithmSpec().set_fields() == frozenset()
    caps.check(algorithm=AlgorithmSpec(), **kwargs)  # default spec: nothing set
    spec = AlgorithmSpec(loss={"eps_clip": 0.3})
    assert "loss.eps_clip" in spec.set_fields()
    with pytest.raises(CapabilityMismatch, match=r"algorithm fields \['loss.eps_clip'\]"):
        caps.check(algorithm=spec, **kwargs)
    wider = fake_capabilities("ports-only", features={"eps_clip"},
                              algorithm_fields={"loss.eps_clip"})
    wider.check(algorithm=spec, **kwargs)
    again = EngineCapabilities.from_json(wider.to_json())
    assert again.algorithm_fields == frozenset({"loss.eps_clip"})
