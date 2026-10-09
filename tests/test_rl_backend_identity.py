"""yeto-framework-decoupling phase 5 (6.1 / 6.2, design D7): backend identity
hash beside the algorithm/contract hashes, bound into the syncer session."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from yeto.protocol import encode_hello, layout_fingerprint
from yeto.rl.adapters.miles import identity as miles_identity
from yeto.rl.adapters.miles.pins import MILES_COMMIT, MILES_NEXT_COMMIT
from yeto.rl.engine.backend_identity import (
    BackendIdentity,
    BackendIdentityMismatch,
    check_identity_match,
    param_map_sha256,
    session_contract_hash,
)

GOLDEN = Path(__file__).resolve().parent / "golden" / "decoupling"
FAKE_VERL = BackendIdentity("verl", "a" * 40, "nvidia", param_map_sha256({"kind": "hf-to-fsdp"}))


def test_miles_identity_is_fixed_and_hash_is_stable():
    ports, legacy = miles_identity.backend_identity("ports"), miles_identity.backend_identity("legacy")
    assert ports.engine == "miles" and ports.device_family == "nvidia"
    assert ports.engine_commit == MILES_NEXT_COMMIT and legacy.engine_commit == MILES_COMMIT
    assert ports.sha256() == miles_identity.backend_identity("ports").sha256()
    assert BackendIdentity.from_dict(ports.to_dict()) == ports
    assert ports.sha256() == hashlib.sha256(ports.canonical_json().encode()).hexdigest()


@pytest.mark.parametrize("field,value", [("engine", "verl"), ("engine_commit", "b" * 40),
                                         ("device_family", "ascend"),
                                         ("param_map_sha256", "c" * 64)])
def test_changing_any_field_changes_the_identity_hash(field, value):
    base = miles_identity.backend_identity("ports")
    assert dataclasses.replace(base, **{field: value}).sha256() != base.sha256()


def test_golden_samples_keep_both_old_hashes_and_carry_the_identity():
    expected = miles_identity.backend_identity("ports")
    for path in sorted(GOLDEN.glob("*.json")):
        if path.name == "fake_engine_tapes.json":
            continue
        sample = json.loads(path.read_text())
        # the identity is a parallel hash: algorithm/contract hashes do not include it
        assert sample["backend_identity"]["sha256"] == expected.sha256(), path.name
        assert sample["algorithm_sha256"] == sample["execution_profile"]["profile"]["algorithm_spec_sha256"]


def test_miles_vs_verl_handshake_is_refused_and_names_both():
    miles = miles_identity.backend_identity("ports")
    with pytest.raises(BackendIdentityMismatch) as err:
        check_identity_match(miles, FAKE_VERL)
    assert "'engine': 'miles'" in str(err.value) and "'engine': 'verl'" in str(err.value)
    check_identity_match(miles, miles_identity.backend_identity("ports"))  # two Miles islands: fine


def _layout():
    from yeto.rl.core import CanonicalTensorSpec, build_rl_fragment_layout

    specs = (CanonicalTensorSpec("base_model.model.layers.0.q.lora_A.weight", (4, 8), "float32", 32),
             CanonicalTensorSpec("base_model.model.layers.0.q.lora_B.weight", (8, 4), "float32", 32))
    return build_rl_fragment_layout(specs, 2)


def test_syncer_session_contract_binds_identity():
    """The syncer admits only byte-identical HELLO session contracts (SessionSpec
    equality in syncer/src/server.rs): different identities -> different contracts."""
    layout = _layout()
    fp = layout_fingerprint(layout)
    miles = miles_identity.backend_identity("ports").sha256()
    a = session_contract_hash(fp, miles)
    assert a == session_contract_hash(fp, miles) and len(a) == 32
    assert a != session_contract_hash(fp, FAKE_VERL.sha256())
    assert a != fp  # differs from the pre-phase-5 layout-only contract
    hello = encode_hello(0, 0, layout, 1, session_contract_hash=a)
    assert a in hello


def test_bridges_send_the_identity_bound_contract():
    from yeto.rl import bridge, decoupled

    layout = _layout()
    sha = miles_identity.backend_identity("ports").sha256()
    for module in (bridge, decoupled):
        assert module._session_contract(layout, None) is None  # legacy callers unchanged
        assert module._session_contract(layout, sha) == session_contract_hash(layout_fingerprint(layout), sha)
    for cls in (bridge.BridgeConfig, decoupled.DecoupledBridgeConfig):
        assert "backend_identity_sha256" in {f.name for f in dataclasses.fields(cls)}


def test_dense_and_sao_training_contracts_reference_the_identity():
    from yeto.rl import local_learner

    src = Path(local_learner.__file__).read_text()
    assert "backend_identity_sha256" in src
    from yeto.rl.adapters.miles.models import full_parameter_dense, sao_streaming

    for module in (full_parameter_dense, sao_streaming):
        assert module._legacy_identity_sha256() == miles_identity.backend_identity("legacy").sha256()
