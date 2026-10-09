"""decoupling 2.6: neutral weight-transport names, translated by the backend adapter."""

from __future__ import annotations

import pytest

from yeto.rl.engine.capabilities import WEIGHT_TRANSPORTS, BackendTraits
from yeto.rl.adapters.miles.entry import miles_capabilities, with_partitioned_serial
from yeto.rl.adapters.miles.traits import MILES_TRAITS

FP = "sha256:" + "1" * 64


def test_neutral_vocabulary():
    assert WEIGHT_TRANSPORTS == {"same-device-ipc", "collective-broadcast", "disk"}


def test_miles_translates_neutral_names_to_the_recorded_miles_names():
    assert MILES_TRAITS.transport_label("same-device-ipc") == "cuda-ipc"
    assert MILES_TRAITS.transport_label("collective-broadcast") == "nccl-broadcast"
    assert MILES_TRAITS.transport_label("disk") == "disk"  # not declared: neutral name recorded
    assert MILES_TRAITS.transport_label("nccl-broadcast") == "nccl-broadcast"  # backend name passes


def test_miles_capabilities_carry_the_traits_without_changing_the_attestation():
    caps = miles_capabilities(FP)
    assert caps.traits is MILES_TRAITS
    assert with_partitioned_serial(caps).traits is MILES_TRAITS
    assert "traits" not in caps.to_attestation_dict()
    assert "weight_transport_names" not in caps.to_json()


def test_unknown_neutral_name_is_refused_and_default_records_neutral():
    with pytest.raises(ValueError, match="unknown neutral weight transports"):
        BackendTraits(weight_transport_names={"cuda-ipc": "x"})
    assert BackendTraits().transport_label("collective-broadcast") == "collective-broadcast"
