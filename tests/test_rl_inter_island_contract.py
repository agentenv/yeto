"""Syncer semantic profile hash: island pool membership stays out of it. No Ray."""

from __future__ import annotations

import pytest

from test_rl_sao_streaming_runtime import _syncer_profile
from yeto.rl.engine.island_ledger import contract_fields
from yeto.syncer_profile import SyncerSemanticProfile

LEGACY = "85a50d0e4fd5103b81a2295536fca0e5ad96cd50531fb1396208af8459be571e"
# Golden for the Rust side (outer_delayed_nesterov_period=4 on the SAO profile).
DELAYED_NESTEROV_4 = "2bf20f25b4d40d6d3db1d91e38d199e0347f9dace3fc481a2948fa7f1c99f126"


def test_fixed_roster_hash_unchanged():
    assert SyncerSemanticProfile.from_mapping(_syncer_profile()).sha256 == LEGACY


@pytest.mark.parametrize("field, value", [
    ("island_scheduling_mode", "legacy"), ("island_scheduling_mode", "elastic"),
    ("quorum_theta", 0.75), ("carry_gamma", 0.5), ("soft_deadline_s", 900), ("q_min", 1),
    ("max_carry_lag", 2),
])
def test_island_scheduling_fields_are_not_profile_fields(field, value):
    raw = _syncer_profile()
    raw[field] = value
    with pytest.raises(ValueError):
        SyncerSemanticProfile.from_mapping(raw)


def test_contract_field_names_agree():
    fields = contract_fields("elastic", soft_deadline_s=900)
    assert set(fields) <= {"island_scheduling_mode", "quorum_theta", "carry_gamma",
                           "soft_deadline_s", "q_min", "max_carry_lag"}
    assert "island_scheduling" not in fields


def test_delayed_nesterov_hash_is_appended_only_when_set():
    # Golden for server.rs::delayed_nesterov_profile_hash_matches_the_python_canonical_vector.
    raw = _syncer_profile()
    raw["outer_delayed_nesterov_period"] = 4
    profile = SyncerSemanticProfile.from_mapping(raw)
    assert profile.canonical_bytes().endswith(
        b"yeto-syncer-delayed-nesterov-v1\0" + (4).to_bytes(4, "little"))
    assert profile.sha256 == DELAYED_NESTEROV_4
    raw["outer_delayed_nesterov_period"] = None
    assert SyncerSemanticProfile.from_mapping(raw).sha256 == LEGACY
    raw["outer_delayed_nesterov_period"] = 1
    with pytest.raises(ValueError):
        SyncerSemanticProfile.from_mapping(raw)  # N = 1 is spelled "absent"
