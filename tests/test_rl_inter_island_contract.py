"""Syncer contract hash with the island-scheduling block (Python side of elastic.rs). No Ray."""

from __future__ import annotations

import pytest

from test_rl_sao_streaming_runtime import _syncer_profile
from yeto.rl.engine.island_ledger import contract_fields
from yeto.syncer_profile import SyncerSemanticProfile

LEGACY = "85a50d0e4fd5103b81a2295536fca0e5ad96cd50531fb1396208af8459be571e"
# Golden for the Rust side (elastic defaults 0.75 / 0.5 / 900 / 1 / 2 on the SAO profile).
ELASTIC_DEFAULT = "c677c127b1ad3e1da179863ee02b6b2e1bc232edb03b088bad62251e5d8d9ba5"
# Golden for the Rust side (outer_delayed_nesterov_period=4 on the SAO profile).
DELAYED_NESTEROV_4 = "2bf20f25b4d40d6d3db1d91e38d199e0347f9dace3fc481a2948fa7f1c99f126"


def _elastic(**kw):
    raw = _syncer_profile()
    raw.update(island_scheduling_mode="elastic", quorum_theta=0.75, carry_gamma=0.5,
               soft_deadline_s=900, q_min=1, max_carry_lag=2)
    raw.update(kw)
    return raw


def test_legacy_hash_unchanged():
    assert SyncerSemanticProfile.from_mapping(_syncer_profile()).sha256 == LEGACY
    raw = _syncer_profile(); raw["island_scheduling_mode"] = "legacy"
    assert SyncerSemanticProfile.from_mapping(raw).sha256 == LEGACY


def test_elastic_default_hash_and_layout():
    p = SyncerSemanticProfile.from_mapping(_elastic())
    assert p.sha256 == ELASTIC_DEFAULT
    tail = p.canonical_bytes()[-(33 + 1 + 22 + 1 + 7 + 8 + 8 + 8 + 4 + 4):]
    assert tail.startswith(b"yeto-syncer-island-scheduling-v1\0\x16island_scheduling_mode\x07elastic")
    assert SyncerSemanticProfile.from_mapping(_elastic(q_min=2)).sha256 != ELASTIC_DEFAULT


def test_invalid_island_scheduling_profiles():
    raw = _syncer_profile(); raw["quorum_theta"] = 0.75
    with pytest.raises(ValueError):
        SyncerSemanticProfile.from_mapping(raw)  # parameter without elastic
    raw = _elastic(); del raw["carry_gamma"]
    with pytest.raises(ValueError):
        SyncerSemanticProfile.from_mapping(raw)
    with pytest.raises(ValueError):
        SyncerSemanticProfile.from_mapping(_elastic(island_scheduling_mode="auto"))
    with pytest.raises(ValueError):
        SyncerSemanticProfile.from_mapping(_elastic(quorum_theta=0.0))


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
