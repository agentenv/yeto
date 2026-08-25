from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from yeto.rl.contracts import LocalStepReceipt
from yeto.rl.local_learner import (
    ComponentIdentity,
    DenseDiLoCoConnector,
    DenseTrainerUpdate,
    ParameterCut,
    ParameterFragmentCut,
    ParameterLayout,
    ParameterSpec,
    advance_parameter_cut_version,
    dense_sweep_session_contract_hash,
    parameter_cut_from_fragment_flats,
)

REVISION = "a" * 40
CONFIG_HASH = "b" * 64
BATCH_HASH = "c" * 64


def actor_specs():
    return (
        ParameterSpec("actor", "model.embed.weight", (2, 3), "float32", 6),
        ParameterSpec("actor", "model.norm.weight", (2,), "float32", 2),
        ParameterSpec("actor", "model.proj.weight", (2, 2), "float32", 4),
    )


class Provider:
    def __init__(self, values):
        self.values = {name: value.clone() for name, value in values.items()}
        self.safe = True
        self.applied = []

    def at_safe_boundary(self):
        return self.safe

    def read_parameters(self, specs):
        return {spec.wire_name: self.values[spec.wire_name] for spec in specs}

    def apply_complete_cut(self, layout, cut):
        staged = {}
        for fragment in cut.fragments:
            offset = 0
            for spec in layout.fragment_specs(fragment.fragment_id):
                staged[spec.wire_name] = (
                    fragment.flat[offset : offset + spec.numel]
                    .reshape(spec.shape)
                    .clone()
                )
                offset += spec.numel
        self.values = staged
        self.applied.append(cut.policy_version)


def grpo_layout(num_fragments=2):
    return ParameterLayout.create(
        algorithm="grpo",
        components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
        specs=actor_specs(),
        num_fragments=num_fragments,
    )


def actor_values(offset=0.0):
    return {
        spec.wire_name: torch.arange(spec.numel, dtype=torch.float32).reshape(
            spec.shape
        )
        + offset
        for spec in actor_specs()
    }


def receipt(anchor):
    return LocalStepReceipt(
        algorithm="grpo",
        learner_id=2,
        learner_generation=4,
        base_policy_version=anchor.policy_version,
        base_policy_hash=anchor.policy_hash,
        input_batch_hash=BATCH_HASH,
        trajectory_ids=("trajectory-1", "trajectory-2"),
        trained_tokens=32,
        optimizer_steps=1,
        optimizer_step_succeeded=True,
        parameter_layout_hash=anchor.layout_hash,
    )


def test_grpo_layout_is_actor_only_deterministic_and_role_qualified():
    first = grpo_layout()
    second = ParameterLayout.create(
        algorithm="grpo",
        components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
        specs=tuple(reversed(actor_specs())),
        num_fragments=2,
    )

    assert first.layout_hash == second.layout_hash
    assert first.fragments.tensor_names() == second.fragments.tensor_names()
    assert set(first.fragments.tensor_names()) == {
        f"actor::global::{spec.name}" for spec in actor_specs()
    }


def test_dense_session_contract_binds_semantics_profile_and_full_roster():
    layout = grpo_layout()
    same_schema_new_model = ParameterLayout.create(
        algorithm="grpo",
        components=(ComponentIdentity("actor", "f" * 40, CONFIG_HASH),),
        specs=actor_specs(),
        num_fragments=2,
    )
    contract = dense_sweep_session_contract_hash(
        layout,
        policy_rounds=3,
        learner_generations={0: 7, 1: 9},
    )
    assert len(contract) == 32
    assert contract != dense_sweep_session_contract_hash(
        same_schema_new_model,
        policy_rounds=3,
        learner_generations={0: 7, 1: 9},
    )
    assert contract != dense_sweep_session_contract_hash(
        layout,
        policy_rounds=4,
        learner_generations={0: 7, 1: 9},
    )
    assert contract != dense_sweep_session_contract_hash(
        layout,
        policy_rounds=3,
        learner_generations={0: 7, 1: 10},
    )
    assert contract != dense_sweep_session_contract_hash(
        layout,
        policy_rounds=3,
        learner_generations={0: 7, 1: 9},
        training_contract_hash="1" * 64,
    )


def test_sao_layout_adds_critic_without_changing_the_connector_contract():
    critic = ParameterSpec("critic", "value_head.weight", (1, 2), "float32", 2)
    layout = ParameterLayout.create(
        algorithm="sao",
        components=(
            ComponentIdentity("actor", REVISION, CONFIG_HASH),
            ComponentIdentity("critic", "d" * 40, "e" * 64),
        ),
        specs=(*actor_specs(), critic),
        num_fragments=2,
    )

    assert {spec.role for spec in layout.specs} == {"actor", "critic"}
    assert "critic::global::value_head.weight" in layout.fragments.tensor_names()
    with pytest.raises(ValueError, match="requires exactly"):
        ParameterLayout.create(
            algorithm="sao",
            components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
            specs=actor_specs(),
            num_fragments=2,
        )


def test_layout_distinguishes_identically_named_topology_shards():
    first = ParameterSpec(
        "actor",
        "model.proj.weight",
        (2, 2),
        "float32",
        4,
        "tp0-of-2",
    )
    second = replace(first, shard_id="tp1-of-2")
    layout = ParameterLayout.create(
        algorithm="grpo",
        components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
        specs=(first, second),
        num_fragments=2,
    )

    assert set(layout.fragments.tensor_names()) == {
        "actor::tp0-of-2::model.proj.weight",
        "actor::tp1-of-2::model.proj.weight",
    }


def test_owner_affine_layout_is_deterministic_and_never_mixes_topology_shards():
    specs = (
        ParameterSpec("actor", "model.a", (8,), "float32", 8, "tp0-of-2"),
        ParameterSpec("actor", "model.b", (4,), "float32", 4, "tp0-of-2"),
        ParameterSpec("actor", "model.c", (2,), "float32", 2, "tp0-of-2"),
        ParameterSpec("actor", "model.a", (7,), "float32", 7, "tp1-of-2"),
        ParameterSpec("actor", "model.b", (3,), "float32", 3, "tp1-of-2"),
    )
    first = ParameterLayout.create(
        algorithm="grpo",
        components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
        specs=specs,
        num_fragments=4,
        fragment_strategy="owner_affine",
    )
    second = ParameterLayout.create(
        algorithm="grpo",
        components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
        specs=tuple(reversed(specs)),
        num_fragments=4,
        fragment_strategy="owner_affine",
    )

    assert first.layout_hash == second.layout_hash
    assert first.fragments.tensor_names() == second.fragments.tensor_names()
    assert first.fragment_strategy == "owner_affine"
    assert [first.fragment_owner(index) for index in range(4)] == [
        ("actor", "tp0-of-2"),
        ("actor", "tp0-of-2"),
        ("actor", "tp1-of-2"),
        ("actor", "tp1-of-2"),
    ]
    for owner in sorted({(spec.role, spec.shard_id) for spec in first.specs}):
        expected = tuple(
            spec.wire_name
            for spec in first.specs
            if (spec.role, spec.shard_id) == owner
        )
        actual = tuple(
            spec.wire_name
            for fragment_id in range(first.fragments.num_fragments)
            if first.fragment_owner(fragment_id) == owner
            for spec in first.fragment_specs(fragment_id)
        )
        assert actual == expected


def test_owner_affine_layout_requires_at_least_one_fragment_per_owner():
    specs = (
        ParameterSpec("actor", "model.a", (1,), "float32", 1, "tp0-of-2"),
        ParameterSpec("actor", "model.a", (1,), "float32", 1, "tp1-of-2"),
    )
    with pytest.raises(ValueError, match="outside"):
        ParameterLayout.create(
            algorithm="grpo",
            components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
            specs=specs,
            num_fragments=1,
            fragment_strategy="owner_affine",
        )


def test_balanced_layout_identity_remains_backward_compatible():
    implicit = grpo_layout()
    explicit = ParameterLayout.create(
        algorithm="grpo",
        components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
        specs=actor_specs(),
        num_fragments=2,
        fragment_strategy="balanced",
    )

    assert implicit.layout_hash == explicit.layout_hash
    assert implicit.fragments.tensor_names() == explicit.fragments.tensor_names()


def test_dense_connector_exports_target_minus_base_and_reconstructs_local_cut():
    provider = Provider(actor_values())
    connector = DenseDiLoCoConnector(
        provider,
        grpo_layout(),
        learner_id=2,
        learner_generation=4,
        initial_policy_version=7,
    )
    anchor = connector.capture()
    provider.values = actor_values(0.25)

    update = connector.export_dense_update(
        anchor,
        receipt(anchor),
        target_policy_version=8,
    )
    reconstructed = connector.reconstruct_target(anchor, update)
    local = connector.capture()

    assert update.manifest.exchange_mode == "dense"
    assert update.manifest.base_policy_version == 7
    assert update.manifest.target_policy_version == 8
    assert update.manifest.fragment_count == 2
    assert all(
        torch.equal(
            fragment.target_minus_base,
            torch.full_like(fragment.target_minus_base, 0.25),
        )
        for fragment in update.fragments
    )
    assert all(
        torch.equal(expected.flat, actual.flat)
        for expected, actual in zip(
            reconstructed.fragments, local.fragments, strict=True
        )
    )


def test_single_learner_cut_promotion_reuses_payload_but_advances_identity():
    provider = Provider(actor_values())
    connector = DenseDiLoCoConnector(
        provider,
        grpo_layout(),
        learner_id=2,
        learner_generation=4,
        initial_policy_version=7,
    )
    local = connector.capture()

    promoted = advance_parameter_cut_version(
        connector.layout,
        local,
        target_policy_version=8,
    )

    assert promoted.policy_version == 8
    assert promoted.fragments is local.fragments
    assert promoted.policy_hash != local.policy_hash
    assert [fragment.payload_hash for fragment in promoted.fragments] == [
        fragment.payload_hash for fragment in local.fragments
    ]
    with pytest.raises(ValueError, match="advance"):
        advance_parameter_cut_version(
            connector.layout,
            local,
            target_policy_version=7,
        )


def test_fragment_flat_cut_matches_named_materialization_exactly():
    provider = Provider(actor_values())
    connector = DenseDiLoCoConnector(
        provider,
        grpo_layout(),
        learner_id=2,
        learner_generation=4,
        initial_policy_version=7,
    )
    expected = connector.capture()
    actual = parameter_cut_from_fragment_flats(
        connector.layout,
        policy_version=7,
        fragments={
            fragment.fragment_id: fragment.flat.clone()
            for fragment in expected.fragments
        },
    )

    assert actual.policy_hash == expected.policy_hash
    assert [fragment.payload_hash for fragment in actual.fragments] == [
        fragment.payload_hash for fragment in expected.fragments
    ]
    assert all(
        torch.equal(left.flat, right.flat)
        for left, right in zip(actual.fragments, expected.fragments, strict=True)
    )


def test_fragment_flat_cut_rejects_missing_and_nonfinite_payloads():
    layout = grpo_layout()
    with pytest.raises(ValueError, match="complete"):
        parameter_cut_from_fragment_flats(
            layout,
            policy_version=0,
            fragments={0: torch.zeros(layout.fragments.fragments[0].numel)},
        )
    payloads = {
        fragment_id: torch.zeros(fragment.numel)
        for fragment_id, fragment in enumerate(layout.fragments.fragments)
    }
    payloads[0][0] = torch.nan
    with pytest.raises(ValueError, match="malformed"):
        parameter_cut_from_fragment_flats(
            layout,
            policy_version=0,
            fragments=payloads,
        )


def test_dense_connector_rejects_wrong_receipt_stale_apply_and_unsafe_boundary():
    provider = Provider(actor_values())
    connector = DenseDiLoCoConnector(
        provider,
        grpo_layout(),
        learner_id=2,
        learner_generation=4,
        initial_policy_version=7,
    )
    anchor = connector.capture()
    provider.values = actor_values(0.5)

    with pytest.raises(ValueError, match="receipt"):
        connector.export_dense_update(
            anchor,
            replace(receipt(anchor), learner_generation=5),
            target_policy_version=8,
        )

    update = connector.export_dense_update(
        anchor,
        receipt(anchor),
        target_policy_version=8,
    )
    target = connector.reconstruct_target(anchor, update)
    connector.apply_global_cut(target, expected_base_version=7)
    assert provider.applied == [8]
    with pytest.raises(ValueError, match="stale"):
        connector.apply_global_cut(target, expected_base_version=7)

    provider.safe = False
    with pytest.raises(RuntimeError, match="safe boundary"):
        connector.capture()


def test_dense_connector_rejects_tampered_or_incomplete_parameter_cut():
    provider = Provider(actor_values())
    connector = DenseDiLoCoConnector(
        provider,
        grpo_layout(),
        learner_id=2,
        learner_generation=4,
        initial_policy_version=7,
    )
    anchor = connector.capture()
    incomplete = replace(anchor, fragments=anchor.fragments[:1])
    with pytest.raises(ValueError, match="complete ordered"):
        connector.apply_global_cut(incomplete, expected_base_version=7)

    first = anchor.fragments[0]
    tampered_fragment = ParameterFragmentCut(
        first.fragment_id,
        first.flat + 1.0,
        first.payload_hash,
    )
    tampered = ParameterCut(
        anchor.policy_version + 1,
        anchor.layout_hash,
        (tampered_fragment, *anchor.fragments[1:]),
        anchor.policy_hash,
    )
    with pytest.raises(ValueError, match="malformed"):
        connector.apply_global_cut(tampered, expected_base_version=7)


def test_dense_connector_rejects_tampered_delta_and_policy_hash():
    provider = Provider(actor_values())
    connector = DenseDiLoCoConnector(
        provider,
        grpo_layout(),
        learner_id=2,
        learner_generation=4,
        initial_policy_version=7,
    )
    anchor = connector.capture()
    provider.values = actor_values(0.5)
    update = connector.export_dense_update(
        anchor,
        receipt(anchor),
        target_policy_version=8,
    )
    first = update.fragments[0]
    corrupted_update = DenseTrainerUpdate(
        update.manifest,
        update.receipt,
        (
            replace(first, target_minus_base=first.target_minus_base + 1.0),
            *update.fragments[1:],
        ),
    )
    with pytest.raises(ValueError, match="fragment is malformed"):
        connector.reconstruct_target(anchor, corrupted_update)

    target = connector.reconstruct_target(anchor, update)
    with pytest.raises(ValueError, match="policy hash"):
        connector.apply_global_cut(
            replace(target, policy_hash="0" * 64),
            expected_base_version=7,
        )
