import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MOD = _load(
    "replay_exact_loo_probe_test",
    ROOT / "scripts" / "replay_exact_loo_probe.py",
)
AGG = _load(
    "aggregate_exact_loo_probe_test",
    ROOT / "scripts" / "aggregate_exact_loo_probe.py",
)


def test_paired_decision_requires_positive_lcb_and_win_rate():
    baseline = [1.0] * 16
    action = [0.999, 0.999] * 8
    result = MOD.paired_decision(
        baseline,
        action,
        panel_size=2,
        min_gain=0.00025,
        lcb_z=2.365,
        min_win_rate=0.75,
    )
    assert result["eligible"]
    assert result["valid"]
    assert result["win_rate"] == 1.0
    assert result["lcb"] > 0.0


def test_paired_decision_abstains_when_panels_disagree():
    baseline = [1.0] * 16
    action = [0.998, 0.998, 1.002, 1.002] * 4
    result = MOD.paired_decision(
        baseline,
        action,
        panel_size=2,
        min_gain=0.0,
        lcb_z=2.365,
        min_win_rate=0.75,
    )
    assert not result["eligible"]
    assert result["valid"]
    assert result["win_rate"] == 0.5


@pytest.mark.parametrize(
    ("baseline", "action"),
    [
        ([1.0, 1.0], [0.9]),
        ([1.0, float("nan")], [0.9, 0.9]),
        ([1.0, 1.0], [0.9, float("inf")]),
    ],
)
def test_paired_decision_fails_closed_on_unequal_or_nonfinite_losses(baseline, action):
    result = MOD.paired_decision(
        baseline,
        action,
        panel_size=1,
        min_gain=0.0,
        lcb_z=0.0,
        min_win_rate=0.0,
    )
    assert result["eligible"] is False
    assert result["valid"] is False
    assert result["fallback_reason"] == "invalid_paired_losses"
    assert result["panels"] == []


def test_utility_center_and_se_share_the_paired_batch_estimand():
    estimate = MOD.utility_estimate([2.0, 4.0], [1.0, 1.0])
    assert estimate["batch_gains"] == [1.0, 3.0]
    assert estimate["center"] == pytest.approx(2.0)
    assert estimate["se"] == pytest.approx(1.0)
    assert MOD.utility_se([2.0, 4.0], [1.0, 1.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="equal length"):
        MOD.utility_estimate([1.0, 2.0], [1.0])


def test_norm_matched_trial_matches_baseline_step():
    current = torch.tensor([1.0, -1.0])
    raw = torch.tensor([2.0, -1.0])
    baseline = torch.tensor([1.0, 1.0])
    trial, scale, valid = MOD.norm_matched_trial(
        current,
        raw,
        baseline,
        min_scale=0.5,
        max_scale=3.0,
    )
    assert valid
    assert scale == 2.0
    assert torch.isclose((trial - current).norm(), (baseline - current).norm())


def _conversation(text):
    return {
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"answer {text}"},
        ]
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_probe_data_provenance_records_file_and_canonical_hashes_and_disjointness(
    tmp_path,
):
    anchor_path = tmp_path / "anchor.jsonl"
    oracle_path = tmp_path / "oracle.jsonl"
    _write_jsonl(anchor_path, [_conversation("a"), _conversation("b")])
    _write_jsonl(oracle_path, [_conversation("c"), _conversation("d")])

    args = SimpleNamespace(
        anchor_data=anchor_path,
        oracle_data=oracle_path,
        max_rows=2,
    )
    metadata = MOD.build_data_provenance(args)

    assert metadata["disjointness"]["verified_disjoint"] is True
    assert metadata["disjointness"]["overlap_count"] == 0
    assert (
        metadata["anchor"]["source_sha256"]
        == hashlib.sha256(anchor_path.read_bytes()).hexdigest()
    )
    assert metadata["anchor"]["canonical_row_hashes"] == [
        MOD.canonical_row_hash(MOD._canonical_row(row, context="test"))
        for row in [_conversation("a"), _conversation("b")]
    ]

    overlap_path = tmp_path / "overlap.jsonl"
    _write_jsonl(overlap_path, [_conversation("d"), _conversation("e")])
    args.oracle_data = overlap_path
    args.anchor_data = oracle_path
    with pytest.raises(ValueError, match="overlap"):
        MOD.build_data_provenance(args)


def _capture_row(learner_id, *, step=1, fragment=0, state="states/one.ckpt"):
    return {
        "schema": MOD.CAPTURE_SCHEMA,
        "oracle_scope": "syncer_current_global_pending_offline",
        "step": step,
        "syncer_global_step": step - 1,
        "fragment": fragment,
        "current_fragment_version": max(0, step - 4),
        "learner_id": learner_id,
        "base_version": max(0, step - 4),
        "local_step": 10,
        "c_steps": 2,
        "c_tokens": 64,
        "weight": 2048.0,
        "state_checkpoint": state,
        "candidate_f32": f"candidates/{step}-{fragment}-{learner_id}.f32",
    }


def test_complete_group_validation_requires_expected_unique_ids_and_metadata():
    rows = [_capture_row(learner_id) for learner_id in range(4)]
    groups = MOD.validate_candidate_groups(rows, 4)
    assert len(groups) == 1
    assert [row["learner_id"] for row in groups[0]] == [0, 1, 2, 3]

    duplicate = copy.deepcopy(rows)
    duplicate[-1]["learner_id"] = 2
    with pytest.raises(ValueError, match="learner IDs"):
        MOD.validate_candidate_groups(duplicate, 4)

    inconsistent = copy.deepcopy(rows)
    inconsistent[-1]["syncer_global_step"] = 9
    with pytest.raises(ValueError, match="inconsistent syncer_global_step"):
        MOD.validate_candidate_groups(inconsistent, 4)

    nonpositive = copy.deepcopy(rows)
    nonpositive[-1]["weight"] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        MOD.validate_candidate_groups(nonpositive, 4)


def test_capture_payload_manifest_digests_state_and_candidate_files(tmp_path):
    (tmp_path / "states").mkdir()
    (tmp_path / "candidates").mkdir()
    (tmp_path / "states" / "one.ckpt").write_bytes(b"state")
    rows = [_capture_row(learner_id) for learner_id in range(4)]
    for row in rows:
        (tmp_path / row["candidate_f32"]).write_bytes(
            f"candidate-{row['learner_id']}".encode()
        )
    groups = MOD.validate_candidate_groups(rows, 4)

    before = MOD.capture_payload_provenance(tmp_path, groups)
    assert len(before["state_checkpoints"]) == 1
    assert len(before["candidate_tensors"]) == 4
    assert (
        before["state_checkpoints"][0]["sha256"] == hashlib.sha256(b"state").hexdigest()
    )

    (tmp_path / rows[0]["candidate_f32"]).write_bytes(b"changed")
    after = MOD.capture_payload_provenance(tmp_path, groups)
    assert after["manifest_sha256"] != before["manifest_sha256"]


def test_group_validation_rejects_incomplete_groups_instead_of_dropping_them():
    rows = [_capture_row(learner_id) for learner_id in range(3)]
    with pytest.raises(ValueError, match="3 candidates, expected 4"):
        MOD.validate_candidate_groups(rows, 4)


def test_expected_group_coverage_requires_the_predeclared_step_range():
    rows = [_capture_row(i, step=1, state="states/one.ckpt") for i in range(4)]
    rows += [_capture_row(i, step=3, state="states/three.ckpt") for i in range(4)]
    groups = MOD.validate_candidate_groups(rows, 4)
    with pytest.raises(ValueError, match=r"missing=\[2\].*extra=\[3\]"):
        MOD.validate_expected_group_coverage(groups, 2)


def test_candidate_tensor_validation_rejects_nan_and_inf():
    MOD.validate_candidate_tensor(torch.tensor([1.0, 2.0]), context="candidate")
    for tensor in (torch.tensor([float("nan")]), torch.tensor([float("inf")])):
        with pytest.raises(ValueError, match="finite"):
            MOD.validate_candidate_tensor(tensor, context="candidate")


def test_capture_order_next_checkpoint_uses_index_order_and_requires_contiguity():
    first = [_capture_row(i, step=1, state="states/one.ckpt") for i in range(4)]
    second = [
        _capture_row(i, step=2, fragment=1, state="states/two.ckpt") for i in range(4)
    ]
    mapping = MOD.capture_order_next_checkpoints(first + second)
    assert mapping[("states/one.ckpt", 1, 0)] == "states/two.ckpt"

    interleaved = first[:2] + second + first[2:]
    with pytest.raises(ValueError, match="not contiguous"):
        MOD.capture_order_next_checkpoints(interleaved)


def test_random_valid_action_is_deterministic_order_and_shard_invariant():
    actions = [
        {"name": "drop_learner_0", "dropped_learner": 0, "valid": False},
        {"name": "drop_learner_1", "dropped_learner": 1, "valid": True},
        {"name": "drop_learner_2", "dropped_learner": 2, "valid": True},
    ]
    forward = MOD.deterministic_random_valid_action(
        actions, random_seed=7, seed=223, stable_group_id="group-17"
    )
    reverse = MOD.deterministic_random_valid_action(
        list(reversed(actions)),
        random_seed=7,
        seed=223,
        stable_group_id="group-17",
    )
    assert forward["name"] == reverse["name"]
    assert forward["valid"] is True
    assert forward["dropped_learner"] in {1, 2}
    assert (
        MOD.deterministic_random_valid_action(
            [actions[0]], random_seed=7, seed=223, stable_group_id="group-17"
        )
        is None
    )


def test_next_state_parity_exposes_step_relative_error():
    current = torch.tensor([0.0, 0.0])
    baseline = torch.tensor([1.0, 0.0])
    captured = torch.tensor([1.000001, 0.0])
    parity = MOD.next_state_parity(current, baseline, captured)
    assert parity["absolute_error"] == pytest.approx(1e-6, rel=0.1)
    assert parity["step_relative_error"] == pytest.approx(1e-6, rel=0.1)


def test_resume_validation_rejects_duplicates_wrong_config_and_partial_json(tmp_path):
    base = {
        "schema": MOD.REPLAY_SCHEMA,
        "seed": 223,
        "group_id": "g0",
        "compatibility_config_sha256": "compat",
        "capture_config_sha256": "capture",
        "replay_config_sha256": "replay",
    }
    seen = MOD._validate_resume_records(
        [base],
        seed=223,
        allowed_group_ids={"g0", "g1"},
        compatibility_config_sha256="compat",
        capture_config_sha256="capture",
        replay_config_sha256="replay",
    )
    assert seen == {"g0"}
    with pytest.raises(ValueError, match="duplicate"):
        MOD._validate_resume_records(
            [base, dict(base)],
            seed=223,
            allowed_group_ids={"g0"},
            compatibility_config_sha256="compat",
            capture_config_sha256="capture",
            replay_config_sha256="replay",
        )
    wrong = dict(base, replay_config_sha256="other")
    with pytest.raises(ValueError, match="different replay_config_sha256"):
        MOD._validate_resume_records(
            [wrong],
            seed=223,
            allowed_group_ids={"g0"},
            compatibility_config_sha256="compat",
            capture_config_sha256="capture",
            replay_config_sha256="replay",
        )

    partial = tmp_path / "partial.jsonl"
    partial.write_text(json.dumps(base) + "\n{" + "\n")
    with pytest.raises(ValueError, match="malformed JSON"):
        MOD.read_replay_artifact(partial)


def _summary_record(seed, group_id, index, digests):
    acted = index == 0
    gain = 0.004 if acted else 0.0
    return {
        "schema": MOD.REPLAY_SCHEMA,
        "seed": seed,
        "group_id": group_id,
        **digests,
        "chosen_gain_vs_baseline": gain,
        "chosen_action": "drop_learner_1" if acted else MOD.BASELINE_ACTION,
        "best_loo_oracle_gain": gain,
        "random_loo_oracle_gain": 0.0,
        "baseline_oracle_negative": True,
        "chosen_oracle_negative": not acted,
        "baseline_oracle_strict_negative": True,
        "chosen_oracle_strict_negative": not acted,
        "production_baseline_next_state_available": index < 3,
        "production_baseline_next_state_step_relative_error": (
            1e-6 if index < 3 else None
        ),
    }


def _write_completed_shard(
    path,
    *,
    seed,
    start,
    stride,
    capture_ids,
    coordinates,
    compatibility_config,
    full_shard_complete=True,
):
    compatibility_digest = MOD._config_sha256(compatibility_config)
    capture_config = {
        "schema": MOD.CONFIG_SCHEMA,
        "compatibility_config_sha256": compatibility_digest,
        "seed": seed,
        "capture_group_ids": capture_ids,
        "capture_group_coordinates": coordinates,
    }
    capture_digest = MOD._config_sha256(capture_config)
    expected_ids = capture_ids[start::stride]
    replay_config = {
        "schema": MOD.CONFIG_SCHEMA,
        "compatibility_config_sha256": compatibility_digest,
        "capture_config_sha256": capture_digest,
        "group_shard": {"start": start, "stride": stride, "max_groups": None},
        "expected_group_ids": expected_ids,
    }
    replay_digest = MOD._config_sha256(replay_config)
    digests = {
        "compatibility_config_sha256": compatibility_digest,
        "capture_config_sha256": capture_digest,
        "replay_config_sha256": replay_digest,
    }
    record_by_id = {
        group_id: _summary_record(seed, group_id, index, digests)
        for index, group_id in enumerate(capture_ids)
    }
    records = [record_by_id[group_id] for group_id in expected_ids]
    completion = {
        "schema": MOD.COMPLETION_SCHEMA,
        "replay_schema": MOD.REPLAY_SCHEMA,
        "complete": True,
        "full_shard_complete": full_shard_complete,
        "seed": seed,
        "compatibility_config": compatibility_config,
        "compatibility_config_sha256": compatibility_digest,
        "capture_config": capture_config,
        "capture_config_sha256": capture_digest,
        "replay_config": replay_config,
        "replay_config_sha256": replay_digest,
        "group_shard": {"start": start, "stride": stride, "max_groups": None},
        "capture_group_count": len(capture_ids),
        "capture_group_ids": capture_ids,
        "capture_group_coordinates": coordinates,
        "full_shard_group_count": len(expected_ids),
        "full_shard_group_ids": expected_ids,
        "expected_group_count": len(expected_ids),
        "expected_group_ids": expected_ids,
        "expected_group_ids_sha256": MOD._ordered_id_digest(expected_ids),
        "completed_group_count": len(expected_ids),
        "completed_group_ids": expected_ids,
        "record_count": len(records),
    }
    _write_jsonl(path, records + [completion])
    return path


def _completed_artifacts(tmp_path):
    compatibility = {
        "schema": MOD.CONFIG_SCHEMA,
        "policy": "exact-loo",
        "anchor_sha256": "anchor",
        "oracle_sha256": "oracle",
    }
    coordinates = [{"step": index + 1, "fragment": index % 4} for index in range(4)]
    paths = []
    for seed in (223, 239):
        capture_ids = [f"seed-{seed}-group-{index}" for index in range(4)]
        for start in (0, 1):
            paths.append(
                _write_completed_shard(
                    tmp_path / f"seed-{seed}-shard-{start}.jsonl",
                    seed=seed,
                    start=start,
                    stride=2,
                    capture_ids=capture_ids,
                    coordinates=coordinates,
                    compatibility_config=compatibility,
                )
            )
    return paths, compatibility, coordinates


def test_aggregate_accepts_only_complete_compatible_full_coverage_and_runs_gate(
    tmp_path,
):
    paths, _, _ = _completed_artifacts(tmp_path)
    result = AGG.aggregate_completed_artifacts(
        paths,
        expected_seeds=[223, 239],
        expected_groups_per_seed=4,
    )

    assert result["schema"] == AGG.AGGREGATE_SCHEMA
    assert result["records"] == 8
    assert result["mean_gain_vs_baseline"] == pytest.approx(0.001)
    assert result["action_rate"] == pytest.approx(0.25)
    assert result["negative_rate_relative_drop"] == pytest.approx(0.25)
    assert result["strict_negative_rate_relative_drop"] == pytest.approx(0.25)
    assert result["next_state_validation"]["max_step_relative_error"] == pytest.approx(
        1e-6
    )
    assert result["offline_gate"]["gate_pass"] is True
    assert all(result["offline_gate"]["checks"].values())


def test_aggregate_rejects_missing_duplicate_incomplete_and_incompatible_shards(
    tmp_path,
):
    paths, compatibility, coordinates = _completed_artifacts(tmp_path)
    with pytest.raises(ValueError, match="missing starts"):
        AGG.aggregate_completed_artifacts(
            paths[:-1],
            expected_seeds=[223, 239],
            expected_groups_per_seed=4,
        )

    duplicate = _write_completed_shard(
        tmp_path / "duplicate-start.jsonl",
        seed=223,
        start=0,
        stride=2,
        capture_ids=[f"seed-223-group-{index}" for index in range(4)],
        coordinates=coordinates,
        compatibility_config=compatibility,
    )
    with pytest.raises(ValueError, match="duplicate shard starts"):
        AGG.aggregate_completed_artifacts(
            paths + [duplicate],
            expected_seeds=[223, 239],
            expected_groups_per_seed=4,
        )

    incomplete = tmp_path / "incomplete.jsonl"
    raw_rows = MOD.read_jsonl(paths[0])
    _write_jsonl(incomplete, raw_rows[:-1])
    with pytest.raises(ValueError, match="missing terminal completion"):
        AGG.aggregate_completed_artifacts(
            [incomplete] + paths[1:],
            expected_seeds=[223, 239],
            expected_groups_per_seed=4,
        )

    incompatible = _write_completed_shard(
        tmp_path / "incompatible.jsonl",
        seed=239,
        start=1,
        stride=2,
        capture_ids=[f"seed-239-group-{index}" for index in range(4)],
        coordinates=coordinates,
        compatibility_config={**compatibility, "policy": "different"},
    )
    with pytest.raises(ValueError, match="compatibility"):
        AGG.aggregate_completed_artifacts(
            paths[:-1] + [incompatible],
            expected_seeds=[223, 239],
            expected_groups_per_seed=4,
        )


def test_aggregate_rejects_max_groups_completion_marker(tmp_path):
    compatibility = {"schema": MOD.CONFIG_SCHEMA, "policy": "exact-loo"}
    path = _write_completed_shard(
        tmp_path / "partial-shard.jsonl",
        seed=223,
        start=0,
        stride=1,
        capture_ids=["g0"],
        coordinates=[{"step": 1, "fragment": 0}],
        compatibility_config=compatibility,
        full_shard_complete=False,
    )
    with pytest.raises(ValueError, match="not its full shard"):
        AGG.aggregate_completed_artifacts(
            [path], expected_seeds=[223], expected_groups_per_seed=1
        )
