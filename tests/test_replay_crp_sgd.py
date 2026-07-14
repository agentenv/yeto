import hashlib
import importlib.util
import json
import math
import random
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "replay_crp_sgd", ROOT / "scripts" / "replay_crp_sgd.py"
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def vector(values, *, expected_sha256=None):
    raw = np.asarray(values, dtype="<f4").tobytes()
    return MOD.ExactVector.from_raw(raw, expected_sha256=expected_sha256)


def event(sequence, stock, residual, *, fragment=0):
    stock_array = np.asarray(stock, dtype=np.float32)
    proposal = stock_array + np.asarray(residual, dtype=np.float32)
    return MOD.ExactEvent(
        f"e{sequence}", sequence, fragment, vector(stock_array), vector(proposal)
    )


def correction_norm(action, stock):
    return MOD._norm64(
        action.values.astype(np.float64) - np.asarray(stock, dtype=np.float64)
    )


def test_fallback_preserves_exact_stock_byte_object_for_random_finite_inputs():
    rng = random.Random(98431)
    engine = MOD.CRPEngine()
    for sequence in range(32):
        values = [rng.uniform(-3.0, 3.0) for _ in range(11)]
        stock = vector(values)
        action = engine.process(
            MOD.ExactEvent(f"random-{sequence}", sequence, sequence, stock, None)
        )
        assert action.fallback
        assert action.raw is stock.raw
        assert action.bit_identical_to_stock
        assert hashlib.sha256(action.raw).digest() == hashlib.sha256(stock.raw).digest()


def test_invalid_stock_fallback_preserves_nan_payload_and_negative_zero_bits():
    raw = struct.pack("<III", 0x7FC12345, 0x80000000, 0x3F800000)
    stock = MOD.ExactVector.from_raw(raw)
    proposal = vector([1.0, 2.0, 3.0])

    action = MOD.CRPEngine().process(MOD.ExactEvent("bad", 0, 0, stock, proposal))

    assert not stock.valid
    assert action.raw is raw
    assert action.raw == raw
    assert struct.unpack("<III", action.raw) == (0x7FC12345, 0x80000000, 0x3F800000)


def test_residual_is_resolved_at_t_plus_one_and_cannot_pulse_until_t_plus_two():
    engine = MOD.CRPEngine()
    stocks = ([1.0, 0.0], [1.0, 0.1], [1.0, 0.2], [1.0, 0.3])
    actions = [
        engine.process(event(index, stock, [0.0, 0.04]))
        for index, stock in enumerate(stocks)
    ]

    assert [action.fallback for action in actions] == [True, True, True, False]
    assert actions[1].admitted_event_id == "e0"
    assert actions[2].admitted_event_id == "e1"
    assert actions[3].contributor_count == 2
    assert actions[3].admitted_event_id == "e2"
    assert MOD.CRP_PULSE_MIN_RATIO <= actions[3].pulse_ratio <= MOD.CRP_PULSE_MAX_RATIO


def test_nonpositive_resolution_clears_bank_before_it_can_pulse():
    engine = MOD.CRPEngine()
    assert engine.process(event(0, [1.0, 0.0], [0.0, 0.04])).fallback
    assert engine.process(event(1, [1.0, 0.1], [0.0, 0.04])).fallback
    assert engine.process(event(2, [1.0, 0.2], [0.0, -0.04])).fallback

    action = engine.process(event(3, [1.0, 0.3], [0.0, 0.04]))

    assert action.fallback
    assert action.reason == "resolution_rejected_bank_cleared"
    assert action.resolution_score is not None and action.resolution_score < 0.0
    assert not engine._states[0].bank


def test_individual_ratio_gate_is_strict_at_one_over_twenty():
    engine = MOD.CRPEngine()
    first = event(0, [1.0, 0.0], [0.0, MOD.CRP_INDIVIDUAL_MAX_RATIO])
    engine.process(first)

    action = engine.process(event(1, [1.0, 0.2], [0.0, 0.01]))

    assert action.reason == "resolution_rejected_bank_cleared"
    assert action.resolution_score is not None and action.resolution_score > 0.0


def test_pulse_is_never_scaled_up_and_is_clipped_down_at_maximum():
    stock = vector([1.0, 0.0])
    proposal = vector([1.0, 0.0])

    no_upscale_engine = MOD.CRPEngine()
    state = no_upscale_engine._states[0]
    state.ordinal = 3
    state.bank = [
        MOD._BankEntry("a", 0, 1, np.array([0.0, 0.03]), 0.1),
        MOD._BankEntry("b", 1, 2, np.array([0.0, 0.03]), 0.1),
    ]
    action = no_upscale_engine.process(MOD.ExactEvent("c", 3, 0, stock, proposal))
    assert not action.fallback
    assert math.isclose(action.pulse_ratio, 0.06, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(correction_norm(action, [1.0, 0.0]), 0.06, abs_tol=1e-7)

    clipped_engine = MOD.CRPEngine()
    state = clipped_engine._states[0]
    state.ordinal = 3
    state.bank = [
        MOD._BankEntry("a", 0, 1, np.array([0.0, 0.09]), 0.1),
        MOD._BankEntry("b", 1, 2, np.array([0.0, 0.09]), 0.1),
    ]
    clipped = clipped_engine.process(MOD.ExactEvent("c", 3, 0, stock, proposal))
    assert not clipped.fallback
    assert clipped.pulse_ratio == MOD.CRP_PULSE_MAX_RATIO
    assert math.isclose(
        correction_norm(clipped, [1.0, 0.0]), MOD.CRP_PULSE_MAX_RATIO, abs_tol=1e-7
    )


def test_bank_age_is_source_based_and_inclusive_at_eight():
    engine = MOD.CRPEngine()
    state = engine._states[0]
    state.ordinal = 8
    state.bank = [
        MOD._BankEntry("keep", 0, 1, np.array([0.0, 0.03]), 0.1),
        MOD._BankEntry("expire", -1, 0, np.array([0.0, 0.03]), 0.1),
    ]
    action = engine.process(
        MOD.ExactEvent("now", 9, 0, vector([1.0, 0.0]), vector([1.0, 0.0]))
    )

    assert action.fallback
    assert action.expired_count == 1
    assert [entry.source_event_id for entry in state.bank] == ["keep"]


def _write_exact_tape(root: Path):
    rows = []
    for sequence, (stock, residual) in enumerate(
        [
            ([1.0, 0.0], [0.0, 0.04]),
            ([1.0, 0.1], [0.0, 0.04]),
            ([1.0, 0.2], [0.0, 0.04]),
            ([1.0, 0.3], [0.0, 0.04]),
        ]
    ):
        stock_array = np.asarray(stock, dtype="<f4")
        proposal_array = stock_array + np.asarray(residual, dtype="<f4")
        stock_path = root / f"stock-{sequence}.f32"
        proposal_path = root / f"proposal-{sequence}.f32"
        stock_path.write_bytes(stock_array.tobytes())
        proposal_path.write_bytes(proposal_array.tobytes())
        rows.append(
            {
                "schema": MOD.EXACT_SCHEMA,
                "event_id": f"e{sequence}",
                "sequence": sequence,
                "fragment": 0,
                "numel": 2,
                "stock_f32le": stock_path.name,
                "stock_f32le_sha256": MOD.sha256_file(stock_path),
                "proposal_f32le": proposal_path.name,
                "proposal_f32le_sha256": MOD.sha256_file(proposal_path),
            }
        )
    index = root / "index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return index


def _read_tape_rows(index):
    return [json.loads(line) for line in index.read_text().splitlines()]


def _write_tape_rows(index, rows):
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_exact_replay_is_deterministic_and_reports_bit_identity(tmp_path):
    index = _write_exact_tape(tmp_path)

    first = MOD.replay_exact_crp(index)
    second = MOD.replay_exact_crp(index)

    assert MOD.canonical_json(first) == MOD.canonical_json(second)
    assert first["summary"]["pulses"] == 1
    assert all(
        action["bit_identical_stock_fallback"]
        for action in first["actions"]
        if action["reason"] != "pulse"
    )


def test_exact_replay_treats_hash_mismatch_as_corrupt_fallback(tmp_path):
    index = _write_exact_tape(tmp_path)
    rows = _read_tape_rows(index)
    rows[0]["proposal_f32le_sha256"] = "0" * 64
    _write_tape_rows(index, rows)

    result = MOD.replay_exact_crp(index)

    assert result["actions"][0]["reason"].startswith("invalid_proposal:sha256_mismatch")
    assert result["actions"][0]["bit_identical_stock_fallback"]
    assert result["summary"]["input_fallback_classes"] == {
        "continuity": 0,
        "nonidentical_fallback": 0,
        "proposal_integrity": 1,
        "shape_integrity": 0,
        "stock_integrity": 0,
    }
    assert result["summary"]["all_fallbacks_bit_identical"]


def test_stock_hash_mismatch_is_separately_accounted_exact_fallback(tmp_path):
    index = _write_exact_tape(tmp_path)
    rows = _read_tape_rows(index)
    rows[0]["stock_f32le_sha256"] = "0" * 64
    _write_tape_rows(index, rows)

    result = MOD.replay_exact_crp(index)

    assert result["actions"][0]["reason"].startswith("invalid_stock:sha256_mismatch")
    assert result["actions"][0]["bit_identical_stock_fallback"]
    assert result["summary"]["input_fallback_classes"]["stock_integrity"] == 1
    assert result["summary"]["all_fallbacks_bit_identical"]


@pytest.mark.parametrize("bad_sequence", [True, "0", 0.0])
def test_exact_index_does_not_coerce_bool_string_or_float_to_integer(
    tmp_path, bad_sequence
):
    index = _write_exact_tape(tmp_path)
    rows = _read_tape_rows(index)
    rows[0]["sequence"] = bad_sequence
    _write_tape_rows(index, rows)

    with pytest.raises(ValueError, match="sequence must be int"):
        MOD.replay_exact_crp(index)


def test_exact_index_rejects_nonstandard_json_constants(tmp_path):
    index = _write_exact_tape(tmp_path)
    lines = index.read_text().splitlines()
    lines[0] = lines[0].replace('"sequence": 0', '"sequence": NaN')
    index.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="non-standard JSON constant NaN"):
        MOD.replay_exact_crp(index)


def test_exact_index_rejects_duplicate_and_unexpected_json_fields(tmp_path):
    index = _write_exact_tape(tmp_path)
    lines = index.read_text().splitlines()
    lines[0] = lines[0][:-1] + ', "sequence": 0}'
    index.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="duplicate JSON field 'sequence'"):
        MOD.replay_exact_crp(index)

    index = _write_exact_tape(tmp_path)
    rows = _read_tape_rows(index)
    rows[0]["ignored"] = "forbidden"
    _write_tape_rows(index, rows)
    with pytest.raises(ValueError, match="unexpected=.*ignored"):
        MOD.replay_exact_crp(index)


@pytest.mark.parametrize(
    "bad_digest",
    ["A" * 64, "a" * 63, "g" * 64, True],
)
def test_exact_index_requires_canonical_lowercase_sha256(tmp_path, bad_digest):
    index = _write_exact_tape(tmp_path)
    rows = _read_tape_rows(index)
    rows[0]["stock_f32le_sha256"] = bad_digest
    _write_tape_rows(index, rows)

    with pytest.raises(ValueError, match="lowercase hexadecimal|must be str"):
        MOD.replay_exact_crp(index)


def test_exact_index_rejects_absolute_parent_and_symlink_vector_paths(tmp_path):
    index = _write_exact_tape(tmp_path)
    rows = _read_tape_rows(index)
    rows[0]["stock_f32le"] = str((tmp_path / "stock-0.f32").resolve())
    _write_tape_rows(index, rows)
    with pytest.raises(ValueError, match="relative and root-contained"):
        MOD.replay_exact_crp(index)

    index = _write_exact_tape(tmp_path)
    rows = _read_tape_rows(index)
    rows[0]["stock_f32le"] = "../stock-0.f32"
    _write_tape_rows(index, rows)
    with pytest.raises(ValueError, match="relative and root-contained"):
        MOD.replay_exact_crp(index)

    index = _write_exact_tape(tmp_path)
    link = tmp_path / "linked-stock.f32"
    link.symlink_to(tmp_path / "stock-0.f32")
    rows = _read_tape_rows(index)
    rows[0]["stock_f32le"] = link.name
    _write_tape_rows(index, rows)
    with pytest.raises(ValueError, match="symlink path component is forbidden"):
        MOD.replay_exact_crp(index)


def test_sequence_gap_forces_accounted_global_bank_clear_and_stock_fallback(tmp_path):
    index = _write_exact_tape(tmp_path)
    rows = _read_tape_rows(index)
    rows[-1]["sequence"] = 4
    _write_tape_rows(index, rows)

    result = MOD.replay_exact_crp(index)

    final = result["actions"][-1]
    assert final["reason"].startswith("continuity_gap_bank_cleared")
    assert final["bit_identical_stock_fallback"]
    assert final["input_continuity_error"] == "expected_sequence_3_observed_4"
    assert result["summary"]["pulses"] == 0
    assert result["summary"]["input_fallback_classes"]["continuity"] == 1
    assert result["summary"]["all_fallbacks_bit_identical"]


def test_fragment_numel_change_clears_state_and_falls_back(tmp_path):
    index = _write_exact_tape(tmp_path)
    rows = _read_tape_rows(index)
    stock_path = tmp_path / "stock-1-wide.f32"
    proposal_path = tmp_path / "proposal-1-wide.f32"
    stock_path.write_bytes(np.asarray([1.0, 0.1, 0.0], dtype="<f4").tobytes())
    proposal_path.write_bytes(np.asarray([1.0, 0.14, 0.0], dtype="<f4").tobytes())
    rows[1].update(
        {
            "numel": 3,
            "stock_f32le": stock_path.name,
            "stock_f32le_sha256": MOD.sha256_file(stock_path),
            "proposal_f32le": proposal_path.name,
            "proposal_f32le_sha256": MOD.sha256_file(proposal_path),
        }
    )
    _write_tape_rows(index, rows)

    result = MOD.replay_exact_crp(index)

    assert result["actions"][1]["reason"] == "fragment_numel_changed_bank_cleared"
    assert result["actions"][1]["bit_identical_stock_fallback"]
    assert result["summary"]["input_fallback_classes"]["shape_integrity"] == 2


def test_cli_publishes_atomic_report_and_lowercase_checksum_completion_marker(tmp_path):
    tape = tmp_path / "tape"
    tape.mkdir()
    index = _write_exact_tape(tape)
    output = tmp_path / "out" / "result.json"

    assert MOD.main(["--exact-crp-index", str(index), "--out", str(output)]) == 0
    first = output.read_bytes()
    checksum_path = Path(f"{output}.sha256")
    expected = hashlib.sha256(first).hexdigest()
    assert checksum_path.read_text() == f"{expected}  {output.name}\n"
    assert MOD.LOWER_SHA256_RE.fullmatch(expected)
    assert not list(output.parent.glob(f".{output.name}.tmp.*"))
    assert not list(output.parent.glob(f".{checksum_path.name}.tmp.*"))

    assert MOD.main(["--exact-crp-index", str(index), "--out", str(output)]) == 0
    assert output.read_bytes() == first
    assert checksum_path.read_text() == f"{expected}  {output.name}\n"


def test_hardening_does_not_change_frozen_policy_digest():
    digest = hashlib.sha256(
        MOD.canonical_json(MOD.policy_contract()).encode()
    ).hexdigest()
    assert digest == "7b8fb30ca98f4b0916f4158824c98246799c61c08d416bfb6bb37d5b2e022710"


def test_bcmp_scalar_tape_is_explicitly_unidentifiable(tmp_path):
    path = tmp_path / "bcmp.jsonl"
    event_id = "l0-f0"
    path.write_text(
        json.dumps(
            {
                "schema": "bcmp_shadow_v1",
                "event_id": event_id,
                "stock_total_step_l2": 1.0,
            }
        )
        + "\n"
        + json.dumps(
            {
                "schema": "bcmp_shadow_resolution_v1",
                "event_id": event_id,
                "candidate": "ray",
                "direction_l2": 0.04,
                "future_gradient_dot": 0.2,
            }
        )
        + "\n"
    )

    audit = MOD.audit_bcmp_scalar([path])

    assert audit["decision"] == "UNIDENTIFIABLE"
    assert not audit["identifiable"]
    assert audit["scalar_coverage"]["by_candidate"]["ray"]["both_proxy_conditions"] == 1
    assert "stock direction vector G_t bytes" in audit["missing_capabilities"]


def _write_checkpoint(path, global_step, version, params):
    params = np.asarray(params, dtype="<f4")
    raw = bytearray()
    raw.extend(struct.pack("<IQI", MOD.CKPT_MAGIC, global_step, 1))
    raw.extend(struct.pack("<QQ", version, params.size))
    raw.extend(params.tobytes())
    raw.extend(np.zeros_like(params).tobytes())
    raw.extend(struct.pack("<I", 0))
    path.write_bytes(raw)


def _write_pti_capture(root: Path):
    (root / "states").mkdir(parents=True)
    directions = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.1], dtype=np.float32),
        np.array([1.0, 0.2], dtype=np.float32),
        np.array([1.0, 0.3], dtype=np.float32),
        np.array([1.0, 0.4], dtype=np.float32),
    ]
    params = np.zeros(2, dtype=np.float32)
    realized = []
    rows = []
    for step in range(1, len(directions) + 2):
        path = root / "states" / f"state_before_step_{step:08}.ckpt"
        _write_checkpoint(path, step - 1, step - 1, params)
        rows.append(
            {
                "schema": "syncer_probe_capture_v1",
                "step": step,
                "fragment": 0,
                "current_fragment_version": step - 1,
                "state_checkpoint": f"states/{path.name}",
            }
        )
        if step <= len(directions):
            following = np.add(params, directions[step - 1], dtype=np.float32)
            realized.append(np.subtract(following, params, dtype=np.float32))
            params = following
    (root / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return realized


def test_pti_screen_materializes_exact_factual_directions_without_policy_invention(
    tmp_path,
):
    capture = tmp_path / "capture"
    expected = _write_pti_capture(capture)

    streams, provenance = MOD.materialize_factual_directions(capture)
    result = MOD.screen_pti_captures([capture])

    assert len(streams[0]) == len(expected)
    for actual, wanted in zip(streams[0], expected):
        assert actual.dtype == np.float32
        assert actual.tobytes() == wanted.tobytes()
    assert provenance["derived_direction_chain_sha256"]
    assert provenance["source_capture_id"] == "capture"
    assert provenance["source_seed"] is None
    assert result["decision"] == "DIRECTION_SCREEN_ONLY"
    assert not result["causal_loss_claim"]
    assert result["valid_scores_by_fragment"] == {"0": 3}
    assert result["coefficient_results"]["0"]["mean_cosine_gain"] == 0.0
    assert result["coefficient_results"]["-0.25"]["post_warmup_opportunities"] == 0
    assert result["coefficient_results"]["-0.25"]["interlock_eligible_fraction"] is None
    assert (
        result["input"][0]["coefficient_results"]["-0.25"]
        == result["coefficient_results"]["-0.25"]
    )
    assert any("tie-break" in item for item in result["limitations"])


def test_pti_capture_rejects_coerced_unknown_duplicate_and_symlinked_inputs(tmp_path):
    capture = tmp_path / "bool"
    _write_pti_capture(capture)
    rows = _read_tape_rows(capture / "index.jsonl")
    rows[0]["step"] = True
    _write_tape_rows(capture / "index.jsonl", rows)
    with pytest.raises(ValueError, match="step must be int"):
        MOD.materialize_factual_directions(capture)

    capture = tmp_path / "unknown"
    _write_pti_capture(capture)
    rows = _read_tape_rows(capture / "index.jsonl")
    rows[0]["unknown"] = 1
    _write_tape_rows(capture / "index.jsonl", rows)
    with pytest.raises(ValueError, match="unexpected fields.*unknown"):
        MOD.materialize_factual_directions(capture)

    capture = tmp_path / "duplicate"
    _write_pti_capture(capture)
    index = capture / "index.jsonl"
    lines = index.read_text().splitlines()
    lines[0] = lines[0][:-1] + ', "step": 1}'
    index.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="duplicate JSON field 'step'"):
        MOD.materialize_factual_directions(capture)

    capture = tmp_path / "symlink"
    _write_pti_capture(capture)
    index = capture / "index.jsonl"
    rows = _read_tape_rows(index)
    target = capture / rows[0]["state_checkpoint"]
    link = capture / "states" / "linked.ckpt"
    link.symlink_to(target)
    rows[0]["state_checkpoint"] = "states/linked.ckpt"
    _write_tape_rows(index, rows)
    with pytest.raises(ValueError, match="symlink path component is forbidden"):
        MOD.materialize_factual_directions(capture)


def test_pti_analytic_scores_match_direct_normalized_vector_construction():
    stream = [
        np.array([0.8, -0.2, 0.1], dtype=np.float32),
        np.array([0.7, 0.3, -0.4], dtype=np.float32),
        np.array([0.5, 0.6, 0.2], dtype=np.float32),
    ]

    outcomes, _ = MOD._pti_scores(stream)

    assert len(outcomes) == 1
    previous, current, following = [
        value.astype(np.float64) / MOD._norm64(value.astype(np.float64))
        for value in stream
    ]
    transverse = previous - MOD._dot64(previous, current) * current
    transverse /= MOD._norm64(transverse)
    stock_cosine = MOD._dot64(current, following)
    for coefficient in MOD.PTI_COEFFICIENTS:
        candidate = current + coefficient * transverse
        candidate /= MOD._norm64(candidate)
        direct = MOD._dot64(candidate, following) - stock_cosine
        assert math.isclose(
            outcomes[0]["scores"][coefficient], direct, rel_tol=0.0, abs_tol=2e-16
        )


def test_mstp_audit_never_converts_missing_state_to_zero_action():
    audit = MOD.mstp_audit()

    assert audit["decision"] == "UNIDENTIFIABLE"
    assert not audit["identifiable"]
    assert any("H/2" in item for item in audit["missing_capabilities"])
