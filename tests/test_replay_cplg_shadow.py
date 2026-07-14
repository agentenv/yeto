"""Fail-closed tests for the CPLG full-vector stock-tape evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

from yeto.cplg_sgd import encode_f32le


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "replay_cplg_shadow",
    ROOT / "scripts" / "replay_cplg_shadow.py",
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def _direction(degrees: float, *, norm: float = 1.0) -> bytes:
    radians = math.radians(degrees)
    return encode_f32le((norm * math.cos(radians), norm * math.sin(radians)))


def _write_tape(root: Path, *, angles: tuple[float, ...] = (0, 10, 20, 30, 40)):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset, angle in enumerate(angles, 1):
        raw = _direction(angle, norm=1.0 + offset)
        vector_path = root / f"stock-{offset}.f32le"
        vector_path.write_bytes(raw)
        rows.append(
            {
                "step": offset,
                "commit_seq": offset,
                "fragment": 0,
                "numel": 2,
                "stock_f32le": vector_path.name,
                "stock_f32le_sha256": hashlib.sha256(raw).hexdigest(),
                # Real syncer tapes contain unrelated fields; they do not
                # weaken the mandatory exact-vector contract.
                "responders": [],
            }
        )
    tape = root / "tape.jsonl"
    tape.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return tape, rows


def _rewrite(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_full_vector_stock_tape_builds_and_resolves_causal_shadows(tmp_path) -> None:
    tape, _rows = _write_tape(tmp_path / "tape")

    first = MOD.evaluate_full_vector_stock_tape(tape)
    second = MOD.evaluate_full_vector_stock_tape(tape)

    assert first == second
    assert first["decision"] == "DIRECTION_SHADOW_ONLY"
    assert first["identifiable"] is True
    assert first["full_vectors_verified"] is True
    assert first["causal_finite_loss_claim"] is False
    assert first["beats_sgd_0_28_claim"] is False
    assert first["summary"]["records"] == 5
    assert first["summary"]["constructed_nonstock_shadows"] == 3
    assert first["summary"]["resolved_shadows"] == 2
    assert first["summary"]["positive_resolved_shadows"] == 2
    assert first["summary"]["unresolved_tail_shadows"] == 1
    assert [record["reason"] for record in first["records"][:2]] == [
        "insufficient_same_fragment_history",
        "insufficient_same_fragment_history",
    ]
    assert first["records"][3]["resolved_source_commit_seq"] == 3
    assert first["records"][3]["resolved_shadow_cosine_gain"] > 0.0


def test_report_makes_sign_angle_and_cross_runtime_limits_unambiguous(tmp_path) -> None:
    tape, _rows = _write_tape(tmp_path / "tape")

    report = MOD.evaluate_full_vector_stock_tape(tape)
    contract = report["reference_contract"]

    assert contract["forward_tangent_sign"] == "rho_times_current_minus_previous"
    assert contract["backward_tangent_sign_not_used"] == (
        "previous_minus_rho_times_current"
    )
    assert contract["command_is_angle_radians_not_tangent_ratio"] is True
    assert contract["angle_cap_f32_bits"] == "0x3e7adbb0"
    assert contract["platform_cap_matches_pinned"] is True
    assert contract["cross_runtime_bit_parity"] == (
        "UNSATISFIED_UNTIL_RUST_FIXTURE_MATCH"
    )
    assert "host C libm" in contract["trig_portability_limitation"]


def test_scalar_only_stock_tape_is_rejected_as_missing_full_vectors(tmp_path) -> None:
    tape = tmp_path / "stock-tape.jsonl"
    tape.write_text(
        json.dumps(
            {
                "commit_seq": 1,
                "fragment": 0,
                "step": 1,
                "pti_stock_sha256": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="full stock vectors are required") as error:
        MOD.evaluate_full_vector_stock_tape(tape)

    message = str(error.value)
    assert "stock_f32le" in message
    assert "stock_f32le_sha256" in message
    assert "numel" in message


def test_missing_full_vectors_never_publishes_a_report(tmp_path) -> None:
    tape = tmp_path / "scalar.jsonl"
    tape.write_text('{"commit_seq":1,"fragment":0}\n', encoding="utf-8")
    output = tmp_path / "result.json"

    with pytest.raises(ValueError, match="full stock vectors are required"):
        MOD.main(["--stock-tape", str(tape), "--out", str(output)])

    assert not output.exists()
    assert not Path(f"{output}.sha256").exists()


def test_full_vector_hash_and_shape_mismatches_are_hard_errors(tmp_path) -> None:
    tape, rows = _write_tape(tmp_path / "hash")
    rows[0]["stock_f32le_sha256"] = "0" * 64
    _rewrite(tape, rows)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MOD.evaluate_full_vector_stock_tape(tape)

    tape, rows = _write_tape(tmp_path / "shape")
    rows[0]["numel"] = 3
    _rewrite(tape, rows)
    with pytest.raises(ValueError, match="requires 12 bytes"):
        MOD.evaluate_full_vector_stock_tape(tape)


@pytest.mark.parametrize("bad_commit_seq", [True, "1", 1.0])
def test_commit_sequence_is_not_coerced(tmp_path, bad_commit_seq) -> None:
    tape, rows = _write_tape(tmp_path / str(type(bad_commit_seq).__name__))
    rows[0]["commit_seq"] = bad_commit_seq
    _rewrite(tape, rows)

    with pytest.raises(ValueError, match="commit_seq must be an exact JSON integer"):
        MOD.evaluate_full_vector_stock_tape(tape)


def test_commit_sequence_gap_and_duplicate_json_are_rejected(tmp_path) -> None:
    tape, rows = _write_tape(tmp_path / "gap")
    rows[1]["commit_seq"] = 3
    _rewrite(tape, rows)
    with pytest.raises(ValueError, match="expected 2, observed 3"):
        MOD.evaluate_full_vector_stock_tape(tape)

    tape, _rows = _write_tape(tmp_path / "duplicate")
    lines = tape.read_text().splitlines()
    lines[0] = lines[0][:-1] + ', "fragment": 0}'
    tape.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON field 'fragment'"):
        MOD.evaluate_full_vector_stock_tape(tape)


def test_absolute_parent_and_symlink_vector_paths_are_rejected(tmp_path) -> None:
    tape, rows = _write_tape(tmp_path / "absolute")
    rows[0]["stock_f32le"] = str((tape.parent / "stock-1.f32le").resolve())
    _rewrite(tape, rows)
    with pytest.raises(ValueError, match="relative and root-contained"):
        MOD.evaluate_full_vector_stock_tape(tape)

    tape, rows = _write_tape(tmp_path / "parent")
    rows[0]["stock_f32le"] = "../stock-1.f32le"
    _rewrite(tape, rows)
    with pytest.raises(ValueError, match="relative and root-contained"):
        MOD.evaluate_full_vector_stock_tape(tape)

    tape, rows = _write_tape(tmp_path / "symlink")
    link = tape.parent / "linked.f32le"
    link.symlink_to(tape.parent / "stock-1.f32le")
    rows[0]["stock_f32le"] = link.name
    _rewrite(tape, rows)
    with pytest.raises(ValueError, match="symlink path component is forbidden"):
        MOD.evaluate_full_vector_stock_tape(tape)


def test_cli_atomically_publishes_checksummed_deterministic_report(tmp_path) -> None:
    tape, _rows = _write_tape(tmp_path / "tape")
    output = tmp_path / "report" / "cplg.json"

    assert MOD.main(["--stock-tape", str(tape), "--out", str(output)]) == 0
    first = output.read_bytes()
    expected_sha256 = hashlib.sha256(first).hexdigest()
    checksum = Path(f"{output}.sha256")
    assert checksum.read_text() == f"{expected_sha256}  {output.name}\n"
    assert not list(output.parent.glob(f".{output.name}.tmp.*"))

    assert MOD.main(["--stock-tape", str(tape), "--out", str(output)]) == 0
    assert output.read_bytes() == first
    assert checksum.read_text() == f"{expected_sha256}  {output.name}\n"
