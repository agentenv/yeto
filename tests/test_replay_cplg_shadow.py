"""Fail-closed tests for the CPLG full-vector stock-tape evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
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


def _canonical_json_line(value: dict) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _publish_tape(path: Path, rows: list[dict]) -> list[dict]:
    """Publish the exact chained tape/manifest shape emitted by the syncer."""

    ledger_head = "0" * 64
    published = []
    for source in rows:
        row = dict(source)
        row.pop("ledger_sha256", None)
        row["ledger_prev_sha256"] = ledger_head
        ledger_head = hashlib.sha256(_canonical_json_line(row)).hexdigest()
        row["ledger_sha256"] = ledger_head
        published.append(row)
    path.write_bytes(b"".join(_canonical_json_line(row) for row in published))

    first = published[0]
    vector_bytes = sum(row["numel"] * 4 for row in published)
    fragment_counts: dict[str, int] = {}
    for row in published:
        key = str(row["fragment"])
        fragment_counts[key] = fragment_counts.get(key, 0) + 1
    tape_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    records = len(published)
    manifest = {
        "schema": "cplg_stock_vector_tape_manifest_v1",
        "status": "COMPLETE",
        "capture_session_uuid": first["capture_session_uuid"],
        "layout_sha256": first["layout_sha256"],
        "initial_state_sha256": "4" * 64,
        "run_config_sha256": first["run_config_sha256"],
        "expected_records": records,
        "records": records,
        "vector_bytes": vector_bytes,
        "fragment_counts": fragment_counts,
        "stock_tape": path.name,
        "stock_tape_sha256": tape_sha256,
        "ledger_head": ledger_head,
        "writer": {
            "state": "closed",
            "accepted_items": records,
            "completed_items": records,
            "accepted_bytes": vector_bytes,
            "completed_bytes": vector_bytes,
            "dropped_items": 0,
            "dropped_bytes": 0,
            "abandoned_items": 0,
            "abandoned_bytes": 0,
            "pending_items": 0,
            "pending_bytes": 0,
            "error": None,
        },
    }
    manifest_path = path.with_name("stock_tape.manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    Path(f"{manifest_path}.sha256").write_text(
        f"{manifest_sha256}  {manifest_path.name}\n", encoding="ascii"
    )
    return published


def _stock_row(
    *,
    commit_seq: int,
    fragment: int,
    version_before: int,
    vector_path: Path,
    raw: bytes,
) -> dict:
    return {
        "schema": "cplg_stock_vector_row_v1",
        "capture_session_uuid": "00000000-0000-4000-8000-000000000001",
        "commit_seq": commit_seq,
        "step": commit_seq,
        "fragment": fragment,
        "fragment_version_before": version_before,
        "fragment_version_after": commit_seq,
        "layout_sha256": "1" * 64,
        "run_config_sha256": "2" * 64,
        "merge_rule": "production_weighted_rda",
        "wire_dtype": "f32_le",
        "numel": len(raw) // 4,
        "responders": [{"id": 0, "weight_f64_bits": "3ff0000000000000"}],
        "stock_f32le": str(vector_path),
        "stock_f32le_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _write_tape(root: Path, *, angles: tuple[float, ...] = (0, 10, 20, 30, 40)):
    root.mkdir(parents=True, exist_ok=True)
    vectors = root / "vectors"
    vectors.mkdir()
    rows = []
    for offset, angle in enumerate(angles, 1):
        raw = _direction(angle, norm=1.0 + offset)
        relative = Path("vectors") / f"commit-{offset:08}-fragment-0000.f32le"
        (root / relative).write_bytes(raw)
        rows.append(
            _stock_row(
                commit_seq=offset,
                fragment=0,
                version_before=offset - 1,
                vector_path=relative,
                raw=raw,
            )
        )
    tape = root / "stock_tape.jsonl"
    rows = _publish_tape(tape, rows)
    return tape, rows


def _rewrite(path: Path, rows: list[dict]) -> None:
    _publish_tape(path, rows)


def _write_gate_tape(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    vectors = root / "vectors"
    vectors.mkdir()
    rows = []
    prior_versions = {fragment: 0 for fragment in range(4)}
    visits = prior_versions.copy()
    for commit_seq, fragment in enumerate((0, 1, 2, 3) * 8, 1):
        visit = visits[fragment]
        visits[fragment] += 1
        raw = _direction(visit * 10.0, norm=1.0 + fragment)
        relative = (
            Path("vectors") / f"commit-{commit_seq:08}-fragment-{fragment:04}.f32le"
        )
        (root / relative).write_bytes(raw)
        rows.append(
            _stock_row(
                commit_seq=commit_seq,
                fragment=fragment,
                version_before=prior_versions[fragment],
                vector_path=relative,
                raw=raw,
            )
        )
        prior_versions[fragment] = commit_seq
    tape = root / "stock_tape.jsonl"
    _rewrite(tape, rows)
    return tape


def _write_overhead(
    path: Path,
    tape: Path,
    *,
    off_ns: int = 10_000,
    on_ns: int = 10_100,
) -> None:
    zero_writer = {
        "state": "disabled",
        "accepted_items": 0,
        "completed_items": 0,
        "accepted_bytes": 0,
        "completed_bytes": 0,
        "dropped_items": 0,
        "dropped_bytes": 0,
        "abandoned_items": 0,
        "abandoned_bytes": 0,
        "pending_items": 0,
        "pending_bytes": 0,
        "error": None,
    }
    on_writer = {
        **zero_writer,
        "state": "closed",
        "accepted_items": 32,
        "completed_items": 32,
        "accepted_bytes": 256,
        "completed_bytes": 256,
    }
    identity = "4" * 64
    common = {
        "interval_start_monotonic_ns": 100,
        "commits": 32,
        "local_steps": 34,
        "fragment_order": list((0, 1, 2, 3) * 8),
        "initial_state_sha256": identity,
        "input_manifest_sha256": "2" * 64,
        "schedule_sha256": "3" * 64,
        "runner_exit_code": 0,
        "evaluation_finite": True,
    }
    document = {
        "schema": "cplg_shadow_overhead_v1",
        "off": {
            **common,
            "capture_enabled": False,
            "interval_end_monotonic_ns": 100 + off_ns,
            "stock_tape_sha256": None,
            "writer": zero_writer,
        },
        "on": {
            **common,
            "capture_enabled": True,
            "interval_end_monotonic_ns": 100 + on_ns,
            "stock_tape_sha256": hashlib.sha256(tape.read_bytes()).hexdigest(),
            "writer": on_writer,
        },
    }
    path.write_text(json.dumps(document, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n")


def test_overhead_evidence_requires_matching_checksum(tmp_path: Path) -> None:
    tape = _write_gate_tape(tmp_path / "gate")
    overhead = tmp_path / "overhead.json"
    _write_overhead(overhead, tape)
    Path(f"{overhead}.sha256").write_text(f"{'0' * 64}  {overhead.name}\n")

    with pytest.raises(ValueError, match="checksum sidecar mismatch"):
        MOD.read_shadow_overhead_evidence(
            overhead,
            stock_tape_sha256=hashlib.sha256(tape.read_bytes()).hexdigest(),
            total_vector_bytes=256,
            initial_state_sha256="4" * 64,
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
    assert first["summary"]["sealed_shadows"] == 3
    assert first["summary"]["constructed_nonstock_shadows"] == 3
    assert first["summary"]["simulated_nonstock_actions"] == 0
    assert first["summary"]["resolved_shadows"] == 2
    assert first["summary"]["positive_resolved_shadows"] == 2
    assert first["summary"]["unresolved_tail_shadows"] == 1
    assert [record["reason"] for record in first["records"][:2]] == [
        "stock_warmup",
        "phase_warmup",
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


def test_rust_libm_authoritative_report_closes_portability_status(tmp_path) -> None:
    subprocess.run(
        ["cargo", "build", "--quiet", "--bin", "cplg_libm_oracle"],
        cwd=ROOT / "syncer",
        check=True,
    )
    helper = ROOT / "syncer" / "target" / "debug" / "cplg_libm_oracle"
    tape, _rows = _write_tape(tmp_path / "tape", angles=(0, 20, 40, 60, 80))
    output = tmp_path / "report.json"

    assert (
        MOD.main(
            [
                "--stock-tape",
                str(tape),
                "--rust-libm-helper",
                str(helper),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    contract = report["reference_contract"]
    assert contract["cross_runtime_bit_parity"] == ("SATISFIED_RUST_LIBM_AUTHORITATIVE")
    assert contract["authoritative_transcendental_runtime"] == "rust"
    assert contract["rust_libm_version"] == "0.2.15"
    assert len(contract["rust_libm_helper_sha256"]) == 64
    assert contract["trig_portability_limitation"] is None


def test_zero_phase_shadow_resolves_zero_and_blocks_interlock(tmp_path) -> None:
    tape, _rows = _write_tape(tmp_path / "zero", angles=(0, 10, 0, 10))
    report = MOD.evaluate_full_vector_stock_tape(tape)

    reversal = report["records"][2]
    following = report["records"][3]
    assert reversal["reason"] == "zero_or_rounded_phase"
    assert reversal["sealed_shadow"] is True
    assert reversal["candidate_is_nonstock"] is False
    assert following["resolved_shadow_cosine_gain"] == 0.0
    assert following["interlock_score_count"] == 1
    assert following["simulated_nonstock_action"] is False


def test_nonacute_boundary_clears_history_and_restarts_warmup(tmp_path) -> None:
    tape, _rows = _write_tape(tmp_path / "clear", angles=(0, 100, 10, 20, 30))
    report = MOD.evaluate_full_vector_stock_tape(tape)

    assert report["records"][1]["reason"] == "nonacute_turn"
    assert report["records"][1]["state_cleared"] is True
    assert report["records"][2]["reason"] == "stock_warmup"
    assert report["records"][3]["reason"] == "phase_warmup"


def test_frozen_shadow_gate_passes_constant_phase_fixture(tmp_path) -> None:
    subprocess.run(
        ["cargo", "build", "--quiet", "--bin", "cplg_libm_oracle"],
        cwd=ROOT / "syncer",
        check=True,
    )
    helper = ROOT / "syncer" / "target" / "debug" / "cplg_libm_oracle"
    tape = _write_gate_tape(tmp_path / "gate")
    overhead = tmp_path / "overhead.json"
    _write_overhead(overhead, tape)

    with MOD.CPLGRustLibmTrig(helper) as trig:
        report = MOD.evaluate_full_vector_stock_tape(
            tape,
            trig=trig,
            enforce_shadow_gate=True,
            overhead_evidence_path=overhead,
        )

    assert report["decision"] == "PASS"
    gate = report["shadow_gate"]
    assert gate["errors"] == []
    assert gate["activity"]["simulated_nonstock_actions"] == 12
    assert gate["direction"]["resolved_scores"] == 20
    assert gate["direction"]["positive_fragment_means"] == 4
    assert gate["direction"]["mean"] > 0.001
    assert gate["direction"]["bootstrap"]["lower_endpoint"] > 0.0
    assert gate["overhead"]["overhead_fraction"] == pytest.approx(0.01)
    assert gate["next_action"] == "write_and_review_separate_cplg_e1_preregistration"


def test_frozen_shadow_gate_fails_cost_without_hiding_other_metrics(tmp_path) -> None:
    subprocess.run(
        ["cargo", "build", "--quiet", "--bin", "cplg_libm_oracle"],
        cwd=ROOT / "syncer",
        check=True,
    )
    helper = ROOT / "syncer" / "target" / "debug" / "cplg_libm_oracle"
    tape = _write_gate_tape(tmp_path / "gate")
    overhead = tmp_path / "overhead.json"
    _write_overhead(overhead, tape, on_ns=10_300)

    with MOD.CPLGRustLibmTrig(helper) as trig:
        report = MOD.evaluate_full_vector_stock_tape(
            tape,
            trig=trig,
            enforce_shadow_gate=True,
            overhead_evidence_path=overhead,
        )

    assert report["decision"] == "FAIL"
    assert report["shadow_gate"]["direction"]["mean"] > 0.001
    assert report["shadow_gate"]["activity"]["simulated_nonstock_actions"] == 12
    assert report["shadow_gate"]["errors"] == [
        "matched_interval_overhead: required <=0.02, observed 0.03"
    ]
    assert report["shadow_gate"]["next_action"] == "kill_cplg_v1"


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


def test_ledger_manifest_and_completion_tampering_are_hard_errors(tmp_path) -> None:
    tape, _rows = _write_tape(tmp_path / "ledger")
    lines = tape.read_text().splitlines()
    first = json.loads(lines[0])
    first["ledger_sha256"] = "0" * 64
    lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    tape.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ledger row SHA-256 mismatch"):
        MOD.evaluate_full_vector_stock_tape(tape)

    tape, _rows = _write_tape(tmp_path / "manifest-checksum")
    checksum = tape.with_name("stock_tape.manifest.json.sha256")
    checksum.write_text(f"{'0' * 64}  stock_tape.manifest.json\n")
    with pytest.raises(ValueError, match="manifest checksum sidecar mismatch"):
        MOD.evaluate_full_vector_stock_tape(tape)

    tape, _rows = _write_tape(tmp_path / "incomplete")
    manifest_path = tape.with_name("stock_tape.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "INCOMPLETE"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    Path(f"{manifest_path}.sha256").write_text(
        f"{manifest_sha256}  {manifest_path.name}\n"
    )
    with pytest.raises(ValueError, match="capture status is not COMPLETE"):
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

    with pytest.raises(FileExistsError, match="not fresh"):
        MOD.main(["--stock-tape", str(tape), "--out", str(output)])
    assert output.read_bytes() == first
    assert checksum.read_text() == f"{expected_sha256}  {output.name}\n"
