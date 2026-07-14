from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_optimizer_capture_parity.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_optimizer_capture_parity", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _checkpoint_bytes(*, global_step: int = 1, version: int = 1) -> bytes:
    raw = bytearray()
    raw += struct.pack("<IQI", 0xD1705A7E, global_step, 1)
    raw += struct.pack("<QQ", version, 2)
    raw += struct.pack("<2f", 1.0, -2.0)
    raw += struct.pack("<2f", 0.25, -0.5)
    raw += struct.pack("<I", 1)
    raw += struct.pack("<IQQQ", 0, 1, 4, 8)
    return bytes(raw)


@dataclass
class Pair:
    off_arm: Path
    on_arm: Path
    off_results: Path
    on_results: Path
    output: Path
    candidate: bytes
    off_arm_name: str = "capture_m4_off"
    on_arm_name: str = "capture_m4_on"


def _probe_row(
    *, step: int, state_name: str, candidate_name: str, update_name: str
) -> dict[str, object]:
    return {
        "schema": "syncer_probe_capture_v1",
        "oracle_scope": "syncer_current_global_pending_offline",
        "step": step,
        "version": step,
        "syncer_global_step": step - 1,
        "fragment": 0,
        "current_fragment_version": step - 1,
        "learner_id": 0,
        "base_version": step - 1,
        "local_step": 4,
        "c_steps": 4,
        "c_tokens": 8,
        "weight": 16.0,
        "state_checkpoint": f"states/{state_name}",
        "candidate_f32": f"candidates/{candidate_name}",
        "applied_update_f32": f"applied_updates/{update_name}",
    }


def _make_probe(root: Path, *, prefix: str, candidate: bytes, update: bytes) -> None:
    for directory in ("states", "candidates", "applied_updates"):
        (root / directory).mkdir(parents=True)
    rows = []
    for step in (1, 2):
        state_name = f"{prefix}-step-{step}-state.ckpt"
        candidate_name = f"{prefix}-step-{step}-candidate.f32"
        update_name = f"{prefix}-step-{step}-update.f32"
        (root / "states" / state_name).write_bytes(
            _checkpoint_bytes(global_step=step - 1, version=step - 1)
        )
        (root / "candidates" / candidate_name).write_bytes(candidate)
        (root / "applied_updates" / update_name).write_bytes(update)
        rows.append(
            _probe_row(
                step=step,
                state_name=state_name,
                candidate_name=candidate_name,
                update_name=update_name,
            )
        )
    _write_jsonl(
        root / "index.jsonl",
        rows,
    )


def _make_transcript(path: Path, candidate: bytes) -> None:
    digest = hashlib.sha256(candidate).hexdigest()
    session = "capture-session-fixture"
    window = "01234567-89ab-cdef-8123-456789abcdef"
    rows = [
        {
            "schema": "syncer_response_transcript_header_v1",
            "capture_session_uuid": session,
            "event_seq": 1,
        }
    ]
    event_seq = 2
    for step in (1, 2):
        rows.extend(
            [
                {
                    "schema": "syncer_push_attempt_v1",
                    "capture_session_uuid": session,
                    "event_seq": event_seq,
                    "request_global_step": step,
                    "fragment_id": 0,
                    "learner_id": 0,
                    "window_uuid": window,
                    "wire_dtype": 1,
                    "base_version": step - 1,
                    "local_step": 4,
                    "c_steps": 4,
                    "c_tokens": 8,
                    "weight": 16.0,
                    "received_payload_sha256": digest,
                    "payload_digest_match": True,
                    "disposition": "admitted_pending",
                },
                {
                    "schema": "syncer_round_commit_v1",
                    "capture_session_uuid": session,
                    "event_seq": event_seq + 1,
                    "request_global_step": step,
                    "fragment_id": 0,
                    "wire_dtype": 1,
                    "responder_count": 1,
                    "responders": [
                        {
                            "responder_index": 0,
                            "learner_id": 0,
                            "source_attempt_event_seq": event_seq,
                            "window_uuid": window,
                            "base_version": step - 1,
                            "local_step": 4,
                            "c_steps": 4,
                            "c_tokens": 8,
                            "weight": 16.0,
                            "received_payload_sha256": digest,
                        }
                    ],
                },
            ]
        )
        event_seq += 2
    _write_jsonl(path, rows)


def _make_export(arm: Path) -> None:
    export = arm / "export"
    export.mkdir(parents=True)
    (export / "adapter_model.safetensors").write_bytes(b"exact tensor payload")
    (export / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": str(arm.resolve() / "base-model"),
                "rank": 4,
                "stable_label": "parity-fixture",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _result_rows(off_wall: float, on_wall: float) -> list[dict[str, object]]:
    return [
        {"arm": "base", "m": 1, "wall_s": 0.0, "eval_loss": 2.0},
        {
            "arm": "capture_m4_off",
            "m": 4,
            "wall_s": off_wall,
            "wall_scope": "syncer_commit_1_to_commit_N",
            "eval_loss": 1.0,
        },
        {
            "arm": "capture_m4_on",
            "m": 4,
            "wall_s": on_wall,
            "wall_scope": "syncer_commit_1_to_commit_N",
            "eval_loss": 1.0,
        },
    ]


def _make_pair(
    tmp_path: Path, *, off_wall: float = 100.0, on_wall: float = 101.8
) -> Pair:
    off_arm = tmp_path / "capture-off" / "work" / "m4"
    on_arm = tmp_path / "capture-on" / "work" / "m4"
    off_arm.mkdir(parents=True)
    on_arm.mkdir(parents=True)
    candidate = struct.pack("<2f", 1.25, -0.75)
    update = struct.pack("<2f", 0.125, -0.25)
    _make_probe(
        off_arm / "syncer_probe", prefix="off", candidate=candidate, update=update
    )
    _make_probe(
        on_arm / "syncer_probe", prefix="on", candidate=candidate, update=update
    )
    _make_transcript(on_arm / "syncer_response_transcript.jsonl", candidate)
    for arm in (off_arm, on_arm):
        (arm / "state.ckpt").write_bytes(_checkpoint_bytes())
        _make_export(arm)
    for arm, wall in ((off_arm, off_wall), (on_arm, on_wall)):
        _write_jsonl(
            arm / "tape.jsonl",
            [
                {
                    "step": 1,
                    "fragment": 0,
                    "commit_seq": 1,
                    "commit_elapsed_ns": 1_000_000_000,
                },
                {
                    "step": 2,
                    "fragment": 0,
                    "commit_seq": 2,
                    "commit_elapsed_ns": 1_000_000_000 + round(wall * 1_000_000_000),
                },
            ],
        )
    off_results = tmp_path / "report" / "results.jsonl"
    on_results = off_results
    _write_jsonl(off_results, _result_rows(off_wall, on_wall))
    return Pair(
        off_arm=off_arm,
        on_arm=on_arm,
        off_results=off_results,
        on_results=on_results,
        output=tmp_path / "report" / "optimizer_state_capture_parity.json",
        candidate=candidate,
    )


def _run(pair: Pair, *, require_barrier_schedule: bool = False) -> dict:
    return MOD.run_gate(
        off_arm_dir=pair.off_arm,
        on_arm_dir=pair.on_arm,
        off_results=pair.off_results,
        on_results=pair.on_results,
        off_arm=pair.off_arm_name,
        on_arm=pair.on_arm_name,
        output=pair.output,
        require_barrier_schedule=require_barrier_schedule,
    )


def _check(result: dict, name: str) -> dict:
    return next(check for check in result["checks"] if check["name"] == name)


def test_matched_pair_passes_with_only_known_path_metadata_canonicalized(tmp_path):
    pair = _make_pair(tmp_path)

    result = _run(pair)

    assert pair.off_results == pair.on_results
    assert result["status"] == "PASS"
    assert all(check["status"] == "PASS" for check in result["checks"])
    persisted = json.loads(pair.output.read_text(encoding="utf-8"))
    assert persisted["status"] == "PASS"
    assert "artifact_sha256" not in persisted
    digest = hashlib.sha256(pair.output.read_bytes()).hexdigest()
    assert result["artifact_sha256"] == digest
    assert (
        pair.output.with_name(pair.output.name + ".sha256").read_text(encoding="ascii")
        == f"{digest}  {pair.output.name}\n"
    )
    input_manifest = pair.output.with_suffix(".inputs.sha256")
    assert input_manifest.is_file()
    input_rows = input_manifest.read_text(encoding="utf-8").splitlines()
    assert len(input_rows) == 24
    sealed = _check(result, "sealed_input_tree")["detail"]
    assert sealed["files"] == len(input_rows)
    assert sealed["sha256"] == hashlib.sha256(input_manifest.read_bytes()).hexdigest()
    for row in input_rows:
        expected, relative = row.split("  ", 1)
        assert (
            hashlib.sha256((input_manifest.parent / relative).read_bytes()).hexdigest()
            == expected
        )


def _set_second_commit_local_step(pair: Pair, local_step: int) -> None:
    for arm in (pair.off_arm, pair.on_arm):
        path = arm / "syncer_probe" / "index.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["local_step"] = local_step
        _write_jsonl(path, rows)
    transcript = pair.on_arm / "syncer_response_transcript.jsonl"
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    for row in rows:
        if row.get("request_global_step") == 2:
            if row.get("schema") == "syncer_push_attempt_v1":
                row["local_step"] = local_step
            elif row.get("schema") == "syncer_round_commit_v1":
                row["responders"][0]["local_step"] = local_step
    _write_jsonl(transcript, rows)


def test_required_barrier_schedule_accepts_exact_fixed_h_waves(tmp_path):
    pair = _make_pair(tmp_path)
    _set_second_commit_local_step(pair, 8)

    result = _run(pair, require_barrier_schedule=True)

    assert result["status"] == "PASS"
    detail = _check(result, "syncer_probe_exact_payload_parity")["detail"]
    assert detail["barrier_schedule"]["off"]["wave_local_steps"] == [4, 8]
    assert detail["barrier_schedule"]["on"]["horizon_steps"] == 4


def test_required_barrier_schedule_rejects_nonbarrier_step_advance(tmp_path):
    pair = _make_pair(tmp_path)

    result = _run(pair, require_barrier_schedule=True)

    check = _check(result, "syncer_probe_exact_payload_parity")
    assert result["status"] == "FAIL"
    assert "advanced 0 local steps; expected H=4" in check["error"]


def test_required_barrier_schedule_rejects_split_responder_boundary(tmp_path):
    rows = {
        (1, 0, learner): {
            "learner_id": learner,
            "local_step": local_step,
            "c_steps": 4,
            "c_tokens": 512,
            "base_version": 0,
        }
        for learner, local_step in ((0, 10), (1, 10), (2, 11), (3, 11))
    }
    probe = MOD.ProbeCapture(
        root=tmp_path,
        canonical_rows=rows,
        candidate_digests={},
        state_digests={(1, 0): "state"},
        update_digests={(1, 0): "update"},
        referenced_manifest={},
    )

    with pytest.raises(MOD.ParityError, match="responders disagree on local_step"):
        MOD.validate_barrier_schedule(probe, label="capture-ON")


def test_input_manifest_detects_post_gate_input_mutation(tmp_path):
    pair = _make_pair(tmp_path)
    result = _run(pair)
    assert result["status"] == "PASS"
    input_manifest = pair.output.with_suffix(".inputs.sha256")

    on_candidate = next((pair.on_arm / "syncer_probe" / "candidates").iterdir())
    on_candidate.write_bytes(struct.pack("<2f", 9.0, -0.75))

    mismatches = []
    for row in input_manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = row.split("  ", 1)
        actual = hashlib.sha256(
            (input_manifest.parent / relative).read_bytes()
        ).hexdigest()
        if actual != expected:
            mismatches.append(relative)
    assert len(mismatches) == 1
    assert mismatches[0].endswith("/" + on_candidate.name)


def test_candidate_byte_difference_fails_and_still_writes_evidence(tmp_path):
    pair = _make_pair(tmp_path)
    on_candidate = next((pair.on_arm / "syncer_probe" / "candidates").iterdir())
    on_candidate.write_bytes(struct.pack("<2f", 1.25, -0.5))

    result = _run(pair)

    assert result["status"] == "FAIL"
    check = _check(result, "syncer_probe_exact_payload_parity")
    assert check["status"] == "FAIL"
    assert "probe payload bytes differ" in check["error"]
    assert pair.output.is_file()
    assert pair.output.with_name(pair.output.name + ".sha256").is_file()


def test_extra_unindexed_probe_payload_fails_closed(tmp_path):
    pair = _make_pair(tmp_path)
    (pair.on_arm / "syncer_probe" / "candidates" / "stale.f32").write_bytes(
        struct.pack("<f", 0.0)
    )

    result = _run(pair)

    check = _check(result, "syncer_probe_exact_payload_parity")
    assert result["status"] == "FAIL"
    assert "payload set mismatch" in check["error"]
    assert "stale.f32" in check["error"]


def test_identical_nonfinite_candidate_payloads_are_still_malformed(tmp_path):
    pair = _make_pair(tmp_path)
    malformed = struct.pack("<2f", float("nan"), 0.0)
    for arm in (pair.off_arm, pair.on_arm):
        candidate = next((arm / "syncer_probe" / "candidates").iterdir())
        candidate.write_bytes(malformed)
    _make_transcript(pair.on_arm / "syncer_response_transcript.jsonl", malformed)

    result = _run(pair)

    check = _check(result, "syncer_probe_exact_payload_parity")
    assert result["status"] == "FAIL"
    assert "non-finite f32" in check["error"]


def test_transcript_must_cover_every_indexed_responder(tmp_path):
    pair = _make_pair(tmp_path)
    for arm in (pair.off_arm, pair.on_arm):
        probe = arm / "syncer_probe"
        rows = [
            json.loads(line)
            for line in probe.joinpath("index.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        second = dict(rows[0])
        second["learner_id"] = 1
        second["candidate_f32"] = "candidates/extra-learner-1.f32"
        (probe / second["candidate_f32"]).write_bytes(pair.candidate)
        rows.append(second)
        _write_jsonl(probe / "index.jsonl", rows)

    result = _run(pair)

    assert _check(result, "syncer_probe_exact_payload_parity")["status"] == "PASS"
    check = _check(result, "capture_on_transcript_join")
    assert result["status"] == "FAIL"
    assert "responder/probe learner sets differ" in check["error"]


def test_exact_producer_interval_over_two_percent_fails(tmp_path):
    pair = _make_pair(tmp_path, off_wall=100.0, on_wall=102.1)

    result = _run(pair)

    check = _check(result, "eval_and_wall_overhead")
    assert result["status"] == "FAIL"
    assert "exact=" in check["error"]
    assert "limit=0.02000000" in check["error"]


def test_cold_start_wall_scope_cannot_pass_overhead_gate(tmp_path):
    pair = _make_pair(tmp_path)
    rows = _result_rows(100.0, 101.0)
    rows[1].pop("wall_scope")
    _write_jsonl(pair.off_results, rows)

    result = _run(pair)

    check = _check(result, "eval_and_wall_overhead")
    assert result["status"] == "FAIL"
    assert "steady-state wall scope" in check["error"]


def test_timing_tape_must_have_exact_commit_sequence(tmp_path):
    pair = _make_pair(tmp_path)
    rows = [
        {
            "step": 1,
            "fragment": 0,
            "commit_seq": 1,
            "commit_elapsed_ns": 1_000_000_000,
        },
        {
            "step": 3,
            "fragment": 0,
            "commit_seq": 3,
            "commit_elapsed_ns": 3_000_000_000,
        },
    ]
    _write_jsonl(pair.on_arm / "tape.jsonl", rows)

    result = _run(pair)

    check = _check(result, "syncer_commit_interval_timing")
    assert result["status"] == "FAIL"
    assert "non-contiguous commit_seq" in check["error"]


def test_results_wall_must_match_sealed_timing_tape(tmp_path):
    pair = _make_pair(tmp_path)
    _write_jsonl(pair.off_results, _result_rows(99.0, 101.8))

    result = _run(pair)

    check = _check(result, "eval_and_wall_overhead")
    assert result["status"] == "FAIL"
    assert "does not match the sealed producer timing" in check["error"]


def test_off_on_timing_commit_order_must_match(tmp_path):
    pair = _make_pair(tmp_path)
    _write_jsonl(
        pair.on_arm / "tape.jsonl",
        [
            {
                "step": 2,
                "fragment": 0,
                "commit_seq": 1,
                "commit_elapsed_ns": 1_000_000_000,
            },
            {
                "step": 1,
                "fragment": 0,
                "commit_seq": 2,
                "commit_elapsed_ns": 102_800_000_000,
            },
        ],
    )

    result = _run(pair)

    check = _check(result, "syncer_commit_interval_timing")
    assert result["status"] == "FAIL"
    assert "different ordered commit identities" in check["error"]


def test_timing_tapes_must_exactly_cover_probe_commit_groups(tmp_path):
    pair = _make_pair(tmp_path)
    rows = [
        {
            "step": 1,
            "fragment": 0,
            "commit_seq": 1,
            "commit_elapsed_ns": 1_000_000_000,
        },
        {
            "step": 3,
            "fragment": 0,
            "commit_seq": 2,
            "commit_elapsed_ns": 101_000_000_000,
        },
    ]
    _write_jsonl(pair.off_arm / "tape.jsonl", rows)
    _write_jsonl(pair.on_arm / "tape.jsonl", rows)

    result = _run(pair)

    check = _check(result, "syncer_commit_interval_timing")
    assert result["status"] == "FAIL"
    assert "timing tape/probe commit groups differ" in check["error"]


def test_identically_truncated_timing_tapes_fail_closed(tmp_path):
    pair = _make_pair(tmp_path)
    row = {
        "step": 1,
        "fragment": 0,
        "commit_seq": 1,
        "commit_elapsed_ns": 1_000_000_000,
    }
    _write_jsonl(pair.off_arm / "tape.jsonl", [row])
    _write_jsonl(pair.on_arm / "tape.jsonl", [row])

    result = _run(pair)

    check = _check(result, "syncer_commit_interval_timing")
    assert result["status"] == "FAIL"
    assert "at least two commits" in check["error"]


def test_unknown_export_metadata_is_not_canonicalized(tmp_path):
    pair = _make_pair(tmp_path)
    for arm in (pair.off_arm, pair.on_arm):
        path = arm / "export" / "adapter_config.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["experiment_label"] = str(arm.resolve())
        path.write_text(json.dumps(value), encoding="utf-8")

    result = _run(pair)

    check = _check(result, "export_payload_parity")
    assert result["status"] == "FAIL"
    assert "adapter_config.json" in check["error"]


def _set_adapter_config_value(arm: Path, key: str, value) -> None:
    path = arm / "export" / "adapter_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config[key] = value
    path.write_text(json.dumps(config), encoding="utf-8")


def test_peft_target_modules_list_order_only_passes_and_preserves_duplicates(tmp_path):
    pair = _make_pair(tmp_path)
    _set_adapter_config_value(
        pair.off_arm, "target_modules", ["q_proj", "v_proj", "v_proj"]
    )
    _set_adapter_config_value(
        pair.on_arm, "target_modules", ["v_proj", "q_proj", "v_proj"]
    )

    result = _run(pair)

    assert result["status"] == "PASS"
    detail = _check(result, "export_payload_parity")["detail"]
    assert detail["canonical_json_set_list_fields"] == [
        {"relative_path": "adapter_config.json", "json_path": ["target_modules"]}
    ]
    assert result["canonicalization"]["export_json_set_list_fields"] == [
        {
            "relative_path": "adapter_config.json",
            "json_path": ["target_modules"],
            "operation": "sort_preserving_multiplicity",
        }
    ]


@pytest.mark.parametrize(
    ("off_value", "on_value"),
    [
        (["q_proj", "v_proj"], ["q_proj", "v_proj", "v_proj"]),
        (["q_proj", "v_proj"], ["q_proj", "o_proj"]),
        (["q_proj", "v_proj"], ["q_proj", 7]),
        (["q_proj", "v_proj"], "q_proj|v_proj"),
    ],
    ids=["multiplicity", "content", "element-type", "container-type"],
)
def test_peft_target_modules_semantic_changes_fail_closed(
    tmp_path, off_value, on_value
):
    pair = _make_pair(tmp_path)
    _set_adapter_config_value(pair.off_arm, "target_modules", off_value)
    _set_adapter_config_value(pair.on_arm, "target_modules", on_value)

    result = _run(pair)

    assert result["status"] == "FAIL"
    check = _check(result, "export_payload_parity")
    assert check["status"] == "FAIL"
    assert "adapter_config.json" in check["error"]


@pytest.mark.parametrize("case", ["other-list", "nested-key", "other-json"])
def test_target_module_exception_does_not_weaken_other_export_exactness(tmp_path, case):
    pair = _make_pair(tmp_path)
    if case == "other-list":
        _set_adapter_config_value(pair.off_arm, "modules_to_save", ["a", "b"])
        _set_adapter_config_value(pair.on_arm, "modules_to_save", ["b", "a"])
    elif case == "nested-key":
        _set_adapter_config_value(
            pair.off_arm, "nested", {"target_modules": ["q_proj", "v_proj"]}
        )
        _set_adapter_config_value(
            pair.on_arm, "nested", {"target_modules": ["v_proj", "q_proj"]}
        )
    elif case == "other-json":
        for arm, values in (
            (pair.off_arm, ["q_proj", "v_proj"]),
            (pair.on_arm, ["v_proj", "q_proj"]),
        ):
            (arm / "export" / "other.json").write_text(
                json.dumps({"target_modules": values}), encoding="utf-8"
            )
    else:  # pragma: no cover
        raise AssertionError(case)

    result = _run(pair)

    assert result["status"] == "FAIL"
    assert _check(result, "export_payload_parity")["status"] == "FAIL"


def test_missing_on_transcript_fails_closed(tmp_path):
    pair = _make_pair(tmp_path)
    (pair.on_arm / "syncer_response_transcript.jsonl").unlink()

    result = _run(pair)

    check = _check(result, "capture_on_transcript_join")
    assert result["status"] == "FAIL"
    assert "missing regular JSONL" in check["error"]


def test_duplicate_probe_json_key_is_malformed_not_last_value_wins(tmp_path):
    pair = _make_pair(tmp_path)
    index = pair.on_arm / "syncer_probe" / "index.jsonl"
    row = index.read_text(encoding="utf-8").rstrip("\n")
    index.write_text(row[:-1] + ',"learner_id":0}\n', encoding="utf-8")

    result = _run(pair)

    check = _check(result, "syncer_probe_exact_payload_parity")
    assert result["status"] == "FAIL"
    assert "duplicate JSON key" in check["error"]


def test_identically_malformed_final_checkpoints_do_not_pass_by_digest(tmp_path):
    pair = _make_pair(tmp_path)
    for arm in (pair.off_arm, pair.on_arm):
        (arm / "state.ckpt").write_bytes(b"same but not a checkpoint")

    result = _run(pair)

    check = _check(result, "final_syncer_checkpoint_parity")
    assert result["status"] == "FAIL"
    assert "invalid final syncer checkpoint" in check["error"]


def test_cli_default_output_is_canonical_report_path():
    args = MOD.parse_args(
        [
            "--off-arm-dir",
            "off",
            "--on-arm-dir",
            "on",
            "--off-results",
            "off-results.jsonl",
            "--on-results",
            "on-results.jsonl",
            "--off-arm",
            "capture_m4_off",
            "--on-arm",
            "capture_m4_on",
        ]
    )

    assert args.output == Path("report/optimizer_state_capture_parity.json")


def test_invalid_overhead_limit_writes_fail_evidence_instead_of_nan_json(tmp_path):
    pair = _make_pair(tmp_path)

    result = MOD.run_gate(
        off_arm_dir=pair.off_arm,
        on_arm_dir=pair.on_arm,
        off_results=pair.off_results,
        on_results=pair.on_results,
        off_arm=pair.off_arm_name,
        on_arm=pair.on_arm_name,
        output=pair.output,
        overhead_limit=float("nan"),
    )

    assert result["status"] == "FAIL"
    assert _check(result, "eval_and_wall_overhead")["status"] == "FAIL"
    persisted = json.loads(pair.output.read_text(encoding="utf-8"))
    assert persisted["thresholds"]["wall_overhead_fraction"] is None
    assert "finite and nonnegative" in persisted["thresholds"]["configuration_error"]
