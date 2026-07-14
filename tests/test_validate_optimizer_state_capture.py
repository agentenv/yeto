from __future__ import annotations

import hashlib
import json
import struct
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scripts.validate_optimizer_state_capture import (
    ArtifactRef,
    CommittedBoundary,
    CommittedBoundaryIndex,
    Expectations,
    ResponderRef,
    ValidationError,
    extract_committed_boundary_index,
    main,
    validate_and_write,
    validate_campaign,
)
from yeto.fragments import Fragment, FragmentLayout, MERGE_RDA
from yeto.optimizer_state_capture import (
    OptimizerStateCapture,
    capture_value_sha256,
)
from yeto.protocol import DTYPE_F32
from yeto.tensor_io import pack_flat


MAX_BYTES = 10_000_000


def _weight(c_tokens: int, c_steps: int) -> float:
    tokens = float(c_tokens)
    return tokens * tokens / float(c_steps)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(capture_dir: Path, manifest: dict) -> None:
    path = capture_dir / "manifest.json"
    raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_suffix(".json.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )


def _mutate_manifest(capture_dir: Path, mutation) -> None:
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    mutation(manifest)
    _write_manifest(capture_dir, manifest)


def _mutate_artifact(capture_dir: Path, kind: str, mutation) -> None:
    artifact = next(capture_dir.glob(f"*-{kind}-*.pt"))
    envelope = torch.load(artifact, map_location="cpu", weights_only=False)
    mutation(envelope)
    envelope["payload_sha256"] = capture_value_sha256(envelope["payload"])
    torch.save(envelope, artifact)
    digest = _sha256(artifact)
    sidecar = artifact.with_suffix(".pt.sha256")
    sidecar.write_text(f"{digest}  {artifact.name}\n", encoding="ascii")

    def update(manifest):
        entry = next(
            row for row in manifest["artifacts"] if row["path"] == artifact.name
        )
        entry["sha256"] = digest
        entry["bytes"] = artifact.stat().st_size
        entry["sidecar_bytes"] = sidecar.stat().st_size
        manifest["counters"]["artifact_bytes"] = sum(
            row["bytes"] + row["sidecar_bytes"] for row in manifest["artifacts"]
        )

    _mutate_manifest(capture_dir, update)


def _write_transcript(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mutate_transcript(arm_dir: Path, mutation) -> None:
    path = arm_dir / "syncer_response_transcript.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows)
    _write_transcript(path, rows)


def _capture_one(
    arm_dir: Path,
    learner_id: int,
    *,
    pull_global_step: int = 44,
    forced_window_uuid: str | None = None,
    background_writer: bool = False,
) -> None:
    capture_dir = arm_dir / f"optimizer_state_capture_learner_{learner_id}"
    params = OrderedDict(
        [("p", torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32)))]
    )
    layout = FragmentLayout([Fragment(MERGE_RDA, [("p", 2)])])
    optimizer = torch.optim.AdamW(params.values(), lr=0.1, weight_decay=0.01)
    capture = OptimizerStateCapture(
        capture_dir,
        params=params,
        layout=layout,
        optimizer=optimizer,
        learner_id=learner_id,
        rank=0,
        every=1,
        max_hmc_events=4,
        max_midpoint_windows=4,
        max_bytes=MAX_BYTES,
        background_writer=background_writer,
        background_writer_max_items=2,
        background_writer_max_bytes=MAX_BYTES,
    )
    if forced_window_uuid is not None:
        capture._make_window_uuid = (  # type: ignore[method-assign]
            lambda _identity, _ordinal, _steps: forced_window_uuid
        )
    window_uuid = capture.note_window_reset(
        0,
        17,
        local_step=30,
        tokens_total=3_000,
        window_steps=2,
        reason="broadcast",
    )
    assert window_uuid is not None
    for local_step in (31, 32):
        capture.capture_first_post_broadcast_gradients(
            local_step_before_update=local_step - 1,
            tokens_total=3_000 + (local_step - 31) * 128,
            clip_total_norm=torch.tensor(0.5),
            clip_max_norm=1.0,
        )
        with torch.no_grad():
            params["p"].add_(torch.tensor([0.25, -0.5]))
        capture.after_optimizer_step(
            local_step=local_step,
            tokens_total=3_000 + (local_step - 30) * 128,
            current_window_steps=2,
        )
    endpoint = params["p"].detach().clone()
    push = {
        "window_uuid": window_uuid,
        "fragment_id": 0,
        "pull_global_step": pull_global_step,
        "base_version": 17,
        "local_step": 32,
        "c_steps": 2,
        "c_tokens": 256,
        "wire_codec": "f32",
        "payload": pack_flat(endpoint, DTYPE_F32),
    }
    first = capture.note_push(**push)
    capture.note_push_enqueued(first["attempt_serial"])
    second = capture.note_push(**push)
    capture.note_push_enqueued(second["attempt_serial"])
    capture.close()


def _make_transcript(arm_dir: Path, learner_ids: tuple[int, ...]) -> None:
    session = "capture-session-fixture"
    rows = [
        {
            "schema": "syncer_response_transcript_header_v1",
            "capture_session_uuid": session,
            "event_seq": 1,
        }
    ]
    source_event_by_learner = {}
    attempts_by_learner = {}
    for learner_id in learner_ids:
        capture_dir = arm_dir / f"optimizer_state_capture_learner_{learner_id}"
        attempts = []
        for artifact in sorted(capture_dir.glob("*-push_candidate-*.pt")):
            attempts.append(
                torch.load(artifact, map_location="cpu", weights_only=False)["metadata"]
            )
        attempts_by_learner[learner_id] = attempts
        for metadata in attempts:
            event_seq = len(rows) + 1
            if metadata["attempt_serial"] == 1:
                source_event_by_learner[learner_id] = event_seq
            admitted = metadata["attempt_serial"] == 1
            weight = _weight(metadata["c_tokens"], metadata["c_steps"])
            rows.append(
                {
                    "schema": "syncer_push_attempt_v1",
                    "capture_session_uuid": session,
                    "event_seq": event_seq,
                    "request_global_step": metadata["pull_global_step"],
                    "fragment_id": metadata["fragment_id"],
                    "learner_id": metadata["learner_id"],
                    "connection_id": 100 + learner_id,
                    "window_uuid": metadata["window_uuid"],
                    "attempt_serial": metadata["attempt_serial"],
                    "base_version": metadata["base_version"],
                    "local_step": metadata["local_step"],
                    "c_steps": metadata["c_steps"],
                    "c_tokens": metadata["c_tokens"],
                    "wire_dtype": DTYPE_F32,
                    "declared_payload_sha256": metadata["payload_sha256"],
                    "received_payload_sha256": metadata["payload_sha256"],
                    "payload_digest_match": True,
                    "weight": weight,
                    "weight_f64_bits": struct.pack(">d", weight).hex(),
                    "disposition": "admitted_pending"
                    if admitted
                    else "rejected_duplicate",
                    "reason": None if admitted else "duplicate learner response",
                    "source_attempt_event_seq": None,
                }
            )
    responders = []
    for responder_index, learner_id in enumerate(sorted(learner_ids)):
        metadata = attempts_by_learner[learner_id][0]
        weight = _weight(metadata["c_tokens"], metadata["c_steps"])
        responders.append(
            {
                "responder_index": responder_index,
                "learner_id": learner_id,
                "connection_id": 100 + learner_id,
                "source_attempt_event_seq": source_event_by_learner[learner_id],
                "window_uuid": metadata["window_uuid"],
                "attempt_serial": metadata["attempt_serial"],
                "base_version": metadata["base_version"],
                "local_step": metadata["local_step"],
                "c_steps": metadata["c_steps"],
                "c_tokens": metadata["c_tokens"],
                "weight": weight,
                "weight_f64_bits": struct.pack(">d", weight).hex(),
                "received_payload_sha256": metadata["payload_sha256"],
            }
        )
    rows.append(
        {
            "schema": "syncer_round_commit_v1",
            "capture_session_uuid": session,
            "event_seq": len(rows) + 1,
            "commit_id": "step-00000044-fragment-0000",
            "request_global_step": 44,
            "fragment_id": 0,
            "previous_fragment_version": 43,
            "committed_fragment_version": 44,
            "syncer_global_step_before": 43,
            "syncer_global_step_after": 44,
            "strict_quorum": True,
            "configured_quorum": len(learner_ids),
            "responder_count": len(responders),
            "wire_dtype": DTYPE_F32,
            "commit_policy": "token_weighted",
            "broadcast_payload_sha256": "b" * 64,
            "responders": responders,
        }
    )
    _write_transcript(arm_dir / "syncer_response_transcript.jsonl", rows)


def _campaign(
    tmp_path: Path, *, background_writer: bool = False
) -> tuple[Path, Expectations]:
    arm_dir = tmp_path / "m2"
    arm_dir.mkdir()
    _capture_one(arm_dir, 0, background_writer=background_writer)
    _capture_one(arm_dir, 1, background_writer=background_writer)
    _make_transcript(arm_dir, (0, 1))
    expectations = Expectations(
        learner_ids=(0, 1),
        fragments=1,
        window_steps=2,
        every=1,
        max_hmc_events=4,
        max_midpoint_windows=4,
        max_bytes=MAX_BYTES,
        min_joined_boundaries=1,
        min_joined_per_fragment=1,
        background_writer=background_writer,
        background_writer_max_items=2 if background_writer else 0,
        background_writer_max_bytes=MAX_BYTES if background_writer else 0,
    )
    return arm_dir, expectations


def _cli(arm_dir: Path) -> list[str]:
    return [
        "--arm-dir",
        str(arm_dir),
        "--expected-learners",
        "0,1",
        "--expected-fragments",
        "1",
        "--expected-h",
        "2",
        "--expected-every",
        "1",
        "--expected-max-hmc-events",
        "4",
        "--expected-max-midpoint-windows",
        "4",
        "--expected-max-bytes",
        str(MAX_BYTES),
        "--min-joined-boundaries",
        "1",
        "--min-joined-per-fragment",
        "1",
    ]


def test_valid_push_linked_campaign_emits_atomic_summary_and_relative_tree(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)
    summary = validate_and_write(arm_dir, expectations)

    assert summary["status"] == "PASS"
    assert summary["join_mode"] == "authoritative_syncer_commit"
    assert summary["joined_boundaries"] == 1
    assert summary["joined_by_fragment"] == {"0": 1}
    assert summary["join_key_fields"] == ["fragment_id", "request_global_step"]
    assert summary["response_transcript"]["primary_attempts"] == 4
    assert summary["response_transcript"]["commits"] == 1

    validation = arm_dir / "optimizer_state_capture_validation.json"
    validation_sidecar = validation.with_suffix(".json.sha256")
    tree = arm_dir / "optimizer_state_capture_artifacts.sha256"
    assert validation.is_file() and validation_sidecar.is_file() and tree.is_file()
    assert (
        validation_sidecar.read_text() == f"{_sha256(validation)}  {validation.name}\n"
    )
    tree_lines = tree.read_text().splitlines()
    assert all(not line.split("  ", 1)[1].startswith("/") for line in tree_lines)
    assert any(line.endswith("/manifest.json") for line in tree_lines)
    assert any("-push_candidate-" in line for line in tree_lines)
    assert any(
        line.endswith("  syncer_response_transcript.jsonl") for line in tree_lines
    )
    assert not list(arm_dir.rglob("*.tmp-*"))


def test_committed_boundary_index_binds_exact_ordered_artifacts_and_writes_canonical(
    tmp_path,
):
    arm_dir, expectations = _campaign(tmp_path)
    index = extract_committed_boundary_index(arm_dir, expectations)

    assert index.capture_session_uuid == "capture-session-fixture"
    assert index.source_transcript.path == "syncer_response_transcript.jsonl"
    assert index.source_transcript.sha256 == _sha256(
        arm_dir / "syncer_response_transcript.jsonl"
    )
    assert len(index.boundaries) == 1
    boundary = index.boundaries[0]
    assert boundary.commit_event_seq == 6
    assert boundary.fragment_id == 0
    assert boundary.request_global_step == 44
    assert [row.responder_index for row in boundary.responders] == [0, 1]
    assert [row.learner_id for row in boundary.responders] == [0, 1]

    for responder in boundary.responders:
        assert not responder.push.path.startswith("/")
        assert not responder.richardson.path.startswith("/")
        push_path = arm_dir / responder.push.path
        richardson_path = arm_dir / responder.richardson.path
        assert _sha256(push_path) == responder.push.sha256
        assert _sha256(richardson_path) == responder.richardson.sha256
        push = torch.load(push_path, map_location="cpu", weights_only=False)
        richardson = torch.load(richardson_path, map_location="cpu", weights_only=False)
        assert push["kind"] == "push_candidate"
        assert push["metadata"]["learner_id"] == responder.learner_id
        assert push["metadata"]["attempt_serial"] == responder.attempt_serial
        assert push["metadata"]["window_uuid"] == responder.window_uuid
        assert richardson["kind"] == "richardson_window"
        assert richardson["metadata"]["learner_id"] == responder.learner_id
        assert richardson["metadata"]["window_uuid"] == responder.window_uuid
        endpoint = richardson["payload"]["endpoint"]["parameters_f32"]
        assert hashlib.sha256(pack_flat(endpoint, DTYPE_F32)).hexdigest() == (
            responder.payload_sha256
        )

    _, tree_text = validate_campaign(arm_dir, expectations)
    assert (
        index.source_tree_manifest_sha256
        == hashlib.sha256(tree_text.encode("ascii")).hexdigest()
    )
    expected_raw = index.canonical_bytes()
    assert expected_raw.endswith(b"\n") and not expected_raw.endswith(b"\n\n")
    assert expected_raw == index.canonical_bytes()

    summary = validate_and_write(arm_dir, expectations)
    output = arm_dir / "optimizer_state_capture_committed_boundaries.json"
    sidecar = output.with_suffix(".json.sha256")
    assert output.read_bytes() == expected_raw
    assert sidecar.read_text() == f"{_sha256(output)}  {output.name}\n"
    assert summary["committed_boundary_index_sha256"] == _sha256(output)
    assert (
        json.loads(output.read_text())["boundaries"][0]["responders"][0]["learner_id"]
        == 0
    )


def test_candidate_join_is_keyed_by_learner_even_when_window_uuid_is_shared(tmp_path):
    arm_dir = tmp_path / "shared-window"
    arm_dir.mkdir()
    shared_uuid = "01234567-89ab-5def-8123-456789abcdef"
    _capture_one(arm_dir, 0, forced_window_uuid=shared_uuid)
    _capture_one(arm_dir, 1, forced_window_uuid=shared_uuid)
    _make_transcript(arm_dir, (0, 1))
    expectations = Expectations(
        learner_ids=(0, 1),
        fragments=1,
        window_steps=2,
        every=1,
        max_hmc_events=4,
        max_midpoint_windows=4,
        max_bytes=MAX_BYTES,
        min_joined_boundaries=1,
        min_joined_per_fragment=1,
    )

    responders = (
        extract_committed_boundary_index(arm_dir, expectations).boundaries[0].responders
    )
    assert {row.window_uuid for row in responders} == {shared_uuid}
    assert responders[0].richardson.path.startswith(
        "optimizer_state_capture_learner_0/"
    )
    assert responders[1].richardson.path.startswith(
        "optimizer_state_capture_learner_1/"
    )


def test_same_payload_retry_resolves_by_source_attempt_not_digest(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)

    def commit_second_attempt(rows):
        attempts = [
            row
            for row in rows
            if row["schema"] == "syncer_push_attempt_v1" and row["learner_id"] == 0
        ]
        first, second = attempts
        first["disposition"] = "rejected_duplicate"
        second["disposition"] = "admitted_pending"
        responder = next(
            row
            for row in next(
                item for item in rows if item["schema"] == "syncer_round_commit_v1"
            )["responders"]
            if row["learner_id"] == 0
        )
        responder["source_attempt_event_seq"] = second["event_seq"]
        responder["attempt_serial"] = second["attempt_serial"]

    _mutate_transcript(arm_dir, commit_second_attempt)
    responder = (
        extract_committed_boundary_index(arm_dir, expectations)
        .boundaries[0]
        .responders[0]
    )
    push = torch.load(
        arm_dir / responder.push.path, map_location="cpu", weights_only=False
    )
    assert responder.attempt_serial == 2
    assert push["metadata"]["attempt_serial"] == 2


@pytest.mark.parametrize(
    ("bits", "weight", "message"),
    [
        ("0" * 15, 0.0, "16 lowercase hexadecimal"),
        ("7ff8000000000000", 0.0, "finite non-negative"),
        (struct.pack(">d", 1.0).hex(), 2.0, "does not round-trip"),
        (struct.pack(">d", 128.0).hex(), 128.0, "production formula"),
    ],
)
def test_transcript_weight_bits_fail_closed(tmp_path, bits, weight, message):
    arm_dir, expectations = _campaign(tmp_path)

    def corrupt_weight(rows):
        commit = next(row for row in rows if row["schema"] == "syncer_round_commit_v1")
        responder = commit["responders"][0]
        source_seq = responder["source_attempt_event_seq"]
        source = next(row for row in rows if row["event_seq"] == source_seq)
        for row in (source, responder):
            row["weight_f64_bits"] = bits
            row["weight"] = weight

    _mutate_transcript(arm_dir, corrupt_weight)
    with pytest.raises(ValidationError, match=message):
        extract_committed_boundary_index(arm_dir, expectations)


@pytest.mark.parametrize("replacement", [None, "not-a-sha256"])
def test_boundary_index_rejects_missing_or_malformed_broadcast_digest(
    tmp_path, replacement
):
    arm_dir, expectations = _campaign(tmp_path)

    def corrupt_digest(rows):
        commit = next(row for row in rows if row["schema"] == "syncer_round_commit_v1")
        if replacement is None:
            del commit["broadcast_payload_sha256"]
        else:
            commit["broadcast_payload_sha256"] = replacement

    _mutate_transcript(arm_dir, corrupt_digest)
    with pytest.raises(ValidationError, match="broadcast_payload_sha256"):
        extract_committed_boundary_index(arm_dir, expectations)


def test_canonical_serialization_preserves_boundary_and_responder_tuple_order():
    artifact = ArtifactRef(path="capture/object.pt", sha256="a" * 64)

    def responder(index: int, learner_id: int) -> ResponderRef:
        return ResponderRef(
            responder_index=index,
            learner_id=learner_id,
            source_attempt_event_seq=10 + index,
            attempt_serial=1,
            window_uuid="01234567-89ab-5def-8123-456789abcdef",
            base_version=1,
            local_step=2,
            c_steps=2,
            c_tokens=4,
            weight_f64_bits=struct.pack(">d", 8.0).hex(),
            payload_sha256="b" * 64,
            push=artifact,
            richardson=artifact,
        )

    first = CommittedBoundary(
        capture_session_uuid="session",
        commit_event_seq=20,
        fragment_id=1,
        request_global_step=3,
        committed_fragment_version=3,
        broadcast_payload_sha256="c" * 64,
        responders=(responder(0, 9), responder(1, 2)),
    )
    second = CommittedBoundary(
        capture_session_uuid="session",
        commit_event_seq=21,
        fragment_id=0,
        request_global_step=4,
        committed_fragment_version=4,
        broadcast_payload_sha256="d" * 64,
        responders=(responder(0, 8), responder(1, 1)),
    )
    index = CommittedBoundaryIndex(
        source_transcript=ArtifactRef("transcript.jsonl", "e" * 64),
        capture_session_uuid="session",
        layout_sha256="f" * 64,
        source_tree_manifest_sha256="1" * 64,
        expected=Expectations((0,), 1, 2, 1, 1, 1, 1),
        boundaries=(first, second),
    )

    rows = json.loads(index.canonical_bytes())["boundaries"]
    assert [(row["commit_event_seq"], row["fragment_id"]) for row in rows] == [
        (20, 1),
        (21, 0),
    ]
    assert [row["learner_id"] for row in rows[0]["responders"]] == [9, 2]


def test_background_writer_campaign_requires_drained_complete_manifests(tmp_path):
    arm_dir, expectations = _campaign(tmp_path, background_writer=True)

    summary, _tree = validate_campaign(arm_dir, expectations)

    assert summary["status"] == "PASS"
    for learner in summary["learners"]:
        manifest = json.loads(
            (arm_dir / learner["manifest_path"]).read_text(encoding="utf-8")
        )
        writer = manifest["background_writer"]
        assert writer["state"] == "closed"
        assert writer["accepted_items"] == writer["completed_items"]
        assert writer["completed_items"] == len(manifest["artifacts"])
        assert writer["abandoned_items"] == 0
        assert writer["worker_alive"] is False


def test_strict_writer_rejects_capacity_or_active_schedule_drops(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)
    capture_dir = arm_dir / "optimizer_state_capture_learner_0"
    _mutate_manifest(
        capture_dir,
        lambda manifest: manifest["counters"]["drop_counts"].update(
            richardson_window_disk_byte_limit=1
        ),
    )

    with pytest.raises(ValidationError, match="strict writer contract"):
        validate_campaign(arm_dir, replace(expectations, strict_writer=True))


def test_strict_writer_allows_only_declared_shutdown_tail_drops(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)
    capture_dir = arm_dir / "optimizer_state_capture_learner_0"
    _mutate_manifest(
        capture_dir,
        lambda manifest: manifest["counters"]["drop_counts"].update(
            hmc_incomplete_at_close=1,
            midpoint_incomplete_at_close=1,
            window_unpushed_at_close=1,
        ),
    )

    summary, _tree = validate_campaign(
        arm_dir, replace(expectations, strict_writer=True)
    )

    assert summary["status"] == "PASS"


def test_manifest_sidecar_and_exact_file_set_fail_closed(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)
    capture_dir = arm_dir / "optimizer_state_capture_learner_0"
    sidecar = capture_dir / "manifest.json.sha256"
    sidecar.write_text(f"{'0' * 64}  manifest.json\n", encoding="ascii")
    with pytest.raises(ValidationError, match="checksum sidecar"):
        validate_campaign(arm_dir, expectations)

    _write_manifest(
        capture_dir,
        json.loads((capture_dir / "manifest.json").read_text()),
    )
    (capture_dir / ".orphan.pt.tmp-123").write_bytes(b"partial")
    with pytest.raises(ValidationError, match="temporary file"):
        validate_campaign(arm_dir, expectations)


def test_nonfinite_tensor_is_rejected_even_when_all_hashes_are_resealed(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)
    capture_dir = arm_dir / "optimizer_state_capture_learner_0"

    def inject_nan(envelope):
        envelope["payload"]["anchor"]["parameters_f32"][0] = float("nan")

    _mutate_artifact(capture_dir, "richardson_window", inject_nan)
    with pytest.raises(ValidationError, match="non-finite tensor"):
        validate_campaign(arm_dir, expectations)


def test_richardson_h_counters_and_history_are_checked_after_resealing(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)
    capture_dir = arm_dir / "optimizer_state_capture_learner_0"

    def break_midpoint(envelope):
        envelope["metadata"]["accepted_midpoint_steps"] = 0

    _mutate_artifact(capture_dir, "richardson_window", break_midpoint)
    with pytest.raises(ValidationError, match="accepted H/2,H counters"):
        validate_campaign(arm_dir, expectations)


def test_push_lifecycle_attempt_mismatch_is_rejected(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)
    capture_dir = arm_dir / "optimizer_state_capture_learner_1"
    _mutate_manifest(
        capture_dir,
        lambda manifest: manifest["window_lifecycles"][0].update(push_attempts=3),
    )
    with pytest.raises(ValidationError, match="push artifact count"):
        validate_campaign(arm_dir, expectations)


def test_transcript_rejects_forged_received_payload_digest(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)

    def forge(rows):
        attempt = next(row for row in rows if row["schema"] == "syncer_push_attempt_v1")
        attempt["received_payload_sha256"] = "0" * 64

    _mutate_transcript(arm_dir, forge)
    with pytest.raises(ValidationError, match="received payload digest"):
        validate_campaign(arm_dir, expectations)


def test_transcript_rejects_commit_with_missing_responder(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)

    def remove_responder(rows):
        commit = next(row for row in rows if row["schema"] == "syncer_round_commit_v1")
        commit["responders"].pop()
        commit["responder_count"] = len(commit["responders"])

    _mutate_transcript(arm_dir, remove_responder)
    with pytest.raises(ValidationError, match="commit responders"):
        validate_campaign(arm_dir, expectations)


def test_transcript_rejects_duplicate_learner_attempt_serial(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)

    def duplicate_serial(rows):
        attempts = [row for row in rows if row["schema"] == "syncer_push_attempt_v1"]
        attempts[1]["attempt_serial"] = attempts[0]["attempt_serial"]

    _mutate_transcript(arm_dir, duplicate_serial)
    with pytest.raises(
        ValidationError, match="duplicate primary learner attempt_serial"
    ):
        validate_campaign(arm_dir, expectations)


def test_transcript_rejects_nonmonotone_learner_attempt_serial(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)

    def reverse_serials(rows):
        attempts = [
            row
            for row in rows
            if row["schema"] == "syncer_push_attempt_v1" and row["learner_id"] == 0
        ]
        attempts[0]["attempt_serial"], attempts[1]["attempt_serial"] = (
            attempts[1]["attempt_serial"],
            attempts[0]["attempt_serial"],
        )

    _mutate_transcript(arm_dir, reverse_serials)
    with pytest.raises(ValidationError, match="not monotone contiguous"):
        validate_campaign(arm_dir, expectations)


def test_transcript_rejects_noncontiguous_event_sequence(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)

    def duplicate_event_seq(rows):
        rows[2]["event_seq"] = rows[1]["event_seq"]

    _mutate_transcript(arm_dir, duplicate_event_seq)
    with pytest.raises(ValidationError, match="event_seq.*not contiguous"):
        validate_campaign(arm_dir, expectations)


def test_normalized_join_rejects_boolean_integer_aliases(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)

    def use_boolean_serial(rows):
        commit = next(row for row in rows if row["schema"] == "syncer_round_commit_v1")
        responder = commit["responders"][0]
        source = next(
            row
            for row in rows
            if row["event_seq"] == responder["source_attempt_event_seq"]
        )
        source["attempt_serial"] = True
        responder["attempt_serial"] = True

    _mutate_transcript(arm_dir, use_boolean_serial)
    with pytest.raises(ValidationError, match="attempt_serial"):
        extract_committed_boundary_index(arm_dir, expectations)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("not-a-uuid", "canonical UUID"),
        ("2F1F3A65-B6A9-5AF6-B44D-BBDBA686402D", "lowercase canonical UUID"),
    ],
)
def test_push_artifact_rejects_malformed_or_noncanonical_window_uuid(
    tmp_path, replacement, message
):
    arm_dir, expectations = _campaign(tmp_path)
    capture_dir = arm_dir / "optimizer_state_capture_learner_0"

    def replace_uuid(envelope):
        envelope["metadata"]["window_uuid"] = replacement

    _mutate_artifact(capture_dir, "push_candidate", replace_uuid)
    with pytest.raises(ValidationError, match=message):
        validate_campaign(arm_dir, expectations)


def test_cli_failure_replaces_pass_summary_and_removes_stale_tree(tmp_path):
    arm_dir, expectations = _campaign(tmp_path)
    validate_and_write(arm_dir, expectations)
    tree = arm_dir / "optimizer_state_capture_artifacts.sha256"
    boundary_index = arm_dir / "optimizer_state_capture_committed_boundaries.json"
    assert tree.exists()
    assert boundary_index.exists()
    assert boundary_index.with_suffix(".json.sha256").exists()
    (arm_dir / "optimizer_state_capture_learner_0" / ".late.tmp-9").write_bytes(
        b"partial"
    )

    assert main(_cli(arm_dir)) == 1
    assert not tree.exists()
    assert not boundary_index.exists()
    assert not boundary_index.with_suffix(".json.sha256").exists()
    validation = arm_dir / "optimizer_state_capture_validation.json"
    failure = json.loads(validation.read_text())
    assert failure["status"] == "FAIL"
    assert failure["errors"]
    assert validation.with_suffix(".json.sha256").read_text() == (
        f"{_sha256(validation)}  {validation.name}\n"
    )
