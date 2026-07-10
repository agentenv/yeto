"""Focused tests for the online action-probe protocol and Python service."""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from yeto.action_probe import (
    ACTION_NAMES,
    LEAVE_ONE_OUT_ACTION_FAMILY,
    PROTOCOL,
    STEP_SCALE_ACTION_FAMILY,
    SUPPORTED_ACTION_FAMILIES,
    ActionProbeReplica,
    ManifestError,
    ProtocolError,
    SelectionConfig,
    build_anchor_panels,
    build_evaluate_frame,
    canonical_anchor_hash,
    decode_frame,
    encode_frame,
    load_anchor_manifest,
    parse_evaluate_request,
    recv_frame,
    send_frame,
    select_paired_lcb,
)
from yeto.action_probe_server import (
    ActionProbeEngine,
    ActionProbeTCPService,
    _parse_listen,
)
from yeto.data import build_packed_dataset


MANIFEST_DIGEST = "a" * 64
PROBE_CONFIG_DIGEST = "c" * 64
LAYOUT_DIGEST = "d" * 64


def _state_and_trials(scale: float = 1.0):
    state = {
        "adapter": torch.tensor([0.0], dtype=torch.float32),
        "other": torch.tensor([0.25, -0.25], dtype=torch.float32),
    }
    trials = {
        "A0": torch.tensor([0.0]),
        "A1": torch.tensor([1.0 * scale]),
        "A2": torch.tensor([-1.0]),
        "A3": torch.tensor([0.5]),
        "A4": torch.tensor([-0.5]),
    }
    return state, trials


def _valid_action_metadata():
    return {
        action: {
            "eligible": True,
            "omitted_responder_id": None if action == "A0" else index - 1,
            "selected_mass": 1.0 if action == "A0" else 0.75,
            "norm_multiplier": 1.0,
            "step_norm_ratio": 1.0,
            "ineligible_reason": None,
        }
        for index, action in enumerate(ACTION_NAMES)
    }


def _valid_step_scale_metadata(scales=(0.875, 0.5, 0.75, 1.0, 1.25)):
    return {
        action: {
            "eligible": True,
            "omitted_responder_id": None,
            "selected_mass": 1.0,
            "norm_multiplier": scale,
            "step_norm_ratio": scale,
            "step_scale": scale,
            "ineligible_reason": None,
        }
        for action, scale in zip(ACTION_NAMES, scales)
    }


def _request_bytes(
    *,
    request_id: str = "req-1",
    scale: float = 1.0,
    action_metadata=None,
    action_family: str = LEAVE_ONE_OUT_ACTION_FAMILY,
) -> bytes:
    state, trials = _state_and_trials(scale)
    if action_family == STEP_SCALE_ACTION_FAMILY and action_metadata is None:
        action_metadata = _valid_step_scale_metadata()
    return build_evaluate_frame(
        request_id=request_id,
        run_uuid="run-1",
        step=7,
        fragment_id=0,
        base_version=3,
        state_epoch=9,
        fragment_versions=[3, 4],
        layout_hash=LAYOUT_DIGEST,
        anchor_manifest_sha256=MANIFEST_DIGEST,
        probe_config_sha256=PROBE_CONFIG_DIGEST,
        current_state=state,
        fragment_names=["adapter"],
        trials=trials,
        action_metadata=action_metadata,
        action_family=action_family,
    )


def test_frame_and_f32_request_round_trip():
    wire = _request_bytes()
    frame = decode_frame(wire)
    request = parse_evaluate_request(frame)

    assert request.request_id == "req-1"
    assert request.action_family == LEAVE_ONE_OUT_ACTION_FAMILY
    assert request.fragment_versions == (3, 4)
    assert tuple(request.current_state) == ("adapter", "other")
    assert torch.equal(request.current_state["other"], torch.tensor([0.25, -0.25]))
    assert set(request.trials) == set(ACTION_NAMES)
    assert (
        request.action_digests["A1"]
        == hashlib.sha256(
            torch.tensor([1.0]).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    )
    assert len(request.request_digest) == 64


def test_leave_one_out_family_defaults_for_legacy_v1_frames():
    frame = decode_frame(_request_bytes())
    assert frame.header["fragment"]["action_family"] == LEAVE_ONE_OUT_ACTION_FAMILY
    del frame.header["fragment"]["action_family"]

    legacy = decode_frame(encode_frame(frame.header, frame.payload))
    request = parse_evaluate_request(legacy)

    assert request.action_family == LEAVE_ONE_OUT_ACTION_FAMILY
    assert request.action_metadata == _valid_action_metadata()


def test_step_scale_family_round_trip_allows_nonunit_a0_fallback():
    request = parse_evaluate_request(
        decode_frame(_request_bytes(action_family=STEP_SCALE_ACTION_FAMILY))
    )

    assert request.action_family == STEP_SCALE_ACTION_FAMILY
    assert request.action_metadata["A0"]["step_scale"] == 0.875
    assert {
        request.action_metadata[action]["step_scale"] for action in ACTION_NAMES
    } == {0.5, 0.75, 0.875, 1.0, 1.25}
    assert all(
        request.action_metadata[action]["omitted_responder_id"] is None
        and request.action_metadata[action]["selected_mass"] == 1.0
        for action in ACTION_NAMES
    )


def test_frame_rejects_truncation_trailing_data_and_payload_corruption():
    wire = _request_bytes()
    with pytest.raises(ProtocolError, match="expected exactly"):
        decode_frame(wire + b"trailing")
    with pytest.raises(ProtocolError):
        decode_frame(wire[:-1])

    corrupted = bytearray(wire)
    corrupted[-1] ^= 0xFF
    with pytest.raises(ProtocolError, match="SHA-256 mismatch"):
        parse_evaluate_request(decode_frame(bytes(corrupted)))


def test_socket_receiver_handles_fragmented_length_prefixed_frame():
    wire = _request_bytes()
    sender, receiver = socket.socketpair()

    def write_fragments():
        try:
            for offset in range(0, len(wire), 17):
                sender.sendall(wire[offset : offset + 17])
        finally:
            sender.close()

    thread = threading.Thread(target=write_fragments)
    thread.start()
    try:
        frame = recv_frame(receiver)
    finally:
        receiver.close()
        thread.join(timeout=5)
    assert parse_evaluate_request(frame).request_id == "req-1"


def test_request_requires_exact_five_action_set_and_finite_tensors():
    state, trials = _state_and_trials()
    trials.pop("A4")
    with pytest.raises(ProtocolError, match="exactly"):
        build_evaluate_frame(
            request_id="x",
            run_uuid="run",
            step=0,
            fragment_id=0,
            base_version=0,
            state_epoch=0,
            fragment_versions=[0],
            layout_hash=LAYOUT_DIGEST,
            anchor_manifest_sha256=MANIFEST_DIGEST,
            probe_config_sha256=PROBE_CONFIG_DIGEST,
            current_state=state,
            fragment_names=["adapter"],
            trials=trials,
        )


def test_request_rejects_unsafe_eligible_action_and_version_mismatch():
    metadata = _valid_action_metadata()
    metadata["A1"]["selected_mass"] = 0.69
    with pytest.raises(ProtocolError, match="minimum selected mass"):
        parse_evaluate_request(decode_frame(_request_bytes(action_metadata=metadata)))

    frame = decode_frame(_request_bytes())
    frame.header["base_version"] = 2
    with pytest.raises(ProtocolError, match="base_version"):
        parse_evaluate_request(frame)

    state, trials = _state_and_trials()
    trials["A4"] = torch.tensor([float("nan")])
    with pytest.raises(ProtocolError, match="NaN or Inf"):
        build_evaluate_frame(
            request_id="x",
            run_uuid="run",
            step=0,
            fragment_id=0,
            base_version=0,
            state_epoch=0,
            fragment_versions=[0],
            layout_hash=LAYOUT_DIGEST,
            anchor_manifest_sha256=MANIFEST_DIGEST,
            probe_config_sha256=PROBE_CONFIG_DIGEST,
            current_state=state,
            fragment_names=["adapter"],
            trials=trials,
        )


def test_step_scale_rejects_duplicate_nonpositive_and_mixed_metadata():
    duplicate = _valid_step_scale_metadata()
    duplicate["A4"]["step_scale"] = duplicate["A3"]["step_scale"]
    with pytest.raises(ProtocolError, match="unique step scales"):
        parse_evaluate_request(
            decode_frame(
                _request_bytes(
                    action_family=STEP_SCALE_ACTION_FAMILY,
                    action_metadata=duplicate,
                )
            )
        )

    nonpositive = _valid_step_scale_metadata()
    nonpositive["A2"]["step_scale"] = 0.0
    with pytest.raises(ProtocolError, match="step_scale must be positive"):
        parse_evaluate_request(
            decode_frame(
                _request_bytes(
                    action_family=STEP_SCALE_ACTION_FAMILY,
                    action_metadata=nonpositive,
                )
            )
        )

    nonfinite = decode_frame(_request_bytes(action_family=STEP_SCALE_ACTION_FAMILY))
    nonfinite.header["fragment"]["actions"][1]["step_scale"] = float("inf")
    with pytest.raises(ProtocolError, match="finite number"):
        parse_evaluate_request(nonfinite)

    omitted = _valid_step_scale_metadata()
    omitted["A1"]["omitted_responder_id"] = 7
    with pytest.raises(ProtocolError, match="must not omit"):
        parse_evaluate_request(
            decode_frame(
                _request_bytes(
                    action_family=STEP_SCALE_ACTION_FAMILY,
                    action_metadata=omitted,
                )
            )
        )

    partial_mass = _valid_step_scale_metadata()
    partial_mass["A3"]["selected_mass"] = 0.75
    with pytest.raises(ProtocolError, match="exactly 1"):
        parse_evaluate_request(
            decode_frame(
                _request_bytes(
                    action_family=STEP_SCALE_ACTION_FAMILY,
                    action_metadata=partial_mass,
                )
            )
        )

    loo_with_scale = _valid_action_metadata()
    loo_with_scale["A2"]["step_scale"] = 1.25
    with pytest.raises(ProtocolError, match="invalid for leave_one_out"):
        parse_evaluate_request(
            decode_frame(_request_bytes(action_metadata=loo_with_scale))
        )


def test_step_scale_requires_exact_action_digest():
    frame = decode_frame(_request_bytes(action_family=STEP_SCALE_ACTION_FAMILY))
    frame.header["fragment"]["actions"][2]["sha256"] = "0" * 64

    with pytest.raises(ProtocolError, match="SHA-256 mismatch"):
        parse_evaluate_request(frame)


def test_paired_lcb_selects_deterministically_and_falls_back():
    losses = {
        "A0": [1.0] * 8,
        "A1": [0.999] * 8,
        "A2": [1.001] * 8,
        "A3": [0.9995] * 8,
        "A4": [1.0] * 8,
    }
    result = select_paired_lcb(losses)
    assert result.selected_action == "A1"
    assert result.fallback_reason is None
    assert next(stat for stat in result.statistics if stat.action == "A1").eligible

    noisy = dict(losses)
    noisy["A1"] = [0.999, 1.001] * 4
    result = select_paired_lcb(noisy)
    assert result.selected_action == "A3"

    malformed = select_paired_lcb({"A0": [1.0, float("nan")]})
    assert malformed.selected_action == "A0"
    assert malformed.fallback_reason == "invalid_losses"

    negative = {action: [1.0] * 8 for action in ACTION_NAMES}
    negative["A1"] = [-1.0] * 8
    assert select_paired_lcb(negative).fallback_reason == "invalid_losses"

    strict = select_paired_lcb(
        losses,
        SelectionConfig(min_gain=0.01, lcb_z=2.365, min_win_rate=0.75),
    )
    assert strict.selected_action == "A0"
    assert strict.fallback_reason == "no_action_passed"


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {
            "input_ids": [(sum(word.encode("utf-8")) % 29) + 3 for word in text.split()]
        }


def _canonical_jsonl(rows):
    return b"".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _anchor_files(tmp_path: Path, *, actual_overlap: bool = False, rows=None):
    if rows is None:
        rows = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": " ".join(f"u{i}_{j}" for j in range(12)),
                    },
                    {
                        "role": "assistant",
                        "content": " ".join(f"a{i}_{j}" for j in range(12)),
                    },
                ]
            }
            for i in range(4)
        ]
    data = tmp_path / "anchor.jsonl"
    payload = _canonical_jsonl(rows)
    data.write_bytes(payload)
    exclusion_rows = (
        [rows[0]]
        if actual_overlap
        else [{"messages": [{"role": "user", "content": "excluded example"}]}]
    )
    exclusion = tmp_path / "excluded.jsonl"
    exclusion_payload = _canonical_jsonl(exclusion_rows)
    exclusion.write_bytes(exclusion_payload)
    manifest = {
        "schema": "disjoint_hf_holdout_v1",
        "canonicalization": "yeto-messages-tools-v1",
        "output_path": str(data),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "selected_count": len(rows),
        "selected_canonical_hashes": [canonical_anchor_hash(row) for row in rows],
        "overlap_count": 0,
        "verified_zero_overlap": True,
        "unique_excluded_canonical_count": len(
            {canonical_anchor_hash(row) for row in exclusion_rows}
        ),
        "exclusions": [
            {
                "path": str(exclusion),
                "sha256": hashlib.sha256(exclusion_payload).hexdigest(),
                "canonical_sha256": hashlib.sha256(exclusion_payload).hexdigest(),
            }
        ],
    }
    manifest_path = tmp_path / "anchor.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path, data, rows


def test_anchor_manifest_and_static_panels_are_verified_and_deterministic(tmp_path):
    manifest_path, data, _ = _anchor_files(tmp_path)
    manifest = load_anchor_manifest(manifest_path)
    panels_a, digest_a = build_anchor_panels(
        manifest,
        FakeTokenizer(),
        seq_len=8,
        panels=3,
        blocks_per_panel=2,
    )
    panels_b, digest_b = build_anchor_panels(
        manifest,
        FakeTokenizer(),
        seq_len=8,
        panels=3,
        blocks_per_panel=2,
    )
    assert digest_a == digest_b
    assert len(panels_a) == 3
    assert all(
        ids.shape == (2, 8) and weights.shape == (2, 8) for ids, weights in panels_a
    )
    assert all(torch.equal(a[0], b[0]) for a, b in zip(panels_a, panels_b))

    data.write_text(data.read_text() + "{}\n")
    with pytest.raises(ManifestError, match="SHA-256 mismatch"):
        load_anchor_manifest(manifest_path)


def test_anchor_panels_skip_prompt_only_packed_blocks(tmp_path):
    rows = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": " ".join(f"prompt_{row}_{i}" for i in range(15)),
                },
                {
                    "role": "assistant",
                    "content": " ".join(f"target_{row}_{i}" for i in range(6)),
                },
            ]
        }
        for row in range(2)
    ]
    manifest_path, _, _ = _anchor_files(tmp_path, rows=rows)
    manifest = load_anchor_manifest(manifest_path)
    tokenizer = FakeTokenizer()

    packed = build_packed_dataset(
        [rows[0]],
        tokenizer,
        learner_id=0,
        num_learners=1,
        seq_len=8,
        max_rows=1,
        train_on="assistant",
    )
    assert float(packed.weights[0, 1:].sum().item()) == 0.0
    first_target_block = next(
        index
        for index in range(len(packed))
        if float(packed.weights[index, 1:].sum().item()) > 0.0
    )

    panels, _ = build_anchor_panels(
        manifest,
        tokenizer,
        seq_len=8,
        panels=2,
        blocks_per_panel=1,
    )

    assert torch.equal(panels[0][0][0], packed.blocks[first_target_block])
    assert all(float(weights[:, 1:].sum().item()) > 0.0 for _, weights in panels)


def test_anchor_manifest_rejects_overlap(tmp_path):
    manifest_path, _, _ = _anchor_files(tmp_path, actual_overlap=True)
    with pytest.raises(ManifestError, match="overlaps"):
        load_anchor_manifest(manifest_path)


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Parameter(torch.tensor([7.0], dtype=torch.float32))
        self.other = nn.Parameter(torch.tensor([8.0, 9.0], dtype=torch.float32))
        self.fail = False

    def forward(self, *, input_ids, use_cache=False):
        assert use_cache is False
        if self.fail:
            raise RuntimeError("synthetic forward failure")
        batch, length = input_ids.shape
        logits = torch.zeros(
            batch, length, 8, dtype=torch.float32, device=input_ids.device
        )
        logits[..., 0] = self.adapter + self.other.sum() * 0.01
        return SimpleNamespace(logits=logits)


def test_replica_evaluates_actions_and_restores_complete_state_exactly():
    model = TinyCausalLM()
    panels = (
        (torch.zeros((1, 4), dtype=torch.long), torch.ones((1, 4))),
        (torch.zeros((1, 4), dtype=torch.long), torch.ones((1, 4))),
    )
    replica = ActionProbeReplica(
        model,
        panels,
        anchor_manifest_sha256=MANIFEST_DIGEST,
        anchor_tensors_sha256="b" * 64,
        probe_config_sha256=PROBE_CONFIG_DIGEST,
        layout_hash=LAYOUT_DIGEST,
        fragment_layout={0: ["adapter"], 1: ["other"]},
        device="cpu",
    )
    request = parse_evaluate_request(decode_frame(_request_bytes()))
    result = replica.evaluate(request, ACTION_NAMES)

    assert set(result["actions"]) == set(ACTION_NAMES)
    assert (
        result["actions"]["A1"]["panel_losses"][0]
        < result["actions"]["A0"]["panel_losses"][0]
    )
    assert result["state_restored"] is True
    assert torch.equal(model.adapter.detach(), torch.tensor([0.0]))
    assert torch.equal(model.other.detach(), torch.tensor([0.25, -0.25]))


def test_replica_restores_complete_state_when_forward_raises():
    model = TinyCausalLM()
    panels = (
        (torch.zeros((1, 4), dtype=torch.long), torch.ones((1, 4))),
        (torch.zeros((1, 4), dtype=torch.long), torch.ones((1, 4))),
    )
    replica = ActionProbeReplica(
        model,
        panels,
        anchor_manifest_sha256=MANIFEST_DIGEST,
        anchor_tensors_sha256="b" * 64,
        probe_config_sha256=PROBE_CONFIG_DIGEST,
        layout_hash=LAYOUT_DIGEST,
        fragment_layout={0: ["adapter"], 1: ["other"]},
        device="cpu",
    )
    request = parse_evaluate_request(decode_frame(_request_bytes()))
    model.fail = True
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        replica.evaluate(request, ["A1"])
    assert torch.equal(model.adapter.detach(), torch.tensor([0.0]))
    assert torch.equal(model.other.detach(), torch.tensor([0.25, -0.25]))


def test_replica_reports_restore_failure_over_forward_failure(monkeypatch):
    model = TinyCausalLM()
    panels = (
        (torch.zeros((1, 4), dtype=torch.long), torch.ones((1, 4))),
        (torch.zeros((1, 4), dtype=torch.long), torch.ones((1, 4))),
    )
    replica = ActionProbeReplica(
        model,
        panels,
        anchor_manifest_sha256=MANIFEST_DIGEST,
        anchor_tensors_sha256="b" * 64,
        probe_config_sha256=PROBE_CONFIG_DIGEST,
        layout_hash=LAYOUT_DIGEST,
        fragment_layout={0: ["adapter"], 1: ["other"]},
        device="cpu",
    )
    request = parse_evaluate_request(decode_frame(_request_bytes()))
    original_apply = replica._apply_state
    calls = 0

    def fail_second_apply(state):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic restore failure")
        original_apply(state)

    monkeypatch.setattr(replica, "_apply_state", fail_second_apply)
    model.fail = True
    with pytest.raises(Exception, match="failed to restore current state"):
        replica.evaluate(request, ["A1"])


class FakeBackend:
    anchor_manifest_sha256 = MANIFEST_DIGEST
    anchor_tensors_sha256 = "b" * 64
    probe_config_sha256 = PROBE_CONFIG_DIGEST
    layout_hash = LAYOUT_DIGEST

    def __init__(self, *, fail=False):
        self.calls = 0
        self.fail = fail

    def describe(self):
        return {"kind": "fake"}

    def evaluate(self, frame, request):
        del frame
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic evaluator failure")
        values = {
            "A0": [1.0] * 8,
            "A1": [0.999] * 8,
            "A2": [1.001] * 8,
            "A3": [0.9995] * 8,
            "A4": [1.0] * 8,
        }
        return {
            "actions": {
                action: {
                    "panel_losses": losses,
                    "trial_sha256": request.action_digests[action],
                    "eval_ms": 1.0,
                }
                for action, losses in values.items()
            },
            "workers": [],
            "dispatch_total_ms": 5.0,
            "state_sha256": request.current_state_digest,
            "anchor_manifest_sha256": self.anchor_manifest_sha256,
            "anchor_tensors_sha256": self.anchor_tensors_sha256,
            "probe_config_sha256": self.probe_config_sha256,
        }


def test_engine_caches_only_exact_retries_and_rejects_request_id_reuse():
    backend = FakeBackend()
    engine = ActionProbeEngine(backend, retry_cache_size=2)
    first = decode_frame(_request_bytes())
    response = engine.handle(first)
    assert response["ok"] is True
    assert response["selected_action"] == "A1"
    assert response["cache_hit"] is False
    assert backend.calls == 1

    retry = engine.handle(decode_frame(_request_bytes()))
    assert retry["ok"] is True
    assert retry["cache_hit"] is True
    assert backend.calls == 1

    changed = engine.handle(decode_frame(_request_bytes(scale=2.0)))
    assert changed["ok"] is False
    assert changed["fail_closed"] is True
    assert changed["selected_action"] == "A0"
    assert changed["fallback_reason"] == "protocol_error"
    assert backend.calls == 1


def test_engine_never_selects_an_ineligible_action():
    metadata = _valid_action_metadata()
    metadata["A1"].update(
        eligible=False,
        selected_mass=0.5,
        ineligible_reason="selected_mass",
    )
    engine = ActionProbeEngine(FakeBackend())
    response = engine.handle(decode_frame(_request_bytes(action_metadata=metadata)))
    assert response["ok"] is True
    assert response["selected_action"] == "A3"
    assert (
        response["selected_action_sha256"] == response["digests"]["action_sha256"]["A3"]
    )


def test_engine_uses_a0_as_the_step_scale_pairing_and_fallback():
    engine = ActionProbeEngine(FakeBackend())
    response = engine.handle(
        decode_frame(_request_bytes(action_family=STEP_SCALE_ACTION_FAMILY))
    )

    assert response["ok"] is True
    assert response["action_family"] == STEP_SCALE_ACTION_FAMILY
    assert response["selected_action"] == "A1"
    assert response["selected_action_metadata"]["step_scale"] == 0.5

    strict = ActionProbeEngine(
        FakeBackend(),
        selection=SelectionConfig(
            min_gain=0.01,
            lcb_z=2.365,
            min_win_rate=0.75,
        ),
    )
    fallback = strict.handle(
        decode_frame(
            _request_bytes(
                request_id="scalar-fallback",
                action_family=STEP_SCALE_ACTION_FAMILY,
            )
        )
    )
    assert fallback["selected_action"] == "A0"
    assert fallback["selected_action_metadata"]["step_scale"] == 0.875


def test_engine_fails_closed_on_backend_error():
    engine = ActionProbeEngine(FakeBackend(fail=True))
    response = engine.handle(decode_frame(_request_bytes()))
    assert response["ok"] is False
    assert response["action_family"] == LEAVE_ONE_OUT_ACTION_FAMILY
    assert response["selected_action"] == "A0"
    assert response["fail_closed"] is True
    assert "synthetic evaluator failure" in response["error"]

    scalar = engine.handle(
        decode_frame(
            _request_bytes(
                request_id="scalar-error",
                action_family=STEP_SCALE_ACTION_FAMILY,
            )
        )
    )
    assert scalar["ok"] is False
    assert scalar["action_family"] == STEP_SCALE_ACTION_FAMILY


def test_ping_has_no_model_request_and_listen_rejects_non_loopback():
    backend = FakeBackend()
    engine = ActionProbeEngine(backend)
    response = engine.handle(
        decode_frame(encode_frame({"protocol": PROTOCOL, "type": "ping"}))
    )
    assert response["type"] == "pong"
    assert response["supported_action_families"] == list(SUPPORTED_ACTION_FAMILIES)
    assert response["service"]["supported_action_families"] == list(
        SUPPORTED_ACTION_FAMILIES
    )
    assert backend.calls == 0
    assert _parse_listen("127.0.0.1:9999") == ("127.0.0.1", 9999)
    with pytest.raises(Exception, match="loopback"):
        _parse_listen("0.0.0.0:9999")


def test_tcp_service_keeps_connection_persistent_for_ping_and_evaluate():
    backend = FakeBackend()
    engine = ActionProbeEngine(backend)
    service = ActionProbeTCPService(("127.0.0.1", 0), engine, client_timeout_s=5)
    thread = threading.Thread(target=service.serve_forever)
    thread.start()
    deadline = time.monotonic() + 5
    while service._listener is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert service._listener is not None
    address = service._listener.getsockname()[:2]

    with socket.create_connection(address, timeout=5) as client:
        send_frame(client, {"protocol": PROTOCOL, "type": "ping"})
        assert recv_frame(client).header["type"] == "pong"
        client.sendall(_request_bytes(action_family=STEP_SCALE_ACTION_FAMILY))
        response = recv_frame(client).header
        assert response["ok"] is True
        assert response["action_family"] == STEP_SCALE_ACTION_FAMILY
        assert response["selected_action"] == "A1"
        service.stop()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert backend.calls == 1
