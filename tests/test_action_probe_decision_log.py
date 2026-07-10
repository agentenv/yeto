"""Focused coverage for successful action-probe decision audit records."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

import torch

from yeto.action_probe import (
    ACTION_NAMES,
    PROTOCOL,
    SelectionConfig,
    build_evaluate_frame,
    decode_frame,
    encode_frame,
)
from yeto.action_probe_server import (
    ACTION_PROBE_DECISION_PREFIX,
    ActionProbeEngine,
)


MANIFEST_DIGEST = "a" * 64
TENSOR_DIGEST = "b" * 64
PROBE_CONFIG_DIGEST = "c" * 64
LAYOUT_DIGEST = "d" * 64


def _evaluate_frame(*, request_id: str = "audit-request"):
    current_state = {
        "adapter": torch.tensor([123456.75], dtype=torch.float32),
        "other": torch.tensor([-987654.25], dtype=torch.float32),
    }
    trials = {
        action: torch.tensor([float(index + 101)], dtype=torch.float32)
        for index, action in enumerate(ACTION_NAMES)
    }
    return decode_frame(
        build_evaluate_frame(
            request_id=request_id,
            run_uuid="audit-run",
            step=17,
            fragment_id=0,
            base_version=11,
            state_epoch=13,
            fragment_versions=[11, 12],
            layout_hash=LAYOUT_DIGEST,
            anchor_manifest_sha256=MANIFEST_DIGEST,
            probe_config_sha256=PROBE_CONFIG_DIGEST,
            current_state=current_state,
            fragment_names=["adapter"],
            trials=trials,
        )
    )


class DecisionBackend:
    anchor_manifest_sha256 = MANIFEST_DIGEST
    anchor_tensors_sha256 = TENSOR_DIGEST
    probe_config_sha256 = PROBE_CONFIG_DIGEST
    layout_hash = LAYOUT_DIGEST

    def __init__(
        self,
        *,
        values: dict[str, list[float]] | None = None,
        fail: bool = False,
    ):
        self.calls = 0
        self.fail = fail
        self.values = values or {
            "A0": [1.0] * 8,
            "A1": [0.999] * 8,
            "A2": [1.001] * 8,
            "A3": [0.9995] * 8,
            "A4": [1.0] * 8,
        }

    def evaluate(self, frame, request):
        del frame
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic decision-log failure")
        return {
            "actions": {
                action: {
                    "panel_losses": self.values[action],
                    "trial_sha256": request.action_digests[action],
                    "eval_ms": float(index + 1),
                    "total_ms": float(index + 6),
                }
                for index, action in enumerate(ACTION_NAMES)
            },
            "paired_baseline_losses": {
                action: [1.0] * 8 for action in ACTION_NAMES[1:]
            },
            "workers": [
                {
                    "gpu_id": 3,
                    "actions": ["A0", "A2", "A4"],
                    "restore_ms": 0.4,
                    "total_ms": 8.0,
                },
                {
                    "gpu_id": 1,
                    "actions": ["A0", "A1", "A3"],
                    "restore_ms": 0.2,
                    "total_ms": 7.0,
                },
            ],
            "dispatch_total_ms": 12.5,
            "state_sha256": request.current_state_digest,
            "anchor_manifest_sha256": self.anchor_manifest_sha256,
            "anchor_tensors_sha256": self.anchor_tensors_sha256,
            "probe_config_sha256": self.probe_config_sha256,
        }


def _decision_records(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == "action-probe"
        and record.getMessage().startswith(ACTION_PROBE_DECISION_PREFIX)
    ]


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_nested_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_nested_keys(item))
        return keys
    return set()


def test_success_log_is_one_compact_canonical_line_with_complete_evidence(caplog):
    values = {action: [1.0] * 8 for action in ACTION_NAMES}
    backend = DecisionBackend(values=values)
    selection = SelectionConfig(
        min_gain=0.0004,
        lcb_z=1.5,
        min_win_rate=0.625,
        min_panels=8,
    )
    engine = ActionProbeEngine(backend, selection=selection)
    caplog.set_level(logging.INFO, logger="action-probe")

    response = engine.handle(_evaluate_frame())

    assert response["ok"] is True
    assert response["selected_action"] == "A0"
    assert response["fallback_reason"] == "no_action_passed"
    records = _decision_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    message = records[0].getMessage()
    assert message.splitlines() == [message]

    encoded = message.removeprefix(ACTION_PROBE_DECISION_PREFIX)
    evidence = json.loads(encoded)
    assert encoded == json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    assert set(evidence) == set(response) | {"selection_config"}
    assert all(evidence[key] == value for key, value in response.items())
    assert evidence["request_id"] == "audit-request"
    assert evidence["run_uuid"] == "audit-run"
    assert evidence["step"] == 17
    assert evidence["fragment_id"] == 0
    assert evidence["action_family"] == "leave_one_out"
    assert evidence["selected_action_metadata"] == response[
        "selected_action_metadata"
    ]
    assert evidence["fallback_reason"] == "no_action_passed"
    assert evidence["selection_config"] == asdict(selection)
    assert evidence["selection"] == response["selection"]
    assert len(evidence["selection"]["statistics"]) == 4
    assert evidence["losses_by_action"] == values
    assert evidence["paired_baseline_losses"] == {
        action: [1.0] * 8 for action in ACTION_NAMES[1:]
    }
    assert evidence["workers"] == response["workers"]
    assert evidence["digests"] == response["digests"]
    assert evidence["timings_ms"] == response["timings_ms"]
    assert {"payload", "current_state", "trials"}.isdisjoint(
        _nested_keys(evidence)
    )
    assert "123456.75" not in encoded
    assert "-987654.25" not in encoded


def test_exact_retry_logs_once_for_each_successful_response(caplog):
    backend = DecisionBackend()
    engine = ActionProbeEngine(backend, retry_cache_size=2)
    frame = _evaluate_frame(request_id="audit-retry")
    caplog.set_level(logging.INFO, logger="action-probe")

    first = engine.handle(frame)
    retry = engine.handle(frame)

    assert first["ok"] is True and first["cache_hit"] is False
    assert retry["ok"] is True and retry["cache_hit"] is True
    assert backend.calls == 1
    records = _decision_records(caplog)
    assert len(records) == 2
    evidence = [
        json.loads(record.getMessage().removeprefix(ACTION_PROBE_DECISION_PREFIX))
        for record in records
    ]
    assert [item["cache_hit"] for item in evidence] == [False, True]
    assert "retry_lookup_ms" not in evidence[0]
    assert evidence[1]["retry_lookup_ms"] >= 0


def test_malformed_and_failed_requests_do_not_emit_success_decisions(caplog):
    backend = DecisionBackend(fail=True)
    engine = ActionProbeEngine(backend, retry_cache_size=2)
    caplog.set_level(logging.INFO, logger="action-probe")

    malformed = decode_frame(
        encode_frame(
            {
                "protocol": PROTOCOL,
                "type": "evaluate",
                "request_id": "malformed-audit-request",
            }
        )
    )
    malformed_response = engine.handle(malformed)
    failed_frame = _evaluate_frame(request_id="failed-audit-request")
    failed_response = engine.handle(failed_frame)
    failed_retry = engine.handle(failed_frame)

    assert malformed_response["ok"] is False
    assert malformed_response["fallback_reason"] == "protocol_error"
    assert failed_response["ok"] is False
    assert failed_response["fallback_reason"] == "evaluation_error"
    assert failed_retry["ok"] is False
    assert failed_retry["cache_hit"] is True
    assert backend.calls == 1
    assert _decision_records(caplog) == []
