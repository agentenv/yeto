"""Regression tests for live stock Codex app-server schema attestation."""

from __future__ import annotations

import hashlib

import pytest

from yeto.rl import learner


def test_live_codex_schema_accepts_reordered_identical_json(tmp_path):
    pinned = tmp_path / "pinned.json"
    generated = tmp_path / "generated.json"
    pinned.write_text(
        '{"methods":{"initialize":{"params":{"b":2,"a":1}}},"revision":2}',
        encoding="utf-8",
    )
    generated.write_text(
        '{\n  "revision": 2,\n  "methods": {"initialize": {"params": '
        '{"a": 1, "b": 2}}}\n}\n',
        encoding="utf-8",
    )
    assert hashlib.sha256(pinned.read_bytes()).digest() != hashlib.sha256(
        generated.read_bytes()
    ).digest()

    learner._verify_live_codex_app_server_schema(pinned, generated)


def test_live_codex_schema_rejects_semantic_drift(tmp_path):
    pinned = tmp_path / "pinned.json"
    generated = tmp_path / "generated.json"
    pinned.write_text('{"revision":2,"methods":{}}', encoding="utf-8")
    generated.write_text('{"methods":{},"revision":3}', encoding="utf-8")

    with pytest.raises(ValueError, match="schema drifted"):
        learner._verify_live_codex_app_server_schema(pinned, generated)


def test_live_codex_schema_rejects_exact_numeric_drift(tmp_path):
    pinned = tmp_path / "pinned.json"
    generated = tmp_path / "generated.json"
    pinned.write_text('{"revision":9007199254740992.0}', encoding="utf-8")
    generated.write_text('{"revision":9007199254740993.0}', encoding="utf-8")

    with pytest.raises(ValueError, match="schema drifted"):
        learner._verify_live_codex_app_server_schema(pinned, generated)


@pytest.mark.parametrize(
    "malformed",
    [
        '{"revision":',
        '{"revision":2,"revision":2}',
        '{"revision":NaN}',
    ],
)
def test_live_codex_schema_rejects_malformed_json(tmp_path, malformed):
    pinned = tmp_path / "pinned.json"
    generated = tmp_path / "generated.json"
    pinned.write_text('{"revision":2}', encoding="utf-8")
    generated.write_text(malformed, encoding="utf-8")

    with pytest.raises(ValueError, match="schema drifted"):
        learner._verify_live_codex_app_server_schema(pinned, generated)
