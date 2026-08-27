"""Regression tests for live stock Codex app-server schema attestation."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from yeto.rl import (
    CODEX_OPENENV_AGENT_MODULES,
    CODEX_OPENENV_IDENTITY_ENV,
    learner,
)


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
    assert (
        hashlib.sha256(pinned.read_bytes()).digest()
        != hashlib.sha256(generated.read_bytes()).digest()
    )

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


def test_codex_openenv_preflight_attests_pinned_adapter_and_environment(
    tmp_path, monkeypatch
):
    adapter_dir = tmp_path / "examples" / "experimental" / "openenv"
    adapter_dir.mkdir(parents=True)
    for name in CODEX_OPENENV_AGENT_MODULES:
        (adapter_dir / name).write_text("# pinned adapter\n", encoding="utf-8")
    identity = {
        name.removeprefix("YETO_CODEX_OPENENV_").lower(): value
        for name, value in CODEX_OPENENV_IDENTITY_ENV.items()
        if name.endswith("_SHA256")
    }
    adapter = SimpleNamespace(
        __file__=str(adapter_dir / "codex_openenv_agent_function.py"),
        _OPENENV_IDENTITY_ENV=dict(CODEX_OPENENV_IDENTITY_ENV),
        codex_openenv_harness_identity=lambda: identity,
    )
    subprocess_adapter = SimpleNamespace(
        __file__=str(adapter_dir / "codex_openenv_subprocess_agent_function.py"),
        run=lambda: None,
    )

    def import_module(name):
        return {
            "codex_openenv_agent_function": adapter,
            "codex_openenv_subprocess_agent_function": subprocess_adapter,
        }[name]

    monkeypatch.setattr(learner.importlib, "import_module", import_module)
    for name, value in CODEX_OPENENV_IDENTITY_ENV.items():
        monkeypatch.setenv(name, value)

    args = SimpleNamespace(miles_root=str(tmp_path))
    learner._preflight_codex_openenv_adapter(args, "qwen35_08b")

    monkeypatch.delenv("YETO_CODEX_OPENENV_MODEL_REVISION")
    with pytest.raises(ValueError, match="environment drifted"):
        learner._preflight_codex_openenv_adapter(args, "qwen35_08b")
    with pytest.raises(ValueError, match="requires backend profile"):
        learner._preflight_codex_openenv_adapter(args, "qwen35")
