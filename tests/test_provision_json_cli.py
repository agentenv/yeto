"""One-request/one-response tests for the yeto-provision-solve JSON CLI."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from yeto.provision import json_cli

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "agentenv_compute_v1"

PLAN_ITEM_ID = "ae-0123456789abcdef0123-000"

FORBIDDEN_MODULES = (
    "torch",
    "transformers",
    "boto3",
    "botocore",
    "sky",
    "yeto.cli",
    "yeto.shape",
    "yeto.shape.ilp",
    "yeto.shape.providers",
    "yeto.launcher",
    "yeto.learner",
    "yeto.protocol",
    "yeto.models",
    "yeto.data",
    "syncer",
)


def fresh_request(launch: dict | None = None) -> dict:
    """A solvable request built from the pinned two-offering fixture."""
    supply = json.loads((FIXTURES / "supply-snapshot.two-offerings.local.json").read_text())
    for offering in supply["offerings"]:
        offering["validUntil"] = "2036-01-01T00:00:00.000Z"
    return {
        "schemaVersion": 1,
        "planItemId": PLAN_ITEM_ID,
        "launch": {"schemaVersion": 1, "nodes": 2, **(launch or {})},
        "supply": supply,
    }


def clean_env() -> dict:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AWS_", "RUNPOD_", "VERDA_"))
    }
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def run_cli(stdin: bytes, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "yeto.provision.json_cli", *args],
        input=stdin,
        capture_output=True,
        cwd=REPO_ROOT,
        env=clean_env(),
        timeout=60,
    )


def parse_single_stdout_json(stdout: bytes) -> dict:
    """stdout must be exactly one JSON document (no NDJSON/event stream)."""
    text = stdout.decode("utf-8")
    assert text.endswith("\n") and text.count("\n") == 1
    return json.loads(text)


# ---------------------------------------------------------------------------
# Typed success (exit 0)
# ---------------------------------------------------------------------------


def test_success_envelope_echoes_plan_item_id_unchanged():
    result = run_cli(json.dumps(fresh_request()).encode())
    assert result.returncode == 0, result.stderr
    envelope = parse_single_stdout_json(result.stdout)
    assert envelope["schemaVersion"] == 1
    assert envelope["ok"] is True
    item = envelope["value"]["items"][0]
    assert item["planItemId"] == PLAN_ITEM_ID
    assert item["offeringId"] == "ae:verda:h100:exact-b"
    assert item["nodes"] == 2
    assert envelope["value"]["catalogSnapshotId"] == "snap-2"


def test_diagnostics_never_reach_stdout():
    result = run_cli(json.dumps(fresh_request()).encode())
    assert result.returncode == 0
    parse_single_stdout_json(result.stdout)  # single bounded JSON document


# ---------------------------------------------------------------------------
# Handled domain errors (typed ok:false, exit 0)
# ---------------------------------------------------------------------------


def test_no_feasible_supply_is_handled_and_exits_zero():
    request = fresh_request(launch={"accelerator": "H200"})
    result = run_cli(json.dumps(request).encode())
    assert result.returncode == 0
    envelope = parse_single_stdout_json(result.stdout)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "no_feasible_supply"
    assert envelope["error"]["retryable"] is False


def test_missing_supply_fails_closed_with_typed_error():
    result = run_cli((FIXTURES / "solve-request.valid.json").read_bytes())
    assert result.returncode == 0
    envelope = parse_single_stdout_json(result.stdout)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "missing_supply"


def test_expired_supply_fails_closed_with_typed_error():
    request = fresh_request()
    for offering in request["supply"]["offerings"]:
        offering["validUntil"] = "2020-01-01T00:00:00.000Z"
    result = run_cli(json.dumps(request).encode())
    assert result.returncode == 0
    envelope = parse_single_stdout_json(result.stdout)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "supply_expired"


def test_malformed_supply_is_a_handled_validation_error():
    request = fresh_request()
    del request["supply"]["offerings"][0]["sshUser"]
    result = run_cli(json.dumps(request).encode())
    assert result.returncode == 0
    envelope = parse_single_stdout_json(result.stdout)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "missing_field"


def test_unknown_request_field_is_a_handled_validation_error():
    request = fresh_request()
    request["orchestrationId"] = "11111111-1111-4111-8111-111111111111"
    result = run_cli(json.dumps(request).encode())
    assert result.returncode == 0
    envelope = parse_single_stdout_json(result.stdout)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "unknown_field"


# ---------------------------------------------------------------------------
# Protocol failures (nonzero exit, stdout untrusted/empty)
# ---------------------------------------------------------------------------


def test_malformed_json_exits_nonzero():
    result = run_cli(b"this is not json")
    assert result.returncode != 0
    assert result.stdout == b""
    assert result.stderr  # diagnostics on stderr only


def test_trailing_second_json_record_exits_nonzero():
    request = json.dumps(fresh_request())
    result = run_cli((request + "\n" + request).encode())
    assert result.returncode != 0
    assert result.stdout == b""


def test_non_object_json_exits_nonzero():
    result = run_cli(b"[1,2,3]")
    assert result.returncode != 0
    assert result.stdout == b""


def test_empty_input_exits_nonzero():
    result = run_cli(b"")
    assert result.returncode != 0
    assert result.stdout == b""


def test_input_byte_limit_enforced():
    request = fresh_request()
    request_bytes = json.dumps(request).encode()
    padding = b" " * (json_cli.MAX_REQUEST_BYTES + 1 - len(request_bytes))
    result = run_cli(request_bytes + padding)
    assert result.returncode != 0
    assert result.stdout == b""
    assert b"exceeds" in result.stderr


def test_arguments_are_a_usage_failure():
    result = run_cli(json.dumps(fresh_request()).encode(), "--apply-json")
    assert result.returncode != 0
    assert result.stdout == b""
    assert b"usage" in result.stderr


def test_output_byte_limit_enforced(monkeypatch, capsys):
    monkeypatch.setattr(json_cli, "MAX_RESPONSE_BYTES", 8)
    stdin = types.SimpleNamespace(buffer=io.BytesIO(json.dumps(fresh_request()).encode()))
    monkeypatch.setattr(json_cli.sys, "stdin", stdin)
    exit_code = json_cli.main([])
    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    assert "exceeds" in captured.err


# ---------------------------------------------------------------------------
# Bootstrap independence from training/provider stacks
# ---------------------------------------------------------------------------


def test_cli_works_when_torch_transformers_boto_and_sky_raise(tmp_path):
    """Blocked heavy imports must not break the provision path, and the
    captured import set must contain no training/provider/Sky module."""
    modules_file = tmp_path / "modules.json"
    driver = tmp_path / "driver.py"
    driver.write_text(
        f"""
import importlib.abc
import json
import sys

BLOCKED = ("torch", "transformers", "boto3", "botocore", "sky")

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            raise ImportError(f"blocked import: {{fullname}}")
        return None

sys.meta_path.insert(0, Blocker())

from yeto.provision.json_cli import main

exit_code = main([])
json.dump(
    {{"exitCode": exit_code, "modules": sorted(sys.modules)}},
    open({str(modules_file)!r}, "w"),
)
sys.exit(exit_code)
"""
    )
    result = subprocess.run(
        [sys.executable, str(driver)],
        input=json.dumps(fresh_request()).encode(),
        capture_output=True,
        cwd=REPO_ROOT,
        env=clean_env(),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    envelope = parse_single_stdout_json(result.stdout)
    assert envelope["ok"] is True
    assert envelope["value"]["items"][0]["planItemId"] == PLAN_ITEM_ID

    captured = json.loads(modules_file.read_text())
    loaded = set(captured["modules"])
    for name in FORBIDDEN_MODULES:
        assert name not in loaded, f"{name} must never load in the provision path"


def test_module_entry_point_matches_console_script_target():
    # The console script target and python -m path are the same callable.
    from yeto.provision.json_cli import main

    assert callable(main)
    text = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'yeto-provision-solve = "yeto.provision.json_cli:main"' in text
