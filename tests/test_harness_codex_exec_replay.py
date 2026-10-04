"""G8 / task 10.3 (reduced): replay the black-box Codex 0.145.0 binary on CPU.

Real stock Codex binary (``YETO_CODEX_BINARY_PATH``) -> in-process
``_ResponsesBridge`` -> fake Miles ``/v1/chat/completions`` endpoint, with the
real TB2 relay (``tb2_provider.LocalProcessBackend``) and trusted verifier.
Skipped when the 0.145.0 binary is not available locally.  The request-shape
summary is written to ``/tmp/t3-g8-codex-replay.json``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import test_harness_tb2_provider as tb2_tests
import test_secrlenv_codex_harness as legacy_tests
from yeto.rl import tbench_outcome
from yeto.rl.engine.tool_wait import HarnessBoard
from yeto.rl.harness.codex import codex_harness_agent as harness
from yeto.rl.harness.codex import codex_openenv_subprocess_agent_function as subprocess_agent

SUMMARY = Path("/tmp/t3-g8-codex-replay.json")


def _stock_env(monkeypatch, binary: Path) -> None:
    for name, value in harness._IDENTITY_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("YETO_CODEX_BINARY_PATH", str(binary))
    monkeypatch.setenv("YETO_CODEX_BINARY_SHA256", hashlib.sha256(binary.read_bytes()).hexdigest())
    monkeypatch.setenv("YETO_CODEX_BINARY_SIZE_BYTES", str(binary.stat().st_size))
    monkeypatch.setenv("YETO_CODEX_VERSION", harness.CODEX_CLI_VERSION)
    monkeypatch.setenv("YETO_CODEX_BACKEND_MAX_TOKENS", "4096")
    monkeypatch.delenv("YETO_CODEX_COMPACTION_ENABLED", raising=False)


def test_stock_codex_0145_single_chain_replay_through_tb2_relay(monkeypatch, tmp_path):
    binary = legacy_tests._stock_codex_binary()
    _stock_env(monkeypatch, binary)
    hb = HarnessBoard()
    provider = tb2_tests._provider(tmp_path)
    tb2_tests._configure(monkeypatch, provider, harness_board=hb)
    monkeypatch.delenv("YETO_CODEX_OPENENV_ALLOW_SCRIPTED_DRIVER", raising=False)

    async def scenario():
        runner, miles_url, requests = await legacy_tests._fake_miles(
            [
                legacy_tests._completion("terminal.exec", '{"command":"echo ok > fixed.txt && ls"}', "call-1", "fix it", 2_000),
                legacy_tests._completion("terminal.exec", '{"command":"cat fixed.txt"}', "call-2", "check", 2_500),
                legacy_tests._completion("submit", '{"evidence":"fixed.txt written"}', "call-3", "done", 3_000),
            ]
        )
        try:
            metadata = {
                "task_id": "fix-git", "trajectory_id": "traj-g8", "prompt": "write ok into fixed.txt",
                "expected_policy_version": "pv-g8",
            }
            result = await asyncio.wait_for(subprocess_agent.run(miles_url, "p", {}, metadata), timeout=120)
        finally:
            await runner.cleanup()
        return result, requests

    result, requests = asyncio.run(scenario())
    outcome, reward = tbench_outcome.verified_outcome(result)
    assert outcome["status"] == "completed", result
    assert reward == 1.0
    assert result["chains_total"] == 1 and result["chain_break_reason"] is None
    assert provider.destroyed == 1 and provider.live == {}
    snap = hb.snapshot()
    assert snap.env_live == 0 and snap.in_flight == 0
    assert snap.tito_session_mismatch == 0
    breaks = snap.tito_chain_breaks
    assert not any(breaks.values()) if isinstance(breaks, dict) else breaks == 0
    # black-box Codex request shape: exactly 3 samples, one chain, tool set fixed, history grows by tool turns
    assert len(requests) == 3
    assert [t["function"]["name"] for t in requests[0]["tools"]] == ["terminal.exec", "submit"]
    roles = [[m["role"] for m in r["messages"]] for r in requests]
    assert roles[1][-1] == "tool" and roles[2][-1] == "tool"
    assert all(len(roles[i]) < len(roles[i + 1]) for i in range(2))
    SUMMARY.write_text(json.dumps({
        "codex_version": harness.CODEX_CLI_VERSION,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "requests": len(requests),
        "tools": [t["function"]["name"] for t in requests[0]["tools"]],
        "message_roles_per_request": roles,
        "status": outcome["status"], "reward": reward, "chains_total": result["chains_total"],
        "tito_session_mismatch": snap.tito_session_mismatch,
    }, indent=2) + "\n")
