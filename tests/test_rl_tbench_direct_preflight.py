from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.rl import CODEX_OPENENV_AGENT, CODEX_OPENENV_IDENTITY_ENV
from yeto.rl import tbench_direct_preflight as preflight
from yeto.rl.codex_backend import QWEN35_08B_MODEL, QWEN35_08B_REVISION


def _argv() -> list[str]:
    return [
        "--custom-generate-function-path",
        "miles.rollout.generate_hub.agentic_tool_call.generate",
        "--custom-agent-function-path",
        CODEX_OPENENV_AGENT,
        "--custom-rm-path",
        "openenv_generate.reward_func",
        "--dynamic-sampling-filter-path",
        "openenv_generate.check_terminal_bench_episode",
        "--input-key",
        "messages",
        "--tito-model",
        "qwen35",
        "--sao-online-recipe",
        "coding",
        "--max-seq-len",
        "8192",
        "--num-gpus-per-node",
        "1",
        "--sglang-mem-fraction-static",
        "0.15",
        "--sglang-max-total-tokens",
        "393216",
        "--sglang-max-mamba-cache-size",
        "256",
        "--sao-compaction",
        "--sao-one-gpu-island",
        "--colocate",
    ]


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    values = {
        **CODEX_OPENENV_IDENTITY_ENV,
        "YETO_CODEX_COMPACTION_ENABLED": "1",
        "YETO_CODEX_COMPACTION_TRIGGER_TOKENS": "6144",
        "YETO_CODEX_COMPACTION_SUMMARY_MAX_TOKENS": "1024",
        "YETO_CODEX_MAX_COMPACTIONS": "3",
        "OPENENV_MAX_ROLLOUT_TIME_SECONDS": "1800",
        "SECRLENV_MAX_ROLLOUT_TIME_SECONDS": "1800",
        "TBENCH_REWARD_HMAC_KEY": "t" * 48,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("TBENCH_REWARD_HMAC_KEY_FILE", raising=False)


def _context() -> dict[str, str]:
    return {
        "benchmark": "terminal-bench-2.1",
        "model": QWEN35_08B_MODEL,
        "base_model_revision": QWEN35_08B_REVISION,
        "rollout_model_revision": QWEN35_08B_REVISION,
    }


def _runtime():
    return SimpleNamespace(
        trajectory_evidence_kind="terminal-bench-2.1",
        trajectory_evidence_schema_version=2,
    )


def test_exact_contract_attests_before_launch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(preflight, "_attest_adapter", lambda root: calls.append(root))
    preflight.preflight_tbench_codex_streaming(
        _argv(),
        sao_context=_context(),
        streaming_runtime=_runtime(),
        miles_root=tmp_path,
    )
    assert calls == [tmp_path.resolve()]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ("messages", "prompt", "input_key"),
        ("qwen35", "qwen35_08b", "tito_model"),
        ("8192", "16384", "max_seq_len"),
        ("0.15", "0.16", "sglang_mem_fraction_static"),
        ("393216", "393217", "sglang_max_total_tokens"),
        ("256", "257", "sglang_max_mamba_cache_size"),
    ),
)
def test_argument_drift_fails_closed(tmp_path, monkeypatch, old, new, message):
    monkeypatch.setattr(preflight, "_attest_adapter", lambda _root: None)
    argv = _argv()
    argv[argv.index(old)] = new
    with pytest.raises(ValueError, match=message):
        preflight.preflight_tbench_codex_streaming(
            argv,
            sao_context=_context(),
            streaming_runtime=_runtime(),
            miles_root=tmp_path,
        )


def test_environment_and_model_drift_fail_before_adapter_attestation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        preflight,
        "_attest_adapter",
        lambda _root: pytest.fail("adapter attestation ran after an earlier gate failed"),
    )
    monkeypatch.setenv("OPENENV_MAX_ROLLOUT_TIME_SECONDS", "1799")
    with pytest.raises(ValueError, match="OPENENV_MAX_ROLLOUT_TIME_SECONDS"):
        preflight.preflight_tbench_codex_streaming(
            _argv(),
            sao_context=_context(),
            streaming_runtime=_runtime(),
            miles_root=tmp_path,
        )

    monkeypatch.setenv("OPENENV_MAX_ROLLOUT_TIME_SECONDS", "1800")
    context = _context()
    context["rollout_model_revision"] = "0" * 40
    with pytest.raises(ValueError, match="model identity drifted"):
        preflight.preflight_tbench_codex_streaming(
            _argv(),
            sao_context=context,
            streaming_runtime=_runtime(),
            miles_root=tmp_path,
        )
