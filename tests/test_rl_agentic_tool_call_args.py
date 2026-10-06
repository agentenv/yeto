"""Task 1.3 [Y]: the fork pin's ``agentic_tool_call.add_arguments`` parses the
IR-1 flags ``miles_adapter.config`` emits (CPU; the parser is loaded from the
pinned checkout with its runtime imports stubbed; no Ray, no sglang).

The pinned Miles checkout (``yeto.rl.MILES_NEXT_COMMIT``) is found through
``YETO_MILES_NEXT_ROOT`` or the local candidates below; skipped when none has
that HEAD.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from yeto.rl import MILES_NEXT_COMMIT
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.miles_adapter import config as cfg

from test_rl_miles_adapter_config import make_config, sub

CANDIDATES = (
    os.environ.get("YETO_MILES_NEXT_ROOT", ""),
    os.path.expanduser("~/work/miles-fr1"),
    os.path.expanduser("~/work/miles-next"),
    "/tmp/miles_up",
)
GENERATE_HUB = "miles/rollout/generate_hub/agentic_tool_call.py"
STUBBED = (
    "httpx",
    "sglang", "sglang.srt", "sglang.srt.entrypoints", "sglang.srt.entrypoints.openai",
    "sglang.srt.entrypoints.openai.protocol",
    "miles", "miles.rollout", "miles.rollout.base_types", "miles.rollout.generate_utils",
    "miles.rollout.generate_utils.openai_endpoint_utils", "miles.rollout.session",
    "miles.rollout.session.v2", "miles.rollout.session.v2.metrics", "miles.utils",
    "miles.utils.function_registry", "miles.utils.types",
)
STUB_NAMES = {
    "sglang.srt.entrypoints.openai.protocol": ("ChatCompletionRequest",),
    "miles.rollout.base_types": ("GenerateFnInput", "GenerateFnOutput"),
    "miles.rollout.generate_utils.openai_endpoint_utils": ("OpenAIEndpointTracer",),
    "miles.rollout.session.v2.metrics": ("SESSION_ROLLOUT_METRICS_KEY",),
    "miles.utils.function_registry": ("load_function",),
    "miles.utils.types": ("Sample",),
}


def pinned_miles_root() -> Path:
    for candidate in CANDIDATES:
        if not candidate or not (Path(candidate) / GENERATE_HUB).is_file():
            continue
        head = subprocess.run(["git", "-C", candidate, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        if head == MILES_NEXT_COMMIT:
            return Path(candidate)
    pytest.skip(f"no local Miles checkout at the fork pin {MILES_NEXT_COMMIT[:12]} (set YETO_MILES_NEXT_ROOT)")


@pytest.fixture
def upstream_add_arguments(monkeypatch):
    root = pinned_miles_root()
    for name in STUBBED:
        module = types.ModuleType(name)
        for attr in STUB_NAMES.get(name, ()):
            setattr(module, attr, type(attr, (), {}) if attr[0].isupper() else (lambda *a, **k: None))
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_pinned_agentic_tool_call", root / GENERATE_HUB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.generate.add_arguments


def _agent_argv() -> list[str]:
    c = sub(make_config(), "agent",
            custom_generate_function_path=cfg.AGENTIC_TOOL_CALL_GENERATE,
            custom_agent_function_path="yeto.rl.harness.codex.codex_openenv_subprocess_agent_function.run",
            agent_max_seq_len=8192, use_session_server=True, tito_model="qwen35")
    return list(cfg.translate_run_config(c, AlgorithmSpec(), tito_parser_resolver=lambda m: ("qwen3", "qwen")).argv)


def test_pinned_add_arguments_recognises_the_ir1_flags(upstream_add_arguments):
    parser = argparse.ArgumentParser()
    upstream_add_arguments(parser)
    argv = _agent_argv()
    i = argv.index("--custom-agent-function-path")
    j = argv.index("--max-seq-len")
    agent_flags = argv[i:i + 2] + argv[j:j + 2]
    ns, unknown = parser.parse_known_args(agent_flags)
    assert unknown == []
    assert ns.custom_agent_function_path == "yeto.rl.harness.codex.codex_openenv_subprocess_agent_function.run"
    assert ns.max_seq_len == 8192
    # The launcher-level spelling (--agent-max-seq-len) is NOT an upstream flag:
    # config.py must keep mapping agent_max_seq_len to --max-seq-len.
    _, unknown = parser.parse_known_args(["--agent-max-seq-len", "8192"])
    assert unknown == ["--agent-max-seq-len", "8192"]
    # Both are the only flags add_arguments owns.
    owned = {a.option_strings[0] for a in parser._actions if a.option_strings and a.dest != "help"}
    assert owned == {"--custom-agent-function-path", "--max-seq-len"}


def test_pinned_session_server_rule_still_quotes_partial_rollout():
    root = pinned_miles_root()
    text = (root / "miles/utils/arguments.py").read_text(encoding="utf-8")
    assert '"--use-session-server does not support --partial-rollout"' in text
    assert "--use-session-server does not support --partial-rollout" in cfg.SESSION_SERVER_PARTIAL_ROLLOUT_RULE
    argv = _agent_argv()
    assert "--use-session-server" in argv and "--partial-rollout" not in argv
    assert argv[argv.index("--tito-model") + 1] == "qwen35"
