"""CompactionRL context compaction inside the Codex Responses bridge (design D8).

Opt-in only (``YETO_CODEX_COMPACTIONRL=1``); with it unset nothing here runs and
the stock path (``codex_harness_agent._drive_codex``) is used unchanged.

Why the bridge can rewrite the context although Codex keeps the full history:
the bridge never forwards Codex's Responses ``input``.  It checks that input is
exactly the expected append-only history (``_expected_input``) and samples Miles
from its own Chat Completions message list (``_messages``).  Codex requests use
``store:false`` and no ``previous_response_id`` (both enforced by
``_validate_codex_request``), so Codex is never told about the compaction: its
own history stays complete and append-only, while each Miles request carries
only the rebuilt context ``s + u_resume(S) + last k steps``.

Training shape (progress.md "S13 cross_segment 每段 sample"): one sample per
segment.  The fork-pinned v1 session server is append-only and would silently
roll back a rewritten history (R-D5a), so every segment uses its own session:
the summary is sampled in the current session (trainable, logprobs recorded by
the session server like any other turn), then the rebuilt context starts in the
next pre-created session.  The trusted parent pre-creates those sessions
(``codex_openenv_agent_function.create_segment_sessions``) and
``codex_openenv_generate`` collects them and writes segment metadata.

Codex 0.145.0's own auto-compaction is switched off explicitly
(``model_auto_compact_token_limit`` = i64 max); any Codex-side compaction still
fails closed in the stock bridge (history item types / ``thread/compacted``).
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from yeto.rl.compaction import (
    MAX_COMPACTIONS,
    RESUME_TEMPLATE,
    SUMMARY_PROMPT,
    T_COMP,
    CompactionConfig,
    extract_summary,
    should_compact,
)

from . import codex_harness_agent as harness
from . import agent as legacy

COMPACTIONRL_ENV = "YETO_CODEX_COMPACTIONRL"
T_COMP_ENV = "YETO_CODEX_COMPACTIONRL_T_COMP"
SEGMENT_URLS_KEY = "segment_base_urls"
METRICS_KEY = "codex_compaction"
SESSIONS_METADATA_KEY = "codex_compaction_sessions"
# Codex 0.145.0 accepts this key under --strict-config (checked against the
# pinned binary: a string value fails with "expected i64").
CODEX_AUTO_COMPACT_DISABLED = "model_auto_compact_token_limit=9223372036854775807"
SUMMARY_PROMPT_TOKEN_RESERVE = len(SUMMARY_PROMPT.encode("utf-8")) + 256
_MESSAGE_TOKEN_OVERHEAD = 256
_FALSE = frozenset({"", "0", "false", "no", "off"})
_TRUE = frozenset({"1", "true", "yes", "on"})


def compactionrl_enabled(env: Mapping[str, str] | None = None) -> bool:
    raw = (os.environ if env is None else env).get(COMPACTIONRL_ENV, "").strip().lower()
    if raw in _FALSE:
        return False
    if raw in _TRUE:
        return True
    raise harness.CodexHarnessError(f"{COMPACTIONRL_ENV} must be a boolean flag")


def compaction_config(max_seq_len: int, env: Mapping[str, str] | None = None) -> CompactionConfig:
    raw = (os.environ if env is None else env).get(T_COMP_ENV, "").strip()
    t_comp = T_COMP
    if raw:
        if not raw.isdigit() or int(raw) <= 0:
            raise harness.CodexHarnessError(f"{T_COMP_ENV} must be a positive integer")
        t_comp = int(raw)
    try:
        return CompactionConfig(context_budget=int(max_seq_len), t_comp=t_comp)
    except ValueError as exc:
        raise harness.CodexHarnessError(f"CompactionRL config rejected: {exc}") from exc


def _message_bound(message: Mapping[str, Any]) -> int:
    """Upper bound on a chat message's tokens: bytes + template overhead."""
    return len(json.dumps(message, ensure_ascii=False).encode("utf-8")) + _MESSAGE_TOKEN_OVERHEAD


def compaction_counters(metrics: Any) -> dict[str, Any]:
    record = getattr(metrics, METRICS_KEY, None)
    return {METRICS_KEY: copy.deepcopy(record)} if isinstance(record, dict) else {}


class CompactionRLBridge(harness._ResponsesBridge):
    """Stock bridge + D8 trigger/summary/rebuild, one session per segment."""

    def __init__(
        self,
        miles_base_url: str,
        prompt: str,
        request_kwargs: dict[str, Any],
        metrics: legacy.AgentMetrics,
        *,
        max_seq_len: int | None,
        segment_base_urls: Sequence[str],
        config: CompactionConfig | None = None,
    ) -> None:
        super().__init__(
            miles_base_url, prompt, request_kwargs, metrics, max_seq_len=max_seq_len
        )
        if self._compaction_enabled:
            raise harness.CodexHarnessError(
                "legacy YETO_CODEX_COMPACTION_ENABLED and CompactionRL are exclusive"
            )
        if max_seq_len is None:
            raise harness.CodexHarnessError("CompactionRL requires max_seq_len (C)")
        self._cfg = config or compaction_config(max_seq_len)
        if self._cfg.context_budget != max_seq_len:
            raise harness.CodexHarnessError("CompactionRL budget must equal max_seq_len")
        if len(segment_base_urls) != self._cfg.max_compactions:
            raise harness.CodexHarnessError(
                "CompactionRL needs one pre-created session per allowed compaction"
            )
        self._segment_urls = [
            f"{legacy._session_url(url).rstrip('/')}/chat/completions"
            for url in segment_base_urls
        ]
        # Paper sec. 5.1: one reply cap for every turn, summaries included.
        self._compaction_summary_max_tokens = self._requested_max_tokens
        self._max_compactions = self._cfg.max_compactions
        self._step_bounds: list[int] = []
        self._record: dict[str, Any] = {
            "schema": 1,
            "context_budget": self._cfg.context_budget,
            "t_comp": self._cfg.t_comp,
            "max_compactions": self._cfg.max_compactions,
            "compactions": 0,
            "segments": [],
        }
        setattr(metrics, METRICS_KEY, self._record)

    # --- stock hooks -----------------------------------------------------
    def _compaction_headers(self) -> dict[str, str]:
        return {}  # the pinned session server has no compaction headers (R-D5a)

    def _should_compact(self, estimated_context_tokens: int) -> bool:
        return (
            self._compaction_count < self._cfg.max_compactions
            and bool(self._atomic_steps)
            and self._pending is None
            and should_compact(estimated_context_tokens, self._cfg)
        )

    def expect_tool_output(self, call_id: str, output: str) -> None:
        super().expect_tool_output(call_id, output)
        assistant, tool = self._atomic_steps[-1]
        self._step_bounds.append(_message_bound(assistant) + _message_bound(tool))

    def _rebuild_compacted_messages(self, retained_step_count: int) -> None:
        if self._resume_summary is None:
            raise harness.CodexHarnessError("compaction resume summary is missing")
        retained = self._atomic_steps[-retained_step_count:] if retained_step_count else []
        bounds = self._step_bounds[-retained_step_count:] if retained_step_count else []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": harness.BASE_INSTRUCTIONS},
            {"role": "user", "content": RESUME_TEMPLATE.format(summary=self._resume_summary)},
        ]
        for assistant, tool in retained:
            messages.extend((copy.deepcopy(assistant), copy.deepcopy(tool)))
        self._messages = messages
        self._atomic_steps = copy.deepcopy(retained)
        self._step_bounds = list(bounds)
        self._retained_step_count = retained_step_count
        if self._record["segments"]:
            self._record["segments"][-1]["kept_steps"] = retained_step_count
        # Nothing of the new window is tokenized yet: bound it from above.
        self._current_context_tokens = 0
        self._unobserved_tool_token_upper_bound = sum(_message_bound(m) for m in messages)

    def _choose_kept_steps(self, summary: str) -> int:
        """Eq. 9: largest k <= cfg.k whose rebuilt context does not trigger again."""
        head = _message_bound({"role": "system", "content": harness.BASE_INSTRUCTIONS}) + _message_bound(
            {"role": "user", "content": RESUME_TEMPLATE.format(summary=summary)}
        )
        for k in range(min(self._cfg.keep_recent_steps, len(self._atomic_steps)), 0, -1):
            if not should_compact(head + sum(self._step_bounds[-k:]), self._cfg):
                return k
        return 0

    async def _compact_context(self) -> None:
        if self._pending is not None or self._last_assistant_message is not None:
            raise harness.CodexHarnessError("compaction attempted inside an atomic tool step")
        used = self._current_context_tokens + self._unobserved_tool_token_upper_bound
        if self._cfg.context_budget - used - SUMMARY_PROMPT_TOKEN_RESERVE <= 0:
            raise harness.CodexSequenceLimit("no room left for the compaction summary")
        # The stock sampler reserves the legacy instruction size; top it up to ours.
        self._unobserved_tool_token_upper_bound += max(
            0, SUMMARY_PROMPT_TOKEN_RESERVE - harness.COMPACTION_SUMMARY_INSTRUCTION_TOKEN_RESERVE
        )
        summary_messages = copy.deepcopy(self._messages)
        summary_messages.append({"role": "user", "content": SUMMARY_PROMPT})
        try:
            completion = await self._sample_miles(messages=summary_messages, summary=True)
        except harness._CompactionContextDoesNotFit as exc:
            raise harness.CodexSequenceLimit("compaction summary does not fit") from exc
        _usage, total = harness._usage(completion)
        if total is None:
            self._metrics.usage_missing += 1
            raise harness.CodexHarnessError("Miles omitted summary token usage")
        self._metrics.max_model_total_tokens = max(self._metrics.max_model_total_tokens, total)
        text, finish_reason = self._summary_text(completion)
        summary, ok = extract_summary(text)
        if not summary:
            raise harness.CodexModelFailure("compaction summary is empty")
        segment = self._compaction_count
        self._record["segments"].append(
            {
                "segment_index": segment,
                "trigger_context_tokens": used,
                "summary_total_tokens": total,
                "summary_finish_reason": finish_reason,
                "summary_ok": ok,
                "kept_steps": None,
            }
        )
        self._miles_url = self._segment_urls[segment]
        self._compaction_count += 1
        self._record["compactions"] = self._compaction_count
        self._context_window += 1
        self._resume_summary = summary
        self._rebuild_compacted_messages(self._choose_kept_steps(summary))

    @staticmethod
    def _summary_text(completion: dict[str, Any]) -> tuple[str, str]:
        choices = completion.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise harness.CodexHarnessError("Miles returned extra or missing summary samples")
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason not in {"stop", "length"}:
            raise harness.CodexModelFailure("compaction summary ended abnormally")
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise harness.CodexModelFailure("compaction summary is not an assistant message")
        content = message.get("content")
        if not isinstance(content, str) or message.get("tool_calls") not in (None, []):
            raise harness.CodexModelFailure("compaction summary must be tool-free text")
        return content, finish_reason


class _CompactionAppServerDriver(harness._AppServerDriver):
    """Stock driver; only the Codex argv gains the auto-compaction off switch."""

    async def __aenter__(self) -> Self:
        isolated_home = tempfile.TemporaryDirectory(prefix="yeto-codex-")
        self._isolated_home = isolated_home
        Path(isolated_home.name).chmod(0o700)
        environment = {
            "CODEX_HOME": isolated_home.name,
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "TZ": "UTC",
            "YETO_CODEX_BRIDGE_TOKEN": self._bridge.token,
        }
        self._process = await asyncio.create_subprocess_exec(
            *codex_argv(self._binary, self._bridge),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
            limit=harness.MAX_APP_SERVER_FRAME_BYTES + 1,
            start_new_session=True,
        )
        self._process_group_id = self._process.pid
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        return self


def codex_argv(binary: Path, bridge: harness._ResponsesBridge) -> list[str]:
    return [*harness._codex_argv(binary, bridge), "-c", CODEX_AUTO_COMPACT_DISABLED]


async def drive_codex_compactionrl(
    binary: Path,
    miles_base_url: str,
    client: Any,
    episode: dict[str, Any],
    request_kwargs: dict[str, Any],
    metrics: legacy.AgentMetrics,
    *,
    max_seq_len: int | None,
    segment_base_urls: Sequence[str],
) -> str:
    """``codex_harness_agent._drive_codex`` with the CompactionRL bridge."""
    episode_id = episode.get("episode_id")
    prompt = episode.get("prompt")
    if not isinstance(episode_id, str) or not isinstance(prompt, str) or not prompt:
        raise legacy.EpisodeClientError("episode daemon returned invalid episode identity")
    async with CompactionRLBridge(
        miles_base_url,
        prompt,
        request_kwargs,
        metrics,
        max_seq_len=max_seq_len,
        segment_base_urls=segment_base_urls,
    ) as bridge, _CompactionAppServerDriver(
        binary, bridge, client, episode_id, prompt, metrics
    ) as driver:
        return await driver.drive()
