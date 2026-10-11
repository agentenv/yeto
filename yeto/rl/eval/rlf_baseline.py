"""RLF-shape Qwen3-30B-A3B baseline runner (rlf-codex-rl task 5.6).

The RLF counterpart of ``codex_infer.py`` (whose docstring covers the TB2
attempt shape).  One run, in this order:

1. admit the CTF task packages into the RLF catalog and render the prompt
   rows (``yeto.rl.rlf.catalog.prompt_rows``);
2. bring up serving on the local GPU: one SGLang ``launch_server`` for the
   bf16 base model (no LoRA) plus Miles' standalone session server — the
   same wire the training rollout uses (Modal deployment shape:
   1xA100-80GB, ``--context-length 32768 --mem-fraction-static 0.9``, model
   from the ``yeto-fn-models`` Volume ``hf/Qwen3-30B-A3B/ad44e777/``);
3. per prompt row: acquire one RLF CTF environment lease
   (``RlfModalProvider``: nested-Docker target + trusted flag verifier),
   create one session, and drive the training Codex subprocess worker
   (``_drive_worker``) with the ``qwen3_30b_a3b`` profile budgets
   (48 turns, 1024 tokens/turn, 8192-byte tool output) under a HARD
   30-minute per-task deadline;
4. finish on the trusted side (``finish_trusted`` -> RLF grade) and write
   one trace record per episode plus a run summary.

Flag hygiene: the trusted flags reach the target containers only as
``-e FLAG`` (W4); here they additionally scrub every persisted trace string
(``yeto.rl.rlf.modal.scrub``).  Submitted flag values are never recorded —
only booleans (``has_flag`` / ``flag_correct``) and a ``flag_seen`` marker
per tool output.  Serving that is not ready within ``start_timeout_s``
(15 min) aborts the whole run as ``infrastructure``; a failed episode
(acquire/drive/grade rejection) is recorded as its own trace with
``classification`` != "success" and never stops the remaining tasks.

CPU tests inject the provider, the drive function and the HTTP client; the
real ones (Modal sandboxes, SGLang, the Miles session server) only exist on
the GPU container.  Modal entry: ``yeto.cloud.modal_rlf_baseline``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from yeto.rl.codex_backend import (
    QWEN3_30B_A3B_MODEL,
    QWEN3_30B_A3B_REVISION,
    stock_codex_backend_profile,
)
from yeto.rl.rlf.catalog import TaskCatalog, import_catalog, prompt_rows
from yeto.rl.rlf.modal import RlfCtfConfig, RlfModalProvider, scrub

from .codex_infer import SESSION_SERVER_BOOT, STATUS_TO_END_REASON, _http, resolve_serving

TRACE_SCHEMA = "yeto.rlf.baseline.trace/v1"
SUMMARY_SCHEMA = "yeto.rlf.baseline.summary/v1"
DEFAULT_PROFILE = "qwen3_30b_a3b"
HARD_EPISODE_SECONDS = 1800.0        # 30 min per task (HARD)
SERVING_START_TIMEOUT_S = 900.0      # 15 min: not ready -> infrastructure, abort


def volume_snapshot_path(model: str = QWEN3_30B_A3B_MODEL, revision: str = QWEN3_30B_A3B_REVISION,
                         mount: str = "/mnt/yeto-models") -> str:
    """The Modal model Volume layout (``hf/<name>/<rev[:8]>/``, dl_hf.py)."""
    return f"{mount}/hf/{model.split('/')[-1]}/{revision[:8]}"


@dataclass
class BaselineConfig:
    """One baseline run.  Defaults come from the ``qwen3_30b_a3b`` profile."""

    tasks_dir: str                                  # admitted CTF packages (task_id -> dir)
    flags: Mapping[str, str]                        # task_id -> trusted flag (never logged)
    profile: str = DEFAULT_PROFILE
    model_path: str = ""                            # filled by __post_init__ (Volume snapshot)
    revision: str | None = QWEN3_30B_A3B_REVISION
    model_identifier: str = QWEN3_30B_A3B_MODEL
    tito_model: str = "qwen3"
    context_length: int = 32_768
    max_tokens: int = 8_192  # default matches the qwen3_30b_a3b profile (widened 2026-10-11)
    max_turns: int = 48
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    mem_fraction_static: float = 0.9
    host: str = "127.0.0.1"
    sglang_port: int = 30000
    session_port: int = 30100
    start_timeout_s: float = SERVING_START_TIMEOUT_S
    episode_seconds: float = HARD_EPISODE_SECONDS
    repeat: int = 1
    task_ids: tuple[str, ...] | None = None
    sglang_extra_args: tuple[str, ...] = ()
    log_dir: str | None = None
    preview_bytes: int = 2048                       # per-tool-output trace preview cap

    def __post_init__(self) -> None:
        if not self.model_path and self.revision:
            self.model_path = volume_snapshot_path(self.model_identifier, str(self.revision))

    @classmethod
    def for_profile(cls, profile: str = DEFAULT_PROFILE, **kwargs: Any) -> "BaselineConfig":
        """Derive every budget from the registered profile (one source of truth)."""
        p = stock_codex_backend_profile(profile)
        # explicit kwargs beat the profile (e.g. context_length=24576 for the
        # qwen3_cyber compaction budget, which the bridge requires == max_seq_len)
        derived = {"revision": p.get("model_revision"),
                   "model_identifier": str(p.get("model_identifier") or ""),
                   "tito_model": str(p.get("tito_model") or p["model"]),
                   "context_length": int(p["max_seq_len"]),
                   "max_tokens": int(p["max_tokens_per_turn"]),
                   "max_turns": int(p["max_turns"])}
        derived.update({k: v for k, v in kwargs.items() if v is not None or k not in derived})
        return cls(profile=profile, **derived)

    @property
    def policy_id(self) -> str:
        """The baseline policy is the pinned base model itself."""
        return f"{self.model_identifier}@{self.revision}"

    @property
    def session_url(self) -> str:
        return f"http://{self.host}:{self.session_port}"

    @property
    def sglang_url(self) -> str:
        return f"http://{self.host}:{self.sglang_port}"

    def request_kwargs(self) -> dict[str, Any]:
        out: dict[str, Any] = {"temperature": self.temperature, "top_p": self.top_p,
                               "max_tokens": self.max_tokens}
        if self.top_k is not None:
            out["top_k"] = self.top_k
        return out

    def worker_budget_env(self) -> dict[str, str]:
        """Per-episode worker env: the profile selection + the launch-time budget knobs
        (docs/RLF_CODEX_RUN.md "Qwen3 cyber baseline profile").  The 8192-byte tool
        output budget needs no env: the worker reads it from the selected profile."""
        return {"YETO_CODEX_CHAT_TEMPLATE": self.profile,
                "SECRLENV_MAX_TURNS": str(self.max_turns),
                "YETO_CODEX_BACKEND_MAX_TOKENS": str(self.max_tokens)}


class InfrastructureError(RuntimeError):
    """Serving never came up (or died) — the run is aborted as infrastructure."""


# --- serving (SGLang + Miles session server, no LoRA) --------------------------------------


def sglang_argv(cfg: BaselineConfig, *, python: str = sys.executable,
                serving: Mapping[str, Any] | None = None) -> list[str]:
    """Baseline serving: same ``launch_server`` shape as the eval island, minus LoRA."""
    argv = [python, "-m", "sglang.launch_server", "--model-path", cfg.model_path,
            "--host", cfg.host, "--port", str(cfg.sglang_port), "--trust-remote-code",
            "--context-length", str(cfg.context_length),
            "--mem-fraction-static", str(cfg.mem_fraction_static)]
    if cfg.revision and not Path(cfg.model_path).exists():
        argv += ["--revision", cfg.revision]
    serving = serving or {}
    if serving.get("reasoning_parser") is not None:
        argv += ["--reasoning-parser", serving["reasoning_parser"]]
    if serving.get("tool_call_parser") is not None:
        argv += ["--tool-call-parser", serving["tool_call_parser"]]
    if serving.get("chat_template_path") is not None:
        argv += ["--chat-template", serving["chat_template_path"]]
    return argv + list(cfg.sglang_extra_args)


def session_server_config(cfg: BaselineConfig, *, serving: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``SessionServerConfig`` fields, as the training island sets them — but LoRA-free.

    ``miles.utils.lora.utils.is_lora_enabled`` = ``lora_rank > 0 or
    lora_adapter_path is not None``; pydantic rejects ``lora_rank=None``.  So the
    only genuinely LoRA-free shape is ``lora_rank=0`` + ``lora_adapter_path=None``
    (second A100 attempt 2026-10-11: rank=32 made the session server request the
    ``miles_lora`` adapter from a LoRA-less SGLang -> HTTP 400)."""
    serving = serving or {}
    return {
        "host": cfg.host, "port": cfg.session_port, "instance_id": "rlf-baseline",
        "backend_url": cfg.sglang_url, "timeout": 1800.0, "hf_checkpoint": cfg.model_path,
        "chat_template_path": serving.get("chat_template_path"), "tito_model": cfg.tito_model,
        "apply_chat_template_kwargs": dict(serving.get("chat_template_kwargs") or {}),
        "use_rollout_routing_replay": False, "use_rollout_indexer_replay": False,
        "use_sampling_support_replay": False, "sglang_speculative_algorithm": None,
        "num_layers": None, "moe_router_topk": None, "save_debug_trajectory_data": None,
        "lora_rank": 0, "lora_adapter_path": None, "lora_train_only": False,
        "use_session_server": True, "session_message_matcher": "strict",
        "pause_generation_mode": None,
        "session_sample_picker_path": None, "session_sample_postprocessor_path": None,
    }


class ServingStack:
    """One SGLang server + one session server on the local GPU; fail closed on slow starts."""

    def __init__(self, cfg: BaselineConfig, *, popen: Callable[..., Any] = subprocess.Popen,
                 http: Callable[..., tuple[int, Any]] = _http,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 serving: Callable[[str], Mapping[str, Any]] = resolve_serving) -> None:
        self.cfg = cfg
        self.serving = serving
        self.popen = popen
        self.http = http
        self.sleep = sleep
        self.clock = clock
        self.procs: dict[str, Any] = {}

    def _spawn(self, name: str, argv: list[str]) -> None:
        log = None
        if self.cfg.log_dir:
            Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)
            log = open(Path(self.cfg.log_dir) / f"{name}.log", "ab")
        self.procs[name] = self.popen(argv, stdout=log, stderr=subprocess.STDOUT if log else None,
                                      start_new_session=True)

    def _wait(self, name: str, url: str) -> float:
        t0 = self.clock()
        while True:
            proc = self.procs.get(name)
            if proc is not None and proc.poll() is not None:
                raise InfrastructureError(f"{name} exited with {proc.returncode} before it was ready")
            try:
                status, _ = self.http("GET", url, timeout=10.0)
            except OSError:
                status = None
            if status == 200:
                return round(self.clock() - t0, 3)
            if self.clock() - t0 > self.cfg.start_timeout_s:
                raise InfrastructureError(
                    f"{name} not ready after {self.cfg.start_timeout_s} s ({url}): infrastructure")
            self.sleep(2.0)

    def start(self) -> dict[str, Any]:
        serving = dict(self.serving(self.cfg.tito_model))
        self._spawn("sglang", sglang_argv(self.cfg, serving=serving))
        sglang_s = self._wait("sglang", f"{self.cfg.sglang_url}/health_generate")
        self._spawn("session-server", [sys.executable, "-c", SESSION_SERVER_BOOT,
                                       json.dumps(session_server_config(self.cfg, serving=serving))])
        session_s = self._wait("session-server", f"{self.cfg.session_url}/health")
        return {"sglang_start_s": sglang_s, "session_server_start_s": session_s,
                "serving": {k: v for k, v in serving.items()}}

    def close(self) -> None:
        for proc in self.procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in self.procs.values():
            try:
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                proc.kill()


# --- per-episode trace recording -------------------------------------------------------------


def _scrubbed(value: Any, secrets: tuple[str, ...]) -> Any:
    """JSON-round-trip any structure and scrub the secrets out of every string."""
    if not secrets:
        return json.loads(json.dumps(value, default=str))
    return json.loads(scrub(json.dumps(value, default=str), *secrets))


class TraceRecorder:
    """Wraps the lease's relay environment: per-tool trace events without the flag.

    Records the scrubbed command, a bounded scrubbed output preview and boolean
    markers (``flag_seen`` in the output, ``flag_correct`` on submission).  The
    submitted flag value itself is never stored.
    """

    def __init__(self, relay: Any, *, secrets: tuple[str, ...] = (), preview_bytes: int = 2048) -> None:
        self._relay = relay
        self._secrets = tuple(s for s in secrets if s)
        self._preview_bytes = int(preview_bytes)
        self.tool_events: list[dict[str, Any]] = []
        self.submissions: list[dict[str, Any]] = []

    def attach(self) -> "TraceRecorder":
        relay = self._relay
        original_execute = relay.execute
        original_submit = relay.submit

        async def execute(episode_id: str, command: str, timeout_seconds: float, output_bytes: int):
            result = await original_execute(episode_id, command, timeout_seconds, output_bytes)
            output = str((result or {}).get("output") or "")
            self.tool_events.append({
                "seq": len(self.tool_events),
                "command": scrub(command, *self._secrets),
                "exit_code": (result or {}).get("exit_code"),
                "timed_out": bool((result or {}).get("timed_out")),
                "truncated": bool((result or {}).get("truncated")),
                "output_bytes": len(output.encode("utf-8")),
                "flag_seen": any(s in output for s in self._secrets),
                "output_preview": scrub(output[: self._preview_bytes], *self._secrets),
            })
            return result

        async def submit(episode_id: str, submission: dict[str, Any]):
            result = await original_submit(episode_id, submission)
            candidate = submission.get("flag") if isinstance(submission, Mapping) else None
            import hmac as _hmac
            self.submissions.append({
                "after_tool_calls": len(self.tool_events),
                "has_flag": isinstance(candidate, str) and bool(candidate),
                "flag_correct": any(isinstance(candidate, str)
                                    and _hmac.compare_digest(candidate, s) for s in self._secrets),
            })
            return result

        relay.execute = execute
        relay.submit = submit
        return self


def _turn_records(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-turn token/byte rows from the harness's capped per-turn metric lists."""
    completion = list(metrics.get("turn_completion_tokens") or [])
    context = list(metrics.get("turn_context_tokens") or [])
    tool_bytes = list(metrics.get("turn_tool_output_bytes") or [])
    n = max(len(completion), len(context), len(tool_bytes))
    return [{"turn": i,
             "completion_tokens": completion[i] if i < len(completion) else None,
             "context_tokens": context[i] if i < len(context) else None,
             "tool_output_bytes": tool_bytes[i] if i < len(tool_bytes) else None}
            for i in range(n)]


def _default_drive() -> Callable[..., Any]:
    from yeto.rl.harness.codex.codex_openenv_subprocess_agent_function import _drive_worker

    return _drive_worker


async def run_episode(cfg: BaselineConfig, provider: Any, row: Mapping[str, Any], *,
                      session_url: str, drive: Callable[..., Any] | None = None,
                      http: Callable[..., tuple[int, Any]] = _http,
                      clock: Callable[[], float] = time.monotonic,
                      hmac_key: str | bytes | None = None) -> dict[str, Any]:
    """One baseline episode: lease -> session -> Codex worker -> trusted finish -> trace.

    Never raises for episode-level failures: they become trace records with
    ``classification`` "infrastructure" (acquire/drive/serving) or "rejected"
    (the trusted side refused the grade).
    """
    from yeto.rl import tbench_outcome
    from yeto.rl.harness.codex import codex_openenv_agent_function as adapter
    from yeto.rl.harness.codex.codex_openenv_subprocess_agent_function import task_prompt
    from yeto.rl.rlf.preflight import RolloutRejected

    drive = drive or _default_drive()
    metadata = dict(row.get("metadata") or {})
    task_id = str(metadata.get("task_id") or "")
    trajectory_id = f"rlf-baseline-{task_id}-{uuid.uuid4().hex[:12]}"
    secrets = (str(cfg.flags[task_id]),) if cfg.flags.get(task_id) else ()
    started = clock()
    trace: dict[str, Any] = {
        "schema": TRACE_SCHEMA, "task_id": task_id, "rlf_task_id": task_id,
        "rlf_trajectory_id": trajectory_id, "policy_id": cfg.policy_id,
        "prompt": scrub(str(metadata.get("prompt") or row.get("prompt") or ""), *secrets),
        "classification": "infrastructure", "success": False, "reward": None,
    }

    def infra(reason: str) -> dict[str, Any]:
        return {**trace, "error": scrub(reason, *secrets), "wall_s": round(clock() - started, 3)}

    lease = None
    session_id = None
    segment_ids: list[str] = []
    try:
        lease = await provider.acquire(task_id, trajectory_id)
    except Exception as exc:  # noqa: BLE001 - a failed acquire is infrastructure, never a 0 reward
        return infra(f"acquire: {type(exc).__name__}: {exc}")
    trace["rlf_episode_id"] = getattr(lease, "episode_id", None)
    trace["rlf_trajectory_id"] = getattr(lease, "trajectory_id", None) or trajectory_id
    recorder = None
    relay = getattr(lease, "environment", None)
    if relay is not None:
        recorder = TraceRecorder(relay, secrets=secrets, preview_bytes=cfg.preview_bytes).attach()
    try:
        try:
            prompt = task_prompt(metadata, row.get("prompt"), lease)
        except Exception as exc:  # noqa: BLE001 - a missing statement is a packaging error
            return infra(f"prompt: {type(exc).__name__}: {exc}")
        body: dict[str, Any] = {"evaluation": True, "temperature": float(cfg.temperature),
                                "top_p": float(cfg.top_p)}
        if cfg.top_k is not None:
            body["top_k"] = int(cfg.top_k)
        status, out = await asyncio.to_thread(http, "POST", f"{session_url}/sessions", body)
        if status != 200 or not isinstance(out, dict) or not out.get("session_id"):
            return infra(f"session server refused a session: HTTP {status} "
                         f"{scrub(str(out)[:200], *secrets)}")
        session_id = str(out["session_id"])
        episode_id = str(getattr(lease, "episode_id", None) or adapter.new_episode_id())
        job = {
            "base_url": f"{session_url}/sessions/{session_id}",
            "prompt": prompt,
            "request_kwargs": cfg.request_kwargs(),
            "episode_id": episode_id,
            "max_seq_len": cfg.context_length,
            "env_url": lease.env_url,
            "env_token": lease.env_token,
            "driver": "stock",
        }
        # CompactionRL only (no-op when disabled): pre-create segment sessions in
        # the trusted layer, same as the training agent function.
        try:
            segments_meta = await adapter.prepare_segment_sessions(job)
        except Exception as exc:  # noqa: BLE001 - session-server failures are infrastructure
            return infra(f"segment sessions: {type(exc).__name__}: {exc}")
        segment_ids = list(segments_meta.get("codex_compaction_sessions") or [])
        worker_env = {**dict(getattr(lease, "worker_env", None) or {}), **cfg.worker_budget_env()}
        deadline = min(float(getattr(lease, "deadline_seconds", 0) or cfg.episode_seconds),
                       float(cfg.episode_seconds))
        agent_t0 = clock()
        try:
            untrusted = await asyncio.wait_for(
                drive(job, trajectory_id, None, worker_env), timeout=deadline)
        except asyncio.TimeoutError:
            untrusted = {"status": "timeout", "metrics": {"timed_out": 1}, "episode_id": episode_id}
        except adapter.harness.CodexHarnessError as exc:
            return {**infra(f"drive: {type(exc).__name__}: {exc}"),
                    "agent_metrics": _scrubbed(getattr(exc, "metrics", None) or {}, secrets)}
        agent_s = round(clock() - agent_t0, 3)
        status = str(untrusted.get("status"))
        metrics = dict(untrusted.get("metrics") or {})
        if status not in adapter.POLICY_STATUSES:
            return {**infra(f"drive: unknown status {status!r}"),
                    "agent_metrics": _scrubbed(metrics, secrets)}
        try:
            trusted = await adapter.finish_trusted(untrusted, lease.verifier, task_id=task_id,
                                                   sample_id=trajectory_id, key=hmac_key)
        except RolloutRejected as exc:
            return {**infra(""), "classification": "rejected",
                    "error": scrub(f"grade rejected: {exc}", secrets),
                    "agent_metrics": _scrubbed(metrics, secrets)}
        reward = trusted.get("reward")
        if reward is None:
            # the timeout path carries the 0 reward only in the signed outcome envelope
            reward = (trusted.get(tbench_outcome.OUTCOME_KEY) or {}).get("reward")
        record: dict[str, Any] = {
            **trace,
            "status": status,
            "end_reason": STATUS_TO_END_REASON.get(status, status),
            "end_kind": metrics.get("end_kind"),
            "classification": "success" if trusted.get("success") else "failed",
            "success": bool(trusted.get("success")),
            "reward": reward,
            "turns": _turn_records(metrics),
            "turn_count": metrics.get("turns"),
            "tool_calls": metrics.get("tool_calls"),
            "submit_calls": metrics.get("submit_calls"),
            "tool_events": recorder.tool_events if recorder else [],
            "submissions": recorder.submissions if recorder else [],
            "max_model_total_tokens": metrics.get("max_model_total_tokens"),
            "max_seq_len_hit": metrics.get("max_seq_len_hit"),
            "compactions": int((metrics.get("codex_compaction") or {}).get("compactions", 0)),
            "segment_sessions": len(segment_ids),
            "last_completion": scrub(json.dumps(metrics.get("last_completion"), default=str)[:1500],
                                     *secrets) if metrics.get("last_completion") is not None else None,
            "grader_evidence": _scrubbed(trusted.get("grader_evidence") or {}, secrets),
            "verifier_log": scrub(str(trusted.get("verifier_log") or "")[-2000:], *secrets) or None,
            "agent_metrics": _scrubbed(metrics, secrets),
            "agent_s": agent_s,
            "wall_s": round(clock() - started, 3),
        }
        return record
    finally:
        for sid in segment_ids:
            try:
                await asyncio.to_thread(http, "DELETE", f"{session_url}/sessions/{sid}")
            except Exception:  # noqa: BLE001 - teardown never masks the episode outcome
                pass
        if session_id is not None:
            try:
                await asyncio.to_thread(http, "DELETE", f"{session_url}/sessions/{session_id}")
            except Exception:  # noqa: BLE001 - teardown never masks the episode outcome
                pass
        if lease is not None:
            try:
                await lease.destroy()
            except Exception:  # noqa: BLE001
                pass


# --- run loop + summary ------------------------------------------------------------------------


async def run_baseline(cfg: BaselineConfig, provider: Any, rows: list[Mapping[str, Any]], *,
                       session_url: str, drive: Callable[..., Any] | None = None,
                       http: Callable[..., tuple[int, Any]] = _http,
                       clock: Callable[[], float] = time.monotonic,
                       hmac_key: str | bytes | None = None,
                       progress: Callable[[str], None] | None = print) -> list[dict[str, Any]]:
    """Sequentially run every prompt row (one 32K window serves one episode at a time)."""
    traces = []
    for i, row in enumerate(rows):
        trace = await run_episode(cfg, provider, row, session_url=session_url, drive=drive,
                                  http=http, clock=clock, hmac_key=hmac_key)
        traces.append(trace)
        if progress is not None:
            progress(f"[rlf-baseline] {i + 1}/{len(rows)} {trace['task_id']}: "
                     f"{trace['classification']} end={trace.get('end_reason')} "
                     f"reward={trace.get('reward')}")
    return traces


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _min_mean_max(values: list[Any]) -> dict[str, Any]:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return {"mean": None, "max": None}
    return {"mean": round(sum(nums) / len(nums), 4), "max": max(nums)}


def summarize(traces: list[Mapping[str, Any]], *, policy_id: str | None = None,
              config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Baseline analysis: success rate, reward distribution, budgets usage per episode."""
    infra = [t for t in traces if t.get("classification") == "infrastructure"]
    rejected = [t for t in traces if t.get("classification") == "rejected"]
    evaluated = [t for t in traces if t.get("classification") in ("success", "failed")]
    successes = [t for t in evaluated if t.get("success")]
    rewards = [float(t["reward"]) for t in evaluated if t.get("reward") is not None]
    end_reasons: dict[str, int] = {}
    for t in evaluated:
        key = str(t.get("end_reason"))
        end_reasons[key] = end_reasons.get(key, 0) + 1
    per_task: dict[str, dict[str, Any]] = {}
    for t in traces:
        bucket = per_task.setdefault(str(t.get("task_id")),
                                     {"episodes": 0, "successes": 0, "rewards": [],
                                      "infrastructure": 0, "rejected": 0})
        bucket["episodes"] += 1
        if t.get("classification") == "infrastructure":
            bucket["infrastructure"] += 1
        elif t.get("classification") == "rejected":
            bucket["rejected"] += 1
        if t.get("reward") is not None:
            bucket["rewards"].append(float(t["reward"]))
        if t.get("success"):
            bucket["successes"] += 1
    for bucket in per_task.values():
        bucket["success_rate"] = _mean([1.0 if s else 0.0 for s in
                                        [r == 1.0 for r in bucket.pop("rewards")]])
    n_seq = sum(1 for t in evaluated if t.get("max_seq_len_hit"))
    return {
        "schema": SUMMARY_SCHEMA,
        "policy_id": policy_id,
        "config": dict(config or {}),
        "episodes": len(traces),
        "evaluated": len(evaluated),
        "infrastructure": len(infra),
        "rejected": len(rejected),
        "success_count": len(successes),
        "success_rate": _mean([1.0 if t.get("success") else 0.0 for t in evaluated]),
        "reward_distribution": {"mean": _mean(rewards),
                                "counts": {str(r): rewards.count(r) for r in sorted(set(rewards))}},
        "end_reasons": end_reasons,
        "max_seq_len_hit": {"count": n_seq,
                            "rate": _mean([1.0] * n_seq + [0.0] * (len(evaluated) - n_seq))},
        "compactions_total": sum(int(t.get("compactions") or 0) for t in evaluated),
        "turns": _min_mean_max([t.get("turn_count") for t in evaluated]),
        "tool_calls": _min_mean_max([t.get("tool_calls") for t in evaluated]),
        "max_model_total_tokens": _min_mean_max([t.get("max_model_total_tokens") for t in evaluated]),
        "per_task": per_task,
    }


def write_outputs(out_dir: str | Path, traces: list[Mapping[str, Any]],
                  summary: Mapping[str, Any]) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traces_path = out / "traces.jsonl"
    traces_path.write_text("".join(json.dumps(t, ensure_ascii=False, default=str) + "\n"
                                   for t in traces))
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    return {"traces": str(traces_path), "summary": str(summary_path)}


# --- wiring: catalog, provider, serving, CLI ----------------------------------------------------


def load_catalog(tasks_dir: str | Path, *, evidence: Mapping[str, Any] | None = None) -> TaskCatalog:
    """Admit every task package (a subdir with task.yaml) under ``tasks_dir``; fail closed."""
    root = Path(tasks_dir)
    packages = sorted(p for p in root.iterdir() if p.is_dir() and (p / "task.yaml").is_file())
    if not packages:
        from yeto.rl.rlf.contracts import AdmissionError

        raise AdmissionError(f"{tasks_dir}: no RLF task packages (subdir with task.yaml)")
    return import_catalog(packages, evidence=evidence)


def run_from_config(config: Mapping[str, Any], *, provider: Any = None,
                    drive: Callable[..., Any] | None = None,
                    http: Callable[..., tuple[int, Any]] | None = None,
                    hmac_key: str | bytes | None = None,
                    progress: Callable[[str], None] | None = print) -> dict[str, Any]:
    """The whole run (in-container body and CLI share this).

    ``config`` keys: ``tasks_dir``, ``flags`` (trusted; never logged),
    ``evidence`` (catalog admission), ``out_dir``, plus any
    :class:`BaselineConfig` field override under ``baseline``.  With
    ``serve`` = "start" (default) the local SGLang + session server are
    brought up first; "external" expects ``session_url`` (CPU smoke).
    """
    base_raw = dict(config.get("baseline") or {})
    if "sglang_extra_args" in base_raw:
        base_raw["sglang_extra_args"] = tuple(base_raw["sglang_extra_args"])
    if "task_ids" in base_raw and base_raw["task_ids"] is not None:
        base_raw["task_ids"] = tuple(base_raw["task_ids"])
    cfg = BaselineConfig.for_profile(str(config.get("profile") or DEFAULT_PROFILE),
                                     tasks_dir=str(config["tasks_dir"]),
                                     flags=dict(config.get("flags") or {}), **base_raw)
    out_dir = Path(str(config.get("out_dir") or "rlf-baseline-out"))
    catalog = load_catalog(cfg.tasks_dir, evidence=config.get("evidence"))
    rows = prompt_rows(catalog, list(cfg.task_ids) if cfg.task_ids else None, repeat=cfg.repeat)
    provider = provider or RlfModalProvider(
        ctf=RlfCtfConfig(tasks_dir=cfg.tasks_dir, flags=cfg.flags,
                         lease_seconds=cfg.episode_seconds))
    stack = None
    session_url = str(config.get("session_url") or "")
    serving_info: dict[str, Any] = {}
    t0 = time.monotonic()
    try:
        if str(config.get("serve") or "start") == "start":
            stack = ServingStack(cfg)
            serving_info = stack.start()
            session_url = cfg.session_url
        if not session_url:
            raise InfrastructureError("serve=external needs config['session_url']")
        traces = asyncio.run(run_baseline(cfg, provider, rows, session_url=session_url,
                                          drive=drive, http=http or _http, hmac_key=hmac_key,
                                          progress=progress))
    except InfrastructureError as exc:
        traces = []
        summary = summarize(traces, policy_id=cfg.policy_id)
        summary["aborted"] = {"classification": "infrastructure", "stage": "serving",
                              "reason": str(exc)[:500]}
        paths = write_outputs(out_dir, traces, summary)
        return {"ok": False, "summary": summary, "paths": paths}
    finally:
        if stack is not None:
            stack.close()
    summary = summarize(traces, policy_id=cfg.policy_id,
                        config={"profile": cfg.profile, "context_length": cfg.context_length,
                                "max_tokens": cfg.max_tokens, "max_turns": cfg.max_turns,
                                "model_path": cfg.model_path, "revision": cfg.revision,
                                "repeat": cfg.repeat, **serving_info})
    summary["wall_s"] = round(time.monotonic() - t0, 3)
    paths = write_outputs(out_dir, traces, summary)
    return {"ok": True, "summary": summary, "paths": paths}


def main(argv: list[str] | None = None) -> int:
    """``python -m yeto.rl.eval.rlf_baseline --tasks-dir DIR --flags-json JSON --out DIR``."""
    parser = argparse.ArgumentParser(description="RLF Qwen3-30B-A3B Codex baseline")
    parser.add_argument("--tasks-dir", default=None,
                        help="admitted CTF packages dir (default $YETO_RLF_CTF_TASKS_DIR)")
    parser.add_argument("--flags-json", default=None,
                        help='{"task_id": "flag{...}"} or @file (default $YETO_RLF_CTF_FLAGS_JSON)')
    parser.add_argument("--evidence", default=None, help="JSON catalog admission evidence "
                        "(must include runtime_quality_gate), or @file")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--task", action="append", default=None, help="task id (repeatable)")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--serve", choices=["start", "external"], default="start")
    parser.add_argument("--session-url", default=None)
    parser.add_argument("--episode-seconds", type=float, default=HARD_EPISODE_SECONDS)
    parser.add_argument("--start-timeout-s", type=float, default=SERVING_START_TIMEOUT_S)
    args = parser.parse_args(argv)

    import os

    def _json_arg(value: str | None, env: str) -> Any:
        raw = value if value is not None else os.environ.get(env)
        if not raw:
            return None
        if raw.startswith("@"):
            raw = Path(raw[1:]).read_text()
        return json.loads(raw)

    tasks_dir = args.tasks_dir or os.environ.get("YETO_RLF_CTF_TASKS_DIR")
    if not tasks_dir:
        parser.error("--tasks-dir or YETO_RLF_CTF_TASKS_DIR is required")
    flags = _json_arg(args.flags_json, "YETO_RLF_CTF_FLAGS_JSON")
    if not isinstance(flags, dict) or not flags:
        parser.error("--flags-json or YETO_RLF_CTF_FLAGS_JSON must be a JSON object of task_id -> flag")
    baseline: dict[str, Any] = {"repeat": args.repeat, "task_ids": args.task,
                                "episode_seconds": args.episode_seconds,
                                "start_timeout_s": args.start_timeout_s}
    if args.model_path:
        baseline["model_path"] = args.model_path
    config = {"tasks_dir": tasks_dir, "flags": flags,
              "evidence": _json_arg(args.evidence, "YETO_RLF_EVIDENCE_JSON"),
              "out_dir": args.out, "profile": args.profile, "serve": args.serve,
              "session_url": args.session_url, "baseline": baseline}
    result = run_from_config(config)
    print(json.dumps(result["summary"], indent=1, ensure_ascii=False, default=str))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
