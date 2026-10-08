"""Eval-island inference agent: SGLang + Miles session server + the training Codex driver
(rl-eval-difficulty-buckets 5.3/5.9).

What runs inside the eval-island container, in this order:

1. :func:`write_peft_dir` -- rebuild the canonical LoRA state from the stored
   ``policy.safetensors`` / ``policy.json`` (the store already checked every
   file's sha256), require its ``policy_tensor_hash`` to equal the manifest's,
   and write a standard PEFT adapter directory;
2. :class:`SglangSessionLoader` (the island's ``PolicyLoader``) -- start one
   SGLang server for the base model with LoRA enabled, load the adapter under
   Miles' served adapter name (``miles_lora``), and start Miles' own standalone
   session server (``miles.rollout.session.server``) in front of it with the
   training TITO settings; the next version only swaps the adapter. ``load``
   returns the manifest's ``rl/policy_token`` only after the tensor-hash check
   and a successful adapter load (fail closed otherwise);
3. :class:`CodexEvalAgent` (the ``tb2_attempt`` agent port) -- one evaluation
   session per attempt, then exactly the training rollout path: the task
   statement from ``task_prompt``, the Codex worker subprocess
   (``codex_openenv_subprocess_agent_function._drive_worker``, the same
   ``drive_untrusted`` + stock Responses bridge), the lease's worker env and
   deadline. Judging stays on the trusted side (``tb2_attempt``).

Nothing here re-implements the harness; the only new pieces are process
supervision and the PEFT writer. Factories for ``island_main``:
``yeto.rl.eval.codex_infer:loader_factory`` / ``:attempt_factory``.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

SERVED_LORA_NAME = "miles_lora"  # miles.utils.lora.utils.LORA_ADAPTER_NAME (the session server sends it)
STATUS_TO_END_REASON = {"completed": "completed", "timeout": "timed_out",
                        "max_turns": "max_turns", "max_seq_len": "max_seq_len"}


# --- 1. stored policy -> PEFT adapter dir ------------------------------------------------


def peft_adapter_config(*, rank: int, targets: list[str], base_model: str, revision: str | None) -> dict[str, Any]:
    """The fields ``yeto.rl.initial_adapter.load_initial_adapter`` accepts (alpha = rank, no dropout)."""
    return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": int(rank), "lora_alpha": int(rank),
            "target_modules": sorted(targets), "lora_dropout": 0.0, "bias": "none", "use_rslora": False,
            "use_dora": False, "fan_in_fan_out": False, "rank_pattern": {}, "alpha_pattern": {},
            "inference_mode": True, "base_model_name_or_path": base_model, "revision": revision}


def lora_rank_and_targets(shapes: Mapping[str, tuple[int, ...]]) -> tuple[int, list[str]]:
    """Rank from the ``lora_A`` rows (must be one value); targets = module leaf names."""
    ranks = {int(shape[0]) for name, shape in shapes.items() if ".lora_A." in name}
    if len(ranks) != 1:
        raise ValueError(f"stored adapter has no single LoRA rank: {sorted(ranks)}")
    targets = sorted({name.rsplit(".lora_", 1)[0].rsplit(".", 1)[-1] for name in shapes})
    return ranks.pop(), targets


def write_peft_dir(files_dir: str | Path, manifest: Mapping[str, Any], out_dir: str | Path, *,
                   base_model: str) -> dict[str, Any]:
    """Recompute ``policy_tensor_hash`` from the stored bytes, then write the PEFT dir. Needs torch."""
    from safetensors.torch import load_file, save_file

    from .export import POLICY_IDENTITY, POLICY_TENSORS, verify_policy_tensor_hash

    t0 = time.monotonic()
    verify_policy_tensor_hash(files_dir, manifest)  # raises EvalIntegrityError on mismatch
    root = Path(files_dir)
    identity = json.loads((root / POLICY_IDENTITY).read_text())
    tensors = load_file(str(root / POLICY_TENSORS))
    rank, targets = lora_rank_and_targets({k: tuple(v.shape) for k, v in tensors.items()})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapter_model.safetensors"))
    config = peft_adapter_config(rank=rank, targets=targets, base_model=base_model,
                                 revision=identity.get("base_model_revision"))
    (out / "adapter_config.json").write_text(json.dumps(config, indent=1, sort_keys=True))
    return {"rank": rank, "targets": targets, "tensors": len(tensors), "path": str(out),
            "seconds": round(time.monotonic() - t0, 3)}


# --- 2. SGLang + session server ----------------------------------------------------------


def _http(method: str, url: str, body: Any = None, *, timeout: float = 60.0) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
    try:
        return status, json.loads(raw) if raw else None
    except ValueError:
        return status, raw.decode("utf-8", "replace")


@dataclass
class InferConfig:
    """Serving settings; the defaults are the S17 M1 codex shape (qwen35_08b, LoRA r16 attention)."""

    base_model: str                      # HF id or local snapshot dir
    revision: str | None = None
    tito_model: str = "qwen35"
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"clear_thinking": False})
    context_length: int = 8192           # --agent-max-seq-len (the bridge's max_seq_len)
    max_tokens: int = 4096               # --rollout-max-response-len (YETO_CODEX_BACKEND_MAX_TOKENS)
    temperature: float = 1.0             # Miles rollout defaults
    top_p: float = 1.0
    top_k: int | None = None
    mem_fraction_static: float = 0.8
    host: str = "127.0.0.1"
    sglang_port: int = 30000
    session_port: int = 30100
    work_dir: str = "/tmp/yeto-eval-infer"
    sglang_extra_args: tuple[str, ...] = ()
    start_timeout_s: float = 900.0
    log_dir: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "InferConfig":
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        if "sglang_extra_args" in known:
            known["sglang_extra_args"] = tuple(known["sglang_extra_args"])
        return cls(**known)

    @property
    def sglang_url(self) -> str:
        return f"http://{self.host}:{self.sglang_port}"

    @property
    def session_url(self) -> str:
        return f"http://{self.host}:{self.session_port}"

    def request_kwargs(self) -> dict[str, Any]:
        out: dict[str, Any] = {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens}
        if self.top_k is not None:
            out["top_k"] = self.top_k
        return out


def sglang_argv(cfg: InferConfig, *, rank: int, targets: list[str], python: str = sys.executable) -> list[str]:
    argv = [python, "-m", "sglang.launch_server", "--model-path", cfg.base_model,
            "--host", cfg.host, "--port", str(cfg.sglang_port), "--trust-remote-code",
            "--context-length", str(cfg.context_length), "--mem-fraction-static", str(cfg.mem_fraction_static),
            "--enable-lora", "--max-lora-rank", str(rank), "--lora-target-modules", *targets,
            "--max-loaded-loras", "2", "--max-loras-per-batch", "1"]
    if cfg.revision and not Path(cfg.base_model).exists():
        argv += ["--revision", cfg.revision]
    return argv + list(cfg.sglang_extra_args)


def session_server_config(cfg: InferConfig, *, rank: int) -> dict[str, Any]:
    """``miles.rollout.session.config.SessionServerConfig`` fields, as the training island sets them
    (``compute_session_server_config``) for a LoRA codex run without replay / speculative decoding."""
    return {
        "host": cfg.host, "port": cfg.session_port, "instance_id": "eval-island",
        "backend_url": cfg.sglang_url, "timeout": 1800.0, "hf_checkpoint": cfg.base_model,
        "chat_template_path": None, "tito_model": cfg.tito_model,
        "apply_chat_template_kwargs": dict(cfg.chat_template_kwargs),
        "use_rollout_routing_replay": False, "use_rollout_indexer_replay": False,
        "use_sampling_support_replay": False, "sglang_speculative_algorithm": None,
        "num_layers": None, "moe_router_topk": None, "save_debug_trajectory_data": None,
        "lora_rank": int(rank), "lora_adapter_path": None, "lora_train_only": False,
        "use_session_server": True, "session_message_matcher": "strict", "pause_generation_mode": None,
        "session_sample_picker_path": None, "session_sample_postprocessor_path": None,
    }


SESSION_SERVER_BOOT = (
    "import json, sys\n"
    "from miles.rollout.session.config import SessionServerConfig\n"
    "from miles.rollout.session.server import run_session_server\n"
    "run_session_server(SessionServerConfig(**json.loads(sys.argv[1])))\n"
)


class SglangSessionLoader:
    """``PolicyLoader`` for ``EvalIsland``: one SGLang + one session server per island."""

    def __init__(self, cfg: InferConfig, *, popen: Callable[..., Any] = subprocess.Popen,
                 http: Callable[..., tuple[int, Any]] = _http, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, emit: Callable[..., None] | None = None) -> None:
        self.cfg = cfg
        self.popen = popen
        self.http = http
        self.sleep = sleep
        self.clock = clock
        self.emit = emit
        self.procs: dict[str, Any] = {}
        self.served: dict[str, Any] | None = None
        self.loads: list[dict[str, Any]] = []

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
                raise RuntimeError(f"{name} exited with {proc.returncode} before it was ready")
            try:
                status, _ = self.http("GET", url, timeout=10.0)
            except OSError:
                status = None
            if status == 200:
                return round(self.clock() - t0, 3)
            if self.clock() - t0 > self.cfg.start_timeout_s:
                raise TimeoutError(f"{name} not ready after {self.cfg.start_timeout_s} s ({url})")
            self.sleep(2.0)

    def _start(self, rank: int, targets: list[str]) -> dict[str, float]:
        self._spawn("sglang", sglang_argv(self.cfg, rank=rank, targets=targets))
        sglang_s = self._wait("sglang", f"{self.cfg.sglang_url}/health_generate")
        self._spawn("session-server", [sys.executable, "-c", SESSION_SERVER_BOOT,
                                       json.dumps(session_server_config(self.cfg, rank=rank))])
        session_s = self._wait("session-server", f"{self.cfg.session_url}/health")
        self.served = {"rank": rank, "targets": list(targets)}
        return {"sglang_start_s": sglang_s, "session_server_start_s": session_s}

    def load(self, manifest: Mapping[str, Any], files_dir: Path) -> str:
        version = int(manifest["policy_version"])
        t0 = self.clock()
        adapter = write_peft_dir(files_dir, manifest, Path(self.cfg.work_dir) / f"v{version:06d}",
                                 base_model=str(manifest.get("base_model") or self.cfg.base_model))
        record: dict[str, Any] = {"policy_version": version, "verify_and_write_s": adapter["seconds"],
                                  "rank": adapter["rank"], "targets": adapter["targets"]}
        if self.served is None:
            record.update(self._start(adapter["rank"], adapter["targets"]))
        elif (adapter["rank"], adapter["targets"]) != (self.served["rank"], self.served["targets"]):
            raise RuntimeError(f"v{version}: LoRA shape {adapter['rank']}/{adapter['targets']} differs from the "
                               f"served {self.served}; restart the island")
        else:
            self.http("POST", f"{self.cfg.sglang_url}/unload_lora_adapter", {"lora_name": SERVED_LORA_NAME})
        t1 = self.clock()
        status, body = self.http("POST", f"{self.cfg.sglang_url}/load_lora_adapter",
                                 {"lora_name": SERVED_LORA_NAME, "lora_path": adapter["path"]}, timeout=600.0)
        if status != 200 or (isinstance(body, dict) and body.get("success") is False):
            raise RuntimeError(f"v{version}: SGLang refused the adapter: HTTP {status} {str(body)[:300]}")
        record["adapter_load_s"] = round(self.clock() - t1, 3)
        record["load_total_s"] = round(self.clock() - t0, 3)
        self.loads.append(record)
        if self.emit is not None:
            self.emit("rl_eval_policy_load", **record)
        return str(manifest["rl/policy_token"])

    def close(self) -> None:
        for proc in self.procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in self.procs.values():
            try:
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                proc.kill()


# --- 3. the agent port --------------------------------------------------------------------


def _default_drive() -> Callable[..., Any]:
    from yeto.rl.harness.codex.codex_openenv_subprocess_agent_function import _drive_worker

    return _drive_worker


class CodexEvalAgent:
    """``agent(lease, task, trial, *, policy_token)`` for ``tb2_attempt``."""

    def __init__(self, cfg: InferConfig, *, drive: Callable[..., Any] | None = None,
                 http: Callable[..., tuple[int, Any]] = _http, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self.drive = drive
        self.http = http
        self.clock = clock

    async def _session(self) -> str:
        body = {"evaluation": True, "temperature": float(self.cfg.temperature), "top_p": float(self.cfg.top_p)}
        if self.cfg.top_k is not None:
            body["top_k"] = int(self.cfg.top_k)
        status, out = await asyncio.to_thread(self.http, "POST", f"{self.cfg.session_url}/sessions", body)
        if status != 200 or not isinstance(out, dict) or not out.get("session_id"):
            raise RuntimeError(f"session server refused a session: HTTP {status} {str(out)[:200]}")
        return str(out["session_id"])

    async def __call__(self, lease: Any, task: Any, trial: int, *, policy_token: str) -> dict[str, Any]:
        from yeto.rl.harness.codex import codex_openenv_agent_function as adapter
        from yeto.rl.harness.codex.codex_openenv_subprocess_agent_function import task_prompt

        drive = self.drive or _default_drive()
        session_id = await self._session()
        episode_id = adapter.new_episode_id()
        job = {
            "base_url": f"{self.cfg.session_url}/sessions/{session_id}",
            "prompt": task_prompt(dict(getattr(task, "row", {}).get("metadata") or {}), None, lease),
            "request_kwargs": self.cfg.request_kwargs(),
            "episode_id": episode_id,
            "max_seq_len": self.cfg.context_length,
            "env_url": lease.env_url,
            "env_token": lease.env_token,
            "driver": "stock",
        }
        trajectory = f"eval-{task.task_id}-t{trial}-{episode_id}"
        t0 = self.clock()
        try:
            try:
                untrusted = await asyncio.wait_for(
                    drive(job, trajectory, None, getattr(lease, "worker_env", None)),
                    timeout=float(lease.deadline_seconds))
            except asyncio.TimeoutError:
                untrusted = {"status": "timeout", "metrics": {"timed_out": 1}, "episode_id": episode_id}
            except adapter.harness.CodexHarnessError as exc:  # training: infrastructure, not a reward
                return {"episode_id": episode_id, "end_reason": "infra_error", "error": str(exc)[:500],
                        "metrics": getattr(exc, "metrics", None), "agent_s": round(self.clock() - t0, 3)}
        finally:
            await asyncio.to_thread(self.http, "DELETE", f"{self.cfg.session_url}/sessions/{session_id}")
        metrics = dict(untrusted.get("metrics") or {})
        status = str(untrusted.get("status"))
        if status not in STATUS_TO_END_REASON:
            return {"episode_id": episode_id, "end_reason": "infra_error", "error": f"unknown status {status!r}"}
        return {"episode_id": untrusted.get("episode_id", episode_id), "end_reason": STATUS_TO_END_REASON[status],
                "turns": metrics.get("turns"), "tokens": metrics.get("max_model_total_tokens"),
                "metrics": metrics, "agent_s": round(self.clock() - t0, 3)}


# --- island_main factories ----------------------------------------------------------------

_LOADERS: dict[int, SglangSessionLoader] = {}


def loader_factory(config: Mapping[str, Any]) -> SglangSessionLoader:
    cfg = InferConfig.from_mapping(config.get("infer") or {})
    loader = SglangSessionLoader(cfg)
    _LOADERS[id(config)] = loader
    return loader


def attempt_factory(config: Mapping[str, Any]) -> Callable[..., Mapping[str, Any]]:
    """``tb2_attempt`` over the Modal reward-env provider (prebaked images) and :class:`CodexEvalAgent`."""
    from .tb2_attempt import tb2_attempt

    provider_spec = config.get("provider") or os.environ.get("YETO_HARNESS_ENVIRONMENT_PROVIDER") \
        or "yeto.cloud.modal_reward_env:modal_provider"
    module, _, name = provider_spec.partition(":")
    import importlib

    provider = getattr(importlib.import_module(module), name)(None)
    return tb2_attempt(provider, CodexEvalAgent(InferConfig.from_mapping(config.get("infer") or {})))
