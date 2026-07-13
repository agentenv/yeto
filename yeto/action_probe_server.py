"""Persistent multi-GPU localhost service for online action probing.

This module intentionally has no syncer or experiment-runner integration.
Run it independently with ``python -m yeto.action_probe_server``; a future
Rust client can use the framing helpers in :mod:`yeto.action_probe`.

Each spawned worker owns one GPU, one model replica, and one immutable copy
of the anchor panels. CUDA is initialized only in ``spawn`` children. The
parent validates requests, fans the five exact actions out to workers, joins
their per-panel losses, applies the paired-LCB rule against A0, and falls back
to A0 on every error path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import ipaddress
import json
import logging
import multiprocessing as mp
import os
import signal
import socket
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any

import torch

from .action_probe import (
    ACTION_NAMES,
    BASELINE_ACTION,
    LEAVE_ONE_OUT_ACTION_FAMILY,
    PROTOCOL,
    STEP_SCALE_ACTION_FAMILY,
    SUPPORTED_ACTION_FAMILIES,
    ActionProbeReplica,
    CttnRequest,
    EvaluateRequest,
    EvaluationError,
    Frame,
    ProtocolError,
    RequestValidationError,
    SelectionConfig,
    build_anchor_panels,
    build_cttn_result_frame,
    build_cttn_shadow_result_frame,
    decode_frame,
    load_anchor_manifest,
    parse_cttn_request,
    parse_evaluate_request,
    probe_config_digest,
    recv_frame,
    select_paired_lcb,
    send_frame,
)


log = logging.getLogger("action-probe")

ACTION_PROBE_DECISION_PREFIX = "ACTION_PROBE_DECISION "


def _log_successful_decision(
    response: dict[str, Any], selection: SelectionConfig
) -> None:
    """Emit one canonical audit record for a successful evaluation response."""

    if response.get("type") != "evaluate_result" or response.get("ok") is not True:
        return
    evidence = dict(response)
    evidence["selection_config"] = asdict(selection)
    encoded = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    log.info("%s%s", ACTION_PROBE_DECISION_PREFIX, encoded)


@dataclass(frozen=True)
class ReplicaConfig:
    gpu_id: int
    model: str
    anchor_manifest: str
    seq_len: int = 128
    panels: int = 8
    blocks_per_panel: int = 2
    train_on: str = "assistant"
    loss_function: str = "cross_entropy"
    lora_r: int = 2
    lora_alpha: int = 4
    lora_targets: str = "auto"
    fragments: int = 4
    fragment_pattern: str = "binpack"
    cache_dir: str | None = None
    selection: SelectionConfig = SelectionConfig()


def _probe_config_sha256(
    static_config: dict[str, Any], selection: SelectionConfig
) -> str:
    bound_config = dict(static_config)
    bound_config["selection"] = asdict(selection)
    return probe_config_digest(bound_config)


def _load_replica(config: ReplicaConfig) -> ActionProbeReplica:
    """Load a base model, PEFT adapter, and static anchors once in a child."""

    import peft
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .learner import (
        _from_pretrained_offline_first,
        accelerator_model_dtype,
        resolve_lora_targets,
    )
    from .fragments import build_layout
    from .models import resolve

    if not torch.cuda.is_available():
        raise EvaluationError("CUDA is unavailable in action-probe worker")
    if config.gpu_id < 0 or config.gpu_id >= torch.cuda.device_count():
        raise EvaluationError(
            f"GPU {config.gpu_id} is outside visible device count {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(config.gpu_id)
    device = torch.device("cuda", config.gpu_id)
    model_id = resolve(config.model)
    load_options: dict[str, Any] = {"trust_remote_code": True}
    if config.cache_dir:
        load_options["cache_dir"] = config.cache_dir
    tokenizer = _from_pretrained_offline_first(AutoTokenizer, model_id, **load_options)
    model = _from_pretrained_offline_first(
        AutoModelForCausalLM,
        model_id,
        torch_dtype=accelerator_model_dtype(device),
        attn_implementation="eager",
        **load_options,
    )
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=resolve_lora_targets(config.lora_targets, model.config),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    manifest = load_anchor_manifest(config.anchor_manifest)
    panels, panel_digest = build_anchor_panels(
        manifest,
        tokenizer,
        seq_len=config.seq_len,
        panels=config.panels,
        blocks_per_panel=config.blocks_per_panel,
        train_on=config.train_on,
        device=device,
    )
    trainable_schema = [
        {
            "name": name,
            "shape": list(param.shape),
            "dtype": str(param.dtype).replace("torch.", ""),
        }
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    trainable_params = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }
    layout = build_layout(
        [(name, int(param.numel())) for name, param in trainable_params.items()],
        config.fragments,
        config.fragment_pattern,
    )
    fragment_layout = {
        fragment_id: [name for name, _numel in fragment.tensors]
        for fragment_id, fragment in enumerate(layout.fragments)
    }
    layout_contract = {
        "pattern": config.fragment_pattern,
        "fragments": [
            {
                "id": fragment_id,
                "merge_mode": int(fragment.merge_mode),
                "tensors": [
                    {
                        "name": name,
                        "shape": list(trainable_params[name].shape),
                        "numel": int(numel),
                    }
                    for name, numel in fragment.tensors
                ],
            }
            for fragment_id, fragment in enumerate(layout.fragments)
        ],
    }
    layout_digest = probe_config_digest(layout_contract)
    static_config = {
        "protocol": PROTOCOL,
        "resolved_model": model_id,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "chat_template_sha256": hashlib.sha256(
            (getattr(tokenizer, "chat_template", None) or "").encode("utf-8")
        ).hexdigest(),
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_targets": config.lora_targets,
        "trainable_schema": trainable_schema,
        "fragment_layout_sha256": layout_digest,
        "anchor_manifest_sha256": manifest.manifest_sha256,
        "anchor_tensors_sha256": panel_digest,
        "seq_len": config.seq_len,
        "panels": config.panels,
        "blocks_per_panel": config.blocks_per_panel,
        "train_on": config.train_on,
        "loss_function": config.loss_function,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
    }
    config_digest = _probe_config_sha256(static_config, config.selection)
    return ActionProbeReplica(
        model,
        panels,
        anchor_manifest_sha256=manifest.manifest_sha256,
        anchor_tensors_sha256=panel_digest,
        probe_config_sha256=config_digest,
        layout_hash=layout_digest,
        fragment_layout=fragment_layout,
        device=device,
        loss_function=config.loss_function,
    )


def _worker_main(connection: Connection, config: ReplicaConfig) -> None:
    replica: ActionProbeReplica | None = None
    try:
        replica = _load_replica(config)
        connection.send(
            {
                "type": "ready",
                "gpu_id": config.gpu_id,
                "pid": os.getpid(),
                "model": config.model,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "lora_targets": config.lora_targets,
                "anchor_manifest_sha256": replica.anchor_manifest_sha256,
                "anchor_tensors_sha256": replica.anchor_tensors_sha256,
                "probe_config_sha256": replica.probe_config_sha256,
                "panel_count": len(replica.panels),
                "layout_hash": replica.layout_hash,
                "fragment_layout": {
                    str(fragment_id): list(names)
                    for fragment_id, names in replica.fragment_layout.items()
                },
                "trainable_tensors": len(replica.params),
                "trainable_numel": sum(
                    param.numel() for param in replica.params.values()
                ),
            }
        )
    except Exception as exc:
        connection.send(
            {
                "type": "startup_error",
                "gpu_id": config.gpu_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=20),
            }
        )
        connection.close()
        return

    while True:
        try:
            message = connection.recv()
        except EOFError:
            break
        if not isinstance(message, dict):
            continue
        if message.get("op") == "shutdown":
            break
        op = message.get("op")
        if op not in ("evaluate", "cttn_step"):
            continue
        request_digest = str(message.get("request_digest", ""))
        try:
            frame = Frame(
                header=message["header"],
                payload=message["payload"],
                digest=request_digest,
            )
            request = (
                parse_cttn_request(frame)
                if op == "cttn_step"
                else parse_evaluate_request(frame)
            )
            if request.request_digest != request_digest:
                raise ProtocolError("worker request digest changed during dispatch")
            if op == "cttn_step":
                result = replica.cttn_step(request)
            else:
                actions = tuple(message["actions"])
                result = replica.evaluate(request, actions)
            connection.send(
                {
                    "type": "result",
                    "gpu_id": config.gpu_id,
                    "request_digest": request_digest,
                    "result": result,
                }
            )
        except Exception as exc:
            # ActionProbeReplica restores the complete current state in its
            # finally block. Any restoration failure is still reported and
            # causes an A0 fallback in the parent.
            tainted = not isinstance(exc, (ProtocolError, RequestValidationError))
            connection.send(
                {
                    "type": "error",
                    "gpu_id": config.gpu_id,
                    "request_digest": request_digest,
                    "error": f"{type(exc).__name__}: {exc}",
                    "tainted": tainted,
                    "traceback": traceback.format_exc(limit=20),
                }
            )
            if tainted:
                # A CUDA exception can leave allocator or kernel state
                # unusable, and an exception during restoration means the
                # model is known dirty. Never return this replica to the pool.
                break
    connection.close()


@dataclass
class _Worker:
    config: ReplicaConfig
    process: mp.Process
    connection: Connection
    ready: dict[str, Any]


class WorkerPoolBackend:
    """Persistent spawned workers implementing the engine backend contract."""

    def __init__(
        self,
        configs: list[ReplicaConfig],
        *,
        startup_timeout_s: float = 1800.0,
        request_timeout_s: float = 30.0,
    ):
        if not configs:
            raise ValueError("at least one replica config is required")
        if len(configs) > len(ACTION_NAMES) - 1:
            raise ValueError("at most four GPU replicas are useful for A1-A4 pairing")
        if startup_timeout_s <= 0 or request_timeout_s <= 0:
            raise ValueError("worker timeouts must be positive")
        replica_shapes = {
            (
                config.model,
                config.anchor_manifest,
                config.seq_len,
                config.panels,
                config.blocks_per_panel,
                config.train_on,
                config.loss_function,
                config.lora_r,
                config.lora_alpha,
                config.lora_targets,
                config.fragments,
                config.fragment_pattern,
                config.selection,
            )
            for config in configs
        }
        if len(replica_shapes) != 1:
            raise ValueError("all action-probe GPU replicas must use identical configs")
        gpu_ids = [config.gpu_id for config in configs]
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("action-probe GPU assignments must be unique")
        self.request_timeout_s = request_timeout_s
        self._context = mp.get_context("spawn")
        self._workers: list[_Worker] = []
        self._healthy = True
        try:
            pending = []
            for config in configs:
                parent, child = self._context.Pipe(duplex=True)
                process = self._context.Process(
                    target=_worker_main,
                    args=(child, config),
                    name=f"action-probe-gpu-{config.gpu_id}",
                    daemon=True,
                )
                process.start()
                child.close()
                pending.append((config, process, parent))
            deadline = time.monotonic() + startup_timeout_s
            for config, process, connection in pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not connection.poll(remaining):
                    raise EvaluationError(
                        f"GPU {config.gpu_id} worker startup timed out"
                    )
                ready = connection.recv()
                if ready.get("type") != "ready":
                    raise EvaluationError(
                        f"GPU {config.gpu_id} worker failed startup: {ready.get('error')}"
                    )
                self._workers.append(_Worker(config, process, connection, ready))
        except Exception:
            for _config, process, connection in locals().get("pending", []):
                connection.close()
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
            raise

        manifest_digests = {
            worker.ready["anchor_manifest_sha256"] for worker in self._workers
        }
        tensor_digests = {
            worker.ready["anchor_tensors_sha256"] for worker in self._workers
        }
        config_digests = {
            worker.ready["probe_config_sha256"] for worker in self._workers
        }
        panel_counts = {worker.ready["panel_count"] for worker in self._workers}
        layout_hashes = {worker.ready["layout_hash"] for worker in self._workers}
        fragment_layouts = {
            json.dumps(worker.ready["fragment_layout"], sort_keys=True)
            for worker in self._workers
        }
        tensor_counts = {
            (worker.ready["trainable_tensors"], worker.ready["trainable_numel"])
            for worker in self._workers
        }
        if (
            len(manifest_digests) != 1
            or len(tensor_digests) != 1
            or len(config_digests) != 1
            or len(panel_counts) != 1
            or len(layout_hashes) != 1
            or len(fragment_layouts) != 1
            or len(tensor_counts) != 1
        ):
            self.close(force=True)
            raise EvaluationError(
                "action-probe replicas disagree on anchor or LoRA layout digests"
            )
        self.anchor_manifest_sha256 = manifest_digests.pop()
        self.anchor_tensors_sha256 = tensor_digests.pop()
        self.probe_config_sha256 = config_digests.pop()
        self.panel_count = panel_counts.pop()
        self.layout_hash = layout_hashes.pop()
        self.fragment_layout = json.loads(fragment_layouts.pop())
        self.gpu_ids = tuple(worker.config.gpu_id for worker in self._workers)
        self.trainable_tensors, self.trainable_numel = tensor_counts.pop()

    @property
    def healthy(self) -> bool:
        return self._healthy and all(
            worker.process.is_alive() for worker in self._workers
        )

    def describe(self) -> dict[str, Any]:
        return {
            "gpus": list(self.gpu_ids),
            "workers": [worker.ready for worker in self._workers],
            "anchor_manifest_sha256": self.anchor_manifest_sha256,
            "anchor_tensors_sha256": self.anchor_tensors_sha256,
            "probe_config_sha256": self.probe_config_sha256,
            "panel_count": self.panel_count,
            "layout_hash": self.layout_hash,
            "fragment_layout": self.fragment_layout,
            "trainable_tensors": self.trainable_tensors,
            "trainable_numel": self.trainable_numel,
            "healthy": self.healthy,
        }

    def _poison(self, reason: str) -> None:
        if not self._healthy:
            return
        self._healthy = False
        log.error("action-probe worker pool is unhealthy: %s", reason)
        # A timed-out CUDA kernel cannot be cancelled through a Pipe. Kill all
        # replicas so neither late results nor possibly contaminated state can
        # enter a later request. The service remains alive and fails closed;
        # an operator may restart it to reload clean model replicas.
        for worker in self._workers:
            if worker.process.is_alive():
                worker.process.terminate()

    def evaluate(self, frame: Frame, request: EvaluateRequest) -> dict[str, Any]:
        if not self.healthy:
            raise EvaluationError("action-probe worker pool is unhealthy")
        if request.anchor_manifest_sha256 != self.anchor_manifest_sha256:
            raise RequestValidationError(
                "request anchor manifest does not match worker pool"
            )
        if request.probe_config_sha256 != self.probe_config_sha256:
            raise RequestValidationError(
                "request probe configuration does not match worker pool"
            )
        if request.layout_hash != self.layout_hash:
            raise RequestValidationError(
                "request fragment layout does not match worker pool"
            )

        # Every alternative is paired with an A0 evaluation on the same
        # replica. This removes constant cross-GPU numerical offsets from the
        # panel gains while retaining two parallel inference waves on 4 GPUs.
        assignments: list[list[str]] = [[] for _ in self._workers]
        for index, action in enumerate(ACTION_NAMES[1:]):
            worker_actions = assignments[index % len(self._workers)]
            if BASELINE_ACTION not in worker_actions:
                worker_actions.append(BASELINE_ACTION)
            worker_actions.append(action)
        active: dict[Connection, _Worker] = {}
        dispatch_started = time.perf_counter()
        deadline = time.monotonic() + self.request_timeout_s
        dispatch_errors: list[tuple[_Worker, Exception]] = []
        dispatch_threads = []

        def dispatch(worker: _Worker, actions: list[str]) -> None:
            try:
                worker.connection.send(
                    {
                        "op": "evaluate",
                        "header": frame.header,
                        "payload": frame.payload,
                        "request_digest": frame.digest,
                        "actions": actions,
                    }
                )
            except (BrokenPipeError, EOFError, OSError) as exc:
                dispatch_errors.append((worker, exc))

        for worker, actions in zip(self._workers, assignments):
            if not actions:
                continue
            active[worker.connection] = worker
            thread = threading.Thread(
                target=dispatch, args=(worker, actions), daemon=True
            )
            thread.start()
            dispatch_threads.append(thread)

        for thread in dispatch_threads:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                thread.join(remaining)
        if any(thread.is_alive() for thread in dispatch_threads):
            self._poison("IPC dispatch timeout")
            for thread in dispatch_threads:
                thread.join(timeout=1)
            raise EvaluationError("action-probe IPC dispatch timed out")
        if dispatch_errors:
            worker, exc = dispatch_errors[0]
            self._poison(f"GPU {worker.config.gpu_id} dispatch failed")
            raise EvaluationError(
                f"GPU {worker.config.gpu_id} worker dispatch failed"
            ) from exc

        worker_results = []
        worker_errors = []
        while active:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._poison("request timeout")
                raise EvaluationError("action-probe worker request timed out")
            ready_connections = wait(list(active), timeout=remaining)
            if not ready_connections:
                self._poison("request timeout")
                raise EvaluationError("action-probe worker request timed out")
            for connection in ready_connections:
                worker = active.pop(connection)
                try:
                    message = connection.recv()
                except (EOFError, OSError) as exc:
                    self._poison(f"GPU {worker.config.gpu_id} disconnected")
                    raise EvaluationError(
                        f"GPU {worker.config.gpu_id} worker disconnected"
                    ) from exc
                if message.get("request_digest") != frame.digest:
                    self._poison(
                        f"GPU {worker.config.gpu_id} returned a stale response"
                    )
                    raise EvaluationError(
                        "worker returned a stale or mismatched response"
                    )
                if message.get("type") != "result":
                    worker_errors.append((worker, message))
                    continue
                worker_results.append(message)

        if worker_errors:
            tainted = any(
                bool(message.get("tainted", True)) for _, message in worker_errors
            )
            if tainted:
                self._poison("tainted GPU evaluation failure")
            details = "; ".join(
                f"GPU {worker.config.gpu_id}: {message.get('error')}"
                for worker, message in worker_errors
            )
            error_type = EvaluationError if tainted else RequestValidationError
            raise error_type(f"action-probe worker request failed: {details}")

        actions: dict[str, Any] = {}
        baseline_copies: list[dict[str, Any]] = []
        paired_baselines: dict[str, list[float]] = {}
        state_digests = set()
        manifest_digests = set()
        tensor_digests = set()
        config_digests = set()
        per_worker = []
        for message in worker_results:
            result = message["result"]
            if result.get("state_restored") is not True:
                self._poison("worker did not confirm exact state restoration")
                raise EvaluationError(
                    "worker did not restore the complete current state"
                )
            state_digests.add(result.get("state_sha256"))
            manifest_digests.add(result.get("anchor_manifest_sha256"))
            tensor_digests.add(result.get("anchor_tensors_sha256"))
            config_digests.add(result.get("probe_config_sha256"))
            result_actions = result.get("actions", {})
            worker_index = self.gpu_ids.index(message["gpu_id"])
            expected_actions = set(assignments[worker_index])
            if set(result_actions) != expected_actions:
                self._poison("worker returned an unexpected action set")
                raise EvaluationError(
                    f"GPU {message['gpu_id']} returned {sorted(result_actions)}, "
                    f"expected {sorted(expected_actions)}"
                )
            baseline_result = result_actions[BASELINE_ACTION]
            if (
                baseline_result.get("trial_sha256")
                != request.action_digests[BASELINE_ACTION]
            ):
                self._poison("worker A0 trial digest mismatch")
                raise EvaluationError("worker A0 trial digest mismatch")
            baseline_copies.append(baseline_result)
            for action, action_result in result_actions.items():
                if action == BASELINE_ACTION:
                    continue
                if action_result.get("trial_sha256") != request.action_digests[action]:
                    self._poison(f"worker {action} trial digest mismatch")
                    raise EvaluationError(f"worker {action} trial digest mismatch")
                if action in actions:
                    self._poison(f"duplicate worker result for {action}")
                    raise EvaluationError(f"duplicate worker result for {action}")
                actions[action] = action_result
                paired_baselines[action] = list(baseline_result["panel_losses"])
            per_worker.append(
                {
                    "gpu_id": message["gpu_id"],
                    "restore_ms": result.get("restore_ms"),
                    "total_ms": result.get("total_ms"),
                    "actions": sorted(result.get("actions", {})),
                }
            )
        if set(actions) != set(ACTION_NAMES[1:]):
            self._poison("worker result action set is incomplete")
            raise EvaluationError(
                f"worker results have actions {sorted(actions)}, "
                f"expected {list(ACTION_NAMES[1:])}"
            )
        if state_digests != {request.current_state_digest}:
            self._poison("worker restored-state digests disagree")
            raise EvaluationError("worker restored-state digests disagree")
        if manifest_digests != {self.anchor_manifest_sha256}:
            self._poison("worker manifest digests disagree")
            raise EvaluationError("worker manifest digests disagree")
        if tensor_digests != {self.anchor_tensors_sha256}:
            self._poison("worker anchor tensor digests disagree")
            raise EvaluationError("worker anchor tensor digests disagree")
        if config_digests != {self.probe_config_sha256}:
            self._poison("worker probe configuration digests disagree")
            raise EvaluationError("worker probe configuration digests disagree")
        baseline_panel_counts = {
            len(result.get("panel_losses", [])) for result in baseline_copies
        }
        if baseline_panel_counts != {self.panel_count}:
            self._poison("worker A0 panel counts are inconsistent")
            raise EvaluationError("worker A0 panel counts are inconsistent")
        averaged_baseline_losses = [
            sum(float(result["panel_losses"][panel]) for result in baseline_copies)
            / len(baseline_copies)
            for panel in range(self.panel_count)
        ]
        actions[BASELINE_ACTION] = {
            "panel_losses": averaged_baseline_losses,
            "trial_sha256": request.action_digests[BASELINE_ACTION],
            "replica_count": len(baseline_copies),
        }
        panel_counts = {
            len(result.get("panel_losses", [])) for result in actions.values()
        }
        if panel_counts != {self.panel_count}:
            self._poison("worker panel counts are inconsistent")
            raise EvaluationError(
                "worker actions returned inconsistent or insufficient panels"
            )
        return {
            "actions": actions,
            "paired_baseline_losses": paired_baselines,
            "workers": per_worker,
            "dispatch_total_ms": (time.perf_counter() - dispatch_started) * 1000.0,
            "state_sha256": request.current_state_digest,
            "anchor_manifest_sha256": self.anchor_manifest_sha256,
            "anchor_tensors_sha256": self.anchor_tensors_sha256,
            "probe_config_sha256": self.probe_config_sha256,
        }

    def cttn_step(self, frame: Frame, request: CttnRequest) -> dict[str, Any]:
        """Run the HVP-heavy CTTN verb on exactly one persistent replica."""

        if not self.healthy:
            raise EvaluationError("action-probe worker pool is unhealthy")
        if request.anchor_manifest_sha256 != self.anchor_manifest_sha256:
            raise RequestValidationError(
                "request anchor manifest does not match worker pool"
            )
        if request.probe_config_sha256 != self.probe_config_sha256:
            raise RequestValidationError(
                "request probe configuration does not match worker pool"
            )
        if request.layout_hash != self.layout_hash:
            raise RequestValidationError(
                "request fragment layout does not match worker pool"
            )

        worker = self._workers[0]
        dispatch_started = time.perf_counter()
        dispatch_error: list[Exception] = []

        def dispatch() -> None:
            try:
                worker.connection.send(
                    {
                        "op": "cttn_step",
                        "header": frame.header,
                        "payload": frame.payload,
                        "request_digest": frame.digest,
                    }
                )
            except (BrokenPipeError, EOFError, OSError) as exc:
                dispatch_error.append(exc)

        thread = threading.Thread(target=dispatch, daemon=True)
        thread.start()
        deadline = time.monotonic() + self.request_timeout_s
        thread.join(max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            self._poison("CTTN IPC dispatch timeout")
            thread.join(timeout=1)
            raise EvaluationError("action-probe CTTN IPC dispatch timed out")
        if dispatch_error:
            self._poison(f"GPU {worker.config.gpu_id} CTTN dispatch failed")
            raise EvaluationError(
                f"GPU {worker.config.gpu_id} worker CTTN dispatch failed"
            ) from dispatch_error[0]

        remaining = deadline - time.monotonic()
        if remaining <= 0 or not wait([worker.connection], timeout=remaining):
            self._poison("CTTN request timeout")
            raise EvaluationError("action-probe CTTN request timed out")
        try:
            message = worker.connection.recv()
        except (EOFError, OSError) as exc:
            self._poison(f"GPU {worker.config.gpu_id} disconnected during CTTN")
            raise EvaluationError(
                f"GPU {worker.config.gpu_id} worker disconnected during CTTN"
            ) from exc
        if message.get("request_digest") != frame.digest:
            self._poison(f"GPU {worker.config.gpu_id} returned a stale CTTN response")
            raise EvaluationError("worker returned a stale or mismatched CTTN response")
        if message.get("type") != "result":
            if bool(message.get("tainted", True)):
                self._poison("tainted GPU CTTN failure")
                error_type = EvaluationError
            else:
                error_type = RequestValidationError
            raise error_type(
                f"action-probe CTTN worker failed: GPU {worker.config.gpu_id}: "
                f"{message.get('error')}"
            )
        result = message["result"]
        if result.get("state_sha256") != request.current_state_digest:
            self._poison("CTTN worker state digest mismatch")
            raise EvaluationError("CTTN worker returned a mismatched state digest")
        if result.get("state_restored") is not True:
            self._poison("CTTN worker did not restore the complete state")
            raise EvaluationError("CTTN worker did not restore the complete state")
        for field, expected in (
            ("anchor_manifest_sha256", self.anchor_manifest_sha256),
            ("anchor_tensors_sha256", self.anchor_tensors_sha256),
            ("probe_config_sha256", self.probe_config_sha256),
        ):
            if result.get(field) != expected:
                self._poison(f"CTTN worker {field} mismatch")
                raise EvaluationError(f"CTTN worker returned a mismatched {field}")
        return {
            **result,
            "gpu_id": worker.config.gpu_id,
            "dispatch_total_ms": (time.perf_counter() - dispatch_started) * 1000.0,
        }

    def close(self, *, force: bool = False) -> None:
        for worker in self._workers:
            if worker.process.is_alive() and not force:
                try:
                    worker.connection.send({"op": "shutdown"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for worker in self._workers:
            worker.process.join(timeout=10 if not force else 1)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=5)
            if worker.process.is_alive() and hasattr(worker.process, "kill"):
                worker.process.kill()
                worker.process.join(timeout=5)
            worker.connection.close()
        self._healthy = False


def _error_response(
    frame: Frame,
    error: Exception | str,
    *,
    error_code: str = "evaluation_error",
    elapsed_ms: float | None = None,
    action_family: str | None = None,
) -> dict[str, Any]:
    message = str(error)
    request_id = frame.header.get("request_id")
    run_uuid = frame.header.get("run_uuid")
    if action_family is None:
        fragment = frame.header.get("fragment")
        declared_family = (
            fragment.get("action_family", LEAVE_ONE_OUT_ACTION_FAMILY)
            if isinstance(fragment, dict)
            else LEAVE_ONE_OUT_ACTION_FAMILY
        )
        action_family = (
            declared_family
            if declared_family in SUPPORTED_ACTION_FAMILIES
            else LEAVE_ONE_OUT_ACTION_FAMILY
        )
    return {
        "protocol": PROTOCOL,
        "type": "evaluate_result",
        "request_id": request_id if isinstance(request_id, str) else None,
        "run_uuid": run_uuid if isinstance(run_uuid, str) else None,
        "action_family": action_family,
        "request_digest": frame.digest,
        "ok": False,
        "fail_closed": True,
        "selected_action": BASELINE_ACTION,
        "fallback_reason": error_code,
        "error": message[:2048],
        "cache_hit": False,
        "timings_ms": {"total": elapsed_ms} if elapsed_ms is not None else {},
    }


def _cttn_error_response(
    frame: Frame,
    error: Exception | str,
    *,
    error_code: str,
    elapsed_ms: float,
) -> dict[str, Any]:
    header = frame.header
    cttn = header.get("cttn")
    shadow = isinstance(cttn, dict) and cttn.get("mode") == "shadow"
    return {
        "protocol": PROTOCOL,
        "type": "cttn_shadow_result" if shadow else "cttn_result",
        "request_id": header.get("request_id"),
        "run_uuid": header.get("run_uuid"),
        "request_digest": frame.digest,
        "ok": False,
        "fallback_reason": error_code,
        "error": str(error)[:2048],
        "timings_ms": {"total": elapsed_ms},
    }


class ActionProbeEngine:
    """Request validation, exact-retry cache, backend join, and selection."""

    def __init__(
        self,
        backend,
        *,
        selection: SelectionConfig = SelectionConfig(),
        retry_cache_size: int = 2,
    ):
        if retry_cache_size < 0:
            raise ValueError("retry_cache_size must be >= 0")
        self.backend = backend
        self.selection = selection
        self.retry_cache_size = retry_cache_size
        self._cache: OrderedDict[tuple[str, str], tuple[str, dict[str, Any]]] = (
            OrderedDict()
        )

    def describe(self) -> dict[str, Any]:
        backend = self.backend.describe() if hasattr(self.backend, "describe") else {}
        return {
            "protocol": PROTOCOL,
            "supported_operations": ["evaluate", "cttn_step"],
            "supported_action_families": list(SUPPORTED_ACTION_FAMILIES),
            "selection": asdict(self.selection),
            "retry_cache_size": self.retry_cache_size,
            "backend": backend,
        }

    def _cache_response(
        self,
        key: tuple[str, str],
        digest: str,
        response: dict[str, Any],
    ) -> None:
        if self.retry_cache_size == 0:
            return
        self._cache[key] = (digest, copy.deepcopy(response))
        self._cache.move_to_end(key)
        while len(self._cache) > self.retry_cache_size:
            self._cache.popitem(last=False)

    def handle(self, frame: Frame) -> dict[str, Any] | tuple[dict[str, Any], bytes]:
        started = time.perf_counter()
        request: EvaluateRequest | None = None
        key: tuple[str, str] | None = None
        if (
            frame.header.get("protocol") == PROTOCOL
            and frame.header.get("type") == "ping"
            and not frame.payload
        ):
            return {
                "protocol": PROTOCOL,
                "type": "pong",
                "ok": True,
                "request_digest": frame.digest,
                "cache_hit": False,
                "supported_action_families": list(SUPPORTED_ACTION_FAMILIES),
                "service": self.describe(),
            }

        if frame.header.get("protocol") == PROTOCOL and frame.header.get("type") == "cttn_step":
            try:
                request_started = time.perf_counter()
                cttn_request = parse_cttn_request(frame)
                parse_ms = (time.perf_counter() - request_started) * 1000.0
                if cttn_request.anchor_manifest_sha256 != self.backend.anchor_manifest_sha256:
                    raise RequestValidationError(
                        "request anchor manifest SHA-256 does not match service"
                    )
                if cttn_request.probe_config_sha256 != self.backend.probe_config_sha256:
                    raise RequestValidationError(
                        "request probe configuration SHA-256 does not match service"
                    )
                if cttn_request.layout_hash != self.backend.layout_hash:
                    raise RequestValidationError(
                        "request fragment layout SHA-256 does not match service"
                    )
                backend_result = self.backend.cttn_step(frame, cttn_request)
                if cttn_request.mode == "shadow":
                    encoded = build_cttn_shadow_result_frame(
                        cttn_request,
                        backend_result["z_matrix"],
                        backend_result["z_scalar"],
                        backend_result["diagnostics"],
                        anchor_tensors_sha256=backend_result[
                            "anchor_tensors_sha256"
                        ],
                    )
                else:
                    encoded = build_cttn_result_frame(
                        cttn_request,
                        backend_result["d"],
                        backend_result["b_new"],
                        backend_result["diagnostics"],
                        anchor_tensors_sha256=backend_result[
                            "anchor_tensors_sha256"
                        ],
                    )
                response = decode_frame(encoded)
                response.header["timings_ms"] = {
                    "parse": parse_ms,
                    "dispatch_and_eval": backend_result.get("dispatch_total_ms"),
                    "total": (time.perf_counter() - started) * 1000.0,
                }
                return response.header, response.payload
            except Exception as exc:
                log.exception("action-probe CTTN request failed closed")
                if isinstance(exc, ProtocolError):
                    code = "protocol_error"
                elif isinstance(exc, RequestValidationError):
                    code = "request_validation_error"
                else:
                    code = "evaluation_error"
                return (
                    _cttn_error_response(
                        frame,
                        exc,
                        error_code=code,
                        elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    ),
                    b"",
                )

        try:
            parse_started = time.perf_counter()
            request = parse_evaluate_request(frame)
            parse_ms = (time.perf_counter() - parse_started) * 1000.0
            key = (request.run_uuid, request.request_id)
            cached = self._cache.get(key)
            if cached is not None:
                cached_digest, cached_response = cached
                if cached_digest != frame.digest:
                    raise ProtocolError(
                        "cached request_id was reused with different header or payload bytes"
                    )
                self._cache.move_to_end(key)
                response = copy.deepcopy(cached_response)
                response["cache_hit"] = True
                response["retry_lookup_ms"] = (time.perf_counter() - started) * 1000.0
                _log_successful_decision(response, self.selection)
                return response
            if request.anchor_manifest_sha256 != self.backend.anchor_manifest_sha256:
                raise RequestValidationError(
                    "request anchor manifest SHA-256 does not match service"
                )
            if request.probe_config_sha256 != self.backend.probe_config_sha256:
                raise RequestValidationError(
                    "request probe configuration SHA-256 does not match service"
                )
            backend_layout_hash = getattr(
                self.backend, "layout_hash", request.layout_hash
            )
            if request.layout_hash != backend_layout_hash:
                raise RequestValidationError(
                    "request fragment layout SHA-256 does not match service"
                )

            backend_result = self.backend.evaluate(frame, request)
            for action in ACTION_NAMES:
                result_digest = backend_result["actions"][action].get("trial_sha256")
                if result_digest != request.action_digests[action]:
                    raise EvaluationError(
                        f"backend returned a mismatched trial digest for {action}"
                    )
            losses = {
                action: backend_result["actions"][action]["panel_losses"]
                for action in ACTION_NAMES
            }
            eligible_actions = [
                action
                for action in ACTION_NAMES[1:]
                if request.action_eligibility[action]
            ]
            selected = select_paired_lcb(
                losses,
                self.selection,
                eligible_actions=eligible_actions,
                baseline_losses_by_action=backend_result.get("paired_baseline_losses"),
                action_multipliers=(
                    {
                        action: request.action_metadata[action]["step_scale"]
                        for action in ACTION_NAMES[1:]
                    }
                    if request.action_family == STEP_SCALE_ACTION_FAMILY
                    else None
                ),
            )
            total_ms = (time.perf_counter() - started) * 1000.0
            response = {
                "protocol": PROTOCOL,
                "type": "evaluate_result",
                "request_id": request.request_id,
                "run_uuid": request.run_uuid,
                "step": request.step,
                "fragment_id": request.fragment_id,
                "action_family": request.action_family,
                "base_version": request.base_version,
                "state_epoch": request.state_epoch,
                "fragment_versions": list(request.fragment_versions),
                "request_digest": frame.digest,
                "ok": True,
                "fail_closed": selected.selected_action == BASELINE_ACTION,
                "selected_action": selected.selected_action,
                "selected_action_sha256": request.action_digests[
                    selected.selected_action
                ],
                "selected_action_metadata": request.action_metadata[
                    selected.selected_action
                ],
                "fallback_reason": selected.fallback_reason,
                "selection": selected.as_dict(),
                "action_metadata": request.action_metadata,
                "losses_by_action": losses,
                "paired_baseline_losses": backend_result.get("paired_baseline_losses"),
                "action_results": backend_result["actions"],
                "workers": backend_result.get("workers", []),
                "digests": {
                    "state_sha256": request.current_state_digest,
                    "action_sha256": request.action_digests,
                    "anchor_manifest_sha256": backend_result["anchor_manifest_sha256"],
                    "anchor_tensors_sha256": backend_result["anchor_tensors_sha256"],
                    "probe_config_sha256": backend_result["probe_config_sha256"],
                    "layout_hash": request.layout_hash,
                },
                "timings_ms": {
                    "parse": parse_ms,
                    "dispatch_and_eval": backend_result.get("dispatch_total_ms"),
                    "total": total_ms,
                },
                "cache_hit": False,
            }
            # Verify JSON finiteness before caching or sending a decision.
            json.dumps(response, allow_nan=False)
            self._cache_response(key, frame.digest, response)
            _log_successful_decision(response, self.selection)
            return response
        except Exception as exc:
            log.exception("action-probe request failed closed")
            if isinstance(exc, ProtocolError):
                code = "protocol_error"
            elif isinstance(exc, RequestValidationError):
                code = "request_validation_error"
            else:
                code = "evaluation_error"
            response = _error_response(
                frame,
                exc,
                error_code=code,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                action_family=request.action_family if request is not None else None,
            )
            if request is not None and key is not None:
                self._cache_response(key, frame.digest, response)
            return response


def _parse_listen(value: str) -> tuple[str, int]:
    try:
        host, port_text = value.rsplit(":", 1)
        port = int(port_text)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("listen address must be HOST:PORT") from exc
    if not host or not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(
            "listen address must contain a valid host and port"
        )
    try:
        if not ipaddress.ip_address(host).is_loopback:
            raise argparse.ArgumentTypeError(
                "action-probe service must bind to a loopback IP"
            )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "listen host must be a numeric loopback IP"
        ) from exc
    return host, port


def _parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "gpus must be a comma-separated integer list"
        ) from exc
    if not gpus or any(gpu < 0 for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError(
            "gpus must be a non-empty unique nonnegative list"
        )
    return gpus


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", type=_parse_listen, default=("127.0.0.1", 49321))
    parser.add_argument("--model", required=True)
    parser.add_argument("--anchor-manifest", required=True, type=Path)
    parser.add_argument("--gpus", required=True, type=_parse_gpus)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--panels", type=int, default=8)
    parser.add_argument("--blocks-per-panel", type=int, default=2)
    parser.add_argument("--train-on", choices=("assistant", "all"), default="assistant")
    parser.add_argument("--loss-function", default="cross_entropy")
    parser.add_argument("--lora-r", type=int, default=2)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument(
        "--lora-targets",
        choices=("auto", "attention", "all-linear"),
        default="auto",
    )
    parser.add_argument("--fragments", type=int, default=4)
    parser.add_argument(
        "--fragment-pattern", choices=("binpack", "strided"), default="binpack"
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--startup-timeout-s", type=float, default=1800.0)
    parser.add_argument("--request-timeout-s", type=float, default=30.0)
    parser.add_argument("--client-timeout-s", type=float, default=120.0)
    parser.add_argument("--retry-cache-size", type=int, default=2)
    parser.add_argument("--min-gain", type=float, default=0.00025)
    parser.add_argument("--lcb-z", type=float, default=2.365)
    parser.add_argument("--min-win-rate", type=float, default=0.75)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.seq_len <= 1 or args.panels < 2 or args.blocks_per_panel <= 0:
        parser.error("seq-len > 1, panels >= 2, and blocks-per-panel > 0 are required")
    if args.lora_r <= 0 or args.lora_alpha <= 0 or args.fragments <= 0:
        parser.error("LoRA rank, alpha, and fragment count must be positive")
    if args.client_timeout_s <= 0:
        parser.error("client-timeout-s must be positive")
    return args


class ActionProbeTCPService:
    def __init__(
        self,
        address: tuple[str, int],
        engine: ActionProbeEngine,
        *,
        client_timeout_s: float = 120.0,
    ):
        self.address = address
        self.engine = engine
        self.client_timeout_s = client_timeout_s
        self._stopping = False
        self._listener: socket.socket | None = None

    def stop(self, *_args) -> None:
        self._stopping = True

    def bind(self) -> tuple[str, int]:
        """Bind before advertising readiness; safe to call more than once."""

        if self._listener is not None:
            bound = self._listener.getsockname()
            return bound[0], bound[1]
        family = socket.AF_INET6 if ":" in self.address[0] else socket.AF_INET
        listener = socket.socket(family, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(self.address)
            listener.listen(4)
            listener.settimeout(1.0)
        except Exception:
            listener.close()
            raise
        self._listener = listener
        bound = listener.getsockname()
        return bound[0], bound[1]

    def _serve_connection(self, connection: socket.socket) -> None:
        connection.settimeout(self.client_timeout_s)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while not self._stopping:
            try:
                frame = recv_frame(connection, timeout_s=self.client_timeout_s)
            except EOFError:
                return
            except (socket.timeout, ProtocolError, OSError) as exc:
                log.warning(
                    "closing malformed/timed-out action-probe connection: %s", exc
                )
                return
            response = self.engine.handle(frame)
            if isinstance(response, tuple):
                response_header, response_payload = response
            else:
                response_header, response_payload = response, b""
            try:
                send_frame(connection, response_header, response_payload)
            except (ProtocolError, OSError) as exc:
                log.warning("failed to return action-probe response: %s", exc)
                return

    def serve_forever(self) -> None:
        self.bind()
        listener = self._listener
        assert listener is not None
        try:
            bound = listener.getsockname()
            log.info("action-probe listening on %s:%s", bound[0], bound[1])
            while not self._stopping:
                try:
                    connection, peer = listener.accept()
                except socket.timeout:
                    continue
                peer_ip = ipaddress.ip_address(peer[0])
                if not peer_ip.is_loopback:
                    log.warning("rejected non-loopback peer %s", peer[0])
                    connection.close()
                    continue
                with connection:
                    self._serve_connection(connection)
        finally:
            listener.close()
            self._listener = None


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    selection = SelectionConfig(
        min_gain=args.min_gain,
        lcb_z=args.lcb_z,
        min_win_rate=args.min_win_rate,
        min_panels=args.panels,
    )
    replica_configs = [
        ReplicaConfig(
            gpu_id=gpu,
            model=args.model,
            anchor_manifest=str(args.anchor_manifest),
            seq_len=args.seq_len,
            panels=args.panels,
            blocks_per_panel=args.blocks_per_panel,
            train_on=args.train_on,
            loss_function=args.loss_function,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_targets=args.lora_targets,
            fragments=args.fragments,
            fragment_pattern=args.fragment_pattern,
            cache_dir=args.cache_dir,
            selection=selection,
        )
        for gpu in args.gpus
    ]
    backend = WorkerPoolBackend(
        replica_configs,
        startup_timeout_s=args.startup_timeout_s,
        request_timeout_s=args.request_timeout_s,
    )
    engine = ActionProbeEngine(
        backend,
        selection=selection,
        retry_cache_size=args.retry_cache_size,
    )
    service = ActionProbeTCPService(
        args.listen, engine, client_timeout_s=args.client_timeout_s
    )
    signal.signal(signal.SIGINT, service.stop)
    signal.signal(signal.SIGTERM, service.stop)
    try:
        bound_host, bound_port = service.bind()
        print(
            "ACTION_PROBE_READY "
            + json.dumps(
                {
                    "listen": f"{bound_host}:{bound_port}",
                    **engine.describe(),
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        service.serve_forever()
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
