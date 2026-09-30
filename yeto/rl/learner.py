"""Entrypoint for one pinned-Miles RL island."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any

from .bridge import BridgeConfig
from .decoupled import DecoupledBridgeConfig
from .deepseek_v4_expert_full import expert_full_specs
from .export import adapter_targets, derive_peft_lora_specs
from .miles import verify_miles_revision


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m yeto.rl.learner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rollout-model", default=None)
    parser.add_argument("--rollout-model-revision", default=None)
    parser.add_argument(
        "--rl-model-recipe",
        choices=["generic", "deepseek-v4-flash"],
        default="generic",
    )
    parser.add_argument("--expert-full-count", type=int, default=0)
    parser.add_argument("--expert-full-lr", type=float, default=1e-6)
    parser.add_argument("--expert-selection-sha256", default=None)
    parser.add_argument("--expert-selection-contract-sha256", default=None)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-revision", default=None)
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--eval-dataset-name", default=None)
    parser.add_argument("--eval-data-sha256", default=None)
    parser.add_argument("--eval-summary-path", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-samples-per-prompt", type=int, default=None)
    parser.add_argument("--eval-temperature", type=float, default=None)
    parser.add_argument("--eval-top-p", type=float, default=None)
    parser.add_argument("--eval-max-prompt-len", type=int, default=None)
    parser.add_argument("--eval-max-response-len", type=int, default=None)
    parser.add_argument("--eval-max-context-len", type=int, default=None)
    # Required unless --rl-single-island-no-sync (checked after parsing).
    parser.add_argument("--syncer", default=None)
    parser.add_argument(
        "--rl-echo-events",
        action="store_true",
        help=(
            "ports: echo every event-tape record to stdout as 'YETO_RL_EVENT <json>' "
            "(the launcher sets it for Modal islands, whose ~/yeto-output is not fetchable)"
        ),
    )
    parser.add_argument(
        "--rl-single-island-no-sync",
        action="store_true",
        help=(
            "ports only: one island with no syncer and no outer sync (G1 smoke "
            "entry for --rl-allow-unverified-mechanism, design D11); refused "
            "with --syncer, several learners or a non-default sync preset"
        ),
    )
    parser.add_argument("--learner-id", type=int, required=True)
    parser.add_argument("--num-learners", type=int, default=1)
    parser.add_argument("--learner-generation", type=int, default=0)
    parser.add_argument("--reward-function", required=True)
    parser.add_argument("--rl-prompt-column", default=None)
    parser.add_argument("--rl-label-column", default=None)
    parser.add_argument("--reward-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--global-rounds", type=int, required=True)
    parser.add_argument(
        "--parameter-mode",
        choices=["lora", "full"],
        default="lora",
    )
    parser.add_argument(
        "--sync-preset",
        choices=["strict-avg", "decoupled", "dense-full"],
        default="strict-avg",
    )
    parser.add_argument("--fragments", type=int, default=1)
    parser.add_argument("--parameter-layout-sha256", default=None)
    parser.add_argument("--pipeline", type=int, default=1)
    parser.add_argument("--local-horizon", type=int, default=1)
    parser.add_argument("--total-fragment-steps", type=int, default=None)
    parser.add_argument("--initial-adapter", default=None)
    parser.add_argument("--initial-adapter-sha256", default=None)
    parser.add_argument("--groups-per-round", type=int, required=True)
    parser.add_argument("--samples-per-group", type=int, required=True)
    parser.add_argument("--over-sampling-batch-size", type=int, required=True)
    parser.add_argument("--dynamic-sampling-filter-path", default=None)
    parser.add_argument("--dynamic-sampling-max-replacements", type=int, default=None)
    parser.add_argument(
        "--secrlenv-max-infrastructure-replacements", type=int, default=None
    )
    parser.add_argument("--rl-offload-train", action="store_true")
    parser.add_argument("--rl-distributed-timeout-minutes", type=int, default=10)
    parser.add_argument("--optimizer-steps", type=int, required=True)
    parser.add_argument("--rollout-max-response-len", type=int, required=True)
    parser.add_argument("--apply-chat-template-kwargs", type=json.loads, default=None)
    parser.add_argument("--custom-generate-function-path", default=None)
    parser.add_argument("--custom-agent-function-path", default=None)
    parser.add_argument("--codex-harness-contract", type=json.loads, default=None)
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=["xhigh"],
        default=None,
    )
    parser.add_argument("--agent-max-seq-len", type=int, default=None)
    parser.add_argument("--use-session-server", action="store_true")
    parser.add_argument("--session-server-ip", default=None)
    parser.add_argument("--session-server-port", type=int, nargs="+", default=None)
    parser.add_argument("--tito-model", default=None)
    parser.add_argument("--codex-backend-profile", default=None)
    parser.add_argument(
        "--tito-allowed-append-roles",
        nargs="+",
        choices=["tool", "user", "system"],
        default=None,
    )
    parser.add_argument("--completed-groups-path", required=True)
    parser.add_argument("--event-tape", required=True)
    parser.add_argument("--audit-dir", default=None)
    parser.add_argument("--actor-num-nodes", type=int, required=True)
    parser.add_argument("--actor-num-gpus-per-node", type=int, required=True)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--pipeline-parallel", type=int, default=1)
    parser.add_argument("--expert-parallel", type=int, default=None)
    parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    parser.add_argument("--rollout-num-gpus", type=int, default=None)
    # rl-infra-spec 2.1 (ports only): LoRA fixed partition + reserved standby
    # GPUs, read by engine.run_config.resolve_rl_run_config.
    parser.add_argument(
        "--rl-placement", choices=["colocated", "fixed-partition"], default="colocated"
    )
    parser.add_argument("--rl-standby-gpus", type=int, default=0)
    # rl-infra-spec 2.3 / 3.x (ports only, off by default; the default island
    # and its Miles argv are unchanged): eval||train overlap and the E1
    # elastic rollout controller.
    parser.add_argument("--rl-overlap-eval", action="store_true")
    parser.add_argument("--rl-elastic", action="store_true")
    parser.add_argument("--rl-elastic-resources", default=None, metavar="PATH")
    parser.add_argument("--rl-elastic-attestation", default=None, metavar="PATH")
    parser.add_argument("--rl-elastic-state-dir", default=None, metavar="PATH")
    parser.add_argument("--rl-elastic-initial-config", default=None, metavar="NAME")
    parser.add_argument("--rl-elastic-cells", default=None, metavar="ID[,ID...]")
    parser.add_argument("--sglang-tp-size", type=int, default=None)
    parser.add_argument("--sglang-dp-size", type=int, default=None)
    parser.add_argument("--sglang-ep-size", type=int, default=None)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.4)
    parser.add_argument("--sglang-attention-backend", default=None)
    parser.add_argument(
        "--sglang-deterministic-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sglang-page-size", type=int, default=None)
    parser.add_argument("--sglang-max-running-requests", type=int, default=None)
    parser.add_argument("--sglang-chunked-prefill-size", type=int, default=None)
    parser.add_argument("--use-rollout-routing-replay", action="store_true")
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument(
        "--lora-targets",
        choices=[
            "auto",
            "attention",
            "attention-routed-experts",
            "all-linear",
        ],
        default=None,
    )
    parser.add_argument("--inner-lr", type=float, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wan-streams", type=int, default=4)
    parser.add_argument("--miles-root", required=True)
    parser.add_argument(
        "--rl-engine",
        choices=["legacy", "ports"],
        default="ports",
        help="RL engine path (default ports); ports rejects unsupported combinations at startup",
    )
    # rl-algorithm-capabilities D8/D9/D11 (ports only; refused on legacy).
    parser.add_argument(
        "--rl-algorithm-spec",
        default=None,
        help="AlgorithmSpec JSON (v1 or v2) for --rl-engine ports",
    )
    parser.add_argument(
        "--rl-expected-algorithm-sha256",
        default=None,
        help="algorithm hash the launcher expects every island to build (ports)",
    )
    parser.add_argument(
        "--rl-allow-unverified-mechanism",
        action="append",
        default=None,
        metavar="NAME",
        help="single-island smoke: admit an expressible but undeclared mechanism (ports)",
    )
    parser.add_argument("--miles-source-sha256", default=None)
    parser.add_argument("--megatron-ref-load", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    from ..wandb_logger import add_arguments as add_wandb_arguments

    add_wandb_arguments(parser)
    args = parser.parse_args(argv)
    if args.parameter_mode == "lora" and (
        args.lora_r is None or args.lora_targets is None
    ):
        parser.error("LoRA mode requires --lora-r and --lora-targets")
    if (args.parameter_mode == "full") != (args.sync_preset == "dense-full"):
        parser.error("--parameter-mode full requires --sync-preset dense-full")
    if not args.eval_only and not args.inner_lr > 0:
        # A zero LR would only surface as the zero-LR invariant failing round 1.
        parser.error(f"--inner-lr must be > 0 for a training run (got {args.inner_lr})")
    if args.rl_engine == "ports":
        try:
            _require_ports_supported(args)
        except ValueError as error:
            parser.error(str(error))
    try:
        _check_ports_infra_switches(args)
        _check_single_island_no_sync(args)
        _check_ports_algorithm_options(
            args, outer_sync=not getattr(args, "rl_single_island_no_sync", False)
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def build_ports_launch(args, run_config, extra_argv=()):
    """The ports learner's spec -> Miles argv step (design D8).

    ``--rl-algorithm-spec`` (else the legacy CLI, as in R0), then the mapped
    extra-argv flags are absorbed by the translation; the run config's
    estimator follows the absorbed spec.
    """

    import dataclasses

    from .engine.algorithm import resolve_ports_algorithm
    from .engine.miles_adapter.algorithm_flags import absorb_extra_argv
    from .engine.miles_adapter.config import translate_run_config

    base_algorithm = resolve_ports_algorithm(args, rl_engine="ports")
    absorbed, _, _ = absorb_extra_argv(base_algorithm, tuple(extra_argv))
    run_config = dataclasses.replace(
        run_config,
        algorithm=dataclasses.replace(
            run_config.algorithm, advantage_estimator=absorbed.advantage_estimator
        ),
    )
    return translate_run_config(run_config, base_algorithm, extra_argv=tuple(extra_argv))


_ELASTIC_COMPANIONS = (
    ("rl_elastic_resources", "--rl-elastic-resources"),
    ("rl_elastic_attestation", "--rl-elastic-attestation"),
    ("rl_elastic_state_dir", "--rl-elastic-state-dir"),
    ("rl_elastic_initial_config", "--rl-elastic-initial-config"),
    ("rl_elastic_cells", "--rl-elastic-cells"),
)
_ELASTIC_REQUIRED = ("rl_elastic_resources", "rl_elastic_state_dir",
                     "rl_elastic_initial_config", "rl_elastic_cells")


def _check_ports_infra_switches(args) -> None:
    """``--rl-overlap-eval`` / ``--rl-elastic``: explicit and ports-only."""

    ports = getattr(args, "rl_engine", "ports") == "ports"
    if getattr(args, "rl_overlap_eval", False) and not ports:
        raise ValueError("--rl-overlap-eval only applies to --rl-engine ports")
    given = [flag for name, flag in _ELASTIC_COMPANIONS if getattr(args, name, None)]
    if not getattr(args, "rl_elastic", False):
        if given:
            raise ValueError(", ".join(given) + " need --rl-elastic")
        return
    if not ports:
        raise ValueError("--rl-elastic only applies to --rl-engine ports")
    missing = [flag for name, flag in _ELASTIC_COMPANIONS
               if name in _ELASTIC_REQUIRED and not getattr(args, name, None)]
    if missing:
        raise ValueError("--rl-elastic needs " + ", ".join(missing))
    if not _elastic_cells(args.rl_elastic_cells):
        raise ValueError("--rl-elastic-cells names no cell")


def _elastic_cells(value: str | None) -> tuple[str, ...]:
    return tuple(c.strip() for c in (value or "").split(",") if c.strip())


def apply_ports_infra_switches(args, miles_args, environ=None) -> None:
    """Carry the opt-in 2.3/3.x switches onto ``miles_args`` before any rollout
    process starts. Without them nothing is set (default path unchanged)."""

    if getattr(args, "rl_overlap_eval", False):
        miles_args.yeto_rl_overlap_eval = True
    if not getattr(args, "rl_elastic", False):
        return
    miles_args.yeto_rl_elastic = {
        "resources": args.rl_elastic_resources,
        "attestation": getattr(args, "rl_elastic_attestation", None),
        "state_dir": args.rl_elastic_state_dir,
        "initial_config": args.rl_elastic_initial_config,
        "declared_cells": _elastic_cells(args.rl_elastic_cells),
    }
    # M1: rollout metadata carries data_cursor/buffer_length only when asked.
    # The attribute covers the driver process; the env var reaches Ray workers
    # (where the metadata hook runs) through connect_island_ray's job-level
    # runtime_env, since workers inherit the raylet's env, not the driver's.
    from .engine.miles_adapter.rollout_meta_hook import ELASTIC_METADATA_ENV

    miles_args.yeto_rl_elastic_metadata = True
    (os.environ if environ is None else environ)[ELASTIC_METADATA_ENV] = "1"


def _check_single_island_no_sync(args) -> None:
    """``--rl-single-island-no-sync``: explicit, ports-only, one island, no syncer."""

    if not getattr(args, "rl_single_island_no_sync", False):
        if getattr(args, "syncer", None) is None:
            raise ValueError("--syncer is required (unless --rl-single-island-no-sync)")
        return
    problems = []
    if getattr(args, "rl_engine", "ports") != "ports":
        problems.append("--rl-engine legacy")
    if getattr(args, "syncer", None) is not None:
        problems.append("--syncer")
    if int(getattr(args, "num_learners", 1) or 1) != 1:
        problems.append(f"--num-learners {args.num_learners}")
    if int(getattr(args, "learner_id", 0)) != 0:
        problems.append(f"--learner-id {args.learner_id}")
    if getattr(args, "sync_preset", "strict-avg") != "strict-avg":
        problems.append(f"--sync-preset {args.sync_preset}")
    if getattr(args, "initial_adapter", None):
        problems.append("--initial-adapter")
    if problems:
        raise ValueError(
            "--rl-single-island-no-sync (one ports island, no syncer, no outer sync) "
            "cannot be combined with: " + ", ".join(problems)
        )


def _check_ports_algorithm_options(args, *, outer_sync: bool = True) -> None:
    """Startup refusal of the ports algorithm options (D8/D11), before any work."""

    from .engine.algorithm import check_unverified_allowance

    rl_engine = getattr(args, "rl_engine", "ports")
    if rl_engine != "ports" and (
        getattr(args, "rl_placement", "colocated") != "colocated"
        or (getattr(args, "rl_standby_gpus", 0) or 0) != 0
    ):
        raise ValueError("--rl-placement/--rl-standby-gpus only apply to --rl-engine ports")
    if rl_engine != "ports":
        from .engine.algorithm import resolve_ports_algorithm

        resolve_ports_algorithm(args, rl_engine=rl_engine)  # raises if any is used
        return
    # D11 as written: any outer sync refuses the allowance. The learner CLI
    # always joins a syncer (--syncer is required), so the island count it
    # receives (--num-learners is not sent to Miles islands) is not relied on.
    check_unverified_allowance(
        getattr(args, "rl_allow_unverified_mechanism", None) or (),
        islands=int(getattr(args, "num_learners", 1) or 1),
        outer_sync=outer_sync,
    )


def install_event_echo() -> bool:
    """Echo every tape record of this island to stdout (``YETO_RL_EVENT``).

    For islands whose ~/yeto-output cannot be fetched (Modal, no-sync): sets
    ``YETO_RL_ECHO_EVENTS=1`` so the single low-level tape writer
    (``yeto.rl.event_echo.append_record``, used by the driver, bridge, learner
    and adapter events) prints each line it writes; Ray workers inherit it
    (``entry.connect_island_ray``). Idempotent; True when enabled now.
    """

    from .engine import driver
    from .event_echo import echo_enabled, enable_echo

    if getattr(driver, "_ECHO_EVENTS", False):
        driver._ECHO_EVENTS = False  # would print driver records a second time
    if echo_enabled():
        return False
    enable_echo()
    from . import miles

    if "append_record" not in miles._append_rl_event.__code__.co_names:
        _wrap_legacy_tape_writer(miles)  # a writer not yet on append_record
    return True


def _wrap_legacy_tape_writer(miles) -> None:
    """Echo for a ``_append_rl_event`` that writes the file itself (pre
    echo-writers patch): read back exactly the whole lines it appended."""

    import threading

    from .event_echo import PREFIX

    original = miles._append_rl_event
    lock = threading.Lock()

    def echo(args, event):
        with lock:
            path = Path(args.yeto_rl_event_tape).expanduser()
            before = path.stat().st_size if path.exists() else 0
            original(args, event)
            with path.open("rb") as handle:
                handle.seek(before)
                data = handle.read()
        written = data[: data.rfind(b"\n") + 1].decode("utf-8")
        for line in written.splitlines():
            if line.strip():
                print(PREFIX + line, flush=True)

    miles._append_rl_event = echo


def _append_ports_event(args, miles_args, event: dict) -> None:
    from .miles import _append_rl_event

    miles_args.yeto_rl_event_tape = args.event_tape
    miles_args.yeto_rl_learner_id = args.learner_id
    _append_rl_event(miles_args, event)


class AlgorithmMismatchError(RuntimeError):
    """This island's algorithm hash differs from the launcher's expectation."""


def verify_ports_algorithm(args, miles_args, launch) -> None:
    """D9/D11: before the island joins outer sync (no bridge exists yet).

    Plugins are re-hashed and imported; the island's algorithm hash is
    compared with ``--rl-expected-algorithm-sha256`` (mismatch: event +
    refusal; missing: warning event, for manually started islands);
    absorbed flags and unverified allowances are recorded.
    """

    algorithm = launch.algorithm
    algorithm.verify_plugins()
    from .engine.algorithm import island_problems

    problems = island_problems(algorithm, {
        "base_model_revision": getattr(args, "model_revision", None),
        "base_model": getattr(args, "model", None),
        "ref_load_override": getattr(args, "megatron_ref_load", None),
    })
    if problems:
        _append_ports_event(args, miles_args, {
            "event": "rl_algorithm_island_rejected",
            "rl/algorithm_spec_sha256": launch.algorithm_sha256,
            "problems": problems,
        })
        raise AlgorithmMismatchError(
            f"island {args.learner_id} algorithm spec rejected before joining outer sync: "
            + "; ".join(problems)
        )
    actual = launch.algorithm_sha256
    expected = getattr(args, "rl_expected_algorithm_sha256", None)
    if expected is None:
        _append_ports_event(args, miles_args, {
            "event": "rl_algorithm_expected_hash_missing",
            "level": "warning",
            "rl/algorithm_spec_sha256": actual,
        })
    elif expected.lower() != actual:
        _append_ports_event(args, miles_args, {
            "event": "rl_algorithm_mismatch",
            "rl/algorithm_spec_sha256": actual,
            "rl/expected_algorithm_spec_sha256": expected.lower(),
            "rl/algorithm_spec": algorithm.canonical_json(),
        })
        raise AlgorithmMismatchError(
            f"island {args.learner_id} algorithm hash {actual} differs from the launcher's "
            f"expected {expected.lower()}; refusing to join outer sync"
        )
    miles_args.yeto_rl_algorithm_absorbed_flags = dict(launch.absorbed_flags)
    miles_args.yeto_rl_outer_sync = not getattr(args, "rl_single_island_no_sync", False)
    # INFRA compares it with the runtime AlgorithmSpec before connect_island_ray
    # (a partitioned run without it is refused there).
    miles_args.yeto_rl_expected_algorithm_sha256 = (
        expected.lower() if expected is not None else None
    )
    miles_args.yeto_rl_unverified_mechanisms = tuple(
        sorted(set(getattr(args, "rl_allow_unverified_mechanism", None) or ()))
    )


def _require_ports_supported(args, extra_argv: Sequence[str] = ()) -> None:
    """Reject combinations outside the R0 ports matrix before any startup."""

    from .engine.selection import require_ports_supported

    require_ports_supported(
        sync_preset=getattr(args, "sync_preset", "strict-avg"),
        parameter_mode=getattr(args, "parameter_mode", "lora"),
        model_recipe=getattr(args, "rl_model_recipe", "generic"),
        lora_targets=getattr(args, "lora_targets", None),
        expert_full_count=getattr(args, "expert_full_count", 0) or 0,
        rollout_num_gpus=getattr(args, "rollout_num_gpus", None),
        placement=getattr(args, "rl_placement", "colocated"),
        extra_argv=tuple(extra_argv),
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


_DENSE_ISLAND_LOCAL_MILES_FLAGS = frozenset(
    {
        "--rollout-seed",
        "--rollout-engine-base-port",
        "--session-server-port",
        "--sglang-router-port",
        "--sglang-router-prometheus-port",
        "--train-master-base-port",
    }
)

_DENSE_ISLAND_LOCAL_MILES_NARGS_FLAGS = frozenset(
    {
        "--session-server-port",
    }
)

_DENSE_FULL_QUANTIZATION_CONFIG_KEYS = frozenset(
    {
        "compression_config",
        "modelopt_quant",
        "quantization",
        "quantization_config",
        "torchao_config",
    }
)


def _require_unquantized_dense_full_rollout_model(model_path: str | Path) -> None:
    """Enable the pinned SGLang BF16 hook fallback only for an exact safe model.

    The public SGLang pin used by the Milestone-1 gate predates the optional
    begin/end weight-update endpoints.  Miles independently rechecks its live
    server arguments before accepting a missing endpoint; this early check
    prevents Yeto from opting into that compatibility path for a quantized
    rollout model in the first place.
    """

    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise ValueError("dense full-parameter rollout model has no config.json")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate rollout config key: {name!r}")
            result[name] = value
        return result

    try:
        config = json.loads(
            config_path.read_bytes(),
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("dense full-parameter rollout config is malformed") from exc
    if not isinstance(config, dict):
        raise TypeError("dense full-parameter rollout config must be an object")
    configured = sorted(
        name
        for name in _DENSE_FULL_QUANTIZATION_CONFIG_KEYS
        if config.get(name) is not None
    )
    if configured:
        raise ValueError(
            "legacy SGLang weight-update hooks require an unquantized rollout "
            f"model; configured fields: {', '.join(configured)}"
        )


def _dense_full_training_contract_sha256(
    args: Any,
    miles_argv: Sequence[str],
    *,
    model_config_sha256: str,
    prompt_data_sha256: str,
) -> str:
    """Hash shared training semantics while excluding island-local routing."""

    shared_argv = []
    index = 0
    while index < len(miles_argv):
        value = miles_argv[index]
        if value in _DENSE_ISLAND_LOCAL_MILES_FLAGS:
            if index + 1 >= len(miles_argv):
                raise ValueError(f"Miles flag has no value: {value}")
            index += 1
            value_count = 0
            while index < len(miles_argv) and not miles_argv[index].startswith("--"):
                index += 1
                value_count += 1
                if value not in _DENSE_ISLAND_LOCAL_MILES_NARGS_FLAGS:
                    break
            if value_count < 1:
                raise ValueError(f"Miles flag has no value: {value}")
            continue
        shared_argv.append(value)
        index += 1
    explicit_rollout_seed = getattr(args, "rollout_seed", None)
    rollout_seed_contract = (
        {"mode": "fixed", "value": explicit_rollout_seed}
        if explicit_rollout_seed is not None
        else {"mode": "base-plus-learner-id", "base_seed": args.seed}
    )
    return _canonical_sha256(
        {
            "schema": "yeto-dense-full-training-contract-v3",
            "shared_miles_argv": shared_argv,
            "rollout_seed_contract": rollout_seed_contract,
            "model": args.model,
            "model_revision": args.model_revision.lower(),
            "model_config_sha256": model_config_sha256,
            "prompt_data_sha256": prompt_data_sha256,
            "data": args.data,
            "data_revision": args.data_revision,
            "evaluation": (
                None
                if getattr(args, "eval_interval", None) is None
                else {
                    "data": getattr(args, "eval_data", None),
                    "data_sha256": getattr(args, "eval_data_sha256", None),
                    "dataset_name": getattr(args, "eval_dataset_name", None),
                    "interval": args.eval_interval,
                    "samples_per_prompt": getattr(
                        args, "eval_samples_per_prompt", None
                    ),
                    "temperature": getattr(args, "eval_temperature", None),
                    "top_p": getattr(args, "eval_top_p", None),
                    "max_prompt_len": getattr(args, "eval_max_prompt_len", None),
                    "max_response_len": getattr(
                        args, "eval_max_response_len", None
                    ),
                    "max_context_len": getattr(args, "eval_max_context_len", None),
                }
            ),
            "yeto_source_sha256": args.source_sha256,
            "miles_source_sha256": args.miles_source_sha256,
            "reward_sha256": args.reward_sha256,
            "codex_harness_contract": getattr(
                args,
                "codex_harness_contract",
                None,
            ),
        }
    )


def _strict_json_file_semantics(path: Path) -> tuple[Any, ...]:
    """Parse exact, typed JSON semantics independent of object-key order."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in pairs:
            if name in value:
                raise ValueError(f"duplicate JSON object key: {name!r}")
            value[name] = item
        return value

    def finite_number(name: str) -> None:
        raise ValueError(f"non-finite JSON number: {name}")

    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=unique_object,
            parse_constant=finite_number,
            parse_float=Decimal,
            parse_int=Decimal,
        )
    except (OSError, UnicodeError, ValueError, DecimalException) as exc:
        raise ValueError("Codex app-server schema is malformed JSON") from exc

    def typed_semantics(item: Any) -> tuple[Any, ...]:
        if item is None:
            return ("null",)
        if isinstance(item, bool):
            return ("boolean", item)
        if isinstance(item, Decimal):
            return ("number", item)
        if isinstance(item, str):
            return ("string", item)
        if isinstance(item, list):
            return ("array", tuple(typed_semantics(value) for value in item))
        if isinstance(item, dict):
            return (
                "object",
                tuple(
                    sorted(
                        (name, typed_semantics(value))
                        for name, value in item.items()
                    )
                ),
            )
        raise TypeError(f"unsupported parsed JSON value: {type(item).__name__}")

    return typed_semantics(value)


def _verify_live_codex_app_server_schema(pinned: Path, generated: Path) -> None:
    try:
        pinned_semantics = _strict_json_file_semantics(pinned)
        generated_semantics = _strict_json_file_semantics(generated)
    except ValueError as exc:
        raise ValueError("live stock Codex app-server schema drifted") from exc
    if generated_semantics != pinned_semantics:
        raise ValueError("live stock Codex app-server schema drifted")


def _preflight_codex_openenv_adapter(args, profile_name: str) -> None:
    """Attest the isolated OpenEnv wrapper inside the pinned Miles source."""

    from . import CODEX_OPENENV_AGENT_MODULES, CODEX_OPENENV_IDENTITY_ENV

    if profile_name != "qwen35_08b":
        raise ValueError(
            "the Codex OpenEnv adapter requires backend profile qwen35_08b"
        )
    adapter_dir = (
        Path(args.miles_root).expanduser().resolve()
        / "examples"
        / "experimental"
        / "openenv"
    )
    for name in CODEX_OPENENV_AGENT_MODULES:
        source = adapter_dir / name
        if source.is_symlink() or not source.is_file():
            raise ValueError("the Codex OpenEnv adapter source is incomplete")
    adapter_root = str(adapter_dir)
    if adapter_root not in sys.path:
        sys.path.insert(0, adapter_root)
    try:
        openenv_adapter = importlib.import_module("codex_openenv_agent_function")
        subprocess_adapter = importlib.import_module(
            "codex_openenv_subprocess_agent_function"
        )
        openenv_identity = openenv_adapter.codex_openenv_harness_identity()
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("cannot attest the Codex OpenEnv adapter") from exc
    for module in (openenv_adapter, subprocess_adapter):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise ValueError("the Codex OpenEnv adapter has no source identity")
        source = Path(module_file)
        if source.is_symlink() or source.resolve().parent != adapter_dir:
            raise ValueError("the Codex OpenEnv adapter resolved outside pinned Miles")
    if not callable(getattr(subprocess_adapter, "run", None)):
        raise ValueError("the Codex OpenEnv subprocess entrypoint is missing")
    if openenv_adapter._OPENENV_IDENTITY_ENV != CODEX_OPENENV_IDENTITY_ENV:
        raise ValueError("the Codex OpenEnv launch identity drifted")
    expected_openenv_identity = {
        name.removeprefix("YETO_CODEX_OPENENV_").lower(): value
        for name, value in CODEX_OPENENV_IDENTITY_ENV.items()
        if name.endswith("_SHA256")
    }
    if openenv_identity != expected_openenv_identity:
        raise ValueError("the Codex OpenEnv surface identity drifted")
    openenv_env_mismatched = [
        name
        for name, expected in CODEX_OPENENV_IDENTITY_ENV.items()
        if os.getenv(name) != expected
    ]
    if openenv_env_mismatched:
        raise ValueError(
            "Codex OpenEnv container environment drifted: "
            + ", ".join(openenv_env_mismatched)
        )


def _preflight_codex_harness(args) -> None:
    """Fail before model allocation if the signed Codex surface has drifted."""

    from . import (
        CODEX_APP_SERVER_PROTOCOL_REVISION,
        CODEX_APP_SERVER_SCHEMA_SHA256,
        CODEX_BASE_INSTRUCTIONS_SHA256,
        CODEX_CLI_VERSION,
        CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH,
        CODEX_CONTAINER_BINARY_PATH,
        CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256,
        CODEX_HARNESS_AGENT,
        CODEX_HARNESS_AGENT_SHA256,
        CODEX_LINUX_BINARY_SHA256,
        CODEX_LINUX_BINARY_SIZE_BYTES,
        CODEX_LINUX_TARGET,
        CODEX_NPM_PACKAGE,
        CODEX_NPM_TARBALL_SHA256,
        CODEX_OPENENV_AGENT,
        CODEX_OPENENV_IDENTITY_ENV,
        CODEX_PACKAGE_MANIFEST_SHA256,
        CODEX_SUBMIT_TOOL_SCHEMA_SHA256,
        CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256,
        SIGNED_CODEX_AGENTS,
    )

    contract = getattr(args, "codex_harness_contract", None)
    if args.custom_agent_function_path not in SIGNED_CODEX_AGENTS:
        if contract is not None:
            raise ValueError(
                "--codex-harness-contract requires a signed Codex agent"
            )
        return
    if not isinstance(contract, dict):
        raise ValueError("the signed Codex agent requires its harness contract")
    required = {
        "agent_function_path",
        "agent_source_sha256",
        "controller_binary_path",
        "controller_package_manifest_path",
        "controller_app_server_schema_path",
        "bundle_binary_path",
        "bundle_package_manifest_path",
        "bundle_app_server_schema_path",
        "container_binary_path",
        "container_app_server_schema_path",
        "binary_sha256",
        "binary_size_bytes",
        "cli_version",
        "npm_package",
        "target",
        "npm_tarball_sha256",
        "package_manifest_sha256",
        "app_server_protocol_revision",
        "app_server_schema_sha256",
        "base_instructions_sha256",
        "terminal_exec_tool_schema_sha256",
        "submit_tool_schema_sha256",
        "dynamic_tools_schema_sha256",
        "reasoning_effort",
        "backend",
    }
    if args.custom_agent_function_path == CODEX_OPENENV_AGENT:
        required.add("openenv_identity_env")
    if set(contract) != required:
        raise ValueError("stock Codex harness contract shape drifted")
    pinned = {
        "agent_function_path": CODEX_HARNESS_AGENT,
        "agent_source_sha256": CODEX_HARNESS_AGENT_SHA256,
        "container_binary_path": CODEX_CONTAINER_BINARY_PATH,
        "container_app_server_schema_path": (
            CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH
        ),
        "binary_sha256": CODEX_LINUX_BINARY_SHA256,
        "binary_size_bytes": CODEX_LINUX_BINARY_SIZE_BYTES,
        "cli_version": CODEX_CLI_VERSION,
        "npm_package": CODEX_NPM_PACKAGE,
        "target": CODEX_LINUX_TARGET,
        "npm_tarball_sha256": CODEX_NPM_TARBALL_SHA256,
        "package_manifest_sha256": CODEX_PACKAGE_MANIFEST_SHA256,
        "app_server_protocol_revision": CODEX_APP_SERVER_PROTOCOL_REVISION,
        "app_server_schema_sha256": CODEX_APP_SERVER_SCHEMA_SHA256,
        "base_instructions_sha256": CODEX_BASE_INSTRUCTIONS_SHA256,
        "terminal_exec_tool_schema_sha256": (
            CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256
        ),
        "submit_tool_schema_sha256": CODEX_SUBMIT_TOOL_SCHEMA_SHA256,
        "dynamic_tools_schema_sha256": CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256,
        "reasoning_effort": "xhigh",
    }
    if any(contract.get(name) != expected for name, expected in pinned.items()):
        raise ValueError("stock Codex runtime identity drifted")
    if (
        args.custom_agent_function_path == CODEX_OPENENV_AGENT
        and contract.get("openenv_identity_env") != CODEX_OPENENV_IDENTITY_ENV
    ):
        raise ValueError("stock Codex OpenEnv surface identity drifted")
    from .codex_backend import (
        stock_codex_backend_contract,
        validate_stock_codex_fields,
    )

    profile_name = getattr(args, "codex_backend_profile", None) or args.tito_model
    expected_backend = stock_codex_backend_contract(
        profile_name,
        args.rollout_max_response_len,
    )
    validate_stock_codex_fields(
        tito_model=args.tito_model,
        codex_backend_profile=profile_name,
        rl_model_recipe=args.rl_model_recipe,
        model=args.model,
        model_revision=args.model_revision,
        rollout_model=getattr(args, "rollout_model", None),
        rollout_model_revision=getattr(args, "rollout_model_revision", None),
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        tito_allowed_append_roles=args.tito_allowed_append_roles,
        codex_reasoning_effort=args.codex_reasoning_effort,
        # The signed Qwen rollout backend profile predates full-parameter
        # training and uses ``attention`` only as its legacy tuning-profile
        # discriminator.  Full mode emits no LoRA runtime flags below.
        lora_targets=(
            args.lora_targets
            if getattr(args, "parameter_mode", "lora") == "lora"
            else "attention"
        ),
        expert_full_count=getattr(args, "expert_full_count", 0),
    )
    if contract.get("backend") != expected_backend:
        raise ValueError("stock Codex backend/TITO identity drifted")
    from ..provenance import file_sha256

    # OpenEnv owns and attests its adapter inside the pinned Miles source.  The
    # legacy SecRLEnv-backed Codex adapter remains an optional integration, but
    # it must not be required for the independent Terminal-Bench path.
    if args.custom_agent_function_path != CODEX_OPENENV_AGENT:
        try:
            from yeto_miles_secrlenv import codex_harness_agent

            live_identity = codex_harness_agent.codex_harness_identity()
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            raise ValueError("cannot attest the Yeto Codex harness adapter") from exc
        identity_names = (
            "base_instructions_sha256",
            "terminal_exec_tool_schema_sha256",
            "submit_tool_schema_sha256",
            "dynamic_tools_schema_sha256",
        )
        if live_identity != {name: contract.get(name) for name in identity_names}:
            raise ValueError("Yeto Codex adapter identity drifted")

        adapter_path = Path(codex_harness_agent.__file__)
        if adapter_path.is_symlink() or not adapter_path.is_file():
            raise ValueError("Yeto Codex adapter source identity drifted")
        adapter_path = adapter_path.resolve()
        if file_sha256(adapter_path) != contract.get("agent_source_sha256"):
            raise ValueError("Yeto Codex adapter source identity drifted")
    binary = Path(CODEX_CONTAINER_BINARY_PATH)
    manifest = Path("/opt/yeto/codex/codex-package.json")
    schema = Path(CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH)
    if (
        binary.is_symlink()
        or not binary.is_file()
        or binary.stat().st_size != CODEX_LINUX_BINARY_SIZE_BYTES
        or file_sha256(binary) != CODEX_LINUX_BINARY_SHA256
        or manifest.is_symlink()
        or not manifest.is_file()
        or file_sha256(manifest) != CODEX_PACKAGE_MANIFEST_SHA256
        or schema.is_symlink()
        or not schema.is_file()
        or file_sha256(schema) != CODEX_APP_SERVER_SCHEMA_SHA256
    ):
        raise ValueError("mounted stock Codex artifact does not match its Yeto pin")
    try:
        version = subprocess.run(
            [str(binary), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("cannot execute the pinned stock Codex binary") from exc
    if version != CODEX_CLI_VERSION:
        raise ValueError("stock Codex executable version drifted")
    try:
        with tempfile.TemporaryDirectory(prefix="yeto-codex-schema-") as temporary:
            root = Path(temporary)
            output = root / "schema"
            subprocess.run(
                [
                    str(binary),
                    "app-server",
                    "generate-json-schema",
                    "--experimental",
                    "--out",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            generated_schema = (
                output / "codex_app_server_protocol.v2.schemas.json"
            )
            if (
                generated_schema.is_symlink()
                or not generated_schema.is_file()
            ):
                raise ValueError("live stock Codex app-server schema drifted")
            _verify_live_codex_app_server_schema(schema, generated_schema)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("cannot verify the stock Codex app-server schema") from exc
    expected_env = {
        "YETO_CODEX_BINARY_PATH": CODEX_CONTAINER_BINARY_PATH,
        "YETO_CODEX_BINARY_SHA256": CODEX_LINUX_BINARY_SHA256,
        "YETO_CODEX_BINARY_SIZE_BYTES": str(CODEX_LINUX_BINARY_SIZE_BYTES),
        "YETO_CODEX_VERSION": CODEX_CLI_VERSION,
        "YETO_CODEX_APP_SERVER_PROTOCOL_REVISION": (
            CODEX_APP_SERVER_PROTOCOL_REVISION
        ),
        "YETO_CODEX_APP_SERVER_SCHEMA_SHA256": CODEX_APP_SERVER_SCHEMA_SHA256,
        "YETO_CODEX_BASE_INSTRUCTIONS_SHA256": contract[
            "base_instructions_sha256"
        ],
        "YETO_CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256": contract[
            "terminal_exec_tool_schema_sha256"
        ],
        "YETO_CODEX_SUBMIT_TOOL_SCHEMA_SHA256": contract[
            "submit_tool_schema_sha256"
        ],
        "YETO_CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256": contract[
            "dynamic_tools_schema_sha256"
        ],
        "YETO_CODEX_REASONING_EFFORT": "xhigh",
        "YETO_CODEX_BACKEND_MAX_TOKENS": str(args.rollout_max_response_len),
        "YETO_CODEX_BACKEND_REASONING_EFFORT": expected_backend[
            "reasoning_effort"
        ],
        "YETO_CODEX_BACKEND_THINKING": "enabled",
        "YETO_CODEX_CHAT_TEMPLATE": expected_backend["chat_template"],
        "YETO_CODEX_CHAT_TEMPLATE_KWARGS": json.dumps(
            expected_backend["chat_template_kwargs"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "YETO_CODEX_TITO_ALLOWED_APPEND_ROLES": "tool,user",
        "YETO_CODEX_HARNESS_CONTRACT_SHA256": _canonical_sha256(contract),
    }
    mismatched = [
        name for name, expected in expected_env.items() if os.getenv(name) != expected
    ]
    if mismatched:
        raise ValueError(
            "stock Codex container environment drifted: " + ", ".join(mismatched)
        )
    if args.custom_agent_function_path == CODEX_OPENENV_AGENT:
        _preflight_codex_openenv_adapter(args, profile_name)


def _provider_value(provider, *names: str):
    for name in names:
        value = getattr(provider, name, None)
        if value is not None:
            return value
    raise ValueError(
        f"Megatron-Bridge provider {type(provider).__name__} lacks {names[0]}"
    )


def _positive_int(provider, *names: str) -> int:
    value = _provider_value(provider, *names)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Megatron-Bridge provider has invalid {names[0]}={value!r}")
    return value


def _text(value: Any) -> str:
    return str(getattr(value, "value", value))


def _miles_callable(spec: str) -> str:
    module, separator, function = spec.partition(":")
    if not separator or not module or not function.isidentifier():
        raise ValueError("RL callable must be package.module:function")
    return f"{module}.{function}"


_ROUTED_EXPERT_LORA_MODULE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.\d+\."
    r"(?P<projection>gate_proj|up_proj|down_proj)$"
)

_MEGATRON_LAYER_MODULE = re.compile(
    r"^(?P<prefix>(?:.*\.)?decoder\.layers\.)\d+(?P<suffix>\..+)$"
)


def _pipeline_local_target(target: str, pipeline_parallel: int) -> str:
    if pipeline_parallel <= 1:
        return target
    # Pipeline stages renumber their physical decoder layers from zero.
    # Bridge's PEFT matcher accepts ``*`` here, so retain the exact module
    # type while making the target independent of that local renumbering.
    # Canonical HF specs still define and validate global layer ownership.
    match = _MEGATRON_LAYER_MODULE.fullmatch(target)
    if match is None:
        return target
    return f"{match.group('prefix')}*{match.group('suffix')}"


def megatron_adapter_targets(
    specs,
    bridge,
    *,
    standard_grouped_experts: bool = False,
    pipeline_parallel: int = 1,
) -> list[str]:
    """Map the exact PEFT contract onto Bridge's Megatron module paths."""

    model_bridge = getattr(bridge, "_model_bridge", None)
    if model_bridge is None:
        raise ValueError("Megatron-Bridge does not expose its model mapping")
    registry = model_bridge.mapping_registry()
    modules = {
        spec.name.removeprefix("base_model.model.").rsplit(".lora_", 1)[0]
        for spec in specs
    }
    targets = set()
    for module in modules:
        expert = _ROUTED_EXPERT_LORA_MODULE.fullmatch(module)
        if standard_grouped_experts and expert is not None:
            branch = (
                "linear_fc2"
                if expert.group("projection") == "down_proj"
                else "linear_fc1"
            )
            targets.add(
                _pipeline_local_target(
                    f"decoder.layers.{expert.group('layer')}.mlp.experts.{branch}",
                    pipeline_parallel,
                )
            )
            continue
        hf_weight = f"{module}.weight"
        mapping = registry.hf_to_megatron_lookup(hf_weight)
        if mapping is None:
            raise ValueError(f"PEFT module {module!r} has no Megatron-Bridge mapping")
        megatron_module = mapping.megatron_param.removesuffix(".weight")
        prefix, separator, leaf = megatron_module.rpartition(".")
        if not separator:
            raise ValueError(f"invalid Megatron adapter module {megatron_module!r}")
        if leaf in {"linear_qkv", "linear_fc1"}:
            hf_params = mapping.hf_param
            component = next(
                (
                    name
                    for name, value in hf_params.items()
                    if value == hf_weight
                ),
                None,
            ) if isinstance(hf_params, dict) else None
            allowed = {"q", "k", "v"} if leaf == "linear_qkv" else {"gate", "up"}
            if component not in allowed:
                raise ValueError(
                    f"PEFT module {module!r} cannot use canonical Megatron LoRA"
                )
            leaf = (
                f"linear_{component}"
                if leaf == "linear_qkv"
                else f"linear_fc1_{component}"
            )
        target = f"{prefix}.{leaf}"
        targets.add(_pipeline_local_target(target, pipeline_parallel))
    return sorted(targets)


def build_miles_argv(
    args,
    *,
    model_path: str | Path,
    rollout_model_path: str | Path | None = None,
    prompt_path: str | Path,
    eval_prompt_path: str | Path | None = None,
    provider,
    target_modules: list[str],
    yeto_policy_sync: bool = True,
) -> list[str]:
    """Construct Miles arguments from Bridge's actual model provider.

    Two layers: ``resolve_rl_run_config`` validates and resolves the
    engine-agnostic run; ``_legacy_miles_argv`` is the pure legacy (agentenv
    Miles fork) translation of that config.
    """

    from .engine.run_config import resolve_rl_run_config

    config = resolve_rl_run_config(
        args,
        model_path=model_path,
        rollout_model_path=rollout_model_path,
        prompt_path=prompt_path,
        eval_prompt_path=eval_prompt_path,
        provider=provider,
        target_modules=target_modules,
        yeto_policy_sync=yeto_policy_sync,
    )
    return _legacy_miles_argv(config)


def _legacy_miles_argv(config) -> list[str]:
    """Translate an ``RLRunConfig`` into legacy Miles argv (order is load-bearing)."""

    from .engine.run_config import (
        RECIPE_DEEPSEEK_V4_FLASH,
        RECIPE_QWEN3_5,
        lr_schedule_argv,
    )

    geometry = config.geometry
    parallel = config.parallel
    trainable = config.trainable
    batch = config.batch
    recipe = config.model_recipe
    serving = config.serving
    agent = config.agent
    columns = config.data.columns
    lora = trainable.parameter_mode == "lora"
    deepseek = recipe.name == RECIPE_DEEPSEEK_V4_FLASH

    if config.algorithm.advantage_estimator != "grpo":
        raise ValueError("legacy Miles translation supports only GRPO")
    if parallel.colocated:
        placement_values = ["--colocate"]
    else:
        placement_values = [
            "--rollout-num-gpus",
            str(parallel.dedicated_rollout_gpus),
            "--bridge-distributed-weight-sync",
            "--allow-missing-unquantized-weight-update-hooks",
            "--update-weight-transfer-mode",
            "broadcast",
            "--rollout-weight-version-format",
            "yeto-policy",
        ]

    model_recipe_values: list[str] = []
    if deepseek:
        model_name = "deepseekv4"
    elif recipe.name == RECIPE_QWEN3_5:
        model_name = "qwen3_5"
        model_recipe_values = [
            "--spec",
            "miles_plugins.models.qwen3_5",
            "get_qwen3_5_spec",
            "--apply-layernorm-1p",
            "--attention-output-gate",
            "--attention-dropout",
            "0.0",
            "--hidden-dropout",
            "0.0",
        ]
    else:
        model_name = recipe.provider_class

    lora_values: list[str] = []
    if lora:
        lora_values = [
            "--lora-rank",
            str(trainable.lora_rank),
            "--lora-alpha",
            str(trainable.lora_rank),
            "--lora-dropout",
            "0",
            "--lora-type",
            "lora" if trainable.routed_expert_lora else "canonical_lora",
            "--target-modules",
            ",".join(trainable.target_modules),
            "--lora-base-cpu-backup",
        ]

    values = [
        "train.py",
        "--train-backend", "megatron",
        "--hf-checkpoint", config.hf_checkpoint,
        "--ref-load", config.ref_load,
        "--megatron-to-hf-mode", "bridge",
        "--model-name", model_name,
        *model_recipe_values,
        "--num-layers", str(geometry.num_layers),
        "--hidden-size", str(geometry.hidden_size),
        "--num-attention-heads", str(geometry.num_attention_heads),
        "--num-query-groups", str(geometry.num_query_groups),
        "--kv-channels", str(geometry.kv_channels),
        "--ffn-hidden-size", str(geometry.ffn_hidden_size),
        "--max-position-embeddings", str(geometry.max_position_embeddings),
        "--seq-length", str(batch.seq_len),
        "--normalization", geometry.normalization,
        "--norm-epsilon", str(geometry.norm_epsilon),
        "--position-embedding-type", geometry.position_embedding_type,
        "--rotary-base", str(geometry.rotary_base),
        "--rotary-percent", str(geometry.rotary_percent),
        "--vocab-size", str(geometry.vocab_size),
        *lora_values,
        "--actor-num-nodes", str(parallel.actor_num_nodes),
        "--actor-num-gpus-per-node", str(parallel.actor_num_gpus_per_node),
        "--num-gpus-per-node", str(parallel.visible_gpus_per_node),
        "--rollout-num-gpus-per-engine", str(parallel.rollout_num_gpus_per_engine),
        *placement_values,
        "--offload-train" if serving.offload_train else "--no-offload-train",
        "--sglang-mem-fraction-static", str(serving.mem_fraction_static),
        "--tensor-model-parallel-size", str(parallel.tensor_parallel),
        "--pipeline-model-parallel-size", str(parallel.pipeline_parallel),
        "--context-parallel-size", "1",
        "--expert-model-parallel-size", str(parallel.expert_parallel),
        "--expert-tensor-parallel-size", "1",
        "--prompt-data", config.data.prompt_path,
        "--input-key", columns.input_key,
        "--label-key", columns.label_key,
        "--metadata-key", columns.metadata_key,
        "--rollout-seed", str(config.algorithm.rollout_seed),
        "--num-rollout", str(0 if batch.eval_only else batch.global_rounds),
        "--rollout-batch-size", str(batch.groups_per_round),
        "--n-samples-per-prompt", str(batch.samples_per_group),
        "--over-sampling-batch-size", str(batch.over_sampling_batch_size),
        "--num-steps-per-rollout", str(batch.optimizer_steps),
        "--global-batch-size", str(batch.global_batch),
        *lr_schedule_argv(config.algorithm.lr_schedule),
        "--balance-data",
        "--rollout-max-context-len", str(batch.seq_len),
        "--rollout-max-response-len", str(batch.rollout_max_response_len),
        "--rollout-function-path",
        (
            "yeto.rl.miles.generate_rollout"
            if config.yeto_policy_sync
            else "miles.rollout.sglang_rollout.generate_rollout"
        ),
        "--custom-rm-path", _miles_callable(config.algorithm.reward_function),
        "--advantage-estimator", "grpo",
        "--lr", str(config.algorithm.lr),
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend", recipe.training_attention_backend,
        "--no-gradient-accumulation-fusion",
        "--bf16",
        "--no-load-optim",
        "--no-load-rng",
        "--no-save-optim",
        "--no-save-rng",
        "--finetune",
        "--seed", str(config.algorithm.seed),
        "--pin-rollout-manager-to-head",
    ]
    if lora:
        values.extend(("--sglang-max-lora-rank", str(trainable.lora_rank)))
    evaluation = config.eval
    if evaluation is not None:
        values.extend(
            (
                "--eval-function-path",
                "yeto.rl.miles.generate_rollout",
                "--eval-prompt-data",
                evaluation.dataset_name,
                evaluation.prompt_path,
                "--eval-interval",
                str(evaluation.interval),
                "--skip-eval-before-train",
                "--n-samples-per-eval-prompt",
                str(evaluation.samples_per_prompt),
                "--log-passrate",
            )
        )
        for flag, value in (
            ("--eval-temperature", evaluation.temperature),
            ("--eval-top-p", evaluation.top_p),
            ("--eval-max-prompt-len", evaluation.max_prompt_len),
            ("--eval-max-response-len", evaluation.max_response_len),
            ("--eval-max-context-len", evaluation.max_context_len),
        ):
            if value is not None:
                values.extend((flag, str(value)))
    if trainable.expert_full:
        values.extend(
            (
                "--optimizer", "adam",
                "--adam-beta1", "0.9",
                "--adam-beta2", "0.98",
                "--adam-eps", str(1e-8),
                "--weight-decay", "0",
                "--clip-grad", "1.0",
                "--kl-coef", "0.001",
            )
        )
    if serving.deterministic_inference:
        values.append("--sglang-enable-deterministic-inference")
    if deepseek:
        values.extend(
            (
                "--transformer-impl", "transformer_engine",
                "--qkv-format", "bshd",
                "--recompute-granularity", "full",
                "--recompute-method", "uniform",
                "--recompute-num-layers", "1",
                "--micro-batch-size", "1",
                "--train-memory-margin-bytes", str(3 * 1024**3),
                "--moe-token-dispatcher-type", "alltoall",
                "--moe-router-freeze-gate",
                "--freeze-e-score-correction-bias",
                "--attention-dropout", "0.0",
                "--hidden-dropout", "0.0",
                "--sglang-moe-runner-backend", "triton",
                "--sglang-disable-shared-experts-fusion",
            )
        )
        if trainable.routed_expert_lora:
            values.append("--no-sglang-lora-use-virtual-experts")
    if parallel.uneven_pipeline_layers is not None:
        first_layers, last_layers = parallel.uneven_pipeline_layers
        values.extend(
            (
                "--decoder-first-pipeline-num-layers",
                str(first_layers),
                "--decoder-last-pipeline-num-layers",
                str(last_layers),
            )
        )
    if config.data.apply_chat_template:
        values.append("--apply-chat-template")
    if parallel.tensor_parallel > 1:
        values.append("--sequence-parallel")
    values.extend(
        ("--distributed-timeout-minutes", str(config.distributed_timeout_minutes))
    )
    if config.data.chat_template_kwargs:
        values.extend(
            (
                "--apply-chat-template-kwargs",
                json.dumps(
                    config.data.chat_template_kwargs,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
    if config.yeto_policy_sync:
        policy_sync_path = (
            "yeto.rl.miles_full_parameter_dense."
            "create_miles_full_parameter_dense_sync"
            if trainable.parameter_mode == "full"
            else "yeto.rl.miles.create_policy_sync"
        )
        values.extend(
            (
                "--rollout-all-samples-process-path",
                "yeto.rl.miles.queue_completed_groups",
                "--external-policy-sync-path",
                policy_sync_path,
            )
        )
    ports = config.ports
    for flag, value in (
        ("--rollout-engine-base-port", ports.rollout_engine_base_port),
        ("--sglang-router-port", ports.sglang_router_port),
        ("--sglang-router-prometheus-port", ports.sglang_router_prometheus_port),
        ("--train-master-base-port", ports.train_master_base_port),
        ("--sglang-tp-size", serving.tp_size),
        ("--sglang-dp-size", serving.dp_size),
        ("--sglang-ep-size", serving.ep_size),
        ("--sglang-attention-backend", serving.attention_backend),
        ("--sglang-page-size", serving.page_size),
        ("--sglang-max-running-requests", serving.max_running_requests),
        ("--sglang-chunked-prefill-size", serving.chunked_prefill_size),
    ):
        if value is not None:
            values.extend((flag, str(value)))
    if agent.use_rollout_routing_replay:
        values.append("--use-rollout-routing-replay")
    if agent.custom_generate_function_path:
        values.extend(
            ("--custom-generate-function-path", agent.custom_generate_function_path)
        )
    if agent.custom_agent_function_path:
        values.extend(
            (
                "--custom-agent-function-path",
                agent.custom_agent_function_path,
                "--max-seq-len",
                str(agent.agent_max_seq_len),
            )
        )
    if agent.dynamic_sampling_filter_path:
        values.extend(
            ("--dynamic-sampling-filter-path", agent.dynamic_sampling_filter_path)
        )
    if agent.use_session_server:
        values.append("--use-session-server")
        if agent.session_server_ip:
            values.extend(("--session-server-ip", agent.session_server_ip))
        if agent.session_server_port:
            values.append("--session-server-port")
            values.extend(str(port) for port in agent.session_server_port)
        if agent.tito_model:
            from miles.utils.chat_template_utils import (
                resolve_reasoning_and_tool_call_parser,
            )

            values.extend(("--tito-model", agent.tito_model))
            reasoning_parser, tool_call_parser = (
                resolve_reasoning_and_tool_call_parser(agent.tito_model)
            )
            if reasoning_parser is not None:
                values.extend(("--sglang-reasoning-parser", reasoning_parser))
            if tool_call_parser is not None:
                values.extend(("--sglang-tool-call-parser", tool_call_parser))
        if agent.tito_allowed_append_roles:
            values.append("--tito-allowed-append-roles")
            values.extend(agent.tito_allowed_append_roles)
    if geometry.group_query_attention:
        values.append("--group-query-attention")
    if geometry.rope_type is not None:
        values.extend(("--rope-type", geometry.rope_type))
    if geometry.mrope_section is not None:
        values.append("--mrope-section")
        values.extend(str(value) for value in geometry.mrope_section)
    if geometry.gated_linear_unit:
        values.append("--swiglu")
    if geometry.untie_embeddings_and_output_weights:
        values.append("--untie-embeddings-and-output-weights")
    if geometry.disable_bias_linear:
        values.append("--disable-bias-linear")
    if geometry.add_qkv_bias:
        values.append("--add-qkv-bias")
    if geometry.qk_layernorm:
        values.append("--qk-layernorm")
    if recipe.gdn.gated_delta_net:
        values.extend(("--qkv-format", recipe.gdn.qkv_format))
    moe = geometry.moe
    if moe is not None:
        values.extend(
            (
                "--num-experts", str(moe.num_experts),
                "--moe-ffn-hidden-size", str(moe.ffn_hidden_size),
                "--moe-router-topk", str(moe.router_topk),
                "--moe-layer-freq", str(moe.layer_freq),
            )
        )
        if moe.shared_expert_intermediate_size is not None:
            values.extend(
                (
                    "--moe-shared-expert-intermediate-size",
                    str(moe.shared_expert_intermediate_size),
                )
            )
    if geometry.multi_latent_attention:
        values.append("--multi-latent-attention")
        for name, value in geometry.mla_dims:
            values.extend((f"--{name.replace('_', '-')}", str(value)))
    return values


def _column(row: dict[str, Any], column: str, flag: str) -> Any:
    """A user-named column; a row without it is an error naming the flag,
    never a silent fallback to another column."""
    if column not in row:
        raise ValueError(f"RL rows have no column {column!r} ({flag})")
    return row[column]


def _messages(row: dict[str, Any], prompt_column: str | None = None) -> list[dict[str, Any]]:
    if prompt_column is not None:
        value = _column(row, prompt_column, "--rl-prompt-column")
    else:
        value = row.get("messages", row.get("prompt", row.get("input")))
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, dict) for item in value)
    ):
        raise ValueError("RL rows must contain messages or a string prompt/input")
    return value


def _prompt_split(source, revision: str | None) -> str:
    """`train`, or a Hub dataset's only split when it has no `train`
    (e.g. HuggingFaceH4/MATH-500 ships just `test`)."""
    if not isinstance(source, str) or os.path.exists(os.path.expanduser(source)):
        return "train"
    try:
        from datasets import get_dataset_split_names

        names = list(get_dataset_split_names(source, revision=revision))
    except Exception:  # noqa: BLE001 - load_rows reports the real error
        return "train"
    if "train" not in names and len(names) == 1:
        return names[0]
    return "train"


def prepare_prompt_data(
    source: str,
    revision: str | None,
    output_path: str | Path,
    prompt_column: str | None = None,
    label_column: str | None = None,
) -> Path:
    from ..data import load_rows

    split = _prompt_split(source, revision)
    if split == "train":
        rows = load_rows(source, revision=revision)
    else:
        rows = load_rows(source, split=split, revision=revision)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for raw in rows:
            row = dict(raw)
            metadata = dict(row.get("metadata") or {})
            metadata.update(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"messages", "metadata"}
                }
            )
            normalized = {
                "messages": _messages(row, prompt_column),
                "label": (
                    _column(row, label_column, "--rl-label-column")
                    if label_column is not None
                    else row.get("label")
                ),
                "metadata": metadata,
            }
            if "tools" in row:
                normalized["tools"] = row["tools"]
            handle.write(
                json.dumps(
                    normalized,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("RL prompt dataset is empty")
    os.replace(temporary, output)
    return output


def _verify_eval_dataset_identity(args) -> Path | None:
    """Verify the immutable source bytes for an eval-only or dense heldout run."""

    if getattr(args, "eval_interval", None) is None:
        if getattr(args, "eval_only", False) or getattr(args, "eval_data", None):
            raise ValueError("--eval-only requires evaluation configuration")
        return None
    eval_only = getattr(args, "eval_only", False)
    dense_train_eval = (
        not eval_only
        and getattr(args, "parameter_mode", None) == "full"
        and getattr(args, "sync_preset", None) == "dense-full"
    )
    if not eval_only and not dense_train_eval:
        raise ValueError(
            "evaluation configuration requires --eval-only or dense full mode"
        )
    expected = str(getattr(args, "eval_data_sha256", "") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("evaluation dataset requires an immutable SHA256")
    source_value = args.data if eval_only else getattr(args, "eval_data", None)
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("dense full evaluation requires --eval-data")
    source = Path(source_value).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ValueError("evaluation requires one regular local dataset file")
    if dense_train_eval and source.resolve() == Path(args.data).expanduser().resolve():
        raise ValueError("dense full evaluation must use a distinct heldout dataset")
    from ..provenance import file_sha256

    actual = file_sha256(source)
    if actual != expected:
        raise ValueError(
            f"evaluation dataset SHA256 mismatch: expected {expected}, got {actual}"
        )
    return source


def _jsonl_record_count(path: str | Path) -> int:
    source = Path(path)
    count = 0
    try:
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    raise ValueError("evaluation prompt data contains a blank row")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("evaluation prompt row is not a JSON object")
                count += 1
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("evaluation prompt data is not canonical JSONL") from exc
    if count < 1:
        raise ValueError("evaluation prompt data is empty")
    return count


def _parse_miles_args(argv: list[str]):
    previous = sys.argv
    try:
        sys.argv = argv
        from miles.utils.arguments import parse_args as parse_miles_args

        return parse_miles_args()
    finally:
        sys.argv = previous


def _syncer_address(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host:
        raise ValueError("--syncer must be HOST:PORT")
    return host, int(port)


EXTERNAL_ROUTER_ENV = "YETO_RL_EXTERNAL_ROUTER"


def require_ports_router_mode(miles_args, environ=None) -> None:
    """Ports path: upstream Miles owns the SGLang router.

    Upstream launches the router as an ``inference-router`` Ray worker with
    a 120 s readiness budget (the legacy 30 s deadline that
    ``start_external_sglang_router`` works around is gone), and it removed
    external router mode: a pre-set ``sglang_router_ip`` without its
    per-model router map is an assertion failure.  So ``YETO_RL_EXTERNAL_ROUTER``
    is a no-op here, and a pre-set router address is refused before launch.
    """

    environ = os.environ if environ is None else environ
    if getattr(miles_args, "sglang_router_ip", None) is not None:
        raise ValueError(
            "--rl-engine ports does not support an external SGLang router "
            "(--sglang-router-ip); upstream Miles launches its own router"
        )
    if environ.get(EXTERNAL_ROUTER_ENV) == "1":
        print(
            f"[rl] {EXTERNAL_ROUTER_ENV}=1 ignored on --rl-engine ports: "
            "upstream Miles launches the SGLang router as a Ray worker",
            flush=True,
        )


def start_external_sglang_router(
    miles_args,
    *,
    timeout_s: float = 300.0,
    popen=subprocess.Popen,
    connect=None,
    sleep=None,
):
    """Start the SGLang router ourselves and hand Miles its address.

    Miles launches the router in a `spawn` child that must re-import
    `miles.utils.http_utils` (which pulls in Megatron-Bridge, ~27 s on a
    Modal H100 container) and then gives it a hard-coded 30 s to listen,
    so on slower CPUs the island dies before the first rollout. Miles
    skips its own launch when `sglang_router_ip` is set; the standalone
    router CLI starts in ~2 s and gets a generous deadline here."""
    import atexit
    import random
    import socket
    import time

    from miles.utils.http_utils import find_available_port, get_host_info

    connect = connect or (lambda host, port: socket.create_connection((host, port), timeout=1).close())
    sleep = sleep or time.sleep
    host = get_host_info()[1]
    port = getattr(miles_args, "sglang_router_port", None) or find_available_port(
        random.randint(3000, 4000)
    )
    command = [
        sys.executable, "-m", "sglang_router.launch_router",
        "--host", host,
        "--port", str(port),
        "--prometheus-port", str(find_available_port(random.randint(4000, 5000))),
        "--log-level", "warn",
        "--request-timeout-secs",
        str(getattr(miles_args, "sglang_router_request_timeout_secs", 14400)),
    ]
    policy = getattr(miles_args, "sglang_router_policy", None)
    if policy:
        command.extend(("--policy", str(policy)))
    process = popen(command)
    atexit.register(process.terminate)
    deadline = time.monotonic() + timeout_s
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang router exited with {process.returncode} before listening")
        try:
            connect(host, port)
            break
        except OSError:
            if time.monotonic() > deadline:
                process.terminate()
                raise RuntimeError(f"SGLang router at {host}:{port} not ready after {timeout_s}s")
            sleep(0.5)
    miles_args.sglang_router_ip = host
    miles_args.sglang_router_port = port
    print(f"[rl] external SGLang router listening at {host}:{port}", flush=True)
    return process


def run_miles(
    args,
    *,
    model_path: str | Path,
    rollout_model_path: str | Path | None = None,
    prompt_path: str | Path,
    eval_prompt_path: str | Path | None = None,
    yeto_policy_sync: bool = True,
    extra_argv: Sequence[str] = (),
) -> None:
    """Run one Miles job, optionally with Yeto's external policy boundary."""

    rl_engine = getattr(args, "rl_engine", "ports")
    if rl_engine not in ("legacy", "ports"):
        raise ValueError(f"unknown rl_engine {rl_engine!r}")
    _check_ports_algorithm_options(args, outer_sync=yeto_policy_sync)
    if rl_engine == "ports" and (
        getattr(args, "rl_single_island_no_sync", False) or getattr(args, "rl_echo_events", False)
    ):
        install_event_echo()
    if rl_engine == "ports":
        _require_ports_supported(args, extra_argv)
        from .engine.miles_adapter.state import require_run_plugin

        require_run_plugin()
    parameter_mode = getattr(args, "parameter_mode", "lora")
    sync_preset = getattr(args, "sync_preset", "strict-avg")
    dense_full = parameter_mode == "full" or sync_preset == "dense-full"
    if dense_full and not (
        parameter_mode == "full" and sync_preset == "dense-full"
    ):
        raise ValueError("dense-full sync and full parameter mode must be selected together")
    if dense_full:
        if not yeto_policy_sync:
            raise ValueError("dense full-parameter GRPO requires Yeto policy sync")
        miles_source_sha256 = getattr(args, "miles_source_sha256", None)
        if (
            type(miles_source_sha256) is not str
            or len(miles_source_sha256) != 64
            or any(value not in "0123456789abcdef" for value in miles_source_sha256)
        ):
            raise ValueError(
                "dense full-parameter GRPO requires an exact Miles source SHA256"
            )
        if (
            args.optimizer_steps != 1
            or getattr(args, "local_horizon", 1) != 1
        ):
            raise ValueError("dense full-parameter GRPO requires H=1")
        if getattr(args, "eval_only", False):
            raise ValueError("Milestone-1 dense full-parameter mode does not run eval-only")
        if (
            type(args.global_rounds) is not int
            or args.global_rounds < 1
            or type(args.fragments) is not int
            or args.fragments < 1
            or args.total_fragment_steps != args.global_rounds * args.fragments
            or not isinstance(args.parameter_layout_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", args.parameter_layout_sha256)
        ):
            raise ValueError(
                "dense full-parameter fragment budget must equal rounds*fragments"
            )
        num_learners = getattr(args, "num_learners", 1)
        generation = getattr(args, "learner_generation", 0)
        if (
            type(num_learners) is not int
            or num_learners < 1
            or type(args.learner_id) is not int
            or args.learner_id not in range(num_learners)
            or type(generation) is not int
            or generation != 0
        ):
            raise ValueError(
                "Milestone-1 dense full-parameter mode requires a fixed generation-0 roster"
            )

    if (
        yeto_policy_sync
        and sync_preset == "decoupled"
        and args.optimizer_steps != 1
    ):
        raise ValueError("decoupled RL requires one optimizer step per rollout")

    clone_only_lora = (
        parameter_mode == "lora"
        and args.lora_targets == "attention-routed-experts"
    )
    expert_full = int(getattr(args, "expert_full_count", 0) or 0) > 0
    if dense_full and expert_full:
        raise ValueError("full-parameter mode cannot also select expert-full tuning")

    def require_env(name: str, value: str) -> None:
        current = os.environ.get(name)
        if current not in (None, value):
            raise ValueError(f"{name} must be unset or {value}, got {current!r}")
        os.environ[name] = value

    if dense_full:
        _require_unquantized_dense_full_rollout_model(
            rollout_model_path or model_path
        )
        require_env("MILES_EXPERIMENTAL_FT_TRAINER", "0")

    if clone_only_lora or expert_full:
        require_env("YETO_DSV4_EXPERT_CLONE", "1")
    if clone_only_lora:
        require_env("YETO_DSV4_CLONE_ONLY_LORA", "1")
    if clone_only_lora or expert_full:
        fuse_wqa_wkv = os.environ.get("SGLANG_OPT_FUSE_WQA_WKV")
        if fuse_wqa_wkv not in (None, "0"):
            raise ValueError(
                "SGLANG_OPT_FUSE_WQA_WKV must be unset or 0 for V4 attention LoRA"
            )
        os.environ["SGLANG_OPT_FUSE_WQA_WKV"] = "0"
    if expert_full:
        require_env("YETO_DSV4_EXPERT_FULL", "1")
        require_env(
            "YETO_DSV4_EXPERT_FULL_COUNT",
            str(args.expert_full_count),
        )
        require_env("YETO_DSV4_EXPERT_FULL_LR", str(args.expert_full_lr))
        require_env("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")

    if getattr(args, "rl_model_recipe", "generic") == "deepseek-v4-flash":
        from .deepseek_v4_bridge import ensure_deepseek_v4_bridge

        ensure_deepseek_v4_bridge()

    from megatron.bridge import AutoBridge

    model_bridge = AutoBridge.from_hf_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    provider = model_bridge.to_megatron_provider(load_weights=False)
    provider.finalize()
    attention_specs = () if dense_full else derive_peft_lora_specs(
        model_path,
        None,
        rank=args.lora_r,
        targets=args.lora_targets,
        trust_remote_code=args.trust_remote_code,
    )
    specs = tuple(attention_specs)
    if expert_full:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=args.trust_remote_code,
        )
        specs = tuple(
            sorted(
                specs
                + expert_full_specs(
                    config,
                    expert_count=args.expert_full_count,
                    expected_selection_sha256=args.expert_selection_sha256,
                    expected_selection_contract_sha256=(
                        args.expert_selection_contract_sha256
                    ),
                )
            )
        )
    canonical_targets = [] if dense_full else adapter_targets(attention_specs)
    miles_targets = [] if dense_full else megatron_adapter_targets(
        attention_specs,
        model_bridge,
        standard_grouped_experts=clone_only_lora,
        pipeline_parallel=getattr(args, "pipeline_parallel", 1),
    )
    ports_launch = ports_algorithm = None
    if rl_engine == "ports":
        # Same engine-agnostic RLRunConfig as legacy; only the translation
        # differs (design D8).
        from .engine.miles_adapter.config import parse_miles_args
        from .engine.run_config import resolve_rl_run_config

        run_config = resolve_rl_run_config(
            args,
            model_path=model_path,
            rollout_model_path=rollout_model_path,
            prompt_path=prompt_path,
            eval_prompt_path=eval_prompt_path,
            provider=provider,
            # Upstream Miles resolves HF module names itself (canonical_lora
            # via Bridge); the fork's per-layer Megatron names have no Bridge
            # mapping upstream.
            target_modules=canonical_targets,
            yeto_policy_sync=yeto_policy_sync,
        )
        ports_launch = build_ports_launch(args, run_config, extra_argv)
        ports_algorithm = ports_launch.algorithm
        miles_argv = list(ports_launch.argv)
        miles_args = parse_miles_args(ports_launch)
        verify_ports_algorithm(args, miles_args, ports_launch)
    else:
        miles_argv = build_miles_argv(
            args,
            model_path=model_path,
            rollout_model_path=rollout_model_path,
            prompt_path=prompt_path,
            eval_prompt_path=eval_prompt_path,
            provider=provider,
            target_modules=miles_targets,
            yeto_policy_sync=yeto_policy_sync,
        )
        _reject_lr_schedule_overrides(extra_argv)
        miles_argv.extend(extra_argv)
        miles_args = _parse_miles_args(miles_argv)

    if yeto_policy_sync:
        if dense_full:
            from ..provenance import file_sha256

            model_config_path = Path(model_path) / "config.json"
            if not model_config_path.is_file():
                raise ValueError("full-parameter model has no config.json")
            model_config_hash = file_sha256(model_config_path)
            training_contract_hash = _dense_full_training_contract_sha256(
                args,
                miles_argv,
                model_config_sha256=model_config_hash,
                prompt_data_sha256=file_sha256(prompt_path),
            )
            layout_hash = model_config_hash
            lora_config_hash = _canonical_sha256(
                {
                    "schema": "yeto-full-parameter-mode-v1",
                    "model_config_sha256": model_config_hash,
                }
            )
        else:
            from .core import (
                canonical_layout_hash,
                canonical_lora_config_hash,
            )

            layout_hash = canonical_layout_hash(specs)
            lora_config_hash = canonical_lora_config_hash(
                rank=args.lora_r,
                target_modules=canonical_targets,
            )
        miles_args.yeto_rl_trust_remote_code = args.trust_remote_code
        miles_args.yeto_rl_model = args.model
        miles_args.yeto_rl_data = args.data
        miles_args.yeto_rl_base_model_revision = args.model_revision
        miles_args.yeto_rl_rollout_model_revision = (
            getattr(args, "rollout_model_revision", None) or args.model_revision
        )
        miles_args.yeto_rl_data_revision = args.data_revision
        miles_args.yeto_rl_lora_config_hash = lora_config_hash
        miles_args.yeto_rl_layout_hash = layout_hash
        miles_args.yeto_rl_parameter_mode = parameter_mode
        miles_args.yeto_rl_clone_only_lora = clone_only_lora
        if expert_full:
            miles_args.yeto_rl_expected_specs = specs
        if clone_only_lora:
            # The Bridge exposes one representative mapping per packed expert
            # parameter.  Miles needs the sparse canonical clone set at the
            # external policy boundary and reconstructs frozen originals as zeros.
            miles_args.yeto_rl_canonical_lora_names = tuple(
                spec.name for spec in specs
            )
        miles_args.yeto_rl_reward_sha256 = args.reward_sha256
        miles_args.yeto_rl_codex_harness_contract = getattr(
            args,
            "codex_harness_contract",
            None,
        )
        miles_args.yeto_rl_codex_backend_profile = getattr(
            args,
            "codex_backend_profile",
            None,
        )
        miles_args.yeto_rl_dynamic_sampling_max_replacements = getattr(
            args, "dynamic_sampling_max_replacements", None
        )
        miles_args.yeto_rl_secrlenv_max_infrastructure_replacements = getattr(
            args, "secrlenv_max_infrastructure_replacements", None
        )
        miles_args.yeto_rl_completed_groups_path = args.completed_groups_path
        miles_args.yeto_rl_event_tape = args.event_tape
        miles_args.yeto_rl_learner_id = args.learner_id
        # W&B reads the same namespace the rest of the yeto context rides in;
        # yeto.rl.wandb_rl starts the island's run off these.
        miles_args.wandb = getattr(args, "wandb", False)
        miles_args.wandb_project = getattr(args, "wandb_project", "yeto")
        miles_args.wandb_entity = getattr(args, "wandb_entity", None)
        miles_args.wandb_mode = getattr(args, "wandb_mode", "online")
        if getattr(args, "eval_only", False):
            miles_args.yeto_rl_eval_policy_version = args.global_rounds
        elif dense_full and getattr(args, "eval_interval", None) is not None:
            if eval_prompt_path is None:
                raise ValueError("dense full evaluation prompt path is unavailable")
            summary_path = Path(
                str(getattr(args, "eval_summary_path", "") or "")
            ).expanduser()
            if (
                not summary_path.is_absolute()
                or summary_path.exists()
                or summary_path.is_symlink()
                or not summary_path.parent.is_dir()
                or summary_path.parent.is_symlink()
            ):
                raise ValueError(
                    "dense full evaluation summary requires a fresh absolute path"
                )
            miles_args.yeto_rl_eval_policy_versions = (0, args.global_rounds)
            miles_args.yeto_rl_eval_dataset_name = args.eval_dataset_name
            miles_args.yeto_rl_eval_prompt_count = _jsonl_record_count(
                eval_prompt_path
            )
            miles_args.yeto_rl_eval_samples_per_prompt = (
                args.eval_samples_per_prompt
            )
            miles_args.yeto_rl_eval_summary_path = str(summary_path)
        miles_args.yeto_rl_sync_preset = sync_preset
        miles_args.external_policy_sync_run_until_stop = sync_preset == "decoupled"
        if dense_full:
            from .dense_sweep_wire import DenseSweepConfig
            from .local_learner import ComponentIdentity
            from .miles_full_parameter_dense import MilesDenseFullParameterConfig

            evidence_parent = (
                Path(args.audit_dir).expanduser()
                if getattr(args, "audit_dir", None)
                else Path(args.completed_groups_path).expanduser().parent
            )
            evidence_parent.mkdir(parents=True, exist_ok=True)
            evidence_directory = Path(
                tempfile.mkdtemp(
                    prefix=f"trajectory-evidence-{args.learner_id}-",
                    dir=evidence_parent,
                )
            )
            if evidence_directory.stat().st_mode & 0o077:
                raise RuntimeError("trajectory evidence directory is not private")
            learner_generations = (0,) * getattr(args, "num_learners", 1)
            miles_args.yeto_rl_model_config_sha256 = model_config_hash
            miles_args.yeto_rl_miles_source_sha256 = args.miles_source_sha256
            miles_args.yeto_rl_dense_training_contract_sha256 = (
                training_contract_hash
            )
            miles_args.external_policy_identity_setter_path = (
                "yeto.rl.miles.set_current_published_policy_identity"
            )
            miles_args.yeto_rl_trajectory_evidence_dir = str(evidence_directory)
            miles_args.yeto_rl_trajectory_evidence_kind = "secrlenv"
            miles_args.yeto_rl_trajectory_evidence_schema_version = (
                2 if bool(getattr(miles_args, "sao_compaction", False)) else 1
            )
            miles_args.yeto_rl_num_fragments = args.fragments
            miles_args.yeto_rl_total_sweeps = args.global_rounds
            miles_args.yeto_rl_total_fragment_steps = args.total_fragment_steps
            miles_args.yeto_rl_learner_generation = 0
            miles_args.yeto_rl_learner_generations = learner_generations
            miles_args.yeto_rl_dense_full_parameter_config = (
                MilesDenseFullParameterConfig(
                    component=ComponentIdentity(
                        role="actor",
                        model_revision=args.model_revision.lower(),
                        config_hash=model_config_hash,
                    ),
                    wire=DenseSweepConfig(
                        syncer_addr=_syncer_address(args.syncer),
                        learner_id=args.learner_id,
                        learner_generation=0,
                        policy_rounds=args.global_rounds,
                        wan_streams=args.wan_streams,
                    ),
                    learner_generations=learner_generations,
                    minimum_fragments=args.fragments,
                    training_contract_hash=training_contract_hash,
                    expected_layout_hash=args.parameter_layout_sha256,
                )
            )
        elif sync_preset == "decoupled":
            from ..protocol import layout_fingerprint
            from .core import build_rl_fragment_layout

            miles_args.yeto_rl_source_sha256 = args.source_sha256
            miles_args.yeto_rl_initial_adapter = getattr(
                args, "initial_adapter", None
            )
            miles_args.yeto_rl_initial_adapter_sha256 = getattr(
                args, "initial_adapter_sha256", None
            )
            total_fragment_steps = args.total_fragment_steps
            if total_fragment_steps != args.global_rounds * args.fragments:
                raise ValueError(
                    "decoupled RL total fragment steps must equal sweeps*fragments"
                )
            layout = build_rl_fragment_layout(specs, args.fragments)
            sync_fingerprint = layout_fingerprint(layout).hex()
            miles_args.yeto_rl_num_fragments = args.fragments
            miles_args.yeto_rl_pipeline = args.pipeline
            miles_args.yeto_rl_local_horizon = args.local_horizon
            miles_args.yeto_rl_total_sweeps = args.global_rounds
            miles_args.yeto_rl_total_fragment_steps = total_fragment_steps
            miles_args.yeto_rl_sync_layout_fingerprint = sync_fingerprint
            miles_args.yeto_rl_learner_budget_steps = getattr(
                args,
                "learner_budget_steps",
                None,
            )
            miles_args.yeto_rl_bridge_config = DecoupledBridgeConfig(
                syncer_addr=_syncer_address(args.syncer),
                learner_id=args.learner_id,
                total_fragment_steps=total_fragment_steps,
                num_fragments=args.fragments,
                pipeline=args.pipeline,
                local_horizon=args.local_horizon,
                expected_specs=specs,
                base_model_revision=args.model_revision,
                lora_config_hash=lora_config_hash,
                canonical_layout_hash=layout_hash,
                wan_streams=args.wan_streams,
                learner_budget_steps=miles_args.yeto_rl_learner_budget_steps,
            )
        else:
            miles_args.yeto_rl_bridge_config = BridgeConfig(
                syncer_addr=_syncer_address(args.syncer),
                learner_id=args.learner_id,
                global_rounds=args.global_rounds,
                groups_per_round=args.groups_per_round,
                samples_per_group=args.samples_per_group,
                local_optimizer_steps=args.optimizer_steps,
                wan_streams=args.wan_streams,
                expected_specs=specs,
                base_model_revision=args.model_revision,
                lora_config_hash=lora_config_hash,
                layout_hash=layout_hash,
                event_tape=args.event_tape,
                audit_dir=args.audit_dir,
                send_initial_params=not getattr(args, "eval_only", False),
            )

    if yeto_policy_sync:
        _configure_applied_lr(
            args, miles_args, rl_engine, dense_full=dense_full
        )
    _configure_grad_audit(args, miles_args, rl_engine)

    if rl_engine == "ports":
        require_ports_router_mode(miles_args)
    elif os.environ.get(EXTERNAL_ROUTER_ENV) == "1":
        start_external_sglang_router(miles_args)

    if rl_engine == "ports":
        _run_ports(
            args,
            miles_args,
            ports_launch,
            ports_algorithm,
            specs=specs,
            canonical_targets=canonical_targets,
            yeto_policy_sync=yeto_policy_sync,
        )
        # Last tape record: a rebuilt (no-sync) tape without it is incomplete.
        _append_ports_event(args, miles_args, {"event": "rl_learner_finalized"})
        print(f"[rl] learner {args.learner_id} finalized (rl_engine=ports)")
        return

    from train import train as miles_train

    asyncio.run(miles_train(miles_args))
    print(f"[rl] learner {args.learner_id} finalized")


def _reject_lr_schedule_overrides(extra_argv: Sequence[str]) -> None:
    """Legacy twin of the ports ``check_extra_argv`` LR-schedule check.

    The schedule is decided by ``resolve_lr_schedule``; an extra-argv override
    (e.g. warmup, or a decay style for decoupled) would silently diverge from
    ports and could trip or defeat the zero-LR invariant.
    """

    from .engine.run_config import LR_SCHEDULE_FLAGS

    for token in extra_argv:
        flag = str(token).split("=", 1)[0]
        if flag in LR_SCHEDULE_FLAGS:
            raise ValueError(
                f"{flag} is owned by yeto's LR schedule and cannot be overridden"
            )


def _configure_applied_lr(args, miles_args, rl_engine: str, *, dense_full: bool) -> bool:
    """Legacy: record each optimizer step's applied LR (zero-LR invariant, D4).

    Ports reads it in the state plugin's ``train_one_step`` recorder. Legacy
    uses the fork's before-train-step hook with the combined
    :data:`yeto.rl.applied_lr.HOOK_PATH`, which also runs the grad audit hook
    when that is configured (``_configure_grad_audit`` keeps it).
    """

    from . import applied_lr

    if rl_engine == "ports" or dense_full or getattr(args, "eval_only", False):
        return False
    existing = getattr(miles_args, "custom_megatron_before_train_step_hook_path", None)
    if existing and existing != applied_lr.HOOK_PATH:
        raise ValueError(
            f"applied-LR recording conflicts with before-train-step hook {existing!r}"
        )
    parent = Path(args.event_tape).expanduser().parent
    parent.mkdir(parents=True, exist_ok=True)
    directory = tempfile.mkdtemp(prefix=f"applied-lr-{args.learner_id}-", dir=parent)
    setattr(miles_args, applied_lr.APPLIED_LR_DIR_ATTR, directory)
    miles_args.custom_megatron_before_train_step_hook_path = applied_lr.HOOK_PATH
    return True


def _configure_grad_audit(args, miles_args, rl_engine: str) -> bool:
    """``YETO_RL_AUDIT_GRADS=1`` (teacher forcing, design D12): export pre-clip LoRA grads.

    Both engines write next to the round audit (``args.audit_dir``). Ports arms
    the capture from its ``train_one_step`` recorder; legacy through Miles'
    before-train-step hook, which the fork already calls (fork code untouched).
    """

    from . import grad_audit

    if not grad_audit.enabled():
        return False
    directory = getattr(args, "audit_dir", None)
    if not directory:
        raise ValueError(f"{grad_audit.GRAD_AUDIT_ENV}=1 requires an audit_dir")
    setattr(miles_args, grad_audit.GRAD_AUDIT_DIR_ATTR, str(Path(directory).expanduser()))
    if rl_engine != "ports":
        from . import applied_lr

        existing = getattr(miles_args, "custom_megatron_before_train_step_hook_path", None)
        if existing == applied_lr.HOOK_PATH:
            # Combined hook: records the applied LR, then runs the grad audit
            # (it reads the audit directory set above).
            return True
        if existing and existing != grad_audit.HOOK_PATH:
            raise ValueError(
                f"{grad_audit.GRAD_AUDIT_ENV}=1 conflicts with before-train-step hook {existing!r}"
            )
        miles_args.custom_megatron_before_train_step_hook_path = grad_audit.HOOK_PATH
    return True


def _run_ports(
    args,
    miles_args,
    launch,
    algorithm,
    *,
    specs,
    canonical_targets,
    yeto_policy_sync: bool,
) -> None:
    """Ports path: yeto's IslandDriver over upstream Miles (design D2)."""

    from .core import canonical_layout_hash, canonical_lora_config_hash
    from .engine.miles_adapter.entry import run_ports_island

    layout_hash = canonical_layout_hash(specs)
    lora_config_hash = canonical_lora_config_hash(
        rank=args.lora_r, target_modules=canonical_targets
    )
    # The event tape and island identity are needed even without outer sync.
    miles_args.yeto_rl_event_tape = args.event_tape
    miles_args.yeto_rl_learner_id = args.learner_id
    miles_args.yeto_rl_engine = "ports"
    apply_ports_infra_switches(args, miles_args)
    run_ports_island(
        miles_args,
        launch,
        algorithm,
        learner_id=args.learner_id,
        base_model_revision=args.model_revision,
        lora_config_hash=lora_config_hash,
        layout_hash=layout_hash,
        yeto_policy_sync=yeto_policy_sync,
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    from ..provenance import (
        is_immutable_commit,
        is_local_reference,
        python_spec_sha256,
        verify_source_tree_sha256,
    )

    verify_source_tree_sha256(args.source_sha256)
    if not is_immutable_commit(args.model_revision):
        raise ValueError("RL model revision must be an immutable commit")
    if not is_local_reference(args.data) and not is_immutable_commit(
        args.data_revision
    ):
        raise ValueError("RL remote dataset revision must be an immutable commit")
    reward_sha256 = python_spec_sha256(args.reward_function)
    if reward_sha256 != args.reward_sha256.lower():
        raise ValueError(
            f"reward source SHA256 mismatch: expected {args.reward_sha256.lower()}, "
            f"got {reward_sha256}"
        )
    miles_root = str(Path(args.miles_root).expanduser().resolve())
    if miles_root not in sys.path:
        sys.path.insert(0, miles_root)
    if getattr(args, "rl_engine", "ports") == "ports":
        from . import MILES_NEXT_PINS

        verify_miles_revision(miles_root, expected=MILES_NEXT_PINS)
    else:
        verify_miles_revision(
            miles_root,
            expected_source_sha256=(
                args.miles_source_sha256
                if args.parameter_mode == "full"
                else None
            ),
        )
    _preflight_codex_harness(args)

    from miles.utils.misc import load_function

    load_function(_miles_callable(args.reward_function))
    if args.custom_generate_function_path:
        load_function(args.custom_generate_function_path)
    if args.custom_agent_function_path:
        load_function(args.custom_agent_function_path)
    from . import CODEX_HARNESS_AGENT

    if args.custom_agent_function_path in {
        "yeto_miles_secrlenv.agent.run",
        CODEX_HARNESS_AGENT,
    }:
        from yeto_miles_secrlenv.client import require_daemon_ready

        require_daemon_ready()
        print("[rl] secrlenv episode daemon ready")
    from huggingface_hub import snapshot_download

    from ..models import resolve
    model = resolve(args.model)
    if is_local_reference(model):
        model_path = str(Path(model).expanduser().resolve())
    else:
        model_path = snapshot_download(repo_id=model, revision=args.model_revision)
    rollout_model = resolve(args.rollout_model) if args.rollout_model else model
    if rollout_model == model:
        rollout_model_path = model_path
    elif is_local_reference(rollout_model):
        rollout_model_path = str(Path(rollout_model).expanduser().resolve())
        if not Path(rollout_model_path).is_dir():
            raise ValueError("--rollout-model local directory does not exist")
    else:
        rollout_model_path = snapshot_download(
            repo_id=rollout_model,
            revision=args.rollout_model_revision,
        )
    eval_source = _verify_eval_dataset_identity(args)
    columns = dict(
        prompt_column=getattr(args, "rl_prompt_column", None),
        label_column=getattr(args, "rl_label_column", None),
    )
    prompt_path = prepare_prompt_data(
        args.data,
        args.data_revision,
        "~/yeto-rl/prompts.jsonl",
        **columns,
    )
    eval_prompt_path = None
    if eval_source is not None:
        eval_prompt_path = (
            prompt_path
            if getattr(args, "eval_only", False)
            else prepare_prompt_data(
                str(eval_source),
                None,
                "~/yeto-rl/eval-prompts.jsonl",
                **columns,
            )
        )
    run_miles(
        args,
        model_path=model_path,
        rollout_model_path=rollout_model_path,
        prompt_path=prompt_path,
        eval_prompt_path=eval_prompt_path,
        # --rl-single-island-no-sync: LocalOnlySync, no syncer connection.
        yeto_policy_sync=not getattr(args, "rl_single_island_no_sync", False),
    )


if __name__ == "__main__":
    main()
