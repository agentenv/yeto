"""Effective breaker values of a config, read through today's code paths (golden lock).

Each value comes from the code that applies it today: the run up monitor (``yeto.rl.up.Run``),
the ``yeto launch`` parser and cost guard config, the launcher's stall rule, the trainer's
buffer reader env, the sampler spec readers (``engine_sampler_entry``, ``agentic_round``,
``store_dispense``), the harness env readers and the pool ``ElasticConfig``. Keys are
``<registry entry>.<param>`` in canonical units (seconds, USD, counts) plus
``<entry>.enabled``; a disabled entry records ``enabled: false`` and no params.

``python -m tests.golden.breakers.capture --write`` rewrites ``golden.json``. Do that only for
a new case or a key rename, never to accept a changed value: a migration must keep every
value of every existing case (test_breakers_golden.py).
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import inspect
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
GOLDEN = Path(__file__).with_name("golden.json")
CASES_DIR = Path(__file__).with_name("cases")
EXAMPLES = REPO / "examples"
# placeholder values a config needs to parse (an int flag, a file the loader opens)
PLACEHOLDERS = {"<total-steps>": "100"}


@dataclass
class Case:
    """One config. ``run``: a run config (path or dict); without it ``launch_argv`` is the
    trainer argv and ``pool`` / ``spec`` stand alone. ``trainer_env`` / ``sampler_env``: the
    environment each component sees (the trainer also gets its argv ``--modal-env`` /
    ``--island-env`` pairs). ``edit``: changes the loaded run config, pool or spec."""

    name: str
    run: str | dict[str, Any] | None = None
    launch_argv: list[str] | None = None
    pool: str | dict[str, Any] | None = None
    spec: str | dict[str, Any] | None = None
    trainer_env: dict[str, str] = field(default_factory=dict)
    sampler_env: dict[str, str] = field(default_factory=dict)
    edit: Callable[[dict[str, Any]], None] | None = None
    edit_pool: Callable[[dict[str, Any]], None] | None = None
    edit_spec: Callable[[dict[str, Any]], None] | None = None


def _load(ref: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    if isinstance(ref, dict):
        return copy.deepcopy(ref)
    return json.loads((REPO / ref).read_text())


def _fill(argv: list[Any]) -> list[str]:
    return [PLACEHOLDERS.get(str(a), str(a)) for a in argv]


def _sibling(ref: str | None, value: str) -> str:
    """A ``<name.json>`` placeholder next to ``ref`` (the example's own file), else ``value``."""
    m = re.fullmatch(r"<([\w.\-]+\.json)>", str(value))
    if ref and m and (REPO / ref).parent.joinpath(m.group(1)).is_file():
        return str((REPO / ref).parent / m.group(1))
    return value


@contextlib.contextmanager
def _environ(env: Mapping[str, str]) -> Iterator[None]:
    """``os.environ`` as exactly ``env`` (the readers below read the process env)."""
    with mock.patch.dict(os.environ, dict(env), clear=True):
        yield


def _pairs(values: list[str] | None) -> dict[str, str]:
    out = {}
    for item in values or ():
        key, _, value = str(item).partition("=")
        out[key] = value
    return out


# ----------------------------------------------------------------------------- run
def run_values(cfg: dict[str, Any], state_root: Path) -> dict[str, Any]:
    from yeto.launch.pool.controller import DEFAULT_STALL_S
    from yeto.rl import up, up_consistency

    run = up.Run(cfg, state_root=state_root)
    b = cfg.get("budget") or {}
    out: dict[str, Any] = {}
    capped = up_consistency.budget_capped(b)
    out["run.budget.enabled"] = capped
    if capped:
        out["run.budget.usd"] = up_consistency.budget_usd(b)
        out["run.budget.force_pools_ratio"] = up.FORCE_FACTOR
        out["run.budget.release_pools_on_trip"] = run.release_nc_on_breach
    wall_h = up_consistency.budget_wall_h(b)
    out["run.wall_clock.enabled"] = wall_h is not None
    if wall_h is not None:
        out["run.wall_clock.hours"] = wall_h
        out["run.wall_clock.implicit"] = "wall_h" not in b
    out["run.trainer_silence.enabled"] = True
    out["run.trainer_silence.effective_limit_s"] = run.silence_limit_s()
    out["run.trainer_silence.floor_s"] = up_consistency.number_or_default(b, "silence_min", 30) * 60
    out["run.trainer_silence.stall_factor_n"] = 4  # the silence limit is max(4 x --rl-stall-timeout, floor_s)
    out["run.first_group.enabled"] = True
    out["run.first_group.timeout_s"] = float((cfg.get("first_group") or {}).get("timeout_s", 5400))
    starve = run.starvation_config
    out["run.starvation.enabled"] = starve.enabled
    if starve.enabled:
        out["run.starvation.alert_s"] = starve.alert_s or None  # 0 (minutes): that verdict off
        out["run.starvation.stop_s"] = starve.stop_s or None
        out["run.starvation.unhealthy_s"] = starve.unhealthy_s
        out["run.starvation.stop_grace_s"] = starve.stop_grace_s
        out["run.starvation.trainer_wait_max_s"] = starve.trainer_wait_max_s
    stall = float(up.rollout_section(cfg).get("stall_alert_s", DEFAULT_STALL_S))
    out["run.rollout_stall_alert.enabled"] = stall > 0
    if stall > 0:
        out["run.rollout_stall_alert.limit_s"] = stall
    sb = cfg.get("sandbox") or {}
    on = bool(sb.get("max_run_containers") or sb.get("max_containers_per_sampler"))
    out["run.sandbox_containers.enabled"] = on
    if on:
        out["run.sandbox_containers.max_run_n"] = int(sb["max_run_containers"]) if sb.get("max_run_containers") else None
        out["run.sandbox_containers.max_per_sampler_n"] = (int(sb["max_containers_per_sampler"])
                                                           if sb.get("max_containers_per_sampler") else None)
        out["run.sandbox_containers.limit_n"] = run.container_limit()
    dash = run.dashboard_args()
    out["monitor.dashboard.budget_usd"] = float(dash[dash.index("--budget") + 1]) if "--budget" in dash else None
    return out


# ----------------------------------------------------------------------------- launch + trainer
def launch_values(argv: list[str]) -> dict[str, Any]:
    from yeto.cost.guard import GuardConfig, guard_config_from_args, guard_enabled
    from yeto.launch.fleet import DEFAULT_RL_STALL_TIMEOUT_S
    from yeto.modal_runner import DEFAULT_RETRIES, DEFAULT_TIMEOUT_S
    from yeto.rl import up_consistency

    args = up_consistency.trainer_args(argv)
    if args is None:
        raise ValueError(f"trainer argv does not parse as an RL launch: {argv}")
    out: dict[str, Any] = {}
    guard = guard_enabled(args)
    cfg = guard_config_from_args(args)
    on = guard and cfg.budget_usd is not None
    out["launch.budget.enabled"] = on
    if on:
        out["launch.budget.usd"] = cfg.budget_usd
        out["launch.budget.warn_usd"] = cfg.warn_usd
        out["launch.budget.stop_grace_s"] = cfg.stop_grace_s
    on = guard and cfg.idle.enabled
    out["launch.idle.enabled"] = on
    if on:
        out["launch.idle.warn_s"] = cfg.idle.idle_after_s
        out["launch.idle.stop_after_warn_s"] = cfg.idle.stop_after_s
        out["launch.idle.load_grace_s"] = cfg.idle.load_grace_s
        out["launch.idle.max_load_s"] = cfg.idle.max_load_s
    out["launch.nc_lease_overdue.enabled"] = guard
    if guard:
        out["launch.nc_lease_overdue.grace_s"] = GuardConfig.nc_overdue_grace_s
    # yeto.launcher: the fleet's stall_timeout
    raw = getattr(args, "rl_stall_timeout", None)
    stall = float(raw or DEFAULT_RL_STALL_TIMEOUT_S) if raw != 0 else 0.0
    out["launch.stall.enabled"] = stall > 0
    if stall > 0:
        out["launch.stall.timeout_s"] = stall
    out["launch.provider_timeout.timeout_s"] = float(args.modal_timeout_s if args.modal_timeout_s is not None
                                                     else DEFAULT_TIMEOUT_S)
    out["launch.provider_timeout.retries_n"] = int(args.modal_retries if args.modal_retries is not None
                                                   else DEFAULT_RETRIES)
    out["switch.cost_guard"] = guard
    return out


def trainer_values(argv: list[str], env: Mapping[str, str]) -> dict[str, Any]:
    from yeto.rl import up_consistency
    from yeto.rl.adapters.miles import buffer_rollout as br
    from yeto.rl.engine.nan_skip import NanSkipPolicy
    from yeto.rl.engine.xhw_guard import XhwGuard

    args = up_consistency.trainer_args(argv)
    # the island env: the component's env, then the argv's --modal-env / --island-env pairs,
    # then what the launch layer exports itself (buffer_rollout.island_env)
    island = {**env, **_pairs(args.modal_env), **_pairs(getattr(args, "island_env", None))}
    age = int(getattr(args, "rl_max_policy_age", 0) or 0)
    island.update(br.island_env(br.TRAINER, store_root="s3://golden/run", max_policy_age=age,
                                wait_s=getattr(args, "rl_sample_wait_s", None)))
    out: dict[str, Any] = {}
    out["trainer.sample_wait.enabled"] = True
    out["trainer.sample_wait.timeout_s"] = float(island.get(br.WAIT_ENV, br.DEFAULT_WAIT_S))
    out["trainer.policy_age.enabled"] = age > 0
    if age > 0:
        out["trainer.policy_age.max_versions"] = age
    nan = NanSkipPolicy.from_env(island)
    out["trainer.nan.enabled"] = True
    out["trainer.nan.skip"] = nan is not None
    if nan is not None:
        out["trainer.nan.skip_max_consecutive_n"] = nan.max_consecutive
        out["trainer.nan.skip_max_fraction"] = nan.max_fraction
        out["trainer.nan.skip_min_rounds_n"] = nan.min_rounds
    xhw = XhwGuard.from_env(island)
    out["trainer.xhw_guard.enabled"] = xhw is not None
    if xhw is not None:
        out["trainer.xhw_guard.action"] = xhw.action
        for lvl in ("warn", "act"):
            for metric in ("abs_diff", "kl", "clip"):
                out[f"trainer.xhw_guard.{lvl}_{metric}"] = getattr(getattr(xhw, lvl), metric)
        out["trainer.xhw_guard.window_n"] = xhw.window
        out["trainer.xhw_guard.min_rounds_n"] = xhw.min_rounds
        out["trainer.xhw_guard.quarantine_rounds_n"] = xhw.quarantine_rounds
        out["trainer.xhw_guard.weight"] = xhw.weight
        out["trainer.xhw_guard.max_cond"] = xhw.max_cond
        out["trainer.xhw_guard.confidence_z"] = xhw.confidence_z
    # buffer_rollout_miles: int(os.environ.get(MAX_DAMAGED_RUN_ENV) or DEFAULT_MAX_DAMAGED_RUN); 0: no limit
    damaged = int(island.get(br.MAX_DAMAGED_RUN_ENV) or br.DEFAULT_MAX_DAMAGED_RUN)
    out["trainer.buffer_damaged.enabled"] = damaged > 0
    if damaged > 0:
        out["trainer.buffer_damaged.max_run_n"] = damaged
    return out


# ----------------------------------------------------------------------------- sampler + harness
def sampler_values(spec: dict[str, Any]) -> dict[str, Any]:
    from yeto.rl.agentic.types import AgentBudget
    from yeto.rl.engine import agentic_round, sampler_loop, store_dispense
    from yeto.rl.engine.engine_sampler import DEFAULT_WRITE_DEFER_MAX_S
    from yeto.rl.engine.incarnation_lease import DEFAULT_MAX_UNREAD_S

    out: dict[str, Any] = {}
    out["sampler.load_failures.enabled"] = True
    out["sampler.load_failures.max_consecutive_n"] = int(spec.get("max_consecutive_load_failures", 3))
    out["sampler.store_read_outage.enabled"] = True
    out["sampler.store_read_outage.limit_s"] = sampler_loop.DEFAULT_MAX_READ_OUTAGE_S
    out["sampler.store_read_outage.max_damaged_latest_n"] = sampler_loop.DEFAULT_MAX_DAMAGED_LATEST_READS
    out["sampler.write_defer.enabled"] = True
    out["sampler.write_defer.limit_s"] = (float(spec["write_defer_max_s"]) if spec.get("write_defer_max_s") is not None
                                          else DEFAULT_WRITE_DEFER_MAX_S)
    lease = spec.get("incarnation_lease", True)
    on = not (lease is False or lease is None)
    out["sampler.incarnation_lease.enabled"] = on
    if on:
        section = {} if lease is True else dict(lease)
        unread = section.get("max_unread_s", DEFAULT_MAX_UNREAD_S)
        out["sampler.incarnation_lease.max_unread_s"] = None if unread is None else float(unread)
    out["sampler.drain.reload_max_defer_s"] = agentic_round.reload_max_defer_s(spec)
    out["sampler.loop_guard.enabled"] = False  # EngineSampler(loop_guard=...) only; no spec or env entry
    agentic = spec.get("agentic") if isinstance(spec.get("agentic"), Mapping) else None
    if agentic is not None:
        timeout, grace = agentic_round.drain_settings(spec, drain=False)  # raw; the cut below
        out["sampler.drain.enabled"] = True
        out["sampler.drain.timeout_s"] = timeout
        out["sampler.drain.abandon_grace_s"] = grace
        out["sampler.drain.effective_cut_s"] = agentic_round.effective_drain_timeout(
            timeout, agentic_round.reload_max_defer_s(spec))
        out["sampler.min_samples.enabled"] = True
        out["sampler.min_samples.min_n"] = int(agentic.get("min_samples", 2))
        out["sampler.infra_retries.enabled"] = True
        out["sampler.infra_retries.max_n"] = int(agentic.get("infra_retries", agentic_round.DEFAULT_INFRA_RETRIES))
        defaults = dict(agentic.get("task_defaults") or {})
        out["sampler.trial_wall_clock.enabled"] = True
        out["sampler.trial_wall_clock.limit_s"] = float(defaults.get("wall_clock_s", AgentBudget.wall_clock_s))
    dispense = (spec.get("prompts") or {}).get("dispense") if isinstance(spec.get("prompts"), Mapping) else None
    if isinstance(dispense, Mapping) and dispense.get("backend") == "store":
        out["sampler.dispense_redo.enabled"] = True
        out["sampler.dispense_redo.max_n"] = int(dispense.get("max_redo", store_dispense.DEFAULT_MAX_REDO))
        out["sampler.dispense_lease.enabled"] = True
        out["sampler.dispense_lease.lease_s"] = float(dispense.get("lease_s", store_dispense.DEFAULT_LEASE_S))
    return out


def _module_constants(module: str, names: tuple[str, ...]) -> dict[str, Any]:
    """Literal module-level constants read from the source (the module imports Harbor, which
    the Python 3.11 CI job does not install)."""
    path = REPO / (module.replace(".", "/") + ".py")
    found = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in names:
            found[node.targets[0].id] = ast.literal_eval(node.value)
    missing = set(names) - set(found)
    if missing:
        raise LookupError(f"{module}: no literal {sorted(missing)}")
    return found


def harness_values(env: Mapping[str, str]) -> dict[str, Any]:
    from types import SimpleNamespace

    from yeto.rl.harness.harbor import agentenv_client, agentenv_network, capacity
    from yeto.rl.harness.harbor.trial import HarborHarness, setup_retries_from_env

    agentenv_env = SimpleNamespace(**_module_constants(
        "yeto.rl.harness.harbor.agentenv_env", ("SKIP_REACH_ENV", "REACH_TRIES_ENV", "REACH_TIMEOUT_ENV")))
    out: dict[str, Any] = {}
    with _environ(env):
        out["harness.setup_retries.enabled"] = True
        out["harness.setup_retries.max_n"] = setup_retries_from_env()
        out["harness.setup_retries.backoff_initial_s"] = (
            inspect.signature(HarborHarness.__init__).parameters["retry_backoff_s"].default)
        skip = os.environ.get(agentenv_env.SKIP_REACH_ENV, "").strip() == "1"
        out["harness.reach_check.enabled"] = not skip
        if not skip:
            tries = os.environ.get(agentenv_env.REACH_TRIES_ENV, "").strip()
            out["harness.reach_check.attempts_n"] = int(tries) if tries else agentenv_network.DEFAULT_TRIES
            out["harness.reach_check.timeout_s"] = float(os.environ.get(agentenv_env.REACH_TIMEOUT_ENV, "").strip()
                                                         or agentenv_network.DEFAULT_TIMEOUT_S)
        out["harness.sandbox_caps.enabled"] = True
        out["harness.sandbox_caps.max_sandboxes"] = capacity.SandboxLimits.from_env().caps or None
        quota = agentenv_client.AgentenvQuota.from_env()
        out["harness.sandbox_caps.agentenv_max_active_n"] = quota.max_active
        out["harness.sandbox_caps.agentenv_starts_per_minute_n"] = quota.starts_per_minute
        out["harness.sandbox_caps.agentenv_starts_per_hour_n"] = quota.starts_per_hour
        out["harness.sandbox_caps.agentenv_create_timeout_s"] = float(
            os.environ.get(agentenv_client.REQUEST_TIMEOUT_ENV, "") or agentenv_client.DEFAULT_REQUEST_TIMEOUT_S)
        out["harness.sandbox_ttl.enabled"] = True
        out["harness.sandbox_ttl.limit_s"] = float(agentenv_client.DEFAULT_TTL_S)
    return out


# ----------------------------------------------------------------------------- pool
def pool_values(pool_cfg: dict[str, Any]) -> dict[str, Any]:
    from yeto.launch.pool.status import ElasticConfig

    e = ElasticConfig.from_dict(pool_cfg.get("elastic"))
    target = int(pool_cfg.get("target", 1))
    out: dict[str, Any] = {}
    out["pool.provision_breaker.enabled"] = True
    out["pool.provision_breaker.failures_n"] = e.breaker_failures
    out["pool.provision_breaker.cooldown_s"] = e.breaker_cooldown_s
    out["pool.provision_breaker.max_cooldown_s"] = e.breaker_max_cooldown_s
    out["pool.provision_breaker.max_trips_n"] = e.breaker_max_trips
    on = e.max_run_spend_usd is not None or e.max_run_spend_ucr is not None
    out["pool.spend_cap.enabled"] = on
    if on:
        out["pool.spend_cap.usd"] = e.max_run_spend_usd
        out["pool.spend_cap.ucr"] = e.max_run_spend_ucr
    out["pool.node_cap.enabled"] = True
    out["pool.node_cap.max_nodes_n"] = e.max_nodes  # null: 2 x target, applied by Budget at run time
    out["pool.node_cap.effective_max_nodes_n"] = e.max_nodes if e.max_nodes is not None else 2 * target
    out["pool.node_cap.max_orders_n"] = e.max_orders
    out["pool.crash_loop.enabled"] = True
    out["pool.crash_loop.action"] = e.crash_loop_before_ready
    out["pool.sampler_restarts.enabled"] = True
    out["pool.sampler_restarts.max_n"] = e.max_sampler_restarts
    out["pool.sampler_restarts.reset_s"] = e.sampler_restart_reset_s
    out["pool.engine_stall.enabled"] = e.engine_stall_s is not None
    if e.engine_stall_s is not None:
        out["pool.engine_stall.limit_s"] = e.engine_stall_s
        out["pool.engine_stall.max_restart_failures_n"] = e.max_stall_restart_failures
    out["pool.sampler_stall.enabled"] = e.stall_s is not None
    if e.stall_s is not None:
        out["pool.sampler_stall.limit_s"] = e.stall_s
    out["pool.start_timeout.enabled"] = True
    out["pool.start_timeout.limit_s"] = e.start_timeout_s
    out["pool.unreachable.enabled"] = True
    out["pool.unreachable.grace_s"] = e.unreachable_grace_s
    out["pool.fence_hold.enabled"] = e.fence_max_hold_s is not None
    if e.fence_max_hold_s is not None:
        out["pool.fence_hold.limit_s"] = e.fence_max_hold_s
    out["pool.lease_overdue.enabled"] = True
    out["pool.lease_overdue.grace_s"] = e.overdue_grace_s
    out["pool.control_plane_alert.enabled"] = True
    out["pool.control_plane_alert.limit_s"] = e.control_plane_alert_s
    out["pool.fence_alert.enabled"] = True
    out["pool.fence_alert.failures_n"] = e.fence_alert_failures
    out["pool.fence_alert.limit_s"] = e.fence_alert_s
    # yeto.launch.nc.cli.pool_config: the top-level lease-end knobs
    out["pool.lease_end.enabled"] = True
    out["pool.lease_end.replace_lead_s"] = float(pool_cfg.get("replace_lead_s", 3600))
    out["pool.lease_end.end_margin_s"] = float(pool_cfg.get("end_margin_s", 600))
    return out


# ----------------------------------------------------------------------------- monitor
def monitor_values() -> dict[str, Any]:
    from yeto.dashboard.alerts import DEFAULT_THRESHOLDS
    from yeto.rl import breaker

    out: dict[str, Any] = {f"monitor.health.{k}": v[0] for k, v in sorted(breaker.LIMITS.items())}
    out["monitor.health.nan_skip_min_rounds_n"] = breaker.NAN_SKIP_MIN_ROUNDS
    for hw, (warn, act) in sorted(breaker.LOGPROB_LIMITS.items()):
        out[f"monitor.health.logprob_{hw}_warn"] = warn
        out[f"monitor.health.logprob_{hw}_act"] = act
    out.update({f"monitor.dashboard.{k}": v for k, v in sorted(DEFAULT_THRESHOLDS.items())})
    return out


# ----------------------------------------------------------------------------- one case
@dataclass
class Loaded:
    """A case's surfaces as the components see them (what ``yeto.breakers.Sources`` takes)."""

    run_config: dict[str, Any] | None
    argv: list[str] | None
    pool: dict[str, Any] | None
    spec: dict[str, Any] | None
    trainer_env: dict[str, str]
    sampler_env: dict[str, str]


def load_case(case: Case, tmp: Path) -> Loaded:
    from yeto.rl import up_consistency

    cfg = _load(case.run)
    pool_ref = case.pool
    pool = _load(case.pool)
    if cfg is not None:
        if case.edit is not None:
            case.edit(cfg)
        cfg["trainer"]["argv"] = _fill(cfg["trainer"]["argv"])
        if pool is None and isinstance(cfg.get("pool"), dict):
            pool_ref = _sibling(case.run if isinstance(case.run, str) else None, cfg["pool"]["config"])
            pool = _load(os.path.relpath(pool_ref, REPO)) if Path(pool_ref).is_file() else None
    spec_ref = case.spec
    if pool is not None:
        if case.edit_pool is not None:
            case.edit_pool(pool)
        if pool.get("spec") and spec_ref is None:
            spec_ref = _sibling(pool_ref if isinstance(pool_ref, str) else None, pool["spec"])
            if Path(spec_ref).is_absolute():
                spec_ref = os.path.relpath(spec_ref, REPO)
    spec = _load(spec_ref) if spec_ref is not None and (isinstance(spec_ref, dict)
                                                       or (REPO / spec_ref).is_file()) else None
    if spec is not None and case.edit_spec is not None:
        case.edit_spec(spec)
    if pool is not None:
        spec_path = tmp / "sampler-spec.json"
        if spec is not None:
            spec_path.write_text(json.dumps(spec))
            pool["spec"] = str(spec_path)
        pool_path = tmp / "pool.json"
        pool_path.write_text(json.dumps(pool))
        if cfg is not None:
            cfg["pool"] = {**(cfg.get("pool") or {}), "config": str(pool_path)}
        merged = up_consistency.effective_spec(pool) if spec is not None else None
        spec = merged if merged is not None else spec
    argv = cfg["trainer"]["argv"] if cfg is not None else (_fill(case.launch_argv) if case.launch_argv else None)
    return Loaded(cfg, argv, pool, spec, dict(case.trainer_env), dict(case.sampler_env))


def capture(case: Case) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        loaded = load_case(case, tmp_path)
        out: dict[str, Any] = {}
        if loaded.run_config is not None:
            out.update(run_values(loaded.run_config, tmp_path / "state"))
        if loaded.argv is not None:
            out.update(launch_values(loaded.argv))
            out.update(trainer_values(loaded.argv, loaded.trainer_env))
        if loaded.spec is not None:
            out.update(sampler_values(loaded.spec))
            out.update(harness_values(loaded.sampler_env))
        if loaded.pool is not None:
            out.update(pool_values(loaded.pool))
    return dict(sorted(out.items()))


def capture_all() -> dict[str, Any]:
    from tests.golden.breakers.cases import CASES

    return {"cases": {c.name: capture(c) for c in CASES}, "constants": monitor_values()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--write", action="store_true", help="rewrite golden.json")
    args = p.parse_args(argv)
    doc = capture_all()
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if args.write:
        GOLDEN.write_text(text)
        print(f"wrote {GOLDEN.relative_to(REPO)}: {len(doc['cases'])} cases")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
