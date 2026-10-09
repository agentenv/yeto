"""Per-island parameter override for negative-test runs (launch-preflight-guards,
capability ``island-param-override``).

``--rl-island-override ISLAND:KEY=VALUE`` (repeatable) gives ONE island a
different value of a whitelisted parameter, to build a negative island on
real hardware (identity mismatch, different LR schedule, different
policy-age limit).  It needs ``--rl-negative-test-run``.  The launcher gives
the named island a copy of ``args`` with the new values (one place, so sky
and Modal islands follow), re-runs the checks that belong to each changed
parameter, writes the differences into the run manifest, and exports
``YETO_ISLAND_OVERRIDE`` (JSON) to the island, which writes an
``rl_island_override`` event at the head of its tape.

A negative-test run must never become a production run:

* the run manifest has ``negative_test: true`` (and the checkpoint store);
* the island writes ``YETO_NEGATIVE_TEST`` at the root of its checkpoint
  store; a launch that is not a negative-test run refuses a store with that
  marker (launcher, before any cloud work, when the store is readable from
  this machine; the island again at start, on every cloud);
* ``yeto merge`` refuses an adapter directory of a negative-test run.

(10-09 ruling of the main agent for the user: the original ``--rl-resume`` /
``yeto export`` do not exist on main; resume is ``--rl-checkpoint-store``,
export is ``yeto merge``.  ``--rl-max-carry-lag`` is a syncer parameter and
cannot be overridden per island.)

This module is imported by the launcher AND by the island (env helpers), so it
must not import the launcher at module level.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

OVERRIDE_ENV = "YETO_ISLAND_OVERRIDE"
NEGATIVE_RUN_ENV = "YETO_NEGATIVE_TEST_RUN"  # on every island of a negative-test run
OVERRIDE_EVENT = "rl_island_override"
NEGATIVE_MARKER = "YETO_NEGATIVE_TEST"
# The syncer takes the session contract / identity from the FIRST admitted
# connection.  An overridden island waits this long before it starts (and so
# before it connects), so the unchanged island sets the contract and the
# overridden one is the one refused (10-09 ruling of the main agent).
DEFAULT_JOIN_DELAY_S = 180.0

LR_SCHEDULES = ("auto", "linear", "constant")


def _lr(value: str) -> str:
    if value not in LR_SCHEDULES:
        raise ValueError(f"rl_lr_schedule must be one of {LR_SCHEDULES}, got {value!r}")
    return value


def _age(value: str) -> int:
    try:
        n = int(value)
    except ValueError as e:
        raise ValueError(f"rl_max_policy_age must be an integer, got {value!r}") from e
    if n < 0:
        raise ValueError("rl_max_policy_age must be >= 0")
    return n


def _salt(value: str) -> str:
    if not value or len(value) > 128:
        raise ValueError("identity_test_salt must be 1-128 characters")
    return value


# key -> (parser, human name for the dashboard / warning)
WHITELIST = {
    "rl_lr_schedule": (_lr, "学习率调度"),
    "rl_max_policy_age": (_age, "落后上限"),
    "identity_test_salt": (_salt, "岛身份测试扰动"),
}
SYNCER_ONLY = {"rl_max_carry_lag": "--rl-max-carry-lag"}


def _norm_key(key: str) -> str:
    return key.strip().lstrip("-").replace("-", "_")


def parse_overrides(items, num_islands: int) -> dict[int, dict[str, Any]]:
    """``["1:rl_lr_schedule=constant", ...]`` -> ``{1: {"rl_lr_schedule": "constant"}}``.
    Raises ValueError on a bad item, a key outside the whitelist or an island
    number out of range."""
    out: dict[int, dict[str, Any]] = {}
    for item in items or ():
        island_s, sep, rest = str(item).partition(":")
        key, sep2, value = rest.partition("=")
        if not sep or not sep2 or not island_s.strip().isdigit():
            raise ValueError(f"--rl-island-override {item!r}: expected ISLAND:KEY=VALUE")
        island = int(island_s)
        if not 0 <= island < num_islands:
            raise ValueError(f"--rl-island-override {item!r}: island {island} out of range "
                             f"(this launch has islands 0..{num_islands - 1})")
        key = _norm_key(key)
        if key in SYNCER_ONLY:
            raise ValueError(f"--rl-island-override {item!r}: {SYNCER_ONLY[key]} is a syncer "
                             "parameter, it cannot be changed per island (syncer 参数，不能按岛换)")
        if key not in WHITELIST:
            raise ValueError(f"--rl-island-override {item!r}: {key!r} cannot be overridden per "
                             f"island; allowed: {sorted(WHITELIST)}")
        if key in out.get(island, {}):
            raise ValueError(f"--rl-island-override: island {island} {key} given twice")
        out.setdefault(island, {})[key] = WHITELIST[key][0](value.strip())
    return out


def overrides_of(args, num_islands: int) -> dict[int, dict[str, Any]]:
    """Validated overrides of a launch (empty without the switch).  Raises before
    any cloud work when the negative-test switch is missing."""
    items = getattr(args, "rl_island_override", None) or []
    if not items:
        return {}
    if getattr(args, "training_mode", "sft") != "rl":
        raise ValueError("--rl-island-override needs --training-mode rl")
    if not getattr(args, "rl_negative_test_run", False):
        raise ValueError("--rl-island-override needs --rl-negative-test-run: a per-island override "
                         "is only for negative tests (负例运行)")
    return parse_overrides(items, num_islands)


def _old_value(args, key: str):
    if key == "identity_test_salt":
        return None
    value = getattr(args, key, None)
    if key == "rl_lr_schedule":
        value = value or "auto"
        if value == "auto":  # resolved as run_config.resolve_lr_schedule does
            mode = getattr(args, "rl_island_scheduling", None) or "legacy"
            run_until_stop = getattr(args, "sync_preset", None) == "decoupled" or mode == "elastic"
            return "constant" if run_until_stop else "linear"
        return value
    if key == "rl_max_policy_age":
        return int(value or 0)
    return value


def records(args, overrides: dict[int, dict[str, Any]]) -> list[dict]:
    return [{"island": island, "key": key, "name": WHITELIST[key][1],
             "old": _old_value(args, key), "new": value}
            for island, kv in sorted(overrides.items()) for key, value in sorted(kv.items())]


def check_island_args(args) -> None:
    """Re-run the checks of the whitelisted parameters on an island's copy."""
    sched = getattr(args, "rl_lr_schedule", None) or "auto"
    _lr(sched)
    if sched == "linear":
        from .launcher import island_scheduling_mode

        if getattr(args, "sync_preset", None) == "decoupled" or island_scheduling_mode(args) == "elastic":
            raise ValueError("--rl-lr-schedule linear needs a known local step count; decoupled and "
                             "elastic islands run until the syncer stops them (use auto or constant)")
    from .launcher import _rl_backend_module
    from .rl.engine.policy_age import validate_limit

    age = validate_limit(getattr(args, "rl_max_policy_age", 0) or 0)
    module = _rl_backend_module(args, "policy_age")
    module.SUPPORT.check(age)
    task_check = getattr(module, "check_task", None)
    if callable(task_check):
        task_check(age, custom_generate=getattr(args, "custom_generate_function_path", None),
                   custom_agent=getattr(args, "custom_agent_function_path", None))


def island_args(args, island: int, overrides: dict[int, dict[str, Any]]):
    """``args`` itself for an island without overrides, else a checked copy."""
    kv = overrides.get(island)
    if not kv:
        return args
    copied = copy.copy(args)
    for key, value in kv.items():
        if key != "identity_test_salt":
            setattr(copied, key, value)
    check_island_args(copied)
    return copied


def island_env(args, island: int, overrides: dict[int, dict[str, Any]]) -> dict[str, str]:
    """Env of one island: nothing for a normal run (island command and env
    unchanged); ``YETO_NEGATIVE_TEST_RUN=1`` on every island of a negative-test
    run; ``YETO_ISLAND_OVERRIDE`` (JSON) on an overridden island."""
    if not getattr(args, "rl_negative_test_run", False):
        return {}
    env = {NEGATIVE_RUN_ENV: "1"}
    kv = overrides.get(island)
    if kv:
        recs = [r for r in records(args, overrides) if r["island"] == island]
        delay = getattr(args, "rl_negative_join_delay_s", None)
        delay = DEFAULT_JOIN_DELAY_S if delay is None else float(delay)
        if delay < 0:
            raise ValueError("--rl-negative-join-delay-s must be >= 0")
        env[OVERRIDE_ENV] = json.dumps({"island": island, "negative_test": True, "overrides": recs,
                                        "join_delay_s": delay},
                                       sort_keys=True, separators=(",", ":"))
    return env


def apply_to_task(task, env: dict[str, str]) -> None:
    if not env:
        return
    if hasattr(task, "update_envs"):
        task.update_envs(env)
    else:  # test doubles
        task.envs = {**(getattr(task, "envs", None) or {}), **env}


def print_warning(args, overrides, out=None) -> None:
    out = sys.stderr if out is None else out
    bar = "!" * 72
    lines = [bar, "!! NEGATIVE-TEST RUN (负例运行): per-island parameter overrides are ON",
             "!! this run is ONLY for negative tests; it cannot be resumed by a normal run",
             "!! and `yeto merge` refuses its adapters"]
    for r in records(args, overrides):
        lines.append(f"!!   island {r['island']}: {r['key']} {r['old']!r} -> {r['new']!r}")
    lines.append(bar)
    print("\n".join(lines), file=out, flush=True)


def manifest_fields(args, overrides) -> dict:
    if not getattr(args, "rl_negative_test_run", False) and not overrides:
        return {}
    return {"negative_test": True, "island_overrides": records(args, overrides),
            "checkpoint_store": getattr(args, "rl_checkpoint_store", None)}


# --- island side --------------------------------------------------------------
def env_override(environ=None) -> dict | None:
    raw = (os.environ if environ is None else environ).get(OVERRIDE_ENV)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def identity_test_salt(environ=None) -> str | None:
    """The island's identity test salt (None unless this island is overridden)."""
    data = env_override(environ)
    for rec in (data or {}).get("overrides", ()):
        if rec.get("key") == "identity_test_salt":
            return str(rec.get("new"))
    return None


def override_event(data: dict, island_id: int) -> dict:
    import time

    return {"event": OVERRIDE_EVENT, "island_id": int(island_id), "time_unix": time.time(),
            "negative_test": True, "overrides": list(data.get("overrides", ()))}


def write_override_event(tape_path, island_id: int, environ=None) -> dict | None:
    """Island start: one ``rl_island_override`` record at the head of the tape."""
    data = env_override(environ)
    if not data:
        return None
    from .rl.event_echo import append_record

    record = override_event(data, island_id)
    append_record(tape_path, record)
    print(f"[yeto-island] NEGATIVE-TEST island {island_id}: overrides {data.get('overrides')}",
          flush=True)
    return record


# --- negative-test marker (resume / merge refusal) ------------------------------
def island_is_negative(environ=None) -> bool:
    return (os.environ if environ is None else environ).get(NEGATIVE_RUN_ENV) == "1"


def island_startup(args, island_id: int, environ=None, *, sleep=None) -> dict | None:
    """Island start (after the tape path is known, before training): write the
    override event, check / mark the checkpoint store, then (overridden island
    only) wait ``join_delay_s`` before anything connects to the syncer."""
    if sleep is None:
        import time

        sleep = time.sleep
    env = os.environ if environ is None else environ
    record = write_override_event(args.event_tape, island_id, env) if getattr(args, "event_tape", None) else None
    for name in ("rl_resume_store", "rl_elastic_checkpoint_store"):
        store = getattr(args, name, None)
        if not store:
            continue
        try:
            from .rl.engine.resume import store_for

            root = store_for(store, environ=env).root
        except Exception as e:  # noqa: BLE001 - an unbuildable store fails later, in its own check
            print(f"[yeto-island] negative-test marker not checked for {store}: {e}", flush=True)
            continue
        check_store_marker(root, negative=island_is_negative(env))
    data = env_override(env)
    delay = float((data or {}).get("join_delay_s") or 0)
    if delay > 0:
        print(f"[yeto-island] negative-test island {island_id}: waiting {delay:.0f} s before it "
              "starts, so the unchanged island sets the session contract", flush=True)
        sleep(delay)
    return record



def check_store_marker(store_root, *, negative: bool) -> None:
    """Refuse a normal run on a store a negative-test run wrote; a negative run
    marks the store.  ``store_root`` must be a readable directory path."""
    root = Path(store_root).expanduser()
    marker = root / NEGATIVE_MARKER
    if negative:
        try:
            root.mkdir(parents=True, exist_ok=True)
            marker.write_text("negative-test run; do not resume as a normal run\n", encoding="utf-8")
        except OSError as e:
            print(f"[yeto] could not write {marker}: {e}", file=sys.stderr)
        return
    if marker.exists():
        raise ValueError(f"checkpoint store {root} was written by a negative-test run "
                         f"({NEGATIVE_MARKER} marker): a normal run must not resume from it")


def local_store_path(store: str | None) -> Path | None:
    """A ``--rl-checkpoint-store`` this machine can read as a directory, else None."""
    if not store or "://" in str(store):
        return None
    return Path(str(store)).expanduser()


def check_resume_before_launch(args, *, runs_root: Path | None = None, out=None) -> None:
    """Launcher, before any cloud work: a normal run with ``--rl-checkpoint-store``
    refuses a store marked by a negative-test run (marker, then the local run
    manifests as a supplement).  Stores this machine cannot read are checked
    by the island at start."""
    out = sys.stderr if out is None else out
    store = getattr(args, "rl_checkpoint_store", None)
    if not store or getattr(args, "rl_negative_test_run", False):
        return
    path = local_store_path(store)
    if path is not None:
        check_store_marker(path, negative=False)
    if runs_root is None:
        from . import runs

        runs_root = runs.run_dir("x").parent
    for manifest in sorted(Path(runs_root).glob("*/run_manifest.json")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("negative_test") and data.get("checkpoint_store") == store:
            raise ValueError(f"--rl-checkpoint-store {store} was used by negative-test run "
                             f"{data.get('cluster_prefix')!r} ({manifest}): a normal run must not "
                             "resume from it")
    if path is None:
        print(f"[yeto] --rl-checkpoint-store {store}: negative-test marker is checked by the "
              "island at start (store not readable from this machine)", file=out, flush=True)


def check_merge_source(adapter_dir) -> None:
    """``yeto merge``: refuse an adapter of a negative-test run (marker file or a
    run manifest with negative_test in the adapter dir or two levels above)."""
    base = Path(adapter_dir).expanduser()
    for d in [base, *list(base.parents)[:2]]:
        if (d / NEGATIVE_MARKER).exists():
            raise ValueError(f"{adapter_dir} belongs to a negative-test run ({d / NEGATIVE_MARKER}); "
                             "its weights must not be exported (负例运行，禁止导出)")
        manifest = d / "run_manifest.json"
        if manifest.is_file():
            try:
                if json.loads(manifest.read_text(encoding="utf-8")).get("negative_test"):
                    raise ValueError(f"{adapter_dir} belongs to a negative-test run ({manifest}); "
                                     "its weights must not be exported (负例运行，禁止导出)")
            except (OSError, ValueError) as e:
                if "negative-test" in str(e):
                    raise
