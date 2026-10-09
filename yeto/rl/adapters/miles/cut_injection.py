"""Test-only fault injection for the E2 GPU acceptance (plan-v3 G-4.5). Off by default.

Every switch is an environment variable, exported into the island run command
by the launcher's ``--rl-test-inject-*`` flags (patch for INFRA-E1) or set by
the E2 harness. Unset (the default) means no code path changes. Values are
validated on read; an invalid value raises instead of being ignored.

Rank-side (read in the Megatron rank process, ``cut_plugin``):

* ``YETO_RL_TEST_INJECT_CUT_SAVE_KILL_RANK=<global rank>`` -- that rank
  ``os._exit(137)`` after writing half of its shard's temp file (G-4.5 "save_cut
  写分片途中 kill"): no shard, no manifest.
* ``YETO_RL_TEST_INJECT_CUT_RESTORE_KILL_RANK=<global rank>`` -- that rank
  ``os._exit(137)`` after writing the adapter tensors and before the optimizer
  (G-4.5 "restore 期间 kill rank").
* ``YETO_RL_TEST_INJECT_CUT_RESTORE_SLEEP=<global rank>:<seconds>`` -- that rank
  sleeps before entering the restore (G-4.5 "restore 前 sleep 超过 distributed
  timeout").

Driver-side (``trainer_rebuild``):

* ``YETO_RL_TEST_INJECT_REBUILD_FAIL=<n>`` -- the first ``n`` fork
  ``rebuild_training_models`` calls fail inside the fork at its
  ``create_training_models`` stage (real ``TrainerRebuildError`` path).
* ``YETO_RL_TEST_INJECT_REBUILD_CURSOR_SHIFT=<groups>`` -- before the rebuild,
  write a Miles dataset state file where ``rollout_executor.load`` (called by
  ``create_training_models``) reads it, with ``sample_offset`` advanced by
  ``groups``: the real rewind path of review H1.

The sixth G-4.5 row (kill the controller before/after the commit CAS) uses
INFRA-E1's ``--rl-test-kill-learner-at REBUILDING_TRAINER|COMMITTED`` with
``--rl-elastic-restart-attempts``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

SAVE_KILL_RANK_ENV = "YETO_RL_TEST_INJECT_CUT_SAVE_KILL_RANK"
RESTORE_KILL_RANK_ENV = "YETO_RL_TEST_INJECT_CUT_RESTORE_KILL_RANK"
RESTORE_SLEEP_ENV = "YETO_RL_TEST_INJECT_CUT_RESTORE_SLEEP"
REBUILD_FAIL_ENV = "YETO_RL_TEST_INJECT_REBUILD_FAIL"
CURSOR_SHIFT_ENV = "YETO_RL_TEST_INJECT_REBUILD_CURSOR_SHIFT"
ALL_ENVS = (SAVE_KILL_RANK_ENV, RESTORE_KILL_RANK_ENV, RESTORE_SLEEP_ENV, REBUILD_FAIL_ENV,
            CURSOR_SHIFT_ENV)
KILL_EXIT_CODE = 137


class InjectionConfigError(ValueError):
    pass


def _int(env: Mapping[str, str], name: str, *, low: int = 0) -> int | None:
    raw = env.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise InjectionConfigError(f"{name}={raw!r} is not an integer") from exc
    if value < low:
        raise InjectionConfigError(f"{name}={value} must be >= {low}")
    return value


def save_kill_rank(env: Mapping[str, str] | None = None) -> int | None:
    return _int(os.environ if env is None else env, SAVE_KILL_RANK_ENV)


def restore_kill_rank(env: Mapping[str, str] | None = None) -> int | None:
    return _int(os.environ if env is None else env, RESTORE_KILL_RANK_ENV)


def restore_sleep(env: Mapping[str, str] | None = None) -> tuple[int, float] | None:
    raw = (os.environ if env is None else env).get(RESTORE_SLEEP_ENV)
    if not raw:
        return None
    rank, sep, seconds = raw.partition(":")
    try:
        parsed = (int(rank), float(seconds))
    except ValueError as exc:
        raise InjectionConfigError(f"{RESTORE_SLEEP_ENV}={raw!r} must be <rank>:<seconds>") from exc
    if not sep or parsed[0] < 0 or not parsed[1] > 0:
        raise InjectionConfigError(f"{RESTORE_SLEEP_ENV}={raw!r} must be <rank>:<seconds>")
    return parsed


def rebuild_fail_count(env: Mapping[str, str] | None = None) -> int:
    return _int(os.environ if env is None else env, REBUILD_FAIL_ENV, low=1) or 0


def cursor_shift(env: Mapping[str, str] | None = None) -> int | None:
    return _int(os.environ if env is None else env, CURSOR_SHIFT_ENV, low=1)


def kill_now(reason: str) -> None:  # pragma: no cover - exits the process
    print(f"[yeto-test-inject] {reason}: os._exit({KILL_EXIT_CODE})", flush=True)
    os._exit(KILL_EXIT_CODE)


def is_rank(coord: Mapping[str, int], rank: int | None) -> bool:
    return rank is not None and int(coord.get("global_rank", -1)) == rank


def write_shifted_dataset_state(args: Any, cursor: Mapping[str, int], groups: int) -> str:
    """Put a dataset state with ``sample_offset + groups`` where Miles'
    ``RolloutDataSource.load(start_rollout_id - 1)`` reads it (its ``args.load``,
    i.e. ``--ref-load`` in bridge mode). Returns the path written.
    """
    import torch

    load = getattr(args, "load", None)
    if not load:
        raise InjectionConfigError(f"{CURSOR_SHIFT_ENV}: args.load is unset, nothing would read the file")
    start = getattr(args, "start_rollout_id", None)
    rollout_id = (int(start) if start is not None else 0) - 1
    path = os.path.join(load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
    if os.path.exists(path):
        raise InjectionConfigError(f"{path} already exists; refusing to overwrite it")
    state = {
        "sample_offset": int(cursor["sample_offset"]) + groups,
        "epoch_id": int(cursor["epoch_id"]),
        "sample_group_index": int(cursor["sample_group_index"]) + groups,
        "sample_index": int(cursor["sample_index"]),
        "metadata": {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    return path
