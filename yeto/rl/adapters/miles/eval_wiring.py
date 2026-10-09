"""Miles adapter wiring for the eval island (rl-eval-difficulty-buckets 2.2/5.2).

Both pieces are opt-in through environment variables, so launch flags (owned by
``island_entry``) and the default tape stay unchanged:

* ``YETO_RL_EVAL_HOLDOUT=path[@sha256],...`` (+ optional ``YETO_RL_EVAL_DATA``):
  the start-time hold-out check against ``--prompt-data``; runs before Ray is
  connected; a non-empty overlap refuses the launch.
* ``YETO_RL_EVAL_STORE=<dir>`` (a Modal Volume mount, or a staging dir that
  ``yeto.cloud.modal_eval_island`` uploads): export eval versions for the eval
  island. ``YETO_RL_EVAL_STORE_INTERVAL`` (default 10) sets the period.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

# Sampling settings pinned per eval version (D3: any change = another ruler).
SAMPLING_ATTRS = ("rollout_temperature", "rollout_top_p", "rollout_top_k", "rollout_max_response_len",
                  "rollout_max_context_len")
SAMPLING_ENVS = ("YETO_CODEX_REASONING_EFFORT", "YETO_CODEX_MAX_TURNS")


def sampling_settings(miles_args: Any, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    out: dict[str, Any] = {a: getattr(miles_args, a, None) for a in SAMPLING_ATTRS}
    out.update({k.lower(): env.get(k) for k in SAMPLING_ENVS})
    return out


def _train_paths(miles_args: Any) -> list[str]:
    value = getattr(miles_args, "prompt_data", None)
    if not value:
        return []
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]


def eval_guard_preflight(miles_args: Any, env: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    from yeto.rl.eval.guard import check_from_env

    return check_from_env(_train_paths(miles_args), env)


def attach(driver: Any, miles_args: Any, guard_report: Mapping[str, Any] | None,
           env: Mapping[str, str] | None = None) -> None:
    env = os.environ if env is None else env
    driver.eval_guard_report = guard_report
    if not env.get("YETO_RL_EVAL_STORE"):
        return
    from yeto.cloud.modal_eval_island import store_from_env
    from yeto.rl.eval.export import DEFAULT_INTERVAL, INTERVAL_ENV, StoreEvalExporter

    driver.eval_export = StoreEvalExporter(
        store_from_env(env), sampling=sampling_settings(miles_args, env),
        interval=int(env.get(INTERVAL_ENV) or DEFAULT_INTERVAL),
        extra={"base_model": getattr(miles_args, "hf_checkpoint", None)})
