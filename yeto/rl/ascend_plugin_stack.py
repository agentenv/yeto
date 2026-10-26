"""The Ascend recipe's Megatron plugin stack, installed outside ms-swift (S27 G-FL).

The user's Nemotron SFT recipe ``18.44pct-lay_r29_clip_buf64_m6_20261024`` reaches
its throughput with a set of environment switches and a list of plugin files that
ms-swift loads through ``--external_plugins``.  Each plugin is a plain module whose
*import* installs it: it reads its own ``ASCEND_*`` switch and monkeypatches
Megatron-Core (or the model code) when the switch is on, and swallows its own
errors so a failed optimisation never stops the run.

Nothing here is ms-swift specific, so a verl Megatron island can install the same
stack: set the switches, then import the files in the recipe's order.  This module
is the table of what to set and what to import, plus :func:`install`, which does it
and reports one line per plugin.  It is **opt-in and off by default**
(``--rl-ascend-plugin-stack <dir>``); without the flag nothing is set or imported.

What this module deliberately does **not** do, and why (every reason read from the
recipe or from the plugin source, S27 G-FL):

* ``clamp`` never reaches ``STACK_REWRITES``.  The recipe exports
  ``STACK_REWRITES=cumsum,squeeze``; the ``clamp`` rewrite deletes the decay clamp
  in ``ssd_recompute_patch._ddAcs``, whose own comment says that produces ``inf``
  for large ``dt`` (S27 G-FK).  :func:`install` refuses any value that names it.
* three plugins are left out (:data:`EXCLUDED`): two import ms-swift at module
  level and cannot be imported without it, one does so under its own switch.  Each
  carries the file and line that says so.
* two recipe variables are left unset (:data:`UNSET`): the per-stage recompute
  policy is written for the recipe's PP3 layout, and ``HCCL_BUFFSIZE=64`` is
  coupled to it (the recipe's ``CAVEATS.md``: at 200 the M=6 policy dies before the
  first step).  A different island shape has to choose both for itself.

The plugin files themselves stay the user's: this module only reads their path and
imports them.  It never edits the delivery.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

FLAG = "--rl-ascend-plugin-stack"
# The island passes the plugin directory to its Ray workers through this variable
# (``verl_main._runtime_env`` forwards every ``YETO_`` name) and points Ray's
# ``worker_process_setup_hook`` at :func:`setup_worker`.
ENV = "YETO_ASCEND_PLUGIN_STACK"
SETUP_HOOK = "yeto.rl.ascend_plugin_stack.setup_worker"

# The rewrite items the recipe enables in the Mamba2 SSD stack. ``clamp`` is not one
# of them and must never be added (see the module docstring).
STACK_REWRITES = "cumsum,squeeze"
REFUSED_REWRITES = ("clamp",)

# What the switch sets: the recipe's kernel, fusion and correctness switches, with
# the recipe's own values (``REPRODUCE.sh`` and ``run_mfu_exp.sh`` of the recipe).
RECIPE_ENV: dict[str, str] = {
    # correctness fixes the recipe keeps on
    "ASCEND_EXPERT_FC1_FIX": "1",
    "ASCEND_UNEVEN_COLL_FIX": "1",
    "ASCEND_GRAD_SCALE_FIX": "1",
    # memory
    "ASCEND_EMPTY_CACHE_CLIP": "0",
    "ASCEND_FREE_MOE_REFS": "1",
    # Mamba2 SSD stack (no clamp)
    "STACK_REWRITES": STACK_REWRITES,
    "MAMBA_FUSE": "1",
    # MoE kernels
    "ASCEND_MOE_FAST_SORT": "1",
    "ASCEND_MOE_PERMUTE_BF16": "1",
    "ASCEND_MOE_FUSED_RELU2": "1",
    "ASCEND_MOE_ROUTER_FASTPATH": "1",
    "ASCEND_MOE_GMM_SPLITS": "1",
    "ASCEND_VEC_MOE_PROBS": "1",
    # attention, convolution and norms
    "ASCEND_CONV_BWD_FUSE": "1",
    "ASCEND_SYNC_ATTN_CUSEQ": "1",
    "ASCEND_SYNC_P2P_SHAPE": "1",
    "ASCEND_VEC_GATED_NORM": "1",
    # head and loss
    "ASCEND_LMHEAD_GATHER": "1",
    "ASCEND_FUSED_VOCAB_CE": "1",
    # vision (a text-only run never reaches it)
    "ASCEND_SKIP_EMPTY_VISION": "1",
}

# Recipe variables this switch does not set, with the reason.
UNSET: dict[str, str] = {
    "ASCEND_RECOMP_POLICY": "the recipe's value names pp0/pp1/pp2, so it belongs to its PP3 layout only",
    "HCCL_BUFFSIZE": "coupled to that recompute policy (the recipe's CAVEATS.md: M=6 dies at 200)",
    "ASCEND_BENCH_NO_SAVE": "suppresses ms-swift checkpoint saving; its plugin is not loaded here",
    "ASCEND_STAGE_TIMER": "a measurement probe whose plugin imports ms-swift when it is set",
    "ASCEND_OUTSIDE_PROBE": "a measurement probe, not a throughput switch",
    "ASCEND_COMM_CENSUS": "a measurement probe, not a throughput switch",
    "ASCEND_RECOMP_MEMLOG": "a measurement probe, not a throughput switch",
}

# The plugin files in the recipe's load order (ms-swift's base three, then
# ``EXTRA_PLUGINS``, then ``EXTRA_PLUGINS_ADD``), minus :data:`EXCLUDED`.
PLUGINS: tuple[str, ...] = (
    "ascend_mcore_fixes.py",
    "ascend_mamba2_kernels.py",
    "ascend_free_moe_refs.py",
    "ascend_grad_scale_fix.py",
    "ascend_empty_cache_clip.py",
    "ascend_uneven_collectives_fix.py",
    "ascend_expert_fc1_etp_fix.py",
    "ascend_chunk_sort_fallback.py",
    "ascend_conv_ragged_fix.py",
    "ascend_skip_empty_vision.py",
    "ascend_moe_fast_chunk_sort.py",
    "ascend_moe_permute_bf16.py",
    "ascend_moe_fused_relu2.py",
    "ascend_vec_moe_probs.py",
    "ascend_moe_router_fastpath.py",
    "ascend_moe_gmm_splits.py",
    "ascend_mamba_bwd_fuse.py",
    "ascend_conv_bwd_fuse.py",
    "ascend_sync_attn_cuseq.py",
    "ascend_sync_p2p_shape.py",
    "ascend_vec_gated_norm.py",
    "ascend_lmhead_gather.py",
    "ascend_fused_vocab_ce.py",
    "ascend_layer_recompute_policy.py",
    "ascend_clip_nosync.py",
)

# Plugins of the recipe's stack this switch leaves out, with the evidence.
EXCLUDED: dict[str, str] = {
    "nemo_template_fix.py": "imports ms-swift at module level (line 9, from swift.template.register)",
    "ascend_bench_pp3_metrics.py": "imports ms-swift at module level (line 14, "
                                   "from swift.megatron.trainers.base)",
    "ascend_stage_timer.py": "imports ms-swift at module level when ASCEND_STAGE_TIMER is set (line 20); "
                             "a measurement probe",
    "optimizer_phase_timing_inject.py": "lives outside the recipe's plugin directory "
                                        "(/data/work/mfu-agents/opt_pipeline) and only times the optimizer",
}


class AscendPluginStackError(ValueError):
    """The Ascend plugin stack cannot be installed as asked."""


def check_env(env: Mapping[str, str]) -> None:
    """Refuse a switch set that would enable a refused rewrite item."""

    items = [item.strip() for item in str(env.get("STACK_REWRITES", "")).split(",") if item.strip()]
    bad = sorted(set(items) & set(REFUSED_REWRITES))
    if bad:
        raise AscendPluginStackError(
            f"STACK_REWRITES must not name {bad}: the clamp rewrite deletes the decay clamp in "
            f"ssd_recompute_patch._ddAcs, which its own comment says makes inf for large dt; the recipe "
            f"exports {STACK_REWRITES!r}")


def recipe_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The switches to set: :data:`RECIPE_ENV` with ``extra`` on top (checked)."""

    env = dict(RECIPE_ENV)
    env.update({str(key): str(value) for key, value in (extra or {}).items()})
    check_env(env)
    return env


def plan(root: str, *, extra_env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """What :func:`install` would set and import, without touching anything.

    ``plugins`` lists ``{"plugin", "path", "present"}`` in load order; ``excluded``
    and ``unset`` carry the reasons."""

    base = str(root).rstrip("/")
    return {
        "root": base,
        "env": recipe_env(extra_env),
        "unset": dict(UNSET),
        "excluded": dict(EXCLUDED),
        "plugins": [{"plugin": name, "path": f"{base}/{name}",
                     "present": os.path.isfile(os.path.join(base, name))} for name in PLUGINS],
    }


def _import_file(name: str, path: str) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"yeto_ascend_plugin_{name[:-3]}", path)
    if spec is None or spec.loader is None:
        raise AscendPluginStackError(f"{path} is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


# The last :func:`install` of this process, read back by the island for its log.
LAST_REPORT: dict[str, Any] | None = None


def install(root: str, *, environ: Any = None, extra_env: Mapping[str, str] | None = None,
            plugins: Sequence[str] | None = None, log=None) -> dict[str, Any]:
    """Set the recipe's switches in ``environ`` and import the plugin files.

    Every plugin installs itself on import and reports its own result in its own
    log line; an import that raises is recorded and the next plugin still runs, so a
    broken optimisation cannot stop the island.  Returns the report (also kept in
    :data:`LAST_REPORT`): ``{"root", "env", "unset", "excluded", "plugins":
    [{"plugin", "installed", "reason"}], "installed", "failed"}``."""

    global LAST_REPORT

    where = os.environ if environ is None else environ
    say = log if log is not None else (lambda line: print(line, flush=True))
    report = plan(root, extra_env=extra_env)
    for key, value in report["env"].items():
        where[key] = value
    say(f"[yeto-ascend-stack] root={report['root']} switches={len(report['env'])} "
        f"plugins={len(report['plugins'])} excluded={len(report['excluded'])}")
    for key, value in sorted(report["env"].items()):
        say(f"[yeto-ascend-stack]   set {key}={value}")
    for name, reason in sorted(report["unset"].items()):
        say(f"[yeto-ascend-stack]   unset {name}: {reason}")
    for name, reason in sorted(report["excluded"].items()):
        say(f"[yeto-ascend-stack]   excluded {name}: {reason}")
    wanted = list(PLUGINS if plugins is None else plugins)
    rows: list[dict[str, Any]] = []
    for name in wanted:
        path = os.path.join(report["root"], name)
        if not os.path.isfile(path):
            rows.append({"plugin": name, "installed": False, "reason": "file not found"})
        else:
            try:
                _import_file(name, path)
                rows.append({"plugin": name, "installed": True, "reason": ""})
            except BaseException as error:  # noqa: BLE001 - a plugin must not stop the island
                rows.append({"plugin": name, "installed": False, "reason": f"{type(error).__name__}: {error}"})
        row = rows[-1]
        say(f"[yeto-ascend-stack]   {'installed' if row['installed'] else 'NOT installed'} "
            f"{row['plugin']}{': ' + row['reason'] if row['reason'] else ''}")
    report["plugins"] = rows
    report["installed"] = sorted(row["plugin"] for row in rows if row["installed"])
    report["failed"] = {row["plugin"]: row["reason"] for row in rows if not row["installed"]}
    say(f"[yeto-ascend-stack] installed {len(report['installed'])}/{len(rows)} plugin(s); "
        f"not installed: {sorted(report['failed']) or 'none'}")
    LAST_REPORT = report
    return report


def setup_worker() -> None:
    """Ray ``worker_process_setup_hook``: install the stack in this worker process
    before the framework builds the model, when the island asked for it (:data:`ENV`).

    Each plugin reports its own line; one that cannot be imported is recorded and the
    others still run (:func:`install`)."""

    root = os.environ.get(ENV, "")
    if root:
        install(root)
