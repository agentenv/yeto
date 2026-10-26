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

* ``clamp`` never reaches ``STACK_REWRITES``, and the step that keeps it out is the
  **write**, not the refusal (S27 G-FW, read off the delivery on npu1 n3).
  :func:`install` puts ``STACK_REWRITES=cumsum,squeeze`` into the worker environment
  **before it imports the first plugin**, because the delivery's own default is the
  opposite value.  ``ascend_mamba2_kernels.apply`` delegates to
  ``/data/work/kern-stack/mamba2_stack.py`` whenever that directory exists on the node,
  and calls ``mamba2_stack.apply(stages=...)`` with no ``rewrites=`` argument, so the
  items come from that module's line 87:
  ``REWRITES = os.environ.get("STACK_REWRITES", "cumsum,squeeze,clamp")``.  An unset
  variable therefore **asks for** the clamp rewrite.
  That rewrite deletes the decay clamp in ``ssd_recompute_patch._ddAcs``.  With the
  clamp gone the margin is zero: S27 G-FR measured about 1.1e5x over the overflow
  bound, independent of the token distribution (a full-vocabulary sweep of layer 0 put
  a single ``'</'`` token 5x over), so the overflow is certain, not unlikely.
  Two further layers stand behind the write: :func:`check_env` refuses any value that
  names ``clamp``, and the delivery's own ``apply_rewrites`` raises unless
  ``KRC_ALLOW_CLAMP_REWRITE=1`` (``kern-recompute/ssd_recompute_patch.py:587-593``,
  a name no ``verl_main.FORWARD_ENV_PREFIXES`` prefix forwards).
  **Do not drop ``STACK_REWRITES`` from :data:`RECIPE_ENV` on the grounds that it is
  "the default anyway": it is not.**
* :func:`setup_worker` -- Ray's ``worker_process_setup_hook`` -- **never imports a
  plugin itself**.  It only arms :func:`install_when_rank_known`, because one plugin
  binds a card at import time and the hook runs before this process has one
  (S27 G-FY; the reason and the measurement are under :func:`install_when_rank_known`).
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

import contextlib
import functools
import importlib
import os
from collections.abc import Mapping, Sequence
from typing import Any

FLAG = "--rl-ascend-plugin-stack"
# The island passes the plugin directory to its Ray workers through this variable
# (``verl_main._runtime_env`` forwards every ``YETO_`` name) and points Ray's
# ``worker_process_setup_hook`` at :func:`setup_worker`.
ENV = "YETO_ASCEND_PLUGIN_STACK"
SETUP_HOOK = "yeto.rl.ascend_plugin_stack.setup_worker"

# The variable that says which card of this node this process owns. A plugin of the
# delivery reads it at import time to pick its device, so :func:`install` must not run
# while it is unset (see :func:`install_when_rank_known`).
RANK_ENV = "LOCAL_RANK"
# The seam that turns "the process started" into "the process owns a card": verl's
# base Worker writes RANK / LOCAL_RANK into the environment and binds the card in
# ``Worker.__init__`` (``verl/single_controller/base/worker.py:180-215`` and
# ``:275-280``), and every verl worker class chains to it. It runs before
# ``init_model`` builds the model, which is the deadline the plugins have to meet.
VERL_WORKER = ("verl.single_controller.base.worker", "Worker")

# The one-variable switch of the S28 G-PIN contrast. The delivery's
# ``ascend_mamba2_kernels.apply`` calls ``torch.npu.set_device(_local_device())`` on its
# kern-stack path (``plugins/ascend_mamba2_kernels.py:352-356``) and again before its
# prewarm (``:290``). Those calls run inside ``Worker.__init__`` once the install is
# deferred (:func:`install_when_rank_known`), that is, inside the Ray actor's
# constructor. Set this variable to ``0`` and yeto replaces ``torch.npu.set_device``
# with a recorder **for the duration of the plugin import only**, then puts the real
# function back. Nothing of the delivery is edited: the switch lives here.
# Default ``1`` keeps the delivery's own behaviour.
SET_DEVICE_ENV = "YETO_ASCEND_STACK_SET_DEVICE"

# Two names that narrow the list of plugins this process imports, so a failing island
# can be bisected without editing the delivery or this table (S28 G-PIN: the island
# dies at 507033 with the stack on and builds its model with the stack off, and the
# cause is not the plugins' own set_device call). Both take a comma separated list of
# file names. ONLY wins when both are set, and an unknown name is reported, not
# silently ignored. Neither is set by the island: they are for a human bisect.
# Where in the worker's life the install happens. "worker_init" (the default) is
# :data:`VERL_WORKER`, the moment verl's base Worker has bound this process's card.
# "init_model" waits one step longer: the start of the verl call that builds the model
# (:data:`VERL_MODEL_WORKERS`), so the plugins' imports and Megatron's own device
# initialisation happen inside the same Ray task rather than in the actor's
# constructor. S28 G-PIN measured that a single plugin import in the constructor is
# enough to make Megatron's first LazySetDevice fail with 507033, with the plugins'
# own set_device call bypassed and the prewarm off, so where the import happens is
# the next thing to vary.
INSTALL_AT_ENV = "YETO_ASCEND_STACK_INSTALL_AT"
VERL_MODEL_WORKERS = (
    ("verl.workers.engine_workers", "ActorRolloutRefWorker", "init_model"),
    ("verl.workers.engine_workers", "TrainingWorker", "init_model"),
)
ONLY_ENV = "YETO_ASCEND_STACK_ONLY"
SKIP_ENV = "YETO_ASCEND_STACK_SKIP"

# The rewrite items the recipe enables in the Mamba2 SSD stack. ``clamp`` is not one
# of them and must never be added, and this entry must stay in RECIPE_ENV: the
# delivery's default (kern-stack/mamba2_stack.py:87) is "cumsum,squeeze,clamp", so
# leaving the variable unset enables the item we refuse (see the module docstring).
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


def _narrow(wanted: list[str], say) -> list[str]:
    """Apply :data:`ONLY_ENV` / :data:`SKIP_ENV` to the list of plugins to import.

    A name in either list that is not in :data:`PLUGINS` is reported on its own line:
    during a bisect a silently ignored typo would read as "this plugin is innocent"."""

    def names(variable: str) -> list[str]:
        return [part.strip() for part in os.environ.get(variable, "").split(",") if part.strip()]

    only, skip = names(ONLY_ENV), names(SKIP_ENV)
    for variable, chosen in ((ONLY_ENV, only), (SKIP_ENV, skip)):
        for name in chosen:
            if name not in wanted:
                say(f"[yeto-ascend-stack] {variable} names {name}, which is not in this list")
    if only:
        narrowed = [name for name in wanted if name in only]
        say(f"[yeto-ascend-stack] {ONLY_ENV}: importing {len(narrowed)} of {len(wanted)} plugin(s): "
            f"{narrowed}")
        return narrowed
    if skip:
        narrowed = [name for name in wanted if name not in skip]
        say(f"[yeto-ascend-stack] {SKIP_ENV}: leaving out {sorted(set(wanted) - set(narrowed))}")
        return narrowed
    return wanted


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
    # Assignment, never setdefault: a Ray worker of an already-running cluster
    # inherits the raylet's environment, and an inherited STACK_REWRITES that names
    # clamp has to lose to the recipe's value (module docstring, first bullet).
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
    wanted = _narrow(wanted, say)
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


def set_device_is_allowed(environ: Mapping[str, str] | None = None) -> bool:
    """False when the island asked yeto to keep the plugins off ``set_device``.

    See :data:`SET_DEVICE_ENV` for what the switch is for. Only the exact values
    ``0``, ``false`` and ``no`` turn it off, so an unset or unreadable value keeps the
    delivery's own behaviour."""

    where = os.environ if environ is None else environ
    return str(where.get(SET_DEVICE_ENV, "1")).strip().lower() not in ("0", "false", "no")


@contextlib.contextmanager
def _set_device_neutralised(say):
    """Replace ``torch.npu.set_device`` with a recorder while the plugins import.

    The recorder prints every call it swallows, so the island log says whether the
    switch changed anything at all. ``torch`` or ``torch_npu`` missing is not an
    error here: then no plugin can call ``set_device`` either, and the block runs
    unchanged."""

    try:
        import torch

        from yeto import accel

        accel.register_backends()  # the one place that imports torch_npu (yeto/accel.py:43)
        real = torch.npu.set_device
    except BaseException as error:  # noqa: BLE001 - a diagnostic switch must not stop the island
        say(f"[yeto-ascend-stack] {SET_DEVICE_ENV}=0 asked for, but torch.npu.set_device is not "
            f"reachable ({type(error).__name__}: {error}); importing the plugins unchanged")
        yield
        return
    calls: list[Any] = []

    def recorder(device=None, *args, **kwargs):
        calls.append(device)
        say(f"[yeto-ascend-stack]   swallowed torch.npu.set_device({device!r}) "
            f"({SET_DEVICE_ENV}=0)")
        return None

    torch.npu.set_device = recorder
    try:
        yield
    finally:
        torch.npu.set_device = real
        say(f"[yeto-ascend-stack] {SET_DEVICE_ENV}=0: restored torch.npu.set_device after "
            f"{len(calls)} swallowed call(s) {calls}")


def rank_is_known(environ: Mapping[str, str] | None = None) -> bool:
    """True when this process has been told which card of the node it owns.

    :data:`RANK_ENV` is how the delivery's plugins ask that question
    (``plugins/ascend_mamba2_kernels.py:62``, ``_local_device``), so it is the question
    :func:`install_when_rank_known` has to answer before it imports one."""

    where = os.environ if environ is None else environ
    return bool(str(where.get(RANK_ENV, "")).strip())


def _install_once(root: str, *, log=None) -> dict[str, Any] | None:
    """:func:`install` at most once per process, and never without a card.

    :data:`LAST_REPORT` is the record of this process's install, so it is also the
    "already done" flag: one Ray worker process builds several verl worker objects
    (``WorkerDict.__init__`` constructs one per role,
    ``verl/single_controller/ray/base.py:1028-1040``), and importing the plugins again
    would re-apply every monkeypatch and re-run the Mamba2 prewarm."""

    say = log if log is not None else (lambda line: print(line, flush=True))
    if LAST_REPORT is not None:
        return None
    if not rank_is_known():
        say(f"[yeto-ascend-stack] NOT installed: no {RANK_ENV} in this process, so a plugin that "
            f"binds a card at import time would bind card 0 (S27 G-FW: eight ranks on card 0)")
        return None
    say(f"[yeto-ascend-stack] installing with {RANK_ENV}={os.environ.get(RANK_ENV)} "
        f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<unset>')} "
        f"{SET_DEVICE_ENV}={'1' if set_device_is_allowed() else '0'}")
    if set_device_is_allowed():
        return install(root, log=log)
    with _set_device_neutralised(say):
        return install(root, log=log)


def install_when_rank_known(root: str, *, log=None) -> str:
    """Arrange for :func:`install` to run at the first moment this process owns a card.

    **Why the install cannot happen in Ray's setup hook** (S27 G-FW, measured on npu1
    n2 with 8 x 910B3): ``ascend_mamba2_kernels.apply`` calls
    ``torch.npu.set_device(_local_device())`` on its kern-stack path
    (``plugins/ascend_mamba2_kernels.py:352-356``), and ``_local_device`` returns **0**
    when :data:`RANK_ENV` is unset (``:62``).  Ray runs
    ``worker_process_setup_hook`` when the worker process starts, which is *before* Ray
    leases that process to an actor, so no one has written :data:`RANK_ENV` yet.  All
    eight trainer ranks therefore bound card 0 -- the island log showed
    ``(Device:0, RankID:0)`` through ``(Device:0, RankID:7)``, cards 1-7 never took a
    single allocation, and HCCL could not build the eight-rank domain, so the island
    did not start.  The delivery is read-only, so the fix is on this side: wait.

    The wait ends at :data:`VERL_WORKER`.  This wraps that class's ``__init__`` so the
    install runs as soon as it returns: by then verl has written ``RANK`` and
    :data:`RANK_ENV` and bound the card, and it has not built the model yet.

    Returns ``"installed"`` (the rank was already known, nothing to wait for),
    ``"deferred"`` (the wrapper is armed) or ``"unavailable"`` (the seam is not there,
    so nothing is installed).  **Not installing is the safe failure and installing
    blind is not**: without the stack the island runs without the recipe's kernels,
    while installing without a card stops the island altogether (above)."""

    say = log if log is not None else (lambda line: print(line, flush=True))
    if rank_is_known():
        _install_once(root, log=log)
        return "installed"
    module_name, class_name = VERL_WORKER
    try:
        worker_cls = getattr(importlib.import_module(module_name), class_name)
    except BaseException as error:  # noqa: BLE001 - a missing seam must not stop the island
        say(f"[yeto-ascend-stack] NOT installed: no {RANK_ENV} yet and {module_name}.{class_name} "
            f"is not available to wait for ({type(error).__name__}: {error})")
        return "unavailable"
    original = worker_cls.__init__
    if getattr(original, "_yeto_ascend_stack", None) is not None:
        return "deferred"

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _install_once(root)

    functools.update_wrapper(__init__, original)  # copies __dict__, so tag afterwards
    __init__._yeto_ascend_stack = root
    worker_cls.__init__ = __init__
    say(f"[yeto-ascend-stack] root={root} deferred: no {RANK_ENV} in this process yet, so the "
        f"install waits for {module_name}.{class_name}.__init__ (it writes {RANK_ENV} and binds "
        f"the card before the model is built)")
    return "deferred"


def _install_at_model_build(root: str, say) -> str:
    """Wrap every verl worker method that builds the model, and install there.

    Returns ``"deferred"`` when at least one seam of :data:`VERL_MODEL_WORKERS` was
    wrapped, ``"unavailable"`` when none was: as everywhere else here, not installing
    is the safe failure."""

    wrapped: list[str] = []
    for module_name, class_name, method_name in VERL_MODEL_WORKERS:
        try:
            worker_cls = getattr(importlib.import_module(module_name), class_name)
            original = getattr(worker_cls, method_name)
        except BaseException as error:  # noqa: BLE001 - a missing seam must not stop the island
            say(f"[yeto-ascend-stack] {module_name}.{class_name}.{method_name} is not available "
                f"to wait for ({type(error).__name__}: {error})")
            continue
        if getattr(original, "_yeto_ascend_stack", None) is not None:
            wrapped.append(f"{class_name}.{method_name}")
            continue

        def make(original=original):
            def method(self, *args, **kwargs):
                _install_once(root)
                return original(self, *args, **kwargs)
            functools.update_wrapper(method, original)
            method._yeto_ascend_stack = root
            return method

        setattr(worker_cls, method_name, make())
        wrapped.append(f"{class_name}.{method_name}")
    if not wrapped:
        say(f"[yeto-ascend-stack] NOT installed: {INSTALL_AT_ENV}=init_model, but no model-building "
            f"seam of this verl is available")
        return "unavailable"
    say(f"[yeto-ascend-stack] root={root} deferred to the model build: {sorted(wrapped)}")
    return "deferred"


def setup_worker() -> None:
    """Ray ``worker_process_setup_hook``: arm the install in this worker process when
    the island asked for the stack (:data:`ENV`).

    It does **not** import a plugin here.  Ray runs this hook before the process owns a
    card, and one plugin binds a card at import time, so the import waits for
    :func:`install_when_rank_known`.  Each plugin then reports its own line; one that
    cannot be imported is recorded and the others still run (:func:`install`)."""

    root = os.environ.get(ENV, "")
    if not root:
        return
    if os.environ.get(INSTALL_AT_ENV, "worker_init").strip() == "init_model":
        _install_at_model_build(root, lambda line: print(line, flush=True))
        return
    install_when_rank_known(root)
