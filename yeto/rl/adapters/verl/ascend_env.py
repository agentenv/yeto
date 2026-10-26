"""vLLM / vllm_ascend environment of a verl island on Ascend (S28 G-M2; W2 and W3 of
SESSION27 ASCEND-ROLLOUT.md).

npu1's working vLLM engine needs three settings that yeto never set (W3):

* ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` -- with fork every TP worker dies in init;
* ``TASK_QUEUE_ENABLE`` in {0, 1} -- graph capture refuses 2;
* ``VLLM_PLUGINS`` naming ``ascend`` (and not ``ascend_service_profiling``).

A NemotronH (Mamba2) model also needs yeto's vLLM plugin
(:mod:`yeto.rl.ascend_vllm_mamba2`), which vLLM loads from a ``vllm.general_plugins``
entry point in every engine process. :func:`write_mamba2_entry_point` writes that
entry point as a dist-info directory; the island puts the directory on ``PYTHONPATH``.

W4: vLLM's sleep mode allocates from a memory pool, and torch_npu refuses a memory pool
when ``PYTORCH_NPU_ALLOC_CONF`` has ``expandable_segments:True`` ("Expandable segments are
not compatible with memory pool", n4 run i3). An island that runs a vLLM rollout therefore
drops that option; a buffer trainer (no rollout) keeps it.

W2: a process started with cwd ``/`` imports an empty namespace package for ``vllm``
(npu1's image has a bare ``/vllm`` directory), so :func:`safe_cwd` moves off ``/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ENV_DEFAULTS = {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TASK_QUEUE_ENABLE": "1"}
ASCEND_PLUGIN = "ascend"
MAMBA2_PLUGIN = "yeto_ascend_mamba2"
MAMBA2_TARGET = "yeto.rl.ascend_vllm_mamba2:register"
REFUSED_PLUGINS = ("ascend_service_profiling",)
DIST_INFO = "yeto_ascend_mamba2-0.1.dist-info"
# names the island forwards into every Ray worker (verl_main.FORWARD_ENV_NAMES)
FORWARD_NAMES = ("VLLM_WORKER_MULTIPROC_METHOD", "VLLM_PLUGINS", "TASK_QUEUE_ENABLE", "PYTORCH_NPU_ALLOC_CONF")


def is_mamba2_model(model_path: str) -> bool:
    """True for a NemotronH checkpoint (a ``config.json`` with a hybrid pattern naming M)."""
    try:
        cfg = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, ValueError):
        return False
    cfg = cfg.get("llm_config", cfg)
    return cfg.get("model_type") == "nemotron_h" and "M" in str(cfg.get("hybrid_override_pattern", ""))


def write_mamba2_entry_point(directory: Path) -> Path:
    """Write the dist-info that registers the Mamba2 plugin; returns the directory to add to sys.path."""
    info = Path(directory) / DIST_INFO
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: yeto-ascend-mamba2\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text(f"[vllm.general_plugins]\n{MAMBA2_PLUGIN} = {MAMBA2_TARGET}\n")
    return Path(directory)


ALLOC_ENV = "PYTORCH_NPU_ALLOC_CONF"
EXPANDABLE = "expandable_segments:True"


def island_env(env: dict, *, mamba2: bool, plugin_dir: Path | None = None, rollout: bool = True) -> list[str]:
    """Set the vLLM engine settings in ``env`` (in place). Returns one note per change.

    A value the caller set is kept, except a ``TASK_QUEUE_ENABLE`` outside {0, 1} and a
    ``VLLM_PLUGINS`` without ``ascend``: those cannot work and are corrected.
    """
    notes = []
    for name, value in ENV_DEFAULTS.items():
        if name not in env:
            env[name] = value
            notes.append(f"{name}={value} (default)")
    if env["TASK_QUEUE_ENABLE"] not in ("0", "1"):
        notes.append(f"TASK_QUEUE_ENABLE={env['TASK_QUEUE_ENABLE']} -> 1 (graph capture needs 0 or 1)")
        env["TASK_QUEUE_ENABLE"] = "1"
    plugins = [p for p in env.get("VLLM_PLUGINS", "").split(",") if p]
    wanted = [ASCEND_PLUGIN] + ([MAMBA2_PLUGIN] if mamba2 else [])
    new = [p for p in plugins if p not in REFUSED_PLUGINS]
    new += [p for p in wanted if p not in new]
    if new != plugins:
        env["VLLM_PLUGINS"] = ",".join(new)
        notes.append(f"VLLM_PLUGINS={env['VLLM_PLUGINS']}")
    if rollout and EXPANDABLE in env.get(ALLOC_ENV, ""):
        # an explicit False, not a deleted variable: Ray workers inherit the raylet's
        # environment, so only a forwarded value reaches them (FORWARD_NAMES)
        kept = [o for o in env[ALLOC_ENV].split(",") if o and o != EXPANDABLE]
        env[ALLOC_ENV] = ",".join(kept or ["expandable_segments:False"])
        notes.append(f"{ALLOC_ENV}={env[ALLOC_ENV]} (dropped {EXPANDABLE}: vLLM sleep mode needs a memory pool)")
    if mamba2:
        if plugin_dir is None:
            raise ValueError("a Mamba2 model needs plugin_dir for the vLLM entry point")
        site = str(write_mamba2_entry_point(plugin_dir))
        parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        if site not in parts:
            env["PYTHONPATH"] = os.pathsep.join([*parts, site])
            notes.append(f"PYTHONPATH += {site} (vLLM Mamba2 plugin entry point)")
    return notes


def rollout_overrides(model_path: str, device_family: str) -> list[str]:
    """Hydra overrides the vLLM rollout needs on this family. A NemotronH model on Ascend runs
    yeto's torch Mamba2 ops, which do not support prefix caching (``mamba_cache_mode='all'``,
    asserted in :mod:`yeto.rl.ascend_vllm_mamba2`); verl turns it on by default."""
    if device_family == "ascend" and is_mamba2_model(model_path):
        return ["actor_rollout_ref.rollout.enable_prefix_caching=False"]
    return []


def safe_cwd(fallback: Path) -> str | None:
    """Leave ``/`` for ``fallback`` (W2). Returns the new cwd, or None when no change was needed."""
    if os.getcwd() != "/":
        return None
    Path(fallback).mkdir(parents=True, exist_ok=True)
    os.chdir(fallback)
    return str(fallback)
