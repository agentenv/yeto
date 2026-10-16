"""Modal deployment shape for the TB2 base-model baseline (``yeto.rl.eval.tb2_baseline``).

One GPU container: the pinned Miles-ports image (same mounts and env as the
eval island), the base model read-only from the models Volume
(``hf/<name>/<rev[:8]>/``), TB2 sandboxes created by the in-container Modal
client.  Evidence (traces, summary, SGLang / session-server logs, nvidia-smi
samples) is copied to the results Volume every ``sync_s`` seconds and once at
the end, so a container killed by its timeout still leaves its traces.

The driver builds the app locally (the spec needs local-only paths) and must
``spawn`` the call inside ``app.run(detach=True)`` and return: killing a client
that blocks on ``.remote()`` cancels the input.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .modal_eval_island import (CODEX_MOUNT, TB2_MOUNT, WORKDIR_MOUNT, _gpu_sampler, _image_paths,
                                default_image_ref)

MODELS_VOLUME = "yeto-fn-models"
MODELS_MOUNT = "/mnt/yeto-models"
RESULTS_VOLUME = "yeto-tb2-baseline-runs"
RESULTS_MOUNT = "/mnt/yeto-tb2-baseline"
FUNCTION_NAME = "tb2_baseline"


@dataclass
class BaselineFunctionSpec:
    """``gpu`` is a Modal GPU string pinned to one type (``H200``; ``H100!`` keeps Modal from upgrading H100 to H200); ``envs`` become one secret
    (codex contract env, Modal token for the TB2 sandboxes, TB2 / worker knobs)."""

    app_name: str
    workdir: str
    codex_dir: str
    tb2_dir: str
    gpu: str = "H200"
    gpus: int = 1
    cpu: float = 16.0
    memory_mib: int = 131072
    timeout_s: int = 4 * 3600
    image_ref: str = field(default_factory=default_image_ref)
    envs: Mapping[str, str] = field(default_factory=dict)

    def container_env(self) -> dict[str, str]:
        # No PYTHONPATH here (see EvalFunctionSpec.container_env).
        return {"HOME": "/root", "PYTHONUNBUFFERED": "1", "YETO_HARNESS_TB2_TASKS_DIR": TB2_MOUNT,
                **{str(k): str(v) for k, v in self.envs.items()}}


def _sync(src: Path, dst: Path, volume: Any) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True)
    volume.commit()


def baseline_body(config: Mapping[str, Any]) -> dict[str, Any]:
    """Inside the container: assert the GPU type, run the baseline, keep the evidence on the results Volume."""
    import subprocess

    import modal

    _image_paths()
    from yeto.rl.eval import tb2_baseline

    names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout.split("\n")
    names = [n.strip() for n in names if n.strip()]
    want = str(config["assert_gpu_name"])
    if not names or not all(want in n for n in names):
        raise RuntimeError(f"baseline asked for {want}, got {names}")
    run_name = str(config["run_name"])
    local = Path("/tmp/tb2-baseline") / run_name
    final = Path(RESULTS_MOUNT) / run_name
    local.mkdir(parents=True, exist_ok=True)
    (local / "body.json").write_text(json.dumps({"gpu_names": names, "container": os.environ.get("MODAL_TASK_ID"),
                                                 "start_ts": time.time()}, sort_keys=True))
    results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
    stop = threading.Event()
    sampler = threading.Thread(target=_gpu_sampler, args=(local / "gpu.jsonl", stop), daemon=True)
    sampler.start()

    def syncer() -> None:
        while not stop.wait(float(config.get("sync_s", 120))):
            try:
                _sync(local, final, results)
            except Exception as exc:  # noqa: BLE001 - the next tick retries
                print(f"[tb2-baseline] sync failed: {exc}", flush=True)

    threading.Thread(target=syncer, daemon=True).start()
    infer = dict(config.get("infer") or {})
    infer.setdefault("log_dir", str(local / "logs"))
    try:
        summary = tb2_baseline.run_from_config({**config, "infer": infer, "out_dir": str(local)})
    finally:
        stop.set()
        sampler.join(timeout=10)
        _sync(local, final, results)
    return {"run_name": run_name, "results": str(final), **{k: summary.get(k) for k in
            ("attempts", "judged", "passed", "pass_rate", "infra_errors", "tasks_mixed", "attempts_wall_s")}}


def build_baseline_app(spec: BaselineFunctionSpec) -> tuple[Any, Any]:
    """(app, baseline GPU function).  Models Volume read-only."""
    import modal

    image = (modal.Image.from_registry(spec.image_ref)
             .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
             .add_local_dir(spec.workdir, WORKDIR_MOUNT, copy=False,
                            ignore=[".git", ".venv", ".claude", "**/__pycache__", "**/*.pyc", "s1-runs",
                                    "syncer/target"])
             .add_local_dir(spec.codex_dir, CODEX_MOUNT, copy=False)
             .add_local_dir(spec.tb2_dir, TB2_MOUNT, copy=False, ignore=[".git"]))
    app = modal.App(spec.app_name)
    models = modal.Volume.from_name(MODELS_VOLUME)
    results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
    gpu = spec.gpu if spec.gpus == 1 else f"{spec.gpu}:{spec.gpus}"
    run = app.function(image=image, gpu=gpu, cpu=spec.cpu, memory=spec.memory_mib, timeout=spec.timeout_s,
                       secrets=[modal.Secret.from_dict(spec.container_env())], retries=0,
                       volumes={MODELS_MOUNT: models.read_only(), RESULTS_MOUNT: results},
                       name=FUNCTION_NAME)(baseline_body)
    return app, run
