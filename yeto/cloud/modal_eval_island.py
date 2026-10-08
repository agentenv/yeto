"""Cloud layer: the eval island on Modal (rl-eval-difficulty-buckets 5.7/5.8, D11.6/D11.7).

First version: Modal only. The authoritative eval store is a Modal Volume
(policy files + manifests, the queue, the unit logs). This module holds

* :func:`store_from_env` -- the training side's ``EvalStore``: either the
  Volume mounted in this container (``commit``/``reload`` hooks) or a local
  staging directory whose new files are uploaded through the Modal API on each
  commit (training island not on Modal);
* :class:`EvalIslandLauncher` -- the scheduler side: start one eval-island GPU
  function, restart it on Modal after a preemption (the unit log resumes where
  it stopped), one ``rl_eval_island`` event per start. One island at a time
  (D11.6: a second island only with user approval; not implemented);
* :func:`island_main` -- what runs inside the eval-island container: mount the
  Volume, build the plan, ``EvalIsland(...).run()``. The loader / attempt
  bindings are dotted factories (``YETO_EVAL_LOADER`` / ``YETO_EVAL_ATTEMPT``).

Imports ``modal`` lazily; CPU tests inject a fake client.
"""

from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from yeto.rl.eval.store import FINISHED_MARKER, EvalStore

VOLUME_ENV = "YETO_RL_EVAL_VOLUME"          # Modal Volume name of the eval store
MOUNTED_ENV = "YETO_RL_EVAL_STORE_MOUNTED"  # "1": YETO_RL_EVAL_STORE is that Volume's mount
STORE_ENV = "YETO_RL_EVAL_STORE"
LOADER_ENV = "YETO_EVAL_LOADER"             # "module:factory" -> PolicyLoader
ATTEMPT_ENV = "YETO_EVAL_ATTEMPT"           # "module:factory" -> attempt callable
ISLAND_EVENT = "rl_eval_island"
DEFAULT_GPU = "H200"
DEFAULT_GPUS = 8


def _volume(name: str) -> Any:
    import modal

    return modal.Volume.from_name(name, create_if_missing=True)


class _Uploader:
    """Commit hook for a staging directory: upload files changed since the last commit."""

    def __init__(self, root: Path, volume: Any) -> None:
        self.root = root
        self.volume = volume
        self.sent: dict[str, float] = {}

    def __call__(self) -> None:
        changed = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                rel = path.relative_to(self.root).as_posix()
                mtime = path.stat().st_mtime
                if self.sent.get(rel) != mtime:
                    changed.append((path, rel, mtime))
        if not changed:
            return
        # manifests last: a reader never sees a manifest before its files
        changed.sort(key=lambda c: (c[1].endswith("manifest.json") or c[1].startswith("queue/")
                                    or c[1] == FINISHED_MARKER, c[1]))
        with self.volume.batch_upload(force=True) as batch:
            for path, rel, _ in changed:
                batch.put_file(str(path), "/" + rel)
        for _, rel, mtime in changed:
            self.sent[rel] = mtime


def store_from_env(env: Mapping[str, str] | None = None, *, volume_factory: Callable[[str], Any] = _volume) -> EvalStore:
    env = os.environ if env is None else env
    root = env.get(STORE_ENV)
    if not root:
        raise RuntimeError(f"{STORE_ENV} is not set")
    name = env.get(VOLUME_ENV)
    if not name:
        return EvalStore(root)  # plain directory (tests, shared disk)
    vol = volume_factory(name)
    if env.get(MOUNTED_ENV) == "1":
        return EvalStore(root, commit=vol.commit, reload=vol.reload)
    Path(root).mkdir(parents=True, exist_ok=True)
    return EvalStore(root, commit=_Uploader(Path(root), vol))


# --- scheduler side ---------------------------------------------------------


class IslandPreempted(RuntimeError):
    """The eval-island container ended without finishing (Modal preemption / crash)."""


@dataclass
class EvalIslandLauncher:
    """Start the eval island on Modal; restart after preemption up to ``max_starts``.

    ``client.start(config) -> handle``; ``handle.wait() -> dict`` returns the
    island's summary or raises (any exception = the container is gone; the
    unit log keeps the finished units). ``price_per_hour`` is recorded in every
    ``rl_eval_island`` event for the ledger."""

    client: Any
    emit: Callable[..., None]
    config: Mapping[str, Any]
    gpu: str = DEFAULT_GPU
    gpus: int = DEFAULT_GPUS
    price_per_hour: float | None = None
    max_starts: int = 5
    clock: Callable[[], float] = time.monotonic
    starts: list[dict[str, Any]] = field(default_factory=list)

    def run(self) -> dict[str, Any]:
        last_error = None
        for attempt in range(1, self.max_starts + 1):
            t0 = self.clock()
            event: dict[str, Any] = {"cloud": "modal", "gpu": self.gpu, "gpus": self.gpus,
                                     "price_per_hour": self.price_per_hour, "attempt": attempt}
            try:
                handle = self.client.start(dict(self.config, gpu=self.gpu, gpus=self.gpus))
            except Exception as exc:  # noqa: BLE001 - capacity / API error
                last_error = f"{type(exc).__name__}: {exc}"
                self._record(event, outcome="start_failed", wait_s=self.clock() - t0, error=last_error)
                continue
            self._record(event, outcome="started", wait_s=self.clock() - t0)
            try:
                summary = dict(handle.wait() or {})
            except Exception as exc:  # noqa: BLE001 - preempted or crashed: resume on a new container
                last_error = f"{type(exc).__name__}: {exc}"
                self._record(event, outcome="preempted", run_s=self.clock() - t0, error=last_error)
                continue
            self._record(event, outcome="finished", run_s=self.clock() - t0,
                         evaluated=summary.get("evaluated"))
            return summary
        raise IslandPreempted(f"eval island did not finish after {self.max_starts} starts: {last_error}")

    def _record(self, base: Mapping[str, Any], **fields: Any) -> None:
        record = {**base, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in fields.items()}}
        self.starts.append(record)
        self.emit(ISLAND_EVENT, **record)


def default_price_per_hour(gpu: str = DEFAULT_GPU, gpus: int = DEFAULT_GPUS) -> float | None:
    try:
        from yeto.shape.providers import modal_node_price_per_hour

        return float(modal_node_price_per_hour(gpu, gpus))
    except Exception:  # noqa: BLE001 - unknown gpu name: leave the price unset
        return None


# --- inside the eval-island container ------------------------------------------


def _factory(spec: str) -> Callable[..., Any]:
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def island_main(config: Mapping[str, Any], env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run the eval island over the mounted store until the queue is drained.

    ``config``: ``store`` (mount path), ``holdout`` (path of the hold-out json
    inside the store or image), ``task_ids`` (optional subset), ``trials_v0`` /
    ``trials``, ``sampling`` (optional; default = the earliest queued
    manifest's), ``island_id``, ``volume`` (name, for commit/reload),
    ``finished_marker`` (path; when present the queue is final) or
    ``wait_for_training`` (true: wait until the training side wrote
    ``training-finished.json`` into the store, 5.10). The loader is closed
    (``close()``, if any) when the island stops; its ``loads`` records ride in
    the summary."""
    from yeto.rl.eval.island import EvalIsland, EvalPlan, jsonl_emitter

    env = os.environ if env is None else env
    root = Path(config["store"])
    vol = _volume(config["volume"]) if config.get("volume") else None
    store = EvalStore(root, commit=vol.commit if vol else None, reload=vol.reload if vol else None)
    store.reload()
    sampling = config.get("sampling")
    if sampling is None:
        queued = store.queued_versions()
        if not queued:
            return {"evaluated": []}
        sampling = store.load_manifest(queued[0], verify=False)["sampling"]
    holdout = json.loads(Path(config["holdout"]).read_text())
    rows = []
    if config.get("eval_data"):
        rows = [json.loads(line) for line in Path(config["eval_data"]).read_text().splitlines() if line.strip()]
    plan = EvalPlan.from_holdout(holdout, sampling=sampling, rows=rows, task_ids=config.get("task_ids"),
                                 trials_v0=int(config.get("trials_v0", 4)), trials=int(config.get("trials", 2)))
    loader = _factory(config.get("loader") or env[LOADER_ENV])(config)
    attempt = _factory(config.get("attempt") or env[ATTEMPT_ENV])(config)
    island_id = str(config.get("island_id", "eval-0"))
    emit = jsonl_emitter(root / "events" / f"{island_id}.jsonl", island=island_id)
    if hasattr(loader, "emit") and getattr(loader, "emit") is None:
        loader.emit = emit
    marker = config.get("finished_marker")
    if marker:
        idle = lambda: Path(marker).exists()  # noqa: E731
    elif config.get("wait_for_training"):
        idle = store.training_finished
    else:
        idle = None
    island = EvalIsland(store, plan, loader=loader, attempt=attempt, emit=emit, island_id=island_id)
    try:
        evaluated = island.run(idle=idle, poll_s=float(config.get("poll_s", 30.0)))
    finally:
        close = getattr(loader, "close", None)
        if close is not None:
            close()
        store.commit()
    return {"evaluated": evaluated, "loads": list(getattr(loader, "loads", []) or [])}


# --- the Modal GPU function (5.10) -------------------------------------------------

EVAL_STORE_MOUNT = "/mnt/yeto-eval-store"     # the eval store Volume inside the eval island
CODEX_MOUNT = "/opt/yeto/codex"                 # yeto.rl.CODEX_CONTAINER_DIR (signed codex bundle)
TB2_MOUNT = "/root/tb2-data"                    # terminal-bench-2 checkout (YETO_HARNESS_TB2_TASKS_DIR)
WORKDIR_MOUNT = "/root/sky_workdir"             # yeto source, same place as the training islands
IMAGE_PYTHONPATH = "/root/miles:/root/sglang/python:/root/sky_workdir"
ISLAND_FUNCTION = "eval_island"
SEED_FUNCTION = "eval_seed"


def default_image_ref() -> str:
    from yeto.rl.adapters.miles.pins import MILES_NEXT_IMAGE

    return MILES_NEXT_IMAGE.removeprefix("docker:")


@dataclass
class EvalFunctionSpec:
    """Shape of the eval-island Modal function. ``gpu`` is a Modal GPU string
    (``H100!`` = exactly H100); ``envs`` become one Modal secret (codex
    contract env, Modal token for the judge sandboxes, TB2 knobs)."""

    app_name: str
    volume: str
    workdir: str
    codex_dir: str
    tb2_dir: str
    gpu: str = "H100!"
    gpus: int = 1
    cpu: float = 8.0
    memory_mib: int = 65536
    timeout_s: int = 3600
    image_ref: str = field(default_factory=default_image_ref)
    envs: Mapping[str, str] = field(default_factory=dict)

    def container_env(self) -> dict[str, str]:
        from yeto.launcher import island_role_env
        from yeto.rl.eval.island import EVAL_ISLAND_ROLE

        return {"HOME": "/root", "PYTHONUNBUFFERED": "1", "PYTHONPATH": IMAGE_PYTHONPATH,
                "YETO_HARNESS_TB2_TASKS_DIR": TB2_MOUNT, STORE_ENV: EVAL_STORE_MOUNT,
                MOUNTED_ENV: "1", VOLUME_ENV: self.volume, **island_role_env(EVAL_ISLAND_ROLE),
                **{str(k): str(v) for k, v in self.envs.items()}}


def _gpu_sampler(path: Path, stop: Any, interval_s: float = 5.0) -> None:
    """nvidia-smi memory / utilization every ``interval_s`` into a jsonl (raw data first)."""
    import subprocess

    path.parent.mkdir(parents=True, exist_ok=True)
    while not stop.wait(interval_s):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        except Exception as exc:  # noqa: BLE001
            out = f"error {exc}"
        with open(path, "a") as fh:
            fh.write(json.dumps({"t": time.time(), "nvidia_smi": out.strip()}) + "\n")


def eval_island_body(config: Mapping[str, Any]) -> dict[str, Any]:
    """Inside the eval-island container: check the GPU, sample it, run ``island_main``."""
    import subprocess
    import threading

    names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout.split("\n")
    names = [n.strip() for n in names if n.strip()]
    want = config.get("assert_gpu_name")
    if want and not all(want in n for n in names):
        raise RuntimeError(f"eval island asked for {want}, got {names}")
    run_dir = Path(config["store"]) / "runs" / str(config.get("island_id", "eval-0"))
    infer = dict(config.get("infer") or {})
    infer.setdefault("log_dir", str(run_dir / "logs"))
    config = {**config, "infer": infer}
    stop = threading.Event()
    sampler = threading.Thread(target=_gpu_sampler, args=(run_dir / "gpu.jsonl", stop), daemon=True)
    sampler.start()
    t0 = time.time()
    try:
        summary = island_main(config)
    finally:
        stop.set()
        sampler.join(timeout=10)
    return {**summary, "gpu_names": names, "container": os.environ.get("MODAL_TASK_ID"),
            "wall_s": round(time.time() - t0, 3)}


def eval_seed_body(config: Mapping[str, Any]) -> dict[str, Any]:
    """CPU function: write version 0 into the mounted store through the training export path."""
    import importlib
    import subprocess
    import sys

    for mod, pin in (("peft", "peft==0.17.1"), ("accelerate", "accelerate")):
        try:
            importlib.import_module(mod)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", "/tmp/seed-site",
                            pin], check=True)
            sys.path.insert(0, "/tmp/seed-site")
    from yeto.rl.eval.export import export_initial_version

    vol = _volume(config["volume"])
    store = EvalStore(config["store"], commit=vol.commit, reload=vol.reload)
    store.reload()
    t0 = time.time()
    manifest = export_initial_version(store, config["model"], config["revision"], rank=int(config["rank"]),
                                      targets=str(config["targets"]), sampling=config["sampling"],
                                      seed=int(config.get("seed", 0)))
    if config.get("training_finished"):
        store.mark_training_finished({"final_version": 0, "note": "seeded smoke"})
    return {"manifest": manifest, "seconds": round(time.time() - t0, 3)}


def build_eval_app(spec: EvalFunctionSpec) -> tuple[Any, Any, Any]:
    """(app, eval-island GPU function, seed CPU function). The Volume is mounted
    at :data:`EVAL_STORE_MOUNT` in both."""
    import modal

    image = (modal.Image.from_registry(spec.image_ref)
             .env({"HOME": "/root", "PYTHONUNBUFFERED": "1", "PYTHONPATH": IMAGE_PYTHONPATH})
             .add_local_dir(spec.workdir, WORKDIR_MOUNT, copy=False,
                            ignore=[".git", ".venv", ".claude", "**/__pycache__", "**/*.pyc", "s1-runs"])
             .add_local_dir(spec.codex_dir, CODEX_MOUNT, copy=False)
             .add_local_dir(spec.tb2_dir, TB2_MOUNT, copy=False, ignore=[".git"]))
    app = modal.App(spec.app_name)
    volume = modal.Volume.from_name(spec.volume, create_if_missing=True)
    secret = modal.Secret.from_dict(spec.container_env())
    gpu = spec.gpu if spec.gpus == 1 else f"{spec.gpu}:{spec.gpus}"
    island = app.function(image=image, gpu=gpu, cpu=spec.cpu, memory=spec.memory_mib, timeout=spec.timeout_s,
                          secrets=[secret], volumes={EVAL_STORE_MOUNT: volume}, name=ISLAND_FUNCTION)(eval_island_body)
    seed = app.function(image=image, cpu=4.0, memory=16384, timeout=1200, secrets=[secret],
                        volumes={EVAL_STORE_MOUNT: volume}, name=SEED_FUNCTION)(eval_seed_body)
    return app, island, seed


class ModalFunctionClient:
    """``EvalIslandLauncher`` client over a Modal function: ``start`` spawns, ``wait`` = ``get``."""

    def __init__(self, function: Any) -> None:
        self.function = function
        self.calls: list[Any] = []

    def start(self, config: Mapping[str, Any]) -> Any:
        call = self.function.spawn({k: v for k, v in config.items() if k not in ("gpu", "gpus")})
        self.calls.append(call)
        return _CallHandle(call)


class _CallHandle:
    def __init__(self, call: Any) -> None:
        self.call = call

    def wait(self) -> dict[str, Any]:
        return self.call.get()
