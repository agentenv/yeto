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

from yeto.rl.eval.store import EvalStore

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
        changed.sort(key=lambda c: (c[1].endswith("manifest.json") or c[1].startswith("queue/"), c[1]))
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
    ``finished_marker`` (path; when present the queue is final)."""
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
    plan = EvalPlan.from_holdout(holdout, sampling=sampling, task_ids=config.get("task_ids"),
                                 trials_v0=int(config.get("trials_v0", 4)), trials=int(config.get("trials", 2)))
    loader = _factory(config.get("loader") or env[LOADER_ENV])(config)
    attempt = _factory(config.get("attempt") or env[ATTEMPT_ENV])(config)
    island_id = str(config.get("island_id", "eval-0"))
    emit = jsonl_emitter(root / "events" / f"{island_id}.jsonl", island=island_id)
    marker = config.get("finished_marker")
    island = EvalIsland(store, plan, loader=loader, attempt=attempt, emit=emit, island_id=island_id)
    evaluated = island.run(idle=(lambda: Path(marker).exists()) if marker else None,
                           poll_s=float(config.get("poll_s", 30.0)))
    return {"evaluated": evaluated}
