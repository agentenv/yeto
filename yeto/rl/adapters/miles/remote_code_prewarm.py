"""Fill transformers' remote-code module cache on every island node before any engine starts.

S27 G-CK. For a ``trust_remote_code`` model, transformers copies the model directory's
``*.py`` files into a node-local module cache (``HF_MODULES_CACHE``, default
``~/.cache/huggingface/modules``) and imports them from there. The copy is not atomic
(``shutil.copyfile`` truncates and rewrites the target) and transformers' own lock only
covers threads of one process. Several SGLang engines (and the TP ranks of each) start
on one node at the same time, so one process can import a file that another process is
rewriting, and the class is missing from the module: s27-ch-d1 ``configuration_nemotron_h``,
s27-bc-a ``processing_nemotron_h_omni``, s27-bc-b ``image_processing_nemotron_h_omni``.

transformers copies a module file only when the cached copy is missing or differs from
the source. Once the cache holds a complete copy of every file, the engines only compare
and read it. So one child interpreter per node runs, under an exclusive file lock in the
cache directory, before Miles creates any engine:

1. every top-level ``*.py`` of the model directory goes through
   ``get_cached_module_file`` (the cache path and copy that an ``auto_map`` entry naming
   that file gets, relative imports included);
2. the loaders that the model directory maps to remote code run once with
   ``trust_remote_code=True``: ``AutoConfig``, ``AutoTokenizer``, ``AutoProcessor``,
   ``AutoImageProcessor`` (the processor pair also when a processor config file exists).

The child is the Ray worker's ``sys.executable``: the interpreter Miles launches
``sglang.launch_server`` with, so its transformers lays out the cache as the engines do.
A failure is only a WARNING and never blocks the start: the engines then load remote code
as they did without this step.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

MODULE = "yeto.rl.adapters.miles.remote_code_prewarm"
RESULT_PREFIX = "YETO_REMOTE_CODE_PREWARM "
LOCK_NAME = ".yeto-remote-code-prewarm.lock"
# One node: interpreter start, transformers import, the loaders. A child that takes
# longer is killed and the node is reported; the driver waits this plus a grace.
NODE_TIMEOUT_S = 300.0
DRIVER_GRACE_S = 60.0
EVENT = "remote_code_prewarm"

_AUTO_MAP_FILES = (
    "config.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
_PROCESSOR_FILES = ("processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json")
_IMAGE_PROCESSOR_FILES = ("preprocessor_config.json",)
_PROCESSOR_CLASSES = frozenset({"AutoProcessor", "AutoImageProcessor", "AutoVideoProcessor",
                                "AutoFeatureExtractor"})


def auto_map_classes(model_path: str | os.PathLike) -> frozenset[str]:
    """The ``auto_map`` keys that the JSON configs of a local model directory declare.

    An empty set means transformers loads no remote code from this directory. A
    tokenizer config's legacy list form maps ``AutoTokenizer``.
    """
    found: set[str] = set()
    for name in _AUTO_MAP_FILES:
        try:
            data = json.loads((Path(model_path) / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        auto_map = data.get("auto_map") if isinstance(data, dict) else None
        if isinstance(auto_map, dict):
            found.update(str(key) for key in auto_map)
        elif isinstance(auto_map, (list, tuple)) and name == "tokenizer_config.json":
            found.add("AutoTokenizer")
    return frozenset(found)


def loader_plan(model_path: str | os.PathLike) -> tuple[str, ...]:
    """transformers ``Auto*`` loaders to run once for ``model_path`` (empty: no remote code)."""
    classes = auto_map_classes(model_path)
    if not classes:
        return ()
    root = Path(model_path)
    has_processor = bool(classes & _PROCESSOR_CLASSES) or any(
        (root / name).is_file() for name in _PROCESSOR_FILES)
    has_image_processor = "AutoImageProcessor" in classes or any(
        (root / name).is_file() for name in _IMAGE_PROCESSOR_FILES)
    plan = []
    if "AutoConfig" in classes:
        plan.append("AutoConfig")
    if "AutoTokenizer" in classes:
        plan.append("AutoTokenizer")
    if has_processor:
        plan.append("AutoProcessor")
    if has_image_processor:
        plan.append("AutoImageProcessor")
    return tuple(plan)


def engine_model_paths(miles_args: Any) -> list[str]:
    """Model directories with remote code that this island's processes load.

    The engines load ``--rollout-hf-checkpoint`` when set, else ``--hf-checkpoint``
    (Miles ``rollout_hf_checkpoint``); the trainer loads ``--hf-checkpoint``. The cache
    path is the directory's base name plus a hash of the module source, so a second
    directory with the same files adds no work.
    """
    paths: list[str] = []
    for value in (getattr(miles_args, "rollout_hf_checkpoint", None),
                  getattr(miles_args, "hf_checkpoint", None)):
        if value and str(value) not in paths and auto_map_classes(value):
            paths.append(str(value))
    return paths


def _short(exc: BaseException, limit: int = 300) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def warm_model_dir(model_path: str, *, transformers_module: Any = None,
                   dynamic_module_utils: Any = None) -> dict[str, Any]:
    """Child side for one directory: cache every top-level ``*.py``, then run the loaders.

    Each step is independent; a failed step is recorded and the next one still runs.
    """
    if transformers_module is None:
        import transformers as transformers_module
    if dynamic_module_utils is None:
        from transformers import dynamic_module_utils
    cached: list[str] = []
    loaded: list[str] = []
    errors: list[dict[str, str]] = []
    for source in sorted(Path(model_path).glob("*.py")):
        try:
            dynamic_module_utils.get_cached_module_file(model_path, source.name)
            cached.append(source.name)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            errors.append({"step": f"cache {source.name}", "error": _short(exc)})
    for loader in loader_plan(model_path):
        try:
            getattr(transformers_module, loader).from_pretrained(model_path, trust_remote_code=True)
            loaded.append(loader)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            errors.append({"step": loader, "error": _short(exc)})
    return {"path": model_path, "cached": cached, "loaded": loaded, "errors": errors}


@contextlib.contextmanager
def exclusive_lock(directory: str | os.PathLike) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``<directory>/LOCK_NAME`` (released when the process ends)."""
    import fcntl

    os.makedirs(directory, exist_ok=True)
    with open(Path(directory) / LOCK_NAME, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main(argv: Sequence[str] | None = None) -> int:
    """Child entry: warm every ``--model-path`` under the lock; print one JSON report line."""
    parser = argparse.ArgumentParser(prog=MODULE)
    parser.add_argument("--model-path", action="append", required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    from transformers.utils import HF_MODULES_CACHE

    with exclusive_lock(HF_MODULES_CACHE):
        locked = time.monotonic()
        models = [warm_model_dir(path) for path in args.model_path]
    report = {
        "cache": str(HF_MODULES_CACHE),
        "import_and_lock_s": round(locked - started, 3),
        "elapsed_s": round(time.monotonic() - started, 3),
        "models": models,
    }
    print(RESULT_PREFIX + json.dumps(report, sort_keys=True), flush=True)
    return 1 if any(model["errors"] for model in models) else 0


def _parse_report(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX):])
            except ValueError:
                return None
    return None


def _tail(text: str, lines: int = 5) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def run_on_this_node(model_paths: Sequence[str], timeout_s: float = NODE_TIMEOUT_S, *,
                     run: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Node side (a Ray task pinned to the node): run :func:`main` in a child interpreter.

    The child gets no GPU (``CUDA_VISIBLE_DEVICES`` empty) and this checkout first on
    ``PYTHONPATH``. Returns ``{"host", "ok", ...}``; never raises.
    """
    host = socket.gethostname()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    root = str(Path(__file__).resolve().parents[4])
    env["PYTHONPATH"] = os.pathsep.join([root, env["PYTHONPATH"]]) if env.get("PYTHONPATH") else root
    cmd = [sys.executable, "-m", MODULE]
    for path in model_paths:
        cmd += ["--model-path", str(path)]
    started = time.monotonic()
    try:
        proc = run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"host": host, "ok": False, "error": f"timed out after {timeout_s:.0f} s"}
    except OSError as exc:
        return {"host": host, "ok": False, "error": _short(exc)}
    report = _parse_report(proc.stdout or "")
    result: dict[str, Any] = {"host": host, "rc": proc.returncode, "report": report,
                              "wall_s": round(time.monotonic() - started, 3)}
    result["ok"] = proc.returncode == 0 and report is not None
    if report is None:
        result["error"] = f"no report (rc {proc.returncode}): {_tail(proc.stderr or '')}"
    return result


def node_task(model_paths: Sequence[str], timeout_s: float) -> dict[str, Any]:
    """The Ray task body: plain positional parameters only (Ray serializes the call)."""
    return run_on_this_node(model_paths, timeout_s)


def _island_nodes(ray_module: Any) -> list[dict[str, Any]]:
    """Alive Ray nodes with GPUs, one per machine (address), in a stable order."""
    nodes = [n for n in ray_module.nodes()
             if n.get("Alive") and (n.get("Resources") or {}).get("GPU", 0)]
    by_address: dict[str, dict[str, Any]] = {}
    for node in sorted(nodes, key=lambda n: (str(n.get("NodeManagerAddress")), str(n.get("NodeID")))):
        by_address.setdefault(str(node.get("NodeManagerAddress")), node)
    return list(by_address.values())


def _run_on_island_nodes(ray_module: Any, model_paths: Sequence[str], *, affinity: Any,
                         timeout_s: float, clock: Callable[[], float]) -> list[dict[str, Any]]:
    nodes = _island_nodes(ray_module)
    if not nodes:
        raise RuntimeError("no alive Ray node with GPUs")
    task = ray_module.remote(num_cpus=0, num_gpus=0)(node_task)
    refs = []
    for node in nodes:
        strategy = affinity(node_id=node["NodeID"], soft=False)
        refs.append((str(node.get("NodeManagerAddress")),
                     task.options(scheduling_strategy=strategy).remote(list(model_paths), timeout_s)))
    deadline = clock() + timeout_s + DRIVER_GRACE_S
    results = []
    for address, ref in refs:
        try:
            result = dict(ray_module.get(ref, timeout=max(1.0, deadline - clock())))
        except Exception as exc:  # noqa: BLE001 - a node that cannot run the task is reported
            with contextlib.suppress(Exception):
                ray_module.cancel(ref, force=True)
            result = {"ok": False, "error": _short(exc)}
        result["node"] = address
        results.append(result)
    return results


def _summary(result: dict[str, Any]) -> str:
    models = (result.get("report") or {}).get("models") or []
    cached = sum(len(m.get("cached") or ()) for m in models)
    loaded = sorted({name for m in models for name in (m.get("loaded") or ())})
    return f"{cached} module file(s) cached, loaders {','.join(loaded) or 'none'}"


def _failures(result: dict[str, Any]) -> str:
    if result.get("error"):
        return str(result["error"])
    models = (result.get("report") or {}).get("models") or []
    return "; ".join(f"{e['step']}: {e['error']}" for m in models for e in (m.get("errors") or ()))


def prewarm_remote_code(miles_args: Any, *, ray_module: Any = None, affinity: Any = None,
                        run_local: Callable[[Sequence[str]], dict[str, Any]] | None = None,
                        emit: Callable[[dict[str, Any]], None] | None = None,
                        timeout_s: float = NODE_TIMEOUT_S, clock: Callable[[], float] = time.monotonic,
                        out: Any = None) -> list[dict[str, Any]] | None:
    """Driver side: warm the module cache once on every island machine; call it after the
    island's Ray is connected and before Miles creates any engine or trainer actor.

    With Ray connected, one task per machine (alive node with GPUs, deduplicated by
    address) is pinned there with ``NodeAffinitySchedulingStrategy``; without a
    connected Ray (Miles starts a local one later), it runs on this machine. Returns the
    per-node results, or ``None`` when the model has no remote code or the step could not
    run. Never raises: each problem is one ``[rl] WARNING`` line.
    """
    out = sys.stderr if out is None else out
    started = clock()
    try:
        paths = engine_model_paths(miles_args)
        if not paths:
            return None
        if ray_module is None:
            try:
                import ray as ray_module
            except ImportError:
                ray_module = None
        if ray_module is not None and ray_module.is_initialized():
            if affinity is None:
                from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy as affinity
            results = _run_on_island_nodes(ray_module, paths, affinity=affinity,
                                           timeout_s=timeout_s, clock=clock)
        else:
            local = run_local or (lambda p: run_on_this_node(p, timeout_s))
            results = [{**local(paths), "node": "local"}]
    except Exception as exc:  # noqa: BLE001 - this step must never block the start
        print(f"[rl] WARNING: remote-code prewarm did not run ({_short(exc)}); engines load "
              "remote code without it and may race on the module cache", file=out, flush=True)
        return None
    for result in results:
        where = f"{result.get('node')} ({result.get('host', '?')})"
        if result.get("ok"):
            print(f"[rl] remote-code prewarm on {where}: {_summary(result)}", flush=True)
        else:
            print(f"[rl] WARNING: remote-code prewarm on {where} failed: {_failures(result)}; "
                  "engines on that node may race on the module cache", file=out, flush=True)
    if emit is not None:
        with contextlib.suppress(Exception):
            emit({"event": EVENT, "paths": paths, "seconds": round(clock() - started, 3),
                  "nodes": [{"node": r.get("node"), "host": r.get("host"), "ok": bool(r.get("ok"))}
                            for r in results]})
    return results


if __name__ == "__main__":
    raise SystemExit(main())
