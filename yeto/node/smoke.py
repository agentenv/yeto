"""``python -m yeto.node.smoke --spec <sampler spec>``: a sampler image's smoke on this node.

``yeto image adopt --role sampler`` (:mod:`yeto.launch.image_smoke`) runs this inside
the image it adopts, started the way the node runtime starts a sampler
(:func:`yeto.node.launch.container_argv`: the profile's device flags, the node's
directories at ``/work`` / ``/models`` / ``/root/.cache``). With the engine of the
pool's sampler spec (:func:`yeto.rl.engine.engine_sampler_entry.build_engine`, the
sampler's own factory) it

1. ``start``: starts the engine on its base weights (policy version 0);
2. ``generate``: one completion;
3. ``reload``: loads the release ``--release`` (default: the engine's base weights)
   as policy version 1, the reload the engine does for every trainer release (in
   place, or a restart for an engine that cannot);
4. ``generate_after_reload``: one completion at version 1;

then stops the engine. Every step prints a ``YETO_SMOKE`` line and the run ends with
``YETO_SMOKE_RESULT {"ok": ..., "step": <the failed step>, ...}``; the exit code is 0
only when every step passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

STEP_MARK = "YETO_SMOKE"
RESULT_MARK = "YETO_SMOKE_RESULT"
STEPS = ("start", "generate", "reload", "generate_after_reload")


def _emit(mark: str, record: Mapping[str, Any], out: Any) -> None:
    print(f"{mark} {json.dumps(dict(record), sort_keys=True, default=str)}", file=out, flush=True)


def base_dir(spec: Mapping[str, Any]) -> str | None:
    """The engine's base weights (``engine.server.base_dir``), if the spec names them."""
    server = (spec.get("engine") or {}).get("server") or {}
    value = server.get("base_dir")
    return str(value) if value else None


def run_smoke(engine: Any, *, release: str | None, prompt: str = "Say hello.", max_tokens: int = 16,
              out: Any = None, clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """The four steps on ``engine`` (a ``SamplerEngine``); the result record. The engine
    is stopped whatever happened."""
    from yeto.rl.engine.sampler_engine import SamplingParams  # noqa: PLC0415

    out = out or sys.stdout
    sampling = SamplingParams(max_tokens=int(max_tokens), temperature=0.0)
    times: dict[str, float] = {}
    record: dict[str, Any] = {"ok": False, "step": None, "seconds": times}

    def text_of(completions: Any) -> str:
        first = completions[0][0]
        return str(getattr(first, "text", "") or "")[:200]

    steps: list[tuple[str, Callable[[], Any]]] = [
        ("start", lambda: engine.start(None, 0)),
        ("generate", lambda: text_of(engine.generate([prompt], sampling))),
        ("reload", lambda: engine.load(Path(release) if release else None, 1)),
        ("generate_after_reload", lambda: text_of(engine.generate([prompt], sampling))),
    ]
    try:
        for name, fn in steps:
            t0 = clock()
            try:
                value = fn()
            except Exception as error:  # noqa: BLE001 - the step's failure is the smoke's answer
                times[name] = round(clock() - t0, 3)
                record.update(step=name, error=f"{type(error).__name__}: {error}"[:600])
                _emit(STEP_MARK, {"step": name, "ok": False, "seconds": times[name], "error": record["error"]}, out)
                return record
            times[name] = round(clock() - t0, 3)
            detail: dict[str, Any] = {}
            if name.startswith("generate"):
                detail["text"] = value
                if not value:
                    record.update(step=name, error="the engine returned an empty completion")
                    _emit(STEP_MARK, {"step": name, "ok": False, "seconds": times[name], "error": record["error"]},
                          out)
                    return record
            elif value is not None and hasattr(value, "reload_mode"):
                detail["reload_mode"] = value.reload_mode
                record[f"{name}_mode"] = value.reload_mode
            _emit(STEP_MARK, {"step": name, "ok": True, "seconds": times[name], **detail}, out)
        served = getattr(engine, "served_version", lambda: None)()
        if served is not None and int(served) != 1:
            record.update(step="reload", error=f"the engine serves version {served} after the reload, not 1")
            return record
        record.update(ok=True, step=None)
        return record
    finally:
        try:
            engine.stop()
        except Exception:  # noqa: BLE001 - a failed stop does not change the answer
            pass


def main(argv: list[str] | None = None, *, out: Any = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m yeto.node.smoke", description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", required=True, help="the sampler spec (JSON) whose engine is smoked")
    parser.add_argument("--release", default=None,
                        help="the release directory the reload loads (default: engine.server.base_dir)")
    parser.add_argument("--prompt", default="Say hello.")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args(argv)
    out = out or sys.stdout
    from yeto.rl.engine.engine_sampler_entry import build_engine  # noqa: PLC0415

    try:
        spec = json.loads(Path(args.spec).read_text())
        engine = build_engine(spec)
    except Exception as error:  # noqa: BLE001 - a spec the engine refuses is the start's failure
        record = {"ok": False, "step": "start", "error": f"{type(error).__name__}: {error}"[:600], "seconds": {}}
        _emit(RESULT_MARK, record, out)
        return 1
    record = run_smoke(engine, release=args.release or base_dir(spec), prompt=args.prompt,
                       max_tokens=args.max_tokens, out=out)
    _emit(RESULT_MARK, record, out)
    return 0 if record["ok"] else 1


def parse_result(text: str) -> dict[str, Any] | None:
    """The last ``YETO_SMOKE_RESULT`` record of a smoke's output (None: there is none)."""
    found = None
    for line in (text or "").splitlines():
        if line.startswith(RESULT_MARK + " "):
            try:
                found = json.loads(line[len(RESULT_MARK) + 1:])
            except ValueError:
                continue
    return found


if __name__ == "__main__":
    raise SystemExit(main())
