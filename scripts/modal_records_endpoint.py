#!/usr/bin/env python
"""A records-proxy model endpoint on Modal for Harbor agent trials (S25 N10.7 smoke).

One GPU container runs the SGLang server adapter (``SGLangServerEngine``, TP1)
on a checkpoint from a Modal Volume and the N10.5 records proxy in front of it
(``RecordsProxy``: Responses -> Chat, token / loss mask / log-prob records).
The proxy is reachable from outside through a Modal tunnel, under a secret path
prefix only (``/k/<secret>/...``; the proxy itself has no authentication).  The
tunnel URL is put into a Modal Dict; the container serves until the Dict's
``stop`` key is set or ``max_serve_s`` passes, then writes every recorded
trajectory to the results Volume.

    python scripts/modal_records_endpoint.py launch --run-name g2-a --secret-file /tmp/secret
    python scripts/modal_records_endpoint.py stop --run-name g2-a

``launch`` spawns the call and returns (killing a client that blocks on
``.remote()`` cancels the input).  The GPU type is pinned (``H100!``) and
asserted inside the container.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

APP_NAME = "yeto-records-endpoint"
MODELS_VOLUME = "yeto-models-qwen38"
MODELS_MOUNT = "/mnt/models"
RESULTS_VOLUME = "yeto-s25-records-endpoint"
RESULTS_MOUNT = "/mnt/results"
CONTROL_DICT = "yeto-records-endpoint-control"
CONTAINER_WORKDIR = "/root/sky_workdir"  # yeto source in the container (modal_eval_island.WORKDIR_MOUNT)
PROXY_PORT = 8000
SGLANG_PORT = 30000

DEFAULTS: dict[str, Any] = {
    "model_subdir": "hf/Qwen3.8-27B/1d4bf0f2",
    "served_model_name": "Qwen3.8-27B",
    "template": "qwen38",
    "template_kwargs": {"reasoning_effort": "medium"},
    "max_model_len": 32768,
    "mem_fraction_static": 0.88,
    "signed_sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 8192},
    "assert_gpu_name": "H100",
    "max_serve_s": 3600,
}


def _gpu_names() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30).stdout
    return [n.strip() for n in out.splitlines() if n.strip()]


def serve(config: dict[str, Any]) -> dict[str, Any]:
    """Inside the container (module doc)."""
    import asyncio
    import dataclasses

    import modal

    sys.path.insert(0, CONTAINER_WORKDIR)
    from yeto.cloud.modal_eval_island import _image_paths

    _image_paths()
    from aiohttp import web

    import sglang

    from yeto.rl.adapters.sampler_engines.server import ServerConfig
    from yeto.rl.adapters.sampler_engines.sglang_server import SGLangOptions, SGLangServerEngine
    from yeto.rl.harness.chat_templates import TEMPLATES, ChatTemplate
    from yeto.rl.harness.gateway.proxy import RecordsProxy

    names = _gpu_names()
    if not names or not all(config["assert_gpu_name"] in n for n in names):
        raise RuntimeError(f"asked for {config['assert_gpu_name']}, got {names}")
    run_dir = Path(RESULTS_MOUNT) / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    results = modal.Volume.from_name(RESULTS_VOLUME)
    control = modal.Dict.from_name(CONTROL_DICT, create_if_missing=True)
    began = time.time()
    meta: dict[str, Any] = {"gpu_names": names, "sglang": sglang.__version__, "config": {
        k: v for k, v in config.items() if k != "secret"}, "container": os.environ.get("MODAL_TASK_ID")}

    model_dir = Path(MODELS_MOUNT) / config["model_subdir"]
    engine = SGLangServerEngine(
        ServerConfig(base_dir=model_dir, card_model=names[0], device_family="cuda",
                     engine_version=sglang.__version__, precision="bf16", quant_scheme="bf16",
                     topology="tp8", max_model_len=int(config["max_model_len"]), port=SGLANG_PORT,
                     python=sys.executable, start_timeout_s=1800.0,
                     extra_args=("--mem-fraction-static", str(config["mem_fraction_static"]))),
        SGLangOptions(tp=1, served_model_name=config["served_model_name"]))
    report = engine.start(None, 0)
    meta["engine_start_s"] = round(report.pause_s, 1)
    spec = TEMPLATES[config["template"]]
    spec = dataclasses.replace(spec, template_kwargs={**spec.template_kwargs, **config["template_kwargs"]})
    template = ChatTemplate(spec, ChatTemplate.load(spec.name, tokenizer_path=str(model_dir)).tokenizer)
    proxy = RecordsProxy(engine, template, model=config["served_model_name"],
                         signed_sampling=config["signed_sampling"])
    trajectories: set[str] = set()

    @web.middleware
    async def note(request: web.Request, handler: Any) -> web.StreamResponse:
        tid = request.match_info.get("tid")
        if tid:
            trajectories.add(tid)
        return await handler(request)

    inner = proxy.app()
    inner.middlewares.append(note)
    outer = web.Application()
    outer.add_subapp(f"/k/{config['secret']}/", inner)

    loop = asyncio.new_event_loop()

    async def start_server() -> web.AppRunner:
        runner = web.AppRunner(outer)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", PROXY_PORT).start()
        return runner

    async def snapshot() -> str:  # runs on the server loop: no concurrent mutation
        records = {}
        for tid in sorted(trajectories):
            try:
                records[tid] = proxy.records(tid)
            except Exception as exc:  # noqa: BLE001 - keep the others
                records[tid] = {"error": repr(exc)}
        return json.dumps(records)

    def dump() -> None:
        text = asyncio.run_coroutine_threadsafe(snapshot(), loop).result(timeout=120)
        (run_dir / "records.json").write_text(text)
        (run_dir / "meta.json").write_text(json.dumps({**meta, "served_s": round(time.time() - began, 1)}))
        results.commit()

    threading.Thread(target=loop.run_forever, daemon=True).start()
    runner = asyncio.run_coroutine_threadsafe(start_server(), loop).result(timeout=60)
    url_key, stop_key = f"{config['run_name']}/url", f"{config['run_name']}/stop"
    try:
        with modal.forward(PROXY_PORT) as tunnel:
            control[url_key] = tunnel.url
            print(f"[records-endpoint] serving, engine start {meta['engine_start_s']} s", flush=True)
            deadline = began + float(config["max_serve_s"])
            while time.time() < deadline and not control.get(stop_key, False):
                time.sleep(15)
                dump()
    finally:
        control.pop(url_key, None)
        dump()
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(timeout=60)
        engine.stop()
    return {"run_name": config["run_name"], "trajectories": len(trajectories)}


def check(config: dict[str, Any]) -> dict[str, Any]:
    """CPU container: image imports, checkpoint files and the template, no engine."""
    sys.path.insert(0, CONTAINER_WORKDIR)
    from yeto.cloud.modal_eval_island import _image_paths

    _image_paths()
    import aiohttp
    import sglang

    from yeto.rl.adapters.sampler_engines.sglang_server import SGLangServerEngine  # noqa: F401
    from yeto.rl.harness.chat_templates import ChatTemplate, check_alignment
    from yeto.rl.harness.gateway.proxy import RecordsProxy  # noqa: F401

    model_dir = Path(MODELS_MOUNT) / config["model_subdir"]
    template = ChatTemplate.load(config["template"], tokenizer_path=str(model_dir))
    messages = [{"role": "system", "content": "a"}, {"role": "system", "content": "b"},
                {"role": "user", "content": "hi"}]
    return {"sglang": sglang.__version__, "aiohttp": aiohttp.__version__,
            "shards": len(list(model_dir.glob("*.safetensors"))),
            "stop_token_ids": template.stop_token_ids,
            "aligned": check_alignment(template, messages).aligned}


def build_app(workdir: str, gpu: str, timeout_s: int) -> tuple[Any, Any, Any]:
    import modal

    sys.path.insert(0, workdir)
    from yeto.cloud.modal_eval_island import WORKDIR_MOUNT
    from yeto.rl.adapters.miles.pins import MILES_NEXT_IMAGE

    assert WORKDIR_MOUNT == CONTAINER_WORKDIR

    image = (modal.Image.from_registry(MILES_NEXT_IMAGE.removeprefix("docker:"))
             .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
             .add_local_dir(workdir, WORKDIR_MOUNT, copy=False,
                            ignore=[".git", ".venv", ".claude", "**/__pycache__", "**/*.pyc", "s1-runs",
                                    "syncer/target"]))
    app = modal.App(APP_NAME)
    models = modal.Volume.from_name(MODELS_VOLUME)
    results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
    fn = app.function(image=image, gpu=gpu, cpu=8.0, memory=65536, timeout=timeout_s, retries=0,
                      volumes={MODELS_MOUNT: models.read_only(), RESULTS_MOUNT: results},
                      name="serve", serialized=True)(serve)
    checker = app.function(image=image, cpu=2.0, memory=8192, timeout=600, retries=0,
                           volumes={MODELS_MOUNT: models.read_only()}, name="check")(check)
    return app, fn, checker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("check", "launch", "stop", "url"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--secret-file", type=Path, help="launch: file holding the path secret")
    parser.add_argument("--gpu", default="H100!")
    parser.add_argument("--max-serve-s", type=int, default=DEFAULTS["max_serve_s"])
    parser.add_argument("--workdir", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    import modal

    control = modal.Dict.from_name(CONTROL_DICT, create_if_missing=True)
    if args.action == "stop":
        control[f"{args.run_name}/stop"] = True
        return
    if args.action == "url":
        print(control.get(f"{args.run_name}/url", ""))
        return
    if args.action == "check":
        app, _, checker = build_app(args.workdir, args.gpu, timeout_s=600)
        with modal.enable_output(), app.run():
            print(json.dumps(checker.remote(dict(DEFAULTS))))
        return
    secret = args.secret_file.read_text().strip()
    if len(secret) < 24:
        raise SystemExit("the path secret must have at least 24 characters")
    config = {**DEFAULTS, "run_name": args.run_name, "secret": secret, "max_serve_s": args.max_serve_s}
    app, fn, _ = build_app(args.workdir, args.gpu, timeout_s=args.max_serve_s + 1800)
    with modal.enable_output(), app.run(detach=True):
        call = fn.spawn(config)
    print(json.dumps({"function_call_id": call.object_id, "run_name": args.run_name}))


if __name__ == "__main__":
    main()
