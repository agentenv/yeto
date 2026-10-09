"""S17 N17: SGLang qwen3_coder single-tool-call probe (old image 4aeafd77 vs new cd920143).

One Modal H100! function per image: start SGLang exactly as the training island does
for --tito-model qwen35 (Miles resolves reasoning/tool-call parser + fixed chat template),
then send N chat completions with tool_choice=auto and parallel_tool_calls false / true,
using a prompt that asks for two tool calls in one reply. Count tool calls per reply.
Raw responses go to the Modal return value -> results-<tag>.json here.
Run with reward-env-venv python (modal client).
"""
import json, os, sys, time, threading
from pathlib import Path
import modal

OUT = Path(__file__).resolve().parent
APP = "yeto-s17-n17-toolprobe"
IMAGES = {"old": "ghcr.io/michaellchung/yeto-miles-ports@sha256:4aeafd7789dbcc02d71f0068c449477e8ab5f6d5bdb037d6d34e8fbf12a5b039",
          "new": "ghcr.io/michaellchung/yeto-miles-ports@sha256:cd920143dd6d336a39c556faa107ca70e96b2871ab8ee8fbab70356b2a1411b5"}
MODEL, REV = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
N = int(os.environ.get("N17_N", "32")); HARD_S = 1500

app = modal.App(APP)

def body(tag: str, n: int) -> dict:
    import subprocess, httpx, sys as _s, json as _j, time as _t
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    from miles.utils.chat_template_utils import resolve_fixed_chat_template, resolve_reasoning_and_tool_call_parser
    reasoning, tool = resolve_reasoning_and_tool_call_parser("qwen35")
    template, kwargs = resolve_fixed_chat_template("qwen35")
    import sglang
    sg = subprocess.run(["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    argv = [_s.executable, "-m", "sglang.launch_server", "--model-path", MODEL, "--revision", REV, "--host", "127.0.0.1",
            "--port", "30000", "--trust-remote-code", "--context-length", "8192", "--mem-fraction-static", "0.8"]
    if reasoning: argv += ["--reasoning-parser", reasoning]
    if tool: argv += ["--tool-call-parser", tool]
    if template: argv += ["--chat-template", template]
    t0 = _t.time()
    log = open("/tmp/sglang.log", "w")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    ready = None
    for _ in range(900):
        try:
            if httpx.get("http://127.0.0.1:30000/health", timeout=5).status_code == 200:
                ready = _t.time() - t0; break
        except Exception:
            pass
        if proc.poll() is not None: break
        _t.sleep(1)
    if ready is None:
        return {"tag": tag, "gpu": gpu, "error": "sglang not ready", "log_tail": open("/tmp/sglang.log").read()[-4000:]}
    tools = [{"type": "function", "function": {"name": "shell", "description": "Run one shell command in the sandbox.",
              "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]
    msgs = [{"role": "system", "content": "You are a terminal agent. You can call the `shell` tool."},
            {"role": "user", "content": "Inspect the machine. In THIS single reply, call the shell tool TWICE: "
             "one call with `ls /` and a second, separate call with `pwd`. Emit both tool calls now, no prose."}]
    import concurrent.futures as cf
    def one(par, i):
        body = {"model": MODEL, "messages": msgs, "tools": tools, "tool_choice": "auto", "parallel_tool_calls": par,
                "max_tokens": 2048, "temperature": 1.0, "seed": 1000 + i, "chat_template_kwargs": dict(kwargs or {})}
        try:
            r = httpx.post("http://127.0.0.1:30000/v1/chat/completions", json=body, timeout=300).json()
            ch = r["choices"][0]
            tc = ch["message"].get("tool_calls") or []
            return {"par": par, "i": i, "n_calls": len(tc), "finish": ch.get("finish_reason"),
                    "names": [c["function"]["name"] for c in tc], "args": [c["function"]["arguments"] for c in tc],
                    "content": (ch["message"].get("content") or "")[:300],
                    "completion_tokens": r.get("usage", {}).get("completion_tokens")}
        except Exception as exc:
            return {"par": par, "i": i, "error": repr(exc)[:300]}
    rows = []
    t1 = _t.time()
    with cf.ThreadPoolExecutor(16) as ex:
        futs = [ex.submit(one, par, i) for par in (False, True) for i in range(n)]
        rows = [f.result() for f in futs]
    proc.terminate()
    return {"tag": tag, "gpu": gpu, "sglang_commit": sg, "sglang_version": getattr(sglang, "__version__", None),
            "reasoning_parser": reasoning, "tool_call_parser": tool, "chat_template": template, "kwargs": kwargs,
            "ready_s": round(ready, 1), "probe_s": round(_t.time() - t1, 1), "rows": rows,
            "log_tail": open("/tmp/sglang.log").read()[-3000:]}

fns = {}
for tag, ref in IMAGES.items():
    img = modal.Image.from_registry(ref).env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
    fns[tag] = app.function(image=img, gpu="H100!", cpu=8.0, memory=65536, timeout=HARD_S, name=f"probe_{tag}",
                            serialized=True)(body)

def watchdog():
    time.sleep(HARD_S + 600)
    os.system(f"/home/michael/work/reward-env-venv/bin/modal app stop {APP} >> {OUT}/watchdog.out 2>&1"); os._exit(3)

if __name__ == "__main__":
    threading.Thread(target=watchdog, daemon=True).start()
    (OUT / "start_utc.txt").write_text(time.strftime("%FT%TZ", time.gmtime()) + "\n")
    with modal.enable_output():
        with app.run():
            (OUT / "app_id.txt").write_text(app.app_id + "\n")
            calls = {tag: fns[tag].spawn(tag, N) for tag in IMAGES}
            for tag, c in calls.items():
                try:
                    res = c.get(timeout=HARD_S + 120)
                except Exception as exc:
                    res = {"tag": tag, "error": repr(exc)[:2000]}
                (OUT / f"results-{tag}.json").write_text(json.dumps(res, indent=1, default=str))
                print(tag, "done", res.get("error"), res.get("gpu"), res.get("ready_s"), flush=True)
    (OUT / "end_utc.txt").write_text(time.strftime("%FT%TZ", time.gmtime()) + "\n")
