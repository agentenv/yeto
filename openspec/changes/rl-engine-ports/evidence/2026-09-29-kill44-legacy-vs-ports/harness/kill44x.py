"""4.4 legacy-vs-ports: one strict island (syncer, learners=1), SIGKILL the learner during
round 2 training, restart it. usage: kill44x.py <legacy|ports> <out>.
Kill point (identical for both engines): completed-groups.pt records policy_version==2 with
local_round_stats None (round-2 rollout generated, round not committed) -> +3s -> SIGKILL pgid.
A watcher logs every distinct completed-groups.pt state to cg-states.jsonl."""
import json, os, signal, subprocess, sys, time, threading, hashlib
from pathlib import Path
ENG = sys.argv[1]; MR = "/opt/miles-legacy" if ENG == "legacy" else "/opt/miles-next"
sys.path[:0] = ["/work/yeto", "/work/harness", MR]
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
import importlib.util
spec = importlib.util.spec_from_file_location("bench", "/work/yeto/scripts/benchmark_rl.py")
B = importlib.util.module_from_spec(spec); sys.modules["bench"] = B; spec.loader.exec_module(B)
from types import SimpleNamespace
from huggingface_hub import snapshot_download
from yeto.provenance import python_spec_sha256
import torch
MODEL, REV = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
model_path = Path(snapshot_download(repo_id=MODEL, revision=REV))
rounds, groups = 4, 4
rows = [json.loads(l) for l in open("/work/data/gsm8k.jsonl")][: groups * rounds]
prompts = out / "prompts.jsonl"
prompts.write_text("".join(json.dumps({**r, "metadata": {"benchmark_prompt_id": i}}) + "\n" for i, r in enumerate(rows)))
args = SimpleNamespace(model=MODEL, model_revision=REV, data="gsm8k", reward_function="gsm8k_reward:score",
    global_rounds=rounds, samples_per_group=8, over_sampling_batch_size=None, dynamic_sampling_filter_path=None,
    optimizer_steps=1, rollout_max_response_len=384, apply_chat_template_kwargs={"enable_thinking": False},
    pipeline_parallel=1, expert_parallel=1, lora_r=16, lora_targets="all-linear", inner_lr=1e-5, seq_len=1024,
    _active_seed=17, miles_port_base=21000, wan_streams=4, miles_root=Path(MR), trust_remote_code=True, rl_engine=ENG)
arm = B.Arm("kill", "single", 1, 1, 1, groups)
worker = B.WorkerSpec(0, 1, groups, prompts, True)
port = B._free_port()
payload = B.worker_payload(args, worker, arm=arm, run_dir=out, model_path=model_path, syncer=f"127.0.0.1:{port}",
                           reward_sha256=python_spec_sha256(args.reward_function))
payload["extra_argv"] = []
(out / "worker.json").write_text(json.dumps(payload, indent=1, default=str))
os.environ["PYTHONPATH"] = f"{MR}:/work/yeto:/work/harness:" + os.environ.get("PYTHONPATH", "")
cg = Path(payload["arguments"]["completed_groups_path"]); events = Path(payload["arguments"]["event_tape"])
def note(**kw):
    kw["t"] = time.time(); print(json.dumps(kw, default=str), flush=True)
    with open(out / "kill44.jsonl", "a") as f: f.write(json.dumps(kw, default=str) + "\n")
def cg_summary(tag):
    try: raw = cg.read_bytes(); p = torch.load(cg, map_location="cpu", weights_only=True)
    except Exception as e: return {"tag": tag, "error": repr(e)[:300]}
    grps = p.get("completed_groups") or []
    def samp(s):
        s = s[0] if isinstance(s, list) else s
        md = s.get("metadata") or {}
        return {"idx": s.get("index"), "gidx": s.get("group_index"), "pid": md.get("benchmark_prompt_id"),
                "status": s.get("status"), "reward": s.get("reward"),
                "tok": md.get("yeto_rl_policy_token") or md.get("policy_token") or s.get("weight_versions")}
    return {"tag": tag, "sha256": hashlib.sha256(raw).hexdigest(), "keys": sorted(p.keys()),
            "schema_version": p.get("schema_version"), "policy_version": p.get("policy_version"),
            "local_round_id": p.get("local_round_id"),
            "local_round_stats": None if p.get("local_round_stats") is None else {k: p["local_round_stats"].get(k) for k in ("local_round_id","base_policy_version","completed_groups","active_groups","completed_trajectories")},
            "rollout_metrics_n": len(p.get("rollout_metrics") or {}),
            "n_completed_groups": len(grps), "group_sizes": [len(g) for g in grps],
            "groups_first_sample": [samp(g[0]) for g in grps if g]}
stop = threading.Event(); state = {"last": None, "tag": "first"}
def watch():
    while not stop.is_set():
        try:
            st = cg.stat(); k = (st.st_mtime_ns, st.st_size)
            if k != state["last"]:
                state["last"] = k; s = cg_summary(state["tag"]); s["t"] = time.time()
                with open(out / "cg-states.jsonl", "a") as f: f.write(json.dumps(s, default=str) + "\n")
        except FileNotFoundError: pass
        time.sleep(0.3)
threading.Thread(target=watch, daemon=True).start()
import ray
ctx = ray.init(num_gpus=1, include_dashboard=True, logging_level="ERROR")
env = B._training_environment(ctx.address_info["address"], Path(MR)); env["KILL_MR"] = MR
sy = subprocess.Popen(B.syncer_command(arm, port, out, rounds=rounds), cwd="/work/yeto",
                      stdout=open(out / "syncer.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
B._wait_for_port(port, sy, out / "syncer.log")
def launch(tag):
    return subprocess.Popen([sys.executable, "/work/harness/worker2.py", str(out / "worker.json")], cwd="/work/yeto",
        stdout=open(out / f"miles-{tag}.log", "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
p = launch("first"); killed = False; t0 = time.time()
while p.poll() is None and time.time() - t0 < 2400:
    if cg.exists():
        s = cg_summary("poll")
        if s.get("policy_version") == 2 and s.get("local_round_stats") is None:
            time.sleep(3)
            ev = events.read_text().splitlines() if events.exists() else []
            before = cg_summary("at_kill")
            os.killpg(p.pid, signal.SIGKILL); killed = True
            note(step="kill", cg_at_kill=before, last_events=[json.loads(l).get("event") + ":" + str(json.loads(l).get("phase", "")) + ":" + str(json.loads(l).get("rollout_id", json.loads(l).get("policy_version", ""))) for l in ev[-4:]])
            break
    time.sleep(0.5)
p.wait(); note(step="first_learner_exit", rc=p.returncode, killed=killed)
if not killed:
    sy.kill(); sys.exit("learner exited before round 2")
time.sleep(20)
note(step="events_before_restart", n=len(events.read_text().splitlines()), cg=cg_summary("before_restart"))
state["tag"] = "restart"
p = launch("restart"); rc = p.wait(timeout=2400); note(step="restart_exit", rc=rc)
try: src = sy.wait(timeout=120)
except subprocess.TimeoutExpired: sy.kill(); src = "killed"
note(step="syncer_exit", rc=src, cg=cg_summary("final"))
time.sleep(1); stop.set(); ray.shutdown()
