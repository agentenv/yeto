"""4.4: one strict island (syncer, learners=1), SIGKILL the learner during round 2, restart it."""
import json, os, signal, subprocess, sys, time
from pathlib import Path
sys.path[:0] = ["/work/yeto", "/work/harness", "/opt/miles-next"]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
import importlib.util
spec = importlib.util.spec_from_file_location("bench", "/work/yeto/scripts/benchmark_rl.py")
B = importlib.util.module_from_spec(spec); sys.modules["bench"] = B; spec.loader.exec_module(B)
from types import SimpleNamespace
from huggingface_hub import snapshot_download
from yeto.provenance import python_spec_sha256
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
    _active_seed=17, miles_port_base=21000, wan_streams=4, miles_root=Path("/opt/miles-next"), trust_remote_code=True, rl_engine="ports")
arm = B.Arm("kill", "single", 1, 1, 1, groups)
worker = B.WorkerSpec(0, 1, groups, prompts, True)
port = B._free_port()
payload = B.worker_payload(args, worker, arm=arm, run_dir=out, model_path=model_path, syncer=f"127.0.0.1:{port}",
                           reward_sha256=python_spec_sha256(args.reward_function))
payload["extra_argv"] = []
(out / "worker.json").write_text(json.dumps(payload, indent=1, default=str))
os.environ["PYTHONPATH"] = "/opt/miles-next:/work/yeto:/work/harness:" + os.environ.get("PYTHONPATH", "")
import ray
ctx = ray.init(num_gpus=1, include_dashboard=True, logging_level="ERROR")
env = B._training_environment(ctx.address_info["address"], Path("/opt/miles-next"))
env.update(PROBE_MODE="none", PROBE_OUT=str(out / "probe.jsonl"))
sy = subprocess.Popen(B.syncer_command(arm, port, out, rounds=rounds), cwd="/work/yeto",
                      stdout=open(out / "syncer.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
B._wait_for_port(port, sy, out / "syncer.log")
events = out / "island-0" / "events.jsonl"
def launch(tag):
    return subprocess.Popen([sys.executable, "/work/harness/worker.py", str(out / "worker.json")], cwd="/work/yeto",
        stdout=open(out / f"miles-{tag}.log", "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
def note(**kw):
    kw["t"] = time.time(); print(json.dumps(kw), flush=True)
    with open(out / "kill44.jsonl", "a") as f: f.write(json.dumps(kw) + "\n")
p = launch("first"); killed = False; t0 = time.time()
while p.poll() is None and time.time() - t0 < 2400:
    if events.exists():
        for l in events.read_text().splitlines():
            r = json.loads(l)
            if r.get("event") == "rl_driver_phase" and r.get("phase") == "train" and r.get("rollout_id") == 2:
                os.killpg(p.pid, signal.SIGKILL); killed = True; break
    if killed: break
    time.sleep(1)
p.wait(); note(step="first_learner_exit", rc=p.returncode, killed=killed)
if not killed:
    sy.kill(); sys.exit("learner exited before round 2")
time.sleep(20)  # let Ray reap the dead job's actors / free the GPU
note(step="events_before_restart", n=len(events.read_text().splitlines()))
p = launch("restart"); rc = p.wait(timeout=2400); note(step="restart_exit", rc=rc)
try: src = sy.wait(timeout=120)
except subprocess.TimeoutExpired: sy.kill(); src = "killed"
note(step="syncer_exit", rc=src)
ray.shutdown()
