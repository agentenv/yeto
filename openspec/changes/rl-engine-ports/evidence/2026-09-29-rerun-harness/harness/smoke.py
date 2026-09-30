"""Single-island ports smoke: 1 GPU, no outer sync, 3 rounds. usage: smoke.py <mode> <outdir>"""
import json, os, sys, subprocess, time
from pathlib import Path
from types import SimpleNamespace
sys.path[:0] = ["/work/yeto", "/work/harness", "/opt/miles-next"]
mode, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
MODEL, REV = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
import importlib.util
spec = importlib.util.spec_from_file_location("bench", "/work/yeto/scripts/benchmark_rl.py")
B = importlib.util.module_from_spec(spec); sys.modules["bench"] = B; spec.loader.exec_module(B)
from yeto.provenance import file_sha256
from huggingface_hub import snapshot_download
model_path = Path(snapshot_download(repo_id=MODEL, revision=REV))
rounds, groups, spg = int(os.environ.get("ROUNDS", 3)), 4, 8
prompts = out / "prompts.jsonl"
if not prompts.exists():
    import datasets
    ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
    rows = [{"messages": [{"role": "user", "content": r["question"] + "\nPut the final numeric answer in \\boxed{}."}],
             "label": r["answer"].split("####")[-1].strip(), "metadata": {"benchmark_prompt_id": i}}
            for i, r in enumerate(ds.select(range(groups * rounds)))]
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
args = SimpleNamespace(model=MODEL, model_revision=REV, data="openai/gsm8k", reward_function="gsm8k_reward:score",
    global_rounds=rounds, samples_per_group=spg, over_sampling_batch_size=None, dynamic_sampling_filter_path=None,
    optimizer_steps=1, rollout_max_response_len=int(os.environ.get("RESP", 384)), apply_chat_template_kwargs={"enable_thinking": False},
    pipeline_parallel=1, expert_parallel=1, lora_r=16, lora_targets="all-linear", inner_lr=1e-5, seq_len=1024,
    _active_seed=17, miles_port_base=20000, wan_streams=4, miles_root=Path("/opt/miles-next"), trust_remote_code=False,
    rl_engine="ports")
arm = B.Arm("smoke", "single", 1, 1, 1, groups)
worker = B.WorkerSpec(0, 1, groups, prompts, False)
from yeto.provenance import python_spec_sha256
reward_sha = python_spec_sha256(args.reward_function)
payload = B.worker_payload(args, worker, arm=arm, run_dir=out, model_path=model_path, syncer=None, reward_sha256=reward_sha)
payload["extra_argv"] = ["--save-debug-rollout-data", str(out / "rollouts" / "{rollout_id}.pt")] + os.environ.get("EXTRA", "").split()
payload["extra_argv"] = [a for a in payload["extra_argv"] if a]
(out / "worker.json").write_text(json.dumps(payload, indent=1, default=str))
import ray
os.environ["PYTHONPATH"] = "/opt/miles-next:/work/yeto:/work/harness:" + os.environ.get("PYTHONPATH", "")
ctx = ray.init(num_gpus=int(os.environ.get("NGPU", 1)), include_dashboard=True, logging_level="ERROR")
env = B._training_environment(ctx.address_info["address"], Path("/opt/miles-next"))
env.update(PROBE_MODE=mode, PROBE_OUT=str(out / "probe.jsonl"))
t0 = time.time()
with open(out / "miles.log", "w") as h:
    rc = subprocess.call([sys.executable, "/work/harness/worker.py", str(out / "worker.json")], cwd="/work/yeto", stdout=h, stderr=subprocess.STDOUT, env=env, timeout=int(os.environ.get("TMO", 3000)))
print("worker rc", rc, "seconds", round(time.time() - t0))
ray.shutdown()
