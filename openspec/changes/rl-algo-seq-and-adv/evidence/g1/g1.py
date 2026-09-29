"""rl-algo-seq-and-adv G1: one single-island ports run (1 GPU, no outer sync).

usage: g1.py <run-name> <outdir>   (inside the ports image, yeto at /work/yeto)
Derived from rl-engine-ports evidence/2026-09-29-rerun-harness/harness/smoke.py.
"""
import json, os, sys, subprocess, time
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = ["/work/yeto", "/work/g1", "/root/miles"]
name, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
EX = "/work/yeto/openspec/changes/rl-algo-seq-and-adv/examples"
A = ["advantage_estimators:gspo", "features:eps_clip", "features:clip_higher"]
RUNS = {  # spec, allowance, reward function, optimizer steps
    "gspo_s2": (f"{EX}/gspo.json", A, "gsm8k_reward:score", 2),
    "gspo_s1": (f"{EX}/gspo.json", A, "gsm8k_reward:score", 1),
    "rpp": (f"{EX}/rpp.json", ["advantage_estimators:reinforce_plus_plus", "features:whiten_advantages"], "gsm8k_reward:score", 1),
    "rpp_baseline": (f"{EX}/rpp_baseline.json", ["advantage_estimators:reinforce_plus_plus_baseline", "features:whiten_advantages"],
                     "gsm8k_reward:score", 1),
    "maxrl": (f"{EX}/maxrl.json", ["features:maxrl", "reward_postprocessors:custom_reward_postprocess", "features:plugins"], "gsm8k_reward:score", 1),
    "mapo": (f"{EX}/mapo.json", ["features:mapo", "reward_postprocessors:custom_reward_postprocess", "features:plugins"], "gsm8k_reward:score", 1),
    "gdpo": (f"{EX}/gdpo.json", ["features:gdpo", "reward_postprocessors:custom_reward_postprocess", "features:plugins"],
             "yeto.rl.algos.gdpo_reward:reward_func", 1),
}
spec_path, allow, reward_fn, opt_steps = RUNS[name]
MODEL, REV = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
import importlib.util
bspec = importlib.util.spec_from_file_location("bench", "/work/yeto/scripts/benchmark_rl.py")
B = importlib.util.module_from_spec(bspec); sys.modules["bench"] = B; bspec.loader.exec_module(B)
from huggingface_hub import snapshot_download
model_path = Path(snapshot_download(repo_id=MODEL, revision=REV))
rounds, groups, spg = 3, 4, 8
prompts = Path("/work/data/prompts.jsonl")  # same 12 prompts for every run
args = SimpleNamespace(model=MODEL, model_revision=REV, data="openai/gsm8k", reward_function=reward_fn,
    global_rounds=rounds, samples_per_group=spg, over_sampling_batch_size=None, dynamic_sampling_filter_path=None,
    optimizer_steps=opt_steps, rollout_max_response_len=384, apply_chat_template_kwargs={"enable_thinking": False},
    pipeline_parallel=1, expert_parallel=1, lora_r=16, lora_targets="all-linear", inner_lr=1e-5, seq_len=1024,
    _active_seed=17, miles_port_base=20000, wan_streams=4, miles_root=Path("/root/miles"), trust_remote_code=False,
    rl_engine="ports")
arm = B.Arm("g1", "single", 1, 1, 1, groups)
worker = B.WorkerSpec(0, 1, groups, prompts, False)
from yeto.provenance import python_spec_sha256
payload = B.worker_payload(args, worker, arm=arm, run_dir=out, model_path=model_path, syncer=None,
                           reward_sha256=python_spec_sha256(reward_fn))
payload["arguments"]["rl_algorithm_spec"] = spec_path
payload["arguments"]["rl_allow_unverified_mechanism"] = list(allow)
payload["extra_argv"] = ["--save-debug-rollout-data", str(out / "rollouts" / "{rollout_id}.pt")]
(out / "worker.json").write_text(json.dumps(payload, indent=1, default=str))
import ray
os.environ["PYTHONPATH"] = "/root/miles:/work/yeto:/work/g1:" + os.environ.get("PYTHONPATH", "")
ctx = ray.init(num_gpus=1, include_dashboard=True, logging_level="ERROR")
env = B._training_environment(ctx.address_info["address"], Path("/root/miles"))
env["PYTHONPATH"] = env["PYTHONPATH"] + ":/work/g1"
t0 = time.time()
with open(out / "miles.log", "w") as h:
    rc = subprocess.call([sys.executable, "-c",
                          "import sys;sys.path[:0]=['/root/miles','/work/yeto','/work/g1'];"
                          "import importlib.util as u;s=u.spec_from_file_location('bench','/work/yeto/scripts/benchmark_rl.py');"
                          "B=u.module_from_spec(s);sys.modules['bench']=B;s.loader.exec_module(B);"
                          "from pathlib import Path;sys.exit(B.run_training_worker(Path(sys.argv[1])))",
                          str(out / "worker.json")],
                         cwd="/work/yeto", stdout=h, stderr=subprocess.STDOUT, env=env, timeout=1500)
print("worker rc", rc, "seconds", round(time.time() - t0), flush=True)
ray.shutdown()
sys.exit(rc)
