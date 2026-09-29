"""One G1/G2 run inside the image: g1.py <run-name> <outdir>  (env ROUNDS, TMO)."""
import json, os, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
sys.path[:0] = ["/work/yeto", "/work/harness", "/root/miles"]
name, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
from yeto.rl.algos import mismatch_correction as mc
from yeto.rl.engine.algorithm import AlgorithmSpec

MIS = lambda **c: {"method": "custom", "function": mc.mis_ref().to_dict(), "mis_level": "token", **c}
RUNS = {
    "observe": ({"method": "custom", "function": mc.observe_ref().to_dict(), "mismatch_metrics": True},
                ["custom", "mismatch_observe", "mismatch_metrics"]),
    "tis": ({"method": "tis", "tis_clip": 2.0, "tis_clip_low": 0.0}, ["tis"]),
    "icepop": ({"method": "custom", "function": mc.icepop_ref().to_dict(), "tis_clip": 5.0,
                "tis_clip_low": 0.5, "mismatch_metrics": True}, ["custom", "icepop", "mismatch_metrics"]),
    "opsm-trainer": ({"method": "opsm", "opsm_delta": 1e-4, "opsm_old_logprob_source": "trainer"},
                     ["opsm", "opsm_trainer"]),
    "mis": (MIS(mis_mode="truncate", mis_upper_bound=2.0), ["custom", "mis"]),
    "mis-mask": (MIS(mis_mode="mask", mis_lower_bound=0.5, mis_upper_bound=2.0), ["custom", "mis_mask"]),
}
correction, allow = RUNS[name]
spec = AlgorithmSpec(correction=correction)
assert spec.rejections() == [], spec.rejections()
spec.verify_plugins()
(out / "spec.json").write_text(spec.canonical_json())
from yeto.rl.engine.miles_adapter.algorithm_flags import algorithm_argv
(out / "expected.json").write_text(json.dumps({"sha256": spec.sha256(), "allow": sorted(allow),
                                               "argv": algorithm_argv(spec)}))

import importlib.util
bs = importlib.util.spec_from_file_location("bench", "/work/yeto/scripts/benchmark_rl.py")
B = importlib.util.module_from_spec(bs); sys.modules["bench"] = B; bs.loader.exec_module(B)
from huggingface_hub import snapshot_download
from yeto.provenance import python_spec_sha256
MODEL, REV = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
model_path = Path(snapshot_download(repo_id=MODEL, revision=REV))
rounds, groups, spg = int(os.environ.get("ROUNDS", 3)), 4, 8
prompts = out / "prompts.jsonl"
import datasets
ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
rows = [{"messages": [{"role": "user", "content": r["question"] + "\nPut the final numeric answer in \\boxed{}."}],
         "label": r["answer"].split("####")[-1].strip(), "metadata": {"benchmark_prompt_id": i}}
        for i, r in enumerate(ds.select(range(groups * rounds)))]
prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
args = SimpleNamespace(model=MODEL, model_revision=REV, data="openai/gsm8k", reward_function="gsm8k_reward:score",
    global_rounds=rounds, samples_per_group=spg, over_sampling_batch_size=None, dynamic_sampling_filter_path=None,
    optimizer_steps=1, rollout_max_response_len=384, apply_chat_template_kwargs={"enable_thinking": False},
    pipeline_parallel=1, expert_parallel=1, lora_r=16, lora_targets="all-linear", inner_lr=1e-5, seq_len=1024,
    _active_seed=17, miles_port_base=20000, wan_streams=4, miles_root=Path("/root/miles"), trust_remote_code=False,
    rl_engine="ports")
arm = B.Arm("g1", "single", 1, 1, 1, groups)
worker = B.WorkerSpec(0, 1, groups, prompts, False)
payload = B.worker_payload(args, worker, arm=arm, run_dir=out, model_path=model_path, syncer=None,
                           reward_sha256=python_spec_sha256(args.reward_function))
payload["arguments"].update(rl_algorithm_spec=str(out / "spec.json"), rl_allow_unverified_mechanism=sorted(allow),
                            rl_expected_algorithm_sha256=spec.sha256())
(out / "worker.json").write_text(json.dumps(payload, indent=1, default=str))
import ray
os.environ["PYTHONPATH"] = "/root/miles:/work/yeto:/work/harness:" + os.environ.get("PYTHONPATH", "")
ctx = ray.init(num_gpus=1, include_dashboard=True, logging_level="ERROR")
env = B._training_environment(ctx.address_info["address"], Path("/root/miles"))
t0 = time.time()
with open(out / "miles.log", "w") as h:
    try:
        rc = subprocess.call([sys.executable, "-c", "import sys; sys.path[:0]=['/root/miles','/work/yeto','/work/harness'];"
                              "import importlib.util as u; s=u.spec_from_file_location('bench','/work/yeto/scripts/benchmark_rl.py');"
                              "B=u.module_from_spec(s); sys.modules['bench']=B; s.loader.exec_module(B);"
                              "from pathlib import Path; sys.exit(B.run_training_worker(Path(sys.argv[1])))",
                              str(out / "worker.json")], cwd="/work/yeto", stdout=h, stderr=subprocess.STDOUT,
                             env=env, timeout=int(os.environ.get("TMO", 1500)))
    except subprocess.TimeoutExpired:
        rc = "timeout"
(out / "rc").write_text(f"{rc}\n")
print("run", name, "rc", rc, "seconds", round(time.time() - t0), flush=True)
ray.shutdown()
