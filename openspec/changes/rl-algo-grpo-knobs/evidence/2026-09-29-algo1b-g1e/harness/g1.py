"""algo1b G1: one mechanism, 1 GPU, single island, no outer sync (--rl-single-island-no-sync),
3 rounds. usage: g1.py <mechanism> <outdir>   (plan.md)"""
import json, os, sys, subprocess, time, threading
from pathlib import Path
sys.path[:0] = ["/work/yeto", "/work/harness", "/opt/miles-next"]
mech, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
MODEL, REV = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
ROUNDS, GROUPS, SPG, RESP = 3, 4, 8, 384
import importlib.util
spec_ = importlib.util.spec_from_file_location("bench", "/work/yeto/scripts/benchmark_rl.py")
B = importlib.util.module_from_spec(spec_); sys.modules["bench"] = B; spec_.loader.exec_module(B)
from huggingface_hub import snapshot_download
from yeto.rl.algos import grpo_knobs as gk, reward_pipeline as rp
from yeto.rl.engine.algorithm import AlgorithmSpec, PluginRef, BOUNDED_NONZERO_STD_FILTER, MECHANISM_DIMENSIONS
from yeto.rl.engine.miles_adapter.entry import miles_capabilities

model_path = Path(snapshot_download(repo_id=MODEL, revision=REV))
over = None; flt = None; repl = None
D = rp.dispatcher_ref().to_dict()
specs = {
    "baseline": {},
    "clip_higher": {"loss": {"eps_clip": 0.2, "eps_clip_high": 0.28}},
    "dual_clip": {"loss": {"eps_clip_c": 3.0}},
    "token": {"loss": {"aggregation": "token"}},
    "drgrpo": {"advantage": {"std_normalization": False},
               "loss": {"aggregation": "constant", "constant_denominator": float(RESP),
                        "reducer": PluginRef.from_path(gk.REDUCER_PATH).to_dict()}},
    "kl_k3": {"kl": {"placement": "loss", "coef": 0.001, "estimator": "k3",
                     "ref_model": {"source": MODEL, "revision": REV}}},
    "entropy": {"entropy_coef": 0.001},
    "no_std": {"advantage": {"std_normalization": False}},
    "clip_sym": {"loss": {"eps_clip": 0.001}},
    "clip_hi": {"loss": {"eps_clip": 0.001, "eps_clip_high": 10.0}},
    "over_sampling": {"sampling": {"filter": BOUNDED_NONZERO_STD_FILTER, "max_replacements": 2,
                                   "over_sampling_batch_size": 8}},
    "overlong_penalty": {"advantage": {"reward_postprocess": D, "reward_shapers": [
        {"name": "overlong_penalty", "max_length": RESP, "cache_length": 128}]}},
}
payload_spec = specs[mech]
spec = gk.with_pipeline_plugins(AlgorithmSpec(**payload_spec)) if payload_spec else AlgorithmSpec()
assert spec.rejections() == [], spec.rejections()
if mech == "over_sampling":
    over, flt, repl = 8, BOUNDED_NONZERO_STD_FILTER, 2
caps = miles_capabilities("sha256:" + "0" * 64)
allow = sorted(f"{d}:{n}" for d, n in spec.required_mechanisms()
               if n not in getattr(caps, d, frozenset()))
spec_path = out / "algorithm_spec.json"; spec_path.write_text(spec.canonical_json())
(out / "g1_meta.json").write_text(json.dumps({"mechanism": mech, "algorithm_spec_sha256": spec.sha256(),
    "allow_unverified": allow, "rounds": ROUNDS}, indent=1))

prompts = out / "prompts.jsonl"
import datasets
ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
rows = [{"messages": [{"role": "user", "content": r["question"] + "\nPut the final numeric answer in \\boxed{}."}],
         "label": r["answer"].split("####")[-1].strip(), "metadata": {"benchmark_prompt_id": i}}
        for i, r in enumerate(ds.select(range(GROUPS * ROUNDS * 3)))]
prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
args = __import__("types").SimpleNamespace(model=MODEL, model_revision=REV, data="openai/gsm8k",
    reward_function="gsm8k_reward:score", global_rounds=ROUNDS, samples_per_group=SPG,
    over_sampling_batch_size=over, dynamic_sampling_filter_path=flt, dynamic_sampling_max_replacements=repl,
    optimizer_steps=int(os.environ.get("OPT_STEPS", 1)), rollout_max_response_len=RESP, apply_chat_template_kwargs={"enable_thinking": False},
    pipeline_parallel=1, expert_parallel=1, lora_r=16, lora_targets="all-linear", inner_lr=float(os.environ.get("INNER_LR", 1e-5)), seq_len=1024,
    _active_seed=17, miles_port_base=20000, wan_streams=4, miles_root=Path("/opt/miles-next"),
    trust_remote_code=False, rl_engine="ports")
arm = B.Arm("g1", "single", 1, 1, 1, GROUPS)
worker = B.WorkerSpec(0, 1, GROUPS, prompts, False)
from yeto.provenance import python_spec_sha256
payload = B.worker_payload(args, worker, arm=arm, run_dir=out, model_path=model_path, syncer=None,
                           reward_sha256=python_spec_sha256(args.reward_function))
payload["arguments"].update(rl_algorithm_spec=str(spec_path), rl_allow_unverified_mechanism=allow,
                            rl_single_island_no_sync=True, rl_expected_algorithm_sha256=spec.sha256(),
                            dynamic_sampling_max_replacements=repl)
(out / "worker.json").write_text(json.dumps(payload, indent=1, default=str))

peak = {"mib": 0}
stop = threading.Event()
def sample():
    while not stop.is_set():
        try:
            v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20).stdout.split()
            peak["mib"] = max(peak["mib"], max(int(x) for x in v))
        except Exception:
            pass
        stop.wait(2)
threading.Thread(target=sample, daemon=True).start()
import ray
os.environ["PYTHONPATH"] = "/opt/miles-next:/work/yeto:/work/harness:" + os.environ.get("PYTHONPATH", "")
ctx = ray.init(num_gpus=1, include_dashboard=True, logging_level="ERROR")
env = B._training_environment(ctx.address_info["address"], Path("/opt/miles-next"))
t0 = time.time()
with open(out / "miles.log", "w") as h:
    rc = subprocess.call([sys.executable, "/work/harness/worker.py", str(out / "worker.json")], cwd="/work/yeto",
                         stdout=h, stderr=subprocess.STDOUT, env=env, timeout=1700)
stop.set()
res = {"mechanism": mech, "rc": rc, "seconds": round(time.time() - t0, 1), "peak_gpu_mib": peak["mib"]}
(out / "g1_result.json").write_text(json.dumps(res))
print("G1_RESULT", json.dumps(res))
ray.shutdown()
