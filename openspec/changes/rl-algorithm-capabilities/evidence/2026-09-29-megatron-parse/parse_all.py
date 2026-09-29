"""Runs inside the ports image: full ports argv -> upstream parse_miles_args (megatron)."""
import dataclasses, json, sys, tempfile, traceback, pathlib
sys.path[:0] = ["/work/yeto", "/work/yeto/tests"]
from test_rl_miles_adapter_config import make_config, sub, _TINY_QWEN3
from test_rl_algorithm_flags_upstream import CASES
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.miles_adapter import config as mc

tmp = pathlib.Path(tempfile.mkdtemp())
(tmp / "config.json").write_text(json.dumps(_TINY_QWEN3))
(tmp / "p.jsonl").write_text('{"messages":[{"role":"user","content":"hi"}],"label":"x"}\n')
base = dataclasses.replace(make_config(), hf_checkpoint=str(tmp), ref_load=str(tmp))
base = sub(base, "data", prompt_path=str(tmp / "p.jsonl"))
base = sub(base, "trainable", target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
results = []
for idx, (change, expected) in enumerate([({}, {"advantage_estimator": "grpo"})] + list(CASES)):
    spec = AlgorithmSpec(**change)
    cfg = sub(base, "algorithm", advantage_estimator=spec.advantage_estimator)
    over = spec.sampling.over_sampling_batch_size
    if over is not None:
        cfg = sub(cfg, "batch", over_sampling_batch_size=over)
    rec = {"case": idx, "spec_sha256": spec.sha256()}
    try:
        launch = mc.translate_run_config(cfg, spec)
        rec["argv"] = list(launch.argv)
        args = mc.parse_miles_args(launch)
        bad = {}
        for k, v in expected.items():
            got = getattr(args, k)
            ok = abs(got - v) <= 1e-12 * max(1, abs(v)) if isinstance(v, float) else got == v
            if not ok:
                bad[k] = [repr(got), repr(v)]
        rec["mismatch"] = bad
        rec["ok"] = not bad
    except BaseException as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = "".join(traceback.format_exception(exc))[-3000:]
    results.append(rec)
    print(f"case {idx}: {'OK' if rec['ok'] else 'FAIL'} {rec.get('mismatch') or rec.get('error','')[-300:]}", flush=True)
json.dump(results, open("/work/results.json", "w"), indent=1)
print("ALL_OK" if all(r["ok"] for r in results) else "SOME_FAILED", sum(r["ok"] for r in results), "/", len(results))
