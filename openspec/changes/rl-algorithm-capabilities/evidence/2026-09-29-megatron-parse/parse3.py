"""2.6 attempt 3. Mode 'local' (yeto venv): write argv per case. Mode 'remote'
(ports image): write argv per case and run upstream parse_miles_args on it.

The argv is built by the learner's own path: learner.parse_args (with
--rl-algorithm-spec <case file>) -> learner.build_ports_launch(args, run_config).
run_config is the tests' make_config fixture (the real one needs a Megatron
model bridge); hf_checkpoint/ref_load/prompt paths are fixed to /tmp/algocap26.
"""
import dataclasses, json, os, pathlib, subprocess, sys, traceback

ROOT = pathlib.Path(sys.argv[2])
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(pathlib.Path(__file__).parent)]
from cases3 import cases  # noqa: E402
from test_rl_engine_selection import _learner_argv  # noqa: E402
from test_rl_miles_adapter_config import _TINY_QWEN3, make_config, sub  # noqa: E402
from yeto.rl import learner  # noqa: E402

mode, out = sys.argv[1], pathlib.Path(sys.argv[3])
tmp = pathlib.Path("/tmp/algocap26")
tmp.mkdir(exist_ok=True)
(tmp / "config.json").write_text(json.dumps(_TINY_QWEN3))
(tmp / "p.jsonl").write_text('{"messages":[{"role":"user","content":"hi"}],"label":"x"}\n')
base = dataclasses.replace(make_config(), hf_checkpoint=str(tmp), ref_load=str(tmp))
base = sub(base, "data", prompt_path=str(tmp / "p.jsonl"))
base = sub(base, "trainable", target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
records = []
for name, spec, expected in cases():
    rec = {"case": name, "spec_sha256": spec.sha256()}
    try:
        spec_file = tmp / f"{name}.json"
        spec_file.write_text(spec.canonical_json())
        args = learner.parse_args(_learner_argv(("--rl-algorithm-spec", str(spec_file))))
        cfg = base
        if spec.sampling.over_sampling_batch_size is not None:
            cfg = sub(cfg, "batch", over_sampling_batch_size=spec.sampling.over_sampling_batch_size)
        launch = learner.build_ports_launch(args, cfg)
        rec["argv"] = list(launch.argv)
        rec["algorithm_sha256"] = launch.algorithm_sha256
        if mode == "remote":
            from yeto.rl.engine.miles_adapter.config import parse_miles_args
            ns = parse_miles_args(launch)
            bad = {}
            for k, v in expected.items():
                got = getattr(ns, k)
                ok = abs(got - v) <= 1e-12 * max(1, abs(v)) if isinstance(v, float) else got == v
                if not ok:
                    bad[k] = [repr(got), repr(v)]
            rec["mismatch"], rec["ok"] = bad, not bad
        else:
            rec["ok"] = True
    except BaseException as exc:  # noqa: BLE001
        rec["ok"], rec["error"] = False, "".join(traceback.format_exception(exc))[-3000:]
    records.append(rec)
    print(f"{name}: {'OK' if rec['ok'] else 'FAIL'} {rec.get('mismatch') or rec.get('error', '')[-300:]}", flush=True)
out.write_text(json.dumps(records, indent=1, sort_keys=True))
print("ALL_OK" if all(r["ok"] for r in records) else "SOME_FAILED", sum(r["ok"] for r in records), "/", len(records))
