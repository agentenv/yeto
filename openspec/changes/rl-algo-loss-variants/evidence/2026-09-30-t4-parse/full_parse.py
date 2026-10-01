"""ALGO-2b-T4 (4.4): inside the pinned image, translate CISPO/SAPO/GMPO specs with
yeto (translate_run_config -> launch.argv, saved verbatim) and run the full upstream
Miles parse_args (Miles + Megatron halves, incl. Miles validate_args) followed by
yeto validate_parsed_args (mc.parse_miles_args). Then check rejected combinations:
yeto-side (AlgorithmSpec.rejections) and Miles-side (full parse_args on the valid
argv with one offending flag appended must raise). No training, no weights.
usage: python full_parse.py <out.json>"""
import dataclasses, json, subprocess, sys, tempfile, traceback
from pathlib import Path
sys.path.insert(0, "/yeto/tests")
from test_rl_miles_adapter_config import _TINY_QWEN3, make_config, sub
from yeto.rl.engine.algorithm import AlgorithmSpec, load_extensions
from yeto.rl.engine.miles_adapter import config as mc
load_extensions()
import miles.utils.arguments as A
root = A.__file__.split("/miles/utils/")[0]
res = {"miles_arguments_file": A.__file__}
try:
    res["miles_git_head"] = subprocess.check_output(["git", f"--git-dir={root}/.git", "rev-parse", "HEAD"], text=True).strip()
except Exception as e:
    res["miles_git_head"] = f"ERR {e!r}"
import megatron.training
res["megatron_training_file"] = megatron.training.__file__
V2 = "yeto-rl-algorithm-spec-v2"
VALID = {
    "cispo": ({"policy_loss_variant": "cispo", "eps_clip": 0.2, "eps_clip_high": 0.28, "aggregation": "token"},
              {"policy_loss_variant": "cispo", "eps_clip": 0.2, "eps_clip_high": 0.28, "calculate_per_token_loss": True}),
    "sapo": ({"policy_loss_variant": "sapo", "sapo_tau_pos": 0.9, "sapo_tau_neg": 1.2},
             {"policy_loss_variant": "sapo", "sapo_tau_pos": 0.9, "sapo_tau_neg": 1.2, "calculate_per_token_loss": False}),
    "gmpo": ({"policy_loss_variant": "gmpo", "gmpo_log_clip_low": 0.3, "gmpo_log_clip_high": 0.5},
             {"policy_loss_variant": "gmpo", "gmpo_log_clip_low": 0.3, "gmpo_log_clip_high": 0.5, "calculate_per_token_loss": False}),
}
YETO_REJECT = {
    "gmpo+token": {"policy_loss_variant": "gmpo", "aggregation": "token"},
    "cispo+default_aggregation": {"policy_loss_variant": "cispo", "eps_clip": 0.2, "eps_clip_high": 0.28},
    "cispo_missing_eps": {"policy_loss_variant": "cispo", "aggregation": "token"},
}
MILES_REJECT = {  # base variant -> appended flags
    "gmpo+--calculate-per-token-loss": ("gmpo", ["--calculate-per-token-loss"]),
    "sapo+--advantage-estimator gspo": ("sapo", ["--advantage-estimator", "gspo"]),
    "cispo+--eps-clip-c 3": ("cispo", ["--eps-clip-c", "3.0"]),
    "sapo+--sapo-tau-pos 0": ("sapo", ["--sapo-tau-pos", "0"]),
    "gmpo+--gmpo-log-clip-low -0.1": ("gmpo", ["--gmpo-log-clip-low=-0.1"]),
}

def cfg_for(tmp):
    (tmp / "config.json").write_text(json.dumps(_TINY_QWEN3))
    (tmp / "p.jsonl").write_text('{"messages":[{"role":"user","content":"hi"}],"label":"x"}\n')
    cfg = dataclasses.replace(make_config(), hf_checkpoint=str(tmp), ref_load=str(tmp))
    cfg = sub(cfg, "data", prompt_path=str(tmp / "p.jsonl"))
    return sub(cfg, "trainable", target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))

launches, ok = {}, True
res["valid"] = {}
for name, (loss, expect) in VALID.items():
    tmp = Path(tempfile.mkdtemp())
    spec = AlgorithmSpec.from_dict({"schema": V2, "loss": loss})
    r = {"loss": loss, "yeto_rejections": spec.rejections(), "sha256": spec.sha256()}
    try:
        launch = mc.translate_run_config(cfg_for(tmp), spec)
        launches[name] = launch
        r["argv"] = list(launch.argv)
        args = mc.parse_miles_args(launch)  # parse_args (Miles+Megatron, validate_args) + validate_parsed_args
        r["parsed"] = {k: getattr(args, k, "<missing>") for k in expect}
        r["mismatch"] = {k: [r["parsed"][k], v] for k, v in expect.items() if r["parsed"][k] != v}
        r["verdict"] = "PASS" if not r["mismatch"] and not r["yeto_rejections"] else "FAIL"
    except BaseException as e:
        r["verdict"], r["error"] = "FAIL", traceback.format_exc()[-3000:]
    ok &= r["verdict"] == "PASS"
    res["valid"][name] = r
    print("VALID", name, r["verdict"], r.get("mismatch"), flush=True)

res["yeto_reject"] = {}
for name, loss in YETO_REJECT.items():
    spec = AlgorithmSpec.from_dict({"schema": V2, "loss": loss})
    rej = spec.rejections()
    res["yeto_reject"][name] = {"loss": loss, "rejections": rej, "verdict": "PASS" if rej else "FAIL"}
    ok &= bool(rej)
    print("YETO_REJECT", name, "PASS" if rej else "FAIL", rej, flush=True)

res["miles_reject"] = {}
for name, (base, extra) in MILES_REJECT.items():
    launch = dataclasses.replace(launches[base], argv=tuple(launches[base].argv) + tuple(extra)) \
        if base in launches else None
    r = {"base": base, "appended": extra}
    if launch is None:
        r["verdict"] = "FAIL"; r["error"] = "base launch missing"
    else:
        r["argv"] = list(launch.argv)
        try:
            mc.parse_miles_args(launch)
            r["verdict"] = "FAIL"; r["error"] = "accepted (expected rejection)"
        except BaseException as e:
            r["verdict"] = "PASS"; r["raised"] = f"{type(e).__name__}: {e}"
    ok &= r["verdict"] == "PASS"
    res["miles_reject"][name] = r
    print("MILES_REJECT", name, r["verdict"], r.get("raised", r.get("error")), flush=True)

res["miles_commit_ok"] = res["miles_git_head"].startswith("5c1b49eb")
ok &= res["miles_commit_ok"]
res["overall"] = "PASS" if ok else "FAIL"
Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=repr))
print("MILES_HEAD", res["miles_git_head"], "OVERALL", res["overall"], flush=True)
