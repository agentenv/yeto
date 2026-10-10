"""Generate the phase-1 G1 spec files (rl-algo-supplement 4.x/5.x) and dry-run each.

usage: python gen_specs.py <outdir>   (run from the yeto repo root)
Writes <outdir>/<case>.json, <case>.allow (one mechanism per line), and
dry-run.json (verdict, sha256, required mechanisms per case).
"""
import json, shlex, subprocess, sys, copy
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
PY = sys.executable
REF = {"source": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"}
BNZ = None  # filled from yeto
from yeto.rl.engine.algorithm import BOUNDED_NONZERO_STD_FILTER as BNZ  # noqa: E402

def dry(extra="", spec=None, allow=()):
    cmd = [PY, "-m", "yeto.rl.adapters.miles.algorithm_flags", "--dry-run", f"--extra={extra}"]
    if spec is not None:
        p = OUT / "_tmp.json"; p.write_text(json.dumps(spec)); cmd += ["--rl-algorithm-spec", str(p)]
    for a in allow:
        cmd += ["--rl-allow-unverified-mechanism", a]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)

def from_extra(extra):
    return dry(extra)["algorithm_spec"]

TIS = "--use-tis --tis-clip 1.01 --tis-clip-low 0.99"
CASES = {
    "base": from_extra(""),
    "cispo": from_extra("--policy-loss-variant cispo --eps-clip 0.2 --eps-clip-high 0.28 --calculate-per-token-loss"),
    "sapo": from_extra("--policy-loss-variant sapo --sapo-tau-pos 1.0 --sapo-tau-neg 1.05"),
    "gmpo": from_extra("--policy-loss-variant gmpo --gmpo-log-clip-low 0.4 --gmpo-log-clip-high 0.4"),
    "cispo-tis": from_extra("--policy-loss-variant cispo --eps-clip 0.2 --eps-clip-high 0.28 --calculate-per-token-loss " + TIS),
    "sapo-tis": from_extra("--policy-loss-variant sapo --sapo-tau-pos 1.0 --sapo-tau-neg 1.05 " + TIS),
    "gmpo-tis": from_extra("--policy-loss-variant gmpo --gmpo-log-clip-low 0.4 --gmpo-log-clip-high 0.4 " + TIS),
    "dual": from_extra("--eps-clip-c 1.01"),
    "nonorm": from_extra("--disable-rewards-normalization"),
    "whiten": from_extra("--normalize-advantages"),
    "clip-sym": from_extra("--eps-clip 0.001 --eps-clip-high 0.001"),
    "clip-hi": from_extra("--eps-clip 0.001 --eps-clip-high 10"),
}
for est in ("k1", "k2", "low_var_kl", "k3"):
    s = from_extra(f"--use-kl-loss --kl-loss-coef 0.01 --kl-loss-type {est}"); s["kl"]["ref_model"] = REF
    CASES[f"kl-{est.replace('_', '')}"] = s
s = from_extra("--use-kl-loss --kl-loss-coef 0.01 --kl-loss-type k3 --use-unbiased-kl"); s["kl"]["ref_model"] = REF
CASES["kl-unbiased"] = s
s = from_extra("--use-opsm --opsm-delta 1e-6")
s["correction"]["opsm_old_logprob_source"] = "rollout"; s["correction"]["use_rollout_logprobs"] = True
CASES["opsm-rollout"] = s
s = copy.deepcopy(CASES["dual"]); s["loss"]["eps_clip_c"] = None  # v2 shape, default loss
s["sampling"].update(filter=BNZ, max_replacements=4, over_sampling_batch_size=8)
CASES["os"] = s

report = {}
for name, spec in CASES.items():
    r0 = dry(spec=spec)
    req = r0.get("required_mechanisms") or []
    # allow exactly the mechanisms the engine rejects (never-declarable ones excluded by policy)
    base_req = set(dry(spec=CASES["base"])["required_mechanisms"])
    allow = []
    r = r0
    for _ in range(6):
        if r["verdict"] == "accepted":
            break
        err = r.get("error", "")
        import re
        m = re.findall(r"(\w+) mechanism '(\w+)' not supported", err)
        if not m:
            break
        dimmap = {"losses": "losses", "features": "features", "corrections": "corrections",
                  "kl_estimators": "kl_estimators", "kl_placements": "kl_placements"}
        for dim, mech in m:
            allow.append(f"{dimmap.get(dim, dim)}:{mech}")
        r = dry(spec=spec, allow=allow)
    (OUT / f"{name}.json").write_text(json.dumps(r.get("algorithm_spec", spec), sort_keys=True))
    (OUT / f"{name}.allow").write_text("".join(a + "\n" for a in allow))
    report[name] = {k: r.get(k) for k in ("verdict", "algorithm_spec_sha256", "required_mechanisms", "error", "launch_warnings")}
    report[name]["allow"] = allow
    print(name, r["verdict"], (r.get("algorithm_spec_sha256") or "")[:12], allow, (r.get("error") or "")[:160])
(OUT / "_tmp.json").unlink(missing_ok=True)
(OUT / "dry-run.json").write_text(json.dumps(report, indent=1, sort_keys=True))
