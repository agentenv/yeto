"""G2 report (task 7.4): report_g2.py <g2-rundir> -> metrics.jsonl + report.md (regenerable)."""
import ast, json, re, statistics, sys
from pathlib import Path
d = Path(sys.argv[1])
log = (d / "miles.log").read_text(errors="replace")
steps = {int(m.group(1)): ast.literal_eval(m.group(2))
         for m in re.finditer(r"log_utils\.py:\d+ - step (\d+): (\{.*\})", log)}
keys = ["train_rollout_kl", "train_rollout_logprob_abs_diff", "tis_abs", "tis_abs_p50", "tis_abs_p90",
        "tis_abs_p99", "tis", "ois", "ess_ratio", "mismatch_outside_0p5_5", "grad_norm", "pg_loss"]
rows = [{"step": s, **{k: steps[s].get(f"train/{k}") for k in keys}} for s in sorted(steps)]
(d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
def q(values, p):
    v = sorted(values); i = (len(v) - 1) * p; lo = int(i); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)
lines = ["# G2 observe-only mismatch report (data only; no recommendation)", "",
         "Execution mode: colocated-serial, single island, LoRA publish over CUDA IPC. Model Qwen/Qwen3-0.6B",
         "@c1899de, LoRA r16 all-linear, gsm8k, 4x8 samples/round, 1 optimizer step/round, bf16, 1x H100.",
         "Not to be extrapolated to the partitioned mode (NCCL broadcast), larger models or MoE.",
         "Per-round values are Miles' logged means; tis_abs_pXX are per-micro-batch token quantiles averaged",
         "by Miles' reducer (not exact batch quantiles). Regenerate: `python report_g2.py <rundir>`.", "",
         "| step | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
for r in rows:
    lines.append(f"| {r['step']} | " + " | ".join("" if r[k] is None else f"{r[k]:.6g}" for k in keys) + " |")
lines += ["", "Across rounds (min / median / max / p90):", ""]
for k in keys:
    vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
    if vals:
        lines.append(f"- {k}: {min(vals):.6g} / {statistics.median(vals):.6g} / {max(vals):.6g} / {q(vals, 0.9):.6g}")
(d / "report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
