"""Task 2.2: back-test yeto.memory_estimate against measured runs.

Measured values (read from /home/michael/work/s1-runs, see `source`):
* peak_gpu_mem_bytes = max of the island tape ``rl_load_sample.peak_gpu_mem_bytes``
  (NVML device memory used, sampled every 10 s) -- a lower bound of the true peak.
* OOM runs: the CUDA OOM message; demand >= in-use + requested.
Run: python memory_backtest.py > memory-backtest.json
"""
import json
import sys
from types import SimpleNamespace

from yeto import memory_estimate as me

GIB = me.GIB
RUNS = [
    # name, role, model, gpu, seq_len, response_len, measured
    ("M1 run b", "calibration", "Qwen/Qwen3.5-4B", "H100", 16384, 8192,
     {"kind": "oom", "demand_gib_at_least": 78.00 + 126 / 1024,
      "source": "s17-m1-20261008b launch log: OOM 'Tried to allocate 126.00 MiB', 78.00 GiB in use of 79.18"}),
    ("M1 run c", "calibration", "Qwen/Qwen3.5-4B", "H200", 16384, 8192,
     {"kind": "oom", "demand_gib_at_least": 133.76 + 7.58, "logits_block_gib": 7.58,
      "source": "s17-m1-20261008c launch log: OOM 'Tried to allocate 7.58 GiB', 133.76 GiB in use of 139.80; "
                "107.75 allocated by PyTorch, 21.04 reserved-unallocated"}),
    ("M1 run d", "calibration", "Qwen/Qwen3.5-4B", "H200", 12288, 6144,
     {"kind": "peak", "peak_bytes": [131.32e9, 132.36e9],
      "source": "s17-m1-20261008d tape-direct l0/l1 rl-island-*.jsonl max peak_gpu_mem_bytes; "
                "S17-MORNING-REPORT says 125-126 GB (source of that figure not found)"}),
    ("N17 A/B a (a)", "test", "Qwen/Qwen3.5-4B", "H200", 12288, 6144,
     {"kind": "peak", "peak_bytes": [136.19e9], "source": "s17-n17-ab-a-20261009a tape-direct l0"}),
    ("N17 A/B a (b)", "test", "Qwen/Qwen3.5-4B", "H200", 12288, 6144,
     {"kind": "peak", "peak_bytes": [136.46e9], "source": "s17-n17-ab-a-20261009b tape-direct l0"}),
    ("N17 A/B b (a)", "test", "Qwen/Qwen3.5-4B", "H200", 12288, 6144,
     {"kind": "peak", "peak_bytes": [132.88e9], "source": "s17-n17-ab-b-20261009a tape-direct l0"}),
    ("N17 A/B b (b)", "test", "Qwen/Qwen3.5-4B", "H200", 12288, 6144,
     {"kind": "peak", "peak_bytes": [136.25e9], "source": "s17-n17-ab-b-20261009b tape-direct l0"}),
    ("N17 codex", "test", "Qwen/Qwen3.5-4B", "H200", 12288, 6144,
     {"kind": "peak", "peak_bytes": [128.32e9], "source": "s17-n17-codex-20261009a tape-direct l0"}),
    ("N17 elastic", "test", "Qwen/Qwen3-0.6B", "H100", 1024, 384,
     {"kind": "peak", "peak_bytes": [37.54e9, 37.56e9], "source": "s17-n17-elastic-20261009a tape-direct l0/l1"}),
    ("N17 strict", "test", "Qwen/Qwen3-0.6B", "H100", 1024, 384,
     {"kind": "peak", "peak_bytes": [37.54e9, 37.56e9], "source": "s17-n17-strict-20261009a tape-direct l0/l1"}),
    ("ARU-2 M0", "test", "Qwen/Qwen3-0.6B", "H100", 2560, 2048,
     {"kind": "peak", "peak_bytes": [39.48e9], "source": "s18-aru2-m0 tape-direct l0"}),
    ("ARU-2 M1", "test", "Qwen/Qwen3-0.6B", "H100", 2560, 2048,
     {"kind": "peak", "peak_bytes": [41.75e9], "source": "s18-aru2-m1 tape-direct l0"}),
    ("ARU-2 MB", "test", "Qwen/Qwen3-0.6B", "H100", 2560, 2048,
     {"kind": "peak", "peak_bytes": [60.10e9], "source": "s18-aru2-mb tape-direct l0"}),
]


def main():
    margin = 0.9
    rows = []
    for name, role, model, gpu, seq, resp, meas in RUNS:
        args = SimpleNamespace(training_mode="rl", model=model, seq_len=seq, rollout_max_response_len=resp,
                               tuning="lora", lora_r=16, lora_targets="attention", sglang_mem_fraction_static=0.4)
        spec = SimpleNamespace(gpu=gpu, total_gpus=1)
        est = me.estimate_for_launch(args, spec, margin=margin)
        row = {"run": name, "role": role, "model": model, "gpu": gpu, "seq_len": seq, "response_len": resp,
               "estimate_peak_gib": est["peak_gib"], "limit_gib": est["limit_gib"], "fits": est["fits"],
               "parts_gib": est["parts_gib"], "measured": meas}
        if meas["kind"] == "peak":
            peak = max(meas["peak_bytes"]) / GIB
            row["measured_peak_gib"] = round(peak, 2)
            row["error_pct"] = round(100 * (est["peak_gib"] - peak) / peak, 1)
        else:
            row["measured_demand_gib_at_least"] = round(meas["demand_gib_at_least"], 2)
            row["estimate_over_limit"] = not est["fits"]
            if "logits_block_gib" in meas:
                lg = est["parts_gib"]["logits_block"]
                row["logits_block_error_pct"] = round(100 * (lg - meas["logits_block_gib"]) / meas["logits_block_gib"], 1)
        rows.append(row)
    out = {"constants": {"K_LOGIT": me.K_LOGIT, "FRAG": round(me.FRAG, 4), "R0_GIB": round(me.R0_GIB, 3),
                         "K_ACT_NO_RECOMPUTE": me.K_ACT_NO_RECOMPUTE, "K_ACT_RECOMPUTE(unverified)": me.K_ACT_RECOMPUTE},
           "margin": margin, "rows": rows}
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
