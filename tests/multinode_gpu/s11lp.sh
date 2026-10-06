#!/bin/bash
# S11 teacher-forced logprob segment (s9-m5lp method): on the kept cluster with the island stopped (s1reset first), serve the BASE
# Qwen3-0.6B with sglang on the island's GPUs and score the fixed token sequences of lp-ref.json (32 prompts x ref outputs):
# labels tp1_g0..tp1_g3 (each card the sweep configs use), tp2_g01, tp4_g0123. Output <outdir>/lp.json + lp-summary.json
# (mean|d| / p99 / max vs tp1_g0). Info item: numerics of the cards/TP shapes, NOT the trained per-config policy.
# usage: s11lp.sh <cluster> <outdir> [gpus on the head: 4 (H200 alloc 4) | 2 (L40S 2x2 head)]
set -u; CL=$1; O=$2; NG=${3:-4}; D=$(cd "$(dirname "$0")" && pwd); export HOME=/home/michael; mkdir -p $O
S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20"
timeout 120 scp -q -o StrictHostKeyChecking=no $D/s1lp.py $D/lp-ref.json $CL:~/ || { echo "LP scp failed"; exit 1; }
timeout 1800 $S $CL "NG=$NG bash -s" > $O/lp-remote.log 2>&1 <<'REMOTE'
PY=; for p in /opt/sglang/bin/python python3; do $p -c "import torch, sglang" 2>/dev/null && { PY=$p; break; }; done
M=Qwen/Qwen3-0.6B; R=c1899de289a04d12100db370d81485cdf75e47ca; rm -f ~/lp.json
health() { for i in $(seq 1 120); do curl -sf 127.0.0.1:30000/health >/dev/null && return 0; sleep 5; done; return 1; }
stop() { pkill -f sglang.launch_server; sleep 5; pkill -9 -f "sglang::"; sleep 10; }
one() { t0=$(date +%s); env CUDA_VISIBLE_DEVICES=$3 $PY -m sglang.launch_server --model-path $M --revision $R --trust-remote-code --tp $2 --host 127.0.0.1 --port 30000 --mem-fraction-static 0.6 --random-seed 17 > ~/srv-$1.log 2>&1 &
  health && { echo "READY $1 $(( $(date +%s) - t0 ))s"; cd ~ && $PY s1lp.py http://127.0.0.1:30000 $1 lp.json lp-ref.json; } || echo "LP_FAIL $1"; stop; }
one tp1_g0 1 0; one tp1_g1 1 1; one tp2_g01 2 0,1
[ "$NG" -ge 4 ] && { one tp1_g2 1 2; one tp1_g3 1 3; one tp4_g0123 4 0,1,2,3; }
echo "LP_JSON $(base64 -w0 ~/lp.json 2>/dev/null)"
REMOTE
grep -o 'LP_JSON .*' $O/lp-remote.log | cut -d' ' -f2 | base64 -d > $O/lp.json 2>/dev/null
python3 - $O <<'PY'
import json, sys
o = sys.argv[1]
try: d = json.load(open(o + "/lp.json"))
except Exception as e: print("LP no data", e); sys.exit(1)
ref = d.get("tp1_g0"); out = {}
for k, v in d.items():
    if ref is None or k == "tp1_g0": continue
    xs = sorted(abs(a - b) for ra, rb in zip(ref, v) for a, b in zip(ra, rb) if a is not None and b is not None)
    out[k] = {"n": len(xs), "mean_abs": sum(xs) / len(xs) if xs else None, "p99": xs[int(0.99 * (len(xs) - 1))] if xs else None, "max": xs[-1] if xs else None}
json.dump(out, open(o + "/lp-summary.json", "w"), indent=1); print("LP summary", json.dumps(out))
PY
