#!/bin/bash
# usage: mkatt.sh <name> <total_steps> [extra learner-affecting cli args...] -> cfg/attestation-<name>.json (fingerprint computed locally with the
# same CLI->learner->Miles argv path, validated against the real 8-card reference; the 4-card L40S value is verified by the smoke run (selfcheck compares it with the island rl_driver_start))
N=$1; S=$2; shift 2
fp=$(/tmp/yeto-venv/bin/python /home/michael/work/gpu-b1-runs/fp_local4.py ${REPO:-/home/michael/work/gpu-b1} --total-steps $S "$@" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['fp'])")
[ -n "$fp" ] || { echo fp-failed; exit 1; }
python3 - "$fp" > /home/michael/work/gpu-b1-runs/cfg/attestation-4-$N.json <<'PY'
import json,sys
print(json.dumps({"runtime_fingerprint":sys.argv[1],"execution_modes":["partitioned-serial"],"certified_edges":[{"source":"T1R1S2","target":"T1R3S0","kind":"rollout-only"},{"source":"T1R3S0","target":"T1R1S2","kind":"rollout-only"}]},separators=(",",":")))
PY
echo $N $S $fp
