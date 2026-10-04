#!/bin/bash
# usage: mkatt.sh <name> <total_steps> [extra learner-affecting cli args...] -> cfg/attestation-<name>.json (fingerprint computed locally with the
# same CLI->learner->Miles argv path, validated against the real 2d0a00f4 value at 47efd25 and 9a06c4b)
N=$1; S=$2; shift 2
fp=$(/tmp/yeto-venv/bin/python /tmp/fpl/fp_local.py /home/michael/work/gpu-b1 --total-steps $S "$@" 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['fp'])")
[ -n "$fp" ] || { echo fp-failed; exit 1; }
python3 - "$fp" > /home/michael/work/gpu-b1-runs/cfg/attestation-$N.json <<'PY'
import json,sys
print(json.dumps({"runtime_fingerprint":sys.argv[1],"execution_modes":["partitioned-serial"],"certified_edges":[{"source":"T4R2S2","target":"T4R4S0","kind":"rollout-only"},{"source":"T4R4S0","target":"T4R2S2","kind":"rollout-only"}]},separators=(",",":")))
PY
echo $N $S $fp
