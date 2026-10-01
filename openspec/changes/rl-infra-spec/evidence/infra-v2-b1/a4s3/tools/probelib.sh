# source me: probe_ok <run dir> <outfile> <mode> [arg] -- run fork probe, retry (up to 12 x 20 s) until it prints its JSON line; never stops the cluster
probe_ok() {
  local R=$1 out=$2; shift 2
  for i in $(seq 1 12); do
    /home/michael/work/gpu-b1-runs/run_probe.sh $R "$@" > $out 2>&1
    grep -q '^{.*"mode"' $out && return 0
    { echo "=== attempt $i failed $(date -u +%T)"; tail -c 1500 $out; } >> $out.attempts; sleep ${PROBE_GAP:-20}
  done
  return 1
}
