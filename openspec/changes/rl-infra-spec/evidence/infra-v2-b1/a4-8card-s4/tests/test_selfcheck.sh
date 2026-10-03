#!/bin/bash
# CPU test of selfcheck.sh markers with stubs (no ssh/cloud)
B=$(cd "$(dirname "$0")/.." && pwd); [ -f $B/cleanup_run.sh ] || B=$B/scripts; T=$(mktemp -d); trap 'rm -rf $T' EXIT; fail=0
printf '#!/bin/bash\necho nstop "$@" >> %s/nstop.calls\n' $T > $T/nstop; chmod +x $T/nstop
printf '#!/bin/bash\necho "$@" > /dev/null\n' > $T/ssh; chmod +x $T/ssh
run() { R=$T/$1; mkdir -p $R/pulled; echo cl > $R/cluster.txt; $2; BDIR=$B NSTOP=$T/nstop SSH=$T/ssh PROBE=$T/probe STARTUP_DEADLINE_S=${DL:-3} SC_POLL_S=1 SC_SETTLE_S=0 bash $B/selfcheck.sh $R p-1; echo $?; }
chk() { [ "$2" = "$3" ] && echo "PASS $1" || { echo "FAIL $1 got=$2 want=$3"; fail=1; }; }
f_ended() { echo rc=1 > $R/rc.txt; }
rc=$(run ended f_ended | tail -1); chk "ended_before_generate rc" $rc 1; [ -f $T/ended/startup_failed ] && grep -q launch_ended $T/ended/startup_failed; chk "ended marker" $? 0
f_to() { :; }; rc=$(run timeout f_to | tail -1); chk "deadline rc" $rc 1; grep -q no_generate_within $T/timeout/startup_failed; chk "deadline marker" $? 0
printf '#!/bin/bash\necho "{\\"probe_attested\\": false}"\n' > $T/probe; chmod +x $T/probe
f_gen() { echo '{"event":"rl_driver_phase","phase":"generate"}' | sed 's/"phase"/"phase"/;s/:"generate"/:"generate"/' | tr -d ' ' > $R/pulled/rl-island-0.jsonl; }
printf '#!/bin/bash\necho ok\n' > $T/ssh; chmod +x $T/ssh
rc=$(run badprobe f_gen | tail -1); chk "unattested probe rc" $rc 1; grep -q "probe_not_attested" $T/badprobe/startup_failed; chk "unattested marker" $? 0
[ -f $T/ended/injection_not_reached ] || [ -f $T/ended/recovery_failed ]; chk "selfcheck never writes judge markers" $? 1
# gpu assertion: 3 L40S + 1 H100 -> startup_failed gpu_assert; 4 L40S -> ok
cat > $T/probe <<'X'
#!/bin/bash
echo '{"probe_attested": true}'
X
cat > $T/ssh <<'X'
#!/bin/bash
echo '{"router":"x"}'; echo '{"data":{"inflight":{}}}'; echo 5
X
chmod +x $T/probe $T/ssh
f_gpu() { f_gen; printf '0,a,NVIDIA L40S,5\n1,b,NVIDIA L40S,5\n2,c,NVIDIA L40S,5\n3,d,NVIDIA H100,5\n' > $R/pulled/gpu.txt; }
rc=$(EXPECT_GPU_NAME=L40S EXPECT_GPU_N=4 run gpuassert f_gpu | tail -1); chk "gpu assert rc" $rc 1; grep -q gpu_assert $T/gpuassert/startup_failed; chk "gpu assert marker" $? 0
f_gpu_ok() { f_gen; printf '0,a,NVIDIA L40S,5\n1,b,NVIDIA L40S,5\n2,c,NVIDIA L40S,5\n3,d,NVIDIA L40S,5\n' > $R/pulled/gpu.txt; }
rc=$(EXPECT_GPU_NAME=L40S EXPECT_GPU_N=4 run gpuok f_gpu_ok | tail -1); chk "gpu ok rc" $rc 0; [ ! -f $T/gpuok/startup_failed ]; chk "no marker when ok" $? 0
[ $fail = 0 ] && echo ALL PASS || { echo SOME FAILED; exit 1; }
