cd /tmp/yeto-rerun; trap '/tmp/modal-venv/bin/python sbx.py kill $(cat sidA); date -u +%FT%TZ > t_endA' EXIT
for m in normal badtoken zerograd pubfail; do ./runcmd.sh sidA smoke-$m "python /work/harness/smoke.py $m /work/out/smoke-$m"; done
./runcmd.sh sidA kill44 "python /work/harness/kill44.py /work/out/kill44"
