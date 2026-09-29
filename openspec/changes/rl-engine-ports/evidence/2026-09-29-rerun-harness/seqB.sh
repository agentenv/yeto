cd /tmp/yeto-rerun; trap '/tmp/modal-venv/bin/python sbx.py kill $(cat sidB); date -u +%FT%TZ > t_endB' EXIT
./runcmd.sh sidB strict2 "bash /work/harness/strict2.sh /work/out/strict2"
./runcmd.sh sidB decoupled "bash /work/harness/decoupled.sh /work/out/decoupled"
