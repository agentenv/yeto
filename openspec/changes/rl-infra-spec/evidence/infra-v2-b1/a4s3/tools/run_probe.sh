#!/bin/bash
# usage: run_probe.sh <run dir> <mode> [arg]  -- runs fork_probe.py on the island with a python that can import ray (probe_remote.sh picks it)
R=$1; shift; CL=$(cat $R/cluster.txt); B=/home/michael/work/gpu-b1-runs
b64=$(base64 -w0 $B/fork_probe.py); rb=$(base64 -w0 $B/probe_remote.sh)
HOME=/home/michael timeout 300 ssh -o StrictHostKeyChecking=no $CL "echo $b64 | base64 -d > ~/yeto-rl/fork_probe.py; echo $rb | base64 -d > ~/yeto-rl/probe_remote.sh; bash ~/yeto-rl/probe_remote.sh $*" 2>&1 | grep -v "^Warning: Permanently"
