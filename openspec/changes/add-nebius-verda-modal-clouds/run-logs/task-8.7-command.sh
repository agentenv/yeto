#!/usr/bin/env bash
# Task 8.7 mixed-fleet acceptance run, 2026-09-23 (attempt 2, prefix yeto-mix87b).
#
# The spec's literal fleet (aws:1xl4@us-east-1 + modal:1xh100) cannot run on this
# AWS account: the "Running On-Demand G and VT instances" vCPU quota is 0 in
# us-east-1 (spot G quota is 0 too), so no g4/g5/g6 L4 instance can ever start.
# Attempt 1 (yeto-mix87) failed with VcpuLimitExceeded in every zone.
#
# Substitution, keeping all three configured clouds in one launch:
#   - head + syncer on AWS us-east-1 (CPU only; standard On-Demand quota is 256)
#   - learner island 0: Nebius 1xH100 @ eu-north1
#   - learner island 1: Modal 1xH100 (unpinned region)
# Both learner islands still sync to one syncer across the public internet.
set -euo pipefail
cd /home/michael/yeto
exec .venv/bin/python -m yeto.cli launch \
  --gpu nebius:1xh100@eu-north1,modal:1xh100 \
  --syncer-region us-east-1 \
  --model qwen3-0.6b \
  --data HuggingFaceH4/no_robots \
  --assistant-mask-mode legacy \
  --max-rows 2000 --seq-len 1024 \
  --total-steps 16 --fragments 8 --quorum 2 \
  --on-demand \
  --cluster-prefix yeto-mix87b
