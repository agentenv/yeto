set -e
E=/home/michael/work/r0-integ/openspec/changes/rl-engine-ports/evidence
S=/home/michael/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
PY="/tmp/yeto-venv/bin/python /home/michael/work/r0-integ/scripts/rl_engine_equivalence.py analyze"
RP="$E/2026-09-29-eq62-v3/replay/legacy-s17/work/seed-17/yeto-federated-m2/rollouts/island-{island}/0.pt"
cd /home/michael/work/gpu-eq/evidence/2026-09-29-eq62-v2
CFG=$(python3 -c "import json;print(json.load(open('report.json'))['meta']['config'])")
$PY --preset strict-avg --out-dir $E/2026-09-29-eq62-v3 \
 --legacy legacy-s17 ../2026-09-29-eq62/legacy-s18 ../2026-09-29-eq62/legacy-s19 ../2026-09-29-eq62/legacy-s20 ../2026-09-29-eq62/legacy-s21 \
 --ports ports-s17 ../2026-09-29-eq62/ports-s18 ../2026-09-29-eq62/ports-s19 ../2026-09-29-eq62/ports-s20 ../2026-09-29-eq62/ports-s21 \
 --tf-reference legacy-s17 --tf-legacy tf-legacy --tf-ports tf-ports --tf-replay "$RP" \
 --fp32-cache-dir /home/michael/.cache/huggingface/hub --fp32-threads 96 --fp32-save-dir /tmp/eq62-v3-fp32 \
 --config "$CFG. v3 (analyzed locally on CPU from /home/michael/work/gpu-eq/evidence/2026-09-29-eq62-v2, relative run dirs below are relative to it): tier 2 anchored on CPU fp32 reference (design D12 option A); replay batch = eq62-v2 legacy-s17 round-1 dumps (gpu-eq/ctl/pt2-strict.tgz, extracted to replay/)"
cd ../2026-09-29-eq63-v2
CFG=$(python3 -c "import json;print(json.load(open('report.json'))['meta']['config'])")
L=""; P=""; for s in 17 18 19 20 21; do L="$L ../2026-09-29-eq63/legacy-s$s"; P="$P ../2026-09-29-eq63/ports-s$s"; done
$PY --preset decoupled --out-dir $E/2026-09-29-eq63-v3 --legacy $L --ports $P \
 --tf-reference ../2026-09-29-eq62-v2/legacy-s17 --tf-legacy ../2026-09-29-eq62-v2/tf-legacy --tf-ports ../2026-09-29-eq62-v2/tf-ports \
 --tf-replay "$RP" --fp32-model-path $S --fp32-threads 96 --peft-base-model $S \
 --config "$CFG. v3 (analyzed locally on CPU from /home/michael/work/gpu-eq/evidence/2026-09-29-eq63-v2, relative run dirs are relative to it): tier 2 = strict-avg TF (eq62-v2) anchored on CPU fp32 reference (design D12 option A); replay batch in ../2026-09-29-eq62-v3/replay/"
