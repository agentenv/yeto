#!/bin/bash
# S11 fnConv (B0-2) on the kept fn H200 cluster, after fnA: HF Qwen/Qwen3.8-Flash-Next@de4b8e4d (already on the FS) -> Megatron
# torch_dist at /mnt/yeto-models/torch_dist/qwen3.8-flash-next_torch_dist, exactly the command `fntrain.sh convert` renders
# (YETO_Q38N_NUM_NODES=4 YETO_Q38N_CONVERT_PP=4 convert_qwen3_8_next.sh --variant full --nproc 8 -> TP2 PP4).
# Gates: HF marker+snapshot present; FS avail >= FNCONV_NEED_G (402 = 1.2 x 335GiB [estimate, FN-TRAIN-PLAN B0-1: bf16 weights
# of the 360GB HF snapshot, no optimizer]). Never deletes. NO RETRY: an output dir without a "release" tracker (a previous failed or
# killed attempt) -> FNCONV_FAIL partial (analyse on CPU first). nvidia-smi samples every 5 s -> per-GPU peak MiB.
# Records start/end, df before/after, log tail, output bytes, per-GPU peak -> <outdir>/fnconv.json.
# usage: s11fnconv.sh <cluster> <outdir> <yeto snapshot dir>      DRY=1: run the remote body locally with --dry-run (no ssh/GPU),
#        FS=<dir> (DRY only) stands in for /mnt/yeto-models.
set -u; CL=$1; O=$2; Y=$3; export HOME=/home/michael; mkdir -p $O
S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20"
REMOTE=$(cat <<'REMOTE'
set -u
FS=${FS:-/mnt/yeto-models}; MODEL=Qwen--Qwen3.8-Flash-Next; REV=de4b8e4d43b917e7706784d8bb445c9af86a3540
SNAP=$FS/hub/models--$MODEL/snapshots/$REV; MARK=$FS/yeto-complete/$MODEL@$REV.json
OUT=$FS/torch_dist/qwen3.8-flash-next_torch_dist; YD=${YD:-$HOME/yeto-fnconv}; CLOG=${CLOG:-$HOME/fnconv-convert.log}
echo "T_START $(date -u +%FT%TZ)"; echo "DF0 $(df -B1 --output=size,used,avail $FS | tail -1)"
[ "${DRY:-0}" = 1 ] || mountpoint -q $FS || { echo "FNCONV_FAIL $FS not mounted"; exit 1; }
[ -f $MARK ] && [ -d $SNAP ] || { echo "FNCONV_FAIL no completed HF snapshot ($MARK)"; exit 1; }
if [ -f $OUT/latest_checkpointed_iteration.txt ] && grep -qx release $OUT/latest_checkpointed_iteration.txt; then
  echo "CONVERT already release"
else
  [ -e $OUT ] && { echo "FNCONV_FAIL partial output $OUT exists without release tracker (no retry; analyse on CPU)"; exit 1; }
  AV=$(df -BG --output=avail $FS | tail -1 | tr -dc 0-9); echo "AVAIL_G $AV"
  [ "$AV" -ge ${FNCONV_NEED_G:-402} ] || { echo "FNCONV_FAIL df: avail ${AV}G < ${FNCONV_NEED_G:-402}G"; exit 1; }
  SMI=${SMI:-$HOME/fnconv-smi.csv}; : > $SMI
  ( while :; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> $SMI 2>/dev/null || exit 0; sleep 5; done ) & SP=$!
  t0=$(date +%s); D=; [ "${DRY:-0}" = 1 ] && D=--dry-run
  cd $YD && YETO_Q38N_NUM_NODES=4 YETO_Q38N_CONVERT_PP=4 timeout ${FNCONV_TIMEOUT:-4800} bash scripts/convert_qwen3_8_next.sh \
    --variant full --nproc 8 --hf-checkpoint $SNAP --output $OUT $D > $CLOG 2>&1; crc=$?
  kill $SP 2>/dev/null; echo "CONVERT_S $(( $(date +%s) - t0 ))"; echo "CONVERT_RC $crc"; tail -20 $CLOG
  echo "GPU_PEAK_MIB $(sort -t, -k1,1n -k2,2n $SMI | awk -F', *' '{m[$1]=$2} END{for(i in m) printf "%s:%s ", i, m[i]}')"
  [ $crc = 0 ] || { echo "FNCONV_FAIL convert rc=$crc"; exit 1; }
fi
echo "T_END $(date -u +%FT%TZ)"; echo "DF1 $(df -B1 --output=size,used,avail $FS | tail -1)"
echo "TRACKER $(cat $OUT/latest_checkpointed_iteration.txt 2>/dev/null)"; du -sb $OUT 2>/dev/null | sed 's/^/TORCHDIST_BYTES /'
[ -f $OUT/yeto-profile-manifest.json ] && echo "MANIFEST yes"
[ "${DRY:-0}" = 1 ] || grep -qx release $OUT/latest_checkpointed_iteration.txt || { echo "FNCONV_FAIL tracker not release"; exit 1; }
echo FNCONV_OK
REMOTE
)
if [ "${DRY:-0}" = 1 ]; then
  echo "$REMOTE" | env DRY=1 SMI=$O/smi.csv YD=$Y CLOG=$O/convert.log FS=${FS:?DRY needs FS=<dir>} FNCONV_NEED_G=${FNCONV_NEED_G:-0} bash -s > $O/fnconv-remote.log 2>&1
else
  tar -C $Y -cf - . | timeout 600 $S $CL 'rm -rf ~/yeto-fnconv && mkdir -p ~/yeto-fnconv && tar -C ~/yeto-fnconv -xf -' || { echo "FNCONV_FAIL copy"; exit 1; }
  echo "$REMOTE" | timeout ${FNCONV_SSH_TIMEOUT:-5400} $S $CL "FNCONV_TIMEOUT=${FNCONV_TIMEOUT:-4800} FNCONV_NEED_G=${FNCONV_NEED_G:-402} bash -s" > $O/fnconv-remote.log 2>&1
  timeout 300 $S $CL 'cat ~/fnconv-convert.log' > $O/convert.log 2>&1
fi
python3 - $O <<'PY'
import json, re, sys
o = sys.argv[1]; t = open(o + "/fnconv-remote.log").read()
g = lambda k: (re.findall(rf"^{k} (.*)$", t, re.M) or [None])[-1]
peak = {k: int(v) for k, v in (p.split(":") for p in (g("GPU_PEAK_MIB") or "").split())}
json.dump({"ok": "FNCONV_OK" in t, "fail": g("FNCONV_FAIL"), "t_start": g("T_START"), "t_end": g("T_END"), "convert_s": g("CONVERT_S"),
           "convert_rc": g("CONVERT_RC"), "avail_g_before": g("AVAIL_G"), "df0": g("DF0"), "df1": g("DF1"), "tracker": g("TRACKER"),
           "torchdist_bytes": g("TORCHDIST_BYTES"), "manifest": g("MANIFEST"), "gpu_peak_mib": peak,
           "already": "CONVERT already release" in t, "log": "convert.log"}, open(o + "/fnconv.json", "w"), indent=1)
PY
grep -q FNCONV_OK $O/fnconv-remote.log && echo FNCONV_OK || { grep -o "FNCONV_FAIL.*" $O/fnconv-remote.log | tail -1 | grep . || echo "FNCONV_FAIL (no marker; see fnconv-remote.log)"; exit 1; }
