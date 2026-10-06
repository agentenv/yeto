#!/bin/bash
# S11 fnPrep on the kept fn H200 cluster (model store FS mounted at /mnt/yeto-models by the fnboot launch):
#  (1) populate CharyZeng/Qwen3.8-Flash-Next-4layer@d19a6b60 into the FS hub layout + yeto-complete marker (same record as
#      scripts/populate_nebius_model_store.sh), (2) scripts/convert_qwen3_8_next.sh --variant 4layer -> torch_dist.
# Never deletes anything on the FS. df gate: free >= 2.1 x snapshot (snapshot + torch_dist of similar size) before (1),
# free >= 1.1 x snapshot before (2); otherwise FNPREP_FAIL (the chain then skips fnA). Timings + df -> <outdir>/fnprep.json.
# usage: s11fnprep.sh <cluster> <outdir> <yeto snapshot dir (git archive of the chain's HEAD)>
set -u; CL=$1; O=$2; Y=$3; export HOME=/home/michael; mkdir -p $O
S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20"
tar -C $Y -cf - . | timeout 600 $S $CL 'rm -rf ~/yeto-fnprep && mkdir -p ~/yeto-fnprep && tar -C ~/yeto-fnprep -xf -' || { echo "FNPREP_FAIL copy"; exit 1; }
timeout 5400 $S $CL 'bash -s' > $O/fnprep-remote.log 2>&1 <<'REMOTE'
set -u
REPO=CharyZeng/Qwen3.8-Flash-Next-4layer; REV=d19a6b60c0df8f90faf92c7c592b37df2e15b060; FS=/mnt/yeto-models
OUT=$FS/torch_dist/qwen3.8-flash-next-4layer_torch_dist; SNAP=$FS/hub/models--CharyZeng--Qwen3.8-Flash-Next-4layer/snapshots/$REV
MARK=$FS/yeto-complete/CharyZeng--Qwen3.8-Flash-Next-4layer@$REV.json
PY=; for p in /opt/sglang/bin/python python3; do $p -c "import huggingface_hub" 2>/dev/null && { PY=$p; break; }; done
echo "DF0 $(df -B1 --output=size,used,avail $FS | tail -1)"; mountpoint -q $FS || { echo "FNPREP_FAIL $FS not mounted"; exit 1; }
SIZE=$($PY -c "from huggingface_hub import HfApi;i=HfApi().model_info('$REPO',revision='$REV',files_metadata=True);print(sum(s.size or 0 for s in i.siblings))")
echo "SNAPSHOT_BYTES $SIZE"
avail() { df -B1 --output=avail $FS | tail -1 | tr -dc 0-9; }
t0=$(date +%s)
if [ -f $MARK ] && [ -d $SNAP ]; then echo "POPULATE already complete"; else
  [ $(avail) -ge $(python3 -c "print(int(2.1*$SIZE))") ] || { echo "FNPREP_FAIL df: avail $(avail) < 2.1 x $SIZE"; exit 1; }
  HF_HUB_ENABLE_HF_TRANSFER=1 $PY - <<PY || { echo "FNPREP_FAIL populate"; exit 1; }
import json, os, time
from huggingface_hub import HfApi, snapshot_download
repo, rev = "$REPO", "$REV"; t0 = time.time()
path = snapshot_download(repo, revision=rev, cache_dir="$FS/hub", max_workers=16); dt = time.time() - t0
info = HfApi().model_info(repo, revision=rev, files_metadata=True); want = {s.rfilename: s.size for s in info.siblings}
bad = [f for f in want if os.path.getsize(os.path.join(path, f)) != want[f]]
assert info.sha == rev and not bad, (info.sha, bad[:5])
os.makedirs("$FS/yeto-complete", exist_ok=True)
rec = dict(repo=repo, revision=rev, files=len(want), bytes=sum(want.values()), seconds=round(dt, 1), snapshot=os.path.relpath(path, "$FS"))
json.dump(rec, open("$MARK", "w")); print("[populate]", json.dumps(rec))
PY
fi
t1=$(date +%s); echo "POPULATE_S $((t1-t0))"; echo "DF1 $(df -B1 --output=size,used,avail $FS | tail -1)"
if [ -f $OUT/latest_checkpointed_iteration.txt ] && grep -qx release $OUT/latest_checkpointed_iteration.txt; then echo "CONVERT already release"; else
  [ $(avail) -ge $(python3 -c "print(int(1.1*$SIZE))") ] || { echo "FNPREP_FAIL df before convert: avail $(avail)"; exit 1; }
  cd ~/yeto-fnprep && bash scripts/convert_qwen3_8_next.sh --variant 4layer --hf-checkpoint $SNAP --output $OUT > ~/fnprep-convert.log 2>&1 || { tail -40 ~/fnprep-convert.log; echo "FNPREP_FAIL convert"; exit 1; }
  tail -5 ~/fnprep-convert.log
fi
t2=$(date +%s); echo "CONVERT_S $((t2-t1))"; echo "DF2 $(df -B1 --output=size,used,avail $FS | tail -1)"
echo "TRACKER $(cat $OUT/latest_checkpointed_iteration.txt 2>/dev/null)"; du -sb $OUT 2>/dev/null | sed 's/^/TORCHDIST_BYTES /'
echo FNPREP_OK
REMOTE
rc=$?
python3 - $O $rc <<'PY'
import json, re, sys
o, rc = sys.argv[1], int(sys.argv[2]); t = open(o + "/fnprep-remote.log").read()
g = lambda k: (re.findall(rf"^{k} (.*)$", t, re.M) or [None])[-1]
json.dump({"rc": rc, "ok": "FNPREP_OK" in t, "fail": g("FNPREP_FAIL"), "snapshot_bytes": g("SNAPSHOT_BYTES"), "populate_s": g("POPULATE_S"),
           "convert_s": g("CONVERT_S"), "df0": g("DF0"), "df1": g("DF1"), "df2": g("DF2"), "tracker": g("TRACKER"),
           "torchdist_bytes": g("TORCHDIST_BYTES")}, open(o + "/fnprep.json", "w"), indent=1)
PY
grep -q FNPREP_OK $O/fnprep-remote.log && echo FNPREP_OK || { echo "FNPREP_FAIL $(grep -o 'FNPREP_FAIL.*' $O/fnprep-remote.log | tail -1)"; exit 1; }
