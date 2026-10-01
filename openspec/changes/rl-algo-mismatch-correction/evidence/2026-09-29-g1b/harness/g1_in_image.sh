#!/bin/bash
# Runs inside the sandbox. usage: g1_in_image.sh <runs...>   (writes /work/out/<run>)
set -u
export PYTHONPATH=/root/miles:/work/yeto:/work/harness
nvidia-smi --query-gpu=name --format=csv,noheader | tee /work/out/gpu_name.txt
grep -q "H100" /work/out/gpu_name.txt || { echo "GPU ASSERT FAILED"; exit 3; }
cat /opt/yeto/image-manifest.json > /work/out/image-manifest.json 2>/dev/null
python /work/harness/predownload.py > /work/out/predownload.log 2>&1 || { echo PREDOWNLOAD FAILED; exit 4; }
for run in "$@"; do
  r=${run#g2-}; rounds=3; [ "$run" != "$r" ] && rounds=20
  ROUNDS=$rounds TMO=$([ $rounds = 3 ] && echo 1500 || echo 2400) python /work/harness/g1.py $r /work/out/$run > /work/out/$run.driver.log 2>&1
  tail -2 /work/out/$run.driver.log
  [ "$r" = observe ] && [ $rounds = 20 ] && python /work/harness/report_g2.py /work/out/$run > /dev/null
  python /work/harness/check_g1.py $r /work/out/$run $rounds no-sync > /work/out/$run/check.json; echo "$run check rc=$?"
done
