#!/bin/bash
# No-progress abort (coordinator review): usage noprogress.sh a|c
# Abort + reclaim when (1) no Miles training step line ("'train/pg_loss'")
# within FIRST_STEP_MIN minutes of t_start, or (2) >= REPEAT_MAX
# "Task raised exception" / "Traceback" lines not followed by any train step
# (i.e. repeated failures before training progresses).
X=$1; P=algo1b-g1f-$X${ATTEMPT:+-$ATTEMPT}; D=/tmp/algo1b-g1f; L=$D/out-$X/launch.log; Y=/home/michael/work/algo-1b-tok
FIRST_STEP_MIN=${FIRST_STEP_MIN:-25}; REPEAT_MAX=${REPEAT_MAX:-20}
start=$(date +%s)
abort() { echo "$(date -u +%FT%TZ) ABORT: $1" >> $D/out-$X/noprogress.log
  (cd $Y && PYTHONPATH=$Y timeout 300 /home/michael/work/gpu-head/venv/bin/python -m yeto.cli down $P) >> $D/out-$X/noprogress.log 2>&1
  /tmp/modal-venv/bin/modal app stop -y yeto-$P >> $D/out-$X/noprogress.log 2>&1; exit 0; }
while [ ! -f $D/out-$X/t_end ]; do
  steps=$(grep -c "'train/pg_loss'" $L 2>/dev/null); steps=${steps:-0}
  errs=$(grep -cE "Task raised exception|Traceback \(most recent" $L 2>/dev/null); errs=${errs:-0}
  now=$(date +%s)
  if [ "$steps" -eq 0 ] && [ $((now - start)) -gt $((FIRST_STEP_MIN * 60)) ]; then abort "no training step within ${FIRST_STEP_MIN} min"; fi
  if [ "$steps" -eq 0 ] && [ "$errs" -ge "$REPEAT_MAX" ]; then abort "$errs repeated exceptions before any training step"; fi
  sleep 30
done
echo "$(date -u +%FT%TZ) run ended; no abort" >> $D/out-$X/noprogress.log
