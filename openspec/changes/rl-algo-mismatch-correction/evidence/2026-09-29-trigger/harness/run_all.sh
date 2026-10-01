#!/bin/bash
# sequential trigger runs, each only when per-user threads <= 3296
cd "$(dirname "$0")/.."
go(){ until [ $(ps -u michael -L --no-headers | wc -l) -le 3296 ]; do sleep 30; done
      echo "$1 start threads=$(ps -u michael -L --no-headers | wc -l) $(date -u +%T)"; OPT_STEPS=$2 bash harness/run_one.sh "${@:1:1}" specs/$1.json "${@:3}"; }
go tis "" corrections:tis
go icepop "" corrections:custom corrections:icepop features:mismatch_metrics
go mis-mask "" corrections:custom corrections:mis_mask
go opsm-trainer 2 corrections:opsm corrections:opsm_trainer
echo ALLDONE
