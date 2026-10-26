#!/usr/bin/env bash
# usage: gm2_hf.sh <tag> <vllm_json_tag> <dtype>  -- HF CPU reference in a container with no NPU attached
W=/data/work/s28-gm2; cd $W
docker run --rm --name gm2_hf_$1 -e ASCEND_VISIBLE_DEVICES= -e PYTHONPATH=$W/tools/hfstub -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 -e MODEL=$W/models/nano -e THREADS=64 -e DTYPE=$3 -v /data:/data -w $W \
  q38/verl-loop:8.5.2 python tools/gm2_hf_ref.py out/$2.json out/$1.json > logs/$1.log 2>&1
echo "rc=$?" > out/$1.rc
