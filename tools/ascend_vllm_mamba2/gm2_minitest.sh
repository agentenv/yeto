#!/usr/bin/env bash
# usage: gm2_minitest.sh <name> <experts> <topk> <dev>  -- build a mini, vLLM greedy gen, HF bf16 ref; print summary
set -u
N=$1; E=$2; K=$3; D=$4; L=${5:-7}; W=/data/work/s28-gm2; cd $W
[ -e models/$N/model.safetensors ] || docker run --rm -e ASCEND_VISIBLE_DEVICES= -e EXPERTS=$E -e TOPK=$K -e LAYERS=$L -v /data:/data -w $W q38/verl-loop:8.5.2 python tools/gm2_mini_nemotron.py models/nano models/$N > logs/mk_$N.log 2>&1
MODEL=$W/models/$N DEVS=$D TP=1 MAXTOK=32 GMU=0.5 bash tools/gm2_run.sh g_$N >/dev/null
MODEL=$W/models/$N bash tools/gm2_hf.sh g_${N}_hf g_$N float32
python3 -c "
import json;f=json.load(open('out/g_${N}_hf.json'))
d=[abs(x-y) for i in f['items'] for x,y in zip(i['ref_logprobs'],i['vllm_logprobs'])];print('$N E=$E K=$K L=$L vllm vs hf fp32',len(d),sum(d)/len(d),max(d))" > out/g_$N.summary 2>&1
