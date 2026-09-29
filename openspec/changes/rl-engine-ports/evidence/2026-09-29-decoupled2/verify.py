import json, glob, hashlib, os, torch
R="/work/out/decoupled2/work/seed-17/yeto-decoupled-m2"
out={}
for i in (0,1):
    ev=[json.loads(l) for l in open(f"{R}/island-{i}/events.jsonl")]
    names=[e.get("event") for e in ev]
    fin=[e for e in ev if "final" in json.dumps(e).lower()]
    applies=[e for e in ev if e.get("event")=="rl_policy_apply"]
    out[f"island{i}"]={"n_events":len(ev),"event_kinds":sorted(set(names)),"last_apply":applies[-1] if applies else None,
      "tail":[{k:v for k,v in e.items() if len(str(v))<200} for e in ev[-8:]]}
print(json.dumps(out,indent=1,default=str)[:12000])
from transformers import AutoModelForCausalLM
from peft import PeftModel
import peft
base=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",revision="c1899de289a04d12100db370d81485cdf75e47ca",torch_dtype=torch.bfloat16)
m=PeftModel.from_pretrained(base,f"{R}/adapter")
lora=[n for n,_ in m.named_parameters() if "lora_" in n]
nz=sum(float(p.abs().sum()) for n,p in m.named_parameters() if "lora_B" in n)
print("PEFT_LOAD_OK peft",peft.__version__,"lora_tensors",len(lora),"sum|lora_B|",nz)
print(os.listdir(f"{R}/adapter"))
