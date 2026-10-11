# CPU-only Modal download of Qwen/Qwen3-30B-A3B@ad44e777 into Volume yeto-fn-models.
# Reuses the fn-modal/dl_hf.py pattern (HF volume layout hf/<name>/<rev[:8]>/ + MANIFEST).
import json, os, hashlib, time
import modal
REPO="Qwen/Qwen3-30B-A3B"; REV="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
ROOT=f"/vol/hf/Qwen3-30B-A3B/{REV[:8]}"
vol=modal.Volume.from_name("yeto-fn-models", create_if_missing=True)
img=modal.Image.debian_slim().pip_install("huggingface_hub[hf_transfer]==0.35.3","requests")
app=modal.App("yeto-rlf-hf-download", image=img)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<24),b""): h.update(b)
    return h.hexdigest()

@app.function(cpu=2, memory=4096, volumes={"/vol":vol}, timeout=3600, retries=2, max_containers=12)
def fetch(files):
    import requests
    out=[]
    for f in files:
        p=os.path.join(ROOT,f["path"]); os.makedirs(os.path.dirname(p),exist_ok=True)
        if not (os.path.exists(p) and os.path.getsize(p)==f["size"]):
            t=time.time(); tmp=p+".part"
            with requests.get(f"https://huggingface.co/{REPO}/resolve/{REV}/{f['path']}",stream=True,timeout=60) as r:
                r.raise_for_status()
                with open(tmp,"wb") as o:
                    for c in r.iter_content(1<<24): o.write(c)
            os.replace(tmp,p)
        s=sha(p); exp=f.get("sha256")
        out.append({"path":f["path"],"size":os.path.getsize(p),"sha256":s,"expected_sha256":exp,
                    "ok": (s==exp) if exp else (os.path.getsize(p)==f["size"])})
        vol.commit()
    return out

@app.function(cpu=0.25, memory=512, volumes={"/vol":vol}, timeout=300)
def write_manifest(m):
    vol.reload(); open(os.path.join(ROOT,"MANIFEST.json"),"w").write(json.dumps(m,indent=1)); vol.commit()

@app.local_entrypoint()
def main():
    import urllib.request
    tree=json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/tree/{REV}?recursive=1"))
    files=[{"path":x["path"],"size":x["size"],"sha256":(x.get("lfs") or {}).get("oid")} for x in tree if x["type"]=="file"]
    files.sort(key=lambda x:-x["size"]); groups=[[] for _ in range(12)]; loads=[0]*12
    for f in files: i=loads.index(min(loads)); groups[i].append(f); loads[i]+=f["size"]
    t=time.time(); res=[r for g in fetch.map(groups) for r in g]
    m={"repo":REPO,"revision":REV,"files":sorted(res,key=lambda r:r["path"]),"n":len(res),
       "total_bytes":sum(r["size"] for r in res),"all_ok":all(r["ok"] for r in res),"wall_s":round(time.time()-t)}
    write_manifest.remote(m)
    json.dump(m,open("/home/michael/work/s19-verl64b-run/evidence/rlf/MANIFEST-hf-qwen3-30b-a3b-ad44e777.json","w"),indent=1)
    print("N",m["n"],"bytes",m["total_bytes"],"all_ok",m["all_ok"],"wall_s",m["wall_s"],"bad",[r["path"] for r in res if not r["ok"]])
