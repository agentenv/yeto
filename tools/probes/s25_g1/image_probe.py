"""CPU-only probe of the Miles ports image for the S25 G1 stage-1 mini rehearsal.

modal run tools/probes/s25_g1/image_probe.py   (no GPU; prints one JSON report)
"""
import modal

# Same digest as yeto.rl.adapters.miles.pins.MILES_NEXT_IMAGE (the container has no yeto).
MILES_NEXT_IMAGE = ("docker:ghcr.io/agentenv/yeto-miles-ports@sha256:"
                    "5b9dc2a9e6e315d73dabfbc6faae428b59cc01aabec6189645a5c8c80d6e3721")

app = modal.App("yeto-s25-g1-probe")
image = modal.Image.from_registry(MILES_NEXT_IMAGE.removeprefix("docker:"))

PROBE = r'''
import importlib, json, subprocess, sys
out = {}
def tryit(key, fn):
    try: out[key] = fn()
    except Exception as e: out[key] = f"ERR {type(e).__name__}: {str(e)[:300]}"
def ver(m): return getattr(importlib.import_module(m), "__version__", "?")
for m in ("torch", "transformer_engine", "megatron.core", "sglang", "fla", "tilelang", "triton",
          "triton_kernels", "mbridge", "compressed_tensors", "flash_attn", "ray", "transformers"):
    tryit("ver:" + m, lambda m=m: ver(m))
def tcfg():
    from megatron.core.transformer import TransformerConfig
    import dataclasses
    names = {f.name for f in dataclasses.fields(TransformerConfig)}
    return {k: k in names for k in ("enable_hyper_connections", "num_residual_streams",
            "mhc_sinkhorn_iterations", "use_fused_mhc", "experimental_attention_variant")}
tryit("megatron_mhc_fields", tcfg)
tryit("sglang_glm5_next", lambda: bool(importlib.import_module("sglang.srt.models.glm5_next")))
tryit("sglang_mxfp4", lambda: bool(importlib.import_module("sglang.srt.layers.quantization.mxfp4").Mxfp4MoEMethod))
tryit("fla_kda", lambda: bool(importlib.import_module("fla.ops.kda")))
def git(d):
    return subprocess.run(["git", "-C", d, "log", "-1", "--format=%H %cd %s"], capture_output=True, text=True).stdout.strip()
for d in ("/root/miles", "/root/Megatron-LM", "/sgl-workspace/sglang"):
    tryit("git:" + d, lambda d=d: git(d))
tryit("manifest", lambda: open("/opt/yeto/image-manifest.json").read()[:1500])
tryit("python", lambda: sys.version)
def sh(c): return subprocess.run(["bash", "-lc", c], capture_output=True, text=True).stdout[-1500:]
tryit("mcore_mhc_grep", lambda: sh("grep -rln 'enable_hyper_connections' /root/Megatron-LM/megatron | head"))
tryit("mcore_remote", lambda: sh("git -C /root/Megatron-LM remote -v | head -2; git -C /root/Megatron-LM log --oneline -3"))
tryit("te_pip", lambda: sh("pip show transformer_engine transformer-engine-cu12 transformer_engine_torch 2>/dev/null | grep -E 'Name|Version'"))
tryit("triton_kernels", lambda: sh("python3 -c 'import triton_kernels,os;print(os.path.dirname(triton_kernels.__file__))'; pip show triton_kernels 2>/dev/null|head -3"))
tryit("mem_cpu", lambda: sh("nproc; free -g | head -2; df -h /root | tail -1"))
print(json.dumps(out, indent=1))
'''


@app.function(image=image, cpu=2, memory=8192, timeout=900)
def probe():
    import subprocess
    r = subprocess.run(["python3", "-c", PROBE], capture_output=True, text=True)
    return r.stdout + "\nSTDERR-TAIL:\n" + r.stderr[-3000:]


@app.local_entrypoint()
def main():
    print(probe.remote())
