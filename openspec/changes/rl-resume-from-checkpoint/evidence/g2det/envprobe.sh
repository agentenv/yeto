# runs inside the Modal container (sh); prints a redacted environment report
echo "=== date"; date -u +%FT%TZ
echo "=== uname"; uname -a
echo "=== cpu"; grep -m1 "model name" /proc/cpuinfo; nproc
echo "=== nvidia-smi"; nvidia-smi
echo "=== nvidia-smi -q (selected)"; nvidia-smi -q | grep -iE "Driver Version|CUDA Version|Product Name|VBIOS|Serial Number|GPU UUID|Bus Id|MIG Mode|Current  *: |ECC Mode|Persistence|Compute Mode|Max Clocks|Graphics  *:|SM  *:|Memory  *:|Power Limit" | head -60
echo "=== python libs"
PYB=$(command -v python3 || command -v python)
$PYB - <<'P' 2>&1
import importlib.metadata as m, json
out = {}
try:
    import torch  # no CUDA context is created here (no is_available / device queries)
    out.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               nccl=".".join(map(str, torch.cuda.nccl.version())),
               tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32)
except Exception as e:
    out["torch_err"] = repr(e)
for d in m.distributions():
    n = (d.metadata["Name"] or "").lower()
    if n.startswith(("nvidia-", "transformer-engine", "transformer_engine", "flash-attn", "flash_attn", "flashinfer", "triton", "sgl-kernel", "torch", "megatron", "apex", "deep-gemm", "deep_gemm", "cuda-")):
        out["pkg:" + n] = d.version
print(json.dumps(out, sort_keys=True, indent=1))
P
echo "=== process environments (redacted, determinism/cuda/nccl/nvte/torch/ray/modal keys)"
for p in /proc/[0-9]*; do
  c=$(tr '\0' ' ' < $p/cmdline 2>/dev/null | cut -c1-160); [ -z "$c" ] && continue
  case "$c" in *python*|*ray*|*sglang*) ;; *) continue;; esac
  echo "--- pid ${p#/proc/}: $c"
  tr '\0' '\n' < $p/environ 2>/dev/null | grep -E '^(NCCL_|CUBLAS|CUDA|NVTE_|NVIDIA_|TORCH|PYTORCH|CUDNN|TF32|SGLANG|SGL_|FLASHINFER|TRITON|OMP_|MKL_|LD_LIBRARY_PATH|RAY_|MODAL_|PYTHONHASHSEED|NCCL|MAMBA|TOKENIZERS|HF_|TRANSFORMERS|YETO_|MILES_)' | grep -viE 'TOKEN|SECRET|PASSWORD|KEY=' | sort
done
