"""Modal launcher for the T-G3 chain (N7.5 / N8.2 / N7.1d) on ``H100!:2``.

usage (from this directory, after the pre-registration in gpu-spend.md):
    G3_RUN=<run-id> G3_MILES=<miles fork checkout> G3_YETO=<yeto checkout> \\
        modal run --detach modal_g3.py [--phases all|n75|n71d]

The entrypoint only ``spawn()``s the chain and returns (killing a detached client would
cancel a blocking ``.remote()``). Results land in the Modal Volume ``t-g3-s25`` under
``/<run>/``; pull them with ``modal volume get t-g3-s25 /<run> <local dir>``.
"""

from __future__ import annotations

import os

import modal

RUN = os.environ.get("G3_RUN", "unset")
MILES = os.environ.get("G3_MILES", "/home/michael/work/miles-s25-g3")
YETO = os.environ.get("G3_YETO", "/home/michael/work/s25-g3-miles")
HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_REF = "radixark/miles:v0.1.1"  # Miles 2806267d + Megatron-LM f148a32b4 (mHC output contraction), TE 2.17
GPU = os.environ.get("G3_GPU", "H100!:2")  # preflight: G3_GPU=L4:2 G3_EXPECT="NVIDIA L4" G3_HARD=900
EXPECT_GPU = os.environ.get("G3_EXPECT", "NVIDIA H100 80GB HBM3")
HARD_S = int(os.environ.get("G3_HARD", "3600"))

image = (
    modal.Image.from_registry(IMAGE_REF).entrypoint([])
    .env({"PYTHONPATH": "/g3/miles:/g3/yeto:/root/Megatron-LM", "CUDA_DEVICE_MAX_CONNECTIONS": "1", "G3_MILES": "/g3/miles",
          "PYTHONUNBUFFERED": "1"})
    .add_local_dir(MILES, "/g3/miles", copy=False, ignore=[".git", "**/__pycache__", "tests", "docs", "examples"])
    .add_local_dir(os.path.join(YETO, "yeto"), "/g3/yeto/yeto", copy=False, ignore=["**/__pycache__"])
    .add_local_dir(HERE, "/g3/harness", copy=False, ignore=["**/__pycache__"])
)
vol = modal.Volume.from_name("t-g3-s25", create_if_missing=True)
app = modal.App(f"t-g3-{RUN}")

CHAIN = r"""
set -uo pipefail
exec > >(tee -a "$OUT/chain.log") 2>&1
H=/g3/harness/harness.py
run() {  # name timeout cmd...
  local name=$1 t=$2; shift 2
  echo "=== PHASE $name start $(date -u +%T)"
  timeout "$t" "$@" > "$OUT/logs/$name.log" 2>&1; local rc=$?
  echo "=== PHASE $name rc=$rc end $(date -u +%T)"; tail -5 "$OUT/logs/$name.log"
  echo "$name $rc" >> "$OUT/phases.txt"
  return $rc
}
mkdir -p "$OUT/logs" "$OUT/n75" "$OUT/n71d"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv | tee "$OUT/nvidia-smi.csv"
n=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c "^$EXPECT_GPU\$")
[ "$n" = 2 ] || { echo "GPU ASSERT FAILED: want 2x $EXPECT_GPU"; exit 3; }
pip list 2>/dev/null | grep -iE '^(torch|transformer.engine|megatron|flash.linear|tilelang|mbridge) ' > "$OUT/versions.txt"
git -C /root/Megatron-LM rev-parse HEAD >> "$OUT/versions.txt"
python $H --phase hfconfig --out "$OUT" --hf /tmp/g3-hf && cp /tmp/g3-hf/config.json "$OUT/hf-config.json"
TR="torchrun --nproc-per-node 2 --master-port 29511"
if [[ "$PHASES" == pre ]]; then
  run pre-n75-te 600 $TR $H --phase n75 --impl te --tag te --out "$OUT/n75"
  run pre-n71d 400 $TR $H --phase n71d --mode cont5 --out "$OUT/n71d" --cut-dir "$OUT/n71d/cut"
fi
if [[ "$PHASES" == n71d-noise ]]; then  # more samples of the run-to-run spread (cut in a second directory)
  export NCCL_ALGO=Ring CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
  for m in cont5c cont5d; do
    run n71d-$m 600 $TR $H --phase n71d --mode $m --out "$OUT/n71d" --cut-dir "$OUT/n71d/cut"
  done
  run n71d-run3b 600 $TR $H --phase n71d --mode run3b --out "$OUT/n71d" --cut-dir "$OUT/n71d/cutb"
  run n71d-resume2b 600 $TR $H --phase n71d --mode resume2b --out "$OUT/n71d" --cut-dir "$OUT/n71d/cutb"
fi
if [[ "$PHASES" == all || "$PHASES" == n75 ]]; then
  run n75-te 1200 $TR $H --phase n75 --impl te --tag te --out "$OUT/n75"
  run n75-te2 900 $TR $H --phase n75 --impl te --tag te2 --weights-from "$OUT/n75" --out "$OUT/n75"
  run n75-local 1200 $TR $H --phase n75 --impl local --tag local --weights-from "$OUT/n75" --out "$OUT/n75"
fi
if [[ "$PHASES" == all || "$PHASES" == n71d ]]; then
  export NCCL_ALGO=Ring CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
  run n71d-cont5 900 $TR $H --phase n71d --mode cont5 --out "$OUT/n71d" --cut-dir "$OUT/n71d/cut"
  run n71d-cont5b 900 $TR $H --phase n71d --mode cont5b --out "$OUT/n71d" --cut-dir "$OUT/n71d/cut"
  run n71d-run3 900 $TR $H --phase n71d --mode run3 --out "$OUT/n71d" --cut-dir "$OUT/n71d/cut"
  run n71d-resume2 900 $TR $H --phase n71d --mode resume2 --out "$OUT/n71d" --cut-dir "$OUT/n71d/cut"
fi
python /g3/harness/compare.py "$OUT" > "$OUT/logs/compare.log" 2>&1; echo "compare rc=$?"
du -sh "$OUT"; echo CHAIN DONE
"""


@app.function(image=image, gpu=GPU, cpu=16.0, memory=65536, timeout=HARD_S, volumes={"/vol": vol})
def chain(run: str, phases: str, expect_gpu: str) -> int:
    import subprocess
    import threading
    import time

    out = f"/vol/{run}"
    os.makedirs(out, exist_ok=True)
    stop = threading.Event()

    def committer():  # partial evidence survives a hard timeout
        while not stop.wait(60):
            try:
                vol.commit()
            except Exception as exc:  # noqa: BLE001
                print("commit failed", exc, flush=True)

    threading.Thread(target=committer, daemon=True).start()
    t0 = time.time()
    rc = subprocess.call(["bash", "-c", CHAIN], env={**os.environ, "OUT": out, "PHASES": phases,
                                                     "EXPECT_GPU": expect_gpu})
    with open(f"{out}/chain.rc", "w") as fh:
        fh.write(f"{rc} {time.time() - t0:.0f}s\n")
    stop.set()
    vol.commit()
    return rc


@app.local_entrypoint()
def main(phases: str = "all"):
    assert RUN != "unset", "set G3_RUN"
    call = chain.spawn(RUN, phases, EXPECT_GPU)  # module globals are re-read in the container
    print("spawned", RUN, call.object_id, flush=True)
