"""algo1b-cpu-img: CPU-only checks inside the pinned ports image (plan.md)."""

import modal

IMAGE = "radixark/miles@sha256:90940828dcd4d54fd907ff668b43537cbd94778047580e4160d6560af548b74d"
image = (
    modal.Image.from_registry(IMAGE)
    .add_local_dir("/home/michael/work/algo-1b", "/yeto", copy=False,
                   ignore=[".git", "**/__pycache__", "openspec/**/evidence/**"])
    .add_local_dir("/home/michael/work/miles-next", "/miles-next", copy=False,
                   ignore=[".git", "**/__pycache__"])
)
import os
GPU = os.environ.get("ALGO1B_GPU") or None  # run 2: T4 (libcuda for the import chain)
RUN2 = bool(GPU)
app = modal.App("algo1b-cpu-img", image=image)

SCRIPT = r"""
set -x
cd /tmp
python -c "import pytest" 2>/dev/null || pip install -q pytest
python -c "import miles, sys; print('image miles', miles.__file__)"
echo '--- 4.1: import examples.experimental.DrGRPO.custom_reducer (image Miles only)'
python - <<'P'
import importlib, traceback
try:
    m = importlib.import_module("examples.experimental.DrGRPO.custom_reducer")
    print("IMPORT_OK", m.__file__, getattr(m, "DIVISOR", None))
except Exception:
    print("IMPORT_FAILED"); traceback.print_exc()
P
python -c "import importlib.util as u; s=u.find_spec('examples'); print('examples spec', s and s.origin, s and list(s.submodule_search_locations or []))"
pip show miles 2>/dev/null | head -3
echo '--- full parse_args (miles-next 0394715 parser, image megatron)'
cd /yeto
PYTHONPATH=/yeto:/miles-next python -m pytest -q -p no:cacheprovider -rs \
  tests/test_rl_grpo_knobs_upstream.py -k "full_parse_args or miles_parser or kl_loss_triggers" 2>&1 | tail -40
PYTHONPATH=/yeto:/miles-next python -m pytest -q -p no:cacheprovider -rs \
  tests/test_rl_miles_adapter_config.py -k upstream_parse 2>&1 | tail -5
"""


@app.function(cpu=2.0, memory=8192, timeout=1200, gpu=GPU)
def run() -> str:
    import subprocess

    script = SCRIPT
    if RUN2:
        script = script.split("echo '--- full parse_args")[0].split("echo '--- 4.1")[0] + \
            "nvidia-smi --query-gpu=name --format=csv\necho '--- full parse_args" + \
            SCRIPT.split("echo '--- full parse_args")[1]
        script = script.replace("-rs", "-rfEs").replace("tail -40", "tail -80")
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return out.stdout + "\n--- stderr ---\n" + out.stderr[-8000:]


@app.local_entrypoint()
def main():
    print(run.remote())
