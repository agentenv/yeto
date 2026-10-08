"""Local patch applied to the verl source at image build time (rl-verl-backend D5).

Not pushed to the verl fork: one hook line after verl's ``add_lora`` in the
vLLM worker extension calls yeto's read-back (``vllm_readback.after_add_lora``)
so the inference side reports a checksum of the adapter it registered.  The
hook is inert unless ``YETO_VERL_READBACK_DIR`` is set and never raises into
vLLM.  The anchor must occur exactly once, otherwise the build fails.

Usage: python patch_verl.py <verl source root>
"""

import sys
from pathlib import Path

ANCHOR = "            self.add_lora(lora_request)\n"
HOOK = (
    "            try:  # yeto rl-verl-backend D5: registered-adapter read-back\n"
    "                from yeto.rl.adapters.verl.vllm_readback import after_add_lora as _yeto_after_add_lora\n"
    "            except ImportError:\n"
    "                _yeto_after_add_lora = None\n"
    "            if _yeto_after_add_lora is not None:\n"
    "                _yeto_after_add_lora(self, VLLM_LORA_INT_ID)\n"
)
TARGET = "verl/workers/rollout/vllm_rollout/utils.py"


def apply(root: Path) -> None:
    path = root / TARGET
    text = path.read_text()
    if HOOK in text:
        return
    if text.count(ANCHOR) != 1:
        raise SystemExit(f"patch_verl: anchor occurs {text.count(ANCHOR)} times in {TARGET}")
    path.write_text(text.replace(ANCHOR, ANCHOR + HOOK))


if __name__ == "__main__":
    apply(Path(sys.argv[1]))
    print("patch_verl: ok")
