"""verl pins (rl-verl-backend design D1, D10; user decision S16).

The user's fork https://github.com/michaellchung/verl at acad9875 (the commit
S16 measured on Modal 1xH100, VERL-MODAL-CHECK-S16.md).  Dependencies come
from the fork's own ``uv.lock`` (``uv sync --frozen --all-packages --extra vllm
--extra fsdp``), the same recipe as upstream ``docker/Dockerfile.uv.cu130``.
"""

VERL_REPOSITORY = "https://github.com/michaellchung/verl"
VERL_COMMIT = "acad9875a8bdfc81afbcb0a50d146b2630f44093"
VERL_BASE_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
VERL_PYTHON = "3.12"
VERL_UV_VERSION = "0.9.5"
VERL_ROOT = "/workspace/verl"
# Versions S16 measured from that lock (env.json); the island asserts them.
EXPECTED_VERSIONS = {
    "torch": "2.13.0+cu130",
    "vllm": "0.29.0",
    "transformers": "5.12.1",
    "peft": "0.19.1",
}
# ``--rl-image`` value that selects the Modal-built verl image (no registry
# digest exists: the image is built on Modal from the pinned commit + lock).
VERL_IMAGE = f"verl-build:{VERL_COMMIT}"
