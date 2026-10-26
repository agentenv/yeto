"""Modal image of the verl engine (rl-verl-backend 1.1, design D10).

Same recipe S16 measured (VERL-MODAL-CHECK-S16.md section 1): CUDA 13.0 devel
base + Python 3.12, the pinned fork checked out at ``VERL_COMMIT``, all
dependencies from the fork's own ``uv.lock`` (``uv sync --frozen
--all-packages --extra vllm --extra fsdp``), plus yeto's one-hook local patch
(``patch_verl``).  ``--rl-image verl-build:<commit>`` selects it; Modal caches
the layers, so only the first run builds. yeto's object-store client (boto3,
``STORE_CLIENT_REQUIREMENTS``) is added to the same venv.  yeto's own image default
(ghcr.io/michaellchung/yeto-miles-ports) is unchanged.
"""

from __future__ import annotations

from pathlib import Path

from .pins import (STORE_CLIENT_REQUIREMENTS, VERL_BASE_IMAGE, VERL_COMMIT, VERL_PYTHON, VERL_REPOSITORY,
                   VERL_ROOT, VERL_UV_VERSION)

VENV_BIN = f"{VERL_ROOT}/.venv/bin"


def build_commands(commit: str = VERL_COMMIT) -> list[str]:
    if commit != VERL_COMMIT:
        raise ValueError(f"verl image commit {commit} is not the pinned {VERL_COMMIT}")
    return [
        f"git clone {VERL_REPOSITORY} {VERL_ROOT} && cd {VERL_ROOT} && git checkout {commit}"
        f" && git rev-parse HEAD > /workspace/verl_sha.txt",
        f"cd {VERL_ROOT} && UV_PYTHON=/usr/local/bin/python{VERL_PYTHON} uv sync --frozen --all-packages"
        " --extra vllm --extra fsdp",
    ]


def store_client_commands() -> list[str]:
    """An island over an s3:// or oss:// store reads it with boto3 (S25 C3V); its own
    layer, so the cached verl build above stays valid."""
    return [f"uv pip install --python {VENV_BIN}/python " + " ".join(STORE_CLIENT_REQUIREMENTS)]


def image_env() -> dict[str, str]:
    return {
        "PATH": f"{VENV_BIN}:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin",
        "VIRTUAL_ENV": f"{VERL_ROOT}/.venv",
        "VERL_USE_UV": "0",
    }


def megatron_commands(max_jobs: int = 16) -> list[str]:
    """S27 G-DC: the Megatron engine layers on top of :func:`build_commands` (variant
    ``megatron``): the fork's ``megatron`` extra (``--inexact``: the vLLM and FSDP
    packages stay), then causal-conv1d and mamba-ssm built for Hopper (the extra does
    not list them; mamba-ssm needs its CUDA selective-scan module, so the build
    cannot be skipped), then the flash-attn FA4 preview removed (it raises
    AttributeError against the locked cutlass-dsl, Megatron-Core only catches
    ImportError; Hopper does not use it). ``megatron_recipe.image_commands`` (the bare-verl
    smoke of S27 G-R) returns these same steps."""
    from .pins import CAUSAL_CONV1D_VERSION, MAMBA_SSM_VERSION

    env = f"MAX_JOBS={max_jobs} TORCH_CUDA_ARCH_LIST=9.0 CC=gcc CXX=g++"
    return [
        f"cd {VERL_ROOT} && UV_PYTHON=/usr/local/bin/python{VERL_PYTHON} uv sync --frozen --all-packages"
        " --inexact --extra vllm --extra fsdp --extra megatron",
        f"uv pip install --python {VENV_BIN}/python ninja packaging wheel setuptools",
        f"{env} CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install --python {VENV_BIN}/python --no-build-isolation"
        f" 'causal-conv1d=={CAUSAL_CONV1D_VERSION}'",
        f"{env} MAMBA_FORCE_BUILD=TRUE uv pip install --python {VENV_BIN}/python"
        f" --no-build-isolation --no-deps 'mamba-ssm=={MAMBA_SSM_VERSION}'",
        f"rm -rf {VERL_ROOT}/.venv/lib/python{VERL_PYTHON}/site-packages/flash_attn/cute",
    ]


VARIANTS = (None, "megatron")
# S27 G-DC: the Megatron island sends the weights from the trainer container to the
# rollout container over NCCL (verl's checkpoint engine, cupy NCCL). Without the
# verbs libraries NCCL cannot use the cluster's RDMA network and falls back to
# sockets: s27-dc-smoke2's first 63 GB update took 57.7 s (about 1.1 GB/s).
RDMA_PACKAGES = ("rdma-core", "ibverbs-providers")


def modal_image(modal, commit: str = VERL_COMMIT, patched: bool = True, variant: str | None = None):
    """The verl image; ``patched=False`` leaves the verl source as the fork ships it
    (the bare-verl arms of the efficiency evaluation, which must not import yeto).
    ``variant="megatron"`` (``--rl-image verl-megatron-build:<commit>``, S27 G-DC) adds
    the Megatron engine layers after the cached FSDP ones and applies the
    ``megatron`` patch profile (``patch_verl``)."""
    if variant not in VARIANTS:
        raise ValueError(f"verl image variant {variant!r} is not one of {VARIANTS}")
    image = (
        modal.Image.from_registry(VERL_BASE_IMAGE, add_python=VERL_PYTHON)
        .apt_install("git", "build-essential", "curl", "libnuma1")
        .pip_install(f"uv=={VERL_UV_VERSION}")
        .run_commands(*build_commands(commit))
        .run_commands(*store_client_commands())
    )
    if variant == "megatron":
        image = image.run_commands(*megatron_commands()).apt_install(*RDMA_PACKAGES)
    if patched:
        patch = Path(__file__).with_name("patch_verl.py")
        profile = "megatron" if variant == "megatron" else "fsdp"
        suffix = f" --profile {profile}" if variant == "megatron" else ""
        image = (image.add_local_file(str(patch), "/workspace/patch_verl.py", copy=True)
                 .run_commands(f"{VENV_BIN}/python /workspace/patch_verl.py {VERL_ROOT}{suffix}"))
    return image.env(image_env())
