"""In-VM docker wrapping for clouds whose VMs cannot run the RL image natively."""

from __future__ import annotations

import shlex

from .storage import RL_INITIAL_ADAPTER_PATH


DOCKER_LOGIN_UNSET = (
    "unset SKYPILOT_DOCKER_USERNAME SKYPILOT_DOCKER_PASSWORD SKYPILOT_DOCKER_SERVER\n"
)


# fix-verda-provider C block (design D6): clouds whose sky adapter refuses
# `image_id: docker:` (sky/clouds/verda.py: DOCKER_IMAGE unsupported). The
# island is launched as a bare VM task instead; the host-side setup logs in
# to the registry and pulls the digest-pinned image, and the host-side run
# executes the ORIGINAL setup + run inside `docker run --gpus all --net=host`
# with the same ~ paths bind-mounted (sky's ssh user on Verda is root, so
# ~ is /root on both sides). Environment variables are forwarded by NAME
# (`-e KEY`), so no value ever appears in a script or a log. Other clouds
# keep sky's own docker runtime: nothing below runs for them.
IN_VM_DOCKER_CLOUDS = frozenset({"verda"})
IN_VM_DOCKER_DIR = "~/yeto-island"
# Host paths bind-mounted into the container at the same path under /root.
IN_VM_DOCKER_MOUNTS = (
    "~/sky_workdir",
    "~/yeto-output",
    "~/yeto-rl",
    "~/.cache/huggingface",
    RL_INITIAL_ADAPTER_PATH,
)
# sky's own variables the task scripts read (set by sky in the host job env).
IN_VM_DOCKER_SKY_ENVS = (
    "SKYPILOT_NODE_IPS",
    "SKYPILOT_NODE_RANK",
    "SKYPILOT_NUM_NODES",
    "SKYPILOT_NUM_GPUS_PER_NODE",
    "SKYPILOT_TASK_ID",
    "SKYPILOT_CLUSTER_INFO",
)
_IN_VM_SETUP_EOF = "YETO_ISLAND_SETUP_EOF"
_IN_VM_RUN_EOF = "YETO_ISLAND_RUN_EOF"
# Inside the container: same conveniences sky's docker runtime provides
# (root without sudo, non-interactive apt) plus git trusting the mounted
# checkouts.
IN_VM_DOCKER_PRELUDE = (
    "#!/bin/bash\n"
    "export HOME=/root DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1\n"
    "command -v sudo >/dev/null 2>&1 || { printf '#!/bin/sh\\nexec \"$@\"\\n' > /usr/local/bin/sudo"
    " && chmod +x /usr/local/bin/sudo; }\n"
    "git config --global --add safe.directory '*' 2>/dev/null || true\n"
    "mkdir -p ~/sky_workdir && cd ~/sky_workdir\n"
)
# Installs the NVIDIA container toolkit when the VM image lacks it (Verda's
# default image ships docker + driver; the toolkit is checked, not assumed).
IN_VM_DOCKER_TOOLKIT_SETUP = (
    'DOCKER=docker; docker info >/dev/null 2>&1 || DOCKER="sudo docker"\n'
    "$DOCKER info 2>/dev/null | grep -qi nvidia || {\n"
    "  echo '[yeto-island] installing nvidia-container-toolkit' >&2\n"
    "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | "
    "sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg\n"
    "  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | "
    "sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | "
    "sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null\n"
    "  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-container-toolkit\n"
    "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker\n"
    "  $DOCKER info 2>/dev/null | grep -qi nvidia || { echo '[yeto-island] docker has no nvidia runtime' >&2; exit 1; }\n"
    "}\n"
)


def in_vm_docker_image(rl_image: str) -> str:
    """The plain image reference for `docker pull` (sky's `docker:` prefix off)."""
    ref = (rl_image or "").strip()
    return ref[len("docker:"):] if ref.startswith("docker:") else ref


def in_vm_docker_setup(rl_image: str, *, login: bool) -> str:
    """Host-side setup for an IN_VM_DOCKER cloud: toolkit check, registry
    login from the SKYPILOT_DOCKER_* secrets sky exported (never echoed),
    then `docker pull` of the digest-pinned image. Ends with the login
    variables unset, like every other yeto task script."""
    image = in_vm_docker_image(rl_image)
    lines = ["set -e", IN_VM_DOCKER_TOOLKIT_SETUP.rstrip("\n")]
    if login:
        lines.append(
            'if [ -n "${SKYPILOT_DOCKER_PASSWORD:-}" ]; then '
            'printf \'%s\' "$SKYPILOT_DOCKER_PASSWORD" | '
            '$DOCKER login "${SKYPILOT_DOCKER_SERVER:-ghcr.io}" -u "$SKYPILOT_DOCKER_USERNAME" --password-stdin; fi'
        )
        lines.append(DOCKER_LOGIN_UNSET.rstrip("\n"))
    lines.append('echo "[yeto-island] docker pull start $(date -u +%FT%TZ)"')
    lines.append(f"$DOCKER pull -q {shlex.quote(image)}")
    lines.append('echo "[yeto-island] docker pull done $(date -u +%FT%TZ)"')
    lines.append(f"$DOCKER run --rm --gpus all {shlex.quote(image)} nvidia-smi -L")
    return "\n".join(lines) + "\n"


def in_vm_docker_run(rl_image: str, setup: str, run: str, env_names, learner_id: int) -> str:
    """Host-side run for an IN_VM_DOCKER cloud: writes the original setup and
    run scripts to IN_VM_DOCKER_DIR (0600) and executes them in one
    `docker run` with the island's ~ paths mounted, GPUs, host network/IPC
    and sky's shm/capability options. The container's exit code is the
    job's (exec). `env_names` are forwarded by name only."""
    image = in_vm_docker_image(rl_image)
    for marker, body in ((_IN_VM_SETUP_EOF, setup), (_IN_VM_RUN_EOF, run)):
        if marker in body:
            raise ValueError(f"island script contains the heredoc marker {marker}")
    names = list(dict.fromkeys(list(IN_VM_DOCKER_SKY_ENVS) + sorted(env_names)))
    env_flags = " ".join(f"-e {n}" for n in names)
    mounts = " ".join(
        f'-v "$HOME/{m[2:]}:/root/{m[2:]}"' for m in IN_VM_DOCKER_MOUNTS
    )
    mkdirs = " ".join(m for m in IN_VM_DOCKER_MOUNTS) + f" {IN_VM_DOCKER_DIR}"
    cname = f"yeto-island-{learner_id}"
    inner = (
        f"bash /root/{IN_VM_DOCKER_DIR[2:]}/setup.sh && "
        'echo "[yeto-island] in-VM docker setup done $(date -u +%FT%TZ)" && '
        f"exec bash /root/{IN_VM_DOCKER_DIR[2:]}/run.sh"
    )
    return (
        "set -e\n"
        'DOCKER=docker; docker info >/dev/null 2>&1 || DOCKER="sudo docker"\n'
        f"mkdir -p {mkdirs}\n"
        f"cat > {IN_VM_DOCKER_DIR}/setup.sh <<'{_IN_VM_SETUP_EOF}'\n"
        f"{IN_VM_DOCKER_PRELUDE}{setup}\n"
        f"{_IN_VM_SETUP_EOF}\n"
        f"cat > {IN_VM_DOCKER_DIR}/run.sh <<'{_IN_VM_RUN_EOF}'\n"
        f"{IN_VM_DOCKER_PRELUDE}{run}\n"
        f"{_IN_VM_RUN_EOF}\n"
        f"chmod 0600 {IN_VM_DOCKER_DIR}/setup.sh {IN_VM_DOCKER_DIR}/run.sh\n"
        f"$DOCKER rm -f {cname} >/dev/null 2>&1 || true\n"
        "SHM=$(awk '/MemTotal/ {printf \"%dm\", $2/2048}' /proc/meminfo)\n"
        'echo "[yeto-island] in-VM docker run start $(date -u +%FT%TZ)"\n'
        f"exec $DOCKER run --rm --name {cname} --gpus all --net=host --ipc=host --shm-size=$SHM "
        "--cap-add=SYS_ADMIN --cap-add=SYS_RESOURCE --ulimit memlock=-1:-1 "
        f"{env_flags} {mounts} "
        f'-v "$HOME/{IN_VM_DOCKER_DIR[2:]}:/root/{IN_VM_DOCKER_DIR[2:]}" '
        f"-w /root/sky_workdir --entrypoint /bin/bash {shlex.quote(image)} -c {shlex.quote(inner)}\n"
    )
