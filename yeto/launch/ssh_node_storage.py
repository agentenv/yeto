"""Node-local storage of an SSH trainer island (``ssh_trainer_island``).

A trainer node of an ``ssh:`` island keeps several large things on its own disk:

* the stage of an ``s3://`` / ``oss://`` store (``s3_store.STAGE_ENV``): each node
  writes its cut shards and node 0 its policy releases there, uploads them by
  object API and then drops the local copy (``evict_local_file`` / ``evict_local``);
* the staging of a full policy release (``YETO_RL_RELEASE_STAGING``, else the
  system temp dir): a 128-expert Kimi-K3.1 release is about 207 GB. It is hard
  linked into the store stage, which only works when both are on one mount;
* temporary files (``TMPDIR``: HF round exports, compile caches), the init
  release (``~/yeto-init-release``: the newest release, downloaded by every node
  before its Ray starts, reused by the next start when it is still there) and
  the island's Ray directory (``~/miles-ray``: session logs, object spilling).

Modal gives each container one local disk for all of them. On a docker node the
defaults land in the container's writable layer: the Docker data root on the
root disk, gone with the container (``docker run --rm``), and a hard link from
``/tmp`` into a bind-mounted stage fails (``EXDEV``), so each release is copied
once more. A node's inventory ``scratch`` (an absolute host path on its NVMe)
moves all of it under ``<scratch>/<run id>``, mounted into a docker trainer at
:data:`CONTAINER_SCRATCH` (one mount: the release staging and the stage share
it, so the link works), and kept across the island's incarnations.

The trainer's ``~/yeto-output`` (the event tape and the files a run writes next
to it) goes into the incarnation's state directory on the node, which a docker
trainer already mounts, so the tape outlives the container like Modal's tape
Volume does.

The object store credential: by default the launcher's store credential goes
into the start script, which reaches the node on the SSH session's stdin (no
argv, no file) and the container by variable name (``docker run -e NAME``). A
node's inventory ``store_env_file`` (an owner-only env file the operator put on
the node) replaces it, as ``--modal-store-secret`` does on Modal: the launcher
ships no credential (:func:`nodes_hold_store_credentials`), a docker trainer
reads the file with ``--env-file`` and a rootfs trainer sources it; the preflight
refuses a file that is not ``0600``-strict (group or others may read it).

Nothing here deletes ``<scratch>/<run id>``: the writers free what they wrote
(evicted cut shards and releases, the release staging at each start), and the
preflight checks the free space (``scratch_min_free_gib``). Files a docker
trainer wrote there belong to the container's user (root).
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping

from .node_inventory import Node

CONTAINER_SCRATCH = "/yeto-scratch"  # the node's <scratch>/<run id> inside a docker trainer
# sampler_island.STAGING_ENV (not imported: the launch layer does not import the Miles adapter)
RELEASE_STAGING_ENV = "YETO_RL_RELEASE_STAGING"
TMPDIR_ENV = "TMPDIR"
# under the scratch root
STORE_STAGE_DIR = "store-stage"
RELEASE_STAGING_DIR = "release-staging"
TMP_DIR = "tmp"
# $HOME entries of a docker trainer that point into the scratch root: (name in $HOME,
# directory under the scratch root); the run script and the island use the $HOME paths
HOME_LINKS = (("yeto-init-release", "init-release"), ("miles-ray", "miles-ray"))
OUTPUT_DIR = "yeto-output"  # $HOME/yeto-output of a docker trainer: in the incarnation's state dir
_GIB = 1 << 30


def run_dir_name(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", str(run_id)) or "run"


def host_scratch(node: Node, run_id: str) -> str | None:
    """``<scratch>/<run id>`` on the node (None: the node declares no scratch)."""
    return None if node.scratch is None else f"{node.scratch}/{run_dir_name(run_id)}"


def job_scratch(node: Node, run_id: str) -> str | None:
    """The scratch root as the trainer sees it: :data:`CONTAINER_SCRATCH` in a docker
    trainer, the host path on a rootfs node."""
    if node.scratch is None:
        return None
    return CONTAINER_SCRATCH if node.runtime == "docker" else host_scratch(node, run_id)


def staged_envs(env: Mapping[str, str]) -> list[str]:
    """The stage variables of the stores ``env`` stages: a staged ``s3://`` store
    (its stage set), an ``oss://`` store (always staged, default stage if unset)."""
    from ..cloud import oss_store, s3_store

    out = [s3_store.STAGE_ENV] if env.get(s3_store.STAGE_ENV) else []
    return out + ([oss_store.STAGE_ENV] if env.get(oss_store.STAGE_ENV) or env.get(oss_store.URI_ENV) else [])


def scratch_env(node: Node, run_id: str, env: Mapping[str, str]) -> dict[str, str]:
    """The variables that move the node-local data under the scratch root: the
    release staging, ``TMPDIR`` and, when ``env`` stages a store, the stage. A
    variable the node's inventory ``env`` sets is left to it. {} without scratch."""
    root = job_scratch(node, run_id)
    if root is None:
        return {}
    out = {RELEASE_STAGING_ENV: f"{root}/{RELEASE_STAGING_DIR}", TMPDIR_ENV: f"{root}/{TMP_DIR}"}
    for name in staged_envs(env):  # a staged store's stage moves; an unstaged one stays unstaged
        out[name] = f"{root}/{STORE_STAGE_DIR}"
    return {k: v for k, v in out.items() if k not in node.env}


# processes of a staged store's many-file upload on an SSH trainer node (s3_store
# UPLOAD_PROCESSES_ENV) unless the launcher or the inventory sets it: node 0 pushes
# each 207 GB Kimi-K3.1 release alone, and one process tops out near 270 MB/s
UPLOAD_PROCESSES = 4


def transfer_env(node: Node, env: Mapping[str, str]) -> dict[str, str]:
    """The upload processes of a staged store on this node ({} when ``env`` stages
    no store, or the run or the node's inventory env already says)."""
    from ..cloud.s3_store import UPLOAD_PROCESSES_ENV

    if not staged_envs(env) or env.get(UPLOAD_PROCESSES_ENV) or UPLOAD_PROCESSES_ENV in node.env:
        return {}
    return {UPLOAD_PROCESSES_ENV: str(UPLOAD_PROCESSES)}


def docker_options(node: Node, run_id: str) -> list[str]:
    """``docker run`` options of the node-local storage: the scratch root's volume and
    the store credential's env file ([] without either)."""
    if node.runtime != "docker":
        return []
    host = host_scratch(node, run_id)
    out = [] if host is None else ["-v", f"{host}:{CONTAINER_SCRATCH}"]
    return out + (["--env-file", node.store_env_file] if node.store_env_file else [])


def nodes_hold_store_credentials(nodes: list[Node]) -> bool:
    """Whether the trainer nodes read the store credential from their own env file
    (inventory ``store_env_file``): then the launcher ships none. All or none of
    the nodes: a mix raises ValueError."""
    holding = [n.name for n in nodes if n.store_env_file]
    if holding and len(holding) != len(nodes):
        raise ValueError(f"inventory store_env_file is set on {holding} but not on "
                         f"{[n.name for n in nodes if not n.store_env_file]}: set it on every trainer node")
    return bool(nodes) and bool(holding)


def host_prepare(node: Node, run_id: str) -> str:
    """Start-script line that creates ``<scratch>/<run id>`` on the node before the job
    ("" without scratch): docker would create a missing bind source as root."""
    host = host_scratch(node, run_id)
    return "" if host is None else f"mkdir -p {shlex.quote(host)}\n"


def job_prelude(node: Node, run_id: str, state_dir: str | None) -> str:
    """Bash lines the trainer job runs first: the scratch directories, and in a docker
    trainer the ``$HOME`` links into them (:data:`HOME_LINKS`) and ``~/yeto-output``
    into the incarnation's state directory ``state_dir`` (its path in the container).
    A ``$HOME`` entry that already exists is left alone."""
    lines = []
    if node.store_env_file and node.runtime != "docker":  # docker reads it with --env-file
        lines.append(f"set -a; . {shlex.quote(node.store_env_file)}; set +a")
    root = job_scratch(node, run_id)
    if root is not None:
        dirs = [TMP_DIR, RELEASE_STAGING_DIR] + ([d for _, d in HOME_LINKS] if node.runtime == "docker" else [])
        lines.append("mkdir -p " + " ".join(shlex.quote(f"{root}/{d}") for d in dirs))
        if node.runtime == "docker":
            lines += [f'[ -e "$HOME/{name}" ] || ln -s {shlex.quote(f"{root}/{d}")} "$HOME/{name}"'
                      for name, d in HOME_LINKS]
    if node.runtime == "docker" and state_dir is not None:
        out = shlex.quote(f"{state_dir}/{OUTPUT_DIR}")
        lines.append(f'mkdir -p {out} && {{ [ -e "$HOME/{OUTPUT_DIR}" ] || ln -s {out} "$HOME/{OUTPUT_DIR}"; }}')
    return "".join(line + "\n" for line in lines)


class ScratchError(RuntimeError):
    """A node's scratch directory cannot hold the run."""


def preflight_script(node: Node, run_id: str) -> str | None:
    """Create ``<scratch>/<run id>``, prove it writable and print its free space in KiB
    (``df -Pk``); None without scratch."""
    host = host_scratch(node, run_id)
    if host is None:
        return None
    q = shlex.quote(host)
    probe = shlex.quote(f"{host}/.yeto-write-probe")
    return (f"mkdir -p {q} && touch {probe} && rm -f {probe} && "
            f"df -Pk {q} | awk 'NR == 2 {{ print $4 }}'")


def check_preflight(node: Node, returncode: int, stdout: str, stderr: str = "") -> float:
    """The free GiB :func:`preflight_script` reported; raises :class:`ScratchError` when
    the directory is not writable or has less than ``scratch_min_free_gib``."""
    if returncode != 0:
        raise ScratchError(f"node {node.name}: scratch {node.scratch} is not a writable directory "
                           f"(rc {returncode}): {(stderr or stdout).strip()[-300:]}")
    lines = [line.strip() for line in str(stdout).splitlines() if line.strip()]
    try:
        free_gib = int(lines[-1]) * 1024 / _GIB
    except (IndexError, ValueError):
        raise ScratchError(f"node {node.name}: df of scratch {node.scratch} gave no free size: "
                           f"{str(stdout).strip()[-200:]!r}") from None
    if node.scratch_min_free_gib is not None and free_gib < node.scratch_min_free_gib:
        raise ScratchError(f"node {node.name}: scratch {node.scratch} has {free_gib:.0f} GiB free, the "
                           f"inventory asks for {node.scratch_min_free_gib:.0f} GiB (a Kimi-K3.1 node 0 "
                           "holds a 207 GB release in staging and a cut shard; free older runs' "
                           "<scratch>/<run id> directories)")
    return free_gib


def env_file_preflight_script(node: Node) -> str | None:
    """Print the mode of the node's store env file (it must be a readable file)."""
    if not node.store_env_file:
        return None
    q = shlex.quote(node.store_env_file)
    return f"test -f {q} && test -r {q} && stat -c %a {q}"


def check_env_file(node: Node, returncode: int, stdout: str) -> None:
    """Raise :class:`ScratchError` unless the env file exists and only its owner may read it."""
    if returncode != 0:
        raise ScratchError(f"node {node.name}: store_env_file {node.store_env_file} is not a file the "
                           f"SSH user can read (rc {returncode})")
    mode = str(stdout).strip().splitlines()[-1:] or [""]
    try:
        bits = int(mode[0], 8)
    except ValueError:
        raise ScratchError(f"node {node.name}: no mode for store_env_file {node.store_env_file}: "
                           f"{str(stdout).strip()[-100:]!r}") from None
    if bits & 0o077:
        raise ScratchError(f"node {node.name}: store_env_file {node.store_env_file} has mode {mode[0]}: it "
                           "holds the object store key; chmod 600 it (group and others must not read it)")


NO_SCRATCH_WARNING = (
    "[launcher] warning: SSH docker trainer node(s) {nodes} declare no inventory scratch: the store "
    "stage, the release staging, temp files and the init release go into the container's writable "
    "layer (the Docker data root, usually the root disk) and are lost with the container; set "
    "scratch: /path/on/nvme for each node")


def no_scratch_warning(nodes: list[Node], *, staged: bool) -> str | None:
    """The warning for docker nodes without scratch on a run with a staged store."""
    missing = [n.name for n in nodes if n.runtime == "docker" and n.scratch is None]
    return NO_SCRATCH_WARNING.format(nodes=missing) if missing and staged else None
