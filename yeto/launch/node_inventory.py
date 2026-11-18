"""Node inventory for machines yeto reaches by SSH.

One file describes every node the launcher may use, on-premises or rented:
the SSH target, an optional jump host, the device family, the CPU
architecture, the card model and count, the HBM size per card, the host
memory (the trainer offload plan reads it), the fabric, shared mounts, the
container image and an optional declared price. The core stays neutral:
provider differences live in the file.

``cpu_arch`` (``x86_64`` or ``aarch64``) selects the image: a node
without ``image`` takes ``images[cpu_arch]`` from the top of the file, and a
node whose image is a yeto catalog image of another architecture or device
family is refused (``yeto.launch.engine_images``). The trainer preflight also
compares ``uname -m`` with ``cpu_arch``.

Example::

    images:
      aarch64: REGISTRY/IMAGE:TAG
      x86_64: REGISTRY/yeto-ppu-train:TAG
    nodes:
      - name: ascend00
        host: user@192.0.2.10
        proxy_jump: user@bastion.example
        device_family: npu
        cpu_arch: aarch64
        card: 910B4
        cards_per_node: 8
        hbm_gib: 64
        host_memory_gib: 1024
        fabric: roce
        interface: eth0
        address: 192.0.2.10
        mounts: {/mnt/shared: /mnt/shared}
        runtime: docker
        docker_args: [--device, /dev/davinci0, ...]
        env: {HCCL_IF_IP: 192.0.2.10}
        price_per_card_hour: 0.0

``address`` is the node's IP on the cluster network (the ``interface``), which
the other nodes dial for rendezvous; ``host`` may be an SSH alias that only the
head resolves. ``docker_args`` are the device flags of ``docker run`` for the
node's device family (a ``runtime: docker`` trainer island refuses to start
without them). ``env`` holds the node's fabric settings that yeto cannot derive
(e.g. ``NCCL_IB_HCA``, ``NCCL_IB_GID_INDEX`` on RoCE, ``NCCL_COLLNET_ENABLE``
with SHARP); the trainer island exports them after the fabric defaults of
``topology.island_comm_env`` and before the run's own environment.
``scratch`` is an absolute host path on the node's local disk (NVMe) that holds
the run's node-local data under ``<scratch>/<run id>``: the stage of an object
store, the staging of policy releases, temporary files, the init release and the
Ray directory (``ssh_node_storage``). Without it a docker trainer writes them
into the container's writable layer (the root disk), and loses them with the
container. ``scratch_min_free_gib``: the trainer preflight refuses a node with
less free space there. ``store_env_file`` is an owner-only (``0600``) env file on
the node with the object store credential (``YETO_RL_STORE_S3_ACCESS_KEY_ID`` /
``..._SECRET_ACCESS_KEY``, or ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``):
like a Modal Secret, the key then never passes through the launcher; a docker
trainer reads it with ``docker run --env-file``. Without it the launcher's store
credential reaches the node in the start script on the SSH session's stdin
(never in an argv or a file) and the container by name (``-e NAME``).

An NVIDIA (``cuda``) multi-node island over ``ib`` / ``roce``, e.g. 6 x 8 B300::

    images:
      x86_64: ghcr.io/OWNER/yeto-miles-k31@sha256:...
    nodes:
      - name: b300-0
        host: ops@10.0.0.10
        address: 10.1.0.10            # IP on the interface below
        interface: bond0
        device_family: cuda
        cpu_arch: x86_64
        card: B300
        cards_per_node: 8
        hbm_gib: 288
        host_memory_gib: 2048
        fabric: ib
        runtime: docker
        docker_args: [--gpus, all, --device, /dev/infiniband, --ulimit, memlock=-1:-1,
                      --ulimit, stack=67108864, --cap-add, IPC_LOCK]
        mounts: {/mnt/nvme/models: /data/models}
        scratch: /mnt/nvme/yeto
        scratch_min_free_gib: 1500
        env: {NCCL_IB_HCA: "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"}
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..gpu_spec import (
    DEVICE_FAMILIES,
    DEVICE_FAMILY_ENV,
    UnknownCard,
    canonical_card,
    device_family_of,
)
from ..hw import card_tools
from ..hw.catalog import UnknownCardType, compat_group
from . import engine_images as image_catalog

# Fabrics an inventory may declare. "tcp" is Ethernet without RDMA; "roce" and
# "ib" are RDMA over Ethernet and InfiniBand; "hccs" is the Ascend on-node
# interconnect (a node declaring it still needs a host fabric for cross-node
# traffic, so hccs is accepted only together with one of the others).
FABRICS = ("tcp", "roce", "ib", "hccs")
RUNTIMES = ("docker", "rootfs")

_REQUIRED = ("name", "host", "device_family", "cpu_arch", "card", "cards_per_node", "hbm_gib",
             "host_memory_gib")
_KNOWN = _REQUIRED + ("proxy_jump", "fabric", "interface", "address", "mounts", "image", "runtime",
                      "docker_args", "env", "price_per_card_hour", "roles", "scratch", "scratch_min_free_gib",
                      "store_env_file")
# characters a host path field (scratch, store_env_file) may not hold: it is spliced
# into shell lines and `docker run` options
_SCRATCH_FORBIDDEN = frozenset("$`\\\"' :,;&|<>*?\n\t")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class InventoryError(ValueError):
    """The node inventory is not usable; the launch is refused before any SSH."""


@dataclass(frozen=True)
class Node:
    """One node of the pool."""

    name: str
    host: str  # SSH target, "user@host" or a host alias
    device_family: str  # "cuda", "npu" or "ppu"
    cpu_arch: str  # "x86_64" or "aarch64" (``uname -m``)
    card: str  # canonical catalog name, e.g. "910B4"
    cards_per_node: int
    hbm_gib: float
    host_memory_gib: float
    proxy_jump: str | None = None
    fabric: str = "tcp"
    interface: str | None = None
    address: str | None = None  # IP on the cluster network (rendezvous); None: not declared
    mounts: Mapping[str, str] = field(default_factory=dict)
    image: str | None = None
    runtime: str = "docker"
    docker_args: tuple[str, ...] = ()  # device flags of `docker run` (runtime docker)
    env: Mapping[str, str] = field(default_factory=dict)  # fabric settings yeto cannot derive
    price_per_card_hour: float | None = None
    roles: tuple[str, ...] = ()
    # an absolute host path on the node's local disk (NVMe) for the run's node-local
    # data: store stage, release staging, temp files, init release (ssh_node_storage)
    scratch: str | None = None
    scratch_min_free_gib: float | None = None  # the trainer preflight refuses less free space
    # an owner-only (0600) env file on the node holding the object store credential: the
    # trainer reads it there (docker --env-file), the launcher ships none (ssh_node_storage)
    store_env_file: str | None = None

    @property
    def hbm_gib_per_node(self) -> float:
        return self.hbm_gib * self.cards_per_node

    @property
    def compat_group(self) -> str:
        """Compatibility group with device family and HBM, e.g. ``ascend-910b4-64g``."""
        return compat_group(self.card, hbm_gib=self.hbm_gib)

    def ssh_options(self) -> list[str]:
        """``ssh`` options that reach this node, ``ProxyJump`` included.

        The executor passes these on the command line; it never writes to the
        user's SSH configuration.
        """
        return ["-o", f"ProxyJump={self.proxy_jump}"] if self.proxy_jump else []

    def preflight_commands(self) -> tuple[tuple[str, ...], ...]:
        """The card-tool commands to try in order (the first that answers wins,
        ssh_trainer_island.preflight); a rocm node falls back to rocm-smi when
        amd-smi is not installed (card_tools.ROCM_SMI_FALLBACK)."""
        commands = card_tools.preflight_commands(self.device_family)
        if self.device_family == "rocm":
            commands = commands + tuple(getattr(card_tools, "ROCM_SMI_FALLBACK", ()))
        return commands

    def check_cards(self, text: str) -> list[card_tools.CardRecord]:
        """Compare the node's card tool output with this inventory entry."""
        return card_tools.check_cards(self.device_family, text, card=self.card,
                                      count=self.cards_per_node, hbm_gib=self.hbm_gib,
                                      node=self.name)

    def check_cpu_arch(self, uname_m: str) -> None:
        """Compare the node's ``uname -m`` output with ``cpu_arch``."""
        try:
            reported = image_catalog.canonical_cpu_arch(uname_m)
        except image_catalog.UnknownCpuArch:
            raise InventoryError(f"node {self.name}: uname -m reports {uname_m.strip()!r}, "
                                 f"the inventory declares cpu_arch {self.cpu_arch}") from None
        if reported != self.cpu_arch:
            raise InventoryError(f"node {self.name}: uname -m reports {reported}, the inventory "
                                 f"declares cpu_arch {self.cpu_arch}")

    def record(self) -> dict[str, Any]:
        """Manifest / event-tape row of this node (no SSH user, no mounts)."""
        return {"name": self.name, "device_family": self.device_family,
                "cpu_arch": self.cpu_arch, "card": self.card,
                "cards_per_node": self.cards_per_node, "hbm_gib": self.hbm_gib,
                "compat_group": self.compat_group, "host_memory_gib": self.host_memory_gib, "fabric": self.fabric,
                "runtime": self.runtime, "behind_jump_host": self.proxy_jump is not None}


def _require_positive(value: Any, node: str, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise InventoryError(f"node {node}: {key} must be a positive number, got {value!r}")
    return float(value)


def _require_text(value: Any, node: str, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InventoryError(f"node {node}: {key} must be a non-empty string, got {value!r}")
    return value.strip()


def _cpu_arch(value: Any, where: str) -> str:
    try:
        return image_catalog.canonical_cpu_arch(_require_text(value, where, "cpu_arch"))
    except image_catalog.UnknownCpuArch as exc:
        raise InventoryError(f"node {where}: {exc}") from None


def _host_path(raw: Mapping[str, Any], node: str, key: str) -> str | None:
    """An absolute host path field (``scratch``, ``store_env_file``) or None: it is
    spliced into shell lines and ``docker run`` options, so plain characters only."""
    value = raw.get(key)
    if value is None:
        return None
    path = _require_text(value, node, key).rstrip("/")
    if not path.startswith("/") or ".." in path.split("/") or set(path) & _SCRATCH_FORBIDDEN:
        raise InventoryError(f"node {node}: {key} must be an absolute host path other than /, "
                             f"without '..', blanks, quotes or shell characters, got {value!r}")
    return path


def parse_images(raw: Any) -> dict[str, str]:
    """The top-level ``images:`` map (CPU architecture -> image reference)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise InventoryError(f"images must be a map of cpu_arch to image, got {raw!r}")
    out: dict[str, str] = {}
    for key, value in raw.items():
        try:
            arch = image_catalog.canonical_cpu_arch(key)
        except image_catalog.UnknownCpuArch as exc:
            raise InventoryError(f"images: {exc}") from None
        if arch in out:
            raise InventoryError(f"images: cpu_arch {arch} is listed twice")
        if not isinstance(value, str) or not value.strip():
            raise InventoryError(f"images: {key} must name an image, got {value!r}")
        out[arch] = value.strip()
    return out


def parse_node(raw: Any, *, index: int, images: Mapping[str, str] | None = None) -> Node:
    """One inventory entry as a :class:`Node`; any gap refuses the launch.

    ``images`` is the inventory's ``images:`` map; a node without ``image``
    takes the entry for its ``cpu_arch``."""
    if not isinstance(raw, dict):
        raise InventoryError(f"inventory entry {index} must be a mapping, got {type(raw).__name__}")
    name = raw.get("name")
    name = _require_text(name, f"#{index}", "name") if name is not None else f"node{index}"
    unknown = sorted(set(raw) - set(_KNOWN))
    if unknown:
        raise InventoryError(f"node {name}: unknown inventory field(s) {unknown} "
                             f"(known: {sorted(_KNOWN)})")
    for key in _REQUIRED:
        if raw.get(key) is None:
            raise InventoryError(f"node {name}: {key} is required; the inventory states no "
                                 f"default")
    family = _require_text(raw["device_family"], name, "device_family")
    if family not in DEVICE_FAMILIES:
        raise InventoryError(f"node {name}: device_family {family!r} is unknown "
                             f"(known: {list(DEVICE_FAMILIES)})")
    cpu_arch = _cpu_arch(raw["cpu_arch"], name)
    try:
        card = canonical_card(_require_text(raw["card"], name, "card"))
    except UnknownCard as exc:
        raise InventoryError(f"node {name}: {exc}") from None
    if device_family_of(card) != family:
        raise InventoryError(f"node {name}: card {card} is a {device_family_of(card)} card, "
                             f"the inventory declares device_family {family}")
    cards = raw["cards_per_node"]
    if isinstance(cards, bool) or not isinstance(cards, int) or cards <= 0:
        raise InventoryError(f"node {name}: cards_per_node must be a positive int, "
                             f"got {cards!r}")
    fabric = _require_text(raw.get("fabric", "tcp"), name, "fabric")
    if fabric not in FABRICS:
        raise InventoryError(f"node {name}: fabric {fabric!r} is unknown "
                             f"(known: {list(FABRICS)})")
    runtime = _require_text(raw.get("runtime", "docker"), name, "runtime")
    if runtime not in RUNTIMES:
        raise InventoryError(f"node {name}: runtime {runtime!r} is unknown "
                             f"(known: {list(RUNTIMES)})")
    image = (_require_text(raw["image"], name, "image") if raw.get("image") is not None
             else (images or {}).get(cpu_arch))
    if runtime == "docker" and image is None:
        raise InventoryError(f"node {name}: runtime docker needs an image (set image, or "
                             f"images.{cpu_arch} at the top of the inventory)")
    if image is not None and (why := image_catalog.image_mismatch(
            image, cpu_arch=cpu_arch, device_family=family)):
        raise InventoryError(f"node {name}: {why}")
    mounts = raw.get("mounts") or {}
    if not isinstance(mounts, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in mounts.items()):
        raise InventoryError(f"node {name}: mounts must be a map of host path to "
                             f"container path, got {mounts!r}")
    address = raw.get("address")
    if address is not None:
        try:
            address = str(ipaddress.ip_address(_require_text(address, name, "address")))
        except ValueError:
            raise InventoryError(f"node {name}: address must be an IP address (the rendezvous "
                                 f"address on the cluster network), got {address!r}") from None
    docker_args = raw.get("docker_args") or ()
    if not isinstance(docker_args, (list, tuple)) or not all(isinstance(a, str) for a in docker_args):
        raise InventoryError(f"node {name}: docker_args must be a list of strings, got {docker_args!r}")
    env = raw.get("env") or {}
    if not isinstance(env, dict) or not all(
            isinstance(k, str) and _ENV_NAME.fullmatch(k) and isinstance(v, (str, int, float))
            and not isinstance(v, bool) for k, v in env.items()):
        raise InventoryError(f"node {name}: env must be a map of variable names to strings or numbers, "
                             f"got {env!r}")
    if DEVICE_FAMILY_ENV in env:
        raise InventoryError(f"node {name}: env must not set {DEVICE_FAMILY_ENV} (device_family decides)")
    price = raw.get("price_per_card_hour")
    if price is not None and (isinstance(price, bool) or not isinstance(price, (int, float))
                              or price < 0):
        raise InventoryError(f"node {name}: price_per_card_hour must be a non-negative "
                             f"number, got {price!r}")
    roles = raw.get("roles") or ()
    if not isinstance(roles, (list, tuple)) or not all(isinstance(r, str) for r in roles):
        raise InventoryError(f"node {name}: roles must be a list of strings, got {roles!r}")
    scratch = _host_path(raw, name, "scratch")
    store_env_file = _host_path(raw, name, "store_env_file")
    min_free = raw.get("scratch_min_free_gib")
    if min_free is not None:
        if scratch is None:
            raise InventoryError(f"node {name}: scratch_min_free_gib needs scratch")
        min_free = _require_positive(min_free, name, "scratch_min_free_gib")
    node = Node(
        name=name,
        host=_require_text(raw["host"], name, "host"),
        device_family=family,
        cpu_arch=cpu_arch,
        card=card,
        cards_per_node=cards,
        hbm_gib=_require_positive(raw["hbm_gib"], name, "hbm_gib"),
        host_memory_gib=_require_positive(raw["host_memory_gib"], name, "host_memory_gib"),
        proxy_jump=(_require_text(raw["proxy_jump"], name, "proxy_jump")
                    if raw.get("proxy_jump") is not None else None),
        fabric=fabric,
        interface=(_require_text(raw["interface"], name, "interface")
                   if raw.get("interface") is not None else None),
        address=address,
        mounts=dict(mounts),
        image=image,
        runtime=runtime,
        docker_args=tuple(docker_args),
        env={k: str(v) for k, v in env.items()},
        price_per_card_hour=(float(price) if price is not None else None),
        roles=tuple(roles),
        scratch=scratch,
        scratch_min_free_gib=min_free,
        store_env_file=store_env_file,
    )
    try:
        node.compat_group  # noqa: B018 - an unknown HBM variant fails here, not at launch
    except UnknownCardType as exc:
        raise InventoryError(f"node {name}: {exc}") from None
    return node


def parse_inventory(document: Any) -> list[Node]:
    """The ``nodes:`` list of a loaded inventory document."""
    images: dict[str, str] = {}
    if isinstance(document, list):
        entries = document
    elif isinstance(document, dict):
        unknown = sorted(set(document) - {"nodes", "images"})
        if unknown:
            raise InventoryError(f"unknown top-level inventory field(s) {unknown} "
                                 f"(known: ['images', 'nodes'])")
        entries = document.get("nodes")
        images = parse_images(document.get("images"))
    else:
        raise InventoryError(f"the inventory must be a mapping with a nodes list, got "
                             f"{type(document).__name__}")
    if not isinstance(entries, list) or not entries:
        raise InventoryError("the inventory must contain at least one node under nodes:")
    nodes = [parse_node(entry, index=index, images=images) for index, entry in enumerate(entries)]
    for key, label in (("name", "name"), ("host", "SSH target")):
        seen: dict[str, str] = {}
        for node in nodes:
            value = getattr(node, key)
            if value in seen:
                raise InventoryError(f"the inventory lists {label} {value!r} twice "
                                     f"(nodes {seen[value]} and {node.name})")
            seen[value] = node.name
    return nodes


def load_inventory(path: str | Path) -> list[Node]:
    """Read ``--rl-node-inventory`` (YAML or JSON) and validate every node."""
    import yaml  # noqa: PLC0415 - optional at import time, like yeto.shape.providers

    file = Path(path)
    try:
        text = file.read_text()
    except OSError as exc:
        raise InventoryError(f"cannot read the node inventory {file}: {exc}") from None
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InventoryError(f"the node inventory {file} is not valid YAML: {exc}") from None
    if document is None:
        raise InventoryError(f"the node inventory {file} is empty")
    try:
        return parse_inventory(document)
    except InventoryError as exc:
        raise InventoryError(f"{file}: {exc}") from None


def nodes_by_name(nodes: list[Node]) -> dict[str, Node]:
    return {node.name: node for node in nodes}


def island_nodes(nodes: list[Node], role: str) -> list[Node]:
    """The nodes of one role (``roles:`` in the file); nodes without roles match."""
    return [node for node in nodes if not node.roles or role in node.roles]
