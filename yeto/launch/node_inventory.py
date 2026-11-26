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
family is refused (``yeto.launch.engine_images``). A ``runtime: docker`` node
of a file without ``images:`` may have no image: an ``ssh:`` trainer island runs
the run's ``--rl-image`` (an inventory image must then name the same one), an SSH
sampler pool its ``container.image``. The trainer preflight also compares ``uname -m`` with
``cpu_arch``. The top-level ``site_images:`` names the site image catalog
(:mod:`yeto.launch.site_images`, default ``~/.yeto/site-images.yaml``; a relative path
is taken from the inventory's directory): an island or pool of these nodes that names
no image takes the catalog's verified image for its model there.

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

``lease_store`` (top level, optional) is a store URI (``s3://`` / ``oss://``, or an
absolute directory / ``file://`` path every launcher of the site shares) where the
nodes of this file are leased across runs (:mod:`yeto.launch.node_lease`): a node
another run holds is skipped. Without it nodes are not leased (as before)::

    lease_store: s3://BUCKET/yeto-node-leases/site-a
    nodes: [...]

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

An NVIDIA (``cuda``) multi-node island over ``ib`` / ``roce``, e.g. 6 x 8 B300
(with ``--model-store local:///data/models --model-store-source s3://...`` the
trainer preflight fills ``/mnt/nvme/models`` on every node, :mod:`.ssh_weights`)::

    images:
      x86_64: ghcr.io/OWNER/yeto-miles-k31@sha256:...
    nodes:
      - name: b300-0
        host: ops@192.0.2.20
        address: 198.51.100.20        # IP on the interface below
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

import contextlib
import ipaddress
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

from ..gpu_spec import (
    DEVICE_FAMILIES,
    DEVICE_FAMILY_ENV,
    UnknownCard,
    UnknownFamily,
    canonical_card,
    device_family_of,
    hw_family,
    launcher_family,
    unified_family,
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
                      "store_env_file", "node_health")
NODE_HEALTH_MODES = ("auto", "off")
# node profile keys (unified-sampler-pool): parse_profile reads them, parse_node accepts and ignores them
_PROFILE_KEYS = ("card_arch", "container", "paths", "model_source", "reach")
# characters a host path field (scratch, store_env_file) may not hold: it is spliced
# into shell lines and `docker run` options
_SCRATCH_FORBIDDEN = frozenset("$`\\\"' :,;&|<>*?\n\t")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class InventoryError(ValueError):
    """The node inventory is not usable; the launch is refused before any SSH."""


class NodeList(list):
    """The nodes of an inventory file (a plain ``list`` of :class:`Node`) and the file's
    top-level ``lease_store`` (None: not declared). A list, so every reader of the nodes
    is unchanged; :func:`lease_store_of` reads the store."""

    lease_store: str | None = None

    def __init__(self, nodes: Any = (), *, lease_store: str | None = None) -> None:
        super().__init__(nodes)
        self.lease_store = lease_store


def lease_store_of(nodes: Any) -> str | None:
    """The ``lease_store`` of an inventory's node list (None: not declared, or a list
    that did not come from :func:`parse_inventory` / :func:`load_inventory`)."""
    return getattr(nodes, "lease_store", None)


_LEASE_SCHEMES = ("s3://", "oss://", "file://")


def parse_lease_store(value: Any) -> str | None:
    """The top-level ``lease_store``: an ``s3://`` / ``oss://`` / ``file://`` URI or an
    absolute directory (None: not declared)."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InventoryError(f"lease_store must be a store URI (s3://, oss://, file://) or an absolute "
                             f"directory, got {value!r}")
    uri = value.strip().rstrip("/")
    if uri.startswith(_LEASE_SCHEMES):
        rest = uri.split("://", 1)[1]
        if not rest or (uri.startswith("file://") and not rest.startswith("/")):
            raise InventoryError(f"lease_store {value!r} names no bucket or directory")
        return uri
    if uri.startswith("/") and ".." not in uri.split("/"):
        return uri
    raise InventoryError(f"lease_store must be a store URI (s3://, oss://, file://) or an absolute "
                         f"directory, got {value!r}")


@dataclass(frozen=True)
class Node:
    """One node of the pool."""

    name: str
    host: str  # SSH target, "user@host" or a host alias
    device_family: str  # launcher family: "cuda", "rocm", "npu" or "ppu" (NodeProfile.family: unified)
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
    # the running node's health check (yeto.hw.node_health): "auto" (on where the family has a
    # checker) or "off"; None: not declared (auto, and nothing about it is printed)
    node_health: str | None = None

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


def parse_node_health(value: Any, where: str) -> str | None:
    """``auto`` / ``off`` (None: not given). YAML reads a bare ``off`` as false and a bare
    ``on`` as true; ``on`` means ``auto`` (on where the family has a checker)."""
    if value is None:
        return None
    if value is False:
        return "off"
    if value is True or value == "on":
        return "auto"
    if value not in NODE_HEALTH_MODES:
        raise InventoryError(f"{where}: node_health must be one of {list(NODE_HEALTH_MODES)}, got {value!r}")
    return str(value)


def parse_node(raw: Any, *, index: int, images: Mapping[str, str] | None = None,
               node_health: str | None = None) -> Node:
    """One inventory entry as a :class:`Node`; any gap refuses the launch.

    ``images`` is the inventory's ``images:`` map; a node without ``image``
    takes the entry for its ``cpu_arch``. ``node_health``: the inventory's top-level
    value, for a node that declares none."""
    if not isinstance(raw, dict):
        raise InventoryError(f"inventory entry {index} must be a mapping, got {type(raw).__name__}")
    name = raw.get("name")
    name = _require_text(name, f"#{index}", "name") if name is not None else f"node{index}"
    unknown = sorted(set(raw) - set(_KNOWN) - set(_PROFILE_KEYS))
    if unknown:
        raise InventoryError(f"node {name}: unknown inventory field(s) {unknown} "
                             f"(known: {sorted(_KNOWN)})")
    ignored = sorted(set(raw) & set(_PROFILE_KEYS))
    if ignored:
        warnings.warn(f"node {name}: {ignored} are node profile keys: read by parse_profile / load_profiles "
                      "(the ssh sampler pool), ignored here (the trainer island)", stacklevel=2)
    for key in _REQUIRED:
        if raw.get(key) is None:
            raise InventoryError(f"node {name}: {key} is required; the inventory states no "
                                 f"default")
    family = _require_text(raw["device_family"], name, "device_family")
    if family not in DEVICE_FAMILIES:
        try:  # an alias (ascend, amd, nvidia, thead) is the launcher family it names
            family = launcher_family(family)
        except UnknownFamily:
            raise InventoryError(f"node {name}: device_family {family!r} is unknown "
                                 f"(known: {list(DEVICE_FAMILIES)})") from None
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
    if runtime == "docker" and image is None and images:
        raise InventoryError(f"node {name}: runtime docker and images has no {cpu_arch} entry (set image, or "
                             f"images.{cpu_arch} at the top of the inventory, or leave images out: an ssh: "
                             "trainer island then runs --rl-image)")
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
        node_health=parse_node_health(raw["node_health"], f"node {name}") if "node_health" in raw else node_health,
    )
    try:
        node.compat_group  # noqa: B018 - an unknown HBM variant fails here, not at launch
    except UnknownCardType as exc:
        raise InventoryError(f"node {name}: {exc}") from None
    return node


def parse_inventory(document: Any) -> NodeList:
    """The ``nodes:`` list of a loaded inventory document (a :class:`NodeList`: with
    the top-level ``lease_store``)."""
    images: dict[str, str] = {}
    health = None
    lease_store = None
    if isinstance(document, list):
        entries = document
    elif isinstance(document, dict):
        unknown = sorted(set(document) - {"nodes", "images", "node_health", "lease_store", "site_images"})
        if unknown:
            raise InventoryError(f"unknown top-level inventory field(s) {unknown} "
                                 f"(known: ['images', 'lease_store', 'node_health', 'nodes', 'site_images'])")
        entries = document.get("nodes")
        images = parse_images(document.get("images"))
        health = parse_node_health(document.get("node_health"), "inventory")
        lease_store = parse_lease_store(document.get("lease_store"))
        catalog = document.get("site_images")  # the site image catalog's path (yeto.launch.site_images)
        if catalog is not None and (not isinstance(catalog, str) or not catalog.strip()):
            raise InventoryError(f"inventory site_images: must be a file path, not {catalog!r}")
    else:
        raise InventoryError(f"the inventory must be a mapping with a nodes list, got "
                             f"{type(document).__name__}")
    if not isinstance(entries, list) or not entries:
        raise InventoryError("the inventory must contain at least one node under nodes:")
    nodes = [parse_node(entry, index=index, images=images, node_health=health)
             for index, entry in enumerate(entries)]
    for key, label in (("name", "name"), ("host", "SSH target")):
        seen: dict[str, str] = {}
        for node in nodes:
            value = getattr(node, key)
            if value in seen:
                raise InventoryError(f"the inventory lists {label} {value!r} twice "
                                     f"(nodes {seen[value]} and {node.name})")
            seen[value] = node.name
    return NodeList(nodes, lease_store=lease_store)


def load_inventory(path: str | Path) -> NodeList:
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


# --- node profile (unified-sampler-pool) ----------------------------------------
#
# A NodeProfile is a Node plus what a sampler start needs to know about the
# machine whatever its source (own SSH node, NC lease, Daytona sandbox, Modal
# function, sky VM): the unified device family, the card architecture, how the
# sampler is started (container engine, plain process, Modal function), the
# node's directories, where the model comes from and how the node is reached.
# Device flags and the visible-card variable are derived from the family here,
# not written into each route.

CONTAINERS = ("docker", "podman", "process", "modal-fn")
"""How the sampler starts: a ``docker`` / ``podman`` container, a ``process`` on
the host (a venv), or a Modal function container (``modal-fn``: the platform
gives it its cards, no device flags)."""
MODEL_SOURCES = ("store", "volume", "hf", "preset")
"""Where the node gets the model: the run's object store, a platform volume, the
Hugging Face hub, or already on the node (``preset``)."""
REACH_MODES = ("direct", "jump", "relay", "outbound", "function")
"""How the control host reaches the node: SSH ``direct``, through a ``jump`` host
(``ProxyJump``), through a ``relay``, not at all (``outbound``: the node only
dials the store), or through the platform's function calls (``function``)."""
PATH_KEYS = ("work", "models", "cache")

# card -> architecture (CUDA ``sm<major><minor>``, ROCm ``gfx...``, Ascend SoC
# family). None: not known (the PPU cards).
CARD_ARCHES: dict[str, str | None] = {
    "V100": "sm70", "T4": "sm75", "A100": "sm80", "A100-80GB": "sm80", "A10G": "sm86",
    "L4": "sm89", "L40S": "sm89", "RTX-6000-Ada": "sm89", "H100": "sm90", "H200": "sm90",
    "B200": "sm100", "B300": "sm103", "RTX-PRO-6000": "sm120",
    "MI300X": "gfx942", "MI325X": "gfx942", "MI355X": "gfx950",
    "910B": "ascend910b", "910B2": "ascend910b", "910B3": "ascend910b", "910B4": "ascend910b",
    "910C": "ascend910c", "810E": None, "M890": None,
}

# What an Ascend container needs from the host besides its /dev/davinci<i> cards:
# npu-smi inside the container is what the TP check and the memory guard read
# (hetero-1114 A02). Only the ones the node has are passed (a missing -v source
# fails the run).
NPU_HOST_DEVICES = ("/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc")
NPU_HOST_MOUNTS = ("/usr/local/Ascend/driver", "/usr/local/dcmi", "/usr/local/bin/npu-smi",
                   "/usr/local/sbin/npu-smi", "/etc/ascend_install.info", "/etc/vnpu.cfg")
ROCM_DEVICES = ("/dev/kfd", "/dev/dri")


def rocm_container_args() -> tuple[str, ...]:
    """The device flags of a ROCm container: the NC route's own constant
    (``yeto.launch.nc.ssh_sampler.ROCM_DOCKER_ARGS``: ``/dev/kfd``, ``/dev/dri``, the
    video group, seccomp off, ``SYS_PTRACE``, unlimited memlock, ``--shm-size 64g``),
    the very tuple the customer's MI355X sampler starts with. Read on use: that module
    imports this one."""
    from .nc.ssh_sampler import ROCM_DOCKER_ARGS  # noqa: PLC0415

    return tuple(ROCM_DOCKER_ARGS)


def card_arch_of(card: str) -> str | None:
    """The architecture of a catalog card (``B300`` -> ``sm103``, ``MI355X`` ->
    ``gfx950``); None when the catalog does not know it."""
    return CARD_ARCHES.get(canonical_card(card))


@dataclass(frozen=True)
class NodePaths:
    """The node's directories (absolute host paths; None: not declared): ``work``
    (the run's work tree: code, specs, logs), ``models`` (model checkpoints and
    releases), ``cache`` (engine caches: compiled kernels, graphs)."""

    work: str | None = None
    models: str | None = None
    cache: str | None = None

    def record(self) -> dict[str, str | None]:
        return {"work": self.work, "models": self.models, "cache": self.cache}


@dataclass(frozen=True)
class Reach:
    """How the control host reaches the node: ``mode`` (:data:`REACH_MODES`) and
    ``via`` (the jump host of ``jump``, the relay address of ``relay``)."""

    mode: str = "direct"
    via: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in REACH_MODES:
            raise InventoryError(f"reach mode {self.mode!r} is unknown (known: {list(REACH_MODES)})")
        if self.mode in ("jump", "relay") and not self.via:
            raise InventoryError(f"reach {self.mode} needs via (the {self.mode} host)")

    @property
    def inbound(self) -> bool:
        """The control host can open a connection to the node (SSH probes work)."""
        return self.mode in ("direct", "jump", "relay")


@dataclass(frozen=True)
class DeviceSetup:
    """The cards of one start: ``args`` are the container engine's device options
    (empty for a process or a Modal function), ``env`` the variables the sampler
    gets (the visible-card variable of the family, numbered as the sampler sees
    the cards: renumbered from 0 when the engine passes only those cards)."""

    args: tuple[str, ...]
    env: dict[str, str]


@dataclass(frozen=True)
class NodeProfile(Node):
    """A :class:`Node` plus what a sampler start needs, for a node of any source.

    Read with :func:`parse_profile` / :func:`load_profiles` (an inventory entry may
    carry the profile keys ``card_arch``, ``container``, ``paths``, ``model_source``,
    ``reach``), built with :meth:`for_card` for a rented node, or wrapped around a
    parsed node with :meth:`from_node`. A NodeProfile is a Node: every Node consumer
    (card check, ``ssh_options``, ``record``) takes it unchanged.

    ``device_family`` stays the launcher spelling (``npu`` for Ascend, what
    ``$YETO_DEVICE_FAMILY`` and the card tools take); :attr:`family` is the unified
    one (``cuda`` / ``rocm`` / ``ascend`` / ``ppu``) the engine image catalog uses.

    Fields beyond Node:

    - ``card_arch``: ``sm90`` / ``sm103`` / ``gfx950`` / ``ascend910b`` ...; derived from
      the card (:data:`CARD_ARCHES`) when not given;
    - ``container``: :data:`CONTAINERS`; derived from ``runtime`` (``docker`` ->
      ``docker``, ``rootfs`` -> ``process``) when not given;
    - ``paths``: :class:`NodePaths` (``work`` defaults to ``scratch``);
    - ``model_source``: :data:`MODEL_SOURCES` or None (not declared);
    - ``reach``: :class:`Reach`; ``jump`` via ``proxy_jump`` when the node has one,
      else ``direct``.

    Derived per family (:meth:`devices`): the container device flags (cuda:
    ``--gpus`` with docker, CDI ``nvidia.com/gpu=<i>`` with podman; rocm: the NC
    route's ``ROCM_DOCKER_ARGS`` (:func:`rocm_container_args`); ascend:
    ``/dev/davinci<i>`` plus the host devices and driver mounts the node has) and
    the visible-card variable of :data:`yeto.gpu_spec.HW_FAMILIES`
    (``CUDA_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES`` / ``ASCEND_RT_VISIBLE_DEVICES``).
    An inventory ``docker_args`` replaces the derived flags (the site knows its
    hardware); the variable then numbers the host's cards."""

    card_arch: str | None = None
    container: str = ""
    paths: NodePaths = field(default_factory=NodePaths)
    model_source: str | None = None
    reach: Reach | None = None

    def __post_init__(self) -> None:
        set_ = lambda key, value: object.__setattr__(self, key, value)  # noqa: E731 - frozen dataclass
        if self.card_arch is None:
            set_("card_arch", CARD_ARCHES.get(self.card))
        if not self.container:
            set_("container", "process" if self.runtime == "rootfs" else "docker")
        if self.container not in CONTAINERS:
            raise InventoryError(f"node {self.name}: container {self.container!r} is unknown "
                                 f"(known: {list(CONTAINERS)})")
        if self.model_source is not None and self.model_source not in MODEL_SOURCES:
            raise InventoryError(f"node {self.name}: model_source {self.model_source!r} is unknown "
                                 f"(known: {list(MODEL_SOURCES)})")
        if self.paths.work is None and self.scratch:
            set_("paths", replace(self.paths, work=self.scratch))
        if self.reach is None:
            set_("reach", Reach("jump", self.proxy_jump) if self.proxy_jump else Reach())
        elif self.reach.mode == "jump" and self.proxy_jump != self.reach.via:
            if self.proxy_jump:
                raise InventoryError(f"node {self.name}: reach via {self.reach.via} and proxy_jump "
                                     f"{self.proxy_jump} name two jump hosts")
            set_("proxy_jump", self.reach.via)  # ssh_options dials through it

    # -- construction
    @classmethod
    def from_node(cls, node: Node, **profile: Any) -> NodeProfile:
        """``node`` with the profile fields ``profile`` (the rest derived)."""
        if isinstance(node, NodeProfile) and not profile:
            return node
        base = {f.name: getattr(node, f.name) for f in fields(Node)}
        if isinstance(node, NodeProfile):
            base.update({k: getattr(node, k) for k in _PROFILE_KEYS})
        return cls(**{**base, **profile})

    @classmethod
    def for_card(cls, name: str, card: str, cards_per_node: int, *, hbm_gib: float, host: str = "",
                 cpu_arch: str = "x86_64", host_memory_gib: float = 0.0, **fields_: Any) -> NodeProfile:
        """A profile of a rented node that has no inventory entry (an NC lease, a
        Daytona sandbox, a Modal function, a sky VM): the family comes from the card.
        ``host`` may be empty when the node is not reached by SSH (``reach``
        ``function`` / ``outbound``)."""
        card = canonical_card(card)
        return cls(name=name, host=host, device_family=device_family_of(card),
                   cpu_arch=image_catalog.canonical_cpu_arch(cpu_arch), card=card, cards_per_node=int(cards_per_node),
                   hbm_gib=float(hbm_gib), host_memory_gib=float(host_memory_gib), **fields_)

    # -- family
    @property
    def family(self) -> str:
        """``cuda`` / ``rocm`` / ``ascend`` / ``ppu`` (the engine image catalog's names)."""
        return unified_family(self.device_family)

    @property
    def hw(self):  # -> yeto.gpu_spec.HwFamily
        return hw_family(self.device_family)

    @property
    def visible_devices_env(self) -> str | None:
        """The variable that limits the visible cards (:data:`yeto.gpu_spec.HW_FAMILIES`)."""
        return self.hw.visible_devices_env

    def family_env(self) -> dict[str, str]:
        """``$YETO_DEVICE_FAMILY``: torch reports ``cuda`` on ROCm and PPU, so the
        sampler's card telemetry and memory guard read the family from here."""
        return {DEVICE_FAMILY_ENV: self.device_family}

    # -- devices
    def host_paths(self) -> tuple[str, ...]:
        """Host paths the device flags may name (a preflight reports which exist)."""
        if self.family == "ascend":
            return NPU_HOST_DEVICES + NPU_HOST_MOUNTS
        if self.family == "rocm":
            return ROCM_DEVICES
        return ()

    def required_host_paths(self) -> tuple[str, ...]:
        """Host paths a container of this family cannot start without."""
        return ROCM_DEVICES if self.family == "rocm" and self.container in ("docker", "podman") else ()

    def devices(self, cards: Sequence[int] | None = None, *, engine: str | None = None,
                host_paths: Sequence[str] | None = None) -> DeviceSetup:
        """Device flags and visible-card variable for a start on ``cards`` (None: every
        card of the node) with container engine ``engine`` (default :attr:`container`).

        ``host_paths``: the paths the node's preflight found (None: assume every
        path of :meth:`host_paths`); only those are passed. ``docker_args`` of the
        inventory replace the derived flags."""
        engine = engine or self.container
        every = list(range(self.cards_per_node))
        chosen = list(every if cards is None else cards)
        var = self.visible_devices_env
        host_numbers = {var: ",".join(str(i) for i in chosen)} if var else {}
        if engine not in ("docker", "podman"):  # a process sees the host's cards; a Modal function its own
            return DeviceSetup((), host_numbers if engine == "process" else {})
        if self.docker_args:
            return DeviceSetup(tuple(self.docker_args), host_numbers)
        family = self.family
        if family == "cuda":
            whole = chosen == every
            if engine == "docker":
                value = "all" if whole else f'"device={",".join(str(i) for i in chosen)}"'
                args: tuple[str, ...] = ("--gpus", value)
            else:  # podman: the NVIDIA Container Toolkit's CDI devices
                args = tuple(a for i in (["all"] if whole else chosen) for a in ("--device", f"nvidia.com/gpu={i}"))
            renumbered = {var: ",".join(str(i) for i in range(len(chosen)))} if var else {}
            return DeviceSetup(args, renumbered)
        if family == "rocm":  # every render node is passed: HIP_VISIBLE_DEVICES picks by host number
            return DeviceSetup(rocm_container_args(), host_numbers)
        if family == "ascend":
            present = set(self.host_paths()) if host_paths is None else set(host_paths)
            out: list[str] = []
            for i in chosen:
                out += ["--device", f"/dev/davinci{i}"]
            for dev in NPU_HOST_DEVICES:
                if dev in present:
                    out += ["--device", dev]
            for path in NPU_HOST_MOUNTS:
                if path in present:
                    out += ["-v", f"{path}:{path}:ro"]
            return DeviceSetup(tuple(out), host_numbers)
        # ppu: no flags are known (vendor docs only); the inventory's docker_args must give them
        return DeviceSetup((), host_numbers)

    def ssh_options(self) -> list[str]:
        """``ssh`` options through :attr:`reach`: the relay's ``ProxyCommand`` for ``relay``
        (:func:`yeto.node.transport.reach_options`); otherwise :meth:`Node.ssh_options`
        (``ProxyJump``: a ``jump`` reach sets ``proxy_jump``)."""
        if self.reach is not None and self.reach.mode == "relay":
            from yeto.node.transport import node_ssh_options  # noqa: PLC0415 - yeto.node imports this module

            return node_ssh_options(self)
        return super().ssh_options()

    def summary(self) -> dict[str, Any]:
        """What a member record and the dashboard show of the machine (no host, no paths)."""
        return {"family": self.family, "card": self.card, "card_arch": self.card_arch,
                "cards": self.cards_per_node, "hbm_gib": self.hbm_gib, "cpu_arch": self.cpu_arch,
                "container": self.container, "reach": self.reach.mode if self.reach else None,
                "model_source": self.model_source}

    def record(self) -> dict[str, Any]:
        """:meth:`Node.record` plus :meth:`summary`'s profile keys."""
        return {**super().record(), "family": self.family, "card_arch": self.card_arch,
                "container": self.container, "model_source": self.model_source,
                "reach": self.reach.mode if self.reach else None}


def _profile_fields(raw: dict[str, Any], node: Node) -> dict[str, Any]:
    """The profile keys of an inventory entry, validated."""
    name = node.name
    out: dict[str, Any] = {}
    if raw.get("card_arch") is not None:
        out["card_arch"] = _require_text(raw["card_arch"], name, "card_arch")
    if raw.get("container") is not None:
        out["container"] = _require_text(raw["container"], name, "container")
    if raw.get("model_source") is not None:
        out["model_source"] = _require_text(raw["model_source"], name, "model_source")
    paths = raw.get("paths")
    if paths is not None:
        if not isinstance(paths, dict) or set(paths) - set(PATH_KEYS):
            raise InventoryError(f"node {name}: paths must be a map with keys {list(PATH_KEYS)}, got {paths!r}")
        out["paths"] = NodePaths(**{k: _host_path(paths, name, k) for k in PATH_KEYS if k in paths})
    reach = raw.get("reach")
    if reach is not None:
        if isinstance(reach, str):
            reach = {"mode": reach}
        if not isinstance(reach, dict) or set(reach) - {"mode", "via"} or "mode" not in reach:
            raise InventoryError(f"node {name}: reach must be a mode or {{mode, via}}, got {reach!r}")
        via = reach.get("via") or (node.proxy_jump if reach["mode"] == "jump" else None)
        try:
            out["reach"] = Reach(str(reach["mode"]), str(via) if via else None)
        except InventoryError as exc:
            raise InventoryError(f"node {name}: {exc}") from None
    return out


@contextlib.contextmanager
def _profile_reader():
    """The profile readers consume the profile keys: no "ignored" warning from parse_node."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"node .*: \[.*\] are node profile keys")
        yield


def parse_profile(raw: Any, *, index: int, images: Mapping[str, str] | None = None) -> NodeProfile:
    """One inventory entry as a :class:`NodeProfile` (:func:`parse_node`, then the
    profile keys)."""
    with _profile_reader():
        node = parse_node(raw, index=index, images=images)
    return NodeProfile.from_node(node, **_profile_fields(raw, node))


def parse_profiles(document: Any) -> NodeList:
    """:func:`parse_inventory` with every node as a :class:`NodeProfile`."""
    with _profile_reader():
        nodes = parse_inventory(document)  # every check of the inventory, the duplicate names included
    entries = document if isinstance(document, list) else document.get("nodes")
    return NodeList([NodeProfile.from_node(node, **_profile_fields(raw, node)) for node, raw in zip(nodes, entries)],
                    lease_store=nodes.lease_store)


def load_profiles(path: str | Path) -> NodeList:
    """:func:`load_inventory` with every node as a :class:`NodeProfile`."""
    import yaml  # noqa: PLC0415

    with _profile_reader():
        nodes = load_inventory(path)
    document = yaml.safe_load(Path(path).read_text())
    entries = document if isinstance(document, list) else document.get("nodes")
    try:
        return NodeList([NodeProfile.from_node(node, **_profile_fields(raw, node))
                         for node, raw in zip(nodes, entries)], lease_store=nodes.lease_store)
    except InventoryError as exc:
        raise InventoryError(f"{path}: {exc}") from None


def profile_of(node: Node) -> NodeProfile:
    """``node`` as a :class:`NodeProfile` (itself when it is one)."""
    return NodeProfile.from_node(node)
