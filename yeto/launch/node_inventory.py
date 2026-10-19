"""Node inventory for machines yeto reaches by SSH (rl-ssh-node-pool D1).

One file describes every node the launcher may use, on-premises or rented:
the SSH target, an optional jump host, the device family, the CPU
architecture, the card model and count, the HBM size per card, the host
memory (the trainer offload plan reads it), the fabric, shared mounts, the
container image and an optional declared price. The core stays neutral:
provider differences live in the file.

``cpu_arch`` (``x86_64`` or ``aarch64``) selects the image (S25 N1.6): a node
without ``image`` takes ``images[cpu_arch]`` from the top of the file, and a
node whose image is a yeto catalog image of another architecture or device
family is refused (``yeto.launch.images``). The trainer preflight also
compares ``uname -m`` with ``cpu_arch``.

Example::

    images:
      aarch64: REGISTRY/yeto-npu-sampler:TAG
      x86_64: REGISTRY/yeto-ppu-train:TAG
    nodes:
      - name: ascend00
        host: user@172.28.11.10
        proxy_jump: user@bastion.example
        device_family: npu
        cpu_arch: aarch64
        card: 910B4
        cards_per_node: 8
        hbm_gib: 64
        host_memory_gib: 1024
        fabric: roce
        interface: eth0
        mounts: {/mnt/shared: /mnt/shared}
        runtime: docker
        price_per_card_hour: 0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..gpu_spec import DEVICE_FAMILIES, UnknownCard, canonical_card, device_family_of
from ..hw import card_tools
from . import images as image_catalog

# Fabrics an inventory may declare. "tcp" is Ethernet without RDMA; "roce" and
# "ib" are RDMA over Ethernet and InfiniBand; "hccs" is the Ascend on-node
# interconnect (a node declaring it still needs a host fabric for cross-node
# traffic, so hccs is accepted only together with one of the others).
FABRICS = ("tcp", "roce", "ib", "hccs")
RUNTIMES = ("docker", "rootfs")

_REQUIRED = ("name", "host", "device_family", "cpu_arch", "card", "cards_per_node", "hbm_gib",
             "host_memory_gib")
_KNOWN = _REQUIRED + ("proxy_jump", "fabric", "interface", "mounts", "image", "runtime",
                      "price_per_card_hour", "roles")


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
    mounts: Mapping[str, str] = field(default_factory=dict)
    image: str | None = None
    runtime: str = "docker"
    price_per_card_hour: float | None = None
    roles: tuple[str, ...] = ()

    @property
    def hbm_gib_per_node(self) -> float:
        return self.hbm_gib * self.cards_per_node

    def ssh_options(self) -> list[str]:
        """``ssh`` options that reach this node, ``ProxyJump`` included.

        The executor passes these on the command line; it never writes to the
        user's SSH configuration (D4).
        """
        return ["-o", f"ProxyJump={self.proxy_jump}"] if self.proxy_jump else []

    def preflight_commands(self) -> tuple[tuple[str, ...], ...]:
        return card_tools.preflight_commands(self.device_family)

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
                "host_memory_gib": self.host_memory_gib, "fabric": self.fabric,
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
                                 f"default (rl-ssh-node-pool D2)")
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
    price = raw.get("price_per_card_hour")
    if price is not None and (isinstance(price, bool) or not isinstance(price, (int, float))
                              or price < 0):
        raise InventoryError(f"node {name}: price_per_card_hour must be a non-negative "
                             f"number, got {price!r}")
    roles = raw.get("roles") or ()
    if not isinstance(roles, (list, tuple)) or not all(isinstance(r, str) for r in roles):
        raise InventoryError(f"node {name}: roles must be a list of strings, got {roles!r}")
    return Node(
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
        mounts=dict(mounts),
        image=image,
        runtime=runtime,
        price_per_card_hour=(float(price) if price is not None else None),
        roles=tuple(roles),
    )


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
