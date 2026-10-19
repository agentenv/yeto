"""Node inventory for machines yeto reaches by SSH (rl-ssh-node-pool D1).

One file describes every node the launcher may use, on-premises or rented:
the SSH target, an optional jump host, the device family, the card model and
count, the HBM size per card, the host memory (the trainer offload plan reads
it), the fabric, shared mounts, the container image and an optional declared
price. The core stays neutral: provider differences live in the file.

Example::

    nodes:
      - name: ascend00
        host: user@172.28.11.10
        proxy_jump: user@bastion.example
        device_family: npu
        card: 910B4
        cards_per_node: 8
        hbm_gib: 64
        host_memory_gib: 1024
        fabric: roce
        interface: eth0
        address: 172.28.11.10
        mounts: {/mnt/shared: /mnt/shared}
        image: REGISTRY/yeto-ascend:TAG
        runtime: docker
        docker_args: [--device, /dev/davinci0, ...]
        price_per_card_hour: 0.0

``address`` is the node's IP on the cluster network (the ``interface``), which
the other nodes dial for rendezvous; ``host`` may be an SSH alias that only the
head resolves. ``docker_args`` are the device flags of ``docker run`` for the
node's device family (a ``runtime: docker`` trainer island refuses to start
without them).
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..gpu_spec import DEVICE_FAMILIES, UnknownCard, canonical_card, device_family_of
from ..hw import card_tools
from ..hw.catalog import UnknownCardType, compat_group

# Fabrics an inventory may declare. "tcp" is Ethernet without RDMA; "roce" and
# "ib" are RDMA over Ethernet and InfiniBand; "hccs" is the Ascend on-node
# interconnect (a node declaring it still needs a host fabric for cross-node
# traffic, so hccs is accepted only together with one of the others).
FABRICS = ("tcp", "roce", "ib", "hccs")
RUNTIMES = ("docker", "rootfs")

_REQUIRED = ("name", "host", "device_family", "card", "cards_per_node", "hbm_gib",
             "host_memory_gib")
_KNOWN = _REQUIRED + ("proxy_jump", "fabric", "interface", "address", "mounts", "image", "runtime",
                      "docker_args", "price_per_card_hour", "roles")


class InventoryError(ValueError):
    """The node inventory is not usable; the launch is refused before any SSH."""


@dataclass(frozen=True)
class Node:
    """One node of the pool."""

    name: str
    host: str  # SSH target, "user@host" or a host alias
    device_family: str  # "cuda", "npu" or "ppu"
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
    price_per_card_hour: float | None = None
    roles: tuple[str, ...] = ()

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

    def record(self) -> dict[str, Any]:
        """Manifest / event-tape row of this node (no SSH user, no mounts)."""
        return {"name": self.name, "device_family": self.device_family, "card": self.card,
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


def parse_node(raw: Any, *, index: int) -> Node:
    """One inventory entry as a :class:`Node`; any gap refuses the launch."""
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
    if runtime == "docker" and raw.get("image") is None:
        raise InventoryError(f"node {name}: runtime docker needs an image")
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
    price = raw.get("price_per_card_hour")
    if price is not None and (isinstance(price, bool) or not isinstance(price, (int, float))
                              or price < 0):
        raise InventoryError(f"node {name}: price_per_card_hour must be a non-negative "
                             f"number, got {price!r}")
    roles = raw.get("roles") or ()
    if not isinstance(roles, (list, tuple)) or not all(isinstance(r, str) for r in roles):
        raise InventoryError(f"node {name}: roles must be a list of strings, got {roles!r}")
    node = Node(
        name=name,
        host=_require_text(raw["host"], name, "host"),
        device_family=family,
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
        image=(_require_text(raw["image"], name, "image")
               if raw.get("image") is not None else None),
        runtime=runtime,
        docker_args=tuple(docker_args),
        price_per_card_hour=(float(price) if price is not None else None),
        roles=tuple(roles),
    )
    try:
        node.compat_group  # noqa: B018 - an unknown HBM variant fails here, not at launch
    except UnknownCardType as exc:
        raise InventoryError(f"node {name}: {exc}") from None
    return node


def parse_inventory(document: Any) -> list[Node]:
    """The ``nodes:`` list of a loaded inventory document."""
    if isinstance(document, list):
        entries = document
    elif isinstance(document, dict):
        entries = document.get("nodes")
    else:
        raise InventoryError(f"the inventory must be a mapping with a nodes list, got "
                             f"{type(document).__name__}")
    if not isinstance(entries, list) or not entries:
        raise InventoryError("the inventory must contain at least one node under nodes:")
    nodes = [parse_node(entry, index=index) for index, entry in enumerate(entries)]
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
