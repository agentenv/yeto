"""Parsers of the host tools' output, shared by ``yeto preflight`` and the node health check.

Pure functions over recorded text (no SSH, no subprocess): the sectioned output of a
probe script (:func:`probe_script` / :func:`parse_sections`), the NVSwitch ``Fabric``
blocks of ``nvidia-smi -q`` (:func:`parse_fabric`), the IB / RoCE ports of ``ibstat``
and ``ibv_devinfo -v`` (:func:`parse_ibstat`, :func:`parse_ibv_devinfo`, :func:`ports_of`)
and the ports an ``NCCL_IB_HCA`` value selects (:func:`hca_filter`).

``yeto preflight`` (:mod:`yeto.launch.preflight.cluster`) checks a node before a run
with them; :mod:`yeto.hw.node_health` checks it while the run goes on with the same
parsers. Standard library only: a node runtime imports this without the trainer stack.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

SECTION = "@@yeto-section"
RC = "@@yeto-rc"


@dataclass(frozen=True)
class Section:
    rc: int | None
    text: str


def parse_sections(text: str) -> dict[str, Section]:
    """Output of a probe script (:func:`probe_script`) by section name."""
    out: dict[str, Section] = {}
    name, lines = None, []
    for line in (text or "").splitlines():
        if line.startswith(SECTION + " "):
            if name is not None:
                out[name] = Section(None, "\n".join(lines))
            name, lines = line[len(SECTION) + 1:].strip(), []
        elif line.startswith(RC + " ") and name is not None:
            try:
                rc = int(line[len(RC) + 1:].strip())
            except ValueError:
                rc = None
            out[name] = Section(rc, "\n".join(lines))
            name, lines = None, []
        elif name is not None:
            lines.append(line)
    if name is not None:
        out[name] = Section(None, "\n".join(lines))
    return out


def probe_script(sections: Sequence[tuple[str, str]]) -> str:
    """Bash that runs every ``(name, command)`` and marks its output and exit code."""
    body = []
    for name, command in sections:
        body.append(f"echo '{SECTION} {name}'\n( {command}\n) 2>&1 < /dev/null\necho \"{RC} $?\"\n")
    return "".join(body)


def parse_fabric(text: str) -> list[tuple[str, str]]:
    """``(state, status)`` per GPU of the ``Fabric`` blocks of ``nvidia-smi -q``."""
    out, state = [], None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "state":
            state = value
        elif key == "status" and state is not None:
            out.append((state, value))
            state = None
    if state is not None:
        out.append((state, ""))
    return out


@dataclass
class IbPort:
    ca: str
    port: int
    state: str  # Active / Down / Init / Armed / Polling
    physical: str = ""
    rate_gbps: float | None = None
    link_layer: str = ""  # InfiniBand / Ethernet
    firmware: str = ""
    active_mtu: int | None = None

    @property
    def active(self) -> bool:
        return self.state.lower() == "active"


def parse_ibstat(text: str) -> list[IbPort]:
    ports: list[IbPort] = []
    ca, fw, cur = None, "", None
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"CA '([^']+)'", line)
        if m:
            ca, fw, cur = m.group(1), "", None
            continue
        m = re.match(r"Port (\d+):$", line)
        if m and ca:
            cur = IbPort(ca=ca, port=int(m.group(1)), state="", firmware=fw)
            ports.append(cur)
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "firmware version":
            fw = value
        if cur is None:
            continue
        if key == "state":
            cur.state = value
        elif key == "physical state":
            cur.physical = value
        elif key == "rate":
            try:
                cur.rate_gbps = float(value.split()[0])
            except (ValueError, IndexError):
                pass
        elif key == "link layer":
            cur.link_layer = value
    return ports


_SPEED = re.compile(r"([\d.]+)\s*Gbps")
_WIDTH = re.compile(r"(\d+)X")


def parse_ibv_devinfo(text: str) -> list[IbPort]:
    ports: list[IbPort] = []
    ca, fw, cur = None, "", None
    width = speed = None

    def close() -> None:
        if cur is not None and width and speed and cur.rate_gbps is None:
            cur.rate_gbps = width * speed

    for raw in text.splitlines():
        line = raw.strip()
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "hca_id":
            close()
            ca, fw, cur, width, speed = value, "", None, None, None
        elif key == "fw_ver":
            fw = value
        elif key == "port" and ca:
            close()
            width = speed = None
            cur = IbPort(ca=ca, port=int(value.split()[0]), state="", firmware=fw)
            ports.append(cur)
        elif cur is None:
            continue
        elif key == "state":
            word = value.split()[0] if value else ""
            cur.state = {"PORT_ACTIVE": "Active", "PORT_DOWN": "Down", "PORT_INIT": "Initializing",
                         "PORT_ARMED": "Armed"}.get(word, word)
        elif key == "link_layer":
            cur.link_layer = value
        elif key == "active_mtu":
            try:
                cur.active_mtu = int(value.split()[0])
            except (ValueError, IndexError):
                pass
        elif key == "active_width":
            m = _WIDTH.search(value)
            width = int(m.group(1)) if m else None
        elif key == "active_speed":
            m = _SPEED.search(value)
            speed = float(m.group(1)) if m else None
    close()
    return ports


def hca_filter(spec: str, ports: Sequence[IbPort]) -> tuple[list[IbPort], list[str]]:
    """The ports ``NCCL_IB_HCA`` selects (prefix match; ``=`` exact; ``^`` excludes),
    and the names of the list that match no port."""
    s = spec.strip()
    exact = s.startswith("=")
    s = s.lstrip("=")
    exclude = s.startswith("^")
    s = s.lstrip("^")
    exact = exact or s.startswith("=")
    s = s.lstrip("=")
    wanted = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, port = item.partition(":")
        wanted.append((name, int(port) if port.isdigit() else None))

    def match(p: IbPort, name: str, port: int | None) -> bool:
        ok = p.ca == name if exact else p.ca.startswith(name)
        return ok and (port is None or p.port == port)

    if exclude:
        return [p for p in ports if not any(match(p, n, q) for n, q in wanted)], []
    selected = [p for p in ports if any(match(p, n, q) for n, q in wanted)]
    missing = [n if q is None else f"{n}:{q}" for n, q in wanted if not any(match(p, n, q) for p in ports)]
    return selected, missing


def natural_order(ports: Iterable[IbPort]) -> list[IbPort]:
    key = lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.ca)] + [p.port]
    return sorted(ports, key=key)


def ports_of(sections: Mapping[str, Section]) -> list[IbPort]:
    """The IB / RoCE ports of a probe: ``ibstat``, with the MTU of ``ibv_devinfo``; else ``ibv_devinfo``."""
    ibstat = parse_ibstat(sections["ibstat"].text) if section_ok(sections.get("ibstat")) else []
    devinfo = parse_ibv_devinfo(sections["ibv_devinfo"].text) if section_ok(sections.get("ibv_devinfo")) else []
    if not ibstat:
        return devinfo
    mtu = {(p.ca, p.port): p.active_mtu for p in devinfo}
    for p in ibstat:
        p.active_mtu = p.active_mtu or mtu.get((p.ca, p.port))
    return ibstat


def section_ok(sec: Section | None) -> bool:
    return sec is not None and sec.rc == 0
