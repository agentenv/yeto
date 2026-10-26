"""Master ports of verl's Ray worker groups without the bind-and-release race (S27 G-DC).

verl gives every ``RayWorkerGroup`` its torch.distributed master port from
``verl.single_controller.ray.base.get_master_addr_port``: bind port 0, read the
port, close the socket, return it. The Megatron island starts its four
standalone vLLM replicas at once (``LLMServerManager.create`` gathers
``init_standalone``), each with its own checkpoint-engine worker group, so
several groups on one node ask at the same time. On Modal the kernel handed
the same released port to two of them; the second ``TCPStore`` failed with
EADDRINUSE (s27-dc-smoke1, port 52589, 2026-10-25).

:func:`pick_free_port` draws the port at random from a range below Linux's
ephemeral range (so the kernel's own port-0 binds never land there) and keeps
the bind test; :func:`spread_master_ports` installs it as verl's remote
function in this process (the driver side that creates the worker groups).
"""

from __future__ import annotations

import random
import socket

PORT_RANGE = (20000, 32000)  # below /proc/sys/net/ipv4/ip_local_port_range (32768-60999)
ATTEMPTS = 200
_installed = False


def pick_free_port(low: int = PORT_RANGE[0], high: int = PORT_RANGE[1], rng=None) -> int:
    rng = rng or random.SystemRandom()
    for _ in range(ATTEMPTS):
        port = rng.randrange(int(low), int(high))
        try:
            with socket.socket() as sock:
                sock.bind(("", port))
            return port
        except OSError:
            continue
    raise RuntimeError(f"no free port in [{low}, {high}) after {ATTEMPTS} random tries")


def spread_master_addr_port(master_port_range=None) -> tuple[str, str]:
    """Same contract as verl's ``get_master_addr_port`` (node address, port as text)."""
    import ray

    low, high = master_port_range or PORT_RANGE
    return ray.util.get_node_ip_address().strip("[]"), str(pick_free_port(low, high))


def spread_master_ports() -> bool:
    """Replace verl's ``get_master_addr_port`` in this process; True when it was installed now."""
    global _installed
    if _installed:
        return False
    import ray
    import verl.single_controller.ray.base as ray_base

    ray_base.get_master_addr_port = ray.remote(spread_master_addr_port)
    _installed = True
    return True
