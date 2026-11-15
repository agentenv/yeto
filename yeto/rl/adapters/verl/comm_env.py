"""Collective-communication variables for a verl island that spans Ray nodes (opt-in).

A verl island's workers are Ray workers, so they inherit the environment of the
``ray start`` shell on their node, never the ``island_comm_env`` exports that the
ssh launcher writes. On one subnet the default route happens to pick the right
interface; across subnets it does not. Measured 2026-11-14 on two nodes on one subnet
plus a third node on another subnet: the 3-node island died in
``gloo/transport/tcp/socket.cc:142 connect: Network is unreachable``.

``YETO_ISLAND_COMM_INTERFACE`` turns this on; unset, nothing here runs and the
Ray runtime env is exactly what it was. Its value is explicit interface names:

* one name (``eth0.100``): every node uses it; the variables travel in the
  runtime env's ``env_vars``.
* a comma list (``eth0.100,eth1.200``): for nodes whose interfaces differ
  (a node on another subnet). Each Ray worker takes the first listed name that
  exists on its own node (:func:`setup_worker`), so nothing is guessed from
  addresses. A node with none of the names fails at worker start, by name.

``auto`` (route probing) is not supported yet and is refused.
``YETO_ISLAND_COMM_FABRIC`` (default ``tcp``) is the fabric passed to
``topology.island_comm_env``.

A value the user already set wins: a variable present in the driver's
environment is forwarded as is and never replaced; in list mode the worker also
keeps a value its node's ``ray start`` shell exported.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping

INTERFACE_ENV = "YETO_ISLAND_COMM_INTERFACE"
FABRIC_ENV = "YETO_ISLAND_COMM_FABRIC"
# list mode: the worker-side recipe, and the setup hook it wraps (if any)
WORKER_ENV = "YETO_ISLAND_COMM_WORKER"
INNER_HOOK_ENV = "YETO_ISLAND_COMM_INNER_HOOK"
SETUP_HOOK = "yeto.rl.adapters.verl.comm_env.setup_worker"
INTERFACE_KEYS = ("HCCL_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME")
SYS_NET = "/sys/class/net"


def _launcher_family(vendor_family: str) -> str:
    from yeto.hw.catalog import DEVICE_FAMILY_BY_VENDOR

    return DEVICE_FAMILY_BY_VENDOR.get(vendor_family, "cuda")


def interfaces(environ: Mapping[str, str] | None = None) -> list[str]:
    """The explicit names of ``YETO_ISLAND_COMM_INTERFACE`` ([] when unset)."""
    raw = (os.environ if environ is None else environ).get(INTERFACE_ENV, "").strip()
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if any(name.lower() == "auto" for name in names):
        raise SystemExit(f"{INTERFACE_ENV}=auto is not supported yet; write the interface name "
                         f"(a comma list when nodes differ, e.g. eth0.100,eth1.200)")
    return names


def island_plan(span: int, *, environ: Mapping[str, str] | None = None,
                device_family: str | None = None) -> dict | None:
    """The plan's ``comm_env`` section, or None (switch unset, or one card node).

    ``device_family`` is the vendor family of ``yeto.hw.catalog.device_family``
    (``ascend`` / ``nvidia``); None reads this node's.
    """
    environ = os.environ if environ is None else environ
    names = interfaces(environ)
    if not names or span <= 1:
        return None
    from yeto.launch.topology import island_comm_env

    if device_family is None:
        from yeto.hw.catalog import device_family as _family

        device_family = _family()
    family = _launcher_family(device_family)
    fabric = environ.get(FABRIC_ENV, "").strip() or "tcp"
    single = names[0] if len(names) == 1 else None
    env = island_comm_env(family, fabric, single, num_nodes=span)
    return {"family": family, "fabric": fabric, "span": span, "interfaces": names, "env": env}


def apply(runtime_env: dict, section: Mapping | None,
          environ: Mapping[str, str] | None = None) -> dict:
    """Add ``section`` (:func:`island_plan`) to a Ray runtime env; None: unchanged."""
    if not section:
        return runtime_env
    environ = os.environ if environ is None else environ
    env_vars = runtime_env.setdefault("env_vars", {})
    for key, value in section["env"].items():
        if key in environ:  # the user's value wins, and reaches the workers too
            env_vars.setdefault(key, environ[key])
        else:
            env_vars.setdefault(key, value)
    if len(section["interfaces"]) > 1:  # per-node names: resolved on each worker
        import json

        env_vars[WORKER_ENV] = json.dumps({k: section[k] for k in ("family", "fabric", "span",
                                                                     "interfaces")})
        for key in INTERFACE_KEYS:  # a driver-side choice still wins
            if key in environ:
                env_vars.setdefault(key, environ[key])
        inner = runtime_env.get("worker_process_setup_hook")
        if inner:
            env_vars[INNER_HOOK_ENV] = inner
        runtime_env["worker_process_setup_hook"] = SETUP_HOOK
    return runtime_env


def local_interface(names, sys_net: str = SYS_NET) -> str:
    for name in names:
        if os.path.exists(os.path.join(sys_net, name)):
            return name
    raise RuntimeError(f"{INTERFACE_ENV}: none of {', '.join(names)} exists on this node "
                       f"({os.uname().nodename}); add this node's interface to the list")


def setup_worker(sys_net: str = SYS_NET) -> None:
    """Ray worker setup hook (list mode): this node's interface, then the wrapped hook."""
    import json

    from yeto.launch.topology import island_comm_env

    spec = json.loads(os.environ[WORKER_ENV])
    name = local_interface(spec["interfaces"], sys_net)
    env = island_comm_env(spec["family"], spec["fabric"], name, num_nodes=spec["span"])
    for key, value in env.items():
        os.environ.setdefault(key, value)
    inner = os.environ.get(INNER_HOOK_ENV)
    if inner:
        module, _, func = inner.rpartition(".")
        getattr(importlib.import_module(module), func)()
