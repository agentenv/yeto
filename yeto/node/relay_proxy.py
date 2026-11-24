"""``ssh`` ``ProxyCommand`` through a site's own WebSocket relay (reach ``relay``).

A node that only a relay host can reach (a cluster behind a gateway that passes
HTTP upgrades and nothing else) runs ``relay.mjs`` (:mod:`yeto.launch.nc`'s
WebSocket -> TCP relay, the same file the National Compute workspace runs) on that
host, with the nodes' ``HOST:22`` in its allow file. The inventory then says::

    reach: {mode: relay, via: wss://relay.example/yeto}

and every ``ssh`` of the pool to the node (code upload, start, probe, stop) dials
through this module (:func:`yeto.node.transport.reach_options`)::

    ProxyCommand python -P -m yeto.node.relay_proxy --url wss://relay.example/yeto %h %p

The framing, the handshake and the pipe are :mod:`yeto.launch.nc.tunnel`'s; only
the endpoint differs (a URL instead of an NC account's gateway cookie).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from yeto.launch.nc import tunnel


def endpoint(url: str) -> tunnel.Endpoint:
    """The relay endpoint of ``url`` (``ws`` / ``wss`` read as ``http`` / ``https``:
    :func:`yeto.launch.nc.tunnel.open_ws` speaks the upgrade itself)."""
    base = url.strip()
    for ws, http in (("wss://", "https://"), ("ws://", "http://")):
        if base.startswith(ws):
            base = http + base[len(ws):]
    if not base.startswith(("http://", "https://")):
        raise tunnel.TunnelError(f"relay url {url!r} must be ws://, wss://, http:// or https://")
    return tunnel.Endpoint(base, "")


def proxy(url: str, host: str, port: int) -> int:
    """Dial ``host:port`` through the relay at ``url`` and pipe stdin / stdout."""
    sock, rest = tunnel.open_ws(endpoint(url), f"{host}:{port}")
    tunnel.pipe(sock, rest)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m yeto.node.relay_proxy",
                                     description="ssh ProxyCommand through a relay.mjs WebSocket relay")
    parser.add_argument("--url", required=True, help="the relay's URL (ws://, wss://, http:// or https://)")
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    args = parser.parse_args(argv)
    return proxy(args.url, args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())
