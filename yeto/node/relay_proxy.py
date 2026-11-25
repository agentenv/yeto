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
the endpoint differs (a URL and a bearer token instead of an NC account's gateway
cookie).

Security. ``relay.mjs`` checks nothing but its allow file: whoever reaches its URL
reaches port 22 of every node it allows. So:

* the relay MUST sit behind a gateway that authenticates each upgrade. This module
  sends ``Authorization: Bearer <token>``; the token comes from the file named by
  ``$YETO_RELAY_TOKEN_FILE`` (owner-only, ``0600``) or from ``$YETO_RELAY_TOKEN``,
  never from the argv or the URL (a URL with ``user:password@`` is refused). Without
  a token the proxy refuses to dial, unless ``$YETO_RELAY_NO_TOKEN=1`` says the
  gateway authenticates otherwise (a VPN, mutual TLS);
* plaintext ``ws://`` / ``http://`` is refused unless ``$YETO_RELAY_ALLOW_PLAINTEXT=1``
  (a relay on a trusted private link): the token would cross in the clear;
* the node's host key is checked strictly (``StrictHostKeyChecking=yes``, added by
  :func:`yeto.node.transport.reach_options`): the relay host must not be able to
  stand in for a node and read the spec and the store credential sent to it. The
  node's key must be in the launcher's ``known_hosts`` beforehand.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit

from yeto.launch.nc import tunnel

TOKEN_ENV = "YETO_RELAY_TOKEN"
TOKEN_FILE_ENV = "YETO_RELAY_TOKEN_FILE"
NO_TOKEN_ENV = "YETO_RELAY_NO_TOKEN"  # "1": the relay's gateway authenticates without a bearer token
PLAINTEXT_ENV = "YETO_RELAY_ALLOW_PLAINTEXT"  # "1": ws:// / http:// accepted
SCHEMES = ("ws", "wss", "http", "https")


def url_problem(url: str, env: Mapping[str, str] | None = None) -> str | None:
    """Why ``url`` is not a relay URL this proxy dials (None: it is)."""
    env = os.environ if env is None else env
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname
    except ValueError:
        return f"relay url {url!r} cannot be parsed"
    if parts.scheme not in SCHEMES or not host:
        return f"relay url {url!r} must be ws://, wss://, http:// or https:// with a host"
    if "@" in parts.netloc:
        return (f"relay url of host {host}: credentials in the URL would sit in the ssh argv; "
                f"put the token in ${TOKEN_FILE_ENV} (a 0600 file) or ${TOKEN_ENV}")
    if parts.scheme in ("ws", "http") and env.get(PLAINTEXT_ENV) != "1":
        return (f"relay url {url!r} is plaintext: the token would cross in the clear; use wss:// / https://, "
                f"or set {PLAINTEXT_ENV}=1 for a relay on a trusted private link")
    return None


def token(env: Mapping[str, str] | None = None) -> str:
    """The relay's bearer token (module doc); "" when ``$YETO_RELAY_NO_TOKEN=1`` and
    none is set. Raises :class:`yeto.launch.nc.tunnel.TunnelError` otherwise."""
    env = os.environ if env is None else env
    path = env.get(TOKEN_FILE_ENV)
    if path:
        file = Path(path).expanduser()
        try:
            mode = file.stat().st_mode
            value = file.read_text().strip()
        except OSError as error:
            raise tunnel.TunnelError(f"{TOKEN_FILE_ENV}: cannot read the relay token file: {error}") from None
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise tunnel.TunnelError(f"{TOKEN_FILE_ENV}: the relay token file must be owner-only (chmod 600)")
        if not value:
            raise tunnel.TunnelError(f"{TOKEN_FILE_ENV}: the relay token file is empty")
        return _header_safe(value, TOKEN_FILE_ENV)
    value = (env.get(TOKEN_ENV) or "").strip()
    if value:
        return _header_safe(value, TOKEN_ENV)
    if env.get(NO_TOKEN_ENV) == "1":
        return ""
    raise tunnel.TunnelError(f"a site relay needs a bearer token: set {TOKEN_FILE_ENV} (a 0600 file) or {TOKEN_ENV}, "
                             f"or {NO_TOKEN_ENV}=1 when the relay's gateway authenticates otherwise (VPN, mutual TLS)")


def _header_safe(value: str, where: str) -> str:
    """A token that fits one HTTP header line: printable ASCII, no blanks (a CR / LF would
    end the header and inject others)."""
    if not all(33 <= ord(c) <= 126 for c in value):
        raise tunnel.TunnelError(f"{where}: the relay token must be one word of printable ASCII "
                                 "(no blanks, CR, LF or other control characters)")
    return value


def endpoint(url: str, env: Mapping[str, str] | None = None) -> tunnel.Endpoint:
    """The relay endpoint of ``url`` with its token (``ws`` / ``wss`` read as ``http`` /
    ``https``: :func:`yeto.launch.nc.tunnel.open_ws` speaks the upgrade itself)."""
    why = url_problem(url, env)
    if why:
        raise tunnel.TunnelError(why)
    base = url.strip()
    for ws, http in (("wss://", "https://"), ("ws://", "http://")):
        if base.startswith(ws):
            base = http + base[len(ws):]
    return tunnel.Endpoint(base, "", authorization=token(env))


def proxy(url: str, host: str, port: int) -> int:
    """Dial ``host:port`` through the relay at ``url`` and pipe stdin / stdout."""
    sock, rest = tunnel.open_ws(endpoint(url), f"{host}:{port}")
    tunnel.pipe(sock, rest)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m yeto.node.relay_proxy",
                                     description="ssh ProxyCommand through a relay.mjs WebSocket relay "
                                                 f"(token: ${TOKEN_FILE_ENV} or ${TOKEN_ENV})")
    parser.add_argument("--url", required=True, help="the relay's URL (wss:// or https://)")
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    args = parser.parse_args(argv)
    return proxy(args.url, args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())
