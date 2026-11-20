"""Resolve and self-check the service URLs an AgentENV microVM must reach.

Under the ``remote-docker`` provider a trial container shares the host's docker
bridge with its task sidecars, so the judge **task-server** and the rollout
**records-proxy** are reachable at ``http://<bridge-ip>:<host-port>`` and
``{{yeto.sidecar.<name>.url}}`` renders to that bridge address
(:mod:`yeto.sandbox.ops`).  AgentENV is different: a trial is a Firecracker
microVM, not a container on a docker bridge, and it reaches those services only
by **egress to a routable address**.  Two AgentENV facts drive this module
(see ``~/work/agentenv-integ/docs/AGENTENV_API.md`` and ``E2B_SDK_CHEATSHEET.md``):

1. A microVM can make outbound TCP connections to an external ``IP:port`` when
   the sandbox is created with ``allow_internet_access=True`` (the default).
2. AgentENV's default egress **denies the private CIDR ranges**.  A sidecar on a
   private address is therefore unreachable unless that address is added to the
   sandbox's egress allow-list (``SandboxNetworkOpts.allow_out`` at create, or
   ``PUT /sandboxes/{id}/network`` at runtime).  AgentENV only authenticates, it
   does not encrypt, so such sidecars must live on a *trusted private network*
   shared with the runtime nodes (or be fronted by a TLS-terminating relay /
   a public HTTPS endpoint, which needs no allow rule).

This module is the single place that turns a logical service endpoint into (a)
the URL a microVM should use, (b) the egress allow targets that make it
reachable, and (c) a reachability self-check run *inside* the microVM.  It owns
no sandbox: callers (the B1 ``agentenv`` provider) inject an ``exec`` callable,
so this module never imports the sandbox / client / pool / env code.

Nothing here opens a socket, resolves DNS, or talks to a cluster; the only I/O
is the injected ``exec_fn`` the caller drives against a live microVM.
"""

from __future__ import annotations

import ipaddress
import re
import shlex
import socket
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

# ``{{yeto.sidecar.<name>.(url|host|port)}}`` -- the agentenv variant of the
# remote-docker placeholder (:data:`yeto.sandbox.ops.PLACEHOLDER`).  Unlike
# remote-docker there is no docker bridge, so ``{{yeto.host.bridge_ip}}`` has no
# meaning for a microVM and :func:`render_task_text` rejects it rather than
# render an address the microVM cannot route to.
PLACEHOLDER = re.compile(
    r"\{\{\s*yeto\.(sidecar\.([A-Za-z0-9_.-]+?)\.(url|host|port)|host\.bridge_ip)\s*\}\}"
)

DEFAULT_TRIES = 15
DEFAULT_SLEEP_S = 2.0
DEFAULT_TIMEOUT_S = 5.0
# curl exit codes that mean "the network could not be reached" (vs. an HTTP
# error, which proves the path works).  6 DNS, 7 connect refused, 28 timeout,
# 35 TLS handshake, 52 empty reply, 56 recv error.
_UNREACHABLE_CURL_EXITS = frozenset({6, 7, 28, 35, 52, 56})
# A health check counts as healthy on any 2xx/3xx; a 4xx/5xx still proves the
# microVM reached the service (reachable) but not that it is serving.
_HEALTHY_CODES = range(200, 400)
RECORDS_PROXY = "records-proxy"
# A records proxy on a sampler node that passes port 22 only (a National Compute
# node behind a 1:1 NAT) times out here; without this check every trial would end
# with no recorded model call and a zero reward.
RECORDS_RELAY_HINT = (
    " The records proxy is unreachable: a sampler node whose cloud firewall passes "
    "port 22 only never sees the microVM's SYN; serve it through "
    "agentic.setup.records_relay (a reverse forward to a relay the microVMs reach) "
    "and check the relay's firewall allows this sandbox's egress address."
)


class NetworkError(RuntimeError):
    """A sidecar endpoint is misconfigured or unreachable from the microVM."""


class ExecResult(Protocol):
    """The shape of :meth:`AgentenvSandbox.exec`'s return value this module reads.

    Matches ``harbor`` ``ExecResult`` (``boat_env.py``): ``stdout`` / ``stderr``
    may be ``None``; ``return_code`` is the command's exit status.
    """

    stdout: str | None
    stderr: str | None
    return_code: int


# An async "run this shell command in the microVM" callable.  The B1 provider
# passes ``AgentenvSandbox.exec`` (or a thin lambda over it); this module only
# needs single-shot string commands and the exit code + stdout.
ExecFn = Callable[[str], Awaitable[ExecResult]]


@dataclass(frozen=True)
class SidecarEndpoint:
    """A logical service a microVM must reach, resolved to a concrete URL.

    ``name`` is the placeholder key (``task-server``, ``records-proxy``, ...).
    ``url`` is the base URL the microVM uses (scheme + host + optional port, no
    trailing slash).  ``health_path`` is an optional path (e.g. ``/health``)
    probed by :func:`check_reachability`.  ``required`` endpoints make
    :func:`self_check` raise when unreachable; optional ones only warn.
    """

    name: str
    url: str
    health_path: str | None = None
    required: bool = True

    @property
    def host(self) -> str:
        """The hostname / IP from ``url`` (what a ``{{...host}}`` placeholder wants)."""
        return urlsplit(self.url).hostname or ""

    @property
    def port(self) -> int | None:
        """The explicit port in ``url``, or the scheme default (80/443), else ``None``."""
        parts = urlsplit(self.url)
        if parts.port is not None:
            return parts.port
        return {"http": 80, "https": 443}.get(parts.scheme)

    @property
    def private(self) -> bool:
        """True when ``host`` is a private/loopback/link-local IP literal.

        Such a host is denied by AgentENV's default egress and needs an allow
        rule (:func:`egress_allow_cidrs`).  A DNS name is treated as public here
        (its address is unknown without resolving); callers that know a name
        maps to a private address should pass the IP, or add the rule anyway.
        """
        return _is_private_host(self.host)

    def probe_url(self) -> str:
        """``url`` joined with ``health_path`` (or ``url`` itself) for a curl probe."""
        if not self.health_path:
            return self.url
        return f"{self.url.rstrip('/')}/{self.health_path.lstrip('/')}"


@dataclass(frozen=True)
class EndpointHealth:
    """One endpoint's reachability as seen from inside the microVM."""

    name: str
    url: str
    reachable: bool  # the microVM's curl reached the service (any HTTP reply, or a 2xx body)
    healthy: bool  # reachable AND (no health_path, or it returned 2xx/3xx)
    http_code: int | None
    latency_s: float | None
    error: str | None


@dataclass(frozen=True)
class ReachabilityReport:
    """The per-endpoint result of a self-check and whether every required one passed."""

    results: tuple[EndpointHealth, ...]
    ok: bool  # every *required* endpoint is healthy

    def by_name(self, name: str) -> EndpointHealth | None:
        for r in self.results:
            if r.name == name:
                return r
        return None

    @property
    def failures(self) -> tuple[EndpointHealth, ...]:
        return tuple(r for r in self.results if not r.healthy)


# --------------------------------------------------------------------------- #
# Address classification and egress allow-list
# --------------------------------------------------------------------------- #


def _is_private_host(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # a DNS name: unknown without resolving, treated as public here
    return ip.is_private or ip.is_loopback or ip.is_link_local


def egress_allow_cidrs(endpoints: Iterable[SidecarEndpoint]) -> list[str]:
    """Host CIDRs the sandbox must allow out so its private sidecars are reachable.

    Returns a de-duplicated, sorted list of ``/32`` (IPv4) or ``/128`` (IPv6)
    CIDRs for every endpoint whose host is a private IP literal.  Pass these to
    ``SandboxNetworkOpts.allow_out`` on create (or ``PUT .../network`` at
    runtime).  Public hosts and DNS names yield nothing: public egress already
    works, and a name must be resolved to an address before it can be allowed.
    """
    cidrs: set[str] = set()
    for ep in endpoints:
        if not ep.private:
            continue
        ip = ipaddress.ip_address(ep.host)
        cidrs.add(f"{ip}/{ip.max_prefixlen}")
    return sorted(cidrs)


# --------------------------------------------------------------------------- #
# Placeholder resolution and rendering
# --------------------------------------------------------------------------- #


def resolve_placeholder_urls(
    endpoints: Iterable[SidecarEndpoint],
) -> dict[str, dict[str, str]]:
    """Map each endpoint name to the ``{url, host, port}`` a placeholder expands to.

    This is the agentenv analogue of the ``sidecars`` map
    :meth:`yeto.sandbox.ops.render` builds from the docker bridge, except the
    values are the routable endpoint URL/host/port instead of a bridge address.
    """
    out: dict[str, dict[str, str]] = {}
    for ep in endpoints:
        out[ep.name] = {
            "url": ep.url,
            "host": ep.host,
            "port": "" if ep.port is None else str(ep.port),
        }
    return out


def render_task_text(text: str, endpoints: Iterable[SidecarEndpoint]) -> str:
    """Substitute ``{{yeto.sidecar.<name>.(url|host|port)}}`` with endpoint values.

    The agentenv counterpart of :meth:`yeto.sandbox.ops.render`.  Raises
    :class:`NetworkError` on an unknown sidecar name or on
    ``{{yeto.host.bridge_ip}}`` (no docker bridge exists for a microVM, so a
    task that needs it is incompatible with the agentenv provider and must be
    rewritten to name a sidecar URL instead).
    """
    table = resolve_placeholder_urls(endpoints)

    def sub(m: re.Match[str]) -> str:
        if m.group(1) == "host.bridge_ip":
            raise NetworkError(
                "task uses {{yeto.host.bridge_ip}}, which has no meaning for an "
                "AgentENV microVM (no docker bridge); rewrite it to name a "
                "sidecar URL, e.g. {{yeto.sidecar.<name>.url}}"
            )
        name, field = m.group(2), m.group(3)
        sc = table.get(name)
        if sc is None:
            raise NetworkError(
                f"task names unknown sidecar {name!r}; known: {sorted(table)}"
            )
        return sc[field]

    return PLACEHOLDER.sub(sub, text)


# --------------------------------------------------------------------------- #
# Reachability self-check (run inside the microVM via the injected exec_fn)
# --------------------------------------------------------------------------- #


def _probe_command(url: str, *, timeout_s: float) -> str:
    """A curl that prints ``<http_code> <time_total>`` and never reads the body.

    ``-sS`` quiet but show errors, ``-o /dev/null`` drop the body, ``--max-time``
    bound the whole call; a non-zero exit means the network failed (DNS/connect/
    timeout), a zero exit with an HTTP code means the service was reached.
    """
    return (
        "curl -sS -o /dev/null "
        f"-w '%{{http_code}} %{{time_total}}' --max-time {int(timeout_s)} "
        f"{shlex.quote(url)}"
    )


def _interpret(ep: SidecarEndpoint, res: ExecResult) -> EndpointHealth:
    """Turn one curl ``ExecResult`` into an :class:`EndpointHealth`."""
    url = ep.probe_url()
    code: int | None = None
    latency: float | None = None
    out = (res.stdout or "").strip()
    if out:
        parts = out.split()
        try:
            code = int(parts[0])
        except (ValueError, IndexError):
            code = None
        if len(parts) > 1:
            try:
                latency = float(parts[1])
            except ValueError:
                latency = None
    if res.return_code != 0 and code is None:
        why = "network unreachable" if res.return_code in _UNREACHABLE_CURL_EXITS else "curl error"
        err = (res.stderr or "").strip()
        return EndpointHealth(
            name=ep.name, url=url, reachable=False, healthy=False,
            http_code=None, latency_s=latency,
            error=f"{why} (curl exit {res.return_code}){f': {err}' if err else ''}",
        )
    # curl exited 0 (or gave an HTTP code): the service was reached.
    healthy = code is not None and (ep.health_path is None or code in _HEALTHY_CODES)
    error = None
    if code is not None and ep.health_path is not None and code not in _HEALTHY_CODES:
        error = f"reached but health {ep.health_path} returned HTTP {code}"
    return EndpointHealth(
        name=ep.name, url=url, reachable=code is not None, healthy=healthy,
        http_code=code, latency_s=latency, error=error,
    )


async def check_reachability(
    exec_fn: ExecFn,
    endpoints: Sequence[SidecarEndpoint],
    *,
    tries: int = DEFAULT_TRIES,
    sleep_s: float = DEFAULT_SLEEP_S,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> ReachabilityReport:
    """Probe every endpoint from inside the microVM and report per-endpoint health.

    For each endpoint, runs ``curl`` via ``exec_fn`` up to ``tries`` times
    (``sleep_s`` apart, bounded by ``timeout_s`` per call), stopping at the first
    healthy reply.  ``sleep`` defaults to :func:`asyncio.sleep`; tests inject a
    no-op.  Never raises for an unreachable endpoint -- that is reported in the
    result; :func:`self_check` is the raising wrapper.

    The probe assumes ``curl`` is on the microVM (true for the pi/codex task
    images; if a task image lacks it, ``exec_fn`` returns a non-zero exit and the
    endpoint reports ``curl error``, which the caller can special-case).
    """
    if tries < 1:
        raise ValueError("tries must be >= 1")
    if sleep is None:
        import asyncio

        sleep = asyncio.sleep
    results: list[EndpointHealth] = []
    for ep in endpoints:
        health = _validate_endpoint_url(ep)
        if health is None:
            cmd = _probe_command(ep.probe_url(), timeout_s=timeout_s)
            for attempt in range(tries):
                health = _interpret(ep, await exec_fn(cmd))
                if health.healthy:
                    break
                if attempt + 1 < tries:
                    await sleep(sleep_s)
        results.append(health)
    required_failed = [r for r in results if not r.healthy and _required(endpoints, r.name)]
    return ReachabilityReport(results=tuple(results), ok=not required_failed)


async def self_check(
    exec_fn: ExecFn,
    endpoints: Sequence[SidecarEndpoint],
    **kwargs: object,
) -> ReachabilityReport:
    """:func:`check_reachability`, but raise :class:`NetworkError` if a required
    endpoint is unreachable.

    This is the one call the B1 provider makes after a microVM is up and before
    it hands the sandbox to a trial: a failed judge/records path is exactly the
    sbx-3-style silent-zero failure this guards against, so it must fail loud and
    early.  The message lists every failing required endpoint.
    """
    report = await check_reachability(exec_fn, endpoints, **kwargs)  # type: ignore[arg-type]
    hard = [r for r in report.failures if _required(endpoints, r.name)]
    if hard:
        lines = "; ".join(f"{r.name} @ {r.url}: {r.error or 'unhealthy'}" for r in hard)
        hint = RECORDS_RELAY_HINT if any(r.name == RECORDS_PROXY and not r.reachable for r in hard) else ""
        raise NetworkError(
            f"microVM cannot reach {len(hard)} required sidecar(s): {lines}. "
            "Check the egress allow-list (egress_allow_cidrs) and that the "
            f"sidecar is up on a microVM-routable address.{hint}"
        )
    return report


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _required(endpoints: Iterable[SidecarEndpoint], name: str) -> bool:
    return any(ep.name == name and ep.required for ep in endpoints)


def _validate_endpoint_url(ep: SidecarEndpoint) -> EndpointHealth | None:
    """Return a failed health for a malformed URL, else ``None`` (probe it)."""
    parts = urlsplit(ep.url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return EndpointHealth(
            name=ep.name, url=ep.url, reachable=False, healthy=False,
            http_code=None, latency_s=None,
            error=f"malformed endpoint URL {ep.url!r} (need http(s)://host[:port])",
        )
    return None


def endpoints_from_mapping(
    spec: Mapping[str, object],
    *,
    optional: Iterable[str] = (),
) -> list[SidecarEndpoint]:
    """Build endpoints from ``{name: url}`` or ``{name: {"url":..., "health":...}}``.

    A convenience for the provider: ``spec`` is whatever the agentic round hands
    down (the records-proxy base URL and any per-task sidecar URLs).  Names in
    ``optional`` get ``required=False``.  Raises :class:`NetworkError` on a value
    that is neither a string URL nor a mapping with a ``url``.
    """
    optional_set = set(optional)
    out: list[SidecarEndpoint] = []
    for name, value in spec.items():
        if isinstance(value, str):
            url, health = value, None
        elif isinstance(value, Mapping):
            raw_url = value.get("url")
            if not isinstance(raw_url, str):
                raise NetworkError(f"sidecar {name!r} spec has no string 'url'")
            url = raw_url
            health_val = value.get("health") or value.get("health_path")
            health = health_val if isinstance(health_val, str) else None
        else:
            raise NetworkError(f"sidecar {name!r} must be a URL string or a mapping, got {type(value).__name__}")
        out.append(SidecarEndpoint(name=name, url=url, health_path=health, required=name not in optional_set))
    return out


def resolve_hostname_to_cidr(host: str) -> str | None:
    """Best-effort DNS resolution of a name to a ``/32`` CIDR for the allow-list.

    Offline-safe: returns ``None`` if ``host`` is already an IP, is empty, or
    does not resolve.  The provider may call this to widen the egress allow-list
    when a sidecar is named by DNS but lives on a private address.  This is the
    only function here that may touch the network (a local resolver lookup); it
    is never called by the reachability path.
    """
    if not host or _is_private_host(host):
        return None
    try:
        ipaddress.ip_address(host)
        return None  # already an IP; egress_allow_cidrs handles it
    except ValueError:
        pass
    try:
        addr = socket.gethostbyname(host)
    except OSError:
        return None
    ip = ipaddress.ip_address(addr)
    return f"{ip}/{ip.max_prefixlen}"
