"""The relay host side of ``agentic.setup.records_relay`` (one idempotent command).

AgentENV microVMs post their model calls to the sampler's records proxy. A sampler
node that passes port 22 only (a National Compute node: a private address behind
a 1:1 NAT whose cloud firewall passes 22 alone) cannot serve it directly, so each
node holds a reverse forward (``ssh -N -R <bind>:<port>:127.0.0.1:<proxy port>``,
:meth:`yeto.launch.agentic_setup.Provisioner.records_relay`) to a relay host the
microVMs reach. This module prepares that relay host once per run, from the same
sampler spec::

    python -m yeto.launch.nc records-relay --spec <spec> [--shards N] [--dry-run]

Three parts, each changed only when it differs from what is there (a second run
changes nothing; nothing here restarts sshd or drops a connection):

``sshd``      a drop-in ``/etc/ssh/sshd_config.d/60-yeto-records-relay.conf`` with
              ``GatewayPorts clientspecified`` (a forward may bind the public
              address) and ``ClientAliveInterval``, checked by ``sshd -t`` and
              ``sshd -T``, then a reload of the ssh service
``keys``      one restricted ``authorized_keys`` line per node key in a managed
              block of the relay user: ``restrict,port-forwarding,permitlisten=
              "<bind>:<port>"`` (a key can forward its shard's port and nothing
              else: no shell, no other port)
``firewall``  an iptables chain ``YETO-RECORDS-RELAY`` on the listen ports: accept
              from ``allow_cidrs`` (the sandbox workers' egress addresses), drop
              the rest; skipped with no ``allow_cidrs`` (the ports then answer
              anyone, the records secret still gates each request)

The relay host's own cloud firewall must pass the listen ports; that is set where
the host is ordered, not here. Only public keys leave the launcher.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import agentic_setup as su

SSHD_DROPIN = "/etc/ssh/sshd_config.d/60-yeto-records-relay.conf"
CHAIN = "YETO-RECORDS-RELAY"
SSHD_CONF = "GatewayPorts clientspecified\nClientAliveInterval 15\nClientAliveCountMax 4\n"
RESULT = "YETO_RELAY"


class RelayError(RuntimeError):
    pass


@dataclass(frozen=True)
class Grant:
    """One ``authorized_keys`` line: a node key and the listen addresses it may bind."""

    shards: tuple[int, ...]
    public_key: str  # "<type> <base64>"
    listens: tuple[str, ...]

    def line(self, label: str) -> str:
        opts = ",".join(["restrict", "port-forwarding"] + [f'permitlisten="{a}"' for a in self.listens])
        shards = ",".join(map(str, self.shards))
        return f"{opts} {self.public_key} {su.RELAY_TAG}:{label}:{shards}"


@dataclass(frozen=True)
class Plan:
    relay: su.RelaySection
    label: str
    grants: tuple[Grant, ...]

    @property
    def ports(self) -> tuple[int, ...]:
        return tuple(sorted({int(a.rpartition(":")[2]) for g in self.grants for a in g.listens}))


def load_section(spec: Mapping[str, Any], *, base_dir: str | Path | None) -> su.RelaySection:
    """``agentic.setup.records_relay`` of a sampler spec; relative key paths resolve
    against ``base_dir`` (``{shard}`` is kept for :meth:`RelaySection.key_of`)."""
    doc = ((spec.get("agentic") or {}).get("setup") or {}) if isinstance(spec.get("agentic"), Mapping) else {}
    if not isinstance(doc, Mapping) or not doc.get("records_relay"):
        raise RelayError("the spec has no agentic.setup.records_relay section")

    def ref(value: Any, what: str) -> su.SecretRef:
        r = su.SecretRef.parse(value, what)
        if r.path:
            p = Path(r.path).expanduser()
            return su.SecretRef(path=str(p if p.is_absolute() or base_dir is None else Path(base_dir) / p))
        return r

    try:
        return su.parse_relay(doc, ref=ref)
    except ValueError as error:
        raise RelayError(str(error)) from None


def public_key(ref: su.SecretRef, *, environ: Mapping[str, str] | None = None,
               run: Callable[..., Any] = subprocess.run) -> str:
    """``<type> <base64>`` of a private key (``ssh-keygen -y`` on the launcher; the
    private key is written to a 0600 temp file only)."""
    problem = ref.problem(os.environ if environ is None else environ, private=True)
    if problem:
        raise RelayError(f"records_relay key {ref.describe()}: {problem}")
    _, pub = su.key_identity(ref, environ=environ, run=run)
    parts = pub.split()
    if len(parts) < 2 or not parts[0].startswith(("ssh-", "ecdsa-", "sk-")):
        raise RelayError(f"records_relay key {ref.describe()}: ssh-keygen gave no public key (value not shown)")
    return f"{parts[0]} {parts[1]}"


def plan(section: su.RelaySection, *, shards: int | None = None, label: str = "default",
         environ: Mapping[str, str] | None = None, run: Callable[..., Any] = subprocess.run) -> Plan:
    """The grants for ``shards`` shards (default: one per listen port)."""
    if not su._NAME.match(label):  # the same plain-name rule as the spec
        raise RelayError(f"--label {label!r} must be a plain name")
    n = len(section.ports) if shards is None else shards
    if n < 1 or n > len(section.ports):
        raise RelayError(f"--shards {n}: records_relay.listen_port {section.port_lo}-{section.port_hi} has "
                         f"{len(section.ports)} port(s), one per shard")
    listen = [f"{section.bind}:{section.port_lo + k}" for k in range(n)]
    if section.per_shard_key:
        grants = tuple(Grant((k,), public_key(section.key_of(k), environ=environ, run=run), (listen[k],))
                       for k in range(n))
        if len({g.public_key for g in grants}) != len(grants):
            raise RelayError("records_relay: two shards share one key file content; give each shard its own key "
                             "(sshd uses the first authorized_keys line of a key)")
    else:
        grants = (Grant(tuple(range(n)), public_key(section.key, environ=environ, run=run), tuple(listen)),)
    return Plan(relay=section, label=label, grants=grants)


def script(p: Plan, *, persist_firewall: bool = False, sshd_dropin: str = SSHD_DROPIN) -> str:
    """The bash script run on the relay host (as its admin login, root or sudo)."""
    r = p.relay
    begin, end = f"# {su.RELAY_TAG} {p.label} begin", f"# {su.RELAY_TAG} {p.label} end"
    block = "\n".join([begin, *(g.line(p.label) for g in p.grants), end]) + "\n"
    lo, hi = min(p.ports), max(p.ports)
    dport = f"{lo}:{hi}" if hi != lo else str(lo)
    rules = [f"-A {CHAIN} -s {c} -p tcp -m tcp --dport {dport} -j ACCEPT" for c in r.allow_cidrs]
    rules.append(f"-A {CHAIN} -p tcp -m tcp --dport {dport} -j DROP")
    keys = " ".join(shlex.quote(g.public_key.split()[1]) for g in p.grants)
    q = shlex.quote
    dropin = q(sshd_dropin)
    return f"""set -eu
if [ "$(id -u)" = 0 ]; then S=; else S=sudo; fi
say() {{ echo "{RESULT} $*"; }}

# -- sshd: GatewayPorts clientspecified (a reload keeps every open session)
want={q(SSHD_CONF)}
if [ "$($S cat {dropin} 2>/dev/null || true)" != "$(printf %s "$want")" ]; then
  $S mkdir -p {q(str(Path(sshd_dropin).parent))}
  printf %s "$want" | $S tee {dropin}.tmp >/dev/null
  $S mv -f {dropin}.tmp {dropin}
  $S sshd -t || {{ $S rm -f {dropin}; say sshd failed "sshd -t rejects the config (drop-in removed)"; exit 3; }}
  $S systemctl reload ssh 2>/dev/null || $S systemctl reload sshd
  sshd_state=changed
else
  sshd_state=ok
fi
if ! $S sshd -T 2>/dev/null | grep -qi '^gatewayports clientspecified'; then
  say sshd failed "sshd -T has no gatewayports clientspecified (no Include of /etc/ssh/sshd_config.d? set it in sshd_config)"
  exit 3
fi
say sshd $sshd_state

# -- keys: a managed block of restricted lines in {r.user}'s authorized_keys
home=$(getent passwd {q(r.user)} | cut -d: -f6)
[ -n "$home" ] || {{ say keys failed "no user {r.user} on the relay"; exit 3; }}
ak="$home/.ssh/authorized_keys"
$S mkdir -p "$home/.ssh"
$S touch "$ak"
tmp=$(mktemp)
$S cat "$ak" | awk -v b={q(begin)} -v e={q(end)} '$0==b{{skip=1;next}} $0==e{{skip=0;next}} !skip' > "$tmp"
for k in {keys}; do
  if grep -v '{su.RELAY_TAG}:' "$tmp" | grep -qF "$k"; then
    say warn "a node key also has an authorized_keys line outside the managed block; sshd uses the first line of a key, so remove the other line"
  fi
done
printf %s {q(block)} >> "$tmp"
if $S cmp -s "$tmp" "$ak"; then keys_state=ok; else
  $S cp "$tmp" "$ak.yeto-tmp" && $S chmod 600 "$ak.yeto-tmp" && $S chown {q(r.user)}: "$ak.yeto-tmp" && $S mv -f "$ak.yeto-tmp" "$ak"
  keys_state=changed
fi
rm -f "$tmp"
say keys $keys_state

# -- firewall: the listen ports {dport} from the sandbox egress addresses only
if [ {len(r.allow_cidrs)} = 0 ]; then
  say firewall skipped "no allow_cidrs: the ports {dport} answer any source"
else
  want_rules={q(chr(10).join([f"-N {CHAIN}", *rules]))}
  $S iptables -N {CHAIN} 2>/dev/null || true
  if [ "$($S iptables -S {CHAIN})" = "$want_rules" ]; then fw=ok; else
    $S iptables -F {CHAIN}
{chr(10).join(f"    $S iptables {rule}" for rule in rules)}
    fw=changed
  fi
  if ! $S iptables -C INPUT -p tcp -m tcp --dport {dport} -j {CHAIN} 2>/dev/null; then
    $S iptables -I INPUT 1 -p tcp -m tcp --dport {dport} -j {CHAIN}
    fw=changed
  fi
  {"command -v netfilter-persistent >/dev/null && $S netfilter-persistent save >/dev/null 2>&1 || true" if persist_firewall else ":"}
  say firewall $fw
fi
"""


def describe(p: Plan, *, ssh_target: str) -> list[str]:
    r = p.relay
    lines = [(f"records relay {ssh_target} (label {p.label}): forwards bind {r.bind}, public base "
              f"http://{r.public_host}:<{min(p.ports)}..{max(p.ports)}>")]
    lines.append(f"  sshd: {SSHD_DROPIN} (GatewayPorts clientspecified, ClientAliveInterval 15), reload if changed")
    for g in p.grants:
        lines.append(f"  keys: shard(s) {','.join(map(str, g.shards))} -> permitlisten {', '.join(g.listens)} "
                     f"({g.public_key.split()[0]} ...{g.public_key.split()[1][-12:]})")
    lines.append(f"  firewall: chain {CHAIN}, accept {list(r.allow_cidrs)} on ports {min(p.ports)}-{max(p.ports)}"
                 if r.allow_cidrs else "  firewall: skipped (no allow_cidrs)")
    return lines


def apply(p: Plan, *, ssh_target: str, identity_file: str | None = None, persist_firewall: bool = False,
          run: Callable[..., Any] = subprocess.run) -> dict[str, str]:
    """Run :func:`script` on the relay; returns ``{part: ok|changed|skipped}``."""
    argv = ["ssh", "-T", "-o", "BatchMode=yes", "-p", str(p.relay.port)]
    if identity_file:
        argv += ["-i", identity_file, "-o", "IdentitiesOnly=yes"]
    out = run([*argv, ssh_target, "bash -s"], input=script(p, persist_firewall=persist_firewall),
              capture_output=True, text=True, timeout=300, check=False)
    report: dict[str, str] = {}
    warnings: list[str] = []
    for ln in (out.stdout or "").splitlines():
        if not ln.startswith(RESULT + " "):
            continue
        part, _, rest = ln[len(RESULT) + 1:].partition(" ")
        if part == "warn":
            warnings.append(rest)
            continue
        report[part] = rest.split()[0] if rest else ""
        if report[part] == "failed":
            raise RelayError(f"records relay [{part}]: {rest.partition(' ')[2]}")
    if out.returncode != 0 or not {"sshd", "keys", "firewall"} <= set(report):
        raise RelayError(f"records relay on {ssh_target}: exit {out.returncode}: "
                         f"{(out.stderr or out.stdout or '').strip()[-300:]}")
    for w in warnings:
        report.setdefault("warnings", "")
        report["warnings"] = (report["warnings"] + "; " if report["warnings"] else "") + w
    return report


def main_cmd(spec_path: str, *, shards: int | None, label: str, ssh_target: str | None,
             identity_file: str | None, dry_run: bool, persist_firewall: bool,
             run: Callable[..., Any] = subprocess.run, out: Callable[[str], None] = print) -> int:
    path = Path(spec_path).resolve()
    spec = json.loads(path.read_text().replace("__RUN_PREFIX__", "run"))
    try:
        section = load_section(spec, base_dir=path.parent)
        p = plan(section, shards=shards, label=label, run=run)
    except RelayError as error:
        out(f"records-relay: {error}")
        return 2
    target = ssh_target or f"{section.user}@{section.host}"
    for line in describe(p, ssh_target=target):
        out(line)
    if not section.allow_cidrs:
        out("note: no records_relay.allow_cidrs: the listen ports answer any source (the records secret in the "
            "URL path still gates each request); list the sandbox workers' egress addresses")
    if dry_run:
        out("records-relay --dry-run: ok (no SSH)")
        return 0
    try:
        report = apply(p, ssh_target=target, identity_file=identity_file, persist_firewall=persist_firewall,
                       run=run)
    except RelayError as error:
        out(f"records-relay: {error}")
        return 1
    out(json.dumps(report, sort_keys=True))
    return 0


__all__: Sequence[str] = ("Grant", "Plan", "RelayError", "apply", "describe", "load_section", "main_cmd", "plan",
                          "public_key", "script")
