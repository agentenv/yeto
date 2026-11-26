"""Store records of the node runtime: assignments and node state.

Under the run's store root (the root that holds ``sample-buffer/``):

* ``assignments/<ticket>.json``: the control plane's assignment of one machine
  (:class:`Assignment`): the shard, the shard it continues, the rendered spec,
  the code hash, the model plan, the node profile, a ``generation`` that grows
  with each rewrite, and the sha256 of the ticket's token. A node reads its
  assignment with the token (:func:`read_assignment`); a machine opened by hand
  joins the same way, so the pool's fence and handoff rules hold for it as for
  an ordered one;
* ``nodes/<ticket>/g<generation>/ready.json``: the sampler joined its shard;
* ``nodes/<ticket>/g<generation>/heartbeat.json``: rewritten every
  ``heartbeat_s`` with the stage (``preflight``, ``code``, ``model``,
  ``starting``, ``engine``, ``running``, ``left``) and an engine summary
  (generation tokens, running / queued requests, rounds);
* ``nodes/<ticket>/g<generation>/exit.json``: exit code and reason.

The control plane judges a node from these records alone (:func:`node_status`,
:func:`probe_of`): a node behind NAT that only dials out is as visible as one
that answers SSH. Node records live under their assignment's generation: a
node still running an older assignment of the same ticket writes under its own
generation and never overwrites the records of a newer one; the control plane
reads the generation of the assignment in the store (the highest written).

The token is a ticket binding, not an authentication: anyone who can write the
run store can write an assignment, and anyone holding a ticket's token can read
its assignment (spec, code hash, model plan). It keeps a node from starting on
another ticket's assignment by mistake and keeps the token itself out of the
store (only its salted sha256 is there). Store credentials are the security
boundary. A token can carry a deadline (``expires_at``): an expired assignment
is refused at read. A hand-opened machine's token has one by default; the token
of a pool-ordered machine has none (the transport hands it over at the start and
the runtime reads it then; a later start writes a new assignment and token).
The runtime keeps the token out of the sampler's environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

ASSIGNMENTS_DIR = "assignments"
NODES_DIR = "nodes"
NODE_RECORDS = ("ready", "heartbeat", "exit")
TOKEN_ENV = "YETO_NODE_TOKEN"
DEFAULT_STALE_S = 300.0  # a heartbeat older than this, without an exit record: stuck

# the stages a heartbeat reports, in order; the sampler writes ``engine`` on, the
# node runtime the ones before
BOOT_STAGES = ("assigned", "preflight", "code", "model", "spec", "starting")
SAMPLER_STAGES = ("engine", "running", "left")


class AssignmentError(RuntimeError):
    """No assignment for the ticket, or the token is not the ticket's."""


def _safe(part: str, what: str) -> str:
    text = str(part)
    if not text or "/" in text or text in (".", "..") or any(c.isspace() for c in text):
        raise ValueError(f"{what} {part!r} must be one plain word (no '/', no blanks)")
    return text


def assignment_key(ticket: str) -> str:
    return f"{ASSIGNMENTS_DIR}/{_safe(ticket, 'ticket')}.json"


def ticket_of_key(key: str) -> str:
    """``assignments/<ticket>.json`` or a bare ticket -> the ticket."""
    text = str(key).strip()
    if text.startswith(f"{ASSIGNMENTS_DIR}/"):
        text = text[len(ASSIGNMENTS_DIR) + 1:]
    return _safe(text.removesuffix(".json"), "ticket")


def node_key(ticket: str, record: str, generation: int) -> str:
    """``nodes/<ticket>/g<generation>/<record>.json``: each assignment generation has its own records."""
    if record not in NODE_RECORDS:
        raise ValueError(f"node record {record!r} is not one of {NODE_RECORDS}")
    if int(generation) < 1:
        raise ValueError(f"generation {generation!r} must be >= 1")
    return f"{NODES_DIR}/{_safe(ticket, 'ticket')}/g{int(generation)}/{record}.json"


def new_token() -> str:
    return secrets.token_urlsafe(24)


def token_digest(token: str) -> str:
    return hashlib.sha256(f"yeto-node:{token}".encode()).hexdigest()


def _read_json(objects: Any, key: str) -> dict[str, Any] | None:
    raw = objects.read(key)
    if raw is None:
        return None
    try:
        doc = json.loads(raw)
    except ValueError:
        return {"torn": True}
    return doc if isinstance(doc, dict) else {"torn": True}


def _write_json(objects: Any, key: str, doc: Mapping[str, Any]) -> None:
    objects.write(key, (json.dumps(dict(doc), sort_keys=True) + "\n").encode())


# -- assignments -------------------------------------------------------------------


@dataclass(frozen=True)
class CodeRef:
    """The yeto tree a node runs: ``sha`` (short commit or content hash), the store
    ``key`` of its tar and the tar's ``sha256``."""

    sha: str
    key: str
    sha256: str
    bytes: int = 0


@dataclass(frozen=True)
class ModelPlan:
    """How the node gets the model. ``source``: ``store`` (``key`` under the bucket),
    ``volume`` / ``preset`` (already at ``dest``), ``hf`` (``repo`` at ``revision``),
    ``none``. ``dest``: a node path (container paths ``/models/...`` map to the
    profile's ``models``)."""

    source: str = "none"
    key: str | None = None
    dest: str | None = None
    include: str | None = None
    repo: str | None = None
    revision: str | None = None
    max_mb_s: int = 2500
    jobs: int = 16
    timeout_s: float = 3600.0


@dataclass(frozen=True)
class Assignment:
    ticket: str
    shard: str
    spec: Mapping[str, Any]
    code: CodeRef
    token_sha256: str
    generation: int = 1
    run: str | None = None
    pool: str | None = None
    source: str | None = None  # the provider kind (nc, daytona, sky, modal, ssh, manual)
    continues_shard: str | None = None
    data_slot: str | None = None
    model: ModelPlan = field(default_factory=ModelPlan)
    profile: Mapping[str, Any] | None = None
    env: Mapping[str, str] = field(default_factory=dict)  # non-secret environment of the sampler
    store_uri: str | None = None  # the run store as the node reaches it (None: the node's --store)
    image: str | None = None  # a container launch's image (None: the profile's)
    python: str | None = None  # the sampler interpreter (None: the image's, or this runtime's)
    sudo: bool = False  # the container engine needs sudo
    cards: tuple[int, ...] | None = None  # the cards the sampler gets (None: every card)
    preflight: Mapping[str, Any] = field(default_factory=dict)  # yeto.node.preflight.Requirements fields
    # "container": the spec is written in container paths (/work, /models: moved for a process
    # launch); "host": in the node's own paths (only the placeholders are filled)
    spec_paths: str = "container"
    created_at: float = 0.0
    expires_at: float | None = None  # the token's deadline (None: none); read refuses after it

    def doc(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any]) -> "Assignment":
        known = set(cls.__dataclass_fields__)
        values = {k: v for k, v in doc.items() if k in known}  # newer fields: ignored
        values["code"] = CodeRef(**{k: v for k, v in dict(doc["code"]).items()
                                    if k in CodeRef.__dataclass_fields__})
        values["model"] = ModelPlan(**{k: v for k, v in dict(doc.get("model") or {}).items()
                                       if k in ModelPlan.__dataclass_fields__})
        if values.get("cards") is not None:
            values["cards"] = tuple(int(c) for c in values["cards"])
        return cls(**values)


def write_assignment(objects: Any, assignment: Assignment) -> str:
    key = assignment_key(assignment.ticket)
    _write_json(objects, key, assignment.doc())
    return key


def assign(objects: Any, *, ticket: str, shard: str, spec: Mapping[str, Any], code: CodeRef,
           token: str | None = None, clock: Any = time.time, **fields_: Any) -> tuple[Assignment, str]:
    """Write ``assignments/<ticket>.json`` (generation one past the one there) and
    return it with the ticket's token (a new token unless ``token``). The token goes
    to the node by the transport, or to an operator for a machine opened by hand."""
    if spec.get("sampler_id") not in (None, shard):
        raise ValueError(f"spec sampler_id {spec.get('sampler_id')!r} is not the shard {shard!r}")
    token = token or new_token()
    generation = next_generation(objects, ticket)
    assignment = Assignment(ticket=ticket, shard=shard, spec=dict(spec), code=code, token_sha256=token_digest(token),
                            generation=generation, created_at=float(clock()), **fields_)
    write_assignment(objects, assignment)
    return assignment, token


def _listed_generation(objects: Any, ticket: str) -> int | None:
    """The highest ``g<N>`` under ``nodes/<ticket>/`` (0: none); None when the store cannot list."""
    lister = getattr(objects, "list", None)
    if lister is None:
        return None
    prefix = f"{NODES_DIR}/{_safe(ticket, 'ticket')}/g"
    found = 0
    for key in lister(prefix):  # a listing failure raises: no guess at the generation
        head = str(key)[len(prefix):].split("/", 1)[0]
        if head.isdigit():
            found = max(found, int(head))
    return found


def next_generation(objects: Any, ticket: str) -> int:
    """One past every generation the ticket has had: the assignment's (a revoked one keeps
    its tombstone) and every ``nodes/<ticket>/g<N>`` a node wrote. A read failure raises
    and an unreadable assignment is refused: a generation that went back would let an
    old node's records pass for the new assignment's."""
    previous = _read_json(objects, assignment_key(ticket))  # a store error raises here
    listed = _listed_generation(objects, ticket)
    if previous is not None and previous.get("torn"):  # a node of the lost generation may still be booting
        raise AssignmentError(f"the assignment of ticket {ticket} is not readable JSON: its generation is "
                              "unknown, not reassigned. Revoke it (a tombstone past every generation its node "
                              f"records show), then assign again: python -m yeto.node revoke --assignment {ticket} "
                              "--store <run store> [--env-file <store env file>]")
    return max(int((previous or {}).get("generation") or 0), listed or 0) + 1


def read_assignment(objects: Any, key: str, token: str, *, clock: Any = time.time) -> Assignment:
    """The assignment of ``key`` (a ticket or ``assignments/<ticket>.json``) when
    ``token`` is the ticket's, the document is that ticket's and its deadline has
    not passed; :class:`AssignmentError` otherwise. A ticket binding, not an
    authentication (see the module notes)."""
    ticket = ticket_of_key(key)
    doc = _read_json(objects, assignment_key(ticket))
    if doc is None:
        raise AssignmentError(f"no assignment for ticket {ticket} in the store")
    if doc.get("torn"):
        raise AssignmentError(f"the assignment of ticket {ticket} is not readable JSON")
    if doc.get("revoked"):
        raise AssignmentError(f"the assignment of ticket {ticket} was revoked")
    if not token or not secrets.compare_digest(str(doc.get("token_sha256") or ""), token_digest(token)):
        raise AssignmentError(f"the token is not the token of ticket {ticket}")
    if doc.get("ticket") != ticket:  # a document copied under another ticket's key
        raise AssignmentError(f"the assignment under ticket {ticket} names ticket {doc.get('ticket')!r}")
    if doc.get("expires_at") is not None and float(clock()) > float(doc["expires_at"]):
        raise AssignmentError(f"the assignment of ticket {ticket} expired (its token is no longer valid)")
    return Assignment.from_doc(doc)


def revoke_assignment(objects: Any, ticket: str) -> None:
    """Replace the assignment with a tombstone that keeps its generation (a later
    assignment of the ticket continues from it; a node can no longer read it)."""
    doc = _read_json(objects, assignment_key(ticket))
    listed = _listed_generation(objects, ticket) or 0
    known = 0 if doc is None or doc.get("torn") else int(doc.get("generation") or 0)
    generation = max(known, listed) + (1 if doc is not None and doc.get("torn") else 0)  # past a torn one
    _write_json(objects, assignment_key(ticket), {"ticket": ticket, "revoked": True, "generation": generation})


# -- node records --------------------------------------------------------------------


class NodeReporter:
    """Writes this node's ``ready`` / ``heartbeat`` / ``exit`` records. Every write is
    best effort for the caller (:meth:`safe`): a store outage never stops a sampler."""

    def __init__(self, objects: Any, ticket: str, generation: int, *, clock: Any = time.time,
                 base: Mapping[str, Any] | None = None) -> None:
        self.objects, self.ticket, self.generation, self.clock = objects, ticket, int(generation), clock
        self.base = dict(base or {})
        self.failures = 0

    def _doc(self, **fields_: Any) -> dict[str, Any]:
        return {"ticket": self.ticket, "generation": self.generation, "time": float(self.clock()),
                "host": socket.gethostname(), "pid": os.getpid(), **self.base, **fields_}

    def write(self, record: str, **fields_: Any) -> dict[str, Any]:
        doc = self._doc(**fields_)
        _write_json(self.objects, node_key(self.ticket, record, self.generation), doc)
        return doc

    def safe(self, record: str, **fields_: Any) -> dict[str, Any] | None:
        try:
            return self.write(record, **fields_)
        except Exception as error:  # noqa: BLE001 - observation only
            self.failures += 1
            if self.failures == 1:
                print(f"[yeto-node] {record} record not written: {type(error).__name__}: {error}", flush=True)
            return None

    def ready(self, **fields_: Any) -> dict[str, Any] | None:
        return self.safe("ready", **fields_)

    def heartbeat(self, stage: str, **fields_: Any) -> dict[str, Any] | None:
        return self.safe("heartbeat", stage=stage, **fields_)

    def exit(self, code: int | None, reason: str, **fields_: Any) -> dict[str, Any] | None:
        return self.safe("exit", exit_code=code, reason=str(reason), **fields_)


def clear_node_records(objects: Any, ticket: str, generation: int) -> None:
    """Remove the node records of one generation of ``ticket`` (a replaced assignment's;
    a node still running it writes them again under its own generation, never the new one's)."""
    for record in NODE_RECORDS:
        try:
            objects.delete(node_key(ticket, record, generation))
        except Exception:  # noqa: BLE001 - absent is fine
            pass


@dataclass(frozen=True)
class NodeStatus:
    """What the store says about one ticket's node. ``state``: ``unassigned``,
    ``assigned`` (nothing written yet), ``booting`` (heartbeats, no ready),
    ``ready``, ``stuck`` (no heartbeat for ``stale_s``, no exit), ``exited``."""

    ticket: str
    state: str
    generation: int | None = None
    stage: str | None = None
    heartbeat_age_s: float | None = None
    progress_age_s: float | None = None  # since the sampler joined or last finished a round (None: unknown)
    exit_code: int | None = None
    reason: str | None = None
    ready: Mapping[str, Any] | None = None
    heartbeat: Mapping[str, Any] | None = None
    exit: Mapping[str, Any] | None = None

    @property
    def engine(self) -> Mapping[str, Any]:
        return dict((self.heartbeat or {}).get("engine") or {})


def node_status(objects: Any, ticket: str, *, now: float | None = None, stale_s: float = DEFAULT_STALE_S,
                generation: int | None = None) -> NodeStatus:
    """The node of ``ticket`` from the store alone. ``generation``: the assignment's
    (default: read from it, the highest generation written); only its records count."""
    if generation is None:
        assignment = _read_json(objects, assignment_key(ticket))
        if assignment is None or assignment.get("torn") or assignment.get("revoked"):
            return NodeStatus(ticket, "unassigned")
        generation = int(assignment.get("generation") or 1)
    now = time.time() if now is None else float(now)
    docs = {}
    for record in NODE_RECORDS:
        doc = _read_json(objects, node_key(ticket, record, int(generation)))
        if doc is not None and not doc.get("torn") and int(doc.get("generation") or 0) == int(generation):
            docs[record] = doc
    ready, beat, ended = docs.get("ready"), docs.get("heartbeat"), docs.get("exit")
    newest = max((float(d.get("time") or 0.0) for d in docs.values()), default=None)
    age = None if newest is None else max(0.0, now - newest)
    stage = None if beat is None else beat.get("stage")
    progress_at = None if beat is None else beat.get("progress_at")
    progress_age = None if progress_at is None else max(0.0, now - float(progress_at))
    common = {"generation": generation, "stage": stage, "heartbeat_age_s": age, "progress_age_s": progress_age,
              "ready": ready, "heartbeat": beat, "exit": ended}
    if ended is not None:
        code = ended.get("exit_code")
        return NodeStatus(ticket, "exited", exit_code=None if code is None else int(code),
                          reason=ended.get("reason"), **common)
    if not docs:
        return NodeStatus(ticket, "assigned", **common)
    if age is not None and age > float(stale_s):
        return NodeStatus(ticket, "stuck", reason=f"no heartbeat for {age:.0f}s (stage {stage})", **common)
    return NodeStatus(ticket, "ready" if ready is not None else "booting", **common)


def probe_of(status: NodeStatus) -> Any:
    """The pool's :class:`yeto.launch.pool.status.Probe` of a store status: the same
    reading an SSH probe gives, from the store. ``progress_age_s`` is the age of the
    sampler's last progress (joined, or a round done), not of its heartbeat: a sampler
    that heartbeats without rounds reads as stalled, as its event tape would. A node
    assigned or booting reads as a starting sampler (running, not joined), as a
    started container does; a stuck node as a running sampler without progress."""
    from yeto.launch.pool.status import PROBE_UNREACHABLE, Probe  # noqa: PLC0415

    if status.state in ("unassigned",):
        return Probe(node=PROBE_UNREACHABLE)
    engine = status.engine
    if status.state == "exited":
        return Probe(node="ok", sampler="exited", exit_code=status.exit_code, joined=status.ready is not None)

    def num(key: str) -> float | None:
        value = engine.get(key)
        return None if value is None else float(value)

    running, queued = num("running"), num("queued")
    progress = status.progress_age_s
    if status.state == "stuck" and progress is None:
        progress = status.heartbeat_age_s
    health = (status.heartbeat or {}).get("health")  # the node runtime's health check (yeto.hw.node_health)
    return Probe(node="ok", sampler="running", joined=status.ready is not None,
                 progress_age_s=progress, gen_tokens=num("gen_tokens"),
                 running=None if running is None else int(running), queued=None if queued is None else int(queued),
                 trainer_wait=bool((status.heartbeat or {}).get("trainer_wait")),
                 health=health if isinstance(health, dict) else None)
