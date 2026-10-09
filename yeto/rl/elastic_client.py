"""Island-side client for the elastic (inter-island scheduling) syncer.

Only used with ``--rl-island-scheduling elastic``; the legacy path never imports
this module. Frame layout mirrors ``syncer/src/elastic.rs`` (ElasticMsg::encode
/ decode) byte for byte:

* wire frame: the usual ``<IBQ`` header (magic, type, length) + payload;
* payload: ``syncer_epoch`` (u64 LE) first, then the message fields (LE), then
  HMAC-SHA256(key, type byte || body) over the body (32 bytes);
* message types: 15 JOIN, 16 JOIN_ACK, 17 LEAVE, 18 LEASE_HEARTBEAT,
  19 SAMPLE_INDEX, 20 DELTA_READY, 21 ELASTIC_INIT, 22 DELTA_TENSOR,
  23 ELASTIC_BASE, 24 SAMPLE_VERDICT (syncer reply to SAMPLE_INDEX).

The key comes from ``YETO_ISLAND_HMAC_KEY`` (the launcher exports it).

JOIN body (yeto-framework-decoupling 6.2a, branch s17-elastic-identity):
``syncer_epoch u64 | island_id u32 | incarnation u64 | capacity f64 |
backend_identity [32]`` -- the island's ``BackendIdentity.sha256()`` (all zero
= not declared).  The syncer pins the first admitted identity and refuses a
JOIN with any other one (Miles vs verl, another engine pin, ...).  Version
boundary: JOIN frames without the field are refused by the new syncer and the
old syncer refuses the new frame (trailing bytes); do not mix the two.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import os
import socket
import struct
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..protocol import MSG_ERROR, read_frame, write_frame

MSG_JOIN = 15
MSG_JOIN_ACK = 16
MSG_LEAVE = 17
MSG_LEASE_HEARTBEAT = 18
MSG_SAMPLE_INDEX = 19
MSG_DELTA_READY = 20
MSG_ELASTIC_INIT = 21
MSG_DELTA_TENSOR = 22
MSG_ELASTIC_BASE = 23
MSG_SAMPLE_VERDICT = 24
# 0.26: syncer reached total_steps. Provisional layout until the Rust side lands:
# syncer_epoch u64 | final_outer_version u64 | policy_hash [32] | HMAC.
MSG_FINISHED = 25
ELASTIC_TYPES = frozenset(range(MSG_JOIN, MSG_FINISHED + 1))

# SAMPLE_VERDICT codes (elastic.rs VERDICT_*).
VERDICT_ACCEPT = 0      # same outer version
VERDICT_ACCEPT_IS = 1   # stale within bound: accepted, needs importance-sampling correction
VERDICT_REJECT = 2
VERDICT_NAMES = {VERDICT_ACCEPT: "ACCEPT", VERDICT_ACCEPT_IS: "ACCEPT_IS", VERDICT_REJECT: "REJECT"}

LEAVE_REASON_REQUESTED = 0
LEAVE_REASON_LEASE_EXPIRED = 1
HMAC_LEN = 32
MAX_ELASTIC_FRAME = 128 * 1024
MAX_ELASTIC_BASE_FRAME = 1 << 36  # a full flat f32 base vector


def _frame_limit(msg_type: int) -> int:
    return MAX_ELASTIC_BASE_FRAME if msg_type == MSG_ELASTIC_BASE else MAX_ELASTIC_FRAME
HMAC_KEY_ENV = "YETO_ISLAND_HMAC_KEY"


class ElasticProtocolError(RuntimeError):
    pass


def hmac_sha256(key: bytes, msg: bytes) -> bytes:
    return _hmac.new(key, msg, hashlib.sha256).digest()


def seal(key: bytes, msg_type: int, body: bytes) -> bytes:
    return body + hmac_sha256(key, bytes([msg_type]) + body)


def open_frame(key: bytes, msg_type: int, payload: bytes) -> bytes:
    if len(payload) < HMAC_LEN:
        raise ElasticProtocolError("elastic frame shorter than its HMAC")
    body, mac = payload[:-HMAC_LEN], payload[-HMAC_LEN:]
    if not _hmac.compare_digest(hmac_sha256(key, bytes([msg_type]) + body), mac):
        raise ElasticProtocolError("elastic frame HMAC mismatch")
    return body


# ----------------------------------------------------------------------------- messages
@dataclass(frozen=True)
class Join:
    syncer_epoch: int
    island_id: int
    incarnation: int
    capacity: float
    backend_identity: bytes = bytes(32)  # BackendIdentity.sha256() bytes; zero = not declared
    TYPE = MSG_JOIN


@dataclass(frozen=True)
class JoinAck:
    syncer_epoch: int
    learner_slot: int
    membership_epoch: int
    base_version: int
    policy_hash: bytes
    catch_up: bool
    TYPE = MSG_JOIN_ACK


@dataclass(frozen=True)
class Leave:
    syncer_epoch: int
    island_id: int
    reason: int = LEAVE_REASON_REQUESTED
    TYPE = MSG_LEAVE


@dataclass(frozen=True)
class LeaseHeartbeat:
    syncer_epoch: int
    island_id: int
    membership_epoch: int
    inner_step: int
    round_wall_s: float
    TYPE = MSG_LEASE_HEARTBEAT


@dataclass(frozen=True)
class SampleIndex:
    """Wire form of ``sample_pool.SampleIndexEntry`` (elastic.rs SampleIndexEntry):
    island_id is the numeric island id, policy_hash the raw 32-byte sha256.
    Convert with :func:`sample_index_from_entry` / :func:`sample_index_to_entry`."""
    syncer_epoch: int
    island_id: int
    outer_version: int
    inner_step: int
    policy_hash: bytes
    group_id: str
    prompt_id: str
    n: int
    uri: str
    size_bytes: int
    sha256: str
    has_behavior_logprob: bool
    advantage_included: bool
    created_at: float
    schema: str
    TYPE = MSG_SAMPLE_INDEX


@dataclass(frozen=True)
class Finished:
    syncer_epoch: int
    final_outer_version: int
    policy_hash: bytes
    TYPE = MSG_FINISHED


class ElasticFinished(Exception):
    """The syncer finished the run (FINISHED, or its connection went away after the
    final version was known): the island ends normally."""

    def __init__(self, final_outer_version: int | None, reason: str) -> None:
        super().__init__(f"elastic syncer finished at outer version {final_outer_version} ({reason})")
        self.final_outer_version = final_outer_version
        self.reason = reason


@dataclass(frozen=True)
class SampleVerdict:
    syncer_epoch: int
    verdict: int  # VERDICT_*
    reason: str
    outer_lag: int
    TYPE = MSG_SAMPLE_VERDICT

    @property
    def verdict_name(self) -> str:
        return VERDICT_NAMES.get(self.verdict, f"UNKNOWN({self.verdict})")


def island_number(island_id: str | int, island_ids: Mapping[str, int] | None = None) -> int:
    """String island name -> u32 wire id: explicit map first, else a decimal string."""
    if isinstance(island_id, int) and not isinstance(island_id, bool):
        n = island_id
    elif island_ids is not None and island_id in island_ids:
        n = int(island_ids[island_id])
    elif isinstance(island_id, str) and island_id.isdigit():
        n = int(island_id)
    else:
        raise ElasticProtocolError(f"island {island_id!r} has no numeric id (pass island_ids)")
    if not 0 <= n < 1 << 32:
        raise ElasticProtocolError(f"island id {n} does not fit u32")
    return n


def policy_hash_bytes(value: str | bytes) -> bytes:
    """Hex sha256 string (sample pool form) -> raw 32 bytes (wire form)."""
    if isinstance(value, (bytes, bytearray)):
        return _hash32(bytes(value))
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ElasticProtocolError(f"policy_hash {value!r} is not hex") from exc
    return _hash32(raw)


def sample_index_from_entry(entry: Any, *, syncer_epoch: int,
                            island_ids: Mapping[str, int] | None = None) -> SampleIndex:
    """``sample_pool.SampleIndexEntry`` -> SAMPLE_INDEX frame."""
    return SampleIndex(int(syncer_epoch), island_number(entry.island_id, island_ids),
                       int(entry.outer_version), int(entry.inner_step),
                       policy_hash_bytes(entry.policy_hash), entry.group_id, entry.prompt_id,
                       int(entry.n), entry.uri, int(entry.size_bytes), entry.sha256,
                       bool(entry.has_behavior_logprob), bool(entry.advantage_included),
                       float(entry.created_at), entry.schema)


def sample_index_to_entry(msg: SampleIndex, *, island_names: Mapping[int, str] | None = None):
    """SAMPLE_INDEX frame -> ``sample_pool.SampleIndexEntry`` (validated)."""
    from .engine.sample_pool import SampleIndexEntry

    name = (island_names or {}).get(msg.island_id, str(msg.island_id))
    return SampleIndexEntry(name, msg.outer_version, msg.inner_step, msg.policy_hash.hex(),
                            msg.group_id, msg.prompt_id, msg.n, msg.uri, msg.size_bytes,
                            msg.sha256, msg.has_behavior_logprob, msg.advantage_included,
                            msg.created_at, msg.schema)


@dataclass(frozen=True)
class DeltaReady:
    syncer_epoch: int
    island_id: int
    base_version: int
    c_tokens: int
    c_steps: int
    TYPE = MSG_DELTA_READY


@dataclass(frozen=True)
class ElasticInit:
    """Initial flat f32 parameter vector (the first one received wins)."""
    syncer_epoch: int
    island_id: int
    params: tuple[float, ...]
    TYPE = MSG_ELASTIC_INIT


@dataclass(frozen=True)
class DeltaTensor:
    """theta - base as a flat f32 vector (same sign convention as PUSH_FRAGMENT)."""
    syncer_epoch: int
    island_id: int
    base_version: int
    c_tokens: int
    c_steps: int
    update: tuple[float, ...]
    TYPE = MSG_DELTA_TENSOR


@dataclass(frozen=True)
class ElasticBase:
    syncer_epoch: int
    outer_version: int
    params: tuple[float, ...]
    TYPE = MSG_ELASTIC_BASE


def _hash32(value: bytes) -> bytes:
    if len(value) != 32:
        raise ElasticProtocolError("policy_hash must be 32 bytes")
    return bytes(value)


def _f32s(values) -> bytes:
    """u64 count + little-endian f32 values (elastic.rs put_f32s)."""
    import array

    arr = array.array("f", values)
    if sys.byteorder != "little":
        arr.byteswap()
    return struct.pack("<Q", len(arr)) + arr.tobytes()


def _str(value: str) -> bytes:
    """u32 length + utf-8 (elastic.rs put_str)."""
    raw = value.encode("utf-8")
    return struct.pack("<I", len(raw)) + raw


def encode(msg: Any, key: bytes) -> tuple[int, bytes]:
    b = struct.pack("<Q", msg.syncer_epoch)
    if isinstance(msg, Join):
        b += struct.pack("<IQd", msg.island_id, msg.incarnation, msg.capacity)
        if len(msg.backend_identity) != 32:
            raise ElasticProtocolError("JOIN backend_identity must be 32 bytes")
        b += bytes(msg.backend_identity)
    elif isinstance(msg, JoinAck):
        b += struct.pack("<IQQ", msg.learner_slot, msg.membership_epoch, msg.base_version)
        b += _hash32(msg.policy_hash) + bytes([int(bool(msg.catch_up))])
    elif isinstance(msg, Leave):
        b += struct.pack("<IB", msg.island_id, msg.reason)
    elif isinstance(msg, LeaseHeartbeat):
        b += struct.pack("<IQQd", msg.island_id, msg.membership_epoch, msg.inner_step,
                         msg.round_wall_s)
    elif isinstance(msg, SampleIndex):
        b += struct.pack("<IQQ", msg.island_id, msg.outer_version, msg.inner_step)
        b += _hash32(msg.policy_hash) + _str(msg.group_id) + _str(msg.prompt_id)
        b += struct.pack("<Q", msg.n) + _str(msg.uri) + struct.pack("<Q", msg.size_bytes)
        b += _str(msg.sha256)
        b += bytes([int(bool(msg.has_behavior_logprob)), int(bool(msg.advantage_included))])
        b += struct.pack("<d", msg.created_at) + _str(msg.schema)
    elif isinstance(msg, Finished):
        b += struct.pack("<Q", msg.final_outer_version) + _hash32(msg.policy_hash)
    elif isinstance(msg, SampleVerdict):
        b += bytes([msg.verdict]) + _str(msg.reason) + struct.pack("<Q", msg.outer_lag)
    elif isinstance(msg, DeltaReady):
        b += struct.pack("<IQQQ", msg.island_id, msg.base_version, msg.c_tokens, msg.c_steps)
    elif isinstance(msg, ElasticInit):
        b += struct.pack("<I", msg.island_id) + _f32s(msg.params)
    elif isinstance(msg, DeltaTensor):
        b += struct.pack("<IQQQ", msg.island_id, msg.base_version, msg.c_tokens, msg.c_steps)
        b += _f32s(msg.update)
    elif isinstance(msg, ElasticBase):
        b += struct.pack("<Q", msg.outer_version) + _f32s(msg.params)
    else:
        raise ElasticProtocolError(f"cannot encode {type(msg).__name__}")
    return msg.TYPE, seal(key, msg.TYPE, b)


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ElasticProtocolError("truncated elastic frame")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt: str):
        s = struct.Struct("<" + fmt)
        return s.unpack(self.take(s.size))

    def f32s(self) -> tuple[float, ...]:
        import array

        (n,) = self.unpack("Q")
        arr = array.array("f")
        arr.frombytes(self.take(4 * n))
        if sys.byteorder != "little":
            arr.byteswap()
        return tuple(arr)

    def text(self) -> str:
        (n,) = self.unpack("I")
        try:
            return self.take(n).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ElasticProtocolError("string is not utf-8") from exc

    def flag(self) -> bool:
        v = self.take(1)[0]
        if v not in (0, 1):
            raise ElasticProtocolError(f"invalid bool byte {v}")
        return bool(v)


def decode(key: bytes, msg_type: int, payload: bytes) -> Any:
    r = _Reader(open_frame(key, msg_type, payload))
    (epoch,) = r.unpack("Q")
    if msg_type == MSG_JOIN:
        island, incarnation, capacity = r.unpack("IQd")
        try:
            identity = r.take(32)
        except ElasticProtocolError as exc:
            raise ElasticProtocolError(
                "JOIN without backend identity (island older than the syncer?)") from exc
        msg = Join(epoch, island, incarnation, capacity, identity)
        if not (msg.capacity > 0 and msg.capacity != float("inf")):
            raise ElasticProtocolError("JOIN capacity must be > 0")
    elif msg_type == MSG_JOIN_ACK:
        slot, mepoch, base = r.unpack("IQQ")
        msg = JoinAck(epoch, slot, mepoch, base, r.take(32), r.flag())
    elif msg_type == MSG_LEAVE:
        msg = Leave(epoch, *r.unpack("IB"))
    elif msg_type == MSG_LEASE_HEARTBEAT:
        msg = LeaseHeartbeat(epoch, *r.unpack("IQQd"))
    elif msg_type == MSG_SAMPLE_INDEX:
        island, version, inner = r.unpack("IQQ")
        h = r.take(32)
        group, prompt = r.text(), r.text()
        (n,) = r.unpack("Q")
        uri = r.text()
        (size,) = r.unpack("Q")
        sha = r.text()
        has_lp, adv = r.flag(), r.flag()
        (created,) = r.unpack("d")
        msg = SampleIndex(epoch, island, version, inner, h, group, prompt, n, uri, size, sha,
                          has_lp, adv, created, r.text())
    elif msg_type == MSG_FINISHED:
        (final,) = r.unpack("Q")
        msg = Finished(epoch, final, r.take(32))
    elif msg_type == MSG_SAMPLE_VERDICT:
        verdict = r.take(1)[0]
        reason = r.text()
        (lag,) = r.unpack("Q")
        msg = SampleVerdict(epoch, verdict, reason, lag)
    elif msg_type == MSG_DELTA_READY:
        msg = DeltaReady(epoch, *r.unpack("IQQQ"))
    elif msg_type == MSG_ELASTIC_INIT:
        (island,) = r.unpack("I")
        msg = ElasticInit(epoch, island, r.f32s())
    elif msg_type == MSG_DELTA_TENSOR:
        island, base, tokens, steps = r.unpack("IQQQ")
        msg = DeltaTensor(epoch, island, base, tokens, steps, r.f32s())
    elif msg_type == MSG_ELASTIC_BASE:
        (version,) = r.unpack("Q")
        msg = ElasticBase(epoch, version, r.f32s())
    else:
        raise ElasticProtocolError(f"message type {msg_type} is not decoded by the island")
    if r.pos != len(r.data):
        raise ElasticProtocolError(f"trailing bytes in elastic frame type {msg_type}")
    return msg


DEBUG_DELAY_ENV = "YETO_RL_ELASTIC_DEBUG_DELAY"


def parse_elastic_debug_delay(spec: str | None) -> dict[int, float]:
    """``"ISLAND:SECONDS[,ISLAND:SECONDS]"`` (bare ``SECONDS`` = every island)."""
    out: dict[int, float] = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        island, sep, seconds = item.rpartition(":")
        key = int(island) if sep else -1
        value = float(seconds)
        if value < 0 or (sep and key < 0):
            raise ValueError(f"--rl-elastic-debug-delay-s: bad item {item!r}")
        out[key] = value
    return out


DEBUG_PAUSE_ENV = "YETO_RL_ELASTIC_DEBUG_PAUSE"


def parse_elastic_debug_pause(spec: str | None) -> dict[int, tuple[int, float]]:
    """``"ISLAND:AFTER_V:SECONDS[,...]"``: once the island applied an ELASTIC_BASE with
    outer version >= AFTER_V, its link to the syncer goes silent for SECONDS (no
    heartbeats, sends held, received frames held) while the process keeps training."""
    out: dict[int, tuple[int, float]] = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        try:
            island, after_v, seconds = int(parts[0]), int(parts[1]), float(parts[2])
            if len(parts) != 3:
                raise ValueError
        except (ValueError, IndexError):
            raise ValueError(f"--rl-elastic-debug-pause: bad item {item!r} (want ISLAND:AFTER_V:SECONDS)") from None
        if island < 0 or after_v < 0 or seconds <= 0:
            raise ValueError(f"--rl-elastic-debug-pause: bad item {item!r}")
        out[island] = (after_v, seconds)
    return out


def hmac_key_from_env(environ: dict | None = None) -> bytes:
    key = (os.environ if environ is None else environ).get(HMAC_KEY_ENV, "")
    if not key:
        raise ElasticProtocolError(f"elastic island needs {HMAC_KEY_ENV}")
    return key.encode("utf-8")


# ----------------------------------------------------------------------------- client
@dataclass
class ElasticClientConfig:
    syncer_addr: tuple[str, int]
    island_id: int
    capacity: float = 1.0
    incarnation: int = 0
    lease_s: float = 30.0
    syncer_epoch: int = 0
    join_timeout_s: float = 60.0
    # 0.22: consecutive failed automatic re-JOINs before the island gives up.
    max_rejoin_failures: int = 5
    # 0.22: heartbeats on their own TCP connection (a multi-MB DELTA_TENSOR /
    # ELASTIC_BASE on the main connection must not delay lease renewal).
    heartbeat_connection: bool = True
    # decoupling 6.2a: BackendIdentity.sha256() hex sent in every JOIN; None = not
    # declared (32 zero bytes). The syncer refuses a JOIN whose identity differs
    # from the one the session's first JOIN pinned.
    backend_identity_sha256: str | None = None

    def backend_identity_bytes(self) -> bytes:
        value = self.backend_identity_sha256
        if value is None:
            return bytes(32)
        if not isinstance(value, str) or len(value) != 64:
            raise ElasticProtocolError("backend_identity_sha256 must be a 64-char sha256 hex digest")
        return bytes.fromhex(value)


REJOIN_MARKER = "rejoin required"
# learner exit code when the island gave up re-joining (not the launcher's 4/6).
ELASTIC_REJOIN_FAILED_EXIT = 7


class ElasticRejoinError(ElasticProtocolError):
    """``max_rejoin_failures`` consecutive automatic re-JOINs failed."""


class ElasticIslandClient:
    """JOIN → JOIN_ACK, LEASE_HEARTBEAT every lease_s/3, DELTA_READY, LEAVE."""

    def __init__(self, config: ElasticClientConfig, key: bytes, *,
                 connect: Callable[[tuple[str, int]], socket.socket] | None = None,
                 on_event: Callable[[dict], None] | None = None) -> None:
        self.config = config
        self.key = key
        self._connect = connect or (lambda addr: socket.create_connection(addr, timeout=20.0))
        self._on_event = on_event or (lambda ev: None)
        self.sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._ack = threading.Event()
        self._left = threading.Event()
        self._base_cv = threading.Condition()
        self.ack: JoinAck | None = None
        self.errors: list[str] = []
        self.latest_base: ElasticBase | None = None
        self._verdict_cv = threading.Condition()
        self._verdicts: list[SampleVerdict] = []
        self._index_lock = threading.Lock()  # one SAMPLE_INDEX in flight: replies are in order
        self.inner_step = 0
        self.round_wall_s = 0.0
        self._threads: list[threading.Thread] = []
        self.hb_sock: socket.socket | None = None
        self._hb_lock = threading.Lock()
        self._rejoin = threading.Event()   # set by a "rejoin required" MSG_ERROR
        self._rejoin_lock = threading.Lock()
        self.rejoins = 0                   # successful automatic re-JOINs
        self.rejoin_failures = 0           # consecutive failed ones
        self.fatal: ElasticRejoinError | None = None
        self.finished: Finished | None = None  # 0.26
        self.disconnected = False
        # 0.26: outer version at which the run ends (the syncer's total_steps); a
        # lost connection once a base at/after it arrived counts as finished.
        self.final_outer_version: int | None = None
        self._base_seq = 0           # ELASTIC_BASE frames received
        self._rejoin_base_seq = 0    # _base_seq when the last re-JOIN was sent
        self._pause_until = 0.0      # debug link pause (monotonic); 0 = never paused

    @property
    def heartbeat_period_s(self) -> float:
        return self.config.lease_s / 3.0

    def pause_link(self, seconds: float) -> None:
        """Test switch (--rl-elastic-debug-pause): simulate a network outage of
        ``seconds``. Heartbeats are skipped, sends wait until the outage ends and
        received frames are held back; nothing else in the process stops."""
        t0 = time.time()
        self._pause_until = time.monotonic() + float(seconds)
        self._on_event({"event": "elastic_debug_pause", "pause_s": float(seconds), "start_unix": t0})

        def _end() -> None:
            if self._stop.wait(float(seconds)):
                return
            self._on_event({"event": "elastic_debug_pause_end", "pause_s": float(seconds),
                            "start_unix": t0, "end_unix": time.time()})
        th = threading.Thread(target=_end, name="elastic-debug-pause", daemon=True)
        th.start()

    def link_paused(self) -> bool:
        return self._pause_until > 0.0 and time.monotonic() < self._pause_until

    def _link_wait(self) -> None:
        while self._pause_until > 0.0 and not self._stop.is_set():
            left = self._pause_until - time.monotonic()
            if left <= 0:
                return
            self._stop.wait(min(left, 0.5))

    def _send(self, msg: Any) -> None:
        self._link_wait()
        t, payload = encode(msg, self.key)
        with self._send_lock:
            if self.sock is None:
                raise ElasticProtocolError("elastic client is not connected")
            try:
                write_frame(self.sock, t, payload)
            except (BrokenPipeError, ConnectionResetError, ConnectionRefusedError):
                if self.finished is not None or self.known_final():
                    raise ElasticFinished(
                        self.finished.final_outer_version if self.finished else self.latest_base.outer_version,
                        "connection closed after the final version") from None
                raise
        self._on_event({"event": "rl_elastic_sent", "type": t, **_fields(msg)})

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                t, payload = read_frame(self.sock, _frame_limit)
            except (OSError, ConnectionError, ValueError) as exc:
                if not self._stop.is_set():
                    self.errors.append(f"connection: {exc}")
                    self.disconnected = True
                    self._ack.set()
                    with self._base_cv:
                        self._base_cv.notify_all()
                return
            self._link_wait()
            if t == MSG_ERROR:
                self._on_error(payload.decode("utf-8", "replace"))
                self._ack.set()  # a refused JOIN must not wait for the timeout
                continue
            try:
                msg = decode(self.key, t, payload)
            except ElasticProtocolError as exc:
                self.errors.append(str(exc))
                continue
            if isinstance(msg, JoinAck):
                self.ack = msg
                self._ack.set()
            elif isinstance(msg, Finished):
                with self._base_cv:
                    self.finished = msg
                    self._base_cv.notify_all()
            elif isinstance(msg, SampleVerdict):
                with self._verdict_cv:
                    self._verdicts.append(msg)
                    self._verdict_cv.notify_all()
            elif isinstance(msg, ElasticBase):
                with self._base_cv:
                    self._base_seq += 1
                    if self.latest_base is None or msg.outer_version >= self.latest_base.outer_version:
                        self.latest_base = msg
                    self._base_cv.notify_all()
            self._on_event({"event": "rl_elastic_received", "type": t, **_fields(msg)})

    def _on_error(self, text: str) -> None:
        self.errors.append(text)
        self._on_event({"event": "rl_elastic_error", "error": text})
        if REJOIN_MARKER in text and self.ack is not None and not self._left.is_set():
            self._rejoin.set()

    def _hb_reader(self, sock: socket.socket) -> None:
        """The heartbeat connection only ever gets MSG_ERROR replies (not a member)."""
        while not self._stop.is_set():
            try:
                t, payload = read_frame(sock, _frame_limit)
            except (OSError, ConnectionError, ValueError):
                return
            self._link_wait()
            if t == MSG_ERROR:
                self._on_error(payload.decode("utf-8", "replace"))

    def _open_heartbeat_connection(self) -> None:
        if not self.config.heartbeat_connection:
            return
        try:
            sock = self._connect(self.config.syncer_addr)
            sock.settimeout(None)
        except OSError as exc:  # fall back to the main connection
            self.errors.append(f"heartbeat connection: {exc}")
            return
        if sock is self.sock:  # test doubles with a single socket
            return
        self.hb_sock = sock
        t = threading.Thread(target=self._hb_reader, args=(sock,), name="elastic-hb-reader", daemon=True)
        t.start()
        self._threads.append(t)

    def rejoin(self) -> bool:
        """0.22: re-JOIN with a new incarnation after a lease expiry; the syncer then
        answers JOIN_ACK (catch_up) + the current ELASTIC_BASE. False on failure;
        after ``max_rejoin_failures`` consecutive failures ``fatal`` is set."""
        with self._rejoin_lock:
            if not self._rejoin.is_set() or self._left.is_set():
                return True
            c = self.config
            c.incarnation += 1
            self._ack.clear()
            old = self.ack
            self.ack = None
            n_err = len(self.errors)
            with self._base_cv:
                self._rejoin_base_seq = self._base_seq
            try:
                self._send(Join(c.syncer_epoch, c.island_id, c.incarnation, float(c.capacity),
                                c.backend_identity_bytes()))
                ok = self._ack.wait(c.join_timeout_s) and self.ack is not None
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"rejoin: {exc}")
                ok = False
            if ok:
                self._rejoin.clear()
                self.rejoins += 1
                self.rejoin_failures = 0
                self._on_event({"event": "elastic_rejoin", "incarnation": c.incarnation,
                                "base_version": self.ack.base_version, "catch_up": self.ack.catch_up,
                                "membership_epoch": self.ack.membership_epoch})
                return True
            self.ack = self.ack or old
            self.rejoin_failures += 1
            self._on_event({"event": "elastic_rejoin_failed", "incarnation": c.incarnation,
                            "failures": self.rejoin_failures, "errors": self.errors[n_err:][-3:]})
            if self.rejoin_failures >= c.max_rejoin_failures:
                self.fatal = ElasticRejoinError(
                    f"{self.rejoin_failures} consecutive re-JOINs failed: " + "; ".join(self.errors[-3:]))
            return False

    def check(self) -> None:
        """Raise the client's fatal error (rejoin exhausted), else re-JOIN if needed."""
        if self.fatal is not None:
            raise self.fatal
        if self._rejoin.is_set():
            self.rejoin()
            if self.fatal is not None:
                raise self.fatal

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.heartbeat_period_s):
            if self._left.is_set():
                return
            if self.link_paused():
                continue  # debug link pause: no lease renewal
            try:
                if self._rejoin.is_set() and self.fatal is None:
                    self.rejoin()  # also while the main thread trains
                elif self.fatal is None:
                    self.heartbeat()
            except Exception as exc:  # reported, the reader decides liveness
                self.errors.append(f"heartbeat: {exc}")

    def join(self) -> JoinAck:
        self.sock = self._connect(self.config.syncer_addr)
        self.sock.settimeout(None)
        reader = threading.Thread(target=self._reader, name="elastic-reader", daemon=True)
        reader.start()
        self._threads.append(reader)
        c = self.config
        self._send(Join(c.syncer_epoch, c.island_id, c.incarnation, float(c.capacity),
                                c.backend_identity_bytes()))
        if not self._ack.wait(c.join_timeout_s) or self.ack is None:
            raise ElasticProtocolError("JOIN refused or timed out: " + "; ".join(self.errors))
        self._open_heartbeat_connection()
        hb = threading.Thread(target=self._heartbeat, name="elastic-heartbeat", daemon=True)
        hb.start()
        self._threads.append(hb)
        return self.ack

    def heartbeat(self) -> None:
        if self.ack is None:
            raise ElasticProtocolError("heartbeat before JOIN_ACK")
        msg = LeaseHeartbeat(self.config.syncer_epoch, self.config.island_id,
                             self.ack.membership_epoch, int(self.inner_step), float(self.round_wall_s))
        if self.hb_sock is None:
            self._send(msg)
            return
        t, payload = encode(msg, self.key)
        with self._hb_lock:
            write_frame(self.hb_sock, t, payload)
        self._on_event({"event": "rl_elastic_sent", "type": t, **_fields(msg)})

    def delta_ready(self, *, base_version: int, c_tokens: int, c_steps: int) -> None:
        self._send(DeltaReady(self.config.syncer_epoch, self.config.island_id,
                              int(base_version), int(c_tokens), int(c_steps)))

    def elastic_init(self, params) -> None:
        self._send(ElasticInit(self.config.syncer_epoch, self.config.island_id, tuple(params)))

    def delta_tensor(self, *, base_version: int, c_tokens: int, c_steps: int, update) -> None:
        self._send(DeltaTensor(self.config.syncer_epoch, self.config.island_id, int(base_version),
                               int(c_tokens), int(c_steps), tuple(update)))

    def sample_index(self, entry: Any, *, island_ids: Mapping[str, int] | None = None,
                     timeout_s: float = 30.0) -> SampleVerdict:
        """Send one ``SampleIndexEntry`` (or a ready SampleIndex) and wait for its SAMPLE_VERDICT."""
        msg = entry if isinstance(entry, SampleIndex) else sample_index_from_entry(
            entry, syncer_epoch=self.config.syncer_epoch, island_ids=island_ids)
        with self._index_lock:
            with self._verdict_cv:
                self._verdicts.clear()
            n_errors = len(self.errors)
            self._send(msg)
            deadline = time.monotonic() + timeout_s
            with self._verdict_cv:
                while not self._verdicts:
                    left = deadline - time.monotonic()
                    if left <= 0 or self._stop.is_set() or len(self.errors) > n_errors:
                        raise ElasticProtocolError("no SAMPLE_VERDICT: " + "; ".join(self.errors))
                    self._verdict_cv.wait(min(left, 0.1))
                return self._verdicts.pop(0)

    def wait_base(self, *, newer_than: int | None, timeout_s: float) -> ElasticBase | None:
        """Block until an ELASTIC_BASE newer than ``newer_than`` (any if None) arrived.

        0.22: after an automatic re-JOIN during the wait (our delta may have been
        refused while we were not a member) the first base received after the
        re-JOIN is returned even when it is not newer (catch-up)."""
        deadline = time.monotonic() + timeout_s
        rejoins0 = self.rejoins
        with self._base_cv:
            while True:
                if self.fatal is not None:
                    raise self.fatal
                b = self.latest_base
                if b is not None and (newer_than is None or b.outer_version > newer_than):
                    return b
                self.raise_if_finished()
                if (b is not None and self.rejoins > rejoins0
                        and self._base_seq > self._rejoin_base_seq):
                    return b
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._base_cv.wait(min(left, 0.1))

    def known_final(self) -> bool:
        b = self.latest_base
        return (self.final_outer_version is not None and b is not None
                and b.outer_version >= self.final_outer_version)

    def raise_if_finished(self) -> None:
        """0.26: FINISHED received, or the syncer is gone after the final version."""
        if self.finished is not None:
            raise ElasticFinished(self.finished.final_outer_version, "FINISHED")
        if self.disconnected and self.known_final():
            raise ElasticFinished(self.latest_base.outer_version, "connection closed after the final version")

    def leave(self, reason: int = LEAVE_REASON_REQUESTED) -> None:
        self._left.set()  # no heartbeat after LEAVE
        if self.ack is not None and self.sock is not None:
            self._send(Leave(self.config.syncer_epoch, self.config.island_id, reason))

    def close(self) -> None:
        self._stop.set()
        if self.hb_sock is not None:
            try:
                self.hb_sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.hb_sock.close()
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
        for t in self._threads:
            t.join(timeout=2.0)


def _fields(msg: Any) -> dict:
    out = {}
    for k, v in vars(msg).items():
        if isinstance(v, bytes):
            v = v.hex()
        elif isinstance(v, tuple) and len(v) > 8:
            v = f"<{len(v)} floats>"
        out[k] = v
    return out
