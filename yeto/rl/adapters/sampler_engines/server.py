"""Shared part of the HTTP-server engine adapters: one engine server process per
sampler instance, started on a release directory, checked for readiness with
a timeout, and stopped with a known exit-code rule.

Guards that apply to every server engine (A-INF §3):

* start timeout (item 15): a server that is not ready in ``start_timeout_s``
  is killed and reported with the adapter's ``hang_hint`` (910B: HCCL hang
  without error when the CANN libprofapi stub is missing);
* host memory (item 17): before a start, the host must have at least
  ``min_host_free_gib`` available (910B W4A8 stages int8 weights in host
  memory, peak about 610 GiB per the report);
* exit codes (item 13): an exit after :meth:`ServerEngine.stop` asked the
  server to stop is normal when the code is in ``exit_codes_after_stop`` (PPU
  SGLang TP8 exits 134 from ``pcclCommAbort``); any exit while serving is a
  failure;
* request timeouts on every HTTP call.

The process and HTTP layers are injected (``spawn``, ``http``) so CPU tests
drive the adapters without a server. Defaults: ``subprocess.Popen`` and httpx.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from yeto.rl.engine.sampler_checks import release_quant_format
from yeto.rl.engine.sampler_engine import (
    RELOAD_IN_PLACE,
    RELOAD_RESTART,
    EngineIdentity,
    LoadReport,
    SamplerEngineError,
)

GIB = 1024 ** 3


class Process(Protocol):
    def poll(self) -> int | None: ...

    def send_signal(self, sig: int) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


Spawn = Callable[[Sequence[str], Mapping[str, str]], Process]
# (method, url, json body or None, timeout seconds) -> (HTTP status, decoded JSON or None)
Http = Callable[[str, str, Any, float], tuple[int, Any]]


def _ignore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def popen_spawn(argv: Sequence[str], env: Mapping[str, str]) -> Process:
    """Start the server with SIGINT ignored (inherited by its workers). A cloud
    reclaim (Modal ``container stop``) sends SIGINT to every process of the
    container: the island turns it into the reclaim marker and leaves after
    the round in progress, so the server must not die under that round
    (T-E2V eval-e2-verl-e2v-a). ``stop()`` still ends it with SIGTERM."""
    return subprocess.Popen(list(argv), env={**os.environ, **env},  # noqa: S603
                            preexec_fn=_ignore_sigint)  # noqa: PLW1509 - only resets one signal


def httpx_call(method: str, url: str, body: Any, timeout_s: float) -> tuple[int, Any]:
    import httpx  # noqa: PLC0415 - optional on hosts that only run tests

    try:
        response = httpx.request(method, url, json=body, timeout=timeout_s)
    except httpx.HTTPError as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"}
    try:
        return response.status_code, response.json() if response.content else None
    except ValueError:
        return response.status_code, {"error": response.text[:300]}


def host_available_gib(meminfo: str = "/proc/meminfo") -> float | None:
    try:
        with open(meminfo, encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / GIB
    except (OSError, ValueError, IndexError):
        return None
    return None


@dataclass
class ServerConfig:
    """Settings every server adapter takes (run config / inventory, not code)."""

    base_dir: Path  # the weights the engine starts on before any release
    card_model: str
    device_family: str
    engine_version: str
    precision: str  # bf16 / w4a8 / mxfp4 ...
    quant_scheme: str  # the release scheme served (derived_from.scheme; bf16 for trained weights)
    topology: str
    max_model_len: int
    host: str = "127.0.0.1"
    port: int = 30000
    python: str = "python"
    extra_args: tuple[str, ...] = ()
    extra_env: Mapping[str, str] = field(default_factory=dict)
    start_timeout_s: float = 3600.0
    request_timeout_s: float = 3600.0
    stop_grace_s: float = 120.0
    min_host_free_gib: float | None = None
    expected_format: str | None = None  # quantization_config.format the release must carry
    in_place_reload: bool = False  # only after an on-card test passed (N4.2)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class ServerEngine:
    """Base of the server adapters. Subclasses set ``ENGINE``, ``HANG_HINT``,
    ``EXIT_CODES_AFTER_STOP`` and implement ``argv``, ``env``, ``generate``,
    ``score``; optionally ``preflight`` and ``reload_in_place``."""

    ENGINE = "server"
    HANG_HINT = ""
    EXIT_CODES_AFTER_STOP: frozenset[int] = frozenset()
    REFUSED_FORMATS: frozenset[str] = frozenset()

    def __init__(self, config: ServerConfig, *, spawn: Spawn = popen_spawn, http: Http = httpx_call,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 host_free_gib: Callable[[], float | None] = host_available_gib) -> None:
        self.config = config
        self._spawn = spawn
        self._http = http
        self._clock = clock
        self._sleep = sleep
        self._host_free_gib = host_free_gib
        self._process: Process | None = None
        self._stopping = False
        self._version: int | None = None
        self.last_exit: dict[str, Any] | None = None

    # ------------------------------------------------------------------ identity
    @property
    def identity(self) -> EngineIdentity:
        c = self.config
        return EngineIdentity(engine=self.ENGINE, engine_version=c.engine_version,
                              device_family=c.device_family, card_model=c.card_model,
                              precision=c.precision, quant_scheme=c.quant_scheme, topology=c.topology)

    @property
    def max_model_len(self) -> int:
        return int(self.config.max_model_len)

    # ------------------------------------------------------------------ subclass hooks
    def argv(self, model_dir: Path) -> list[str]:
        raise NotImplementedError

    def env(self) -> dict[str, str]:
        return dict(self.config.extra_env)

    def preflight(self, model_dir: Path) -> None:
        """Refuse a release or configuration the engine would serve wrongly."""
        self.check_format(model_dir)

    def reload_in_place(self, model_dir: Path, version: int) -> None:
        raise SamplerEngineError(f"{self.ENGINE}: in-place reload is not implemented")

    def after_ready(self) -> dict[str, Any]:
        """Work after the server is ready and before it is checked (warm-up)."""
        return {}

    # ------------------------------------------------------------------ guards
    def check_format(self, model_dir: Path) -> None:
        method, fmt = release_quant_format(model_dir)
        want = self.config.expected_format
        if fmt in self.REFUSED_FORMATS:
            raise SamplerEngineError(f"{model_dir}: quantization_config.format {fmt!r} is refused "
                                     f"by {self.ENGINE} (known to load wrong weights without error)")
        if want is not None and fmt != want:
            raise SamplerEngineError(f"{model_dir}: quantization_config.format is {fmt!r} "
                                     f"(quant_method {method!r}); this engine is configured for {want!r}")
        if want is None and self.config.quant_scheme == "bf16" and method is not None:
            raise SamplerEngineError(f"{model_dir}: a bf16 engine got a quantized checkpoint "
                                     f"({method}/{fmt})")

    def _check_host_memory(self) -> None:
        need = self.config.min_host_free_gib
        if need is None:
            return
        free = self._host_free_gib()
        if free is None or free < need:
            raise SamplerEngineError(f"{self.ENGINE}: host has {free} GiB available, the load needs "
                                     f">= {need} GiB (weight staging in host memory)")

    # ------------------------------------------------------------------ http
    def call(self, method: str, path: str, body: Any = None, timeout_s: float | None = None) -> Any:
        self._check_alive()
        status, data = self._http(method, f"{self.config.url}{path}", body,
                                  float(timeout_s or self.config.request_timeout_s))
        if status != 200:
            raise SamplerEngineError(f"{self.ENGINE} {method} {path}: HTTP {status}: {str(data)[:300]}")
        return data

    def _check_alive(self) -> None:
        if self._process is None:
            raise SamplerEngineError(f"{self.ENGINE} is not running")
        code = self._process.poll()
        if code is not None and not self._stopping:
            self._process = None
            self._version = None
            self.last_exit = {"code": code, "during": "service", "normal": False}
            raise SamplerEngineError(f"{self.ENGINE} exited with code {code} while serving")

    # ------------------------------------------------------------------ lifecycle
    def start(self, release_dir: Path | None, version: int) -> LoadReport:
        model_dir = Path(release_dir) if release_dir is not None else Path(self.config.base_dir)
        self.preflight(model_dir)
        began = self._clock()
        self.stop()
        self._check_host_memory()
        self._stopping = False
        self._version = None
        self._process = self._spawn(self.argv(model_dir), self.env())
        self._wait_ready()
        detail = self.after_ready()
        self._version = int(version)
        return LoadReport(reload_mode=RELOAD_RESTART, pause_s=self._clock() - began,
                          detail={"model_dir": str(model_dir), **detail})

    def load(self, release_dir: Path, version: int) -> LoadReport:
        if not self.config.in_place_reload or self._process is None:
            return self.start(release_dir, version)
        self.preflight(Path(release_dir))
        began = self._clock()
        self.reload_in_place(Path(release_dir), int(version))
        self._version = int(version)
        return LoadReport(reload_mode=RELOAD_IN_PLACE, pause_s=self._clock() - began,
                          detail={"model_dir": str(release_dir)})

    def _wait_ready(self) -> None:
        deadline = self._clock() + float(self.config.start_timeout_s)
        while True:
            code = None if self._process is None else self._process.poll()
            if code is not None:
                self._process = None
                self.last_exit = {"code": code, "during": "start", "normal": False}
                raise SamplerEngineError(f"{self.ENGINE} exited with code {code} during start")
            status, _ = self._http("GET", f"{self.config.url}/health", None, 10.0)
            if status == 200:
                return
            if self._clock() >= deadline:
                self._kill()
                raise SamplerEngineError(f"{self.ENGINE} not ready after {self.config.start_timeout_s:g} s; "
                                         f"killed. {self.HANG_HINT}".strip())
            self._sleep(5.0)

    def _kill(self) -> None:
        if self._process is not None:
            self._stopping = True
            self._process.kill()
            self._process.wait(timeout=30)
            self._process = None
            self._version = None

    def stop(self) -> None:
        """Ask the server to stop; an exit code in ``EXIT_CODES_AFTER_STOP`` is normal."""
        process, self._process = self._process, None
        self._version = None
        if process is None:
            return
        self._stopping = True
        code = process.poll()
        if code is None:
            process.send_signal(signal.SIGTERM)
            try:
                code = process.wait(timeout=self.config.stop_grace_s)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait(timeout=30)
        known = code in self.EXIT_CODES_AFTER_STOP
        self.last_exit = {"code": code, "during": "stop",
                          "normal": code in (0, -signal.SIGTERM) or known,
                          **({"note": "known exit code after stop"} if known else {})}

    def served_version(self) -> int | None:
        return self._version if self._process is not None else None

    # The servers report neither count; the post-load check then uses the
    # configured weight memory or fails (sampler_checks.PostLoadChecks).
    def loaded_expert_matrices(self) -> int | None:
        return None

    def weight_gib_per_card(self) -> float | None:
        return None
