"""Nebius preemptible reclaim notice: host SIGTERM -> learner in the sky container.

rl-spot-cost-saving (S19 real-spot experiment, option C). Nebius docs: a
preemptible VM gets SIGTERM 60 s before it is stopped (ACPI shutdown of the VM),
then SIGKILL. The learner runs inside sky's docker container (``sky_container``,
PID 1 is an interactive bash), so the VM shutdown never reaches it as a signal:
docker only signals the container's PID 1 and kills the rest 10 s later.

Path (two halves):

* Host: a systemd unit ``yeto-reclaim-notice.service`` ordered ``After=docker.service``.
  On shutdown systemd stops it BEFORE docker; its ExecStop writes the host time
  into a notice file inside the container (``docker exec``), then waits (bounded)
  for the learner's ack file so docker is not stopped before the LEAVE.
  The launcher installs the unit over ssh on the VM host (:func:`install_host_hook`).
* Container: :class:`NoticeFilePoller` polls the notice file every second and calls
  ``on_notice`` once (``droppable.install_reclaim_listener`` -> ``handle_notice`` ->
  LEAVE -> ``spot_reclaim``), then writes the ack file.

Unverified on a real VM until the S19 stop drill: whether ``nebius compute
instance stop`` takes the same SIGTERM path, and the end-to-end latency.
"""

from __future__ import annotations

import os
import shlex
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

NOTICE_FILE = "/tmp/yeto-reclaim-notice"
ACK_FILE = "/tmp/yeto-reclaim-ack"
NOTICE_ENV = "YETO_RECLAIM_NOTICE_FILE"
ACK_ENV = "YETO_RECLAIM_ACK_FILE"
CONTAINER = "sky_container"          # sky.skylet.constants.DEFAULT_DOCKER_CONTAINER_NAME
UNIT = "yeto-reclaim-notice.service"
HOOK_BIN = "/usr/local/bin/yeto-reclaim-notice"
POLL_S = 1.0
ACK_WAIT_S = 45                      # < 60 s Nebius notice; docker still gets its 10 s stop
SOURCE = "nebius_host_sigterm"


def notice_paths(environ=None) -> tuple[str, str]:
    env = os.environ if environ is None else environ
    return env.get(NOTICE_ENV) or NOTICE_FILE, env.get(ACK_ENV) or ACK_FILE


def parse_notice(text: str, *, now: float) -> float:
    """The host time written by the hook; ``now`` when the file is empty or garbled
    (the notice still counts, only the timestamp is lost)."""
    try:
        value = float((text or "").strip().split()[0])
    except (ValueError, IndexError):
        return now
    return value if 0 < value <= now + 5 else now


def hook_script(container: str = CONTAINER, notice: str = NOTICE_FILE, ack: str = ACK_FILE,
                wait_s: int = ACK_WAIT_S) -> str:
    """The host ExecStop script. Never fails the shutdown (always exit 0)."""
    c, n, a = shlex.quote(container), shlex.quote(notice), shlex.quote(ack)
    return (
        "#!/bin/bash\n"
        "t=$(date +%s.%N)\n"
        'logger -t yeto-reclaim "stop at $t"\n'
        "D=docker; docker info >/dev/null 2>&1 || D='sudo docker'\n"
        f"$D exec {c} sh -c \"echo $t > {n}.tmp && mv {n}.tmp {n}\" "
        '|| { logger -t yeto-reclaim "notice write failed"; exit 0; }\n'
        f"for i in $(seq 1 {int(wait_s)}); do\n"
        f"  if $D exec {c} test -e {a}; then logger -t yeto-reclaim \"ack after ${{i}}s\"; exit 0; fi\n"
        "  sleep 1\n"
        "done\n"
        'logger -t yeto-reclaim "no ack"\n'
        "exit 0\n"
    )


def unit_text(wait_s: int = ACK_WAIT_S) -> str:
    return (
        "[Unit]\n"
        "Description=yeto: forward the VM stop (Nebius preemption SIGTERM) to the learner\n"
        "After=docker.service\n"
        "Wants=docker.service\n"
        "[Service]\n"
        "Type=oneshot\n"
        "RemainAfterExit=yes\n"
        "ExecStart=/bin/true\n"
        f"ExecStop={HOOK_BIN}\n"
        f"TimeoutStopSec={int(wait_s) + 10}\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def install_command(container: str = CONTAINER, notice: str = NOTICE_FILE, ack: str = ACK_FILE) -> str:
    """One host shell command (sudo) that installs and starts the unit. Idempotent."""
    script, unit = hook_script(container, notice, ack), unit_text()
    return (
        f"printf %s {shlex.quote(script)} | sudo tee {HOOK_BIN} >/dev/null && "
        f"sudo chmod 0755 {HOOK_BIN} && "
        f"printf %s {shlex.quote(unit)} | sudo tee /etc/systemd/system/{UNIT} >/dev/null && "
        "sudo systemctl daemon-reload && "
        f"sudo systemctl enable --now {UNIT} && systemctl is-active {UNIT}"
    )


def wants_host_hook(task_envs) -> bool:
    """True for a droppable island on Nebius (from the task's YETO_ISLAND_ROLE)."""
    from yeto.cloud import droppable

    doc = droppable.env_role(dict(task_envs or {}))
    return bool(doc) and doc.get("role") == droppable.DROPPABLE and doc.get("cloud") == "nebius"


def _host_runner(handle: Any) -> Any:
    """A sky SSH runner on the VM HOST (docker_user=None), not in the container."""
    from sky.backends import backend_utils
    from sky.utils import command_runner

    creds = backend_utils.ssh_credential_from_yaml(handle.cluster_yaml, None, handle.ssh_user)
    creds.pop("docker_user", None)
    ip = handle.head_ip
    return command_runner.SSHCommandRunner((ip, 22), **creds)


def install_host_hook(handle: Any, *, name: str = "", runner: Any = None,
                      log: Callable[[str], Any] = print) -> bool:
    """Install the host unit on ``handle``'s VM. Returns success; never raises
    (without the hook the island falls back to lease expiry, as before)."""
    try:
        r = runner if runner is not None else _host_runner(handle)
        rc, out, err = r.run(install_command(), require_outputs=True, stream_logs=False)
    except Exception as exc:  # noqa: BLE001
        log(f"[launcher] {name}: Nebius reclaim hook NOT installed ({type(exc).__name__}: {exc})")
        return False
    ok = rc == 0 and "active" in (out or "")
    log(f"[launcher] {name}: Nebius reclaim hook {'installed' if ok else 'NOT installed'} (rc={rc})"
        + ("" if ok else f" {str(err)[-300:]}"))
    return ok


@dataclass
class NoticeFilePoller:
    """Poll the notice file; on first sight call ``on_notice(notice)`` once, then ack."""

    island: str
    on_notice: Callable[[Any], Any]
    region: str | None = None
    role: str = "droppable"
    notice_path: str = NOTICE_FILE
    ack_path: str = ACK_FILE
    deadline_s: float | None = 60.0
    interval_s: float = POLL_S
    clock: Callable[[], float] = time.time
    fired: bool = False
    _stop: threading.Event = field(default_factory=threading.Event)

    def poll_once(self) -> Any:
        from yeto.cloud.preemption import ReclaimNotice

        if self.fired:
            return None
        try:
            with open(self.notice_path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return None
        self.fired = True
        now = self.clock()
        notice = ReclaimNotice(self.island, "nebius", self.region, "spot", self.role, SOURCE,
                               parse_notice(text, now=now), self.deadline_s)
        try:
            self.on_notice(notice)
        finally:
            try:
                with open(self.ack_path, "w", encoding="utf-8") as f:
                    f.write(f"{self.clock():.3f}\n")
            except OSError:
                pass
        return notice

    def run(self) -> None:
        while not self.fired and not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.interval_s)

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, daemon=True, name=f"nebius-reclaim-{self.island}")
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()
