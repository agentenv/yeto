"""The ``ssh`` transport of the launcher's SSH nodes (trainer islands, sampler pools).

Two things make a flaky route to a node's public address survivable:

* connection reuse: every ``ssh`` the launcher runs to a node rides one master
  connection (``ControlMaster=auto``, ``ControlPersist``), so the many short
  status, kill and preflight calls of a run do not each open a new TCP + key
  exchange. The control sockets live under the launcher's own state directory
  (:func:`control_dir`), never in the user's SSH configuration; a host whose
  SSH configuration already sets ``ControlMaster`` / ``ControlPath`` keeps its
  own settings (:func:`mux_options` adds nothing there). ``YETO_SSH_MUX=off``
  turns reuse off;
* a short retry of a connection the far end dropped before the remote command
  ran (:func:`transient_failure`: ``ssh`` exit 255 with ``Connection closed by
  <host> port <n>``, ``Connection reset by <host> port <n>`` or
  ``kex_exchange_identification`` on stderr: the messages of a connection that
  never got a session). A session that drops mid-command reports otherwise
  (``Connection to <host> closed by remote host``, ``client_loop: ...``) and is not
  retried, so a retried command never ran. A streamed input (a pipe) is sent once.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

MUX_ENV = "YETO_SSH_MUX"  # "off": no connection reuse
CONTROL_DIR_ENV = "YETO_SSH_CONTROL_DIR"  # where the control sockets go
DEFAULT_CONTROL_DIR = "~/.yeto/ssh"
CONTROL_PERSIST = "10m"
# A unix socket path is at most 104 bytes on macOS (108 on Linux); ssh appends a
# 17-character temporary suffix to ControlPath while it sets the master up, and
# %C expands to 40 hex characters.
_SOCKET_PATH_MAX = 100
_CONTROL_NAME_LEN = 40 + 17 + 1

# ssh exit 255 with one of these on stderr: the far end dropped the connection before
# a session existed (key exchange, authentication); a new connection usually gets through
TRANSIENT_PATTERNS = (re.compile(r"Connection closed by \S+ port \d+"),
                      re.compile(r"Connection reset by \S+ port \d+"),
                      re.compile(r"kex_exchange_identification"))
RETRY_DELAYS_S = (1.0, 3.0)  # sleeps before the 2nd and 3rd attempt
SSH_FAILURE_RC = 255

_user_controls: dict[tuple[str, ...], bool] = {}


def control_dir() -> Path:
    """The directory of the control sockets (created owner-only). A configured or
    home directory too long for a socket path falls back to the system temp dir."""
    configured = os.environ.get(CONTROL_DIR_ENV) or DEFAULT_CONTROL_DIR
    path = Path(configured).expanduser()
    if len(str(path)) + _CONTROL_NAME_LEN > _SOCKET_PATH_MAX:
        path = Path(tempfile.gettempdir()) / f"yeto-ssh-{os.getuid()}"
        if len(str(path)) + _CONTROL_NAME_LEN > _SOCKET_PATH_MAX:
            path = Path("/tmp") / f"yeto-ssh-{os.getuid()}"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _probe_user_controls(target: Sequence[str]) -> bool:
    """Whether the user's SSH configuration sets connection reuse for ``target``
    (``ssh -G``: the effective configuration, nothing is dialled)."""
    try:
        out = subprocess.run(["ssh", "-G", *target], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    for line in out.splitlines():
        key, _, value = line.strip().partition(" ")
        key, value = key.lower(), value.strip().lower()
        if key == "controlmaster" and value not in ("", "false", "no"):
            return True
        if key == "controlpath" and value not in ("", "none"):
            return True
    return False


def user_controls(target: Sequence[str]) -> bool:
    """:func:`_probe_user_controls`, once per target."""
    key = tuple(target)
    if key not in _user_controls:
        _user_controls[key] = _probe_user_controls(key)
    return _user_controls[key]


def mux_options(target: Sequence[str]) -> list[str]:
    """``-o`` options that make ``ssh ... <target>`` reuse one master connection
    (module doc); ``target``: the options and host the call dials (``ProxyJump`` and
    the host), so the user's configuration is read for exactly that route."""
    if any(str(o).startswith("ProxyCommand=") for o in target):
        # ControlPath's %C does not hash the ProxyCommand: two relay nodes with one inner
        # address (two sites' 192.0.2.7) would share a master, and the second would skip its
        # host key check. A relayed route never shares a connection, the user's config included.
        return ["-o", "ControlMaster=no", "-o", "ControlPath=none"]
    if os.environ.get(MUX_ENV, "").strip().lower() in ("off", "0", "false", "no"):
        return []
    if user_controls(target):
        return []
    return ["-o", "ControlMaster=auto", "-o", f"ControlPath={control_dir()}/%C",
            "-o", f"ControlPersist={CONTROL_PERSIST}"]


def transient_failure(returncode: int, stderr: Any) -> bool:
    """An ``ssh`` exit that a new connection is likely to get past (module doc)."""
    if returncode != SSH_FAILURE_RC:
        return False
    text = stderr.decode(errors="replace") if isinstance(stderr, bytes) else str(stderr or "")
    return any(pattern.search(text) for pattern in TRANSIENT_PATTERNS)


def run_with_retry(run: Callable[..., Any], argv: Sequence[str], *,
                   delays: Sequence[float] = RETRY_DELAYS_S,
                   sleep: Callable[[float], None] = time.sleep,
                   log: Callable[[str], None] | None = None, label: str = "", **kwargs: Any) -> Any:
    """``run(argv, **kwargs)`` (a ``subprocess.run``), again after each of ``delays``
    while it ends in a :func:`transient_failure`; the last result. The input must be
    re-sendable (``input=``, not a stream). ``label``: the node, for the retry line."""
    done = run(list(argv), **kwargs)
    for delay in delays:
        if not transient_failure(getattr(done, "returncode", 0), getattr(done, "stderr", "")):
            break
        if log is not None:
            err = getattr(done, "stderr", "")
            err = err.decode(errors="replace") if isinstance(err, bytes) else str(err or "")
            log(f"[launcher] ssh{' to ' + label if label else ''}: {err.strip()[-200:]}; "
                f"retrying in {delay:g} s")
        sleep(delay)
        done = run(list(argv), **kwargs)
    return done
