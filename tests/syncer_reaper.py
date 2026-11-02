"""Pytest plugin: no ``yeto-syncer`` process outlives the test that started it.

Many tests start a real syncer (directly, or through ``bash -c``) and stop it
in their own cleanup. A failing assertion, a timeout or a ``bash`` wrapper
that is killed without its child used to leave the syncer running, reparented
to init. This plugin makes that impossible for syncers started from the test
process:

* every ``subprocess.Popen`` whose command line names ``yeto-syncer`` is
  started in its own process group (unless the caller chose its own session,
  group or ``preexec_fn``) and recorded;
* a syncer started by a function-scoped fixture or by the test itself is
  killed (the whole group, so a ``bash -c`` wrapper's child too) and waited
  for when that test's teardown is done, pass or fail;
* one started by a module- or session-scoped fixture is killed when the last
  test of that module, or the session, is torn down;
* at session end, orphaned ``yeto-syncer`` processes of this checkout
  (reparented to init or a subreaper) that did not exist when the session started are killed too
  and named in a warning: a syncer started by a grandchild process is not
  seen by the ``Popen`` hook.
"""

from __future__ import annotations

import os
import signal
import subprocess
import warnings
from pathlib import Path

import pytest

SYNCER_NAME = "yeto-syncer"
REPO_ROOT = Path(__file__).resolve().parents[1]

_ORIGINAL_POPEN_INIT = subprocess.Popen.__init__
# (process, owner): owner is "item", ("module", <module path>) or "session"
_TRACKED: list[tuple[subprocess.Popen, object]] = []
_OWNER: list[object] = ["item"]
_PREEXISTING: set[int] = set()


def _command_text(args) -> str:
    if isinstance(args, (str, bytes, os.PathLike)):
        return os.fsdecode(args)
    try:
        return " ".join(os.fsdecode(a) for a in args)
    except TypeError:
        return ""


def names_syncer(args) -> bool:
    """True when a Popen command line runs ``yeto-syncer``."""
    return SYNCER_NAME in _command_text(args)


def _tracked_popen_init(self, args, *rest, **kwargs):
    syncer = names_syncer(args)
    if syncer and not rest and not any(kwargs.get(k) for k in
                                       ("start_new_session", "process_group", "preexec_fn")):
        kwargs["start_new_session"] = True
    _ORIGINAL_POPEN_INIT(self, args, *rest, **kwargs)
    if syncer:
        try:  # its own group: the whole group can be killed, even after the leader exited
            self._yeto_own_group = os.getpgid(self.pid) == self.pid
        except OSError:
            self._yeto_own_group = False
        _TRACKED.append((self, _OWNER[-1]))


def _kill(process: subprocess.Popen) -> None:
    """Kill the process and its group (its own group only), then reap it."""
    if getattr(process, "_yeto_own_group", False):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _group_alive(process: subprocess.Popen) -> bool:
    if process.poll() is None:
        return True
    if not getattr(process, "_yeto_own_group", False):
        return False
    try:  # the leader exited; a child of a bash wrapper may still hold the group
        os.killpg(process.pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True


def reap(owner_matches) -> int:
    """Kill every tracked syncer whose owner matches; the number still alive."""
    alive = 0
    for entry in list(_TRACKED):
        process, owner = entry
        if not owner_matches(owner):
            continue
        _TRACKED.remove(entry)
        if _group_alive(process):
            alive += 1
            _kill(process)
    return alive


def _ps() -> list[tuple[int, int, str]]:
    try:
        # -ww: procps cuts lines to 80 columns when stdout is not a terminal, which
        # drops the binary path of a syncer under a long checkout or tmp path
        out = subprocess.run(["ps", "-ww", "-eo", "pid=,ppid=,args="], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return rows


def checkout_syncers(root: Path = REPO_ROOT) -> list[tuple[int, int, str]]:
    """``yeto-syncer`` processes whose binary is under ``root``: (pid, ppid, args)."""
    prefix = str(root)
    return [(pid, ppid, args) for pid, ppid, args in _ps()
            if args.split(" ", 1)[0].startswith(prefix)
            and os.path.basename(args.split(" ", 1)[0]) == SYNCER_NAME]


def orphan_parents() -> set[int]:
    """Parents an orphan of this process tree is reparented to: init (pid 1) or a
    child subreaper, which is always an ancestor of the process that left it
    (CI runners and container inits are subreapers)."""
    parent_of = {pid: ppid for pid, ppid, _ in _ps()}
    parents, pid = {1}, parent_of.get(os.getpid())
    while pid and pid not in parents:
        parents.add(pid)
        pid = parent_of.get(pid)
    return parents


def reap_orphans(root: Path = REPO_ROOT, *, keep: set[int] = frozenset()) -> list[int]:
    """Kill orphaned syncers of ``root`` (parent: init or a subreaper above this
    process) not in ``keep``; their pids."""
    killed = []
    parents = orphan_parents()
    for pid, ppid, _ in checkout_syncers(root):
        if ppid in parents and pid not in keep:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except OSError:
                pass
    return killed


def pytest_sessionstart(session):
    subprocess.Popen.__init__ = _tracked_popen_init
    _PREEXISTING.update(pid for pid, _, _ in checkout_syncers())


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    scope = fixturedef.scope
    if scope == "function":
        owner = "item"
    elif scope in ("module", "class"):
        owner = ("module", str(getattr(request.node, "path", "")))
    else:
        owner = "session"
    _OWNER.append(owner)
    try:
        yield
    finally:
        _OWNER.pop()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    reap(lambda owner: owner == "item")
    if nextitem is None or getattr(nextitem, "path", None) != getattr(item, "path", None):
        module = ("module", str(getattr(item, "path", "")))
        reap(lambda owner: owner == module)


def pytest_sessionfinish(session, exitstatus):
    reap(lambda owner: True)
    subprocess.Popen.__init__ = _ORIGINAL_POPEN_INIT
    if hasattr(session.config, "workerinput"):  # an xdist worker: the controller sweeps
        return
    killed = reap_orphans(keep=_PREEXISTING)
    if killed:
        warnings.warn(f"killed {len(killed)} orphaned {SYNCER_NAME} process(es) left by this "
                      f"session: {killed}", stacklevel=1)
