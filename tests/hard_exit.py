"""Leave the test process without running native static destructors.

pyarrow's thread pool launches worker threads during interpreter shutdown (the
SkyPilot imports of ``tests/test_launch_auto.py`` pull in pandas and pyarrow,
and ``sky`` keeps touching them from its own atexit handlers). libarrow's
static destruction then tears that pool down while the ``std::thread`` objects
are still joinable, so libstdc++ calls ``terminate``::

    terminate called without an active exception
    Thread ... received signal SIGABRT

The process dies with 134 *after* pytest has already printed a green summary,
so a run whose every test passed is reported as a failed job. Caught under gdb
in 6 of 6 runs of ``--shard 3/4`` on ubuntu-latest / Python 3.11 (2026-11-16);
the pool threads are created later than every thread of the session itself.

Nothing of ours owns those threads, and the abort is in the C++ runtime, past
the point where Python can join anything. So once pytest has computed its exit
status and its output is flushed, the session leaves through :func:`os._exit`,
which skips atexit handlers and static destructors. The status pytest computed
is what the process exits with, so a failing run still fails.

``YETO_TEST_SOFT_EXIT=1`` keeps the normal exit (for anything that needs the
atexit handlers, e.g. a coverage run).

Only the process's own (outermost) pytest session leaves this way. A test that
runs pytest in process (``pytester.runpytest``, with a copy of this conftest)
starts a nested session in the same interpreter; its end must return to the
test, not end the process with the nested run's status (which would stop the
outer run silently, exit status 0, with the rest of its tests never run).
"""

from __future__ import annotations

import os
import sys

_STATUS: dict[int, int] = {}  # id(config) -> the exit status its session computed
_ROOT: list[int] = []  # id() of the first (outermost) config of this process


def note_configure(config) -> None:
    """Remember the first config of the process; called from ``pytest_configure``."""
    if not _ROOT:
        _ROOT.append(id(config))


def is_outermost(config) -> bool:
    """Whether ``config`` is the process's own session, not a nested in-process run."""
    return bool(_ROOT) and id(config) == _ROOT[0]


def note_exitstatus(session, exitstatus) -> None:
    """Record what pytest will exit with; called from ``pytest_sessionfinish``."""
    _STATUS[id(session.config)] = int(exitstatus)


def pytest_unconfigure(config):
    """Last pytest hook of the process: leave before the native teardown runs."""
    if os.environ.get("YETO_TEST_SOFT_EXIT") == "1":
        return
    if hasattr(config, "workerinput"):  # an xdist worker: the controller collects its result
        return
    if not _ROOT or id(config) != _ROOT[0]:  # a nested in-process run (pytester): back to its test
        _STATUS.pop(id(config), None)
        return
    status = _STATUS.get(id(config))
    if status is None:  # no session ran (e.g. --collect-only, a usage error)
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(status)
