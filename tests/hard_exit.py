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
"""

from __future__ import annotations

import os
import sys

_STATUS: dict[str, int] = {}


def note_exitstatus(exitstatus) -> None:
    """Record what pytest will exit with; called from ``pytest_sessionfinish``."""
    _STATUS["exitstatus"] = int(exitstatus)


def pytest_unconfigure(config):
    """Last pytest hook of the process: leave before the native teardown runs."""
    if os.environ.get("YETO_TEST_SOFT_EXIT") == "1":
        return
    if hasattr(config, "workerinput"):  # an xdist worker: the controller collects its result
        return
    status = _STATUS.get("exitstatus")
    if status is None:  # no session ran (e.g. --collect-only, a usage error)
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(status)
