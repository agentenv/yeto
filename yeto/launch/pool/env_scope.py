"""The run's environment around the provider calls of a pool driven in this process.

A pool run as its own loop (``python -m yeto.launch.nc pool``, the run's
``controller: "subprocess"``) is a child process started with the run's environment
(``run.json`` ``env`` on top of the launching shell's). A pool driven by a
:class:`~.controller.RolloutController` inside ``yeto run up`` (``controller:
"in_process"``) runs in the launching process instead, whose ``os.environ`` is the
shell's: its provider SDKs (boto3, the sky client, ...) read ``AWS_PROFILE`` and the
like from ``os.environ`` and accept no mapping, so the variables of ``run.json`` never
reached them.

:class:`EnvScopedOps` wraps a pool's ops: every call through it runs inside
:func:`scoped_environ`, which sets the overlay in ``os.environ`` for the duration of
the call and restores the previous values after it. ``os.environ`` is one per
process, so one overlay at a time is active in a process: calls of that overlay that
overlap (the pool's tick thread, a background sampler start, a drain from the main
thread, other pools of the same run) share one application, put in place by the
first and restored by the last, counted under a lock that is never held across a
call (a sampler start of many minutes blocks nothing). A scope of any other overlay
waits a bounded time (``wait_s``; 0 by default) for the active one to leave, then
raises :class:`EnvOverlayConflict`: no call ever runs with another overlay's values,
and nothing waits without a bound. ``yeto run up`` builds one overlay for all its
pools (``Run.pool_environ``), so its pools never conflict. Outside a pool call the
process environment is the shell's, as before. How the rollout controller handles a
conflict (skip a tick, report it in ``check()``, leave a drain to ``yeto run down``
rather than release with other credentials) is in
:meth:`~.controller.RolloutController._pool_env`.

Limits:

* While a pool call runs, threads of the process that are not the pool's (``yeto run
  up``'s monitor loop, a dashboard thread, a pool built without an overlay) read the
  overlay too. In ``yeto run up`` the overlay is ``Run.env`` minus the process
  environment, and that process hands ``Run.env`` explicitly to what depends on these
  variables: every child (``env=self.env``) and every store client
  (``head_objects(..., environ=self.env)``, whose profile and keys come from that
  mapping, an ``AWS_PROFILE`` of the run included). What still reads ``os.environ``
  implicitly may see the run's value during a pool call and the shell's outside one:
  boto3 / botocore at client creation (e.g. ``AWS_CONFIG_FILE``), and the sampler
  layer of the breakers (``max_backlog_groups: "auto"``: the cap run up derived during
  a pool call, the fallback outside; run up resolves it in its preflight, before any
  pool call).
* A thread a provider leaves running after its call returns reads the environment
  without the overlay, and so does a credential an SDK refreshes lazily later (SSO,
  assume-role, ``credential_process``): it reads the process environment of the
  moment it refreshes.
* Waiting is not fair: when the active overlay leaves, whichever waiting scope takes
  the lock first goes, so a waiter can lose to newer scopes of another overlay until
  its ``wait_s`` runs out (``yeto run up`` has one overlay: nothing waits there).
* Something else that writes one of the overlay's variables during a scope is read by
  the pool's calls from then on; at the end a value it wrote that differs from the
  overlay's is kept, one equal to the overlay's is replaced by the earlier value.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any


class EnvOverlayConflict(RuntimeError):
    """A scope of an overlay other than the one active in this process."""


# how long a call through EnvScopedOps (a background start or release) waits for another
# overlay to leave before it raises EnvOverlayConflict
OPS_WAIT_S = 120.0

_COND = threading.Condition(threading.Lock())
_active: dict[str, str | None] | None = None  # the overlay applied now
_count = 0  # scopes of it open
_saved: dict[str, str | None] = {}  # each of its variables before it was applied (None: unset)


def _set(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _restore(overlay: Mapping[str, str | None], saved: Mapping[str, str | None]) -> None:
    for name in saved:
        if os.environ.get(name) == overlay[name]:  # else written by something else meanwhile: kept
            _set(name, saved[name])


def _conflict(overlay: Mapping[str, str | None]) -> EnvOverlayConflict:
    active = _active or {}
    differ = sorted(k for k in set(active) | set(overlay) if active.get(k, ...) != overlay.get(k, ...))
    return EnvOverlayConflict(f"another environment overlay is active in this process (variables that differ: "
                              f"{', '.join(differ)}): one overlay at a time per process")


@contextmanager
def scoped_environ(overlay: Mapping[str, str | None] | None, *, wait_s: float = 0.0) -> Iterator[None]:
    """``os.environ`` with ``overlay`` (a value None: the variable unset) inside the block.

    Scopes of the overlay active in the process (equal mappings) nest and overlap
    freely. Another overlay (a subset of the active one included) waits up to
    ``wait_s`` (default 0: not at all) for the active one to leave, then raises
    :class:`EnvOverlayConflict` (variable names, never values): a call never runs with
    another overlay's values. A thread that holds an overlay and opens another one waits
    on itself: it raises after ``wait_s``. An empty or None overlay changes nothing.
    Applying is all or nothing: a variable that cannot be set restores the ones set
    before it and raises."""
    global _active, _count, _saved
    if not overlay:
        yield
        return
    overlay = dict(overlay)
    counted = False  # this scope holds a count (set before the count is taken, see below)
    try:
        # the try opens before the count is taken, and ``counted`` is set before it: an
        # asynchronous exception (a signal) at any point still reaches the finally below
        with _COND:
            deadline = time.monotonic() + max(0.0, wait_s)
            while _active is not None and _active != overlay:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise _conflict(overlay)
                _COND.wait(left)
            if _active is None:
                saved: dict[str, str | None] = {}
                try:
                    for name, value in overlay.items():
                        saved[name] = os.environ.get(name)
                        _set(name, value)
                except BaseException:
                    _restore(overlay, saved)
                    raise
                counted = True
                _active, _count, _saved = overlay, 1, saved
            else:
                counted = True
                _count += 1
        yield
    finally:
        if counted:
            with _COND:
                _count -= 1
                if _count == 0:
                    _restore(_active or {}, _saved)
                    _active, _saved = None, {}
                    _COND.notify_all()


class EnvScopedOps:
    """A pool's ops whose every call runs in :func:`scoped_environ` of ``overlay``.
    Attributes that are not callable pass through; a missing attribute stays missing
    (the pool probes optional ops with ``getattr(ops, name, None)``).

    A call made inside a scope of the same overlay (the controller's tick of the pool)
    never waits. A call on another thread that starts while another overlay is active
    (a background start or release submitted in a tick that has ended since) waits up
    to ``wait_s`` (:data:`OPS_WAIT_S`) for it to leave, then raises
    :class:`EnvOverlayConflict` without calling the provider: the pool takes it as a
    failed start or release (retried by its own paths, never run with other values)."""

    def __init__(self, ops: Any, overlay: Mapping[str, str | None], *, wait_s: float = OPS_WAIT_S) -> None:
        self.__dict__["_ops"], self.__dict__["_overlay"] = ops, dict(overlay)
        self.__dict__["_wait_s"] = float(wait_s)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.__dict__["_ops"], name)
        if not callable(value):
            return value
        overlay, wait_s = self.__dict__["_overlay"], self.__dict__["_wait_s"]

        def scoped(*args: Any, **kw: Any) -> Any:
            with scoped_environ(overlay, wait_s=wait_s):
                return value(*args, **kw)

        return scoped

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.__dict__["_ops"], name, value)
