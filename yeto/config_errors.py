"""Deterministic configuration errors: refused before any work, never retried.

Standard library only: the island entrypoints, the Modal container runner and the
launcher's fleet controller all import it.

An island that refuses its own configuration (:class:`ConfigError`, raised by a
pure check of the launch options) exits with :data:`CONFIG_ERROR_EXIT` and prints
:data:`CONFIG_ERROR_MARKER` with the reason. The same options fail the same way on
every attempt, so neither the cloud platform (Modal's function retries) nor the
fleet controller (island relaunch) runs the island again, and the launcher ends
the run with :data:`CONFIG_ERROR_EXIT` (registered with the other launcher exit
codes in ``yeto.launch.fleet``).
"""

from __future__ import annotations

# Island process exit code and launcher exit code of a configuration error.
CONFIG_ERROR_EXIT = 10
# Log marker of an island's configuration error (the fleet controller's log probes read it).
CONFIG_ERROR_MARKER = "[yeto-config-error]"
# Island process exit code of a deterministic stop: a trainer breaker stopped it (no new group
# within trainer.sample_wait, a damaged buffer, ...), a strict check failed, or the syncer refused
# it. A relaunch stops the same way and bills a machine meanwhile, so neither Modal's function
# retries nor the fleet controller run the island again (``yeto.launch.island_lifecycle``).
ISLAND_STOP_EXIT = 11
# Log marker of a deterministic stop: ``[yeto-island-stop] <cause> <Error>: <message>``.
ISLAND_STOP_MARKER = "[yeto-island-stop]"
# The island's switch for both (``launch.deterministic_stop``; the launch sets it to 1 on every
# island of a fleet, or of a single trainer with --rl-deterministic-stop). Unset: the island
# raises a deterministic stop as any error (exit 1; Modal retries it, the controller relaunches).
DETERMINISTIC_STOP_ENV = "YETO_RL_DETERMINISTIC_STOP"


def deterministic_stop_on(environ=None) -> bool:
    """Whether this island process ends a deterministic stop with :data:`ISLAND_STOP_EXIT`."""
    import os  # noqa: PLC0415

    env = os.environ if environ is None else environ
    return str(env.get(DETERMINISTIC_STOP_ENV, "")).strip().lower() in ("1", "true", "on", "yes")


class ConfigError(ValueError):
    """A launch option combination that cannot run: the same on every attempt.

    A ``ValueError`` so callers that refuse bad options with ``ValueError`` keep
    catching it."""


def config_error_in(error: BaseException | None) -> ConfigError | None:
    """The :class:`ConfigError` in ``error``'s cause / context chain; None without one."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ConfigError):
            return error
        error = error.__cause__ or error.__context__
    return None


def island_stop_exit(error: BaseException, stream=None) -> int | None:
    """:data:`ISLAND_STOP_EXIT` when ``error`` is a deterministic stop of the island (a trainer
    breaker's stop such as ``trainer.sample_wait``, a strict failure, the syncer's refusal:
    ``yeto.launch.island_lifecycle.island_stop_line``, whose whitelist decides), after its
    traceback and the ``[yeto-island-stop]`` line the launcher's log probes read. A relaunch
    would stop the same way and bill a machine meanwhile, so the exit is not retried (Modal's
    function retries and the fleet controller both see it). None for any other error, and for
    every error while the launch's ``launch.deterministic_stop`` is off
    (:data:`DETERMINISTIC_STOP_ENV` unset: a single trainer by default): raised as before.
    Every island entry ends its deterministic stops here (the Miles island entry, the verl
    training process ``yeto.rl.adapters.verl.verl_main``)."""
    import sys  # noqa: PLC0415
    import traceback  # noqa: PLC0415

    if not deterministic_stop_on():  # launch.deterministic_stop off: raised as before (exit 1)
        return None
    from yeto.launch.island_lifecycle import island_stop_line  # noqa: PLC0415 - stdlib-only module top

    line = island_stop_line(error)
    if line is None:
        return None
    stream = sys.stderr if stream is None else stream
    traceback.print_exception(error, file=stream)
    print(line, file=stream, flush=True)
    return ISLAND_STOP_EXIT


def config_error_line(error: BaseException) -> str:
    """The marker line an island prints before it exits :data:`CONFIG_ERROR_EXIT`."""
    return (f"{CONFIG_ERROR_MARKER} {type(error).__name__}: {error} "
            f"(configuration error: exit {CONFIG_ERROR_EXIT}, not retried)")
