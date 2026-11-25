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


def config_error_line(error: BaseException) -> str:
    """The marker line an island prints before it exits :data:`CONFIG_ERROR_EXIT`."""
    return (f"{CONFIG_ERROR_MARKER} {type(error).__name__}: {error} "
            f"(configuration error: exit {CONFIG_ERROR_EXIT}, not retried)")
