"""Backend-neutral engine exceptions (yeto-framework-decoupling 2.1). Import-light.

Exceptions the engine core reacts to by type live here so the core never
imports a backend adapter to recognise them; adapters raise these objects
(the old adapter paths re-export the same class).
"""

from __future__ import annotations

from typing import Any


class RecoveryRequired(RuntimeError):
    """The trainer cannot be brought back from the cut; stop data consumption."""

    def __init__(self, message: str, *, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = attempts
