"""Runtime check: a learner island must not hold cloud credentials
(secret-handling-hardening, design D4).

The launcher never ships cloud credentials to a learner island (the head keeps
them). This module checks that promise inside the island process at start-up:

* a cloud credential in the process environment (``MODAL_TOKEN_ID`` and the
  other names in :data:`CLOUD_CREDENTIAL_ENV_NAMES`) is an error;
* a cloud credential file under ``$HOME`` (``~/.modal.toml``, ``~/.aws/...``)
  is a warning, because some images may ship such a file that the island
  never reads.

``YETO_ALLOW_ISLAND_CLOUD_CREDENTIALS=1`` turns the error into a warning (for
a developer running an island by hand). Messages name the variable or file,
never the value.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

# Same names as yeto.launcher.CLOUD_CREDENTIAL_ENV (a test keeps them equal);
# kept here so the island does not import the launcher.
CLOUD_CREDENTIAL_ENV_NAMES = frozenset({
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
    "RUNPOD_API_KEY",
    "NEBIUS_IAM_TOKEN", "NEBIUS_TENANT_ID",
    "VERDA_CLIENT_ID", "VERDA_CLIENT_SECRET",
    "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET",
})
CLOUD_CREDENTIAL_FILES = (".modal.toml", ".aws/credentials", ".runpod", ".nebius", ".verda")
ALLOW_ENV = "YETO_ALLOW_ISLAND_CLOUD_CREDENTIALS"


class IslandCloudCredentialError(RuntimeError):
    """A learner island process holds a cloud credential in its environment."""


def find_cloud_credentials(environ: Mapping[str, str] | None = None,
                           home: str | os.PathLike | None = None) -> tuple[list[str], list[str]]:
    """(credential env names that are set, credential files that exist)."""
    environ = os.environ if environ is None else environ
    names = sorted(n for n in CLOUD_CREDENTIAL_ENV_NAMES if environ.get(n))
    base = Path(home if home is not None else environ.get("HOME") or Path.home())
    files = [f"~/{rel}" for rel in CLOUD_CREDENTIAL_FILES if (base / rel).exists()]
    return names, files


def check_island_credentials(environ: Mapping[str, str] | None = None,
                             home: str | os.PathLike | None = None,
                             stream=None) -> None:
    """Raise IslandCloudCredentialError for a credential env var (unless
    YETO_ALLOW_ISLAND_CLOUD_CREDENTIALS=1); warn on stderr for credential files."""
    environ = os.environ if environ is None else environ
    stream = sys.stderr if stream is None else stream
    names, files = find_cloud_credentials(environ, home)
    if files:
        print(f"[yeto-island] WARNING: cloud credential files present on this island: "
              f"{', '.join(files)} (the island must not use them)", file=stream, flush=True)
    if not names:
        return
    message = (f"cloud credentials in the learner island environment: {', '.join(names)}; "
               "the launcher must not ship them to an island (secret-handling-hardening)")
    if environ.get(ALLOW_ENV) == "1":
        print(f"[yeto-island] WARNING: {message} ({ALLOW_ENV}=1)", file=stream, flush=True)
        return
    raise IslandCloudCredentialError(message)
