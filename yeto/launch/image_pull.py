"""A ``docker pull`` of an island's image that fails the same way on every attempt.

sky pulls an island's ``docker:`` image on the VM while it sets the cluster up; a pull
that fails ends the launch with sky's error, whose text carries the pull's output. Some
failures belong to the image or its reference, not to the machine or the network: a
layer the daemon cannot register (an image whose layer whiteouts a directory and writes
into it again), a manifest the registry does not have, a reference it cannot parse.
Ordering the same image again, on the same target or another one, fails the same way and
bills a machine for each attempt, so such a pull is a configuration error: not
relaunched (``yeto.config_errors``), and the error says it is the image.

A registry that refuses the login (unauthorized, denied, an expired token) is a
credential or permission problem. Every provision runs ``docker login`` on its new VM
(sky's docker setup) with the login the launch read when it built its task (the
``SKYPILOT_DOCKER_*`` variables, else ``~/.docker/config.json``); yeto mints no registry
token itself. A launch's first provision has just read it: refused there, it is a
configuration error ("check the registry login"). A relaunch inside a running launch
reuses the login read at its start, which may have expired since (an ECR password lasts
12 hours): it reads the login again and tries once more only when that login differs
from the refused one (by fingerprint); the same login fails the same way, so it gives up
at once (``sky_ops.relaunch``).

Only docker's own error lines count (``Error response from daemon: ...``, ``failed to
register layer: ...``): a line of another tool in the same text (a Hugging Face 401, git's
"Authentication required") says nothing about the image. A docker line that names a
transient or host problem (a timeout, a rate limit, a registry 5xx, a reset, an
input/output error, a read-only or full disk) stays an infrastructure failure and keeps
its relaunches: another attempt or another machine may pull the image.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

IMAGE = "image"
CREDENTIAL = "credential"
# docker's own error lines in a pull's output (lower case, after leading space)
DOCKER_LINE_PREFIXES = ("error response from daemon:", "docker: error response from daemon:",
                        "failed to register layer:", "error pulling image configuration:")
# A property of the image or its reference: the same on every machine.
IMAGE_MARKERS = (
    "failed to register layer",
    "manifest unknown",
    "no matching manifest for",
    "invalid reference format",
)
# The registry refused the login or the image's access.
CREDENTIAL_MARKERS = (
    "unauthorized",
    "authentication required",
    "no basic auth credentials",
    "pull access denied",
    "requested access to the resource is denied",
    "authorization token has expired",
)
# On a docker line, a transient or host problem wins: another attempt or machine may pull.
TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "toomanyrequests",
    "too many requests",
    " 429",
    " 500 ",
    " 502 ",
    " 503 ",
    " 504 ",
    "internal server error",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
    "connection reset",
    "connection refused",
    "unexpected eof",
    "temporary failure in name resolution",
    # the machine's own disk, not the image
    "no space left on device",
    "input/output error",
    "read-only file system",
)
_PULL = re.compile(r"docker\s+pull\b|pulling from\b", re.IGNORECASE)


@dataclass(frozen=True)
class PullFailure:
    """A deterministic pull failure: ``kind`` :data:`IMAGE` or :data:`CREDENTIAL`, and docker's line."""

    kind: str
    line: str


class ImagePullRefused(RuntimeError):
    """A relaunch whose image pull failed deterministically (:func:`deterministic_pull_error`):
    raised by ``sky_ops.relaunch`` so the fleet controller ends the island as a
    configuration error instead of retrying until its recovery timeout."""


def _docker_lines(body: str) -> list[str]:
    return [line.strip() for line in body.splitlines()
            if line.strip().lower().startswith(DOCKER_LINE_PREFIXES)]


def pull_failure(text: object) -> PullFailure | None:
    """The deterministic failure of the image pull in ``text`` (a launch error with the
    pull's output): docker's first error line naming an image marker (:data:`IMAGE`) or a
    credential marker (:data:`CREDENTIAL`). None when ``text`` is no image pull, no docker
    line names one, or a docker line names a transient or host problem."""
    body = str(text or "")
    if not _PULL.search(body):
        return None
    lines = _docker_lines(body)
    if any(marker in line.lower() for line in lines for marker in TRANSIENT_MARKERS):
        return None
    for line in lines:
        lowered = line.lower()
        if any(marker in lowered for marker in IMAGE_MARKERS):
            return PullFailure(IMAGE, line)
        if any(marker in lowered for marker in CREDENTIAL_MARKERS):
            return PullFailure(CREDENTIAL, line)
    return None


def deterministic_pull_error(text: object) -> str | None:
    """docker's line of :func:`pull_failure`, or None."""
    found = pull_failure(text)
    return None if found is None else found.line


def image_error_message(cluster: str, failure: PullFailure | str, *, note: str = "") -> str:
    """The configuration error of a deterministic pull failure on ``cluster`` (``note``: why a
    credential failure was not tried again)."""
    if isinstance(failure, str):
        failure = PullFailure(IMAGE, failure)
    if failure.kind == CREDENTIAL:
        return (f"the island image cannot be pulled on {cluster}: {failure.line} (credential or image "
                "permission problem: check the registry login (SKYPILOT_DOCKER_USERNAME / "
                "SKYPILOT_DOCKER_PASSWORD / SKYPILOT_DOCKER_SERVER, else ~/.docker/config.json; an ECR "
                "password lasts 12 hours) and the image's access, then launch again; not relaunched"
                + (f"; {note}" if note else "") + ")")
    return (f"the island image cannot be pulled on {cluster}: {failure.line} (a property of the image or its "
            "reference, the same on every machine: fix or replace the image, then launch again; "
            "not relaunched)")


def login_fingerprint(login: dict | None) -> str:
    """A digest of a registry login triple (user, password, server), to compare two logins
    without keeping or showing either."""
    import hashlib

    from ..modal_runner import DOCKER_LOGIN_ENV_VARS

    login = login or {}
    raw = "\0".join(str(login.get(key) or "") for key in DOCKER_LOGIN_ENV_VARS)
    return hashlib.sha256(raw.encode()).hexdigest()


def fresh_registry_login(server: str | None, environ=None) -> dict[str, str] | None:
    """The registry login for ``server`` read now: the ``SKYPILOT_DOCKER_*`` variables, else
    its ``~/.docker/config.json`` entry; None without one."""
    import os

    from ..modal_runner import registry_credentials
    from .credentials import docker_config_login

    if not server:
        return None
    ref = f"{server.rstrip('/')}/image"
    try:
        found = registry_credentials(ref, os.environ if environ is None else environ)
    except ValueError:
        found = None
    return found or docker_config_login(ref)
