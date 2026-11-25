"""A ``docker pull`` of an island's image that fails the same way on every attempt.

sky pulls an island's ``docker:`` image on the VM while it sets the cluster up; a pull
that fails ends the launch with sky's error, whose text carries the pull's output. Some
failures belong to the image or its reference, not to the machine or the network: a
layer the daemon cannot register (an image whose layer whiteouts a directory and writes
into it again), a manifest the registry does not have, a reference it refuses
(unauthorized, denied). Ordering the same image again, on the same target or another one,
fails the same way and bills a machine for each attempt, so such a pull is a
configuration error: not relaunched (``yeto.config_errors``), and the error says it is
the image. A timeout, a rate limit (429) or a registry 5xx stays an infrastructure
failure and keeps its relaunches.
"""

from __future__ import annotations

import re

# Lines of a pull's output that belong to the image or its reference (lower case).
DETERMINISTIC_MARKERS = (
    "failed to register layer",
    "manifest unknown",
    "no matching manifest for",
    "unauthorized",
    "authentication required",
    "pull access denied",
    "requested access to the resource is denied",
    "invalid reference format",
)
# A transient failure anywhere in the same text wins: the pull may succeed next time.
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
    "no space left on device",
    "temporary failure in name resolution",
)
_PULL = re.compile(r"docker\s+pull\b|pulling from\b", re.IGNORECASE)


class ImagePullRefused(RuntimeError):
    """A relaunch whose image pull failed deterministically (:func:`deterministic_pull_error`):
    raised by ``sky_ops.relaunch`` so the fleet controller ends the island as a
    configuration error instead of retrying until its recovery timeout."""


def deterministic_pull_error(text: object) -> str | None:
    """The line of ``text`` (a launch error, with the pull's output) that makes its image
    pull fail on every attempt (:data:`DETERMINISTIC_MARKERS`); None when ``text`` is no
    image pull, names none of them, or also names a transient failure
    (:data:`TRANSIENT_MARKERS`)."""
    body = str(text or "")
    if not _PULL.search(body):
        return None
    lowered = body.lower()
    if any(marker in lowered for marker in TRANSIENT_MARKERS):
        return None
    for line in body.splitlines():
        if any(marker in line.lower() for marker in DETERMINISTIC_MARKERS):
            return line.strip()
    return None


def image_error_message(cluster: str, line: str) -> str:
    """The configuration error of a deterministic pull failure on ``cluster``."""
    return (f"the island image cannot be pulled on {cluster}: {line} (a property of the image or its "
            "reference, the same on every machine: fix or replace the image, then launch again; "
            "not relaunched)")
