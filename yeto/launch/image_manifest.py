"""Is a digest-pinned image in its registry? A manifest HEAD, nothing downloaded.

A trainer island that pulls its image learns that the image is missing only after the
machine is ordered and the pull fails. This asks the registry first, over the Docker
Registry HTTP API v2: ``HEAD /v2/<repository>/manifests/<digest>``, with the token flow
the registry asks for (``WWW-Authenticate: Bearer``: ghcr.io, Docker Hub; ``Basic``:
ECR), anonymous unless a registry login is given (the ``SKYPILOT_DOCKER_*`` triple the
island would pull with). The login is sent to the registry's own token endpoint only
and never printed.

Redirects are not followed by urllib (its redirect handler copies the
``Authorization`` header to whatever host the registry points at: an ECR or Docker
Hub registry redirects to a presigned storage URL). They are followed here, at most
:data:`MAX_REDIRECTS` hops, and the ``Authorization`` header goes only to the host it
was meant for, over https.

``YETO_IMAGE_MANIFEST_CHECK=off`` turns the check off (unit tests; a host that cannot
reach the registry while the nodes can).
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

CHECK_ENV = "YETO_IMAGE_MANIFEST_CHECK"
ACCEPT = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))
PRESENT, MISSING, DENIED, UNKNOWN = "present", "missing", "denied", "unknown"
MAX_REDIRECTS = 5
_REDIRECTS = (301, 302, 303, 307, 308)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect comes back as an HTTPError (3xx) instead of being followed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _open(url: str, *, method: str, headers: dict[str, str], timeout: float, urlopen):
    """``urlopen`` of ``url``, following redirects with the ``Authorization`` header
    dropped as soon as a hop leaves the original host or https."""
    origin = urllib.parse.urlsplit(url)
    for _ in range(MAX_REDIRECTS + 1):
        try:
            return urlopen(urllib.request.Request(url, method=method, headers=headers), timeout=timeout)
        except urllib.error.HTTPError as error:
            location = (error.headers or {}).get("Location") if error.code in _REDIRECTS else None
            if not location:
                raise
            url = urllib.parse.urljoin(url, location)
            hop = urllib.parse.urlsplit(url)
            if hop.scheme != "https" or hop.netloc.lower() != origin.netloc.lower():
                headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
    raise urllib.error.URLError(f"more than {MAX_REDIRECTS} redirects from {origin.netloc}")


def enabled() -> bool:
    return os.environ.get(CHECK_ENV, "on").strip().lower() not in ("off", "0", "false", "no")


def split_ref(ref: str) -> tuple[str, str, str]:
    """``docker:<host>/<repo>@sha256:<hex>`` -> (host, repo, digest); Docker Hub names
    (no host part) map to ``registry-1.docker.io`` (``library/`` for a bare name)."""
    name, sep, digest = ref.removeprefix("docker:").partition("@")
    if not sep or not digest.startswith("sha256:"):
        raise ValueError(f"{ref!r} is not a digest-pinned image reference")
    first, slash, rest = name.partition("/")
    if slash and ("." in first or ":" in first or first == "localhost"):
        host, repo = first, rest
    else:
        host, repo = "registry-1.docker.io", name if slash else f"library/{name}"
    if host == "docker.io":
        host = "registry-1.docker.io"
    return host, repo, digest


def _basic(login: dict[str, str] | None) -> str | None:
    if not login:
        return None
    raw = f"{login['SKYPILOT_DOCKER_USERNAME']}:{login['SKYPILOT_DOCKER_PASSWORD']}".encode()
    return base64.b64encode(raw).decode()


def manifest_status(ref: str, login: dict[str, str] | None = None, *, timeout: float = 10.0,
                    urlopen=None) -> tuple[str, str]:
    """``(status, detail)``: ``present`` (HTTP 200), ``missing`` (404), ``denied`` (401/403:
    absent, or private and no / a wrong login: a registry does not tell which; and a 400 to
    a request that carried the login: ECR answers a malformed or wrong login so) or
    ``unknown`` (the registry was not reached, or answered something else)."""
    host, repo, digest = split_ref(ref)
    url = f"https://{host}/v2/{repo}/manifests/{digest}"
    basic = _basic(login)
    urlopen = urlopen or _OPENER.open

    def head(auth: str | None) -> tuple[int, dict]:
        headers = {"Accept": ACCEPT, **({"Authorization": auth} if auth else {})}
        try:
            with _open(url, method="HEAD", headers=headers, timeout=timeout, urlopen=urlopen) as r:
                return int(r.status), dict(r.headers)
        except urllib.error.HTTPError as error:
            return int(error.code), dict(error.headers or {})

    sent_login = False  # the code below answers a request (or token request) that carried the login
    try:
        code, headers = head(None)
        if code == 401:
            challenge = next((v for k, v in headers.items() if k.lower() == "www-authenticate"), "")
            if challenge.lower().startswith("bearer"):
                params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
                if "realm" not in params:
                    return UNKNOWN, f"{host}: a Bearer challenge without a realm"
                query = urllib.parse.urlencode({k: params[k] for k in ("service", "scope") if k in params})
                try:
                    with _open(f"{params['realm']}?{query}", method="GET", timeout=timeout, urlopen=urlopen,
                               headers={"Authorization": f"Basic {basic}"}
                               if basic and params["realm"].startswith("https://") else {}) as r:
                        body = json.loads(r.read() or b"{}")
                except urllib.error.HTTPError as error:
                    code, sent_login = int(error.code), bool(basic and params["realm"].startswith("https://"))
                else:
                    token = body.get("token") or body.get("access_token")
                    code = head(f"Bearer {token}")[0] if token else 401
            elif basic:
                code, sent_login = head(f"Basic {basic}")[0], True
    except (urllib.error.URLError, OSError, ValueError) as error:
        return UNKNOWN, f"{host} not reached ({type(error).__name__}: {error})"[:300]
    if code == 200:
        return PRESENT, f"{host}/{repo}@{digest}"
    if code == 404:
        return MISSING, f"{host}/{repo} has no manifest {digest}"
    if code == 400 and sent_login:
        return DENIED, (f"{host} rejected the registry login given for {repo}@{digest} (HTTP 400): a malformed "
                        "or wrong password (an ECR password is `aws ecr get-login-password`, valid 12 hours); a registry "
                        f"that answers 400 for another reason: {CHECK_ENV}=off skips this check")
    if code in (401, 403):
        how = "with the registry login given" if login else "anonymously"
        return DENIED, (f"{host}/{repo}@{digest} is not readable {how} (HTTP {code}): it is not in that "
                        "registry, or it is private and needs a registry login")
    return UNKNOWN, f"{host} answered HTTP {code} for {repo}@{digest}"
