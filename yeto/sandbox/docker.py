"""The docker CLI of one sandbox host (``DOCKER_HOST`` per call) and its SSH shell.

Every call goes through one ``runner(argv, env, stdin) -> (rc, stdout, stderr)``
so tests drive a fake daemon. The process environment is never changed: the
daemon is named in each call's own environment, so one process drives several
hosts at once.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

Runner = Callable[[Sequence[str], Mapping[str, str], "bytes | None"], tuple[int, str, str]]

LABEL_RUN = "yeto.sandbox.run"
LABEL_TASK = "yeto.sandbox.task"
LABEL_SIDECAR = "yeto.sandbox.sidecar"
LABEL_SPEC = "yeto.sandbox.spec"
_DAEMON_SELECTORS = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")


class DockerError(RuntimeError):
    pass


def subprocess_runner(timeout_s: float = 7200.0) -> Runner:
    def run(argv: Sequence[str], env: Mapping[str, str], stdin: bytes | None) -> tuple[int, str, str]:
        try:
            proc = subprocess.run(list(argv), env=dict(env), input=stdin, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return 124, "", f"timeout after {timeout_s:.0f} s"
        except FileNotFoundError as exc:
            return 127, "", str(exc)
        return proc.returncode, proc.stdout.decode(errors="replace"), proc.stderr.decode(errors="replace")

    return run


class DockerHost:
    """``docker`` against ``docker_host``; ``ssh`` (optional) for host-side probes."""

    def __init__(self, docker_host: str, *, ssh: str | None = None, runner: Runner | None = None,
                 docker_bin: str = "docker") -> None:
        self.docker_host = docker_host
        self.ssh = ssh
        self.runner = runner or subprocess_runner()
        self.docker_bin = docker_bin
        base = {k: v for k, v in os.environ.items() if k not in _DAEMON_SELECTORS}
        self.env = {**base, "DOCKER_HOST": docker_host}

    def docker(self, *args: str, stdin: bytes | None = None, check: bool = True) -> tuple[int, str, str]:
        rc, out, err = self.runner([self.docker_bin, *args], self.env, stdin)
        if check and rc != 0:
            raise DockerError(f"docker {' '.join(args[:3])}: rc={rc}: {err.strip()[-500:]}")
        return rc, out, err

    def shell(self, command: str) -> tuple[int, str, str]:
        """A command on the host itself (needs ``ssh``)."""
        if not self.ssh:
            return 127, "", "no ssh target for this host"
        return self.runner(["ssh", "-o", "BatchMode=yes", self.ssh, command], self.env, None)

    # -- probes ---------------------------------------------------------
    def info(self) -> dict[str, Any]:
        _, out, _ = self.docker("info", "--format", "{{json .}}")
        return json.loads(out or "{}")

    def bridge_ip(self) -> str | None:
        rc, out, _ = self.docker("network", "inspect", "bridge", "--format",
                                 "{{range .IPAM.Config}}{{.Gateway}} {{end}}", check=False)
        for word in out.split() if rc == 0 else []:
            if word.count(".") == 3:
                return word
        return None

    def image_id(self, ref: str) -> str | None:
        rc, out, _ = self.docker("image", "inspect", "--format", "{{.Id}}", ref, check=False)
        return out.strip() or None if rc == 0 else None

    def pull(self, ref: str) -> None:
        self.docker("pull", ref)

    def build(self, tag: str, context: str | None, *, dockerfile: str | None = None,
              labels: Mapping[str, str] | None = None) -> None:
        args = ["build", "-t", tag]
        for k, v in dict(labels or {}).items():
            args += ["--label", f"{k}={v}"]
        if dockerfile is not None:
            self.docker(*args, "-", stdin=dockerfile.encode())
        else:
            self.docker(*args, str(context))

    def containers(self, *filters: str, all_: bool = True) -> list[dict[str, Any]]:
        args = ["ps", "--no-trunc", "--format", "{{json .}}"]
        if all_:
            args.insert(1, "-a")
        for f in filters:
            args += ["--filter", f]
        _, out, _ = self.docker(*args)
        return [json.loads(line) for line in out.splitlines() if line.strip()]

    def count_running(self) -> int:
        _, out, _ = self.docker("ps", "-q")
        return sum(1 for line in out.splitlines() if line.strip())

    def labelled_containers(self, keys: Sequence[str], *filters: str,
                            all_: bool = False) -> list[dict[str, Any]]:
        """``[{"id", "labels": {key: value}}]`` of the containers matching ``filters``
        (running only unless ``all_``); one ``docker ps`` call."""
        fmt = "\t".join(["{{.ID}}", *('{{.Label "%s"}}' % k for k in keys)])
        args = ["ps", "--no-trunc", "--format", fmt]
        if all_:
            args.insert(1, "-a")
        for f in filters:
            args += ["--filter", f]
        _, out, _ = self.docker(*args)
        rows = []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            parts += [""] * (len(keys) + 1 - len(parts))
            rows.append({"id": parts[0], "labels": {k: v for k, v in zip(keys, parts[1:]) if v}})
        return rows

    def remove_networks(self, *filters: str) -> None:
        """Remove the networks matching ``filters`` (a network still in use stays)."""
        args = ["network", "ls", "-q"]
        for f in filters:
            args += ["--filter", f]
        rc, out, _ = self.docker(*args, check=False)
        ids = out.split() if rc == 0 else []
        if ids:
            self.docker("network", "rm", *ids, check=False)

    def inspect(self, name: str) -> dict[str, Any] | None:
        rc, out, _ = self.docker("container", "inspect", name, check=False)
        if rc != 0:
            return None
        data = json.loads(out or "[]")
        return data[0] if data else None

    def published_ports(self) -> set[int]:
        used: set[int] = set()
        for c in self.containers(all_=False):
            for part in str(c.get("Ports") or "").split(","):
                part = part.strip()
                if "->" in part:
                    host = part.split("->", 1)[0].rsplit(":", 1)[-1]
                    for piece in host.split("-"):
                        if piece.isdigit():
                            used.add(int(piece))
        return used

    def run_detached(self, name: str, image: str, *, publish: str, env: Iterable[tuple[str, str]],
                     labels: Mapping[str, str]) -> str:
        args = ["run", "-d", "--name", name, "--restart", "unless-stopped", "-p", publish]
        for k, v in env:
            args += ["-e", f"{k}={v}"]
        for k, v in labels.items():
            args += ["--label", f"{k}={v}"]
        _, out, _ = self.docker(*args, image)
        return out.strip()

    def remove(self, ids: Sequence[str]) -> None:
        if ids:
            self.docker("rm", "-f", *ids)

    def http_ok(self, url: str) -> bool | None:
        """``curl`` on the host (published ports bind the bridge address); None: no ssh."""
        if not self.ssh:
            return None
        rc, _, _ = self.shell(f"curl -sf --max-time 5 {shlex.quote(url)} >/dev/null")
        return rc == 0

    def disk(self, path: str) -> dict[str, int] | None:
        rc, out, _ = self.shell(f"df -Pk {shlex.quote(path)} | tail -n1")
        if rc != 0:
            return None
        parts = out.split()
        try:
            return {"total_kb": int(parts[1]), "used_kb": int(parts[2]), "free_kb": int(parts[3])}
        except (IndexError, ValueError):
            return None
