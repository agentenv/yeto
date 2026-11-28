"""The launcher's half of the node runtime on a pool route.

A route that switched to the node runtime (the ssh pool, the sky route of the
SSH + docker pool) keeps everything it already does to reach and prepare a node
(preflight, code upload, env file, the container's flags and mounts) and changes
one thing: what the node runs. Instead of ``python -m yeto.rl.engine.engine_sampler
--spec`` it runs ``python -m yeto.node run --assignment <ticket>`` (:func:`runtime_command`),
after :meth:`NodeRoute.assign` wrote the assignment to the run store and the
route put the ticket's token and the node's profile next to the spec. The
sampler's start, its records and its exit are then the node runtime's, the same
on every source; the pool reads them from the store (:meth:`NodeRoute.probe`).

``node_runtime: false`` in the pool config (or ``provider.node_runtime`` of an
ssh pool) keeps the route's own sampler start for one release.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from yeto.launch.node_inventory import NodeProfile

from . import code as node_code
from .assign import assign_member
from .profile import profile_doc
from .records import Assignment, CodeRef, node_status, probe_of


def enabled(cfg: Mapping[str, Any], *, default: bool, section: Mapping[str, Any] | None = None) -> bool:
    """Whether a pool route starts its samplers through the node runtime: ``node_runtime``
    of the provider section, else of the pool config, else ``default``."""
    for doc in (section or {}, cfg):
        value = doc.get("node_runtime")
        if value is not None:
            if not isinstance(value, bool):
                raise ValueError(f"node_runtime must be true or false, not {value!r}")
            return value
    return default


def runtime_command(python: str, ticket: str, store: str, *, env_file: str | None, token_file: str,
                    profile_file: str | None = None, child_pid_file: str | None = None, log: str | None = None,
                    skip_preflight: bool = True, health_every_s: float | None = None) -> str:
    """``exec <python> -m yeto.node run ...``: the node's start line (the token in a file, never
    in this line). ``skip_preflight``: the route ran its own preflight already.
    ``health_every_s``: the node health check's period (None: the runtime's default; 0: off)."""
    argv = [python, "-m", "yeto.node", "run", "--assignment", ticket, "--store", store, "--token-file", token_file]
    if env_file:
        argv += ["--env-file", env_file]
    if profile_file:
        argv += ["--profile", profile_file]
    if child_pid_file:
        argv += ["--child-pid-file", child_pid_file]
    if log:
        argv += ["--log", log]
    if skip_preflight:
        argv.append("--skip-preflight")
    if health_every_s is not None:
        argv += ["--health-every-s", f"{float(health_every_s):g}"]
    return "exec " + shlex.join(argv)


class NodeRoute:
    """Assignments of one pool in its run store. ``objects``: the store's object API
    (a callable: opened on first use); ``store``: the store as the node names it
    (``s3://<bucket>/runs/<prefix>``, or a directory the node reads)."""

    def __init__(self, cfg: Mapping[str, Any], *, kind: str, objects: Callable[[], Any], store: Callable[[], str],
                 code_ref: str = "HEAD", run: Callable[..., Any] = subprocess.run, publish_code: bool = True) -> None:
        """``publish_code`` False: the route puts the tree on the node itself (a store on the
        node: the launcher cannot write a tar there by object), the assignment names its hash."""
        self.cfg, self.kind, self._objects, self._store = cfg, kind, objects, store
        self.code_ref, self._run, self.publish_code = code_ref, run, publish_code
        self._code: CodeRef | None = None
        self._objects_value: Any = None

    @property
    def objects(self) -> Any:
        if self._objects_value is None:
            self._objects_value = self._objects()
        return self._objects_value

    @property
    def store(self) -> str:
        return self._store()

    def code(self) -> CodeRef:
        """The route's yeto tree in the store (``node-code/<sha>.tar``), published once;
        ``<sha>`` is the short commit the route's own code upload uses."""
        if self._code is None:
            if self.publish_code:
                self._code = node_code.publish_code(self.objects, *node_code.git_code_tar(self.code_ref, run=self._run))
            else:
                sha = node_code.code_sha(self.code_ref, repo=node_code.REPO_ROOT, run=self._run)
                self._code = CodeRef(sha=sha, key=node_code.code_key(sha), sha256="")
        return self._code

    def assign(self, member: Any, shards: Sequence[str], profile: NodeProfile, spec: Mapping[str, Any], *,
               cfg: Mapping[str, Any] | None = None, **extra: Any) -> tuple[Assignment, str]:
        """``cfg``: the pool config as the node sees it (default the route's)."""
        return assign_member(self.objects, self.cfg if cfg is None else cfg, member, shards, profile, self.code(),
                             source=self.kind, spec=spec, store_uri=self.store if "://" in self.store else None,
                             **extra)

    @staticmethod
    def profile_file_text(profile: NodeProfile, **overrides: Any) -> str:
        """The profile as the runtime on the node reads it (``--profile``): a container-wrapped
        runtime sees its own paths and starts the sampler as a process there."""
        return json.dumps({**profile_doc(profile), **overrides}, sort_keys=True)

    def probe(self, member: Any, *, stale_s: float | None = None) -> Any:
        """The pool's probe of ``member`` from the store alone (``yeto.node.records``)."""
        kw = {} if stale_s is None else {"stale_s": stale_s}
        try:
            return probe_of(node_status(self.objects, member.ticket, **kw))
        except Exception:  # noqa: BLE001 - the store did not answer: no reading this tick
            from yeto.launch.pool.status import PROBE_UNREACHABLE, Probe  # noqa: PLC0415

            return Probe(node=PROBE_UNREACHABLE)


class ShellObjects:
    """The object API of a store directory on a node, through the route's own shell
    (``sh(command, input=None) -> result``): what the launcher writes there (the
    assignment) and reads back (its generation, the node records)."""

    def __init__(self, sh: Callable[..., Any], root: str) -> None:
        self.sh, self.root = sh, root.rstrip("/")

    def _path(self, key: str) -> str:
        return shlex.quote(f"{self.root}/{key}")

    def read(self, key: str) -> bytes | None:
        """The object, or None when the file is absent; an ssh failure raises (it is not
        an absent object: the assignment's generation would restart at 1)."""
        out = self.sh(f"cat {self._path(key)} 2>/dev/null || echo YETO_NODE_ABSENT_$?", None)
        text = out.stdout if isinstance(out.stdout, str) else (out.stdout or b"").decode()
        if out.returncode != 0:
            raise RuntimeError(f"reading {self.root}/{key} on the node failed (exit {out.returncode})")
        return None if text.startswith("YETO_NODE_ABSENT_") else text.encode()

    def write(self, key: str, data: bytes) -> None:
        q = self._path(key)
        out = self.sh(f'mkdir -p "$(dirname {q})" && cat > {q}.tmp && mv -f {q}.tmp {q}', data.decode())
        if out.returncode != 0:
            raise RuntimeError(f"writing {self.root}/{key} on the node failed (exit {out.returncode})")

    def delete(self, key: str) -> None:
        self.sh(f"rm -f {self._path(key)}", None)
