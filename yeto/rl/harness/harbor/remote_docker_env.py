"""Harbor's docker environment on a docker daemon of another host.

Harbor's ``DockerEnvironment`` declares ``mounted=True``: the trial's log
directories (``/logs/agent``, ``/logs/verifier``) are bind mounts of the trial
directory on the host that runs the trial, and the verifier's reward is read
from there. With ``DOCKER_HOST=ssh://user@host`` the docker CLI talks to the
remote daemon, but a bind mount then names a directory on the *remote* host:
the reward never reaches the trial process and every trial ends without a
verifier result.

:class:`RemoteDockerEnvironment` is the same environment with two changes:
``capabilities.mounted`` is false, so Harbor's ``Trial`` downloads the agent
and verifier directories (``docker compose cp``, through ``DOCKER_HOST``), as
for its cloud providers; and the mount targets are created inside the
container without a bind volume. Everything else (build, compose, exec,
upload, teardown) is Harbor's code.

The daemon comes from the process environment (``DOCKER_HOST``); every docker
CLI call of Harbor inherits it, so one trial process drives one sandbox host.
The trial host needs the docker CLI with the compose plugin; for ``ssh://`` a
non-interactive SSH login to the sandbox host (an ``ssh_config`` entry with
``ControlMaster`` keeps one connection for the many CLI calls).

``provider: remote-docker`` (:data:`.trial.CUSTOM_PROVIDERS`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harbor.environments.docker import write_mounts_compose_file
from harbor.environments.docker.docker import DockerEnvironment

from yeto.rl.agentic import HarnessConfigError

DOCKER_HOST_ENV = "DOCKER_HOST"
# A run label on every sandbox container (``docker ps --filter label=ai.yeto.run=<run>``):
# teardown on a shared sandbox host removes only the containers of its own run.
RUN_LABEL_ENV = "YETO_SANDBOX_RUN_LABEL"
RUN_LABEL_KEY = "ai.yeto.run"


class RemoteDockerEnvironment(DockerEnvironment):
    """``DockerEnvironment`` without host bind mounts (module doc)."""

    def __init__(self, *args: Any, docker_host: str | None = None, **kwargs: Any) -> None:
        # provider_kwargs {"docker_host": ...} (yeto sandbox validate, HARBOR_ENV_KWARGS): the docker CLI
        # reads DOCKER_HOST from the process environment, so a process drives one sandbox host
        if docker_host and not os.environ.get(DOCKER_HOST_ENV, "").strip():
            os.environ[DOCKER_HOST_ENV] = docker_host
        elif docker_host and os.environ[DOCKER_HOST_ENV].strip() != docker_host:
            raise HarnessConfigError(f"docker_host {docker_host} differs from {DOCKER_HOST_ENV}="
                                     f"{os.environ[DOCKER_HOST_ENV]}: one process drives one sandbox host")
        if not os.environ.get(DOCKER_HOST_ENV, "").strip():
            raise HarnessConfigError(f"provider remote-docker needs {DOCKER_HOST_ENV} (e.g. ssh://sbxdock@sandbox-host: a non-root user with rootless docker)")
        super().__init__(*args, **kwargs)

    @property
    def capabilities(self) -> Any:
        return super().capabilities.model_copy(update={"mounted": False})

    def _write_mounts_compose_file(self) -> Path:
        # The targets stay in self._mounts (start() creates them in the container); no volume binds them.
        self._cleanup_mounts_compose_file()
        import tempfile

        self._mounts_compose_temp_dir = tempfile.TemporaryDirectory()
        path = Path(self._mounts_compose_temp_dir.name) / "docker-compose-mounts.json"
        write_mounts_compose_file(path, [])
        # run / shard / incarnation / trial (yeto.sandbox.trial_labels): a later incarnation of
        # the same shard removes what a killed one left (sweep_stale)
        from yeto.sandbox.trial_labels import trial_labels  # noqa: PLC0415

        labels = trial_labels(getattr(self, "session_id", None))
        if labels:
            import json

            doc = json.loads(path.read_text())
            doc["services"]["main"]["labels"] = labels
            path.write_text(json.dumps(doc, indent=2))
        return path

    async def prepare_logs_for_host(self) -> None:
        """Nothing to chown: the trial host has no view of the container's files."""
