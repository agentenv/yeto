"""Harbor's pi agent pointed at yeto's records proxy; see :mod:`.pi_config`."""

from __future__ import annotations

from typing import Any

from harbor.agents.installed.node_install import nvm_node_install_snippet
from harbor.agents.installed.pi import Pi

from .pi_config import DEFAULT_HTTP_IDLE_TIMEOUT_S, DEFAULT_MAX_RETRIES, UNBOUNDED_CONTEXT_WINDOW, setup_command
from .pins import PI_PACKAGE, PI_VERSION


class YetoPi(Pi):
    """Harbor's ``Pi`` with two changes.

    * ``install``: the package pi is published under now (Harbor's agent
      installs ``@mariozechner/pi-coding-agent``, deprecated upstream in favour
      of :data:`~.pins.PI_PACKAGE`), at the pinned version by default.
    * ``run``: writes pi's ``models.json`` / ``settings.json`` first, then runs
      Harbor's command unchanged. The model name must be ``yeto/<model>``, so
      Harbor's ``--provider`` selects the ``yeto`` provider and copies no host
      API key into the sandbox.
    """

    def __init__(self, *args: Any, max_tokens: int, context_window: int = UNBOUNDED_CONTEXT_WINDOW,
                 http_idle_timeout_s: int = DEFAULT_HTTP_IDLE_TIMEOUT_S, max_retries: int = DEFAULT_MAX_RETRIES,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._max_tokens = int(max_tokens)
        self._context_window = int(context_window)
        self._http_idle_timeout_s = int(http_idle_timeout_s)
        self._max_retries = int(max_retries)

    def preinstalled_command(self) -> str:
        """Exits 0 when the image already has the wanted pi under nvm (a prebuilt sandbox image)."""
        version = self._version or PI_VERSION
        return f'. ~/.nvm/nvm.sh >/dev/null 2>&1; [ "$(pi --version 2>/dev/null | tail -n1)" = "{version}" ]'

    async def install(self, environment: Any) -> None:
        check = await environment.exec(command=self.preinstalled_command())
        if check.return_code == 0:
            return  # the image ships this pi build: no apt, nvm or npm per trial
        await self.exec_as_root(environment, command="apt-get update && apt-get install -y curl",
                                env={"DEBIAN_FRONTEND": "noninteractive"})
        await self.exec_as_agent(environment, command=(
            f"set -euo pipefail; {nvm_node_install_snippet()} && "
            f"npm install -g {PI_PACKAGE}@{self._version or PI_VERSION} && pi --version"))

    def config_command(self) -> str:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        return setup_command(self.model_name.split("/", 1)[1], self._max_tokens,
                             context_window=self._context_window,
                             http_idle_timeout_s=self._http_idle_timeout_s, max_retries=self._max_retries)

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        await self.exec_as_agent(environment, command=self.config_command())
        await super().run(instruction, environment, context)  # renders the prompt template
