"""Harbor as the yeto agent harness: AgentTask -> TrialConfig -> Trial.run() -> TrialVerdict.

Harbor does the work (task loading, sandbox provider, agent install and
launch, tool exec, verifier, teardown).  This module only translates at the
two edges and is shared by both backends:

* :class:`HarborHarness` implements ``yeto.rl.agentic.AgentHarness``; the
  ``TrialConfig`` builder is injected (verl: :func:`build_trial_config`;
  Miles: the Miles example's builder for Harbor's built-in providers,
  :func:`build_trial_config` for :data:`CUSTOM_PROVIDERS`), so is the
  ``Trial`` class (CPU tests).
* :func:`verdict_from_result` reads a Harbor ``TrialResult`` into the neutral
  :class:`~yeto.rl.agentic.TrialVerdict`; it uses attribute access only, so
  any object of the same shape works.

Harbor is imported lazily: the yeto core and CPU tests do not need it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yeto.rl.agentic import AgentTask, HarnessConfigError, TrialVerdict
from yeto.rl.agentic.trust import check_agent_env, task_dir

from .pins import CODEX_VERSION

TrialConfigBuilder = Callable[[AgentTask, str], Any]

# Providers Harbor does not ship, loaded by ``EnvironmentConfig.import_path``
# (Harbor unchanged).  Key = ``AgentTask.provider``.
CUSTOM_PROVIDERS: Mapping[str, str] = {
    "boat": "yeto.rl.harness.harbor.boat_env:BoatEnvironment",
}

# The agent sees a dummy key: the endpoint is yeto's recorder / Miles' session
# server, which does not authenticate the agent.
AGENT_API_KEY = "dummy"

# Harbor codex CLI flags: the gateway serves no reasoning summaries (it refuses
# ``reasoning.summary`` other than none) and no hosted web search.
CODEX_KWARGS: Mapping[str, str] = {"reasoning_summary": "none", "web_search": "disabled"}


@dataclass(frozen=True)
class HarborSettings:
    """Where Harbor finds tasks and writes trials, and backend-specific provider kwargs."""

    tasks_dir: Path
    trials_dir: Path = Path("/tmp/harbor_trials")
    provider_kwargs: Mapping[str, Any] = field(default_factory=dict)
    timeout_multiplier: float = 1.0

    @classmethod
    def from_env(cls) -> HarborSettings:
        """Worker settings from the environment; both backends read the same variables.

        ``HARBOR_TASKS_DIR`` (required), ``HARBOR_TRIALS_DIR`` (default
        ``/tmp/harbor_trials``), ``HARBOR_ENV_KWARGS`` (JSON object, provider
        kwargs), ``HARBOR_TIMEOUT_MULTIPLIER`` (default 1.0).
        """
        tasks_dir = os.environ.get("HARBOR_TASKS_DIR", "").strip()
        if not tasks_dir:
            raise HarnessConfigError("set HARBOR_TASKS_DIR to the Harbor task directories")
        try:
            provider_kwargs = json.loads(os.environ.get("HARBOR_ENV_KWARGS", "") or "{}")
            multiplier = float(os.environ.get("HARBOR_TIMEOUT_MULTIPLIER", "") or 1.0)
        except ValueError as exc:
            raise HarnessConfigError(f"bad Harbor environment variable: {exc}") from exc
        if not isinstance(provider_kwargs, dict):
            raise HarnessConfigError("HARBOR_ENV_KWARGS must be a JSON object")
        return cls(
            tasks_dir=Path(tasks_dir),
            trials_dir=Path(os.environ.get("HARBOR_TRIALS_DIR", "") or "/tmp/harbor_trials"),
            provider_kwargs=provider_kwargs,
            timeout_multiplier=multiplier,
        )


def agent_env(agent: str, endpoint: str) -> dict[str, str]:
    """Environment that points an installed agent at ``endpoint`` (OpenAI base URL).

    Harbor's codex agent copies ``OPENAI_BASE_URL`` into ``$CODEX_HOME/config.toml``
    (``openai_base_url``) itself.  Other OpenAI-protocol agents read one of the two
    names.  Agents with another protocol (claude-code) are not wired here yet.
    """
    if agent == "claude-code":
        raise HarnessConfigError("claude-code needs an Anthropic endpoint; not wired on the yeto path")
    return check_agent_env({"OPENAI_BASE_URL": endpoint, "OPENAI_API_BASE": endpoint,
                            "OPENAI_API_KEY": AGENT_API_KEY})


def build_trial_config(task: AgentTask, endpoint: str, settings: HarborSettings) -> Any:
    """yeto's own ``TrialConfig`` builder.

    verl uses it for every provider; Miles uses it for :data:`CUSTOM_PROVIDERS`
    (the Miles example only accepts Harbor's built-in ``EnvironmentType``).
    """
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig

    try:
        path = task_dir(settings.tasks_dir, task)
        if task.provider in CUSTOM_PROVIDERS:
            env_target: dict[str, Any] = {"import_path": CUSTOM_PROVIDERS[task.provider]}
        else:
            env_target = {"type": EnvironmentType(task.provider)}
    except (FileNotFoundError, ValueError) as exc:  # TrustViolation (escape) is not caught: it stops the run
        raise HarnessConfigError(str(exc)) from exc
    kwargs: dict[str, Any] = {}
    version = task.agent_version or (CODEX_VERSION if task.agent == "codex" else None)
    if version is not None:
        kwargs["version"] = version
    if task.agent == "codex":
        kwargs.update(CODEX_KWARGS)
    if task.budget.max_seq_len is not None and task.agent.startswith("terminus"):
        kwargs["max_seq_len"] = task.budget.max_seq_len
    return TrialConfig(
        task=TaskConfig(path=path),
        agent=AgentConfig(
            name=task.agent,
            model_name=f"openai/{task.model_name}",
            override_timeout_sec=task.budget.agent_timeout_s,
            env=agent_env(task.agent, endpoint),
            kwargs=kwargs,
        ),
        environment=EnvironmentConfig(**env_target, delete=True, kwargs=dict(settings.provider_kwargs)),
        trials_dir=settings.trials_dir,
        timeout_multiplier=settings.timeout_multiplier,
    )


def verdict_from_result(result: Any, *, trial_dir: str | None = None) -> TrialVerdict:
    """Harbor ``TrialResult`` -> :class:`TrialVerdict` (no classification here)."""
    exc = getattr(result, "exception_info", None)
    verifier = getattr(result, "verifier_result", None)
    rewards = getattr(verifier, "rewards", None) if verifier is not None else None
    timings: dict[str, float] = {}
    for name, phase in (("total_time", result),
                        ("env_setup_time", getattr(result, "environment_setup", None)),
                        ("agent_setup_time", getattr(result, "agent_setup", None)),
                        ("agent_run_time", getattr(result, "agent_execution", None)),
                        ("eval_time", getattr(result, "verifier", None))):
        started = getattr(phase, "started_at", None)
        finished = getattr(phase, "finished_at", None)
        if started is not None and finished is not None:
            timings[name] = (finished - started).total_seconds()
    metrics: dict[str, Any] = {}
    agent_result = getattr(result, "agent_result", None)
    for name in ("n_input_tokens", "n_output_tokens", "n_steps"):
        if (value := getattr(agent_result, name, None)) is not None:
            metrics[name] = value
    return TrialVerdict(
        rewards=dict(rewards) if rewards is not None else None,
        exception_type=getattr(exc, "exception_type", None) if exc is not None else None,
        exception_message=str(getattr(exc, "exception_message", "")) if exc is not None else "",
        timings_s=timings,
        agent_metrics=metrics,
        trial_dir=trial_dir,
    )


class HarborHarness:
    """``AgentHarness`` over Harbor ``Trial``; one sandbox per call."""

    def __init__(self, build_config: TrialConfigBuilder, *, trial_cls: Any = None) -> None:
        self._build_config = build_config
        self._trial_cls = trial_cls

    def _trial(self) -> Any:
        if self._trial_cls is None:
            from harbor.trial.trial import Trial

            self._trial_cls = Trial
        return self._trial_cls

    async def run(self, task: AgentTask, endpoint: str) -> TrialVerdict:
        config = self._build_config(task, endpoint)  # config errors raise: they fail every sample
        trial = await self._trial().create(config)
        # Trial.run stops its environment on cancellation (the outer wall-clock cap).
        result = await trial.run()
        paths = getattr(trial, "paths", None)
        trial_dir = getattr(paths, "trial_dir", None)
        return verdict_from_result(result, trial_dir=None if trial_dir is None else str(trial_dir))
