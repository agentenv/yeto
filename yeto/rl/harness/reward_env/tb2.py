"""Terminal-Bench 2 adapter for the reward environment (change rl-agentic-reward-env).

Reuses the task resolution and the judge command of the existing codex TB2
provider (``yeto.rl.harness.codex.tb2_provider``; imported, never modified) and
adds what that provider lacks:

- ``prebake_plan``: the dependency setup that each task's ``tests/test.sh``
  performs on every judge run (``apt-get install curl``, the uv installer,
  ``uvx -p 3.13 -w pytest==… …`` or ``pip install pytest…``), extracted so it can
  run once at image build time.  ``test.sh`` itself is never changed: at judge
  time it runs the same steps again but they hit the baked caches.
- ``PrebakedModalSandboxBackend`` + ``modal_provider``: the existing
  ``Tb2EnvironmentProvider`` with sandboxes started from the prebaked image,
  under a separate Modal app (default ``yeto-reward-env``; never the
  production ``yeto-tbench2`` name for builds).
- held-out evaluation split helpers (30 tasks by default).
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any

from yeto.rl.harness.codex import tb2_provider as tb2

from .benchmark import PrebakePlan, TaskSpec, register

NAME = "tb2"
DEFAULT_APP = "yeto-reward-env"
PROTECTED_APPS = frozenset({tb2.DEFAULT_MODAL_APP})  # "yeto-tbench2": production, never built into
APP_ENV = "YETO_REWARD_ENV_MODAL_APP"
PREBAKE_ENV = "YETO_REWARD_ENV_PREBAKE"  # "0" disables prebaking (A/B timing)

_INSTALL_PREFIXES = ("apt-get ", "apt ", "curl ", "wget ", "source ", ". ", "export ", "pip install ",
                     "pip3 install ", "python -m pip install ", "python3 -m pip install ", "uv ")
_CONTROL = re.compile(r"^(if|then|else|elif|fi|echo|exit|set)\b")


def _logical_lines(script: str) -> list[str]:
    lines: list[str] = []
    buffer = ""
    for raw in script.splitlines():
        stripped = raw.strip()
        if not buffer and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        lines.append((buffer + stripped).strip())
        buffer = ""
    if buffer.strip():
        lines.append(buffer.strip())
    return lines


_UVX_VALUE_FLAGS = frozenset({"-p", "--python", "-w", "--with", "--from", "--index", "--default-index",
                              "--index-url", "--extra-index-url", "-i", "--with-requirements",
                              "--with-editable", "--index-strategy", "--prerelease"})


def _uvx_warmup(line: str) -> str | None:
    """``uvx -p 3.13 -w a -w b pytest --ctrf …`` -> ``uvx -p 3.13 -w a -w b pytest --version``."""
    try:
        tokens = shlex.split(line)
    except ValueError:
        return None
    if not tokens or tokens[0] != "uvx":
        return None
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token in _UVX_VALUE_FLAGS:
            i += 2
            continue
        if token.startswith("-"):
            i += 1
            continue
        return shlex.join(tokens[: i + 1] + ["--version"])
    return None


def _is_test_run(line: str) -> bool:
    return line.startswith(("uvx ", "uv run ")) or bool(re.match(r"^(python3? -m )?pytest\b", line))


def prebake_from_test_sh(script: str, base_image: str) -> PrebakePlan:
    """Setup lines before the test run, verbatim, plus a cache warm-up of the uvx tool env."""
    commands: list[str] = []
    skipped: list[str] = []
    for line in _logical_lines(script):
        if _is_test_run(line):
            # shlex re-quoting would freeze shell variables (e.g. ${COMMIT_HASH}
            # set by a task-specific line): no warm-up then, judge installs it.
            warm = None if "$" in line else _uvx_warmup(line)
            if warm:
                commands.append(warm)
            elif line.startswith("uvx "):
                skipped.append(line)  # tool env not warmed
            break
        if line.startswith(("curl ", "wget ")) and not re.search(r"\|\s*(ba)?sh\b", line):
            skipped.append(line)  # a data download into the workdir, not a dependency
        elif line.startswith(_INSTALL_PREFIXES):
            commands.append(line)
        elif _CONTROL.match(line) or line.startswith("[") or line.startswith("cd "):
            continue  # PWD guards etc.: not dependencies
        else:
            skipped.append(line)
    else:
        # no recognisable test-run line: do not guess
        return PrebakePlan(base_image=base_image, commands=(), complete=False, skipped=tuple(skipped) or ("<no test run line>",))
    return PrebakePlan(base_image=base_image, commands=tuple(commands), complete=not skipped, skipped=tuple(skipped))


def prebake_script(plan: PrebakePlan) -> str:
    """One bash script (env from ``source`` must survive into the warm-up)."""
    return "set -e\ncd /tmp\n" + "\n".join(plan.commands) + "\n"


def _checkout_version(path: Path) -> str:
    import subprocess

    try:
        rev = subprocess.run(["git", "-C", str(path), "rev-parse", "--short=7", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "tb2@unknown"
    return f"tb2@{rev}" if rev else "tb2@unknown"


def _official_difficulty(task_dir: Path) -> str:
    try:
        meta = tb2.tomllib.loads((task_dir / "task.toml").read_text()).get("metadata", {})
    except (OSError, ValueError):
        return "unknown"
    value = meta.get("difficulty")
    return str(value) if value else "unknown"


class Tb2Benchmark:
    name = NAME

    def __init__(self, tasks_dir: str | os.PathLike[str] | None = None, version: str | None = None) -> None:
        raw = tasks_dir or os.environ.get(tb2.TASKS_DIR_ENV)
        if not raw:
            raise RuntimeError(f"{tb2.TASKS_DIR_ENV} is not set (terminal-bench-2 checkout root)")
        self.tasks_dir = Path(raw).expanduser().resolve()
        self.version = version or _checkout_version(self.tasks_dir)

    def task_ids(self) -> list[str]:
        return sorted(p.parent.name for p in self.tasks_dir.glob("*/task.toml"))

    def _task(self, task_id: str) -> tb2.Tb2Task:
        return tb2.resolve_task(task_id, self.tasks_dir)

    def task_spec(self, task_id: str) -> TaskSpec:
        t = self._task(task_id)
        difficulty = _official_difficulty(t.task_dir)
        return TaskSpec(benchmark=NAME, task_id=t.task_id, base_image=t.docker_image, cpus=t.cpus,
                        memory_mb=t.memory_mb, workdir=t.workdir, agent_timeout_s=t.agent_timeout_s,
                        judge_timeout_s=t.verifier_timeout_s, instruction=t.instruction,
                        benchmark_version=self.version, difficulty=difficulty,
                        difficulty_source="tb2-task.toml",
                        eval_bucket=f"tb2-{difficulty}" if difficulty in ("easy", "medium", "hard") else None)

    def prebake_plan(self, task_id: str) -> PrebakePlan:
        t = self._task(task_id)
        return prebake_from_test_sh((t.tests_dir / "test.sh").read_text(), t.docker_image)

    def judge_command(self, task_id: str, submission: str | None = None) -> str:
        del submission  # TB2 judges the sandbox the agent worked in (official stage-at-verify)
        return tb2.verifier_command(self._task(task_id))

    def parse_judge(self, task_id: str, output: str, exit_code: int, timed_out: bool) -> dict[str, Any]:
        reward = tb2._REWARD_RE.search(output)
        rc = tb2._RC_RE.search(output)
        passed = bool(reward) and reward.group(1) == "1"
        return {"reward": 1.0 if passed else 0.0, "passed": passed,
                "testsh_rc": int(rc.group(1)) if rc else exit_code,
                "reward_txt": reward.group(1) if reward else None}


register(NAME, Tb2Benchmark)


# ----------------------------------------------------------------------------- Modal backend


def modal_image(plan: PrebakePlan, *, prebake: bool = True) -> Any:
    from modal import Image

    image = Image.from_registry(plan.base_image)
    if prebake and not plan.empty:
        image = image.run_commands("bash -lc " + shlex.quote(prebake_script(plan)))
    return image


def _prebake_enabled() -> bool:
    return os.environ.get(PREBAKE_ENV, "1") != "0"


class PrebakedModalSandboxBackend(tb2.ModalSandboxBackend):
    """``tb2_provider.ModalSandboxBackend`` with the task's prebaked image."""

    def __init__(self, *, prebake: bool = True, **kwargs: Any) -> None:
        kwargs.setdefault("app_name", DEFAULT_APP)
        super().__init__(**kwargs)
        self.prebake = prebake

    def create(self, task: tb2.Tb2Task, trajectory_id: str) -> tb2.ModalSandbox:
        from modal import Sandbox

        plan = prebake_from_test_sh((task.tests_dir / "test.sh").read_text(), task.docker_image)
        tags = {"yeto-tb2-task": task.task_id, "yeto-trajectory": trajectory_id[:63],
                "yeto-prebake": plan.digest()[:16] if self.prebake and not plan.empty else "none"}
        if self.run_id:
            tags["yeto-run-id"] = self.run_id[:63]
        sandbox = Sandbox.create(
            "sleep", "infinity", app=self._get_app(), image=modal_image(plan, prebake=self.prebake),
            timeout=self.ttl_s, idle_timeout=self.idle_timeout_s, cpu=float(task.cpus),
            memory=int(task.memory_mb), workdir=task.workdir, tags=tags,
        )
        return tb2.ModalSandbox(sandbox, task.workdir)


def modal_provider(miles_args: Any = None) -> tb2.Tb2EnvironmentProvider:
    """``YETO_HARNESS_ENVIRONMENT_PROVIDER=yeto.rl.harness.reward_env.tb2:modal_provider``."""
    del miles_args
    tb2.require_modal_client()
    backend = PrebakedModalSandboxBackend(
        prebake=_prebake_enabled(),
        app_name=os.environ.get(APP_ENV, DEFAULT_APP),
        ttl_s=int(tb2._env_float(tb2.SANDBOX_TTL_ENV, tb2.DEFAULT_SANDBOX_TTL_S) or tb2.DEFAULT_SANDBOX_TTL_S),
        idle_timeout_s=int(tb2._env_float(tb2.IDLE_TIMEOUT_ENV, tb2.DEFAULT_IDLE_TIMEOUT_S)
                           or tb2.DEFAULT_IDLE_TIMEOUT_S),
    )
    return tb2._provider(backend)


def check_build_app(app_name: str) -> str:
    if app_name in PROTECTED_APPS:
        raise ValueError(f"refusing to build into production Modal app {app_name!r}")
    return app_name


# ----------------------------------------------------------------------------- hold-out (WP3 D2/D6b)

HOLDOUT_SEED = 20261008
HOLDOUT_QUOTAS = {"tb2-easy": 2, "tb2-medium": 18, "tb2-hard": 10}
HOLDOUT_RULE = "stratified by difficulty: easy 2 / medium 18 / hard 10"


def build_holdout(adapter: "Tb2Benchmark", **kwargs: Any) -> dict[str, Any]:
    from .benchmark import build_holdout as _build

    return _build(adapter, HOLDOUT_QUOTAS, seed=HOLDOUT_SEED, rule=HOLDOUT_RULE, **kwargs)
