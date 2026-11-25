"""A trial whose sandbox provider or harness cannot import a module (the sampler image lacks
the provider's dependency, e.g. a data row's ``provider: boat`` with no ``boat_sdk``) is a
configuration error: never retried as infrastructure, and a few in a row stop the sampler
with exit 2 (its pool halts until ``pool-retry``), naming the provider and the module."""

from __future__ import annotations

import pytest

from test_agentic_engine_sampler import Agent, _entries, _island, qwen  # noqa: F401 (fixture)
from yeto.rl.agentic import SandboxDependencyError, TrialVerdict
from yeto.rl.engine import agentic_round

MISSING = TrialVerdict(rewards=None, exception_type="ModuleNotFoundError",
                       exception_message="No module named 'boat_sdk'")
# another trial's first import of a package with an import cycle was still running: the
# next import succeeds (harbor's own, before #830)
PARTIAL = TrialVerdict(rewards=None, exception_type="ImportError",
                       exception_message="cannot import name 'Provider' from partially initialized module "
                                         "'acme_sandbox' (most likely due to a circular import)")


class ImportFailing(Agent):
    """Trials in ``missing`` end as Harbor reports a provider module missing from the image
    (ABORTED before the agent ran, no model call); the others run as :class:`Agent`'s."""

    def __init__(self, rewards, *, missing: set[int], verdict: TrialVerdict = MISSING) -> None:
        super().__init__(rewards)
        self.missing = set(missing)
        self.verdict = verdict

    async def run(self, task, endpoint):
        index = self.calls
        if index in self.missing:
            self.calls += 1
            return self.verdict
        return await super().run(task, endpoint)


def test_a_missing_provider_module_is_not_retried_as_infrastructure(tmp_path, qwen):  # noqa: F811
    """One trial that could not import boat_sdk: lost, not run again (infra_retries 2); the
    trials after it run, so the sampler goes on."""
    agent = ImportFailing([1.0] * 5, missing={0})
    island, store, events, recorder, engine = _island(tmp_path, qwen, agent, min_samples=1, infra_retries=2)
    try:
        assert island.run() == 1
    finally:
        island.round.close()
    assert not [f for n, f in events if n == "rl_agentic_trial_retry"]
    assert agent.calls == 4  # no attempt again
    (round_event,) = [f for n, f in events if n == "rl_sampler_engine_round"]
    assert round_event["agentic"]["infra_retries"] == {}


def test_missing_provider_modules_in_a_row_stop_the_sampler_naming_the_provider_and_module(tmp_path, qwen):  # noqa: F811
    agent = ImportFailing([1.0] * 8, missing=set(range(8)))
    island, store, events, recorder, engine = _island(tmp_path, qwen, agent, groups=4, min_samples=1,
                                                      infra_retries=2)
    try:
        with pytest.raises(SandboxDependencyError) as raised:
            island.run()
    finally:
        island.round.close()
    text = str(raised.value)
    assert "'boat'" in text and "boat_sdk" in text and "ModuleNotFoundError" in text
    assert f"{agentic_round.DEPENDENCY_STREAK_N} trials in a row" in text
    assert agent.calls <= agentic_round.DEPENDENCY_STREAK_N + 1  # stopped, not spinning
    assert not [f for n, f in events if n == "rl_agentic_trial_retry"]


def test_a_trial_that_gets_further_resets_the_count(tmp_path, qwen):  # noqa: F811
    """Two import failures, then a trial that runs, then two more: never three in a row."""
    agent = ImportFailing([1.0] * 8, missing={0, 1, 3, 4})
    island, store, events, recorder, engine = _island(tmp_path, qwen, agent, groups=4, min_samples=1,
                                                      infra_retries=0)
    try:
        assert island.run() == 1
    finally:
        island.round.close()
    assert agent.calls == 8


def test_a_racing_import_of_a_circular_package_is_retried_never_stopping(tmp_path, qwen):  # noqa: F811
    """REV on #844: the first wave of concurrent trials importing a provider package with an
    import cycle fails with "partially initialized module"; later imports succeed. Those
    trials stay infrastructure failures (retried), and no number of them stops the sampler."""
    agent = ImportFailing([1.0] * 12, missing={0, 1, 2, 3}, verdict=PARTIAL)
    island, store, events, recorder, engine = _island(tmp_path, qwen, agent, groups=4, min_samples=1,
                                                      infra_retries=2)
    try:
        assert island.run() == 1
    finally:
        island.round.close()
    assert len([f for n, f in events if n == "rl_agentic_trial_retry"]) >= agentic_round.DEPENDENCY_STREAK_N


def test_a_lost_trial_neither_counts_nor_resets_the_streak():
    """REV on #844: a cancelled or session-collision trial (_Lost) between two import
    failures says nothing about the image: the third failure still stops the sampler."""
    from types import SimpleNamespace

    from yeto.rl.agentic import AgentTask, AgentTrajectory, Outcome

    task = AgentTask(task_id="t", agent="codex", provider="boat")
    failed = type("Finished", (), {"turns": (), "trajectory": AgentTrajectory(
        task=task, outcome=Outcome.ABORTED, reward=0.0, reason="ModuleNotFoundError", verdict=MISSING)})()
    state = SimpleNamespace(_dependency_streak=0, _error=None, _task=lambda item, why: task)
    group = SimpleNamespace(item=SimpleNamespace(prompt_id="p0"))
    step = agentic_round.AgenticRound._dependency
    for result in (failed, failed, agentic_round._Lost("cancelled"), failed):
        step(state, group, result)
    assert isinstance(state._error, SandboxDependencyError) and "'boat'" in str(state._error)


def test_missing_dependency_reads_harbor_s_verdict_and_nothing_else():
    from yeto.rl.agentic import AgentTask, AgentTrajectory, Outcome

    task = AgentTask(task_id="t", agent="codex", provider="boat")

    def finished(outcome, verdict, reason=""):
        traj = AgentTrajectory(task=task, outcome=outcome, reward=0.0, reason=reason, verdict=verdict)
        return type("Finished", (), {"trajectory": traj, "turns": ()})()

    assert agentic_round.missing_dependency(finished(Outcome.ABORTED, MISSING, "ModuleNotFoundError")) == \
        "ModuleNotFoundError: No module named 'boat_sdk'"
    assert agentic_round.infra_failure(finished(Outcome.ABORTED, MISSING, "ModuleNotFoundError")) is None
    other = TrialVerdict(rewards=None, exception_type="RuntimeError", exception_message="sandbox lost")
    assert agentic_round.missing_dependency(finished(Outcome.ABORTED, other, "RuntimeError")) is None
    assert agentic_round.infra_failure(finished(Outcome.ABORTED, other, "RuntimeError")) == "RuntimeError"
    assert agentic_round.missing_dependency(finished(Outcome.ABORTED, PARTIAL, "ImportError")) is None
    assert agentic_round.infra_failure(finished(Outcome.ABORTED, PARTIAL, "ImportError")) == "ImportError"
    version = TrialVerdict(rewards=None, exception_type="ImportError",
                           exception_message="cannot import name 'Sandbox' from 'e2b' (/opt/e2b/__init__.py)")
    assert agentic_round.missing_dependency(finished(Outcome.ABORTED, version, "ImportError")) == \
        "ImportError: cannot import name 'Sandbox' from 'e2b' (/opt/e2b/__init__.py)"
    no_verdict = finished(Outcome.ABORTED, None, "ModuleNotFoundError: No module named 'e2b'")
    assert agentic_round.missing_dependency(no_verdict) == "ModuleNotFoundError: No module named 'e2b'"


def test_harbor_never_setup_retries_a_missing_module():
    from types import SimpleNamespace

    from yeto.rl.harness.harbor.trial import retryable_setup_failure

    def result(name, message=""):
        return SimpleNamespace(exception_info=SimpleNamespace(exception_type=name, exception_message=message),
                               agent_execution=None, verifier_result=None)

    assert retryable_setup_failure(result("ModuleNotFoundError", "No module named 'boat_sdk'")) is None
    assert retryable_setup_failure(result("ImportError", "libcrypto.so.3: cannot open shared object file")) is None
    assert retryable_setup_failure(result("RuntimeError")) == "RuntimeError"
    assert retryable_setup_failure(result("ImportError", PARTIAL.exception_message)) == "ImportError"
    assert retryable_setup_failure(result("_DeadlockError", "deadlock detected by _ModuleLock('acme')")) == \
        "_DeadlockError"


def test_the_sampler_exits_with_the_configuration_error_code(monkeypatch, tmp_path, capsys):
    """exit 2: the pool reads it as a configuration error (status.CONFIG_EXIT_CODES) and halts."""
    from yeto.launch.pool.status import CONFIG_EXIT_CODES
    from yeto.rl.engine import engine_sampler_entry as entry

    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    monkeypatch.setattr(entry, "load_spec", lambda path: {})

    def run(spec, **kw):
        raise SandboxDependencyError("sandbox provider 'boat': ModuleNotFoundError: No module named 'boat_sdk'")

    monkeypatch.setattr(entry, "run", run)
    assert entry.main(["--spec", str(spec)]) == 2 and 2 in CONFIG_EXIT_CODES
    assert "configuration error: sandbox provider 'boat'" in capsys.readouterr().err
    monkeypatch.setattr(entry, "run", lambda spec, **kw: (_ for _ in ()).throw(RuntimeError("other")))
    with pytest.raises(RuntimeError):  # any other error keeps its exit
        entry.main(["--spec", str(spec)])
