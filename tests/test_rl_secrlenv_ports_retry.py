"""P0-1.7b.4-Q1: the SecRLEnv group check and same-task retry on ports.

CPU only: the algorithm spec accepts the group check, the Miles translation
binds the retry classes, the retry data source keeps the legacy contract, the
terminal cleanup sanitizes errors, and the Miles classes load against stub
Miles modules.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from test_rl_miles_adapter_config import flag_value, make_config, sub
from yeto.rl import (
    SECRLENV_AGENT,
    SECRLENV_GENERATE,
    SECRLENV_GROUP_FILTER,
)
from yeto.rl.adapters.miles import config as cfg
from yeto.rl.adapters.miles import secrlenv_retry as retry
from yeto.rl.adapters.miles.binding import check_filter
from yeto.rl.adapters.miles.entry import miles_capabilities
from yeto.rl.engine.algorithm import SECRLENV_GROUP_CHECK_FILTER, AlgorithmSpec
from yeto.rl.harness.codex.reward import (
    SecRLEnvReplacementExhausted,
    SecRLEnvRolloutCleanupError,
    SecRLEnvUntrustedEvidence,
    UntrustedOutcome,
)

TARGET = 2


# ------------------------------------------------------------------ spec
def _secrlenv_spec() -> AlgorithmSpec:
    return AlgorithmSpec(
        advantage_estimator="grpo",
        dynamic_sampling_filter=SECRLENV_GROUP_FILTER,
        dynamic_sampling_max_replacements=0,
    )


def test_spec_accepts_the_group_check_under_its_neutral_name():
    spec = _secrlenv_spec()
    assert spec.dynamic_sampling_filter == SECRLENV_GROUP_CHECK_FILTER
    assert spec.sha256() == AlgorithmSpec(
        advantage_estimator="grpo",
        dynamic_sampling_filter=SECRLENV_GROUP_CHECK_FILTER,
        dynamic_sampling_max_replacements=0,
    ).sha256()
    legacy_args = SimpleNamespace(dynamic_sampling_filter_path=SECRLENV_GROUP_FILTER,
                                  dynamic_sampling_max_replacements=0, kl_coef=None)
    assert AlgorithmSpec.from_legacy_args(legacy_args).sha256() == spec.sha256()
    capabilities = miles_capabilities("sha256:" + "0" * 64)
    assert SECRLENV_GROUP_CHECK_FILTER in capabilities.dynamic_sampling_filters
    assert ("dynamic_sampling_filters", SECRLENV_GROUP_CHECK_FILTER) in spec.required_mechanisms()


def test_binding_check_skips_only_the_evidence_call():
    row = check_filter(SECRLENV_GROUP_CHECK_FILTER)
    assert (row["mapped"], row["argv"]) == (SECRLENV_GROUP_FILTER, "same")
    assert row["identity"]["source_sha256"]
    assert row["call"] == "skipped: needs environment evidence"


# ----------------------------------------------------------- translation
def _secrlenv_config(**batch):
    config = sub(make_config(), "agent",
                 custom_generate_function_path=SECRLENV_GENERATE,
                 custom_agent_function_path=SECRLENV_AGENT,
                 dynamic_sampling_filter_path=SECRLENV_GROUP_FILTER)
    groups = make_config().batch.groups_per_round
    return sub(config, "batch", **{"over_sampling_batch_size": groups + 1, **batch})


def test_translation_binds_the_retry_classes():
    argv = list(cfg.translate_run_config(_secrlenv_config(), _secrlenv_spec()).argv)
    assert flag_value(argv, "--dynamic-sampling-filter-path") == SECRLENV_GROUP_FILTER
    assert flag_value(argv, "--data-source-path") == retry.DATA_SOURCE_PATH
    assert flag_value(argv, "--rollout-function-path") == retry.ROLLOUT_FUNCTION_PATH
    assert flag_value(argv, "--custom-generate-function-path") == SECRLENV_GENERATE
    assert flag_value(argv, "--custom-agent-function-path") == SECRLENV_AGENT


def test_translation_leaves_other_runs_unchanged():
    argv = cfg.translate_run_config(make_config(), AlgorithmSpec()).argv
    assert "--data-source-path" not in argv and "--rollout-function-path" not in argv


def test_retry_classes_are_adapter_owned():
    assert {"--data-source-path", "--rollout-function-path"} <= cfg.ADAPTER_OWNED_FLAGS


@pytest.mark.parametrize("config, spec, match", [
    (sub(_secrlenv_config(), "agent", dynamic_sampling_filter_path=None), AlgorithmSpec(),
     "go together"),
    (sub(make_config(), "agent", dynamic_sampling_filter_path=SECRLENV_GROUP_FILTER),
     _secrlenv_spec(), "go together"),
    (_secrlenv_config(over_sampling_batch_size=make_config().batch.groups_per_round),
     _secrlenv_spec(), "exactly one refill slot"),
])
def test_translation_refuses_a_broken_retry_contract(config, spec, match):
    with pytest.raises(cfg.UnmappedConfigError, match=match):
        cfg.translate_run_config(config, spec)


def test_retry_refuses_a_policy_age_limit():
    with pytest.raises(ValueError, match="policy-age limit"):
        retry.rollout_argv(custom_agent=SECRLENV_AGENT,
                           dynamic_filter=SECRLENV_GROUP_CHECK_FILTER,
                           rollout_batch_size=TARGET, over_sampling_batch_size=TARGET + 1,
                           max_policy_age=1)


# ------------------------------------------------------- retry data source
class _Sample:
    def __init__(self, index, status="completed"):
        self.index = index
        self.status = status
        self.metadata = {"secrlenv_trusted_outcome": {"index": index}}
        self.resets = 0

    def reset_for_retry(self):
        self.status = "pending"
        self.resets += 1


class _Source:
    """Stands in for Miles' RolloutDataSourceWithBuffer."""

    def __init__(self, args):
        self.args = args
        self.next_index = 0
        self.buffer = []
        self.draws = []

    def get_samples(self, num_samples):
        self.draws.append(num_samples)
        groups = [[_Sample(self.next_index + g * 10 + s) for s in range(2)]
                  for g in range(num_samples)]
        self.next_index += 100
        return groups

    def add_samples(self, samples):
        self.buffer.extend(samples)


class _RetrySource(retry.SecRLEnvTaskRetry, _Source):
    pass


def _args(**overrides):
    values = dict(rollout_batch_size=TARGET, over_sampling_batch_size=TARGET + 1,
                  yeto_rl_secrlenv_max_infrastructure_replacements=1)
    values.update(overrides)
    return SimpleNamespace(**values)


def _ledger(group, value=0):
    return {sample.index: value for sample in group}


def _request(source, group, *, key="evidence", reason="secrlenv_infrastructure_failure",
             ledger=None):
    source.request_retry(group, evidence_key=key, reason=reason,
                         infrastructure_503_retries=_ledger(group) if ledger is None else ledger)


@pytest.mark.parametrize("overrides", [
    dict(over_sampling_batch_size=TARGET),
    dict(rollout_batch_size=0, over_sampling_batch_size=1),
    dict(yeto_rl_secrlenv_max_infrastructure_replacements=2),
    dict(yeto_rl_secrlenv_max_infrastructure_replacements=None),
])
def test_data_source_refuses_a_broken_contract(overrides):
    with pytest.raises(SecRLEnvReplacementExhausted, match="contract is invalid"):
        _RetrySource(_args(**overrides))


def test_round_draws_the_exact_batch_then_only_the_queued_retry():
    args = _args()
    source = _RetrySource(args)
    assert getattr(args, retry.RETRY_CALLBACK_ATTR) == source.request_retry
    groups = source.get_samples(TARGET + 1)
    assert len(groups) == TARGET and source.draws == [TARGET]
    failed = groups[1]
    failed[0].status = "failed"
    failed[0].metadata["secrlenv_trusted_outcome"] = {"status": "infrastructure_error"}
    _request(source, failed, ledger={failed[0].index: 1, failed[1].index: 0})
    _request(source, failed, ledger={failed[0].index: 1, failed[1].index: 0})  # repeated callback
    (rerun,) = source.get_samples(TARGET + 1)
    assert [s.index for s in rerun] == [s.index for s in failed]
    assert [s.resets for s in rerun] == [1, 1]
    assert [s.metadata for s in rerun] == [
        {"secrlenv_infrastructure_503_retries": 1},
        {"secrlenv_infrastructure_503_retries": 0},
    ]
    assert source.draws == [TARGET]  # no new task was drawn for the retry
    assert source.retry_stats() == dict(
        replacement_attempts=1, nontrainable_groups=1, infrastructure_groups=1,
        pending_groups=0, failed_groups=1)
    assert retry.retry_metrics(args._yeto_secrlenv_retry_stats)[
        "rl/secrlenv/replacement_attempts"] == 1.0


def test_refill_without_a_queued_retry_is_terminal():
    source = _RetrySource(_args())
    source.get_samples(TARGET + 1)
    with pytest.raises(SecRLEnvReplacementExhausted, match="no authenticated task retry"):
        source.get_samples(TARGET + 1)


def test_refill_size_is_part_of_the_contract():
    source = _RetrySource(_args())
    with pytest.raises(SecRLEnvReplacementExhausted, match="refill contract"):
        source.get_samples(TARGET)


def test_second_retry_in_a_round_is_terminal():
    source = _RetrySource(_args())
    groups = source.get_samples(TARGET + 1)
    _request(source, groups[0], key="a")
    source.get_samples(TARGET + 1)
    with pytest.raises(SecRLEnvReplacementExhausted, match="budget was exhausted"):
        _request(source, groups[1], key="b")


@pytest.mark.parametrize("ledger, error, match", [
    ({}, SecRLEnvUntrustedEvidence, "ledger authentication"),
    ("bad", SecRLEnvUntrustedEvidence, "ledger authentication"),
    (None, SecRLEnvReplacementExhausted, "identity did not match"),
])
def test_retry_must_name_a_drawn_task_and_its_ledger(ledger, error, match):
    source = _RetrySource(_args())
    groups = source.get_samples(TARGET + 1)
    group = groups[0] if ledger is not None else [_Sample(9999), _Sample(9998)]
    with pytest.raises(error, match=match):
        _request(source, group, ledger=ledger)


def test_ledger_values_are_bounded():
    source = _RetrySource(_args())
    group = source.get_samples(TARGET + 1)[0]
    with pytest.raises(SecRLEnvUntrustedEvidence):
        _request(source, group, ledger=_ledger(group, value=2))


def test_round_end_resets_the_budget_and_the_filter_memo():
    args = _args()
    source = _RetrySource(args)
    groups = source.get_samples(TARGET + 1)
    _request(source, groups[0])
    source.get_samples(TARGET + 1)
    args._yeto_secrlenv_filter_state = {"rollout_id": None, "decisions": {"k": (False, "x")}}
    source.add_samples([])  # Miles returns the round's aborted groups
    next_round = source.get_samples(TARGET + 1)
    assert len(next_round) == TARGET and source.draws == [TARGET, TARGET]
    assert args._yeto_secrlenv_filter_state is None
    assert source.retry_stats()["replacement_attempts"] == 0
    with pytest.raises(SecRLEnvReplacementExhausted, match="identity did not match"):
        _request(source, groups[1], key="old-round")


def test_each_round_has_a_fresh_budget():
    source = _RetrySource(_args())
    _request(source, source.get_samples(TARGET + 1)[0])
    source.get_samples(TARGET + 1)
    source.add_samples([])
    _request(source, source.get_samples(TARGET + 1)[0])
    assert len(source.get_samples(TARGET + 1)) == 1


def test_group_check_queues_the_retry_through_the_data_source(monkeypatch):
    """The real group check, fed a signed infrastructure outcome, asks the
    data source for the rerun (the callback attribute is the contract)."""

    from yeto.rl.harness.codex import reward

    monkeypatch.setattr(reward, "_verified_outcome", lambda metadata: (metadata["outcome"], 0.0))
    output_module = types.ModuleType("miles.rollout.filter_hub.base_types")
    output_module.DynamicFilterOutput = lambda keep, reason: SimpleNamespace(keep=keep, reason=reason)
    for name in ("miles", "miles.rollout", "miles.rollout.filter_hub"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "miles.rollout.filter_hub.base_types", output_module)

    args = _args(yeto_rl_dynamic_sampling_max_replacements=0)
    source = _RetrySource(args)
    group = source.get_samples(TARGET + 1)[0]
    for sample in group:
        sample.status = SimpleNamespace(value="failed")
        sample.metadata = {"outcome": {"status": reward.INFRASTRUCTURE_STATUS,
                                       reward.INFRASTRUCTURE_503_RETRIES_KEY: 0}}
    result = reward.check_group(args, group)
    assert (result.keep, result.reason) == (False, "secrlenv_infrastructure_failure")
    (rerun,) = source.get_samples(TARGET + 1)
    assert [s.index for s in rerun] == [s.index for s in group]


# ------------------------------------------------------- terminal cleanup
def _cleanup(error, *, abort=None, require_clean=lambda: None, timeout_s=1.0):
    calls = []

    async def default_abort():
        calls.append("abort")

    asyncio.run(retry.raise_after_terminal_cleanup(
        error, abort=abort or default_abort, in_flight=list, require_clean=require_clean,
        timeout_s=timeout_s))
    return calls


def test_terminal_error_is_raised_after_the_abort():
    error = SecRLEnvReplacementExhausted("budget")
    with pytest.raises(SecRLEnvReplacementExhausted) as raised:
        _cleanup(error)
    assert raised.value is error and raised.value.__cause__ is None


def test_untrusted_outcome_is_sanitized():
    with pytest.raises(SecRLEnvUntrustedEvidence, match="authentication failed") as raised:
        _cleanup(UntrustedOutcome("payload with episode secrets"))
    assert "secrets" not in str(raised.value) and raised.value.__cause__ is None


@pytest.mark.parametrize("kind", ["abort", "residue", "timeout"])
def test_cleanup_failure_replaces_the_error(kind):
    async def failing_abort():
        raise RuntimeError("engine said no")

    async def slow_abort():
        await asyncio.sleep(5)

    def residue():
        raise RuntimeError("episode left")

    options = {"abort": dict(abort=failing_abort), "residue": dict(require_clean=residue),
               "timeout": dict(abort=slow_abort, timeout_s=0.01)}[kind]
    with pytest.raises(SecRLEnvRolloutCleanupError) as raised:
        _cleanup(SecRLEnvReplacementExhausted("x"), **options)
    assert raised.value.__cause__ is None


# ------------------------------------------------------- Miles classes
@pytest.fixture
def stub_miles(monkeypatch):
    """Minimal Miles modules the binding module imports."""

    calls = []

    class RolloutDataSourceWithBuffer(_Source):
        pass

    class InferenceRolloutFn:
        def __init__(self, input):
            self.state = SimpleNamespace(args=input.args)
            self.outcome = input.outcome

        async def _call_train(self, input):
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            if callable(self.outcome):  # a driver standing in for generate_rollout_async
                return await self.outcome(self.state.args)
            return self.outcome

    async def abort(state, pendings, rollout_id):
        calls.append(("abort", pendings, rollout_id))
        return []

    modules = {
        "miles": {}, "miles.rollout": {}, "miles.rollout.inference_rollout": {},
        "miles.rollout.data_source": {"RolloutDataSourceWithBuffer": RolloutDataSourceWithBuffer},
        "miles.rollout.inference_rollout.inference_rollout_common": {
            "InferenceRolloutFn": InferenceRolloutFn},
        "miles.rollout.inference_rollout.inference_rollout_train": {"abort": abort},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "yeto.rl.adapters.miles.secrlenv_miles", raising=False)
    module = importlib.import_module("yeto.rl.adapters.miles.secrlenv_miles")
    yield module, calls
    sys.modules.pop("yeto.rl.adapters.miles.secrlenv_miles", None)


def test_miles_paths_name_the_binding_classes(stub_miles):
    module, _ = stub_miles
    for path in (retry.DATA_SOURCE_PATH, retry.ROLLOUT_FUNCTION_PATH):
        module_name, _, name = path.rpartition(".")
        assert module_name == module.__name__ and hasattr(module, name)
    source = module.SecRLEnvRetryDataSource(_args())
    assert len(source.get_samples(TARGET + 1)) == TARGET


def test_rollout_function_reports_the_retry_counters(stub_miles):
    module, _ = stub_miles
    args = _args()
    module.SecRLEnvRetryDataSource(args)
    output = SimpleNamespace(metrics={"rollout/x": 1.0})
    fn = module.SecRLEnvRolloutFn(SimpleNamespace(args=args, outcome=output))
    result = asyncio.run(fn._call_train(SimpleNamespace(rollout_id=3)))
    assert result.metrics["rl/secrlenv/replacement_attempts"] == 0.0
    assert result.metrics["rollout/x"] == 1.0


def test_rollout_function_drains_before_a_terminal_error(stub_miles, monkeypatch):
    module, calls = stub_miles
    monkeypatch.setattr(module, "_require_no_episode_residue", lambda: calls.append("clean"))
    fn = module.SecRLEnvRolloutFn(SimpleNamespace(
        args=_args(), outcome=SecRLEnvReplacementExhausted("budget")))
    with pytest.raises(SecRLEnvReplacementExhausted):
        asyncio.run(fn._call_train(SimpleNamespace(rollout_id=7)))
    assert calls == [("abort", set(), 7), "clean"]


# ------------------------------------------------------------- launcher
def test_launcher_ports_preflight_accepts_a_secrlenv_run():
    """Before P0-1.7b.4-Q1 the ports spec refused the group check here; the
    expected spec hash is the one the island derives from the bound filter."""

    from test_rl_launcher import _args as launcher_args

    from yeto.launcher import _prepare_rl_args

    args = launcher_args([
        "--reward-function", "yeto.rl.harness.codex.reward:reward_func",
        "--custom-generate-function-path", SECRLENV_GENERATE,
        "--custom-agent-function-path", SECRLENV_AGENT,
        "--use-session-server", "--tito-model", "org/model",
    ])
    _prepare_rl_args(args)
    assert args.dynamic_sampling_filter_path == SECRLENV_GROUP_FILTER
    assert args.rl_expected_algorithm_sha256 == _secrlenv_spec().sha256()
    # Regression: computed before the contract bound the filter, the hash was
    # that of the filterless spec and the island refused the run.
    assert args.rl_expected_algorithm_sha256 != AlgorithmSpec(advantage_estimator="grpo").sha256()
    assert args.over_sampling_batch_size == args.rollout_batch_size + 1


def _miles_like_round(source, *, stuck):
    """Stands in for Miles' generate_rollout_async: group tasks run as
    ``generate_and_rm_group``; an exception inside a task is swallowed (the
    group only counts as failed) and a dropped group triggers a refill."""

    from yeto.rl.harness.codex.reward import remember_terminal_error

    async def generate_and_rm_group(args, group, fail):
        if fail:
            error = UntrustedOutcome("forged outcome with episode payload")
            remember_terminal_error(args, error)  # what reward_func/generate do
            raise error
        await asyncio.sleep(3600)

    async def driver(args):
        groups = source.get_samples(TARGET + 1)
        pendings = {asyncio.create_task(generate_and_rm_group(args, g, i == 0))
                    for i, g in enumerate(groups)}
        stuck.extend(pendings)
        done, pendings = await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                task.result()
            except Exception:  # noqa: BLE001 - Miles: logged, group counted as failed
                continue
        source.get_samples(TARGET + 1)  # refill for the dropped group: no retry queued

    return driver


def test_rollout_function_raises_the_first_terminal_error_and_stops_group_tasks(
        stub_miles, monkeypatch):
    module, calls = stub_miles
    monkeypatch.setattr(module, "_require_no_episode_residue", lambda: calls.append("clean"))
    args = _args()
    source = module.SecRLEnvRetryDataSource(args)
    stuck = []
    fn = module.SecRLEnvRolloutFn(SimpleNamespace(
        args=args, outcome=_miles_like_round(source, stuck=stuck)))

    async def run():
        with pytest.raises(SecRLEnvUntrustedEvidence) as raised:
            await fn._call_train(SimpleNamespace(rollout_id=4))
        # checked inside the loop: asyncio.run would cancel leftovers on exit
        assert len(stuck) == TARGET and all(task.done() for task in stuck)
        return raised.value

    error = asyncio.run(run())
    assert "payload" not in str(error)  # sanitized first error, not the refill error
    assert calls == [("abort", set(), 4), "clean"]
    assert sum(task.cancelled() for task in stuck) == TARGET - 1


def test_round_start_forgets_the_previous_terminal_error():
    from yeto.rl.harness.codex.reward import TERMINAL_ERROR_ATTR

    args = _args()
    source = _RetrySource(args)
    setattr(args, TERMINAL_ERROR_ATTR, UntrustedOutcome("old"))
    source.get_samples(TARGET + 1)
    assert getattr(args, TERMINAL_ERROR_ATTR) is None
    fallback = SecRLEnvReplacementExhausted("x")
    assert retry.first_terminal_error(args, fallback) is fallback
