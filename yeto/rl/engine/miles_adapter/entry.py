"""``--rl-engine ports`` composition root (tasks 5.1, design D2).

``run_ports_island`` is what ``yeto.rl.learner.run_miles`` calls on the ports
path, after the legacy-shared validation has produced the parsed upstream Miles
namespace (with the ``yeto_rl_*`` attributes the bridges read):

1. refuse a Miles checkout without ``run_plugin`` (before any component);
2. upstream init exactly as ``train.py`` does it -- ``init_orchestration_script``,
   ``create_rollout_components``, ``create_training_models`` -- inside one
   upstream ``Disposer`` on the island's persistent loop;
3. :func:`compose_island`: the adapter ports + sync session + ``IslandDriver``;
4. ``driver.run()``, then teardown through the disposer.

:func:`compose_island` takes the upstream objects as arguments so the CPU
integration test drives the real adapter classes over stubs.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from typing import Any

from ..algorithm import BOUNDED_NONZERO_STD_FILTER, STOCK_NONZERO_STD_FILTER, AlgorithmSpec
from ..capabilities import R0_MECHANISMS, EngineCapabilities, ExecutionCapabilities
from . import LoopRunner

ENGINE_NAME = "miles-upstream"


def runtime_fingerprint(launch: Any, miles_commit: str) -> str:
    """Identity of the engine runtime this island drives."""

    payload = {"miles_commit": miles_commit, "argv": list(launch.argv)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


# Declared beyond R0: "dimension:name" -> evidence that the mechanism takes
# effect on GPU (declaration policy, rl-infra-spec alignment §7b; may be
# overridden by the user). One entry per mechanism, added in its own commit.
_E1A = "openspec/changes/rl-algo-mismatch-correction/evidence"
_E1B = "openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1"
_E1B_B = "openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1b"
_E1B_C = "openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1c"
_E2A = "openspec/changes/rl-algo-seq-and-adv/evidence/g1"
MILES_DECLARED: dict[str, str] = {
    "corrections:tis": f"{_E1A}/2026-09-29-g1c + 2026-09-29-trigger (tis_clipfrac > 0)",
    "corrections:opsm": (
        f"{_E1A}/2026-09-29-trigger (opsm_clipfrac > 0, optimizer_steps 2); the OPSM "
        "dimension that every source-specific OPSM mechanism also requires"
    ),
    "corrections:opsm_trainer": f"{_E1A}/2026-09-29-trigger (opsm_clipfrac > 0)",
    "features:maxrl": f"{_E2A}/attempt4 (maxrl)",
    "features:mapo": f"{_E2A}/attempt4 (mapo)",
    "loss_aggregations:constant": f"{_E1B}/g1_report_v2.json drgrpo (constant-denominator aggregation)",
    "kl_placements:loss": f"{_E1B}/g1_report_v2.json kl_k3 (kl_loss 0 / 0.00079 / 0.00082)",
    "features:kl_loss_ref_model": f"{_E1B}/g1_report_v2.json kl_k3 (ref model loaded, kl_loss > 0)",
    "features:entropy_bonus": f"{_E1B}/g1_report_v2.json entropy (entropy_loss 0.30/0.38/0.45)",
    "reward_postprocessors:custom_reward_postprocess": f"{_E1B}/g1_report_v2.json overlong_penalty (dispatcher shaped 4/7/21 of 32 samples)",
    "features:overlong_penalty": f"{_E1B}/g1_report_v2.json overlong_penalty (shaped_samples 4/7/21)",
    "advantage_estimators:gspo": f"{_E2A}/attempt6 gspo_s2 (optimizer_steps 2: second-step clipfrac 0.1875/0.5/0.5; steps 1: 0)",
    "advantage_estimators:reinforce_plus_plus": (
        f"{_E2A}/attempt6 rpp; plan.md 'Attempt 6 addenda': rollout/advantages mean "
        "0.0155/0.0742/-0.0217 (non-zero where GRPO's group-normalized mean is ~0), "
        "ref_log_probs scored every round, diverging from round 1 (reward KL active)"
    ),
    "advantage_estimators:reinforce_plus_plus_baseline": (
        f"{_E2A}/attempt6 rpp_baseline; plan.md 'Attempt 6 addenda': rollout/advantages "
        "mean 0.0374/0.1242/0.1093 (vs ~0 for GRPO), ref_log_probs scored, diverging from "
        "round 1"
    ),
    "features:gdpo": f"{_E2A}/attempt6 gdpo (per-round nonzero_advantages 32/24/32 match the dispatcher)",
    "corrections:mismatch_observe": f"{_E1A}/2026-09-29-g1b observe + g2-observe (observation only; weights constant 1)",
    "corrections:icepop": f"{_E1A}/2026-09-29-trigger icepop [0.99,1.01] (masked tis_clipfrac 0.192/0.225/0.267)",
    "corrections:mis_mask": f"{_E1A}/2026-09-29-trigger mis-mask token [0.99,1.01] (mask fraction 0.192/0.225/0.267)",
    "features:eps_clip": f"{_E1B_B}/plan.md run A-r1 (eps_clip 0.001 / eps_clip_high 0.002, test values to trigger the clip, not recommendations): step-2 pg_clipfrac 0.1046/0.1107/0.1046",
    "features:no_grpo_std_normalization": f"{_E1B_C}/g1c_report.json no_std (isolated paired step 1: grad_norm 0.2428 vs baseline 0.6349; effective, paired_valid; analyze.py 12b592f)",
}


# Declarations whose evidence holds only for specific Miles pins (exact
# commits; a new pin must be re-verified before it is added here).
MILES_DECLARED_PINS: dict[str, frozenset[str]] = {}


def declared_by_dimension(miles_commit: str | None = None) -> dict[str, set[str]]:
    """R0 mechanism sets plus :data:`MILES_DECLARED`, per dimension.

    An entry of :data:`MILES_DECLARED_PINS` is declared only when the Miles
    pin (``miles_commit``, default ``yeto.rl.MILES_NEXT_COMMIT``) is one of
    its verified commits.
    """

    from ..capabilities import R0_MECHANISMS

    if miles_commit is None:
        from yeto.rl import MILES_NEXT_COMMIT as miles_commit
    out = {dim: set(names) for dim, names in R0_MECHANISMS.items()}
    out["advantage_estimators"] = {"grpo"}
    for mechanism in MILES_DECLARED:
        pins = MILES_DECLARED_PINS.get(mechanism)
        if pins is not None and miles_commit not in pins:
            continue
        dimension, name = mechanism.split(":", 1)
        out.setdefault(dimension, set()).add(name)
    return out


def miles_capabilities(
    fingerprint: str, *, unverified_mechanisms=()
) -> EngineCapabilities:
    """R0 ports capabilities (spec rl-engine-selection support matrix).

    Mechanism dimensions are the R0 set (``EngineCapabilities`` defaults:
    grpo, policy_loss, default aggregation, KL none/reward, no correction,
    the nonzero-std filters); a follow-up algorithm change adds a mechanism
    here only after its single-GPU smoke (G1) passed. ``execution``
    (rl-algorithm-capabilities D4, alignment A1): no critic, serial
    colocated produces policy age 0, and upstream SGLang rollouts return
    logprobs (``sglang_rollout.py`` ``return_logprob=True``).
    ``unverified_mechanisms`` is the single-island D11 allowance.
    """

    capabilities = EngineCapabilities(
        engine=ENGINE_NAME,
        runtime_fingerprint=fingerprint,
        parameter_layouts={"lora"},
        placements={"colocated"},
        dynamic_sampling_filters={BOUNDED_NONZERO_STD_FILTER, STOCK_NONZERO_STD_FILTER},
        execution_modes={"colocated-serial"},
        execution=ExecutionCapabilities(
            critic=False, max_policy_staleness=0, rollout_logprobs=True
        ),
        **declared_by_dimension(),
    )
    if unverified_mechanisms:
        capabilities = capabilities.with_unverified(unverified_mechanisms)
    return capabilities


def receipt_role_family(algorithm: AlgorithmSpec) -> str:
    """``LocalStepReceipt.algorithm``: the TRAINING ROLE FAMILY, not the estimator.

    It must equal ``ParameterLayout.algorithm`` (``local_learner.py`` checks
    both; the layout hash covers it), whose families are grpo / sao. Every
    critic-free estimator (grpo, gspo, reinforce_plus_plus[_baseline]) trains
    the single actor role -> ``"grpo"``; the estimator itself is identified by
    ``algorithm_spec_sha256``. Critic estimators (ppo) have no family in the
    layout contract and are refused.
    """
    from ..algorithm import CRITIC_ESTIMATORS

    estimator = algorithm.advantage_estimator
    if estimator in CRITIC_ESTIMATORS:
        raise ValueError(
            f"advantage estimator {estimator!r} needs a critic role family, which the "
            "receipt/layout contract does not define"
        )
    return "grpo"


def with_partitioned_serial(capabilities: EngineCapabilities) -> EngineCapabilities:
    """Infra declaration (rl-infra-spec 2.1/2.2): the fixed-partition placement and
    the partitioned-serial driver mode are implemented on the ports path.

    Kept apart from :func:`miles_capabilities` (the algorithm/R0 declaration).
    Declaration is not certification: the #66 attestation for a runtime
    fingerprint is issued only after the 2.1/2.2 GPU acceptance.
    """
    import dataclasses

    return dataclasses.replace(
        capabilities,
        placements=capabilities.placements | {"fixed-partition"},
        execution_modes=capabilities.execution_modes | {"partitioned-serial"},
        partitioned_driver=True,
    )


def outer_protocol(miles_args: Any, *, yeto_policy_sync: bool) -> str:
    if not yeto_policy_sync:
        return "none"
    return (
        "decoupled"
        if getattr(miles_args, "yeto_rl_sync_preset", "strict-avg") == "decoupled"
        else "strict-avg"
    )


EXPECTED_ALGORITHM_ENV = "YETO_RL_EXPECTED_ALGORITHM_SHA256"


def expected_algorithm_sha256(miles_args: Any, environ: Any = None) -> str | None:
    """The AlgorithmSpec hash the LAUNCHER intended (external to this process).

    Sources: ``miles_args.yeto_rl_expected_algorithm_sha256`` (learner flag
    ``--rl-expected-algorithm-sha256``) or the ``YETO_RL_EXPECTED_ALGORITHM_SHA256``
    environment variable. Comparing it with the runtime spec is what makes the
    A1 check non-circular (review F2).
    """
    environ = os.environ if environ is None else environ
    value = getattr(miles_args, "yeto_rl_expected_algorithm_sha256", None) or environ.get(
        EXPECTED_ALGORITHM_ENV
    )
    return str(value) if value else None


def execution_profile_for(
    miles_args: Any,
    launch: Any,
    algorithm: AlgorithmSpec,
    *,
    yeto_policy_sync: bool,
    expected_sha256: str | None = None,
):
    """The run's :class:`ExecutionProfile`, bound to the EXTERNAL algorithm hash.

    colocated placement -> ``colocated-serial``; fixed partition ->
    ``partitioned-serial`` (no overlap is certified, task 2.3). The profile is
    bound to ``expected_sha256`` (launcher-provided); :func:`preflight` then
    compares it with the runtime ``AlgorithmSpec``. Without an external hash a
    partitioned run is refused; a colocated (R0) run binds to the runtime spec
    and records that the hash source was the runtime.
    """
    from ..execution_profile import ExecutionProfile, ProfileError

    mode = "colocated-serial" if launch.placement.kind == "colocated" else "partitioned-serial"
    if expected_sha256 is None:
        if mode != "colocated-serial":
            raise ProfileError(
                f"{mode} needs the launcher's expected AlgorithmSpec hash "
                f"(--rl-expected-algorithm-sha256 or {EXPECTED_ALGORITHM_ENV}); refusing"
            )
        expected_sha256, source = algorithm.sha256(), "runtime"
    else:
        source = "launcher"
    return ExecutionProfile(
        name=f"miles-lora-{mode}",
        execution_mode=mode,
        outer_protocol=outer_protocol(miles_args, yeto_policy_sync=yeto_policy_sync),
        groups_per_batch=int(miles_args.rollout_batch_size),
        samples_per_group=int(miles_args.n_samples_per_prompt),
        optimizer_steps_per_round=int(getattr(miles_args, "num_steps_per_rollout", 1) or 1),
        algorithm_spec_sha256=expected_sha256,
        extra={"algorithm_hash_source": source},
    )


def preflight(profile: Any, algorithm: AlgorithmSpec, capabilities: EngineCapabilities) -> None:
    """A1: the launcher-bound profile agrees with the runtime AlgorithmSpec and the
    declared capabilities, before any GPU process exists (before connect_island_ray)."""
    from ..execution_profile import check_algorithm_contract

    check_algorithm_contract(profile, algorithm)
    placement = "colocated" if profile.execution_mode == "colocated-serial" else "fixed-partition"
    capabilities.check(
        layout="lora",
        placement=placement,
        execution_mode=profile.execution_mode,
        algorithm=algorithm,
        max_policy_age=profile.max_policy_age,
    )


def build_sync(miles_args: Any, *, yeto_policy_sync: bool) -> tuple[Any, Any]:
    """(sync session, progress store) for the preset the learner selected."""

    from ..bridges import (
        DecoupledIslandProgress,
        DecoupledSync,
        LocalOnlySync,
        StrictAvgSync,
        StrictIslandProgress,
    )

    if not yeto_policy_sync:
        return LocalOnlySync(int(miles_args.num_rollout)), None
    if getattr(miles_args, "yeto_rl_sync_preset", "strict-avg") == "decoupled":
        return DecoupledSync(miles_args), DecoupledIslandProgress(miles_args)
    progress = StrictIslandProgress(miles_args)
    return StrictAvgSync(miles_args.yeto_rl_bridge_config, progress=progress), progress


def compose_island(
    *,
    miles_args: Any,
    launch: Any,
    algorithm: AlgorithmSpec,
    inference_controller: Any,
    rollout_executor: Any,
    actor_model: Any,
    learner_id: int,
    base_model_revision: str,
    lora_config_hash: str,
    layout_hash: str,
    sync: Any,
    progress: Any,
    metadata: Any,
    capabilities: EngineCapabilities,
    runner: LoopRunner,
    events: Any = None,
    evaluate: Callable[[int], Any] | None = None,
    eval_interval: int | None = None,
    update_weights: Callable[..., Any] | None = None,
    release_refs: Callable[[Any, Any], None] | None = None,
    flatten_checksums: Callable[[Any], list[dict[str, Any]]] | None = None,
    verify_engine_checksums: bool = True,
    placement: Any = None,
    profile: Any = None,
    observe: bool = False,
):
    """Wire the adapter ports into an ``IslandDriver`` (no upstream imports)."""

    from yeto.rl.core import parse_policy_snapshot_token

    from ..driver import DriverError, EventTape, IslandDriver
    from .placement import MilesPlacement
    from .publish import MilesPublisher
    from .rollout import MilesRolloutPool
    from .state import MilesPolicyState
    from .trainer import MilesTrainerGroup

    holder: dict[str, IslandDriver] = {}

    def expected_policy() -> tuple[int, str]:
        driver = holder["driver"]
        if driver.expected_token is None:
            raise DriverError("rollout requested before any complete publication")
        return parse_policy_snapshot_token(driver.expected_token)

    policy_state = MilesPolicyState(
        actor_model=actor_model,
        base_model_revision=base_model_revision,
        config_hash=lora_config_hash,
        expected_layout_hash=layout_hash,
        runner=runner,
    )
    driver = IslandDriver(
        learner_id=learner_id,
        rollout=MilesRolloutPool(
            inference_controller=inference_controller,
            rollout_executor=rollout_executor,
            metadata=metadata,
            expected_policy=expected_policy,
            runner=runner,
            args=miles_args,
        ),
        trainer=MilesTrainerGroup(
            args=miles_args,
            actor_model=actor_model,
            learner_id=learner_id,
            learner_generation=0,
            parameter_layout_hash=lambda: layout_hash,
            algorithm=receipt_role_family(algorithm),
            spec=algorithm,
            release_refs=release_refs,
            runner=runner,
        ),
        policy_state=policy_state,
        publisher=MilesPublisher(
            args=miles_args,
            actor_model=actor_model,
            rollout_executor=rollout_executor,
            inference_controller=inference_controller,
            export_trainer_state=policy_state.export,
            verify_engine_checksums=verify_engine_checksums,
            update_weights=update_weights,
            flatten_checksums=flatten_checksums,
            runner=runner,
        ),
        placement=placement
        or MilesPlacement.from_parsed_args(launch.placement, miles_args),
        capabilities=capabilities,
        algorithm=algorithm,
        sync=sync,
        events=events
        or EventTape(miles_args.yeto_rl_event_tape, learner_id, args=miles_args),
        progress=progress,
        evaluate=evaluate,
        eval_interval=eval_interval,
        profile=profile,
        observe=observe,
    )
    holder["driver"] = driver
    return driver


def selection_event(
    *,
    launch: Any,
    algorithm: AlgorithmSpec,
    miles_commit: str,
    unverified_mechanisms: Any = (),
    outer_sync: bool | None = None,
) -> dict[str, Any]:
    """Run event carrying the algorithm identity (D9) and its provenance.

    ``rl/algorithm_absorbed_flags`` lists flags absorbed from extra argv
    (D3); ``rl/unverified_mechanisms`` the D11 allowances (an artifact of
    such a run contains unverified mechanisms).
    """

    event = {
        "event": "rl_engine_selected",
        "rl_engine": "ports",
        "miles_commit": miles_commit,
        "rl/algorithm_spec_sha256": algorithm.sha256(),
        "rl/algorithm_spec": algorithm.canonical_json(),
        "rl/algorithm_absorbed_flags": dict(getattr(launch, "absorbed_flags", None) or {}),
        "placement": launch.placement.kind,
    }
    if unverified_mechanisms:
        event["rl/unverified_mechanisms"] = sorted(unverified_mechanisms)
        event["rl/contains_unverified_mechanisms"] = True
    # Always recorded; an unknown mode (None) is the R0 default: outer sync on.
    event["rl/outer_sync"] = True if outer_sync is None else bool(outer_sync)
    return event


def connect_island_ray(*, environ=None, ray_module=None) -> str | None:
    """Connect the driver to the island's own Ray and pin every actor to it.

    A SkyPilot machine runs two Ray instances: the island's (6379, started by
    the island task) and SkyPilot's runtime Ray (6380).  The launcher gives
    the driver ``RAY_ADDRESS``, but Ray workers inherit the raylet's
    environment, not the driver's.  Legacy Miles only resolved the address in
    the driver (``compute_ray_pin_head_options`` ran in placement_group.py);
    upstream Miles calls ``ray.util.state.list_nodes()`` inside the
    ``RayWorkerManager`` actor, which then sees both instances and fails with
    "Found multiple active Ray instances".  A job-level ``runtime_env``
    ``env_vars`` entry is merged into every actor and task the job creates,
    so they resolve the same address as the driver.  ``PYTHONPATH`` travels
    with it so actors import the pinned Miles checkout, not the image's.
    """

    environ = os.environ if environ is None else environ
    address = environ.get("RAY_ADDRESS")
    if not address:
        return None
    if ray_module is None:
        import ray as ray_module
    if ray_module.is_initialized():
        raise RuntimeError(
            "Ray was initialized before the ports island pinned RAY_ADDRESS; "
            "its actors could resolve the wrong Ray instance"
        )
    env_vars = {"RAY_ADDRESS": address}
    from yeto.rl.event_echo import ECHO_ENV

    if environ.get(ECHO_ENV):
        env_vars[ECHO_ENV] = environ[ECHO_ENV]  # Ray workers echo their tape writes too
    if environ.get("PYTHONPATH"):
        env_vars["PYTHONPATH"] = environ["PYTHONPATH"]
    ray_module.init(address=address, runtime_env={"env_vars": env_vars})
    return address


def run_ports_island(
    miles_args: Any,
    launch: Any,
    algorithm: AlgorithmSpec,
    *,
    learner_id: int,
    base_model_revision: str,
    lora_config_hash: str,
    layout_hash: str,
    yeto_policy_sync: bool,
):
    """Run one ports-path island to completion on real upstream Miles."""

    from yeto.rl import MILES_NEXT_COMMIT
    from yeto.rl.miles import _append_rl_event

    from .state import require_run_plugin

    require_run_plugin()  # before any upstream component or model exists
    capabilities = with_partitioned_serial(
        miles_capabilities(
            runtime_fingerprint(launch, MILES_NEXT_COMMIT),
            unverified_mechanisms=getattr(miles_args, "yeto_rl_unverified_mechanisms", ()),
        )
    )
    profile = execution_profile_for(
        miles_args,
        launch,
        algorithm,
        yeto_policy_sync=yeto_policy_sync,
        expected_sha256=expected_algorithm_sha256(miles_args),
    )
    preflight(profile, algorithm, capabilities)  # A1: before any GPU process
    connect_island_ray()

    from miles.ray.placement_group import create_rollout_components, create_training_models
    from miles.ray.rollout.eval_dispatch import EvalDispatcher
    from miles.utils.async_utils import Disposer
    from miles.utils.orchestration_utils import init_orchestration_script

    from .rollout import RayMetadataSink

    _append_rl_event(
        miles_args,
        selection_event(
            launch=launch,
            algorithm=algorithm,
            miles_commit=MILES_NEXT_COMMIT,
            unverified_mechanisms=getattr(miles_args, "yeto_rl_unverified_mechanisms", ()),
            outer_sync=getattr(miles_args, "yeto_rl_outer_sync", None),
        ),
    )
    runner = LoopRunner()
    disposer = Disposer()

    async def init():
        await disposer.__aenter__()
        init_orchestration_script(miles_args, disposer=disposer)
        controller, executor, _ = await create_rollout_components(miles_args)
        disposer.add(controller, executor)
        actor, critic = await create_training_models(miles_args, executor)
        if critic is not None:
            raise RuntimeError("the ports engine does not drive a critic")
        disposer.add(actor)
        dispatcher = EvalDispatcher(miles_args, actor, executor)
        disposer.add(dispatcher.drain)
        return controller, executor, actor, dispatcher

    error: BaseException | None = None
    try:
        controller, executor, actor, dispatcher = runner.run(init())
        if not callable(getattr(actor, "run_plugin", None)):
            raise RuntimeError("upstream actor group exposes no run_plugin")

        async def evaluate(rollout_id: int) -> dict[str, float]:
            await controller.prepare_eval()
            await dispatcher.dispatch(rollout_id)
            return {}

        sync, progress = build_sync(miles_args, yeto_policy_sync=yeto_policy_sync)
        driver = compose_island(
            miles_args=miles_args,
            launch=launch,
            algorithm=algorithm,
            inference_controller=controller,
            rollout_executor=executor,
            actor_model=actor,
            learner_id=learner_id,
            base_model_revision=base_model_revision,
            lora_config_hash=lora_config_hash,
            layout_hash=layout_hash,
            sync=sync,
            progress=progress,
            metadata=RayMetadataSink(),
            capabilities=capabilities,
            runner=runner,
            evaluate=lambda rollout_id: runner.run(evaluate(rollout_id)),
            eval_interval=getattr(miles_args, "eval_interval", None),
            profile=profile,
            observe=bool(getattr(miles_args, "yeto_rl_observe_timeline", False)),
        )
        return driver.run()
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            runner.run(
                disposer.__aexit__(
                    None if error is None else type(error),
                    error,
                    None if error is None else error.__traceback__,
                )
            )
        finally:
            runner.close()
