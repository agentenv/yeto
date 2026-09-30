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
from pathlib import Path
import os
import time
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


def ports_runtime_fingerprint(launch: Any) -> str:
    """The island's runtime fingerprint (attestation ``runtime_fingerprint``,
    ``rl_driver_start``): pinned Miles commit + the full Miles argv. The ONE
    function both ``run_ports_island`` and ``--rl-print-attestation-fingerprint``
    call, so the printed value is the value the island will check."""
    from yeto.rl import MILES_NEXT_COMMIT

    return runtime_fingerprint(launch, MILES_NEXT_COMMIT)


# Declared beyond R0: "dimension:name" -> evidence that the mechanism takes
# effect on GPU (declaration policy, rl-infra-spec alignment §7b; may be
# overridden by the user). One entry per mechanism, added in its own commit.
_E1A = "openspec/changes/rl-algo-mismatch-correction/evidence"
_E1B = "openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1"
_E1B_B = "openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1b"
_E1B_C = "openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1c"
_E1B_G1F = "openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1f"
_E1B_G1H = "openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1h"
_E1B_G1I = "openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1i"
_E1B_G1J = "openspec/changes/rl-algo-grpo-knobs/evidence/2026-09-29-algo1b-g1j"
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
    "loss_aggregations:token": (
        f"{_E1B_G1F} (branch algo-1b-token 604078e..9f6f8a6, YETO_SHA 8d30ad2): paired "
        "step 1 (raw_reward 0.90625 both) grad_norm 0.4310 vs baseline 0.4867; ONLY on "
        "Miles 0af62f4d+ (LoRA bridge sets calculate_per_token_loss); image "
        "sha256:c6f5455c... inferred from the pin and verified from source by the main "
        "agent (launch.log printed no digest)"
    ),
    "features:over_sampling": (
        f"{_E1B_G1H} (branch algo-1b-os d53397d, conclusion rewritten 3fec259): the "
        "pre-registered criteria (a)(b) are literally met but discriminate weakly; the "
        "decisive evidence is the over_sampling arm's rollout 1 (submitted 8, aborted 4, "
        "filtered 0 -- inferred afterwards from the rollout_meta_hook formula, one round "
        "only) and the two arms' parameter tables differing only in "
        "over_sampling_batch_size; strong evidence awaits a rerun once Miles records the "
        "batch size of every submission. Miles 0af62f4d only"
    ),
    "features:overlong_filter": (
        f"{_E1B_G1I} (branch algo-1b-os bf9f914): paired (step-1 raw_reward 0.65625 both "
        "arms, truncation rate 0.5); (a) of_on filtered_samples 16/16/31, (b) of_off None, "
        "(c) step-1 grad_norm 0.5647 vs 0.6329; round 3 (31/32 filtered) raised no "
        "zero-gradient false alarm. Requires the 1b hook (integ-decl 21912fe or later: "
        "rollout_meta_hook applies the sample filter before recording trained groups). "
        "Miles 0af62f4d only"
    ),
    "features:clip_higher": (
        f"{_E1B_G1J} (branch algo-1b 53cb477): 6 groups x 3 steps, seed 17; step-1 "
        "grad_norm bit-identical in both arms (1.1293506622314453), round-1 step-3 "
        "grad_norm A 0.5642 vs B 0.6572 -- pre-registered criterion met. Nature of the "
        "evidence: a deterministic same-seed reproduction of the post-hoc g1e observation "
        "(step-2 grad_norm differs; g1j's first two steps are bit-identical to g1e), not "
        "an independent confirmation; the attribution holds (the arms differ only in "
        "eps_clip_high, step 1 bit-identical). RAN ON Miles "
        "0394715, not the pinned 0af62f4d: transferred because `git diff 0394715..0af62f4d "
        "-- miles` (13 files, checked by the main agent and ALGO-CAP) touches no "
        "loss/policy file and leaves the clip path unchanged -- a code-diff argument, not "
        "a run on 0af62f4d. Open: suspected pg_clipfrac vs loss inconsistency "
        "(2026-09-29-clipfrac-offline/report.md). Miles 0af62f4d only"
    ),
}


# Declarations whose evidence holds only for specific Miles pins (exact
# commits; a new pin must be re-verified before it is added here).
# 5c1b49eb = 0af62f4d + the 2b loss variants: carried over by code diff, not
# by a rerun -- `git diff 0af62f4d..5c1b49eb -- miles` touches only
# loss_hub/{losses,math_utils}.py and arguments.py; with the default
# --policy-loss-variant policy_loss the loss path calls the same
# compute_policy_loss with the same arguments, need_full_log_probs is
# unchanged, and the new flags only add parser entries/validation.
# 2f23a0fc = 5c1b49eb + F-R1, same basis: `git diff --stat 5c1b49eb..2f23a0fc
# -- miles` touches only miles/ray/{placement_group,rollout/inference_controller,
# specs/inference}.py, miles/utils/workers/* and the --yeto-placement-map help
# text in arguments.py -- no loss_hub/backends file; without rollout_cells /
# deferred cells the startup path starts the same cells (evidence
# openspec/changes/rl-infra-spec/evidence/2026-09-30-img-2f23a0f).
_PINS_0AF62F4D_PLUS = frozenset({
    "0af62f4d48ed6a5b185c257578d8f7e22312aa87",
    "5c1b49ebccbc7508c1d9ef89eacc2db3e448b6ba",
    "2f23a0fca9b80f6a7300da401703c343014b03c0",
})
MILES_DECLARED_PINS: dict[str, frozenset[str]] = {
    # before 0af62f4d the LoRA bridge ignored calculate_per_token_loss (g1c:
    # grad_norm bit-identical to the baseline)
    "loss_aggregations:token": _PINS_0AF62F4D_PLUS,
    "features:over_sampling": _PINS_0AF62F4D_PLUS,
    "features:overlong_filter": _PINS_0AF62F4D_PLUS,
    "features:clip_higher": _PINS_0AF62F4D_PLUS,
}


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
        # partitioned-overlap = eval||train/outer_sync only (2.3, overlap.py).
        execution_modes=capabilities.execution_modes
        | {"partitioned-serial", "partitioned-overlap"},
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
    ``partitioned-serial``, or ``partitioned-overlap`` (eval||train/outer_sync
    only, task 2.3) when ``yeto_rl_overlap_eval`` is set. The profile is
    bound to ``expected_sha256`` (launcher-provided); :func:`preflight` then
    compares it with the runtime ``AlgorithmSpec``. Without an external hash a
    partitioned run is refused; a colocated (R0) run binds to the runtime spec
    and records that the hash source was the runtime.
    """
    from ..execution_profile import ExecutionProfile, ProfileError, check_overlap_eval

    from ..overlap import IMPLEMENTED_OVERLAP

    mode = "colocated-serial" if launch.placement.kind == "colocated" else "partitioned-serial"
    overlap = frozenset()
    if getattr(miles_args, "yeto_rl_overlap_eval", False):
        check_overlap_eval(
            placement_kind="colocated" if mode == "colocated-serial" else "fixed-partition",
            eval_uses_snapshots=bool(getattr(miles_args, "eval_uses_snapshots", False)),
            eval_interval=getattr(miles_args, "eval_interval", None),
        )
        mode, overlap = "partitioned-overlap", IMPLEMENTED_OVERLAP
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
        allowed_overlap=overlap,
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
    elastic: Any = None,
    evaluate_start: Callable[[int], Any] | None = None,
):
    """Wire the adapter ports into an ``IslandDriver`` (no upstream imports).

    ``elastic`` (:class:`.elastic_wiring.ElasticWiring`, rl-infra-spec 3.x):
    declared rollout cells for the E1 membership verbs, the reconfiguration
    controller and the batch ledger. The controller reconciles the fork's
    membership epoch with its journal before the driver runs. None keeps the
    island unchanged.
    """

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
            **(
                {"declared_cells": resolve_declared_cells(
                    inference_controller, runner, elastic.declared_cells),
                 "track_timeout_s": elastic.track_timeout_s,
                 "tool_wait_board": elastic.tool_wait_board}
                if elastic is not None
                else {}
            ),
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
        **(
            {"controller": elastic.controller, "ledger": elastic.ledger}
            if elastic is not None
            else {}
        ),
        **({"evaluate_start": evaluate_start} if evaluate_start is not None else {}),
    )
    if elastic is not None:
        from .elastic_placement import ElasticPlacement

        driver.placement = ElasticPlacement(
            driver.placement, pool_gpus=elastic.pool_gpus,
            epoch=elastic.controller.journal.epochs.config_epoch,
        )
        epochs = elastic.controller.journal.epochs
        driver.config_epoch = epochs.config_epoch
        if epochs.config_epoch > 0:
            # restart after a commit: the journal's config is authoritative
            committed = elastic.controller.configs[epochs.config_id].placement or {}
            if committed.get("rollout"):
                driver.placement.restore_committed(tuple(committed["rollout"]),
                                                   epoch=epochs.config_epoch)
        elastic.controller.open(driver.rollout)
        _wire_trainer_rebuild(driver, elastic=elastic, miles_args=miles_args, algorithm=algorithm,
                              actor_model=actor_model, rollout_executor=rollout_executor,
                              runner=runner, base_model_revision=base_model_revision)
        if (getattr(miles_args, "yeto_rl_elastic", None) or {}).get("trainer_edges"):
            _wire_trainer_edges(driver, elastic=elastic, miles_args=miles_args, launch=launch,
                                algorithm=algorithm, actor_model=actor_model,
                                rollout_executor=rollout_executor, runner=runner,
                                base_model_revision=base_model_revision)
    holder["driver"] = driver
    return driver


def _wire_trainer_rebuild(driver, *, elastic, miles_args, algorithm, actor_model,
                          rollout_executor, runner, base_model_revision) -> None:
    """4.4: give the controller a same-shape trainer rebuilder when the actor is
    the swappable proxy (a rebuild still has to be requested explicitly)."""
    from .rebuild_wiring import make_trainer_rebuilder
    from .trainer_rebuild import SwappableActor, rebuild_preconditions, rebuild_same_shape

    if not isinstance(actor_model, SwappableActor) or not hasattr(elastic.controller,
                                                                  "trainer_rebuilder"):
        return
    if rebuild_preconditions(miles_args):
        # review F3: not a same-shape rebuild path (e.g. --load given): leave the
        # rebuilder unwired, so a request is rejected at plan time
        return
    ref_load = getattr(miles_args, "ref_load", None)
    elastic.controller.trainer_rebuilder = make_trainer_rebuilder(
        trainer=driver.trainer,
        rollout=driver.rollout,
        ledger=elastic.ledger,
        algorithm=algorithm,
        backend_fingerprint=elastic.controller.runtime_fingerprint or "",
        cut_root=str(elastic.controller.state_dir / "cuts"),
        global_batch_size=int(miles_args.global_batch_size),
        rebuild_same_shape=lambda *, restore: rebuild_same_shape(
            driver.trainer, args=miles_args, rollout_executor=rollout_executor,
            actor=actor_model, run=runner.run, restore=restore, rollout=driver.rollout,
            # YETO_RL_TEST_INJECT_REBUILD_FAIL is applied by trainer_rebuild's default
            # rebuild (cut_injection.rebuild_fail_count; one implementation).
        ),
        ref_model=(None if not ref_load
                   else {"ref_load": str(ref_load), "base_model_revision": base_model_revision}),
        preconditions=lambda: rebuild_preconditions(miles_args),
    )


def _role_map(request: Any) -> dict[str, Any] | None:
    """The role -> logical bundle part of the ``--yeto-placement-map`` Miles got."""
    pm = getattr(request, "placement_map_arg", None)
    if pm is None:
        pm = getattr(request, "placement_map", None)
    return None if pm is None else {k: v for k, v in pm.items() if k in ("trainer", "rollout", "standby")}


def _startup_views(manager: Any, runner: Any) -> dict[str, Any]:
    """The fork's startup placement-group views, read once before any is re-pointed."""
    views = {}
    for name in ("actor", "rollout", "standby"):
        try:
            views[name] = runner.run(_await_ref(manager.get_pg_view.remote(name)))
        except Exception:  # noqa: BLE001 - a role without a view (e.g. no standby)
            continue
    return views


async def _await_ref(ref: Any) -> Any:
    return await ref


def _wire_trainer_edges(driver, *, elastic, miles_args, launch, algorithm, actor_model,
                        rollout_executor, runner, base_model_revision, manager=None) -> bool:
    """E3 (4.7): the startup bundle map on the rollout pool (bind_members /
    member_gpus) and ``MilesTrainerOps`` behind ``IslandController.trainer_edges``.

    Skipped (trainer edges stay refused) without the SwappableActor proxy,
    without pool GPU ids (``ElasticWiring.pool_gpus``), when E3's modules are
    not in the tree, or when the startup views cannot be read. Returns whether
    the edges were wired. Attested trainer edges are still required per edge.
    """
    from .trainer_rebuild import SwappableActor, rebuild_preconditions

    controller = elastic.controller
    if rebuild_preconditions(miles_args):
        # review L2: a trainer edge rebuilds the trainer; not possible on this run
        return False
    if (not isinstance(actor_model, SwappableActor) or elastic.pool_gpus is None
            or not callable(getattr(controller, "set_trainer_edges", None))):
        return False
    try:
        from .trainer_resize import MilesTrainerOps
    except ImportError:  # E3 not integrated in this tree
        return False
    from .bundles import StartupBundles
    from .rebuild_wiring import CutSource

    if manager is None:
        from miles.utils.workers.ray_worker_manager import RayWorkerManager

        manager = RayWorkerManager.get_handle()
    views = _startup_views(manager, runner)
    try:
        bundles = StartupBundles(pool_gpus=elastic.pool_gpus, views=views,
                                 placement_map=_role_map(launch.placement))
    except Exception:  # noqa: BLE001 - no usable map: leave trainer edges refused
        return False
    pool = driver.rollout
    pool._bundles = bundles
    pool._worker_manager = manager
    pool._gpus_per_engine = int(launch.placement.gpus_per_engine)
    ref_load = getattr(miles_args, "ref_load", None)
    source = CutSource(
        driver=lambda: driver, trainer=driver.trainer, rollout=pool, ledger=elastic.ledger,
        algorithm=algorithm, backend_fingerprint=controller.runtime_fingerprint or "",
        cut_root=str(controller.state_dir / "cuts"),
        global_batch_size=int(miles_args.global_batch_size),
        ref_model=(None if not ref_load
                   else {"ref_load": str(ref_load), "base_model_revision": base_model_revision}),
    )
    ops = MilesTrainerOps(
        trainer=driver.trainer, actor=actor_model, rollout_executor=rollout_executor,
        run=runner.run, rollout=pool, root=source.cut_root, context_for=source.context,
        expect_for=lambda layout: source.expectation(
            layout, epoch=controller.journal.epochs.config_epoch),
        view_for=bundles.view_for,
        policy_hash_fn=lambda: driver.policy_state.export().policy_tensor_hash(),
        certified_for=lambda plan: controller.attestation.algorithms_for(
            (plan.source, plan.target, plan.kind)),
        worker_manager=manager,
    )
    controller.set_trainer_edges(lambda: {
        "spec": algorithm, "args": driver.trainer._args,
        "global_batch_size": int(miles_args.global_batch_size),
        "micro_batch_size": int(getattr(miles_args, "micro_batch_size", 1) or 1),
        "ops": ops,
    })
    return True


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
    with it so actors import the pinned Miles checkout, not the image's;
    ``YETO_RL_ELASTIC_METADATA`` (only when ``--rl-elastic`` set it) so the
    rollout metadata hook in the workers reports the data cursor.
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
    from .rollout_meta_hook import ELASTIC_METADATA_ENV

    from yeto.rl.tool_wait_workload import TOOL_DELAY_ENV

    for key, value in DETERMINISM_ENV.items():  # --rl-deterministic-trainer set them
        if environ.get(key) == value:
            env_vars[key] = value
    from .cut_injection import ALL_ENVS as CUT_INJECTION_ENVS

    for key in CUT_INJECTION_ENVS:  # TEST ONLY (E2 G-4.5): rank-side switches, off unless set
        if environ.get(key):
            env_vars[key] = environ[key]

    if environ.get(TOOL_DELAY_ENV):  # test tool-wait workload runs in Ray workers
        env_vars[TOOL_DELAY_ENV] = environ[TOOL_DELAY_ENV]
    if environ.get(ELASTIC_METADATA_ENV) == "1":
        # --rl-elastic: the rollout metadata hook runs inside Ray workers, which
        # inherit the raylet's environment, not the driver's.
        env_vars[ELASTIC_METADATA_ENV] = "1"
    ray_module.init(address=address, runtime_env={"env_vars": env_vars})
    return address


# E2 plan-v2 §0 determinism environment (with Megatron --deterministic-mode).
DETERMINISM_ENV = {"NCCL_ALGO": "Ring", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                   "NVIDIA_TF32_OVERRIDE": "0"}


def resolve_declared_cells(inference_controller: Any, runner: Any,
                           explicit: Any = ()) -> tuple[str, ...]:
    """The fork cell ids the E1 verbs manage.

    With a fork that lists its declared cells (``describe_cells``, F-R1), each
    explicit ``--rl-elastic-cells`` name is resolved to the fork cell id: by the
    ``alias`` the fork reports (the yeto name declared in placement map
    ``rollout_cells``), else by the cell id itself; an unknown name is refused.
    Without explicit names every declared cell is managed. A fork without
    ``describe_cells`` needs explicit names, taken as its cell ids.
    """
    explicit = tuple(str(c) for c in (explicit or ()))
    describe = getattr(inference_controller, "describe_cells", None)
    if not callable(describe):
        if not explicit:
            raise ValueError("--rl-elastic-cells is required: this Miles fork cannot list its "
                             "declared cells (describe_cells, F-R1)")
        return explicit
    cells = dict(runner.run(_awaitable(describe())) or {})
    if not cells:
        raise ValueError("the fork declares no rollout cells")
    if not explicit:
        return tuple(sorted(cells))
    by_alias = {str(d.get("alias")): cid for cid, d in cells.items()
                if isinstance(d, dict) and d.get("alias")}
    out, unknown = [], []
    for name in explicit:
        cid = by_alias.get(name) or (name if name in cells else None)
        (out.append(cid) if cid else unknown.append(name))
    if unknown:
        raise ValueError(f"--rl-elastic-cells {unknown} are not cells the fork declares "
                         f"(aliases {sorted(by_alias)}, ids {sorted(cells)})")
    return tuple(out)


async def _awaitable(value: Any) -> Any:
    return await value if hasattr(value, "__await__") else value


def elastic_wiring_for(miles_args: Any, *, profile: Any, fingerprint: str):
    """``miles_args.yeto_rl_elastic`` (learner ``--rl-elastic``) -> ElasticWiring.

    None when the switch is off: the island is then composed exactly as before.
    """
    config = getattr(miles_args, "yeto_rl_elastic", None)
    if not config:
        return None
    check_elastic_miles_args(miles_args)
    from .elastic_wiring import LazyBoardActor, build_elastic

    return build_elastic(
        state_dir=config["state_dir"],
        resources=config["resources"],
        attestation=config.get("attestation"),
        profile=profile,
        initial_config=config["initial_config"],
        runtime_fingerprint=fingerprint,
        declared_cells=tuple(config["declared_cells"]),
        # 3.3 X5: the island's named ToolWaitBoard actor, created lazily after
        # connect_island_ray (only when the learner asked for it).
        **({"tool_wait_board": LazyBoardActor(int(getattr(miles_args, "yeto_rl_learner_id", 0)))}
           if config.get("tool_wait_board") else {}),
        # 3.8 pause-budget inputs, only when the learner was given them.
        **{k: config[k] for k in ("quorum_timeout_s", "idle_flow_timeout_s", "pause_margin")
           if config.get(k) is not None},
        # 4.7: pool GPU ids (manifest resources.gpus, in logical-bundle order), only
        # with trainer edges; every other elastic run keeps the described pool.
        **({"pool_gpus": manifest_pool_gpus(config["resources"])}
           if config.get("trainer_edges") else {}),
    )


def check_elastic_miles_args(miles_args: Any) -> None:
    """Miles preconditions of the fork verbs the E1 controller uses, refused before
    Ray: cordon / drain_cells / admit_cells / cordoned update_weights need the
    Miles router; member publication needs a resident partitioned rollout."""
    problems = []
    if not getattr(miles_args, "use_miles_router", False):
        problems.append("--use-miles-router (fork cordon/drain/admit_cordoned need the Miles router)")
    if getattr(miles_args, "colocate", False):
        problems.append("no --colocate (elastic needs a fixed partition)")
    if getattr(miles_args, "offload_rollout", False):
        problems.append("no rollout offload (member publication needs resident engines)")
    if problems:
        raise ValueError("--rl-elastic needs " + "; ".join(problems))


def manifest_pool_gpus(resources: Any) -> tuple[str, ...]:
    """``resources.gpus[*].uuid`` in manifest order = logical bundle 0..N-1 of the
    fork-M1 placement map (the manifest must list the pool in that order)."""
    if not isinstance(resources, dict):
        resources = json.loads(Path(resources).expanduser().read_text(encoding="utf-8"))
    gpus = [g.get("uuid") for g in (resources.get("gpus") or [])]
    if not gpus or not all(isinstance(g, str) and g for g in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("--rl-elastic-trainer-edges needs the manifest's resources.gpus "
                         "(distinct uuids, in logical bundle order)")
    return tuple(gpus)


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
    from ..overlap import loop_eval_starter

    fingerprint = ports_runtime_fingerprint(launch)
    capabilities = with_partitioned_serial(
        miles_capabilities(
            fingerprint,
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
    # E1 (3.x), opt-in: a bad manifest/attestation fails here, before Ray.
    elastic = elastic_wiring_for(miles_args, profile=profile, fingerprint=fingerprint)
    from .e2_harness import load_plan as load_e2_harness_plan

    e2_plan = load_e2_harness_plan()  # TEST ONLY: None unless an E2 harness snapshot
    if e2_plan is not None:
        from .rollout_meta_hook import ELASTIC_METADATA_ENV

        # the harness cuts need the rollout data cursor (rollout-side metadata)
        miles_args.yeto_rl_elastic_metadata = True
        os.environ[ELASTIC_METADATA_ENV] = "1"
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
        # rl-infra-spec 4.3/4.4: one swappable handle shared by trainer,
        # policy state, publisher and the eval dispatcher (EvalDispatcher keeps
        # self.actor_model and resolves methods per call, so the proxy is
        # enough); the disposer gets the proxy, whose async dispose() resolves
        # the CURRENT target (Disposer.add binds item.dispose when added).
        from .trainer_rebuild import SwappableActor

        actor = SwappableActor(actor)
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
            evaluate_start=(
                loop_eval_starter(runner, evaluate, time.monotonic)
                if profile.execution_mode == "partitioned-overlap"
                else None
            ),
            elastic=elastic,
        )
        if e2_plan is not None:  # TEST ONLY: E2 GPU harness instead of the training loop
            from .e2_harness import HarnessContext, run_harness

            run_harness(HarnessContext(
                driver=driver, actor=actor, miles_args=miles_args, rollout_executor=executor,
                runner=runner, algorithm=algorithm, base_model_revision=base_model_revision,
                backend_fingerprint=fingerprint, plan=e2_plan,
            ))
            return driver.published_state
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
