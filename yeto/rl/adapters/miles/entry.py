"""``--rl-engine ports`` composition root.

``run_ports_island`` is what ``yeto.rl.adapters.miles.island_entry.run_miles`` calls on the ports
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
import logging
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, NamedTuple

from yeto.rl.engine.resume import DEFAULT_CUT_EVERY
from yeto.rl.engine.algorithm import (
    BOUNDED_NONZERO_STD_FILTER,
    STOCK_NONZERO_STD_FILTER,
    AlgorithmSpec,
)
from yeto.rl.engine.capabilities import R0_MECHANISMS, EngineCapabilities, ExecutionCapabilities
from yeto.rl.engine.events import append_tape_event
from .traits import MILES_TRAITS
from . import LoopRunner
from .config import MilesConfigError, check_harness_reward_scope
from .pins import miles_fork_lineage_from
from .remote_code_prewarm import prewarm_remote_code

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


# Declared beyond R0: "dimension:name" -> why the mechanism is declared, i.e. the
# GPU observation showing it takes effect (declaration policy; may be overridden
# by the user). One entry per mechanism.
MILES_DECLARED: dict[str, str] = {
    "corrections:tis": "GPU run: tis_clipfrac > 0",
    "corrections:opsm": (
        "GPU run: opsm_clipfrac > 0 with 2 optimizer steps; the OPSM dimension that every "
        "source-specific OPSM mechanism also requires"
    ),
    "corrections:opsm_trainer": "GPU run: opsm_clipfrac > 0",
    "features:maxrl": "GPU run: maxrl advantages applied",
    "features:mapo": "GPU run: mapo advantages applied",
    "loss_aggregations:constant": "GPU run: constant-denominator aggregation (Dr. GRPO)",
    "kl_placements:loss": "GPU run: kl_loss 0 / 0.00079 / 0.00082 with k3",
    "features:kl_loss_ref_model": "GPU run: reference model loaded, kl_loss > 0",
    "features:entropy_bonus": "GPU run: entropy_loss 0.30 / 0.38 / 0.45",
    "reward_postprocessors:custom_reward_postprocess": "GPU run: dispatcher shaped 4/7/21 of 32 samples",
    "features:overlong_penalty": "GPU run: shaped_samples 4/7/21",
    "advantage_estimators:gspo": "GPU run: second-step clipfrac 0.1875/0.5/0.5 with 2 optimizer steps",
    "advantage_estimators:reinforce_plus_plus": (
        "GPU run: rollout/advantages mean non-zero where GRPO's group-normalized mean is ~0; "
        "ref_log_probs scored every round and diverging from round 1 (reward KL active)"
    ),
    "advantage_estimators:reinforce_plus_plus_baseline": (
        "GPU run: rollout/advantages mean non-zero (vs ~0 for GRPO); ref_log_probs scored "
        "and diverging from round 1"
    ),
    "features:gdpo": "GPU run: per-round nonzero_advantages match the dispatcher",
    "corrections:mismatch_observe": "GPU run: observation only, weights constant 1",
    "corrections:icepop": "GPU run: icepop [0.99, 1.01] masks tokens (masked tis_clipfrac > 0)",
    "corrections:mis_mask": "GPU run: token mask [0.99, 1.01], mask fraction > 0",
    "corrections:mis": (
        "GPU run: token truncation with upper bound 1.01 (a trigger value, not a "
        "recommendation): truncate fraction > 0, weight after the bound <= before in every "
        "step, final max ratio 1.0100"
    ),
    "features:eps_clip": "GPU run: eps_clip 0.001 / eps_clip_high 0.002 (trigger values): step-2 pg_clipfrac > 0.1",
    "features:no_grpo_std_normalization": "GPU run: paired step-1 grad_norm 0.2428 vs baseline 0.6349",
    "loss_aggregations:token": (
        "GPU run: paired step-1 grad_norm 0.4310 vs baseline 0.4867; only on Miles "
        "0af62f4d+ (the LoRA bridge sets calculate_per_token_loss)"
    ),
    "features:over_sampling": (
        "GPU run: aborted submissions in the over_sampling arm while the two arms differ "
        "only in over_sampling_batch_size (weak evidence until Miles records the batch "
        "size of every submission). Miles 0af62f4d only"
    ),
    "features:overlong_filter": (
        "GPU run: filtered_samples 16/16/31 with the filter on, none with it off; step-1 "
        "grad_norm 0.5647 vs 0.6329; a round with 31/32 filtered raised no zero-gradient "
        "false alarm. Miles 0af62f4d only"
    ),
    "features:clip_higher": (
        "GPU run: step-1 grad_norm bit-identical in both arms, later steps differ "
        "(0.5642 vs 0.6572) with the arms differing only in eps_clip_high. Ran on Miles "
        "0394715; the diff to 0af62f4d leaves the loss and clip path unchanged. Miles "
        "0af62f4d only"
    ),
    # Small LoRA GPU run (3 rounds x 2 steps); Miles 64b591a4b only (see MILES_DECLARED_PINS).
    "losses:cispo": "GPU run: round-1 step-2 pg_loss -0.0324 vs GRPO 0.000166",
    "losses:sapo": "GPU run: round-1 step-2 pg_loss -0.0148 vs GRPO 0.000166",
    "losses:gmpo": "GPU run: clip_fraction equals the gmpo num/den ratio",
    "features:dual_clip": (
        "GPU run: eps_clip_c 1.01, dual_clipfrac > 0 (ports-v1 64b591a4b: the fork's loss metric "
        "dual_clipfrac; on k31 the evidence is yeto/dual_clipfrac, which the yeto state plugin "
        "reports with the fork's definition)"
    ),
    "features:kl_unbiased": "GPU run: kl_loss differs from k3",
    "corrections:opsm_rollout": "GPU run: opsm_clipfrac > 0 with opsm_old_logprob_source=rollout",
    "features:rollout_logprobs_as_old": "GPU run: use_rollout_logprobs=true in the event spec",
}


# User-code mechanisms that are never declared:
# GPU evidence covers one piece of code, not arbitrary user code.  The vendored
# Dr.GRPO reducer is claimed by ``loss_aggregations:constant`` through
# ``register_named_reducer`` and so never requires
# ``features:custom_pg_loss_reducer``; every other reducer does.  The yeto
# dispatcher (``reward_postprocessors:custom_reward_postprocess``) is yeto code
# and is not in this list.  ``--rl-allow-unverified-mechanism`` still admits
# these on a single island without outer sync only.
NEVER_DECLARABLE: frozenset[str] = frozenset({
    "corrections:custom",
    "features:plugins",
    "losses:custom_loss",
    "features:custom_pg_loss_reducer",
})
if NEVER_DECLARABLE & set(MILES_DECLARED):  # pragma: no cover - guarded by tests too
    raise RuntimeError(
        f"MILES_DECLARED must not declare user-code mechanisms: {sorted(NEVER_DECLARABLE & set(MILES_DECLARED))}"
    )

# GPU evidence behind a declaration is bound to a Miles lineage: it holds for
# ``from_commit`` and the later commits of that lineage (the carry-over
# arguments by code diff live in the lineage comments of pins.py), never for
# a commit of another lineage. On the k31 lineage every mechanism below is
# undeclared until its GPU re-verification lands a k31 EvidenceRef here
# (yeto-adapter.md 4.5; K31_REVERIFY says what each re-run has to show).
class EvidenceRef(NamedTuple):
    lineage: str  # key of pins.MILES_LINEAGES
    from_commit: str  # first commit of that lineage the evidence holds for
    run_id: str  # the GPU run (or carry-over argument) that produced the judgment
    judgment_path: str  # where the judgment is recorded (repo-relative or external)


_PORTS_V1_0AF62F4D = EvidenceRef(
    "ports-v1", "0af62f4d48ed6a5b185c257578d8f7e22312aa87",
    "lora-per-token-loss-smoke", "docs/RL_ALGORITHMS.md#declarations",
)
_PORTS_V1_64B591A4B = EvidenceRef(
    "ports-v1", "64b591a4bec1ffa37fb089d3e3b99c84773b1ffc",
    "rl-algo-supplement-phase1-gpu", "rl-algo-supplement/evidence/phase1-gpu (archived outside the repo)",
)
MILES_DECLARED_EVIDENCE: dict[str, tuple[EvidenceRef, ...]] = {
    # before 0af62f4d the LoRA bridge ignored calculate_per_token_loss
    # (grad_norm bit-identical to the baseline)
    "loss_aggregations:token": (_PORTS_V1_0AF62F4D,),
    "features:over_sampling": (_PORTS_V1_0AF62F4D,),
    "features:overlong_filter": (_PORTS_V1_0AF62F4D,),
    "features:clip_higher": (_PORTS_V1_0AF62F4D,),
    # the loss-supplement GPU smoke ran on 64b591a4b; clip_higher was re-run there too.
    **{m: (_PORTS_V1_64B591A4B,) for m in ("losses:cispo", "losses:sapo", "losses:gmpo", "features:dual_clip",
                                            "features:kl_unbiased", "corrections:opsm_rollout",
                                            "features:rollout_logprobs_as_old")},
}


# The pin-bound mechanisms whose ports-v1 GPU evidence does not carry over to the
# k31 lineage (B8 report 5.3), with why each needs its own k31 GPU re-run. On a
# k31 pin they are undeclared: requesting one is refused like any unverified
# mechanism (a single island without outer sync may pass
# --rl-allow-unverified-mechanism <name>); default GRPO needs none of them. An
# entry stops applying once a k31 EvidenceRef in MILES_DECLARED_EVIDENCE covers
# the pin (:func:`needs_k31_reverification`).
K31_REVERIFY: dict[str, str] = {
    **{m: ("loss and metrics bitwise equal to the fork on CPU (k31/b8 4ff093a8), but the flags reach "
           "the loss through the typed config slice and the CUDA torch.compile path is unverified")
       for m in ("losses:cispo", "losses:sapo", "losses:gmpo")},
    "features:dual_clip": ("loss bitwise equal; the evidence metric moved from the fork's dual_clipfrac "
                           "to yeto/dual_clipfrac (state plugin)"),
    "features:kl_unbiased": "upstream code, loss bitwise equal on CPU; a new pin is re-run on GPU",
    "features:clip_higher": "upstream code, loss bitwise equal on CPU; a new pin is re-run on GPU",
    "loss_aggregations:token": ("the LoRA path is the upstream adapter library on a newer Megatron-Bridge "
                                "(the ports-v1 evidence was the bridge's calculate_per_token_loss fix)"),
    "features:over_sampling": "rollout side: upstream rewrote the DataSource / RolloutFn path",
    "features:overlong_filter": "rollout side: upstream rewrote the DataSource / RolloutFn path",
    "corrections:opsm_rollout": ("loss bitwise equal; rollout_log_probs transport, CP split and dtype "
                                 "changed upstream"),
    "features:rollout_logprobs_as_old": ("rollout_log_probs transport, CP split and dtype changed "
                                         "upstream"),
}


def needs_k31_reverification(miles_commit: str | None = None) -> dict[str, str]:
    """The :data:`K31_REVERIFY` mechanisms still undeclared on ``miles_commit``
    (default :func:`pins.current_miles_commit`): empty off the k31 lineage, and
    for a mechanism a k31 EvidenceRef already covers."""
    from .pins import current_miles_commit, lineage_of

    commit = current_miles_commit() if miles_commit is None else miles_commit
    lineage = lineage_of(commit)
    if lineage is None or lineage.name != "k31":
        return {}
    return {m: why for m, why in K31_REVERIFY.items()
            if not any(evidence_covers(ref, commit) for ref in MILES_DECLARED_EVIDENCE.get(m, ()))}


def evidence_covers(ref: EvidenceRef, miles_commit: str) -> bool:
    """``miles_commit`` is ``ref.from_commit`` or later on ``ref.lineage``."""
    from .pins import MILES_LINEAGES

    lineage = MILES_LINEAGES[ref.lineage]
    position = lineage.index(miles_commit)
    return position is not None and position >= lineage.commits.index(ref.from_commit)


def _declared_pins() -> dict[str, frozenset[str]]:
    return {
        mechanism: frozenset().union(*(miles_fork_lineage_from(ref.from_commit) for ref in refs))
        for mechanism, refs in MILES_DECLARED_EVIDENCE.items()
    }


# mechanism -> every commit its evidence covers (derived from MILES_DECLARED_EVIDENCE).
MILES_DECLARED_PINS: dict[str, frozenset[str]] = _declared_pins()


def declared_by_dimension(miles_commit: str | None = None) -> dict[str, set[str]]:
    """R0 mechanism sets plus :data:`MILES_DECLARED`, per dimension.

    A mechanism with :data:`MILES_DECLARED_EVIDENCE` is declared only when
    one of its evidence references covers the Miles pin (``miles_commit``,
    default :func:`pins.current_miles_commit`): same lineage, at or after the
    evidence commit. A pin of another lineage (the k31 sync) declares none of
    them until re-verified there.
    """

    from yeto.rl.engine.capabilities import R0_MECHANISMS

    from .pins import current_miles_commit

    if miles_commit is None:
        miles_commit = current_miles_commit()
    out = {dim: set(names) for dim, names in R0_MECHANISMS.items()}
    out["advantage_estimators"] = {"grpo"}
    for mechanism in MILES_DECLARED:
        refs = MILES_DECLARED_EVIDENCE.get(mechanism)
        if refs is not None and not any(evidence_covers(ref, miles_commit) for ref in refs):
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
    here only after its single-GPU smoke passed. Exception: the SecRLEnv
    group check moved over from the removed legacy engine; its ports run is
    still to be scheduled. ``execution``: no critic, serial
    colocated produces policy age 0, and upstream SGLang rollouts return
    logprobs (``sglang_rollout.py`` ``return_logprob=True``).
    ``unverified_mechanisms`` is the single-island allowance.
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
        traits=MILES_TRAITS,
        **declared_by_dimension(),
    )
    if unverified_mechanisms:
        capabilities = capabilities.with_unverified(unverified_mechanisms)
    return capabilities


def receipt_role_family(algorithm: AlgorithmSpec) -> str:
    """``LocalStepReceipt.algorithm``: the TRAINING ROLE FAMILY, not the estimator.

    It must equal ``ParameterLayout.algorithm`` (``local_learner.py`` checks
    both; the layout hash covers it), whose families are grpo / sao / ppo. Every
    critic-free estimator (grpo, gspo, reinforce_plus_plus[_baseline]) trains
    the single actor role -> ``"grpo"``; the estimator itself is identified by
    ``algorithm_spec_sha256``. A critic algorithm (estimator ppo with
    ``execution.needs_critic``) trains actor + critic -> ``"ppo"``.
    """
    from yeto.rl.engine.algorithm import CRITIC_ESTIMATORS

    estimator = algorithm.advantage_estimator
    if estimator in CRITIC_ESTIMATORS:
        if not getattr(getattr(algorithm, "execution", None), "needs_critic", False):
            raise ValueError(
                f"advantage estimator {estimator!r} needs execution.needs_critic=true"
            )
        return "ppo"
    return "grpo"


def with_partitioned_serial(capabilities: EngineCapabilities) -> EngineCapabilities:
    """Infra declaration: the fixed-partition placement and
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
    contract check non-circular.
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
    only) when ``yeto_rl_overlap_eval`` is set. The profile is
    bound to ``expected_sha256`` (launcher-provided); :func:`preflight` then
    compares it with the runtime ``AlgorithmSpec``. Without an external hash a
    partitioned run is refused; a colocated (R0) run binds to the runtime spec
    and records that the hash source was the runtime.
    """
    from yeto.rl.engine.execution_profile import ExecutionProfile, ProfileError, check_overlap_eval

    from yeto.rl.engine.overlap import IMPLEMENTED_OVERLAP

    from .lr_schedule import miles_lr_schedule_sha256

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
        **_policy_age_fields(miles_args),
        lr_schedule_sha256=miles_lr_schedule_sha256(miles_args),
        extra={"algorithm_hash_source": source},
    )


def _policy_age_fields(miles_args: Any) -> dict[str, Any]:
    """agentic-rollout-utilization 2.1: limit 0 (default) adds nothing to the profile."""
    from yeto.rl.engine.policy_age import BOUNDED_STALENESS_CONTRACT

    age = int(getattr(miles_args, "yeto_rl_max_policy_age", 0) or 0)
    return ({"max_policy_age": age, "algorithm_contract": BOUNDED_STALENESS_CONTRACT}
            if age else {})


def load_tool_wait_source(miles_args: Any, elastic: Any = None) -> Any:
    """1.7: where load samples read the in-flight tool-wait count from.

    The elastic drain board when wired; the island's named board when the
    tool-wait workload generate is configured (it counts on that board); 0
    (``TOOL_WAIT_NO_BOARD_STOCK``) for Miles' stock generate, which makes no
    tool calls; None (unknown) for any other custom generate.
    """
    from .rollout import TOOL_WAIT_NO_BOARD_STOCK

    board = getattr(elastic, "tool_wait_board", None) if elastic is not None else None
    if board is not None:
        return board
    custom = custom_function_path(getattr(miles_args, "custom_generate_function_path", None))
    if not custom:
        return TOOL_WAIT_NO_BOARD_STOCK
    from yeto.rl.tool_wait_workload import GENERATE_PATH

    if custom == GENERATE_PATH:
        from .elastic_wiring import LazyBoardActor

        return LazyBoardActor(int(getattr(miles_args, "yeto_rl_learner_id", 0) or 0))
    return None


def side_effect_log_kwargs(elastic: Any) -> dict[str, Any]:
    """3.3 X5 evidence switch (YETO_RL_TEST_TOOL_SIDE_EFFECT_LOG, launcher
    --rl-test-tool-side-effect-log): the pool journals every execution of the
    injected tool in ``<elastic state dir>/side_effects.jsonl``."""
    from .rollout import SIDE_EFFECT_LOG_FILE, side_effect_log_enabled

    if elastic is None or not side_effect_log_enabled():
        return {}
    return {"side_effect_log": Path(elastic.controller.state_dir) / SIDE_EFFECT_LOG_FILE}


def harness_source(miles_args: Any, elastic: Any = None) -> Any:
    """Where the drain probe / load sample read harness counts from.

    ``elastic.harness_board`` when wired; the island's named ``HarnessBoard``
    actor when a custom agent function (agentic generate) is configured;
    ``HARNESS_NOT_AGENTIC`` (explicit zeros) otherwise. Never None here: a
    pool built with ``harness=None`` reports "harness counts unknown".
    """
    from .rollout import HARNESS_NOT_AGENTIC

    board = getattr(elastic, "harness_board", None) if elastic is not None else None
    if board is not None:
        return board
    if custom_function_path(getattr(miles_args, "custom_agent_function_path", None)):
        from .elastic_wiring import LazyBoardActor
        from yeto.rl.engine.tool_wait import harness_board_actor

        return LazyBoardActor(int(getattr(miles_args, "yeto_rl_learner_id", 0) or 0),
                              factory=harness_board_actor)
    return HARNESS_NOT_AGENTIC


# Harness preflight hook. Called by ``run_ports_island`` after the
# contract preflight and BEFORE connect_island_ray / any placement or model
# allocation; an exception aborts the island with zero allocate calls.
HarnessPreflight = Callable[[Any, Any], None]  # (miles_args, launch) -> None
HARNESS_PREFLIGHT_ENV = "YETO_HARNESS_PREFLIGHT"  # "module:callable" / "module.callable"


def resolve_harness_preflight(miles_args: Any, environ: Any = None) -> HarnessPreflight | None:
    """The hook from ``miles_args.yeto_harness_preflight`` or the environment (dotted path)."""
    environ = os.environ if environ is None else environ
    spec = getattr(miles_args, "yeto_harness_preflight", None) or environ.get(HARNESS_PREFLIGHT_ENV)
    if not spec:
        return None
    if callable(spec):
        return spec
    import importlib

    module, sep, name = str(spec).partition(":")
    if not sep:
        module, _, name = module.rpartition(".")
    if not module or not name:
        raise MilesConfigError(f"harness preflight {spec!r} is not module:callable")
    hook = getattr(importlib.import_module(module), name)
    if not callable(hook):
        raise MilesConfigError(f"harness preflight {spec!r} is not callable")
    return hook


# The island publishes its own INFRA member cell
# so the Codex harness admits sessions under ``rollout.member_id(cell_id)`` instead
# of the global key.  Single-island (--rl-single-island-no-sync) islands publish 0.
MEMBER_CELL_ENV = "YETO_RL_CELL_ID"


def publish_member_cell(miles_args: Any, learner_id: int | None = None, environ: Any = None) -> str:
    """Set ``miles_args.yeto_rl_cell_id`` and export ``YETO_RL_CELL_ID``; return the cell id.

    Source, in order: an explicit ``miles_args.yeto_rl_cell_id`` (INFRA may pre-set it),
    ``learner_id`` (the island number the launcher already assigns; ``island_id`` in the
    ``rl_engine_selected`` event), ``miles_args.yeto_rl_learner_id``, else 0.  Rollout-side
    code reads the env export (``engine:<cell>``; ``engine:0`` on a single island).
    """
    environ = os.environ if environ is None else environ
    cell = getattr(miles_args, "yeto_rl_cell_id", None)
    if cell is None or str(cell) == "":
        if learner_id is None:
            learner_id = getattr(miles_args, "yeto_rl_learner_id", None)
        cell = int(learner_id or 0)
    cell = str(cell)
    miles_args.yeto_rl_cell_id = cell
    environ[MEMBER_CELL_ENV] = cell
    return cell


def preflight_stage(
    miles_args: Any,
    launch: Any,
    algorithm: AlgorithmSpec,
    *,
    yeto_policy_sync: bool,
    harness_preflight: HarnessPreflight | None = None,
) -> tuple[str, EngineCapabilities, Any, Any]:
    """Everything ``run_ports_island`` checks before Ray / placement / allocation.

    Returns ``(fingerprint, capabilities, profile, elastic)``. The harness
    preflight runs last, after the contract preflight and the elastic
    wiring checks; a failure here means no allocate call ever happens.
    """
    publish_member_cell(miles_args)  # member cell before the harness preflight resolves it
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
    preflight(profile, algorithm, capabilities)  # before any GPU process
    # Elastic wiring, opt-in: a bad manifest/attestation fails here, before Ray.
    elastic = elastic_wiring_for(miles_args, profile=profile, fingerprint=fingerprint)
    hook = harness_preflight if harness_preflight is not None else resolve_harness_preflight(miles_args)
    if hook is not None:
        # 7.1 in Miles' error type; the neutral hook repeats it with the core error.
        check_harness_reward_scope(getattr(miles_args, "yeto_harness_reward_scope", None))
        hook(miles_args, launch)  # harness preflight before placement/allocation
    return fingerprint, capabilities, profile, elastic


def preflight(profile: Any, algorithm: AlgorithmSpec, capabilities: EngineCapabilities) -> None:
    """The launcher-bound profile agrees with the runtime AlgorithmSpec and the
    declared capabilities, before any GPU process exists (before connect_island_ray)."""
    from yeto.rl.engine.execution_profile import check_algorithm_contract

    check_algorithm_contract(profile, algorithm)
    # neutral names bind to the pre-rename implementation, or no launch.
    from .binding import check_spec

    print("[yeto] backend binding " + json.dumps(check_spec(algorithm), sort_keys=True), flush=True)
    placement = "colocated" if profile.execution_mode == "colocated-serial" else "fixed-partition"
    capabilities.check(
        layout="lora",
        placement=placement,
        execution_mode=profile.execution_mode,
        algorithm=algorithm,
        max_policy_age=profile.max_policy_age,
    )


def build_sync(
    miles_args: Any,
    *,
    yeto_policy_sync: bool,
    runner: Any = None,
    groups: Mapping[str, Any] | None = None,
) -> tuple[Any, Any]:
    """(sync session, progress store) for the preset the learner selected.

    A full-parameter island gets the one :class:`FullParameterSync`
    over its trainable-role ``groups``; its progress lives in the syncer.
    """

    from .full_parameter.session import CONFIG_ATTRIBUTE, build_full_parameter_sync

    if getattr(miles_args, CONFIG_ATTRIBUTE, None) is not None:
        if not yeto_policy_sync:
            raise ValueError("full-parameter training requires Yeto policy sync")
        if runner is None or groups is None:
            raise ValueError("full-parameter sync needs the event-loop runner and trainer groups")
        return build_full_parameter_sync(miles_args, runner=runner, groups=groups), None

    from yeto.rl.engine.bridges import (
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
    if getattr(miles_args, "yeto_rl_island_pool", False):
        # Dynamic island pool; absent attribute = fixed roster (unchanged below).
        from yeto.rl.engine.bridges import PoolAvgSync

        if getattr(miles_args, "use_critic", False):
            raise ValueError("island pool options do not support a critic yet")
        return PoolAvgSync(miles_args.yeto_rl_bridge_config, progress=progress,
                              syncer_epoch=int(getattr(miles_args, "yeto_rl_syncer_epoch", 0)),
                              lease_s=getattr(miles_args, "yeto_rl_island_lease_s", None),
                              groups_per_round=int(getattr(miles_args, "rollout_batch_size", 0) or 0) or None,
                              ), progress
    critic_syncer = getattr(miles_args, "yeto_rl_critic_syncer_addr", None)
    if critic_syncer is not None:
        # second syncer channel for the
        # critic, one atomic commit for both roles.
        from yeto.rl.engine.bridges import DualStrictAvgSync

        return DualStrictAvgSync(miles_args.yeto_rl_bridge_config, critic_syncer_addr=critic_syncer,
                                 progress=progress), progress
    if getattr(miles_args, "use_critic", False):
        raise ValueError("strict-avg with a critic needs the critic syncer (--critic-syncer); "
                         "the actor syncer alone would leave the critics unaveraged")
    return StrictAvgSync(miles_args.yeto_rl_bridge_config, progress=progress), progress


def data_position_log_for(miles_args: Any, sync: Any, *, learner_id: int) -> Any:
    """The data position log of a full-parameter island in
    the run store (``--rl-data-position-store``), None otherwise. Only a
    full-parameter session restarts its round counter at 0 in every process; the
    other sessions resume their data cursor by global rollout id."""
    store = getattr(miles_args, "yeto_rl_data_position_store", None)
    if not store or not _is_full_parameter_sync(sync):
        return None
    from yeto.rl.engine.data_position import data_position_log

    return data_position_log(str(store), learner_id=int(learner_id),
                             slot=getattr(miles_args, "yeto_rl_data_slot", None))


HF_EXPORT_STAGING = "yeto-hf-export"  # under the system temp dir (local disk)


def hf_exports_for(miles_args: Any, *, actor_model: Any, runner: Any,
                   environ: Mapping[str, str] | None = None) -> Any:
    """``--rl-export-hf-rounds`` (``yeto.rl.engine.hf_export``): the island's
    ``HfRoundExports``, None when off. With the trainer on one node every rank
    writes a share of the files into the local staging directory; across nodes
    Miles' ``export_hf`` writes them on rank 0."""
    config = getattr(miles_args, "yeto_rl_hf_exports", None)
    if not config:
        return None
    from yeto.rl.engine.hf_export import EXPORT_HF_FLAG, ExportRounds, HfRoundExports
    from yeto.rl.engine.resume import store_for

    from .publish import lora_mode
    from .release_plugin import full_checkpoint_writer

    if lora_mode(miles_args):
        raise ValueError(f"{EXPORT_HF_FLAG} writes full weights; a LoRA island has no full export")
    one_node = int(getattr(miles_args, "actor_num_nodes", 1) or 1) == 1
    return HfRoundExports(ExportRounds.parse(config["rounds"]), store_for(config["store"], environ=environ),
                          full_checkpoint_writer(actor_model, runner, all_ranks=one_node,
                                                 base_dir=_hf_dir(miles_args)),
                          staging=Path(tempfile.gettempdir()) / HF_EXPORT_STAGING)


def _hf_dir(miles_args: Any) -> Path | None:
    path = getattr(miles_args, "hf_checkpoint", None)
    return Path(path) if path and Path(path).is_dir() else None


def _is_full_parameter_sync(sync: Any) -> bool:
    from yeto.rl.engine.full_parameter_sync import FullParameterSync

    return isinstance(sync, FullParameterSync)


def _policy_state_for(sync: Any, *, full_parameter: bool = False, **lora: Any) -> Any:
    """The full-parameter actor cut for a FullParameterSync island; a trainer-side
    full-parameter digest for a no-sync full-parameter island; else LoRA state."""

    from .full_parameter.session import MilesFullParameterPolicyState
    from .state import MilesFullParameterLocalPolicyState, MilesPolicyState

    if _is_full_parameter_sync(sync):
        return MilesFullParameterPolicyState(sync)
    if full_parameter:
        return MilesFullParameterLocalPolicyState(
            actor_model=lora["actor_model"], base_model_revision=lora["base_model_revision"],
            config_hash=lora["config_hash"], expected_layout_hash=lora["expected_layout_hash"],
            runner=lora.get("runner"))
    return MilesPolicyState(**lora)


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
    layout_hash: str | None,
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
    critic_model: Any = None,
    orchestrator_args: Any = None,
):
    """Wire the adapter ports into an ``IslandDriver`` (no upstream imports).

    ``orchestrator_args``: the upstream ``OrchestratorConfig`` slice the
    components were built from (``ArgvOrchestratorStartupInfo.args``); the
    publisher hands it to upstream ``update_weights``. Defaults to ``miles_args``.

    ``critic_model``: the upstream critic
    ``TrainGroup`` of a critic algorithm (shared actor/critic PPO), trained
    before the actor in every round; None for every critic-free algorithm.

    ``elastic`` (:class:`.elastic_wiring.ElasticWiring`):
    declared rollout cells for the elastic membership verbs, the reconfiguration
    controller and the batch ledger. The controller reconciles the fork's
    membership epoch with its journal before the driver runs. None keeps the
    island unchanged.
    """

    from yeto.rl.core import parse_policy_snapshot_token

    from yeto.rl.engine.driver import DriverError, EventTape, IslandDriver
    from .placement import MilesPlacement
    from .publish import MilesPublisher
    from .rollout import MilesRolloutPool
    from .trainer import MilesTrainerGroup

    holder: dict[str, IslandDriver] = {}
    # rl-resume-from-checkpoint: round cuts without --rl-elastic (single island, no sync)
    resume = resume_wiring_for(miles_args, sync=sync) if elastic is None else None

    def expected_policy() -> tuple[int, str]:
        driver = holder["driver"]
        if driver.expected_token is None:
            raise DriverError("rollout requested before any complete publication")
        return parse_policy_snapshot_token(driver.expected_token)

    full_parameter = _is_full_parameter_sync(sync) or getattr(
        miles_args, "yeto_rl_parameter_mode", None) == "full"
    policy_state = _policy_state_for(
        sync,
        full_parameter=full_parameter,
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
            load_tool_wait=load_tool_wait_source(miles_args, elastic),
            harness=harness_source(miles_args, elastic),
            **(
                {"declared_cells": resolve_declared_cells(
                    inference_controller, runner, elastic.declared_cells),
                 "track_timeout_s": elastic.track_timeout_s,
                 "tool_wait_board": elastic.tool_wait_board,
                 **side_effect_log_kwargs(elastic)}
                if elastic is not None
                else {}
            ),
        ),
        trainer=MilesTrainerGroup(
            args=miles_args,
            actor_model=actor_model,
            learner_id=learner_id,
            learner_generation=0,
            parameter_layout_hash=lambda: policy_state.layout_hash,
            algorithm=receipt_role_family(algorithm),
            spec=algorithm,
            critic_model=critic_model,
            release_refs=release_refs,
            runner=runner,
            install_actor_recorders=full_parameter,
        ),
        policy_state=policy_state,
        publisher=MilesPublisher(
            args=miles_args,
            orchestrator_args=orchestrator_args,
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
            else {"ledger": resume.ledger} if resume is not None
            else {}
        ),
        **({"evaluate_start": evaluate_start} if evaluate_start is not None else {}),
    )
    driver.round_phases = True  # per-round time breakdown (rl_round_phases) for MFU
    driver.ledger_set_aside_before = getattr(miles_args, "yeto_rl_ledger_set_aside_before", None)
    # publication checks on every round (default: cut, first and every-K rounds)
    driver.verify_every_round = bool(getattr(miles_args, "yeto_rl_verify_every_round", False))
    # --rl-verify-every N (no-sync island): hash the policy only every N-th version
    verify_every = getattr(miles_args, "yeto_rl_verify_every", None)
    if verify_every is not None:
        driver.verify_every = int(verify_every)
        allow = getattr(driver.policy_state, "allow_unverified_rounds", None)
        if callable(allow):
            allow()
    # the engines generate the next round's batch while the trainer trains
    driver.rollout_ahead = int(getattr(miles_args, "yeto_rl_rollout_ahead", 0) or 0)
    if elastic is not None:
        from .elastic_placement import ElasticPlacement

        driver.publisher.perturb_trainer = lora_perturber(driver)  # TEST injection hook only
        # TEST injection records (test_injection / test_hold) -> journal + tape
        driver.publisher.event_sink = (
            lambda event, **f: elastic.controller.record_test_event(driver, event, **f))
        if hasattr(driver.rollout, "event_sink"):
            driver.rollout.event_sink = driver.publisher.event_sink

        _topo = getattr(launch.placement, "topology", None)
        driver.placement = ElasticPlacement(
            driver.placement, pool_gpus=elastic.pool_gpus,
            epoch=elastic.controller.journal.epochs.config_epoch,
            gpus_per_node=getattr(_topo, "gpus_per_node", None),
        )
        epochs = elastic.controller.journal.epochs
        driver.config_epoch = epochs.config_epoch
        if epochs.config_epoch > 0:
            # restart after a commit: the journal's config is authoritative
            committed = elastic.controller.configs[epochs.config_id].placement or {}
            if committed.get("rollout"):
                driver.placement.restore_committed(tuple(committed["rollout"]),
                                                   epoch=epochs.config_epoch)
        topology = getattr(launch.placement, "topology", None)
        if topology is not None and callable(getattr(elastic.controller, "set_topology", None)):
            # node loss is observed through Ray, fail closed
            elastic.controller.set_topology((topology.nodes, topology.gpus_per_node), _ray_alive_nodes,
                                            layout=island_layout_of(miles_args, topology, launch.placement))
        elastic.controller.open(driver.rollout)
        _wire_trainer_rebuild(driver, elastic=elastic, miles_args=miles_args, algorithm=algorithm,
                              actor_model=actor_model, rollout_executor=rollout_executor,
                              runner=runner, base_model_revision=base_model_revision)
        _wire_round_cuts(driver, elastic=elastic, miles_args=miles_args, algorithm=algorithm,
                         base_model_revision=base_model_revision)
        if (getattr(miles_args, "yeto_rl_elastic", None) or {}).get("trainer_edges"):
            _wire_trainer_edges(driver, elastic=elastic, miles_args=miles_args, launch=launch,
                                algorithm=algorithm, actor_model=actor_model,
                                rollout_executor=rollout_executor, runner=runner,
                                base_model_revision=base_model_revision)
        # --rl-recommend-mode / --rl-edge-costs-path / --rl-elastic-window-s
        from .elastic_hook import elastic_hook_for

        driver.elastic_hook = elastic_hook_for(miles_args, controller=elastic.controller,
                                               profile=profile, observe=observe)
        # dashboard 5.1/5.2: controller cell-snapshot / reconfig-phase events; only when the
        # observation path is on so the legacy (observe=False) tape stays byte-identical.
        if observe:
            elastic.controller.set_event_sink(driver.emit, journal=True)
    if resume is not None:
        if resume.stop_after is not None:
            sync.stop_after = resume.stop_after
        from yeto.rl.engine.resume import install_preempt_handler, watch_preempt_marker

        install_preempt_handler(resume.controller, emit=driver.emit)
        # on Modal the signal reaches the runner, which writes the reclaim marker
        watch_preempt_marker(resume.controller)
        _wire_round_cuts(driver, elastic=resume, miles_args=miles_args, algorithm=algorithm,
                         base_model_revision=base_model_revision)
    _wire_droppable_reclaim(sync, driver)
    from .sampler_island import wire_policy_releases

    # a trainer island (--rl-sample-role trainer) releases every publish to the samplers
    wire_policy_releases(driver, miles_args=miles_args, actor_model=actor_model, runner=runner)
    # opt-in (YETO_SPOT_INFLIGHT_SAVE_DIR): timed in-flight save
    from .inflight_save import install as _install_inflight_save

    _install_inflight_save(driver, rollout_executor, island=str(learner_id))
    holder["driver"] = driver
    return driver


def _wire_droppable_reclaim(sync: Any, driver: Any, environ: Any = None) -> Any:
    """A droppable island (``YETO_ISLAND_ROLE``) LEAVEs the
    elastic pool on a reclaim notice and writes ``spot_reclaim`` with role
    droppable. Nothing is saved: it loses only its own round increment and comes
    back from the syncer base. No-op on every other island."""
    from yeto.cloud import droppable

    if droppable.env_role(environ) is None or not hasattr(sync, "_client"):
        return None

    def leave() -> bool:
        from yeto.rl.pool_client import LeaveReason

        return sync.leave(LeaveReason.PREEMPTED)  # once per process; the driver's close won't resend

    island = str(getattr(getattr(sync, "config", None), "learner_id", "?"))
    return droppable.install_reclaim_listener(island, leave=leave, emit=driver.emit,
                                              environ=environ)


# rl-resume-from-checkpoint design §3: what must be identical between a cut and the
# launch resuming it (Miles argument names; num_rollout is NOT part of it: extending a
# run is the normal resume). Missing attributes are recorded as None.
RUN_FINGERPRINT_ARGS = (
    "hf_checkpoint", "ref_load", "lora_rank", "lora_alpha", "lora_dropout", "target_modules",
    "lr", "lr_decay_style", "min_lr", "lr_warmup_iters", "lr_decay_iters", "weight_decay",
    "adam_beta1", "adam_beta2", "clip_grad", "optimizer", "seed", "prompt_data", "input_key",
    "label_key", "rollout_shuffle", "rollout_batch_size", "n_samples_per_prompt",
    "global_batch_size", "rollout_max_response_len", "rollout_temperature",
    "tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size",
    "expert_model_parallel_size", "advantage_estimator", "eps_clip", "eps_clip_high", "kl_coef",
    "use_kl_loss", "bf16", "fp16",
)


def run_fingerprint_of(miles_args: Any, algorithm: Any) -> dict[str, Any]:
    def plain(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return str(value)

    out = {name: plain(getattr(miles_args, name, None)) for name in RUN_FINGERPRINT_ARGS}
    sha = getattr(algorithm, "sha256", None)
    out["algorithm_spec_sha256"] = sha() if callable(sha) else None
    # a partial-experts run: the cut's trainable slots belong to this selection (the argv
    # names only the file's path); absent for every other recipe, so their fingerprints stay
    selection = (getattr(miles_args, "yeto_rl_release_manifest", None) or {}).get("expert_selection_sha256")
    if selection is not None:
        out["expert_selection_sha256"] = str(selection)
    return out


def next_lr_for(miles_args: Any):
    """Design §4.5 ``lr_at_next_round``: the lr the next round will train at, from the
    run's schedule with Megatron's ``OptimizerParamScheduler.get_lr`` arithmetic
    (constant / linear / cosine, warmup 0 -- what yeto's lr_schedule emits). The round
    after ``local_step`` optimizer steps trains at scheduler step ``local_step``
    (e.g. rounds 0..9 at 1e-5 * (1 - k/10)). None when the schedule is not one of
    these (the cut still restores the scheduler step, checked in restore_cut_shard)."""
    import math

    style = getattr(miles_args, "lr_decay_style", None)
    lr = getattr(miles_args, "lr", None)
    warmup = int(getattr(miles_args, "lr_warmup_iters", 0) or 0)
    min_lr = float(getattr(miles_args, "min_lr", 0.0) or 0.0)
    decay = getattr(miles_args, "lr_decay_iters", None)
    if lr is None or warmup != 0 or style not in ("constant", "linear", "cosine"):
        return None
    if style != "constant" and not decay:
        return None
    max_lr = float(lr)

    def next_lr(driver: Any) -> Any:
        k = int(getattr(driver, "local_step", 0))
        if style == "constant":
            value = max_lr
        elif k > int(decay):
            value = min_lr
        else:
            ratio = float(k) / float(int(decay))
            coeff = (1.0 - ratio) if style == "linear" else 0.5 * (math.cos(math.pi * ratio) + 1.0)
            value = min_lr + coeff * (max_lr - min_lr)
        return [value]

    return next_lr


class ResumeWiring:
    """The round-cut controller + batch ledger of a run with ``--rl-resume-store`` and
    no ``--rl-elastic`` (same attribute names as ElasticWiring for _wire_round_cuts).
    ``interval``: when a cut is due (:mod:`yeto.rl.engine.cut_interval`)."""

    def __init__(self, *, controller: Any, ledger: Any, every: int, stop_after: int | None,
                 allow_config_change: bool, interval: Any = None) -> None:
        self.controller = controller
        self.ledger = ledger
        self.every = every
        self.stop_after = stop_after
        self.allow_config_change = allow_config_change
        self.interval = interval


def resume_wiring_for(miles_args: Any, *, sync: Any, environ: Any = None) -> ResumeWiring | None:
    config = getattr(miles_args, "yeto_rl_resume", None)
    if not config:
        return None
    from yeto.rl.engine.bridges import LocalOnlySync, RoundCutHooks
    from yeto.rl.engine.cut_interval import cut_interval_for

    if config.get("write_only"):
        # a synced island writes round cuts; its syncer stays authoritative
        if isinstance(sync, LocalOnlySync) or not isinstance(sync, RoundCutHooks):
            raise ValueError(f"--rl-island-cut-store: {type(sync).__name__} takes no island round cuts "
                             "(strict, decoupled or full-parameter outer sync only)")
    elif not isinstance(sync, LocalOnlySync):
        raise ValueError("--rl-resume-store needs --rl-single-island-no-sync (multi-island resume "
                         "goes through the syncer checkpoint)")
    from yeto.rl.engine.ledger import BatchLedger, island_cut_ledger
    from yeto.rl.engine.resume import ResumeController, store_for

    if not config.get("runtime_fingerprint"):
        raise ValueError("--rl-resume-store: no runtime fingerprint (a cut without it is refused)")
    store = store_for(config["store"], environ=environ)
    controller = ResumeController(state_dir=config["state_dir"], store=store,
                                  runtime_fingerprint=str(config.get("runtime_fingerprint") or ""),
                                  keep=int(config.get("keep", 2)))
    interval = cut_interval_for(every=config.get("every"), auto_mtbf_s=config.get("auto_mtbf_s"),
                                max_interval_s=config.get("max_interval_s"))
    ledger = (island_cut_ledger(controller.state_dir, tag=controller.incarnation["id"],
                                rounds_restart=bool(getattr(sync, "local_rounds_restart", False)))
              if config.get("write_only") else BatchLedger(controller.state_dir))
    return ResumeWiring(controller=controller, ledger=ledger,
                        every=interval.every, stop_after=config.get("stop_after"),
                        allow_config_change=bool(config.get("allow_config_change", False)),
                        interval=interval)


def _wire_round_cuts(driver, *, elastic, miles_args, algorithm, base_model_revision) -> None:
    """With a checkpoint store and no outer syncer, keep a round
    cut in the store at every safe point and resume from it (round_cut module doc)."""
    from .rebuild_wiring import CutSource
    from .round_cut import wire_round_cuts

    controller = elastic.controller
    if getattr(controller, "checkpoint_store", None) is None:
        return
    ref_load = getattr(miles_args, "ref_load", None)
    source = CutSource(
        driver=lambda: driver, trainer=driver.trainer, rollout=driver.rollout, ledger=elastic.ledger,
        algorithm=algorithm, backend_fingerprint=controller.runtime_fingerprint or "",
        cut_root="", global_batch_size=int(miles_args.global_batch_size),
        ref_model=(None if not ref_load
                   else {"ref_load": str(ref_load), "base_model_revision": base_model_revision}),
    )
    config = getattr(miles_args, "yeto_rl_resume", None) or {}
    checkpoint = wire_round_cuts(
        driver, controller=controller, source=source,
        every=int(getattr(elastic, "every", None) or config.get("every")
                  or getattr(miles_args, "yeto_rl_resume_every", None) or DEFAULT_CUT_EVERY),
        fingerprint=run_fingerprint_of(miles_args, algorithm),
        allow_config_change=bool(getattr(elastic, "allow_config_change", False)
                                 or config.get("allow_config_change", False)),
        next_lr=next_lr_for(miles_args), interval=_cut_interval_of(elastic, miles_args))
    if config.get("write_only"):
        # a synced trainer's buffer lane keeps its own position
        _commit_cuts_inline_for_buffer_reader(checkpoint, miles_args=miles_args)
        return
    _wire_sample_buffer(checkpoint, driver=driver, miles_args=miles_args)


def _commit_cuts_inline_for_buffer_reader(checkpoint: Any, *, miles_args: Any) -> None:
    """A write-only cut store on a trainer that reads the sample buffer: commit
    each cut inline. A deferred cut holds the store writer guard until the next
    cut, so the buffer reader skips every Volume reload in between, never sees
    the sampler's new groups and fails the round ("0/32 groups after
    900 s")."""
    from .buffer_rollout import READER_FUNCTION_PATHS

    store = getattr(getattr(checkpoint, "controller", None), "store", None)
    if getattr(store, "cut_commits_beside_buffer_reads", False):
        return  # a staged object store: the buffer reader never waits on the cut (round_cut._defer)
    if checkpoint is not None and custom_function_path(
            getattr(miles_args, "rollout_function_path", None)) in READER_FUNCTION_PATHS:
        checkpoint.background_commit = False


def _cut_interval_of(wiring: Any, miles_args: Any) -> Any:
    """The resume wiring's interval, else a timed one (``--rl-cut-auto-mtbf-s`` on an
    ``--rl-elastic`` run), else None (every N rounds)."""
    interval = getattr(wiring, "interval", None)
    auto = getattr(miles_args, "yeto_rl_cut_auto", None)
    if interval is not None or not auto:
        return interval
    from yeto.rl.engine.cut_interval import cut_interval_for

    return cut_interval_for(every=None, auto_mtbf_s=auto["auto_mtbf_s"],
                            max_interval_s=auto.get("max_interval_s"))


def _wire_sample_buffer(checkpoint: Any, *, driver: Any, miles_args: Any) -> None:
    """A trainer whose rollout reads the durable sample buffer writes the
    consumed watermark into each round cut and continues there after a restore
    (also a convertible node, whose rollout function reads in trainer mode)."""
    from .buffer_rollout import READER_FUNCTION_PATHS, wire_sample_buffer

    if custom_function_path(getattr(miles_args, "rollout_function_path", None)) not in READER_FUNCTION_PATHS:
        return
    if checkpoint is None:
        raise RuntimeError("a trainer that reads the sample buffer needs round cuts "
                           "(a checkpoint store and no outer syncer) to keep its watermark")
    wire_sample_buffer(checkpoint, driver.rollout)


def run_commanded_trainer(driver: Any) -> Any:
    """One incarnation of a trainer the head's scheduler starts and stops
    (``trainer_remote``): commands come through the round-cut store."""
    from yeto.rl.engine.trainer_remote import run_commanded_island

    controller = getattr(getattr(driver.sync, "round_cuts", None), "controller", None)
    store = getattr(controller, "store", None)
    if store is None:
        raise RuntimeError("a trainer started by the head's scheduler needs round cuts "
                           "(a checkpoint store and no outer syncer)")
    run_commanded_island(driver, store=store)
    drain = getattr(driver.publisher, "drain_releases", None)
    if callable(drain):  # a staged policy release still uploading reaches the store
        drain()
    return driver.published_state


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


def lora_perturber(driver: Any) -> Callable[[float | None], Awaitable[None]]:
    """TEST ONLY (``YETO_RL_TEST_INJECT_LORA_PERTURB``): ``await perturb(eps)``
    applies the published LoRA adapter + eps to the trainer (optimizer state and
    local step preserved); ``await perturb(None)`` applies the saved original back
    exactly. Used around one member-scoped update_weights. It is a coroutine
    because the publisher calls it inside its running event loop, where the
    synchronous ``policy_state.export/apply`` (``LoopRunner.run_until_complete``)
    raise "This event loop is already running"; it uses the async
    ``aexport``/``aapply`` when the policy state has them."""
    from dataclasses import replace

    saved: dict[str, Any] = {}
    ps = driver.policy_state

    async def _export() -> Any:
        return await ps.aexport() if hasattr(ps, "aexport") else ps.export()

    async def _apply(state: Any) -> None:
        kw = {"optimizer": "preserve", "local_step": int(driver.local_step)}
        if hasattr(ps, "aapply"):
            await ps.aapply(state, **kw)
        else:
            ps.apply(state, **kw)

    async def perturb(scale: float | None) -> None:
        if scale is not None:
            state = await _export()
            saved["state"] = state
            tensors = {name: value + float(scale) for name, value in state.tensors.items()}
            await _apply(replace(state, tensors=tensors, _lora=None))
        else:
            state = saved.pop("state")
            await _apply(state)
            if (await _export()).policy_tensor_hash() != state.policy_tensor_hash():
                raise RuntimeError("LoRA perturbation injection: trainer not restored exactly")

    return perturb


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
    """The startup bundle map on the rollout pool (bind_members /
    member_gpus) and ``MilesTrainerOps`` behind ``IslandController.trainer_edges``.

    Skipped (trainer edges stay refused) without the SwappableActor proxy,
    without pool GPU ids (``ElasticWiring.pool_gpus``), when the trainer-resize modules are
    not in the tree, or when the startup views cannot be read. Returns whether
    the edges were wired. Attested trainer edges are still required per edge.
    """
    from .trainer_rebuild import SwappableActor, rebuild_preconditions

    controller = elastic.controller
    if rebuild_preconditions(miles_args):
        # a trainer edge rebuilds the trainer; not possible on this run
        return False
    if (not isinstance(actor_model, SwappableActor) or elastic.pool_gpus is None
            or not callable(getattr(controller, "set_trainer_edges", None))):
        return False
    try:
        from .trainer_resize import MilesTrainerOps
    except ImportError:  # trainer resize not integrated in this tree
        return False
    from .bundles import StartupBundles
    from .rebuild_wiring import CutSource

    if manager is None:
        from miles.utils.workers.ray_worker_manager import RayWorkerManager

        manager = RayWorkerManager.get_handle()
    views = _startup_views(manager, runner)
    gpus_per_node = getattr(launch.placement, "gpus_per_node", None)
    try:
        bundles = StartupBundles(pool_gpus=elastic.pool_gpus, views=views,
                                 placement_map=_role_map(launch.placement),
                                 gpus_per_node=gpus_per_node,
                                 node_resolver=_ray_bundle_node,
                                 head_node=_ray_head_node()[0] if gpus_per_node is not None else None)
    except Exception:  # noqa: BLE001 - no usable map: leave trainer edges refused
        if gpus_per_node is not None:
            raise  # a multi-node island fails closed, never silently
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
    """Run event carrying the algorithm identity and its provenance.

    ``rl/algorithm_absorbed_flags`` lists flags absorbed from extra argv;
    ``rl/unverified_mechanisms`` the unverified-mechanism allowances (an artifact of
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


def connect_island_ray(*, environ=None, ray_module=None, miles_args=None) -> str | None:
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
    rollout metadata hook in the workers reports the data cursor;
    ``YETO_RL_BACKEND=miles`` so neutral rollout-side code finds this adapter.
    """

    environ = os.environ if environ is None else environ
    quant_env = rollout_quant_env(miles_args, environ)  # FP8 tier 1: checked before any engine starts
    address = environ.get("RAY_ADDRESS")
    if not address:
        environ.update(quant_env)  # a Ray the driver starts inherits the driver's environment
        return None
    if ray_module is None:
        import ray as ray_module
    if ray_module.is_initialized():
        raise RuntimeError(
            "Ray was initialized before the ports island pinned RAY_ADDRESS; "
            "its actors could resolve the wrong Ray instance"
        )
    from yeto.rl.engine.backends import BACKEND_ENV

    # The rollout actors and codex
    # subprocesses look up the backend's rollout-metadata port by this name.
    env_vars = {"RAY_ADDRESS": address, BACKEND_ENV: "miles"}
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

    for key in CUT_INJECTION_ENVS:  # TEST ONLY: rank-side switches, off unless set
        if environ.get(key):
            env_vars[key] = environ[key]

    if environ.get(TOOL_DELAY_ENV):  # test tool-wait workload runs in Ray workers
        env_vars[TOOL_DELAY_ENV] = environ[TOOL_DELAY_ENV]
    from yeto.island_overrides import NEGATIVE_RUN_ENV, OVERRIDE_ENV

    for key in (OVERRIDE_ENV, NEGATIVE_RUN_ENV):  # launch-preflight-guards: negative-test island
        if environ.get(key):  # (identity_test_salt is read where the sync client is built)
            env_vars[key] = environ[key]
    from yeto.rl.engine.resume import STORE_ENV_PREFIX, STORE_IMPL_ENV

    from .buffer_rollout import ISLAND_ENVS as SAMPLE_BUFFER_ENVS

    for key in (*SAMPLE_BUFFER_ENVS, STORE_IMPL_ENV):  # the buffer rollout functions
        if environ.get(key):  # open the store inside the rollout executor actor
            env_vars[key] = environ[key]
    for key in environ:  # the store's own settings (yeto.cloud.*: Volume name, s3 URI/credentials)
        if key.startswith(STORE_ENV_PREFIX) and environ[key]:
            env_vars[key] = environ[key]
    if environ.get(ELASTIC_METADATA_ENV) == "1":
        # --rl-elastic: the rollout metadata hook runs inside Ray workers, which
        # inherit the raylet's environment, not the driver's.
        env_vars[ELASTIC_METADATA_ENV] = "1"
    env_vars.update(quant_env)
    ray_module.init(address=address, runtime_env={"env_vars": env_vars})
    return address


def rollout_quant_env(miles_args: Any, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """FP8 tier 1: the SGLang ignore list for ``miles_args.hf_checkpoint``, after the
    scope check (:func:`yeto.rl.quant.tiers.online_plan` raises before any engine starts).
    Empty for every other tier and for a trainer-only island (no rollout engine).

    With the trainer FP8 source (``yeto_rollout_quant_fp8_source``) the engines start on
    Miles' block-FP8 conversion instead (:func:`prepare_native_fp8_base`, which sets
    ``miles_args.rollout_hf_checkpoint``); its config carries the ignore list, so no
    environment entry is needed."""
    from yeto.rl.quant import tiers

    if miles_args is None or getattr(miles_args, "yeto_rollout_quant_tier", 0) != tiers.TIER_ONLINE_FP8:
        return {}
    plan = tiers.online_plan(Path(miles_args.hf_checkpoint))
    print(f"[rl] {tiers.describe(tiers.TIER_ONLINE_FP8)}: {json.dumps(plan.summary())}", flush=True)
    if getattr(miles_args, "yeto_rollout_quant_fp8_source", tiers.FP8_SOURCE_ENGINE) == tiers.FP8_SOURCE_TRAINER:
        environ = os.environ if environ is None else environ
        prepare_native_fp8_base(miles_args, plan, Path(environ.get("YETO_RL_WORK_DIR") or Path.home() / "yeto-rl"))
        return {}
    return tiers.sglang_online_env(plan)


def prepare_native_fp8_base(miles_args: Any, plan: Any, work_dir: Path, *, run: Any = subprocess.run) -> Path:
    """FP8 tier 1, trainer source: convert the BF16 base with Miles' own converter
    (``tools/convert_hf_to_fp8.py``, 128x128 blocks) keeping the plan's ignore list in BF16,
    check the result against the scope rule, and point the engines at it
    (``--rollout-hf-checkpoint``). The trainer keeps ``hf_checkpoint`` (BF16) and Miles
    block-quantizes each weight update by the converted config (``quantize_params_fp8``)."""
    from yeto.rl.quant import tiers

    if not hasattr(miles_args, "rollout_hf_checkpoint"):
        raise RuntimeError("this Miles has no --rollout-hf-checkpoint (needs a newer fork commit)")
    work_dir.mkdir(parents=True, exist_ok=True)
    ignore_file, out = work_dir / "fp8-native-ignore.json", work_dir / "fp8-native-base"
    ignore_file.write_text(json.dumps(tiers.native_ignore_list(plan)))
    converter = Path.home() / "miles" / "tools" / "convert_hf_to_fp8.py"
    run([sys.executable, str(converter), "--model-dir", str(miles_args.hf_checkpoint), "--save-dir", str(out),
         "--strategy", "block", "--block-size", *map(str, tiers.NATIVE_BLOCK),
         "--ignored-layers-file", str(ignore_file)], check=True, stdout=subprocess.DEVNULL)
    count = tiers.check_native_base(out)
    print(f"[rl] {tiers.describe(tiers.TIER_ONLINE_FP8)} trainer source: {miles_args.hf_checkpoint} -> {out} "
          f"({count} FP8 modules)", flush=True)
    from .island_args import override_native

    override_native(miles_args, rollout_hf_checkpoint=str(out))  # a native Miles field: before the slice
    return out


# Determinism environment (with Megatron --deterministic-mode); the table row
# lives in the core (yeto.rl.engine.determinism). It reaches every rank through
# connect_island_ray.
from yeto.rl.engine.determinism import determinism_env as _determinism_env  # noqa: E402

DETERMINISM_ENV = _determinism_env("miles", "nvidia")


def resolve_declared_cells(inference_controller: Any, runner: Any,
                           explicit: Any = ()) -> tuple[str, ...]:
    """The fork cell ids the elastic verbs manage.

    With a fork that lists its declared cells (``describe_cells``), each
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
                             "declared cells (describe_cells)")
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
        # 3.7 restart recovery budget (--rl-elastic-max-recovery-attempts)
        **({"max_recovery_attempts": int(config["max_recovery_attempts"])}
           if config.get("max_recovery_attempts") is not None else {}),
        # off-island copy of the state dir (--rl-elastic-checkpoint-store)
        **({"checkpoint_store": config["checkpoint_store"]} if config.get("checkpoint_store") else {}),
        # 4.7: pool GPU ids (manifest resources.gpus, in logical-bundle order), only
        # with trainer edges; every other elastic run keeps the described pool.
        **({"pool_gpus": manifest_pool_gpus(config["resources"])}
           if config.get("trainer_edges") else {}),
        **_elastic_timeouts(config),
        # absent = legacy
        **({"island_scheduling": config["island_scheduling"]}
           if config.get("island_scheduling") else {}),
    )


def _elastic_timeouts(config: Any) -> dict:
    """``--rl-elastic-drain-timeout-s`` / ``-recovery-timeout-s`` -> controller
    ``Timeouts`` (only when given; otherwise build_elastic's defaults)."""
    given = {k: float(config[f"{k}_timeout_s"]) for k in ("drain", "recovery")
             if config.get(f"{k}_timeout_s") is not None}
    if not given:
        return {}
    from dataclasses import replace

    from yeto.rl.engine.controller import Timeouts

    return {"timeouts": replace(Timeouts(), **given)}


def uses_miles_router(miles_args: Any) -> bool:
    """Upstream (#4322) selects the router with ``--router {miles,...}``;
    ``--use-miles-router`` is an alias of ``--router miles`` and the namespace
    has no ``use_miles_router`` any more."""
    return getattr(miles_args, "router", None) == "miles"


def custom_function_path(value: Any) -> str | None:
    """The dotted path of a ``*-path`` argument. Upstream (#3522) parses every
    custom-function flag into a ``CustomFunctionConfig`` (``.path``); a plain
    string is what an older pin, or a test namespace, holds."""
    if value is None:
        return None
    path = getattr(value, "path", value)
    return str(path) if path else None


def check_elastic_miles_args(miles_args: Any) -> None:
    """Miles preconditions of the fork verbs the elastic controller uses, refused before
    Ray: cordon / drain_cells / admit_cells / cordoned update_weights need the
    Miles router; member publication needs a resident partitioned rollout."""
    problems = []
    if not uses_miles_router(miles_args):
        problems.append("--router miles (fork cordon/drain/admit_cordoned need the Miles router)")
    if getattr(miles_args, "colocate", False):
        problems.append("no --colocate (elastic needs a fixed partition)")
    if getattr(miles_args, "offload_rollout", False):
        problems.append("no rollout offload (member publication needs resident engines)")
    if problems:
        raise ValueError("--rl-elastic needs " + "; ".join(problems))


def island_layout_of(miles_args: Any, topology: Any, placement: Any = None) -> dict[str, Any]:
    """The parallel layout this learner launches: the
    Megatron parallel sizes and actor shape from ``miles_args`` plus the role ->
    logical bundle map of the placement request (None = leading-bundle layout).
    Journaled with the ``topology`` record and compared against the baseline on a
    rebuild: a cut written under another layout is not restorable without conversion."""
    def size(name: str, default: int = 1) -> int:
        return int(getattr(miles_args, name, None) or default)

    nodes = size("actor_num_nodes", 1)
    per = size("actor_num_gpus_per_node", 0)
    layout: dict[str, Any] = {
        "tp": size("tensor_model_parallel_size"), "pp": size("pipeline_model_parallel_size"),
        "cp": size("context_parallel_size"), "ep": size("expert_model_parallel_size"),
        "nodes": int(topology.nodes), "gpus_per_node": int(topology.gpus_per_node),
    }
    if per:
        layout["trainer"] = nodes * per
    bundle_map = getattr(placement, "bundle_map", None) if placement is not None else None
    layout["bundle_map"] = None if bundle_map is None else {str(k): list(v) for k, v in bundle_map.items()}
    # ruling 2026-10-04 v2: the explicit cross-node TP opt-ins are part of the layout
    # (journal topology.layout.cross_node_tp / cross_node_engine_tp) and WARN at startup
    cross_tp = bool(getattr(placement, "allow_cross_node_tp", False)
                    or getattr(miles_args, "yeto_rl_allow_cross_node_tp", False))
    cross_engine = bool(getattr(placement, "allow_cross_node_engine_tp", False)
                        or getattr(miles_args, "yeto_rl_allow_cross_node_engine_tp", False))
    layout["cross_node_tp"] = int(cross_tp)
    layout["cross_node_engine_tp"] = int(cross_engine)
    if cross_tp:
        logging.getLogger(__name__).warning(
            "cross_node_tp=1: the trainer TP*CP group (tp=%s cp=%s) may span nodes "
            "(--rl-allow-cross-node-tp); NCCL TP collectives cross the inter-node fabric",
            layout["tp"], layout["cp"])
    if cross_engine:
        logging.getLogger(__name__).warning(
            "cross_node_engine_tp=1: rollout engines may span whole nodes (sglang nnodes>1, "
            "--rl-allow-cross-node-engine-tp); engines scale as whole replicas")
    return layout


def refuse_partial_island_preflight(elastic: Any, topology: Any, miles_args: Any, *,
                                    node_probe: Callable[[], Any] | None = None,
                                    placement: Any = None) -> None:
    """Multi-node startup precondition, run right
    after ``connect_island_ray()`` and before any Miles placement group exists: the
    controller writes its ``topology`` journal record from the live Ray node table and,
    with fewer alive nodes than declared, enters RECOVERY_REQUIRED; the learner then
    emits ``rl_reconfiguration`` (RECOVERY_REQUIRED) on the tape and exits non-zero
    instead of blocking on a PENDING placement group. No-op without elastic/topology."""
    controller = getattr(elastic, "controller", None) if elastic is not None else None
    if controller is None or topology is None or int(topology.nodes) <= 1:
        return
    if not callable(getattr(controller, "refuse_partial_island", None)):
        return
    controller.set_topology((int(topology.nodes), int(topology.gpus_per_node)),
                            node_probe or _ray_alive_nodes,
                            layout=island_layout_of(miles_args, topology, placement))
    why = controller.refuse_partial_island()
    if why is None:
        return
    error = getattr(controller, "recovery_required", None) or f"island topology: {why}"
    try:
        epochs = getattr(getattr(controller, "journal", None), "epochs", None)
        append_tape_event(miles_args, {
            "event": "rl_reconfiguration", "rollout_id": None, "result": "RECOVERY_REQUIRED",
            "error": str(error), "config_epoch": getattr(epochs, "config_epoch", None),
        })
    except Exception as exc:  # noqa: BLE001 - the tape must not mask the refusal
        import logging

        logging.getLogger(__name__).warning(
            "rl_reconfiguration (partial island) not written to the tape: %r", exc)
    raise RuntimeError(f"island is RECOVERY_REQUIRED: {error}")


def _ray_alive_nodes() -> dict[str, int]:
    """``{node_id: GPUs}`` of the alive Ray nodes (the controller's node probe)."""
    import ray

    return {n["NodeID"]: int((n.get("Resources") or {}).get("GPU", 0))
            for n in ray.nodes() if n.get("Alive")}


INCARNATION_MARKER_DIR = "/tmp/yeto-rl-incarnation"


def reconcile_gpu_pool_preflight(elastic: Any, topology: Any, miles_args: Any, *,
                                 resources: Any = None, accept_rebind: bool | None = None,
                                 gpu_probe: Callable[[Any], Any] | None = None,
                                 placement: Any = None,
                                 incarnation_probe: Callable[[Any, Any], Mapping[str, str]] | None = None,
                                 other_islands: Mapping[str, Any] | None = None,
                                 marker_writer: Callable[[Any, Any, str], None] | None = None) -> Any:
    """Multi-node GPU uuid reconciliation (ruling 2026-10-04),
    run right after :func:`refuse_partial_island_preflight` and before any placement
    group: ``nvidia-smi --query-gpu=index,uuid`` is collected on every island node (a
    Ray task per node; an unreachable node fails closed), the pool is reconciled against
    the cfg's uuids and/or the journal's binding baseline
    (:func:`multinode.reconcile_gpu_pool`), and the controller journals ``gpu_pool``.
    A refused pool -> tape ``rl_reconfiguration`` RECOVERY_REQUIRED + ``RuntimeError``.
    A new pool is accepted (and becomes the baseline) only with
    ``--rl-elastic-accept-rebind``. Single-node islands are untouched (None)."""
    controller = getattr(elastic, "controller", None) if elastic is not None else None
    if controller is None or topology is None or int(topology.nodes) <= 1:
        return None
    if not callable(getattr(controller, "record_gpu_pool", None)):
        return None
    from yeto.rl.engine import multinode as mn

    config = getattr(miles_args, "yeto_rl_elastic", None) or {}
    if resources is None:
        resources = config.get("resources")
    if accept_rebind is None:
        accept_rebind = bool(config.get("accept_rebind", False))
    topo = mn.Topology(int(topology.nodes), int(topology.gpus_per_node))

    def refuse(error: str) -> None:
        try:
            epochs = getattr(getattr(controller, "journal", None), "epochs", None)
            append_tape_event(miles_args, {
                "event": "rl_reconfiguration", "rollout_id": None, "result": "RECOVERY_REQUIRED",
                "error": str(error), "config_epoch": getattr(epochs, "config_epoch", None),
            })
        except Exception as exc:  # noqa: BLE001 - the tape must not mask the refusal
            import logging

            logging.getLogger(__name__).warning(
                "rl_reconfiguration (gpu_pool) not written to the tape: %r", exc)
        raise RuntimeError(f"island is RECOVERY_REQUIRED: {error}")

    try:
        if resources is not None and not isinstance(resources, dict):
            resources = json.loads(Path(resources).expanduser().read_text(encoding="utf-8"))
        cfg_pool = mn.declared_gpu_pool(resources or {}, topo)
        observed = mn.observed_gpu_pool((gpu_probe or _ray_gpu_uuids)(topo), topo)
    except Exception as exc:  # noqa: BLE001 - probe/cfg failure: fail closed, journaled
        error = f"gpu_pool: GPU probe failed: {exc}"
        controller.record_gpu_pool(mn.ReconcileResult(ok=False, observed=(), error=str(exc)),
                                   source="probe", accept_rebind=bool(accept_rebind))
        refuse(getattr(controller, "recovery_required", None) or error)
    baseline = controller.gpu_pool_baseline()
    declared = mn.merge_declared_pool(cfg_pool, baseline)
    cfg_named = any(u is not None for node in cfg_pool for u in node)
    source = "+".join(n for n, on in (("cfg", cfg_named), ("journal", baseline is not None)) if on) or "none"
    result = mn.reconcile_gpu_pool(declared, observed, accept_rebind=bool(accept_rebind))
    roles: dict[str, str] = {}
    if result.ok:
        # Ruling 2026-10-04 v2, rebind safety: (a) duplicate occupation, (b) role conflict,
        # (c) stale re-entry of an older incarnation's processes on the same GPUs.
        mine = str((getattr(controller, "incarnation", None) or {}).get("id", ""))
        why3 = None
        try:
            if placement is not None:
                counts = (int(getattr(placement, "trainer_gpus", 0) or 0),
                          int(getattr(placement, "rollout_gpus", 0) or 0),
                          int(getattr(placement, "standby_gpus", 0) or 0))
                roles = mn.role_uuid_map(getattr(placement, "bundle_map", None), result.flat, counts=counts)
            why3 = mn.occupation_rejection(result.flat, roles or {u: "unassigned" for u in result.flat},
                                           other_islands)
            if why3 is None:
                live = (incarnation_probe or _ray_live_incarnations)(topo, result.flat)
                why3 = mn.stale_incarnation_rejection(
                    tuple(u for node in (baseline or ()) for u in node), result.flat, live, mine)
        except Exception as exc:  # noqa: BLE001 - fail closed on an unknown state
            why3 = f"rebind safety check failed: {exc}"
        if why3 is not None:
            result = mn.ReconcileResult(ok=False, observed=result.observed, rebind=result.rebind,
                                        mapping=result.mapping, diffs=result.diffs, error=why3)
    why = controller.record_gpu_pool(result, source=source, accept_rebind=bool(accept_rebind), roles=roles)
    if why is not None:
        refuse(getattr(controller, "recovery_required", None) or f"gpu_pool: {why}")
    mine = str((getattr(controller, "incarnation", None) or {}).get("id", ""))
    if mine:
        try:  # mark every bound GPU with this incarnation (read back by the next incarnation's (c))
            (marker_writer or _ray_write_incarnation_markers)(topo, result.observed, mine)
        except Exception as exc:  # noqa: BLE001 - a missing marker only weakens the next check
            logging.getLogger(__name__).warning("incarnation markers not written: %r", exc)
    return result


def _marker_rows(uuids: Sequence[str], marker_dir: str = INCARNATION_MARKER_DIR) -> dict[str, str]:
    """uuid -> incarnation id for every marker in ``marker_dir`` whose owning pid is alive."""
    import os as _os

    out: dict[str, str] = {}
    for uuid in uuids:
        path = Path(marker_dir) / f"{uuid}.json"
        if not path.exists():
            continue
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            pid = int(info.get("pid", 0))
            _os.kill(pid, 0)  # raises when the process is gone
        except (OSError, ValueError, TypeError):
            continue
        out[uuid] = str(info.get("incarnation", ""))
    return out


def _write_markers(uuids: Sequence[str], incarnation: str, pid: int,
                   marker_dir: str = INCARNATION_MARKER_DIR) -> None:
    Path(marker_dir).mkdir(parents=True, exist_ok=True)
    for uuid in uuids:
        (Path(marker_dir) / f"{uuid}.json").write_text(
            json.dumps({"incarnation": incarnation, "pid": int(pid)}), encoding="utf-8")


def _ray_live_incarnations(topology: Any, observed_flat: Sequence[str]) -> dict[str, str]:
    """(c) stale re-entry probe: per-node Ray tasks read the incarnation marker files
    (``INCARNATION_MARKER_DIR/<uuid>.json``, written by :func:`_ray_write_incarnation_markers`)
    and keep those whose recorded pid is still alive on that node. Without ``ray``
    importable there is no Ray cluster (no old Ray process can hold a GPU): {}."""
    try:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    except ImportError:
        return {}

    nodes = [n for n in ray.nodes() if n.get("Alive") and (n.get("Resources") or {}).get("GPU", 0)]

    @ray.remote(num_cpus=0, num_gpus=0)
    def _read(uuids: list[str]) -> dict[str, str]:
        return _marker_rows(uuids)

    refs = [_read.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        node_id=n["NodeID"], soft=False)).remote(list(observed_flat)) for n in nodes]
    live: dict[str, str] = {}
    for rows in ray.get(refs, timeout=120):
        live.update(rows)
    return live


def _ray_write_incarnation_markers(topology: Any, observed: Sequence[Sequence[str]], incarnation: str) -> None:
    """Write this incarnation's marker for every bound GPU on its node (the learner pid on
    the head stands for the whole incarnation: its Ray job owns every worker process)."""
    import os as _os

    try:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    except ImportError:
        return
    nodes = [n for n in ray.nodes() if n.get("Alive") and (n.get("Resources") or {}).get("GPU", 0)]
    head_id, _ip = _ray_head_node(nodes)
    order = sorted(nodes, key=lambda n: (0 if n["NodeID"] == head_id else 1,
                                         str(n.get("NodeManagerAddress")), str(n["NodeID"])))
    pid = _os.getpid()

    @ray.remote(num_cpus=0, num_gpus=0)
    def _write(uuids: list[str], inc: str, owner: int) -> None:
        import os as _o

        _write_markers(uuids, inc, _o.getppid() if owner < 0 else owner)

    refs = [_write.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        node_id=n["NodeID"], soft=False)).remote(list(uuids), incarnation, pid)
            for n, uuids in zip(order, observed)]
    ray.get(refs, timeout=120)


def island_smi_rows(out: str, use: str | None = None) -> list[tuple[int, str]]:
    """``nvidia-smi --query-gpu=index,uuid`` csv -> ``[(index, uuid)]`` of the island's
    GPUs. nvidia-smi ignores ``CUDA_VISIBLE_DEVICES`` and always lists every card of
    the machine; under ``--rl-island-use-gpus-per-node M`` (the launcher exports
    ``YETO_ISLAND_USE_GPUS_PER_NODE=M`` before ``ray start``, inherited by Ray workers)
    only local GPUs ``0..M-1`` belong to the island, the rest are unallocated."""
    import os

    rows = [(int(a.strip()), b.strip()) for a, b in
            (line.split(",", 1) for line in out.splitlines() if line.strip())]
    use = os.environ.get("YETO_ISLAND_USE_GPUS_PER_NODE") if use is None else use
    if use:
        rows = [r for r in rows if r[0] < int(use)]
    return rows


def _ray_gpu_uuids(topology: Any) -> list[list[tuple[int, str]]]:
    """``[(index, uuid), ...]`` per island node in logical order (Ray head first,
    then the workers by address/node id), each collected by a Ray task pinned to that
    node (``NodeAffinitySchedulingStrategy``, hard). A node whose ``nvidia-smi`` count
    disagrees with its Ray ``GPU`` resource, or a node that cannot run the task, raises
    (fail closed)."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    nodes = [n for n in ray.nodes() if n.get("Alive")]
    head_id, _ip = _ray_head_node(nodes)
    order = sorted(nodes, key=lambda n: (0 if n["NodeID"] == head_id else 1,
                                         str(n.get("NodeManagerAddress")), str(n["NodeID"])))
    order = [n for n in order if n["NodeID"] == head_id or (n.get("Resources") or {}).get("GPU", 0)]
    if len(order) != int(topology.nodes):
        raise RuntimeError(f"{len(order)} alive Ray nodes with GPUs, topology has {topology.nodes}")

    @ray.remote(num_cpus=0, num_gpus=0)
    def _smi() -> list[tuple[int, str]]:
        import subprocess

        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             check=True, capture_output=True, text=True, timeout=60).stdout
        return island_smi_rows(out)

    refs = [_smi.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
        node_id=n["NodeID"], soft=False)).remote() for n in order]
    rows = ray.get(refs, timeout=120)
    for n, got in zip(order, rows):
        ray_gpus = int((n.get("Resources") or {}).get("GPU", 0))
        if len(got) != ray_gpus:
            raise RuntimeError(f"node {n['NodeID']} nvidia-smi lists {len(got)} GPUs, Ray exposes {ray_gpus}")
    return [list(r) for r in rows]


def attach_engine_outage(driver: Any, topology: Any, *, environ: Mapping[str, str] | None = None) -> Any:
    """With ``YETO_ISLAND_ENGINE_RECOVERY_S`` (the launcher sets it
    for a two-cluster island) the driver survives the loss of its engine nodes
    (:mod:`yeto.rl.engine.engine_outage`, :mod:`.engine_recovery`). Unset: nothing."""
    from yeto.rl.engine.engine_outage import arm_engine_outage, outage_budget_s

    budget = outage_budget_s(os.environ if environ is None else environ)
    if budget is None or topology is None or int(topology.nodes) <= 1:
        return None
    from .engine_recovery import MilesEngineRecovery

    return arm_engine_outage(driver, MilesEngineRecovery(driver.rollout, nodes=int(topology.nodes)),
                             budget_s=budget, nodes=int(topology.nodes))


def wait_island_nodes(nodes: int) -> dict[str, Any] | None:
    """:func:`yeto.rl.engine.multinode.wait_for_island_nodes` on this Ray."""
    from yeto.rl.engine.multinode import ISLAND_NODE_WAIT_ENV, wait_for_island_nodes

    if not os.environ.get(ISLAND_NODE_WAIT_ENV):
        return None
    import ray

    return wait_for_island_nodes(int(nodes), environ=os.environ, ray_nodes=ray.nodes,
                                 clock=time.monotonic, sleep=time.sleep,
                                 log=lambda line: print(line, flush=True))


def _ray_head_node(nodes: Any = None) -> tuple[Any, str]:
    """``(node_id, ip)`` of the Ray head: the one alive node carrying Ray's
    built-in ``node:__internal_head__`` resource (head pin). Exactly one such node, else ``RuntimeError`` (fail closed)."""
    from yeto.rl.engine.multinode import HEAD_RESOURCE

    if nodes is None:
        import ray

        nodes = ray.nodes()
    heads = [n for n in nodes if n.get("Alive") and HEAD_RESOURCE in (n.get("Resources") or {})]
    if len(heads) != 1:
        raise RuntimeError(f"expected exactly one alive Ray head node ({HEAD_RESOURCE}), found {len(heads)}")
    return heads[0]["NodeID"], str(heads[0].get("NodeManagerAddress"))


def pin_placement_group_to_head(gpus_per_node: int, *, pg_module: Any = None, head: Any = None,
                                table: Any = None) -> Any:
    """Make the fork's
    startup placement group put logical node 0 (the trainer's block) on the Ray
    head, and refuse startup when it did not.

    Patches ``miles.ray.placement_group`` in the driver before the
    ``RayWorkerManager`` is launched: ``placement_group`` (the Ray call) gets
    :func:`head_pinned_bundles`, ``sort_key`` sorts the head's bundles first,
    and ``_create_placement_group`` checks ``placement_group_table`` afterwards
    (block 0 of the reordered bundles on the head node, else ``RuntimeError``).
    Returns the head node id. ``pg_module``/``head``/``table`` are test seams."""
    from yeto.rl.engine.multinode import assert_head_block, head_first_sort_key, head_pinned_bundles

    if pg_module is None:
        import importlib

        pg_module = importlib.import_module("miles.ray.placement_group")
    head_id, head_ip = head if head is not None else _ray_head_node()
    ray_placement_group, base_key, base_create = (
        pg_module.placement_group, pg_module.sort_key, pg_module._create_placement_group)
    if getattr(base_create, "_yeto_head_pinned", None) is not None:
        return base_create._yeto_head_pinned  # already installed (one PG per driver)

    def pinned_placement_group(bundles, *args, **kwargs):
        return ray_placement_group(head_pinned_bundles(len(bundles), gpus_per_node), *args, **kwargs)

    def checked_create(num_gpus, *args, **kwargs):
        info = base_create(num_gpus, *args, **kwargs)
        pg, reordered = info[0], list(info[1])
        if pg is None:
            return info
        if table is None:
            import ray

            node_of_bundle = ray.util.placement_group_table(pg)["bundles_to_node_id"]
        else:
            node_of_bundle = table(pg)
        blocks = [node_of_bundle[b] for b in reordered[:gpus_per_node]]
        if len(set(map(str, blocks))) != 1:
            raise RuntimeError(f"head pin: logical node 0 bundles span nodes {sorted(set(map(str, blocks)))}")
        assert_head_block(blocks, head_id)
        return info

    checked_create._yeto_head_pinned = head_id
    pg_module.placement_group = pinned_placement_group
    pg_module.sort_key = head_first_sort_key(head_ip, base_key)
    pg_module._create_placement_group = checked_create
    return head_id


def _ray_bundle_node(pg: Any, bundle: int) -> Any:
    """The Ray node id hosting ``bundle`` of ``pg``
    (``ray.util.placement_group_table``), the runtime source of the node-block
    assertion; raises when Ray cannot tell (fail closed)."""
    import ray

    table = ray.util.placement_group_table(pg)
    return table["bundles_to_node_id"][bundle]


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


TRAINER_ACTOR_INIT_SUFFIX = "TrainRayActor.__init__"


def trainer_actor_import_done_unix(list_tasks: Callable[..., Any] | None = None) -> float | None:
    """When the last trainer actor finished its creation task (Unix seconds).

    Ray runs the actor class import and ``__init__`` in the actor creation
    task; Miles' trainer ``__init__`` is light (the model is built in
    ``init``), so its end is the import end.  Read from the Ray state API
    (dashboard on, see the island run script); ``None`` when Ray has no such
    record.  Unverified on GPU.
    """

    try:
        if list_tasks is None:
            from ray.util.state import list_tasks
        tasks = list_tasks(filters=[("type", "=", "ACTOR_CREATION_TASK")], detail=True,
                           limit=10000, raise_on_missing_output=False)
    except Exception:  # noqa: BLE001 - a missing record is reported as a missing phase
        return None
    ends = [
        getattr(task, "end_time_ms", None) for task in tasks
        if str(getattr(task, "name", "")).endswith(TRAINER_ACTOR_INIT_SUFFIX)
    ]
    ends = [end for end in ends if end]
    return max(ends) / 1000.0 if ends else None


def run_ports_island(
    miles_args: Any,
    launch: Any,
    algorithm: AlgorithmSpec,
    *,
    learner_id: int,
    base_model_revision: str,
    lora_config_hash: str,
    layout_hash: str | None,
    yeto_policy_sync: bool,
    harness_preflight: HarnessPreflight | None = None,
):
    """Run one ports-path island to completion on real upstream Miles.

    ``harness_preflight``: injectable hook run by :func:`preflight_stage`
    before Ray is connected and before any placement/model allocation.
    """

    from yeto.rl import MILES_NEXT_COMMIT

    from .state import require_run_plugin

    require_run_plugin()  # before any upstream component or model exists
    from yeto.rl.engine.overlap import loop_eval_starter

    publish_member_cell(miles_args, learner_id)  # island cell -> harness member key
    fingerprint, capabilities, profile, elastic = preflight_stage(
        miles_args, launch, algorithm, yeto_policy_sync=yeto_policy_sync,
        harness_preflight=harness_preflight,
    )  # contract preflight(...) + harness preflight: before connect_island_ray()
    if getattr(miles_args, "yeto_rl_resume", None):
        # the round cut's backend fingerprint (the elastic controller gets the same one;
        # a cut without it is refused "runtime: backend_fingerprint missing")
        miles_args.yeto_rl_resume["runtime_fingerprint"] = fingerprint

    from . import eval_wiring

    eval_guard = eval_wiring.eval_guard_preflight(miles_args)  # before any GPU action
    from yeto.rl.engine.telemetry import StartupTelemetry

    # fleet-dashboard 8.4: heartbeat/resource sampling from here until driver.run()
    # takes over (same opt-in intervals as the driver's; tape unchanged when unset)
    startup = StartupTelemetry(
        lambda event, **fields: append_tape_event(miles_args, {"event": event, **fields}),
        heartbeat_interval_s=getattr(miles_args, "yeto_rl_heartbeat_interval_s", None),
        resource_interval_s=getattr(miles_args, "yeto_rl_resource_sample_interval_s", None),
    ).__enter__()  # stopped before driver.run() and in the finally below
    from yeto.rl.engine.start_phases import StartPhases

    # submitted / container_ready / setup_done /
    # ray_ready come from the launcher env and the setup shell's mark files.
    start_phases = StartPhases(
        lambda event, **fields: append_tape_event(miles_args, {"event": event, **fields}))
    start_phases.replay()
    connect_island_ray(miles_args=miles_args)
    topology = getattr(launch.placement, "topology", None)
    if topology is not None and topology.nodes > 1:
        # two-cluster island: every node in the Ray before anything below
        # (None without YETO_ISLAND_NODE_WAIT_S: no wait, no event)
        nodes_ready = wait_island_nodes(topology.nodes)
        if nodes_ready is not None:
            append_tape_event(miles_args, {"event": "rl_island_nodes_ready", **nodes_ready})
        # a restarted learner on a partial island (worker node DEAD in the GCS)
        # must fail closed here; a placement group asking for the dead node's GPUs
        # would stay PENDING forever
        refuse_partial_island_preflight(elastic, topology, miles_args, placement=launch.placement)
        # runtime GPU uuid reconciliation, journal gpu_pool; a changed
        # pool is accepted only with --rl-elastic-accept-rebind (fail closed otherwise)
        reconcile_gpu_pool_preflight(elastic, topology, miles_args, placement=launch.placement)
        # trainer block = node 0 = Ray head, fail closed
        pin_placement_group_to_head(topology.gpus_per_node)
    startup.step("ray_connected")
    # fill the remote-code module cache on every island node before any
    # engine or trainer actor imports from it (concurrent first imports raced).
    prewarm_remote_code(miles_args, emit=lambda event: append_tape_event(miles_args, event))

    from miles.ray.placement_group import (
        create_rollout_components,
        create_trainer_handles,
        create_training_models,
    )
    from miles.ray.rollout.eval_dispatch import EvalDispatcher
    from miles.ray.specs.train import compute_trainer_configs
    from miles.utils.async_utils import Disposer
    from miles.utils.orchestration_utils import ArgvOrchestratorStartupInfo, init_orchestration_script

    from .island_args import miles_config
    from .rollout import RayMetadataSink

    append_tape_event(
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
    # Upstream assembly order (miles train.py): the orchestrator works on the
    # OrchestratorConfig slice of the parsed AllConfig; create_training_models
    # writes start_rollout_id back onto that slice, so it is the authority
    # for the resumed rollout id from here on.
    # miles_args may be the driver facade (island_args.IslandArgs, immutable Miles
    # config): Miles' own API takes the configuration behind it.
    startup_info = ArgvOrchestratorStartupInfo.create(miles_config(miles_args))
    orch = startup_info.args

    async def init():
        await disposer.__aenter__()
        capability = init_orchestration_script(startup_info, disposer=disposer)
        handles = create_trainer_handles(
            orch, trainer_configs=compute_trainer_configs(orch), capability=capability)
        controller, executor, _ = await create_rollout_components(
            orch, capability=capability, trainer_handles=handles)
        disposer.add(controller, executor)
        startup.step("engine_ready")
        actor, critic = await create_training_models(orch, executor, handles=handles)
        startup.step("weights_loaded")
        imported = trainer_actor_import_done_unix()
        if imported is None:
            start_phases.missing("actor_import_done", "Ray state API has no trainer actor creation time")
        else:
            start_phases.mark("actor_import_done", imported)
        start_phases.mark("actor_init_done")
        start_phases.forward_setup_events()  # e.g. a late model_prefetch_failed
        # Miles creates a critic iff use_critic
        # (estimator ppo); it must agree with the AlgorithmSpec.
        if (critic is not None) != bool(algorithm.execution.needs_critic):
            raise RuntimeError(
                f"Miles created {'a' if critic is not None else 'no'} critic but the algorithm "
                f"spec says needs_critic={algorithm.execution.needs_critic}"
            )
        # one swappable handle shared by trainer,
        # policy state, publisher and the eval dispatcher (EvalDispatcher keeps
        # self.actor_model and resolves methods per call, so the proxy is
        # enough); the disposer gets the proxy, whose async dispose() resolves
        # the CURRENT target (Disposer.add binds item.dispose when added).
        from .trainer_rebuild import SwappableActor

        actor = SwappableActor(actor)
        disposer.add(actor)
        if critic is not None:
            critic = SwappableActor(critic)
            disposer.add(critic)
        dispatcher = EvalDispatcher(orch, actor, executor)
        disposer.add(dispatcher.drain)
        return controller, executor, actor, critic, dispatcher

    error: BaseException | None = None
    try:
        controller, executor, actor, critic, dispatcher = runner.run(init())
        if not callable(getattr(actor, "run_plugin", None)):
            raise RuntimeError("upstream actor group exposes no run_plugin")

        async def evaluate(rollout_id: int) -> dict[str, float]:
            await controller.prepare_eval()
            await dispatcher.dispatch(rollout_id)
            return {}

        groups = {"actor": actor} if critic is None else {"actor": actor, "critic": critic}

        def build_driver() -> Any:
            """A fresh island driver over the components (one per trainer incarnation
            of a convertible node; once otherwise)."""
            sync, progress = build_sync(
                miles_args, yeto_policy_sync=yeto_policy_sync, runner=runner, groups=groups
            )
            driver = compose_island(
                miles_args=miles_args,
                orchestrator_args=orch,
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
                critic_model=critic,
            )
            # fleet-dashboard 2.1/2.2: opt-in heartbeat / resource sampler periods
            driver.heartbeat_interval_s = getattr(miles_args, "yeto_rl_heartbeat_interval_s", None)
            driver.resource_sample_interval_s = getattr(
                miles_args, "yeto_rl_resource_sample_interval_s", None)
            eval_wiring.attach(driver, miles_args, eval_guard)
            driver.island_tasks = getattr(miles_args, "yeto_rl_island_tasks", None)
            driver.data_slot = ("dispense" if getattr(miles_args, "yeto_rl_data_dispense", None)
                                else getattr(miles_args, "yeto_rl_data_slot", None))
            dispense = getattr(miles_args, "yeto_rl_data_dispense", None) or {}
            if dispense.get("stream_record"):
                driver.data_streams = {"streams": dispense["stream_record"],
                                       "accept": dispense.get("accept")}
            driver.data_positions = data_position_log_for(miles_args, sync, learner_id=learner_id)
            driver.hf_exports = hf_exports_for(miles_args, actor_model=actor, runner=runner)
            driver.code_pack_digest = getattr(miles_args, "yeto_rl_code_pack_digest", None)
            driver.code_pack_mode = getattr(miles_args, "yeto_rl_code_pack_mode", None)
            attach_engine_outage(driver, topology)  # no-op without the env
            driver.start_phases = start_phases  # first_step_done
            return driver

        from .convertible import convertible_node_name

        if convertible_node_name(miles_args) is not None:
            # rollout node until the head converts it
            from .convertible import run_convertible_node

            startup.__exit__(None, None, None)
            return run_convertible_node(miles_args, build_driver=build_driver,
                                        controller=controller, executor=executor, actor=actor,
                                        runner=runner)
        driver = build_driver()
        startup.__exit__(None, None, None)  # the driver's own heartbeat starts in run()
        from yeto.rl.engine import trainer_remote

        if trainer_remote.commanded():  # the head starts/stops us
            return run_commanded_trainer(driver)
        state = driver.run()
        from .sampler_island import mark_trainer_end

        mark_trainer_end(miles_args, learner_id=learner_id)  # its samplers may stop now
        return state
    except BaseException as exc:
        error = exc
        raise
    finally:
        startup.__exit__(None, None, None)
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
