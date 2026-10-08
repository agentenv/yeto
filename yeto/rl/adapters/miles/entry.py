"""``--rl-engine ports`` composition root (tasks 5.1, design D2).

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
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from yeto.rl.engine.algorithm import BOUNDED_NONZERO_STD_FILTER, STOCK_NONZERO_STD_FILTER, AlgorithmSpec
from yeto.rl.engine.capabilities import R0_MECHANISMS, EngineCapabilities, ExecutionCapabilities
from .traits import MILES_TRAITS
from . import LoopRunner
from .config import MilesConfigError, check_harness_reward_scope

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
# fb04d6ff / e3a11ab3 = 2f23a0fc + M5, same basis: `git diff --stat
# 2f23a0fc..e3a11ab3
# -- miles` touches only megatron_utils/lora/dp_invariant_state.py, reached
# only with --lora-dp-invariant-state (default off); default training/loss
# path unchanged (evidence .../2026-09-30-img-e3a11ab).
# 1023269 = e3a11ab3 + M3 + A27 (image-m3a27), same basis: `git diff
# --name-only e3a11ab38..1023269 -- miles miles_plugins` touches
# megatron_utils/model.py (a new is_qwen3_8_next_model LoRA-injection branch
# only), update_weight/hf_weight_iterator_direct.py, utils/lora/*,
# miles_plugins/models/qwen3_8_next/lora.py, sglang_utils/sglang_api_client.py,
# weight_update/protocols/broadcast.py, utils/workers/ray_worker_manager.py and
# parser entries in arguments.py -- no loss_hub/training-loss file (evidence
# .../evidence/ports-image/2026-10-02-m3a27).
_PINS_0AF62F4D_PLUS = frozenset({
    "0af62f4d48ed6a5b185c257578d8f7e22312aa87",
    "5c1b49ebccbc7508c1d9ef89eacc2db3e448b6ba",
    "2f23a0fca9b80f6a7300da401703c343014b03c0",
    "fb04d6ffa30edc28c7ba0a2e88802a84bbbd28f9",
    "e3a11ab38cbb7fd911b23fdd62a4eb6dfbb1c841",
    "1023269412bf4e54a1d95c8d2deaee795871aa72",
    # c35702e = 1023269 + A27-2 (857fc9592): megatron_utils/actor.py, weight_update/
    # updater.py + protocols/broadcast.py, ray/train/{cell,group}.py,
    # utils/workers/worker_handle.py -- engine-failure propagation only, no loss path
    # (evidence .../evidence/ports-image/2026-10-02-m3a27b).
    "c35702eefcf2862cee155e46870e6ad30568d2c6",
    # 8bc52237a = c35702e + s16-raw-lora-disagg (weight_update/protocols/broadcast.py placement +
    # megatron_utils/actor.py guard + update_weight/hf_weight_iterator.py comment): no loss path.
    "8bc52237a1102306abd8f89a2ea2090aa2df6850",
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

    from yeto.rl.engine.capabilities import R0_MECHANISMS

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
    ``execution.needs_critic``) trains actor + critic -> ``"ppo"``
    (rl-algo-critic-family D4).
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
    from yeto.rl.engine.execution_profile import ExecutionProfile, ProfileError, check_overlap_eval

    from yeto.rl.engine.overlap import IMPLEMENTED_OVERLAP

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
    custom = getattr(miles_args, "custom_generate_function_path", None)
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
    """IR-2: where the drain probe / load sample read harness counts from.

    ``elastic.harness_board`` when wired; the island's named ``HarnessBoard``
    actor when a custom agent function (agentic generate) is configured;
    ``HARNESS_NOT_AGENTIC`` (explicit zeros) otherwise. Never None here: a
    pool built with ``harness=None`` reports "harness counts unknown".
    """
    from .rollout import HARNESS_NOT_AGENTIC

    board = getattr(elastic, "harness_board", None) if elastic is not None else None
    if board is not None:
        return board
    if getattr(miles_args, "custom_agent_function_path", None):
        from .elastic_wiring import LazyBoardActor
        from yeto.rl.engine.tool_wait import harness_board_actor

        return LazyBoardActor(int(getattr(miles_args, "yeto_rl_learner_id", 0) or 0),
                              factory=harness_board_actor)
    return HARNESS_NOT_AGENTIC


# IR-1: harness preflight hook. Called by ``run_ports_island`` after the A1
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


# A16 / D5 (rl-fn-codex-rollout 0.6): the island publishes its own INFRA member cell
# so the Codex harness admits sessions under ``rollout.member_id(cell_id)`` instead
# of the global key.  Single-island (--rl-single-island-no-sync) islands publish 0.
MEMBER_CELL_ENV = "YETO_RL_CELL_ID"


def publish_member_cell(miles_args: Any, learner_id: int | None = None, environ: Any = None) -> str:
    """Set ``miles_args.yeto_rl_cell_id`` and export ``YETO_RL_CELL_ID``; return the cell id.

    Source, in order: an explicit ``miles_args.yeto_rl_cell_id`` (INFRA may pre-set it),
    ``learner_id`` (the island number the launcher already assigns; ``island_id`` in the
    ``rl_engine_selected`` event), ``miles_args.yeto_rl_learner_id``, else 0.  The env
    export reaches rollout-worker subprocesses through ``worker_runtime_env`` and
    ``preflight.configure_rollout_worker``; ``preflight.resolve_member`` then yields
    ``engine:<cell>`` (``engine:0`` on a single island).
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
    preflight (IR-1) runs last, after the contract preflight and the elastic
    wiring checks; a failure here means no allocate call ever happens.
    """
    publish_member_cell(miles_args)  # A16: member cell before the harness preflight resolves it
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
    hook = harness_preflight if harness_preflight is not None else resolve_harness_preflight(miles_args)
    if hook is not None:
        # 7.1 in Miles' error type; the neutral hook repeats it with the core error.
        check_harness_reward_scope(getattr(miles_args, "yeto_harness_reward_scope", None))
        hook(miles_args, launch)  # IR-1: harness preflight before placement/allocation
    return fingerprint, capabilities, profile, elastic


def preflight(profile: Any, algorithm: AlgorithmSpec, capabilities: EngineCapabilities) -> None:
    """A1: the launcher-bound profile agrees with the runtime AlgorithmSpec and the
    declared capabilities, before any GPU process exists (before connect_island_ray)."""
    from yeto.rl.engine.execution_profile import check_algorithm_contract

    check_algorithm_contract(profile, algorithm)
    # 4.4a (design D6a): neutral names bind to the pre-rename implementation, or no launch.
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


def build_sync(miles_args: Any, *, yeto_policy_sync: bool) -> tuple[Any, Any]:
    """(sync session, progress store) for the preset the learner selected."""

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
    if getattr(miles_args, "yeto_rl_island_scheduling", "legacy") == "elastic":
        # rl-inter-island-scheduling 0.15: absent attribute = legacy (unchanged below).
        from yeto.rl.engine.bridges import ElasticAvgSync

        if getattr(miles_args, "use_critic", False):
            raise ValueError("--rl-island-scheduling elastic does not support a critic yet")
        return ElasticAvgSync(miles_args.yeto_rl_bridge_config, progress=progress,
                              syncer_epoch=int(getattr(miles_args, "yeto_rl_syncer_epoch", 0)),
                              groups_per_round=int(getattr(miles_args, "rollout_batch_size", 0) or 0) or None,
                              ), progress
    critic_syncer = getattr(miles_args, "yeto_rl_critic_syncer_addr", None)
    if critic_syncer is not None:
        # rl-algo-critic-family 4.2.3 (design D4 plan a): second syncer channel for the
        # critic, one atomic commit for both roles.
        from yeto.rl.engine.bridges import DualStrictAvgSync

        return DualStrictAvgSync(miles_args.yeto_rl_bridge_config, critic_syncer_addr=critic_syncer,
                                 progress=progress), progress
    if getattr(miles_args, "use_critic", False):
        raise ValueError("strict-avg with a critic needs the critic syncer (--critic-syncer); "
                         "the actor syncer alone would leave the critics unaveraged")
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
):
    """Wire the adapter ports into an ``IslandDriver`` (no upstream imports).

    ``critic_model`` (rl-algo-critic-family 3.1): the upstream critic
    ``TrainGroup`` of a critic algorithm (shared actor/critic PPO), trained
    before the actor in every round; None for every critic-free algorithm.

    ``elastic`` (:class:`.elastic_wiring.ElasticWiring`, rl-infra-spec 3.x):
    declared rollout cells for the E1 membership verbs, the reconfiguration
    controller and the batch ledger. The controller reconciles the fork's
    membership epoch with its journal before the driver runs. None keeps the
    island unchanged.
    """

    from yeto.rl.core import parse_policy_snapshot_token

    from yeto.rl.engine.driver import DriverError, EventTape, IslandDriver
    from .placement import MilesPlacement
    from .publish import MilesPublisher
    from .rollout import MilesRolloutPool
    from .state import MilesPolicyState
    from .trainer import MilesTrainerGroup

    holder: dict[str, IslandDriver] = {}
    # rl-resume-from-checkpoint: round cuts without --rl-elastic (single island, no sync)
    resume = resume_wiring_for(miles_args, sync=sync) if elastic is None else None

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
            else {"ledger": resume.ledger} if resume is not None
            else {}
        ),
        **({"evaluate_start": evaluate_start} if evaluate_start is not None else {}),
    )
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
            # rl-multinode-island D9/Q6: node loss is observed through Ray, fail closed
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
        # D2 (6.5): --rl-recommend-mode / --rl-edge-costs-path / --rl-elastic-window-s
        from .elastic_hook import elastic_hook_for

        driver.elastic_hook = elastic_hook_for(miles_args, controller=elastic.controller,
                                               profile=profile, observe=observe)
        # dashboard 5.1/5.2: controller cell-snapshot / reconfig-phase events; only when the
        # observation path is on so the legacy (observe=False) tape stays byte-identical.
        if observe:
            elastic.controller.set_event_sink(driver.emit, journal=True)
    if resume is not None:
        sync.stop_after = resume.stop_after
        from yeto.rl.engine.resume import install_preempt_handler

        install_preempt_handler(resume.controller, emit=driver.emit)
        _wire_round_cuts(driver, elastic=resume, miles_args=miles_args, algorithm=algorithm,
                         base_model_revision=base_model_revision)
    holder["driver"] = driver
    return driver


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
    return out


def next_lr_for(miles_args: Any):
    """Design §4.5 ``lr_at_next_round``: the lr the next round will train at, from the
    run's schedule with Megatron's ``OptimizerParamScheduler.get_lr`` arithmetic
    (constant / linear / cosine, warmup 0 -- what yeto's lr_schedule emits). The round
    after ``local_step`` optimizer steps trains at scheduler step ``local_step``
    (G1 tape: rounds 0..9 at 1e-5 * (1 - k/10)). None when the schedule is not one of
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
    no ``--rl-elastic`` (same attribute names as ElasticWiring for _wire_round_cuts)."""

    def __init__(self, *, controller: Any, ledger: Any, every: int, stop_after: int | None,
                 allow_config_change: bool) -> None:
        self.controller = controller
        self.ledger = ledger
        self.every = every
        self.stop_after = stop_after
        self.allow_config_change = allow_config_change


def resume_wiring_for(miles_args: Any, *, sync: Any, environ: Any = None) -> ResumeWiring | None:
    config = getattr(miles_args, "yeto_rl_resume", None)
    if not config:
        return None
    from yeto.rl.engine.bridges import LocalOnlySync

    if not isinstance(sync, LocalOnlySync):
        raise ValueError("--rl-resume-store needs --rl-single-island-no-sync (multi-island resume "
                         "goes through the syncer checkpoint; see rl-resume-from-checkpoint 3.3)")
    from yeto.rl.engine.ledger import BatchLedger
    from yeto.rl.engine.resume import ResumeController, store_for

    if not config.get("runtime_fingerprint"):
        raise ValueError("--rl-resume-store: no runtime fingerprint (a cut without it is refused)")
    store = store_for(config["store"], environ=environ)
    controller = ResumeController(state_dir=config["state_dir"], store=store,
                                  runtime_fingerprint=str(config.get("runtime_fingerprint") or ""),
                                  keep=int(config.get("keep", 2)))
    return ResumeWiring(controller=controller, ledger=BatchLedger(controller.state_dir),
                        every=int(config.get("every", 1)), stop_after=config.get("stop_after"),
                        allow_config_change=bool(config.get("allow_config_change", False)))


def _wire_round_cuts(driver, *, elastic, miles_args, algorithm, base_model_revision) -> None:
    """rl-multinode-island M4: with a checkpoint store and no outer syncer, keep a round
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
    wire_round_cuts(driver, controller=controller, source=source,
                    every=int(getattr(elastic, "every", None) or config.get("every")
                              or getattr(miles_args, "yeto_rl_resume_every", None) or 1),
                    fingerprint=run_fingerprint_of(miles_args, algorithm),
                    allow_config_change=bool(getattr(elastic, "allow_config_change", False)
                                             or config.get("allow_config_change", False)),
                    next_lr=next_lr_for(miles_args))


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
    """TEST ONLY (``YETO_RL_TEST_INJECT_LORA_PERTURB``, 3.5 E1-B): ``await perturb(eps)``
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
    gpus_per_node = getattr(launch.placement, "gpus_per_node", None)
    try:
        bundles = StartupBundles(pool_gpus=elastic.pool_gpus, views=views,
                                 placement_map=_role_map(launch.placement),
                                 gpus_per_node=gpus_per_node,
                                 node_resolver=_ray_bundle_node,
                                 head_node=_ray_head_node()[0] if gpus_per_node is not None else None)
    except Exception:  # noqa: BLE001 - no usable map: leave trainer edges refused
        if gpus_per_node is not None:
            raise  # rl-multinode-island D3/Q6: a multi-node island fails closed, never silently
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
    from yeto.rl.engine.backends import BACKEND_ENV

    # Decoupling 4.11 (design D11 方案 A): the rollout actors and codex
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

    for key in CUT_INJECTION_ENVS:  # TEST ONLY (E2 G-4.5): rank-side switches, off unless set
        if environ.get(key):
            env_vars[key] = environ[key]

    if environ.get(TOOL_DELAY_ENV):  # test tool-wait workload runs in Ray workers
        env_vars[TOOL_DELAY_ENV] = environ[TOOL_DELAY_ENV]
    if environ.get(ELASTIC_METADATA_ENV) == "1":
        # --rl-elastic: the rollout metadata hook runs inside Ray workers, which
        # inherit the raylet's environment, not the driver's.
        env_vars[ELASTIC_METADATA_ENV] = "1"
    if miles_args is not None:
        # Codex harness: the agent function runs in RolloutExecutor actors and
        # configures itself from this env (preflight.configure_rollout_worker).
        from yeto.rl.harness.codex.preflight import worker_runtime_env

        env_vars.update(worker_runtime_env(miles_args, environ))
    ray_module.init(address=address, runtime_env={"env_vars": env_vars})
    return address


# E2 plan-v2 §0 determinism environment (with Megatron --deterministic-mode);
# the table row lives in the core (decoupling 4.9, yeto.rl.engine.determinism).
# It reaches every rank through connect_island_ray (A2 follow-up, L-2.3).
from yeto.rl.engine.determinism import determinism_env as _determinism_env  # noqa: E402

DETERMINISM_ENV = _determinism_env("miles", "nvidia")


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
        # 3.7 restart recovery budget (--rl-elastic-max-recovery-attempts)
        **({"max_recovery_attempts": int(config["max_recovery_attempts"])}
           if config.get("max_recovery_attempts") is not None else {}),
        # Q4 (C5): off-island copy of the state dir (--rl-elastic-checkpoint-store)
        **({"checkpoint_store": config["checkpoint_store"]} if config.get("checkpoint_store") else {}),
        # 4.7: pool GPU ids (manifest resources.gpus, in logical-bundle order), only
        # with trainer edges; every other elastic run keeps the described pool.
        **({"pool_gpus": manifest_pool_gpus(config["resources"])}
           if config.get("trainer_edges") else {}),
        **_elastic_timeouts(config),
        # rl-inter-island-scheduling 0.13: absent = legacy
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


def island_layout_of(miles_args: Any, topology: Any, placement: Any = None) -> dict[str, Any]:
    """The parallel layout this learner launches (rl-multinode-island Q4, C5): the
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
    """Multi-node startup precondition (rl-multinode-island D9, tasks 3.3), run right
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
        from yeto.rl.adapters.miles.legacy.engine import _append_rl_event

        epochs = getattr(getattr(controller, "journal", None), "epochs", None)
        _append_rl_event(miles_args, {
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
    """Multi-node GPU uuid reconciliation (rl-multinode-island Q6, ruling 2026-10-04),
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
            from yeto.rl.adapters.miles.legacy.engine import _append_rl_event

            epochs = getattr(getattr(controller, "journal", None), "epochs", None)
            _append_rl_event(miles_args, {
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
    """``[(index, uuid), ...]`` per island node in logical order (Ray head first, D3,
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


def _ray_head_node(nodes: Any = None) -> tuple[Any, str]:
    """``(node_id, ip)`` of the Ray head: the one alive node carrying Ray's
    built-in ``node:__internal_head__`` resource (rl-multinode-island D3 head
    pin). Exactly one such node, else ``RuntimeError`` (fail closed)."""
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
    """rl-multinode-island D3 head pin (ruling 2026-10-03): make the fork's
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
            raise RuntimeError(f"D3 head pin: logical node 0 bundles span nodes {sorted(set(map(str, blocks)))}")
        assert_head_block(blocks, head_id)
        return info

    checked_create._yeto_head_pinned = head_id
    pg_module.placement_group = pinned_placement_group
    pg_module.sort_key = head_first_sort_key(head_ip, base_key)
    pg_module._create_placement_group = checked_create
    return head_id


def _ray_bundle_node(pg: Any, bundle: int) -> Any:
    """rl-multinode-island D3/Q6: the Ray node id hosting ``bundle`` of ``pg``
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

    ``harness_preflight`` (IR-1): injectable hook run by :func:`preflight_stage`
    before Ray is connected and before any placement/model allocation.
    """

    from yeto.rl import MILES_NEXT_COMMIT
    from yeto.rl.adapters.miles.legacy.engine import _append_rl_event

    from .state import require_run_plugin

    require_run_plugin()  # before any upstream component or model exists
    from yeto.rl.engine.overlap import loop_eval_starter

    publish_member_cell(miles_args, learner_id)  # A16 (D5): island cell -> harness member key
    fingerprint, capabilities, profile, elastic = preflight_stage(
        miles_args, launch, algorithm, yeto_policy_sync=yeto_policy_sync,
        harness_preflight=harness_preflight,
    )  # contract preflight(...) + harness preflight: before connect_island_ray()
    if getattr(miles_args, "yeto_rl_resume", None):
        # the round cut's backend fingerprint (the elastic controller gets the same one;
        # G2 A: a cut without it is refused "runtime: backend_fingerprint missing")
        miles_args.yeto_rl_resume["runtime_fingerprint"] = fingerprint
    from .e2_harness import load_plan as load_e2_harness_plan

    e2_plan = load_e2_harness_plan()  # TEST ONLY: None unless an E2 harness snapshot
    if e2_plan is not None:
        from .rollout_meta_hook import ELASTIC_METADATA_ENV

        # the harness cuts need the rollout data cursor (rollout-side metadata)
        miles_args.yeto_rl_elastic_metadata = True
        os.environ[ELASTIC_METADATA_ENV] = "1"
    connect_island_ray(miles_args=miles_args)
    topology = getattr(launch.placement, "topology", None)
    if topology is not None and topology.nodes > 1:
        # tasks 3.3: a restarted learner on a partial island (worker node DEAD in the GCS)
        # must fail closed here; a placement group asking for the dead node's GPUs
        # would stay PENDING forever
        refuse_partial_island_preflight(elastic, topology, miles_args, placement=launch.placement)
        # Q6 (2026-10-04): runtime GPU uuid reconciliation, journal gpu_pool; a changed
        # pool is accepted only with --rl-elastic-accept-rebind (fail closed otherwise)
        reconcile_gpu_pool_preflight(elastic, topology, miles_args, placement=launch.placement)
        # rl-multinode-island D3 head pin: trainer block = node 0 = Ray head, fail closed
        pin_placement_group_to_head(topology.gpus_per_node)

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
        # rl-algo-critic-family 3.1: Miles creates a critic iff use_critic
        # (estimator ppo); it must agree with the AlgorithmSpec.
        if (critic is not None) != bool(algorithm.execution.needs_critic):
            raise RuntimeError(
                f"Miles created {'a' if critic is not None else 'no'} critic but the algorithm "
                f"spec says needs_critic={algorithm.execution.needs_critic}"
            )
        # rl-infra-spec 4.3/4.4: one swappable handle shared by trainer,
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
        dispatcher = EvalDispatcher(miles_args, actor, executor)
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
            critic_model=critic,
        )
        # fleet-dashboard 2.1/2.2: opt-in heartbeat / resource sampler periods
        driver.heartbeat_interval_s = getattr(miles_args, "yeto_rl_heartbeat_interval_s", None)
        driver.resource_sample_interval_s = getattr(
            miles_args, "yeto_rl_resource_sample_interval_s", None)
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
