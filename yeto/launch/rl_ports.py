"""Launch checks and island flags for the ports RL backend: algorithm,
infrastructure switches, elastic membership, held-out eval and resume.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path

from ..gpu_spec import parse_gpu_spec
from ..rl.engine.run_flags import CLI, rl_forward_pairs, rl_option, rl_option_pairs
from .rl_backend import _rl_backend_module
from .remote_setup import read_island_pre_run
from .topology import rl_island_count, rl_island_placement_args
from .syncer import island_lease_s, island_pool, island_syncer_epoch
from .storage import ELASTIC_ISLAND_STATE_DIR, rl_checkpoint_store_plan
from .island_tasks import _check_island_tasks


def _prepare_ports_algorithm(args) -> None:
    """rl-algorithm-capabilities D8/D9/D11, before any cloud or GPU work.

    ports: build the run's AlgorithmSpec once (``--rl-algorithm-spec`` or the
    legacy CLI; the launcher has no extra Miles argv, so nothing is absorbed
    here), refuse the rejection matrix, the registered launch checks, anything
    the Miles adapter does not declare, and every unverified-mechanism
    allowance (a launched run always has outer sync, design D11); keep the
    canonical JSON + expected hash every island receives. An island whose
    learner absorbs extra argv into a different hash is refused there
    (``rl_algorithm_mismatch``).
    """

    from ..rl.engine.algorithm import (
        AlgorithmSpecError,
        check_unverified_allowance,
        launch_problems,
        resolve_ports_algorithm,
    )
    from ..rl.engine.capabilities import CapabilityMismatch

    _check_ports_infra_switches(args)
    steps = getattr(args, "rl_optimizer_steps", 1)
    if steps is None:
        steps = 1
    if type(steps) is not int or steps < 1:
        raise ValueError(f"--rl-optimizer-steps must be a positive int (got {steps!r})")
    if steps != 1:
        samples = int(args.rollout_batch_size) * int(args.n_samples_per_prompt)
        if samples % steps:
            # run_config: rollout_batch_size * n_samples == global_batch * optimizer_steps
            raise ValueError(
                f"--rl-optimizer-steps {steps} must divide --rollout-batch-size * "
                f"--n-samples-per-prompt ({samples}) into equal optimizer batches"
            )
    try:
        spec = resolve_ports_algorithm(args)
        problems = spec.rejections()
        # Run-configuration checks of the follow-up changes and the Miles
        # adapter's declaration, before any cloud resource exists (the learner
        # repeats both on the island).
        problems += launch_problems(spec, {
            "rollout_batch_size": getattr(args, "rollout_batch_size", None),
            "rollout_max_response_len": getattr(args, "rollout_max_response_len", None),
            "context_parallel_size": int(getattr(args, "context_parallel", 1) or 1),
            "multi_lora": False,
            # rl-algo-critic-family 2.3 (critic run-level rejections)
            "sync_preset": getattr(args, "rl_sync_preset", "strict-avg"),
            "elastic": bool(getattr(args, "rl_elastic", False)),
            "reward_function": getattr(args, "reward_function", None),
        })
        if problems:
            raise AlgorithmSpecError("algorithm spec rejected: " + "; ".join(problems))
        islands = (
            len(parse_gpu_spec(args.gpu)) if getattr(args, "gpu", None) else 1
        ) + max(0, getattr(args, "external_learners", 0) or 0)
        no_sync = bool(getattr(args, "rl_single_island_no_sync", False))
        from ..rl.engine import sample_roles

        # L5.4: a store-coupled pair (one trainer, one or more samplers: S25 C1) and a
        # convertible pool (L7.5) are two or more no-sync islands (sample_roles holds
        # their own rules).
        no_sync_islands = (islands if sample_roles.is_convertible_pool(args)
                           or sample_roles.is_pair(args) else 1)
        if no_sync and (islands != no_sync_islands or getattr(args, "rl_sync_preset", "strict-avg")
                        != "strict-avg" or getattr(args, "rl_initial_adapter", None)):
            raise AlgorithmSpecError(
                "--rl-single-island-no-sync needs exactly one island (one --gpu entry, no "
                "external learners; one per island with --rl-island-sample-role), the default sync "
                "preset and no initial adapter"
            )
        args.rl_allow_unverified_mechanism = list(
            check_unverified_allowance(
                getattr(args, "rl_allow_unverified_mechanism", None) or (),
                islands=islands,
                outer_sync=not no_sync,  # a launched run has a syncer unless no-sync
                sync_preset=getattr(args, "rl_sync_preset", "strict-avg"),
            )
        )
        _entry = _rl_backend_module(args, "entry")
        miles_capabilities, with_partitioned_serial = (
            _entry.miles_capabilities, _entry.with_partitioned_serial)

        # Each island checks its own placement; the AlgorithmSpec hash does not
        # depend on it (P4-Q4: placement capabilities stay out of the hash).
        for partitioned in sorted({getattr(iargs, "rl_placement", "colocated") == "fixed-partition"
                                   for iargs in rl_island_placement_args(args)}):
            capabilities = miles_capabilities(
                "sha256:" + "0" * 64, unverified_mechanisms=args.rl_allow_unverified_mechanism
            )
            if partitioned:
                capabilities = with_partitioned_serial(capabilities)
            capabilities.check(
                layout="lora",
                placement="fixed-partition" if partitioned else "colocated",
                execution_mode="partitioned-serial" if partitioned else "colocated-serial",
                algorithm=spec,
            )
    except (AlgorithmSpecError, CapabilityMismatch) as error:
        raise ValueError(str(error)) from error
    _check_critic_warmup_options(args, spec)
    args.rl_algorithm_spec_json = spec.canonical_json()
    args.rl_expected_algorithm_sha256 = spec.sha256()
    args._offpolicy_correction_manifest = _offpolicy_correction_manifest(args, spec)


def _offpolicy_correction_manifest(args, spec) -> dict:
    """rl-offpolicy-correction-passthrough D8/L6.5: every island's correction
    record. A warning (unmappable field, island difference) refuses the launch
    (phase 2, L6b.2)."""

    from ..rl.engine.offpolicy_correction import (refuse_correction_warnings,
                                                  run_correction_manifest)
    from .topology import rl_island_args

    backends = {str(m): getattr(rl_island_args(args, m), "rl_backend", None) or "miles"
                for m in range(rl_island_count(args))}
    section = run_correction_manifest(spec, backends)
    refuse_correction_warnings(section["warnings"])
    return section


def _check_critic_warmup_options(args, spec) -> None:
    """--rl-critic-load/--rl-critic-init-sha256/--rl-critic-baseline-rounds (5.3)."""

    load = getattr(args, "rl_critic_load", None)
    digest = getattr(args, "rl_critic_init_sha256", None)
    rounds = getattr(args, "rl_critic_baseline_rounds", 0) or 0
    if load is None and digest is None and not rounds:
        return
    if (load is None) != (digest is None):
        raise ValueError("--rl-critic-load and --rl-critic-init-sha256 go together")
    if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("--rl-critic-init-sha256 must be a lowercase hex SHA256")
    critic = spec.critic
    warm = (spec.execution.needs_critic and critic.init == "copy_actor_backbone"
            and bool(critic.warmup_steps))
    if not warm:
        raise ValueError(
            "--rl-critic-load/--rl-critic-baseline-rounds need a critic algorithm with "
            "critic.init='copy_actor_backbone' and critic.warmup_steps > 0"
        )
    if type(rounds) is not int or rounds < 0:
        raise ValueError(f"--rl-critic-baseline-rounds must be >= 0 (got {rounds!r})")
    if rounds and (load is not None or not getattr(args, "rl_single_island_no_sync", False)):
        raise ValueError(
            "--rl-critic-baseline-rounds needs --rl-single-island-no-sync and no --rl-critic-load"
        )


_ELASTIC_LAUNCH_FLAGS = rl_option_pairs(
    "rl_elastic_resources",
    "rl_elastic_attestation",
    "rl_elastic_initial_config",
    "rl_elastic_cells",
    layer=CLI,
)
# 3.8 strict pause budget: forwarded to the learner; the quorum timeout also
# goes to the syncer so both sides use the same value.
_ELASTIC_PAUSE_FLAGS = rl_forward_pairs(
    "rl_elastic_quorum_timeout_s",
    "rl_elastic_idle_flow_timeout_s",
    "rl_elastic_pause_margin",
    # controller D4 timeouts T_drain / T_recovery (plan.md E1-C, E1-D 4)
    "rl_elastic_drain_timeout_s",
    "rl_elastic_recovery_timeout_s",
    # 3.7 restart recovery budget (recovery-design.md §3)
    "rl_elastic_max_recovery_attempts",
)
# Test-only fault injection for GPU acceptance runs (gpu-plan-v2 A5 quorum case):
# exported into the island run command; off unless given.
_ELASTIC_TEST_FLAGS = rl_option_pairs(
    "rl_test_inject_start_delay_s",
    "rl_test_inject_update_weights_block_s",
    "rl_test_inject_stop_failures",
    "rl_test_hold_before_check_s",
    "rl_test_inject_tool_wait_s",
    "rl_test_inject_undrain_fail",
    "rl_elastic_restart_attempts",
    layer=CLI,
)
# (attr, flag, env) of the test-only switches exported into the island run
# command; each needs --rl-elastic, all are off by default.
_ELASTIC_TEST_EXPORTS = tuple(
    (dest, rl_option(dest, CLI), env) for dest, env in (
        ("rl_test_hold_before_check_s", "YETO_RL_TEST_HOLD_BEFORE_CHECK_S"),
        ("rl_test_inject_tool_wait_s", "YETO_RL_TEST_INJECT_TOOL_WAIT_S"),
        ("rl_test_inject_undrain_fail", "YETO_RL_TEST_INJECT_UNDRAIN_FAIL"),
        # A4bc / 3.3 X5: side-effect journal of the injected tool (store_true -> "1")
        ("rl_test_tool_side_effect_log", "YETO_RL_TEST_TOOL_SIDE_EFFECT_LOG"),
        ("rl_test_inject_lora_perturb", "YETO_RL_TEST_INJECT_LORA_PERTURB"),
        ("rl_test_inject_stop_failures", "YETO_RL_TEST_INJECT_STOP_FAILURES"),
        ("rl_test_kill_learner_at", "YETO_RL_TEST_KILL_LEARNER_AT"),
        ("rl_test_inject_rebuild_fail", "YETO_RL_TEST_INJECT_REBUILD_FAIL"),
        # E2 G-4.5 (plan-v3), read by miles_adapter.cut_injection
        ("rl_test_inject_cut_save_kill_rank", "YETO_RL_TEST_INJECT_CUT_SAVE_KILL_RANK"),
        ("rl_test_inject_cut_restore_kill_rank", "YETO_RL_TEST_INJECT_CUT_RESTORE_KILL_RANK"),
        ("rl_test_inject_cut_restore_sleep", "YETO_RL_TEST_INJECT_CUT_RESTORE_SLEEP"),
        ("rl_test_inject_rebuild_cursor_shift", "YETO_RL_TEST_INJECT_REBUILD_CURSOR_SHIFT"),
    )
)
KILL_PHASES = ("QUIESCING", "TRANSFERRING", "INITIALIZING", "VERIFYING", "COMMITTED",
               "RESUMING", "REBUILDING_TRAINER")
# In-place learner restarts (E1-D ⑤⑥⑦): the learner command runs in a loop that
# re-executes it with the same arguments (same --rl-elastic-state-dir) after a
# non-zero exit, at most N times. The Ray head stays up; the old driver's job
# (fork controller, engines, trainer) dies with it, so the fork restarts at
# membership epoch 0 and the journal reconciles it.
RESTART_LOOP_FN = (
    "yeto_rl_restart_loop() {\n"
    "  local attempt=0 rc=0\n"
    "  while :; do\n"
    "    \"$@\" && return 0\n"
    "    rc=$?\n"
    "    attempt=$((attempt + 1))\n"
    "    if [ \"$attempt\" -gt \"$YETO_RL_RESTART_ATTEMPTS\" ]; then return $rc; fi\n"
    "    echo \"[yeto] learner exited $rc; in-place restart $attempt/$YETO_RL_RESTART_ATTEMPTS\" >&2\n"
    "  done\n"
    "}\n"
)


_CUT_LAUNCH_FLAGS = (("rl_cut_every", "--rl-cut-every"), ("rl_cut_keep", "--rl-cut-keep"),
                     ("rl_stop_after_rounds", "--rl-stop-after-rounds"),
                     ("rl_cut_auto_mtbf_s", "--rl-cut-auto-mtbf-s"),
                     ("rl_cut_max_interval_s", "--rl-cut-max-interval-s"))
# what turns on the round cuts of a synced launch (S27 R-2)
_ISLAND_CUT_TRIGGERS = ("rl_cut_every", "rl_cut_auto_mtbf_s")


def island_cuts_requested(args) -> bool:
    """S27 R-2: a synced launch (a syncer, no ``--rl-elastic``) with
    ``--rl-checkpoint-store`` and a cut interval: every island writes its own round
    cuts under ``<store>/island-cuts`` (forwarded as ``--rl-island-cut-store``)."""
    return (bool(getattr(args, "rl_checkpoint_store", None))
            and not getattr(args, "rl_elastic", False)
            and not getattr(args, "rl_single_island_no_sync", False)
            and any(getattr(args, name, None) is not None for name in _ISLAND_CUT_TRIGGERS))


def island_cut_store(store_path: str) -> str:
    """Where the islands of a synced launch keep their round cuts (one subdir each)."""
    return f"{store_path.rstrip('/')}/island-cuts"


def _check_resume_launch_flags(args) -> bool:
    """rl-resume-from-checkpoint: True when ``--rl-checkpoint-store`` is the non-elastic
    resume store (``--rl-single-island-no-sync``, ports). Validates the cadence flags."""
    from ..rl.engine.cut_interval import cut_interval_for

    store = getattr(args, "rl_checkpoint_store", None)
    elastic = bool(getattr(args, "rl_elastic", False))
    for name, flag in _CUT_LAUNCH_FLAGS:
        value = getattr(args, name, None)
        if value is not None and not value > 0:
            raise ValueError(f"{flag} must be positive" if isinstance(value, float) else f"{flag} must be >= 1")
        if value is not None and not store:
            raise ValueError(f"{flag} needs --rl-checkpoint-store")
    if getattr(args, "rl_resume_allow_config_change", False) and not store:
        raise ValueError("--rl-resume-allow-config-change needs --rl-checkpoint-store")
    cut_interval_for(every=getattr(args, "rl_cut_every", None),
                     auto_mtbf_s=getattr(args, "rl_cut_auto_mtbf_s", None),
                     max_interval_s=getattr(args, "rl_cut_max_interval_s", None))  # exclusive modes
    from ..rl.engine.sample_roles import is_synced_trainer

    if store and not elastic and not getattr(args, "rl_single_island_no_sync", False):
        # S25 C3: the store holds the sample buffer and the releases of synced trainers
        # (forwarded as --rl-sample-store). S27 R-2: with a cut interval every synced
        # island also writes its own round cuts there; it does not resume from them
        # (the syncer is authoritative), so the resume switches stay refused.
        for name, flag in (("rl_stop_after_rounds", "--rl-stop-after-rounds"),
                           ("rl_resume_allow_config_change", "--rl-resume-allow-config-change")):
            if getattr(args, name, None):
                raise ValueError(f"{flag}: a synced launch does not resume from round cuts (the "
                                 "syncer is authoritative; multi-island resume: rl-resume-from-checkpoint 3.3)")
        # S27 E-3: --rl-peer-recovery keeps its peer snapshots in the store
        if not (is_synced_trainer(args) or island_cuts_requested(args)
                or getattr(args, "rl_peer_recovery", False)):
            raise ValueError("--rl-checkpoint-store without --rl-elastic needs --rl-single-island-no-sync, "
                             "synced trainers, a cut interval (--rl-cut-every / --rl-cut-auto-mtbf-s: "
                             "island round cuts) or --rl-peer-recovery")
        if getattr(args, "rl_cut_keep", None) is not None and not island_cuts_requested(args):
            raise ValueError("--rl-cut-keep needs a cut interval on a synced launch")
        if island_cuts_requested(args) and (getattr(args, "rl_backend", None) or "miles") != "miles":
            raise ValueError("island round cuts (--rl-cut-every / --rl-cut-auto-mtbf-s on a synced "
                             "launch) are wired for --rl-backend miles only")
        rl_checkpoint_store_plan(args)  # validates the form
        return False
    if elastic:
        for name, flag in (("rl_cut_keep", "--rl-cut-keep"), ("rl_stop_after_rounds", "--rl-stop-after-rounds")):
            if getattr(args, name, None) is not None:
                raise ValueError(f"{flag} applies to a non-elastic --rl-checkpoint-store run")
        return False
    if not store:
        return False
    rl_checkpoint_store_plan(args)  # validates the form (single island, no sync)
    return True


_EVAL_LAUNCH_FLAGS = rl_option_pairs(
    "rl_eval_data", "rl_eval_dataset_name", "rl_eval_samples_per_prompt", layer=CLI,
)
# optional eval knobs: launcher flag attr -> learner flag (forwarded when given)
_EVAL_OPTIONAL_FLAGS = rl_forward_pairs(
    ("rl_eval_temperature", "eval_temperature"),
    ("rl_eval_top_p", "eval_top_p"),
    ("rl_eval_max_prompt_len", "eval_max_prompt_len"),
    ("rl_eval_max_response_len", "eval_max_response_len"),
    ("rl_eval_max_context_len", "eval_max_context_len"),
)
EVAL_ISLAND_DATA_PATH = "~/yeto-rl/eval-heldout.jsonl"
EVAL_INLINE_MAX_BYTES = 96 * 1024  # shipped inline in the run command (ARG_MAX headroom)


def _check_ports_eval(args) -> int | None:
    """``--rl-eval-*``: the launcher's ONE eval source (ports LoRA heldout eval).

    Returns the eval interval (None when eval is off) — the same value the
    learner receives as ``--eval-interval`` — and stages the heldout bytes and
    their SHA256 on ``args`` for :func:`_ports_infra_flags`.
    """

    interval = getattr(args, "rl_eval_interval", None)
    given = [flag for name, flag in _EVAL_LAUNCH_FLAGS if getattr(args, name, None) is not None]
    given += [rl_option(name, CLI) for name, _flag in _EVAL_OPTIONAL_FLAGS
              if getattr(args, name, None) is not None]
    args.rl_eval_data_text = None
    if interval is None:
        if given:
            raise ValueError(", ".join(given) + " need --rl-eval-interval")
        return None
    if type(interval) is not int or interval <= 0:
        raise ValueError(f"--rl-eval-interval must be a positive int (got {interval!r})")
    missing = [flag for name, flag in _EVAL_LAUNCH_FLAGS if getattr(args, name, None) is None]
    if missing:
        raise ValueError("--rl-eval-interval needs " + ", ".join(missing))
    samples = args.rl_eval_samples_per_prompt
    if type(samples) is not int or samples <= 0:
        raise ValueError("--rl-eval-samples-per-prompt must be a positive int")
    source = Path(args.rl_eval_data).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ValueError("--rl-eval-data must be one regular local file")
    data = getattr(args, "data", None)
    if isinstance(data, str) and Path(data).expanduser().exists() and (
        Path(data).expanduser().resolve() == source.resolve()
    ):
        raise ValueError("--rl-eval-data must be a heldout file distinct from --data")
    raw = source.read_bytes()
    if len(raw) > EVAL_INLINE_MAX_BYTES:
        raise ValueError(f"--rl-eval-data is larger than {EVAL_INLINE_MAX_BYTES} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("--rl-eval-data is not UTF-8") from error
    rows = [line for line in text.splitlines() if line.strip()]
    if not rows or len(rows) != len(text.splitlines()):
        raise ValueError("--rl-eval-data must be non-empty JSONL without blank rows")
    args.rl_eval_data_text = text
    args.rl_eval_data_sha256 = hashlib.sha256(raw).hexdigest()
    return interval


def _check_ports_infra_switches(args) -> None:
    """``--rl-overlap-eval`` / ``--rl-elastic`` (rl-infra-spec 2.3 / 3.x): opt-in;
    the elastic manifest is read here so a bad file fails locally."""

    from ..rl.engine import island_placement
    from ..rl.engine.execution_profile import UNKNOWN, check_elastic_placement, check_overlap_eval

    placements = island_placement.check(args, rl_island_count(args),
                                        island_scheduling="elastic" if island_pool(args) else "legacy")
    # The checks below need a fixed partition; any colocated island fails them.
    placement = (island_placement.FIXED_PARTITION
                 if all(p.placement == island_placement.FIXED_PARTITION for p in placements)
                 else island_placement.COLOCATED)
    _check_test_tool_delay(args)
    eval_interval = _check_ports_eval(args)
    _check_island_tasks(args, eval_interval)
    if getattr(args, "rl_overlap_eval", False):
        # Same check the island's execution profile applies, run before any
        # resource is provisioned, on the values the launcher really forwards:
        # --rl-placement is the island's launch.placement.kind and
        # --rl-eval-interval is the learner's --eval-interval (-> Miles
        # eval_interval). --eval-uses-snapshots has no launcher source (the
        # learner never sets it), so that one stays with the island's check.
        check_overlap_eval(placement_kind=placement, eval_interval=eval_interval,
                           eval_uses_snapshots=UNKNOWN)
    sample_s = getattr(args, "rl_resource_sample_interval", None)
    if sample_s is not None and sample_s < 0:
        raise ValueError("--rl-resource-sample-interval needs a value >= 0")
    _rl_backend_module(args, "elastic_hook").check_recommend_flags(args)
    dropout = getattr(args, "rl_lora_dropout", None)
    if dropout is not None and not 0.0 <= dropout < 1.0:
        raise ValueError("--rl-lora-dropout needs a value in [0, 1)")
    expert_rank = getattr(args, "rl_lora_expert_rank", None)
    if expert_rank is not None and expert_rank < 0:
        raise ValueError("--rl-lora-expert-rank needs a value >= 0")
    ref_load = getattr(args, "rl_megatron_ref_load", None)
    if ref_load is not None and (not ref_load.startswith("/")
                                 or any(c.isspace() for c in ref_load)):
        raise ValueError("--rl-megatron-ref-load needs an absolute path without whitespace")
    if getattr(args, "rl_boot_only", False) and not getattr(args, "rl_single_island_no_sync", False):
        raise ValueError("--rl-boot-only needs --rl-single-island-no-sync")
    from ..rl.engine import sample_roles

    if sample_roles.is_convertible_pool(args):  # L7.5: every island convertible
        sample_roles.check_pool(args, rl_island_count(args))
    elif sample_roles.is_pair(args):  # L5.4 --rl-island-sample-role: per-island roles
        sample_roles.check_pair(args, rl_island_count(args))
    else:
        sample_roles.check_synced_trainers(args, rl_island_count(args))  # S25 C3
        sample_roles.check_launch(args)  # L4.6 --rl-sample-role (both checks nothing)
    if getattr(args, "rl_sampler_spot", False) and not sample_roles.is_pair(args):
        raise ValueError("--rl-sampler-spot needs --rl-island-sample-role (it bills the sampler "
                         "islands of a pair as spot)")
    from .storage import store_object_envs

    for cloud in {s.cloud for s in parse_gpu_spec(args.gpu)} if getattr(args, "gpu", None) else ():
        store_object_envs(args, cloud)  # raises without the s3 store credentials
    pre_run = getattr(args, "rl_island_pre_run", None)
    if pre_run is not None:
        if not getattr(args, "rl_single_island_no_sync", False):
            raise ValueError("--rl-island-pre-run needs --rl-single-island-no-sync")
        args.rl_island_pre_run_script = read_island_pre_run(pre_run)
    given = [flag for name, flag in _ELASTIC_LAUNCH_FLAGS + _ELASTIC_PAUSE_FLAGS
             if getattr(args, name, None) is not None]
    if getattr(args, "rl_elastic_tool_wait_board", False):
        given.append("--rl-elastic-tool-wait-board")
    if getattr(args, "rl_elastic_trainer_edges", False):
        given.append("--rl-elastic-trainer-edges")
    if getattr(args, "rl_elastic_accept_rebind", False):
        given.append("--rl-elastic-accept-rebind")
    resume_store = _check_resume_launch_flags(args)
    if getattr(args, "rl_checkpoint_store", None) and not resume_store \
            and not sample_roles.is_synced_trainer(args) and not island_cuts_requested(args) \
            and not getattr(args, "rl_peer_recovery", False):  # S27 E-3: peer snapshots
        given.append("--rl-checkpoint-store")
    if getattr(args, "rl_elastic_declare_cells", False):
        given.append("--rl-elastic-declare-cells")
    for name, flag in _ELASTIC_PAUSE_FLAGS + _ELASTIC_TEST_FLAGS:
        value = getattr(args, name, None)
        if value is not None and not value > 0:
            raise ValueError(f"{flag} must be positive")
    given += [flag for name, flag in _ELASTIC_TEST_FLAGS if getattr(args, name, None) is not None]
    given += [flag for name, flag, _ in _ELASTIC_TEST_EXPORTS
              if getattr(args, name, None) is not None and getattr(args, name) is not False
              and flag not in given]  # rank 0 is a valid value (E2 cut injections)
    if getattr(args, "rl_elastic_state_dir", None) is not None:
        given.append("--rl-elastic-state-dir")
    if getattr(args, "rl_test_hold_before_check_s", None) is not None and not any(
            getattr(args, n, None) not in (None, False)
            for n, _f, e in _ELASTIC_TEST_EXPORTS if n != "rl_test_hold_before_check_s"):
        raise ValueError("--rl-test-hold-before-check-s only acts together with another "
                         "--rl-test-* injection (test-injection mode)")
    if (getattr(args, "rl_test_inject_tool_wait_s", None) is not None
            and not getattr(args, "rl_elastic_tool_wait_board", False)):
        raise ValueError("--rl-test-inject-tool-wait-s needs --rl-elastic-tool-wait-board")
    if (getattr(args, "rl_test_tool_side_effect_log", False)
            and getattr(args, "rl_test_inject_tool_wait_s", None) is None):
        raise ValueError("--rl-test-tool-side-effect-log needs --rl-test-inject-tool-wait-s "
                         "(the injected tool is the journaled tool)")
    kill_at = getattr(args, "rl_test_kill_learner_at", None)
    if kill_at is not None and kill_at not in KILL_PHASES:
        raise ValueError(f"--rl-test-kill-learner-at must be one of {list(KILL_PHASES)}")
    if kill_at is not None and not getattr(args, "rl_elastic_restart_attempts", None):
        raise ValueError("--rl-test-kill-learner-at needs --rl-elastic-restart-attempts")
    if not getattr(args, "rl_elastic", False):
        if given:
            raise ValueError(", ".join(given) + " need --rl-elastic")
        return
    missing = [flag for name, flag in _ELASTIC_LAUNCH_FLAGS
               if name not in ("rl_elastic_attestation", "rl_elastic_cells")
               and not getattr(args, name, None)]
    if missing:
        raise ValueError("--rl-elastic needs " + ", ".join(missing))
    check_elastic_placement(placement)
    if args.rl_elastic_cells is not None:
        cells = [c.strip() for c in args.rl_elastic_cells.split(",") if c.strip()]
        if not cells or any(not re.fullmatch(r"[A-Za-z0-9_.:@+-]+", c) for c in cells):
            raise ValueError(f"--rl-elastic-cells: bad cell ids {args.rl_elastic_cells!r}")
    if getattr(args, "rl_elastic_declare_cells", False) and args.rl_elastic_cells is None:
        raise ValueError("--rl-elastic-declare-cells needs --rl-elastic-cells (the names)")
    from ..rl.elastic_benchmark.capabilities import load_attestation, parse_configs

    resources = json.loads(Path(args.rl_elastic_resources).expanduser().read_text(encoding="utf-8"))
    configs = parse_configs(resources)  # the island's build_elastic parses the same way
    if args.rl_elastic_initial_config not in configs:
        raise ValueError(f"--rl-elastic-initial-config {args.rl_elastic_initial_config!r} "
                         f"is not a manifest config ({sorted(configs)})")
    args.rl_elastic_resources_json = json.dumps(resources, sort_keys=True, separators=(",", ":"))
    # Q2: the initial config's explicit (node, gpu) placement, when the manifest
    # declares a topology; rl_island_layout derives the trainer shape from it.
    args.rl_elastic_initial_placement_slots = getattr(
        configs[args.rl_elastic_initial_config], "placement_slots", None)
    # ruling 2026-10-04 v2: cfg parallel.allow_cross_node_* count like the CLI flags
    args.rl_elastic_initial_cfg_switches = {
        k: bool(getattr(configs[args.rl_elastic_initial_config], k, False))
        for k in ("allow_cross_node_tp", "allow_cross_node_engine_tp")}
    attestation = getattr(args, "rl_elastic_attestation", None)
    args.rl_elastic_attestation_json = None
    if attestation:
        path = Path(attestation).expanduser()
        load_attestation(path)  # malformed attestation fails locally
        args.rl_elastic_attestation_json = json.dumps(
            json.loads(path.read_text(encoding="utf-8")), sort_keys=True, separators=(",", ":"),
        )


def _check_test_tool_delay(args) -> None:
    """``--rl-test-tool-delay-s`` (TEST ONLY): the tool-wait workload's delay."""
    from ..rl.tool_wait_workload import GENERATE_PATH

    delay = getattr(args, "rl_test_tool_delay_s", None)
    if delay is None:
        return
    if not delay > 0:
        raise ValueError("--rl-test-tool-delay-s must be positive")
    if getattr(args, "custom_generate_function_path", None) != GENERATE_PATH:
        raise ValueError(f"--rl-test-tool-delay-s needs --custom-generate-function-path {GENERATE_PATH}")


def _ports_infra_flags(args) -> tuple[str, str]:
    """(prelude, learner flags) for the opt-in 2.3/3.x switches; ("", "") by default."""

    prelude, flags = "", ""
    if getattr(args, "rl_test_tool_delay_s", None) is not None:
        from ..rl.tool_wait_workload import TOOL_DELAY_ENV

        prelude += f"export {TOOL_DELAY_ENV}={float(args.rl_test_tool_delay_s)!r}\n"
    if getattr(args, "rl_eval_interval", None) is not None:
        prelude += (
            "mkdir -p ~/yeto-rl && printf '%s' "
            f"{shlex.quote(args.rl_eval_data_text)} > {EVAL_ISLAND_DATA_PATH}\n"
        )
        flags += (
            f" --eval-interval {int(args.rl_eval_interval)}"
            f" --eval-data {EVAL_ISLAND_DATA_PATH}"
            f" --eval-data-sha256 {args.rl_eval_data_sha256}"
            f" --eval-dataset-name {shlex.quote(args.rl_eval_dataset_name)}"
            f" --eval-samples-per-prompt {int(args.rl_eval_samples_per_prompt)}"
        )
        for name, flag in _EVAL_OPTIONAL_FLAGS:
            value = getattr(args, name, None)
            if value is not None:
                flags += f" {flag} {value!r}"
    if getattr(args, "rl_overlap_eval", False):
        flags += " --rl-overlap-eval"
    if getattr(args, "rl_observe_timeline", False):
        flags += " --rl-observe-timeline"
    if getattr(args, "rl_debug_dump", False):
        flags += " --rl-debug-dump"
    if getattr(args, "rl_profile_trainer", False):
        flags += " --rl-profile-trainer"
    if getattr(args, "rl_verify_every_round", False):
        flags += " --rl-verify-every-round"
    if getattr(args, "rl_resource_sample_interval", None) is not None:
        flags += f" --rl-resource-sample-interval {float(args.rl_resource_sample_interval)!r}"
    flags += _rl_backend_module(args, "elastic_hook").recommend_flags(args)
    if getattr(args, "rl_deterministic_trainer", False):
        flags += " --rl-deterministic-trainer"
    if getattr(args, "rl_lora_dropout", None) is not None:
        flags += f" --rl-lora-dropout {float(args.rl_lora_dropout)!r}"
    if getattr(args, "rl_lora_expert_rank", None) is not None:
        flags += f" --rl-lora-expert-rank {int(args.rl_lora_expert_rank)}"
    if getattr(args, "rl_megatron_ref_load", None) is not None:
        flags += f" --megatron-ref-load {shlex.quote(args.rl_megatron_ref_load)}"
    if getattr(args, "rl_print_attestation_fingerprint", False):
        flags += " --rl-print-attestation-fingerprint"
    if getattr(args, "rl_boot_only", False):
        flags += " --rl-boot-only"
    if getattr(args, "rl_max_policy_age", 0):  # 0 (default) adds nothing
        flags += f" --rl-max-policy-age {int(args.rl_max_policy_age)}"
    if getattr(args, "rl_async_buffer_rounds", None) is not None:  # None adds nothing
        flags += f" --rl-async-buffer-rounds {int(args.rl_async_buffer_rounds)}"
    from ..rl.engine.sample_roles import island_flags as sample_role_flags
    from ..rl.engine.sample_roles import is_synced_trainer as sample_role_is_synced_trainer

    flags += sample_role_flags(args)  # L4.6: role both (default) adds nothing
    if (getattr(args, "rl_lr_schedule", None) or "auto") != "auto":  # auto adds nothing
        flags += f" --rl-lr-schedule {args.rl_lr_schedule}"
    if island_pool(args):  # a fixed roster adds nothing
        flags += (" --rl-island-pool"
                  f" --rl-syncer-epoch {island_syncer_epoch(args)}"
                  f" --rl-sync-island-lease-s {island_lease_s(args)!r}")
    resume_store = (rl_checkpoint_store_plan(args)
                    if getattr(args, "rl_checkpoint_store", None) and not getattr(args, "rl_elastic", False)
                    else None)
    if resume_store is not None and island_cuts_requested(args):
        # S27 R-2: every synced island writes its round cuts (island<learner id> below)
        flags += f" --rl-island-cut-store {shlex.quote(island_cut_store(resume_store[0]))}"
        if getattr(args, "rl_cut_keep", None) is not None:
            flags += f" --rl-cut-keep {int(args.rl_cut_keep)}"
    if resume_store is not None and sample_role_is_synced_trainer(args):
        # S25 C3: synced trainers mount the store for the buffer and the releases
        flags += (f" --rl-sample-store {shlex.quote(resume_store[0])}"
                  f" --rl-sample-lanes {getattr(args, 'rl_synced_trainers', None) or rl_island_count(args)}")
        resume_store = None
    if resume_store is not None and not getattr(args, "rl_single_island_no_sync", False):
        resume_store = None  # a synced launch: island cuts only (above)
    if resume_store is not None:  # rl-resume-from-checkpoint: non-elastic resume store
        flags += f" --rl-resume-store {shlex.quote(resume_store[0])}"
        for name, flag in (("rl_cut_keep", "--rl-cut-keep"), ("rl_stop_after_rounds", "--rl-stop-after-rounds")):
            if getattr(args, name, None) is not None:
                flags += f" {flag} {int(getattr(args, name))}"
    if getattr(args, "rl_checkpoint_store", None) and getattr(args, "rl_cut_every", None) is not None:
        flags += f" --rl-cut-every {int(args.rl_cut_every)}"
    for name, flag in (("rl_cut_auto_mtbf_s", "--rl-cut-auto-mtbf-s"),
                       ("rl_cut_max_interval_s", "--rl-cut-max-interval-s")):
        if getattr(args, "rl_checkpoint_store", None) and getattr(args, name, None) is not None:
            flags += f" {flag} {float(getattr(args, name))!r}"
    if getattr(args, "rl_checkpoint_store", None) and getattr(args, "rl_resume_allow_config_change", False):
        flags += " --rl-resume-allow-config-change"
    if getattr(args, "rl_elastic", False):
        prelude += (
            "mkdir -p ~/yeto-rl && printf '%s' "
            f"{shlex.quote(args.rl_elastic_resources_json)} > ~/yeto-rl/elastic_resources.json\n"
        )
        state_dir = (shlex.quote(args.rl_elastic_state_dir)
                     if getattr(args, "rl_elastic_state_dir", None) else ELASTIC_ISLAND_STATE_DIR)
        flags += (
            " --rl-elastic --rl-elastic-resources ~/yeto-rl/elastic_resources.json"
            f" --rl-elastic-state-dir {state_dir}"
            f" --rl-elastic-initial-config {shlex.quote(args.rl_elastic_initial_config)}"
        )
        if args.rl_elastic_cells is not None:
            flags += f" --rl-elastic-cells {shlex.quote(args.rl_elastic_cells)}"
        if getattr(args, "rl_elastic_declare_cells", False):
            flags += " --rl-elastic-declare-cells"
        if getattr(args, "rl_elastic_tool_wait_board", False):
            flags += " --rl-elastic-tool-wait-board"
        if getattr(args, "rl_elastic_trainer_edges", False):
            flags += " --rl-elastic-trainer-edges"
        if getattr(args, "rl_elastic_accept_rebind", False):
            flags += " --rl-elastic-accept-rebind"
        store = rl_checkpoint_store_plan(args)
        if store is not None:
            flags += f" --rl-elastic-checkpoint-store {shlex.quote(store[0])}"
        for name, flag in _ELASTIC_PAUSE_FLAGS:
            value = getattr(args, name, None)
            if value is not None:
                flags += f" {flag} {value!r}"
        delay = getattr(args, "rl_test_inject_start_delay_s", None)
        if delay is not None:
            INJECT_START_DELAY_ENV = _rl_backend_module(args, "rollout").INJECT_START_DELAY_ENV

            prelude += f"export {INJECT_START_DELAY_ENV}={float(delay)!r}\n"
        for name, _flag, env in _ELASTIC_TEST_EXPORTS:
            value = getattr(args, name, None)
            if value is True:
                value = 1
            if value is not None and value is not False:  # rank 0 is valid
                prelude += f"export {env}={shlex.quote(str(value))}\n"
        attempts = getattr(args, "rl_elastic_restart_attempts", None)
        if attempts:
            prelude += f"export YETO_RL_RESTART_ATTEMPTS={int(attempts)}\n" + RESTART_LOOP_FN
            args.rl_learner_launch_prefix = "yeto_rl_restart_loop "
        block = getattr(args, "rl_test_inject_update_weights_block_s", None)
        if block is not None:
            INJECT_UPDATE_BLOCK_ENV = _rl_backend_module(args, "publish").INJECT_UPDATE_BLOCK_ENV

            prelude += f"export {INJECT_UPDATE_BLOCK_ENV}={float(block)!r}\n"
        if getattr(args, "rl_elastic_attestation_json", None):
            prelude += (
                "printf '%s' "
                f"{shlex.quote(args.rl_elastic_attestation_json)} > ~/yeto-rl/elastic_attestation.json\n"
            )
            flags += " --rl-elastic-attestation ~/yeto-rl/elastic_attestation.json"
    return prelude, flags


def _ports_algorithm_flags(args) -> tuple[str, str]:
    """(run prelude, learner flags) carrying the algorithm to a ports island."""

    expected = getattr(args, "rl_expected_algorithm_sha256", None)
    if expected is None:
        return "", ""
    prelude = ""
    flags = f" --rl-expected-algorithm-sha256 {expected}"
    if getattr(args, "rl_algorithm_spec", None):
        prelude = (
            "mkdir -p ~/yeto-rl && printf '%s' "
            f"{shlex.quote(args.rl_algorithm_spec_json)} > ~/yeto-rl/algorithm_spec.json\n"
        )
        flags += " --rl-algorithm-spec ~/yeto-rl/algorithm_spec.json"
    # rl-infra-spec 2.1 placement options, forwarded only when non-default
    # (the default argv is unchanged).
    if getattr(args, "rl_placement", "colocated") != "colocated":
        flags += f" --rl-placement {shlex.quote(args.rl_placement)}"
    if getattr(args, "rl_standby_gpus", 0):
        flags += f" --rl-standby-gpus {int(args.rl_standby_gpus)}"
    if getattr(args, "rl_placement", "colocated") == "fixed-partition":
        flags += f" --rollout-num-gpus {int(args.rollout_num_gpus)}"
    if getattr(args, "rl_island_placement", None):
        flags += " --rl-per-island-placement"
    allowed = list(getattr(args, "rl_allow_unverified_mechanism", None) or ())
    for name in allowed:
        flags += f" --rl-allow-unverified-mechanism {shlex.quote(name)}"
    # rl-algo-critic-family 5.3: critic warm-up options, forwarded only when set
    # (every other argv is unchanged).
    if getattr(args, "rl_critic_load", None):
        flags += (f" --rl-critic-load {shlex.quote(args.rl_critic_load)}"
                  f" --rl-critic-init-sha256 {shlex.quote(args.rl_critic_init_sha256)}")
    if getattr(args, "rl_critic_baseline_rounds", 0):
        flags += f" --rl-critic-baseline-rounds {int(args.rl_critic_baseline_rounds)}"
    infra_prelude, infra_flags = _ports_infra_flags(args)
    return prelude + infra_prelude, flags + infra_flags
