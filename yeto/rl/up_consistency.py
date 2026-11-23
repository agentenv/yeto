"""Trainer / sampler consistency checks of ``yeto run up`` preflight.

The trainer (``trainer.argv``, a ``yeto launch``) and the samplers (each pool's
sampler spec) are configured in different files, and each side's own dry run
checks only itself. A value the two sides must share and do not would otherwise
surface after the trainer island loaded its weights (a K3.1 B300 island: about
21 minutes) or, worse, never fail and train on the wrong data:

* group size: trainer ``--n-samples-per-prompt`` vs spec ``sampling.n``. A larger
  sampler group is consumed and never trained (the reader takes at most
  ``group_size`` samples per group); a smaller one is padded with loss-free
  samples every round;
* response width: spec ``agentic.max_response_len`` (or ``sampling.max_tokens``)
  vs trainer ``--rollout-max-response-len``: the buffer reader does not cut a
  longer response to the trainer's width;
* row width (agentic specs): ``agentic.max_prompt_len`` + ``agentic.max_response_len``,
  at most the engine context length ``max_model_len`` (a trial ends there), vs the
  trainer's ``--seq-len`` (the memory plan of the island). A plain spec's
  ``max_prompt_tokens`` cuts no prompt (only the engine capacity check reads it), so
  it bounds no row and is not compared;
* routing replay (R3): a trainer with ``--use-rollout-routing-replay`` (always on
  in the ``kimi_k31`` recipe) needs ``rollout_fields.routed_experts`` in the spec;
* expert selection: the sha256 of ``--rl-expert-selection`` vs the spec's
  ``expert_selection_sha256`` / ``expert_trainable_sha256`` (the sampler refuses
  every release of another selection and keeps serving the base policy);
* the store: ``run.json`` ``store``, the trainer's ``--rl-checkpoint-store`` and
  each pool's ``s3://<bucket>/runs/<run_prefix>`` must be one store (the trainer
  reads the buffer and writes the releases under its own; the first-group wait
  lists the run config's);
* the release directory a sampler follows must be one the trainer side writes;
* context compaction: a spec with ``agentic.compaction`` needs a Miles trainer with a
  GRPO-family estimator, the rollout-side schedule (dynamic batch size, none of
  :data:`SPLIT_BY_SAMPLE_FLAGS`) and rows of at least ``context_budget``
  (:func:`compaction_problem`; verl and PPO CompactionRL are refused);
* backlog cap vs release cadence (:func:`backlog_findings`, a single trainer):
  a sampler's ``max_backlog_groups`` counts from the newest release's read
  position, so the samplers together must hold the groups the trainer reads
  between two releases. Refused when the trainer can never read up to its first
  release (``--rl-release-every`` x ``--rollout-batch-size`` groups) even with every
  sampler at its cap; a warning with the value to set when the samplers would wait
  at their caps between releases;
* prices (:func:`price_problem`): with a spend cap (``budget.usd``) every pool of a
  USD-priced provider (Modal / Daytona / Sky) needs a real hourly price, or the run's
  spend estimate counts its islands at 0 and the cap never fires on them.

Only local files are read (the run config, pool configs, specs, the store env
file's bucket name, the selection file); no cloud API, no store access.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
from typing import Any

SHA256_HEX = frozenset("0123456789abcdef")


def trainer_args(argv: list[str]) -> Any | None:
    """``trainer.argv`` through the ``yeto launch`` parser (its defaults included);
    None when it does not parse (``yeto launch --dry-run`` reports that)."""
    from yeto.cli import build_parser  # noqa: PLC0415

    try:
        # stdout too: an argv with --help / --version would print to the run's log
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            args = build_parser().parse_args(["launch", *[str(a) for a in argv]])
    except SystemExit:
        return None
    if getattr(args, "training_mode", None) != "rl":
        return None
    return args


def effective_spec(pool_cfg: dict[str, Any]) -> dict[str, Any] | None:
    """The pool's sampler spec with the pool's ``sampler`` knobs (``n``,
    ``max_tokens``, ...) laid over it, as the pool renders it for a node; None
    when there is none or it does not load (the pool dry-run reports that)."""
    if not pool_cfg.get("spec"):
        return None
    try:
        text = Path(os.path.expanduser(str(pool_cfg["spec"]))).read_text()
        text = text.replace("__RUN_PREFIX__", str(pool_cfg.get("run_prefix") or "").strip("/"))
        spec = json.loads(text)
        if not isinstance(spec, dict):
            return None
        from yeto.launch.nc import cli as nc_cli  # noqa: PLC0415
        from yeto.launch.nc import tasks  # noqa: PLC0415

        return tasks.apply_sampler_overrides(spec, **nc_cli._overrides(pool_cfg.get("sampler") or {}))
    except (OSError, ValueError, TypeError):
        return None


def _sha256(value: Any) -> str | None:
    """A 64-hex sha256 (a ``<placeholder>`` or anything else: None)."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if len(v) == 64 and set(v) <= SHA256_HEX else None


def _norm_store(uri: str) -> str:
    uri = str(uri).strip().rstrip("/")
    return "s3://" + uri[len("r2://"):] if uri.startswith("r2://") else uri


def pool_store(pool_cfg: dict[str, Any]) -> str | None:
    """``s3://<bucket>/runs/<run_prefix>``: where the pool's samplers write (the
    bucket from the store env file, as the pool reads it); None when unknown."""
    prefix, env_file = pool_cfg.get("run_prefix"), pool_cfg.get("r2_env_file")
    if not prefix or not env_file:
        return None
    from yeto.launch.nc import elastic  # noqa: PLC0415

    try:
        bucket = elastic.read_env_file(os.path.expanduser(str(env_file))).get(pool_cfg.get("bucket_env", "R2_BUCKET"))
    except OSError:
        return None
    return f"s3://{bucket}/runs/{str(prefix).strip('/')}" if bucket else None


def group_problem(args: Any, spec: dict[str, Any]) -> str | None:
    n = (spec.get("sampling") or {}).get("n")
    want = int(getattr(args, "n_samples_per_prompt", 0) or 0)
    if n is None or not want or int(n) == want:
        return None
    effect = ("every group is larger than the trainer's group and is consumed without being trained"
              if int(n) > want else
              f"every round is padded with {want - int(n)} loss-free samples per group")
    return (f"the sampler spec's sampling.n {n} != the trainer's --n-samples-per-prompt {want}"
            f"{' (the launcher default)' if '--n-samples-per-prompt' not in getattr(args, '_argv', ()) else ''}: "
            f"{effect}; set both to the same group size")


def _caps(spec: dict[str, Any]) -> tuple[str, int | None, str, int | None]:
    agentic = spec.get("agentic")
    if isinstance(agentic, dict):
        p, r = agentic.get("max_prompt_len"), agentic.get("max_response_len")
        return ("agentic.max_prompt_len", None if p is None else int(p),
                "agentic.max_response_len", None if r is None else int(r))
    p, r = spec.get("max_prompt_tokens"), (spec.get("sampling") or {}).get("max_tokens")
    return ("max_prompt_tokens", None if p is None else int(p),
            "sampling.max_tokens", None if r is None else int(r))


def length_problems(args: Any, spec: dict[str, Any]) -> list[str]:
    try:
        p_name, prompt, r_name, response = _caps(spec)
    except (TypeError, ValueError):
        return []
    problems = []
    width = int(getattr(args, "rollout_max_response_len", 0) or 0)
    if response is not None and width and response > width:
        problems.append(f"the sampler spec's {r_name} {response} > the trainer's --rollout-max-response-len "
                        f"{width}: the trainer would get responses longer than its row width (the buffer "
                        f"reader does not cut them); lower {r_name} or raise --rollout-max-response-len")
    seq = max(int(getattr(args, "seq_len", 0) or 0), width)  # yeto launch raises --seq-len to the response
    # Only the agentic caps bound a row (a longer first prompt is ABORTED, the response is
    # cut). A plain spec's max_prompt_tokens is only the engine capacity claim
    # (max_tokens + max_prompt_tokens <= max_model_len; no prompt is cut or dropped at it),
    # often set to max_model_len - max_tokens: it says nothing about the rows.
    if not p_name.startswith("agentic."):
        return problems
    row = None if response is None or prompt is None else prompt + response
    limit = spec_max_model_len_of(spec)
    if row is not None and limit is not None:
        # an agentic stream (first prompt + every turn after it) never outgrows the engine context: the
        # records proxy ends the trial at max_model_len (context_full), with or without compaction. So
        # max_response_len may be the engine context (as a compacting spec needs: compaction_problem
        # checks the trainer's row against it) and the sum overstates the row
        row = min(row, limit)
    if row is not None and seq and row > seq:
        capped = f" (capped at the engine context length max_model_len {limit})" if row < prompt + response else ""
        problems.append(f"the sampler spec's {p_name} {prompt} + {r_name} {response} = {prompt + response}"
                        f"{capped} > the trainer's --seq-len {seq}: the island is planned for shorter rows than "
                        f"the samplers write; raise --seq-len or lower the sampler caps")
    return problems


def trainer_r3(args: Any) -> bool:
    return bool(getattr(args, "use_rollout_routing_replay", False)) or \
        getattr(args, "rl_model_recipe", None) == "kimi_k31"  # RECIPE_ARGV always replays the routing


def r3_problem(args: Any, spec: dict[str, Any]) -> str | None:
    if not trainer_r3(args):
        return None
    if (spec.get("rollout_fields") or {}).get("routed_experts"):
        return None
    why = "the kimi_k31 recipe" if getattr(args, "rl_model_recipe", None) == "kimi_k31" \
        else "--use-rollout-routing-replay"
    return (f"the trainer replays the rollout routing (R3, {why}) but the sampler spec has no "
            "rollout_fields.routed_experts: its groups carry no routed experts; set "
            "rollout_fields.routed_experts true")


def selection_problem(args: Any, spec: dict[str, Any]) -> str | None:
    path = getattr(args, "rl_expert_selection", None)
    want_sel, want_tr = _sha256(spec.get("expert_selection_sha256")), _sha256(spec.get("expert_trainable_sha256"))
    if not path or not (want_sel or want_tr):
        return None
    local = Path(os.path.expanduser(str(path)))
    if not local.is_file():
        return None  # a path on the trainer node: the launch checks it there
    from yeto.rl.adapters.miles.models.partial_experts.selection import ExpertSelection  # noqa: PLC0415

    try:
        selection = ExpertSelection.load_file(local)
    except Exception:  # noqa: BLE001 - yeto launch --dry-run reports a bad selection file
        return None
    tail = ("the sampler refuses every release of this trainer and keeps serving the base policy, "
            "so the trainer waits on --rl-max-policy-age")
    if want_tr:  # replaces the selection check on the sampler (engine_sampler)
        if selection.trainable_sha256 != want_tr:
            return (f"the sampler spec's expert_trainable_sha256 {want_tr[:12]}.. != the trainable sha256 "
                    f"{selection.trainable_sha256[:12]}.. of --rl-expert-selection {path}: {tail}")
        return None
    if selection.sha256 != want_sel:
        return (f"the sampler spec's expert_selection_sha256 {want_sel[:12]}.. != the sha256 "
                f"{selection.sha256[:12]}.. of --rl-expert-selection {path}: {tail}")
    return None


def release_dir_problem(spec: dict[str, Any]) -> str | None:
    from yeto.rl.quant.schemes import ALL_SCHEMES, release_dir_for  # noqa: PLC0415

    want = spec.get("release_dir")
    known = sorted({release_dir_for(s) for s in ALL_SCHEMES})
    if want is None or str(want).strip("/") in known:
        return None
    return (f"the sampler spec's release_dir {want!r} is no release directory the trainer side writes "
            f"({', '.join(known)}): the sampler would never see a release")


# Miles flags under which its training side splits a round by sample count instead of
# the rollout-side DP schedule (one rollout per trial, all of a trial's segments in one
# step): a compacted round is then refused on the trainer at its first group
# (buffer_rollout.segmented_round_problems). None of them is a yeto launch flag (yeto
# refuses --indep-dp with the fault-tolerance flags); they are named here so a trainer
# argv that carries one is refused before anything is ordered.
SPLIT_BY_SAMPLE_FLAGS = {
    "--indep-dp": "--indep-dp",
    "--multi-lora": "multi-LoRA",
    "--multimodal-keys": "--multimodal-keys",
    "--use-dynamic-global-batch-size": "--use-dynamic-global-batch-size",
}


def segment_schedule_problems(args: Any | None, argv_flags: Any = ()) -> list[str]:
    """Why the trainer of ``trainer.argv`` would not train compacted trials one rollout
    each ([] when it does): a non-Miles backend (Miles trains on Megatron), no dynamic
    batch size (``--rl-max-tokens-per-gpu``: a static micro batch asserts when the step's
    sample count varies with the segments), or a flag of :data:`SPLIT_BY_SAMPLE_FLAGS`.
    ``args``: the parsed argv (None: it does not parse); ``argv_flags``: its flag names."""
    from yeto.rl.engine.backends import DEFAULT_BACKEND  # noqa: PLC0415

    flags = {str(a).split("=", 1)[0] for a in argv_flags}
    out = [why for flag, why in SPLIT_BY_SAMPLE_FLAGS.items() if flag in flags]
    if args is None:
        return out
    backend = getattr(args, "rl_backend", None) or DEFAULT_BACKEND
    if backend != DEFAULT_BACKEND:
        out.append(f"--rl-backend {backend} (not the Megatron trainer of the Miles backend)")
    if getattr(args, "rl_max_tokens_per_gpu", None) is None:
        out.append("no --rl-max-tokens-per-gpu (Miles --use-dynamic-batch-size)")
    return out


def segment_schedule_text(problems: list[str]) -> str:
    return ("the sampler spec's agentic.compaction writes several samples per trial, but this trainer would "
            f"split a round by sample count ({'; '.join(problems)}): a trial's segments could land in different "
            "steps and the trainer stops at the first compacted group; use the Miles backend with "
            "--rl-max-tokens-per-gpu and none of the listed flags")


def compaction_problem(args: Any, spec: dict[str, Any]) -> str | None:
    """A bad ``agentic.compaction`` section (:func:`yeto.rl.compaction.config_from_spec`),
    a trainer that cannot train its segment samples
    (:func:`yeto.launch.harness_env.check_grpo_compaction`), or a trainer row too
    narrow for a segment: a segment is up to ``context_budget`` tokens, and a row
    shorter than that cuts the summary that ends it."""
    from yeto.launch.harness_env import check_grpo_compaction  # noqa: PLC0415
    from yeto.rl.compaction import config_from_spec  # noqa: PLC0415

    try:
        cfg = config_from_spec(spec)
        check_grpo_compaction(args, cfg)
    except (TypeError, ValueError) as exc:
        return str(exc)
    if cfg is None:
        return None
    schedule = segment_schedule_problems(args, getattr(args, "_argv", ()))
    if schedule:
        return segment_schedule_text(schedule)
    from yeto.rl.compaction import segment_width  # noqa: PLC0415

    width, _warning = segment_width(spec, cfg)  # the warning: compaction_lines
    seq = max(int(getattr(args, "seq_len", 0) or 0), int(getattr(args, "rollout_max_response_len", 0) or 0))
    if seq < width:
        what = ("engine context length max_model_len" if spec_max_model_len_of(spec) is not None
                else "agentic.compaction.context_budget")
        return (f"the trainer's --seq-len {seq} < the sampler spec's {what} {width}: a compaction segment can "
                f"grow to the engine context (a large tool output after the last step below the trigger, then "
                f"the summary) and a shorter row cuts the summary at its end; raise --seq-len (and "
                f"--rollout-max-response-len) to at least {width}")
    return None


def spec_max_model_len_of(spec: dict[str, Any]) -> int | None:
    from yeto.rl.compaction import spec_max_model_len  # noqa: PLC0415

    try:
        return spec_max_model_len(spec)
    except (TypeError, ValueError):
        return None


def compaction_switch_warnings(spec: dict[str, Any], args: Any | None) -> list[str]:
    """Preflight warnings of a compacting spec that only the trainer and the sampler
    together decide. A compacted trial runs several segments, often longer than a
    release interval; a drain does not spare it. On a drain-mode sampler:
    (a) an effective drain cut below the suggested floor; (b) the trainer's
    ``--rl-max-policy-age`` at most 2 (a long group ages past it); (c) a release every
    version (no ``--rl-release-every N`` > 1, no ``--rl-release-skip-busy``), so a
    drain starts at every step. On any sampler: the trial wall clock
    (``agentic.task_defaults.wall_clock_s``) left at the default."""
    from yeto.rl.agentic.types import AgentBudget  # noqa: PLC0415
    from yeto.rl.engine import agentic_round as ar  # noqa: PLC0415

    section = spec.get("agentic") if isinstance(spec.get("agentic"), dict) else {}
    if section.get("compaction") is None:
        return []
    out = []
    switch = str(section.get("weight_switch", "auto"))
    if not (switch == "live" or (switch == "auto" and ar.spec_live_engine(spec))):
        floor = float(ar.COMPACTION_DRAIN_FLOOR_S)  # the smallest drain cut suggested with agentic.compaction
        try:
            cut, _grace = ar.drain_settings(spec, drain=True)
        except ValueError:
            cut = None  # the pool dry-run reports a bad drain setting
        if cut is not None and cut < floor:
            out.append(f"agentic.compaction on a drain-mode sampler: the effective drain cut {cut:g} s "
                       f"(min of agentic.drain_timeout_s, reload_max_defer_s) is below {floor:g} s (an estimate for "
                       "a 262k budget and 3 compactions): compacted trials running at a release are cancelled "
                       "with all their segments; raise drain_timeout_s and reload_max_defer_s together, or use "
                       "weight_switch live on an engine with live_update")
        age = getattr(args, "rl_max_policy_age", None)
        if args is not None and age is not None and int(age) <= 2:
            out.append(f"agentic.compaction on a drain-mode sampler with the trainer's --rl-max-policy-age "
                       f"{int(age)}: a compacted group is as old as its oldest segment and ages past it; raise it to "
                       "about the release interval N + 6 (upload and download lag)")
        if args is not None and int(getattr(args, "rl_release_every", 1) or 1) <= 1 \
                and not getattr(args, "rl_release_skip_busy", False):
            out.append("agentic.compaction on a drain-mode sampler with a release every version: every step "
                       "starts a drain that cuts the long trials; release less often (--rl-release-every N, N about "
                       "the compacted trials' p90 wall time / step time) or --rl-release-skip-busy")
    defaults = section.get("task_defaults") if isinstance(section.get("task_defaults"), dict) else {}
    try:
        wall = None if defaults.get("wall_clock_s") is None else float(defaults["wall_clock_s"])
    except (TypeError, ValueError):
        return out  # the sampler refuses a bad wall_clock_s at its first trial
    if wall is None or wall <= AgentBudget.wall_clock_s:
        out.append(f"agentic.compaction with the trial wall clock at "
                   f"{AgentBudget.wall_clock_s if wall is None else wall:g} s"
                   f"{' (the default)' if wall is None else ''}: a compacted trial that runs longer is aborted "
                   "(wall_clock_cap) and not retried; set agentic.task_defaults.wall_clock_s (and agent_timeout_s) "
                   "to the compacted trials' p99 wall time")
    return out


def compaction_lines(pool_cfg: dict[str, Any], trainer_argv: list[str] | None = None) -> list[str]:
    """Preflight log lines of the pool's ``agentic.compaction``: its effective values
    and the warnings of :func:`compaction_switch_warnings` (empty: off, or the spec
    does not load; the pool dry-run reports a bad section)."""
    from yeto.rl.compaction import config_from_spec, describe  # noqa: PLC0415

    spec = effective_spec(pool_cfg)
    try:
        cfg = None if spec is None else config_from_spec(spec)
    except (TypeError, ValueError):
        return []
    if cfg is None:
        return []
    from yeto.rl.compaction import segment_width  # noqa: PLC0415

    args = trainer_args(trainer_argv) if trainer_argv is not None else None
    _width, width_warning = segment_width(spec, cfg)
    micro = micro_batch_warning(args)
    return [f"agentic.compaction {describe(cfg)}"] + [
        f"WARNING: {w}" for w in [*([width_warning] if width_warning else []), *([micro] if micro else []),
                                  *compaction_switch_warnings(spec, args)]]


def micro_batch_warning(args: Any | None) -> str | None:
    """A Miles micro batch (``--rl-max-tokens-per-gpu`` x CP tokens, packed over the CP
    ranks: dp_schedule) wider than one ``--seq-len`` row: it can pack two full-length
    rows (two segments of a compacted trial, say), more activation memory than one row
    at ``--seq-len``, the shape the memory estimate and the 262k layouts are sized for.
    None: no such trainer, or ``--rl-max-tokens-per-gpu`` x CP <= ``--seq-len``."""
    if args is None or getattr(args, "rl_max_tokens_per_gpu", None) is None:
        return None
    try:
        per_gpu = int(args.rl_max_tokens_per_gpu)
        cp = max(1, int(getattr(args, "context_parallel", 1) or 1))
        seq = int(getattr(args, "seq_len", 0) or 0)
    except (TypeError, ValueError):
        return None
    if seq <= 0 or per_gpu * cp <= seq:
        return None
    return (f"--rl-max-tokens-per-gpu {per_gpu} x CP {cp} = {per_gpu * cp} tokens per micro batch > --seq-len "
            f"{seq}: Miles packs up to that many tokens into one micro batch, so it can hold more than one "
            f"full-length row (two compaction segments) and run out of memory; set --rl-max-tokens-per-gpu "
            f"to --seq-len / CP ({-(-seq // cp)})")


def _group_tokens(spec: dict[str, Any]) -> int | None:
    """The most stream tokens one trial / sample of the spec carries (None: unknown):
    the rows the sampler writes (``agentic.max_prompt_len + max_response_len``, or a
    plain spec's ``max_prompt_tokens + sampling.max_tokens``), at most the engine
    context length."""
    from yeto.rl.compaction import spec_max_model_len  # noqa: PLC0415

    try:
        p_name, prompt, _r_name, response = _caps(spec)
        limit = spec_max_model_len(spec)
    except (TypeError, ValueError):
        return None
    rows = None if prompt is None or response is None else prompt + response
    known = [v for v in (rows, limit) if v is not None]
    return min(known) if known else None


def payload_size_warning(pool_cfg: dict[str, Any], trainer_argv: list[str] | None = None,
                         environ: dict[str, str] | None = None) -> str | None:
    """A warning when one buffer group of the pool's samplers can exceed one object
    PUT (5 GiB on S3 / R2 / OSS): ``sampling.n`` trials x (1 + ``max_compactions``)
    segments x the segment width (:func:`yeto.rl.compaction.segment_width`) with compaction, else x the row tokens
    (:func:`_group_tokens`), sized by ``buffer_payload.k31_group_payload_bytes`` (the
    K3.1 bytes per token: R3 + top-128). Only for specs that return routed experts
    (R3 is most of the bytes). The codec is ``YETO_SAMPLE_BUFFER_PAYLOAD_CODEC`` of
    ``environ`` (the run's env). None: below the limit or not a K3.1-shaped spec."""
    from yeto.rl.engine import buffer_payload  # noqa: PLC0415

    size_of = buffer_payload.k31_group_payload_bytes
    limit = buffer_payload.SINGLE_PUT_LIMIT_BYTES
    spec = effective_spec(pool_cfg)
    if spec is None or not (spec.get("rollout_fields") or {}).get("routed_experts"):
        return None
    from yeto.rl.compaction import config_from_spec  # noqa: PLC0415

    try:
        cfg = config_from_spec(spec)
    except (TypeError, ValueError):
        return None  # the pool dry-run reports a bad section
    args = trainer_args(trainer_argv) if trainer_argv is not None else None
    trials = (spec.get("sampling") or {}).get("n") or getattr(args, "n_samples_per_prompt", None)
    from yeto.rl.compaction import segment_width  # noqa: PLC0415

    tokens = segment_width(spec, cfg)[0] if cfg is not None else _group_tokens(spec)  # a segment: up to max_model_len
    if not trials or not tokens:
        return None
    compactions = 0 if cfg is None else cfg.max_compactions
    env_name = buffer_payload.PAYLOAD_CODEC_ENV
    codec = str((environ or {}).get(env_name) or "json").strip().lower()
    try:
        size = size_of(int(trials), int(tokens), max_compactions=compactions, codec=codec)
        zstd = size_of(int(trials), int(tokens), max_compactions=compactions, codec="zstd")
    except (KeyError, TypeError, ValueError):
        return None
    if size <= limit:
        return None
    shape = (f"{int(trials)} trials x {1 + compactions} segments x {int(tokens)} tokens (agentic.compaction)"
             if cfg is not None else f"{int(trials)} trials x {int(tokens)} tokens")
    return (f"one buffer group can reach {size / 2**30:.1f} GiB ({shape}, codec {codec}), over the "
            f"{limit / 2**30:.0f} GiB single-object PUT limit of S3 / R2 / OSS: such a group needs a multipart "
            f"write; set {env_name}=zstd on the samplers (about {zstd / 2**30:.1f} GiB) or lower the rows"
            + (" / context_budget / max_compactions" if cfg is not None else ""))


def _positive_int(value: Any) -> int | None:
    """``value`` as an int > 0 (a ``<placeholder>``, a bool or anything else: None)."""
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def pool_backlog(pool_cfg: dict[str, Any]) -> dict[str, int] | None:
    """The backlog shape of one pool's samplers: ``cap`` (``max_backlog_groups`` as a
    node runs it: the spec's, or the pool's ``backlog`` section, ``nc.cli``),
    ``target`` (samplers the pool keeps running), ``most`` (the most it runs at
    once: ``elastic.max_nodes`` when larger) and ``over`` (the groups one sampler
    can write past its cap: a round, or the groups of one agentic call, upper
    bound). None when the pool does not say (no spec, a placeholder)."""
    spec = effective_spec(pool_cfg)
    target = _positive_int(pool_cfg.get("target"))
    if spec is None or target is None:
        return None
    prompts = spec.get("prompts") or {}
    per_round = _positive_int(prompts.get("groups_per_round")) or 1
    cap = _positive_int(spec.get("max_backlog_groups"))
    backlog = pool_cfg.get("backlog")
    if backlog:
        from yeto.launch.nc import tasks  # noqa: PLC0415

        try:
            cap = int(tasks.backlog_groups(groups_per_round=per_round,
                                           samplers=len(spec.get("buffer_shards") or [None]), **backlog))
        except (TypeError, ValueError):
            return None  # the pool dry-run reports a bad backlog section
    if cap is None:
        return None
    agentic = spec.get("agentic") if isinstance(spec.get("agentic"), dict) else {}
    in_flight = _positive_int(agentic.get("max_in_flight"))
    n = _positive_int((spec.get("sampling") or {}).get("n")) or 1
    over = max(per_round, -(-in_flight // n) if in_flight else 0)
    most = max(target, _positive_int((pool_cfg.get("elastic") or {}).get("max_nodes")) or 0)
    return {"cap": cap, "target": target, "most": most, "over": over}


def backlog_findings(trainer_argv: list[str], pools: list[tuple[dict[str, Any], str]]
                     ) -> tuple[list[str], list[str]]:
    """``(problems, warnings)`` of the samplers' backlog caps against a single
    buffer trainer's release cadence (``pools``: each pool's config and data path).

    A sampler's backlog counts from the read position of the newest release it
    sees, 0 before the first (``SamplerLoop.backlog``): until a release lands, the
    samplers together write at most the sum of their caps (plus the groups each
    can write past it). Checked for a trainer with ``--rl-sample-role trainer``
    and ``--rl-single-island-no-sync`` over ``r2`` pools (synced trainers size
    their samplers by the pool's ``backlog`` section; relay pools have no cap).

    * problem: even with every pool at its most samplers, each at its cap plus
      the most it writes past it, the trainer cannot read the ``release_every x
      groups_per_round`` groups up to its first release; it would wait for groups
      until ``--rl-sample-wait-s`` ends the run (only a replacement node's new
      shard could move it on), so nothing should be ordered;
    * warning: the target samplers hold fewer than
      :func:`yeto.rl.engine.sample_roles.release_backlog_need` groups: they wait
      at their caps between releases (sampling time lost every release, a trainer
      waiting for groups). The value to set is that need over the samplers,
      rounded up, and over one sampler fewer (one down or being replaced)."""
    from yeto.rl.engine import sample_roles  # noqa: PLC0415

    args = trainer_args(trainer_argv)
    if args is None or getattr(args, "rl_sample_role", None) != "trainer" \
            or not getattr(args, "rl_single_island_no_sync", False):
        return [], []
    groups = _positive_int(getattr(args, "rollout_batch_size", None))
    every = _positive_int(getattr(args, "rl_release_every", None)) or 1
    if groups is None:
        return [], []
    shapes = [shape for cfg, data_path in pools if data_path == "r2" and (shape := pool_backlog(cfg))]
    if not shapes:
        return [], []
    staged = getattr(args, "tuning", None) != "lora" and not getattr(args, "rl_release_inline", False)
    skip_busy = bool(staged and getattr(args, "rl_release_skip_busy", False))
    rounds = sample_roles.release_record_rounds(release_every=every, staged=staged, skip_busy=skip_busy)
    need = sample_roles.release_backlog_need(groups_per_round=groups, release_every=every, record_rounds=rounds)
    first = every * groups
    most = sum(s["most"] * (s["cap"] + s["over"]) for s in shapes)
    caps = ", ".join(f"{s['target']} x {s['cap']}" for s in shapes)
    samplers = sum(s["target"] for s in shapes)
    held = sum(s["target"] * s["cap"] for s in shapes)
    fewer = max(1, samplers - 1)
    suggest, suggest_fewer = -(-need // samplers), -(-need // fewer)
    advice = (f"set max_backlog_groups to at least {suggest} per sampler"
              + (f" ({suggest_fewer} to keep up with one of the {samplers} samplers down or being replaced)"
                 if samplers > 1 else "")
              + f" = ceil(((release_every {every} + m {rounds} + 1) x {groups} + "
                f"{sample_roles.RELEASE_BACKLOG_MARGIN_GROUPS}) / samplers), or release more often")
    if most < first:
        return [(f"the samplers ({caps} max_backlog_groups; at most {most} groups with every node a pool may run, "
                 f"each past its cap) count their backlog from the newest release's read position, 0 before the "
                 f"first release, but the trainer reads --rl-release-every {every} x --rollout-batch-size {groups} "
                 f"= {first} groups before it can release: it would wait for groups and never release; "
                 + advice)], []
    if held >= need:
        return [], []
    upload = (f"m = {rounds}: the round after a publication plus a staged upload assumed "
              f"{sample_roles.ASSUMED_UPLOAD_ROUNDS} rounds long; re-derive with the measured upload / step time"
              if staged else "m = 0: releases written inline")
    return [], [f"the samplers ({caps} max_backlog_groups = {held} groups) count their backlog from the newest "
                f"release's read position, and the trainer reads up to {need} groups between two releases "
                f"(--rl-release-every {every}, --rollout-batch-size {groups}, {upload}): the samplers would wait "
                f"at their caps between releases; " + advice]


def store_problems(run_store: str, args: Any, pool_cfg: dict[str, Any], *, data_path: str = "r2") -> list[str]:
    problems = []
    run = _norm_store(run_store)
    trainer = getattr(args, "rl_checkpoint_store", None)
    trainer = _norm_store(trainer) if trainer else None
    if trainer and trainer != run:
        problems.append(f"the trainer's --rl-checkpoint-store {trainer} != the run config's store {run}: "
                        "the first-group wait and the dashboard read another store than the trainer")
    if data_path != "r2":
        return problems  # the samplers write through the syncer relay, not the store
    pool = pool_store(pool_cfg)
    if pool is None:
        return problems
    root = trainer or run
    sub = pool[len(root) + 1:] if pool.startswith(root + "/") else ""
    synced_substore = bool(trainer) and sub.startswith("trainer") and sub[len("trainer"):].isdigit()
    if pool != root and not synced_substore:  # samplers that follow their trainer: <store>/trainer<k>
        problems.append(f"the pool's samplers write to {pool} (store env bucket + runs/<run_prefix>) but the "
                        f"trainer reads {root}: no group would reach the trainer; fix run_prefix / the bucket "
                        "or the store URIs")
    return problems


def problems(run_store: str, trainer_argv: list[str], pool_cfg: dict[str, Any], *,
             data_path: str = "r2") -> list[str]:
    """Every trainer / sampler mismatch of one pool (empty: consistent, or not checkable)."""
    args = trainer_args(trainer_argv)
    if args is None:  # yeto launch --dry-run reports the argv; a compacting pool also names the split flags
        spec = effective_spec(pool_cfg)
        compacting = isinstance((spec or {}).get("agentic"), dict) and spec["agentic"].get("compaction") is not None
        split = segment_schedule_problems(None, trainer_argv) if compacting else []
        return [segment_schedule_text(split)] if split else []
    args._argv = [str(a).split("=", 1)[0] for a in trainer_argv]
    out = store_problems(run_store, args, pool_cfg, data_path=data_path)
    spec = effective_spec(pool_cfg)
    if spec is None:
        return out
    out += [p for p in (group_problem(args, spec), r3_problem(args, spec), selection_problem(args, spec),
                        release_dir_problem(spec), compaction_problem(args, spec)) if p]
    out += length_problems(args, spec)
    return out


# where an hourly price comes from, per USD-priced provider kind (the preflight hint)
PRICE_SOURCES = {
    "modal": "the Modal pricing page (modal.com/pricing): GPU price per second x 3600 x gpus, plus CPU / memory",
    "daytona": "your Daytona contract / dashboard rate for the sandbox class (spot or on-demand)",
    "sky": "the cloud's price for the instance type (`sky show-gpus --all` lists them)",
}
NO_CAP_USD = 1e9  # budget.usd absent or null: no spend cap (the monitor's default)


def number_or_default(section: dict[str, Any] | None, key: str, default: float) -> float:
    """``section[key]`` as a float; absent or ``null`` gives ``default`` (a run config's
    ``trainer.usd_per_h: null`` must not crash the monitor's spend estimate mid-run)."""
    value = (section or {}).get(key)
    return float(default) if value is None else float(value)


def budget_usd(budget: dict[str, Any] | None) -> float:
    """The spend cap ``budget.usd``; absent or ``null`` means no cap (:data:`NO_CAP_USD`).
    A value that is not a number raises ValueError naming the field."""
    usd = (budget or {}).get("usd")
    if usd is None:
        return NO_CAP_USD
    try:
        return float(usd)
    except (TypeError, ValueError):
        raise ValueError(f"budget.usd must be a number of dollars or null (no cap), got {usd!r}") from None


DEFAULT_WALL_H = 3.0  # budget.wall_h absent: the wall clock stops the trainer after 3 h


def budget_wall_h(budget: dict[str, Any] | None) -> float | None:
    """The wall-clock cap ``budget.wall_h`` in hours. Absent means :data:`DEFAULT_WALL_H`;
    ``null`` means no wall-clock cap (``None``, like ``budget.usd: null``). A value that is
    not a positive number raises ValueError naming the field."""
    b = budget or {}
    if "wall_h" not in b:
        return DEFAULT_WALL_H
    wall = b["wall_h"]
    if wall is None:
        return None
    try:
        hours = float(wall)
    except (TypeError, ValueError):
        hours = float("nan")
    if isinstance(wall, bool) or not hours > 0 or hours == float("inf"):
        raise ValueError(f"budget.wall_h must be a positive number of hours or null (no wall-clock cap), "
                         f"got {wall!r}")
    return hours


def budget_capped(budget: dict[str, Any] | None) -> bool:
    """Whether the run sets a spend cap (``budget.usd`` a finite number below :data:`NO_CAP_USD`)."""
    usd = budget_usd(budget)
    return usd == usd and usd < NO_CAP_USD


def price_problem(name: str, entry_usd_per_h: float | None, pool_cfg: dict[str, Any]) -> str | None:
    """A pool of a USD-priced provider without a positive hourly price (with a spend cap set):
    the run's spend estimate would count its islands at $0. The entry's ``usd_per_h`` wins
    over the provider section's ``price_usd_h``; National Compute (ucr) is priced by
    ``budget.nc_usd_per_h`` and never refused here."""
    from yeto.launch.nc.cli import pool_price  # noqa: PLC0415
    from yeto.launch.pool import provider as providers  # noqa: PLC0415

    kind = providers.kind_of(pool_cfg) or "factory"
    entry = providers.registered(kind)
    if entry is None or entry.caps.currency != "usd":
        return None
    if entry_usd_per_h is not None:
        if entry_usd_per_h > 0:
            return None
        where, value = f"samplers[{name}].usd_per_h", entry_usd_per_h
    else:
        where = "provider.price_usd_h" + (" (or offers[0].price_usd_h)" if kind == "sky" else "")
        try:
            value = pool_price(pool_cfg)[0]
        except (TypeError, ValueError):
            value = (pool_cfg.get("provider") or {}).get("price_usd_h")
        if isinstance(value, (int, float)) and value > 0:
            return None
    state = "missing" if value is None else f"{value!r}"
    hint = f"; drop samplers[{name}].usd_per_h (it overrides the provider price) or" if where.startswith("samplers") \
        else "; set"
    return (f"pool {name} ({kind}, priced in USD): {where} is {state} while budget.usd caps the run: its islands "
            f"would be counted at $0 and the budget guard would not stop them{hint} a real USD price of one "
            f"island hour (source: {PRICE_SOURCES.get(kind, 'the provider price list')}); or set budget.usd "
            "to null to run without a spend cap")
