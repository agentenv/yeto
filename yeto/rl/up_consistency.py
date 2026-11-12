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
* row width (agentic specs): ``agentic.max_prompt_len`` + ``agentic.max_response_len``
  vs the trainer's ``--seq-len`` (the memory plan of the island). A plain spec's
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
* the release directory a sampler follows must be one the trainer side writes.

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
        with contextlib.redirect_stderr(io.StringIO()):
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
    if response is not None and prompt is not None and seq and prompt + response > seq:
        problems.append(f"the sampler spec's {p_name} {prompt} + {r_name} {response} = {prompt + response} > "
                        f"the trainer's --seq-len {seq}: the island is planned for shorter rows than the "
                        f"samplers write; raise --seq-len or lower the sampler caps")
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
    if args is None:
        return []
    args._argv = [str(a).split("=", 1)[0] for a in trainer_argv]
    out = store_problems(run_store, args, pool_cfg, data_path=data_path)
    spec = effective_spec(pool_cfg)
    if spec is None:
        return out
    out += [p for p in (group_problem(args, spec), r3_problem(args, spec), selection_problem(args, spec),
                        release_dir_problem(spec)) if p]
    out += length_problems(args, spec)
    return out
