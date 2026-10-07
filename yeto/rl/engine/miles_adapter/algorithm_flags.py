"""Miles algorithm flags <-> ``AlgorithmSpec`` fields (change
``rl-algorithm-capabilities``, design D3).

One table drives three things:

1. :func:`algorithm_argv` translates a spec into the upstream Miles argv
   fragment (values equal to the v1 default are not emitted, so the default
   GRPO argv stays byte-identical);
2. :func:`absorb_extra_argv` absorbs mapped flags found in extra argv into the
   spec (they then enter validation, the capability check and the hash); a
   value that disagrees with the spec is a conflict and startup is refused;
3. every mapped flag is adapter-owned (``config.ADAPTER_OWNED_FLAGS``).

:data:`OBJECTIVE_FLAGS` lists every upstream flag that changes the training
objective (loss / advantage / rollout groups of Miles ``arguments.py``). It
is a superset of the table; an objective flag without a mapping row is
refused in extra argv (:data:`UNMAPPED_OBJECTIVE_FLAGS`). When Miles is
upgraded, re-review this list against the new parser (``docs/MILES_RL.md``);
``tests/test_rl_algorithm_flags_upstream.py`` pins its existence upstream.

Follow-up changes add rows with :func:`register_flag` from their own module
(listed in ``yeto.rl.algos.EXTENSION_MODULES``) and remove the flag from the
unmapped set in the same call.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..algorithm import (
    CRITIC_ESTIMATORS,
    AlgorithmSpec,
    AlgorithmSpecError,
    PluginRef,
    load_extensions,
)

Assignment = tuple[str, Any]  # (spec field path, value)


class AlgorithmFlagConflict(AlgorithmSpecError):
    """An extra-argv algorithm flag disagrees with the algorithm spec."""


class UnmappedAlgorithmFlag(AlgorithmSpecError):
    """An objective-changing flag that the algorithm spec cannot express yet."""


@dataclass(frozen=True)
class FlagMapping:
    """One Miles flag. ``switch`` flags take no value (store_true)."""

    flag: str
    field: str  # primary spec field path
    switch: bool
    parse: Callable[[str], Any]  # raw argv string -> python value (value flags)
    absorb: Callable[[Any], list[Assignment]]  # parsed value -> spec assignments
    translate: Callable[[AlgorithmSpec], list[str]]  # spec -> argv fragment
    # Emitted in fixed R0 positions by config.translate_run_config itself.
    emitted_by_config: bool = False


def _float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise AlgorithmSpecError(f"expected a number, got {raw!r}") from exc


def _int(raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise AlgorithmSpecError(f"expected an integer, got {raw!r}") from exc


def _str(raw: str) -> str:
    return raw


def _plugin(raw: str) -> PluginRef:
    return PluginRef.from_path(raw)


def _num(value: float) -> str:
    return str(float(value))


def _value_row(flag: str, path: str, parse, *, emit: Callable[[Any], str] = str,
               default: Any = None) -> FlagMapping:
    def translate(spec: AlgorithmSpec) -> list[str]:
        value = spec.get_path(path)
        if value == default or value is None:
            return []
        if isinstance(value, PluginRef):
            return [flag, value.path]
        return [flag, emit(value)]

    return FlagMapping(flag, path, False, parse, lambda v: [(path, v)], translate)


def _switch_row(flag: str, path: str, value: Any) -> FlagMapping:
    def translate(spec: AlgorithmSpec) -> list[str]:
        return [flag] if spec.get_path(path) == value else []

    return FlagMapping(flag, path, True, lambda raw: True, lambda _: [(path, value)], translate)


def _kl_rows() -> list[FlagMapping]:
    def kl_coef(spec):  # reward placement; R0 emits it (config.translate_run_config)
        return []

    def use_kl_loss(spec):
        if spec.kl.placement != "loss":
            return []
        argv = ["--use-kl-loss", "--kl-loss-coef", _num(spec.kl.coef),
                "--kl-loss-type", spec.kl.estimator]
        if spec.kl.unbiased:
            argv.append("--use-unbiased-kl")
        return argv

    return [
        FlagMapping("--kl-coef", "kl.coef", False, _float,
                    lambda v: [("kl.placement", "reward"), ("kl.coef", v)], kl_coef,
                    emitted_by_config=True),
        FlagMapping("--use-kl-loss", "kl.placement", True, lambda raw: True,
                    lambda _: [("kl.placement", "loss")], use_kl_loss),
        FlagMapping("--kl-loss-coef", "kl.coef", False, _float,
                    lambda v: [("kl.coef", v)], lambda spec: []),
        FlagMapping("--kl-loss-type", "kl.estimator", False, _str,
                    lambda v: [("kl.estimator", v)], lambda spec: []),
        FlagMapping("--use-unbiased-kl", "kl.unbiased", True, lambda raw: True,
                    lambda _: [("kl.unbiased", True)], lambda spec: []),
    ]


def _correction_rows() -> list[FlagMapping]:
    def use_tis(spec):
        c = spec.correction
        if c.method not in ("tis", "custom"):
            return []
        argv = ["--use-tis"]
        if c.tis_clip is not None:
            argv += ["--tis-clip", _num(c.tis_clip)]
        if c.tis_clip_low is not None:
            argv += ["--tis-clip-low", _num(c.tis_clip_low)]
        if c.function is not None:
            argv += ["--custom-tis-function-path", c.function.path]
        return argv

    def use_opsm(spec):
        c = spec.correction
        if c.opsm_delta is None:
            return []
        return ["--use-opsm", "--opsm-delta", _num(c.opsm_delta)]

    return [
        FlagMapping("--use-tis", "correction.method", True, lambda raw: True,
                    lambda _: [("correction.method", "tis")], use_tis),
        FlagMapping("--tis-clip", "correction.tis_clip", False, _float,
                    lambda v: [("correction.tis_clip", v)], lambda spec: []),
        FlagMapping("--tis-clip-low", "correction.tis_clip_low", False, _float,
                    lambda v: [("correction.tis_clip_low", v)], lambda spec: []),
        FlagMapping("--custom-tis-function-path", "correction.function", False, _plugin,
                    lambda v: [("correction.method", "custom"), ("correction.function", v)],
                    lambda spec: []),
        _switch_row("--use-rollout-logprobs", "correction.use_rollout_logprobs", True),
        _switch_row("--get-mismatch-metrics", "correction.mismatch_metrics", True),
        FlagMapping("--use-opsm", "correction.method", True, lambda raw: True,
                    lambda _: [("correction.method", "opsm")], use_opsm),
        FlagMapping("--opsm-delta", "correction.opsm_delta", False, _float,
                    lambda v: [("correction.opsm_delta", v)], lambda spec: []),
    ]


def _estimator_absorb(value: str) -> list[Assignment]:
    pairs: list[Assignment] = [("advantage.estimator", value)]
    if value in CRITIC_ESTIMATORS:  # Miles: use_critic = estimator == "ppo" (:3591)
        pairs.append(("execution.needs_critic", True))
    return pairs


def _builtin_rows() -> list[FlagMapping]:
    return [
        FlagMapping("--advantage-estimator", "advantage.estimator", False, _str,
                    _estimator_absorb, lambda spec: [],
                    emitted_by_config=True),
        _value_row("--eps-clip", "loss.eps_clip", _float, emit=_num),
        _value_row("--eps-clip-high", "loss.eps_clip_high", _float, emit=_num),
        _value_row("--eps-clip-c", "loss.eps_clip_c", _float, emit=_num),
        _switch_row("--calculate-per-token-loss", "loss.aggregation", "token"),
        _switch_row("--disable-grpo-std-normalization", "advantage.std_normalization", False),
        _switch_row("--disable-rewards-normalization", "advantage.rewards_normalization", False),
        _switch_row("--normalize-advantages", "advantage.whiten", True),
        *_kl_rows(),
        _value_row("--entropy-coef", "entropy_coef", _float, emit=_num, default=0.0),
        *_correction_rows(),
        _value_row("--custom-pg-loss-reducer-function-path", "loss.reducer", _plugin),
        _value_row("--custom-reward-post-process-path", "advantage.reward_postprocess", _plugin),
        _value_row("--loss-type", "loss.variant", _str, default="policy_loss"),
        _value_row("--custom-loss-function-path", "loss.custom_loss", _plugin),
        FlagMapping("--dynamic-sampling-filter-path", "sampling.filter", False, _str,
                    lambda v: [("sampling.filter", v)], lambda spec: [],
                    emitted_by_config=True),
        FlagMapping("--over-sampling-batch-size", "sampling.over_sampling_batch_size", False,
                    _int, lambda v: [("sampling.over_sampling_batch_size", v)],
                    lambda spec: [], emitted_by_config=True),
    ]


# Objective-changing upstream flags that the spec cannot express yet (Miles
# arguments.py algo / rollout / reward groups at MILES_NEXT_COMMIT).
_UNMAPPED = [
    # critic flags: rows registered by yeto.rl.algos.critic (rl-algo-critic-family)
    "--gamma",
    "--lambd",
    "--value-clip",
    "--num-critic-only-steps",
    "--critic-load",
    "--critic-lr",
    "--critic-lr-warmup-iters",
    # fork-only (yeto-gae-variant ce96fc060, yeto-vapo cbf8c4737): rows registered by
    # yeto.rl.algos.critic (rl-algo-critic-family 7.2, VAPO); critic.FORK_FLAGS
    "--gae-variant",
    "--gae-lambd-mode",
    "--gae-length-alpha",
    "--gae-critic-lambd",
    "--positive-example-lm-loss-coef",
    "--positive-example-reward-threshold",
    "--ref-update-interval",
    "--disable-compute-advantages-and-returns",
    "--use-rollout-entropy",
    "--skip-actor-forward-only",
    "--reset-optimizer-states",
    "--use-routing-replay",
    "--opd-kl-coef",
    "--clip-grad",
    "--partial-rollout",
    "--mask-offpolicy-in-partial-rollout",
    "--max-weight-staleness",
    "--keep-old-actor",
    "--update-weights-interval",
    "--rollout-temperature",
    "--rollout-top-p",
    "--rollout-top-k",
    "--rollout-data-postprocess-path",
    "--reward-key",
    "--group-rm",
    # A YAML whose keys miles_validate_args setattr()s onto the namespace
    # (arguments.py:3194-3199), i.e. arbitrary overrides incl. use_tis/eps_clip:
    # it would bypass the spec entirely, so it is refused like any unmapped flag.
    "--custom-config-path",
    # Miles fork (michaellchung/miles yeto/ports, rl-algo-loss-variants route
    # B): mapped by yeto.rl.algos.loss_variants; listed here so they are
    # adapter-owned from import time. Upstream only once the pin carries them
    # (loss_variants.FORK_COMMITS).
    "--policy-loss-variant",
    "--sapo-tau-pos",
    "--sapo-tau-neg",
    "--gmpo-log-clip-low",
    "--gmpo-log-clip-high",
    # Miles fork SAO port (yeto-sao; rl-algo-critic-family 8.2/8.3): mapped by
    # yeto.rl.algos.sao.
    "--policy-objective",
    "--sao-dis-eps-low",
    "--sao-dis-eps-high",
]

MAPPINGS: dict[str, FlagMapping] = {row.flag: row for row in _builtin_rows()}
# Reviewed P0 rows (design D3) and rows added by extension modules; the
# table must always equal their union (tests/test_rl_algorithm_flags.py).
BUILTIN_FLAGS: frozenset[str] = frozenset(MAPPINGS)
EXTENSION_FLAGS: set[str] = set()
UNMAPPED_OBJECTIVE_FLAGS: set[str] = set(_UNMAPPED)


def objective_flags() -> frozenset[str]:
    """Every objective-changing Miles flag (mapped or not)."""

    load_extensions()
    return frozenset(MAPPINGS) | frozenset(UNMAPPED_OBJECTIVE_FLAGS)


def mapped_flags() -> frozenset[str]:
    load_extensions()
    return frozenset(MAPPINGS)


# Kept for ``config.ADAPTER_OWNED_FLAGS`` (import-time constant).
OBJECTIVE_FLAGS = frozenset(MAPPINGS) | frozenset(UNMAPPED_OBJECTIVE_FLAGS)


def register_flag(row: FlagMapping) -> None:
    """Add a mapping row (and move the flag out of the unmapped set)."""

    if row.flag in MAPPINGS:
        raise ValueError(f"{row.flag} is already mapped")
    MAPPINGS[row.flag] = row
    EXTENSION_FLAGS.add(row.flag)
    UNMAPPED_OBJECTIVE_FLAGS.discard(row.flag)


# --------------------------------------------------------------------------
# translation
# --------------------------------------------------------------------------


def algorithm_argv(spec: AlgorithmSpec) -> list[str]:
    """Non-default algorithm flags beyond the R0-positioned ones."""

    load_extensions()
    argv: list[str] = []
    for row in MAPPINGS.values():
        if not row.emitted_by_config:
            argv.extend(row.translate(spec))
    return argv


# --------------------------------------------------------------------------
# absorption (design D3 / D8 step 3)
# --------------------------------------------------------------------------


def _split(extra_argv: Sequence[str]) -> list[tuple[str, str | None, list[str]]]:
    """[(flag, value, raw tokens)] for mapped flags; others as (None...)."""

    tokens = list(extra_argv)
    out = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        flag, eq, inline = token.partition("=")
        row = MAPPINGS.get(flag) if token.startswith("--") else None
        if row is None:
            out.append(("", None, [token]))
            i += 1
            continue
        if row.switch:
            if eq:
                raise AlgorithmSpecError(f"{flag} takes no value, got {token!r}")
            out.append((flag, None, [token]))
            i += 1
        elif eq:
            out.append((flag, inline, [token]))
            i += 1
        else:
            if i + 1 >= len(tokens):
                raise AlgorithmSpecError(f"{flag} requires a value")
            out.append((flag, tokens[i + 1], [token, tokens[i + 1]]))
            i += 2
    return out


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return float(a) == float(b)
    return a == b


def absorb_extra_argv(
    spec: AlgorithmSpec, extra_argv: Sequence[str]
) -> tuple[AlgorithmSpec, tuple[str, ...], dict[str, Any]]:
    """Absorb mapped algorithm flags of ``extra_argv`` into ``spec``.

    Returns ``(spec', remaining_argv, absorbed)`` where ``absorbed`` maps each
    absorbed flag to its argv value (``True`` for switches). A flag whose value
    disagrees with a value the spec already sets (non-default), or with
    another absorbed flag, raises :class:`AlgorithmFlagConflict` naming both
    values. Objective-changing unmapped flags raise
    :class:`UnmappedAlgorithmFlag`.
    """

    load_extensions()
    split = _split(extra_argv)
    flags_present = {flag for flag, _, _ in split if flag}
    custom_tis = "--custom-tis-function-path" in flags_present
    if custom_tis and "--use-tis" not in flags_present:
        raise AlgorithmSpecError(
            "--custom-tis-function-path takes effect only with --use-tis; pass both "
            "(correction.method='custom') or neither"
        )
    remaining: list[str] = []
    absorbed: dict[str, Any] = {}
    assignments: dict[str, tuple[Any, str]] = {}  # path -> (value, source flag)
    for flag, raw, tokens in split:
        if not flag:
            name = tokens[0].split("=", 1)[0]
            if tokens[0].startswith("--") and name in UNMAPPED_OBJECTIVE_FLAGS:
                raise UnmappedAlgorithmFlag(
                    f"{name} changes the training objective but is not part of the "
                    "algorithm spec yet; it cannot be passed through extra argv"
                )
            remaining.extend(tokens)
            continue
        row = MAPPINGS[flag]
        try:
            value = row.parse(raw) if not row.switch else True
        except AlgorithmSpecError as exc:
            raise AlgorithmSpecError(f"{flag}: {exc}") from exc
        if flag in absorbed and not _values_equal(absorbed[flag], raw if raw is not None else True):
            raise AlgorithmFlagConflict(
                f"{flag} given twice in extra argv ({absorbed[flag]!r} vs {raw!r})"
            )
        absorbed[flag] = raw if raw is not None else True
        pairs = row.absorb(value)
        if flag == "--use-tis" and custom_tis:
            pairs = []  # --use-tis + --custom-tis-function-path = correction.method custom
        for path, new in pairs:
            current = spec.get_path(path)
            default = spec.effective_default_at(path)
            shown = new.path if isinstance(new, PluginRef) else new
            if current != default and current is not None and not _values_equal(current, new):
                shown_current = current.path if isinstance(current, PluginRef) else current
                raise AlgorithmFlagConflict(
                    f"extra argv {flag} sets {path}={shown!r} but the algorithm spec has "
                    f"{path}={shown_current!r}"
                )
            if path in assignments and not _values_equal(assignments[path][0], new):
                other_value, other_flag = assignments[path]
                other_shown = other_value.path if isinstance(other_value, PluginRef) else other_value
                if path == "kl.placement":
                    raise AlgorithmFlagConflict(
                        f"reward KL ({'--kl-coef'}) and loss KL (--use-kl-loss) are both set "
                        f"in extra argv ({other_flag}, {flag}); choose one kl.placement"
                    )
                raise AlgorithmFlagConflict(
                    f"extra argv {flag} sets {path}={shown!r} but {other_flag} sets "
                    f"{path}={other_shown!r}"
                )
            assignments[path] = (new, flag)
    if not assignments:
        return spec, tuple(remaining), absorbed
    payload = spec.structured_dict()
    for path, (value, _flag) in assignments.items():
        head, _, tail = path.partition(".")
        if isinstance(value, PluginRef):
            value = value.to_dict()
        if tail:
            payload.setdefault(head, {})[tail] = value
        else:
            payload[head] = value
    # Placement switched to loss: a reward KL coefficient already in the spec
    # would be carried over; the reward->loss switch is a conflict only if
    # the spec itself set a reward KL (caught above via kl.placement).
    return AlgorithmSpec.from_dict(payload), tuple(remaining), absorbed


# --------------------------------------------------------------------------
# dry run (docs/MILES_RL.md "Algorithm specs")
# --------------------------------------------------------------------------


def dry_run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Resolve, absorb, check and translate an algorithm without any engine.

    Mirrors the ports learner: ``--rl-algorithm-spec`` (else R0 default GRPO),
    absorption of ``--extra`` argv, the rejection matrix and the Miles
    adapter's capability declaration (with ``--rl-allow-unverified-mechanism``
    as a single-island allowance).
    """

    import argparse
    import shlex

    from ..algorithm import check_unverified_allowance, resolve_ports_algorithm
    from ..capabilities import CapabilityMismatch
    from .entry import miles_capabilities

    parser = argparse.ArgumentParser(prog="python3 -m yeto.rl.engine.miles_adapter.algorithm_flags")
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--rl-algorithm-spec", default=None)
    parser.add_argument("--extra", default="", help="extra Miles argv (one shell string)")
    parser.add_argument("--rl-allow-unverified-mechanism", action="append", default=None)
    args = parser.parse_args(argv)
    result: dict[str, Any] = {}
    try:
        base = resolve_ports_algorithm(args, rl_engine="ports")
        spec, remaining, absorbed = absorb_extra_argv(base, shlex.split(args.extra))
        # models a single-island G1 smoke without outer sync (D11)
        allow = check_unverified_allowance(
            args.rl_allow_unverified_mechanism or (), islands=1, outer_sync=False
        )
        result.update(
            schema=spec.schema,
            algorithm_spec=json.loads(spec.canonical_json()),
            algorithm_spec_sha256=spec.sha256(),
            absorbed_flags=absorbed,
            remaining_extra_argv=list(remaining),
            miles_argv=["--advantage-estimator", spec.advantage_estimator]
            + (["--kl-coef", str(spec.kl_coef)] if spec.kl_coef is not None else [])
            + algorithm_argv(spec),
            required_mechanisms=sorted(f"{d}:{n}" for d, n in spec.required_mechanisms()),
        )
        # Launch checks need the run configuration; the dry run reports them
        # with the ports defaults (CP 1) as warnings -- a real launch refuses them.
        from ..algorithm import launch_problems

        result["launch_warnings"] = launch_problems(spec, {
            "rollout_batch_size": None, "rollout_max_response_len": None,
            "context_parallel_size": 1, "multi_lora": False,
        })
        caps = miles_capabilities("sha256:" + "0" * 64, unverified_mechanisms=allow)
        caps.check(layout="lora", placement="colocated", execution_mode="colocated-serial",
                   algorithm=spec)
        result["verdict"] = "accepted"
    except (AlgorithmSpecError, CapabilityMismatch) as exc:
        result["verdict"] = "rejected"
        result["error"] = str(exc)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    result = dry_run(argv)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "accepted" else 1


if __name__ == "__main__":
    # Run the package module, not this ``__main__`` copy: extension modules
    # (``register_flag``) register rows into the package module's tables.
    from yeto.rl.engine.miles_adapter import algorithm_flags as _package_module

    raise SystemExit(_package_module.main())
