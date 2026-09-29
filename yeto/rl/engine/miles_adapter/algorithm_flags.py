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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..algorithm import AlgorithmSpec, AlgorithmSpecError, PluginRef, load_extensions

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
        if c.method != "opsm":
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


def _builtin_rows() -> list[FlagMapping]:
    return [
        FlagMapping("--advantage-estimator", "advantage.estimator", False, _str,
                    lambda v: [("advantage.estimator", v)], lambda spec: [],
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
    "--gamma",
    "--lambd",
    "--value-clip",
    "--num-critic-only-steps",
    "--critic-load",
    "--critic-lr",
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
]

MAPPINGS: dict[str, FlagMapping] = {row.flag: row for row in _builtin_rows()}
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
    remaining: list[str] = []
    absorbed: dict[str, Any] = {}
    assignments: dict[str, tuple[Any, str]] = {}  # path -> (value, source flag)
    for flag, raw, tokens in _split(extra_argv):
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
        for path, new in row.absorb(value):
            current = spec.get_path(path)
            default = AlgorithmSpec.default_at(path)
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
            payload[head][tail] = value
        else:
            payload[head] = value
    # Placement switched to loss: a reward KL coefficient already in the spec
    # would be carried over; the reward->loss switch is a conflict only if
    # the spec itself set a reward KL (caught above via kl.placement).
    return AlgorithmSpec.from_dict(payload), tuple(remaining), absorbed
