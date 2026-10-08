"""Miles wrappers for neutral rewards and group filters (decoupling 3.2, 3.3, 3.10).

Translation only: Miles ``Sample`` -> :class:`Trajectory`, neutral
:class:`RewardResult` -> return value + ``sample.metadata`` +
``Sample.Status.ABORTED``, :class:`FilterDecision` -> ``DynamicFilterOutput``.

A registered neutral reward is reachable by Miles as
``--custom-rm-path yeto.rl.engine.miles_adapter.rewards.<name>`` (module-level
``__getattr__``; Miles' ``load_function`` splits the path at the last dot).
:func:`miles_custom_rm_path` validates a reference before launch and returns
that path; a ``module:function`` reference is encoded so it survives the split.
"""

from __future__ import annotations

from typing import Any, Callable

from yeto.rl.rewards.registry import RewardRegistryError, load_reward
from yeto.rl.rewards.types import FilterDecision, RewardResult, Trajectory

_MODULE = __name__
_REF_PREFIX = "ref__"


def _status_text(status: Any) -> str | None:
    if status is None:
        return None
    value = getattr(status, "value", status)
    if isinstance(value, str):
        return value.lower()
    name = getattr(status, "name", None)
    return str(name or status).lower()


def trajectory_from_sample(sample: Any) -> Trajectory:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}  # attached to the sample only if the reward writes into it
    return Trajectory(
        group_index=getattr(sample, "group_index", None),
        index=getattr(sample, "index", None),
        prompt=getattr(sample, "prompt", None),
        response=getattr(sample, "response", None) or "",
        label=getattr(sample, "label", None),
        tokens=getattr(sample, "tokens", None),
        rollout_logprobs=getattr(sample, "rollout_log_probs", None),
        loss_mask=getattr(sample, "loss_mask", None),
        status=_status_text(getattr(sample, "status", None)),
        metadata=metadata,
        reward=getattr(sample, "reward", None),
    )


def _aborted_status(sample: Any) -> Any:
    try:
        from miles.utils.types import Sample
    except ImportError:  # CPU tests without the Miles package
        status_type = getattr(type(sample), "Status", None)
        return getattr(status_type, "ABORTED", "ABORTED")
    return Sample.Status.ABORTED


def apply_reward_result(sample: Any, result: RewardResult) -> float:
    if not isinstance(result, RewardResult):
        raise TypeError(f"neutral reward must return RewardResult, got {type(result).__name__}")
    if result.metadata:
        if not isinstance(getattr(sample, "metadata", None), dict):
            sample.metadata = {}
        sample.metadata.update(result.metadata)
    if result.aborted:
        sample.status = _aborted_status(sample)
    return result.value


def _call(fn, sample: Any) -> float:
    trajectory = trajectory_from_sample(sample)
    result = fn(trajectory)
    if trajectory.metadata and trajectory.metadata is not getattr(sample, "metadata", None):
        sample.metadata = trajectory.metadata
    return apply_reward_result(sample, result)


def miles_reward(fn: Callable[[Trajectory], RewardResult]):
    """Miles ``--custom-rm-path`` callable for neutral ``fn`` (single sample or list)."""

    async def reward_func(args: Any, samples: Any, **_kwargs: Any):
        if isinstance(samples, list):
            return [_call(fn, s) for s in samples]
        return _call(fn, samples)

    reward_func.__name__ = getattr(fn, "__name__", "reward_func")
    reward_func.__qualname__ = reward_func.__name__
    reward_func.__wrapped_neutral__ = fn
    return reward_func


def filter_output(decision: FilterDecision):
    try:
        from miles.rollout.filter_hub.base_types import DynamicFilterOutput
    except ImportError:  # CPU tests without the Miles package
        from types import SimpleNamespace as DynamicFilterOutput  # type: ignore[assignment]
    return DynamicFilterOutput(keep=decision.keep, reason=decision.reason)


def _encode(ref: str) -> str:
    return _REF_PREFIX + ref.replace(".", "__dot__").replace(":", "__colon__")


def _decode(attr: str) -> str:
    return attr[len(_REF_PREFIX):].replace("__colon__", ":").replace("__dot__", ".")


def miles_custom_rm_path(ref: str) -> str:
    """Validate ``ref`` (registered name or ``module:function``) and return its Miles path.

    Raises :class:`RewardRegistryError` listing the registered names when the
    name is unknown -- call this before launching.
    """

    load_reward(ref)
    attr = ref if ":" not in ref else _encode(ref)
    return f"{_MODULE}.{attr}"


def __getattr__(attr: str):
    if attr.startswith("__"):
        raise AttributeError(attr)
    ref = _decode(attr) if attr.startswith(_REF_PREFIX) else attr
    try:
        plugin = load_reward(ref)
    except RewardRegistryError as exc:
        raise AttributeError(str(exc)) from exc
    return miles_reward(plugin.fn)
