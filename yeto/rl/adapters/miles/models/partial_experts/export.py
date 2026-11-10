"""Filter an HF export down to the trainable experts.

Two Miles export paths exist and both are filtered by HF parameter name:

* **Bridge** (``HfWeightIteratorBridge``): conversion *tasks* carry one or
  more HF names (``task.mapping.hf_param``); :func:`filter_tasks` keeps a task
  when it has names and every name is accepted. Every EP rank must keep the
  same local expert offsets so the export collectives stay aligned -- the
  model runtime's predicate encodes that;
* **Direct** (``HfWeightIteratorDirect``, raw mode): the iterator yields
  ``(hf_name, weight, ...)`` items; :func:`select_named_weights` keeps the
  accepted ones.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

NamePredicate = Callable[[str], bool]


def mapping_hf_names(task: Any) -> tuple[str, ...]:
    """The HF parameter names a Bridge conversion task produces."""

    if task is None:
        return ()
    value = getattr(getattr(task, "mapping", None), "hf_param", None)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping) and all(isinstance(item, str) for item in value.values()):
        return tuple(value.values())
    return ()


def filter_tasks(tasks: Iterable[Any], accept: NamePredicate) -> list[Any]:
    """Tasks whose HF names are non-empty and all accepted."""

    selected = []
    for task in tasks:
        names = mapping_hf_names(task)
        if names and all(accept(name) for name in names):
            selected.append(task)
    return selected


def select_named_weights(items: Iterable[tuple], accept: NamePredicate) -> Iterator[tuple]:
    """``(hf_name, ...)`` items whose name is accepted."""

    return (item for item in items if accept(item[0]))
