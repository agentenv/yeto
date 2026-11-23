"""Golden lock of every breaker's effective value (decision: no behavior change for existing configs).

``golden.json`` holds, per config in ``cases.py``, the value each breaker has today (``capture.py``
reads it through the code that applies it). Two checks keep it:

1. today's code paths still produce exactly those values (a migration that changes a reader,
   a default or an alias fails here);
2. ``yeto.breakers.resolve`` gives the same values for every registered entry (skipped while
   the registry package is not on main; entries not registered yet are skipped one by one).

Run: ``PYTHONPATH=. python -m pytest tests/golden/breakers -q``. Regenerate only for a new case
or a key rename: ``PYTHONPATH=. python -m tests.golden.breakers.capture --write``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests.golden.breakers.capture import EXAMPLES, GOLDEN, REPO, capture, load_case, monitor_values
from tests.golden.breakers.cases import CASES

GOLDEN_DOC = json.loads(GOLDEN.read_text())
BY_NAME = {c.name: c for c in CASES}

# examples/k31 files that hold no breaker knob (algorithm specs: the policy age rule reads their staleness;
# the inventory and the shell launchers carry the same flags as the run configs)
NO_BREAKERS = {"algorithm-spec.async-age12-tis.json", "algorithm-spec.async-age12.json",
               "algorithm-spec.async-age2-tis.json", "algorithm-spec.async-age2.json",
               "inventory.b300-6x8-baremetal.yaml", "launch-convertible-modal.sh",
               "launch-trainer-modal-b300.sh", "sampler-node-nc-mi355x.sh", "README.md"}

# golden keys that are derived views, not registry params: checked by (1) only
DERIVED = {
    "sampler.drain.effective_cut_s",  # min(timeout_s, reload_max_defer_s): the drain cut rule prints it
    "run.sandbox_containers.limit_n",  # max_run_n raised to max_per_sampler_n x billed nodes, at run time
    "monitor.dashboard.budget_usd",  # the dashboard's --budget, passed from run.budget.usd by run up
    "pool.node_cap.effective_max_nodes_n",  # max_nodes_n, or 2 x the pool target when null
}
# golden keys a layer chose not to declare as a registry param, with the reason (append here;
# never drop a key from capture.py: it still locks today's reader)
NOT_IN_REGISTRY: dict[str, str] = {}


def _jsonable(values: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(values, sort_keys=True))


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def _diff(got: dict[str, Any], want: dict[str, Any]) -> list[str]:
    lines = [f"  {k}: golden {want[k]!r}, now missing" for k in sorted(set(want) - set(got))]
    lines += [f"  {k}: not in golden, now {got[k]!r}" for k in sorted(set(got) - set(want))]
    lines += [f"  {k}: golden {want[k]!r}, now {got[k]!r}" for k in sorted(set(want) & set(got))
              if not _same(got[k], want[k])]
    return lines


def test_every_example_config_has_a_case():
    covered = set()
    for case in CASES:
        for ref in (case.run, case.pool, case.spec):
            if isinstance(ref, str):
                covered.add((REPO / ref).name)
    k31 = {p.name for p in (EXAMPLES / "k31").iterdir() if p.is_file()}
    # a run config's pool and spec are covered through it as well, but each also has its own case
    missing = sorted(k31 - covered - NO_BREAKERS)
    assert not missing, f"examples/k31 files without a golden case (add them to cases.py): {missing}"
    assert set(GOLDEN_DOC["cases"]) == set(BY_NAME), "golden.json and cases.py list different cases: regenerate"


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_todays_code_paths_match_golden(name):
    problems = _diff(_jsonable(capture(BY_NAME[name])), GOLDEN_DOC["cases"][name])
    assert not problems, (f"{name}: effective breaker values changed (a config without breakers: must resolve "
                          "exactly as before):\n" + "\n".join(problems))


def test_monitor_constants_match_golden():
    problems = _diff(_jsonable(monitor_values()), GOLDEN_DOC["constants"])
    assert not problems, "monitor thresholds changed:\n" + "\n".join(problems)


# ----------------------------------------------------------------------------- registry
def _registry():
    breakers = pytest.importorskip("yeto.breakers", reason="registry package not on this branch yet")
    if not hasattr(breakers, "resolve") or not hasattr(breakers, "Sources"):
        pytest.skip("yeto.breakers has no resolve() yet")
    return breakers


def _sources(breakers: Any, name: str, tmp: Path) -> Any:
    loaded = load_case(BY_NAME[name], tmp)
    surfaces: dict[str, Any] = {"run_config": loaded.run_config, "argv": loaded.argv, "spec": loaded.spec,
                                "env": {**loaded.trainer_env, **loaded.sampler_env}}
    if loaded.pool is not None:
        surfaces["pool_config"] = loaded.pool
        surfaces["pool_elastic"] = loaded.pool.get("elastic") or {}
        surfaces["pool_dispense"] = (loaded.pool.get("sampler") or {}).get("dispense") or {}
    known = getattr(breakers.Sources, "__dataclass_fields__", surfaces)
    return breakers.Sources(**{k: v for k, v in surfaces.items() if k in known})


def _split(key: str) -> tuple[str, str]:
    entry, _, param = key.rpartition(".")
    return entry, param


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_registry_resolves_golden(name):
    breakers = _registry()
    registry = breakers.default_registry()
    with tempfile.TemporaryDirectory() as tmp:
        rb = breakers.resolve(_sources(breakers, name, Path(tmp)))
    want = GOLDEN_DOC["cases"][name]
    disabled = {_split(k)[0] for k, v in want.items() if k.endswith(".enabled") and v is False}
    problems, checked = [], 0
    for key, value in sorted(want.items()):
        if key in DERIVED or key in NOT_IN_REGISTRY:
            continue
        if key.startswith("switch."):
            group = key.split(".", 1)[1]
            if group in rb.switches():
                checked += 1
                if not _same(rb.switches()[group], value):
                    problems.append(f"  {key}: golden {value!r}, registry {rb.switches()[group]!r}")
            continue
        entry, param = _split(key)
        if entry not in registry:
            continue  # its layer has not landed yet
        if entry in disabled and param != "enabled":
            continue
        resolved = rb.entry(entry)
        checked += 1
        if param == "enabled":
            got = rb.enabled(entry)
        elif param == "action":
            got = resolved.action
        elif param == "implicit":
            got = resolved.spec.implicit and resolved.source.location == "default (implicit)"
        elif param not in {p.name for p in resolved.spec.params}:
            problems.append(f"  {key}: entry {entry} declares no param {param!r} (declare it, rename the golden "
                            "key in capture.py keeping its value, or list it in NOT_IN_REGISTRY with a reason)")
            continue
        else:
            got = rb.get(key)
        if not _same(got, value):
            problems.append(f"  {key}: golden {value!r}, registry {got!r} (source {resolved.sources.get(param)})")
    assert not problems, f"{name}: the registry resolves differently from today:\n" + "\n".join(problems)
    if checked == 0:
        pytest.skip("no golden key of this case is registered yet")
