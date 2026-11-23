"""``yeto launch --breakers FILE``: a ``breakers:`` section applied to one launch.

The file holds a bare ``breakers`` object or a whole run config (its ``breakers`` key).
The section is resolved against the launch's own flags (the legacy aliases of the
``launch`` and ``trainer`` layers) and written back into the argv, so the launch reads
exactly the knobs it reads today: a section value becomes its flag, a disabled entry
becomes the flag's legacy off spelling (``--idle-warn-minutes 0``, ``--budget-usd off``,
``--no-cost-guard``). A flag on the command line that disagrees with the section is a
conflict and the launch is refused. The other layers are listed as not applied here.

``launch.budget`` inherits ``run.budget.usd`` when the section sets it and the launch
does not set its own budget (decision 3); rule D11 (``yeto launch --dry-run``) warns when a
launch budget is larger.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

APPLIED_LAYERS = ("launch", "trainer")
APPLIED_SURFACES = ("launch_flag", "trainer_flag")
RESOLVED_FILE = "breakers.resolved.json"


def load_section(path: str | Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """(breakers section, whole run config or None) of ``path``."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"--breakers {path}: expected a JSON object")
    if "breakers" in doc:
        section = doc["breakers"]
        if not isinstance(section, dict):
            raise ValueError(f"--breakers {path}: 'breakers' must be an object")
        return section, doc
    return doc, None


def apply(path: str | Path, argv: Sequence[str], *, log: Callable[[str], None] = print
          ) -> tuple[list[str], Any]:
    """(the launch argv with the section written back, the resolved breakers).

    ``argv`` is the ``yeto launch`` argument list (without the ``launch`` word). Raises
    ``ValueError`` (``BreakerConfigError``) on a bad section or a conflict."""
    from yeto import breakers

    section, run_config = load_section(path)
    sources = breakers.Sources(argv=list(argv), section=section, run_config=run_config)
    resolved = breakers.resolve(sources)
    out = breakers.materialize(resolved, sources, surfaces=APPLIED_SURFACES)
    for change in out.changes:
        log(f"[yeto] --breakers: {change}")
    skipped = sorted(layer for layer in section
                     if layer not in (*APPLIED_LAYERS, "switches", "deps"))
    if skipped:
        log(f"[yeto] --breakers: not applied in this process: {', '.join(skipped)}")
    return list(out.argv if out.argv is not None else argv), resolved


def print_deprecations(argv: Sequence[str], resolved: Any = None, *,
                       log: Callable[[str], None] = print) -> None:
    """Print once, at the launch CLI, every deprecated legacy knob the launch uses (decision 9).
    Never fails a launch: without ``--breakers`` the flags are resolved leniently."""
    try:
        if resolved is None:
            from yeto import breakers

            resolved = breakers.resolve(breakers.Sources(argv=list(argv)), layers=APPLIED_LAYERS, strict=False)
        lines = resolved.deprecations()
    except Exception:  # noqa: BLE001 - a note, not a check
        return
    for line in lines:
        log(f"[yeto] DEPRECATED: {line}")


def save_resolved(run: str, resolved: Any) -> Path | None:
    """``breakers.resolved.json`` in the run directory (for ``show`` and the dashboard)."""
    from .. import runs

    path = runs.run_dir(run) / RESOLVED_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(resolved.to_json(), indent=1, default=str) + "\n", encoding="utf-8")
    except OSError:
        return None
    return path
