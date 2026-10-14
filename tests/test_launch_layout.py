"""Layout guard for the launcher split: ``yeto/launch`` modules depend one way.

``yeto.launcher`` composes the modules in ``yeto/launch``; none of them may
import ``yeto.launcher`` back, and their imports among themselves form no cycle.
"""

from __future__ import annotations

import ast
from pathlib import Path

LAUNCH_DIR = Path(__file__).resolve().parents[1] / "yeto" / "launch"


def _launch_imports(path: Path) -> set[str]:
    """Modules of ``yeto.launch`` (and ``yeto.launcher``) that ``path`` imports."""
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1:  # from .x import ... / from . import ...
                out.add(node.module or "__init__")
            elif node.level == 2 and node.module == "launcher":
                out.add("yeto.launcher")
            elif node.level == 0 and node.module == "yeto.launcher":
                out.add("yeto.launcher")
            elif node.level == 2 and node.module is None:
                out |= {f"yeto.{a.name}" for a in node.names if a.name == "launcher"}
        elif isinstance(node, ast.Import):
            out |= {a.name for a in node.names if a.name == "yeto.launcher"}
    return out


def _graph() -> dict[str, set[str]]:
    return {p.stem: _launch_imports(p) for p in LAUNCH_DIR.glob("*.py")}


def test_launch_modules_do_not_import_the_launcher():
    offenders = sorted(mod for mod, deps in _graph().items() if "yeto.launcher" in deps)
    assert offenders == []


def test_launch_module_imports_form_no_cycle():
    graph = _graph()
    state: dict[str, int] = {}

    def visit(mod: str, path: list[str]) -> None:
        state[mod] = 1
        for dep in sorted(graph.get(mod, set()) & set(graph)):
            assert state.get(dep) != 1, "import cycle: " + " -> ".join(path + [dep])
            if dep not in state:
                visit(dep, path + [dep])
        state[mod] = 2

    for mod in sorted(graph):
        if mod not in state:
            visit(mod, [mod])
