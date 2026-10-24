#!/usr/bin/env python3
"""Install Harbor into an engine image without moving the engine's packages (S26 E0.4).

Harbor runs in the engine's own Python (yeto's HarborAgentLoop and the Miles
Harbor example import it in-process), so it shares one environment with
SGLang / Miles / vLLM / verl / Ray. The expected result of that install is
kept in ``versions.json`` next to this file:

* every package of the base image is held at its base version (uv
  constraint), except the ones listed under ``backends.<b>.changes``; Harbor's
  dependency bounds force those to move, each to one exact version;
* ``pins`` fixes Harbor-only packages whose choice decides the moves above.

Commands (run inside the image; stdlib only):

* ``constraints --backend B BEFORE``: the uv constraint file, to stdout;
* ``check --backend B BEFORE AFTER``: exit 1 when the install moved, removed
  or downgraded anything ``versions.json`` does not allow;
* ``lock --backend B AFTER --out LOCK``: the start lock that
  ``docker/image-assert/assert_image_versions.py`` checks at container start.

BEFORE / AFTER are ``uv pip freeze`` outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SPEC_FILE = Path(__file__).with_name("versions.json")
_PINNED = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)$")
_DIRECT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*@\s*(\S+)$")
# Module each asserted package is imported as, where it is light enough to
# import at container start (no CUDA context, no engine side effects).
START_IMPORTS = {
    "harbor": "harbor", "boat-sdk": "boat_sdk", "openai": "openai", "pyyaml": "yaml",
    "websockets": "websockets", "importlib-metadata": "importlib_metadata", "filelock": "filelock",
    "litellm": None,  # imports slowly and fetches a cost table; version only
}


def canonical(name: str) -> str:
    """PEP 503 name: ``PyYAML`` / ``importlib_metadata`` -> ``pyyaml`` / ``importlib-metadata``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_freeze(text: str) -> dict[str, str]:
    """``uv pip freeze`` -> ``{name: version}``; a direct (VCS, path) install maps to ``@ <url>``.

    Editable lines (``-e ...``) carry no name and are skipped: the engines
    install their own sources editable and Harbor's install leaves them alone
    (``check`` would not see such a move; the start lock's commits do).
    """
    found: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith(("#", "-")):
            continue
        if match := _PINNED.match(line):
            found[canonical(match.group(1))] = match.group(2)
        elif match := _DIRECT.match(line):
            found[canonical(match.group(1))] = "@ " + match.group(2)
    return found


def load_spec(backend: str, spec_file: Path = SPEC_FILE) -> tuple[dict, dict]:
    spec = json.loads(spec_file.read_text())
    if backend not in spec["backends"]:
        raise SystemExit(f"[harbor-layer] unknown backend {backend!r}; known: {sorted(spec['backends'])}")
    return spec, spec["backends"][backend]


def constraints(before: dict[str, str], spec: dict, backend_spec: dict) -> list[str]:
    """Hold every pinned base package, except the allowed changes (at their layer version) and the pins."""
    changes = {canonical(n): c for n, c in backend_spec["changes"].items()}
    pins = {canonical(n): p for n, p in spec["pins"].items()}
    lines = [f"{name}=={version}" for name, version in sorted(before.items())
             if not version.startswith("@") and name not in changes and name not in pins]
    lines += [f"{name}=={change['layer']}" for name, change in sorted(changes.items())]
    lines += [f"{name}=={pin['version']}" for name, pin in sorted(pins.items())]
    return lines


def check(before: dict[str, str], after: dict[str, str], spec: dict, backend_spec: dict) -> list[str]:
    """Every way ``after`` differs from what ``versions.json`` allows, one line each (empty: as expected)."""
    changes = {canonical(n): c for n, c in backend_spec["changes"].items()}
    problems: list[str] = []
    for name, change in sorted(changes.items()):
        if before.get(name) != change["base"]:
            problems.append(f"{name}: base image has {before.get(name)}, versions.json expects base "
                            f"{change['base']} (the base image moved; review the change and update versions.json)")
        if after.get(name) != change["layer"]:
            problems.append(f"{name}: installed {after.get(name)}, versions.json allows only {change['layer']}")
    for name, version in sorted(before.items()):
        if name in changes:
            continue
        if name not in after:
            problems.append(f"{name}: removed by the Harbor install (base {version})")
        elif after[name] != version:
            problems.append(f"{name}: moved {version} -> {after[name]} by the Harbor install, "
                            "not allowed by versions.json")
    for name, pin in sorted(spec["pins"].items()):
        if after.get(canonical(name)) != pin["version"]:
            problems.append(f"{canonical(name)}: installed {after.get(canonical(name))}, pinned {pin['version']}")
    harbor = after.get("harbor", "")
    if not harbor.endswith("@" + spec["harbor"]["commit"]):
        problems.append(f"harbor: installed {harbor or 'nothing'}, pinned commit {spec['harbor']['commit']}")
    if after.get("boat-sdk") != spec["boat_sdk"]:
        problems.append(f"boat-sdk: installed {after.get('boat-sdk')}, pinned {spec['boat_sdk']}")
    return problems


def start_lock(after: dict[str, str], spec: dict, backend: str, *, cpu_arch: str = "x86_64") -> dict:
    """The lock ``assert_image_versions.py`` checks at start: engine packages at their built version,
    the allowed changes and pins at their layer version, Harbor and boat-sdk importable."""
    backend_spec = spec["backends"][backend]
    packages: dict[str, dict] = {}
    for name in spec["assert_packages"]:
        version = after.get(canonical(name))
        if version is not None and not version.startswith("@"):
            packages[canonical(name)] = {"version": version}
    expected = {canonical(n): c["layer"] for n, c in backend_spec["changes"].items()}
    expected |= {canonical(n): p["version"] for n, p in spec["pins"].items()}
    expected["filelock"] = after.get("filelock")
    expected["harbor"] = spec["harbor"]["version"]
    expected["boat-sdk"] = spec["boat_sdk"]
    for name, version in expected.items():
        if version is None:
            continue
        row = {"version": version}
        if START_IMPORTS.get(name):
            row["import"] = START_IMPORTS[name]
        packages[name] = row
    return {
        "image": backend_spec["image"],
        "cpu_arch": cpu_arch,
        "device_family": "cuda",
        "python": None,
        "packages": packages,
        "commits": {},
        "required_files": [],
        "decisions": [f"{n}: {c['base']} -> {c['layer']} ({c['why']})" for n, c in backend_spec["changes"].items()]
                     + [f"{n}: pinned {p['version']} ({p['why']})" for n, p in spec["pins"].items()],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("constraints", "check", "lock"))
    ap.add_argument("--backend", required=True)
    ap.add_argument("freeze", type=Path, nargs="+", help="BEFORE (constraints, check) and/or AFTER (check, lock)")
    ap.add_argument("--out", type=Path, help="lock: where to write the start lock")
    ap.add_argument("--spec", type=Path, default=SPEC_FILE)
    args = ap.parse_args(argv)
    spec, backend_spec = load_spec(args.backend, args.spec)
    freezes = [parse_freeze(path.read_text()) for path in args.freeze]
    if args.command == "constraints":
        print("\n".join(constraints(freezes[0], spec, backend_spec)))
        return 0
    if args.command == "check":
        if len(freezes) != 2:
            ap.error("check takes BEFORE and AFTER")
        problems = check(freezes[0], freezes[1], spec, backend_spec)
        for line in problems:
            print(f"[harbor-layer] FAIL {line}", file=sys.stderr)
        if problems:
            print(f"[harbor-layer] {len(problems)} problem(s): the Harbor install changed the {args.backend} "
                  f"image beyond {args.spec.name}; refusing the image", file=sys.stderr)
            return 1
        print(f"[harbor-layer] ok: {args.backend} image changed only as {args.spec.name} allows", file=sys.stderr)
        return 0
    if args.out is None:
        ap.error("lock needs --out")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(start_lock(freezes[-1], spec, args.backend), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
