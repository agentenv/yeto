#!/usr/bin/env python3
"""Check an engine image against its version lock at container start (S25 N0.2).

Usage: ``assert_image_versions.py LOCK.json`` (the image entrypoint runs it
before the command). Exit 0 when every check passes, 1 otherwise; the report
goes to stderr, one line per check, and to ``--report`` as JSON when given.

The lock (``docker/<image>/versions.lock.json``) lists:

* ``cpu_arch``: must equal ``platform.machine()``;
* ``python``: ``major.minor`` prefix of the interpreter version (null: record);
* ``packages``: ``{dist: {"version", "import", "optional"}}``. ``version`` is
  compared with the public part of the installed version (``2.10.0+abc`` ->
  ``2.10.0``); null means "record only". ``import`` names a module the image
  must import. ``optional`` true turns a missing package or failed import into
  a report line instead of a failure (used where the probe on the real machine
  decides, e.g. TransformerEngine on PPU);
* ``commits``: ``{name: sha}`` compared with ``/opt/yeto/image/commits.json``
  written at build time (null: record);
* ``cann``: ``{"version", "version_files"}``: the toolkit version read from the
  first existing file (``version=`` / ``Version=`` line) must start with
  ``version``;
* ``driver``: ``{"min", "enforce", "version_file"}``: the host driver mounted
  into the container; ``min`` null means record only; ``enforce`` false
  reports a driver below ``min`` without failing (kept while a probe on the
  real host decides, see the NPU lock);
* ``required_files``: paths that must exist (patches, stubs).

Stdlib only: it runs before anything else in the image is trusted.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

COMMITS_FILE = Path("/opt/yeto/image/commits.json")
_VERSION_LINE = re.compile(r"^\s*(?:version|Version)\s*=\s*(\S+)", re.MULTILINE)


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, check: str, ok: bool, detail: str, *, optional: bool = False) -> None:
        status = "ok" if ok else ("not-enforced" if optional else "FAIL")
        self.rows.append({"check": check, "status": status, "detail": detail})
        print(f"[image-assert] {status:16} {check}: {detail}", file=sys.stderr)

    @property
    def failed(self) -> list[dict]:
        return [row for row in self.rows if row["status"] == "FAIL"]


def public_version(version: str) -> str:
    return version.split("+", 1)[0]


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def check_packages(packages: dict, report: Report) -> None:
    for dist, spec in packages.items():
        optional = bool(spec.get("optional"))
        want = spec.get("version")
        try:
            have = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            report.add(f"package {dist}", False, "not installed", optional=optional)
            continue
        if want is None:
            report.add(f"package {dist}", True, f"{have} (recorded, not pinned)")
        else:
            # A lock with a local version (torch 2.13.0+cu130) pins that build; one without
            # accepts any local build of the public version.
            ok = have == want if "+" in want else public_version(have) == want
            report.add(f"package {dist}", ok, f"{have} (lock {want})", optional=optional)
            if not ok:
                continue
        module = spec.get("import")
        if module:
            try:
                importlib.import_module(module)
            except Exception as exc:  # noqa: BLE001 - any import failure is the finding
                report.add(f"import {module}", False, f"{type(exc).__name__}: {exc}", optional=optional)
            else:
                report.add(f"import {module}", True, "imported")


def check_commits(commits: dict, report: Report, commits_file: Path) -> None:
    if not commits:
        return
    try:
        built = json.loads(commits_file.read_text())
    except (OSError, ValueError) as exc:
        report.add("commits", False, f"cannot read {commits_file}: {exc}")
        return
    for name, want in commits.items():
        have = built.get(name)
        if want is None:
            report.add(f"commit {name}", have is not None, f"{have} (recorded, not pinned)")
        else:
            report.add(f"commit {name}", have == want, f"{have} (lock {want})")


def _read_version(files: list[str]) -> tuple[str | None, str | None]:
    for name in files:
        path = Path(name)
        if path.is_file():
            match = _VERSION_LINE.search(path.read_text(errors="replace"))
            if match:
                return match.group(1), name
    return None, None


def check_cann(cann: dict | None, report: Report) -> None:
    if not cann:
        return
    have, source = _read_version(cann["version_files"])
    if have is None:
        report.add("cann", False, f"no version in {cann['version_files']}")
    else:
        report.add("cann", have.startswith(cann["version"]), f"{have} from {source} (lock {cann['version']})")


def check_driver(driver: dict | None, report: Report) -> None:
    if not driver:
        return
    have, source = _read_version([driver["version_file"]])
    want = driver.get("min")
    if have is None:
        report.add("driver", False, f"no version in {driver['version_file']} (is the host driver mounted?)")
    elif want is None:
        report.add("driver", True, f"{have} from {source} (recorded; no minimum in the lock)")
    else:
        report.add("driver", _version_tuple(have) >= _version_tuple(want), f"{have} (minimum {want})",
                   optional=not driver.get("enforce", True))


def run(lock: dict, *, commits_file: Path = COMMITS_FILE, machine: str | None = None) -> Report:
    report = Report()
    machine = machine or platform.machine()
    report.add("cpu_arch", machine == lock["cpu_arch"], f"{machine} (lock {lock['cpu_arch']})")
    want_py = lock.get("python")
    have_py = platform.python_version()
    report.add("python", want_py is None or have_py.startswith(want_py + ".") or have_py == want_py,
               f"{have_py} (lock {want_py})")
    check_packages(lock.get("packages", {}), report)
    check_commits(lock.get("commits", {}), report, commits_file)
    check_cann(lock.get("cann"), report)
    check_driver(lock.get("driver"), report)
    for name in lock.get("required_files", []):
        report.add(f"file {name}", Path(name).exists(), "present" if Path(name).exists() else "absent")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lock", type=Path)
    ap.add_argument("--report", type=Path, help="also write the rows as JSON here")
    ap.add_argument("--commits-file", type=Path, default=COMMITS_FILE)
    args = ap.parse_args(argv)
    lock = json.loads(args.lock.read_text())
    report = run(lock, commits_file=args.commits_file)
    if args.report:
        args.report.write_text(json.dumps({"image": lock.get("image"), "rows": report.rows}, indent=2))
    if report.failed:
        print(f"[image-assert] {len(report.failed)} check(s) failed; refusing to start "
              f"image {lock.get('image')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
