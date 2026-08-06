#!/usr/bin/env python3
"""Prepare target-specific CyberGym coverage runners."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


PARSER_BLOCK = r'''
coverage_file=$(printenv CYBERGYM_COVERAGE_FILE 2>/dev/null || true)
if [ -n "$coverage_file" ]; then
  /usr/local/bin/python3 - "$output" "$coverage_file" <<'PY_COVERAGE'
import json
import re
import sys

covered = []
for line in open(sys.argv[1], errors="replace"):
    match = re.match(r"COVERED_FUNC: hits: (\d+) edges: \d+/\d+ ([^ ]+)", line)
    if match and int(match.group(1)) > 0:
        covered.append(match.group(2))
    match = re.match(r"COVERED_FUNC: in (.+)$", line)
    if match:
        covered.append(match.group(1))
json.dump({"functions": sorted(set(covered)), "blocks": []}, open(sys.argv[2], "w"))
PY_COVERAGE
fi
'''


ARVO_COVERAGE_BLOCK = r'''
coverage_file=$(printenv CYBERGYM_COVERAGE_FILE 2>/dev/null || true)
if [ -n "$coverage_file" ] && [ -s "$map" ]; then
  /usr/local/bin/python3 - "$map" /out/coverage-map.json "$coverage_file" <<'PY_COVERAGE'
import json
import sys
from pathlib import Path

coverage_map = json.loads(Path(sys.argv[2]).read_text())
functions = set()
for line in Path(sys.argv[1]).read_text(errors="replace").splitlines():
    index, _, _hits = line.partition(":")
    name = coverage_map.get(str(int(index)))
    if not name:
        continue
    functions.add(name)
    base = name.split("(", 1)[0].strip()
    functions.add(base)
    if "::" in base:
        functions.add(base.rsplit("::", 1)[-1])
Path(sys.argv[3]).write_text(
    json.dumps({"functions": sorted(functions), "blocks": []})
)
PY_COVERAGE
fi
'''


def remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def link(source: str, destination: Path) -> None:
    remove(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source, destination)


def _arvo_target(vulnerable_runner: Path) -> str:
    text = vulnerable_runner.read_text(errors="replace")
    match = re.search(r"/out/([^\s]+)\s+/tmp/poc", text)
    if match is None:
        raise ValueError(f"cannot find target binary in {vulnerable_runner}")
    return match.group(1)


def _arvo_symbols(binary: Path) -> str:
    return subprocess.run(
        ["nm", "-an", binary],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout


def _is_aflpp(binary: Path) -> bool:
    return re.search(r"\s__afl_persistent_loop$", _arvo_symbols(binary), re.MULTILINE) is not None


def _arvo_coverage_map(binary: Path) -> dict[str, str]:
    symbols = _arvo_symbols(binary)
    guard_symbols: dict[str, int] = {}
    for line in symbols.splitlines():
        match = re.match(r"^([0-9a-fA-F]+)\s+[A-Za-z]\s+(__(?:start|stop)___sancov_guards)$", line)
        if match:
            guard_symbols[match.group(2)] = int(match.group(1), 16)
    try:
        guard_start = guard_symbols["__start___sancov_guards"]
        guard_stop = guard_symbols["__stop___sancov_guards"]
    except KeyError:
        return {}

    disassembly = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", "-C", binary],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    coverage: dict[str, str] = {}
    function = ""
    for line in disassembly.splitlines():
        function_match = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
        if function_match:
            function = function_match.group(2)
            continue
        if not function or "movslq" not in line or "#" not in line:
            continue
        address_match = re.search(r"#\s*([0-9a-f]+)", line)
        if address_match is None:
            continue
        guard_address = int(address_match.group(1), 16)
        if not guard_start <= guard_address < guard_stop:
            continue
        offset = guard_address - guard_start
        if offset % 4:
            continue
        # AFL++ reserves map slots 0..5 before assigning sanitizer guards.
        guard_id = 6 + offset // 4
        coverage[str(guard_id)] = function
    return coverage


def arvo_wrapper(vulnerable_runner: Path) -> str:
    target = _arvo_target(vulnerable_runner)
    text = vulnerable_runner.read_text(errors="replace")
    exports = "\n".join(line for line in text.splitlines() if line.startswith("export "))
    return (
        "#!/bin/bash\n"
        "set +e\n"
        f"{exports}\n"
        "export AFL_INST_RATIO=100\n"
        "output=$(mktemp)\n"
        "map=$(mktemp)\n"
        "/src/aflplusplus/afl-showmap -q -r -t 5000 -o \"$map\" -- "
        f"env LD_LIBRARY_PATH=/out-libs /out/{target} /tmp/poc >\"$output\" 2>&1\n"
        "status=$?\n"
        f"{ARVO_COVERAGE_BLOCK}"
        "cat \"$output\"\n"
        "rm -f \"$output\" \"$map\"\n"
        "exit \"$status\"\n"
    )


def _legacy_arvo_wrapper(vulnerable_runner: Path) -> str:
    target = _arvo_target(vulnerable_runner)
    text = vulnerable_runner.read_text(errors="replace")
    exports = "\n".join(line for line in text.splitlines() if line.startswith("export "))
    return (
        "#!/bin/bash\n"
        "set +e\n"
        f"{exports}\n"
        "output=$(mktemp)\n"
        f"/out/{target} -print_coverage=1 -runs=1 -timeout=5 /tmp/poc >\"$output\" 2>&1\n"
        "status=$?\n"
        f"{PARSER_BLOCK}"
        "rm -f \"$output\"\n"
        "exit \"$status\"\n"
    )


def oss_fuzz_wrapper(target: str) -> str:
    return (
        "#!/bin/bash\n"
        "set +e\n"
        "testcase=\"${!#}\"\n"
        "output=$(mktemp)\n"
        f"/out/{target}.real -print_coverage=1 -runs=1 -timeout=5 \"$testcase\" >\"$output\" 2>&1\n"
        "status=$?\n"
        f"{PARSER_BLOCK}"
        "rm -f \"$output\"\n"
        "exit \"$status\"\n"
    )


def _link_output_files(vulnerable_out: Path, coverage_out: Path) -> None:
    remove(coverage_out)
    coverage_out.mkdir(parents=True, exist_ok=True)
    for source in sorted(vulnerable_out.iterdir()):
        link(f"../../vul/out/{source.name}", coverage_out / source.name)


def prepare(root: Path, task_ids: list[str], runner_image: str) -> dict[str, object]:
    errors: list[str] = []
    counts = {"arvo": 0, "oss-fuzz": 0}
    for task_id in task_ids:
        subset, identifier = task_id.split(":", 1)
        base = root / subset / identifier
        vulnerable = base / "vul"
        coverage = base / "vul-cov"
        try:
            if subset == "arvo":
                target = _arvo_target(vulnerable / "arvo")
                target_binary = vulnerable / "out" / target
                link("../vul/libs", coverage / "libs")
                target_script = coverage / "arvo"
                if _is_aflpp(target_binary):
                    _link_output_files(vulnerable / "out", coverage / "out")
                    coverage_map = _arvo_coverage_map(target_binary)
                    (coverage / "out" / "coverage-map.json").write_text(
                        json.dumps(coverage_map, sort_keys=True) + "\n"
                    )
                    (coverage / "runner").write_text(runner_image + "\n")
                    target_script.write_text(arvo_wrapper(vulnerable / "arvo"))
                else:
                    link("../vul/out", coverage / "out")
                    runner = vulnerable / "runner"
                    if runner.is_file():
                        (coverage / "runner").write_text(runner.read_text())
                    else:
                        remove(coverage / "runner")
                    target_script.write_text(_legacy_arvo_wrapper(vulnerable / "arvo"))
                target_script.chmod(0o755)
            elif subset == "oss-fuzz":
                metadata = json.loads((vulnerable / "metadata.json").read_text())
                target_name = str(metadata["fuzz_target"])
                coverage.mkdir(parents=True, exist_ok=True)
                link("../vul/metadata.json", coverage / "metadata.json")
                remove(coverage / "out")
                (coverage / "out").mkdir()
                for source in sorted((vulnerable / "out").iterdir()):
                    if source.name == target_name:
                        os.symlink(
                            f"../../vul/out/{target_name}",
                            coverage / "out" / f"{target_name}.real",
                        )
                        wrapper = coverage / "out" / target_name
                        wrapper.write_text(oss_fuzz_wrapper(target_name))
                        wrapper.chmod(0o755)
                    else:
                        os.symlink(
                            f"../../vul/out/{source.name}",
                            coverage / "out" / source.name,
                        )
                runner = vulnerable / "runner"
                if runner.is_file():
                    (coverage / "runner").write_text(runner.read_text())
                else:
                    remove(coverage / "runner")
            else:
                raise ValueError(f"unsupported task subset {subset}")
            counts[subset] += 1
        except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
            errors.append(f"{task_id}: {exc}")
    return {"requested": len(task_ids), "counts": counts, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--runner-image", required=True)
    args = parser.parse_args()
    task_ids = json.loads(args.tasks.read_text())
    result = prepare(args.root, task_ids, args.runner_image)
    print(json.dumps(result, sort_keys=True))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
