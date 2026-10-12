"""Run logs never enter the repository (rl-code-quality-cleanup, spec
"运行日志不入库").

Evidence logs live outside the repository under
/home/michael/work/infra-drafts/evidence/<change>/ ; each old directory keeps
an EVIDENCE.md pointer. Test fixtures under tests/ and data/ are inputs, not
run logs, and are out of scope.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LOG_PATTERN = re.compile(
    r"(^|/)run-logs/(?!EVIDENCE\.md$)|\.log$|\.stderr$|\.stdout$|metrics[^/]*\.json$|\.jsonl$|\.out$|\.err$")
FIXTURE_ROOTS = ("tests/", "data/")


def _git(*args: str) -> str:
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


def test_no_tracked_run_logs():
    tracked = _git("ls-files").splitlines()
    hits = [f for f in tracked if LOG_PATTERN.search(f) and not f.startswith(FIXTURE_ROOTS)]
    assert hits == [], f"{len(hits)} tracked run-log files: {hits[:20]}"


@pytest.mark.parametrize("name", ["openspec/changes/x/run-logs/a.log",
                                  "openspec/changes/x/run-logs/b.txt",
                                  "openspec/changes/x/evidence/c/events.jsonl",
                                  "openspec/changes/x/evidence/c/run.stderr",
                                  "openspec/changes/x/evidence/c/train_metrics.json"])
def test_gitignore_blocks_new_logs(name):
    out = _git("check-ignore", "--no-index", "-v", name)
    assert out.strip() and not out.split("\t")[0].rsplit(":", 1)[-1].startswith("!"), out


def test_gitignore_has_no_log_whitelist():
    for line in (REPO / ".gitignore").read_text().splitlines():
        if line.startswith("!") and line != "!**/run-logs/EVIDENCE.md":
            assert not LOG_PATTERN.search(line[1:]), line
