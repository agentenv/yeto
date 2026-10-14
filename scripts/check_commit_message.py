#!/usr/bin/env python3
"""commit-msg check (standard library only): commit messages must not contain CJK text.

Rejects Han ideographs, kana, hangul, CJK punctuation and full-width forms in the
subject and in the body. Lines starting with '#' (git comments) and everything after
a scissors line are ignored. Symbols such as '->' arrows and accented Latin letters
pass.

Usage (git passes the message file path as the only argument):
  python3 scripts/check_commit_message.py .git/COMMIT_EDITMSG
Exit code 1 when the message contains a rejected character.
"""
from __future__ import annotations

import re
import sys

SCISSORS = "# ------------------------ >8 ------------------------"

# Han (incl. extension A and compatibility), CJK symbols and punctuation, kana,
# hangul, and half-width / full-width forms.
CJK_PATTERN = re.compile(
    "[　-〿぀-ヿ㐀-䶿一-鿿"
    "가-힯豈-﫿＀-￯]"
)


def find_violations(message: str) -> list[tuple[int, str]]:
    """Return (1-based line number, line) for each message line with CJK text."""
    violations = []
    for number, line in enumerate(message.splitlines(), start=1):
        if line.startswith(SCISSORS):
            break
        if line.startswith("#"):
            continue
        if CJK_PATTERN.search(line):
            violations.append((number, line))
    return violations


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as handle:
        violations = find_violations(handle.read())
    for number, line in violations:
        print(f"commit message line {number} contains CJK text: {line}", file=sys.stderr)
    if violations:
        print("Commit messages must be English only.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
