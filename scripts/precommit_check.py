#!/usr/bin/env python3
"""Pre-commit check (standard library only).

Checks staged files (or the paths given on the command line):
  1. size: files larger than 500 KB are rejected unless listed in
     .precommit-allow-large (one path or fnmatch glob per line, '#' comments);
  2. secrets: 15 simple regexes (AWS, GitHub, HF, W&B, Modal, OpenAI/Anthropic,
     private keys, island HMAC key, URL credentials, bearer, Nebius, Verda, ...);
  3. personal e-mail addresses (noreply / example domains are allowed);
  4. private-network IPv4 addresses (10/8, 172.16/12, 192.168/16) outside test
     fake data (tests/test_*.py, tests/fixtures/, tests/golden/); use a
     placeholder or a documentation address (192.0.2.0/24) instead;
  5. per-user home paths (/home/<user>/, /Users/<user>/); use ~, $HOME or an
     environment variable instead;
  6. cloud tenant IDs (tenant-<20+ lowercase alphanumerics>);
  7. personal container registries: ghcr.io/<owner> where <owner> is neither
     an organization in REGISTRY_ORGS nor a placeholder (x, o, <owner>,
     $VAR, ...);
  8. CJK characters: the repository is English-only.

Warnings (printed, never fail the check):
  - phase and defect tags such as S29, s30_, G1, E2, P0-b, D17 (S3 is not a
    tag and is ignored).

Usage:
  python scripts/precommit_check.py            # check staged files
  python scripts/precommit_check.py FILE...    # check given files
Exit code 1 when any problem is found.
"""
from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys

MAX_BYTES = 500 * 1024
ALLOW_FILE = ".precommit-allow-large"

SECRET_PATTERNS = {
    "aws_akid": rb"\b(AKIA|ASIA)[0-9A-Z]{16}\b",
    "aws_secret": rb"(?i)aws_secret_access_key[\"'\s:=]+([A-Za-z0-9/+]{40})",
    "github": rb"\b(ghp_|gho_|ghs_|ghu_|github_pat_)[A-Za-z0-9_]{20,}",
    "hf": rb"\bhf_[A-Za-z0-9]{30,}",
    "wandb": rb"(?i)WANDB_API_KEY[\"'\s:=]+([A-Za-z0-9]{20,})",
    "modal": rb"\b(ak|as)-[A-Za-z0-9]{20,}",
    "modal_env": rb"MODAL_TOKEN_(ID|SECRET)[\"'\s:=]+([A-Za-z0-9-]{10,})",
    "sk": rb"\bsk-(ant-)?[A-Za-z0-9_-]{20,}",
    "privkey": rb"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    # a call such as secrets.token_hex(16) or self._token() is code, not a literal secret
    "hmac": rb"YETO_ISLAND_HMAC_KEY[\"'\s:=]+([^\s\"'$()]{8,})(?![^\s\"'$()]|\()",
    "urltok": rb"https?://[^/\s:@]+:[^/\s@]{8,}@",
    "bearer": rb"Authorization:\s*Bearer\s+([A-Za-z0-9._-]{16,})",
    "nebius": rb"(?i)nebius[^\n]{0,30}(token|secret|key)[\"'\s:=]+([A-Za-z0-9._-]{16,})",
    "verda": rb"(?i)verda[^\n]{0,30}(secret|key|token)[\"'\s:=]+([A-Za-z0-9._-]{16,})(?![A-Za-z0-9._(-])",
    "hex40kv": rb"(?i)(api_?key|secret|token)[\"'\s:=]+([0-9a-f]{40})\b",
}
SECRET_RE = {k: re.compile(v) for k, v in SECRET_PATTERNS.items()}
EMAIL_RE = re.compile(rb"[A-Za-z0-9._%+-]+@(gmail|googlemail|outlook|hotmail|yahoo|qq|163|126|icloud|proton(mail)?)\.(com|me|cn)\b", re.I)
PRIVATE_IP_RE = re.compile(
    rb"(?<![\d.])(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}(?!\.?\d)")
# Lock files are generated; their four-part package versions (10.3.9.90) look like IPs.
IP_FAKE_DATA_GLOBS = ("tests/test_*.py", "tests/fixtures/*", "tests/golden/*", "uv.lock")
HOME_PATH_RE = re.compile(rb"/(?:home|Users)/([A-Za-z0-9_][A-Za-z0-9_.-]*)(?:/|\b)")
# Generic account names that do not identify a person (cloud images, CI, containers).
# The project's own service accounts (yeto, sky) are listed exactly: a prefix match
# would also let a personal name such as /home/skyler through.
GENERIC_HOMES = {b"ubuntu", b"user", b"runner", b"ec2-user", b"admin", b"root", b"Shared", b"yeto", b"sky"}
TENANT_RE = re.compile(rb"tenant-[a-z0-9]{20,}")
# ghcr.io/<owner>: organizations may appear; any other concrete name is a
# personal namespace. Placeholders (one letter, <owner>, $VAR, ${VAR}, ...)
# are fine.
REGISTRY_ORGS = {b"agentenv"}
GHCR_RE = re.compile(rb"ghcr\.io/([A-Za-z0-9][A-Za-z0-9_.-]*)")
GHCR_PLACEHOLDERS = {b"owner", b"org", b"example", b"your-org", b"your-user", b"user"}
# The published engine images still live in a personal namespace until they move
# to an organization registry. Only the files that pin those images, and the
# tests and fixtures that check the pins, may name it.
REGISTRY_PIN_FILES = (
    "yeto/rl/adapters/miles/pins.py",
    "yeto/rl/adapters/verl/image.py",
    "tests/test_rl_backend_registry.py",
    "tests/fixtures/*",
)
# CJK ideographs, CJK symbols/punctuation and full-width forms.
CJK_RE = re.compile("[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")
# Phase tags (S25, s29next, G1, E2b, P0-b) and defect numbers (D1-D40). S3 is
# Amazon S3, never a phase tag; lowercase s0..s9 are ordinary variable names.
PHASE_TAG_RE = re.compile(
    rb"(?<![A-Za-z0-9])(?:(?:[Ss][1-9][0-9]|S[0-24-9])(?:next)?|[GEP][0-9][a-z]?(?:-[a-z])?|D(?:[1-9]|[1-3][0-9]|40))(?![A-Za-z0-9])")
# The checker and its tests necessarily contain samples of every pattern.
SELF = {os.path.normpath("scripts/precommit_check.py"), os.path.normpath("tests/test_precommit_check.py")}


def _personal_home(data: bytes) -> bool:
    for m in HOME_PATH_RE.finditer(data):
        name = m.group(1)
        if name not in GENERIC_HOMES:
            return True
    return False


def _personal_registry(data: bytes) -> list[str]:
    owners = []
    for m in GHCR_RE.finditer(data):
        owner = m.group(1).rstrip(b".")
        if len(owner) <= 1 or owner in REGISTRY_ORGS or owner.lower() in GHCR_PLACEHOLDERS:
            continue
        owners.append(owner.decode())
    return sorted(set(owners))


def _cjk_count(data: bytes) -> int:
    if b"\0" in data:
        return 0  # binary file
    return len(CJK_RE.findall(data.decode("utf-8", errors="ignore")))


def phase_tags(data: bytes) -> list[str]:
    return sorted({m.group(0).decode() for m in PHASE_TAG_RE.finditer(data)})


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        capture_output=True, check=True,
    ).stdout.decode()
    return [p for p in out.split("\0") if p]


def load_allow() -> list[str]:
    try:
        with open(ALLOW_FILE, encoding="utf-8") as fh:
            return [l.strip() for l in fh if l.strip() and not l.lstrip().startswith("#")]
    except FileNotFoundError:
        return []


def check(paths: list[str], warnings: list[str] | None = None) -> list[str]:
    allow = load_allow()
    problems: list[str] = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        if size > MAX_BYTES and not any(fnmatch.fnmatch(path, g) for g in allow):
            problems.append(f"{path}: {size} bytes > {MAX_BYTES} (upload raw data to the Modal Volume, or add to {ALLOW_FILE})")
        if os.path.normpath(path) in SELF:
            continue
        with open(path, "rb") as fh:
            data = fh.read(8 * 1024 * 1024)
        for name, rx in SECRET_RE.items():
            if rx.search(data):
                problems.append(f"{path}: possible secret ({name})")
        if EMAIL_RE.search(data):
            problems.append(f"{path}: personal e-mail address (replace with <redacted-email>)")
        norm = os.path.normpath(path).replace(os.sep, "/")
        if PRIVATE_IP_RE.search(data) and not any(fnmatch.fnmatch(norm, g) for g in IP_FAKE_DATA_GLOBS):
            problems.append(f"{path}: private-network IP address (use a placeholder or 192.0.2.x)")
        if _personal_home(data):
            problems.append(f"{path}: per-user home path (use ~, $HOME or an environment variable)")
        if TENANT_RE.search(data):
            problems.append(f"{path}: cloud tenant ID (pass it as a parameter or environment variable)")
        owners = _personal_registry(data)
        if owners and not any(fnmatch.fnmatch(norm, g) for g in REGISTRY_PIN_FILES):
            problems.append(f"{path}: personal container registry ghcr.io/{','.join(owners)} (make the registry a parameter)")
        cjk = _cjk_count(data)
        if cjk:
            problems.append(f"{path}: {cjk} CJK characters (the repository is English-only)")
        if warnings is not None:
            tags = phase_tags(data)
            if tags:
                warnings.append(f"{path}: phase/defect tags {' '.join(tags[:10])}")
    return problems


def main(argv: list[str]) -> int:
    paths = argv[1:] or staged_files()
    warnings: list[str] = []
    problems = check(paths, warnings)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
