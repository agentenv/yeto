"""Hugging Face hub repos whose small files (tokenizer, chat template, config)
some unit tests load.

This list is the single source for ``scripts/prefetch_hf_test_assets.py`` and
for the CI cache key of the HF hub directory. Model weights are never fetched.

Tests call :func:`require_hf_repo` to get the local snapshot directory. Without
a cached snapshot the test is skipped, or fails when ``YETO_REQUIRE_HF_ASSETS=1``
(CI sets it after restoring/prefetching the cache, so a missing repo is loud).
"""

from __future__ import annotations

import os

import pytest

# repo id -> pinned revision (commit sha)
HF_TEST_REPOS: dict[str, str] = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3-4B-Instruct-2507": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "zai-org/GLM-4.7-Flash": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16": "fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe",
}

# Repos that need an access token (none today). CI passes secrets.HF_TOKEN to the
# prefetch step; without it these repos are not fetched and their tests skip.
HF_GATED_REPOS: frozenset[str] = frozenset()

# Only tokenizer / template / config files; never weights.
HF_ALLOW_PATTERNS: list[str] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
    "chat_template.json",
]

REQUIRE_ENV = "YETO_REQUIRE_HF_ASSETS"


def hf_snapshot_dir(repo: str) -> str | None:
    """Local snapshot directory of ``repo`` at its pinned revision, or None."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:
        return snapshot_download(repo, revision=HF_TEST_REPOS[repo], allow_patterns=HF_ALLOW_PATTERNS,
                                 local_files_only=True)
    except Exception:
        return None


def require_hf_repo(repo: str) -> str:
    """Return the cached snapshot of ``repo``; skip (or fail in CI) when absent."""
    if repo not in HF_TEST_REPOS:
        raise KeyError(f"{repo} is not declared in tests/hf_assets.py:HF_TEST_REPOS")
    path = hf_snapshot_dir(repo)
    if path is not None:
        return path
    msg = f"{repo} tokenizer/config not in the local HF cache (run scripts/prefetch_hf_test_assets.py)"
    gated_without_token = repo in HF_GATED_REPOS and not os.environ.get("HF_TOKEN")
    if os.environ.get(REQUIRE_ENV) == "1" and not gated_without_token:
        pytest.fail(msg)
    pytest.skip(msg)
