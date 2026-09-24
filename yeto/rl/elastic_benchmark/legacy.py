"""Adapter over the existing ``scripts/benchmark_rl.py`` harness.

The legacy harness is a script, not a package, so it is loaded by path. Only
its prompt pairing, normalisation and JSONL helpers are reused; its CLI and
result contract stay untouched. Loading is lazy so ``plan``/``validate`` never
pay for the script's imports.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_SCRIPT = REPO_ROOT / "scripts" / "benchmark_rl.py"
_MODULE_NAME = "yeto_legacy_benchmark_rl"


@lru_cache(maxsize=1)
def load_legacy_harness(script: Path = LEGACY_SCRIPT) -> ModuleType:
    if not script.is_file():
        raise FileNotFoundError(f"legacy RL benchmark harness is missing: {script}")
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy harness from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_legacy_harness()._read_jsonl(path)


def paired_streams(rows: list[dict[str, Any]], *, islands: int, groups: int, rounds: int):
    """Round-major paired prompt streams, identical to the legacy arms' pairing."""
    return load_legacy_harness().paired_prompt_streams(rows, islands=islands, groups=groups, rounds=rounds)


def write_prompt_files(streams, evaluation_rows: list[dict[str, Any]], directory: Path):
    return load_legacy_harness().write_prompt_files(streams, evaluation_rows, directory)


def normalized_prompt(row: dict[str, Any], prompt_id: int | str) -> dict[str, Any]:
    return load_legacy_harness()._normalized_prompt(row, prompt_id)


def materialize_paired_inputs(
    rows: list[dict[str, Any]],
    *,
    train_ids: list[int],
    held_out_ids: list[int],
    assignment: list[list[int]],
    directory: Path,
) -> dict[str, Any]:
    """Write the elastic study's per-update prompt stream in the legacy file format.

    ``assignment`` lists train row indexes per logical update (see
    ``work.assign_groups``). Every arm reads the same ``combined.jsonl``;
    ``eval.jsonl`` holds held-out rows only.
    """
    leaked = sorted(set(held_out_ids) & {i for update in assignment for i in update})
    if leaked:
        raise ValueError(f"held-out rows leaked into training assignment: {leaked[:4]}")
    stream_rows = [dict(rows[i]) for update in assignment for i in update]
    stream_ids = [i for update in assignment for i in update]
    harness = load_legacy_harness()
    streams = harness.PromptStreams(
        combined_rows=tuple(stream_rows),
        island_rows=(tuple(stream_rows),),
        combined_ids=tuple(stream_ids),
        island_ids=(tuple(stream_ids),),
    )
    combined, island_paths, evaluation = harness.write_prompt_files(
        streams, [dict(rows[i]) for i in held_out_ids], directory
    )
    return {
        "combined": str(combined),
        "islands": [str(p) for p in island_paths],
        "eval": str(evaluation),
        "train_rows": len(stream_rows),
        "held_out_rows": len(held_out_ids),
        "updates": len(assignment),
        "train_ids_unique": len(set(train_ids)),
    }
