"""Training-time heldout eval on a verl island (S27 G-EN, VERL-PARITY #34).

The launcher sends the Miles island's eval flags to every island
(``--rl-eval-*`` -> ``--eval-interval --eval-data --eval-data-sha256
--eval-dataset-name --eval-samples-per-prompt`` and the optional sampling and
length flags, ``launch/rl_ports._ports_infra_flags``) and writes the heldout
JSONL on the island.  The schedule is the engine driver's: ``IslandDriver``
calls ``evaluate(rollout_id)`` after the publication of rollout 0, of every
``interval``-th rollout and of the last one, exactly as on a Miles island.
This module is the verl port behind that hook:

* the island checks the file against the launcher's SHA256 and reads the
  flags with the engine's ``run_config.verify_eval_dataset_file`` and
  ``run_config.eval_config_from_args`` (the Miles island's own reading);
* the heldout rows become verl's validation parquet (``data.val_files``) in the
  schema of the island's training rows: chat ``messages`` and a ``label``
  (``island_entry.prepare_data``), scored by the island's reward function;
* sampling: ``val_kwargs.n`` is the samples per prompt; temperature and top-p
  are the eval flags, else the training rollout's values (the Miles fallback,
  ``miles/utils/eval_config.py``).  verl's validation shares the training
  prompt and response lengths (``data.max_prompt_length``,
  ``rollout.response_length``), so the eval length flags are accepted only at
  those values and refused otherwise;
* :class:`VerlValidate` runs verl's own ``_validate()`` on the policy the
  driver just published (a colocated island's replicas are awake right after
  ``update_weights``) and returns verl's validation metrics plus
  ``<dataset name>``: the mean reward over every eval sample, the value a Miles
  island logs as ``eval/<dataset name>``.  The driver writes them as one
  ``rl_eval`` event (``eval/<key>``).

Only the synchronous verl trainer (``trainer.run_island``: LoRA and
full-parameter islands) has the hook; the fully_async and buffer-trainer
paths refuse the eval flags before verl starts.
"""

from __future__ import annotations

import json
import math
import numbers
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PLAN_KEY = "eval"
DATA_SOURCE_PREFIX = "yeto-eval/"
VAL_FILES_KEY = "data.val_files"
VAL_KWARGS = "actor_rollout_ref.rollout.val_kwargs"
ASSERTED_KEYS = (VAL_FILES_KEY, f"{VAL_KWARGS}.n", f"{VAL_KWARGS}.temperature",
                 f"{VAL_KWARGS}.top_p", f"{VAL_KWARGS}.top_k", f"{VAL_KWARGS}.do_sample")

# The Miles island's eval flags (adapters/miles: island_entry + Miles' own parser).
_FLAGS = (
    ("--eval-interval", {"type": int, "default": None}),
    ("--eval-data", {"default": None}),
    ("--eval-data-sha256", {"default": None}),
    ("--eval-dataset-name", {"default": None}),
    ("--eval-samples-per-prompt", {"type": int, "default": None}),
    ("--eval-temperature", {"type": float, "default": None}),
    ("--eval-top-p", {"type": float, "default": None}),
    ("--eval-max-prompt-len", {"type": int, "default": None}),
    ("--eval-max-response-len", {"type": int, "default": None}),
    ("--eval-max-context-len", {"type": int, "default": None}),
)


class EvalPortError(ValueError):
    """The eval flags ask for something the verl island cannot run."""


def add_island_flags(parser) -> None:
    """The eval flags, in the Miles island spellings; a parser that already
    defines one keeps its own."""

    for flag, kwargs in _FLAGS:
        if flag not in parser._option_string_actions:
            parser.add_argument(flag, **kwargs)


def eval_requested(args: Any) -> bool:
    """True when the island got any eval flag (``--eval-interval`` or another)."""

    return any(getattr(args, flag[2:].replace("-", "_"), None) is not None for flag, _ in _FLAGS)


def refuse_unwired(args: Any, path: str) -> None:
    """Refuse the eval flags on a verl path without the evaluate hook (``path``
    names it), before any work; nothing when the island got none."""

    if eval_requested(args):
        raise EvalPortError(f"--eval-interval: the verl {path} path has no evaluate hook "
                            "(only the synchronous verl trainer runs verl's _validate)")


def heldout_rows(path: Path, *, data_source: str) -> list[dict]:
    """verl validation rows of the heldout JSONL ``path``: one row per line,
    each with chat ``messages`` (role and content) and a ``label``."""

    rows = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except ValueError as error:
            raise EvalPortError(f"--eval-data row {number} is not JSON: {error}") from None
        messages = raw.get("messages") if isinstance(raw, dict) else None
        if (not isinstance(messages, list) or not messages
                or not all(isinstance(m, dict) and {"role", "content"} <= set(m) for m in messages)):
            raise EvalPortError(f"--eval-data row {number}: verl reads chat 'messages' "
                                "(a list of role/content) as the island's training rows do")
        if raw.get("label") is None:
            raise EvalPortError(f"--eval-data row {number} has no 'label'")
        rows.append({
            "data_source": data_source,
            "prompt": [{"role": m["role"], "content": m["content"]} for m in messages],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(raw["label"])},
            "extra_info": {"split": "eval", "index": number},
        })
    if not rows:
        raise EvalPortError("--eval-data has no rows")
    return rows


def _write_parquet(rows: list[dict], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)


def eval_section(args: Any, *, work: Path, max_prompt_len: int, max_response_len: int,
                 temperature: float, top_p: float, write=_write_parquet) -> dict | None:
    """The island plan's ``eval`` section (None without ``--eval-interval``):
    the checked heldout file written as verl's validation parquet under
    ``work``, the schedule and the sampling.  Raises :class:`EvalPortError`
    (or the engine's ValueError) for a run the verl island cannot evaluate."""

    from yeto.rl.engine.run_config import eval_config_from_args, verify_eval_dataset_file

    interval = getattr(args, "eval_interval", None)
    if interval is None:
        if eval_requested(args):
            raise EvalPortError("the eval flags need --eval-interval")
        return None
    if type(interval) is not int or interval <= 0:
        raise EvalPortError(f"--eval-interval must be a positive int (got {interval!r})")
    source = verify_eval_dataset_file(args)
    config = eval_config_from_args(args, prompt_path=source)
    shared = (("--eval-max-prompt-len", config.max_prompt_len, int(max_prompt_len)),
              ("--eval-max-response-len", config.max_response_len, int(max_response_len)),
              ("--eval-max-context-len", config.max_context_len, int(max_prompt_len) + int(max_response_len)))
    for flag, value, island in shared:
        if value is not None and int(value) != island:
            raise EvalPortError(f"{flag} {value}: verl's validation shares the training lengths "
                                f"(this island: {island}); give {island} or leave it out")
    temp = float(temperature if config.temperature is None else config.temperature)
    top = float(top_p if config.top_p is None else config.top_p)
    if not (math.isfinite(temp) and temp >= 0.0) or not (math.isfinite(top) and 0.0 < top <= 1.0):
        raise EvalPortError(f"eval sampling out of range: temperature {temp}, top_p {top}")
    data_source = DATA_SOURCE_PREFIX + config.dataset_name
    rows = heldout_rows(source, data_source=data_source)
    val_file = Path(work) / "eval.parquet"
    write(rows, val_file)
    return {"interval": interval, "dataset_name": config.dataset_name, "data_source": data_source,
            "data_sha256": str(args.eval_data_sha256).lower(), "val_file": str(val_file),
            "rows": len(rows), "samples_per_prompt": int(config.samples_per_prompt),
            "temperature": temp, "top_p": top}


def _replace(overrides: Sequence[str], key: str, value: Any) -> list[str]:
    item = f"{key}={value}"
    out = [item if o.startswith(key + "=") else o for o in overrides]
    return out if item in out else [*out, item]


def apply_eval_overrides(section: dict | None, overrides: Sequence[str],
                         asserted_keys: Sequence[str]) -> tuple[list[str], tuple[str, ...]]:
    """``(overrides, asserted keys)`` with the eval section's validation data and
    sampling (unchanged without a section).  The keys are asserted against
    verl's resolved config at island start."""

    if section is None:
        return list(overrides), tuple(asserted_keys)
    out = _replace(overrides, VAL_FILES_KEY, section["val_file"])
    temp = section["temperature"]
    for key, value in (("n", section["samples_per_prompt"]), ("temperature", temp),
                       ("top_p", section["top_p"]), ("top_k", -1), ("do_sample", temp > 0.0)):
        out = _replace(out, f"{VAL_KWARGS}.{key}", value)
    return out, tuple(dict.fromkeys((*asserted_keys, *ASSERTED_KEYS)))


class VerlValidate:
    """``evaluate(rollout_id)`` of the engine driver over verl's ``_validate()``."""

    def __init__(self, trainer: Any, section: dict) -> None:
        loaded = len(trainer.val_dataset)
        if loaded != int(section["rows"]):
            # verl drops rows longer than data.max_prompt_length (filter_overlong_prompts):
            # a smaller eval set would be a different ruler, so the island stops here
            raise EvalPortError(f"verl loaded {loaded} of the {section['rows']} eval rows "
                                "(prompts over data.max_prompt_length are dropped)")
        self.trainer = trainer
        self.section = section
        # verl files the reward mean under val-core, or under val-aux when the reward
        # function also reports "acc" (the reward loop adds it for a float score:
        # experimental/reward_loop/reward_manager/naive.py); s27-en-s1 met the second.
        tail = f"/{section['data_source']}/reward/mean@{int(section['samples_per_prompt'])}"
        self.mean_keys = ("val-core" + tail, "val-aux" + tail)

    def __call__(self, rollout_id: int) -> dict[str, float]:
        del rollout_id  # the driver records it with the metrics
        t = self.trainer
        started = time.monotonic()
        t.on_validate_begin()
        metrics = t._validate()
        t.on_validate_end()
        out = {str(key): float(value) for key, value in dict(metrics).items()
               if isinstance(value, numbers.Real)}
        mean_key = next((key for key in self.mean_keys if key in out), None)
        if mean_key is None:
            raise RuntimeError(f"verl validation returned none of {self.mean_keys} (keys: {sorted(out)})")
        out[self.section["dataset_name"]] = out[mean_key]
        out["verl_validate_seconds"] = time.monotonic() - started
        return out


def driver_options(trainer: Any, plan: dict) -> dict[str, Any]:
    """``IslandDriver`` keyword arguments for the plan's eval section (empty without one)."""

    section = plan.get(PLAN_KEY)
    if not section:
        return {}
    return {"evaluate": VerlValidate(trainer, section), "eval_interval": int(section["interval"])}
