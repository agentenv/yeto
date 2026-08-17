"""Privacy-bounded extraction of scalar RL training metrics.

This module deliberately does not deserialize complete log records.  It only
recognizes a small allowlist of scalar keys in Miles train/eval logs and Yeto
``rl_local_round`` events.  In particular, unknown fields and their values are
never copied into the output summaries.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

try:  # Linux and macOS production hosts provide advisory file locking.
    import fcntl
except ImportError:  # pragma: no cover - unsupported production platform
    fcntl = None


SCHEMA_VERSION = 1
MAX_LOG_LINE_BYTES = 1 << 20
TRAIN_METRIC_FIELDS = (
    "loss",
    "pg_loss",
    "grad_norm",
    "kl",
    "ess",
    "clipfrac",
    "lr",
    "reward",
    "pass_rate",
)
EVAL_METRIC_FIELDS = (
    "eval_score",
    "eval_pass_at_1",
    "eval_truncated_ratio",
)
METRIC_FIELDS = (*TRAIN_METRIC_FIELDS, *EVAL_METRIC_FIELDS)
CSV_FIELDS = ("source", "step", *METRIC_FIELDS)

_EVAL_NUMBER = (
    r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|"
    r"(?i:nan|inf(?:inity)?))"
)
# Metric records may be read directly or through ``docker logs --timestamps``.
# The latter adds an RFC3339 timestamp before Ray's optional ANSI-coloured
# ``(Actor pid=N)`` prefix.  Keep every accepted component anchored and narrow:
# arbitrary logger/payload prose before ``step`` or ``eval`` must never match.
_DOCKER_TIMESTAMP = r"(?:\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z[ \t]+)?"
_ANSI_SGR = r"(?:\x1b\[[0-9;]*m)*"
_RAY_PREFIX = (
    rf"(?:{_ANSI_SGR}\([A-Za-z][A-Za-z0-9_.<>-]* pid=[0-9]+\)"
    rf"{_ANSI_SGR}[ \t]+)?"
)
_MILES_LOGGER_PREFIX = (
    r"(?:\[[0-9]{4}-[0-9]{2}-[0-9]{2} [^\]{}'\"\r\n]+\] "
    r"[A-Za-z0-9_./<>-]+:[0-9]+ -[ \t]+)?"
)
_METRIC_RECORD_PREFIX = _DOCKER_TIMESTAMP + _RAY_PREFIX + _MILES_LOGGER_PREFIX
_STEP_PREFIX_RE = re.compile(rf"^{_METRIC_RECORD_PREFIX}step (?P<step>[0-9]+):\s*\{{")
_EVAL_PREFIX_RE = re.compile(rf"^{_METRIC_RECORD_PREFIX}eval (?P<step>[0-9]+):\s*\{{")
_SAFE_SOURCE_RE = re.compile(r"[A-Za-z0-9_.-]+\Z")
_FINGERPRINT_SOURCE_RE = re.compile(r"[0-9a-fA-F]{32,128}\Z")
_SAFE_DATASET = r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,62}[A-Za-z0-9])?"
_SAFE_DATASET_RE = re.compile(rf"{_SAFE_DATASET}\Z")
_RESERVED_EVAL_SCORE_SUFFIXES = ("-none_reward_ratio", "-truncated_ratio")

# Aliases are ordered by preference.  The first finite value on a line wins.
# Bare keys such as ``loss`` and ``reward`` are intentionally excluded: they
# occur in rollout payloads and are not unambiguous training telemetry.
_ALIASES = {
    "loss": ("train/loss",),
    "pg_loss": ("train/pg_loss",),
    "grad_norm": ("train/grad_norm",),
    "kl": (
        "train/train_rollout_kl",
        "train/ppo_kl",
        "train/approx_kl",
        "train/mean_kl",
        "train/kl",
        "rl/current_vs_rollout_kl",
    ),
    "ess": ("train/ess_ratio", "rl/ess_ratio"),
    "clipfrac": (
        "train/pg_clipfrac",
        "train/clipfrac",
        "train/clip_fraction",
        "rl/clip_fraction",
    ),
    "lr": (
        "train/lr-pg_0",
        "train/lr",
        "train/learning_rate",
        "train/policy_lr",
    ),
    "reward": (
        "train/reward_mean",
        "rl/reward_mean",
        "eval/reward_mean",
    ),
    "pass_rate": (
        "train/pass_rate",
        "rl/pass_rate",
        "eval/pass_rate",
        "eval/pass@1",
    ),
}
_STEP_ALIASES = ("train/step", "global_step", "optimizer_step")


def _item_value_re(key_pattern: str) -> re.Pattern[str]:
    return re.compile(
        rf"\A\s*(?P<quote>['\"])(?:{key_pattern})(?P=quote)\s*[:=]\s*"
        rf"(?:"
        rf"(?P<number_plain>{_EVAL_NUMBER})"
        rf"|(?:np\.)?float(?:16|32|64)?\(\s*"
        rf"(?P<number_float>{_EVAL_NUMBER})\s*\)"
        rf"|tensor\(\s*(?P<number_tensor>{_EVAL_NUMBER})(?:\s*,[^)]*)?\)"
        rf")\s*\Z"
    )


_ITEM_SCALAR_RES = {
    field: tuple(_item_value_re(re.escape(alias)) for alias in aliases)
    for field, aliases in _ALIASES.items()
}
_ITEM_SCALAR_RES["lr"] += (_item_value_re(r"train/lr-pg_[0-9]+"),)
_ITEM_STEP_RES = tuple(_item_value_re(re.escape(alias)) for alias in _STEP_ALIASES)
_ITEM_ROUND_STEP_RES = (_item_value_re(r"(?:local_round_id|rl/local_round_id)"),)
_EVENT_ITEM_RE = re.compile(
    r"\A\s*(?P<quote>['\"])event(?P=quote)\s*:\s*"
    r"(?P<value_quote>['\"])(?P<event>[A-Za-z0-9_.-]+)(?P=value_quote)\s*\Z"
)


def _eval_item_re(suffix: str) -> re.Pattern[str]:
    key = rf"eval/(?P<dataset>{_SAFE_DATASET}){re.escape(suffix)}"
    return re.compile(
        rf"\A\s*(?P<quote>['\"]){key}(?P=quote)\s*:\s*"
        rf"(?:"
        rf"(?P<number_plain>{_EVAL_NUMBER})"
        rf"|(?:np\.)?float(?:16|32|64)?\(\s*"
        rf"(?P<number_float>{_EVAL_NUMBER})\s*\)"
        rf"|tensor\(\s*(?P<number_tensor>{_EVAL_NUMBER})(?:\s*,[^)]*)?\)"
        rf")\s*\Z",
    )


_EVAL_ITEM_RES = {
    "eval_pass_at_1": _eval_item_re("-pass@1"),
    "eval_truncated_ratio": _eval_item_re("-truncated_ratio"),
    "eval_score": _eval_item_re(""),
}


@dataclass(frozen=True)
class MetricRow:
    """One deduplicated source/optimizer-step point."""

    source: str
    step: int
    loss: float | None = None
    pg_loss: float | None = None
    grad_norm: float | None = None
    kl: float | None = None
    ess: float | None = None
    clipfrac: float | None = None
    lr: float | None = None
    reward: float | None = None
    pass_rate: float | None = None
    eval_score: float | None = None
    eval_pass_at_1: float | None = None
    eval_truncated_ratio: float | None = None

    def __post_init__(self) -> None:
        validate_source_label(self.source)
        if (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or self.step < 0
        ):
            raise ValueError("metric step must be a non-negative integer")
        for name in METRIC_FIELDS:
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when present")

    @property
    def key(self) -> tuple[str, int]:
        return self.source, self.step


def validate_source_label(label: str) -> str:
    """Require an opaque, output-safe source label rather than a path."""

    if (
        len(label) > 64
        or not _SAFE_SOURCE_RE.fullmatch(label)
        or _FINGERPRINT_SOURCE_RE.fullmatch(label)
    ):
        raise ValueError(
            "source labels must be short opaque names (for example, island-0), "
            "not paths or fingerprints"
        )
    return label


def validate_eval_dataset(dataset: str) -> str:
    """Require the exact output-safe Miles eval dataset label."""

    if not _SAFE_DATASET_RE.fullmatch(dataset):
        raise ValueError("eval dataset must be a short alphanumeric label")
    return dataset


def _top_level_dict_items(line: str, opening_brace: int) -> list[str]:
    """Split one complete flat-looking repr without decoding any values."""

    items: list[str] = []
    depth = 1
    quote: str | None = None
    escaped = False
    item_start = opening_brace + 1
    for index in range(item_start, len(line)):
        char = line[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char in "{[(":
            depth += 1
        elif char in "}])":
            depth -= 1
            if depth == 0:
                items.append(line[item_start:index])
                return items
            if depth < 0:
                return []
        elif char == "," and depth == 1:
            items.append(line[item_start:index])
            item_start = index + 1
    return []


def _eval_scalar(match: re.Match[str]) -> float | None:
    raw = next(
        (
            match.group(name)
            for name in ("number_plain", "number_float", "number_tensor")
            if match.group(name) is not None
        ),
        None,
    )
    if raw is None:
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _finite_item_match(
    patterns: Iterable[re.Pattern[str]], items: Iterable[str]
) -> float | None:
    for pattern in patterns:
        for item in items:
            match = pattern.fullmatch(item)
            if match is None:
                continue
            value = _eval_scalar(match)
            if value is not None:
                return value
    return None


def _integer_item_match(
    patterns: Iterable[re.Pattern[str]], items: Iterable[str]
) -> int | None:
    value = _finite_item_match(patterns, items)
    if value is None or value < 0 or not value.is_integer():
        return None
    return int(value)


def _parse_eval_line(
    line: str,
    source: str,
    prefix: re.Match[str],
    expected_dataset: str | None,
) -> MetricRow | None:
    items = _top_level_dict_items(line, prefix.end() - 1)
    recognized: list[tuple[int, str, str, float | None]] = []
    for item_index, item in enumerate(items):
        for field in (
            "eval_pass_at_1",
            "eval_truncated_ratio",
            "eval_score",
        ):
            match = _EVAL_ITEM_RES[field].fullmatch(item)
            if match is None:
                continue
            dataset = match.group("dataset")
            if expected_dataset is not None and dataset != expected_dataset:
                continue
            if field == "eval_score" and dataset.endswith(
                _RESERVED_EVAL_SCORE_SUFFIXES
            ):
                continue
            recognized.append((item_index, dataset, field, _eval_scalar(match)))
            break
    if not recognized:
        return None

    by_dataset: dict[str, set[str]] = {}
    first_index: dict[str, int] = {}
    for item_index, item_dataset, field, _ in recognized:
        by_dataset.setdefault(item_dataset, set()).add(field)
        first_index.setdefault(item_dataset, item_index)
    complete_datasets = [
        item_dataset
        for item_dataset, item_fields in by_dataset.items()
        if "eval_score" in item_fields
        and item_fields.intersection({"eval_pass_at_1", "eval_truncated_ratio"})
    ]
    if complete_datasets:
        dataset = min(complete_datasets, key=first_index.__getitem__)
    else:
        # A suffix is unambiguous evidence of the dataset name.  Prefer it over
        # an earlier unsuffixed extra metric; otherwise use the first score.
        companion = next(
            (
                item
                for item in recognized
                if item[2] in ("eval_pass_at_1", "eval_truncated_ratio")
            ),
            None,
        )
        dataset = (companion or recognized[0])[1]
    values: dict[str, float | None] = {field: None for field in EVAL_METRIC_FIELDS}
    for _, item_dataset, field, value in recognized:
        if item_dataset == dataset and value is not None:
            values[field] = value
    if not any(value is not None for value in values.values()):
        return None
    return MetricRow(source=source, step=int(prefix.group("step")), **values)


def parse_metric_line(
    line: str, source: str, *, eval_dataset: str | None = None
) -> MetricRow | None:
    """Extract only allowlisted finite scalars from one bounded log line.

    Accepted records are Miles ``step N: {...}`` and exact ``eval N: {...}``
    lines, tracker dictionaries containing ``train/step``, or Yeto
    ``rl_local_round`` events.  Requiring one of those contexts avoids
    interpreting scalar-looking text inside rollout payload logs as telemetry.
    """

    validate_source_label(source)
    if eval_dataset is not None:
        validate_eval_dataset(eval_dataset)
    if len(line.encode("utf-8", errors="ignore")) > MAX_LOG_LINE_BYTES:
        return None

    eval_prefix = _EVAL_PREFIX_RE.search(line)
    if eval_prefix is not None:
        return _parse_eval_line(line, source, eval_prefix, eval_dataset)

    prefix = _STEP_PREFIX_RE.match(line)
    if prefix is not None:
        items = _top_level_dict_items(line, prefix.end() - 1)
        dictionary_step = _integer_item_match(_ITEM_STEP_RES, items)
        # Miles always includes train/step in the logged dictionary.  Requiring
        # it to agree with the human-readable prefix sharply reduces the chance
        # of treating payload text containing ``step N`` as telemetry.
        prefix_step = int(prefix.group("step"))
        if dictionary_step != prefix_step:
            return None
        step = prefix_step
    else:
        # Structured Yeto events are consumed from the event tape as raw JSON
        # objects.  Do not accept a logger/payload line which merely contains
        # an ``event`` key somewhere in its text.
        opening_brace = len(line) - len(line.lstrip())
        if opening_brace >= len(line) or line[opening_brace] != "{":
            return None
        items = _top_level_dict_items(line, opening_brace)
        event = next(
            (
                match.group("event")
                for item in items
                if (match := _EVENT_ITEM_RE.fullmatch(item)) is not None
            ),
            None,
        )
        if event != "rl_local_round":
            return None
        step = _integer_item_match(_ITEM_ROUND_STEP_RES, items)
        if step is None:
            return None

    values = {
        field: _finite_item_match(_ITEM_SCALAR_RES[field], items)
        for field in TRAIN_METRIC_FIELDS
    }
    if not any(value is not None for value in values.values()):
        return None
    return MetricRow(source=source, step=step, **values)


def merge_rows(*groups: Iterable[MetricRow]) -> list[MetricRow]:
    """Merge by ``(source, step)``, filling/updating only finite scalars."""

    merged: dict[tuple[str, int], MetricRow] = {}
    for group in groups:
        for row in group:
            previous = merged.get(row.key)
            if previous is None:
                merged[row.key] = row
                continue
            values = {
                name: (
                    getattr(row, name)
                    if getattr(row, name) is not None
                    else getattr(previous, name)
                )
                for name in METRIC_FIELDS
            }
            merged[row.key] = MetricRow(row.source, row.step, **values)
    return sorted(merged.values(), key=lambda row: (row.source, row.step))


def _consume_oversized_line(handle: BinaryIO, chunk: bytes) -> bool:
    while chunk and not chunk.endswith(b"\n"):
        chunk = handle.readline(MAX_LOG_LINE_BYTES + 1)
    return chunk.endswith(b"\n")


def scan_log(
    path: Path,
    source: str,
    *,
    offset: int = 0,
    complete_lines_only: bool = False,
    eval_dataset: str | None = None,
) -> tuple[list[MetricRow], int]:
    """Scan bounded lines starting at ``offset`` and return the next offset."""

    validate_source_label(source)
    if eval_dataset is not None:
        validate_eval_dataset(eval_dataset)
    rows: list[MetricRow] = []
    with path.open("rb") as handle:
        handle.seek(offset)
        next_offset = offset
        while True:
            line_start = handle.tell()
            raw = handle.readline(MAX_LOG_LINE_BYTES + 1)
            if not raw:
                break
            if len(raw) > MAX_LOG_LINE_BYTES:
                line_complete = _consume_oversized_line(handle, raw)
                if complete_lines_only and not line_complete:
                    # Revisit from the beginning after the writer terminates
                    # the line.  Otherwise a later tail of one oversized
                    # payload could be mistaken for a fresh metric record.
                    next_offset = line_start
                    break
                next_offset = handle.tell()
                continue
            if complete_lines_only and not raw.endswith(b"\n"):
                next_offset = line_start
                break
            row = parse_metric_line(
                raw.decode("utf-8", errors="replace"),
                source,
                eval_dataset=eval_dataset,
            )
            if row is not None:
                rows.append(row)
            next_offset = handle.tell()
    return merge_rows(rows), next_offset


def iter_metric_stream(
    handle: BinaryIO, source: str, *, eval_dataset: str | None = None
) -> Iterator[MetricRow]:
    """Yield scalar-only rows from a seekless byte stream.

    This is intended for ``docker logs --follow`` input.  Oversized records are
    consumed and discarded as a unit, so a payload tail cannot be reinterpreted
    as a new telemetry line.
    """

    validate_source_label(source)
    if eval_dataset is not None:
        validate_eval_dataset(eval_dataset)
    while True:
        raw = handle.readline(MAX_LOG_LINE_BYTES + 1)
        if not raw:
            return
        if len(raw) > MAX_LOG_LINE_BYTES:
            _consume_oversized_line(handle, raw)
            continue
        row = parse_metric_line(
            raw.decode("utf-8", errors="replace"),
            source,
            eval_dataset=eval_dataset,
        )
        if row is not None:
            yield row


def load_csv(path: Path) -> list[MetricRow]:
    if not path.exists():
        return []
    rows: list[MetricRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"existing CSV has an incompatible schema: {path}")
        for raw in reader:
            values = {
                name: (None if raw[name] == "" else float(raw[name]))
                for name in METRIC_FIELDS
            }
            rows.append(MetricRow(raw["source"], int(raw["step"]), **values))
    return merge_rows(rows)


def _csv_text(rows: Iterable[MetricRow]) -> str:
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        raw = asdict(row)
        writer.writerow(
            {
                name: (
                    ""
                    if raw[name] is None
                    else format(raw[name], ".17g")
                    if isinstance(raw[name], float)
                    else raw[name]
                )
                for name in CSV_FIELDS
            }
        )
    return output.getvalue()


def summary_document(rows: Iterable[MetricRow]) -> dict[str, object]:
    ordered = merge_rows(rows)
    sources: dict[str, dict[str, int]] = {}
    for row in ordered:
        summary = sources.setdefault(row.source, {"points": 0, "latest_step": row.step})
        summary["points"] += 1
        summary["latest_step"] = max(summary["latest_step"], row.step)
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_fields": list(METRIC_FIELDS),
        "sources": sources,
        "rows": [asdict(row) for row in ordered],
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


@contextmanager
def _output_lock(csv_path: Path) -> Iterator[None]:
    lock_path = csv_path.with_name(f".{csv_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def persist_summaries(
    rows: Iterable[MetricRow], csv_path: Path, json_path: Path
) -> list[MetricRow]:
    """Atomically merge new points into append-safe CSV and JSON summaries."""

    with _output_lock(csv_path):
        merged = merge_rows(load_csv(csv_path), rows)
        _atomic_write(csv_path, _csv_text(merged))
        document = summary_document(merged)
        _atomic_write(
            json_path,
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    return merged


def write_png(rows: Iterable[MetricRow], path: Path) -> bool:
    """Write an atomic scalar-only plot when matplotlib is already installed."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    ordered = merge_rows(rows)
    plotted_fields = [
        name
        for name in METRIC_FIELDS
        if any(getattr(row, name) is not None for row in ordered)
    ]
    if not plotted_fields:
        return False

    columns = 2
    row_count = (len(plotted_fields) + columns - 1) // columns
    figure, axes = plt.subplots(
        row_count, columns, squeeze=False, figsize=(12, 3.5 * row_count)
    )
    sources = sorted({row.source for row in ordered})
    for axis, name in zip(axes.flat, plotted_fields, strict=False):
        for source in sources:
            points = [
                row
                for row in ordered
                if row.source == source and getattr(row, name) is not None
            ]
            if points:
                axis.plot(
                    [row.step for row in points],
                    [getattr(row, name) for row in points],
                    marker="o",
                    markersize=3,
                    label=source,
                )
        axis.set_title(name)
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes.flat[len(plotted_fields) :]:
        axis.set_visible(False)
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        figure.savefig(temp_name, format="png", dpi=140)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    finally:
        plt.close(figure)
    return True
