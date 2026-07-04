"""Prepare Gemini clip labels for NAVA training manifests.

Input is the exported gzip JSON array produced under the lora_labels S3 prefix.
Output is NAVA's JSONL training format. The converter is intentionally strict:
ambiguous age-safety or rejected/low-quality rows are excluded by default.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

from .utils import is_s3_uri, materialize_uri, split_s3_uri


DEFAULT_AUDIO_HINT = "Audio follows the visible scene and contains the sounds described by the source label."


def normalize_label_input(uri: str) -> str:
    """Accept either the label file itself or the export prefix directory."""
    suffixes = (".json", ".jsonl", ".json.gz", ".jsonl.gz")
    if uri.endswith(suffixes):
        return uri
    if uri.endswith("/") or is_s3_uri(uri) or Path(uri).is_dir():
        return uri.rstrip("/") + "/clip_s3_labels.json.gz"
    return uri


def _open_text(uri: str, cache_dir: str | None = None):
    local = materialize_uri(uri, cache_dir or os.environ.get("YETO_NAVA_LABEL_CACHE") or tempfile.gettempdir())
    if local.endswith(".gz"):
        return gzip.open(local, "rt", encoding="utf-8")
    return open(local, "r", encoding="utf-8")


def load_label_records(uri: str, cache_dir: str | None = None) -> Iterable[dict]:
    with _open_text(uri, cache_dir) as f:
        # Main export is a JSON array. JSONL is accepted for easier tests and
        # manual curation.
        first = f.read(1)
        f.seek(0)
        if first == "[":
            for item in json.load(f):
                yield item
        else:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _age_safety_text(age_safety) -> str:
    if age_safety is None:
        return ""
    if isinstance(age_safety, str):
        return age_safety
    if isinstance(age_safety, dict):
        return " ".join(_age_safety_text(v) for v in age_safety.values())
    if isinstance(age_safety, (list, tuple, set)):
        return " ".join(_age_safety_text(v) for v in age_safety)
    return str(age_safety)


def _safe_adult(age_safety) -> bool:
    text = _age_safety_text(age_safety)
    if not text:
        return False
    low = text.lower()
    bad = ("minor", "under", "child", "teen", "young-looking", "uncertain", "unknown", "not sure")
    if any(x in low for x in bad):
        return False
    return "adult" in low or "18" in low


def keep_record(
    record: dict,
    *,
    qualities: set[str],
    require_trainable: bool,
    strict_age: bool,
    min_duration: float | None,
    max_duration: float | None,
) -> tuple[bool, str]:
    label = record.get("label") or {}
    if require_trainable and label.get("trainable") is not True:
        return False, "not_trainable"
    if label.get("reject_reasons"):
        return False, "reject_reasons"
    quality = str(label.get("quality") or "").lower()
    if qualities and quality not in qualities:
        return False, "quality"
    if strict_age and not _safe_adult(label.get("age_safety")):
        return False, "age_safety"
    if not record.get("clip_s3_uri"):
        return False, "missing_clip_uri"
    duration = record.get("duration")
    if duration is not None:
        duration = float(duration)
        if min_duration is not None and duration < min_duration:
            return False, "duration_short"
        if max_duration is not None and duration > max_duration:
            return False, "duration_long"
    if not (label.get("dense_lora_caption") or label.get("short_lora_caption") or label.get("lora_tags")):
        return False, "missing_caption"
    return True, "ok"


def compose_caption(label: dict, field: str) -> str:
    if field == "dense_lora_caption":
        return str(label.get("dense_lora_caption") or "")
    if field == "short_lora_caption":
        return str(label.get("short_lora_caption") or label.get("dense_lora_caption") or "")
    if field == "lora_tags":
        return str(label.get("lora_tags") or label.get("dense_lora_caption") or "")
    if field != "composed":
        raise ValueError(f"unknown caption field {field!r}")
    parts = []
    dense = label.get("dense_lora_caption") or label.get("short_lora_caption")
    if dense:
        parts.append(str(dense))
    for key, prefix in [
        ("camera", "Camera"),
        ("background", "Background"),
        ("actions", "Actions"),
        ("visual_pairing", "Visual pairing"),
        ("audio", "Audio"),
        ("lora_tags", "Tags"),
    ]:
        val = label.get(key)
        if not val:
            continue
        if isinstance(val, list):
            val = ", ".join(map(str, val))
        parts.append(f"{prefix}: {val}")
    if not any(p.startswith("Audio:") for p in parts):
        parts.append(DEFAULT_AUDIO_HINT)
    return "\n".join(parts).strip()


def ffprobe_video(uri: str, cache_dir: str | None = None) -> dict:
    """Return width/height/fps/duration when ffprobe is available."""
    path = materialize_uri(uri, cache_dir)
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate:format=duration",
                "-of",
                "json",
                path,
            ],
            text=True,
            timeout=60,
        )
        meta = json.loads(out)
        stream = (meta.get("streams") or [{}])[0]
        frac = stream.get("r_frame_rate") or "0/1"
        num, _, den = frac.partition("/")
        fps = float(num) / max(1.0, float(den or 1))
        return {
            "image_width": int(stream.get("width") or 0),
            "image_height": int(stream.get("height") or 0),
            "fps": fps or 24.0,
            "duration": float((meta.get("format") or {}).get("duration") or 0),
        }
    except Exception:
        return {}


def to_nava_row(record: dict, *, caption_field: str, probe: bool, cache_dir: str | None) -> dict:
    label = record.get("label") or {}
    clip_uri = record["clip_s3_uri"]
    probed = ffprobe_video(clip_uri, cache_dir) if probe else {}
    duration = float(probed.get("duration") or record.get("duration") or 5.0)
    width = int(probed.get("image_width") or label.get("image_width") or 0)
    height = int(probed.get("image_height") or label.get("image_height") or 0)
    fps = float(probed.get("fps") or 24.0)
    return {
        "data_id": record.get("clip_id") or label.get("clip_id"),
        "video_info": [
            {
                "data_path": clip_uri,
                "fps": fps,
                "duration": duration,
                "image_width": width,
                "image_height": height,
                "is_valid": True,
            }
        ],
        "text_list": [
            {
                "text": compose_caption(label, caption_field),
                "text_type": "caption",
                "is_valid": True,
            }
        ],
        "meta": {
            "source": "gemini3_full_v3",
            "clip_id": record.get("clip_id"),
            "source_video_id": record.get("source_video_id"),
            "request_id": record.get("request_id"),
            "quality": label.get("quality"),
        },
    }


def _write_text(uri: str, text: str) -> None:
    if is_s3_uri(uri):
        import boto3

        bucket, key = split_s3_uri(uri)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))
    else:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def prepare_labels(
    input_uri: str,
    output_uri: str,
    *,
    report_uri: str | None = None,
    caption_field: str = "composed",
    qualities: set[str] | None = None,
    require_trainable: bool = True,
    strict_age: bool = True,
    min_duration: float | None = None,
    max_duration: float | None = None,
    max_output_rows: int | None = None,
    probe: bool = False,
    cache_dir: str | None = None,
) -> dict:
    input_uri = normalize_label_input(input_uri)
    qualities = {q.lower() for q in (qualities or {"good", "high"})}
    counts = Counter()
    rows = []
    for rec in load_label_records(input_uri, cache_dir):
        counts["input"] += 1
        ok, reason = keep_record(
            rec,
            qualities=qualities,
            require_trainable=require_trainable,
            strict_age=strict_age,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        if not ok:
            counts[f"filtered_{reason}"] += 1
            continue
        try:
            rows.append(to_nava_row(rec, caption_field=caption_field, probe=probe, cache_dir=cache_dir))
            counts["output"] += 1
            if max_output_rows is not None and counts["output"] >= max_output_rows:
                counts["stopped_max_output_rows"] = 1
                break
        except Exception:
            counts["filtered_convert_error"] += 1
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    _write_text(output_uri, body)
    report = dict(counts)
    report.update(
        {
            "output_uri": output_uri,
            "input_uri": input_uri,
            "caption_field": caption_field,
            "qualities": sorted(qualities),
            "min_duration": min_duration,
            "max_duration": max_duration,
            "max_output_rows": max_output_rows,
        }
    )
    if report_uri:
        _write_text(report_uri, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Convert Gemini clip labels to NAVA JSONL")
    p.add_argument("--input", required=True, help="s3:// or local clip_s3_labels.json.gz/JSONL")
    p.add_argument("--output", required=True, help="s3:// or local output NAVA JSONL")
    p.add_argument("--report", default=None)
    p.add_argument("--caption-field", choices=["composed", "dense_lora_caption", "short_lora_caption", "lora_tags"], default="composed")
    p.add_argument("--quality", default="good,high", help="comma-separated allowed quality labels")
    p.add_argument("--allow-untrainable", action="store_true")
    p.add_argument("--allow-ambiguous-age", action="store_true")
    p.add_argument("--min-duration", type=float, default=None)
    p.add_argument("--max-duration", type=float, default=None)
    p.add_argument("--max-output-rows", type=int, default=None)
    p.add_argument("--ffprobe", action="store_true", help="ffprobe each clip to fill width/height/fps")
    p.add_argument("--cache-dir", default=os.environ.get("YETO_NAVA_DATA_CACHE"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = prepare_labels(
        args.input,
        args.output,
        report_uri=args.report,
        caption_field=args.caption_field,
        qualities={q.strip() for q in args.quality.split(",") if q.strip()},
        require_trainable=not args.allow_untrainable,
        strict_age=not args.allow_ambiguous_age,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_output_rows=args.max_output_rows,
        probe=args.ffprobe,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
