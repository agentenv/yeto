import gzip
import json

from yeto.nava.labels import prepare_labels


def test_prepare_labels_filters_and_writes_nava_jsonl(tmp_path):
    records = [
        {
            "clip_id": "clip:seg000001",
            "clip_s3_uri": "s3://bucket/path/seg000001.mp4",
            "duration": 5.0,
            "source_video_id": "clip",
            "request_id": "req1",
            "label": {
                "trainable": True,
                "quality": "good",
                "reject_reasons": [],
                "age_safety": "The participant appears to be an adult over the age of 18.",
                "dense_lora_caption": "A cinematic scene with synchronized audio.",
                "camera": "static medium shot",
                "audio": "ambient room tone",
            },
        },
        {
            "clip_id": "bad:seg000002",
            "clip_s3_uri": "s3://bucket/path/seg000002.mp4",
            "duration": 5.0,
            "label": {
                "trainable": False,
                "quality": "good",
                "reject_reasons": [],
                "age_safety": "adult",
                "dense_lora_caption": "Should be filtered.",
            },
        },
    ]
    src = tmp_path / "labels.json.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        json.dump(records, f)
    out = tmp_path / "train.jsonl"
    report = tmp_path / "report.json"

    stats = prepare_labels(str(src), str(out), report_uri=str(report), probe=False)

    assert stats["input"] == 2
    assert stats["output"] == 1
    assert stats["filtered_not_trainable"] == 1
    row = json.loads(out.read_text().strip())
    assert row["video_info"][0]["data_path"] == "s3://bucket/path/seg000001.mp4"
    assert row["video_info"][0]["duration"] == 5.0
    assert row["text_list"][0]["text_type"] == "caption"
    assert "Camera:" in row["text_list"][0]["text"]
    assert json.loads(report.read_text())["output"] == 1


def test_prepare_labels_accepts_prefix_and_duration_filters(tmp_path):
    records = [
        {
            "clip_id": "ok",
            "clip_s3_uri": "s3://bucket/ok.mp4",
            "duration": 3.0,
            "label": {
                "trainable": True,
                "quality": "good",
                "reject_reasons": [],
                "age_safety": "adult 18+",
                "dense_lora_caption": "ok",
            },
        },
        {
            "clip_id": "long",
            "clip_s3_uri": "s3://bucket/long.mp4",
            "duration": 30.0,
            "label": {
                "trainable": True,
                "quality": "good",
                "reject_reasons": [],
                "age_safety": "adult 18+",
                "dense_lora_caption": "too long",
            },
        },
    ]
    with gzip.open(tmp_path / "clip_s3_labels.json.gz", "wt", encoding="utf-8") as f:
        json.dump(records, f)
    out = tmp_path / "train.jsonl"

    stats = prepare_labels(str(tmp_path) + "/", str(out), max_duration=10)

    assert stats["output"] == 1
    assert stats["filtered_duration_long"] == 1
    assert json.loads(out.read_text())["data_id"] == "ok"


def test_prepare_labels_can_stop_after_max_output_rows(tmp_path):
    records = []
    for idx in range(3):
        records.append(
            {
                "clip_id": f"clip{idx}",
                "clip_s3_uri": f"s3://bucket/{idx}.mp4",
                "duration": 3.0,
                "label": {
                    "trainable": True,
                    "quality": "good",
                    "reject_reasons": [],
                    "age_safety": "adult 18+",
                    "dense_lora_caption": "caption",
                },
            }
        )
    src = tmp_path / "labels.json.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        json.dump(records, f)
    out = tmp_path / "train.jsonl"

    stats = prepare_labels(str(src), str(out), max_output_rows=2)

    assert stats["output"] == 2
    assert stats["stopped_max_output_rows"] == 1
    assert len(out.read_text().splitlines()) == 2


def test_prepare_labels_accepts_structured_age_safety(tmp_path):
    records = [
        {
            "clip_id": "ok",
            "clip_s3_uri": "s3://bucket/ok.mp4",
            "duration": 3.0,
            "label": {
                "trainable": True,
                "quality": "good",
                "reject_reasons": [],
                "age_safety": {
                    "status": "safe",
                    "reason": "Both participants appear to be adults over 18.",
                },
                "dense_lora_caption": "ok",
            },
        },
        {
            "clip_id": "bad",
            "clip_s3_uri": "s3://bucket/bad.mp4",
            "duration": 3.0,
            "label": {
                "trainable": True,
                "quality": "good",
                "reject_reasons": [],
                "age_safety": {"status": "uncertain"},
                "dense_lora_caption": "bad",
            },
        },
    ]
    src = tmp_path / "labels.json.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        json.dump(records, f)
    out = tmp_path / "train.jsonl"

    stats = prepare_labels(str(src), str(out))

    assert stats["output"] == 1
    assert stats["filtered_age_safety"] == 1
    assert json.loads(out.read_text())["data_id"] == "ok"
