import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    name = "build_disjoint_hf_holdout"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "build_disjoint_hf_holdout.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


holdout = _load_script()


def _row(label: str, *, tools=None, **extra) -> dict:
    row = {
        "messages": [
            {"role": "user", "content": f"question {label}"},
            {"role": "assistant", "content": f"answer {label}"},
        ],
        **extra,
    }
    if tools is not None:
        row["tools"] = tools
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_scans_source_order_and_excludes_canonical_examples(tmp_path):
    source = _write_jsonl(
        tmp_path / "source.jsonl",
        [
            _row("a", source_metadata=0),
            _row("b", source_metadata=1),
            _row("b", source_metadata=2),  # duplicate of the selected index 1
            _row("c", tools=[], source_metadata=3),
            _row(
                "d",
                tools=[{"type": "function", "function": {"name": "lookup"}}],
                source_metadata=4,
            ),
            _row("e", source_metadata=5),
        ],
    )
    exclude_a = _write_jsonl(
        tmp_path / "exclude-a.jsonl",
        [
            _row("a", unrelated="ignored"),
            _row("a", tools=None, another="duplicate exclusion"),
        ],
    )
    exclude_c = _write_jsonl(
        tmp_path / "exclude-c.jsonl",
        [_row("c", unrelated="also ignored")],
    )
    out = tmp_path / "out" / "holdout.jsonl"
    manifest_path = tmp_path / "out" / "holdout.manifest.json"

    assert holdout.main(
        [
            "--data",
            str(source),
            "--exclude-jsonl",
            str(exclude_a),
            "--exclude-jsonl",
            str(exclude_c),
            "--search-start",
            "0",
            "--rows",
            "3",
            "--out-jsonl",
            str(out),
            "--manifest-out",
            str(manifest_path),
        ]
    ) == 0

    selected = _read_jsonl(out)
    assert [row["messages"][0]["content"] for row in selected] == [
        "question b",
        "question d",
        "question e",
    ]
    assert all("source_metadata" not in row for row in selected)
    assert "tools" not in selected[0]
    assert selected[1]["tools"][0]["function"]["name"] == "lookup"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == str(source)
    assert manifest["selected_source_indices"] == [1, 4, 5]
    assert manifest["selected_count"] == 3
    assert manifest["excluded_count"] == 2
    assert manifest["duplicate_count"] == 1
    assert manifest["scanned_count"] == 6
    assert manifest["overlap_count"] == 0
    assert manifest["verified_zero_overlap"] is True
    assert manifest["output_sha256"] == _sha256(out)
    assert manifest["exclusion_paths"] == [
        str(exclude_a.resolve()),
        str(exclude_c.resolve()),
    ]
    assert manifest["exclusion_hashes"] == [_sha256(exclude_a), _sha256(exclude_c)]
    assert manifest["exclusions"][0]["row_count"] == 2
    assert manifest["exclusions"][0]["unique_canonical_count"] == 1
    assert manifest["exclusions"][0]["duplicate_canonical_count"] == 1
    assert len(manifest["exclusions"][0]["canonical_sha256"]) == 64
    assert manifest["selected_canonical_hashes"] == [
        holdout.canonical_hash(row) for row in selected
    ]

    selected_hashes = {holdout.canonical_hash(row) for row in selected}
    excluded_hashes = {
        holdout.canonical_hash(holdout.canonical_example(row))
        for path in (exclude_a, exclude_c)
        for row in _read_jsonl(path)
    }
    assert selected_hashes.isdisjoint(excluded_hashes)


def test_output_is_deterministic_and_search_start_is_inclusive(tmp_path):
    source = _write_jsonl(
        tmp_path / "source.jsonl",
        [_row("a"), _row("b"), _row("c"), _row("d")],
    )
    exclusion = _write_jsonl(tmp_path / "exclude.jsonl", [_row("b")])

    manifests = []
    outputs = []
    for suffix in ("one", "two"):
        out = tmp_path / suffix / "holdout.jsonl"
        manifest_path = tmp_path / suffix / "manifest.json"
        manifests.append(
            holdout.build_holdout(
                data=str(source),
                exclude_jsonl=[exclusion],
                search_start=1,
                rows=2,
                out_jsonl=out,
                manifest_out=manifest_path,
            )
        )
        outputs.append(out.read_bytes())

    assert outputs[0] == outputs[1]
    assert manifests[0]["output_sha256"] == manifests[1]["output_sha256"]
    assert manifests[0]["selected_source_indices"] == [2, 3]
    assert manifests[1]["selected_source_indices"] == [2, 3]
    assert manifests[0]["excluded_count"] == 1
    assert manifests[0]["duplicate_count"] == 0


def test_hf_dataset_id_uses_repository_loader(monkeypatch, tmp_path):
    calls = []

    class FakeDataset(list):
        _fingerprint = "capybara-fixture-fingerprint"

    def fake_load_rows(data):
        calls.append(data)
        return FakeDataset([_row("a"), _row("b"), _row("c")])

    monkeypatch.setattr(holdout, "load_rows", fake_load_rows)
    out = tmp_path / "holdout.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = holdout.build_holdout(
        data="trl-lib/Capybara",
        exclude_jsonl=[],
        search_start=1,
        rows=2,
        out_jsonl=out,
        manifest_out=manifest_path,
    )

    assert calls == ["trl-lib/Capybara"]
    assert manifest["source"] == "trl-lib/Capybara"
    assert manifest["source_fingerprint"] == "capybara-fixture-fingerprint"
    assert manifest["selected_source_indices"] == [1, 2]
    assert len(_read_jsonl(out)) == 2


def test_insufficient_unique_rows_fails_without_outputs(tmp_path):
    source = _write_jsonl(
        tmp_path / "source.jsonl",
        [_row("a"), _row("a"), _row("b")],
    )
    exclusion = _write_jsonl(tmp_path / "exclude.jsonl", [_row("b")])
    out = tmp_path / "holdout.jsonl"
    manifest = tmp_path / "manifest.json"

    with pytest.raises(holdout.HoldoutError, match="selected 1 of 2"):
        holdout.build_holdout(
            data=str(source),
            exclude_jsonl=[exclusion],
            search_start=0,
            rows=2,
            out_jsonl=out,
            manifest_out=manifest,
        )
    assert not out.exists()
    assert not manifest.exists()


@pytest.mark.parametrize(
    ("search_start", "rows", "message"),
    [(-1, 1, "search_start"), (0, 0, "rows must be > 0"), (9, 1, "outside source")],
)
def test_invalid_selection_bounds_are_rejected(
    tmp_path, search_start, rows, message
):
    source = _write_jsonl(tmp_path / "source.jsonl", [_row("a")])
    with pytest.raises(holdout.HoldoutError, match=message):
        holdout.build_holdout(
            data=str(source),
            exclude_jsonl=[],
            search_start=search_start,
            rows=rows,
            out_jsonl=tmp_path / "out.jsonl",
            manifest_out=tmp_path / "manifest.json",
        )


def test_canonicalization_requires_conversation_rows():
    with pytest.raises(holdout.HoldoutError, match="non-empty messages"):
        holdout.canonical_example({"text": "not a chat row"})
