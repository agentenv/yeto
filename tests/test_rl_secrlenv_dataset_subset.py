import hashlib
import json

import pytest

from yeto.rl.secrlenv_dataset_subset import DatasetSubsetError, build_subset

TASK_PACK_SHA256 = "a" * 64


def _row(task_id: str, *, spacing: bool = False) -> bytes:
    if spacing:
        return (
            f'{{ "messages": [], "metadata": {{ "task_id": "{task_id}" }} }}\n'
        ).encode()
    return (
        json.dumps(
            {"messages": [], "metadata": {"task_id": task_id}},
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _parent(tmp_path):
    rows = {
        "task-a": _row("task-a", spacing=True),
        "task-b": _row("task-b"),
        "task-c": _row("task-c", spacing=True),
    }
    parent = tmp_path / "parent.jsonl"
    parent.write_bytes(b"".join(rows.values()))
    return parent, rows


def _parent_attestation(parent):
    value = parent.read_bytes()
    return {
        "expected_parent_sha256": hashlib.sha256(value).hexdigest(),
        "expected_parent_count": len(value.splitlines()),
    }


def test_explicit_order_is_exact_and_manifest_binds_all_inputs(tmp_path):
    parent, rows = _parent(tmp_path)
    ordered_ids = tmp_path / "ordered.ids"
    ordered_id_bytes = b"task-c\ntask-a\n"
    ordered_ids.write_bytes(ordered_id_bytes)
    output = tmp_path / "subset.jsonl"
    manifest_output = tmp_path / "subset.manifest.json"

    manifest = build_subset(
        parent,
        output,
        manifest_output,
        **_parent_attestation(parent),
        expected_count=2,
        task_pack_sha256=TASK_PACK_SHA256,
        selection_rule="externally attested flaky-task list",
        selection_seed=41,
        ordered_task_ids=ordered_ids,
    )

    expected_subset = rows["task-c"] + rows["task-a"]
    assert output.read_bytes() == expected_subset
    assert manifest_output.read_text() == (
        json.dumps(manifest, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
    )
    assert manifest == {
        "schema": "secrlenv-dataset-subset/v1",
        "id_field": "metadata.task_id",
        "parent": {
            "row_count": 3,
            "sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        },
        "subset": {
            "row_count": 2,
            "sha256": hashlib.sha256(expected_subset).hexdigest(),
        },
        "ordered_task_ids_sha256": hashlib.sha256(ordered_id_bytes).hexdigest(),
        "ordered_task_ids_sha256_basis": "ordered_task_id_file_bytes",
        "task_pack_sha256": TASK_PACK_SHA256,
        "selection": {
            "mode": "explicit_ordered_task_ids",
            "rule": "externally attested flaky-task list",
            "seed": 41,
            "expected_count": 2,
        },
    }


def test_first_n_is_explicit_and_reproducible(tmp_path):
    parent, rows = _parent(tmp_path)
    output_a = tmp_path / "a.jsonl"
    manifest_a = tmp_path / "a.manifest.json"
    output_b = tmp_path / "b.jsonl"
    manifest_b = tmp_path / "b.manifest.json"

    first = build_subset(
        parent,
        output_a,
        manifest_a,
        **_parent_attestation(parent),
        expected_count=2,
        task_pack_sha256=TASK_PACK_SHA256,
        selection_rule="parent order first two",
        first_n=True,
    )
    second = build_subset(
        parent,
        output_b,
        manifest_b,
        **_parent_attestation(parent),
        expected_count=2,
        task_pack_sha256=TASK_PACK_SHA256,
        selection_rule="parent order first two",
        first_n=True,
    )

    assert output_a.read_bytes() == rows["task-a"] + rows["task-b"]
    assert output_b.read_bytes() == output_a.read_bytes()
    assert manifest_b.read_bytes() == manifest_a.read_bytes()
    assert first == second
    assert first["selection"] == {
        "mode": "parent_order_first_n",
        "rule": "parent order first two",
        "seed": None,
        "expected_count": 2,
    }
    assert (
        first["ordered_task_ids_sha256"]
        == hashlib.sha256(b"task-a\ntask-b\n").hexdigest()
    )
    assert first["ordered_task_ids_sha256_basis"] == "generated_utf8_lf_bytes"


@pytest.mark.parametrize("first_n", [False, True])
def test_exactly_one_selection_mode_is_required(tmp_path, first_n):
    parent, _ = _parent(tmp_path)
    ids = tmp_path / "ordered.ids"
    ids.write_text("task-a\n")

    with pytest.raises(DatasetSubsetError, match="exactly one"):
        build_subset(
            parent,
            tmp_path / "subset.jsonl",
            tmp_path / "manifest.json",
            **_parent_attestation(parent),
            expected_count=1,
            task_pack_sha256=TASK_PACK_SHA256,
            selection_rule="test",
            ordered_task_ids=ids if first_n else None,
            first_n=first_n,
        )


@pytest.mark.parametrize(
    ("ids", "expected_count", "message"),
    [
        ("task-a\ntask-a\n", 2, "duplicate"),
        ("task-a\ntask-missing\n", 2, "absent from parent"),
        ("task-a\ntask-b\n", 1, "exact expected count"),
    ],
)
def test_explicit_ids_reject_duplicates_missing_and_extras(
    tmp_path, ids, expected_count, message
):
    parent, _ = _parent(tmp_path)
    ordered_ids = tmp_path / "ordered.ids"
    ordered_ids.write_text(ids)
    output = tmp_path / "subset.jsonl"
    manifest = tmp_path / "manifest.json"

    with pytest.raises(DatasetSubsetError, match=message):
        build_subset(
            parent,
            output,
            manifest,
            **_parent_attestation(parent),
            expected_count=expected_count,
            task_pack_sha256=TASK_PACK_SHA256,
            selection_rule="test",
            ordered_task_ids=ordered_ids,
        )

    assert not output.exists()
    assert not manifest.exists()


def test_parent_ids_must_be_unique_and_rows_newline_terminated(tmp_path):
    duplicate_parent = tmp_path / "duplicate.jsonl"
    duplicate_parent.write_bytes(_row("task-a") + _row("task-a"))
    unterminated_parent = tmp_path / "unterminated.jsonl"
    unterminated_parent.write_bytes(_row("task-a").rstrip(b"\n"))

    for parent, message in (
        (duplicate_parent, "duplicate task IDs"),
        (unterminated_parent, "terminate every row"),
    ):
        with pytest.raises(DatasetSubsetError, match=message):
            build_subset(
                parent,
                tmp_path / f"{parent.stem}.subset.jsonl",
                tmp_path / f"{parent.stem}.manifest.json",
                **_parent_attestation(parent),
                expected_count=1,
                task_pack_sha256=TASK_PACK_SHA256,
                selection_rule="test",
                first_n=True,
            )


def test_parent_sha_and_count_must_match_attested_canonical_input(tmp_path):
    parent, _ = _parent(tmp_path)
    contract = _parent_attestation(parent)

    with pytest.raises(DatasetSubsetError, match="SHA-256"):
        build_subset(
            parent,
            tmp_path / "wrong-sha.jsonl",
            tmp_path / "wrong-sha.manifest.json",
            expected_parent_sha256="b" * 64,
            expected_parent_count=contract["expected_parent_count"],
            expected_count=1,
            task_pack_sha256=TASK_PACK_SHA256,
            selection_rule="test",
            first_n=True,
        )
    with pytest.raises(DatasetSubsetError, match="row count"):
        build_subset(
            parent,
            tmp_path / "wrong-count.jsonl",
            tmp_path / "wrong-count.manifest.json",
            expected_parent_sha256=contract["expected_parent_sha256"],
            expected_parent_count=4,
            expected_count=1,
            task_pack_sha256=TASK_PACK_SHA256,
            selection_rule="test",
            first_n=True,
        )


def test_parent_task_ids_reject_control_characters(tmp_path):
    parent = tmp_path / "control.jsonl"
    parent.write_bytes(_row("task-a\u0001"))
    with pytest.raises(DatasetSubsetError, match="valid metadata.task_id"):
        build_subset(
            parent,
            tmp_path / "subset.jsonl",
            tmp_path / "manifest.json",
            **_parent_attestation(parent),
            expected_count=1,
            task_pack_sha256=TASK_PACK_SHA256,
            selection_rule="test",
            first_n=True,
        )


def test_refuses_to_overwrite_outputs_or_accept_unattested_task_pack(tmp_path):
    parent, _ = _parent(tmp_path)
    output = tmp_path / "subset.jsonl"
    manifest = tmp_path / "manifest.json"
    output.write_text("do not replace")

    with pytest.raises(DatasetSubsetError, match="overwrite"):
        build_subset(
            parent,
            output,
            manifest,
            **_parent_attestation(parent),
            expected_count=1,
            task_pack_sha256=TASK_PACK_SHA256,
            selection_rule="test",
            first_n=True,
        )
    assert output.read_text() == "do not replace"
    assert not manifest.exists()

    with pytest.raises(DatasetSubsetError, match="64 lowercase hex"):
        build_subset(
            parent,
            tmp_path / "other.jsonl",
            tmp_path / "other.manifest.json",
            **_parent_attestation(parent),
            expected_count=1,
            task_pack_sha256="not-attested",
            selection_rule="test",
            first_n=True,
        )


def test_first_n_rejects_misleading_seed_metadata(tmp_path):
    parent, _ = _parent(tmp_path)

    with pytest.raises(DatasetSubsetError, match="does not use a seed"):
        build_subset(
            parent,
            tmp_path / "subset.jsonl",
            tmp_path / "manifest.json",
            **_parent_attestation(parent),
            expected_count=1,
            task_pack_sha256=TASK_PACK_SHA256,
            selection_rule="parent order",
            selection_seed=7,
            first_n=True,
        )
