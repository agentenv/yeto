"""Model and data preparation for the bare-framework arms (no yeto import).

Run inside the container as ``python eff_prep.py <spec json>``. It downloads
the pinned model snapshot and the pinned dataset revision, then writes the
training prompts in the dataset's own row order:

* ``/root/eff-data/train.parquet`` and ``test.parquet`` -- verl rows, the same
  shape as ``yeto.rl.adapters.verl.island_entry.prepare_data`` (the yeto+verl
  arm's data), so bare verl and yeto+verl read identical rows;
* ``/root/eff-data/train.jsonl`` -- Miles rows (``prompt`` = chat messages,
  ``label``), for ``--input-key prompt --label-key label --apply-chat-template``;
* ``/root/eff-data/order.json`` -- sha256 of the label sequence (row order) and
  the row count, so a report can show that two arms read the same rows.

Shuffling is left to each framework (seeded by the spec's ``seed``), as in the
yeto arm of the same framework.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

DATA_ROOT = Path("/root/eff-data")
MODEL_ROOT = Path("/root/eff-models")
VAL_ROWS = 64


def source_rows(spec: dict, download) -> dict[str, list[tuple[list, str]]]:
    """Split -> ``(chat messages, label)`` in the dataset's own row order.

    Two layouts: ``train.parquet``/``test.parquet`` with ``messages`` and ``label``
    (zhuzilin/gsm8k), or one JSONL file ``spec["data_file"]`` with ``prompt`` (chat
    messages) and ``label`` (zhuzilin/dapo-math-17k), whose test rows are the
    ``VAL_ROWS`` rows after the training rows. ``spec["train_rows"]`` (0: all) keeps
    the first rows only, so every arm trains on the same prompt set in one epoch.
    """
    import pandas as pd

    n = int(spec.get("train_rows") or 0)
    if spec.get("data_file"):
        path = download(spec["data"], spec["data_file"], repo_type="dataset", revision=spec["data_revision"])
        with open(path) as fh:
            records = [(r["prompt"], str(r["label"])) for r in map(json.loads, fh)]
        if not n:
            raise ValueError("a JSONL data_file needs train_rows (its test rows follow them)")
        return {"train": records[:n], "test": records[n:n + VAL_ROWS]}
    out = {}
    for split in ("train", "test"):
        frame = pd.read_parquet(download(spec["data"], f"{split}.parquet", repo_type="dataset",
                                         revision=spec["data_revision"]))
        records = [(list(row["messages"]), str(row["label"])) for _, row in frame.iterrows()]
        out[split] = records[:n] if (split == "train" and n) else records[:VAL_ROWS] if split == "test" else records
    return out


def yeto_task_rows(spec: dict, download) -> list[dict]:
    """The training rows as a yeto local task file (``messages``, ``label``, task id
    ``gsm<row+1>``): the yeto arms read the same rows as the bare arms' ``train.jsonl``."""
    return [{"messages": [{"role": m["role"], "content": m["content"]} for m in messages], "label": label,
             "metadata": {"task_id": f"gsm{i + 1}"}}
            for i, (messages, label) in enumerate(source_rows(spec, download)["train"])]


def main(spec_path: str) -> int:
    import pandas as pd
    from huggingface_hub import hf_hub_download, snapshot_download

    spec = json.loads(Path(spec_path).read_text())
    model_dir = MODEL_ROOT / spec["model"].split("/")[-1]
    snapshot_download(spec["model"], revision=spec["model_revision"], local_dir=str(model_dir))
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    order = {}
    for split, records in source_rows(spec, hf_hub_download).items():
        rows = [{
            "data_source": "yeto/" + spec["data"],
            "prompt": [{"role": m["role"], "content": m["content"]} for m in messages],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(label)},
            "extra_info": {"split": split, "index": i},
        } for i, (messages, label) in enumerate(records)]
        pd.DataFrame(rows).to_parquet(DATA_ROOT / f"{split}.parquet")
        labels = [r["reward_model"]["ground_truth"] for r in rows]
        order[split] = {"rows": len(rows), "label_sha256": hashlib.sha256("\n".join(labels).encode()).hexdigest()}
        if split == "train":
            with (DATA_ROOT / "train.jsonl").open("w") as fh:
                for r in rows:
                    fh.write(json.dumps({"prompt": r["prompt"], "label": r["reward_model"]["ground_truth"]}) + "\n")
    (DATA_ROOT / "order.json").write_text(json.dumps({**order, "model_dir": str(model_dir)}, indent=1))
    print(json.dumps({"model_dir": str(model_dir), **order}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
