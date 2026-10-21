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


def main(spec_path: str) -> int:
    import pandas as pd
    from huggingface_hub import hf_hub_download, snapshot_download

    spec = json.loads(Path(spec_path).read_text())
    model_dir = MODEL_ROOT / spec["model"].split("/")[-1]
    snapshot_download(spec["model"], revision=spec["model_revision"], local_dir=str(model_dir))
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    order = {}
    for split in ("train", "test"):
        src = hf_hub_download(spec["data"], f"{split}.parquet", repo_type="dataset", revision=spec["data_revision"])
        frame = pd.read_parquet(src)
        rows = [{
            "data_source": "yeto/" + spec["data"],
            "prompt": [{"role": m["role"], "content": m["content"]} for m in row["messages"]],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(row["label"])},
            "extra_info": {"split": split, "index": int(i)},
        } for i, row in frame.iterrows()]
        if split == "test":
            rows = rows[:VAL_ROWS]
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
