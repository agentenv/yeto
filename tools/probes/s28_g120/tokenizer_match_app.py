"""S28 G-120B: do the Ascend trainer and the Modal sampler tokenize identically?

Health criterion 4 compares the trainer's log-probs with the sampler's on the
same tokens, so the two sides must produce the same ids for the same text. The
text-120B base carries npu1's patched ``tokenizer.json``, but transformers in
the Modal image still warns about the mistral regex, so equality has to be
measured rather than assumed.

CPU only, no GPU: this loads the tokenizer from the base directory on the
Volume and prints the ids, which are compared against the ids the same script
produced on npu1 (``/tmp/g120b-tokcheck.py``, result in
``/tmp/g120b-tok-npu1.json`` on the dev machine).

    modal run tools/probes/s28_g120/tokenizer_match_app.py
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve()
REPO = HERE.parents[3] if len(HERE.parents) > 3 else Path("/")
WORKDIR = "/root/yeto"
IMAGE = ("ghcr.io/michaellchung/yeto-miles-ports@sha256:"
         "d639e258709b9d4ae4a980c63818562daf62137c9e364f779010bc0b753e867a")
BASE_DIR = "/root/yeto-checkpoint-store/xc120/base/nemotron-120b-text"
CHAT_PROMPT = "What is 17 times 23? Answer with the number only."
PROBE = "Hello World, 7 times 6 is 42.\nSecond line."

app = modal.App(os.environ.get("YETO_G120_APP", "yeto-s28-g120-tokmatch"))
xc = modal.Volume.from_name("yeto-s27-xc", create_if_missing=False)
image = (
    modal.Image.from_registry(IMAGE)
    .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(REPO / "yeto"), f"{WORKDIR}/yeto", copy=False,
                   ignore=["**/__pycache__", "**/*.pyc"])
)


@app.function(image=image, volumes={"/root/yeto-checkpoint-store": xc}, cpu=4.0,
              memory=16384, timeout=900)
def tokenize() -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_DIR, trust_remote_code=True)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": CHAT_PROMPT}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return {
        "vocab_size": len(tok),
        "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "rendered_len": len(rendered),
        "chat_ids": tok(rendered)["input_ids"],
        "probe_ids": tok(PROBE)["input_ids"],
    }


@app.local_entrypoint()
def main(reference: str = "/tmp/g120b-tok-npu1.json") -> None:
    got = tokenize.remote()
    got["chat_ids_n"] = len(got["chat_ids"])
    Path("/tmp/g120b-tok-modal.json").write_text(json.dumps(got, sort_keys=True))
    ref = json.loads(Path(reference).read_text())

    checks = []
    for key in ("vocab_size", "rendered_len", "rendered_sha256", "chat_ids_n"):
        checks.append((key, ref.get(key) == got.get(key), f"npu1={ref.get(key)!r} modal={got.get(key)!r}"))
    for key in ("chat_ids", "probe_ids"):
        same = ref.get(key) == got.get(key)
        note = ""
        if not same:
            a, b = ref.get(key) or [], got.get(key) or []
            diff = [i for i in range(max(len(a), len(b)))
                    if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None)]
            note = f"{len(diff)} position(s) differ, first {diff[:5]}"
        checks.append((key, same, note))

    print("\n=== tokenizer id equality, npu1 trainer vs Modal sampler ===")
    for name, ok, note in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {note}")
    if not all(ok for _, ok, _ in checks):
        raise SystemExit(1)
    print("ALL-EQUAL: both sides produce the same token ids")
