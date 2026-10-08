#!/usr/bin/env python
"""rl-fn-codex-rollout task 0.3: render one multi-turn conversation with the
Flash-Next HF ``chat_template.jinja`` (de4b8e4d) and the Miles fork fixed
template ``qwen3.8_small_and_flash_next_fixed.jinja`` and diff the token ids.

Offline, no Ray.  Needs only ``transformers`` (its Jinja renderer is the same
code path SGLang and the fork's session server use) and the FN tokenizer files
(tokenizer.json / tokenizer_config.json from the same HF snapshot).

    OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python tools/probes/fn_template_diff.py \
        --out /tmp/fn_template_diff.json

Defaults locate both templates and the tokenizer on this machine; pass
``--hf-template``/``--fork-template``/``--tokenizer`` to override.  With
``--download`` the single HF files are fetched with ``huggingface_hub`` when
the local snapshot is missing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any

HF_REPO = "Qwen/Qwen3.8-Flash-Next"
HF_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
FORK_TEMPLATE_GLOB = (
    "/home/michael/work/miles-critic-c357-tied/miles/utils/chat_template_utils/"
    "templates/qwen3.8_small_and_flash_next_fixed.jinja"
)

# Profile kwargs under test (design D1 / codex_backend.py "qwen38").
PROFILE_KWARGS = {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "xhigh"}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}, "timeout": {"type": "integer"}},
                "required": ["cmd"],
            },
        },
    }
]

MESSAGES = [
    {"role": "system", "content": "You are a terminal agent. Reply with one command."},
    {"role": "user", "content": "List the repo root."},
    {
        "role": "assistant",
        "reasoning_content": "I should look at the files first.\n\nls is enough.",
        "content": "Listing the root.",
        "tool_calls": [
            {"type": "function", "function": {"name": "bash", "arguments": {"cmd": "ls -la /repo", "timeout": 30}}}
        ],
    },
    {"role": "tool", "content": "total 0\ndrwxr-xr-x 2 root root 40 README.md"},
    {
        "role": "assistant",
        "reasoning_content": "Only a README. Task done.",
        "content": "TASK_COMPLETE",
    },
    {"role": "user", "content": "Double check please."},
]

# Only tool_response "user" turns (no real user query): HF raises, fork does not.
MESSAGES_TOOL_ONLY = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "<tool_response>\nok\n</tool_response>"},
]

SCENARIOS: dict[str, dict[str, Any]] = {
    "profile(xhigh,preserve,think)": dict(PROFILE_KWARGS),
    "D1-alt(enable_thinking only)": {"enable_thinking": True},
    "no kwargs": {},
    "effort=low": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "low"},
    "effort=medium": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "medium"},
    "preserve_thinking=False": {"enable_thinking": True, "preserve_thinking": False, "reasoning_effort": "xhigh"},
    "enable_thinking=False": {"enable_thinking": False, "preserve_thinking": True, "reasoning_effort": "xhigh"},
    "enable_thinking=False,no effort": {"enable_thinking": False},
}


def _find_snapshot() -> str | None:
    pats = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/{HF_REVISION}"))
    return pats[0] if pats else None


def _download(files: list[str]) -> str:
    from huggingface_hub import hf_hub_download

    out = None
    for f in files:
        p = hf_hub_download(HF_REPO, f, revision=HF_REVISION, cache_dir="/tmp/fn-hf-cache")
        out = os.path.dirname(p)
    assert out
    return out


def render(tok, template: str, messages, *, add_generation_prompt: bool, tools=None, **kwargs) -> str:
    return tok.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        **kwargs,
    )


def first_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def comparator_cases(tok, fork_template_path: str) -> dict[str, Any]:
    import importlib.util

    cmp_path = os.path.join(os.path.dirname(os.path.dirname(fork_template_path)), "token_seq_comparator.py")
    spec = importlib.util.spec_from_file_location("token_seq_comparator", cmp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["token_seq_comparator"] = mod
    spec.loader.exec_module(mod)
    comparator = mod.TokenSeqComparator(
        tok,
        assistant_start_str="<|im_start|>assistant",
        trim_trailing_ids={tok.encode("\n", add_special_tokens=False)[0]},
    )
    head = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
    enc = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731
    expected = enc(head + "plan\n</think>\n\n```bash\nls\n```<|im_end|>\n")
    cases = {
        "exact": enc(head + "plan\n</think>\n\n```bash\nls\n```<|im_end|>\n"),
        "(a) reasoning trailing blank line before </think>": enc(head + "plan\n\n</think>\n\n```bash\nls\n```<|im_end|>\n"),
        "(b) content trailing newline before <|im_end|>": enc(head + "plan\n</think>\n\n```bash\nls\n```\n<|im_end|>\n"),
        "(b) single newline after </think>": enc(head + "plan\n</think>\n```bash\nls\n```<|im_end|>\n"),
        "(d) truncated: no <|im_end|>": enc(head + "plan\n</think>\n\n```bash\nls"),
        "bpe drift: same text, '\\n\\n' as two ids": (
            enc(head + "plan\n</think>") + enc("\n") + enc("\n") + enc("```bash\nls\n```<|im_end|>\n")
        ),
    }
    out = {}
    for name, actual in cases.items():
        ms = comparator.compare_sequences(expected, actual)
        out[name] = [m.type.value for m in ms] or ["<no mismatch>"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-template")
    ap.add_argument("--fork-template", default=FORK_TEMPLATE_GLOB)
    ap.add_argument("--tokenizer")
    ap.add_argument("--download", action="store_true", help="fetch single files from HF when snapshot missing")
    ap.add_argument("--out", default="/tmp/fn_template_diff.json")
    args = ap.parse_args()

    snap = _find_snapshot()
    if snap is None and args.download:
        snap = _download(["chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"])
    if snap is None and (args.hf_template is None or args.tokenizer is None):
        print("no FN snapshot in HF cache; pass --download or --hf-template/--tokenizer", file=sys.stderr)
        return 2
    hf_path = args.hf_template or os.path.join(snap, "chat_template.jinja")
    tok_path = args.tokenizer or snap

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_path)
    hf_tpl = open(hf_path).read()
    fork_tpl = open(args.fork_template).read()

    report: dict[str, Any] = {
        "hf_template": hf_path,
        "fork_template": args.fork_template,
        "tokenizer": tok_path,
        "tokenizer_class": type(tok).__name__,
        "template_text_identical_modulo_user_query_guard": hf_tpl.replace(
            "{%- if ns.multi_step_tool %}\n    {{- raise_exception('No user query found in messages.') }}\n{%- endif %}\n",
            "",
        ).rstrip("\n")
        == fork_tpl.rstrip("\n"),
        "think_tokens_special": {
            t: {"id": tok.convert_tokens_to_ids(t), "special": tok.convert_tokens_to_ids(t) in set(tok.all_special_ids)}
            for t in ("<think>", "</think>", "<|im_start|>", "<|im_end|>")
        },
        "scenarios": {},
    }

    for name, kw in SCENARIOS.items():
        for with_tools in (False, True):
            key = f"{name}|tools={with_tools}"
            tools = TOOLS if with_tools else None
            out: dict[str, Any] = {"kwargs": kw, "tools": with_tools}
            try:
                hf_txt = render(tok, hf_tpl, MESSAGES, add_generation_prompt=True, tools=tools, **kw)
                fk_txt = render(tok, fork_tpl, MESSAGES, add_generation_prompt=True, tools=tools, **kw)
                hf_ids = tok.encode(hf_txt, add_special_tokens=False)
                fk_ids = tok.encode(fk_txt, add_special_tokens=False)
                d = first_diff(hf_ids, fk_ids)
                out.update(
                    text_equal=hf_txt == fk_txt,
                    tokens_equal=hf_ids == fk_ids,
                    n_tokens=len(fk_ids),
                    first_diff_index=d,
                    reasoning_effort_in_prompt=("Reasoning effort is set to" in fk_txt),
                    effort_line=next((ln for ln in fk_txt.splitlines() if ln.startswith("Reasoning effort")), None),
                    first_assistant_has_think=("<|im_start|>assistant\n<think>\nI should look" in fk_txt),
                    second_assistant_has_think=("<|im_start|>assistant\n<think>\nOnly a README" in fk_txt),
                    gen_prompt_tail=fk_txt[-40:],
                )
                # append-only invariant (fork template): prefix(N, no gen) is prefix of full(with gen)
                pre = render(tok, fork_tpl, MESSAGES[:5], add_generation_prompt=False, tools=tools, **kw)
                out["fork_append_only_ok"] = fk_txt.startswith(pre)
                if d is not None:
                    out["diff_context"] = {
                        "hf": tok.decode(hf_ids[max(0, d - 5): d + 10]),
                        "fork": tok.decode(fk_ids[max(0, d - 5): d + 10]),
                    }
            except Exception as e:  # template raise_exception etc.
                out["error"] = f"{type(e).__name__}: {e}"[:300]
            report["scenarios"][key] = out

    # Edge: tool_response-only history.
    edge: dict[str, Any] = {}
    for label, tpl in (("hf", hf_tpl), ("fork", fork_tpl)):
        try:
            render(tok, tpl, MESSAGES_TOOL_ONLY, add_generation_prompt=True, **PROFILE_KWARGS)
            edge[label] = "renders"
        except Exception as e:
            edge[label] = f"raises: {str(e)[:120]}"
    report["tool_response_only_history"] = edge

    # Illustrative BPE-drift check for D2 (a)/(b): canonical re-tokenization of
    # an assistant turn vs. piecewise ids (think, boundary, content tokenized
    # separately the way generated tokens + template boundary are inherited).
    think, content = "Only a README. Task done.", "TASK_COMPLETE"
    canonical = tok.encode(f"<|im_start|>assistant\n<think>\n{think}\n</think>\n\n{content}<|im_end|>\n", add_special_tokens=False)
    piecewise = (
        tok.encode("<|im_start|>assistant\n<think>\n", add_special_tokens=False)
        + tok.encode(think, add_special_tokens=False)
        + tok.encode("\n</think>", add_special_tokens=False)
        + tok.encode("\n\n", add_special_tokens=False)
        + tok.encode(content, add_special_tokens=False)
        + tok.encode("<|im_end|>\n", add_special_tokens=False)
    )
    report["bpe_drift_demo"] = {
        "canonical_ids": canonical,
        "piecewise_ids": piecewise,
        "equal": canonical == piecewise,
        "decode_equal": tok.decode(canonical) == tok.decode(piecewise),
        "newline_ids": {"\\n": tok.encode("\n", add_special_tokens=False), "\\n\\n": tok.encode("\n\n", add_special_tokens=False)},
    }

    # D2 (a)/(b)/(d) reproduction through the fork's own TokenSeqComparator
    # (pure-python module, loaded by path so no ``miles`` package import).
    report["comparator_cases"] = comparator_cases(tok, args.fork_template)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    print(f"templates identical modulo user-query guard: {report['template_text_identical_modulo_user_query_guard']}")
    print("think tokens:", report["think_tokens_special"])
    for k, v in report["scenarios"].items():
        if "error" in v:
            print(f"{k:50s} ERROR {v['error'][:80]}")
        else:
            print(
                f"{k:50s} tok_eq={v['tokens_equal']} n={v['n_tokens']} effort_in_prompt={v['reasoning_effort_in_prompt']} "
                f"think1={v['first_assistant_has_think']} think2={v['second_assistant_has_think']} append_ok={v['fork_append_only_ok']}"
            )
    print("tool_response-only history:", edge)
    print("bpe drift demo equal:", report["bpe_drift_demo"]["equal"], "decode_equal:", report["bpe_drift_demo"]["decode_equal"])
    print("comparator cases:")
    for k, v in report["comparator_cases"].items():
        print(f"  {k:55s} -> {v}")
    print("report:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
