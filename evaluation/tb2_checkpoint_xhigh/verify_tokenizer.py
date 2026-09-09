"""CPU-only native-template preflight on the actual served HF export."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def verify(model_dir: Path, *, expected_template_sha256: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from transformers import AutoTokenizer

    template = model_dir / "chat_template.jinja"
    digest = hashlib.sha256(template.read_bytes()).hexdigest()
    if digest != expected_template_sha256:
        raise ValueError("Served export template does not match the pinned training model")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                                               trust_remote_code=False)
    if tokenizer.chat_template != template.read_text():
        raise ValueError("AutoTokenizer selected a different chat template")
    tools = [{"type": "function", "function": {
        "name": "terminal", "description": "Run a terminal command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    }}]
    conversation = [{"role": "system", "content": "Complete the task."},
                    {"role": "user", "content": "List the current directory."}]
    rounds = [conversation, conversation + [
        {"role": "assistant", "content": "", "reasoning_content": "I should inspect the directory.",
         "tool_calls": [{"id": "call1", "type": "function", "function": {
             "name": "terminal", "arguments": {"command": "pwd"}}}]},
        {"role": "tool", "tool_call_id": "call1", "content": "/workspace"},
    ]]
    receipts = []
    for messages in rounds:
        kwargs = dict(tools=tools, add_generation_prompt=True, tokenize=False, preserve_thinking=True)
        actual = tokenizer.apply_chat_template(messages, **kwargs)
        explicit = tokenizer.apply_chat_template(messages, **kwargs,
                                                  enable_thinking=True, reasoning_effort="xhigh")
        if actual != explicit or "Reasoning effort is set to xhigh." not in actual:
            raise ValueError("Default serving path does not render explicit native xhigh")
        if not actual.endswith("<|im_start|>assistant\n<think>\n"):
            raise ValueError("Generation prompt does not start an open thinking block")
        ids = tokenizer.encode(actual, add_special_tokens=False)
        if ids != tokenizer.encode(explicit, add_special_tokens=False):
            raise ValueError("Explicit and default rendered token IDs differ")
        receipts.append({"events": len(messages), "input_tokens": len(ids),
                         "render_sha256": hashlib.sha256(actual.encode()).hexdigest()})
    return {"schema": "yeta.tb2-native-xhigh-preflight/v1", "status": "passed",
            "model_dir": str(model_dir), "chat_template_sha256": digest,
            "tokenizer_json_sha256": hashlib.sha256((model_dir / "tokenizer.json").read_bytes()).hexdigest(),
            "default_equals_explicit_xhigh": True, "generation_thinking_enabled": True,
            "synthetic_cases": receipts, "gpu_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-template-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.model_dir, expected_template_sha256=args.expected_template_sha256),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
