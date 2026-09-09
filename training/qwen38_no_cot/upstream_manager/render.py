"""Pinned Qwen3.8 non-thinking SFT rendering with explicit token-loss spans.

No model weights, GPU, network, or remote Python code is used here. Callers own
source normalization and any role adaptations. This renderer accepts canonical
HF text/tool messages and fails closed on unsupported or ambiguous shapes.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from transformers import AutoTokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template, _render_with_assistant_indices

VERSION = "qwen3.8-nonthinking-assistant-body-loss/v1"
REPOSITORY = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
TOKENIZER_CONFIG_SHA256 = "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
OFFICIAL_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
ESCAPE_VERSION = "reserved-source-leading-angle-unicode-escape-with-edits/v1"
EMPTY_THINK = "<think>\n\n</think>\n\n"
EOT = "<|im_end|>"
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]*$")
_PRIVATE_KEYS = {"reasoning", "reasoning_content", "reasoning_details", "thinking", "redacted_thinking",
                 "chain_of_thought", "cot", "analysis", "signature", "signature_delta", "ciphertext"}
_PRIVATE_BLOCKS = _PRIVATE_KEYS | {"reasoning_text", "reasoning_summary", "summary_text", "encrypted_content",
                                   "encrypted_reasoning"}
_LEADING_REASONING = re.compile(r"\A\s*<(think|thinking|analysis|reasoning)\b[^>]*>.*?(?:</\1\s*>|\Z)", re.I | re.S)


class UnsupportedMessage(ValueError):
    pass


class UnsafeTokenBoundary(ValueError):
    pass


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _private_key(key):
    key = key.lower()
    return key in _PRIVATE_KEYS or key.startswith("encrypted_") or key.endswith("_encrypted")


def _json_safe(value):
    if value is None or type(value) in {str, bool, int}:
        return True
    if type(value) is float:
        return value == value and abs(value) != float("inf")
    if isinstance(value, list):
        return all(_json_safe(child) for child in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_safe(child) for key, child in value.items())
    return False


def adapt_custom_tool(name, raw_input, *, description):
    """Explicit raw-tool adapter; caller must retain this schema and provenance.

    It preserves the raw input string as the value of an explicitly declared
    ``input: string`` parameter. It never parses/guesses a custom tool language.
    """
    if not isinstance(name, str) or not _NAME.fullmatch(name) or not isinstance(raw_input, str):
        raise UnsupportedMessage("Custom tools require a safe function name and exact string input")
    if not isinstance(description, str):
        raise UnsupportedMessage("Custom tools require an explicit schema description")
    schema = {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": {"input": {"type": "string"}},
                       "required": ["input"], "additionalProperties": False}}}
    call = {"type": "function", "function": {"name": name, "arguments": {"input": raw_input}}}
    return {"version": "custom-tool-raw-input-string/v1", "tool": schema, "tool_call": call}


def _replace_once(source, before, after):
    if source.count(before) != 1:
        raise UnsupportedMessage("Pinned template adaptation anchor no longer matches")
    return source.replace(before, after, 1)


def _templates(official):
    later_system = _replace_once(official,
        "{{- raise_exception('System message must be at the beginning.') }}",
        "{{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}")
    before = """        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}"""
    after = """        {{- '<|im_start|>' + message.role + '\\n<think>\\n\\n</think>\\n\\n' }}
        {%- generation %}
        {{- content }}"""
    instrumented = _replace_once(later_system, before, after)
    instrumented = _replace_once(instrumented,
        "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}",
        "        {{- '<|im_end|>' }}\n        {%- endgeneration %}\n        {{- '\\n' }}\n    {%- elif message.role == \"tool\" %}")
    return later_system, instrumented


def align_loss_mask(offsets, spans):
    """Map trusted renderer character spans to full-string tokenizer offsets."""
    previous = 0
    for start, end in spans:
        if type(start) is not int or type(end) is not int or start < previous or end <= start:
            raise UnsafeTokenBoundary("Invalid or overlapping assistant character spans")
        previous = end
    mask = []
    cursor = 0
    for start, end in offsets:
        if start < 0 or end <= start:
            raise UnsafeTokenBoundary("A token has no usable nonempty source offset")
        while cursor < len(spans) and spans[cursor][1] <= start:
            cursor += 1
        if cursor == len(spans) or end <= spans[cursor][0]:
            mask.append(0)
        elif start >= spans[cursor][0] and end <= spans[cursor][1]:
            mask.append(1)
        else:
            raise UnsafeTokenBoundary("A token crosses a supervised/unsupervised character boundary")
    return mask


class Qwen38Renderer:
    def __init__(self, tokenizer_dir, *, max_tokens=262144):
        self.directory = Path(tokenizer_dir)
        pins = {"tokenizer.json": TOKENIZER_SHA256, "tokenizer_config.json": TOKENIZER_CONFIG_SHA256,
                "chat_template.jinja": OFFICIAL_TEMPLATE_SHA256}
        for filename, expected in pins.items():
            if _sha((self.directory / filename).read_bytes()) != expected:
                raise UnsupportedMessage("Tokenizer/template file does not match the pinned Qwen revision")
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise ValueError("max_tokens must be positive or None for explicit token-audit-only use")
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.directory, local_files_only=True, trust_remote_code=False)
        if not self.tokenizer.is_fast:
            raise UnsupportedMessage("Exact loss spans require the pinned fast tokenizer")
        official = (self.directory / "chat_template.jinja").read_text()
        plain, instrumented = _templates(official)
        self._plain = _compile_jinja_template(plain)
        self._instrumented = _compile_jinja_template(instrumented)
        controls = set(self.tokenizer.get_added_vocab()) | {"<function=", "</function>", "<parameter=", "</parameter>"}
        self._controls = re.compile("|".join(re.escape(value) for value in sorted(controls, key=len, reverse=True)))
        self.identity = {"version": VERSION, "repository": REPOSITORY, "revision": REVISION,
            "tokenizer_sha256": TOKENIZER_SHA256, "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "official_template_sha256": OFFICIAL_TEMPLATE_SHA256,
            "adapted_template_sha256": _sha(plain.encode()), "loss_template_sha256": _sha(instrumented.encode()),
            "renderer_sha256": _sha(Path(__file__).read_bytes()), "escape_version": ESCAPE_VERSION,
            "adaptations": ["later_system_messages_preserve_chronology", "assistant_body_and_eot_only_generation_spans"],
            "enable_thinking": False, "preserve_thinking": True, "add_generation_prompt": False,
            "loss_0": ["system", "user", "tool_observations", "assistant_header", "empty_think_wrapper", "post_eot_newline"],
            "loss_1": ["assistant_visible_content", "assistant_tool_calls", "assistant_eot"]}

    def _escape(self, value, path, edits):
        if isinstance(value, str):
            parts, cursor, length = [], 0, 0
            for match in self._controls.finditer(value):
                before = value[cursor:match.start()]
                parts.append(before)
                length += len(before)
                # Literal six-character escape, not a tokenizer control token.
                replacement = "\\u003c" + match.group()[1:]
                edits.append({"path": path, "escaped_offset": length, "escaped_length": 6, "original": "<"})
                parts.append(replacement)
                length += len(replacement)
                cursor = match.end()
            parts.append(value[cursor:])
            return "".join(parts)
        if isinstance(value, list):
            return [self._escape(child, f"{path}/{i}", edits) for i, child in enumerate(value)]
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                escaped_key = self._escape(key, path + "/<key>", edits)
                if escaped_key in result:
                    raise UnsupportedMessage("Control escaping would collide object keys")
                result[escaped_key] = self._escape(child, path + "/" + key.replace("~", "~0").replace("/", "~1"), edits)
            return result
        return value

    def _content(self, content, role, counts):
        if content is None:
            return ""
        if isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    raise UnsupportedMessage("Canonical content blocks must be typed text objects")
                if str(block.get("type", "")).lower() in _PRIVATE_BLOCKS:
                    counts["reasoning_blocks_removed"] += 1
                    continue
                if block.get("type") != "text" or set(block) != {"type", "text"} or not isinstance(block["text"], str):
                    raise UnsupportedMessage("Only canonical text blocks are supported; real media needs a processor")
                parts.append(block["text"])
            content = "".join(parts)
        if not isinstance(content, str):
            raise UnsupportedMessage("Canonical message content must be text or text blocks")
        if role == "assistant":
            # Explicit leading reasoning blocks are source reasoning, not targets.
            while (cleaned := _LEADING_REASONING.sub("", content, count=1)) != content:
                content = cleaned
                counts["leading_reasoning_blocks_removed"] += 1
        return content

    def _normalize(self, messages, tools):
        if not isinstance(messages, list) or not messages:
            raise UnsupportedMessage("A nonempty canonical message list is required")
        if tools is None:
            tools = []
        if not isinstance(tools, list) or not _json_safe(tools):
            raise UnsupportedMessage("Tool schemas must be JSON-safe function-schema objects")
        counts = Counter()
        cleaned, edits = [], []
        for i, original in enumerate(messages):
            if not isinstance(original, dict):
                raise UnsupportedMessage("Each canonical message must be an object")
            item = {key: deepcopy(value) for key, value in original.items() if not _private_key(key)}
            counts["reasoning_or_encrypted_fields_removed"] += len(original) - len(item)
            if item.get("role") not in {"system", "user", "assistant", "tool"}:
                raise UnsupportedMessage("Unsupported source role; explicit caller-side role adaptation is required")
            if set(item) - {"role", "content", "tool_calls", "tool_call_id", "name"}:
                raise UnsupportedMessage("Unsupported canonical message fields")
            role = item["role"]
            item["content"] = self._content(item.get("content"), role, counts)
            if "tool_calls" in item:
                if role != "assistant" or not isinstance(item["tool_calls"], list):
                    raise UnsupportedMessage("Only assistant messages can contain canonical tool calls")
                for call in item["tool_calls"]:
                    if (not isinstance(call, dict) or call.get("type") != "function"
                            or set(call) - {"id", "type", "function"}):
                        raise UnsupportedMessage("Custom tool calls require the explicit raw-input schema adapter")
                    fn = call.get("function")
                    if (not isinstance(fn, dict) or set(fn) != {"name", "arguments"}
                            or not isinstance(fn.get("name"), str) or not _NAME.fullmatch(fn["name"])
                            or not isinstance(fn.get("arguments"), dict) or not _json_safe(fn["arguments"])):
                        raise UnsupportedMessage("Tool calls require a safe name and JSON-object arguments, not serialized JSON")
                    if any(not _NAME.fullmatch(key) for key in fn["arguments"]):
                        raise UnsupportedMessage("Tool parameter names cannot be represented safely in the official XML form")
            item = self._escape(item, f"/messages/{i}", edits)
            cleaned.append(item)
        for tool in tools:
            if (not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict)
                    or not isinstance(tool["function"].get("name"), str)
                    or not _NAME.fullmatch(tool["function"]["name"])
                    or not isinstance(tool["function"].get("parameters"), dict)):
                raise UnsupportedMessage("Canonical tools require declared function names and parameter schemas")
        safe_tools = self._escape(deepcopy(tools), "/tools", edits)
        return cleaned, safe_tools, dict(counts), edits

    def render(self, messages, tools=None):
        """Return unshifted HF labels: target token IDs or -100 for ignored loss.

        Transformers performs its own causal shift. The returned ``loss_mask``
        is binary metadata, not the labels themselves. No content is truncated.
        Escaping edits refer to normalized canonical values before template trim.
        """
        clean, safe_tools, counts, edits = self._normalize(messages, tools)
        kwargs = {"enable_thinking": False, "preserve_thinking": True, "add_vision_id": False}
        text, spans = _render_with_assistant_indices(
            self._instrumented, clean, safe_tools or None, None, False, **kwargs)
        plain = self._plain.render(messages=clean, tools=safe_tools or None, documents=None,
                                    add_generation_prompt=False, **kwargs)
        if text != plain:
            raise UnsupportedMessage("Loss instrumentation changed official-compatible rendered bytes")
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                                 padding=False, truncation=False)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if self.max_tokens is not None and len(ids) > self.max_tokens:
            raise UnsupportedMessage("Complete trace exceeds configured context; no source text was truncated")
        mask = align_loss_mask(offsets, spans)
        if len(spans) != sum(message["role"] == "assistant" for message in clean):
            raise UnsafeTokenBoundary("Assistant generation-span coverage mismatch")
        for start, end in spans:
            if text[end - len(EOT):end] != EOT:
                raise UnsafeTokenBoundary("Assistant span does not terminate at its EOT token")
        return {"input_ids": ids, "attention_mask": [1] * len(ids),
                "labels": [token if active else -100 for token, active in zip(ids, mask)],
                "loss_mask": mask, "rendered_text": text,
                "assistant_spans": [{"start": start, "end": end} for start, end in spans],
                "token_offsets": offsets, "normalization": counts, "control_escape_edits": edits,
                "input_tokens": len(ids), "supervised_tokens": sum(mask), "identity": self.identity}
