"""Chat template registry and the token alignment check (S25 N10.5/N10.6).

The records proxy (:mod:`yeto.rl.harness.gateway.records`) renders canonical
chat messages (the gateway form, :mod:`.gateway.translate`) with the served
model's own chat template, sends the token ids to the engine and parses the
generated text back into an assistant message. A :class:`TemplateSpec` holds
the model-specific facts for that: tokenizer, template kwargs, the tokens that
end a turn, whether history reasoning is kept, and the output parser.

Registered templates:

* ``glm5``: the GLM XML tool-call format (``<tool_call>name<arg_key>k</arg_key>
  <arg_value>v</arg_value></tool_call>``), ``clear_thinking=False`` so every
  turn keeps its reasoning (the stream stays append-only). The GLM5.3-Flash
  tokenizer is not on this machine: the default tokenizer is the GLM-4.7-Flash
  one (same template family). Load the real checkpoint with
  ``ChatTemplate.load("glm5", tokenizer_path=...)``; whether GLM5.3 uses the
  same template is not verified;
* ``qwen3``: the Qwen3 hermes tool-call format; the small CPU stand-in.

:func:`check_alignment` is the N10.6 check: the template rendering, tokenized
the way the proxy sends it, must equal the tokenizer's own
``apply_chat_template`` ids token by token, and it reports how many tokens an
engine would add if it got the text and tokenized it with special tokens.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]


class TemplateError(ValueError):
    """The template cannot render or parse a turn (fail closed)."""


# ---------------------------------------------------------------------- parsers
def _call_id(text: str, index: int) -> str:
    return f"call_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}_{index}"


def _tool_call(call_id: str, name: str, arguments: Mapping[str, Any]) -> Message:
    # Key order is kept: the template renders the arguments in this order.
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(dict(arguments), ensure_ascii=False)}}


def _assistant(content: str, reasoning: str | None, calls: list[Message]) -> Message:
    message: Message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    return message


def _split_reasoning(text: str, *, opened: bool) -> tuple[str | None, str]:
    """``(reasoning, rest)``. ``opened``: the generation prompt already wrote
    ``<think>``. Without ``</think>`` an opened block is all reasoning."""
    if "</think>" in text:
        head, rest = text.split("</think>", 1)
        return head.split("<think>", 1)[-1].strip("\n"), rest
    if opened:
        return text.strip("\n"), ""
    return None, text


_GLM_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_GLM_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def _glm_value(raw: str) -> Any:
    # The template writes strings raw and other values as JSON (``tojson``).
    try:
        value = json.loads(raw)
    except ValueError:
        return raw
    return raw if isinstance(value, str) else value


def parse_glm(text: str) -> Message:
    reasoning, rest = _split_reasoning(text, opened=True)
    calls = []
    for index, match in enumerate(_GLM_CALL.finditer(rest)):
        body = match.group(1)
        name = body.split("<arg_key>", 1)[0].strip()
        if not name:
            raise TemplateError(f"GLM tool call without a name: {body[:80]!r}")
        calls.append(_tool_call(_call_id(text, index), name,
                                {k.strip(): _glm_value(v) for k, v in _GLM_ARG.findall(body)}))
    content = rest.split("<tool_call>", 1)[0].strip()
    return _assistant(content, reasoning, calls)


_QWEN_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_qwen3(text: str) -> Message:
    reasoning, rest = _split_reasoning(text, opened=False)
    calls = []
    for index, match in enumerate(_QWEN_CALL.finditer(rest)):
        try:
            payload = json.loads(match.group(1))
            name, arguments = payload["name"], payload.get("arguments", {})
        except (ValueError, KeyError, TypeError) as exc:
            raise TemplateError(f"Qwen3 tool call is not {{name, arguments}} JSON: {match.group(1)[:80]!r}") from exc
        if not isinstance(arguments, Mapping):
            raise TemplateError("Qwen3 tool call arguments are not an object")
        calls.append(_tool_call(_call_id(text, index), str(name), arguments))
    content = rest.split("<tool_call>", 1)[0].strip("\n")
    return _assistant(content, reasoning, calls)


# ---------------------------------------------------------------------- registry
@dataclass(frozen=True)
class TemplateSpec:
    """Model-specific template facts. ``stop_tokens``: special tokens that end
    an assistant turn (sent to the engine as stop token ids).
    ``keeps_history_reasoning``: with ``template_kwargs`` the template renders
    the reasoning of every earlier turn (gateway ``ChainRegistry`` input).
    ``verified``: what the token alignment check ran on (free text)."""

    name: str
    tokenizer: str
    parser: Callable[[str], Message]
    stop_tokens: tuple[str, ...]
    template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    keeps_history_reasoning: bool = True
    verified: str = "not verified"


TEMPLATES: dict[str, TemplateSpec] = {}


def register_template(spec: TemplateSpec) -> TemplateSpec:
    if spec.name in TEMPLATES and TEMPLATES[spec.name] != spec:
        raise TemplateError(f"template {spec.name!r} is already registered")
    TEMPLATES[spec.name] = spec
    return spec


register_template(TemplateSpec(
    name="glm5", tokenizer="zai-org/GLM-4.7-Flash", parser=parse_glm,
    stop_tokens=("<|endoftext|>", "<|user|>", "<|observation|>"),
    template_kwargs={"clear_thinking": False},
    verified="GLM-4.7-Flash tokenizer as a stand-in; GLM5.3-Flash tokenizer not verified"))
register_template(TemplateSpec(
    name="qwen3", tokenizer="Qwen/Qwen3-0.6B", parser=parse_qwen3,
    stop_tokens=("<|im_end|>", "<|endoftext|>"),
    template_kwargs={"enable_thinking": True},
    # The Qwen3 template drops the reasoning of turns before the last user turn.
    keeps_history_reasoning=False,
    verified="Qwen3-0.6B tokenizer (CPU stand-in)"))


# ---------------------------------------------------------------------- template
def _render_ready(messages: Sequence[Message]) -> list[Message]:
    """Canonical messages -> template input: tool-call arguments as objects."""
    out = []
    for message in messages:
        message = dict(message)
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = dict(call["function"])
                if isinstance(function.get("arguments"), str):
                    try:
                        function["arguments"] = json.loads(function["arguments"] or "{}")
                    except ValueError as exc:
                        raise TemplateError(f"tool call arguments are not JSON: {function['arguments'][:80]!r}") from exc
                calls.append({**call, "function": function})
            message["tool_calls"] = calls
        out.append(message)
    return out


class ChatTemplate:
    """A :class:`TemplateSpec` bound to a tokenizer (HF ``PreTrainedTokenizer``)."""

    def __init__(self, spec: TemplateSpec, tokenizer: Any) -> None:
        self.spec = spec
        self.tokenizer = tokenizer
        ids = []
        for token in spec.stop_tokens:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) != 1:
                raise TemplateError(f"stop token {token!r} is not one token in this tokenizer")
            ids.append(token_ids[0])
        self.stop_token_ids: tuple[int, ...] = tuple(ids)
        self._stop_by_id = dict(zip(ids, spec.stop_tokens))

    @classmethod
    def load(cls, name: str, *, tokenizer_path: str | None = None, local_files_only: bool = True) -> "ChatTemplate":
        try:
            spec = TEMPLATES[name]
        except KeyError:
            raise TemplateError(f"unknown chat template {name!r} (known: {sorted(TEMPLATES)})") from None
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or spec.tokenizer, local_files_only=local_files_only)
        return cls(spec, tokenizer)

    def render(self, messages: Sequence[Message], *, tools: Sequence[Message] | None = None,
               add_generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            _render_ready(messages), tools=list(tools) if tools else None, tokenize=False,
            add_generation_prompt=add_generation_prompt, **dict(self.spec.template_kwargs))

    def encode(self, text: str) -> list[int]:
        """Token ids of rendered text; the template writes its own BOS tokens."""
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def stop_text(self, token_id: int) -> str | None:
        return self._stop_by_id.get(int(token_id))

    def parse(self, text: str) -> Message:
        return self.spec.parser(text)


# ---------------------------------------------------------------------- N10.6 check
@dataclass(frozen=True)
class AlignmentReport:
    """``aligned``: the proxy's ids equal ``apply_chat_template`` ids.
    ``first_diff``: first differing position (None when aligned).
    ``engine_text_extra``: tokens an engine adds when it tokenizes the rendered
    text with special tokens (a text prompt would carry them twice)."""

    aligned: bool
    first_diff: int | None
    tokens: int
    engine_text_extra: int


def first_difference(a: Sequence[int], b: Sequence[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def check_alignment(template: ChatTemplate, messages: Sequence[Message], *,
                    tools: Sequence[Message] | None = None, add_generation_prompt: bool = True) -> AlignmentReport:
    text = template.render(messages, tools=tools, add_generation_prompt=add_generation_prompt)
    ours = template.encode(text)
    reference = template.tokenizer.apply_chat_template(
        _render_ready(messages), tools=list(tools) if tools else None, tokenize=True,
        add_generation_prompt=add_generation_prompt, **dict(template.spec.template_kwargs))
    if isinstance(reference, Mapping):
        reference = reference["input_ids"]
    engine_text = template.tokenizer.encode(text, add_special_tokens=True)
    diff = first_difference(ours, list(reference))
    return AlignmentReport(aligned=diff is None, first_diff=diff, tokens=len(ours),
                           engine_text_extra=len(engine_text) - len(ours))
