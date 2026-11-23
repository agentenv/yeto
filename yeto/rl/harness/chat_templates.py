"""Chat template registry and the token alignment check.

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
* ``qwen3``: the Qwen3 hermes tool-call format; the small CPU stand-in;
* ``qwen3_2507``: Qwen3-4B-Instruct-2507 (non-thinking Qwen3, hermes JSON tool
  calls, no reasoning block); the Codex system messages are merged into one;
* ``deepseek_v41``: DeepSeek-V4.1 (no HF chat template): rendered by the
  checkpoint's own ``encoding/encoding.py`` ``encode_messages`` (what vLLM's
  ``deepseek_v41`` tokenizer mode runs: tools on the first system message, one
  inserted when there is none; ``thinking`` / ``reasoning_effort`` /
  ``drop_thinking`` template kwargs), tokenized with ``add_special_tokens=False``;
  DSML tool calls (``<\uff5cDSML\uff5c invoke name=...>``) parsed by :func:`parse_deepseek_v41`.
  Reasoning of every turn after the last real user turn is kept (tool-result
  turns do not count, and any conversation with tools keeps all of it);
* ``qwen38``: Qwen3.8 (Qwen3.5 family) XML tool calls (``<tool_call><function=name>
  <parameter=k>v</parameter></function></tool_call>``), thinking kept in every
  turn (``preserve_thinking``). Load with the checkpoint's tokenizer
  (``tokenizer_path``); a deployment may set ``reasoning_effort`` by
  replacing ``template_kwargs`` (xhigh / medium / low; the template's default
  is xhigh).

:func:`check_alignment` is the drift check: the template rendering, tokenized
the way the proxy sends it, must equal the tokenizer's own
``apply_chat_template`` ids token by token, and it reports how many tokens an
engine would add if it got the text and tokenized it with special tokens.
"""

from __future__ import annotations

import array
import bisect
import collections
import hashlib
import json
import re
import sys
import threading
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


def parse_glm(text: str, *, thinking: bool = True) -> Message:
    """``thinking=False`` (``enable_thinking: false``): the generation prompt
    closed the reasoning block, the output has none."""
    reasoning, rest = _split_reasoning(text, opened=thinking)
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


_QWEN38_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
_QWEN38_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_qwen38(text: str, *, thinking: bool = True) -> Message:
    """Qwen3.8 output after the ``<think>\\n`` generation prompt (``thinking=False``:
    after the closed ``<think>\\n\\n</think>\\n\\n`` prefill, no reasoning)."""
    reasoning, rest = _split_reasoning(text, opened=thinking)
    calls = []
    for index, match in enumerate(_QWEN38_CALL.finditer(rest)):
        arguments = {k.strip(): _glm_value(v) for k, v in _QWEN38_PARAM.findall(match.group(2))}
        calls.append(_tool_call(_call_id(text, index), match.group(1).strip(), arguments))
    if rest.count("<tool_call>") != len(calls):
        raise TemplateError(f"Qwen3.8 tool call is not <function=...> XML: {rest[rest.find('<tool_call>'):][:80]!r}")
    content = rest.split("<tool_call>", 1)[0].strip()
    return _assistant(content, reasoning, calls)


_K3_OPEN, _K3_CLOSE, _K3_SEP = "<|open|>", "<|close|>", "<|sep|>"
_K3_CALL = re.compile(r'<\|open\|>call tool="([^"]*)" index="(\d+)"<\|sep\|>(.*?)<\|close\|>call<\|sep\|>', re.S)
_K3_ARG = re.compile(r'<\|open\|>argument key="([^"]*)" type="([^"]*)"<\|sep\|>(.*?)<\|close\|>argument<\|sep\|>', re.S)
_K3_JSON = re.compile(r'<\|open\|>json type="object"<\|sep\|>(.*?)<\|close\|>json<\|sep\|>', re.S)
_K3_RESPONSE = re.compile(r"<\|open\|>response<\|sep\|>(.*?)<\|close\|>response<\|sep\|>", re.S)


def _k3_attr(value: str) -> str:
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _k3_arguments(body: str) -> str:
    """The call's arguments as JSON text whose one-level parse (the template's
    ``normalize_tool_arguments``) gives back each ``(key, type, text)``: string
    values JSON-quoted, other values their literal text."""
    block = _K3_JSON.fullmatch(body)
    if block is not None:
        return block.group(1)
    args = _K3_ARG.findall(body)
    if _K3_ARG.sub("", body) != "":
        raise TemplateError(f"Kimi K3 tool call body is not <argument> XTML: {body[:80]!r}")
    parts = []
    for key, kind, text in args:
        value = json.dumps(text, ensure_ascii=False) if kind == "string" else text
        if kind != "string":
            try:
                json.loads(text)
            except ValueError as exc:
                raise TemplateError(f"Kimi K3 argument {key!r} of type {kind} is not JSON: {text[:80]!r}") from exc
        parts.append(f"{json.dumps(_k3_attr(key), ensure_ascii=False)}:{value}")
    return "{" + ",".join(parts) + "}"


def parse_kimi_k3(text: str, *, thinking: bool = True) -> Message:
    """Kimi K3 XTML output after the ``<message role="assistant"><think>``
    generation prompt: reasoning, the ``response`` channel and ``tools`` calls.
    Reasoning and content are kept byte for byte (the template renders them
    raw), so the next turn's rendering repeats the sampled text.

    ``thinking=False`` (template kwarg ``thinking: false``): the generation
    prompt opened the ``response`` channel instead, the output has no think
    channel and the message no reasoning (the template renders none)."""
    reasoning: str | None
    if thinking:
        marker = f"{_K3_CLOSE}think{_K3_SEP}"
        if marker not in text:
            return _assistant("", text, [])  # cut inside the think channel
        reasoning, rest = text.split(marker, 1)
    else:
        reasoning, rest = None, f"{_K3_OPEN}response{_K3_SEP}{text}"
    response = _K3_RESPONSE.search(rest)
    content = response.group(1) if response else ""
    calls = []
    for index, match in enumerate(_K3_CALL.finditer(rest)):
        calls.append({"id": _call_id(text, index), "type": "function",
                      "function": {"name": _k3_attr(match.group(1)), "arguments": _k3_arguments(match.group(3))}})
    if rest.count(f"{_K3_OPEN}call ") != len(calls):
        raise TemplateError(f"Kimi K3 tool call is not <call> XTML: {rest[rest.find(_K3_OPEN + 'call '):][:80]!r}")
    return _assistant(content, reasoning, calls)


_DS_EOS = "<\uff5cend\u2581of\u2581sentence\uff5c>"
_DS_DSML = "\uff5cDSML\uff5c"
_DS_CALLS = f"\n\n<{_DS_DSML} calls>"
_DS_INVOKE = re.compile(rf'<{_DS_DSML} invoke name="(.*?)">\n(.*?)</{_DS_DSML} invoke>', re.S)
_DS_PARAM = re.compile(rf'<{_DS_DSML} parameter name="(.*?)" string="(true|false)">(.*?)</{_DS_DSML} parameter>', re.S)


def _ds_value(raw: str, is_string: str) -> Any:
    if is_string == "true":
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def parse_deepseek_v41(text: str, *, thinking: bool = True) -> Message:
    """DeepSeek-V4.1 output after the ``<\uff5cAssistant\uff5c><think>`` generation prompt
    (``thinking=False``: after ``</think>``): reasoning up to ``</think>``, the
    content, then DSML tool calls in a ``\\n\\n<\uff5cDSML\uff5c calls>`` block. Reasoning and
    content are kept byte for byte (the encoder renders them raw)."""
    reasoning: str | None = None
    rest = text
    if thinking:
        if "</think>" not in text:
            return _assistant("", text, [])  # cut inside the reasoning
        reasoning, rest = text.split("</think>", 1)
    if rest.endswith(_DS_EOS):
        rest = rest[: -len(_DS_EOS)]
    content, sep, block = rest.partition(_DS_CALLS)
    calls = []
    if sep:
        for index, match in enumerate(_DS_INVOKE.finditer(block)):
            arguments = {name: _ds_value(value, flag) for name, flag, value in _DS_PARAM.findall(match.group(2))}
            calls.append(_tool_call(_call_id(text, index), match.group(1), arguments))
        if block.count(f"<{_DS_DSML} invoke ") != len(calls):
            raise TemplateError(f"DeepSeek-V4.1 tool call is not DSML: {block[:80]!r}")
    return _assistant(content, reasoning, calls)


# ---------------------------------------------------------------------- registry
@dataclass(frozen=True)
class TemplateSpec:
    """Model-specific template facts. ``stop_tokens``: special tokens that end
    an assistant turn (sent to the engine as stop token ids).
    ``keeps_history_reasoning``: with ``template_kwargs`` the template renders
    the reasoning of every earlier turn (gateway ``ChainRegistry`` input).
    ``verified``: what the token alignment check ran on (free text).
    ``merge_leading_system``: join the leading system messages into one (the
    template accepts a single system message; Codex sends its instructions
    and a developer message, both system in the gateway form)."""

    name: str
    tokenizer: str
    parser: Callable[..., Message]
    stop_tokens: tuple[str, ...]
    template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    keeps_history_reasoning: bool = True
    verified: str = "not verified"
    merge_leading_system: bool = False
    # the tokenizer is custom code shipped with the checkpoint (Kimi's TikTokenTokenizer): loading it
    # needs the caller's explicit trust_remote_code opt-in (``ChatTemplate.load``), never a default
    needs_remote_code: bool = False
    # tool-call arguments go to the template as the JSON text they were parsed from
    # (the template reads them itself and keeps each value's literal text)
    raw_arguments: bool = False
    # the template kwarg that turns the reasoning block off (default on): the parser takes it as
    # ``thinking=`` (with it off the generation prompt closes or omits the block, the output has no
    # reasoning; a parser that assumes an open block would read the whole output as reasoning)
    thinking_kwarg: str | None = None
    # the tokenizer encodes a rendering segment by segment (structural markers as
    # control tokens, message text as plain BPE even where it spells a marker):
    # ids come from the rendering's segments, never from re-encoding its text
    segment_encoding: bool = False
    # the checkpoint ships its prompt encoder instead of a chat template (DeepSeek-V4.1
    # ``encoding/encoding.py``, relative to the tokenizer directory): it renders every turn
    checkpoint_encoder: str | None = None


TEMPLATES: dict[str, TemplateSpec] = {}


def register_template(spec: TemplateSpec) -> TemplateSpec:
    if spec.name in TEMPLATES and TEMPLATES[spec.name] != spec:
        raise TemplateError(f"template {spec.name!r} is already registered")
    TEMPLATES[spec.name] = spec
    return spec


register_template(TemplateSpec(
    name="glm5", tokenizer="zai-org/GLM-4.7-Flash", parser=parse_glm,
    stop_tokens=("<|endoftext|>", "<|user|>", "<|observation|>"),
    template_kwargs={"clear_thinking": False}, thinking_kwarg="enable_thinking",
    verified="GLM-4.7-Flash tokenizer as a stand-in; GLM5.3-Flash tokenizer not verified"))
register_template(TemplateSpec(
    name="qwen3", tokenizer="Qwen/Qwen3-0.6B", parser=parse_qwen3,
    stop_tokens=("<|im_end|>", "<|endoftext|>"),
    template_kwargs={"enable_thinking": True},
    # The Qwen3 template drops the reasoning of turns before the last user turn.
    keeps_history_reasoning=False,
    verified="Qwen3-0.6B tokenizer (CPU stand-in)"))
register_template(TemplateSpec(
    name="qwen3_2507", tokenizer="Qwen/Qwen3-4B-Instruct-2507", parser=parse_qwen3,
    stop_tokens=("<|im_end|>", "<|endoftext|>"),
    verified="Qwen3-4B-Instruct-2507@cdbee75f tokenizer", merge_leading_system=True))
register_template(TemplateSpec(
    name="qwen38", tokenizer="Qwen/Qwen3.8-27B", parser=parse_qwen38,
    stop_tokens=("<|im_end|>", "<|endoftext|>"),
    template_kwargs={"enable_thinking": True, "preserve_thinking": True}, thinking_kwarg="enable_thinking",
    verified="Qwen3.8-27B@1d4bf0f2 tokenizer", merge_leading_system=True))
register_template(TemplateSpec(
    # Kimi-K3.1 XTML (``encoding_k3.py`` of the checkpoint); ``thinking_effort``
    # comes from the deployment (agentic.template_kwargs; the tokenizer's default is max)
    name="kimi_k31", tokenizer="moonshotai/Kimi-K3.1-Pro", parser=parse_kimi_k3,
    stop_tokens=("<|end_of_msg|>",), template_kwargs={"thinking": True}, thinking_kwarg="thinking",
    verified="Kimi-K3.1-Pro@d9e99b07 tokenizer (CPU alignment test)",
    needs_remote_code=True, raw_arguments=True, segment_encoding=True))
register_template(TemplateSpec(
    # DeepSeek-V4.1-Flash: the checkpoint's encoding/encoding.py (as vLLM's deepseek_v41
    # tokenizer mode); reasoning_effort high is the encoder's default
    name="deepseek_v41", tokenizer="deepseek-ai/DeepSeek-V4.1-Flash-1017", parser=parse_deepseek_v41,
    stop_tokens=(_DS_EOS,), template_kwargs={"thinking": True, "reasoning_effort": "high"},
    thinking_kwarg="thinking", checkpoint_encoder="encoding/encoding.py",
    verified="DeepSeek-V4.1-Flash-1017 tokenizer + encoding.py vs vLLM 0.32.1rc1.dev36 /tokenize (MI355X)"))
register_template(TemplateSpec(
    name="nemotron3", tokenizer="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", parser=parse_qwen38,
    stop_tokens=("<|im_end|>",),
    template_kwargs={"enable_thinking": True, "truncate_history_thinking": False}, thinking_kwarg="enable_thinking",
    verified="Nemotron-3.5-Super-VL-120B tokenizer + chat template (CPU alignment test)",
    merge_leading_system=True))


# ---------------------------------------------------------------------- template
def _join(pieces: Sequence[array.array]) -> list[int]:
    out = array.array("I")
    for piece in pieces:
        out.extend(piece)
    return out.tolist()


def _merge_leading_system(messages: Sequence[Message]) -> list[Message]:
    head = 0
    while head < len(messages) and messages[head].get("role") == "system":
        head += 1
    if head < 2:
        return list(messages)
    if any(not isinstance(m.get("content"), str) for m in messages[:head]):
        raise TemplateError("cannot merge system messages with non-text content")
    merged = {"role": "system", "content": "\n\n".join(m["content"] for m in messages[:head])}
    return [merged, *messages[head:]]


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
        # segment_encoding: this thread's newest renderings, (text, segment end offsets, ids per segment)
        self._recent = threading.local()
        # segment_encoding: (segment text, allow_special) -> ids, shared by every rendering (a turn
        # renders the whole history again, and the earlier segments have the same text)
        self._pieces: "collections.OrderedDict[tuple[str, bool], array.array]" = collections.OrderedDict()
        self._pieces_bytes = 0
        self._pieces_lock = threading.Lock()

    @classmethod
    def load(cls, name: str, *, tokenizer_path: str | None = None, local_files_only: bool = True,
             trust_remote_code: bool = False) -> "ChatTemplate":
        """``trust_remote_code``: the caller's opt-in to run the checkpoint's tokenizer code (like
        ``--trust-remote-code``); required by a template whose tokenizer needs it, ignored otherwise."""
        try:
            spec = TEMPLATES[name]
        except KeyError:
            raise TemplateError(f"unknown chat template {name!r} (known: {sorted(TEMPLATES)})") from None
        if spec.needs_remote_code and not trust_remote_code:
            from .gateway.records_env import TRUST_REMOTE_CODE_ENV

            raise TemplateError(f"chat template {name!r}: its tokenizer is custom code shipped with the checkpoint; "
                                "opt in with trust_remote_code (agentic.trust_remote_code: true, "
                                f"{TRUST_REMOTE_CODE_ENV}=1 for a records proxy)")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or spec.tokenizer, local_files_only=local_files_only,
                                                  trust_remote_code=spec.needs_remote_code and trust_remote_code)
        return cls(spec, tokenizer)

    def render(self, messages: Sequence[Message], *, tools: Sequence[Message] | None = None,
               add_generation_prompt: bool) -> str:
        import jinja2  # comes with transformers

        if self.spec.segment_encoding:
            return self._render_segments(messages, tools=tools, add_generation_prompt=add_generation_prompt)
        if self.spec.checkpoint_encoder:
            return self._render_encoder(messages, tools=tools, add_generation_prompt=add_generation_prompt)
        try:
            return self.tokenizer.apply_chat_template(
                self._ready(messages), tools=list(tools) if tools else None, tokenize=False,
                add_generation_prompt=add_generation_prompt, **dict(self.spec.template_kwargs))
        except jinja2.TemplateError as exc:  # raise_exception(...): the template refuses the conversation
            raise TemplateError(f"{self.spec.name} template refused the conversation: {exc}") from exc

    def _encoder(self) -> Any:
        """The checkpoint's prompt encoder module (``spec.checkpoint_encoder`` under the
        tokenizer directory), loaded once."""
        module = getattr(self, "_encoder_module", None)
        if module is None:
            import importlib.util
            from pathlib import Path

            path = Path(str(getattr(self.tokenizer, "name_or_path", ""))) / str(self.spec.checkpoint_encoder)
            if not path.is_file():
                raise TemplateError(f"{self.spec.name}: no prompt encoder at {path} (load the template with the "
                                    "checkpoint's tokenizer directory)")
            loader = importlib.util.spec_from_file_location(f"_yeto_{self.spec.name}_encoding", path)
            if loader is None or loader.loader is None:
                raise TemplateError(f"{self.spec.name}: {path} is not a Python module")
            module = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(module)
            self._encoder_module = module
        return module

    def _render_encoder(self, messages: Sequence[Message], *, tools: Sequence[Message] | None,
                        add_generation_prompt: bool) -> str:
        """Render with the checkpoint's ``encode_messages`` (vLLM's deepseek_v41 tokenizer
        mode: tools on the first system message, one inserted when there is none)."""
        conversation = [dict(m) for m in self._ready(messages)]
        for message in conversation:
            if "reasoning" in message and "reasoning_content" not in message:
                message["reasoning_content"] = message.pop("reasoning")
            if isinstance(message.get("content"), list):
                message["content"] = "\n\n".join(str(b.get("text", "")) for b in message["content"]
                                                  if b.get("type") in ("text", "input_text", "output_text"))
        if tools:
            system = next((m for m in conversation if m.get("role") == "system"), None)
            if system is None:
                system = {"role": "system", "content": ""}
                conversation.insert(0, system)
            system["tools"] = list(tools)
        kwargs = dict(self.spec.template_kwargs)
        thinking = bool(kwargs.pop("thinking", True))
        try:
            text = self._encoder().encode_messages(conversation, thinking_mode="thinking" if thinking else "chat",
                                                   drop_thinking=bool(kwargs.pop("drop_thinking", True)),
                                                   reasoning_effort=kwargs.pop("reasoning_effort", None))
        except (ValueError, TypeError, KeyError, AssertionError, NotImplementedError) as exc:
            raise TemplateError(f"{self.spec.name} encoder refused the conversation: {exc}") from exc
        if not add_generation_prompt:  # the encoder always opens the next assistant turn after a user turn
            for suffix in ("<\uff5cAssistant\uff5c><think>", "<\uff5cAssistant\uff5c></think>"):
                if text.endswith(suffix) and conversation and conversation[-1].get("role") != "assistant":
                    return text[: -len(suffix)]
        return str(text)

    def _ready(self, messages: Sequence[Message]) -> list[Message]:
        ready = [dict(m) for m in messages] if self.spec.raw_arguments else _render_ready(messages)
        return _merge_leading_system(ready) if self.spec.merge_leading_system else ready

    # segment_encoding: renderings kept per thread for encode(). The records proxy encodes a
    # rendering, or a tail of one, before the same thread renders more than three other texts
    # (RecordingBackend._prompt renders full, start and head, then encodes the tail of full).
    RECENT_RENDERINGS = 4
    # segment_encoding: memory bound of the shared per-segment ids (str + array sizes, LRU)
    PIECE_MEMO_BYTES = 1 << 30

    def encode(self, text: str) -> list[int]:
        """Token ids of rendered text; the template writes its own BOS tokens.

        ``segment_encoding``: ``text`` must be a stop token, or one of this
        thread's last :attr:`RECENT_RENDERINGS` renderings or a tail of one that
        starts at a segment boundary (what the records proxy encodes); anything
        else is refused. Text that is such a tail of two kept renderings with
        different ids (message text spelling the markers) is refused too."""
        if not self.spec.segment_encoding:
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        if text in self.spec.stop_tokens:
            return [self.stop_token_ids[self.spec.stop_tokens.index(text)]]
        if text == "":
            return []
        found: list[int] | None = None
        for rendered, ends, pieces in reversed(self._renderings()):
            if len(rendered) < len(text) or not rendered.endswith(text):
                continue
            start = len(rendered) - len(text)
            first = 0
            if start:
                first = bisect.bisect_left(ends, start)
                if first == len(ends) or ends[first] != start:
                    continue  # not a segment boundary of this rendering
                first += 1
            ids = _join(pieces[first:])
            if found is None:
                found = ids
            elif ids != found:
                raise TemplateError(f"{self.spec.name}: text to encode is a tail of two renderings "
                                    "with different ids; encode right after rendering it")
        if found is None:
            raise TemplateError(f"{self.spec.name}: text to encode is not a tail of a recent rendering "
                                "at a segment boundary")
        return found

    def _renderings(self) -> "collections.deque[tuple[str, list[int], list[array.array]]]":
        recent = getattr(self._recent, "renderings", None)
        if recent is None:
            recent = self._recent.renderings = collections.deque(maxlen=self.RECENT_RENDERINGS)
        return recent

    def _piece(self, text: str, allow_special: bool) -> array.array:
        """Ids of one segment: the tokenizer's ``_encode_text_piece`` (a function of
        the text and the flag), kept in an LRU bounded by :attr:`PIECE_MEMO_BYTES`."""
        key = (text, allow_special)
        with self._pieces_lock:
            ids = self._pieces.get(key)
            if ids is not None:
                self._pieces.move_to_end(key)
                return ids
        ids = array.array("I", self.tokenizer._encode_text_piece(text, allow_special_tokens=allow_special))
        size = sys.getsizeof(text) + sys.getsizeof(ids) + 160  # key tuple, dict entry
        with self._pieces_lock:
            if key not in self._pieces:
                self._pieces[key] = ids
                self._pieces_bytes += size
            while self._pieces_bytes > self.PIECE_MEMO_BYTES and self._pieces:
                (old_text, _), old = self._pieces.popitem(last=False)
                self._pieces_bytes -= sys.getsizeof(old_text) + sys.getsizeof(old) + 160
        return ids

    def _render_segments(self, messages: Sequence[Message], *, tools: Sequence[Message] | None,
                         add_generation_prompt: bool) -> str:
        """Render through the tokenizer's own segment builder (Kimi ``encoding_k3``)
        and keep each segment's ids for :meth:`encode`."""
        module = sys.modules[type(self.tokenizer).__module__]
        kwargs = dict(self.spec.template_kwargs)
        thinking = bool(kwargs.pop("thinking", True))
        kwargs.setdefault("thinking_effort", "max")  # as the tokenizer's apply_chat_template
        try:
            segments = module.build_chat_segments(self._ready(messages), tools=list(tools) if tools else None,
                                                  add_generation_prompt=add_generation_prompt, thinking=thinking,
                                                  **kwargs)
        except (ValueError, TypeError, KeyError, AssertionError) as exc:
            raise TemplateError(f"{self.spec.name} template refused the conversation: {exc}") from exc
        ends, pieces, offset = [], [], 0
        for segment in segments:
            offset += len(segment.text)
            ends.append(offset)
            pieces.append(self._piece(segment.text, bool(segment.allow_special)))
        text = "".join(segment.text for segment in segments)
        self._renderings().append((text, ends, pieces))
        return text

    def text_ids(self, text: str) -> list[int]:
        """Token ids of plain message text, outside any rendering: to measure and
        cut message content (the records proxy's compaction), never a stream's ids
        (:meth:`encode` gives those)."""
        if self.spec.segment_encoding:
            return list(self._piece(text, False))
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def stop_text(self, token_id: int) -> str | None:
        return self._stop_by_id.get(int(token_id))

    def parse(self, text: str) -> Message:
        if self.spec.thinking_kwarg is not None:
            return self.spec.parser(text, thinking=bool(self.spec.template_kwargs.get(self.spec.thinking_kwarg, True)))
        return self.spec.parser(text)


# -------------------------------------------------------------------- drift check
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
        template._ready(messages), tools=list(tools) if tools else None, tokenize=True,
        add_generation_prompt=add_generation_prompt, **dict(template.spec.template_kwargs))
    if isinstance(reference, Mapping):
        reference = reference["input_ids"]
    engine_text = template.tokenizer.encode(text, add_special_tokens=True)
    diff = first_difference(ours, list(reference))
    return AlignmentReport(aligned=diff is None, first_diff=diff, tokens=len(ours),
                           engine_text_extra=len(engine_text) - len(ours))
