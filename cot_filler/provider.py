"""Explicit opt-in teacher client. Demo uses no network and no model."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import importlib.util
from pathlib import Path

from .core import PROMPT_VERSION, digest, prompt_messages


class ContextOverflow(ValueError):
    pass


def check_budget(prompt_tokens, max_output_tokens, context_limit, safety_margin=256):
    if min(prompt_tokens, max_output_tokens, context_limit) < 1 or safety_margin < 0:
        raise ValueError("Token budgets must be positive")
    total = prompt_tokens + max_output_tokens + safety_margin
    if total > context_limit:
        raise ContextOverflow(f"Full prompt ({prompt_tokens}) + output ({max_output_tokens}) + margin ({safety_margin}) exceeds context {context_limit}; no truncation was performed")
    return total


class FakeProvider:
    model = "fixture-only-no-llm"

    def generate(self, gap):
        target = gap["target"]
        text = f"DEMO FIXTURE — no language model generated this text. The next original action is event {target['event_id']}. Replace this fixture with a real reviewed candidate before evaluating rationale quality."
        return text, {"generator": {"model": self.model, "provider": "fake", "prompt_version": PROMPT_VERSION},
                      "prompt_hash": gap["prompt_hash"], "flags": ["demo_fixture_not_quality_evidence"],
                      "finish_reason": "stop", "synthetic": True, "lookahead_conditioned": True}


class LocalTokenizer:
    def __init__(self, path):
        path = Path(path).resolve()
        if not path.is_dir():
            raise ValueError("Tokenizer must be an existing local directory; downloads are disabled")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ValueError("Real generation requires transformers and the serving model's local tokenizer") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        hashes = {str(p.relative_to(path)): __import__("hashlib").sha256(p.read_bytes()).hexdigest() for p in sorted(path.rglob("*")) if p.is_file() and ("token" in p.name or p.suffix in {".jinja", ".model"})}
        if not hashes:
            raise ValueError("Tokenizer identity has no recognized tokenizer/template files")
        self.identity = digest(hashes)

    def count(self, messages, template_options=None):
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **(template_options or {})))


class DeepSeekV4Tokenizer:
    """Count prompts using the serving copy's native V4 renderer and tokenizer."""
    def __init__(self, path):
        path = Path(path).resolve()
        tok = path / "tokenizer.json"
        enc = path / "encoding" / "encoding_dsv4.py"
        if not tok.is_file() or not enc.is_file():
            raise ValueError("DeepSeek V4 tokenizer.json and encoding/encoding_dsv4.py are required")
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ValueError("DeepSeek V4 counting requires the tokenizers package") from exc
        spec = importlib.util.spec_from_file_location("cot_filler_deepseek_v4_encoding", enc)
        if not spec or not spec.loader:
            raise ValueError("Unable to load DeepSeek V4 encoding module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.tokenizer = Tokenizer.from_file(str(tok))
        self.encoding = module
        self.identity = digest({"tokenizer": __import__("hashlib").sha256(tok.read_bytes()).hexdigest(), "encoding": __import__("hashlib").sha256(enc.read_bytes()).hexdigest()})

    def count(self, messages, template_options=None):
        options = dict(template_options or {})
        thinking = "thinking" if options.pop("thinking", True) else "chat"
        rendered = self.encoding.encode_messages(messages, thinking_mode=thinking,
                                                  drop_thinking=True,
                                                  reasoning_effort=options.pop("reasoning_effort", None))
        if options:
            raise ValueError("Unsupported DeepSeek V4 template options: " + ", ".join(sorted(options)))
        return len(self.tokenizer.encode(rendered).ids)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Teacher redirects are disabled")


class OpenAICompatibleProvider:
    def __init__(self, config, tokenizer=None):
        required = {"model", "base_url", "tokenizer_path", "context_limit", "max_output_tokens", "chat_template_matches_server"}
        if not required <= config.keys() or config.get("chat_template_matches_server") is not True:
            raise ValueError("Real teacher config needs model, base_url, local tokenizer, limits, and verified matching server chat template")
        url = urllib.parse.urlparse(config["base_url"])
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("Invalid teacher base URL; credentials/query/fragment are prohibited")
        loopback = url.hostname in {"localhost", "127.0.0.1", "::1"}
        if not loopback and (not config.get("allow_remote_endpoint") or url.scheme != "https"):
            raise ValueError("Remote teacher requires explicit allow_remote_endpoint and HTTPS; use localhost SSH forwarding for the fleet")
        self.config = config
        options = config.get("chat_template_kwargs", {})
        if not isinstance(options, dict) or set(options) & {"messages", "conversation", "tokenize", "add_generation_prompt", "return_tensors", "return_dict"}:
            raise ValueError("Unsupported chat template controls")
        if "reasoning_effort" in config and options.get("reasoning_effort") != config["reasoning_effort"]:
            raise ValueError("reasoning_effort must also be explicitly included in matching chat_template_kwargs for token counting")
        if tokenizer:
            self.tokenizer = tokenizer
        elif config.get("tokenizer_type") == "deepseek_v4":
            self.tokenizer = DeepSeekV4Tokenizer(config["tokenizer_path"])
        else:
            self.tokenizer = LocalTokenizer(config["tokenizer_path"])

    def generate(self, gap):
        config = self.config
        messages = prompt_messages(gap)
        template_options = config.get("chat_template_kwargs", {})
        count = self.tokenizer.count(messages, template_options)
        check_budget(count, int(config["max_output_tokens"]), int(config["context_limit"]), int(config.get("safety_margin", 256)))
        body = {"model": config["model"], "messages": messages, "max_tokens": int(config["max_output_tokens"]), "temperature": config.get("temperature", 0.3), "stream": False}
        # Only explicit model controls; cannot override messages or output budget.
        for key in ("reasoning_effort", "chat_template_kwargs"):
            if key in config:
                body[key] = config[key]
        key = os.environ.get(config.get("api_key_env", "COT_TEACHER_API_KEY"))
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(config["base_url"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(request, timeout=int(config.get("timeout_seconds", 300))) as response:
                payload = response.read(4 * 1024 * 1024 + 1)
                if len(payload) > 4 * 1024 * 1024:
                    raise ValueError("Teacher response exceeds safety bound")
                result = json.loads(payload)
        except urllib.error.HTTPError as exc:
            # Never print server body: it can echo private transcript content.
            raise ValueError(f"Teacher returned HTTP {exc.code}") from None
        except urllib.error.URLError:
            raise ValueError("Teacher connection failed; inspect endpoint access separately") from None
        choices = result.get("choices", [])
        if len(choices) != 1:
            raise ValueError("Expected exactly one teacher completion")
        choice = choices[0]
        text = choice.get("message", {}).get("content")
        if not isinstance(text, str):
            raise ValueError("Teacher returned no textual rationale")
        finish = choice.get("finish_reason")
        flags = [] if finish == "stop" else ["truncated_output" if finish == "length" else "incomplete_output"]
        return text, {"generator": {"provider": "openai-compatible", "model": config["model"], "prompt_version": PROMPT_VERSION,
                                     "tokenizer_sha256": self.tokenizer.identity, "parameters": {k: body[k] for k in body if k not in {"messages", "stream"}}},
                      "prompt_hash": digest(messages), "prompt_tokens_local": count, "counted_chat_template_kwargs": template_options, "usage": {k: v for k, v in result.get("usage", {}).items() if isinstance(v, (int, float))},
                      "finish_reason": finish, "flags": flags, "synthetic": True, "lookahead_conditioned": True}
