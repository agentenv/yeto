"""DeepSeek V4 Jinja renderer pinned to the upstream template commit."""
from __future__ import annotations

import copy
import json
from pathlib import Path


class DeepSeekJinjaRenderer:
    def __init__(self, template_path):
        try:
            from jinja2 import Environment, FileSystemLoader
        except ImportError as exc:
            raise ValueError("Jinja rendering requires the Jinja2 package") from exc
        path = Path(template_path).resolve()
        if not path.is_file():
            raise ValueError("DeepSeek Jinja template is missing")
        # The upstream template's tojson expressions are concatenated into DSML
        # markup. Return plain JSON text so Jinja does not HTML-escape it.
        env = Environment(loader=FileSystemLoader(str(path.parent)), autoescape=False)
        env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        self.template = env.get_template(path.name)
        self.path = path

    @staticmethod
    def _jinja_messages(messages):
        result = copy.deepcopy(messages)
        for message in result:
            for call in message.get("tool_calls", []):
                function = call.get("function", {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise ValueError("Tool-call arguments must be valid JSON for Jinja rendering") from exc
        return result

    def render(self, messages, *, thinking_mode="thinking", drop_thinking=True,
               add_generation_prompt=False, tools=None):
        kwargs = {"messages": self._jinja_messages(messages), "thinking_mode": thinking_mode,
                  "drop_thinking": drop_thinking, "add_generation_prompt": add_generation_prompt}
        if tools:
            kwargs["tools"] = copy.deepcopy(tools)
        return self.template.render(**kwargs)
