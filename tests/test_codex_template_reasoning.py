"""rl-fn-codex-rollout 0.2 (upstream 5.1): offline check that each codex profile's
``keeps_history_reasoning`` declaration matches its fork fixed TITO template.

Renders a six-message history (system / user / assistant(think+text) / tool /
assistant(think+text) / user) with the fork pin's template for ``qwen35`` and
``qwen4exp`` and asserts the *first* assistant's think text survives the render
exactly when the profile declares ``True``.

The fork checkout is located via ``YETO_MILES_FORK_DIR`` or the usual
``~/work`` checkouts; the template file must be byte-identical to the one at
``MILES_NEXT_COMMIT`` (checked via ``git rev-parse <commit>:<path>``), so a
checkout whose HEAD moved on is still accepted as long as the jinja did not.
Rendering goes through the fork's ``chat_template_verify.get_standard_result``
when the fork package imports (it needs ``sglang``); otherwise through the very
``transformers.render_jinja_template`` call that helper wraps.  Missing
checkout -> skip, never fail.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from yeto.rl import MILES_NEXT_COMMIT
from yeto.rl.codex_backend import stock_codex_backend_profile, stock_codex_keeps_history_reasoning, stock_codex_tito_model

TEMPLATE_DIR = "miles/utils/chat_template_utils/templates"
FIXED_TEMPLATES = {  # fork tito_tokenizer.py @ MILES_NEXT_COMMIT: TITOTokenizerType -> FixedTemplate
    "qwen35": ("qwen3.5_fixed.jinja", {"preserve_thinking": True}),
    "qwen4exp": ("qwen3.8_small_and_flash_next_fixed.jinja", {"preserve_thinking": True}),
}
CANDIDATES = [os.environ.get("YETO_MILES_FORK_DIR"), "~/work/miles-fr1", "~/work/miles-critic-c357-tied"]

FIRST_THINK = "FIRST-TURN-REASONING-7f3a"
SECOND_THINK = "SECOND-TURN-REASONING-9c1e"
HISTORY = [
    {"role": "system", "content": "You are a terminal agent."},
    {"role": "user", "content": "List the files."},
    # tool_call arguments are a dict, as the fork's normalize_tool_arguments(..., "dict") hands HF-Jinja templates
    {"role": "assistant", "content": "", "reasoning_content": FIRST_THINK,
     "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": {"cmd": "ls"}}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "a.txt\nb.txt"},
    {"role": "assistant", "content": "Two files.", "reasoning_content": SECOND_THINK},
    # A later user query: both Qwen fixed templates only ever drop reasoning
    # *before* the last user turn (``loop.index0 > ns.last_query_index`` keeps
    # the tail), so without this the preserve_thinking switch is untestable.
    {"role": "user", "content": "Now count them."},
]
TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}}]


def _pinned_template(name: str) -> tuple[Path, Path] | None:
    rel = f"{TEMPLATE_DIR}/{name}"
    for cand in CANDIDATES:
        if not cand:
            continue
        root = Path(cand).expanduser()
        path = root / rel
        if not path.is_file():
            continue
        try:
            pinned = subprocess.run(["git", "-C", str(root), "rev-parse", f"{MILES_NEXT_COMMIT}:{rel}"],
                                    capture_output=True, text=True, check=True).stdout.strip()
            actual = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            continue
        if pinned == actual:
            return root, path
    return None


def _render(root: Path, template: str, messages, **kwargs) -> str:
    try:
        sys.path.insert(0, str(root))
        from miles.utils.test_utils.chat_template_verify import get_standard_result  # fork helper
    except Exception:  # fork needs sglang; use the same HF call the helper wraps
        from transformers.utils.chat_template_utils import render_jinja_template

        rendered, _ = render_jinja_template(conversations=[messages], chat_template=template, add_generation_prompt=True, tools=TOOLS, **kwargs)
        return rendered[0]
    finally:
        sys.path.remove(str(root))
    return get_standard_result(template, messages, tools=TOOLS, **kwargs)


@pytest.mark.parametrize("profile_name", ["qwen35", "qwen35_08b", "qwen38_next", "qwen38_next_4layer"])
def test_keeps_history_reasoning_declaration_matches_fork_template(profile_name):
    tito = stock_codex_tito_model(profile_name)
    jinja, extra_kwargs = FIXED_TEMPLATES[tito]
    found = _pinned_template(jinja)
    if found is None:
        pytest.skip(f"no fork checkout with {jinja} at MILES_NEXT_COMMIT {MILES_NEXT_COMMIT[:7]}")
    root, path = found
    kwargs = {**stock_codex_backend_profile(profile_name)["chat_template_kwargs"], **extra_kwargs}
    text = _render(root, path.read_text(), HISTORY, **kwargs)
    declared = stock_codex_keeps_history_reasoning(profile_name)
    assert (FIRST_THINK in text) is declared, f"{profile_name}: declared {declared}, render:\n{text[-600:]}"
    assert (SECOND_THINK in text) is declared
    if declared:
        assert text.index(FIRST_THINK) < text.index("a.txt") < text.index(SECOND_THINK) < text.index("Now count them.")


def test_dropping_preserve_thinking_would_falsify_the_declaration():
    """Scenario 声明与模板不符: the same qwen4exp template with preserve_thinking=False drops the first think."""
    found = _pinned_template(FIXED_TEMPLATES["qwen4exp"][0])
    if found is None:
        pytest.skip("no fork checkout at MILES_NEXT_COMMIT")
    root, path = found
    kwargs = {**stock_codex_backend_profile("qwen38_next")["chat_template_kwargs"], "preserve_thinking": False}
    text = _render(root, path.read_text(), HISTORY, **kwargs)
    assert FIRST_THINK not in text and SECOND_THINK not in text and "Now count them." in text
