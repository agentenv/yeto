import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_evalplus_qwen.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("benchmark_evalplus_qwen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    def __init__(self):
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "rendered"


def test_render_code_prompt_uses_native_non_thinking_assistant_prefill():
    tokenizer = FakeTokenizer()

    assert MODULE.render_code_prompt(tokenizer, "def add(a, b):\n    pass") == "rendered"
    assert tokenizer.kwargs == {
        "tokenize": False,
        "continue_final_message": True,
        "enable_thinking": False,
    }
    assert tokenizer.messages[0]["role"] == "user"
    assert "def add(a, b)" in tokenizer.messages[0]["content"]
    assert tokenizer.messages[1]["role"] == "assistant"
    assert tokenizer.messages[1]["content"].endswith("```python\n")


def test_completed_task_ids_supports_resumable_jsonl(tmp_path):
    output = tmp_path / "samples.jsonl"
    output.write_text(
        '{"task_id": "HumanEval/0", "solution": "pass"}\n'
        '{"task_id": "HumanEval/1", "solution": "pass"}\n'
    )

    assert MODULE._completed_task_ids(output) == {"HumanEval/0", "HumanEval/1"}
