"""Tests for scripts/check_commit_message.py (the commit-msg language check)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_commit_message.py"
_spec = importlib.util.spec_from_file_location("check_commit_message", _SCRIPT)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def _run(tmp_path, message: str) -> int:
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(message, encoding="utf-8")
    return check.main(["check_commit_message.py", str(path)])


def test_plain_english_passes(tmp_path):
    assert _run(tmp_path, "Add hook\n\nExplain why the hook exists.\n") == 0


def test_han_in_subject_fails(tmp_path):
    assert _run(tmp_path, "Add 钩子\n\nBody.\n") == 1


def test_han_in_body_fails(tmp_path):
    assert _run(tmp_path, "Add hook\n\n这是正文。\n") == 1


def test_fullwidth_punctuation_in_body_fails(tmp_path):
    assert _run(tmp_path, "Add hook\n\nNote：see below\n") == 1


@pytest.mark.parametrize("text", ["kana ひらがな", "hangul 한글", "ideographic　space"])
def test_other_cjk_scripts_fail(tmp_path, text):
    assert _run(tmp_path, f"Add hook\n\n{text}\n") == 1


def test_arrow_and_accented_letters_pass(tmp_path):
    assert _run(tmp_path, "Rename a → b\n\nCafé résumé.\n") == 0


def test_comment_lines_and_scissors_section_are_ignored(tmp_path):
    message = (
        "Add hook\n"
        "# 请输入提交信息\n"
        f"{check.SCISSORS}\n"
        "diff 中文\n"
    )
    assert _run(tmp_path, message) == 0


def test_violation_reports_line_number(capsys, tmp_path):
    _run(tmp_path, "Add hook\n\n中文\n")
    assert "line 3" in capsys.readouterr().err


def test_wrong_argument_count_is_usage_error():
    assert check.main(["check_commit_message.py"]) == 2
