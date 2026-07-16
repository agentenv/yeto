import json

from yeto.codex_traces import convert_paths, row_from_atif, row_from_session_events


def test_atif_trajectory_becomes_yeto_chat_row():
    doc = {
        "schema_version": "ATIF-v1.6",
        "session_id": "s1",
        "agent": {"model_name": "gpt-test"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "fix the bug"},
            {
                "step_id": 2,
                "source": "agent",
                "model_name": "gpt-test",
                "message": "I'll inspect the file.",
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "function_name": "Read",
                        "arguments": {"path": "src/main.rs"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "call-1",
                            "content": "fn main() {}",
                            "success": True,
                        }
                    ]
                },
                "extra": {"thinking": "private plan"},
            },
            {"step_id": 3, "source": "agent", "message": "Done."},
        ],
        "extra": {"success": True},
    }

    row = row_from_atif(doc)

    assert row["messages"][0] == {"role": "user", "content": "fix the bug"}
    assistant = row["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "I'll inspect the file."
    assert assistant["tool_calls"][0]["id"] == "call-1"
    assert assistant["tool_calls"][0]["function"]["name"] == "Read"
    assert "private plan" not in assistant["content"]
    assert row["messages"][2] == {
        "role": "tool",
        "content": "fn main() {}",
        "tool_call_id": "call-1",
    }
    assert row["messages"][3] == {"role": "assistant", "content": "Done."}
    assert row["metadata"]["source"] == "atif"


def test_failed_atif_is_filtered_by_default():
    doc = {
        "schema_version": "ATIF-v1.6",
        "steps": [
            {"source": "user", "message": "task"},
            {"source": "agent", "message": "failed"},
        ],
        "extra": {"success": False},
    }

    assert row_from_atif(doc) is None
    assert row_from_atif(doc, include_failures=True)["messages"][1]["content"] == "failed"


def test_include_thinking_keeps_plain_text_reasoning():
    doc = {
        "schema_version": "ATIF-v1.6",
        "steps": [
            {"source": "user", "message": "task"},
            {
                "source": "agent",
                "message": "answer",
                "extra": {"thinking": "plain plan"},
            },
        ],
        "extra": {"success": True},
    }

    row = row_from_atif(doc, include_thinking=True)

    assert row["messages"][1]["content"] == "answer\n\n[thinking]\nplain plan"
    assert "reasoning_status" not in row["metadata"]


def test_include_thinking_skips_encrypted_reasoning():
    doc = {
        "schema_version": "ATIF-v1.6",
        "steps": [
            {"source": "user", "message": "task"},
            {
                "source": "agent",
                "message": "answer",
                "extra": {
                    "thinking": {
                        "encrypted": True,
                        "ciphertext": "abc123",
                        "algorithm": "test",
                    }
                },
            },
        ],
        "extra": {"success": True},
    }

    row = row_from_atif(doc, include_thinking=True)

    assert row["messages"][1]["content"] == "answer"
    assert "ciphertext" not in row["messages"][1]["content"]
    assert row["metadata"]["reasoning_status"] == "encrypted_skipped"
    assert row["metadata"]["encrypted_reasoning_skipped"] == 1


def test_teacher_backfill_replaces_encrypted_atif_reasoning():
    seen = {}

    def teacher(messages, assistant_content):
        seen["messages"] = messages
        seen["assistant_content"] = assistant_content
        return "synthetic rationale"

    doc = {
        "schema_version": "ATIF-v1.6",
        "steps": [
            {"source": "user", "message": "task"},
            {
                "source": "agent",
                "message": "answer",
                "extra": {"thinking": {"encrypted_content": "secret-blob"}},
            },
        ],
        "extra": {"success": True},
    }

    row = row_from_atif(
        doc,
        include_thinking=True,
        reasoning_policy="teacher-backfill",
        teacher_backfill_fn=teacher,
    )

    assert row["messages"][1]["content"] == "answer\n\n[thinking]\nsynthetic rationale"
    assert seen["assistant_content"] == "answer"
    assert "secret-blob" not in json.dumps(seen)
    assert row["metadata"]["reasoning_status"] == "teacher_backfilled"
    assert row["metadata"]["synthetic_reasoning"] is True


def test_session_events_become_chat_row():
    row = row_from_session_events(
        [
            {"type": "user_message", "text": "hello"},
            {
                "type": "tool_invocation",
                "call_id": "c1",
                "tool_id": "Read",
                "input": "{\"path\":\"a\"}",
                "output": "contents",
                "success": True,
            },
            {"type": "assistant_message", "text": "hi"},
        ]
    )

    assert [m["role"] for m in row["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert row["messages"][1]["content"] == ""
    assert row["messages"][1]["tool_calls"][0]["id"] == "c1"
    assert row["messages"][1]["tool_calls"][0]["function"]["name"] == "Read"
    assert "Tool Read [ok]" in row["messages"][2]["content"]


def test_codex_session_events_skip_encrypted_reasoning():
    row = row_from_session_events(
        [
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "hello"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "gAAAAABqTrNaturalEncryptedBlob==",
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "hi",
                    "phase": "final_answer",
                },
            },
        ]
    )

    assert row["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert row["metadata"]["reasoning_status"] == "encrypted_skipped"
    assert row["metadata"]["encrypted_reasoning_skipped"] == 1


def test_teacher_backfill_replaces_codex_encrypted_reasoning():
    seen = {}

    def teacher(messages, assistant_content):
        seen["messages"] = messages
        seen["assistant_content"] = assistant_content
        return "synthetic codex rationale"

    row = row_from_session_events(
        [
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "hello"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "gAAAAABqTrNaturalEncryptedBlob==",
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "message": "hi",
                    "phase": "final_answer",
                },
            },
        ],
        reasoning_policy="teacher-backfill",
        teacher_backfill_fn=teacher,
    )

    assert row["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi\n\n[thinking]\nsynthetic codex rationale"},
    ]
    assert seen["messages"] == [{"role": "user", "content": "hello"}]
    assert seen["assistant_content"] == "hi"
    assert "gAAAAA" not in json.dumps(seen)
    assert row["metadata"]["reasoning_status"] == "teacher_backfilled"
    assert row["metadata"]["teacher_backfilled_reasoning"] == 1


def test_convert_paths_reads_directories_and_writes_rows(tmp_path):
    trace = tmp_path / "trajectory.json"
    trace.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "steps": [
                    {"source": "user", "message": "task"},
                    {"source": "agent", "message": "answer"},
                ],
                "extra": {"success": True},
            }
        )
    )

    rows = convert_paths([tmp_path])

    assert len(rows) == 1
    assert rows[0]["messages"][1]["content"] == "answer"
