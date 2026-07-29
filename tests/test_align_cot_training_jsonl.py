import json

from scripts import align_cot_training_jsonl as aligner


def test_aligns_selected_reasoning_to_real_user_and_final_answer():
    events = [
        {
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "Please fix the failing test."},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "id": "rs-selected",
                "summary": [{"type": "summary_text", "text": "Inspecting the failure"}],
                "encrypted_content": "gAAAAA-secret-must-not-appear",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "phase": "commentary",
                "message": "I will inspect it now.",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "phase": "final_answer",
                "message": "I fixed the failing assertion and verified the focused test suite.",
            },
        },
    ]

    rows = aligner.aligned_rows_from_events(
        events,
        {"rs-selected"},
        session_id="session-1",
        include_public_summaries=True,
        min_assistant_chars=1,
    )

    assert len(rows) == 1
    assert rows[0]["messages"] == [
        {"role": "user", "content": "Please fix the failing test."},
        {
            "role": "assistant",
            "content": "I fixed the failing assertion and verified the focused test suite.",
            "reasoning_content": "Inspecting the failure",
        },
    ]
    serialized = json.dumps(rows[0])
    assert "gAAAAA-secret" not in serialized
    assert rows[0]["metadata"]["extracted_replay_used"] is False


def test_requires_a_selected_reasoning_item_in_the_turn():
    events = [
        {
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "Please explain this behavior."},
        },
        {
            "type": "response_item",
            "payload": {"type": "reasoning", "id": "rs-other", "summary": []},
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "phase": "final_answer",
                "message": "This answer is long enough but belongs to an unselected reasoning item.",
            },
        },
    ]

    assert not aligner.aligned_rows_from_events(
        events,
        {"rs-selected"},
        min_assistant_chars=1,
    )


def test_mixed_reasoning_prefers_validated_replay_and_records_provenance():
    events = [
        {
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "Please diagnose the timeout."},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "id": "rs-selected",
                "summary": [{"type": "summary_text", "text": "Public fallback"}],
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "phase": "final_answer",
                "message": "The timeout came from a stalled collective; the focused retry passed.",
            },
        },
    ]

    rows = aligner.aligned_rows_from_events(
        events,
        {"rs-selected"},
        reasoning_source="mixed",
        replay_text_by_id={"rs-selected": "Validated replay candidate"},
        min_assistant_chars=1,
    )

    assert rows[0]["messages"][1]["reasoning_content"] == "Validated replay candidate"
    assert rows[0]["metadata"]["extracted_replay_used"] is True
    assert rows[0]["metadata"]["reasoning_segments"] == [
        {
            "reasoning_id": "rs-selected",
            "source": "replay_text",
            "validation": "strict_text_filter",
            "authenticity": "unverified",
        }
    ]


def test_manifest_only_promotes_strict_recording_text(tmp_path):
    path = tmp_path / "manifest.jsonl"
    valid = "A careful candidate explanation. " * 6
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "rs-valid",
                        "src": "rollout-one",
                        "method": "recording",
                        "text": valid,
                    }
                ),
                json.dumps(
                    {
                        "id": "rs-ciphertext",
                        "src": "rollout-two",
                        "method": "recording",
                        "text": "gAAAAA" + ("x" * 300),
                    }
                ),
                json.dumps(
                    {
                        "id": "rs-rawlog",
                        "src": "rollout-three",
                        "method": "rawlog",
                        "text": valid,
                    }
                ),
            ]
        )
    )

    ids, sources, replay = aligner.load_manifest(path)

    assert ids == {"rs-valid", "rs-ciphertext", "rs-rawlog"}
    assert sources == {"rollout-one", "rollout-two", "rollout-three"}
    assert replay == {"rs-valid": valid.strip()}
