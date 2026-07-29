from scripts import filter_codex_training_jsonl as filtering


def test_rejects_fernet_blob_from_user_or_assistant():
    blob = "gAAAAA" + "a" * 120

    assert not filtering.is_good_user(
        f"Decrypt this: {blob}", min_chars=1, max_chars=1_000
    )
    assert not filtering.is_good_assistant(
        f"Recovered value: {blob}", min_chars=1, max_chars=1_000
    )


def test_accepts_normal_debugging_pair():
    user = "Why does this test pass alone but fail in the full suite?"
    assistant = (
        "Check for shared global state, environment mutations, and fixtures that do not "
        "restore their changes. Then bisect the preceding tests to find the contaminating case."
    )

    assert filtering.is_good_user(user, min_chars=1, max_chars=1_000)
    assert filtering.is_good_assistant(assistant, min_chars=1, max_chars=1_000)
