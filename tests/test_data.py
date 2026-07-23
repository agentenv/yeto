"""Streaming tokenization tests using fake tokenizers and in-memory rows."""

import json

import pytest
import torch

from yeto.data import (
    ExactAssistantMaskError,
    StreamingPackedBlocks,
    _row_tokens,
    build_packed_dataset,
)


class FakeTokenizer:
    """Small model-native tokenizer with faithful assistant-mask semantics."""

    name_or_path = "fake/native"
    bos_token_id = 1
    eos_token_id = 2
    chat_template = (
        "{% for message in messages %}"
        "{% if message.role == 'assistant' %}"
        "{% generation %}{{ message.content }}{% endgeneration %}"
        "{% else %}{{ message.content }}{% endif %}"
        "{% endfor %}"
    )

    def __init__(self):
        self.last_apply_kwargs = None
        self.last_messages = None
        self.last_tools = None
        self.last_encoded = None
        self.last_spans = []

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        import zlib

        words = text.split()
        out = {"input_ids": [zlib.crc32(w.encode()) % 5000 + 3 for w in words]}
        if return_offsets_mapping:
            offsets = []
            cursor = 0
            for word in words:
                start = text.index(word, cursor)
                end = start + len(word)
                offsets.append((start, end))
                cursor = end
            out["offset_mapping"] = offsets
        return out

    def get_chat_template(self, chat_template=None, tools=None):
        assert chat_template is None
        return self.chat_template

    @staticmethod
    def _content_text(content):
        if isinstance(content, list):
            return json.dumps(content, sort_keys=True)
        return str(content or "")

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        **kwargs,
    ):
        self.last_messages = messages
        self.last_tools = tools
        self.last_apply_kwargs = dict(kwargs, tokenize=tokenize)
        if not tokenize:
            return " ".join(
                f"native-{message['role']} {self._content_text(message.get('content'))}"
                for message in messages
            )

        ids = [self.bos_token_id]
        mask = [0]
        self.last_spans = []
        role_ids = {"system": 6001, "user": 6002, "assistant": 6003, "tool": 6004}
        for message in messages:
            role = message["role"]
            ids.append(role_ids[role])
            mask.append(0)  # role/control token: the native template owns this choice
            text = self._content_text(message.get("content"))
            if message.get("tool_calls"):
                text += " " + json.dumps(message["tool_calls"], sort_keys=True)
            payload = self(text, add_special_tokens=False)["input_ids"]
            start = len(ids)
            ids.extend(payload)
            mask.extend([1 if role == "assistant" else 0] * len(payload))
            end = len(ids)
            ids.append(self.eos_token_id if role == "assistant" else 6005)
            mask.append(1 if role == "assistant" else 0)
            self.last_spans.append((role, start, end, len(ids) - 1))

        self.last_encoded = {"input_ids": ids, "assistant_masks": mask}
        return self.last_encoded


class LegacyTokenizer:
    """The pre-native whitespace tokenizer, used only by explicit legacy tests."""

    name_or_path = "fake/legacy"
    bos_token_id = 1
    eos_token_id = 2
    chat_template = None

    def __init__(self):
        self.plain_tokenize_calls = 0

    def __call__(self, text, add_special_tokens=False):
        import zlib

        self.plain_tokenize_calls += 1
        return {"input_ids": [zlib.crc32(w.encode()) % 5000 + 3 for w in text.split()]}


def make_rows(n, words_per_row=50):
    return [
        {"messages": [{"role": "user", "content": " ".join(f"w{i}x{j}" for j in range(words_per_row))}]}
        for i in range(n)
    ]


def stream_blocks(ds, k, **kw):
    """First k (input_ids, weights) pairs from a fresh stream."""
    it = iter(
        StreamingPackedBlocks(ds, FakeTokenizer(), seq_len=32, **kw)
    )
    return [next(it) for _ in range(k)]


def stream_ids(ds, k, **kw):
    return [ids for ids, _ in stream_blocks(ds, k, **kw)]


def test_stream_yields_full_blocks_forever():
    rows = make_rows(4)
    blocks = stream_blocks(rows, 40, learner_id=0, num_learners=1, train_on="all")
    assert all(ids.shape == (32,) and ids.dtype == torch.long for ids, _ in blocks)
    assert all(w.shape == (32,) and w.dtype == torch.float32 for _, w in blocks)
    # 4 rows x ~52 tokens < 40 blocks x 32 tokens: the stream must have cycled.


def test_stream_respects_learner_sharding():
    rows = make_rows(6, words_per_row=31)  # one 31+2-token row -> ~1 block each
    # learner 1 of 3 owns rows 1 and 4; with distinct word lengths the block
    # contents must differ from learner 0's.
    a = torch.stack(stream_ids(rows, 4, learner_id=0, num_learners=3, train_on="all"))
    b = torch.stack(stream_ids(rows, 4, learner_id=1, num_learners=3, train_on="all"))
    assert not torch.equal(a, b)


def test_stream_ranks_get_disjoint_rows():
    rows = make_rows(8)
    r0 = stream_ids(rows, 3, learner_id=0, num_learners=1, rank=0, world=2, train_on="all")
    r1 = stream_ids(rows, 3, learner_id=0, num_learners=1, rank=1, world=2, train_on="all")
    assert not any(torch.equal(x, y) for x in r0 for y in r1)


def test_stream_errors_when_oversplit():
    rows = make_rows(1)
    with pytest.raises(ValueError):
        stream_blocks(rows, 1, learner_id=0, num_learners=1, rank=1, world=2)


def test_preload_matches_row_budget():
    rows = make_rows(4)
    ds = build_packed_dataset(rows, FakeTokenizer(), 0, 1, seq_len=32, max_rows=2, train_on="all")
    # 2 rows x 52 tokens = 104 tokens -> 3 full blocks of 32
    assert len(ds) == 3
    ids, weights = ds[0]
    assert ids.shape == (32,)
    assert weights.shape == (32,)


def test_stream_works_under_dataloader_workers():
    rows = make_rows(8)
    ds = StreamingPackedBlocks(rows, FakeTokenizer(), 0, 1, seq_len=32, train_on="all")
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2)
    it = iter(loader)
    batches = [next(it) for _ in range(6)]
    assert all(ids.shape == (2, 32) and w.shape == (2, 32) for ids, w in batches)


# --- per-token loss weights (train_on) ---

CHAT_ROW = {
    "messages": [
        {"role": "user", "content": "u1 u2 u3"},
        {"role": "assistant", "content": "a1 a2 a3"},
    ]
}
def test_native_multi_turn_tools_results_and_multipart_content_are_preserved():
    tools = [
        {
            "type": "function",
            "function": {"name": "weather", "parameters": {"type": "object"}},
        }
    ]
    row = {
        "tools": tools,
        "messages": [
            {"role": "system", "content": "be concise"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "weather here?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is sunny."}],
            },
        ],
    }
    tokenizer = FakeTokenizer()

    ids, weights = _row_tokens(tokenizer, row)

    assert tokenizer.last_messages == row["messages"]
    assert tokenizer.last_tools == tools
    assert tokenizer.last_apply_kwargs == {
        "add_generation_prompt": False,
        "return_dict": True,
        "return_assistant_tokens_mask": True,
        "tokenize": True,
    }
    assert ids == tokenizer.last_encoded["input_ids"]
    assert weights == [float(value) for value in tokenizer.last_encoded["assistant_masks"]]
    assert ids[0] == tokenizer.bos_token_id and weights[0] == 0.0

    for role, start, end, terminator in tokenizer.last_spans:
        expected = 1.0 if role == "assistant" else 0.0
        assert all(value == expected for value in weights[start:end])
        assert weights[terminator] == expected
    assistant_terminators = [
        terminator for role, _start, _end, terminator in tokenizer.last_spans if role == "assistant"
    ]
    assert len(assistant_terminators) == 2
    assert all(ids[index] == tokenizer.eos_token_id for index in assistant_terminators)


def test_native_ids_and_masks_stay_aligned_across_packed_block_boundaries():
    rows = [CHAT_ROW] * 4
    tokenizer = FakeTokenizer()
    row_ids, row_weights = _row_tokens(tokenizer, CHAT_ROW)
    expected_ids = (row_ids * len(rows))
    expected_weights = (row_weights * len(rows))

    ds = build_packed_dataset(rows, tokenizer, 0, 1, seq_len=7)
    flat_ids = ds.blocks.flatten().tolist()
    flat_weights = ds.weights.flatten().tolist()
    valid_pairs = set(zip(expected_ids, expected_weights))
    assert all(pair in valid_pairs for pair in zip(flat_ids, flat_weights))
    assert torch.all(ds.weights.sum(dim=1) > 0)

    stream = iter(StreamingPackedBlocks(rows, FakeTokenizer(), 0, 1, seq_len=7))
    blocks = [next(stream) for _ in range(4)]
    stream_ids = torch.cat([ids for ids, _weights in blocks]).tolist()
    stream_weights = torch.cat([weights for _ids, weights in blocks]).tolist()
    assert all(pair in valid_pairs for pair in zip(stream_ids, stream_weights))
    assert all(weights.sum() > 0 for _ids, weights in blocks)


def test_native_path_does_not_inject_extra_bos_or_eos_tokens():
    tokenizer = FakeTokenizer()
    ids, weights = _row_tokens(tokenizer, CHAT_ROW)

    assert ids == tokenizer.last_encoded["input_ids"]
    assert weights == [float(value) for value in tokenizer.last_encoded["assistant_masks"]]
    assert ids.count(tokenizer.bos_token_id) == 1
    assert ids.count(tokenizer.eos_token_id) == 1


def test_explicit_legacy_mode_preserves_synthetic_roles_and_bos_eos_weights():
    tokenizer = LegacyTokenizer()
    ids, weights = _row_tokens(tokenizer, CHAT_ROW, assistant_mask_mode="legacy")
    assistant_ids = set(tokenizer("<|assistant|> a1 a2 a3")["input_ids"])
    user_ids = set(tokenizer("<|user|> u1 u2 u3")["input_ids"])

    assert ids[0] == tokenizer.bos_token_id and weights[0] == 0.0
    assert ids[-1] == tokenizer.eos_token_id and weights[-1] == 1.0
    for token_id, weight in zip(ids[1:-1], weights[1:-1]):
        assert weight == (1.0 if token_id in assistant_ids else 0.0)
        assert token_id in assistant_ids | user_ids


@pytest.mark.parametrize(
    "template,error",
    [
        (None, "has no native chat template"),
        ("{% for message in messages %}{{ message.content }}{% endfor %}", "does not contain"),
    ],
)
def test_native_mode_rejects_templates_without_exact_mask_support(template, error):
    class UnsupportedTokenizer(LegacyTokenizer):
        chat_template = template

        def get_chat_template(self, chat_template=None, tools=None):
            return self.chat_template

    with pytest.raises(ExactAssistantMaskError, match=error) as exc:
        _row_tokens(UnsupportedTokenizer(), CHAT_ROW)
    assert "--assistant-mask-mode legacy" in str(exc.value)


def test_native_template_failure_never_silently_uses_plain_tokenization():
    class BrokenNativeTokenizer(FakeTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            raise RuntimeError("template rejected tool result")

        def __call__(self, text, add_special_tokens=False):
            raise AssertionError("plain tokenization is a forbidden implicit fallback")

    with pytest.raises(ExactAssistantMaskError, match="template rejected tool result") as exc:
        _row_tokens(BrokenNativeTokenizer(), CHAT_ROW)
    assert "--assistant-mask-mode legacy" in str(exc.value)


def test_native_mode_rejects_missing_or_all_zero_assistant_masks():
    class BadMaskTokenizer(FakeTokenizer):
        def __init__(self, returned):
            super().__init__()
            self.returned = returned

        def apply_chat_template(self, *args, **kwargs):
            return self.returned

    with pytest.raises(ExactAssistantMaskError, match="did not return assistant_masks"):
        _row_tokens(BadMaskTokenizer({"input_ids": [1, 2]}), CHAT_ROW)
    with pytest.raises(ExactAssistantMaskError, match="all-zero assistant mask"):
        _row_tokens(
            BadMaskTokenizer({"input_ids": [1, 2], "assistant_masks": [0, 0]}),
            CHAT_ROW,
        )


@pytest.mark.parametrize(
    "returned,error",
    [
        ([], "instead of a mapping"),
        ({"assistant_masks": [1]}, "did not return input_ids"),
        ({"input_ids": [[1, 2]], "assistant_masks": [0, 1]}, "batched input_ids"),
        ({"input_ids": [1, 2], "assistant_masks": [[0, 1]]}, "batched assistant_masks"),
        ({"input_ids": [1, 2], "assistant_masks": [1]}, "2 input ids but 1"),
        ({"input_ids": [1, 2], "assistant_masks": [0, 0.5]}, "non-binary"),
    ],
)
def test_native_mode_rejects_malformed_ids_and_masks_with_legacy_hint(returned, error):
    class MalformedTokenizer(FakeTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return returned

    with pytest.raises(ExactAssistantMaskError, match=error) as exc:
        _row_tokens(MalformedTokenizer(), CHAT_ROW)
    assert "--assistant-mask-mode legacy" in str(exc.value)


def test_assistant_packing_skips_zero_target_windows():
    row = {
        "messages": [
            {"role": "user", "content": " ".join(f"u{i}" for i in range(60))},
            {"role": "assistant", "content": "a1 a2 a3 a4"},
        ]
    }
    ds = build_packed_dataset([row] * 3, FakeTokenizer(), 0, 1, seq_len=8, train_on="assistant")
    assert len(ds) < 12  # old raw packing would emit roughly every user-only window
    assert torch.all(ds.weights.sum(dim=1) > 0)

    stream = StreamingPackedBlocks([row] * 3, FakeTokenizer(), 0, 1, seq_len=8, train_on="assistant")
    for _, weights in [next(iter(stream)) for _ in range(3)]:
        assert weights.sum() > 0


def test_train_on_all_gives_all_ones():
    rows = [CHAT_ROW] * 4
    # A tokenizer without native mask support retains the historical all-token
    # fallback behavior; assistant_mask_mode is irrelevant to train_on="all".
    ds = build_packed_dataset(rows, LegacyTokenizer(), 0, 1, seq_len=7, train_on="all")
    assert torch.equal(ds.weights, torch.ones_like(ds.weights))
    it = iter(StreamingPackedBlocks(rows, LegacyTokenizer(), 0, 1, seq_len=7, train_on="all"))
    ids, weights = next(it)
    assert torch.equal(weights, torch.ones(7))


def test_train_on_rejects_unknown_mode():
    with pytest.raises(ValueError):
        StreamingPackedBlocks([CHAT_ROW], FakeTokenizer(), 0, 1, seq_len=7, train_on="user")
    with pytest.raises(ValueError):
        build_packed_dataset([CHAT_ROW] * 2, FakeTokenizer(), 0, 1, seq_len=7, train_on="user")


def test_assistant_mask_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="assistant_mask_mode"):
        StreamingPackedBlocks(
            [CHAT_ROW], FakeTokenizer(), 0, 1, seq_len=7, assistant_mask_mode="guess"
        )
    with pytest.raises(ValueError, match="assistant_mask_mode"):
        build_packed_dataset(
            [CHAT_ROW] * 2,
            FakeTokenizer(),
            0,
            1,
            seq_len=7,
            assistant_mask_mode="guess",
        )


def test_load_rows_local_jsonl(tmp_path):
    import json as _json

    from yeto.data import load_rows

    f = tmp_path / "rows.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"hi {i}"}]} for i in range(3)]
    f.write_text("\n".join(_json.dumps(r) for r in rows))
    ds = load_rows(str(f))
    assert len(ds) == 3 and ds[0]["messages"][0]["content"] == "hi 0"
    # A directory of jsonl files loads the same way.
    ds = load_rows(str(tmp_path))
    assert len(ds) == 3


def test_load_rows_local_rejects_unknown_extension(tmp_path):
    import pytest as _pytest

    from yeto.data import load_rows

    f = tmp_path / "rows.csv"
    f.write_text("a,b\n")
    with _pytest.raises(ValueError, match="unsupported data file type"):
        load_rows(str(f))
