"""Streaming tokenization tests using a fake tokenizer and in-memory rows."""

import pytest
import torch

from yeto.data import StreamingPackedBlocks, build_packed_dataset


class FakeTokenizer:
    """Whitespace tokenizer: token id = crc of the word (deterministic, no I/O)."""

    bos_token_id = 1
    eos_token_id = 2
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        import zlib

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
    blocks = stream_blocks(rows, 40, learner_id=0, num_learners=1)
    assert all(ids.shape == (32,) and ids.dtype == torch.long for ids, _ in blocks)
    assert all(w.shape == (32,) and w.dtype == torch.float32 for _, w in blocks)
    # 4 rows x ~52 tokens < 40 blocks x 32 tokens: the stream must have cycled.


def test_stream_respects_learner_sharding():
    rows = make_rows(6, words_per_row=31)  # one 31+2-token row -> ~1 block each
    # learner 1 of 3 owns rows 1 and 4; with distinct word lengths the block
    # contents must differ from learner 0's.
    a = torch.stack(stream_ids(rows, 4, learner_id=0, num_learners=3))
    b = torch.stack(stream_ids(rows, 4, learner_id=1, num_learners=3))
    assert not torch.equal(a, b)


def test_stream_ranks_get_disjoint_rows():
    rows = make_rows(8)
    r0 = stream_ids(rows, 3, learner_id=0, num_learners=1, rank=0, world=2)
    r1 = stream_ids(rows, 3, learner_id=0, num_learners=1, rank=1, world=2)
    assert not any(torch.equal(x, y) for x in r0 for y in r1)


def test_stream_errors_when_oversplit():
    rows = make_rows(1)
    with pytest.raises(ValueError):
        stream_blocks(rows, 1, learner_id=0, num_learners=1, rank=1, world=2)


def test_preload_matches_row_budget():
    rows = make_rows(4)
    ds = build_packed_dataset(rows, FakeTokenizer(), 0, 1, seq_len=32, max_rows=2)
    # 2 rows x 52 tokens = 104 tokens -> 3 full blocks of 32
    assert len(ds) == 3
    ids, weights = ds[0]
    assert ids.shape == (32,)
    assert weights.shape == (32,)


def test_stream_works_under_dataloader_workers():
    rows = make_rows(8)
    ds = StreamingPackedBlocks(rows, FakeTokenizer(), 0, 1, seq_len=32)
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
# Fallback segments tokenize to 4 tokens each ("<|role|>" + 3 words); with
# BOS/EOS every row is 10 tokens, so seq_len=7 puts a block boundary inside
# the assistant span of the first row.


def expected_weight_by_id():
    tok = FakeTokenizer()
    one_ids = set(tok("<|assistant|> a1 a2 a3")["input_ids"]) | {tok.eos_token_id}
    zero_ids = set(tok("<|user|> u1 u2 u3")["input_ids"]) | {tok.bos_token_id}
    assert not one_ids & zero_ids  # distinct words -> distinct fake ids
    return one_ids, zero_ids


def check_weights_match_ids(ids, weights, one_ids, zero_ids):
    for t, w in zip(ids.tolist(), weights.tolist()):
        assert t in one_ids or t in zero_ids
        assert w == (1.0 if t in one_ids else 0.0)


def test_assistant_weights_stay_aligned_across_block_boundary():
    rows = [CHAT_ROW] * 4
    one_ids, zero_ids = expected_weight_by_id()
    it = iter(StreamingPackedBlocks(rows, FakeTokenizer(), 0, 1, seq_len=7))
    blocks = [next(it) for _ in range(4)]  # 4 x 7 tokens spans several 10-token rows
    for ids, weights in blocks:
        check_weights_match_ids(ids, weights, one_ids, zero_ids)
    # rows straddle block boundaries, so some block must mix 0- and 1-weights
    assert any(0.0 < w.mean() < 1.0 for _, w in blocks)


def test_assistant_weights_in_preload_mode():
    rows = [CHAT_ROW] * 4
    one_ids, zero_ids = expected_weight_by_id()
    ds = build_packed_dataset(rows, FakeTokenizer(), 0, 1, seq_len=7, train_on="assistant")
    for i in range(len(ds)):
        ids, weights = ds[i]
        check_weights_match_ids(ids, weights, one_ids, zero_ids)


def test_train_on_all_gives_all_ones():
    rows = [CHAT_ROW] * 4
    ds = build_packed_dataset(rows, FakeTokenizer(), 0, 1, seq_len=7, train_on="all")
    assert torch.equal(ds.weights, torch.ones_like(ds.weights))
    it = iter(StreamingPackedBlocks(rows, FakeTokenizer(), 0, 1, seq_len=7, train_on="all"))
    ids, weights = next(it)
    assert torch.equal(weights, torch.ones(7))


def test_train_on_rejects_unknown_mode():
    with pytest.raises(ValueError):
        StreamingPackedBlocks([CHAT_ROW], FakeTokenizer(), 0, 1, seq_len=7, train_on="user")
    with pytest.raises(ValueError):
        build_packed_dataset([CHAT_ROW] * 2, FakeTokenizer(), 0, 1, seq_len=7, train_on="user")


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
