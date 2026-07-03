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
    it = iter(
        StreamingPackedBlocks(ds, FakeTokenizer(), seq_len=32, **kw)
    )
    return [next(it) for _ in range(k)]


def test_stream_yields_full_blocks_forever():
    rows = make_rows(4)
    blocks = stream_blocks(rows, 40, learner_id=0, num_learners=1)
    assert all(b.shape == (32,) and b.dtype == torch.long for b in blocks)
    # 4 rows x ~52 tokens < 40 blocks x 32 tokens: the stream must have cycled.


def test_stream_respects_learner_sharding():
    rows = make_rows(6, words_per_row=31)  # one 31+2-token row -> ~1 block each
    # learner 1 of 3 owns rows 1 and 4; with distinct word lengths the block
    # contents must differ from learner 0's.
    a = torch.stack(stream_blocks(rows, 4, learner_id=0, num_learners=3))
    b = torch.stack(stream_blocks(rows, 4, learner_id=1, num_learners=3))
    assert not torch.equal(a, b)


def test_stream_ranks_get_disjoint_rows():
    rows = make_rows(8)
    r0 = stream_blocks(rows, 3, learner_id=0, num_learners=1, rank=0, world=2)
    r1 = stream_blocks(rows, 3, learner_id=0, num_learners=1, rank=1, world=2)
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
    assert ds[0].shape == (32,)


def test_stream_works_under_dataloader_workers():
    rows = make_rows(8)
    ds = StreamingPackedBlocks(rows, FakeTokenizer(), 0, 1, seq_len=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=2)
    it = iter(loader)
    batches = [next(it) for _ in range(6)]
    assert all(b.shape == (2, 32) for b in batches)
