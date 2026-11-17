"""Sample-buffer payload format v2 (opt-in raw / zstd array blobs) next to the default v1."""

import hashlib
import json

import numpy
import pytest

from yeto.rl.engine import buffer_payload as bp
from yeto.rl.engine.buffer_payload import (
    PAYLOAD_CODEC_ENV,
    PAYLOAD_SCHEMA,
    PAYLOAD_SCHEMA_V2,
    ROLLOUT_ROUTED_EXPERTS,
    ROLLOUT_TOP_LOGPROBS,
    ROLLOUT_TOP_LOGPROBS_IDS,
    ROUTED_EXPERTS_PAD,
    DamagedPayload,
    K31_PAYLOAD_BYTES_PER_TOKEN,
    SINGLE_PUT_LIMIT_BYTES,
    GroupWriter,
    decode_group,
    k31_group_payload_bytes,
    encode_group,
    payload_codec,
)

# sha256 of encode_group(_golden_group()) before format v2 existed
GOLDEN_V1 = "b821899b0fdbb2fae4edc97e332339a6b24eb7acca0a2aa028309bc449ee8e23"


def _golden_group():
    rng = numpy.random.default_rng(7)
    out = []
    for i in range(2):
        out.append({"index": i, "group_index": 3, "tokens": [1, 2, 3, 4, 5, 6], "response_length": 3,
                    "loss_mask": [1, 0, 1], "rollout_log_probs": [-0.5, 0.0, -1.25], "reward": 1.0,
                    "status": "completed", "metadata": {"weight_version": "3"},
                    ROLLOUT_ROUTED_EXPERTS: rng.integers(0, 896, (5, 4, 2)).astype(numpy.uint16),
                    ROLLOUT_TOP_LOGPROBS_IDS: rng.integers(0, 163840, (3, 4)).astype(numpy.int32),
                    ROLLOUT_TOP_LOGPROBS: -rng.random((3, 4), dtype=numpy.float32)})
    return out


def _segment(rng, *, trial, segment, segments, prompt, response, layers=6, top_k=4, k=8):
    """One compacted segment of an agentic K3.1-like trial: its own R3 over its own
    stream, its own top-K over its response (tool tokens -1 / -inf)."""
    stream = prompt + response
    routed = rng.integers(0, 896, (stream - 1, layers, top_k)).astype(numpy.uint16)
    routed[:2] = ROUTED_EXPERTS_PAD
    ids = rng.integers(0, 163840, (response, k)).astype(numpy.int32)
    vals = numpy.sort(-rng.exponential(2.0, (response, k)).astype(numpy.float32), axis=1)[:, ::-1].copy()
    mask = [int(x) for x in rng.integers(0, 2, response)]
    for pos, m in enumerate(mask):
        if not m:
            ids[pos], vals[pos] = -1, -numpy.inf
    return {"group_index": 0, "index": trial, "tokens": rng.integers(0, 163840, stream).tolist(),
            "response_length": response, "loss_mask": mask,
            "rollout_log_probs": [float(-x) if m else 0.0 for x, m in zip(rng.random(response), mask)],
            "reward": 1.0, "status": "completed" if segment + 1 < segments else "truncated",
            "metadata": {"weight_version": "4", "compaction": {"schema": 1, "segment_index": segment,
                                                               "num_segments": segments}},
            ROLLOUT_ROUTED_EXPERTS: routed, ROLLOUT_TOP_LOGPROBS_IDS: ids, ROLLOUT_TOP_LOGPROBS: vals,
            "wide": numpy.array([1, 2, 70000], dtype=numpy.int64),  # narrowed, cast back
            "f64": numpy.array([numpy.nan, -numpy.inf, 1.5]), "u8": numpy.arange(3, dtype=numpy.uint8),
            "empty": numpy.zeros((0, 3), dtype=numpy.int32)}


def _compacted_group(trials=3, segments=(1, 2, 4)):
    rng = numpy.random.default_rng(11)
    return [_segment(rng, trial=t, segment=s, segments=segments[t], prompt=40 if s == 0 else 25, response=60)
            for t in range(trials) for s in range(segments[t])]


def _assert_same(expected, actual):
    assert len(expected) == len(actual)
    for a, b in zip(expected, actual):
        assert a.keys() == b.keys()
        for key in a:
            if hasattr(a[key], "dtype"):
                assert (a[key].dtype, a[key].shape) == (b[key].dtype, b[key].shape), key
                assert a[key].tobytes() == b[key].tobytes(), key
                assert b[key].flags.writeable and b[key].flags.c_contiguous, key
            else:
                assert json.dumps(a[key], sort_keys=True) == json.dumps(b[key], sort_keys=True), key


def test_the_default_payload_is_byte_identical_to_v1(monkeypatch):
    monkeypatch.delenv(PAYLOAD_CODEC_ENV, raising=False)
    data = encode_group(_golden_group())
    assert hashlib.sha256(data).hexdigest() == GOLDEN_V1
    assert encode_group(_golden_group(), codec="json") == data
    monkeypatch.setenv(PAYLOAD_CODEC_ENV, "json")
    assert encode_group(_golden_group()) == data
    assert json.loads(data)["schema"] == PAYLOAD_SCHEMA


def test_the_codec_comes_from_the_environment(monkeypatch):
    assert payload_codec({}) == "json"
    assert payload_codec({PAYLOAD_CODEC_ENV: " ZSTD "}) == "zstd"
    with pytest.raises(ValueError, match=PAYLOAD_CODEC_ENV):
        payload_codec({PAYLOAD_CODEC_ENV: "gzip"})
    with pytest.raises(ValueError, match="expected one of"):
        encode_group(_golden_group(), codec="lz4")
    monkeypatch.setenv(PAYLOAD_CODEC_ENV, "zstd")
    assert encode_group(_golden_group()).startswith(bp._V2_MAGIC)


@pytest.mark.parametrize("codec", ["raw", "zstd"])
def test_v2_decodes_to_what_v1_decodes(codec):
    group = _compacted_group()
    expected = decode_group(encode_group(group, codec="json"))
    data = encode_group(group, codec=codec)
    assert data.startswith(bp._V2_MAGIC)
    actual = decode_group(data)
    _assert_same(expected, actual)
    # the compaction metadata and the per-segment R3 rows ride through unchanged
    assert [s["metadata"]["compaction"]["segment_index"] for s in actual] == [0, 0, 1, 0, 1, 2, 3]
    assert all(len(s[ROLLOUT_ROUTED_EXPERTS]) == len(s["tokens"]) - 1 for s in actual)
    # a v2 payload re-encoded as v1 (a relay) gives the v1 bytes
    assert encode_group(actual, codec="json") == encode_group(group, codec="json")


def test_v2_is_smaller_and_zstd_smaller_still():
    rng = numpy.random.default_rng(3)  # K3.1 widths: 92 MoE layers x top-16, top-128 log-probs
    group = [_segment(rng, trial=0, segment=s, segments=2, prompt=40, response=60, layers=92, top_k=16, k=128)
             for s in range(2)]
    v1, raw, zstd = (len(encode_group(group, codec=c)) for c in ("json", "raw", "zstd"))
    assert raw < v1 * 0.8  # no base64 third
    assert zstd < raw * 0.8  # expert ids < 1024: the high byte plane compresses


def test_a_reader_before_v2_refuses_a_v2_payload_as_not_json():
    with pytest.raises(ValueError):
        json.loads(encode_group(_golden_group(), codec="zstd"))


def _v2_parts(data):
    start = len(bp._V2_MAGIC) + bp._V2_HEADER_LEN
    end = start + int.from_bytes(data[len(bp._V2_MAGIC):start], "little")
    return json.loads(data[start:end]), data[end:]


def _v2(header, blobs):
    raw = json.dumps(header).encode()
    return bp._V2_MAGIC + len(raw).to_bytes(bp._V2_HEADER_LEN, "little") + raw + blobs


def _first_array(header):
    return header["samples"][0][ROLLOUT_ROUTED_EXPERTS]


@pytest.mark.parametrize("damage", ["cut_header_length", "cut_header", "cut_blobs", "blob_outside",
                                    "codec", "zstd_frame", "zstd_size", "raw_size", "schema", "not_object"])
def test_a_damaged_v2_payload_is_a_damaged_payload(damage):
    data = encode_group(_golden_group(), codec="zstd")
    header, blobs = _v2_parts(data)
    if damage == "cut_header_length":
        data = data[:len(bp._V2_MAGIC) + 3]
    elif damage == "cut_header":
        data = data[:len(bp._V2_MAGIC) + bp._V2_HEADER_LEN + 10]
    elif damage == "cut_blobs":
        data = data[:-5]
    elif damage == "blob_outside":
        _first_array(header)["__blob__"] = [len(blobs), 4]
        data = _v2(header, blobs)
    elif damage == "codec":
        _first_array(header)["codec"] = "brotli"
        data = _v2(header, blobs)
    elif damage == "zstd_frame":
        offset, nbytes = _first_array(header)["__blob__"]
        data = _v2(header, blobs[:offset] + b"\x00" * nbytes + blobs[offset + nbytes:])
    elif damage == "zstd_size":
        _first_array(header)["shape"] = [6, 4, 2]
        data = _v2(header, blobs)
    elif damage == "raw_size":
        header, blobs = _v2_parts(encode_group(_golden_group(), codec="raw"))
        _first_array(header)["shape"] = [4, 4, 2]
        data = _v2(header, blobs)
    elif damage == "schema":
        header["schema"] = PAYLOAD_SCHEMA
        data = _v2(header, blobs)
    elif damage == "not_object":
        data = _v2([1, 2], b"")
    with pytest.raises(DamagedPayload):
        decode_group(data)


def test_a_v2_header_with_the_v2_schema_decodes():
    header, blobs = _v2_parts(encode_group(_golden_group(), codec="raw"))
    assert header["schema"] == PAYLOAD_SCHEMA_V2
    record = _first_array(header)
    assert record["codec"] == "raw" and record["pad"] == ROUTED_EXPERTS_PAD and "__ndarray__" not in record
    _assert_same(decode_group(encode_group(_golden_group())), decode_group(_v2(header, blobs)))


class _Buffer:
    def __init__(self):
        self.payloads, self.entries = {}, []

    def write_payload(self, sample_id, data):
        self.payloads[sample_id] = data
        return sample_id

    def append(self, entry):
        self.entries.append(entry)


def test_the_group_writer_writes_the_codec_the_environment_names(monkeypatch):
    group = _compacted_group()
    buffer = _Buffer()
    monkeypatch.setenv(PAYLOAD_CODEC_ENV, "zstd")
    entry = GroupWriter(buffer, "s1").write(group, group_key="g0")
    data = buffer.payloads[entry.payload_ref]
    assert data.startswith(bp._V2_MAGIC)
    _assert_same(decode_group(encode_group(group, codec="json")), decode_group(data))
    monkeypatch.delenv(PAYLOAD_CODEC_ENV)
    entry = GroupWriter(buffer, "s1").write(group, group_key="g1")
    assert buffer.payloads[entry.payload_ref] == encode_group(group, codec="json")


def test_the_k31_bound_matches_the_measured_bytes_per_token():
    # 8 trials x 262k: v1 is over one PUT already, zstd just fits; one compaction is over in any codec
    assert k31_group_payload_bytes(8, 262144) > SINGLE_PUT_LIMIT_BYTES
    assert k31_group_payload_bytes(8, 262144, codec="zstd") < SINGLE_PUT_LIMIT_BYTES
    assert k31_group_payload_bytes(8, 262144, max_compactions=1, codec="zstd") > SINGLE_PUT_LIMIT_BYTES
    assert k31_group_payload_bytes(8, 131072, max_compactions=1, codec="zstd") < SINGLE_PUT_LIMIT_BYTES
    # the per-token bound covers a K3.1-width segment in each codec
    rng = numpy.random.default_rng(5)
    seg = _segment(rng, trial=0, segment=0, segments=1, prompt=40, response=400, layers=92, top_k=16, k=128)
    for codec, per_token in K31_PAYLOAD_BYTES_PER_TOKEN.items():
        assert len(encode_group([seg], codec=codec)) <= per_token * len(seg["tokens"]) * 1.05, codec


def test_the_group_writer_logs_a_payload_near_the_put_limit(monkeypatch, capsys):
    monkeypatch.delenv(PAYLOAD_CODEC_ENV, raising=False)
    monkeypatch.setattr(bp, "PAYLOAD_WARN_BYTES", 1000)
    buffer = _Buffer()
    entry = GroupWriter(buffer, "s1").write(_golden_group(), group_key="g0")
    assert "single PUT over 5 GiB" in capsys.readouterr().out
    assert buffer.payloads[entry.payload_ref] == encode_group(_golden_group(), codec="json")
    monkeypatch.setattr(bp, "PAYLOAD_WARN_BYTES", 10**9)
    GroupWriter(buffer, "s1").write(_golden_group(), group_key="g1")
    assert capsys.readouterr().out == ""
