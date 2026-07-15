import struct

from yeto.fragments import MERGE_ISO, MERGE_RDA, Fragment, FragmentLayout
from yeto.layout_metadata import build_fragment_order_metadata
from yeto.protocol import DTYPE_F32, encode_hello


def test_encode_hello_appends_optional_layout_metadata():
    layout = FragmentLayout([Fragment(MERGE_RDA, [("x.weight", 4)])])
    old = encode_hello(7, DTYPE_F32, layout, 2)
    meta = {"task": "nava", "fragments": []}
    new = encode_hello(7, DTYPE_F32, layout, 2, meta)

    assert new.startswith(old)
    (n,) = struct.unpack_from("<I", new, len(old))
    raw = new[len(old) + 4 :]
    assert n == len(raw)
    assert b'"task":"nava"' in raw


def test_encode_hello_iso_fragment_carries_row_col_pairs():
    # Wire contract with syncer/src/state.rs Layout::decode: iso fragments
    # append one (rows, cols) u64 pair per tensor, in tensor order, right
    # after the numels array; rda fragments stay byte-identical.
    layout = FragmentLayout(
        [
            Fragment(MERGE_RDA, [("x.weight", 4)]),
            Fragment(
                MERGE_ISO,
                [("a.weight", 6), ("b.weight", 8)],
                shapes={"a.weight": (2, 3), "b.weight": (4, 2)},
            ),
        ]
    )
    payload = encode_hello(7, DTYPE_F32, layout, 2)
    off = struct.calcsize("<IBI")  # learner_id, dtype, num_fragments
    mode, n = struct.unpack_from("<BI", payload, off)
    assert (mode, n) == (MERGE_RDA, 1)
    off += struct.calcsize("<BI") + n * 8  # no shape block for rda
    mode, n = struct.unpack_from("<BI", payload, off)
    assert (mode, n) == (MERGE_ISO, 2)
    off += struct.calcsize("<BI")
    assert struct.unpack_from("<2Q", payload, off) == (6, 8)
    off += n * 8
    assert struct.unpack_from("<4Q", payload, off) == (2, 3, 4, 2)
    off += 2 * n * 8
    (num_streams,) = struct.unpack_from("<H", payload, off)
    assert num_streams == 2
    assert len(payload) == off + 2


def test_fragment_order_metadata_preserves_exact_learner_tensor_order():
    layout = FragmentLayout(
        [
            Fragment(MERGE_RDA, [("z.weight", 3), ("a.weight", 5)]),
            Fragment(MERGE_RDA, [("m.weight", 7)]),
        ]
    )
    metadata = build_fragment_order_metadata(layout)
    assert [
        [tensor["name"] for tensor in fragment["tensors"]]
        for fragment in metadata["fragments"]
    ] == [["z.weight", "a.weight"], ["m.weight"]]
    assert [
        [tensor["numel"] for tensor in fragment["tensors"]]
        for fragment in metadata["fragments"]
    ] == [[3, 5], [7]]
