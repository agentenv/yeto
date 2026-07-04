import struct

from yeto.fragments import MERGE_RDA, Fragment, FragmentLayout
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
