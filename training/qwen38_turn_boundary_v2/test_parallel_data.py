"""Parallel qualification retains the exact serial output and all global gates."""
from copy import deepcopy
import hashlib

import pytest

from . import data
from .test_data_recipe import dataset, make_row, renderer


def test_parallel_index_exact_byte_parity_with_serial(tmp_path, renderer):
    manifest, index, serial = dataset(tmp_path, renderer)
    serial_receipt = data.qualify(manifest, index, serial)
    parallel = tmp_path / 'parallel-qualified.json'
    parallel_receipt = data.qualify(manifest, index, parallel, workers=2)
    assert serial.read_bytes() == parallel.read_bytes()
    assert serial_receipt == parallel_receipt
    assert len(data.TurnTokenDataset(manifest, index_path=parallel,
                                    index_sha256=parallel_receipt['index_sha256'])) == 1


@pytest.mark.parametrize('mutation', ['normalized', 'source', 'retained', 'group'])
def test_parallel_detects_duplicates_and_overlap_across_different_shards(tmp_path, renderer, mutation):
    first, second = make_row(renderer, 'first'), make_row(renderer, 'second')
    if mutation == 'normalized':
        second['metadata']['provenance']['messages_sha256'] = first['metadata']['provenance']['messages_sha256']
    elif mutation == 'source':
        second['metadata']['provenance']['sha256'] = first['metadata']['provenance']['sha256']
    elif mutation == 'retained':
        second = deepcopy(first)
        second['metadata']['provenance']['messages_sha256'] = 'a' * 64
        second['metadata']['provenance']['sha256'] = 'b' * 64
        second['group_id'] = 'distinct-second'
    else:
        second['group_id'] = first['group_id']
    m, i, q = dataset(tmp_path, renderer, rows=[first], validation=[second])
    with pytest.raises(ValueError, match='Duplicate|overlap'):
        data.qualify(m, i, q, workers=2)
    assert not q.exists()


@pytest.mark.parametrize('field,value', [('original_baseline_group_id', 'different-original-group'),
                                        ('original_baseline_split', 'validation')])
def test_parallel_preserves_original_source_group_and_split(tmp_path, renderer, field, value):
    first = make_row(renderer)
    first['metadata']['provenance'][field] = value
    m, i, q = dataset(tmp_path, renderer, rows=[first], validation=[])
    with pytest.raises(ValueError, match='Original baseline source'):
        data.qualify(m, i, q, workers=2)
    assert not q.exists()


def test_parallel_rejects_bad_labels_even_when_hashes_are_rebuilt(tmp_path, renderer):
    first = make_row(renderer)
    first['labels'][0] = first['input_ids'][0]
    m, i, q = dataset(tmp_path, renderer, rows=[first], validation=[])
    with pytest.raises(ValueError):
        data.qualify(m, i, q, workers=2)
    assert not q.exists()


@pytest.mark.parametrize('workers', [0, 65, False, 2.0])
def test_parallel_worker_limit_is_explicit(tmp_path, workers):
    with pytest.raises(ValueError, match='workers'):
        data.qualify(tmp_path/'unused', tmp_path/'unused-index', tmp_path/'never', workers=workers)
