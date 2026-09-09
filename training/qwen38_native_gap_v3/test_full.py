import hashlib,json
from pathlib import Path
import pytest
from training.qwen38_turn_boundary_v2.test_masked_full import unsafe_fixture
from training.qwen38_cot_experimental.test_prepare_full import freeze_fixture
from training.qwen38_cot_experimental.test_render import ASSETS
from . import prepare_masked_full as prep,masked_data as data,masked_recipe as recipe


def test_full_trace_retained_after_only_conflicting_cot_omitted(tmp_path):
    freeze=unsafe_fixture(tmp_path)
    result=prep.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),output=tmp_path/'out',workers=1,shard_rows=1)
    m=json.loads(Path(result['manifest']).read_text())
    assert sum(m['counts'].values())==3
    assert m['included_valid_candidate_count']==0 and m['omitted_valid_candidate_count']==1
    assert m['excluded_valid_candidate_count']==0 and m['all_frozen_generations_accounted_for']
    assert m['turn_boundary_excluded_sources']=={} and result['index']['omitted_valid_candidate_count']==1
    assert (tmp_path/'out/COMPLETE.json').exists()
    cfg=recipe.make_recipe(model_dir='/data/sft_baseline_20260908/models/Qwen3.8-27B',manifest=result['manifest'],index_path=tmp_path/'out/index.json',index_sha256=result['index']['sha256'],output_dir=tmp_path/'run')
    assert cfg['training_contract']==data.TRAINING_CONTRACT and cfg['optimizer']['lr']==1e-5
    assert cfg['distributed']['cp_size']==8


def test_full_safe_reasoning_stays_and_unknown_sessions_train_only(tmp_path):
    freeze=freeze_fixture(tmp_path,missing_group=True)
    result=prep.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),output=tmp_path/'out',workers=1,shard_rows=1)
    m=json.loads(Path(result['manifest']).read_text())
    assert m['included_valid_candidate_count']==1 and m['omitted_valid_candidate_count']==0
    assert m['unresolved_session_sources']==2
    for split,shards in m['splits'].items():
        for shard in shards:
            row=json.loads(Path(shard['path']).read_text());data.validate_row(row,m['renderer_identity'])
            if row['metadata']['provenance'].get('session_identity_verified') is False:assert split=='train'


def test_changed_source_quality_still_blocks_full_export(tmp_path):
    with pytest.raises(ValueError,match='original-trace/generation exclusions'):
        prep.prepare(freeze_receipt=freeze_fixture(tmp_path,low_quality=True),tokenizer_dir=str(ASSETS),output=tmp_path/'out',workers=1)
