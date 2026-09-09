from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
from unittest.mock import patch

import pytest
from cot_filler.core import canonical,digest
from cot_filler.corpus_worker import gap_at
from training.qwen38_cot_experimental.test_prepare_full import freeze_fixture
from training.qwen38_cot_experimental.test_render import ASSETS
from . import prepare_masked_full as prepare,masked_data as data,masked_recipe as recipe,masked_train as train


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unsafe_fixture(tmp_path):
    path=freeze_fixture(tmp_path)
    receipt=json.loads(path.read_text());source_path=Path(receipt['source_path'])
    sources=[json.loads(line) for line in source_path.read_text().splitlines()]
    sources[0]['events'].insert(1,{'event_id':'preceding-action','kind':'message','role':'assistant','content':'I will inspect it now.'})
    gap=gap_at(sources[0],2)
    rows=[canonical(s).encode()+b'\n' for s in sources];source_path.write_bytes(b''.join(rows))
    db=sqlite3.connect(receipt['snapshot_path']);offset=0
    for i,(s,raw) in enumerate(zip(sources,rows),1):
        db.execute('UPDATE sources SET offset=?,length=?,row_sha256=?,source_digest=? WHERE id=?',(offset,len(raw),hashlib.sha256(raw).hexdigest(),digest(s),i));offset+=len(raw)
    db.execute('UPDATE gaps SET id=?,event_index=?',(gap['id'],2))
    for table in ('candidates','responses'):
        old=json.loads(db.execute('SELECT data FROM '+table).fetchone()[0]);old['prompt_hash']=gap['prompt_hash']
        db.execute('UPDATE '+table+' SET gap_id=?,data=?',(gap['id'],canonical(old)))
    db.execute('UPDATE candidates SET prefix_hash=?,lookahead_hash=?,prompt_hash=?',tuple(gap[k] for k in ('prefix_hash','lookahead_hash','prompt_hash')))
    identity=json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0]);identity['source_sha256']=sha(source_path)
    db.execute("UPDATE meta SET value=? WHERE key='identity'",(canonical(identity),));db.commit();db.close()
    receipt.update(source_sha256=sha(source_path),snapshot_sha256=sha(receipt['snapshot_path']),original_identity_sha256=digest(identity));path.write_text(canonical(receipt))
    return path


@pytest.fixture
def prepared(tmp_path):
    freeze=freeze_fixture(tmp_path)
    result=prepare.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),output=tmp_path/'out',workers=1,all_train=True,shard_rows=1)
    manifest=json.loads(Path(result['manifest']).read_text())
    return tmp_path,result,manifest


def test_corrected_full_source_renderer_index_loader_and_once_shift(prepared):
    folder,result,manifest=prepared
    assert sum(manifest['counts'].values())==3
    assert manifest['included_valid_candidate_count']==1
    assert manifest['excluded_valid_candidate_count']==0
    assert manifest['all_frozen_generations_accounted_for'] is True
    assert manifest['turn_boundary_excluded_sources']=={}
    index=folder/'out/index.json'
    dataset=data.GeneratedMaskedTokenDataset(result['manifest'],index_path=index,index_sha256=sha(index),split='train')
    assert len(dataset)==3
    for shard in manifest['splits']['train']:
        row=json.loads(Path(shard['path']).read_text());shifted=data.validate_row(row,manifest['renderer_identity'])
        assert shifted['input_ids']==row['input_ids']
        assert shifted['labels']==row['labels'][1:]+[-100]
        assert row['metadata']['provenance']['generated_cot']['reviewed'] is False
        bad=deepcopy(row);bad['metadata']['turn_boundary_audit']['excluded']=True
        with pytest.raises(ValueError):data.validate_row(bad,manifest['renderer_identity'])


def test_unsafe_later_reasoning_is_whole_source_exclusion_with_exact_frozen_count(tmp_path):
    freeze=unsafe_fixture(tmp_path)
    before=sha(json.loads(freeze.read_text())['snapshot_path'])
    result=prepare.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),output=tmp_path/'out',workers=1,all_train=True,shard_rows=1)
    manifest=json.loads(Path(result['manifest']).read_text())
    assert manifest['included_valid_candidate_count']==0 and manifest['excluded_valid_candidate_count']==1
    assert manifest['turn_boundary_excluded_sources']=={'reasoning_after_visible_content':1}
    assert manifest['turn_boundary_excluded_generations']=={'reasoning_after_visible_content':1}
    assert manifest['all_frozen_generations_accounted_for'] is True
    assert manifest['all_frozen_generations_included_before_cutoff'] is False
    assert sum(manifest['counts'].values())==2
    assert sha(json.loads(freeze.read_text())['snapshot_path'])==before
    assert (tmp_path/'out/COMPLETE.json').exists()


def test_low_quality_cannot_be_reclassified_as_boundary_exclusion(tmp_path):
    freeze=freeze_fixture(tmp_path,low_quality=True)
    with pytest.raises(ValueError,match='original-trace/generation exclusions'):
        prepare.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),output=tmp_path/'out',workers=1,all_train=True)
    assert not (tmp_path/'out/COMPLETE.json').exists()


def make_config(prepared):
    folder,result,manifest=prepared
    index=folder/'out/index.json'
    return recipe.make_recipe(model_dir='/data/sft_baseline_20260908/models/Qwen3.8-27B',manifest=result['manifest'],index_path=index,index_sha256=sha(index),output_dir=folder/'run')


def test_recipe_retains_exact_cp8_full_run_and_synchronous_checkpoints(prepared):
    config=make_config(prepared);recipe.require_recipe(config)
    assert config['distributed']['cp_size']==8
    assert config['step_scheduler']['global_batch_size']==8 and config['step_scheduler']['local_batch_size']==1
    assert config['optimizer']['lr']==1e-5 and config['dataset']['seq_len']==262144
    assert config['checkpoint']['cpu_offload'] is True and config['checkpoint']['is_async'] is False
    assert 'max_samples' not in config['dataset'] and config['step_scheduler'].get('max_steps') is None
    assert 'validation_dataset' not in config


@pytest.mark.parametrize('section,key,value',[
    ('loss_fn','shift_labels',True),('model','state_dict',{}),('optimizer','_target_','wrong.Optimizer'),
    ('clip_grad_norm','max_norm',.5),('distributed','cp_size',4),('checkpoint','is_async',True)])
def test_hidden_runtime_overrides_rejected(prepared,section,key,value):
    config=make_config(prepared);config[section][key]=value
    with pytest.raises(ValueError):recipe.require_recipe(config)


def test_wrapper_restores_original_baseline_guards_on_exception(prepared):
    config=make_config(prepared);path=prepared[0]/'config.json';path.write_text(json.dumps(config))
    original=(train.baseline.require_recipe,train.baseline.contract_identity)
    with pytest.raises(RuntimeError,match='deliberate'):
        with train.bound_baseline(config,path,'train'):
            assert train.baseline.require_recipe is not original[0]
            raise RuntimeError('deliberate')
    assert (train.baseline.require_recipe,train.baseline.contract_identity)==original


def test_unknown_session_sources_stay_train_only_under_known_session_split(tmp_path):
    freeze=freeze_fixture(tmp_path,missing_group=True)
    result=prepare.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),output=tmp_path/'out',workers=1,shard_rows=1)
    manifest=json.loads(Path(result['manifest']).read_text())
    assert manifest['all_train'] is False and manifest['unresolved_session_sources']==2
    assert manifest['unknown_session_policy']=='capture-identity-train-only-no-holdout-claim/v2'
    for split,shards in manifest['splits'].items():
        for shard in shards:
            row=json.loads(Path(shard['path']).read_text())
            if row['metadata']['provenance'].get('session_identity_verified') is False:
                assert split=='train'
            else: assert split==prepare.split_for_group(row['group_id'])


def test_full_native_mask_validator_rejects_wrong_roles_headers_body_and_eot(prepared):
    _,_,manifest=prepared
    row=json.loads(Path(manifest['splits']['train'][0]['path']).read_text())
    data.validate_native_mask(row)
    ids,labels=row['input_ids'],row['labels']
    user=next(i for i in range(len(ids)-3) if ids[i:i+3]==[248045,846,198])+3
    assistant=next(i for i in range(len(ids)-3) if ids[i:i+3]==[248045,74455,198])
    action=next(i for i,t in enumerate(labels) if t!=-100)
    eot=next(i for i,t in enumerate(ids) if t==248046 and labels[i]!=-100)
    for i in (user,assistant,action,eot):
        bad=deepcopy(row);bad['labels'][i]=-100 if labels[i]!=-100 else ids[i]
        with pytest.raises(ValueError,match='Saved labels'):
            data.validate_native_mask(bad)


def test_exact_replay_selection_preserves_baseline_group_and_rejects_changed_identity(tmp_path):
    group='original-replay-session'
    job={'kind':'replay','source':'replay','identity':'replay:'+'a'*64,'sha256':'a'*64,'format':'rollout'}
    selected={**job,'original_baseline_group_id':group,'original_baseline_split':prepare.split_for_group(group)}
    path=tmp_path/'selected.jsonl';path.write_text(json.dumps(selected)+'\n')
    extra={**job,'identity':'replay:'+'b'*64,'sha256':'b'*64}
    kept,receipt=prepare.select_baseline_replay([job,extra],path,sha(path))
    assert len(kept)==1 and kept[0]['original_baseline_group_id']==group
    assert receipt['selected_replay_sources']==1 and receipt['excluded_replay_sources']==1
    with pytest.raises(ValueError,match='selection changed'):
        prepare.select_baseline_replay([job,extra],path,'0'*64)
