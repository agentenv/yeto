from copy import deepcopy
import hashlib,json,os,shutil,sqlite3
from pathlib import Path
import pytest
from cot_filler.core import canonical,digest
from cot_filler.corpus_worker import gap_at
from training.qwen38_turn_boundary_v2.test_masked_full import unsafe_fixture
from training.qwen38_turn_boundary_v2.test_masked import events as boundary_events
from training.qwen38_cot_experimental.test_prepare_full import freeze_fixture
from training.qwen38_cot_experimental.test_render import ASSETS
from . import prepare_masked_full as prep,masked_data as data,masked_recipe as recipe
from . import masked_train as train


def _fail_worker_init(*_):
    raise RuntimeError('fixture worker initializer failure')


@pytest.fixture(autouse=True)
def pinned_export_runtime(monkeypatch):
    monkeypatch.setenv('YETA_TRAINING_IMAGE',prep.RUNTIME_IMAGE)
    monkeypatch.setattr(prep,'_path_on_read_only_mount',lambda _:True)


def test_full_trace_retained_after_only_conflicting_cot_omitted(tmp_path):
    freeze=unsafe_fixture(tmp_path)
    result=prep.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    m=json.loads(Path(result['manifest']).read_text())
    assert sum(m['counts'].values())==3
    assert m['included_valid_candidate_count']==0 and m['omitted_valid_candidate_count']==1
    assert m['excluded_valid_candidate_count']==0 and m['all_frozen_generations_accounted_for']
    assert m['turn_boundary_excluded_sources']=={} and result['index']['omitted_valid_candidate_count']==1
    assert m['cross_arm_no_cot_parity']['verified'] is True
    assert m['cross_arm_no_cot_parity']['adapter_divergence_rows']==0
    assert m['cross_arm_no_cot_parity']['actual_target_tokens']<=m['cross_arm_no_cot_parity']['zero_cot_target_tokens']<=m['cross_arm_no_cot_parity']['baseline_target_tokens']
    assert m['replay_source_membership']=={'schema':'qwen38-exact-replay-membership/v1',
        'verified':True,'expected_selected_sources':1,'accepted_sources':1,
        'excluded_sources':{},'excluded_source_count':0}
    assert m['dynamic_trace_exclusion_audit']['verified'] is True
    assert m['dynamic_trace_exclusion_audit']['observed_record_count']==0
    assert (tmp_path/'out/COMPLETE.json').exists()
    cfg=recipe.make_recipe(model_dir='/data/sft_baseline_20260908/models/Qwen3.8-27B',manifest=result['manifest'],index_path=tmp_path/'out/index.json',index_sha256=result['index']['sha256'],output_dir=tmp_path/'run')
    assert cfg['training_contract']==data.TRAINING_CONTRACT and cfg['optimizer']['lr']==1e-5
    assert cfg['distributed']['cp_size']==8


def test_full_safe_reasoning_stays_and_unknown_sessions_train_only(tmp_path):
    freeze=freeze_fixture(tmp_path,missing_group=True)
    result=prep.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    m=json.loads(Path(result['manifest']).read_text())
    assert m['included_valid_candidate_count']==1 and m['omitted_valid_candidate_count']==0
    assert m['unresolved_session_sources']==2
    for split,shards in m['splits'].items():
        for shard in shards:
            row=json.loads((Path(result['manifest']).parent/shard['path']).read_text());data.validate_row(row,m['renderer_identity'])
            if row['metadata']['provenance'].get('session_identity_verified') is False:assert split=='train'


def test_full_export_requires_explicit_all_train_before_creating_output(tmp_path):
    freeze=freeze_fixture(tmp_path)
    output=tmp_path/'out'
    with pytest.raises(ValueError,match='--all-train'):
        prep.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),
            output=output,workers=1,shard_rows=1)
    assert not output.exists()


@pytest.mark.parametrize('field,value',[('source_path','/wrong/source.jsonl'),
                                         ('source_sha256','0'*64),('source_bytes',1)])
def test_freeze_receipt_redundant_source_identity_is_bound_to_snapshot(tmp_path,field,value):
    freeze=freeze_fixture(tmp_path)
    receipt=json.loads(Path(freeze).read_text());receipt[field]=value
    Path(freeze).write_text(canonical(receipt))
    with pytest.raises(ValueError,match='source identity differs'):
        prep.load_freeze(freeze)


def test_frozen_journal_accounts_preserved_generation_invalid_candidate(tmp_path):
    freeze=freeze_fixture(tmp_path)
    receipt=json.loads(Path(freeze).read_text())
    db=sqlite3.connect(receipt['snapshot_path'])
    gap=db.execute('SELECT * FROM gaps').fetchone()
    candidate=db.execute('SELECT * FROM candidates').fetchone()
    response=db.execute('SELECT * FROM responses').fetchone()
    invalid_gap='generation-invalid-fixture'
    db.execute('INSERT INTO gaps VALUES(?,?,?,?,?,?,?,?)',
        (2,invalid_gap,gap[2],gap[3],gap[4],'generation_invalid',None,'now'))
    db.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,?)',
        (invalid_gap,*candidate[1:]))
    db.execute('INSERT INTO responses VALUES(?,?,?,?,?,?)',
        (2,invalid_gap,response[2],response[3],response[4],'now'))
    db.commit();db.close()
    receipt['snapshot_sha256']=hashlib.sha256(Path(receipt['snapshot_path']).read_bytes()).hexdigest()
    receipt['gap_states']={'review_pending':1,'generation_invalid':1}
    receipt['invalid_candidate_count_preserved']=1
    Path(freeze).write_text(canonical(receipt))
    _,_,jobs=prep.load_freeze(freeze)
    assert len(jobs)==3

    db=sqlite3.connect(receipt['snapshot_path'])
    db.execute("UPDATE gaps SET state='queued' WHERE id=?",(invalid_gap,))
    db.commit();db.close()
    receipt['snapshot_sha256']=hashlib.sha256(Path(receipt['snapshot_path']).read_bytes()).hexdigest()
    receipt['gap_states']={'review_pending':1,'queued':1}
    receipt['invalid_candidate_count_preserved']=0
    Path(freeze).write_text(canonical(receipt))
    with pytest.raises(ValueError,match='Frozen source/candidate membership'):
        prep.load_freeze(freeze)


@pytest.mark.parametrize('mutation',[
    lambda manifest: manifest['cross_arm_no_cot_parity'].__setitem__(
        'actual_target_tokens',manifest['cross_arm_no_cot_parity']['actual_target_tokens']+1),
    lambda manifest: manifest['cross_arm_parity_excluded_sources'].__setitem__('trace',1),
    lambda manifest: manifest['cross_arm_parity_excluded_generations'].__setitem__('trace',1),
])
def test_manifest_rejects_cross_arm_parity_tampering(tmp_path,mutation):
    freeze=unsafe_fixture(tmp_path)
    result=prep.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),
        output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    manifest_path=Path(result['manifest'])
    manifest=json.loads(manifest_path.read_text())
    mutation(manifest)
    tampered=manifest_path.with_name('tampered-manifest.json')
    tampered.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='full-corpus baseline-visible target parity'):
        data.read_manifest(tampered)


def test_manifest_rejects_missing_selected_replay(tmp_path):
    freeze=unsafe_fixture(tmp_path)
    result=prep.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),
        output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    manifest_path=Path(result['manifest'])
    manifest=json.loads(manifest_path.read_text())
    manifest['replay_source_membership']['accepted_sources']-=1
    tampered=manifest_path.with_name('tampered-replay-manifest.json')
    tampered.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='replay source membership'):
        data.read_manifest(tampered)


def test_any_replay_conversion_failure_blocks_complete(tmp_path):
    freeze=freeze_fixture(tmp_path)
    receipt=json.loads(Path(freeze).read_text())
    inventory=Path(receipt['replay_manifests'][0]['path'])
    job=json.loads(inventory.read_text())
    replay_path=Path(job['path'])
    replay_path.write_bytes(b'{invalid-json\n')
    job['sha256']=hashlib.sha256(replay_path.read_bytes()).hexdigest()
    job['identity']='replay:'+job['sha256']
    inventory.write_text(canonical(job)+'\n')
    receipt['replay_manifests'][0]['sha256']=hashlib.sha256(inventory.read_bytes()).hexdigest()
    Path(freeze).write_text(canonical(receipt))
    with pytest.raises(ValueError,match='selected replay membership'):
        prep.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),
            output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    assert (tmp_path/'out/BLOCKED.json').exists()
    assert not (tmp_path/'out/COMPLETE.json').exists()


def dynamic_exclusion_fixture(tmp_path):
    freeze=freeze_fixture(tmp_path)
    receipt=json.loads(Path(freeze).read_text())
    source_path=Path(receipt['source_path'])
    sources=[json.loads(line) for line in source_path.read_text().splitlines()]
    source=sources[0]
    source['events']=boundary_events()
    source['events'].insert(3,{'event_id':'late-comment','kind':'message','role':'assistant',
        'content':'A source comment after the call.'})
    source['gap_targets']=[{'event_id':'a'}]
    gap=gap_at(source,1)
    rows=[canonical(row).encode()+b'\n' for row in sources]
    source_path.write_bytes(b''.join(rows))
    db=sqlite3.connect(receipt['snapshot_path']);offset=0
    for index,(row,raw) in enumerate(zip(sources,rows),1):
        db.execute('UPDATE sources SET offset=?,length=?,row_sha256=?,source_digest=? WHERE id=?',
            (offset,len(raw),hashlib.sha256(raw).hexdigest(),digest(row),index));offset+=len(raw)
    db.execute('UPDATE gaps SET id=?,event_index=?,event_id=?',(gap['id'],1,'a'))
    for table in ('candidates','responses'):
        old=json.loads(db.execute('SELECT data FROM '+table).fetchone()[0])
        old['prompt_hash']=gap['prompt_hash']
        db.execute('UPDATE '+table+' SET gap_id=?,data=?',(gap['id'],canonical(old)))
    db.execute('UPDATE candidates SET prefix_hash=?,lookahead_hash=?,prompt_hash=?',
        tuple(gap[key] for key in ('prefix_hash','lookahead_hash','prompt_hash')))
    identity=json.loads(db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
    identity['source_sha256']=hashlib.sha256(source_path.read_bytes()).hexdigest()
    db.execute("UPDATE meta SET value=? WHERE key='identity'",(canonical(identity),))
    db.commit();db.close()
    receipt.update(source_sha256=identity['source_sha256'],source_bytes=source_path.stat().st_size,
        snapshot_sha256=hashlib.sha256(Path(receipt['snapshot_path']).read_bytes()).hexdigest(),
        original_identity_sha256=digest(identity))
    Path(freeze).write_text(canonical(receipt))
    return freeze


def test_dynamic_trace_exclusion_needs_exact_external_allowlist(tmp_path):
    freeze=dynamic_exclusion_fixture(tmp_path)
    with pytest.raises(ValueError,match='Dynamic trace exclusions differ'):
        prep.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),
            output=tmp_path/'blocked',workers=1,shard_rows=1,all_train=True)
    assert (tmp_path/'blocked/BLOCKED.json').exists()
    assert not (tmp_path/'blocked/COMPLETE.json').exists()
    candidate=tmp_path/'blocked/dynamic-trace-exclusions.candidate.json'
    policy=tmp_path/'approved-dynamic-exclusions.json'
    policy.write_bytes(candidate.read_bytes())
    result=prep.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),
        output=tmp_path/'approved',workers=1,shard_rows=1,all_train=True,
        dynamic_trace_exclusions=policy,
        dynamic_trace_exclusions_sha256=hashlib.sha256(policy.read_bytes()).hexdigest())
    manifest=json.loads(Path(result['manifest']).read_text())
    audit=manifest['dynamic_trace_exclusion_audit']
    assert audit['verified'] is True and audit['observed_record_count']==1
    assert manifest['turn_boundary_excluded_sources']=={'unrepresentable_text_after_tool':1}
    assert (tmp_path/'approved/COMPLETE.json').exists()


def test_changed_source_quality_still_blocks_full_export(tmp_path):
    with pytest.raises(ValueError,match='original-trace/generation exclusions'):
        prep.prepare(freeze_receipt=freeze_fixture(tmp_path,low_quality=True),tokenizer_dir=str(ASSETS),output=tmp_path/'out',workers=1,all_train=True)


def test_runtime_contract_binds_native_gap_modules_not_parent_copies(tmp_path,monkeypatch):
    freeze=unsafe_fixture(tmp_path)
    result=prep.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),
        output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    config=recipe.make_recipe(model_dir=str(tmp_path/'model'),manifest=result['manifest'],
        index_path=tmp_path/'out/index.json',index_sha256=result['index']['sha256'],
        output_dir=tmp_path/'run')
    monkeypatch.setattr(train,'verify_base_assets',lambda _:{'fixture':True})
    monkeypatch.setattr(train,'_path_on_read_only_mount',lambda _:True)
    monkeypatch.setattr(train,'_ORIGINAL_IDENTITY',lambda _:{})
    identity=train.contract_identity(config)
    paths=set(identity['filled_cot_code_sha256'])
    assert 'training/qwen38_native_gap_v3/masked_train.py' in paths
    assert 'training/qwen38_native_gap_v3/masked.py' in paths
    assert not any(path.startswith('training/qwen38_turn_boundary_v2/') for path in paths)


def test_wandb_name_is_unique_to_corrected_manifest(tmp_path):
    freeze=unsafe_fixture(tmp_path)
    result=prep.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),
        output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    config=recipe.make_recipe(model_dir=str(tmp_path/'model'),manifest=result['manifest'],
        index_path=tmp_path/'out/index.json',index_sha256=result['index']['sha256'],
        output_dir=tmp_path/'run')
    manifest_sha=hashlib.sha256(Path(result['manifest']).read_bytes()).hexdigest()
    assert config['wandb']['name']==recipe.WANDB_NAME_PREFIX+manifest_sha[:12]+'-train'
    assert 'masked-turns-v2' not in config['wandb']['name']
    config['wandb']['name']='qwen38-27b-generated-cot-masked-turns-v2-train'
    with pytest.raises(ValueError,match='manifest identity'):
        recipe.require_recipe(config)


def test_production_recipe_rejects_non_full_or_non_all_train_manifest(tmp_path):
    result=prep.prepare(freeze_receipt=unsafe_fixture(tmp_path),tokenizer_dir=str(ASSETS),
        output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    original=json.loads(Path(result['manifest']).read_text())
    for field,value in (('full_export',False),('all_train',False),('internal_validation',True)):
        changed=deepcopy(original);changed[field]=value
        path=tmp_path/'out'/('bad-'+field+'.json');path.write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='complete frozen source export|Corrected full export'):
            recipe.make_recipe(model_dir='/data/sft_baseline_20260908/models/Qwen3.8-27B',
                manifest=path,index_path=tmp_path/'out/index.json',index_sha256=result['index']['sha256'],
                output_dir=tmp_path/('run-'+field))


def _open_dataset(result, split='train'):
    return data.GeneratedMaskedTokenDataset(
        result['manifest'], index_path=Path(result['manifest']).parent/'index.json',
        index_sha256=result['index']['sha256'],
        complete_path=result['complete']['path'], complete_sha256=result['complete']['sha256'],
        split=split)


def test_index_recomputes_manifest_counts_and_token_totals(tmp_path):
    freeze=unsafe_fixture(tmp_path)
    result=prep.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),
        output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    manifest=json.loads(Path(result['manifest']).read_text())
    manifest['token_counts']['input_tokens']+=1
    tampered=tmp_path/'out/tampered-manifest.json'
    tampered.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='aggregate token counts'):
        data.build_index(tampered,tmp_path/'out/tampered-index.json',workers=1)

    manifest=json.loads(Path(result['manifest']).read_text())
    trace=next(key for key in manifest['counts'] if key.endswith('/trace'))
    replay=next(key for key in manifest['counts'] if key.endswith('/replay'))
    manifest['counts'][trace]+=1;manifest['counts'][replay]-=1
    manifest['replay_source_membership']['accepted_sources']-=1
    manifest['replay_source_membership']['expected_selected_sources']-=1
    manifest['end_of_build_input_stability']['selected_replay_files_rehashed']-=1
    tampered2=tmp_path/'out/tampered-counts-manifest.json'
    tampered2.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='selected frozen trace source|source/split row counts'):
        data.build_index(tampered2,tmp_path/'out/tampered-counts-index.json',workers=1)

    manifest=json.loads(Path(result['manifest']).read_text())
    manifest['raw_tool_cardinality_audit']['raw_tool_call_event_content_bytes']+=1
    tampered3=tmp_path/'out/tampered-tools-manifest.json'
    tampered3.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='raw-event tool-cardinality totals'):
        data.build_index(tampered3,tmp_path/'out/tampered-tools-index.json',workers=1)


def test_loader_requires_atomic_complete_and_rejects_index_aggregate_tamper(tmp_path):
    result=prep.prepare(freeze_receipt=unsafe_fixture(tmp_path),tokenizer_dir=str(ASSETS),
        output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    with pytest.raises(ValueError,match='Atomic COMPLETE'):
        data.GeneratedMaskedTokenDataset(result['manifest'],index_path=tmp_path/'out/index.json',
            index_sha256=result['index']['sha256'])
    copied=tmp_path/'out/COMPLETE-tampered.json'
    complete=json.loads(Path(result['complete']['path']).read_text());complete['status']='partial'
    copied.write_text(json.dumps(complete))
    with pytest.raises(ValueError,match='Completion marker'):
        data.GeneratedMaskedTokenDataset(result['manifest'],index_path=tmp_path/'out/index.json',
            index_sha256=result['index']['sha256'],complete_path=copied,
            complete_sha256=hashlib.sha256(copied.read_bytes()).hexdigest())
    index=json.loads((tmp_path/'out/index.json').read_text())
    index['token_counts']['input_tokens']+=1
    bad_index=tmp_path/'out/index-tampered.json';bad_index.write_text(json.dumps(index))
    complete=json.loads(Path(result['complete']['path']).read_text())
    complete['index']={'path':bad_index.name,'sha256':hashlib.sha256(bad_index.read_bytes()).hexdigest()}
    bad_complete=tmp_path/'out/COMPLETE-index-tampered.json';bad_complete.write_text(json.dumps(complete))
    with pytest.raises(ValueError,match='validator/manifest contract'):
        data.GeneratedMaskedTokenDataset(result['manifest'],index_path=bad_index,
            index_sha256=complete['index']['sha256'],complete_path=bad_complete,
            complete_sha256=hashlib.sha256(bad_complete.read_bytes()).hexdigest())


def test_portable_tree_keeps_manifest_index_complete_bytes_and_loads(tmp_path):
    result=prep.prepare(freeze_receipt=unsafe_fixture(tmp_path),tokenizer_dir=str(ASSETS),
        output=tmp_path/'source',workers=1,shard_rows=1,all_train=True)
    destination=tmp_path/'relocated';shutil.copytree(tmp_path/'source',destination)
    for name in ('manifest.json','index.json','COMPLETE.json'):
        assert (destination/name).read_bytes()==(tmp_path/'source'/name).read_bytes()
    manifest=json.loads((destination/'manifest.json').read_text())
    index=json.loads((destination/'index.json').read_text())
    assert all(Path(entry['path']).name==entry['path'] for entries in manifest['splits'].values() for entry in entries)
    assert all(Path(entry['path']).name==entry['path'] for entries in index['splits'].values() for entry in entries)
    checked=data.GeneratedMaskedTokenDataset(destination/'manifest.json',index_path=destination/'index.json',
        index_sha256=result['index']['sha256'],complete_path=destination/'COMPLETE.json',
        complete_sha256=result['complete']['sha256'])
    assert len(checked)==sum(value for key,value in manifest['counts'].items() if key.startswith('train/'))


@pytest.mark.parametrize('bad_path',['/absolute.jsonl','../escape.jsonl','nested/file.jsonl','nested\\file.jsonl'])
def test_manifest_rejects_nonportable_shard_paths(tmp_path,bad_path):
    result=prep.prepare(freeze_receipt=unsafe_fixture(tmp_path),tokenizer_dir=str(ASSETS),
        output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    manifest=json.loads(Path(result['manifest']).read_text())
    first=next(iter(manifest['splits'].values()))[0];first['path']=bad_path
    changed=tmp_path/'out/nonportable.json';changed.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='path, SHA256'):
        data.read_manifest(changed)


def test_manifest_rejects_casefold_shard_collision(tmp_path):
    result=prep.prepare(freeze_receipt=unsafe_fixture(tmp_path),tokenizer_dir=str(ASSETS),
        output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    manifest=json.loads(Path(result['manifest']).read_text())
    split=next(iter(manifest['splits']))
    clone=deepcopy(manifest['splits'][split][0]);clone['path']=clone['path'].upper()
    manifest['splits'][split].append(clone)
    changed=tmp_path/'out/case-collision.json';changed.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='path, SHA256'):
        data.read_manifest(changed)


def test_actual_loader_shift_and_pinned_collator_one_batch(tmp_path):
    torch=pytest.importorskip('torch')
    result=prep.prepare(freeze_receipt=unsafe_fixture(tmp_path),tokenizer_dir=str(ASSETS),
        output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    checked=_open_dataset(result)
    path,offset,size,expected,_=checked.refs[0]
    with path.open('rb') as stream:stream.seek(offset);stored=data._strict_json(stream.read(size))
    loaded=checked[0]
    assert loaded['input_ids']==stored['input_ids']
    assert loaded['labels']==stored['labels'][1:]+[-100]
    batch=data.collate_exact([loaded])
    assert batch['input_ids'].dtype==torch.long and batch['labels'].dtype==torch.long
    n=len(loaded['input_ids'])
    assert batch['input_ids'][0,:n].tolist()==loaded['input_ids']
    assert batch['labels'][0,:n].tolist()==loaded['labels']
    assert batch['attention_mask'][0,:n].tolist()==[1]*n
    assert batch['labels'][0,n:].eq(-100).all().item()
    invalid=deepcopy(stored);invalid['input_ids'][0]=data.MAX_TOKEN_ID+1
    invalid['metadata']['retained_token_ids_sha256']=hashlib.sha256(
        json.dumps(invalid['input_ids'],separators=(',',':')).encode()).hexdigest()
    with pytest.raises(ValueError,match='pinned Qwen vocabulary'):
        data.validate_row(invalid,invalid['metadata']['renderer_identity'])


def test_end_of_build_input_mutation_blocks_all_admission_metadata(tmp_path,monkeypatch):
    freeze=unsafe_fixture(tmp_path)
    original=prep.verify_frozen_inputs
    def mutate_then_verify(**kwargs):
        source=Path(kwargs['identity']['source_path'])
        source.write_bytes(source.read_bytes()+b' ')
        return original(**kwargs)
    monkeypatch.setattr(prep,'verify_frozen_inputs',mutate_then_verify)
    with pytest.raises(ValueError,match='changed during export'):
        prep.prepare(freeze_receipt=freeze,tokenizer_dir=str(ASSETS),
            output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    assert (tmp_path/'out/BLOCKED.json').exists()
    assert not any((tmp_path/'out'/name).exists() for name in ('manifest.json','index.json','COMPLETE.json'))


def test_process_pool_initializer_failure_leaves_sanitized_blocked_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(prep,'init_worker',_fail_worker_init)
    with pytest.raises(Exception):
        prep.prepare(freeze_receipt=unsafe_fixture(tmp_path),tokenizer_dir=str(ASSETS),
            output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    blocked=json.loads((tmp_path/'out/BLOCKED.json').read_text())
    assert blocked['reason']=='build_failed_before_atomic_completion'
    assert blocked['error_type'] in {'BrokenProcessPool','RuntimeError'}
    assert 'fixture worker initializer failure' not in (tmp_path/'out/BLOCKED.json').read_text()
    assert not (tmp_path/'out/COMPLETE.json').exists()


def test_v3_manifest_cannot_enter_corrected_v4_loader(tmp_path):
    result=prep.prepare(freeze_receipt=unsafe_fixture(tmp_path),tokenizer_dir=str(ASSETS),
        output=tmp_path/'out',workers=1,shard_rows=1,all_train=True)
    old=json.loads(Path(result['manifest']).read_text())
    old['schema']='qwen38-generated-cot-native-gap-manifest/v3'
    path=tmp_path/'out/invalid-v3-manifest.json';path.write_text(json.dumps(old))
    with pytest.raises(ValueError,match='distinct experimental'):
        data.read_manifest(path)


def test_strict_json_rejects_nonfinite_numbers():
    with pytest.raises(ValueError,match='Non-finite'):
        data._strict_json(b'{"metric":NaN}')
    with pytest.raises(ValueError,match='Non-finite'):
        data._strict_json(b'{"metric":1e9999}')


def test_full_byte_base_receipt_rehashes_and_rejects_mutation_or_membership(tmp_path,monkeypatch):
    model=(tmp_path/'model').resolve();model.mkdir()
    controls={name:index+1 for index,name in enumerate(data.PINNED_CONTROL_TOKEN_IDS)}
    tokenizer={'model':{'vocab':{'plain':0}},
               'added_tokens':[{'content':name,'id':token_id} for name,token_id in controls.items()]}
    (model/'config.json').write_text('{"synthetic":true}\n')
    (model/'model.safetensors.index.json').write_text(json.dumps(
        {'weight_map':{'weight':'model-00001-of-00001.safetensors'}}))
    shard=model/'model-00001-of-00001.safetensors';shard.write_bytes(b'0123456789abcdef')
    (model/'tokenizer.json').write_text(json.dumps(tokenizer))
    (model/'tokenizer_config.json').write_text('{"synthetic":true}\n')
    (model/'chat_template.jinja').write_text('synthetic template\n')
    records=[{'path':path.name,'bytes':path.stat().st_size,
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(model.iterdir())]
    source_manifest=''.join(item['sha256']+'  ./'+item['path']+'\n' for item in records)
    native={'repository':'Qwen/synthetic','revision':'synthetic-revision',
        'tokenizer_sha256':next(x['sha256'] for x in records if x['path']=='tokenizer.json'),
        'tokenizer_config_sha256':next(x['sha256'] for x in records if x['path']=='tokenizer_config.json'),
        'official_template_sha256':next(x['sha256'] for x in records if x['path']=='chat_template.jinja')}
    receipt={'schema':'qwen38-base-full-byte-verification/v1','status':'passed','root':str(model),
        'repository':native['repository'],'revision':native['revision'],
        'source_manifest_sha256':hashlib.sha256(source_manifest.encode()).hexdigest(),
        'file_count':len(records),'weight_shards':1,'total_bytes':sum(x['bytes'] for x in records),
        'exact_file_membership':True,'full_weight_bytes_rehashed':True,
        'config_sha256':next(x['sha256'] for x in records if x['path']=='config.json'),
        'weight_index_sha256':next(x['sha256'] for x in records if x['path']=='model.safetensors.index.json'),
        'tokenizer_sha256':native['tokenizer_sha256'],
        'tokenizer_config_sha256':native['tokenizer_config_sha256'],
        'chat_template_sha256':native['official_template_sha256'],'files':records}
    receipt_path=(tmp_path/'verification.json').resolve();receipt_path.write_text(json.dumps(receipt))
    binding={'path':str(receipt_path),'sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        'schema':receipt['schema'],'source_manifest_sha256':receipt['source_manifest_sha256'],
        'file_count':len(records),'weight_shards':1,'total_bytes':receipt['total_bytes']}
    monkeypatch.setattr(data,'BASE_ASSET_RECEIPT',binding)
    monkeypatch.setattr(data,'NATIVE_ASSET_IDENTITY',native)
    monkeypatch.setattr(data,'TOKENIZER_ID_COUNT',len(controls)+1)
    monkeypatch.setattr(data,'MAX_TOKEN_ID',len(controls))
    monkeypatch.setattr(data,'PINNED_CONTROL_TOKEN_IDS',controls)
    monkeypatch.setattr(train,'BASE_MODEL_DIR',model)
    monkeypatch.setattr(train,'MODEL_CONFIG_SHA256',receipt['config_sha256'])
    monkeypatch.setattr(train,'MODEL_INDEX_SHA256',receipt['weight_index_sha256'])
    monkeypatch.setattr(train,'_path_on_read_only_mount',lambda _:True)
    train._BASE_VERIFICATION_CACHE.clear()
    monkeypatch.setenv('YETA_TRAINING_IMAGE',train.RUNTIME_IMAGE)
    config={'model':{'pretrained_model_name_or_path':str(model),
        '_target_':'nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained',
        'local_files_only':True,'trust_remote_code':False},
        'dataset_contract':{'base_asset_receipt':binding}}
    verified=train.verify_base_assets(config)
    assert verified['full_weight_bytes_rehashed'] is True
    original=shard.read_bytes();shard.write_bytes(b'X'+original[1:])
    with pytest.raises(ValueError,match='full-byte receipt'):
        train.verify_base_assets(config)
    shard.write_bytes(original)
    extra=model/'unexpected';extra.mkdir()
    with pytest.raises(ValueError,match='regular top-level file'):
        train.verify_base_assets(config)
    extra.rmdir();shard.unlink()
    with pytest.raises(ValueError,match='membership'):
        train.verify_base_assets(config)
