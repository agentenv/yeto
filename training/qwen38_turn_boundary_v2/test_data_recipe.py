"""CPU contract tests for actual native rows, frozen indices and fresh training."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from . import data, recipe, train
from .test_end_to_end import ASSETS, events, message, selected
from .source_adapters import canonical_messages
from .render import Qwen38Renderer, training_row


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def packed(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


@pytest.fixture(scope="module")
def renderer():
    return Qwen38Renderer(ASSETS)


def make_row(renderer, label="first", group=None, long_prefix=False):
    source = events()
    source[-1]["content"] += " " + label
    if long_prefix:
        source[2]["data"]["block"]["arguments"]["cmd"] = "word " * 270000
    messages, audit = canonical_messages(source)
    result = renderer.render(messages, turn_boundary_audit=audit["turn_boundary_audit"])
    row = training_row(result, group_id=group or "session-"+label, source="trace", provenance={
        "messages_sha256":audit["turn_boundary_audit"]["output_messages_sha256"], "sha256":sha(label.encode())})
    row["metadata"]["retained_training_arrays_sha256"] = sha(json.dumps([row["input_ids"],row["labels"]],separators=(",", ":")).encode())
    return row


def dataset(tmp_path, renderer, rows=None, validation=None):
    rows = rows if rows is not None else [make_row(renderer)]
    validation = validation if validation is not None else [make_row(renderer,"second")]
    manifest={"schema":"qwen38-corrected-turn-dataset/v2", "renderer_identity":renderer.identity,
              "splits":{"train":[],"validation":[]}, "counts":{}, "exclusions":{}, "source_jobs":len(rows)+len(validation)}
    index={"schema":"turn-normalized-token-index/v2", "renderer_identity":renderer.identity,
           "seq_len":262144,"splits":{"train":[],"validation":[]}}
    for split,items in (("train",rows),("validation",validation)):
        if not items:continue
        offset=0;refs=[];blob=b""
        for row in items:
            raw=packed(row)+b"\n";refs.append([offset,len(raw),sha(raw),len(row["input_ids"])]);offset+=len(raw);blob+=raw
        name=split+".jsonl";(tmp_path/name).write_bytes(blob)
        manifest["splits"][split].append({"path":name,"sha256":sha(blob),"rows":len(items)})
        manifest["counts"][split+"/trace"]=len(items)
        index["splits"][split].append({"path":name,"sha256":sha(blob),"size_bytes":len(blob),"rows":refs})
    manifest_path=tmp_path/"manifest.json";manifest_path.write_bytes(packed(manifest))
    index["manifest_sha256"]=data.digest_file(manifest_path)
    index_path=tmp_path/"index.json";index_path.write_bytes(packed(index))
    return manifest_path,index_path,tmp_path/"qualified.json"


def config(tmp_path, renderer):
    manifest,index,qualified=dataset(tmp_path,renderer)
    receipt=data.qualify(manifest,index,qualified)
    run=tmp_path/"run";run.mkdir()
    c=recipe.make_recipe(model_dir='/data/sft_baseline_20260908/models/Qwen3.8-27B',manifest=manifest,
        index=qualified,index_sha256=receipt['index_sha256'],output_dir=run,wandb_project='synthetic-contract-test')
    path=run/"train.json";path.write_bytes(packed(c))
    return c,path


def test_full_native_row_qualification_and_exact_loader_shift(tmp_path,renderer):
    row=make_row(renderer)
    m,i,q=dataset(tmp_path,renderer,rows=[row])
    r=data.qualify(m,i,q)
    assert r['rows_validated']==2 and r['full_mask_audit'] is True and r['split_group_overlap']==0
    loaded=data.TurnTokenDataset(m,index_path=q,index_sha256=r['index_sha256'])[0]
    assert loaded['input_ids']==row['input_ids']
    assert loaded['labels']==row['labels'][1:]+[-100]


def test_row_bytes_cannot_change_after_qualification(tmp_path,renderer):
    m,i,q=dataset(tmp_path,renderer);r=data.qualify(m,i,q)
    loaded=data.TurnTokenDataset(m,index_path=q,index_sha256=r['index_sha256'])
    p=tmp_path/'train.jsonl';raw=p.read_bytes();p.write_bytes(raw.replace(b'session-first',b'session-other'))
    with pytest.raises(ValueError,match='changed'):loaded[0]
    with pytest.raises(ValueError,match='identity'):data.TurnTokenDataset(m,index_path=q,index_sha256=r['index_sha256'])


def test_cross_split_session_overlap_rejected(tmp_path,renderer):
    m,i,q=dataset(tmp_path,renderer,rows=[make_row(renderer,'a',group='same-session')],validation=[make_row(renderer,'b',group='same-session')])
    with pytest.raises(ValueError,match='overlap'):data.qualify(m,i,q)
    assert not q.exists()


def test_distinct_full_sources_with_duplicate_retained_arrays_rejected(tmp_path,renderer):
    # A full-message/source SHA alone cannot detect identical post-cutoff rows.
    a=make_row(renderer,'a',long_prefix=True);b=make_row(renderer,'b',long_prefix=True)
    assert a["metadata"]["provenance"]["messages_sha256"] != b["metadata"]["provenance"]["messages_sha256"]
    assert len(a["input_ids"]) == len(b["input_ids"]) == 262144
    assert a["input_ids"] == b["input_ids"] and a["labels"] == b["labels"]
    m,i,q=dataset(tmp_path,renderer,rows=[a],validation=[b])
    with pytest.raises(ValueError,match='Duplicate.*retained'):data.qualify(m,i,q)
    assert not q.exists()


def test_retained_array_receipt_cannot_be_forged(tmp_path,renderer):
    row=make_row(renderer);row['metadata']['retained_training_arrays_sha256']='0'*64
    m,i,q=dataset(tmp_path,renderer,rows=[row],validation=[])
    with pytest.raises(ValueError,match='[Rr]etained'):data.qualify(m,i,q)


def test_missing_index_bytes_are_not_qualified(tmp_path,renderer):
    m,i,q=dataset(tmp_path,renderer);index=json.loads(i.read_bytes());index['splits']['train'][0]['rows'][0][1]-=1;i.write_bytes(packed(index))
    with pytest.raises(ValueError):data.qualify(m,i,q)


def test_empty_assistant_cannot_pass_loader_mask_validation(renderer):
    row=make_row(renderer)
    # Use an exact syntactically-native assistant header immediately followed by EOT.
    ids=[data.IM_START,8678,198,100,data.IM_END,198]+data.ASSISTANT_PREFIX+[data.IM_END,198]
    row.update(input_ids=ids,labels=[-100]*(len(ids)-2)+[data.IM_END,-100],attention_mask=[1]*len(ids))
    row['metadata']['sequence_audit']['retained_input_tokens']=len(ids)
    with pytest.raises(ValueError,match='Empty assistant'):data.validate_row(row,renderer.identity)


def test_recipe_is_full_one_epoch_with_same_memory_policy(tmp_path,renderer):
    c,_=config(tmp_path,renderer);recipe.require_recipe(c)
    assert c['step_scheduler']['num_epochs']==1 and c['step_scheduler']['global_batch_size']==8
    assert c['step_scheduler']['local_batch_size']==1 and c['step_scheduler'].get('max_steps') is None
    assert c['distributed']['cp_size']==8 and c['distributed']['tp_size']==1
    assert c['distributed']['activation_checkpointing'] and c['distributed']['reshard_after_forward']
    assert c['distributed']['enable_fsdp2_prefetch'] is False
    assert c['optimizer']['lr']==1e-5 and c['optimizer']['master_weight_dtype']=='torch.float32'
    assert c['checkpoint']['cpu_offload'] and c['checkpoint']['is_async'] is False
    assert c['dataloader']['shuffle'] and not c['dataloader']['drop_last']
    assert not any(k in c['dataset'] for k in ['max_samples','max_input_tokens'])
    assert c['initialization']=='pinned-original-base-fresh-optimizer'


@pytest.mark.parametrize('section,key,value',[
    ('distributed','cp_size',4),('distributed','activation_checkpointing',False),
    ('distributed','reshard_after_forward',False),('distributed','enable_fsdp2_prefetch',True),
    ('distributed','defer_fsdp_grad_sync',True),('distributed','sequence_parallel',True),
    ('optimizer','lr',2e-5),('optimizer','_target_','torch.optim.AdamW'),
    ('optimizer','exp_avg_dtype','torch.bfloat16'),('checkpoint','cpu_offload',False),
    ('checkpoint','restore_from','/old/checkpoint'),('dataset','max_samples',2),
    ('step_scheduler','max_steps',2),('step_scheduler','local_batch_size',2),
    ('step_scheduler','global_batch_size',1),('loss_fn','shift',True),
    ('model','state_dict',{}),('model','_target_','other.from_pretrained'),
])
def test_wrong_memory_shift_initialization_or_subset_is_rejected(tmp_path,renderer,section,key,value):
    c,_=config(tmp_path,renderer);c[section][key]=value
    with pytest.raises(ValueError):recipe.require_recipe(c)


def test_old_output_and_alternate_base_rejected_before_dispatch(tmp_path,renderer):
    c,p=config(tmp_path,renderer);checkpoint=Path(c['checkpoint']['checkpoint_dir']);checkpoint.mkdir();(checkpoint/'old').touch()
    with pytest.raises(ValueError,match='fresh'):train.prepare_run(c,config_path=p)
    c['model']['pretrained_model_name_or_path']='/old/step299'
    with pytest.raises(ValueError,match='original base'):train.verify_base(c)


def test_delegation_is_configuration_bound_and_restores_after_error(tmp_path,renderer):
    c,_=config(tmp_path,renderer);old=(train.baseline.require_recipe,train.baseline.contract_identity)
    with pytest.raises(RuntimeError,match='sentinel'):
        with train.bound_baseline(c):
            train.baseline.require_recipe(deepcopy(c))
            modified=deepcopy(c);modified['clip_grad_norm']['max_norm']=0.25
            with pytest.raises(ValueError,match='changed'):train.baseline.require_recipe(modified)
            raise RuntimeError('sentinel')
    assert (train.baseline.require_recipe,train.baseline.contract_identity)==old
    with train.bound_baseline(c):pass


def test_same_contract_resume_identity_binds_loss_scheduler_and_clip(tmp_path,renderer):
    c,_=config(tmp_path,renderer)
    with patch.object(train,'verify_base',return_value={'synthetic':True}),patch.object(train,'_ORIGINAL_IDENTITY',return_value={'old':'identity'}):
        before=train.contract_identity(c)
        for section,key,value in [('loss_fn','shift',True),('clip_grad_norm','max_norm',0.5),('lr_scheduler','lr_warmup_steps',10),('step_scheduler','ckpt_every_steps',10)]:
            changed=deepcopy(c);changed[section][key]=value
            assert train.contract_identity(changed)!=before


def test_tiny_real_source_cpu_rebuild_qualify_and_load(tmp_path):
    from . import prepare_data
    from training.qwen38_no_cot.prepare_data import SEED
    groups={}
    for n in range(10000):
        group="synthetic-session-"+str(n)
        split="validation" if int(hashlib.sha256((SEED+":split:"+group).encode()).hexdigest()[:8],16)%100==0 else "train"
        groups.setdefault(split,group)
        if len(groups)==2:break
    jobs=[]
    for split,group in sorted(groups.items()):
        source=events();source[-1]["content"] += " "+split
        raw=b"".join(packed(event)+b"\n" for event in source)
        path=tmp_path/(split+"-source.jsonl");path.write_bytes(raw)
        jobs.append({"identity":"synthetic-"+split,"path":str(path),"sha256":sha(raw),"format":"canonical","source":"trace","group_id":group})
    source_raw=b"".join(packed(job)+b"\n" for job in jobs)
    inventory=tmp_path/"source_jobs.jsonl";inventory.write_bytes(source_raw)
    output=tmp_path/"rebuilt"
    receipt=prepare_data.rebuild(source_jobs=inventory,source_jobs_sha256=sha(source_raw),tokenizer_dir=str(ASSETS),output=output,workers=1,shards=2)
    assert receipt["accepted"]==2 and receipt["excluded"]==0
    assert receipt["source_accounting_complete"] and receipt["training_ready"] is False
    q=output/"qualified-index.json"
    checked=data.qualify(output/"manifest.json",output/"index.json",q)
    assert checked["rows_validated"]==2 and checked["split_group_overlap"]==0
    for split in ("train","validation"):
        loader=data.TurnTokenDataset(output/"manifest.json",index_path=q,index_sha256=checked["index_sha256"],split=split)
        assert len(loader)==1 and any(t!=-100 for t in loader[0]["labels"])
