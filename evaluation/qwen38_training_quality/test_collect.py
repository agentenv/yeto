import hashlib
import json
from pathlib import Path
import shutil
import pytest
from . import collect,native,protocol,suite
from .prepare import verify_run
from .test_quality import frozen,prepared,outcomes


def setup_collector(prepared,tmp_path,monkeypatch):
    source,inventory=outcomes(prepared,tmp_path);manifest=verify_run(prepared)
    hf=prepared/'hf';hf.mkdir();(hf/'export-receipt.json').write_text('{}')
    launch=prepared/'launch';launch.mkdir();(launch/'native-qualification-tokens.json').write_text('{}')
    cpu={'native_template_sha256':protocol.TEMPLATE_SHA,'qualified_mode':'qwen3_6',
        'initial_explicit_xhigh_parity':True,'incremental_tool_result_parity':True,'thinking_prefix_correct':True}
    proof={'schema':'yeta.qwen38-quality-live/v1','checkpoint':manifest['checkpoint'],'model_name':manifest['model_name'],
        'native_template_sha256':protocol.TEMPLATE_SHA,'cpu':cpu,
        'actual_initial_native_xhigh_parity':True,'actual_incremental_tool_result_parity':True,
        'supplied_synthetic_tool_history':True,'model_action_success_required':False,
        'excluded_from_quality_score':True,'qualification_requests':2,
        'verifier_sha256':suite.digest(native.__file__),'shared_verifier_sha256':suite.digest(native.shared.__file__),
        'export_receipt_sha256':suite.digest(hf/'export-receipt.json'),
        'token_record_sha256':suite.digest(launch/'native-qualification-tokens.json')}
    (launch/'native-live-proof.json').write_text(json.dumps(proof))
    bundles=[]
    for row in inventory['outcomes']:
        trial=prepared/'jobs/quality'/(row['task_id']+'__original');trial.mkdir(parents=True)
        shutil.copyfile(row['result']['path'],trial/'result.json')
        if row.get('trajectory'):
            (trial/'agent').mkdir();shutil.copyfile(row['trajectory']['path'],trial/'agent/trajectory.json')
        if row.get('bundle'):
            bundle=json.loads(Path(row['bundle']['path']).read_bytes())
            bundle['segments'][0]['messages']=[{'role':'user','content':'Synthetic task.'},{'role':'assistant','content':'Synthetic response.'}]
            bundle['segments'][0]['tokens'][:2]=[1,2];bundle['segments'][0]['tools']=[]
            target=prepared/'gateway-artifacts/quality/group'/(row['task_id']+'-bundle.json');target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(bundle));bundles.append({'trial_name':trial.name,'bundle':str(target)})
    (prepared/'gateway-artifacts/quality/group/manifest.json').write_text(json.dumps({'final_trials':bundles,'attempts':bundles}))
    class Tito:
        def _render_messages(self,*args,**kwargs):return 'same-native-input'
        def _encode_text(self,text):return [1,2]
    monkeypatch.setattr(native.shared,'runtime',lambda model:(None,Tito()))
    return inventory


def test_collector_original16_results_native_checks_and_automatic_summary(prepared,tmp_path,monkeypatch):
    setup_collector(prepared,tmp_path,monkeypatch)
    summary=collect.collect(prepared)
    assert summary['complete_coverage'] and summary['outcome_count']==16
    assert summary['counts']['passed']==5 and summary['counts']['infrastructure_failure']==1
    assert summary['counts']['native_parity_verified']==15
    assert (prepared/'outcomes.json').is_file() and (prepared/'quality-summary.json').is_file()
    with pytest.raises(FileExistsError):collect.collect(prepared)


def test_collector_rejects_multiple_attempts_without_bestof(prepared,tmp_path,monkeypatch):
    setup_collector(prepared,tmp_path,monkeypatch)
    original=next((prepared/'jobs/quality').glob('*/result.json'))
    duplicate=original.parent.with_name(original.parent.name+'-retry');duplicate.mkdir()
    shutil.copyfile(original,duplicate/'result.json')
    with pytest.raises(ValueError,match='Multiple attempts'):collect.collect(prepared)


def test_collector_does_not_claim_parity_when_saved_input_mismatches(prepared,tmp_path,monkeypatch):
    setup_collector(prepared,tmp_path,monkeypatch)
    path=next((prepared/'gateway-artifacts/quality/group').glob('*-bundle.json'))
    bundle=json.loads(path.read_bytes());bundle['segments'][0]['tokens'][0]=999;path.write_text(json.dumps(bundle))
    summary=collect.collect(prepared)
    assert summary['counts']['native_parity_verified']==14 and not summary['fully_native_verified']
    assert summary['counts']['passed']==5
