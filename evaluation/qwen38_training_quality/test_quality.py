from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from . import protocol, suite
from .prepare import prepare, verify_run
from .aggregate import aggregate, INVENTORY_SCHEMA


@pytest.fixture
def frozen(tmp_path):
    path=tmp_path/'frozen';suite.build(path);return path


@pytest.fixture
def prepared(tmp_path,frozen):
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps({'models':[{'slug':'preserved-old-model',
        'display_name':'preserved-old-model','supports_reasoning_summaries':True}]}))
    identity=protocol.checkpoint_identity(arm='no-cot-turn-v2',completed_updates=25,
        checkpoint_sha256='a'*64,cumulative_supervised_tokens=1500000,cumulative_input_tokens=12000000)
    output=tmp_path/'prepared'
    prepare(suite_dir=frozen,output=output,remote_root='/data/evals/quality-no-cot-u25',
        checkpoint=identity,model_catalog=catalog)
    return output


def save(path,value):
    path.write_text(json.dumps(value));return {'path':str(path),'sha256':suite.digest(path)}


def outcomes(prepared,tmp_path):
    run=verify_run(prepared)
    rows=[]
    for index,task in enumerate(run['task_ids']):
        passed=index<5;infra=index==15;cap=index==14;tools=index<7
        result={'id':'original-trial-'+str(index),'finished_at':'2026-09-09T00:00:00Z',
            'config':{'task':{'path':'/data/evals/suite/tasks/'+task,'source':suite.SOURCE},
                      'agent':{'name':'codex','model_name':run['model_name'],
                          'kwargs':{'version':'0.142.5','reasoning_effort':'xhigh',
                           'model_context_window':262144,'model_auto_compact_token_limit':196608}}},
            'verifier_result':{'rewards':{'reward':1 if passed else 0}},
            'exception_info':{'exception_type':'EnvironmentStartError'} if infra else None}
        if infra:result['verifier_result']=None
        row={'task_id':task,'result':save(tmp_path/(task+'-result.json'),result)}
        if not infra:
            calls=[{'tool_call_id':'call-1','function_name':'exec_command'}] if tools else []
            trajectory={'steps':[{'source':'agent','llm_call_count':2 if tools else 1,
                'tool_calls':calls,'metrics':{'prompt_tokens':1000},
                'observation':{'results':[{'source_call_id':'call-1','content':'redacted'}] if tools else []}}]}
            row['trajectory']=save(tmp_path/(task+'-trajectory.json'),trajectory)
            bundle={'segments':[{'tokens':[248045,12,33,248046],'full_loss_mask':[0,0,1,1],
                'finish_reason':'length' if cap else 'stop'}],
                'warnings':['invalid_tool_call parser error'] if index==6 else [],'failures':[]}
            row['bundle']=save(tmp_path/(task+'-bundle.json'),bundle)
            proof={'schema':'yeta.qwen38-quality-native-parity/v1',
                'checkpoint_manifest_sha256':run['checkpoint']['checkpoint_manifest_sha256'],
                'result_sha256':row['result']['sha256'],'bundle_sha256':row['bundle']['sha256'],
                'native_template_sha256':protocol.TEMPLATE_SHA,
                'actual_initial_native_xhigh_parity':True,'actual_incremental_tool_result_parity':True}
            row['native_parity']=save(tmp_path/(task+'-parity.json'),proof)
        rows.append(row)
    inventory={'schema':INVENTORY_SCHEMA,'run_sha256':suite.digest(prepared/'run.json'),
        'first_attempts_only':True,'best_of_selection':False,'outcomes':rows}
    path=tmp_path/'outcomes.json';path.write_text(json.dumps(inventory));return path,inventory


def test_fixed_schedule_and_actual_step_index():
    assert protocol.checkpoints_through(424)==[25,50,100,200,300,400]
    assert protocol.next_checkpoint(24)==25
    assert protocol.next_checkpoint(25)==50
    assert protocol.next_checkpoint(50)==100
    assert protocol.next_checkpoint(100)==200
    assert not protocol.scheduled(24) and not protocol.scheduled(True)
    assert protocol.checkpoint_identity(arm='masked-cot-native-gap-v3',completed_updates=50,
        checkpoint_sha256='b'*64,cumulative_supervised_tokens=123)['zero_based_checkpoint_step']==49


@pytest.mark.parametrize('kwargs',[{'arm':'old-baseline'},{'completed_updates':24},
    {'checkpoint_sha256':'not-pinned'},{'cumulative_supervised_tokens':None},{'cumulative_supervised_tokens':-1}])
def test_checkpoint_requires_real_identity_and_token_budget(kwargs):
    params={'arm':'no-cot-turn-v2','completed_updates':25,'checkpoint_sha256':'a'*64,'cumulative_supervised_tokens':1}
    params.update(kwargs)
    with pytest.raises(ValueError):protocol.checkpoint_identity(**params)


def test_freeze_unique_private_expected_outputs_and_startup(frozen):
    manifest=suite.verify_suite(frozen)
    assert len(set(manifest['task_ids']))==16
    assert manifest['training_data'] is False and manifest['terminal_bench_score'] is False
    for task in manifest['task_ids']:
        docker=(frozen/'tasks'/task/'environment/Dockerfile').read_text()
        assert 'ENTRYPOINT []' in docker and 'CMD ["sleep", "infinity"]' in docker
        assert 'COPY inputs/' in docker and 'COPY tests' not in docker
        assert (frozen/'tasks'/task/'tests/expected.json').is_file()
    with pytest.raises(FileExistsError):suite.build(frozen)


def test_freeze_detects_changed_and_extra_files(frozen):
    path=frozen/'tasks/write-marker/instruction.md';path.write_text('different task')
    with pytest.raises(ValueError,match='changed'):suite.verify_suite(frozen)


def test_all16_real_verifier_correct_and_bad_outputs(tmp_path):
    for spec in suite.specifications():
        task=tmp_path/spec['id'];task.mkdir()
        output=task/spec['output']
        if spec['verifier_kind']=='python-function':output.write_text('def total_even(values):\n return sum(v for v in values if v%2==0)\n')
        elif spec['verifier_kind']=='json':output.write_text(json.dumps(spec['expected']))
        else:output.write_text(spec['expected'])
        expected=task/'expected.json';expected.write_text(json.dumps(spec))
        reward_dir=task/'verifier'
        # Execute exactly the shipped verifier, relocating only its absolute sandbox roots for CPU tests.
        code=suite.VERIFY.replace("Path('/tests/expected.json')",'Path('+repr(str(expected))+')')
        code=code.replace("Path('/work/quality')/spec['id']",'Path('+repr(str(task))+')')
        code=code.replace("'/logs/verifier'",repr(str(reward_dir))).replace("'/logs/verifier/reward.txt'",repr(str(reward_dir/'reward.txt')))
        subprocess.run([sys.executable,'-c',code],check=True)
        assert (reward_dir/'reward.txt').read_text()=='1',spec['id']
        output.unlink()
        subprocess.run([sys.executable,'-c',code],check=True)
        assert (reward_dir/'reward.txt').read_text()=='0',spec['id']


def test_checkpoint_specific_preparation_preserves_native_harness(prepared):
    run=verify_run(prepared);job=json.loads((prepared/'job.json').read_bytes())
    services=json.loads((prepared/'runtime/services.json').read_bytes())
    assert run['dispatch_ready'] is False and run['gpu_allocated'] is False
    assert job['n_attempts']==1 and job['retry']['max_retries']==0 and len(job['tasks'])==16
    assert all(task['source']==suite.SOURCE and 'git_url' not in task for task in job['tasks'])
    assert job['agents'][0]['kwargs']['model_auto_compact_token_limit']==196608
    assert job['agents'][0]['kwargs']['model_context_window']==262144
    assert job['agents'][0]['kwargs']['reasoning_effort']=='xhigh'
    assert job['agents'][0]['model_name']==run['model_name']
    proxy=next(service for service in services['services_in_dependency_order'] if service['name']=='proxy')
    assert proxy['argv'][proxy['argv'].index('--token-build-model')+1]=='qwen3_6'
    assert proxy['argv'][proxy['argv'].index('--tito-model')+1]=='qwen3_6'
    assert proxy['argv'][proxy['argv'].index('--model-mask-type')+1]=='qwen3_5'
    assert '/data/evals/quality-no-cot-u25/runtime/qwen38-codex-models.json' in (prepared/'runtime/host-gateway.compose.yaml').read_text()
    assert 'step299' not in json.dumps([job,services['harbor_commands'],services['model_path']])


def test_prepared_file_mutation_rejected(prepared):
    path=prepared/'job.json';job=json.loads(path.read_bytes());job['n_attempts']=2;path.write_text(json.dumps(job))
    with pytest.raises(ValueError,match='changed'):verify_run(prepared)


def test_aggregate_preserves_all16_and_separates_failure_types(prepared,tmp_path):
    path,inventory=outcomes(prepared,tmp_path);result=aggregate(prepared,path)
    assert result['fixed_denominator']==16 and result['outcome_count']==16 and result['complete_coverage']
    assert result['counts']['passed']==5 and result['pass_fraction_all16']==5/16
    assert result['counts']['model_outcome']==15 and result['counts']['infrastructure_failure']==1
    assert result['counts']['no_tool']==8 and result['counts']['premature_eot_no_tool']==7
    assert result['counts']['output_length_limited']==1 and result['counts']['parser_error_reported']==1
    assert result['counts']['native_parity_verified']==15 and not result['fully_native_verified']
    assert result['outcomes'][0]['linked_tool_observations']==1
    assert result['unavailable_observations']['llm_calls']==1
    assert not result['improvement_established']
    assert 'redacted' not in json.dumps(result)


def test_missing_outcomes_do_not_shrink_denominator(prepared,tmp_path):
    path,inventory=outcomes(prepared,tmp_path);inventory['outcomes']=inventory['outcomes'][:5];path.write_text(json.dumps(inventory))
    result=aggregate(prepared,path)
    assert not result['complete_coverage'] and result['fixed_denominator']==16
    assert len(result['missing_tasks'])==11 and result['pass_fraction_all16']==5/16


@pytest.mark.parametrize('mutation',['duplicate','best_of','changed_result','wrong_model','live_result','mask_length'])
def test_rejects_identity_mutation_and_best_of(prepared,tmp_path,mutation):
    path,inventory=outcomes(prepared,tmp_path);row=inventory['outcomes'][0]
    if mutation=='duplicate':inventory['outcomes'].append(deepcopy(row))
    elif mutation=='best_of':inventory['best_of_selection']=True
    elif mutation=='changed_result':Path(row['result']['path']).write_text('{}')
    elif mutation in ('wrong_model','live_result'):
        original=json.loads(Path(row['result']['path']).read_bytes())
        if mutation=='wrong_model':original['config']['agent']['model_name']='old-model'
        else:original.pop('finished_at')
        row['result']=save(Path(row['result']['path']),original)
    elif mutation=='mask_length':
        original=json.loads(Path(row['bundle']['path']).read_bytes());original['segments'][0]['full_loss_mask'].pop()
        row['bundle']=save(Path(row['bundle']['path']),original)
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError):aggregate(prepared,path)


def test_aggregate_never_overwrites_prior_outcomes(prepared,tmp_path):
    path,_=outcomes(prepared,tmp_path);output=tmp_path/'summary.json';aggregate(prepared,path,output)
    before=output.read_bytes()
    with pytest.raises(FileExistsError):aggregate(prepared,path,output)
    assert output.read_bytes()==before


def test_paired_comparison_preserves_task_identities_and_token_budget(prepared,tmp_path):
    from .compare import compare
    path,_=outcomes(prepared,tmp_path);first=aggregate(prepared,path)
    left=tmp_path/'left.json';right=tmp_path/'right.json';left.write_text(json.dumps(first))
    second=deepcopy(first);second['checkpoint']['completed_updates']=50;second['checkpoint']['zero_based_checkpoint_step']=49
    second['checkpoint']['cumulative_supervised_tokens']*=2
    second['outcomes'][7]['passed']=True;second['outcomes'][0]['passed']=False
    right.write_text(json.dumps(second));result=compare(left,right)
    assert len(result['gained_passes'])==len(result['lost_passes'])==1
    assert result['supervised_token_budget_ratio_right_to_left']==2
    assert result['equal_supervised_token_budget'] is False and result['equal_completed_updates'] is False
    assert result['best_checkpoint_selected'] is False and result['statistical_significance_claimed'] is False


def test_paired_comparison_refuses_incomplete_or_different_suite(prepared,tmp_path):
    from .compare import compare
    path,_=outcomes(prepared,tmp_path);first=aggregate(prepared,path)
    left=tmp_path/'left.json';right=tmp_path/'right.json';left.write_text(json.dumps(first))
    for key,value in [('complete_coverage',False),('suite_sha256','f'*64)]:
        second=deepcopy(first);second[key]=value;right.write_text(json.dumps(second))
        with pytest.raises(ValueError):compare(left,right)
