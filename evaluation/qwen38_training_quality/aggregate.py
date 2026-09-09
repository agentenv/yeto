"""Aggregate all fixed first-attempt outcomes without exporting transcript text."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from . import protocol
from .prepare import verify_run
from .suite import digest, SOURCE

INVENTORY_SCHEMA='yeta.qwen38-training-quality-outcomes/v1'
SUMMARY_SCHEMA='yeta.qwen38-training-quality-summary/v1'


def read_bound(ref):
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or not protocol.hash_string(ref['sha256']):
        raise ValueError('Each original artifact requires its actual path and SHA256')
    path=Path(ref['path']);raw=path.read_bytes()
    if path.is_symlink() or digest(path)!=ref['sha256']:
        raise ValueError('Original outcome artifact changed')
    return json.loads(raw)


def measure(task, row, run):
    result=read_bound(row['result']);config=result.get('config',{})
    task_config=config.get('task',{})
    if Path(task_config.get('path','')).name!=task or task_config.get('source')!=SOURCE:
        raise ValueError('Actual trial task/source differs from frozen synthetic suite')
    agent=config.get('agent',{})
    kwargs=agent.get('kwargs',{})
    if (agent.get('name')!='codex' or agent.get('model_name')!=run['model_name']
        or any(kwargs.get(k)!=v for k,v in {'version':'0.142.5','reasoning_effort':'xhigh',
        'model_context_window':protocol.CONTEXT,'model_auto_compact_token_limit':protocol.COMPACTION}.items())):
        raise ValueError('Actual agent settings differ from checkpoint protocol')
    if not result.get('finished_at'):
        raise ValueError('An in-progress result cannot be sealed as an outcome')
    reward=(result.get('verifier_result') or {}).get('rewards',{}).get('reward')
    if reward not in (0,1,None) or type(reward) is bool:
        raise ValueError('Expected a binary synthetic verifier reward')
    exception=result.get('exception_info')
    exception_type=exception.get('exception_type','unspecified') if exception else None
    agent_timeout=exception_type in {'AgentTimeoutError','AgentExecutionTimeoutError'}
    rec={'task_id':task,'trial_id':result.get('id'),'reward':reward,'exception_type':exception_type,
         'result_sha256':row['result']['sha256'],'trajectory_available':False,'bundle_available':False,
         'native_parity_verified':False,'tool_calls':None,'tool_observations':None,'linked_tool_observations':None,
         'llm_calls':None,'no_tool':None,'premature_eot_no_tool':None,'finish_reasons':[],
         'output_length_limited':None,'parser_error_reported':False,
         'infrastructure_failure':bool(exception) and not agent_timeout,'agent_execution_timeout':agent_timeout,'model_outcome':not exception and reward in (0,1),
         'passed':not exception and reward==1}
    if row.get('trajectory'):
        trajectory=read_bound(row['trajectory']);steps=trajectory.get('steps')
        if not isinstance(steps,list):raise ValueError('Unsupported trajectory steps')
        assistant=[step for step in steps if step.get('source') in ('agent','assistant')]
        if any(not isinstance(step.get('tool_calls') or [],list) for step in assistant):
            raise ValueError('Malformed tool-call list')
        calls=[call for step in assistant for call in (step.get('tool_calls') or [])]
        observations=[obs for step in steps if isinstance(step.get('observation'),dict)
                      for obs in (step['observation'].get('results') or [])]
        call_ids={call.get('tool_call_id',call.get('id')) for call in calls}-{None}
        linked=sum(obs.get('source_call_id') in call_ids for obs in observations)
        counts=[step.get('llm_call_count') for step in assistant]
        llm_calls=sum(counts) if counts and all(type(count) is int and count>=0 for count in counts) else None
        rec.update(trajectory_available=True,trajectory_sha256=row['trajectory']['sha256'],
                   llm_calls=llm_calls,tool_calls=len(calls),tool_observations=len(observations),
                   linked_tool_observations=linked,
                   max_prompt_tokens=max([step.get('metrics',{}).get('prompt_tokens',0) or 0 for step in assistant] or [0]),
                   no_tool=(llm_calls>0 and not calls) if llm_calls is not None else None)
    if row.get('bundle'):
        bundle=read_bound(row['bundle']);segments=bundle.get('segments')
        if not isinstance(segments,list):raise ValueError('Unsupported saved bundle')
        outputs=[]
        for segment in segments:
            tokens=segment.get('tokens');mask=segment.get('full_loss_mask')
            if (not isinstance(tokens,list) or not isinstance(mask,list) or len(tokens)!=len(mask)
                or any(type(token) is not int for token in tokens) or any(value not in (0,1) for value in mask)):
                raise ValueError('Malformed saved token/mask arrays')
            outputs.append([token for token,selected in zip(tokens,mask) if selected])
        reasons=[segment.get('finish_reason') for segment in segments]
        warnings=bundle.get('warnings') or [];failures=bundle.get('failures') or []
        parser=bool(re.search(r'(?i)(?:tool.?call.?pars|parser.?error|invalid_tool_call)',json.dumps([warnings,failures])))
        single_stop=(len(segments)==1 and reasons==['stop'] and 248046 in outputs[0])
        rec.update(bundle_available=True,bundle_sha256=row['bundle']['sha256'],finish_reasons=reasons,
                   output_token_counts=[len(output) for output in outputs],
                   output_length_limited='length' in reasons,parser_error_reported=parser,
                   bundle_failure_count=len(failures),bundle_warning_count=len(warnings),
                   generated_tool_open_count=sum(output.count(248058) for output in outputs),
                   generated_tool_close_count=sum(output.count(248059) for output in outputs),
                   premature_eot_no_tool=(rec['no_tool'] and rec['llm_calls']==1 and single_stop)
                       if rec['no_tool'] is not None else None)
    if row.get('native_parity'):
        proof=read_bound(row['native_parity'])
        expected={'schema':'yeta.qwen38-quality-native-parity/v1',
            'checkpoint_manifest_sha256':run['checkpoint']['checkpoint_manifest_sha256'],
            'result_sha256':row['result']['sha256'],'bundle_sha256':row.get('bundle',{}).get('sha256'),
            'native_template_sha256':protocol.TEMPLATE_SHA,'actual_initial_native_xhigh_parity':True,
            'actual_incremental_tool_result_parity':True}
        rec['native_parity_verified']=all(proof.get(key)==value for key,value in expected.items())
        rec['native_parity_sha256']=row['native_parity']['sha256']
    if not rec['infrastructure_failure'] and reward is None:
        rec['model_outcome']=False
        rec['incomplete_verifier_result']=True
    return rec


def aggregate(run_dir, inventory_path, output=None):
    run=verify_run(run_dir,allow_runtime_outputs=True);inventory=json.loads(Path(inventory_path).read_bytes())
    if (inventory.get('schema')!=INVENTORY_SCHEMA or inventory.get('run_sha256')!=digest(Path(run_dir)/'run.json')
        or inventory.get('first_attempts_only') is not True or inventory.get('best_of_selection') is not False):
        raise ValueError('Inventory must bind this run and preserve all first attempts')
    outcomes=inventory.get('outcomes')
    if not isinstance(outcomes,list):raise ValueError('Missing outcome inventory')
    task_ids=[row.get('task_id') for row in outcomes]
    if len(task_ids)!=len(set(task_ids)) or set(task_ids)-set(run['task_ids']):
        raise ValueError('Duplicate attempts or unknown task; best-of selection is forbidden')
    rows=[measure(row['task_id'],row,run) for row in outcomes]
    trial_ids=[row['trial_id'] for row in rows]
    if any(not isinstance(value,str) or not value for value in trial_ids) or len(set(trial_ids))!=len(trial_ids):
        raise ValueError('Missing or repeated original trial identity')
    counts={key:sum(row.get(key) is True for row in rows) for key in ('passed','model_outcome',
        'infrastructure_failure','agent_execution_timeout','trajectory_available','bundle_available','native_parity_verified',
        'no_tool','premature_eot_no_tool','output_length_limited','parser_error_reported')}
    observed=len(rows);missing=[task for task in run['task_ids'] if task not in set(task_ids)]
    summary={'schema':SUMMARY_SCHEMA,'run_sha256':digest(Path(run_dir)/'run.json'),
        'inventory_sha256':digest(inventory_path),'aggregator_sha256':digest(__file__),
        'checkpoint':run['checkpoint'],'suite_sha256':run['suite_sha256'],'fixed_denominator':16,
        'outcome_count':observed,'missing_tasks':missing,'complete_coverage':not missing,
        'counts':counts,'pass_fraction_all16':counts['passed']/16,
        'pass_fraction_executed_model_outcomes':counts['passed']/counts['model_outcome'] if counts['model_outcome'] else None,
        'fully_native_verified':not missing and counts['native_parity_verified']==16,
        'all_first_attempts_preserved':True,'best_of_selection':False,
        'benchmark_generalization_established':False,'improvement_established':False,
        'unavailable_observations':{key:sum(row.get(key) is None for row in rows) for key in
            ('llm_calls','tool_calls','no_tool','premature_eot_no_tool','output_length_limited')},
        'exceptions':dict(Counter(row['exception_type'] for row in rows if row['exception_type'])),
        'finish_reasons':dict(Counter(reason for row in rows for reason in row['finish_reasons'])),
        'outcomes':rows}
    if output:
        with Path(output).open('x') as file:json.dump(summary,file,indent=2,sort_keys=True);file.write('\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('run-dir','inventory','output'):parser.add_argument('--'+field,required=True)
    args=parser.parse_args();summary=aggregate(args.run_dir,args.inventory,args.output)
    print(json.dumps({key:summary[key] for key in ('counts','outcome_count','fixed_denominator','complete_coverage')}))
