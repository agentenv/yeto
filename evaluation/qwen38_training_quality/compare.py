"""Paired descriptive comparison; it never selects a best checkpoint."""
import argparse
import json
from pathlib import Path
from .aggregate import SUMMARY_SCHEMA
from .protocol import hash_string
from .suite import digest


def compare(left_path,right_path):
    left=json.loads(Path(left_path).read_bytes());right=json.loads(Path(right_path).read_bytes())
    for summary in (left,right):
        if (summary.get('schema')!=SUMMARY_SCHEMA or summary.get('complete_coverage') is not True
            or summary.get('fixed_denominator')!=16 or summary.get('all_first_attempts_preserved') is not True
            or summary.get('best_of_selection') is not False or not hash_string(summary.get('suite_sha256'))):
            raise ValueError('Compare complete fixed first-attempt suites only')
    if left['suite_sha256']!=right['suite_sha256']:raise ValueError('Task suite differs')
    def outcomes(summary):
        rows=summary['outcomes']
        if len(rows)!=16 or len({row['task_id'] for row in rows})!=16:raise ValueError('Wrong task coverage')
        return {row['task_id']:row for row in rows}
    a,b=outcomes(left),outcomes(right)
    if set(a)!=set(b):raise ValueError('Task identities differ')
    gains=[task for task in a if not a[task]['passed'] and b[task]['passed']]
    losses=[task for task in a if a[task]['passed'] and not b[task]['passed']]
    left_budget=left['checkpoint']['cumulative_supervised_tokens'];right_budget=right['checkpoint']['cumulative_supervised_tokens']
    fully_measured=left['fully_native_verified'] and right['fully_native_verified'] and all(
        row['model_outcome'] for row in [*a.values(),*b.values()])
    return {'schema':'yeta.qwen38-training-quality-comparison/v1','left_sha256':digest(left_path),
        'right_sha256':digest(right_path),'left_checkpoint':left['checkpoint'],'right_checkpoint':right['checkpoint'],
        'suite_sha256':left['suite_sha256'],'task_count':16,'gained_passes':gains,'lost_passes':losses,
        'pass_fraction_delta_right_minus_left':(len(gains)-len(losses))/16,
        'no_tool_delta_right_minus_left':right['counts']['no_tool']-left['counts']['no_tool'],
        'premature_eot_delta_right_minus_left':right['counts']['premature_eot_no_tool']-left['counts']['premature_eot_no_tool'],
        'supervised_token_budget_ratio_right_to_left':right_budget/left_budget if left_budget else None,
        'equal_supervised_token_budget':left_budget==right_budget,
        'equal_completed_updates':left['checkpoint']['completed_updates']==right['checkpoint']['completed_updates'],
        'all_outcomes_native_verified_and_without_infra_failure':fully_measured,
        'interpretation':'descriptive paired synthetic action check; no generalization or causal improvement claim',
        'best_checkpoint_selected':False,'statistical_significance_claimed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('left','right','output'):parser.add_argument('--'+field,required=True)
    args=parser.parse_args();result=compare(args.left,args.right)
    with Path(args.output).open('x') as file:json.dump(result,file,indent=2,sort_keys=True);file.write('\n')
    print(json.dumps({key:result[key] for key in ('task_count','gained_passes','lost_passes','pass_fraction_delta_right_minus_left')}))
