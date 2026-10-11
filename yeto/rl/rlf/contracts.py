from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import hashlib, json, time, uuid

class AdmissionError(ValueError): pass

def _sha(path: Path) -> str:
    h=hashlib.sha256()
    if path.is_file(): h.update(path.read_bytes())
    else:
        for p in sorted(path.rglob('*')):
            if p.is_file(): h.update(str(p.relative_to(path)).encode()); h.update(p.read_bytes())
    return h.hexdigest()

@dataclass(frozen=True)
class TaskAdmission:
    task_id: str; grader_mode: str; package_digest: str; image_digest: str|None
    source_commit: str|None; evidence: Mapping[str,Any]; manifest: Mapping[str,Any]

def admit_task_package(root: str|Path, *, evidence: Mapping[str,Any]|None=None, image_digest: str|None=None, source_commit: str|None=None) -> TaskAdmission:
    root=Path(root); public=root/'public'; yaml=root/'task.yaml'; manifest=root/'MANIFEST.json'
    if not public.is_dir() or not yaml.is_file() or not manifest.is_file(): raise AdmissionError('package must contain public/, task.yaml and MANIFEST.json')
    try: m=json.loads(manifest.read_text()); text=yaml.read_text()
    except Exception as e: raise AdmissionError(f'invalid package metadata: {e}') from e
    task_id=m.get('task_id') or next((x.split(':',1)[1].strip() for x in text.splitlines() if x.startswith('task_id:')),None)
    mode=m.get('grader_mode') or next((x.split(':',1)[1].strip() for x in text.splitlines() if x.startswith('grader_mode:')),None)
    if not task_id or mode not in {'flag','hacked'}: raise AdmissionError('task_id and grader_mode flag/hacked are required')
    ev=dict(evidence or {})
    if not ev.get('runtime_quality_gate'): raise AdmissionError('runtime_quality_gate evidence is required')
    if ev.get('agentflow_node_output'): raise AdmissionError('AgentFlow node output is not admission evidence')
    return TaskAdmission(task_id, mode, _sha(root), image_digest, source_commit, ev, m)

@dataclass
class EpisodeLease:
    task_id: str; trajectory_id: str; episode_id: str=field(default_factory=lambda: str(uuid.uuid4())); started_at: float=field(default_factory=time.time); closed: bool=False
    secret: str|None=None
    def teardown(self): self.closed=True; self.secret=None

@dataclass(frozen=True)
class TrustedGrade:
    status: str; reward: float; passed: bool; grader_evidence: Mapping[str,Any]; error_class: str|None=None
@dataclass(frozen=True)
class RewardEvidence:
    value: float; metadata: Mapping[str,Any]

def finish_episode(lease: EpisodeLease, grade: TrustedGrade, *, timed_out: bool=False) -> RewardEvidence:
    if lease.closed: raise RuntimeError('episode already torn down')
    status='timeout' if timed_out else grade.status
    md={'success': bool(grade.passed), 'reward': float(grade.reward), 'rlf_episode_id':lease.episode_id, 'rlf_task_id':lease.task_id, 'rlf_trajectory_id':lease.trajectory_id, 'rlf_status':status, 'grader_evidence':dict(grade.grader_evidence)}
    if grade.error_class: md['rejected_data_reason']=grade.error_class
    lease.teardown(); return RewardEvidence(float(grade.reward), md)

def rl_gate(rows: list[Mapping[str,Any]]) -> dict[str,Any]:
    rewards=[float(r.get('reward',0)) for r in rows]; adv=[float(r.get('advantage',0)) for r in rows]
    total=sum(rewards); mean=total/len(rewards) if rewards else 0.0
    variance=sum((x-mean)**2 for x in rewards)/len(rewards) if rewards else 0.0
    result={'reward_sum':total,'reward_std':variance**0.5,'success_count':sum(bool(r.get('metadata',{}).get('success')) for r in rows),'advantage_abs_sum':sum(abs(x) for x in adv),'policy_update':any(bool(r.get('policy_update')) for r in rows)}
    criteria={'non_empty':bool(rows),'reward_std_positive':result['reward_std']>0,'success_count_positive':result['success_count']>0,'advantage_abs_sum_positive':result['advantage_abs_sum']>0,'policy_update':result['policy_update']}
    result['passed']=all(criteria.values())
    if not result['passed']:
        failed=', '.join(k for k,v in criteria.items() if not v)
        raise AdmissionError(f'RL gate failed ({failed}): {result}')
    return result

def group_advantages(rewards: list[float]) -> list[float]:
    """GRPO-style group baseline: advantage_i = reward_i - mean(group).

    This is the gate-runner/evidence-entry computation of record for
    ``advantage_abs_sum`` on paths where the learner does not export
    per-sample advantages; callers must not hand-fill the value.
    """
    values=[float(r) for r in rewards]
    if not values: return []
    mean=sum(values)/len(values)
    return [r-mean for r in values]

def _params_digest(params: Mapping[str,float]) -> str:
    blob=json.dumps({k:float(v) for k,v in sorted(params.items())},sort_keys=True,separators=(',',':'))
    return hashlib.sha256(blob.encode()).hexdigest()

@dataclass(frozen=True)
class PolicyUpdateEvidence:
    """Structured proof that an optimizer step changed policy parameters."""
    step: int; updated: bool; before_digest: str; after_digest: str; changed: tuple[str,...]

def policy_update_evidence(before: Mapping[str,float], after: Mapping[str,float], *, step: int) -> PolicyUpdateEvidence:
    """Record a learner update by digesting parameters before and after the step.

    ``updated`` is true only when the parameter snapshot actually changed;
    gate rows must read ``policy_update`` from this evidence, never from a
    hand-written literal.
    """
    keys=set(before)|set(after)
    changed=tuple(sorted(k for k in keys if float(before.get(k,0.0))!=float(after.get(k,0.0))))
    bd=_params_digest(before); ad=_params_digest(after)
    return PolicyUpdateEvidence(int(step), bd!=ad, bd, ad, changed)
