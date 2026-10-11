from __future__ import annotations
from typing import Any, Mapping
from .contracts import TrustedGrade
_REQUIRED=('rlf_episode_id','rlf_task_id','rlf_trajectory_id','grader_evidence','success')
class RolloutRejected(ValueError): pass
def validate_rollout(row: Mapping[str,Any]) -> None:
    md=row.get('metadata')
    if not isinstance(md, Mapping): raise RolloutRejected('missing metadata')
    missing=[x for x in _REQUIRED if x not in md]
    if missing: raise RolloutRejected('missing trusted fields: '+', '.join(missing))
    if not md.get('rlf_episode_id') or not md.get('rlf_task_id'): raise RolloutRejected('missing RLF identity')
    if not isinstance(md['grader_evidence'], Mapping): raise RolloutRejected('invalid grader evidence')
    if row.get('cleanup_ok') is False or md.get('cleanup_error'): raise RolloutRejected('cleanup evidence failed')

RLF_GRADE_KEY = 'rlf_grade'
_GRADE_REQUIRED = ('status', 'reward', 'passed', 'grader_evidence')

def trusted_grade_from_evaluation(evaluation: Mapping[str, Any]) -> TrustedGrade | None:
    """Interface for the harness trusted finish (RLF grade first, TB2 HMAC fallback).

    A ``TrustedVerifier.evaluate()`` payload that carries ``rlf_grade`` (or is
    itself a full grade mapping) is parsed into a ``TrustedGrade`` so the
    trusted finish can close the episode via ``finish_episode``; anything else
    returns ``None`` and the caller falls back to the TB2 HMAC-signed outcome.
    A present but malformed grade fails closed (``RolloutRejected``) instead of
    silently downgrading to HMAC.

    Not yet wired into ``codex_openenv_agent_function.finish_trusted``; that
    call site is outside this change's write scope.
    """
    if not isinstance(evaluation, Mapping): return None
    payload = evaluation.get(RLF_GRADE_KEY)
    if payload is None and all(k in evaluation for k in _GRADE_REQUIRED): payload = evaluation
    if payload is None: return None
    if not isinstance(payload, Mapping) or not all(k in payload for k in _GRADE_REQUIRED):
        raise RolloutRejected('malformed RLF grade in trusted evaluation')
    evidence = payload['grader_evidence']
    if not isinstance(evidence, Mapping): raise RolloutRejected('RLF grade evidence must be a mapping')
    error = payload.get('error_class')
    return TrustedGrade(str(payload['status']), float(payload['reward']), bool(payload['passed']),
                        dict(evidence), str(error) if error else None)
