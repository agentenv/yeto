from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
@dataclass
class CodexTrajectory:
    task_id: str; episode_id: str; trajectory_id: str
    tokens: list[int]=field(default_factory=list); logprobs: list[float]=field(default_factory=list)
    loss_mask: list[int]=field(default_factory=list); segments: list[Mapping[str,Any]]=field(default_factory=list)
    policy_id: str|None=None; metadata: dict[str,Any]=field(default_factory=dict)
    def as_metadata(self):
        return {'rlf_task_id':self.task_id,'rlf_episode_id':self.episode_id,'rlf_trajectory_id':self.trajectory_id,'policy_id':self.policy_id,'segments':self.segments}

def from_codex_result(result: Mapping[str,Any], *, task_id: str, episode_id: str, trajectory_id: str, policy_id: str|None=None) -> CodexTrajectory:
    return CodexTrajectory(task_id,episode_id,trajectory_id,list(result.get('tokens',[])),list(result.get('logprobs',result.get('rollout_logprobs',[]))),list(result.get('loss_mask',[])),list(result.get('segments',[])),policy_id,dict(result.get('metadata',{})))
