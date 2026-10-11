from __future__ import annotations
import json, uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from .contracts import AdmissionError, EpisodeLease, TaskAdmission, admit_task_package

@dataclass
class TaskCatalog:
    tasks: dict[str, TaskAdmission]
    roots: dict[str, Path] = field(default_factory=dict)
    def add(self, package: str|Path, **kwargs: Any) -> TaskAdmission:
        item = admit_task_package(package, **kwargs)
        if item.task_id in self.tasks: raise ValueError(f'duplicate RLF task: {item.task_id}')
        self.tasks[item.task_id] = item
        self.roots[item.task_id] = Path(package)
        return item
    def get(self, task_id: str) -> TaskAdmission:
        return self.tasks[task_id]
    def episode(self, task_id: str, *, trajectory_id: str|None=None) -> EpisodeLease:
        """Start one rollout episode for an admitted task (RLF identity producer)."""
        admission = self.get(task_id)
        return EpisodeLease(task_id=admission.task_id,
                            trajectory_id=trajectory_id or f'rlf-traj-{uuid.uuid4().hex}')

def import_catalog(packages: list[str|Path], **kwargs: Any) -> TaskCatalog:
    catalog=TaskCatalog({})
    for package in packages: catalog.add(package, **kwargs)
    return catalog

RLF_IDENTITY_KEYS = ('rlf_task_id', 'rlf_episode_id', 'rlf_trajectory_id', 'policy_id')

def rlf_identity(lease: EpisodeLease, *, policy_id: str|None=None) -> dict[str, str]:
    """The RLF identity quadruple for one episode lease (prompt/chain metadata)."""
    identity = {'rlf_task_id': lease.task_id, 'rlf_episode_id': lease.episode_id,
                'rlf_trajectory_id': lease.trajectory_id}
    if policy_id is not None: identity['policy_id'] = str(policy_id)
    return identity

def task_statement(catalog: TaskCatalog, task_id: str) -> str:
    """The task's own statement (public README); never fabricated."""
    root = catalog.roots.get(task_id)
    if root is None: raise AdmissionError(f'no package root recorded for task {task_id!r}')
    for name in ('README.md', 'README'):
        candidate = root/'public'/name
        if candidate.is_file():
            text = candidate.read_text().strip()
            if text: return text
    raise AdmissionError(f'task {task_id!r} has no public README task statement')

def target_endpoint(catalog: TaskCatalog, task_id: str) -> str | None:
    """The agent-visible target address from the package's task.yaml.

    RLF CTF targets run on the episode VM's host network, so the address is
    always 127.0.0.1:<target_port>.  Returns None when the package declares no
    port (non-network task).
    """
    root = catalog.roots.get(task_id)
    if root is None:
        return None
    text = (root/'task.yaml').read_text()
    for line in text.splitlines():
        if line.startswith('target_port:'):
            port = line.split(':', 1)[1].strip()
            return f'127.0.0.1:{port}' if port else None
    return None


def prompt_rows(catalog: TaskCatalog, task_ids: list[str]|None=None, *, repeat: int=1) -> list[dict[str, Any]]:
    """Catalog -> Miles ``--prompt-data`` rows (the catalog consumer for rollout).

    Each row carries ``metadata.task_id`` (read by the harness ``task_identity``)
    and ``rlf_task_id``; the per-rollout episode/trajectory ids are produced
    later by ``codex_openenv_generate.produce_rlf_identity`` so episodes are
    never shared across rollouts.
    """
    rows: list[dict[str, Any]] = []
    for task_id in (task_ids or list(catalog.tasks)):
        admission = catalog.get(task_id)
        statement = task_statement(catalog, task_id)
        endpoint = target_endpoint(catalog, task_id)
        if endpoint:
            statement += ('\n\nThe target service listens on '
                          + endpoint + ' (reachable from your shell).')

        metadata: dict[str, Any] = {'task_id': admission.task_id, 'rlf_task_id': admission.task_id,
                                    'benchmark': 'rlf', 'grader_mode': admission.grader_mode,
                                    'package_digest': admission.package_digest, 'prompt': statement}
        if admission.image_digest: metadata['image_digest'] = admission.image_digest
        for _ in range(max(1, repeat)):
            rows.append({'metadata': dict(metadata), 'prompt': statement})
    return rows

def write_prompt_data(catalog: TaskCatalog, path: str|Path, task_ids: list[str]|None=None, *, repeat: int=1) -> Path:
    """Write ``prompt_rows`` as the JSONL file Miles ``--prompt-data`` expects."""
    path = Path(path)
    rows = prompt_rows(catalog, task_ids, repeat=repeat)
    if not rows: raise AdmissionError('catalog produced no prompt rows')
    path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    return path

def main(argv: list[str]|None=None) -> None:
    """``python -m yeto.rl.rlf.catalog PKG... --out prompt.jsonl [--repeat N]``."""
    import argparse
    parser = argparse.ArgumentParser(description='RLF catalog -> Miles prompt data')
    parser.add_argument('packages', nargs='+')
    parser.add_argument('--out', required=True)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--evidence', default=None, help='JSON admission evidence (must include runtime_quality_gate)')
    args = parser.parse_args(argv)
    evidence = json.loads(args.evidence) if args.evidence else None
    catalog = import_catalog(args.packages, evidence=evidence)
    write_prompt_data(catalog, args.out, repeat=args.repeat)

if __name__ == '__main__':
    main()
