"""Preserve a completed model snapshot for the fixed training-quality suite.

Only model and small scheduler/contract state are hard-linked. No optimizer
files are copied, and the trainer's checkpoint retention policy is unchanged.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor

from . import protocol


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def budget_from_events(text, completed_updates):
    """Require every exact optimizer update; rounded stdout is insufficient."""
    updates = {}
    for line in text.splitlines():
        if not line.startswith('{"event": "verified_optimizer_update",'): continue
        row = json.loads(line)
        step, metrics = row['step'], row['measurements']
        if type(step) is not int or step < 0: raise ValueError('Invalid optimizer cursor')
        if step >= completed_updates: continue
        count = metrics['num_label_tokens']
        if type(count) is float and math.isfinite(count) and count.is_integer(): count = int(count)
        inputs = metrics.get('num_tokens_per_step')
        if type(count) is not int or count <= 0: raise ValueError('Invalid supervised token count')
        if inputs is not None and (type(inputs) is not int or inputs < count):
            raise ValueError('Invalid input token count')
        item = {'supervised': count, 'input': inputs}
        if step in updates and updates[step] != item: raise ValueError('Optimizer event changed')
        updates[step] = item
    if set(updates) != set(range(completed_updates)):
        raise ValueError('Exact cumulative optimizer history is incomplete')
    return {'cumulative_supervised_tokens': sum(x['supervised'] for x in updates.values()),
            'cumulative_input_tokens': (sum(x['input'] for x in updates.values())
                if all(x['input'] is not None for x in updates.values()) else None)}


def snapshot(source, destination, *, arm, completed_updates, budget, config):
    source, destination, config = Path(source), Path(destination), Path(config)
    if arm not in protocol.ARMS or not protocol.scheduled(completed_updates):
        raise ValueError('Use a scheduled corrected-arm checkpoint')
    match = re.fullmatch(r'epoch_(\d+)_step_(\d+)', source.name)
    if not match or int(match[1]) != 0 or int(match[2]) != completed_updates - 1:
        raise ValueError('Checkpoint name does not match the requested completed updates')
    if not all(x.is_absolute() for x in (source, destination, config)):
        raise ValueError('Use explicit absolute paths')
    if source.resolve(strict=True) != source or (source / '.incomplete').exists():
        raise ValueError('Checkpoint must be complete and cannot be a symlink')
    if destination == source or source in destination.parents:
        raise ValueError('Snapshot must be outside the original checkpoint')
    settings = json.loads(config.read_bytes())
    expected_target = ('training.qwen38_turn_boundary_v2.data.TurnTokenDataset' if arm == protocol.ARMS[0]
                       else 'training.qwen38_native_gap_v3.masked_data.GeneratedMaskedTokenDataset')
    if settings['dataset']['_target_'] != expected_target:
        raise ValueError('Configuration does not identify the requested corrected arm')
    if Path(settings['checkpoint']['checkpoint_dir']) != source.parent:
        raise ValueError('Checkpoint belongs to another training configuration')
    paths = sorted((source / 'model').iterdir())
    if ({p.name for p in paths if p.suffix == '.distcp'} != {f'__{rank}_0.distcp' for rank in range(8)}
            or {p.name for p in paths if p.suffix != '.distcp'} != {'.metadata'}):
        raise ValueError('Unexpected eight-rank DCP model inventory')
    paths += [source / 'step_scheduler.pt', source / 'resume_guard.pt']
    if any(not p.is_file() or p.resolve(strict=True) != p for p in paths):
        raise ValueError('Invalid model or scheduler file')
    before = {str(p.relative_to(source)): (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.incomplete')
    if destination.exists() or partial.exists(): raise FileExistsError('Preserve the existing snapshot')
    partial.mkdir()
    for p in paths:
        target = partial / p.relative_to(source); target.parent.mkdir(exist_ok=True)
        os.link(p, target)
    (partial / 'train.json').write_bytes(config.read_bytes())
    def entry(relative):
        p = partial / relative
        return {'path': relative, 'bytes': p.stat().st_size, 'sha256': digest(p)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(entry, sorted(before)))
    for relative, state in before.items():
        p = partial / relative
        if (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) != state:
            raise ValueError('Checkpoint file changed while snapshotting')
    if (source / '.incomplete').exists(): raise ValueError('Source completion changed')
    manifest = {'schema': 'yeta.qwen38-quality-checkpoint-snapshot/v1', 'arm': arm,
                'completed_updates': completed_updates, 'source': str(source),
                'config_sha256': digest(config), 'files': files,
                'optimizer_copied': False, 'gpu_used': False}
    write_json(partial / 'snapshot-manifest.json', manifest)
    identity = protocol.checkpoint_identity(arm=arm, completed_updates=completed_updates,
        checkpoint_sha256=digest(partial / 'snapshot-manifest.json'), **budget)
    write_json(partial / 'checkpoint.json', identity)
    partial.rename(destination)
    return identity


def verify_snapshot(path):
    path = Path(path)
    manifest = json.loads((path / 'snapshot-manifest.json').read_bytes())
    checkpoint = json.loads((path / 'checkpoint.json').read_bytes())
    if (manifest['schema'] != 'yeta.qwen38-quality-checkpoint-snapshot/v1'
            or checkpoint['checkpoint_manifest_sha256'] != digest(path / 'snapshot-manifest.json')
            or checkpoint['arm'] != manifest['arm'] or checkpoint['completed_updates'] != manifest['completed_updates']
            or digest(path / 'train.json') != manifest['config_sha256']):
        raise ValueError('Snapshot identity changed')
    for item in manifest['files']:
        p = path / item['path']
        if (not p.resolve(strict=True).is_relative_to(path.resolve()) or p.is_symlink()
                or p.stat().st_size != item['bytes'] or digest(p) != item['sha256']):
            raise ValueError('Snapshot file changed')
    return checkpoint
