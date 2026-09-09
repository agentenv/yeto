"""Finite, on-host checkpoint preservation and CPU export; never launches evaluation.

The immutable plan binds an already-qualified, already-running trainer.  A
separate detached CPU container converts one snapshot at a time while this
process continues preserving later checkpoints.  Restarting this supervisor
resumes its ledger; it never restarts a trainer or retries a failed conversion.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from . import checkpoint, protocol

SCHEMA = 'yeta.qwen38-quality-supervisor-plan/v1'
STATE_SCHEMA = 'yeta.qwen38-quality-supervisor-state/v1'
IMAGE = 'sha256:b0bd0e50b29cb0a3ec69c3cbbb49652d3c1aabedbfe22ee0fa6d053b6bb33aee'
ORIGINAL = Path('/data/sft_baseline_20260908/models/Qwen3.8-27B')
GIB = 1024**3
SCHEDULE = [25, 50, 100]
ARMS = {
    'no-cot-turn-v2': ('training.qwen38_turn_boundary_v2.data.TurnTokenDataset',
                     'training.qwen38_turn_boundary_v2.train'),
    'masked-cot-native-gap-v3': ('training.qwen38_native_gap_v3.masked_data.GeneratedMaskedTokenDataset',
                               'training.qwen38_native_gap_v3.masked_train'),
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(argv, *, timeout=30):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        # Docker diagnostics may contain private paths or log text. Keep those
        # out of the public status; the command's durable receipt identifies it.
        raise RuntimeError('Command failed: ' + argv[0] + ' exit ' + str(result.returncode))
    return result.stdout


def absolute(value, *, exists=True):
    path = Path(value)
    if not path.is_absolute() or path.resolve(strict=exists) != path:
        raise ValueError('Use canonical absolute paths without symlinks')
    return path


def bound_file(value, sha):
    path = absolute(value)
    if not path.is_file() or not protocol.hash_string(sha) or checkpoint.digest(path) != sha:
        raise ValueError('Bound file identity changed')
    return path


def verify_code(root, expected):
    root = absolute(root)
    manifest = bound_file(root / 'code-manifest.json', expected)
    entries = json.loads(manifest.read_bytes())['files']
    seen = set()
    if not entries: raise ValueError('Empty code manifest')
    for item in entries:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
            raise ValueError('Escaping or duplicate code path')
        seen.add(str(relative))
        actual = bound_file(root / relative, item['sha256'])
        if actual.stat().st_size != item['bytes']: raise ValueError('Code size changed')
    return root, seen


def validate_plan(plan):
    if (plan.get('schema') != SCHEMA or plan.get('arm') not in ARMS
            or plan.get('arm') not in protocol.ARMS or plan.get('completed_updates') != SCHEDULE
            or plan.get('runtime_image') != IMAGE):
        raise ValueError('Expected the finite qualified corrected-arm plan')
    if not protocol.hash_string(plan.get('trainer_container')):
        raise ValueError('Pin the full trainer container ID')
    if not isinstance(plan.get('trainer_started_at'), str) or not plan['trainer_started_at']:
        raise ValueError('Pin the actual trainer start identity')
    args = plan.get('trainer_argv')
    if not isinstance(args, list) or not args or any(not isinstance(x, str) for x in args):
        raise ValueError('Pin the exact trainer command')
    config = bound_file(plan['config_path'], plan['config_sha256'])
    settings = json.loads(config.read_bytes())
    target, module = ARMS[plan['arm']]
    if (settings['dataset']['_target_'] != target or args.count('-m') != 1
            or args[args.index('-m') + 1] != module or args.count('--config') != 1
            or args[args.index('--config') + 1] != str(config)):
        raise ValueError('Wrong training arm or configuration argument')
    checkpoints = absolute(settings['checkpoint']['checkpoint_dir'], exists=False)
    if checkpoints != config.parent / 'checkpoints' or settings['checkpoint'].get('restore_from'):
        raise ValueError('Expected the fresh corrected run checkpoint directory')
    code, _ = verify_code(plan['training_code_dir'], plan['training_code_manifest_sha256'])
    quality, files = verify_code(plan['quality_code_dir'], plan['quality_code_manifest_sha256'])
    required = {'evaluation/qwen38_training_quality/' + name + '.py'
                for name in ('__init__', 'checkpoint', 'export', 'protocol', 'supervisor')}
    required.add('monitoring/export_masked_cot_pilot_checkpoint_20260909.py')
    if not required <= files or Path(__file__).resolve() != quality / 'evaluation/qwen38_training_quality/supervisor.py':
        raise ValueError('The executing quality runtime is not the bound code')
    receipt = json.loads(bound_file(plan['cpu_preflight_path'], plan['cpu_preflight_sha256']).read_bytes())
    expected = {'status': 'passed', 'runtime_image': IMAGE, 'config_sha256': plan['config_sha256'],
                'code_manifest_sha256': plan['training_code_manifest_sha256'], 'cuda_initialized': False}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Actual-image CPU preflight does not qualify this trainer')
    output = absolute(plan['output_dir'], exists=False)
    if output == checkpoints or checkpoints in output.parents or output in checkpoints.parents:
        raise ValueError('Quality output must be separate from trainer checkpoints')
    if any(output == protected or protected in output.parents or output in protected.parents
           for protected in (code, quality, ORIGINAL)):
        raise ValueError('Quality output must be separate from immutable code and original model')
    if not 5 <= plan.get('poll_seconds', 10) <= 60:
        raise ValueError('Poll interval must be between 5 and 60 seconds')
    return {'config': config, 'checkpoints': checkpoints, 'quality': quality, 'training': code, 'output': output}


def inspect_container(container):
    # Explicit non-secret fields only: do not inspect the full Config or Env.
    template = ('{"id":{{json .Id}},"image":{{json .Image}},"running":{{json .State.Running}},'
                '"started_at":{{json .State.StartedAt}},"exit_code":{{json .State.ExitCode}},'
                '"oom_killed":{{json .State.OOMKilled}},"path":{{json .Path}},"args":{{json .Args}}}')
    return json.loads(run(['docker', 'inspect', '--format', template, container]))


def verify_trainer(plan):
    info = inspect_container(plan['trainer_container'])
    if (info['id'] != plan['trainer_container'] or info['image'] != IMAGE
            or info['started_at'] != plan['trainer_started_at']
            or [info['path'], *info['args']] != plan['trainer_argv']):
        raise ValueError('Trainer identity changed; never follow a replacement automatically')
    bound_file(plan['config_path'], plan['config_sha256'])
    return info


def optimizer_history(container):
    result = subprocess.run(['docker', 'logs', container], capture_output=True, text=True, timeout=60)
    if result.returncode: raise RuntimeError('Cannot read the bound trainer optimizer history')
    # Docker preserves the application's stdout/stderr streams separately.
    # Event order is checked by cursor, so concatenation loses no budget data.
    raw = result.stdout + '\n' + result.stderr
    lines, steps = [], set()
    for line in raw.splitlines():
        if not line.startswith('{"event": "verified_optimizer_update",'): continue
        row = json.loads(line)
        step = row.get('step')
        if type(step) is not int or step < 0: raise ValueError('Invalid optimizer event cursor')
        # Only retain numeric budget fields, never training text or generic logs.
        metrics = row['measurements']
        item = {'event': 'verified_optimizer_update', 'step': step,
                'measurements': {'num_label_tokens': metrics['num_label_tokens']}}
        if 'num_tokens_per_step' in metrics: item['measurements']['num_tokens_per_step'] = metrics['num_tokens_per_step']
        lines.append(json.dumps(item, allow_nan=False)); steps.add(step)
    return '\n'.join(lines), max(steps, default=-1) + 1


def resource_status(output, model_bytes):
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name == 'MemAvailable': memory[name] = int(value.strip().split()[0]) * 1024
    free = shutil.disk_usage(output).free
    # Reserve host headroom for the trainer's CPU optimizer/checkpoint work.
    required_disk = 512 * GIB + max(128 * GIB, model_bytes * 2)
    available = memory.get('MemAvailable', 0)
    return {'ready': available >= 320 * GIB and free >= required_disk and (os.cpu_count() or 0) >= 32,
            'memory_available_bytes': available, 'minimum_memory_available_bytes': 320 * GIB,
            'disk_free_bytes': free, 'minimum_disk_free_bytes': required_disk,
            'cpu_count': os.cpu_count(), 'export_cpus': 16, 'export_memory_bytes': 192 * GIB}


def export_command(plan, number, source, destination, name):
    code = plan['quality_code_dir']
    return ['docker', 'create', '--name', name, '--label', 'yeta.quality.plan=' + protocol.sha(plan),
            '--label', 'yeta.quality.completed_updates=' + str(number), '--runtime', 'runc',
            '--network', 'none', '--cpus', '16', '--memory', '192g', '--memory-swap', '192g',
            '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--tmpfs', '/tmp:rw,nosuid,nodev,size=8g', '--env', 'NVIDIA_VISIBLE_DEVICES=void',
            '--env', 'CUDA_VISIBLE_DEVICES=', '--env', 'PYTHONDONTWRITEBYTECODE=1',
            '--env', 'OMP_NUM_THREADS=8', '--env', 'PYTHONPATH=' + code,
            '--volume', code + ':' + code + ':ro',
            '--volume', str(ORIGINAL) + ':' + str(ORIGINAL) + ':ro',
            '--volume', str(source) + ':' + str(source) + ':ro',
            '--volume', str(destination.parent) + ':' + str(destination.parent) + ':rw',
            '--workdir', code, '--entrypoint', 'python3', IMAGE,
            '-m', 'evaluation.qwen38_training_quality.export',
            '--source', str(source), '--destination', str(destination)]


def atomic_json(path, value):
    partial = path.with_name(path.name + '.pending')
    with partial.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    partial.replace(path)


def publish_once(path, value):
    if path.exists():
        if path.is_symlink() or json.loads(path.read_bytes()) != value:
            raise ValueError('Existing immutable publication differs')
        return
    checkpoint.write_json(path, value)


def verify_export_isolation(container, plan, number):
    template = ('{"runtime":{{json .HostConfig.Runtime}},"network":{{json .HostConfig.NetworkMode}},'
                '"cpus":{{json .HostConfig.NanoCpus}},"memory":{{json .HostConfig.Memory}},'
                '"memory_swap":{{json .HostConfig.MemorySwap}},"readonly":{{json .HostConfig.ReadonlyRootfs}},'
                '"devices":{{json .HostConfig.Devices}},"device_requests":{{json .HostConfig.DeviceRequests}},'
                '"plan":{{json (index .Config.Labels "yeta.quality.plan")}},'
                '"updates":{{json (index .Config.Labels "yeta.quality.completed_updates")}}}')
    info = json.loads(run(['docker', 'inspect', '--format', template, container]))
    expected = {'runtime': 'runc', 'network': 'none', 'cpus': 16 * 10**9,
                'memory': 192 * GIB, 'memory_swap': 192 * GIB, 'readonly': True,
                'plan': protocol.sha(plan), 'updates': str(number)}
    if any(info.get(k) != v for k, v in expected.items()) or info['devices'] or info['device_requests']:
        raise ValueError('CPU export isolation differs; do not start or accept it')


class Supervisor:
    def __init__(self, plan):
        self.plan = plan
        self.paths = validate_plan(plan)
        self.root = self.paths['output']
        self.root.mkdir(parents=True, exist_ok=True)
        # A second output directory must not accidentally create another CPU
        # export queue for the very same trainer/configuration.
        self.trainer_lock = (self.paths['config'].parent / 'quality-supervisor.lock').open('a')
        fcntl.flock(self.trainer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock = (self.root / 'supervisor.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.plan_sha = protocol.sha(plan)
        self.state_path = self.root / 'state.json'
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_bytes())
            if self.state.get('schema') != STATE_SCHEMA or self.state.get('plan_sha256') != self.plan_sha:
                raise ValueError('Do not reuse a supervisor directory for another plan')
        else:
            checkpoint.write_json(self.root / 'plan.json', plan)
            self.state = {'schema': STATE_SCHEMA, 'plan_sha256': self.plan_sha, 'created_at': now(),
                          'status': 'running', 'checkpoints': {str(n): {'state': 'waiting'} for n in SCHEDULE}}
        self.stop_requested = False
        self.save()

    def save(self):
        self.state['updated_at'] = now()
        atomic_json(self.state_path, self.state)

    def capture(self, number, text, completed, trainer):
        item = self.state['checkpoints'][str(number)]
        if item['state'] != 'waiting': return
        source = self.paths['checkpoints'] / ('epoch_0_step_' + str(number - 1))
        destination = self.root / 'snapshots' / ('updates-' + str(number))
        if destination.exists():
            identity = checkpoint.verify_snapshot(destination)
            budget = checkpoint.budget_from_events(text, number)
            manifest = json.loads((destination / 'snapshot-manifest.json').read_bytes())
            if (identity['arm'] != self.plan['arm'] or identity['completed_updates'] != number
                    or any(identity[key] != value for key, value in budget.items())
                    or manifest['source'] != str(source)
                    or checkpoint.digest(destination / 'train.json') != self.plan['config_sha256']):
                raise ValueError('Existing snapshot belongs to another source or token budget')
        elif source.is_dir() and not (source / '.incomplete').exists() and completed >= number:
            budget = checkpoint.budget_from_events(text, number)
            item.update(state='snapshotting', started_at=now()); self.save()
            identity = checkpoint.snapshot(source, destination, arm=self.plan['arm'], completed_updates=number,
                                           budget=budget, config=self.paths['config'])
        else:
            # Three newer every-five checkpoints prove retention passed this
            # scheduled source. An exited trainer may also leave a coverage gap.
            newer = [p for p in self.paths['checkpoints'].glob('epoch_0_step_*')
                     if re.fullmatch(r'epoch_0_step_\d+', p.name) and int(p.name.rsplit('_', 1)[1]) > number - 1
                     and p.is_dir() and not (p / '.incomplete').exists()]
            if not trainer['running'] or len(newer) >= 3:
                item.update(state='unavailable', reason='trainer_ended_before_capture' if not trainer['running']
                            else 'checkpoint_retention_passed', observed_completed_updates=completed)
            return
        item.update(state='snapshot_complete', snapshot=str(destination), checkpoint=identity,
                    snapshot_manifest_sha256=checkpoint.digest(destination / 'snapshot-manifest.json'), captured_at=now())
        self.save()

    def finish_export(self, number, item):
        info = inspect_container(item['export_container'])
        if (info['id'] != item['export_container'] or info['image'] != IMAGE
                or [info['path'], *info['args']] != item['export_argv']):
            raise ValueError('Export container identity changed')
        verify_export_isolation(item['export_container'], self.plan, number)
        if info['running']: return True
        destination = Path(item['export_destination'])
        receipt_path = destination / 'export-receipt.json'
        if info['exit_code'] != 0 or info['oom_killed'] or not receipt_path.is_file():
            item.update(state='export_failed', exit_code=info['exit_code'], oom_killed=info['oom_killed'],
                        finished_at=now(), reason='conversion_failed_no_automatic_retry')
            return False
        receipt = json.loads(receipt_path.read_bytes())
        if (receipt.get('schema') != 'yeta.qwen38-training-quality-hf-export/v1'
                or receipt.get('status') != 'complete' or receipt.get('exact_tensor_readback') is not True
                or receipt.get('gpu_used') is not False or receipt.get('checkpoint') != item['checkpoint']
                or receipt.get('source_snapshot_manifest_sha256') != item['snapshot_manifest_sha256']):
            raise ValueError('Export receipt did not bind the captured checkpoint')
        receipt_sha = checkpoint.digest(receipt_path)
        publish_once(self.root / ('ready-updates-' + str(number) + '.json'), {
            'schema': 'yeta.qwen38-quality-checkpoint-ready/v1', 'checkpoint': item['checkpoint'],
            'model_path': str(destination), 'export_receipt_sha256': receipt_sha,
            'source_snapshot_manifest_sha256': item['snapshot_manifest_sha256'],
            'plan_sha256': self.plan_sha, 'evaluation_launched': False})
        item.update(state='export_complete', export_receipt_sha256=receipt_sha, finished_at=now())
        return False

    def exports(self):
        active = False
        for number in SCHEDULE:
            item = self.state['checkpoints'][str(number)]
            if item['state'] == 'export_running': active = self.finish_export(number, item) or active
        if active or self.stop_requested: return
        for number in SCHEDULE:
            item = self.state['checkpoints'][str(number)]
            if item['state'] != 'snapshot_complete': continue
            source = Path(item['snapshot'])
            manifest = json.loads((source / 'snapshot-manifest.json').read_bytes())
            resources = resource_status(self.root, sum(x['bytes'] for x in manifest['files']))
            self.state['resources'] = resources
            if not resources['ready']: return
            verify_code(self.plan['quality_code_dir'], self.plan['quality_code_manifest_sha256'])
            destination = self.root / 'exports' / ('updates-' + str(number))
            destination.parent.mkdir(exist_ok=True)
            name = 'quality-cpu-' + self.plan_sha[:16] + '-' + str(number)
            command = export_command(self.plan, number, source, destination, name)
            # Record the intent before creating a detached container. A crash
            # during create/start is an explicit reconciliation state, not a retry.
            item.update(state='export_launch_pending', export_name=name, export_command=command,
                        export_destination=str(destination), export_started_at=now())
            self.save()
            container = run(command).strip()
            if not protocol.hash_string(container): raise ValueError('Invalid export container ID')
            info = inspect_container(container)
            expected_args = ['python3', '-m', 'evaluation.qwen38_training_quality.export',
                             '--source', str(source), '--destination', str(destination)]
            if info['image'] != IMAGE or [info['path'], *info['args']] != expected_args:
                raise ValueError('Created CPU export command does not match the requested converter')
            verify_export_isolation(container, self.plan, number)
            item.update(state='export_created', export_container=container,
                        export_argv=[info['path'], *info['args']]); self.save()
            run(['docker', 'start', container])
            item['state'] = 'export_running'; self.save()
            return

    def tick(self):
        trainer = verify_trainer(self.plan)
        text, completed = optimizer_history(self.plan['trainer_container'])
        self.state.update(trainer_running=trainer['running'], observed_completed_updates=completed)
        for number in SCHEDULE: self.capture(number, text, completed, trainer)
        self.exports()
        states = {x['state'] for x in self.state['checkpoints'].values()}
        if states == {'export_complete'}: self.state['status'] = 'complete'
        elif states <= {'export_complete', 'export_failed', 'unavailable'}: self.state['status'] = 'incomplete'
        else: self.state['status'] = 'running'
        self.save()
        return self.state['status'] in ('complete', 'incomplete')

    def run(self, *, once=False):
        interrupted = {'snapshotting', 'export_launch_pending', 'export_created'}
        if any(x['state'] in interrupted for x in self.state['checkpoints'].values()):
            raise ValueError('An interrupted mutation needs explicit identity reconciliation; no automatic retry')
        while not self.stop_requested:
            finished = self.tick()
            if finished or once: break
            deadline = time.monotonic() + self.plan.get('poll_seconds', 10)
            while not self.stop_requested and time.monotonic() < deadline: time.sleep(0.25)
        if self.stop_requested:
            self.state['status'] = 'paused'; self.save()
        return {'status': self.state['status'], 'arm': self.plan['arm'],
                'observed_completed_updates': self.state.get('observed_completed_updates'),
                'checkpoints': {n: row['state'] for n, row in self.state['checkpoints'].items()},
                'gpu_evaluation_launched': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    plan_path = bound_file(args.plan, args.plan_sha256)
    supervisor = Supervisor(json.loads(plan_path.read_bytes()))
    def stop(_signum, _frame): supervisor.stop_requested = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        print(json.dumps(supervisor.run(once=args.once)), flush=True)
    except Exception as error:
        supervisor.state.update(status='needs_attention', error_type=type(error).__name__, error=str(error))
        supervisor.save()
        raise


if __name__ == '__main__': main()
