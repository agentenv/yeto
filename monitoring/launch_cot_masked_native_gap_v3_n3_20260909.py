"""Launch the authorized full masked-CoT experiment on an idle n3 host.

The plan binds a fresh recipe, exact code bundle and CPU preflight receipt.
No existing containers or checkpoints are stopped, removed, or reused.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

IMAGE = 'sha256:b0bd0e50b29cb0a3ec69c3cbbb49652d3c1aabedbfe22ee0fa6d053b6bb33aee'
ROOT = Path('/data/sft_baseline_20260908')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def launch(plan):
    raise RuntimeError('Retired invalid v3 launcher; use launch_cot_masked_native_gap_v4_n3_20260910.py')
    # Kept below only as historical evidence; it is deliberately unreachable.
    assert plan['schema'] == 'qwen38-native-gap-masked-launch-plan/v3'
    run, code = Path(plan['run']), Path(plan['code'])
    assert run.parent == ROOT/'runs' and code.parent == ROOT/'code'
    assert run.resolve(strict=True) == run and code.resolve(strict=True) == code
    assert run.name.startswith('cot-masked-native-gap-v3-')
    for path, sha in [(run/'train.json', plan['config_sha256']),
                      (Path(plan['cpu_receipt']), plan['cpu_receipt_sha256']),
                      (code/'code-manifest.json', plan['code_manifest_sha256'])]:
        assert digest(path) == sha, str(path)
    manifest = json.loads((code/'code-manifest.json').read_bytes())
    assert isinstance(manifest['files'], list) and manifest['files']
    seen = set()
    for entry in manifest['files']:
        relative = Path(entry['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert entry['path'] not in seen
        seen.add(entry['path'])
        path = code/relative
        assert path.is_file() and path.resolve(strict=True) == path
        assert path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256']
    cpu = json.loads(Path(plan['cpu_receipt']).read_bytes())
    assert cpu['status'] == 'passed' and cpu['config_sha256'] == plan['config_sha256']
    assert cpu['runtime_image'] == IMAGE
    assert cpu['code_manifest_sha256']==plan['code_manifest_sha256'] and cpu['cuda_initialized'] is False
    assert cpu['loss_internal_shift'] is False and cpu['causal_shift']=='dataset-once-before-CP'
    assert cpu['loss_source_sha256']=='fd4754cb1eb4f28373a75fff9efef0524768bf3a779659f8aa2c3df3b392cbad'
    assert cpu['training_contract'] == 'qwen38-xhigh-native-gap-cot-loss-zero/v3'
    assert cpu['semantic_quality_qualified'] is False
    assert cpu['dataset_rows'].get('train',0)>0 and cpu['dataset_rows'].get('validation',0)>0
    assert cpu['runtime_dataset_rows']['validation']>0
    assert cpu['frozen_valid_candidate_count']==31504
    assert cpu['included_valid_candidate_count']+cpu['omitted_valid_candidate_count']+cpu['excluded_valid_candidate_count']==31504
    assert cpu['gap_omission_policy']=='keep-native-leading-gap-omit-only-conflicting-cot/v3'
    assert cpu['unreviewed_gap_count']==cpu['included_valid_candidate_count']
    assert cpu['full_native_mask_validation'] and cpu['original_gap_positions_preserved']
    assert cpu['reasoning_relocated'] is False and cpu['known_session_split_verified']
    assert cpu['unknown_sessions_train_only']
    config = json.loads((run/'train.json').read_bytes())
    dataset = Path(config['dataset']['path_or_dataset']).parent
    assert dataset.parent == ROOT/'datasets' and dataset.resolve(strict=True) == dataset
    assert digest(dataset/'manifest.json') == cpu['manifest_sha256']
    assert digest(Path(config['dataset']['index_path'])) == cpu['index_sha256'] == config['dataset']['index_sha256']
    assert digest(dataset/'source-manifest.json')==cpu['source_manifest_sha256']
    assert digest(dataset/'source-COMPLETE.json')==cpu['source_complete_sha256']
    assert config['training_contract']=='qwen38-xhigh-native-gap-cot-loss-zero/v3'
    assert config['dataset']['_target_']=='training.qwen38_native_gap_v3.masked_data.GeneratedMaskedTokenDataset'
    assert config['validation_dataset']['_target_']==config['dataset']['_target_']
    assert config['experiment_phase'] == 'train'
    assert config['validation_dataset']['max_samples']==32 and config['validation_dataset']['max_input_tokens']==32768
    assert config['checkpoint']['checkpoint_dir'] == str(run/'checkpoints')
    assert not config['checkpoint'].get('restore_from')
    assert config['step_scheduler']['num_epochs'] == 1
    assert config['step_scheduler'].get('max_steps') is None
    name = 'qwen38-' + run.name.lower()
    command = ['docker','run','-d','--name',name,'--gpus','all','--ipc','host',
        '--ulimit','memlock=-1','--ulimit','stack=67108864','--cpus','64','--memory','1400g',
        '-v',str(ROOT)+':'+str(ROOT),'-v',str(run/'triton-cache')+':/root/.triton/cache',
        '-w',str(code),'-e','PYTHONPATH='+str(code),'-e','YETA_TRAINING_IMAGE='+IMAGE,
        '-e','WANDB_MODE=offline','-e','WANDB_DISABLE_CODE=true','-e','WANDB_LOG_MODEL=false',
        '-e','TOKENIZERS_PARALLELISM=false','-e','OMP_NUM_THREADS=4',
        '-e','PYTORCH_ALLOC_CONF=expandable_segments:True','--entrypoint','torchrun',IMAGE,
        '--standalone','--nnodes=1','--nproc-per-node=8','-m','training.qwen38_native_gap_v3.masked_train',
        '--config',str(run/'train.json'),'--mode','train','--validate-memory-repair-in-production']
    with (run.parent/'cot-masked-native-gap-v3-launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        compute = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                          '--format=csv,noheader,nounits'],text=True).strip()
        assert not compute, 'GPU compute processes remain'
        gpus = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total',
                                       '--format=csv,noheader,nounits'],text=True).strip().splitlines()
        assert len(gpus) == 8 and all(int(row.split(',')[1]) < 1024 for row in gpus)
        names = subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines()
        assert name not in names, 'Inspect the existing container instead of duplicating it'
        assert not (run/'launch.json').exists(), 'Launch receipt already exists'
        assert not (run/'checkpoints').exists() or not any((run/'checkpoints').iterdir())
        (run/'triton-cache').mkdir(exist_ok=True)
        receipt = {**plan, 'schema':'qwen38-native-gap-masked-gpu-launch/v3',
            'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'host':'ubuntu@100.66.92.111','name':name,'argv':command,'prelaunch_gpus':gpus}
        (run/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
        result = subprocess.run(command,text=True,capture_output=True)
        receipt['docker_exit_code'] = result.returncode
        if result.returncode == 0:
            receipt['container'] = result.stdout.strip()
            assert len(receipt['container']) == 64
        else:
            receipt['launch_error'] = result.stderr[:1500]
        (run/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
        return result.returncode


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    raise SystemExit(launch(json.loads(Path(parser.parse_args().plan).read_bytes())))
