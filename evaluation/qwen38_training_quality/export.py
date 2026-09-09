"""Export a hash-bound corrected checkpoint using the qualified CPU converter."""
import argparse
import json
from pathlib import Path

from .checkpoint import digest, verify_snapshot, write_json
from monitoring import export_masked_cot_pilot_checkpoint_20260909 as converter

CONVERTER_SHA = 'ab01af426b57e511f550d2768f77ba440ef1ec874a2e14a6505fdbd424c9a036'


def export(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    checkpoint = verify_snapshot(source)
    if digest(converter.__file__) != CONVERTER_SHA:
        raise ValueError('Qualified CPU conversion code changed')
    converter.main(['--source',str(source),'--destination',str(destination),
        '--expected-metadata-sha256',digest(source/'model/.metadata'),
        '--completed-updates',str(checkpoint['completed_updates'])])
    # Keep the converter's original evidence, and publish this purpose-specific
    # binding separately before atomically replacing the public receipt.
    original = destination/'export-receipt.json'
    receipt = json.loads(original.read_bytes())
    if (receipt['status'] != 'complete' or receipt['exact_tensor_readback'] is not True
            or receipt['completed_optimizer_updates'] != checkpoint['completed_updates']):
        raise ValueError('Converter did not qualify the requested checkpoint')
    with (destination/'cpu-converter-receipt.json').open('xb') as stream:stream.write(original.read_bytes())
    receipt.update(schema='yeta.qwen38-training-quality-hf-export/v1',
        purpose='fixed-training-quality-evaluation',checkpoint=checkpoint,
        checkpoint_manifest_sha256=checkpoint['checkpoint_manifest_sha256'],
        source_snapshot_manifest_sha256=digest(source/'snapshot-manifest.json'),
        cpu_converter_receipt_sha256=digest(destination/'cpu-converter-receipt.json'),
        binding_code_sha256=digest(__file__))
    pending=destination/'quality-export-receipt.pending.json'
    write_json(pending,receipt)
    pending.replace(original)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--destination',required=True,type=Path)
    args=parser.parse_args()
    receipt=export(args.source,args.destination)
    print(json.dumps({'status':receipt['status'],'checkpoint':receipt['checkpoint']}))
