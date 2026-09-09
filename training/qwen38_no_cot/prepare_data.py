"""CPU-only source adapters and deterministic no-CoT SFT preparation.

Source data is never rewritten. Every exclusion is counted and recorded by
hashed identity; unrecognized trace shapes do not silently become training text.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re

SEED = 'qwen38-no-cot-sft-20260908/v1'
VERSION = 'normalized-codex-and-replay-nocot/v1'
PRIVATE = {'reasoning', 'thinking', 'analysis', 'chain_of_thought', 'reasoning_text',
           'redacted_thinking', 'encrypted_reasoning', 'encrypted_content'}


class UnsupportedTrace(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def visible_text(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out = []
        for block in value:
            if not isinstance(block, dict):
                raise UnsupportedTrace('untyped_content_block')
            kind = block.get('type')
            if kind in PRIVATE:
                continue
            if kind in {'text', 'input_text', 'output_text'} and isinstance(block.get('text'), str):
                out.append(block['text'])
            elif kind == 'refusal' and isinstance(block.get('refusal'), str):
                out.append(block['refusal'])
            else:
                raise UnsupportedTrace('nontext_content_block')
        return ''.join(out)
    raise UnsupportedTrace('nontext_message')


def observation(value):
    if isinstance(value, (str, list)) or value is None:
        return visible_text(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    raise UnsupportedTrace('unsupported_observation')


def tool_call(block, *, name=None, call_id=None, counts=None):
    counts = counts if counts is not None else Counter()
    fn = block.get('function', block.get('custom', block))
    name = name or fn.get('name') or fn.get('function_name')
    identity = call_id or block.get('call_id') or block.get('tool_call_id') or block.get('id')
    if not isinstance(name, str) or not name or not isinstance(identity, str) or not identity:
        raise UnsupportedTrace('tool_call_missing_identity')
    custom = block.get('type') in {'custom', 'custom_tool_call'} or 'custom' in block
    if custom:
        value = fn.get('input')
        if not isinstance(value, str):
            raise UnsupportedTrace('custom_input_not_string')
        args = {'input': value}
        counts['custom_raw_input_function_adaptations'] += 1
    elif block.get('type') == 'web_search_call' and isinstance(block.get('action'), dict):
        args = {'action': block['action']}
        counts['builtin_web_action_as_function'] += 1
    else:
        args = fn.get('arguments', fn.get('input'))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                raise UnsupportedTrace('nonjson_function_arguments') from exc
        if not isinstance(args, dict):
            raise UnsupportedTrace('nonobject_function_arguments')
    return {'id': identity, 'type': 'function', 'function': {'name': name, 'arguments': args}}


def canonical_messages(events):
    messages, counts = [], Counter()
    previous_key = None
    for event_index, event in enumerate(events):
        kind, role = event.get('kind'), event.get('role')
        if kind in {'reasoning', 'empty_message', 'capture_metadata'} or event.get('channel') in PRIVATE:
            counts['reasoning_or_nonvisible_events_removed'] += 1
            continue
        if kind == 'tool_definition':
            counts['capability_schemas_omitted'] += 1
            continue
        data = event.get('data', {})
        block = data.get('block', data) if isinstance(data, dict) else data
        pointer = event.get('source', {}).get('pointer', event.get('event_id'))
        if kind == 'unknown' and isinstance(block, dict) and block.get('type') == 'tool_search_output':
            kind, role = 'tool_result', 'tool'
            counts['tool_search_output_as_observation'] += 1
        elif (kind == 'unknown' and block is None and not event.get('content')
              and event_index == len(events) - 1 and isinstance(pointer, str) and pointer.startswith('/response/')):
            counts['empty_terminal_capture_event_omitted'] += 1
            continue
        elif kind in {'media', 'unknown', 'compaction'}:
            raise UnsupportedTrace('unsupported_canonical_' + kind)
        if role == 'agent':
            role = 'assistant'
        if role == 'developer':
            role = 'system'
            counts['developer_as_chronological_system'] += 1
        key = re.sub(r'/content/\d+$', '', pointer) if isinstance(pointer, str) else None
        if kind == 'message':
            if role not in {'system', 'user', 'assistant'}:
                raise UnsupportedTrace('unsupported_message_role')
            if isinstance(block, dict) and block.get('type') not in {None, 'message', 'agent_message', 'text', 'input_text', 'output_text', 'refusal'}:
                raise UnsupportedTrace('unsupported_message_block')
            text = visible_text(event.get('content'))
            if not text:
                continue
            if previous_key == (key, role) and messages and messages[-1]['role'] == role and 'tool_calls' not in messages[-1]:
                messages[-1]['content'] += text
                counts['same_source_message_blocks_joined'] += 1
            else:
                messages.append({'role': role, 'content': text})
            previous_key = (key, role)
        elif kind == 'tool_call':
            if role != 'assistant' or not isinstance(block, dict):
                raise UnsupportedTrace('invalid_canonical_tool_call')
            if (event_index == len(events) - 1 and isinstance(pointer, str) and pointer.startswith('/response/output/')
                    and block.get('type') == 'function_call' and 'arguments' not in block
                    and 'input' not in block and 'function' not in block):
                counts['incomplete_terminal_call_omitted'] += 1
                continue
            call = tool_call(block, name=event.get('name'), call_id=event.get('call_id'), counts=counts)
            messages.append({'role': 'assistant', 'content': '', 'tool_calls': [call]})
            previous_key = None
        elif kind == 'tool_result':
            if not isinstance(block, dict):
                raise UnsupportedTrace('invalid_canonical_tool_result')
            identity = event.get('call_id') or block.get('call_id') or block.get('tool_call_id') or block.get('tool_use_id')
            if not isinstance(identity, str) or not identity:
                raise UnsupportedTrace('tool_result_missing_identity')
            # Source-normalized content is the original observation text, not
            # the judge's shortened or reasoning-stripped projection.
            messages.append({'role': 'tool', 'tool_call_id': identity, 'content': observation(event.get('content'))})
            previous_key = None
        else:
            raise UnsupportedTrace('unknown_canonical_kind')
    validate_messages(messages)
    return messages, dict(counts)


def validate_messages(messages):
    calls = {}
    seen_observations = set()
    if not any(m['role'] == 'user' for m in messages):
        raise UnsupportedTrace('no_user_message')
    if not any(m['role'] == 'assistant' for m in messages):
        raise UnsupportedTrace('no_assistant_message')
    for message in messages:
        for call in message.get('tool_calls', []):
            identity = call['id']
            if identity in calls:
                raise UnsupportedTrace('duplicate_tool_call_id')
            calls[identity] = call['function']['name']
        if message['role'] == 'tool':
            identity = message['tool_call_id']
            if identity not in calls or identity in seen_observations:
                raise UnsupportedTrace('orphan_or_duplicate_tool_observation')
            seen_observations.add(identity)


def rollout_messages(events):
    messages, counts = [], Counter()
    session_id = None
    # Response items are authoritative; event_msg copies would double-count.
    if not any(e.get('type') == 'response_item' for e in events):
        raise UnsupportedTrace('rollout_without_response_items')
    for event in events:
        payload = event.get('payload')
        if not isinstance(payload, dict):
            continue
        if event.get('type') == 'session_meta':
            session_id = payload.get('session_id') or payload.get('id') or session_id
            base = payload.get('base_instructions')
            if isinstance(base, dict):
                base = base.get('text')
            if base:
                messages.append({'role': 'system', 'content': visible_text(base)})
        if event.get('type') != 'response_item':
            continue
        kind = payload.get('type')
        if kind in PRIVATE or payload.get('channel') in PRIVATE:
            counts['reasoning_items_removed'] += 1
            continue
        if kind == 'message':
            role = payload.get('role')
            if role == 'developer':
                role = 'system'
                counts['developer_as_chronological_system'] += 1
            if role not in {'system', 'user', 'assistant'}:
                raise UnsupportedTrace('unsupported_rollout_role')
            content = visible_text(payload.get('content'))
            if content:
                messages.append({'role': role, 'content': content})
        elif kind in {'function_call', 'custom_tool_call'}:
            messages.append({'role': 'assistant', 'content': '', 'tool_calls': [tool_call(payload, counts=counts)]})
        elif kind in {'function_call_output', 'custom_tool_call_output'}:
            identity = payload.get('call_id')
            if not isinstance(identity, str) or not identity:
                raise UnsupportedTrace('tool_result_missing_identity')
            messages.append({'role': 'tool', 'tool_call_id': identity, 'content': observation(payload.get('output'))})
        else:
            raise UnsupportedTrace('unsupported_rollout_response_item')
    if not isinstance(session_id, str) or not session_id:
        raise UnsupportedTrace('rollout_missing_session_id')
    validate_messages(messages)
    return messages, dict(counts), 'session:' + session_id


def atif_messages(doc):
    if not isinstance(doc.get('steps'), list):
        raise UnsupportedTrace('not_atif')
    messages, counts = [], Counter()
    for step in doc['steps']:
        role = {'agent': 'assistant', 'developer': 'system'}.get(step.get('source'), step.get('source'))
        if role not in {'system', 'user', 'assistant'}:
            raise UnsupportedTrace('unsupported_atif_step_source')
        content = visible_text(step.get('message'))
        message = {'role': role, 'content': content}
        raw = step.get('tool_calls') or step.get('extra', {}).get('requested_tool_calls') or []
        if raw:
            message['tool_calls'] = [tool_call(c, counts=counts) for c in raw]
        if content or raw:
            messages.append(message)
        if 'thinking' in step.get('extra', {}):
            counts['atif_thinking_omitted'] += 1
        observation_block = step.get('observation') or {}
        for result in observation_block.get('results', []):
            identity = result.get('source_call_id')
            if not isinstance(identity, str) or not identity:
                raise UnsupportedTrace('atif_result_missing_call_id')
            messages.append({'role': 'tool', 'tool_call_id': identity, 'content': observation(result.get('content'))})
    validate_messages(messages)
    identity = doc.get('session_id')
    if not isinstance(identity, str) or not identity:
        raise UnsupportedTrace('atif_missing_session_id')
    return messages, dict(counts), 'session:' + identity


_renderer = None
_normalizer_sha = None


def init_worker(tokenizer_dir):
    global _renderer, _normalizer_sha
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from .render import Qwen38Renderer
    _renderer = Qwen38Renderer(tokenizer_dir)
    _normalizer_sha = sha_file(__file__)


def convert_job(job):
    from .render import training_row
    try:
        raw = Path(job['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != job['sha256']:
            raise UnsupportedTrace('source_file_hash_mismatch')
        if job['format'] == 'canonical':
            messages, counts = canonical_messages([json.loads(l) for l in raw.splitlines() if l.strip()])
            group = job['group_id']
            if not isinstance(group, str) or not group:
                raise UnsupportedTrace('missing_source_session_group')
        elif job['format'] == 'rollout':
            messages, counts, group = rollout_messages([json.loads(l) for l in raw.splitlines() if l.strip()])
        elif job['format'] == 'atif':
            messages, counts, group = atif_messages(json.loads(raw))
        else:
            raise UnsupportedTrace('unknown_job_format')
        # No capability tool schemas go into either API or the historical
        # rendering. Actual calls/results remain fully present in messages.
        rendered = _renderer.render(messages)
        provenance = {**job, 'normalization_version': VERSION, 'source_normalization_counts': counts,
                      'normalizer_sha256': _normalizer_sha,
                      'messages_sha256': digest(messages), 'original_reasoning_used': False,
                      'capability_schemas_included': False, 'custom_adapter': 'raw-input-as-function-input-string/v1'}
        row = training_row(rendered, group_id=group, source=job['source'], provenance=provenance)
        return {'ok': True, 'job': job, 'row': row, 'messages_sha256': digest(messages)}
    except Exception as exc:
        reason = str(exc) if isinstance(exc, UnsupportedTrace) else type(exc).__name__
        return {'ok': False, 'identity': job['identity'], 'source': job['source'], 'reason': reason}


def collect_jobs(args):
    jobs = []
    if args.codex_manifest:
        expected = args.codex_manifest_sha256
        if not expected or sha_file(args.codex_manifest) != expected:
            raise ValueError('Frozen Codex manifest hash mismatch')
        for lineno, line in enumerate(Path(args.codex_manifest).open(), 1):
            row = json.loads(line)
            if row['quality_score'] < 4 or row['overall_confidence_score'] < 3:
                raise ValueError('Source manifest violates authorized score thresholds')
            jobs.append({'identity': row['capture_id'], 'path': row['archive_path'], 'sha256': row['archive_file_sha256'],
                         'group_id': row['group_id'], 'format': 'canonical', 'source': 'trace',
                         'manifest_sha256': expected, 'manifest_line': lineno,
                         'label_sha256': row['label_file_sha256'], 'global_seal_sha256': row['global_seal_sha256']})
    if args.replay_root:
        replay_root = Path(args.replay_root)
        for inventory in sorted(replay_root.glob('*_files.jsonl')):
            family = inventory.name.removesuffix('_files.jsonl')
            receipt = json.loads((replay_root / (family + '_extraction.json')).read_text())
            if sha_file(inventory) != receipt['inventory_sha256']:
                raise ValueError('Replay inventory hash mismatch')
            for lineno, line in enumerate(inventory.open(), 1):
                record = json.loads(line)
                relative = record['path']
                is_rollout = Path(relative).name.startswith('rollout-') and relative.endswith('.jsonl')
                is_atif = Path(relative).name == 'trajectory.json'
                if not (is_rollout or is_atif):
                    continue
                jobs.append({'identity': 'replay:' + record['sha256'],
                             'path': str(Path(receipt['directory']) / relative), 'sha256': record['sha256'],
                             'format': 'rollout' if is_rollout else 'atif', 'source': 'replay', 'family': family,
                             'inventory_sha256': receipt['inventory_sha256'], 'inventory_line': lineno})
    unique = {}
    for job in jobs:
        unique.setdefault((job['source'], job['sha256']), job)
    jobs = list(unique.values())
    for earlier_manifest in getattr(args, 'exclude_source_jobs', []) or []:
        with Path(earlier_manifest).open() as earlier:
            done = {(j['source'], j['sha256']) for j in map(json.loads, earlier)}
        jobs = [j for j in jobs if (j['source'], j['sha256']) not in done]
    jobs.sort(key=lambda j: hashlib.sha256((SEED + ':shuffle:' + j['identity']).encode()).hexdigest())
    if args.limit:
        jobs = jobs[:args.limit]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codex-manifest')
    ap.add_argument('--codex-manifest-sha256')
    ap.add_argument('--replay-root')
    ap.add_argument('--tokenizer-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--exclude-source-jobs', action='append', default=[])
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    jobs = collect_jobs(args)
    (out / 'source_jobs.jsonl').write_text(''.join(json.dumps(j, sort_keys=True) + '\n' for j in jobs))
    counts, source_tokens, reasons = Counter(), Counter(), Counter()
    seen_messages = set()
    paths = {'train': out / 'train.jsonl', 'validation': out / 'validation.jsonl'}
    with paths['train'].open('x') as train, paths['validation'].open('x') as valid, (out / 'exclusions.jsonl').open('x') as excluded:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args.tokenizer_dir,)) as pool:
            def bounded_results():
                pending = deque()
                source = iter(jobs)
                for _ in range(args.workers * 2):
                    job = next(source, None)
                    if job is not None:
                        pending.append(pool.submit(convert_job, job))
                while pending:
                    yield pending.popleft().result()
                    job = next(source, None)
                    if job is not None:
                        pending.append(pool.submit(convert_job, job))

            for i, result in enumerate(bounded_results(), 1):
                if not result['ok']:
                    reasons[result['reason']] += 1
                    excluded.write(json.dumps(result, sort_keys=True) + '\n')
                    continue
                if result['messages_sha256'] in seen_messages:
                    reasons['duplicate_normalized_messages'] += 1
                    continue
                seen_messages.add(result['messages_sha256'])
                row = result['row']
                split = 'validation' if int(hashlib.sha256((SEED + ':split:' + row['group_id']).encode()).hexdigest()[:8], 16) % 100 == 0 else 'train'
                stream = valid if split == 'validation' else train
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                counts[split + '/' + row['source']] += 1
                audit = row['metadata']['sequence_audit']
                source_tokens[split + '/' + row['source'] + '/input'] += len(row['input_ids'])
                source_tokens[split + '/' + row['source'] + '/targets'] += sum(t != -100 for t in row['labels'])
                source_tokens['dropped_input_tokens'] += audit['dropped_input_tokens']
                source_tokens['dropped_target_tokens'] += audit['dropped_supervised_tokens']
                if i % 100 == 0:
                    train.flush(); valid.flush(); excluded.flush()
                    print(json.dumps({'jobs_processed': i, 'counts': dict(counts), 'exclusions': dict(reasons)}), flush=True)
    summary = {'version': VERSION, 'jobs': len(jobs), 'counts': dict(counts), 'token_counts': dict(source_tokens),
               'exclusions': dict(reasons), 'seed': SEED, 'shuffle': 'source-identity-seeded-sha256-order',
               'group_split': 'sha256(seed:split:group_id) mod100 ==0 validation',
               'shards': {name: {'path': str(path), 'sha256': sha_file(path)} for name, path in paths.items()}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
