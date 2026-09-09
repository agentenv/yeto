"""Fixed checkpoint and outcome policy; no scheduling or model calls."""
import hashlib
import json
import re

SCHEMA = 'yeta.qwen38-training-quality-protocol/v1'
ARMS = ('no-cot-turn-v2', 'masked-cot-native-gap-v3')
CONTEXT = 262144
COMPACTION = 196608
TEMPLATE_SHA = 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'


def scheduled(completed_updates):
    return type(completed_updates) is int and (completed_updates in (25, 50) or completed_updates >= 100 and completed_updates % 100 == 0)


def checkpoints_through(completed_updates):
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError('Completed updates must be a nonnegative integer')
    return [i for i in (25, 50, *range(100, completed_updates + 1, 100)) if i <= completed_updates]


def next_checkpoint(completed_updates):
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError('Completed updates must be a nonnegative integer')
    return next(i for i in (25, 50, *range(100, (completed_updates // 100 + 2) * 100, 100)) if i > completed_updates)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def hash_string(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def checkpoint_identity(*, arm, completed_updates, checkpoint_sha256, cumulative_supervised_tokens,
                        cumulative_input_tokens=None, training_gpu_hours=None):
    if arm not in ARMS or not scheduled(completed_updates):
        raise ValueError('Use a named corrected arm and scheduled completed-update checkpoint')
    if not hash_string(checkpoint_sha256):
        raise ValueError('Bind the actual completed checkpoint artifact manifest SHA256')
    for name, value in [('supervised', cumulative_supervised_tokens), ('input', cumulative_input_tokens)]:
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError('Cumulative ' + name + ' tokens must be nonnegative integers')
    if cumulative_supervised_tokens is None:
        raise ValueError('Supervised-token budget is required for fair comparisons')
    if training_gpu_hours is not None and (type(training_gpu_hours) not in (int, float) or training_gpu_hours < 0):
        raise ValueError('Invalid training GPU hours')
    return {'arm': arm, 'completed_updates': completed_updates, 'zero_based_checkpoint_step': completed_updates - 1,
            'checkpoint_manifest_sha256': checkpoint_sha256,
            'cumulative_supervised_tokens': cumulative_supervised_tokens,
            'cumulative_input_tokens': cumulative_input_tokens, 'training_gpu_hours': training_gpu_hours,
            'equal_updates_imply_equal_token_budget': False}


def policy():
    return {'schema': SCHEMA, 'completed_updates': [25, 50, 100], 'thereafter_every_updates': 100,
            'single_attempt_per_task': True, 'best_of_selection': False, 'model_failure_retries': 0,
            'synthetic_tasks': 16, 'same_suite_for_both_arms': True, 'full_terminal_bench': False,
            'checkpoint_freshness_required': True, 'native_initial_and_incremental_parity_required': True,
            'model_context_window': CONTEXT, 'model_auto_compact_token_limit': COMPACTION,
            'reasoning_effort': 'xhigh', 'codex_version': '0.142.5', 'temperature': 1.0, 'top_p': 1.0,
            'expected_quality_improvement_step': None, 'training_data': False}
