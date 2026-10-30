"""S28 CP2: the THD ``seq_idx`` patch must pass the global pack length at every CP size.

Megatron-core's Mamba context parallel is all-to-all, not a sequence split:
``MambaContextParallel.pre_conv_ssm`` (``megatron/core/ssm/mamba_context_parallel.py``)
gathers the whole global sequence onto every CP rank and shards the heads instead, so the
convolution and the SSM scan see every token of the pack and ``seq_idx`` has to span the
global pack. The patch used to return ``{}`` when ``cp_size != 1``, which left ``seq_idx``
unset and let one packed sample's Mamba state flow into the next.

The numerical evidence sits outside pytest, because it needs torch and 2 to 4 gloo ranks:
``/home/michael/work/s1-runs/s28-g120/cp_seqidx_check.py``, log ``/tmp/cp2-check.log``.
"""

from yeto.rl.adapters.verl import patch_verl


def _helper(supports_total_tokens: bool = True):
    """Exec the patched helper's source and return the function object."""
    scope: dict = {"_packed_seq_params_supports": lambda name: supports_total_tokens}
    exec(compile(patch_verl.SEQ_IDX_HELPER, "seq_idx_helper.py", "exec"), scope)
    return scope["_yeto_seq_idx_args"]


def test_seq_idx_patch_passes_the_global_pack_length_at_every_cp_size():
    helper = _helper()
    lengths = [24, 40]  # the global padded sample lengths of one THD pack
    for cp_size in (1, 2, 4, 8):
        assert helper(cp_size, lengths) == {"total_tokens": 64}, cp_size


def test_seq_idx_patch_stays_quiet_when_megatron_has_no_total_tokens_field():
    # An older Megatron-core builds PackedSeqParams without ``total_tokens``; passing it
    # would raise a TypeError on every forward, so the patch must return no kwargs.
    helper = _helper(supports_total_tokens=False)
    assert helper(1, [24, 40]) == {}
    assert helper(2, [24, 40]) == {}


def test_seq_idx_patch_no_longer_bails_out_on_cp_above_one():
    # Pin the regression: the old body was ``if cp_size != 1 or not ...: return {}``.
    # The docstring may still name that old guard, so look for the statement only.
    assert "if cp_size != 1" not in patch_verl.SEQ_IDX_HELPER
    assert patch_verl.SEQ_IDX_HELPER.count("def _yeto_seq_idx_args(") == 1
    # Both call sites keep the same signature, so the helper stays the single place that
    # decides the pack length (tests/test_rl_verl_megatron.py pins the call strings).
    assert "_yeto_seq_idx_args(cp_size, seqlens_in_batch_padded_cpu)" in patch_verl.SEQ_IDX_PACKED_HOOK
    assert "_yeto_seq_idx_args(cp_size, seqlens_in_batch_padded_cpu)" in patch_verl.SEQ_IDX_ENGINE_HOOK
