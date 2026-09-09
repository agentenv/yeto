"""Reduce temporary CP gather storage without changing collective values.

Only the pinned NeMo implementation is patched. For the production B=1,
sequence-dimension gather, a single contiguous output replaces list gather,
its internal flatten buffer, and concatenation. The existing autograd backward
is retained unchanged. Dense reordering selects the local rows directly so a
small local input never keeps an entire global hidden-state allocation alive.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ORIGINAL_CP_SHA256 = '1a183963614b9d97ee5283a0d219c53955e7e688c1123b855c45d60de974958a'


def contiguous_forward(original_forward):
    def forward(ctx, local_tensor, group, dim):
        import torch
        import torch.distributed as dist
        dim = dim if dim >= 0 else local_tensor.ndim + dim
        if not (dim == 0 or (dim == 1 and local_tensor.shape[0] == 1)):
            return original_forward(ctx, local_tensor, group, dim)
        world = dist.get_world_size(group)
        shape = (world * local_tensor.shape[0], *local_tensor.shape[1:])
        gathered = torch.empty(shape, device=local_tensor.device, dtype=local_tensor.dtype)
        dist.all_gather_into_tensor(gathered, local_tensor.contiguous(), group=group)
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.dim = dim
        ctx.local_dim_size = local_tensor.size(dim)
        if dim == 1:
            return gathered.reshape(1, world * local_tensor.shape[1], *local_tensor.shape[2:])
        return gathered
    return forward


def local_dense_reorder(self, hidden_states, original_positions, cp_group):
    import torch
    import torch.distributed as dist
    rank, length = dist.get_rank(cp_group), hidden_states.shape[1]
    gathered = self._all_gather_concat(hidden_states, cp_group, dim=1, differentiable=True)
    positions = self._all_gather_concat(original_positions, cp_group, dim=0)
    order = torch.argsort(positions)
    sorted_positions = positions.index_select(0, order)
    expected = torch.arange(sorted_positions.numel(), device=positions.device, dtype=positions.dtype)
    if not torch.equal(sorted_positions, expected):
        raise RuntimeError('CP positions must cover the complete dense sequence')
    # Same rows and gradient as index_select(all rows)[:, start:end], with a
    # local-sized allocation rather than a view retaining global storage.
    local_order = order.narrow(0, rank * length, length)
    return gathered.index_select(1, local_order), sorted_positions


def install():
    from nemo_automodel.components.models.qwen3_5_moe import cp_linear_attn as cp
    if hashlib.sha256(Path(cp.__file__).read_bytes()).hexdigest() != ORIGINAL_CP_SHA256:
        raise RuntimeError('CP memory patch does not match the pinned runtime source')
    if getattr(cp, '_yeta_cp_memory_v1', False):
        return
    cp._yeta_original_gather_forward = cp._AllGatherConcatFn.forward
    cp._yeta_original_dense_reorder = cp.CPAwareGatedDeltaNet._undo_attention_load_balancing
    cp._AllGatherConcatFn.forward = staticmethod(contiguous_forward(cp._AllGatherConcatFn.forward))
    cp.CPAwareGatedDeltaNet._undo_attention_load_balancing = local_dense_reorder
    cp._yeta_cp_memory_v1 = True


def verify_cuda_parity():
    """Exercise the original and replacement on the actual eight-rank NCCL group."""
    import torch
    import torch.distributed as dist
    from nemo_automodel.components.models.qwen3_5_moe import cp_linear_attn as cp
    if dist.get_world_size() != 8 or dist.get_backend() != 'nccl':
        raise RuntimeError('CUDA parity requires the actual eight-rank NCCL group')
    rank, world = dist.get_rank(), dist.get_world_size()
    class Reference(torch.autograd.Function):
        forward = staticmethod(cp._yeta_original_gather_forward)
        backward = staticmethod(cp._AllGatherConcatFn.backward)
    class Fake:
        layer_idx = 0
        def __init__(self, function): self.function = function
        def _all_gather_concat(self, x, group, *, dim, differentiable=False):
            if differentiable: return self.function.apply(x, group, dim)
            chunks = [torch.empty_like(x) for _ in range(world)]
            dist.all_gather(chunks, x.contiguous(), group=group)
            return torch.cat(chunks, dim=dim)
    cases = 0
    for dtype in (torch.float32, torch.bfloat16):
        for shape, dim in (((1, 8, 5), 1), ((2, 8, 5), 1), ((4, 5), 0)):
            base = (torch.arange(__import__('math').prod(shape), device='cuda').reshape(shape) % 17).to(dtype)+rank
            xs = [base.clone().requires_grad_() for _ in range(2)]
            ys = [Reference.apply(xs[0], dist.group.WORLD, dim), cp._AllGatherConcatFn.apply(xs[1], dist.group.WORLD, dim)]
            if not torch.equal(*ys): raise RuntimeError('NCCL gather forward parity failed')
            for y in ys: y.backward(torch.full_like(y, rank+1))
            if not torch.equal(xs[0].grad, xs[1].grad): raise RuntimeError('NCCL gather backward parity failed')
            cases += 1
        positions = torch.cat((torch.arange(rank*4,(rank+1)*4,device='cuda'),
                               torch.arange((2*world-1-rank)*4,(2*world-rank)*4,device='cuda')))
        base = (positions[:,None]*5+torch.arange(5,device='cuda')[None,:]).unsqueeze(0).to(dtype)
        xs = [base.clone().requires_grad_() for _ in range(2)]
        old, old_pos = cp._yeta_original_dense_reorder(Fake(Reference),xs[0],positions,dist.group.WORLD)
        new, new_pos = local_dense_reorder(Fake(cp._AllGatherConcatFn),xs[1],positions,dist.group.WORLD)
        if not torch.equal(old,new) or not torch.equal(old_pos,new_pos): raise RuntimeError('NCCL reorder forward parity failed')
        for y in (old,new): y.backward(torch.full_like(y,rank+1))
        if not torch.equal(xs[0].grad,xs[1].grad): raise RuntimeError('NCCL reorder backward parity failed')
        cases += 1
    dist.barrier()
    return {'ranks': world, 'backend': 'nccl', 'exact_forward_backward_cases': cases}
