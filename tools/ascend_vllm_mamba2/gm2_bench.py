"""Micro-benchmark of the yeto torch Mamba2 ops on one NPU (TP4 shard shapes of Nano)."""
import time, torch, torch_npu  # noqa: F401
from yeto.rl import ascend_vllm_mamba2 as m2

def t(fn, n=20):
    fn(); torch.npu.synchronize(); a = time.time()
    for _ in range(n): fn()
    torch.npu.synchronize(); return (time.time() - a) / n * 1e3

def main():
    d = "npu"; H, P, N, G, b, S = 16, 64, 128, 2, 256, 300
    st = torch.zeros(S, H, P, N, device=d); x = torch.randn(b, H, P, device=d, dtype=torch.bfloat16)
    dt = torch.randn(b, H, device=d, dtype=torch.bfloat16)[:, :, None].expand(-1, -1, P)
    A = -torch.rand(H, device=d)[:, None, None].expand(-1, P, N); B = torch.randn(b, G, N, device=d, dtype=torch.bfloat16)
    C = torch.randn_like(B); D = torch.randn(H, device=d)[:, None].expand(-1, P); bias = torch.randn(H, device=d)[:, None].expand(-1, P)
    idx = torch.arange(b, device=d, dtype=torch.int32); out = torch.empty_like(x)
    print("ssd_decode ms", t(lambda: m2.ssd_decode(st, x, dt, A, B, C, D, dt_bias=bias, dt_softplus=True, state_batch_indices=idx, out=out)), flush=True)
    dim = H * P + 2 * G * N; cs = torch.zeros(S, 3, dim, device=d, dtype=torch.bfloat16).transpose(-1, -2)
    xc = torch.randn(b, dim, device=d, dtype=torch.bfloat16); w = torch.randn(dim, 4, device=d, dtype=torch.bfloat16)
    print("conv_decode ms", t(lambda: m2.conv_decode(xc, cs, w, None, "silu", conv_state_indices=idx)), flush=True)
    T = 2048; xp = torch.randn(T, H, P, device=d, dtype=torch.bfloat16); dtp = torch.randn(T, H, device=d, dtype=torch.bfloat16)
    Bp = torch.randn(T, G, N, device=d, dtype=torch.bfloat16); Cp = torch.randn_like(Bp); op = torch.empty_like(xp)
    cu = torch.tensor([0, 512, 1024, 2048], device=d, dtype=torch.int32)
    print("ssd_prefill 3 seqs 2048 tok ms", t(lambda: m2.ssd_prefill(xp, dtp, -torch.rand(H, device=d), Bp, Cp, 128, cu, out=op, D=torch.randn(H, device=d), dt_bias=torch.randn(H, device=d), dt_softplus=True), 5), flush=True)

if __name__ == "__main__":
    main()
