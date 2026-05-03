"""
Scaling benchmark: CUDA kernel vs PyTorch batched ops at different problem sizes.
Measures forward+backward (full training iteration equivalent).
"""
import torch
import torch.nn as nn
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pseudo_completion_cuda
from DeLUCA import FusedPseudoCompletionFn


def bench_cuda(B, F, N=50):
    x = torch.randn(B, F, device="cuda")
    w = nn.Parameter(torch.randn(F, B, B, device="cuda") * 0.01)
    b = nn.Parameter(torch.zeros(F, B, device="cuda"))
    p = nn.Parameter(torch.full((F, 1), 0.25, device="cuda"))

    # warmup
    for _ in range(5):
        out = FusedPseudoCompletionFn.apply(x, w, b, p)
        out.sum().backward()
        w.grad = b.grad = p.grad = None
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N):
        out = FusedPseudoCompletionFn.apply(x, w, b, p)
        out.sum().backward()
        w.grad = b.grad = p.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N * 1000


def bench_pytorch(B, F, N=50):
    x = torch.randn(B, F, device="cuda")
    w = nn.Parameter(torch.randn(F, B, B, device="cuda") * 0.01)
    b = nn.Parameter(torch.zeros(F, B, device="cuda"))
    p = nn.Parameter(torch.full((F, 1), 0.25, device="cuda"))

    def fwd_bwd():
        x_t = x.t().unsqueeze(2)
        out = torch.bmm(w, x_t).squeeze(2) + b
        pos = torch.clamp(out, min=0)
        neg = p * torch.clamp(out, max=0)
        result = (pos + neg).t()
        result.sum().backward()
        w.grad = b.grad = p.grad = None

    for _ in range(5):
        fwd_bwd()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N):
        fwd_bwd()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N * 1000


if __name__ == "__main__":
    configs = [
        (500,  100),   # current: K=5, n=100
        (1000, 100),   # K=10, n=100
        (1000, 200),   # K=5, n=200
        (2000, 200),   # K=10, n=200
        (2000, 500),   # K=4, n=500
        (5000, 500),   # K=10, n=500
    ]

    print(f"{'B':>6} x {'F':>4} | {'CUDA (ms)':>10} | {'PyTorch (ms)':>12} | {'Speedup':>8} | Winner")
    print("-" * 65)

    for B, F in configs:
        # Check if weight tensor fits in VRAM: F*B*B*4 bytes
        mem_gb = F * B * B * 4 / 1e9
        if mem_gb > 6.0:
            print(f"{B:>6} x {F:>4} | {'SKIP':>10} | {'SKIP':>12} | {'':>8} | OOM risk ({mem_gb:.1f}GB)")
            continue

        try:
            cuda_ms = bench_cuda(B, F)
            pt_ms = bench_pytorch(B, F)
            speedup = pt_ms / cuda_ms
            winner = "CUDA" if speedup > 1.0 else "PyTorch"
            print(f"{B:>6} x {F:>4} | {cuda_ms:>10.2f} | {pt_ms:>12.2f} | {speedup:>7.2f}x | {winner}")
        except RuntimeError as e:
            print(f"{B:>6} x {F:>4} | ERROR: {e}")

        torch.cuda.empty_cache()
