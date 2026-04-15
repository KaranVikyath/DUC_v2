"""
Correctness test for the fused pseudo-completion CUDA kernels.
Tests both forward and backward against PyTorch reference implementations.

Run after building the extension:
    cd cuda_kernels && python setup.py install
    python test_kernel.py
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def reference_forward(x, weight, bias, prelu_weight):
    """PyTorch reference: batched matmul + bias + PReLU."""
    x_clean = torch.nan_to_num(x, nan=0.0).float()
    x_t = x_clean.t().unsqueeze(2)                         # (F, B, 1)
    out = torch.bmm(weight, x_t).squeeze(2)                # (F, B)
    z = out + bias                                          # pre-activation
    pos = torch.clamp(z, min=0)
    neg = prelu_weight * torch.clamp(z, max=0)
    return (pos + neg).t(), z                               # (B, F), (F, B)


def reference_backward(grad_output, x_clean, prelu_weight, pre_act):
    """PyTorch reference backward."""
    grad_t = grad_output.t()                                # (F, B)
    prelu_w = prelu_weight.view(-1, 1)                      # (F, 1)
    mask = (pre_act > 0).float()                            # (F, B)
    dz = grad_t * (mask + prelu_w * (1.0 - mask))           # (F, B)

    d_prelu = (grad_t * pre_act * (1.0 - mask)).sum(dim=1, keepdim=True)  # (F, 1)
    d_bias = dz                                             # (F, B)
    x_col = x_clean.t()                                     # (F, B)
    d_weight = dz.unsqueeze(2) * x_col.unsqueeze(1)         # (F, B, B)
    return d_weight, d_bias, d_prelu


def load_extension():
    try:
        import pseudo_completion_cuda
        return pseudo_completion_cuda
    except ImportError:
        pass
    try:
        from torch.utils.cpp_extension import load
        return load(
            name="pseudo_completion_cuda",
            sources=[os.path.join(os.path.dirname(__file__), "pseudo_completion.cu")],
            verbose=True,
        )
    except Exception as e:
        print(f"Cannot load CUDA extension: {e}")
        print("Build first: cd cuda_kernels && python setup.py install")
        return None


def test_forward(ext):
    print("=== Forward Test ===")
    torch.manual_seed(42)
    B, F = 500, 100

    x = torch.randn(B, F, device="cuda")
    x[0, 0] = float("nan")
    x[10, 50] = float("nan")

    weight = torch.randn(F, B, B, device="cuda") * 0.01
    bias = torch.zeros(F, B, device="cuda")
    prelu_weight = torch.full((F, 1), 0.25, device="cuda")

    x_clean = torch.nan_to_num(x, nan=0.0).float().contiguous()

    out_cuda, pre_act = ext.forward(
        x_clean, weight.contiguous(), bias.contiguous(),
        prelu_weight.contiguous())

    out_ref, pre_act_ref = reference_forward(x, weight, bias, prelu_weight)

    max_diff = (out_cuda - out_ref).abs().max().item()
    mean_diff = (out_cuda - out_ref).abs().mean().item()
    pa_diff = (pre_act - pre_act_ref).abs().max().item()

    print(f"  Output  — max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e}")
    print(f"  PreAct  — max diff: {pa_diff:.2e}")
    ok = max_diff < 1e-3
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok, x_clean, weight, bias, prelu_weight, pre_act


def test_backward(ext, x_clean, weight, bias, prelu_weight, pre_act):
    print("\n=== Backward Test ===")
    if not hasattr(ext, 'backward'):
        print("  SKIP — no backward() in extension")
        return True

    B, F = x_clean.shape
    grad_output = torch.randn(B, F, device="cuda")

    # CUDA backward
    dw_cuda, db_cuda, dp_cuda = ext.backward(
        grad_output.contiguous(), x_clean.contiguous(),
        prelu_weight.contiguous(), pre_act.contiguous())

    # Reference backward
    dw_ref, db_ref, dp_ref = reference_backward(
        grad_output, x_clean, prelu_weight, pre_act)

    dw_diff = (dw_cuda - dw_ref).abs().max().item()
    db_diff = (db_cuda - db_ref).abs().max().item()
    dp_diff = (dp_cuda - dp_ref).abs().max().item()

    print(f"  d_weight — max diff: {dw_diff:.2e}")
    print(f"  d_bias   — max diff: {db_diff:.2e}")
    print(f"  d_prelu  — max diff: {dp_diff:.2e}")

    ok = dw_diff < 1e-2 and db_diff < 1e-3 and dp_diff < 1e-2
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_autograd(ext):
    print("\n=== Autograd Integration Test ===")
    from DeLUCA import FusedPseudoCompletionFn

    B, F = 500, 100
    torch.manual_seed(42)
    x_clean = torch.randn(B, F, device="cuda")
    w = nn.Parameter(torch.randn(F, B, B, device="cuda") * 0.01)
    b = nn.Parameter(torch.zeros(F, B, device="cuda"))
    p = nn.Parameter(torch.full((F, 1), 0.25, device="cuda"))

    out = FusedPseudoCompletionFn.apply(x_clean, w, b, p)
    loss = out.sum()
    loss.backward()

    print(f"  weight.grad norm: {w.grad.norm().item():.4f}")
    print(f"  bias.grad norm:   {b.grad.norm().item():.4f}")
    print(f"  prelu.grad norm:  {p.grad.norm().item():.4f}")
    ok = w.grad is not None and b.grad is not None and p.grad is not None
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_timing(ext):
    print("\n=== Timing Comparison ===")
    B, F = 500, 100
    torch.manual_seed(0)
    x = torch.randn(B, F, device="cuda")
    w = torch.randn(F, B, B, device="cuda") * 0.01
    b = torch.zeros(F, B, device="cuda")
    p = torch.full((F, 1), 0.25, device="cuda")

    # Warmup
    for _ in range(10):
        ext.forward(x, w, b, p)
    torch.cuda.synchronize()

    import time
    N = 100

    # CUDA kernel
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        ext.forward(x, w, b, p)
    torch.cuda.synchronize()
    cuda_ms = (time.perf_counter() - t0) / N * 1000

    # PyTorch batched
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        xt = x.t().unsqueeze(2)
        out = torch.bmm(w, xt).squeeze(2) + b
        pos = torch.clamp(out, min=0)
        neg = p * torch.clamp(out, max=0)
        _ = pos + neg
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - t0) / N * 1000

    speedup = torch_ms / cuda_ms if cuda_ms > 0 else float('inf')
    print(f"  CUDA kernel:   {cuda_ms:.3f} ms/call")
    print(f"  PyTorch batch: {torch_ms:.3f} ms/call")
    print(f"  Speedup:       {speedup:.2f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — skipping")
        sys.exit(0)

    ext = load_extension()
    if ext is None:
        sys.exit(1)

    ok1, x_clean, w, b, p, pa = test_forward(ext)
    ok2 = test_backward(ext, x_clean, w, b, p, pa)
    ok3 = test_autograd(ext)
    test_timing(ext)

    print("\n" + "=" * 40)
    all_ok = ok1 and ok2 and ok3
    print(f"Overall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)
