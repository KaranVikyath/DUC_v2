"""
Tests for fused masked-loss CUDA extension.

Usage:
    cd cuda_kernels
    python test_masked_loss.py
"""

import sys, time, math
import torch

sys.path.insert(0, "..")
import masked_loss_cuda as ext
from DeLUCA import MaskedLossFn


def pytorch_ref(x, Xc, decoded):
    """PyTorch reference — the original 7-op computation."""
    nan_mask = torch.isnan(x)
    x_omega = torch.where(nan_mask, torch.zeros_like(x), x)
    mask_tensor = torch.where(nan_mask, torch.zeros_like(x), torch.ones_like(x))
    Xc_m  = Xc * mask_tensor
    dec_m = decoded * mask_tensor
    d1 = Xc_m - x_omega
    d2 = Xc_m - dec_m
    d3 = dec_m - x_omega
    return (torch.norm(d1, p='fro') + torch.norm(d2, p='fro') + torch.norm(d3, p='fro'))


def make_inputs(B, F, missing_pct=0.3, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(B, F, device="cuda")
    if missing_pct > 0:
        n_nan = int(missing_pct * B * F)
        idx = torch.randperm(B * F)[:n_nan]
        x.view(-1)[idx] = float("nan")
    Xc      = torch.randn(B, F, device="cuda")
    decoded = torch.randn(B, F, device="cuda")
    return x, Xc, decoded


# ── Forward correctness ───────────────────────────────────────
def test_forward_correctness():
    ok = True
    for missing_pct, tag in [(0.0, "0% miss"), (0.3, "30% miss"), (0.7, "70% miss")]:
        x, Xc, decoded = make_inputs(500, 100, missing_pct)
        loss_ref  = pytorch_ref(x, Xc, decoded).item()
        out = ext.forward(x.float().contiguous(),
                          Xc.float().contiguous(),
                          decoded.float().contiguous())
        loss_cuda = out[0].item()
        diff = abs(loss_cuda - loss_ref)
        rel  = diff / (abs(loss_ref) + 1e-8)
        flag = "PASS" if rel < 1e-4 else "FAIL"
        if flag == "FAIL": ok = False
        print(f"  {tag}: ref={loss_ref:.4f}  cuda={loss_cuda:.4f}  rel={rel:.2e}  {flag}")
    print(f"FORWARD CORRECTNESS: {'PASS' if ok else 'FAIL'}")
    return ok


# ── Backward correctness via autograd ────────────────────────
def test_backward_correctness():
    B, F = 200, 50
    x, Xc_base, dec_base = make_inputs(B, F, 0.3)

    # --- Reference: PyTorch autograd ---
    Xc_ref  = Xc_base.float().clone().requires_grad_(True)
    dec_ref = dec_base.float().clone().requires_grad_(True)
    loss_ref = pytorch_ref(x, Xc_ref, dec_ref)
    loss_ref.backward()
    dXc_ref  = Xc_ref.grad.clone()
    ddec_ref = dec_ref.grad.clone()

    # --- Custom kernel backward ---
    Xc_c  = Xc_base.float().clone().requires_grad_(True)
    dec_c = dec_base.float().clone().requires_grad_(True)
    loss_c = MaskedLossFn.apply(x, Xc_c, dec_c)
    loss_c.backward()

    diff_Xc  = (Xc_c.grad - dXc_ref).abs().max().item()
    diff_dec = (dec_c.grad - ddec_ref).abs().max().item()
    ok = diff_Xc < 1e-4 and diff_dec < 1e-4
    print(f"  d_Xc max diff:      {diff_Xc:.2e}")
    print(f"  d_decoded max diff: {diff_dec:.2e}")
    print(f"BACKWARD CORRECTNESS: {'PASS' if ok else 'FAIL'}")
    return ok


# ── Gradient flow through autograd.Function ──────────────────
def test_gradient_flow():
    B, F = 500, 100
    x, Xc_b, dec_b = make_inputs(B, F, 0.3)
    Xc  = Xc_b.requires_grad_(True)
    dec = dec_b.requires_grad_(True)
    loss = MaskedLossFn.apply(x, Xc, dec)
    loss.backward()
    ok = (Xc.grad is not None and Xc.grad.shape == (B, F) and Xc.grad.norm() > 0 and
          dec.grad is not None and dec.grad.shape == (B, F) and dec.grad.norm() > 0)
    print(f"  Xc.grad norm:  {Xc.grad.norm().item():.4f}")
    print(f"  dec.grad norm: {dec.grad.norm().item():.4f}")
    print(f"GRADIENT FLOW: {'PASS' if ok else 'FAIL'}")
    return ok


# ── Edge cases ────────────────────────────────────────────────
def test_edge_cases():
    ok = True
    B, F = 500, 100

    # All NaN → loss should be 0
    x_all_nan = torch.full((B, F), float("nan"), device="cuda")
    Xc  = torch.randn(B, F, device="cuda")
    dec = torch.randn(B, F, device="cuda")
    out = ext.forward(x_all_nan, Xc.float().contiguous(), dec.float().contiguous())
    loss = out[0].item()
    flag = "PASS" if abs(loss) < 1e-6 else "FAIL"
    if flag == "FAIL": ok = False
    print(f"  All-NaN loss = {loss:.2e}  {flag}")

    # No NaN (0% missing)
    x_clean = torch.randn(B, F, device="cuda")
    loss_ref  = pytorch_ref(x_clean, Xc, dec).item()
    out = ext.forward(x_clean.float().contiguous(), Xc.float().contiguous(), dec.float().contiguous())
    loss_cuda = out[0].item()
    rel = abs(loss_cuda - loss_ref) / (abs(loss_ref) + 1e-8)
    flag = "PASS" if rel < 1e-4 else "FAIL"
    if flag == "FAIL": ok = False
    print(f"  No-NaN rel err = {rel:.2e}  {flag}")

    print(f"EDGE CASES: {'PASS' if ok else 'FAIL'}")
    return ok


# ── Timing ───────────────────────────────────────────────────
def test_timing():
    B, F, N_iter = 500, 100, 1000
    x, Xc, dec = make_inputs(B, F, 0.3)
    xf  = x.float().contiguous()
    xcf = Xc.float().contiguous()
    df  = dec.float().contiguous()

    # Warmup
    for _ in range(20):
        ext.forward(xf, xcf, df)
    torch.cuda.synchronize()

    # Fused CUDA
    t0 = time.perf_counter()
    for _ in range(N_iter):
        ext.forward(xf, xcf, df)
    torch.cuda.synchronize()
    cuda_ms = (time.perf_counter() - t0) / N_iter * 1000

    # PyTorch reference (7 ops)
    t0 = time.perf_counter()
    for _ in range(N_iter):
        pytorch_ref(xf, xcf, df)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / N_iter * 1000

    print(f"  Fused CUDA:  {cuda_ms:.3f} ms")
    print(f"  PyTorch ref: {pt_ms:.3f} ms")
    print(f"  Speedup:     {pt_ms/cuda_ms:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("  Fused Masked-Loss Kernel Tests")
    print("=" * 60)

    print("\n--- Forward Correctness ---")
    test_forward_correctness()

    print("\n--- Backward Correctness ---")
    test_backward_correctness()

    print("\n--- Gradient Flow ---")
    test_gradient_flow()

    print("\n--- Edge Cases ---")
    test_edge_cases()

    print("\n--- Timing ---")
    test_timing()

    print("\nDone.")
