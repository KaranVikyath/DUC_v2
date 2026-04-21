"""
Tests for CFS cuSOLVER SVD + fused projection CUDA extension.

Usage:
    cd cuda_kernels
    python test_cfs.py
"""

import sys
import time
import torch
import numpy as np

sys.path.insert(0, ".")

def test_forward_correctness():
    """Compare cuSOLVER CFS vs PyTorch svd_lowrank projection."""
    import cfs_solver_cuda as ext

    B, F_enc, rank = 500, 40, 25
    torch.manual_seed(42)
    Z = torch.randn(B, F_enc, device="cuda")

    # --- cuSOLVER path ---
    PZ_cuda, V_cuda, Coef_cuda = ext.forward(Z.contiguous(), rank)

    # --- PyTorch reference ---
    Zt = Z.t()
    U, S, V_ref = torch.svd_lowrank(Zt, q=rank)
    PZ_ref = (V_ref @ V_ref.t()) @ Z
    Coef_ref = V_ref @ V_ref.t()

    # Compare projections (sign-invariant — PZ = VV^T Z is unique)
    pz_diff = (PZ_cuda - PZ_ref).abs().max().item()
    coef_diff = (Coef_cuda - Coef_ref).abs().max().item()

    print(f"PZ max diff:   {pz_diff:.2e}")
    print(f"Coef max diff: {coef_diff:.2e}")

    # Tolerance higher due to different SVD algorithms (Jacobi vs randomized)
    ok = pz_diff < 0.1
    print(f"FORWARD CORRECTNESS: {'PASS' if ok else 'FAIL'}")
    if not ok:
        # Also check that projections are close in Frobenius norm sense
        rel_err = (PZ_cuda - PZ_ref).norm() / PZ_ref.norm()
        print(f"  Relative Frobenius error: {rel_err:.4e}")
    return ok


def test_projection_properties():
    """VV^T should be idempotent and symmetric."""
    import cfs_solver_cuda as ext

    B, F_enc, rank = 500, 40, 25
    Z = torch.randn(B, F_enc, device="cuda")

    _, _, Coef = ext.forward(Z.contiguous(), rank)

    # Idempotent: P^2 = P
    PP = Coef @ Coef
    idem_diff = (PP - Coef).abs().max().item()
    print(f"Idempotency max diff: {idem_diff:.2e}")

    # Symmetric: P = P^T
    sym_diff = (Coef - Coef.t()).abs().max().item()
    print(f"Symmetry max diff:    {sym_diff:.2e}")

    ok = idem_diff < 1e-3 and sym_diff < 1e-5
    print(f"PROJECTION PROPERTIES: {'PASS' if ok else 'FAIL'}")
    return ok


def test_autograd():
    """Verify gradients flow through CusolverCFSFn."""
    # Import autograd wrapper from DeLUCA
    sys.path.insert(0, "..")
    from DeLUCA import CusolverCFSFn

    B, F_enc, rank = 500, 40, 25
    Z = torch.randn(B, F_enc, device="cuda", requires_grad=True)

    PZ, Coef = CusolverCFSFn.apply(Z, rank)
    loss = PZ.sum()
    loss.backward()

    ok = Z.grad is not None and Z.grad.shape == (B, F_enc) and Z.grad.norm().item() > 0
    print(f"Z.grad shape: {Z.grad.shape if Z.grad is not None else 'None'}")
    print(f"Z.grad norm:  {Z.grad.norm().item():.4f}" if Z.grad is not None else "")
    print(f"AUTOGRAD: {'PASS' if ok else 'FAIL'}")
    return ok


def test_backward_correctness():
    """Compare backward gradient vs manual VV^T @ grad_PZ."""
    sys.path.insert(0, "..")
    from DeLUCA import CusolverCFSFn
    import cfs_solver_cuda as ext

    B, F_enc, rank = 500, 40, 25
    torch.manual_seed(42)
    Z = torch.randn(B, F_enc, device="cuda", requires_grad=True)

    PZ, Coef = CusolverCFSFn.apply(Z, rank)
    loss = PZ.sum()
    loss.backward()

    # Manual: d_Z = Coef @ grad_PZ where grad_PZ = ones
    grad_PZ = torch.ones_like(PZ)
    d_Z_manual = Coef @ grad_PZ

    diff = (Z.grad - d_Z_manual).abs().max().item()
    print(f"Backward max diff vs manual: {diff:.2e}")
    ok = diff < 1e-4
    print(f"BACKWARD CORRECTNESS: {'PASS' if ok else 'FAIL'}")
    return ok


def test_timing():
    """Compare cuSOLVER CFS vs torch.svd_lowrank timing."""
    import cfs_solver_cuda as ext

    B, F_enc, rank = 500, 40, 25
    Z = torch.randn(B, F_enc, device="cuda")
    N = 500

    # Warmup
    for _ in range(20):
        ext.forward(Z.contiguous(), rank)
    torch.cuda.synchronize()

    # cuSOLVER
    t0 = time.perf_counter()
    for _ in range(N):
        ext.forward(Z.contiguous(), rank)
    torch.cuda.synchronize()
    cuda_ms = (time.perf_counter() - t0) / N * 1000

    # PyTorch svd_lowrank + VV^T @ Z (old way)
    t0 = time.perf_counter()
    for _ in range(N):
        Zt = Z.t()
        U, S, V = torch.svd_lowrank(Zt, q=rank)
        P = V @ V.t()
        PZ = P @ Z
    torch.cuda.synchronize()
    pt_old_ms = (time.perf_counter() - t0) / N * 1000

    # PyTorch svd_lowrank + fused projection (new fallback)
    t0 = time.perf_counter()
    for _ in range(N):
        Zt = Z.t()
        U, S, V = torch.svd_lowrank(Zt, q=rank)
        VtZ = V.t() @ Z
        PZ = V @ VtZ
    torch.cuda.synchronize()
    pt_fused_ms = (time.perf_counter() - t0) / N * 1000

    print(f"cuSOLVER CFS:            {cuda_ms:.3f} ms")
    print(f"PyTorch old (VV^T @ Z):  {pt_old_ms:.3f} ms")
    print(f"PyTorch fused (V@V^TZ):  {pt_fused_ms:.3f} ms")
    print(f"Speedup vs old:          {pt_old_ms/cuda_ms:.2f}x")
    print(f"Speedup vs fused:        {pt_fused_ms/cuda_ms:.2f}x")


def test_various_sizes():
    """Test with different B, F_enc, rank combinations."""
    import cfs_solver_cuda as ext

    configs = [
        (100, 20, 5),
        (500, 40, 25),
        (1000, 80, 10),
        (200, 50, 15),
    ]

    all_ok = True
    for B, F_enc, rank in configs:
        Z = torch.randn(B, F_enc, device="cuda")
        try:
            PZ, V_rank, Coef = ext.forward(Z.contiguous(), rank)
            assert PZ.shape == (B, F_enc), f"PZ shape {PZ.shape}"
            assert V_rank.shape == (B, rank), f"V_rank shape {V_rank.shape}"
            assert Coef.shape == (B, B), f"Coef shape {Coef.shape}"
            print(f"  ({B:>4}, {F_enc:>3}, {rank:>2}): PASS")
        except Exception as e:
            print(f"  ({B:>4}, {F_enc:>3}, {rank:>2}): FAIL — {e}")
            all_ok = False

    print(f"VARIOUS SIZES: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("  CFS cuSOLVER SVD Extension Tests")
    print("=" * 60)

    print("\n--- Forward Correctness ---")
    test_forward_correctness()

    print("\n--- Projection Properties ---")
    test_projection_properties()

    print("\n--- Various Sizes ---")
    test_various_sizes()

    print("\n--- Autograd ---")
    test_autograd()

    print("\n--- Backward Correctness ---")
    test_backward_correctness()

    print("\n--- Timing ---")
    test_timing()

    print("\nDone.")
