"""
Correctness tests for deluca_v3's low-rank pseudo-completion and CFS module.

The v3 forward/backward is a hand-written autograd Function wrapping v2's fused
CUDA kernel, so its gradients are checked against a pure-PyTorch reference.

Run:  py -3.10 test_deluca_v3.py
"""

import numpy as np
import torch

from deluca_v3 import (LowRankPseudoCompletionFn, LowRankPseudoCompletion,
                       EfficientCFSModule)
from DeLUCA import _load_pseudo_cuda


def reference_forward(x, A, B_mat, bias, prelu_w):
    """Pure-PyTorch equivalent of LowRankPseudoCompletionFn.forward -> (B, F)."""
    x_t = x.t().unsqueeze(2)
    tmp = torch.bmm(A, x_t)
    pre = torch.bmm(B_mat, tmp).squeeze(2) + bias
    out = torch.clamp(pre, min=0) + prelu_w * torch.clamp(pre, max=0)
    return out.t()


def test_forward_backward(B=64, F=32, r=5, device="cuda", tol=2e-4):
    torch.manual_seed(0)
    x = torch.randn(B, F, device=device)
    init = dict(device=device, dtype=torch.float32)
    A0 = torch.randn(F, r, B, **init) * 0.1
    B0 = torch.randn(F, B, r, **init) * 0.1
    bias0 = torch.randn(F, B, **init) * 0.1
    p0 = torch.full((F, 1), 0.25, **init)
    grad_seed = torch.randn(B, F, device=device)

    def run(fn):
        A = A0.clone().requires_grad_(True)
        Bm = B0.clone().requires_grad_(True)
        bias = bias0.clone().requires_grad_(True)
        p = p0.clone().requires_grad_(True)
        out = fn(x, A, Bm, bias, p)
        (out * grad_seed).sum().backward()
        return out, A.grad, Bm.grad, bias.grad, p.grad

    ref = run(reference_forward)
    got = run(LowRankPseudoCompletionFn.apply)

    names = ["output", "d_A", "d_B_mat", "d_bias", "d_prelu"]
    ok = True
    for name, a, b in zip(names, ref, got):
        err = (a - b).abs().max().item()
        scale = max(a.abs().max().item(), 1e-8)
        rel = err / scale
        passed = rel < tol
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<8} max|diff|={err:.3e}  rel={rel:.3e}")
    return ok


def test_module_matches_fallback(B=48, F=24, r=4, device="cuda", tol=2e-4):
    """The CUDA path and the portable fallback must agree."""
    torch.manual_seed(1)
    mod = LowRankPseudoCompletion((B, F), [B], r).to(device)
    x = torch.randn(B, F, device=device)

    with torch.no_grad():
        cuda_out = mod(x).clone()
        ext = _load_pseudo_cuda()
        import deluca_v3
        saved = deluca_v3._load_pseudo_cuda
        deluca_v3._load_pseudo_cuda = lambda: None   # force fallback branch
        try:
            fb_out = mod(x).clone()
        finally:
            deluca_v3._load_pseudo_cuda = saved

    err = (cuda_out - fb_out).abs().max().item()
    rel = err / max(fb_out.abs().max().item(), 1e-8)
    passed = rel < tol
    print(f"  {'PASS' if passed else 'FAIL'}  cuda-vs-fallback max|diff|={err:.3e} rel={rel:.3e}")
    return passed


def test_affinity_matches_dense(B=128, Fenc=32, rank=8, device="cuda", tol=1e-5):
    """Tiled C must equal the dense V V^T that v1/v2 build, for any tile size."""
    torch.manual_seed(2)
    Z = torch.randn(B, Fenc, device=device)
    cfs = EfficientCFSModule(rank).to(device)
    PZ, _ = cfs(Z)                 # forward returns no B-sized factor any more
    V = cfs.V_rank                 # reconstructed on demand

    dense = (V @ V.t()).cpu()                       # what v1/v2 materialize
    ok = True
    for tile in (B, 64, 17):
        tiled = cfs.affinity_tiled(tile=tile)
        err = (dense - tiled).abs().max().item()
        passed = err < tol
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  tile={tile:<4} max|diff| vs dense V V^T = {err:.3e}")

    # v1 stores Coef = P.T; P is symmetric so the two must coincide.
    sym = (dense - dense.t()).abs().max().item()
    passed = sym < tol
    ok &= passed
    print(f"  {'PASS' if passed else 'FAIL'}  C symmetric (C == C.T): max|diff|={sym:.3e}")

    # PZ must equal the dense projector applied to Z.
    err = (dense.to(device) @ Z - PZ).abs().max().item()
    rel = err / max(PZ.abs().max().item(), 1e-8)
    passed = rel < 1e-4
    ok &= passed
    print(f"  {'PASS' if passed else 'FAIL'}  PZ == (V V^T) Z: rel={rel:.3e}")
    return ok


def test_cfs_backends_agree(B=200, Fe=40, rank=25, device="cuda"):
    """gram, lowrank and the exact SVD must span the same subspace and give the same PZ."""
    torch.manual_seed(4)
    Z = (torch.randn(B, rank, device=device) @ torch.randn(rank, Fe, device=device)
         + 0.05 * torch.randn(B, Fe, device=device))

    Zd = Z.double()
    U, S, Vh = torch.linalg.svd(Zd.t(), full_matrices=False)
    V_ref = Vh.transpose(-2, -1)[:, :rank]
    PZ_ref = (V_ref @ (V_ref.t() @ Zd)).float()
    P_ref = (V_ref @ V_ref.t()).float()

    ok = True
    for backend, eigh_dev in (("gram", "cpu"), ("gram", "gpu"), ("lowrank", "cpu")):
        cfs = EfficientCFSModule(rank, backend=backend, eigh_device=eigh_dev).to(device)
        PZ, _ = cfs(Z)
        V = cfs.V_rank
        pz_err = (PZ - PZ_ref).abs().max().item() / PZ_ref.abs().max().item()
        # compare projectors, not bases: the basis is only defined up to rotation
        proj_err = ((V @ V.t()) - P_ref).abs().max().item()
        passed = pz_err < 1e-3 and proj_err < 1e-3
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {backend:<8} eigh={eigh_dev:<4} "
              f"rel PZ err={pz_err:.2e}  projector err={proj_err:.2e}")
    return ok


def test_gram_gradients(B=200, Fe=40, rank=25, device="cuda", tol=2e-3):
    """d PZ / d Z from the gram path must match the lowrank path (and be finite)."""
    torch.manual_seed(5)
    Z0 = (torch.randn(B, rank, device=device) @ torch.randn(rank, Fe, device=device)
          + 0.05 * torch.randn(B, Fe, device=device))
    g = torch.randn(B, Fe, device=device)

    grads = {}
    for backend in ("gram", "lowrank"):
        Z = Z0.clone().requires_grad_(True)
        cfs = EfficientCFSModule(rank, backend=backend).to(device)
        PZ, _ = cfs(Z)
        (PZ * g).sum().backward()
        grads[backend] = Z.grad.clone()

    finite = bool(torch.isfinite(grads["gram"]).all())
    err = (grads["gram"] - grads["lowrank"]).abs().max().item()
    rel = err / max(grads["lowrank"].abs().max().item(), 1e-8)
    passed = finite and rel < tol
    print(f"  {'PASS' if passed else 'FAIL'}  d PZ/d Z gram vs lowrank rel={rel:.2e}  "
          f"all finite={finite}")
    return passed


def test_encoder_nonan_equivalence(device="cuda"):
    """EncoderNoNaN must equal Encoder whenever the input has no NaN."""
    from DeLUCA import Encoder
    from deluca_v3 import EncoderNoNaN
    torch.manual_seed(6)
    ok = True
    for shape, enc_size, ks in (((8, 50), [40], None),
                                ((8, 32, 32, 1), [3, 3, 5], [3, 3, 3])):
        x = torch.randn(*shape, device=device)
        torch.manual_seed(7)
        a = Encoder(shape, enc_size, ks).to(device)
        torch.manual_seed(7)
        b = EncoderNoNaN(shape, enc_size, ks).to(device)
        with torch.no_grad():
            err = (a(x) - b(x)).abs().max().item()
        passed = err < 1e-6
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  shape {str(shape):<16} max|diff|={err:.2e}")
    return ok


def test_loss_variants(B=64, F=32, device="cuda"):
    """The PyTorch 'frobenius' path must equal the fused kernel it replaces,
    and every variant must be finite with finite gradients."""
    from deluca_v3 import masked_reconstruction_loss, RECON_LOSSES
    from DeLUCA import MaskedLossFn, _load_masked_loss_cuda

    torch.manual_seed(8)
    x = torch.randn(B, F, device=device) * 10
    x[torch.rand_like(x) < 0.3] = float("nan")          # 30% missing
    Xc0 = torch.randn(B, F, device=device)
    dec0 = torch.randn(B, F, device=device)

    ok = True
    if _load_masked_loss_cuda() is not None:
        Xc = Xc0.clone().requires_grad_(True)
        dec = dec0.clone().requires_grad_(True)
        MaskedLossFn.apply(x, Xc, dec).backward()
        gk = (Xc.grad.clone(), dec.grad.clone())
        ref = MaskedLossFn.apply(x, Xc0, dec0).item()

        Xc = Xc0.clone().requires_grad_(True)
        dec = dec0.clone().requires_grad_(True)
        lt = masked_reconstruction_loss(x, Xc, dec, "frobenius")
        lt.backward()
        gt = (Xc.grad.clone(), dec.grad.clone())

        lerr = abs(lt.item() - ref) / max(abs(ref), 1e-8)
        gerr = max((a - b).abs().max().item() / max(a.abs().max().item(), 1e-8)
                   for a, b in zip(gk, gt))
        passed = lerr < 1e-5 and gerr < 1e-4
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  frobenius torch vs fused kernel: "
              f"loss rel={lerr:.2e}  grad rel={gerr:.2e}")

    for kind in RECON_LOSSES:
        Xc = Xc0.clone().requires_grad_(True)
        dec = dec0.clone().requires_grad_(True)
        v = masked_reconstruction_loss(x, Xc, dec, kind)
        v.backward()
        good = (torch.isfinite(v).all() and torch.isfinite(Xc.grad).all()
                and torch.isfinite(dec.grad).all())
        ok &= bool(good)
        print(f"  {'PASS' if good else 'FAIL'}  {kind:<10} loss={v.item():>12.4f}  "
              f"max|grad|={Xc.grad.abs().max().item():.3e}")
    return ok


def test_no_b2_allocation(B=4096, F=16, r=4, device="cuda"):
    """A full fwd+bwd must never allocate anything of order B**2.

    Shape chosen so the two hypotheses are far apart: a dense (B,B) is 64 MB
    while the whole legitimate O(F*B*r) working set is ~1 MB per tensor. With
    B**2 comparable to F*B*r the test would pass vacuously.
    """
    torch.manual_seed(3)
    mod = LowRankPseudoCompletion((B, F), [B], r).to(device)
    x = torch.randn(B, F, device=device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = mod(x)
    out.sum().backward()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 1024 ** 2

    b2_mb = B * B * 4 / 1024 ** 2                     # one dense (B,B) float32
    budget = 6 * F * B * r * 4 / 1024 ** 2 + 4        # grads + workspace, O(F*B*r)
    assert budget < b2_mb / 4, "shape does not separate O(F*B*r) from O(B**2)"
    passed = peak < budget
    print(f"  {'PASS' if passed else 'FAIL'}  peak={peak:.1f}MB  "
          f"O(F*B*r) budget={budget:.1f}MB  one dense (B,B) would be {b2_mb:.1f}MB")
    return passed


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if _load_pseudo_cuda() is None:
        raise SystemExit("pseudo_completion_cuda extension not built")

    print("[1] low-rank fwd/bwd vs pure-PyTorch reference")
    r1 = test_forward_backward()
    print("[2] CUDA path vs portable fallback")
    r2 = test_module_matches_fallback()
    print("[3] tiled affinity vs dense V V^T")
    r3 = test_affinity_matches_dense()
    print("[4] no B**2 allocation in fwd+bwd")
    r4 = test_no_b2_allocation()
    print("[5] CFS backends agree with the exact SVD")
    r5 = test_cfs_backends_agree()
    print("[6] gram-path gradients match the lowrank path")
    r6 = test_gram_gradients()
    print("[7] EncoderNoNaN == Encoder on NaN-free input")
    r7 = test_encoder_nonan_equivalence()
    print("[8] reconstruction-loss variants")
    r8 = test_loss_variants()

    print("\nALL PASS" if all([r1, r2, r3, r4, r5, r6, r7, r8])
          else "\nFAILURES PRESENT")
