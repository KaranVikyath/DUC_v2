"""Quick test: hybrid kernel correctness + forward-only timing."""
import torch
import time
import pseudo_completion_cuda as ext

print("hybrid_forward:", hasattr(ext, "hybrid_forward"))
print("hybrid_backward:", hasattr(ext, "hybrid_backward"))

B, F = 500, 100
torch.manual_seed(42)
x = torch.randn(B, F, device="cuda")
w = torch.randn(F, B, B, device="cuda") * 0.01
b = torch.zeros(F, B, device="cuda")
p = torch.full((F, 1), 0.25, device="cuda")

# Reference: PyTorch
x_t = x.t().unsqueeze(2)
bmm_out = torch.bmm(w, x_t).squeeze(2)
z = bmm_out + b
pos = torch.clamp(z, min=0)
neg = p * torch.clamp(z, max=0)
ref = (pos + neg).t()

# Hybrid: cuBLAS bmm + CUDA kernel
out, pre_act = ext.hybrid_forward(bmm_out, b, p)
diff = (out - ref).abs().max().item()
print(f"hybrid fwd max diff: {diff:.2e}")
print("FWD:", "PASS" if diff < 1e-5 else "FAIL")

# Backward test
grad = torch.randn(B, F, device="cuda")
dz, dp = ext.hybrid_backward(grad, p, pre_act)

# Reference backward
grad_t = grad.t()
pw = p.view(-1, 1)
mask = (pre_act > 0).float()
dz_ref = grad_t * (mask + pw * (1.0 - mask))
dp_ref = (grad_t * pre_act * (1.0 - mask)).sum(dim=1, keepdim=True)
dz_diff = (dz - dz_ref).abs().max().item()
dp_diff = (dp - dp_ref).abs().max().item()
print(f"dz max diff:    {dz_diff:.2e}")
print(f"d_prelu diff:   {dp_diff:.2e}")
print("BWD:", "PASS" if dz_diff < 1e-5 and dp_diff < 1e-2 else "FAIL")

# Autograd integration via DeLUCA
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from DeLUCA import HybridPseudoCompletionFn
import torch.nn as nn

x2 = torch.randn(B, F, device="cuda")
w2 = nn.Parameter(torch.randn(F, B, B, device="cuda") * 0.01)
b2 = nn.Parameter(torch.zeros(F, B, device="cuda"))
p2 = nn.Parameter(torch.full((F, 1), 0.25, device="cuda"))

out2 = HybridPseudoCompletionFn.apply(x2, w2, b2, p2)
out2.sum().backward()
print(f"w.grad norm: {w2.grad.norm().item():.4f}")
print(f"b.grad norm: {b2.grad.norm().item():.4f}")
print(f"p.grad norm: {p2.grad.norm().item():.4f}")
ok = w2.grad is not None and b2.grad is not None and p2.grad is not None
print("AUTOGRAD:", "PASS" if ok else "FAIL")

# Timing: forward only
N = 200
for _ in range(10):
    x_t = x.t().unsqueeze(2)
    torch.bmm(w, x_t).squeeze(2)
torch.cuda.synchronize()

# PyTorch full fwd
t0 = time.perf_counter()
for _ in range(N):
    xt = x.t().unsqueeze(2)
    o = torch.bmm(w, xt).squeeze(2)
    z = o + b
    pos = torch.clamp(z, min=0)
    neg = p * torch.clamp(z, max=0)
    _ = (pos + neg).t()
torch.cuda.synchronize()
pt_ms = (time.perf_counter() - t0) / N * 1000

# Hybrid fwd
t0 = time.perf_counter()
for _ in range(N):
    xt = x.t().unsqueeze(2)
    bmm_out = torch.bmm(w, xt).squeeze(2)
    out, _ = ext.hybrid_forward(bmm_out, b, p)
torch.cuda.synchronize()
hy_ms = (time.perf_counter() - t0) / N * 1000

print(f"\nPyTorch full fwd: {pt_ms:.3f} ms")
print(f"Hybrid fwd:       {hy_ms:.3f} ms")
print(f"Speedup:          {pt_ms/hy_ms:.2f}x")
