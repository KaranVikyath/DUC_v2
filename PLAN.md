# DeLUCA GPU Acceleration Plan (ME759 Final Project)

## Context
ME759 course project: accelerate DeLUCA (AAAI-25 published) using CUDA/OpenMP. Current PyTorch impl has bottlenecks in pseudo-completion loop, SVD, loss computation. Goal = incremental speedups on synthetic data (m=100, n=100, r=5, K=5), committed to https://github.com/KaranVikyath/DUC_v2.git regularly.

## Workspace Setup
1. Clone `https://github.com/KaranVikyath/DUC_v2.git` → `e:\ACADEMIA\Research\duc_v2\`
2. Copy from LLR-DeLUCA: `DeLUCA.py`, `custom_funcs.py`, `dataset_params.py`
3. Save this plan file into `duc_v2/PLAN.md`
4. All new work happens in `duc_v2/` — NO modifications to LLR-DeLUCA

## Synthetic Benchmark Setup
- `generate_data(m=100, n=100, r=5, K=5, noise=0)` → matrix shape (500, 100), 5 classes of 100 samples each
- Missing data: 0%, 25%, 50%, 80% NaN
- Run on both CPU + GPU, measure wall-clock per iteration + total training time
- Baseline: current DeLUCA.py unmodified

## Incremental Optimization Steps

### Step 0: Baseline Benchmark Script
- Create `benchmark.py` — runs current DeLUCA on synthetic data, records per-iteration time, total time, peak memory
- This is the "before" measurement. Commit to repo.
- **Files**: new `benchmark.py`

### Step 1: Pure PyTorch Optimizations (no CUDA kernels yet)
**Target: PseudoCompletion layer** — biggest bottleneck (Python loop over n=100 features)
- Replace sequential `for i in range(feature_size)` loop with batched matmul
- Stack all feature linear transforms into single `nn.Linear(batch_size, batch_size * feature_size)` or use `torch.bmm`
- Fuse PReLU into batched op
- **Also**: remove `all_params` CPU extraction in `finetune_fit` line 106 (dead weight)
- **Files**: modify [DeLUCA.py](DeLUCA.py) (PseudoCompletion class + finetune_fit)
- **Expected speedup**: 2-5x on pseudo-completion, ~1.5-2x overall

### Step 2: SVD Optimization in CFSModule
- Replace `torch.linalg.svd(full_matrices=False)` with `torch.svd_lowrank(q=rank)` — randomized SVD, O(mnr) vs O(mn*min(m,n))
- For m=100, n=500, r=5 → massive reduction
- Remove numpy fallback (not needed with lowrank)
- **Files**: modify [DeLUCA.py](DeLUCA.py) (CFSModule class)
- **Expected speedup**: 3-10x on CFS step

### Step 3: Loss Computation Optimization
- Current: 3 separate Frobenius norms + mask creation
- Fuse into single pass: compute masked diff + norm in one op
- Avoid creating intermediate tensors (`x_tilde_omega`, `x_omega_hat`)
- **Files**: modify [DeLUCA.py](DeLUCA.py) (forward method)
- **Expected speedup**: minor (~10-15% on loss step)

### Step 4: Custom CUDA Kernel — Fused Pseudo-Completion
- Write C++/CUDA extension via `torch.utils.cpp_extension`
- Single kernel: batch all n feature transforms + PReLU activation
- Launch n thread blocks, each handling one feature's matmul
- **Files**: new `cuda_kernels/pseudo_completion.cu`, `cuda_kernels/setup.py`
- **ME759 concepts**: thread/block config, shared memory, kernel fusion

### Step 5: cuSOLVER SVD + cuBLAS GEMM
- Replace PyTorch SVD with direct cuSOLVER call for truncated SVD
- Use cuBLAS for VV^T and PZ matrix products
- **Files**: new `cuda_kernels/cfs_module.cu`
- **ME759 concepts**: cuSOLVER/cuBLAS library usage, device memory management

### Step 6: Fused Loss Kernel
- Single CUDA kernel: masked diff + parallel reduction (sum-of-squares)
- Avoid 3 separate norm calls
- **Files**: new `cuda_kernels/loss_kernel.cu`
- **ME759 concepts**: parallel reduction (HW06)

### Step 7: OpenMP Data Preprocessing
- Parallelize NaN masking + batch assembly on host side
- Use `#pragma omp parallel for` across features
- **Files**: new `cuda_kernels/data_loader.cpp`
- **ME759 concepts**: OpenMP scheduling (HW07/08)

## Git Commit Schedule
Each step = 1 commit. Target: push every 2-3 days.
```
Commit 1: Step 0 — baseline benchmark
Commit 2: Step 1 — batched pseudo-completion
Commit 3: Step 2 — randomized SVD
Commit 4: Step 3 — fused loss
Commit 5: Step 4 — CUDA pseudo-completion kernel
Commit 6: Step 5 — cuSOLVER/cuBLAS CFS
Commit 7: Step 6 — CUDA loss kernel
Commit 8: Step 7 — OpenMP data loader
Commit 9: Final benchmarks + report
```

## Verification
- After each step: run `benchmark.py`, compare time vs baseline
- Check clustering accuracy + completion error within 1% of original
- Produce timing bar chart at end (per-iteration + total, across steps)

## Key Files to Modify
- [DeLUCA.py](DeLUCA.py) — main model (Steps 1-3)
- [custom_funcs.py](custom_funcs.py) — `generate_data()` needs update for m=100,n=100,r=5,K=5
- New: `benchmark.py` — timing harness
- New: `cuda_kernels/` — CUDA/C++ extensions (Steps 4-7)
