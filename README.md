<div align="center">

# DUC v2 — GPU-Accelerated Deep Union Completion

**High-throughput simultaneous matrix completion + subspace clustering**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue?style=flat-square)](LICENSE)

---

**9.9&times; faster per-iteration &nbsp;·&nbsp; 13.9&times; total wall-clock &nbsp;·&nbsp; +13.6 pp accuracy at 70% missing**

</div>

---

## Paper

> **Deep Union Completion for Subspace Clustering with Missing Data**
> Karan Vikyath Veeranna Rupashree, Siddharth Baskar, et al.
> *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-25)*, 2025.

This repository is the GPU-accelerated implementation of **DeLUCA** (Deep Union Completion Architecture) published at AAAI-25. The original PyTorch code bottlenecked on sequential Python loops, full SVD, and redundant tensor allocations. DUC v2 rewrites the computational pipeline using **CUDA kernels**, **cuBLAS**, **cuSOLVER**, **kernel fusion**, and **OpenMP** — achieving an order-of-magnitude speedup while simultaneously improving accuracy at high missing-data rates.

---

## Architecture

```
Input X (with NaN)
       │
       ▼
┌──────────────────────────────────────┐
│         PseudoCompletion             │  cuBLAS bmm + fused bias/PReLU/transpose
│                                      │  Python for-loop → single GEMM  (9× speedup)
└──────────────────┬───────────────────┘
                   │  Xc  (B × F)
                   ▼
┌──────────────────────────────────────┐
│            Encoder                   │  Conv2D / Linear + PReLU
└──────────────────┬───────────────────┘
                   │  Z  (B × Z_f)
                   ▼
┌──────────────────────────────────────┐
│           CFS Module                 │  cuSOLVER gesvdj + fused V@(VᵀZ)
│                                      │  avoids B×B projection matrix
└──────────────────┬───────────────────┘
                   │  PZ  (B × Z_f)
                   ▼
┌──────────────────────────────────────┐
│            Decoder                   │  ConvTranspose2D / Linear + PReLU
└──────────────────┬───────────────────┘
                   │  decoded  (B × F)
                   ▼
┌──────────────────────────────────────┐
│       Reconstruction Loss            │  Single-pass warp-shuffle reduction
│                                      │  NaN mask + 3 Frobenius norms, 0 allocs
└──────────────────────────────────────┘

OpenMP  →  parallel NaN insert (1.7×)  +  parallel ‖A−B‖_F (12.6×)
```

---

## What's New in v2

| # | Optimization | Key Technique | Speedup |
|---|-------------|---------------|---------|
| 1 | **Batched pseudo-completion** | `torch.bmm` — single cuBLAS call replaces Python loop | **9.0×** |
| 2+3 | **Randomized SVD + fused loss** | `torch.svd_lowrank`, tensor reuse | 6.6× |
| 4 | **CUDA fused kernel (naive)** | Shared memory, warp-shuffle, `__ldg` | 5.0× |
| 4b | **Hybrid cuBLAS+CUDA** | cuBLAS matmul + fused bias/PReLU custom kernel | 6.6× |
| 5 | **cuSOLVER CFS module** | `cusolverDnSgesvdj` + fused low-rank projection | 5.2× |
| 6 | **Fused masked-loss kernel** | Single-pass warp-shuffle + `atomicAdd` | 6.1× |
| **7** | **OpenMP data preprocessing** | `#pragma omp parallel for reduction` | **9.9×** |

---

## Results

### Per-Iteration Time (ms) — lower is better

| Method | 0% Missing | 30% Missing | 70% Missing | vs Baseline |
|--------|:----------:|:-----------:|:-----------:|:-----------:|
| Baseline (DUC v1) | 249.2 | 243.2 | 259.4 | 1.0× |
| Batched bmm | 27.6 | 26.8 | 27.1 | **9.0×** |
| rSVD + fused | 37.6 | 39.1 | 36.5 | 6.6× |
| CUDA naive kernel | 49.9 | 49.4 | 49.3 | 5.0× |
| Hybrid cuBLAS+CUDA | 37.8 | 36.5 | 35.6 | 6.6× |
| cuSOLVER CFS | 47.5 | 45.8 | 46.7 | 5.2× |
| Fused masked-loss | 40.6 | 40.4 | 39.7 | 6.1× |
| **DUC v2 (final)** | **26.7** | **24.9** | **25.2** | **9.9×** |

### Accuracy — higher is better

| Method | Compl % (0%) | Compl % (30%) | Compl % (70%) | Clust % (70%) |
|--------|:------------:|:-------------:|:-------------:|:-------------:|
| Baseline (DUC v1) | 100.0 | 99.3 | 71.4 | 93.4 |
| Batched bmm | 100.0 | 99.3 | 68.5 | 93.8 |
| rSVD + fused | 100.0 | 95.9 | 82.8 | 97.2 |
| Hybrid cuBLAS+CUDA | 100.0 | 95.6 | 82.6 | 96.8 |
| cuSOLVER CFS | 100.0 | 99.2 | 84.8 | **99.8** |
| Fused masked-loss | 100.0 | 99.1 | 82.9 | 99.2 |
| **DUC v2 (final)** | **100.0** | **99.3** | **85.0** | **99.2** |

### Total Training Time

| Method | Total Time (s) | Speedup |
|--------|:--------------:|:-------:|
| Baseline (DUC v1) | 580.98 | 1.0× |
| Batched bmm | 72.05 | 8.1× |
| rSVD + fused | 49.95 | 11.6× |
| Hybrid cuBLAS+CUDA | 49.25 | 11.8× |
| Fused masked-loss | 70.03 | 8.3× |
| **DUC v2 (final)** | **41.94** | **13.9×** |

> Benchmark config: synthetic data m=100, n=100, r=5, K=5, noise=0 → matrix (500×100).
> GPU: NVIDIA RTX sm_86. `torch.manual_seed(17)`, deterministic=True.

---

## Repository Layout

```
DUC_v2/
├── DeLUCA.py                   # Optimized model — all optimizations active
├── DeLUCA_original.py          # Unmodified DUC v1 baseline for comparison
├── custom_funcs.py             # Data utils, clustering, OpenMP loader
├── benchmark.py                # Benchmark harness (run & compare)
├── benchmark_results.json      # Saved results for all optimization stages
│
└── cuda_kernels/
    ├── pseudo_completion.cu    # Fused pseudo-completion (naive + hybrid)
    ├── cfs_solver.cu           # cuSOLVER SVD + fused low-rank projection
    ├── masked_loss.cu          # Fused masked-loss reduction kernel
    ├── data_prep.cpp           # OpenMP data preprocessing
    ├── setup.py                # Builds all 4 extensions in one command
    │
    ├── test_kernel.py          # Correctness + timing: naive CUDA kernels
    ├── test_hybrid.py          # Correctness + timing: hybrid cuBLAS+CUDA
    ├── test_cfs.py             # Correctness + timing: cuSOLVER SVD
    ├── test_masked_loss.py     # Correctness + timing: fused loss kernel
    └── test_data_prep.py       # Correctness + timing: OpenMP ops
```

---

## Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| PyTorch | 2.x with CUDA |
| CUDA Toolkit | 12.x |
| Compiler | MSVC 2019+ (Windows) / GCC 11+ (Linux) |
| OpenMP | MSVC `/openmp` or GCC `-fopenmp` |

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/KaranVikyath/DUC_v2.git
cd DUC_v2

# 2. Install Python dependencies
pip install torch numpy scipy scikit-learn munkres tensorboard

# 3. Build all CUDA + OpenMP extensions (one command)
cd cuda_kernels && python setup.py install && cd ..
```

> **Windows note:** CUDA 12.1 + VS2026 flags (`--allow-unsupported-compiler`,
> `-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`) are pre-configured in `setup.py`.

---

## Usage

### Run Benchmark

```bash
# Run optimized pipeline and save results
python benchmark.py --step "DUC v2"

# Run original baseline for comparison
python benchmark.py --original --step "DUC v1 baseline"

# View comparison table across all saved steps
python benchmark.py --report
```

### Run Tests

```bash
python cuda_kernels/test_hybrid.py       # hybrid cuBLAS+CUDA kernels
python cuda_kernels/test_cfs.py          # cuSOLVER SVD
python cuda_kernels/test_masked_loss.py  # fused loss kernel
python cuda_kernels/test_data_prep.py    # OpenMP ops
```

### Quick-Start in Python

```python
import torch
from DeLUCA import DeLUCA
from custom_funcs import generate_data, missing_data_generation

# Generate synthetic data
data, full_data, input_shape, batch_size, total_datapoints, \
    flat_layer_size, enc_layer_size, deco_layer_size, \
    kernel_size, output_padding, K, reg1, reg2, \
    alpha1, alpha2, d, lr, rank, true_labels = generate_data(
        m=100, n=100, r=5, k=5, noise=0)

# Simulate missing data (30% entries missing)
data_missing = missing_data_generation(data, int(0.3 * total_datapoints))

device = torch.device("cuda")

model = DeLUCA(
    input_shape, flat_layer_size, enc_layer_size, deco_layer_size,
    kernel_size, output_padding, lr=lr, K=K, rank=rank,
    batch_size=batch_size, cluster_model="CFS", device=device
).to(device)

# Training loop — all CUDA extensions auto-detected at runtime
for step in range(1000):
    C, loss, complete_data, current_lr = model.finetune_fit(data_missing)
    if step % 100 == 0:
        print(f"step {step:4d}  loss={loss:.4f}  lr={current_lr:.2e}")
```

> All CUDA extensions are **auto-detected at import time** with graceful CPU fallback
> if extensions are not built or unavailable.

---

## Implementation Details

| Technique | File | Description |
|-----------|------|-------------|
| cuBLAS batched GEMM | `DeLUCA.py`, `cfs_solver.cu` | `torch.bmm`, `torch::mm` |
| Shared memory caching | `pseudo_completion.cu` | B-float x[:,f] column cache |
| Warp-shuffle reduction | `pseudo_completion.cu`, `masked_loss.cu` | Two-level cross-warp reduction |
| `__ldg` read-only cache | All `.cu` files | Weight/bias/input reads |
| Kernel fusion | `pseudo_completion.cu`, `masked_loss.cu` | Eliminate intermediate allocations |
| `atomicAdd` | `masked_loss.cu` | 3 global scalar accumulators |
| cuSOLVER `cusolverDnSgesvdj` | `cfs_solver.cu` | Jacobi SVD, economy mode |
| OpenMP `parallel for reduction` | `data_prep.cpp` | Frobenius norm — 12.6× speedup |
| `torch.autograd.Function` | `DeLUCA.py` | 4 custom wrappers with CPU fallback |

---

## Citation

If you use this code, please cite the original DUC paper:

```bibtex
@inproceedings{veerannarupashree2025duc,
  title     = {Deep Union Completion for Subspace Clustering with Missing Data},
  author    = {Veeranna Rupashree, Karan Vikyath and Baskar, Siddharth and others},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025}
}
```

---

## License

[BSD 3-Clause](LICENSE) — released as open-source for unfettered use by any interested party.

---

<div align="center">

University of Wisconsin–Madison

</div>
