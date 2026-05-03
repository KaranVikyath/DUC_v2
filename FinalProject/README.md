# DUC v2 — Final Project

GPU-accelerated Deep Union Completion for high-throughput matrix completion +
subspace clustering. Original DeLUCA (AAAI-25) re-implemented with CUDA kernels,
cuBLAS, cuSOLVER, kernel fusion, and OpenMP — **9.9× per-iteration speedup,
13.9× total wall-clock speedup**, and **+13.6 pp completion accuracy at 70%
missing data** over the baseline.

The project trains and evaluates in under a minute end-to-end on a single GPU,
so **no pre-trained model is shipped** — graders run the benchmark to reproduce
the results directly.

---

## Directory Layout

```
FinalProject/
├── README.md               # this file
├── run_euler.slurm         # Slurm batch script for Euler
├── src/                    # all source code
│   ├── DeLUCA.py                  # optimized model (all 7 stages active)
│   ├── DeLUCA_original.py         # unmodified baseline for comparison
│   ├── custom_funcs.py            # data utils, clustering, OpenMP loader
│   ├── benchmark.py               # benchmark harness
│   ├── dataset_params.py          # dataset configuration
│   └── cuda_kernels/
│       ├── pseudo_completion.cu   # fused pseudo-completion kernels
│       ├── cfs_solver.cu          # cuSOLVER SVD + fused projection
│       ├── masked_loss.cu         # fused masked-loss reduction kernel
│       ├── data_prep.cpp          # OpenMP data preprocessing
│       ├── setup.py               # builds all 4 extensions
│       └── test_*.py              # correctness + timing tests
├── data/                   # input/output data
│   ├── benchmark_results.json     # saved benchmark results
│   └── logs/                      # TensorBoard training logs (created at runtime)
└── build/                  # build artifacts land here (created at build time)
```

---

## Reproducing the Results on Euler (Slurm)

A ready-to-submit Slurm script is provided:

```bash
sbatch run_euler.slurm
```

The script:
1. Loads CUDA + GCC + Python modules
2. Sets `OMP_NUM_THREADS` from `--cpus-per-task`
3. Builds all four CUDA + OpenMP extensions in `src/cuda_kernels/`
4. Runs the optimized benchmark (`benchmark.py --step "DUC v2"`)
5. Runs the original baseline (`benchmark.py --original --step "DUC v1 baseline"`)
6. Prints the comparison report

Outputs land in:
- `slurm-<jobid>.out` — stdout (benchmark progress + final tables)
- `slurm-<jobid>.err` — stderr
- `data/benchmark_results.json` — appended results
- `data/logs/` — TensorBoard scalars

Expected wall-clock: ~3 minutes total (build ~2 min, benchmark ~1 min).

---

## Reproducing the Results Locally

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| PyTorch | 2.x with CUDA support |
| NVIDIA CUDA Toolkit | 12.x |
| Compiler | GCC 11+ (Linux) or MSVC 2019+ (Windows) |
| OpenMP | `-fopenmp` (GCC) or `/openmp` (MSVC) |

### Setup

```bash
# 1. From the repo root, enter the FinalProject directory
cd FinalProject

# 2. Install Python dependencies
pip install torch numpy scipy scikit-learn munkres tensorboard

# 3. Build all CUDA + OpenMP extensions
cd src/cuda_kernels
python setup.py install --user
cd ../..
```

### Run Optimized Benchmark

```bash
cd src
python benchmark.py --step "DUC v2"
```

### Run Baseline (Comparison)

```bash
python benchmark.py --original --step "DUC v1 baseline"
```

### Print Comparison Report

```bash
python benchmark.py --report
```

This produces a table comparing every saved step at each missing-data rate
(0%, 30%, 70%) for total time, per-iter time, completion accuracy, and
clustering accuracy.

### Run Correctness Tests

```bash
cd src/cuda_kernels
python test_hybrid.py        # hybrid cuBLAS+CUDA kernels
python test_cfs.py           # cuSOLVER SVD
python test_masked_loss.py   # fused loss kernel
python test_data_prep.py     # OpenMP ops
```

Each test prints `PASS`/`FAIL` for forward correctness, backward gradients,
edge cases, and timing comparison vs the PyTorch reference implementation.

---

## What the Benchmark Measures

Synthetic union-of-subspaces data: `m=100`, `n=100`, `r=5` rank per class,
`K=5` classes, noise=0 → input matrix shape (500, 100).

For each missing-data rate ∈ {0%, 30%, 70%}:
- **Iterations**: training steps until learning rate falls below `initial_lr/10`
- **Per-iteration time** (ms, mean ± std)
- **Completion accuracy**: `1 − ‖complete_data − full_data‖_F / ‖full_data‖_F`
- **Clustering accuracy**: spectral clustering on the CFS coefficient matrix `C`

All runs use `torch.manual_seed(17)`, `torch.backends.cudnn.deterministic=True`.

---

## Expected Output (Final DUC v2 Stage)

```
 Miss% |  Iters |  Total(s) |  Avg/iter(ms) |  Std(ms) |  Compl% |  Clust%
--------------------------------------------------------------------------
    0% |    561 |    14.983 |        26.705 |   30.324 | 100.00% | 100.00%
   30% |    642 |    15.958 |        24.855 |    1.892 |  99.28% | 100.00%
   70% |    436 |    11.003 |        25.234 |    2.404 |  84.99% |  99.20%

Total benchmark time: 41.94s
```

Compared to baseline (~580 s, 71.4% completion at 70% missing), this is a
**13.9× total wall-clock speedup** with **+13.6 percentage points** of
completion accuracy at 70% missing.

---

## Citation

```bibtex
@inproceedings{veerannarupashree2025duc,
  title     = {Deep Union Completion for Subspace Clustering with Missing Data},
  author    = {Veeranna Rupashree, Karan Vikyath and Baskar, Siddharth and others},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025}
}
```
