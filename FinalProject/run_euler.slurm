#!/usr/bin/env bash
# ============================================================================
#  DUC v2  —  Slurm batch script for the Euler cluster (UW–Madison)
# ============================================================================
#  Submit:   sbatch run_euler.slurm
#  Watch:    squeue -u $USER       /  tail -f slurm-<jobid>.out
#  Cancel:   scancel <jobid>
#
#  Builds all CUDA + OpenMP extensions, then runs the optimized benchmark
#  followed by the original baseline, and prints a comparison report.
#  Total expected wall-clock: ~3 minutes (build ~2 min, benchmark ~1 min).
# ============================================================================

#SBATCH --job-name=duc_v2
#SBATCH --partition=research          # adjust if a class partition is required
#SBATCH --gres=gpu:1                  # one GPU
#SBATCH --time=00:20:00               # 20 minutes wall-clock cap
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # OpenMP threads — Step 7 scales to ~16
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
echo "======================================================================="
echo "  DUC v2 — Job ${SLURM_JOB_ID} on $(hostname)"
echo "  Partition: ${SLURM_JOB_PARTITION}   GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "  Started:   $(date)"
echo "======================================================================="

# ─── Module environment ───────────────────────────────────────────────────
module purge
module load nvidia/cuda/12.1.0    || module load cuda/12.1
module load gcc/11.3.0            || module load gcc/11
module load python/3.10           || module load anaconda

# ─── OpenMP threading (Step 7 reduction) ──────────────────────────────────
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

# ─── Project paths ────────────────────────────────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
SRC_DIR="${PROJECT_ROOT}/src"
KERNEL_DIR="${SRC_DIR}/cuda_kernels"
DATA_DIR="${PROJECT_ROOT}/data"

mkdir -p "${DATA_DIR}/logs"

# ─── Python deps (user-local install if missing) ──────────────────────────
echo "--- Installing Python dependencies (--user) ---"
python -m pip install --quiet --user --upgrade pip
python -m pip install --quiet --user torch numpy scipy scikit-learn munkres tensorboard

# ─── Build all CUDA + OpenMP extensions ───────────────────────────────────
echo "--- Building CUDA + OpenMP extensions ---"
cd "${KERNEL_DIR}"
python setup.py install --user

# ─── Sanity tests (fast) ──────────────────────────────────────────────────
echo "--- Correctness checks ---"
python test_hybrid.py        || echo "WARN: hybrid test failed"
python test_cfs.py           || echo "WARN: cfs test failed"
python test_masked_loss.py   || echo "WARN: masked_loss test failed"
python test_data_prep.py     || echo "WARN: data_prep test failed"

# ─── Run optimized benchmark ──────────────────────────────────────────────
echo "--- Optimized benchmark (DUC v2) ---"
cd "${SRC_DIR}"
python benchmark.py --step "DUC v2 (Euler ${SLURM_JOB_ID})"

# ─── Run baseline for comparison ──────────────────────────────────────────
echo "--- Baseline benchmark (DUC v1) ---"
python benchmark.py --original --step "DUC v1 baseline (Euler ${SLURM_JOB_ID})"

# ─── Print comparison report ──────────────────────────────────────────────
echo "--- Comparison report ---"
python benchmark.py --report

echo "======================================================================="
echo "  Finished: $(date)"
echo "  Results saved to: ${DATA_DIR}/benchmark_results.json"
echo "======================================================================="
