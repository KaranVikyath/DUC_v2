"""
Tests for OpenMP data-prep extension (Step 7).

Usage:
    cd cuda_kernels
    python test_data_prep.py
"""

import sys, time
import numpy as np

import data_prep_omp as ext


def test_thread_count():
    n = ext.omp_thread_count()
    print(f"  OpenMP threads available: {n}")
    ok = n >= 1
    print(f"THREAD COUNT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_nan_insert_correctness():
    np.random.seed(42)
    N = 500 * 100
    data = np.random.randn(N).astype(np.float64)
    n_nan = int(0.3 * N)
    indices = np.random.choice(N, size=n_nan, replace=False).astype(np.int64)

    # Reference
    ref = data.copy()
    ref[indices] = np.nan

    # OpenMP version
    omp = data.copy()
    ext.parallel_nan_insert(omp, indices)

    nan_match = np.array_equal(np.isnan(ref), np.isnan(omp))
    val_match  = np.allclose(ref[~np.isnan(ref)], omp[~np.isnan(omp)])
    ok = nan_match and val_match
    print(f"  NaN positions match: {nan_match}")
    print(f"  Non-NaN values match: {val_match}")
    print(f"NAN INSERT CORRECTNESS: {'PASS' if ok else 'FAIL'}")
    return ok


def test_frob_correctness():
    np.random.seed(7)
    A = np.random.randn(500, 100).astype(np.float64)
    B = np.random.randn(500, 100).astype(np.float64)

    ref  = np.linalg.norm(A - B)
    omp  = ext.parallel_frob_diff_norm(
        np.ascontiguousarray(A), np.ascontiguousarray(B))

    rel = abs(omp - ref) / (ref + 1e-12)
    ok  = rel < 1e-10
    print(f"  numpy ref:  {ref:.6f}")
    print(f"  omp result: {omp:.6f}  rel err: {rel:.2e}")
    print(f"FROBENIUS CORRECTNESS: {'PASS' if ok else 'FAIL'}")
    return ok


def test_cast_f32_correctness():
    A = np.random.randn(500, 100)  # float64
    out = ext.parallel_cast_f32(np.ascontiguousarray(A))
    diff = np.abs(out.astype(np.float64) - A).max()
    ok = diff < 1e-6
    print(f"  Max abs diff f64->f32: {diff:.2e}")
    print(f"CAST F32 CORRECTNESS: {'PASS' if ok else 'FAIL'}")
    return ok


def test_timing():
    np.random.seed(0)
    N     = 500 * 100
    n_nan = int(0.3 * N)
    data  = np.random.randn(N).astype(np.float64)
    idx   = np.random.choice(N, size=n_nan, replace=False).astype(np.int64)
    A     = np.random.randn(500, 100)
    B     = np.random.randn(500, 100)
    REPS  = 2000

    # --- NaN insert ---
    t0 = time.perf_counter()
    for _ in range(REPS):
        d = data.copy(); d[idx] = np.nan
    seq_nan_ms = (time.perf_counter() - t0) / REPS * 1000

    t0 = time.perf_counter()
    for _ in range(REPS):
        d = data.copy(); ext.parallel_nan_insert(d, idx)
    omp_nan_ms = (time.perf_counter() - t0) / REPS * 1000

    print(f"  NaN insert  — seq: {seq_nan_ms:.3f} ms  omp: {omp_nan_ms:.3f} ms  "
          f"speedup: {seq_nan_ms/omp_nan_ms:.2f}x")

    # --- Frobenius norm ---
    Ac = np.ascontiguousarray(A)
    Bc = np.ascontiguousarray(B)

    t0 = time.perf_counter()
    for _ in range(REPS):
        np.linalg.norm(Ac - Bc)
    seq_frob_ms = (time.perf_counter() - t0) / REPS * 1000

    t0 = time.perf_counter()
    for _ in range(REPS):
        ext.parallel_frob_diff_norm(Ac, Bc)
    omp_frob_ms = (time.perf_counter() - t0) / REPS * 1000

    print(f"  Frob norm   — seq: {seq_frob_ms:.3f} ms  omp: {omp_frob_ms:.3f} ms  "
          f"speedup: {seq_frob_ms/omp_frob_ms:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("  OpenMP Data-Prep Extension Tests")
    print("=" * 60)

    print("\n--- Thread Count ---")
    test_thread_count()

    print("\n--- NaN Insert Correctness ---")
    test_nan_insert_correctness()

    print("\n--- Frobenius Norm Correctness ---")
    test_frob_correctness()

    print("\n--- Float32 Cast Correctness ---")
    test_cast_f32_correctness()

    print("\n--- Timing ---")
    test_timing()

    print("\nDone.")
