"""
Benchmark script for DeLUCA performance profiling.
Runs synthetic data (m=100, n=100, r=5, K=5) at multiple missing-data levels.
Measures per-iteration time, total training time, and reports accuracy.

Usage:
    python benchmark.py --step "Step 0: Baseline"          # run & save results
    python benchmark.py --original --step "Step 0: Baseline"  # run original
    python benchmark.py --report                            # show all saved results
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import warnings
import sys

warnings.filterwarnings("ignore", category=UserWarning)

# Results file lives in FinalProject/data/ (one level up from src/)
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(_HERE, "..", "data", "benchmark_results.json")

# --- Config ---
CFG_M, CFG_N, CFG_R, CFG_K, CFG_NOISE = 100, 100, 5, 5, 0
MISSING_PERCENTS = [0.0, 0.30, 0.70]
STOPPING_FACTOR = 10  # stop when lr < initial_lr / STOPPING_FACTOR


def run_benchmark(DeLUCA_cls, generate_data_fn, missing_data_generation_fn,
                  thrC_fn, post_proC_fn, err_rate_fn, convert_nan_fn,
                  missing_pct, device):
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(17)

    data, full_data, input_shape, batch_size, total_datapoints, \
        flat_layer_size, enc_layer_size, deco_layer_size, kernel_size, \
        output_padding, K, reg1, reg2, alpha1, alpha2, d, lr, rank, \
        true_labels = generate_data_fn(m=CFG_M, n=CFG_N, r=CFG_R, k=CFG_K, noise=CFG_NOISE)

    if missing_pct > 0:
        missing_data_input = missing_data_generation_fn(
            data, int(missing_pct * total_datapoints))
    else:
        missing_data_input = data.copy()

    model = DeLUCA_cls(
        input_shape, flat_layer_size, enc_layer_size, deco_layer_size,
        kernel_size, output_padding, lr, K, rank,
        reg_const1=reg1, reg_const2=reg2,
        batch_size=batch_size, model_path=None,
        logs_path=os.path.join(_HERE, "..", "data", "logs",
                               f"benchmark_{int(missing_pct*100)}pct"),
        cluster_model="CFS", device=device)
    model.to(device)

    stopping_lr = lr / STOPPING_FACTOR
    data_norm = np.linalg.norm(full_data)

    iter_times = []
    iteration = 0

    total_start = time.perf_counter()
    while True:
        iter_start = time.perf_counter()
        C, cost, complete_data, current_lr = model.finetune_fit(missing_data_input)
        iter_end = time.perf_counter()

        iter_times.append(iter_end - iter_start)
        iteration += 1

        if current_lr < stopping_lr:
            break
    total_end = time.perf_counter()

    # Completion accuracy — Step 7: parallel Frobenius norm via OpenMP
    try:
        from custom_funcs import _load_data_prep_omp
        _omp = _load_data_prep_omp()
    except Exception:
        _omp = None

    def _frob_diff(a, b):
        if _omp is not None:
            return _omp.parallel_frob_diff_norm(
                np.ascontiguousarray(a, dtype=np.float64),
                np.ascontiguousarray(b, dtype=np.float64))
        return np.linalg.norm(a - b)

    accuracy = 1 - _frob_diff(complete_data, full_data) / data_norm
    complete_data[~np.isnan(missing_data_input)] = missing_data_input[~np.isnan(missing_data_input)]
    mod_acc = 1 - _frob_diff(complete_data, full_data) / data_norm

    # Clustering accuracy
    cluster_acc = 0.0
    if true_labels is not None:
        C_thr = thrC_fn(C, alpha1)
        y_x, _ = post_proC_fn(C_thr, K, d, alpha2)
        cluster_acc = 1 - err_rate_fn(true_labels, y_x)

    total_time = total_end - total_start
    avg_iter = np.mean(iter_times)
    std_iter = np.std(iter_times)

    return {
        "missing_pct": missing_pct,
        "iterations": iteration,
        "total_time_s": total_time,
        "avg_iter_ms": avg_iter * 1000,
        "std_iter_ms": std_iter * 1000,
        "completion_acc": mod_acc * 100,
        "cluster_acc": cluster_acc * 100,
    }


def load_history():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return []


def save_history(history):
    with open(RESULTS_FILE, "w") as f:
        json.dump(history, f, indent=2)


def save_run(step_name, device_name, results):
    history = load_history()
    entry = {
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        "device": device_name,
        "config": {"m": CFG_M, "n": CFG_N, "r": CFG_R, "K": CFG_K, "noise": CFG_NOISE},
        "results": results,
    }
    history.append(entry)
    save_history(history)
    print(f"\nResults saved to {RESULTS_FILE} under step: \"{step_name}\"")


def print_report():
    history = load_history()
    if not history:
        print("No saved results yet. Run a benchmark with --step first.")
        return

    pcts = sorted({r["missing_pct"] for entry in history for r in entry["results"]})

    # Per missing-pct comparison
    for pct in pcts:
        print(f"\n{'='*80}")
        print(f"  Missing {pct*100:.0f}%")
        print(f"{'='*80}")
        header = f"  {'Step':<35} | {'Total(s)':>9} | {'Avg/iter(ms)':>13} | {'Compl%':>7} | {'Clust%':>7} | {'Speedup':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        baseline_time = None
        for entry in history:
            r = next((r for r in entry["results"] if r["missing_pct"] == pct), None)
            if r is None:
                continue
            if baseline_time is None:
                baseline_time = r["total_time_s"]
            speedup = baseline_time / r["total_time_s"] if r["total_time_s"] > 0 else float("inf")
            print(f"  {entry['step']:<35} | {r['total_time_s']:>9.3f} | "
                  f"{r['avg_iter_ms']:>13.3f} | {r['completion_acc']:>6.2f}% | "
                  f"{r['cluster_acc']:>6.2f}% | {speedup:>6.2f}x")

    # Summary: total time across all missing pcts
    print(f"\n{'='*80}")
    print(f"  Total Time (sum across all missing %)")
    print(f"{'='*80}")
    baseline_total = None
    for entry in history:
        total = sum(r["total_time_s"] for r in entry["results"])
        if baseline_total is None:
            baseline_total = total
        speedup = baseline_total / total if total > 0 else float("inf")
        print(f"  {entry['step']:<35} | {total:>9.3f}s | speedup: {speedup:.2f}x")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", action="store_true",
                        help="Use DeLUCA_original.py instead of DeLUCA.py")
    parser.add_argument("--step", type=str, default=None,
                        help="Name for this optimization step (e.g. 'Step 1: Batched PseudoCompletion')")
    parser.add_argument("--report", action="store_true",
                        help="Print comparison report across all saved steps")
    args = parser.parse_args()

    if args.report:
        print_report()
        return

    # Import correct module
    if args.original:
        from DeLUCA_original import DeLUCA
        label = "ORIGINAL"
    else:
        from DeLUCA import DeLUCA
        label = "OPTIMIZED"

    from custom_funcs import (generate_data, missing_data_generation,
                              thrC, post_proC, err_rate, convert_nan)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step_display = f" | Step: {args.step}" if args.step else ""
    print(f"=== DeLUCA Benchmark ({label}{step_display}) ===")
    print(f"Device: {device}")
    print(f"Synthetic data: m={CFG_M}, n={CFG_N}, r={CFG_R}, K={CFG_K}, noise={CFG_NOISE}")
    print(f"Matrix shape: ({CFG_K*CFG_N}, {CFG_M}) = ({CFG_K*CFG_N} samples, {CFG_M} features)")
    print()

    header = f"{'Miss%':>6} | {'Iters':>6} | {'Total(s)':>9} | {'Avg/iter(ms)':>13} | {'Std(ms)':>8} | {'Compl%':>7} | {'Clust%':>7}"
    print(header)
    print("-" * len(header))

    results = []
    for pct in MISSING_PERCENTS:
        r = run_benchmark(
            DeLUCA, generate_data, missing_data_generation,
            thrC, post_proC, err_rate, convert_nan,
            pct, device)
        results.append(r)
        print(f"{r['missing_pct']*100:>5.0f}% | {r['iterations']:>6d} | "
              f"{r['total_time_s']:>9.3f} | {r['avg_iter_ms']:>13.3f} | "
              f"{r['std_iter_ms']:>8.3f} | {r['completion_acc']:>6.2f}% | "
              f"{r['cluster_acc']:>6.2f}%")

    print()
    print(f"Total benchmark time: {sum(r['total_time_s'] for r in results):.2f}s")

    if args.step:
        save_run(args.step, str(device), results)


if __name__ == "__main__":
    main()
