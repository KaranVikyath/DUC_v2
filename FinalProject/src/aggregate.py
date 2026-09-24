"""
Aggregate result shards into one table. Standard library only, so it runs on
the CHTC login node where scipy/torch are not installed.

    python src/aggregate.py "results/shards/*.json"
    python src/aggregate.py "results/shards/*.json" --csv out.csv

Held-out metrics only: NMAE (per-feature scaled), compl* (held-out completion
in original units), clustering. Groups by (dataset, method, missing rate, GPU
model) and reports mean +- std over seeds. Splitting by GPU keeps a v3 paired
with v1 on an H200 out of the L40S row, so every row's time/VRAM is
same-hardware. `ok` counts seeds with a finite NMAE — a diverged run is NaN and
drops out of the mean, so `ok` below `n` means some seeds failed.
"""

import argparse
import glob
import json
import statistics
from collections import defaultdict


def _stats(vals):
    v = [x for x in vals if x is not None and x == x]        # drop None/NaN
    if not v:
        return float("nan"), float("nan"), 0
    return (statistics.fmean(v), statistics.pstdev(v) if len(v) > 1 else 0.0, len(v))


def load(pattern):
    shards, failed = [], 0
    for p in glob.glob(pattern):
        try:
            d = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            failed += 1
            continue
        if isinstance(d, dict) and {"dataset", "tag", "missing_pct"} <= d.keys():
            shards.append(d)
    return shards, failed


def report(pattern, csv_path=None):
    shards, bad = load(pattern)
    if not shards:
        print(f"no result shards matched {pattern}")
        return
    devices = sorted({str(s.get("device", "?")) for s in shards})
    print(f"  {len(shards)} shards" + (f", {bad} unreadable" if bad else ""))
    print("  gpu: " + ", ".join(f"{i}={d}" for i, d in enumerate(devices)))

    groups = defaultdict(list)
    for s in shards:
        groups[(s["dataset"], s["tag"], float(s["missing_pct"]),
                str(s.get("device", "?")))].append(s)

    hdr = (f"  {'dataset':<10} | {'method':<20} | {'Miss':>5} | {'ok/n':>5} | "
           f"{'time(s)':>8} | {'peak MB':>8} | {'NMAE':>16} | "
           f"{'compl*%':>15} | {'Clust%':>15} | gpu")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows_csv = []
    last_ds = None
    for key in sorted(groups, key=lambda k: (k[0], k[2], k[1], k[3])):
        g = groups[key]
        if last_ds not in (None, key[0]):
            print("  " + "-" * (len(hdr) - 2))
        last_ds = key[0]
        n_m, n_s, n_ok = _stats([x.get("nmae_unobs") for x in g])
        c_m, c_s, _ = _stats([x.get("completion_unobs_raw") for x in g])
        k_m, k_s, _ = _stats([x.get("cluster_acc") for x in g])
        t_m, _, _ = _stats([x.get("total_time_s") for x in g])
        p_m, _, _ = _stats([x.get("peak_train_mb") for x in g])
        dev = devices.index(key[3])
        ok = f"{n_ok}/{len(g)}"
        print(f"  {key[0]:<10} | {key[1]:<20} | {key[2]*100:>4.0f}% | {ok:>5} | "
              f"{t_m:>8.1f} | {p_m:>8.1f} | {n_m:>7.4f} +-{n_s:>7.4f} | "
              f"{c_m:>7.2f} +-{c_s:>5.2f} | {k_m:>7.2f} +-{k_s:>5.2f} | {dev}")
        rows_csv.append([key[0], key[1], key[2], len(g), n_ok, t_m, p_m,
                         n_m, n_s, c_m, c_s, k_m, k_s, key[3]])

    if csv_path:
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "method", "missing", "n", "n_ok", "time_s",
                        "peak_mb", "nmae", "nmae_std", "compl", "compl_std",
                        "clust", "clust_std", "device"])
            w.writerows(rows_csv)
        print(f"\n  wrote {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--csv", help="also write the table as CSV")
    args = ap.parse_args()
    report(args.pattern, args.csv)


if __name__ == "__main__":
    main()
