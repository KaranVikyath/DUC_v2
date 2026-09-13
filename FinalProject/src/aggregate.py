"""
Aggregate result shards into one table. Standard library only, so it runs on
the CHTC login node where scipy/torch are not installed.

    python src/aggregate.py "results/shards/*.json"
    python src/aggregate.py "results/shards/*.json" --csv out.csv

Held-out metrics only: NMAE (per-feature scaled), compl* (held-out completion
in original units), clustering. Groups by (dataset, method, missing rate),
reports mean +- std over seeds, and flags any group whose seeds ran on more
than one GPU model — those time/VRAM numbers are not comparable.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--csv", help="also write the table as CSV")
    args = ap.parse_args()

    shards, bad = load(args.pattern)
    if not shards:
        print(f"no result shards matched {args.pattern}")
        return
    devices = sorted({str(s.get("device", "?")) for s in shards})
    print(f"  {len(shards)} shards" + (f", {bad} unreadable" if bad else ""))
    print(f"  devices: {', '.join(devices)}"
          + ("" if len(devices) == 1 else
             "   <-- MIXED: time/VRAM are not comparable across rows on different GPUs"))

    groups = defaultdict(list)
    for s in shards:
        groups[(s["dataset"], s["tag"], float(s["missing_pct"]))].append(s)

    hdr = (f"  {'dataset':<10} | {'method':<12} | {'Miss':>5} | {'n':>2} | "
           f"{'time(s)':>8} | {'peak MB':>8} | {'NMAE':>16} | "
           f"{'compl*%':>15} | {'Clust%':>15} | gpu")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows_csv = []
    last_ds = None
    for key in sorted(groups, key=lambda k: (k[0], k[2], k[1])):
        g = groups[key]
        if last_ds not in (None, key[0]):
            print("  " + "-" * (len(hdr) - 2))
        last_ds = key[0]
        n_m, n_s, n = _stats([x.get("nmae_unobs") for x in g])
        c_m, c_s, _ = _stats([x.get("completion_unobs_raw") for x in g])
        k_m, k_s, _ = _stats([x.get("cluster_acc") for x in g])
        t_m, _, _ = _stats([x.get("total_time_s") for x in g])
        p_m, _, _ = _stats([x.get("peak_train_mb") for x in g])
        devs = {str(x.get("device", "?")) for x in g}
        dev = "MIX!" if len(devs) > 1 else str(devices.index(next(iter(devs))))
        print(f"  {key[0]:<10} | {key[1]:<12} | {key[2]*100:>4.0f}% | {len(g):>2} | "
              f"{t_m:>8.1f} | {p_m:>8.1f} | {n_m:>7.4f} +-{n_s:>7.4f} | "
              f"{c_m:>7.2f} +-{c_s:>5.2f} | {k_m:>7.2f} +-{k_s:>5.2f} | {dev}")
        rows_csv.append([key[0], key[1], key[2], len(g), t_m, p_m,
                         n_m, n_s, c_m, c_s, k_m, k_s, ";".join(sorted(devs))])

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "method", "missing", "n", "time_s", "peak_mb",
                        "nmae", "nmae_std", "compl", "compl_std", "clust",
                        "clust_std", "devices"])
            w.writerows(rows_csv)
        print(f"\n  wrote {args.csv}")


if __name__ == "__main__":
    main()
