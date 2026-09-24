"""
Paper comparison tables: DUC v3 vs v1, classical baselines and the SOTA
completion / clustering-with-completion methods. Standard library only.

    python src/compare.py ROOT [ROOT ...] [--legacy ROOT ...] [--md out.md] [--csv out.csv]

Each ROOT is searched recursively for result shards (*.json with dataset/tag/
missing_pct). A --legacy root is the Sep 13-22 CHTC bundle: its real-data
v1/v3 shards predate the 2026-09-23 fixes (eigh NaN, all-seed-17, overwritten
pairs) and are dropped; its baseline and synthetic shards are valid and kept.

Rows are keyed (dataset, method, config, missing, seed). `config` separates
runs that differ in something other than the seed:
  - v3 at the published stop rule (lr0/10, 900 iters, "pair") vs the tuned
    rule (lr0/100, 3000 iters, "best"), and the GPU it ran on;
  - rank-dependent methods (v3, svd_impute, soft_impute) on COIL100/DSDD/VED/
    VED10k under the original rank vs the rank-validity rule
    (notes/rank_validity_rule.md).
Duplicate keys (the same shard downloaded twice) keep the first copy; if two
copies disagree on NMAE the key is reported as a conflict.

Metrics: NMAE on held-out entries (lower is better) and clustering accuracy
(%) from the shared spectral pipeline on each completed matrix. Methods that
cluster on their own also report `cluster_acc_own`, shown separately.
"""

import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict

DATASETS = ["synthetic", "ORL", "COIL20", "EYaleB", "COIL100", "Flowers",
            "OxfordPet", "HARUS", "DSDD", "VED10k", "VED"]
RATES = [0.1, 0.3, 0.5, 0.7, 0.9]         # the grid every method shares

# Ranks set by the rank-validity rule; anything else on these datasets is the
# original (invalid) config.
RULE_RANK = {"COIL100": 205, "DSDD": 8, "VED": 2, "VED10k": 2}
RANK_METHODS = {"svd_impute", "soft_impute"}

CLASSICAL = ["mean", "knn", "mice", "svd_impute", "soft_impute"]
SOTA_COMPLETION = ["sota_diffputer", "sota_cacti", "sota_miri", "sota_newimp",
                   "sota_aemc_ne", "sota_refidiff", "sota_kfmc", "sota_cagi"]
SOTA_CLUSTERING = ["sota_kfsc", "sota_altpzf", "sota_kcsc"]
V3 = ["v3", "v3-identity", "v3-tanh"]
METHODS = CLASSICAL + SOTA_COMPLETION + SOTA_CLUSTERING + V3

SHORT = {"mean": "Mean", "knn": "KNN", "mice": "MICE", "svd_impute": "SVD",
         "soft_impute": "Soft", "sota_diffputer": "DiffPuter",
         "sota_cacti": "CACTI", "sota_miri": "MIRI", "sota_newimp": "NewImp",
         "sota_aemc_ne": "AEMC-NE", "sota_refidiff": "RefiDiff",
         "sota_kfmc": "KFMC", "sota_cagi": "CAGI", "sota_kfsc": "k-FSC",
         "sota_altpzf": "AltPZF", "sota_kcsc": "KCSC", "v3": "v3-lin",
         "v3-identity": "v3-id", "v3-tanh": "v3-tanh", "v1": "v1"}

V3_TAGS = {"v3(r_p=1)": "v3", "v3-cola-identity(r_p=1)": "v3-identity",
           "v3-cola-tanh(r_p=1)": "v3-tanh"}


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _stats(vals):
    v = [x for x in vals if _finite(x)]
    if not v:
        return None
    return statistics.fmean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0), len(v)


def classify(s, legacy=False):
    """(method, config) for a shard, or None to drop it."""
    tag, ds = s["tag"], s["dataset"]
    if s.get("version") in ("v1", "v3"):
        if legacy and ds != "synthetic":
            return None                                   # pre-fix real-data run
        if s["version"] == "v1":
            return "v1", "pair@" + str(s.get("device"))
        method = V3_TAGS.get(tag)
        if method is None:
            return None
        sf = s.get("stop_factor", 10)      # shards before the flag used lr0/10
        if float(sf) == 10.0:
            return method, "pair@" + str(s.get("device"))
        cfg = "best"
        if ds in RULE_RANK:
            cfg += "/rule" if s.get("rank") == RULE_RANK[ds] else "/orig"
        return method, cfg
    if tag in RANK_METHODS and ds in RULE_RANK:
        return tag, "rule" if s.get("rank") == RULE_RANK[ds] else "orig"
    return tag, ""


def load(roots, legacy=()):
    rows, conflicts, dropped = {}, [], 0
    for root in list(roots) + list(legacy):
        for p in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
            try:
                s = json.load(open(p))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            if not (isinstance(s, dict) and {"dataset", "tag", "missing_pct"} <= s.keys()):
                continue
            mc = classify(s, legacy=root in legacy)
            if mc is None:
                dropped += 1
                continue
            key = (s["dataset"], mc[0], mc[1], round(float(s["missing_pct"]), 2),
                   int(s.get("seed", -1)))
            if key in rows:
                a, b = rows[key].get("nmae_unobs"), s.get("nmae_unobs")
                if _finite(a) and _finite(b) and abs(a - b) > 1e-6:
                    conflicts.append((key, p, a, b))
                continue
            s["_path"] = p
            rows[key] = s
    return rows, conflicts, dropped


def group(rows):
    g = defaultdict(list)
    for (ds, m, cfg, miss, _seed), s in rows.items():
        g[(ds, m, cfg, miss)].append(s)
    return g


def pick_config(g, ds, m, pref):
    """Config to report for (ds, m): the first of `pref` present, else ''."""
    have = {k[2] for k in g if k[0] == ds and k[1] == m}
    for c in pref:
        if c in have:
            return c
    return None


def cell(g, ds, m, cfg, miss, metric):
    shards = g.get((ds, m, cfg, miss), [])
    st = _stats([x.get(metric) for x in shards])
    return st, len(shards)


def fmt(st, n, pct=False):
    if st is None:
        return "-" if n == 0 else "NaN"
    mean, std, ok = st
    s = f"{mean:.1f}" if pct else f"{mean:.3f}"
    if ok < n:
        s += f" ({ok}/{n})"
    return s


def main_table(g, metric, config_pref, methods, lower_better, pct):
    """Markdown table: rows dataset x rate, columns methods; best in bold."""
    out = []
    hdr = "| dataset | miss | " + " | ".join(SHORT[m] for m in methods) + " |"
    out += [hdr, "|" + "---|" * (len(methods) + 2)]
    for ds in DATASETS:
        for miss in RATES:
            cells, vals = [], []
            for m in methods:
                cfg = pick_config(g, ds, m, config_pref(ds, m))
                st, n = cell(g, ds, m, cfg, miss, metric) if cfg is not None else (None, 0)
                cells.append(fmt(st, n, pct))
                vals.append(st[0] if st else None)
            if all(v is None for v in vals):
                continue
            finite = [v for v in vals if v is not None]
            best = (min if lower_better else max)(finite)
            cells = [f"**{c}**" if v is not None and abs(v - best) < 1e-12 else c
                     for c, v in zip(cells, vals)]
            out.append(f"| {ds} | {int(miss*100)}% | " + " | ".join(cells) + " |")
    return "\n".join(out)


def rule_pref(ds, m):
    """Main tables report the rank-rule config where it exists."""
    if m in V3:
        return ["best/rule", "best"] if ds in RULE_RANK else ["best"]
    if m in RANK_METHODS and ds in RULE_RANK:
        return ["rule"]
    return [""]


def orig_pref(ds, m):
    if m in V3:
        return ["best/orig", "best"] if ds in RULE_RANK else ["best"]
    if m in RANK_METHODS and ds in RULE_RANK:
        return ["orig"]
    return [""]


def v3_rank_summary(g, config_pref):
    """For each (dataset, rate): the best v3 variant's place among all methods."""
    out = ["| dataset | miss | best v3 | NMAE | place | best method | NMAE |",
           "|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        for miss in RATES:
            res = {}
            for m in METHODS:
                cfg = pick_config(g, ds, m, config_pref(ds, m))
                if cfg is None:
                    continue
                st, _ = cell(g, ds, m, cfg, miss, "nmae_unobs")
                if st:
                    res[m] = st[0]
            v3s = {m: v for m, v in res.items() if m in V3}
            if not v3s or len(res) <= len(v3s):
                continue
            bv3 = min(v3s, key=v3s.get)
            others = sorted(v for m, v in res.items() if m not in V3)
            place = 1 + sum(v < v3s[bv3] for v in others)
            bm = min(res, key=res.get)
            out.append(f"| {ds} | {int(miss*100)}% | {SHORT[bv3]} | {v3s[bv3]:.3f} | "
                       f"{place}/{len(others)+1} | {SHORT[bm]} | {res[bm]:.3f} |")
    return "\n".join(out)


def own_cluster_table(g):
    out = ["| dataset | miss | " + " | ".join(
        f"{SHORT[m]} own / spectral" for m in SOTA_CLUSTERING) + " |",
        "|" + "---|" * (len(SOTA_CLUSTERING) + 2)]
    for ds in DATASETS:
        for miss in RATES:
            cells, any_ = [], False
            for m in SOTA_CLUSTERING:
                a, n = cell(g, ds, m, "", miss, "cluster_acc_own")
                b, _ = cell(g, ds, m, "", miss, "cluster_acc")
                any_ |= n > 0
                cells.append(f"{fmt(a, n, True)} / {fmt(b, n, True)}")
            if any_:
                out.append(f"| {ds} | {int(miss*100)}% | " + " | ".join(cells) + " |")
    return "\n".join(out)


def v1_table(g):
    out = ["| dataset | miss | gpu | method | n | time (s) | peak MB | NMAE | clust % |",
           "|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(k[0], k[3], k[2]) for k in g if k[1] == "v1" and k[0] != "synthetic"})
    for ds, miss, cfg in keys:
        for m in ["v1", "v3"]:
            shards = g.get((ds, m, cfg, miss), [])
            if not shards:
                continue
            t = _stats([x.get("total_time_s") for x in shards])
            p = _stats([x.get("peak_train_mb") for x in shards])
            e = _stats([x.get("nmae_unobs") for x in shards])
            c = _stats([x.get("cluster_acc") for x in shards])
            gpu = cfg.split("@", 1)[1].replace("NVIDIA ", "")
            out.append(f"| {ds} | {int(miss*100)}% | {gpu} | {m} | {len(shards)} | "
                       f"{t[0]:.1f} | {p[0]:.0f} | {fmt(e, len(shards))} | "
                       f"{fmt(c, len(shards), True)} |")
    return "\n".join(out)


def coverage(g):
    out = ["| method | " + " | ".join(DATASETS) + " |", "|" + "---|" * (len(DATASETS) + 1)]
    for m in METHODS + ["v1"]:
        row = []
        for ds in DATASETS:
            n = sum(len(v) for k, v in g.items() if k[0] == ds and k[1] == m)
            row.append(str(n) if n else ".")
        out.append(f"| {SHORT[m]} | " + " | ".join(row) + " |")
    return "\n".join(out)


def write_csv(g, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "config", "missing", "n", "n_ok",
                    "nmae", "nmae_std", "clust", "clust_std", "clust_own",
                    "time_s", "peak_mb", "devices"])
        for (ds, m, cfg, miss), shards in sorted(g.items()):
            e = _stats([x.get("nmae_unobs") for x in shards]) or (float("nan"),) * 2 + (0,)
            c = _stats([x.get("cluster_acc") for x in shards]) or (float("nan"),) * 3
            o = _stats([x.get("cluster_acc_own") for x in shards]) or (float("nan"),) * 3
            t = _stats([x.get("total_time_s") for x in shards]) or (float("nan"),) * 3
            p = _stats([x.get("peak_train_mb") for x in shards]) or (float("nan"),) * 3
            devs = sorted({str(x.get("device")) for x in shards})
            w.writerow([ds, m, cfg, miss, len(shards), e[2], e[0], e[1], c[0], c[1],
                        o[0], t[0], p[0], ";".join(devs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--legacy", nargs="*", default=[],
                    help="Sep 13-22 bundle(s): keep only baselines and synthetic")
    ap.add_argument("--md", help="write the tables as markdown")
    ap.add_argument("--csv", help="write every (dataset, method, config, rate) group")
    args = ap.parse_args()

    rows, conflicts, dropped = load(args.roots, args.legacy)
    g = group(rows)
    parts = [f"{len(rows)} runs; {dropped} pre-fix real-data v1/v3 shards dropped; "
             f"{len(conflicts)} conflicting duplicates"]
    for key, p, a, b in conflicts[:20]:
        parts.append(f"  conflict {key}: {a:.4f} vs {b:.4f} ({p})")
    parts += ["", "## Coverage (runs per method x dataset)", coverage(g),
              "", "## Completion NMAE (held-out, lower is better; rank-rule configs)",
              main_table(g, "nmae_unobs", rule_pref, METHODS, True, False),
              "", "## Clustering accuracy % (shared spectral pipeline; rank-rule configs)",
              main_table(g, "cluster_acc", rule_pref, METHODS, False, True),
              "", "## Clustering methods: own labels / spectral on their completion",
              own_cluster_table(g),
              "", "## Where the best v3 variant places (NMAE, rank-rule configs)",
              v3_rank_summary(g, rule_pref),
              "", "## v1 vs v3 at the published stop rule, same GPU",
              v1_table(g),
              "", "## Original-config NMAE for the rank-rule datasets",
              main_table({k: v for k, v in g.items() if k[0] in RULE_RANK},
                         "nmae_unobs", orig_pref, METHODS, True, False)]
    text = "\n".join(parts)
    print(text)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if args.csv:
        write_csv(g, args.csv)


if __name__ == "__main__":
    main()
