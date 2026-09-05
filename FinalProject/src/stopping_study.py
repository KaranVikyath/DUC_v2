"""
Is `lr < lr0/10` a good place to stop, and what would be better?

DUC is fit to a single corrupted matrix with no labels and no validation split,
so the usual early-stopping machinery does not apply directly. This measures the
full training trajectory and asks how well each *deployable* stopping signal
(one that never sees ground truth) locates the point of best held-out accuracy.

Signals compared
  lr/10            the current rule: stop when ReduceLROnPlateau has decayed
                   the learning rate below lr0/10.
  val-mask         hold out a further slice of the OBSERVED entries, train
                   without them, and monitor reconstruction there. Fully
                   unsupervised — it only uses values we were given — and it is
                   the standard cross-validation construction for matrix
                   completion.
  ES-WMV           windowed moving variance of successive reconstructions, from
                   the Deep Image Prior early-stopping literature (Wang et al.,
                   arXiv:2112.06074). Needs neither labels nor a held-out slice:
                   it detects when the fit stops tracking signal and starts
                   tracking noise.
  train loss       plateau of the training objective, for reference.

Oracle (diagnostic only, never usable in practice): held-out NMAE on the truly
missing entries.

Run:  py -3.10 stopping_study.py
"""

import argparse
import json
import os

import numpy as np
import torch

import bench_v1_v3 as B
from custom_funcs import missing_data_generation, thrC, post_proC, err_rate
from deluca_v3 import DeLUCAV3

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "..", "data", "stopping_study.json")


def nmae(pred, truth, sd_f, mask):
    """Per-feature-normalised MAE: scale-free, every feature weighted equally."""
    e = np.abs(pred[mask] - truth[mask]) / sd_f[mask]
    return float(np.mean(e))


def trajectory(P, missing_pct, seed=17, max_iters=1500, val_frac=0.10,
               every=25, window=20, device="cuda", rank_pseudo=1):
    B.seed_all(seed)
    data, full = P["data"], P["full_data"]
    miss = missing_data_generation(data, int(missing_pct * P["total_dp"]))
    obs = ~np.isnan(miss)

    # Carve a validation slice out of the OBSERVED entries. The model never sees
    # them, but we know their values, so error there is measurable without labels.
    rng = np.random.RandomState(seed)
    obs_idx = np.flatnonzero(obs.ravel())
    val_idx = rng.choice(obs_idx, size=int(val_frac * obs_idx.size), replace=False)
    val_mask = np.zeros(miss.size, bool)
    val_mask[val_idx] = True
    val_mask = val_mask.reshape(miss.shape)

    if val_frac <= 0:                    # control: train on every observed entry
        val_mask = np.zeros_like(val_mask)
    train_in = miss.copy()
    train_in[val_mask] = np.nan          # hidden from training
    test_mask = np.isnan(miss)           # truly missing — the oracle target

    shape = P["batch_size"], -1
    sd_f = np.nanstd(train_in.reshape(shape), axis=0)
    sd_f = np.where((sd_f < 1e-8) | ~np.isfinite(sd_f), 1.0, sd_f)
    sd_full = np.broadcast_to(sd_f, (P["batch_size"], sd_f.size)).reshape(full.shape)

    # Feed the model the normalized matrix when the problem asks for it, and map
    # its output back to raw units before scoring. Omitting this trains on the
    # raw 6-decade data, which is a different (and catastrophic) experiment.
    mu, sd = 0.0, 1.0
    if P.get("normalize"):
        mu = np.nanmean(train_in, axis=0, keepdims=True)
        sd = np.nanstd(train_in, axis=0, keepdims=True)
        sd = np.where((sd < 1e-8) | ~np.isfinite(sd), 1.0, sd)
    model_in = (train_in - mu) / sd

    B.seed_all(seed)
    model = DeLUCAV3(
        P["input_shape"], P["flat_layer_size"], P["enc_layer_size"],
        P["deco_layer_size"], P["kernel_size"], P["output_padding"],
        P["lr"], P["K"], P["rank"], reg_const1=P["reg1"], reg_const2=P["reg2"],
        batch_size=P["batch_size"], logs_path=os.path.join(B.LOGS_ROOT, "stop"),
        cluster_model="CFS", device=device, rank_pseudo=rank_pseudo,
        log_every=0).to(device)

    # Several decay thresholds, not just lr0/10. Each factor-of-10 needs ~22
    # more ReduceLROnPlateau events (factor 0.9), so lr0/100 roughly doubles the
    # budget and lr0/1000 triples it.
    thresholds = {10: P["lr"] / 10.0, 100: P["lr"] / 100.0, 1000: P["lr"] / 1000.0}
    lr_stop = {k: None for k in thresholds}
    hist, ring = [], []
    for it in range(1, max_iters + 1):
        _, loss, _, cur_lr = model.finetune_fit(model_in)
        for k, thr in thresholds.items():
            if lr_stop[k] is None and cur_lr < thr:
                lr_stop[k] = it
        if it % every and it != 1:
            continue

        pred = model.completed_data() * sd + mu       # back to raw units
        ring.append(pred.copy())
        if len(ring) > window:
            ring.pop(0)
        # ES-WMV: variance across the window of recent reconstructions
        wmv = float(np.mean(np.var(np.stack(ring), axis=0))) if len(ring) > 1 else np.nan

        rec = dict(iter=it, lr=float(cur_lr), train_loss=float(loss),
                   val_nmae=(nmae(pred, miss, sd_full, val_mask)
                             if val_mask.any() else float("nan")),
                   test_nmae=nmae(pred, full, sd_full, test_mask),
                   wmv=wmv, cluster=float("nan"))
        try:
            C = model.coefficient_matrix()
            y, _ = post_proC(thrC(C, P["alpha1"]), P["K"], P["d"], P["alpha2"])
            rec["cluster"] = (1 - err_rate(P["true_labels"], y)) * 100
        except Exception:
            pass
        hist.append(rec)

    model.summary_writer.close()
    del model
    B.gpu_reset()
    return hist, lr_stop


def summarise(name, hist, lr_stop):
    it = np.array([h["iter"] for h in hist])
    test = np.array([h["test_nmae"] for h in hist])
    wmv = np.array([h["wmv"] for h in hist])
    clu = np.array([h["cluster"] for h in hist])

    best_i = int(np.nanargmin(test))
    picks = {f"lr/{k}": v for k, v in sorted(lr_stop.items()) if v is not None}
    picks["oracle (best held-out)"] = it[best_i]
    if np.isfinite(wmv).any():
        picks["ES-WMV argmin"] = it[int(np.nanargmin(wmv))]

    print(f"\n=== {name} ===")
    print(f"  ran {it[-1]} iters | best held-out NMAE {test[best_i]:.4f} at iter "
          f"{it[best_i]} (clustering there {clu[best_i]:.1f}%)")
    for k, v in sorted(lr_stop.items()):
        if v is None:
            print(f"  lr/{k}: never reached within {it[-1]} iters")
    print(f"  {'signal':<24}{'stops at':>9}{'NMAE there':>12}{'vs best':>10}{'clust':>8}")
    for k, v in picks.items():
        j = int(np.argmin(np.abs(it - v)))
        print(f"  {k:<24}{int(v):>9}{test[j]:>12.4f}"
              f"{100*(test[j]/test[best_i]-1):>9.1f}%{clu[j]:>8.1f}")
    best_c = int(np.nanargmax(clu)) if np.isfinite(clu).any() else None
    if best_c is not None:
        print(f"  best clustering {clu[best_c]:.1f}% at iter {it[best_c]} "
              f"(held-out NMAE there {test[best_c]:.4f})")

    # Where does the curve actually flatten? Last iteration at which held-out
    # NMAE is still >0.5% above its final value — i.e. real headroom remaining.
    fin = test[-1]
    still = np.flatnonzero(test > fin * 1.005)
    if len(still):
        print(f"  held-out NMAE still improving >0.5% until iter {it[still[-1]]}"
              f"  (final {fin:.4f})")
    return picks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=8000)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--seeds", default="17,18,19")
    args = ap.parse_args()
    dev = torch.device("cuda")

    cases = [
        ("hetero(z) 30% missing", lambda sd: dict(B.hetero_problem(seed=sd)), 0.3),
        ("ORL 30% missing", lambda sd: B.orl_problem(sd), 0.3),
        ("ORL 70% missing", lambda sd: B.orl_problem(sd), 0.7),
    ]
    allout = {}
    seeds = [int(s) for s in args.seeds.split(",")]
    for name, fn, pct in cases:
        per_seed = []
        for sd in seeds:
            P = fn(sd) if fn.__code__.co_argcount else fn()
            P["max_iters"] = args.max_iters
            # val_frac=0: train on every observed entry, the deployed setting.
            hist, lr_stop = trajectory(P, pct, seed=sd, max_iters=args.max_iters,
                                       every=args.every, device=dev, val_frac=0.0)
            allout[f"{name} seed{sd}"] = dict(hist=hist, lr_stop=lr_stop)
            per_seed.append((hist, lr_stop))
            summarise(f"{name} seed{sd}", hist, lr_stop)
            with open(OUT, "w") as f:
                json.dump(allout, f, indent=2)

        # Aggregate the thing we actually care about: what each lr threshold
        # costs relative to the best point on the curve, averaged over seeds.
        print(f"\n  --- {name}: mean over {len(seeds)} seeds ---")
        print(f"  {'threshold':<12}{'iters':>8}{'NMAE':>10}{'vs best':>10}"
              f"{'clustering':>12}")
        for k in (10, 100, 1000):
            it_, nm, pen, cl = [], [], [], []
            for hist, lr_stop in per_seed:
                v = lr_stop.get(k)
                if v is None:
                    continue
                itr = np.array([h["iter"] for h in hist])
                te = np.array([h["test_nmae"] for h in hist])
                cu = np.array([h["cluster"] for h in hist])
                j = int(np.argmin(np.abs(itr - v)))
                b = np.nanmin(te)
                it_.append(v); nm.append(te[j]); pen.append(100 * (te[j] / b - 1))
                cl.append(cu[j])
            if it_:
                print(f"  lr/{k:<9}{np.mean(it_):>8.0f}{np.mean(nm):>10.4f}"
                      f"{np.mean(pen):>9.1f}%{np.mean(cl):>11.1f}%")
            else:
                print(f"  lr/{k:<9}{'never reached':>38}")
    print(f"\nSaved {os.path.normpath(OUT)}")
