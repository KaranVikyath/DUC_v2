"""
bench_v1_v3.py — DUC v1 vs DUC v3: VRAM, time, completion and clustering accuracy.

  v1  DeLUCA_original.DeLUCA  — per-feature nn.Linear loop, full SVD, dense (B,B) Coef
  v3  deluca_v3.DeLUCAV3      — low-rank W_f = B_mat @ A on v2's fused CUDA kernels,
                                no dense (B,B) anywhere on the training path

Datasets
  synthetic  200x50 union-of-subspaces (k=5 subspaces of rank 5, noise=1), 10%..90% missing
  ORL        400 faces of 32x32 (40 classes x 10), config copied from dataset_params.py

v3 never builds the coefficient matrix during training; C = P = V_rank @ V_rank.T is
materialized exactly once, after the loop, and that step is timed separately.

Metrics
  peak/avg VRAM   max / mean over iterations of per-iteration peak allocated bytes,
                  net of a stable baseline (see gpu_reset + warmup for why that matters)
  time            total wall clock, mean / median / std per iteration
  completion      1 - ||X_hat - X_full||_F / ||X_full||_F, observed entries restored
  clustering      spectral clustering on C, Hungarian-matched to true labels

Usage
    py -3.10 bench_v1_v3.py                    # all parts
    py -3.10 bench_v1_v3.py --parts syn
    py -3.10 bench_v1_v3.py --parts orl,attrib
    py -3.10 bench_v1_v3.py --report
"""

import argparse
import contextlib
import copy
import gc
import io
import json
import os
import time
import warnings
from datetime import datetime

import numpy as np
import scipy.io as sio
from collections import defaultdict
import torch

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
RESULTS_FILE = os.path.join(_DATA, "v1_vs_v3.json")
LOGS_ROOT = os.path.join(_DATA, "logs", "v1_vs_v3")
ORL_MAT = os.path.join(_DATA, "ORL_32x32.mat")

from custom_funcs import (generate_data, missing_data_generation,
                          thrC, post_proC, err_rate)

SYN_CFG = dict(m=50, n=40, r=5, k=5, noise=1)      # -> (200, 50)
MISSING_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ORL_V1_RATES = [0.3, 0.5, 0.7]     # v1 costs ~1.3 s/iter on ORL; full sweep is ~1.5 h
RANK_PSEUDO = 10
ORL_RANK_PSEUDO_SWEEP = [1, 5, 10, 25]
MULTISEED_SEEDS = [17, 18, 19, 20, 21]
# Gradient magnitudes differ ~100x between these norms, so a fixed-lr comparison
# would mostly measure effective step size. Each loss gets its own lr search.
LOSS_LR_MULT = [0.1, 1.0, 10.0, 100.0]
LOSS_SEEDS = [17, 18, 19, 20, 21]
LOSS_MISSING = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SIZES = [250, 500, 1000, 2000]
RP_SWEEP = [1, 2, 5, 10, 25, 50]
# (label, enc_layer_size, deco_layer_size, rank). rank=None keeps k*r=25.
# The CFS projector spans min(rank, bottleneck) dims, so a bottleneck <= rank
# makes it the identity — included deliberately to show that.
ARCHS = [
    ("50-40-50 (baseline)",            [40],            [50],           None),
    ("50-512-25-512-50 (as asked)",    [512, 25],       [512, 50],      None),
    ("50-512-25-512-50 rank=15",       [512, 25],       [512, 50],      15),
    ("50-512-40-512-50",               [512, 40],       [512, 50],      None),
    ("50-256-64-256-50",               [256, 64],       [256, 50],      None),
    ("50-512-128-40-128-512-50",       [512, 128, 40],  [128, 512, 50], None),
    ("50-128-32-128-50",               [128, 32],       [128, 50],      None),
]
SYN_MAX_ITERS = 3000
ORL_MAX_ITERS = 900
STOPPING_FACTOR = 10
MB = 1024.0 ** 2


def seed_all(seed=17):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def gpu_reset():
    """Drop the previous run's tensors before measuring the next one.

    nn.Module <-> optimizer <-> Parameter are reference cycles, so `del model`
    alone leaves the old model resident until a cyclic-GC pass. Since
    max_memory_allocated() is absolute, that residue inflates every later run
    and compresses the version-to-version ratios.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


# ------------------------------------------------------------------ problems --
def synthetic_problem(cfg=None, seed=17):
    cfg = SYN_CFG if cfg is None else cfg
    seed_all(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        out = generate_data(**cfg)
    (data, full_data, input_shape, batch_size, total_dp, flat_layer_size,
     enc_layer_size, deco_layer_size, kernel_size, output_padding, K, reg1,
     reg2, alpha1, alpha2, d, lr, rank, true_labels) = out
    return dict(name="synthetic", data=data, full_data=full_data,
                input_shape=input_shape, batch_size=batch_size,
                total_dp=total_dp, flat_layer_size=flat_layer_size,
                enc_layer_size=enc_layer_size, deco_layer_size=deco_layer_size,
                kernel_size=kernel_size, output_padding=output_padding, K=K,
                reg1=reg1, reg2=reg2, alpha1=alpha1, alpha2=alpha2, d=d,
                lr=lr, rank=rank, true_labels=true_labels,
                max_iters=SYN_MAX_ITERS)


def sized_problem(B, F=50, r=5, k=5, noise=1, enc=None, deco=None, rank=None, seed=17):
    """Synthetic problem at an arbitrary sample count, optionally re-architected.

    generate_data builds (k*n, m), so B = k*n and F = m.
    `enc`/`deco` override the autoencoder layer sizes; `rank` overrides the CFS rank.
    """
    P = synthetic_problem(dict(m=F, n=B // k, r=r, k=k, noise=noise), seed)
    if enc is not None:
        P["enc_layer_size"] = list(enc)
        P["deco_layer_size"] = list(deco) if deco else [P["input_shape"][-1]]
    if rank is not None:
        P["rank"] = rank
    # The CFS projector spans min(rank, F_enc) directions. When rank >= F_enc it
    # spans the whole latent space, so P = I and the subspace projection is a no-op.
    P["f_enc"] = P["enc_layer_size"][-1]
    P["cfs_identity"] = P["rank"] >= P["f_enc"]
    return P


def hetero_problem(B=200, F=50, r=5, k=5, noise=0.05, scale_decades=6,
                   normalize=True, seed=17):
    """Union-of-subspaces data with per-feature scales spanning many decades.

    Real sensor/telemetry matrices mix units — RPM in the thousands, voltage
    around 12, fuel rate under 30 — so an unnormalized Frobenius objective is
    dominated by whichever column happens to be largest. This generator makes
    that explicit:

        X = (S diag(s)) + 1 o^T + noise * diag(s) * E

    S is the union of k rank-r subspaces; s_j spans `scale_decades` orders of
    magnitude; o_j is a per-feature offset. Column scaling maps a union of
    subspaces to another union of the same dimensions, so the structure the
    method looks for is preserved. Noise is scaled per feature too, so every
    column has the same SNR and the low-magnitude ones are not pure noise.
    """
    seed_all(seed)
    n = B // k
    parts, labels = [], []
    for i in range(k):
        U = np.random.randn(F, r)
        V = np.random.randn(n, r)
        parts.append(V @ U.T)
        labels.append(np.repeat(i + 1, n))
    S = np.vstack(parts)                                    # (B, F)

    half = scale_decades / 2.0
    s = 10.0 ** np.random.uniform(-half, half, size=F)       # per-feature scale
    o = s * np.random.uniform(-5, 5, size=F)                 # per-feature offset
    X = S * s + o + noise * s * np.random.randn(B, F)

    P = dict(name="hetero", data=X, full_data=copy.deepcopy(X),
             input_shape=X.shape, batch_size=B, total_dp=X.size,
             flat_layer_size=[B], enc_layer_size=[40], deco_layer_size=[F],
             kernel_size=None, output_padding=None, K=k, reg1=1.0, reg2=1e-3,
             alpha1=1, alpha2=1, d=r, lr=0.005, rank=k * r,
             true_labels=np.concatenate(labels), max_iters=SYN_MAX_ITERS,
             normalize=normalize)
    P["f_enc"] = 40
    P["cfs_identity"] = P["rank"] >= P["f_enc"]
    P["feature_scales"] = s
    return P


def orl_problem(seed=17):
    """ORL config transcribed from dataset_params.Dataset_params('ORL')."""
    seed_all(seed)
    mat = sio.loadmat(ORL_MAT)
    Img = np.reshape(mat['fea'], [mat['fea'].shape[0], 32, 32, 1])
    num_class, num_sa = 40, 10
    data = np.array(Img[0:num_sa * num_class, :]).astype(float)
    lab = np.array(mat['gnd'][0:num_sa * num_class])
    lab = lab - lab.min() + 1
    true_labels = np.squeeze(lab)

    input_shape = np.shape(data)
    batch_size = num_class * num_sa
    return dict(name="ORL", data=data, full_data=copy.deepcopy(data),
                input_shape=input_shape, batch_size=batch_size,
                total_dp=input_shape[0] * input_shape[1] * input_shape[2],
                flat_layer_size=[batch_size], enc_layer_size=[3, 3, 5],
                deco_layer_size=[3, 3, input_shape[-1]],
                kernel_size=[3, 3, 3], output_padding=[1, 1, 1],
                K=int(true_labels.max()), reg1=1.0, reg2=1e-5,
                alpha1=0.2, alpha2=3.5, d=10, lr=4e-2,
                rank=num_class * 1, true_labels=true_labels,
                max_iters=ORL_MAX_ITERS)


def mat_problem(name, seed=17):
    """Flat .mat datasets, transcribed from dataset_params.Dataset_params."""
    seed_all(seed)
    spec = {
        # name: (file, fea, lab, enc, K, d, lr, rank, alpha1, alpha2)
        "HARUS": ("HARUS.mat", "fea", "lab", [512, 256], 6, 20, 7e-3, 120, 1, 1),
        "DSDD": ("Dataset_for_Sensorless_Drive_diagnosis.mat", "fea", "lab",
                 [40], 11, 4, 7e-3, 44, 1, 1),
    }[name]
    fn, fk, lk, enc, K, d, lr, rank, a1, a2 = spec
    mat = sio.loadmat(os.path.join(_DATA, fn))
    data = np.asarray(mat[fk]).astype(float)
    lab = np.asarray(mat[lk]).ravel()
    lab = lab - lab.min() + 1

    input_shape = data.shape
    B = input_shape[0]
    P = dict(name=name, data=data, full_data=copy.deepcopy(data),
             input_shape=input_shape, batch_size=B,
             total_dp=int(np.prod(input_shape)),
             flat_layer_size=[B], enc_layer_size=list(enc),
             deco_layer_size=[input_shape[-1]], kernel_size=None,
             output_padding=None, K=K, reg1=1.0, reg2=1e-5,
             alpha1=a1, alpha2=a2, d=d, lr=lr, rank=rank,
             true_labels=lab, max_iters=ORL_MAX_ITERS)
    P["f_enc"] = enc[-1]
    P["cfs_identity"] = rank >= P["f_enc"]
    return P


def conv_problem(name, seed=17):
    """Image datasets with conv encoders, transcribed from dataset_params.py.

    NOTE: hyperparameters follow the CODE (dataset_params.py), not the paper's
    Table 1, because two of the table's Rank entries are degenerate — see
    paper_configs.check_configs(). ORL: table says rank 400, code says 40, and
    F_enc is 80, so only the code value leaves the CFS projection active.
    """
    seed_all(seed)
    if name == "COIL20":
        mat = sio.loadmat(os.path.join(_DATA, "COIL20.mat"))
        Img = np.reshape(mat['fea'], (mat['fea'].shape[0], 32, 32, 1))
        data = Img.astype(float)
        lab = np.squeeze(mat['gnd'])
        lab = lab - lab.min() + 1
        cfg = dict(K=20, d=12, rank=20 * 12, lr=4e-2, a1=1, a2=8,
                   enc=[15], deco=[1], ks=[3], opad=[1])
    elif name == "COIL100":
        # Not present in dataset_params.py — modelled on the COIL20 block
        # (same 32x32 imagery, same encoder), with K=100 classes of 72 views.
        mat = sio.loadmat(os.path.join(_DATA, "COIL100.mat"))
        Img = np.reshape(mat['fea'], (mat['fea'].shape[0], 32, 32, 1))
        data = Img.astype(float)
        lab = np.squeeze(mat['gnd'])
        lab = lab - lab.min() + 1
        cfg = dict(K=100, d=12, rank=100 * 12, lr=4e-2, a1=1, a2=8,
                   enc=[15], deco=[1], ks=[3], opad=[1])
    elif name in ("Flowers", "OxfordPet"):
        fn, K, d, lr = (("flowers.mat", 102, 2, 1e-3) if name == "Flowers"
                        else ("oxford_pet.mat", 37, 3, 9e-3))
        mat = sio.loadmat(os.path.join(_DATA, fn))
        data = np.asarray(mat['fea']).astype(float)          # (N, 32, 32, 3)
        lab = np.squeeze(np.asarray(mat['lab']))
        lab = lab - lab.min() + 1
        cfg = dict(K=K, d=d, rank=K * d, lr=lr, a1=1, a2=8,
                   enc=[32, 64, 128], deco=[64, 32, data.shape[-1]],
                   ks=[3, 3, 3], opad=[1, 1, 1])
    elif name == "EYaleB":
        mat = sio.loadmat(os.path.join(_DATA, "YaleBCrop025.mat"))
        img = mat['Y']                      # (2016, 64, 38)
        I, L = [], []
        for i in range(img.shape[2]):
            for j in range(img.shape[1]):
                I.append(np.reshape(img[:, j, i], [42, 48]))
                L.append(i)
        Img = np.expand_dims(np.transpose(np.array(I), [0, 2, 1]), 3)
        num_class = 20
        data = np.array(Img[:64 * num_class]).astype(float)
        lab = np.array(L[:64 * num_class])
        lab = lab - lab.min() + 1
        cfg = dict(K=20, d=10, rank=20 * 10, lr=5e-2, a1=1, a2=3.5,
                   enc=[10, 20, 30], deco=[20, 10, 1], ks=[5, 3, 3],
                   opad=[1, (0, 1), 1])
    else:
        raise ValueError(f"no conv config for {name}")

    input_shape = np.shape(data)
    B = input_shape[0]
    P = dict(name=name, data=data, full_data=copy.deepcopy(data),
             input_shape=input_shape, batch_size=B,
             total_dp=input_shape[0] * input_shape[1] * input_shape[2],
             flat_layer_size=[B], enc_layer_size=cfg["enc"],
             deco_layer_size=cfg["deco"], kernel_size=cfg["ks"],
             output_padding=cfg["opad"], K=cfg["K"], reg1=1.0, reg2=1e-5,
             alpha1=cfg["a1"], alpha2=cfg["a2"], d=cfg["d"], lr=cfg["lr"],
             rank=cfg["rank"], true_labels=np.squeeze(lab),
             max_iters=ORL_MAX_ITERS)
    return P


def get_problem(name, B=None, F=50, seed=17):
    if name == "synthetic":
        return sized_problem(B or 200, F=F, seed=seed) if B else synthetic_problem(seed=seed)
    if name == "ORL":
        return orl_problem(seed)
    if name in ("COIL20", "COIL100", "EYaleB", "Flowers", "OxfordPet"):
        return conv_problem(name, seed)
    return mat_problem(name, seed)


def build_model(version, P, device, logs, rank_pseudo=None, **extra):
    args = (P["input_shape"], P["flat_layer_size"], P["enc_layer_size"],
            P["deco_layer_size"], P["kernel_size"], P["output_padding"],
            P["lr"], P["K"], P["rank"])
    kwargs = dict(reg_const1=P["reg1"], reg_const2=P["reg2"],
                  batch_size=P["batch_size"], model_path=None,
                  logs_path=logs, cluster_model="CFS", device=device)
    if version == "v1":
        from DeLUCA_original import DeLUCA as Model
    elif version == "v3":
        from deluca_v3 import DeLUCAV3 as Model
        kwargs.update(rank_pseudo=rank_pseudo, **extra)
    else:
        raise ValueError(version)
    return Model(*args, **kwargs)


# ------------------------------------------------------------------ one run ---
def run_one(version, P, missing_pct, device, rank_pseudo=None,
            tile=4096, verbose=True, label=None, skip_cluster=False,
            seed=17, pattern="mcar", **extra):
    # `seed` drives BOTH the missing mask and the model init. It used to be
    # hardwired to 17, which meant that on a fixed dataset like ORL every
    # "seed" produced an identical run and the error bars measured only GPU
    # non-determinism. Datasets whose contents vary with the seed (the
    # synthetic generators) were unaffected.
    seed_all(seed)
    if missing_pct <= 0:
        missing = P["data"].copy()
    elif pattern == "mcar":
        missing = missing_data_generation(P["data"], int(missing_pct * P["total_dp"]))
    else:
        missing = make_missing(P["data"], missing_pct, pattern, seed)
    full_data = P["full_data"]

    # Per-feature z-scoring, computed from OBSERVED entries only — using the
    # complete matrix would leak the held-out values into the statistics.
    # Applied after masking so the mask (NaN) is preserved by the transform.
    full_raw = full_data                 # kept in ORIGINAL units for scoring
    missing_raw = missing                # masked input, before any transform
    mu, sd = 0.0, 1.0
    if P.get("normalize"):
        with warnings.catch_warnings():      # all-NaN column under block patterns
            warnings.simplefilter("ignore")
            mu = np.nanmean(missing, axis=0, keepdims=True)
            sd = np.nanstd(missing, axis=0, keepdims=True)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd >= 1e-8), sd, 1.0)
        missing = (missing - mu) / sd
        full_data = (full_data - mu) / sd

    data_norm = float(np.linalg.norm(full_data))

    tag = label or (version if version != "v3" else f"v3(r_p={rank_pseudo})")
    logs = os.path.join(LOGS_ROOT, f"{P['name']}_{tag}_{int(missing_pct*100)}")

    seed_all(seed)                       # model init, same seed as the mask
    base_alloc = gpu_reset()
    model = build_model(version, P, device, logs, rank_pseudo, **extra).to(device)
    torch.cuda.synchronize()
    n_params = sum(p.numel() for p in model.parameters())
    pseudo_params = sum(p.numel() for p in model.pseudo.parameters())

    stopping_lr = P["lr"] / STOPPING_FACTOR
    iter_times, iter_peaks = [], []
    it, hit_cap = 0, False
    C_train, complete_data = None, None

    total_t0 = time.perf_counter()
    while True:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        C_train, cost, complete_data, cur_lr = model.finetune_fit(missing)
        torch.cuda.synchronize()
        iter_times.append(time.perf_counter() - t0)
        iter_peaks.append(torch.cuda.max_memory_allocated())
        it += 1
        if cur_lr < stopping_lr:
            break
        if it >= P["max_iters"]:
            hit_cap = True
            break
    total_time = time.perf_counter() - total_t0
    if complete_data is None:          # v3 defers the device->host copy
        complete_data = model.completed_data()

    peak_train_mb = (max(iter_peaks) - base_alloc) / MB
    avg_vram_mb = (float(np.mean(iter_peaks)) - base_alloc) / MB

    # ---- accuracy ------------------------------------------------------------
    obs = ~np.isnan(missing)
    unobs = ~obs
    pred = complete_data.copy()          # model output, nothing pasted back

    # Held-out-only metrics: scored on the entries the model never saw. The
    # legacy completion_acc below pastes the observed entries back in first,
    # which scores the model partly on values it was given.
    if unobs.any():
        err = pred[unobs] - full_data[unobs]
        mae_unobs = float(np.mean(np.abs(err)))
        rmse_unobs = float(np.sqrt(np.mean(err ** 2)))
        ref = float(np.linalg.norm(full_data[unobs]))
        nrmse_unobs = float(np.linalg.norm(err)) / ref if ref > 0 else float("nan")
        completion_unobs = (1 - nrmse_unobs) * 100

        # Scored in ORIGINAL data units by inverting the normalisation. Without
        # this a normalized run reports z-units and an unnormalized one reports
        # pixels, so the two could not be compared at all.
        pred_raw = pred * sd + mu
        err_r = pred_raw[unobs] - full_raw[unobs]
        mae_unobs_raw = float(np.mean(np.abs(err_r)))
        ref_r = float(np.linalg.norm(full_raw[unobs]))
        completion_unobs_raw = ((1 - float(np.linalg.norm(err_r)) / ref_r) * 100
                                if ref_r > 0 else float("nan"))

        # NMAE: each error divided by its own feature's observed spread before
        # averaging. Absolute-error metrics are dominated by whichever features
        # have the largest units — on 6-decade data, Frobenius^2 draws 90% of its
        # value from 4 of 50 features and raw MAE from 6, while NMAE spreads
        # across 44. This is the only one of the three that is scale-free, and it
        # does not depend on the model having been trained in a normalized space.
        Bn = P["batch_size"]
        mm2 = missing_raw.reshape(Bn, -1)
        sd_f = np.nanstd(mm2, axis=0)
        sd_f = np.where((sd_f < 1e-8) | ~np.isfinite(sd_f), 1.0, sd_f)
        sd_full = np.broadcast_to(sd_f, mm2.shape).reshape(full_raw.shape)
        nmae_unobs = float(np.mean(np.abs(err_r) / sd_full[unobs]))
    else:
        mae_unobs = rmse_unobs = nrmse_unobs = float("nan")
        completion_unobs = mae_unobs_raw = completion_unobs_raw = float("nan")
        nmae_unobs = float("nan")

    complete_data = pred.copy()
    complete_data[obs] = missing[obs]
    completion_acc = 1 - float(np.linalg.norm(complete_data - full_data)) / data_norm

    # ---- coefficient matrix: v3 materializes it only here --------------------
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    alloc_before_C = torch.cuda.memory_allocated()
    tc0 = time.perf_counter()
    # The dense (B,B) is ~B^2*4 bytes on the HOST — 149 GB at B=200k. Past roughly
    # 20k samples completion is still fine but clustering has to be skipped.
    C = None if skip_cluster else (
        model.coefficient_matrix(tile=tile) if version == "v3" else C_train)
    torch.cuda.synchronize()
    cluster_build_ms = (time.perf_counter() - tc0) * 1000
    cluster_extra_mb = (torch.cuda.max_memory_allocated() - alloc_before_C) / MB

    tp0 = time.perf_counter()
    try:
        if C is None:
            cluster_acc = float("nan")
        else:
            y_pred, _ = post_proC(thrC(C, P["alpha1"]), P["K"], P["d"], P["alpha2"])
            cluster_acc = (1 - err_rate(P["true_labels"], y_pred)) * 100
    except Exception as e:                      # ARPACK can fail to converge
        cluster_acc = float("nan")
        print(f"    [clustering failed: {type(e).__name__}]")
    spectral_ms = (time.perf_counter() - tp0) * 1000

    res = dict(dataset=P["name"], version=version, tag=tag,
               missing_pct=missing_pct, rank_pseudo=rank_pseudo,
               iterations=it, hit_cap=hit_cap, total_time_s=total_time,
               avg_iter_ms=float(np.mean(iter_times)) * 1000,
               median_iter_ms=float(np.median(iter_times)) * 1000,
               std_iter_ms=float(np.std(iter_times)) * 1000,
               peak_train_mb=peak_train_mb, avg_vram_mb=avg_vram_mb,
               n_params=int(n_params), pseudo_params=int(pseudo_params),
               cluster_extra_mb=cluster_extra_mb,
               coef_dense_mb=(P["batch_size"] ** 2 * 4) / MB,
               cluster_build_ms=cluster_build_ms, spectral_ms=spectral_ms,
               completion_acc=completion_acc * 100, cluster_acc=cluster_acc,
               mae_unobs=mae_unobs, rmse_unobs=rmse_unobs,
               nrmse_unobs=nrmse_unobs, completion_unobs=completion_unobs,
               mae_unobs_raw=mae_unobs_raw, nmae_unobs=nmae_unobs,
               completion_unobs_raw=completion_unobs_raw,
               normalized=bool(P.get("normalize")),
               final_loss=float(cost), base_alloc_mb=base_alloc / MB,
               lr=P["lr"], B=P["batch_size"], f_enc=P.get("f_enc"),
               cfs_identity=P.get("cfs_identity"),
               enc_layer_size=list(P["enc_layer_size"]),
               deco_layer_size=list(P["deco_layer_size"]), rank=P["rank"])

    if verbose:
        cap = " CAP" if hit_cap else ""
        print(f"  {P['name']:<9} {tag:<16} miss={missing_pct*100:>3.0f}%  it={it:>4}{cap}  "
              f"{total_time:>7.1f}s  {res['avg_iter_ms']:>7.2f} ms/it  "
              f"peak={peak_train_mb:>7.1f}MB  MAE*={mae_unobs:>8.4f}  "
              f"compl*={completion_unobs:>6.2f}%  clust={cluster_acc:>6.2f}%")

    model.summary_writer.close()
    del model, C, C_train
    gpu_reset()
    return res


# ------------------------------------------------------- memory model ---------
def predict_mb(version, B, F, r_p=RANK_PSEUDO, rank=25):
    """Analytic training-peak VRAM, in MB.

    v1  pseudo weight is (F, B, B). Resident copies: param + grad + Adam's
        exp_avg + exp_avg_sq, plus one (F,B,B) gradient temporary  -> 5*F*B^2.
        CFS additionally forms the dense projector P = V V^T        -> +B^2.
    v3  A (F,r_p,B) + B_mat (F,B,r_p) = 2*F*B*r_p, times 4 for param+grad+Adam
        -> 8*F*B*r_p, plus bias (F,B) x4, plus the CFS statistic Z@U (B,rank).
        The gram CFS works in F_enc-space, so nothing here is quadratic in B.
    """
    f = 4.0 / MB
    if version == "v1":
        return (5.0 * F * B * B + B * B) * f
    return (8.0 * F * B * r_p + 4.0 * F * B + 4.0 * B * rank) * f


def scaling_study(device, F=50, r=5, k=5, rp=RANK_PSEUDO,
                  v1_batches=(200, 400, 800, 1600, 3200),
                  v3_batches=(200, 800, 3200, 12800, 51200, 102400)):
    """Measure peak VRAM vs sample count B, then check it against the model."""
    rows = []
    total_vram = torch.cuda.get_device_properties(0).total_memory / MB

    for version, batches in (("v1", v1_batches), ("v3", v3_batches)):
        for B in batches:
            seed_all(17)
            with contextlib.redirect_stdout(io.StringIO()):
                o = generate_data(m=F, n=B // k, r=r, k=k, noise=1)
            (data, full, ishape, bs, tdp, flat, enc, deco, ks, opad,
             K, reg1, reg2, a1, a2, d, lr, rank, labels) = o
            missing = missing_data_generation(data, int(0.3 * tdp))
            P = dict(input_shape=ishape, flat_layer_size=flat, enc_layer_size=enc,
                     deco_layer_size=deco, kernel_size=ks, output_padding=opad,
                     K=K, reg1=reg1, reg2=reg2, lr=lr, rank=rank, batch_size=bs)

            base = gpu_reset()
            model, status, peak = None, "ok", float("nan")
            try:
                model = build_model(version, P, device,
                                    os.path.join(LOGS_ROOT, f"scale_{version}_{B}"),
                                    rp if version == "v3" else None).to(device)
                for _ in range(3):
                    model.finetune_fit(missing)
                torch.cuda.synchronize()
                peak = (torch.cuda.max_memory_allocated() - base) / MB
                # WDDM backs oversized allocations with host RAM instead of failing,
                # so "no exception" does not mean "fit in VRAM".
                status = "ok" if peak <= total_vram else "SPILL"
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                status = "OOM" if "out of memory" in str(e).lower() else type(e).__name__
            finally:
                if model is not None:
                    model.summary_writer.close()
                del model
                gpu_reset()

            pred = predict_mb(version, B, F, rp, rank)
            rows.append(dict(version=version, B=B, F=F, rank_pseudo=rp, rank=rank,
                             peak_mb=peak, predicted_mb=pred, status=status,
                             total_vram_mb=total_vram))
            err = "" if np.isnan(peak) else f"  model {pred:>10.1f} MB ({pred/peak:>5.2f}x)"
            print(f"  {version:<3} B={B:<7} peak={peak:>10.1f} MB  {status:<5}{err}")
    return rows


def print_scaling(rows):
    print(f"\n{'='*100}\n  VRAM SCALING — 3 training steps, F=50 features, growing sample count B"
          f"\n{'='*100}")
    hdr = (f"  {'B':>8} | {'v1 measured':>13} | {'v1 model':>11} | "
           f"{'v3 measured':>13} | {'v3 model':>11} | {'ratio':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for B in sorted({r["B"] for r in rows}):
        c = {r["version"]: r for r in rows if r["B"] == B}
        def cell(v):
            r = c.get(v)
            if r is None:
                return f"{'-':>13}", f"{'-':>11}"
            m = (f"{r['peak_mb']:>10.1f} MB" + ("*" if r["status"] == "SPILL" else " ")
                 if not np.isnan(r["peak_mb"]) else f"{r['status']:>13}")
            return m, f"{r['predicted_mb']:>8.1f} MB"
        a, ap = cell("v1")
        b, bp = cell("v3")
        ratio = ("-" if "v1" not in c or "v3" not in c
                 else f"{c['v1']['predicted_mb']/c['v3']['predicted_mb']:>6.0f}x")
        print(f"  {B:>8} | {a:>13} | {ap:>11} | {b:>13} | {bp:>11} | {ratio:>7}")
    tv = rows[0]["total_vram_mb"]
    print(f"\n  * exceeds the {tv:.0f} MB on this card — no OOM only because the Windows WDDM"
          f"\n    driver spills into host RAM; on Linux this is a hard OOM.")


def print_extrapolation(F=50, rp=RANK_PSEUDO, rank=25):
    print(f"\n{'='*100}\n  EXTRAPOLATION from the validated model (F={F} features, "
          f"r_p={rp})\n{'='*100}")
    hdr = f"  {'B':>10} | {'v1 (F*B^2)':>16} | {'v3 (F*B*r_p)':>14} | {'ratio':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    def human(mb):
        for unit, div in (("TB", 1024**2), ("GB", 1024), ("MB", 1)):
            if mb >= div:
                return f"{mb/div:.1f} {unit}"
        return f"{mb:.2f} MB"

    for B in (200, 1000, 10_000, 50_000, 100_000, 200_000):
        a, b = predict_mb("v1", B, F, rp, rank), predict_mb("v3", B, F, rp, rank)
        print(f"  {B:>10,} | {human(a):>16} | {human(b):>14} | {a/b:>8.0f}x")
    print(f"\n  v1 grows as F*B^2, v3 as F*B*r_p — the ratio itself grows linearly in B.")
    print(f"  At B=200,000 the v1 pseudo-completion weight alone is "
          f"{human(predict_mb('v1', 200_000, F, rp, rank))}, and its dense CFS")
    print(f"  projector P = V V^T is another {human(200_000**2*4/MB)}.")


def warmup(device, P):
    """Allocate the one-time cuBLAS/cuSOLVER workspace before measuring.

    It is cached per device and never released, so whichever run touches it
    first would otherwise absorb its whole cost (~16 MB here).
    """
    saved = P["max_iters"]
    P["max_iters"] = 3
    for version in ("v1", "v3"):
        run_one(version, P, 0.3, device,
                rank_pseudo=RANK_PSEUDO if version == "v3" else None, verbose=False)
    P["max_iters"] = saved
    return gpu_reset() / MB


# ------------------------------------------------------------------- report ---
def _fmt(rows, title, cols_extra=""):
    print(f"\n{'='*118}\n  {title}\n{'='*118}")
    hdr = (f"  {'Miss':>5} | {'Version':<16} | {'Iters':>5} | {'Total(s)':>8} | "
           f"{'ms/iter':>8} | {'PeakVRAM':>9} | {'AvgVRAM':>9} | {'Compl%':>7} | {'Clust%':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pct in sorted({r["missing_pct"] for r in rows}):
        for r in [x for x in rows if x["missing_pct"] == pct]:
            cap = "*" if r["hit_cap"] else " "
            print(f"  {pct*100:>4.0f}% | {r['tag']:<16} | {r['iterations']:>4}{cap} | "
                  f"{r['total_time_s']:>8.1f} | {r['avg_iter_ms']:>8.2f} | "
                  f"{r['peak_train_mb']:>7.1f}MB | {r['avg_vram_mb']:>7.1f}MB | "
                  f"{r['completion_acc']:>6.2f}% | {r['cluster_acc']:>6.2f}%")
        print("  " + "-" * (len(hdr) - 2))
    if any(r["hit_cap"] for r in rows):
        print("  * hit the iteration cap before the LR stopping criterion")


def print_ratio_table(rows, title):
    pairs = []
    for pct in sorted({r["missing_pct"] for r in rows}):
        sub = {r["version"]: r for r in rows if r["missing_pct"] == pct
               and r.get("rank_pseudo") in (None, RANK_PSEUDO)}
        if "v1" in sub and "v3" in sub:
            pairs.append((pct, sub["v1"], sub["v3"]))
    if not pairs:
        return
    print(f"\n  {title}")
    hdr = (f"  {'Miss':>5} | {'Total time':>10} | {'ms/iter':>9} | {'PeakVRAM':>9} | "
           f"{'AvgVRAM':>9} | {'Params':>8} | {'d Compl':>8} | {'d Clust':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pct, a, b in pairs:
        print(f"  {pct*100:>4.0f}% | {a['total_time_s']/b['total_time_s']:>9.1f}x | "
              f"{a['avg_iter_ms']/b['avg_iter_ms']:>8.1f}x | "
              f"{a['peak_train_mb']/b['peak_train_mb']:>8.1f}x | "
              f"{a['avg_vram_mb']/b['avg_vram_mb']:>8.1f}x | "
              f"{a['n_params']/b['n_params']:>7.1f}x | "
              f"{b['completion_acc']-a['completion_acc']:>+7.2f}pp | "
              f"{b['cluster_acc']-a['cluster_acc']:>+7.2f}pp")
    print("  (ratios are v1/v3, so >1x means v3 is better; d columns are v3 minus v1)")


MISSING_PATTERNS = ("mcar", "burst", "channel", "block", "mnar")


def make_missing(data, frac, pattern="mcar", seed=17):
    """Delete a `frac` fraction of entries under different missingness mechanisms.

    Everything so far has used MCAR — entries dropped uniformly at random. Real
    sensor and telemetry data almost never fails that way: a link drops for a
    stretch of time, a channel is absent on a whole class of vehicles, a sensor
    saturates and clips exactly when the value is extreme. Those are strictly
    harder, because the missing entries are correlated and can remove a whole
    subspace direction at once rather than thinning it uniformly.

      mcar     entries uniformly at random (the usual assumption)
      burst    contiguous runs down the sample axis within one feature —
               a sensor or link dropping out for a while
      channel  a whole feature absent across a long span of samples —
               e.g. no HV-battery signal on a petrol vehicle
      block    rectangular blocks of samples x features
      mnar     value-dependent: large values are likelier to go missing,
               the way a saturating sensor clips its own extremes

    All patterns are driven to the same overall fraction so completion numbers
    stay comparable across mechanisms.
    """
    rng = np.random.RandomState(seed)
    X = np.array(data, dtype=float, copy=True)
    B = X.shape[0]
    flat = X.reshape(B, -1)
    F = flat.shape[1]
    target = int(frac * flat.size)
    if target <= 0:
        return X
    mask = np.zeros(flat.shape, dtype=bool)

    def guard(limit=200000):
        """Stop the accept/reject loops if a pattern cannot reach the target."""
        return limit

    if pattern == "mcar":
        idx = rng.choice(flat.size, size=target, replace=False)
        mask.reshape(-1)[idx] = True

    elif pattern in ("burst", "channel"):
        # burst: short runs; channel: long runs covering much of the sample axis
        scale = max(2.0, B * (0.05 if pattern == "burst" else 0.35))
        tries = guard()
        while mask.sum() < target and tries > 0:
            tries -= 1
            f = rng.randint(F)
            L = min(B, max(1, int(rng.exponential(scale))))
            s = rng.randint(0, max(1, B - L + 1))
            mask[s:s + L, f] = True

    elif pattern == "block":
        tries = guard()
        while mask.sum() < target and tries > 0:
            tries -= 1
            h = min(B, max(1, int(rng.exponential(max(2.0, B * 0.12)))))
            w = min(F, max(1, int(rng.exponential(max(2.0, F * 0.12)))))
            s = rng.randint(0, max(1, B - h + 1))
            t = rng.randint(0, max(1, F - w + 1))
            mask[s:s + h, t:t + w] = True

    elif pattern == "mnar":
        # Per feature, the probability of being dropped rises with the value's
        # rank, so the largest readings vanish first.
        order = np.argsort(np.argsort(flat, axis=0), axis=0)      # 0..B-1 ranks
        w = (order + 1.0) ** 3
        w = w / w.sum()
        idx = rng.choice(flat.size, size=target, replace=False, p=w.reshape(-1))
        mask.reshape(-1)[idx] = True
    else:
        raise ValueError(f"unknown missingness pattern: {pattern}")

    # A feature with nothing observed cannot be imputed or normalised by any
    # method; keep one entry so the comparison stays about the mechanism.
    dead = mask.all(axis=0)
    if dead.any():
        mask[rng.randint(0, B, size=int(dead.sum())), np.flatnonzero(dead)] = False

    flat[mask] = np.nan
    return flat.reshape(X.shape)


def mean_impute_row(P, missing_pct, seed, pattern="mcar"):
    """Predict each feature's observed mean. The floor every result must clear.

    Reported as an ordinary row so it is impossible to read a loss table without
    seeing it — a completion figure is meaningless without this reference.
    """
    # Baseline is defined in ORIGINAL units so it is comparable either way:
    # predicting a feature's observed mean is the same prediction whether or
    # not the model happens to train in a normalized space.
    seed_all(seed)
    B = P["batch_size"]
    mm = (missing_data_generation(P["data"], int(missing_pct * P["total_dp"]))
          if pattern == "mcar" else make_missing(P["data"], missing_pct, pattern, seed))
    mm = mm.reshape(B, -1)
    ff = P["full_data"].reshape(B, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        col = np.nanmean(mm, axis=0)
    col = np.where(np.isfinite(col), col, 0.0)
    un = np.isnan(mm)
    err = np.repeat(col[None, :], B, 0)[un] - ff[un]
    ref = float(np.linalg.norm(ff[un]))
    return dict(dataset=P["name"], version="baseline", tag="mean-impute",
                missing_pct=missing_pct, loss_fn="mean-impute", lr_mult=1.0,
                seed=seed, iterations=0, avg_iter_ms=0.0, peak_train_mb=0.0,
                mae_unobs=float(np.mean(np.abs(err))),
                rmse_unobs=float(np.sqrt(np.mean(err ** 2))),
                nrmse_unobs=float(np.linalg.norm(err)) / ref if ref else float("nan"),
                completion_unobs=(1 - float(np.linalg.norm(err)) / ref) * 100
                if ref else float("nan"),
                completion_acc=float("nan"), cluster_acc=float("nan"))


def loss_sweep(problem_fn, device, rows, save, rank_pseudo=1, seeds=None,
               rates=None, lr_mults=None, select_at=0.3):
    """Norm x lr search, then each norm at its own best lr across missing rates.

    Shared by the synthetic and real-dataset sweeps so the protocol cannot drift
    between them. Resumable: rows already present are skipped.
    """
    from deluca_v3 import RECON_LOSSES
    seeds = seeds or LOSS_SEEDS
    rates = rates or LOSS_MISSING
    lr_mults = lr_mults or LOSS_LR_MULT

    def key(stage, kind, mult, miss, sd):
        return (stage, kind, float(mult), round(float(miss), 4), int(sd))

    have = {key(r.get("stage"), r.get("loss_fn"), r.get("lr_mult", 1.0),
                r["missing_pct"], r.get("seed", -1))
            for r in rows if r.get("seed") is not None}
    if have:
        print(f"  resuming — {len(have)} runs already saved")

    print("  [baseline] mean imputation")
    for m in rates:
        for sd in seeds:
            if key(2, "mean-impute", 1.0, m, sd) in have:
                continue
            r = mean_impute_row(problem_fn(sd), m, sd)
            r.update(stage=2)
            rows.append(r)
    save()

    print(f"  [1] norm x lr @ {int(select_at*100)}% missing, {len(seeds)} seeds")
    for kind in RECON_LOSSES:
        for mult in lr_mults:
            for sd in seeds:
                if key(1, kind, mult, select_at, sd) in have:
                    continue
                P = problem_fn(sd)
                P["lr"] = P["lr"] * mult
                r = run_one("v3", P, select_at, device, rank_pseudo=rank_pseudo,
                            verbose=False, label=f"{kind} lr x{mult:g}", seed=sd,
                            loss_fn=kind, use_fused_loss=False)
                r.update(loss_fn=kind, lr_mult=mult, seed=sd, stage=1)
                rows.append(r)
                save()
        b = _best_lr(rows, kind, "mae_unobs", minimize=True)
        print(f"    {kind:<10} best lr x{b[0]:g} -> MAE(held-out) {b[1]:.4f}")

    print("\n  [2] each norm at its best lr, across missing rates")
    for kind in RECON_LOSSES:
        mult = _best_lr(rows, kind, "mae_unobs", minimize=True)[0]
        for m in rates:
            for sd in seeds:
                if key(2, kind, mult, m, sd) in have:
                    continue
                P = problem_fn(sd)
                P["lr"] = P["lr"] * mult
                r = run_one("v3", P, m, device, rank_pseudo=rank_pseudo,
                            verbose=False, label=kind, seed=sd,
                            loss_fn=kind, use_fused_loss=False)
                r.update(loss_fn=kind, lr_mult=mult, seed=sd, stage=2)
                rows.append(r)
                save()
        print(f"    {kind:<10} done (lr x{mult:g})")
    return rows


def _best_lr(rows, kind, metric="completion_acc", minimize=False, stage=1):
    """(lr_mult, best mean metric) for one loss, over stage-`stage` rows."""
    cand = defaultdict(list)
    for r in rows:
        if r.get("loss_fn") == kind and r.get("stage") == stage:
            cand[r["lr_mult"]].append(r[metric])
    # A too-large lr can diverge to NaN; those candidates must not win by
    # accident, and max() over NaN is order-dependent.
    scored = []
    for k, v in cand.items():
        a = np.array(v, dtype=float)
        a = a[np.isfinite(a)]
        if len(a):
            scored.append((float(a.mean()), k))
    if not scored:
        return (1.0, float("nan"))
    best = min(scored) if minimize else max(scored)
    return (best[1], best[0])



def _cell(vals, fmt, mark):
    if not vals:
        return f"{'diverged':>15} "
    return (fmt.format(np.mean(vals)) + "+-" + f"{np.std(vals):<5.3g}" + mark)


def print_hetloss(rows, title="heterogeneous-scale synthetic"):
    ctl = [r for r in rows if r.get("stage") == 0]
    s1 = [r for r in rows if r.get("stage") == 1]
    s2 = [r for r in rows if r.get("stage") == 2]
    kinds = sorted({r["loss_fn"] for r in s1 or s2})

    if s1:
        print("\n" + "=" * 104)
        print(f"  {title.upper()} — reconstruction norm x learning rate, 30% missing")
        print("  MAE on HELD-OUT entries only (lower is better)")
        print("=" * 104)
        mults = sorted({r["lr_mult"] for r in s1})
        hdr = "  " + f"{'loss':<11}" + "".join(f"{'lr x' + f'{m:g}':>15}" for m in mults)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for kind in kinds:
            best = _best_lr(rows, kind, "mae_unobs", minimize=True)[0]
            cells = []
            for m in mults:
                v = [r["mae_unobs"] for r in s1
                     if r["loss_fn"] == kind and r["lr_mult"] == m]
                v = [x for x in v if np.isfinite(x)]
                mark = "*" if m == best else " "
                cells.append(f"{np.mean(v):>14.4f}{mark}" if v else f"{'diverged':>14} ")
            print(f"  {kind:<11}" + "".join(cells))
        print("  * best lr for that norm, carried into stage 2")

    if ctl:
        c = [r["mae_unobs"] for r in ctl]
        f1 = [r["mae_unobs"] for r in s1
              if r["loss_fn"] == "frobenius" and r["lr_mult"] == 1.0]
        if f1:
            print(f"\n  control — frobenius WITHOUT normalization: MAE {np.mean(c):.4f}"
                  f"   vs normalized {np.mean(f1):.4f}"
                  f"   ({np.mean(c) / max(np.mean(f1), 1e-12):.1f}x worse)")

    s3 = [r for r in rows if r.get("stage") == 3]
    if s2:
        s2 = s2 + s3                      # show v1 as an extra row for reference
        kinds = kinds + (["frobenius(v1)"] if s3 else [])
        rates = sorted({r["missing_pct"] for r in s2})
        for metric, lower, fmt, title in (
                ("mae_unobs", True, "{:>8.4f}", "MAE on HELD-OUT entries (lower is better)"),
                ("completion_unobs", False, "{:>8.2f}", "completion % on HELD-OUT entries"),
                ("cluster_acc", False, "{:>8.2f}", "clustering %")):
            print("\n" + "=" * 124)
            print(f"  EACH NORM AT ITS BEST LR — {title}, mean +- std over "
                  f"{len(LOSS_SEEDS)} seeds")
            print("=" * 124)
            hdr = ("  " + f"{'loss':<11}{'lr':>6}"
                   + "".join(f"{str(int(p * 100)) + '%':>16}" for p in rates))
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            best_at = {}
            for p in rates:
                vals = []
                for k in kinds:
                    a = [r[metric] for r in s2
                         if r["loss_fn"] == k and r["missing_pct"] == p]
                    a = [x for x in a if np.isfinite(x)]
                    if a:
                        vals.append((np.mean(a), k))
                best_at[p] = (min(vals) if lower else max(vals))[1] if vals else None
            for kind in kinds:
                mult = next((r["lr_mult"] for r in s2 if r["loss_fn"] == kind), 1.0)
                cells = []
                for p in rates:
                    a = [r[metric] for r in s2
                         if r["loss_fn"] == kind and r["missing_pct"] == p]
                    a = [x for x in a if np.isfinite(x)]
                    cells.append(_cell(a, fmt, "*" if best_at[p] == kind else " "))
                print(f"  {kind:<11}{'x' + f'{mult:g}':>6}" + "".join(cells))
            print("  * best at that missing rate")


def print_loss(rows):
    s1 = [r for r in rows if r.get("stage") == 1]
    s2 = [r for r in rows if r.get("stage") == 2]
    if s1:
        print(f"\n{'='*104}\n  RECONSTRUCTION NORM x LEARNING RATE — 30% missing, "
              f"mean over {len(LOSS_SEEDS)} seeds\n{'='*104}")
        mults = sorted({r["lr_mult"] for r in s1})
        hdr = "  " + f"{'loss':<11}" + "".join(f"{'lr x'+f'{m:g}':>16}" for m in mults)
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for kind in sorted({r["loss_fn"] for r in s1}):
            cells = []
            best = _best_lr(rows, kind)[0]
            for m in mults:
                v = [r["completion_acc"] for r in s1
                     if r["loss_fn"] == kind and r["lr_mult"] == m]
                mark = "*" if m == best else " "
                cells.append(f"{np.mean(v):>14.2f}%{mark}" if v else f"{'-':>15} ")
            print(f"  {kind:<11}" + "".join(cells))
        print("  * best lr for that norm, carried into stage 2")
    if s2:
        print(f"\n{'='*116}\n  EACH NORM AT ITS BEST LR — completion %, "
              f"mean +- std over {len(LOSS_SEEDS)} seeds\n{'='*116}")
        rates = sorted({r["missing_pct"] for r in s2})
        hdr = ("  " + f"{'loss':<11}{'lr':>7}" +
               "".join(f"{str(int(p*100))+'%':>16}" for p in rates) + f"{'ms/iter':>10}")
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        best_by_rate = {}
        for p in rates:
            vals = [(np.mean([r["completion_acc"] for r in s2
                              if r["loss_fn"] == k and r["missing_pct"] == p]), k)
                    for k in sorted({r["loss_fn"] for r in s2})]
            best_by_rate[p] = max(vals)[1]
        for kind in sorted({r["loss_fn"] for r in s2}):
            mult = next(r["lr_mult"] for r in s2 if r["loss_fn"] == kind)
            cells = []
            for p in rates:
                v = [r["completion_acc"] for r in s2
                     if r["loss_fn"] == kind and r["missing_pct"] == p]
                mark = "*" if best_by_rate[p] == kind else " "
                cells.append(f"{np.mean(v):>9.2f}+-{np.std(v):<4.2f}{mark}"
                             if v else f"{'-':>15} ")
            ms = np.mean([r["avg_iter_ms"] for r in s2 if r["loss_fn"] == kind])
            print(f"  {kind:<11}{'x'+f'{mult:g}':>7}" + "".join(cells) + f"{ms:>10.2f}")
        print("  * best norm at that missing rate")


def print_opt(rows):
    print(f"\n{'='*112}\n  OPTIMIZATION ABLATION — cumulative, synthetic 30% missing"
          f"\n{'='*112}")
    hdr = (f"  {'#':>2} {'variant':<34} | {'Iters':>5} | {'ms/iter':>8} | {'vs base':>8} | "
           f"{'PeakVRAM':>9} | {'Compl%':>7} | {'Clust%':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    base = next((r["avg_iter_ms"] for r in rows if r.get("cumulative")), None)
    for i, r in enumerate(rows):
        speed = f"{base/r['avg_iter_ms']:.2f}x" if base else "-"
        mark = " " if r.get("cumulative") else "x"
        print(f"  {mark:>2} {r['tag']:<34} | {r['iterations']:>5} | "
              f"{r['avg_iter_ms']:>8.2f} | {speed:>8} | {r['peak_train_mb']:>7.1f}MB | "
              f"{r['completion_acc']:>6.2f}% | {r['cluster_acc']:>6.2f}%")
    print("  " + "-" * (len(hdr) - 2))
    print("  rows marked 'x' are measured-and-rejected alternatives, not cumulative")
    print("  ms/iter is the reliable speed metric: each variant changes the numerics")
    print("  slightly, so the LR schedule stops at a different iteration count.")


def print_attrib(rows):
    print(f"\n{'='*100}\n  v3 KERNEL ATTRIBUTION — which parts of v2's CUDA stack v3 uses"
          f"\n{'='*100}")
    hdr = (f"  {'Variant':<34} | {'Iters':>5} | {'ms/iter':>8} | {'PeakVRAM':>9} | "
           f"{'Compl%':>7} | {'Clust%':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['tag']:<34} | {r['iterations']:>5} | {r['avg_iter_ms']:>8.2f} | "
              f"{r['peak_train_mb']:>7.1f}MB | {r['completion_acc']:>6.2f}% | "
              f"{r['cluster_acc']:>6.2f}%")


def print_cluster_table(rows):
    print(f"\n{'='*104}\n  COEFFICIENT MATRIX — when the (B,B) is built\n{'='*104}")
    hdr = (f"  {'Dataset':<10} | {'Miss':>5} | {'Version':<16} | {'built':<20} | "
           f"{'x per run':>9} | {'build(ms)':>9} | {'extra VRAM':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        v3 = r["version"] == "v3"
        print(f"  {r['dataset']:<10} | {r['missing_pct']*100:>4.0f}% | {r['tag']:<16} | "
              f"{'once, after training' if v3 else 'every iteration':<20} | "
              f"{1 if v3 else r['iterations']:>9} | {r['cluster_build_ms']:>9.2f} | "
              f"{r['cluster_extra_mb']:>8.2f}MB")


def report(saved):
    if saved.get("syn"):
        _fmt(saved["syn"], "SYNTHETIC 200x50 — union of subspaces (k=5, rank 5, noise=1)")
        print_ratio_table(saved["syn"], "v1 -> v3 on synthetic")
    if saved.get("orl"):
        _fmt(saved["orl"], "ORL — 400 faces, 32x32, 40 classes")
        print_ratio_table(saved["orl"], "v1 -> v3 on ORL")
    if saved.get("orl_sweep"):
        _fmt(saved["orl_sweep"], "ORL — v3 rank_pseudo sweep @ 30% missing")
    if saved.get("orlloss"):
        print_hetloss(saved["orlloss"], title="ORL (pixel units)")
    if saved.get("hetloss"):
        print_hetloss(saved["hetloss"], title="heterogeneous-scale synthetic")
    if saved.get("loss"):
        print_loss(saved["loss"])
    if saved.get("multiseed"):
        from collections import defaultdict
        g = defaultdict(list)
        for r in saved["multiseed"]:
            g[(r["missing_pct"], r["version"])].append(r)
        print(f"\n{'='*122}\n  SYNTHETIC 200x50 — v1 vs v3, mean +- std over "
              f"{len(MULTISEED_SEEDS)} seeds\n{'='*122}")
        hdr = (f"  {'Miss':>5} | {'Ver':<3} | {'n':>2} | {'Total(s)':>16} | "
               f"{'ms/iter':>14} | {'PeakVRAM':>9} | {'Compl%':>16} | {'Clust%':>16}")
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for pct in sorted({k[0] for k in g}):
            for v in ("v1", "v3"):
                rs = g.get((pct, v))
                if not rs:
                    continue
                def st(f):
                    a = np.array([x[f] for x in rs], float); a = a[~np.isnan(a)]
                    return (np.mean(a), np.std(a)) if len(a) else (float('nan'),)*2
                t, ts = st("total_time_s"); m, msd = st("avg_iter_ms")
                c, cs = st("completion_acc"); k_, ks = st("cluster_acc")
                print(f"  {pct*100:>4.0f}% | {v:<3} | {len(rs):>2} | "
                      f"{t:>8.1f} +-{ts:>5.1f} | {m:>7.2f} +-{msd:>4.2f} | "
                      f"{np.mean([x['peak_train_mb'] for x in rs]):>7.1f}MB | "
                      f"{c:>8.2f} +-{cs:>5.2f} | {k_:>8.2f} +-{ks:>5.2f}")
            print("  " + "-" * (len(hdr) - 2))
    if saved.get("rpsweep"):
        print(f"\n{'='*112}\n  PSEUDO-RANK r_p x PROBLEM SIZE — synthetic F=50, 30% missing"
              f"\n{'='*112}")
        hdr = (f"  {'B':>6} | {'r_p':>4} | {'Iters':>5} | {'ms/iter':>8} | "
               f"{'PeakVRAM':>9} | {'W_f params':>11} | {'Compl%':>7} | {'Clust%':>7}")
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for B in sorted({r["B"] for r in saved["rpsweep"]}):
            for r in [x for x in saved["rpsweep"] if x["B"] == B]:
                print(f"  {B:>6} | {r['rank_pseudo']:>4} | {r['iterations']:>5} | "
                      f"{r['avg_iter_ms']:>8.2f} | {r['peak_train_mb']:>7.1f}MB | "
                      f"{r['pseudo_params']:>11,} | {r['completion_acc']:>6.2f}% | "
                      f"{r['cluster_acc']:>6.2f}%")
            print("  " + "-" * (len(hdr) - 2))
    if saved.get("arch"):
        print(f"\n{'='*118}\n  AUTOENCODER SHAPE — B=250, F=50, r_p=1, 30% missing"
              f"\n{'='*118}")
        hdr = (f"  {'architecture':<30} | {'rank':>4} | {'bott':>4} | {'Iters':>5} | "
               f"{'ms/iter':>8} | {'VRAM':>8} | {'Compl%':>7} | {'Clust%':>7} | CFS")
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for r in saved["arch"]:
            print(f"  {r['tag']:<30} | {r['rank']:>4} | {r['f_enc']:>4} | "
                  f"{r['iterations']:>5} | {r['avg_iter_ms']:>8.2f} | "
                  f"{r['peak_train_mb']:>6.1f}MB | {r['completion_acc']:>6.2f}% | "
                  f"{r['cluster_acc']:>6.2f}% | "
                  f"{'IDENTITY' if r['cfs_identity'] else 'active'}")
        print("  " + "-" * (len(hdr) - 2))
        print("  CFS=IDENTITY means rank >= bottleneck, so P = I and the subspace")
        print("  projection does nothing — the model degenerates to a plain autoencoder.")
    if saved.get("scale"):
        print_scaling(saved["scale"])
        print_extrapolation()
    if saved.get("opt"):
        print_opt(saved["opt"])
    if saved.get("attrib"):
        print_attrib(saved["attrib"])
    cl = [r for r in saved.get("syn", []) + saved.get("orl", [])
          if r["missing_pct"] in (0.3, 0.5)]
    if cl:
        print_cluster_table(cl)


# --------------------------------------------------------------------- main ---
def single_run(args):
    """One (version, dataset, missing, seed) config -> one JSON.

    This is the CHTC-native entry point: the sweep is embarrassingly parallel
    over the cross product, so HTCondor queues one job per line instead of one
    long serial job. Aggregate the shards afterwards with --aggregate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = get_problem(args.dataset, B=args.B, F=args.F, seed=args.seed)
    if args.max_iters:
        P["max_iters"] = args.max_iters
    if device.type == "cuda":
        # Warmup exists only to allocate the cuBLAS/cuSOLVER workspace before the
        # measured run. It must never cluster: that would build the dense (B,B)
        # affinity on the HOST — 10 GB at B=50k, 160 GB at B=200k — regardless of
        # --no-cluster, which previously applied to the real run only.
        warm = dict(P)
        warm["max_iters"] = 3
        run_one(args.version, warm, args.missing, device,
                rank_pseudo=args.rank_pseudo, verbose=False, skip_cluster=True)
    r = run_one(args.version, P, args.missing, device,
                rank_pseudo=args.rank_pseudo, skip_cluster=args.no_cluster)
    r["seed"] = args.seed
    r["device"] = (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    out = args.out or os.path.join(
        _DATA, "shards",
        f"{args.dataset}_{args.version}_rp{args.rank_pseudo}_"
        f"m{int(args.missing*100)}_s{args.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(r, f, indent=2)
    print(f"wrote {out}")


def aggregate(pattern):
    """Merge per-job shards, averaging repeated seeds and reporting spread."""
    import glob
    from collections import defaultdict
    shards = [json.load(open(p)) for p in glob.glob(pattern)]
    if not shards:
        print(f"no shards matched {pattern}")
        return
    groups = defaultdict(list)
    for s in shards:
        groups[(s["dataset"], s["tag"], s["missing_pct"])].append(s)

    hdr = (f"  {'dataset':<10} | {'version':<14} | {'Miss':>5} | {'n':>2} | "
           f"{'ms/iter':>8} | {'PeakVRAM':>9} | {'Compl%':>15} | {'Clust%':>15}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key in sorted(groups):
        g = groups[key]
        def ms(f):
            v = np.array([x[f] for x in g], dtype=float)
            v = v[~np.isnan(v)]
            return (np.mean(v), np.std(v)) if len(v) else (float("nan"),) * 2
        c_m, c_s = ms("completion_acc")
        k_m, k_s = ms("cluster_acc")
        print(f"  {key[0]:<10} | {key[1]:<14} | {key[2]*100:>4.0f}% | {len(g):>2} | "
              f"{np.mean([x['avg_iter_ms'] for x in g]):>8.2f} | "
              f"{np.mean([x['peak_train_mb'] for x in g]):>7.1f}MB | "
              f"{c_m:>7.2f} +-{c_s:>5.2f} | {k_m:>7.2f} +-{k_s:>5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="syn,orl,attrib")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep saved rows and skip configurations already run")
    ap.add_argument("--single", action="store_true", help="run one config (CHTC)")
    ap.add_argument("--aggregate", metavar="GLOB", help="merge shards into a table")
    ap.add_argument("--version", default="v3", choices=["v1", "v3"])
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "ORL", "COIL20", "COIL100", "EYaleB",
                             "Flowers", "OxfordPet", "HARUS", "DSDD"])
    ap.add_argument("--B", type=int, default=None, help="sample count (synthetic)")
    ap.add_argument("--F", type=int, default=50, help="feature count (synthetic)")
    ap.add_argument("--missing", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--rank-pseudo", type=int, default=RANK_PSEUDO, dest="rank_pseudo")
    ap.add_argument("--max-iters", type=int, default=None, dest="max_iters")
    ap.add_argument("--no-cluster", action="store_true",
                    help="skip clustering (the dense (B,B) is infeasible past ~20k)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.aggregate)
        return
    if args.single:
        single_run(args)
        return
    if args.report:
        with open(RESULTS_FILE) as f:
            report(json.load(f))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("This benchmark measures VRAM — a CUDA device is required.")
    parts = {p.strip() for p in args.parts.split(",")}

    syn = synthetic_problem()
    print("=== DUC v1 vs DUC v3 ===")
    print(f"Device     : {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/MB:.0f} MB)")
    print(f"torch      : {torch.__version__}")
    print(f"synthetic  : {syn['data'].shape}  rank={syn['rank']}  lr={syn['lr']}")
    if os.path.exists(ORL_MAT):
        orl = orl_problem()
        print(f"ORL        : {orl['data'].shape}  K={orl['K']}  rank={orl['rank']}  "
              f"lr={orl['lr']}")
    else:
        orl = None
        print(f"ORL        : NOT FOUND at {ORL_MAT} — ORL parts will be skipped")
    from deluca_v3 import DeLUCAV3 as _V3
    import inspect
    _d = inspect.signature(_V3.__init__).parameters
    print(f"v3         : rank_pseudo={RANK_PSEUDO}, "
          f"cfs_backend={_d['cfs_backend'].default}, "
          f"eigh_device={_d['eigh_device'].default}\n")

    # Keep results for parts we are not re-running, so `--parts opt` does not
    # discard an expensive sweep produced by an earlier `--parts syn`.
    out = {"syn": [], "orl": [], "orl_sweep": [], "opt": [], "attrib": [],
           "scale": [], "rpsweep": [], "arch": [], "multiseed": [], "loss": [],
           "hetloss": [], "orlloss": [], "orlnorm": [], "flatab": [], "patterns": [],
           "checks": {}}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                out.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    out["timestamp"] = datetime.now().isoformat()
    out["device"] = torch.cuda.get_device_name(0)
    if not args.resume:
        for p in parts:                  # only the requested parts are refreshed
            if p in out:
                out[p] = []

    def save():
        with open(RESULTS_FILE, "w") as f:
            json.dump(out, f, indent=2)

    print("[warmup]")
    out["checks"]["warmup_resident_mb"] = warmup(device, syn)
    print(f"  stable baseline {out['checks']['warmup_resident_mb']:.2f} MB "
          f"(subtracted from every run)\n")

    if "syn" in parts:
        print("[1] synthetic sweep, 10% -> 90% missing")
        for pct in MISSING_SWEEP:
            for version in ("v1", "v3"):
                out["syn"].append(run_one(
                    version, syn, pct, device,
                    rank_pseudo=RANK_PSEUDO if version == "v3" else None))
                save()
            print()

    if "orl" in parts and orl is not None:
        print("[2] ORL — v3 sweep 10% -> 90% missing")
        for pct in MISSING_SWEEP:
            out["orl"].append(run_one("v3", orl, pct, device, rank_pseudo=RANK_PSEUDO))
            save()
        print(f"\n[3] ORL — v1 at {[int(p*100) for p in ORL_V1_RATES]}% "
              f"(v1 costs ~1.3 s/iter here; the full sweep would take ~1.5 h)")
        for pct in ORL_V1_RATES:
            out["orl"].append(run_one("v1", orl, pct, device))
            save()
        print("\n[4] ORL — v3 rank_pseudo sweep @ 30% missing")
        for rp in ORL_RANK_PSEUDO_SWEEP:
            out["orl_sweep"].append(run_one("v3", orl, 0.3, device, rank_pseudo=rp))
            save()
        print()

    if "patterns" in parts:
        print("[patterns] missingness mechanism sweep — MCAR vs structured")
        print("  Everything so far assumed MCAR. Real telemetry loses data in")
        print("  bursts, whole channels, and value-dependent clips instead.\n")
        cases = [("hetero(z)", lambda sd: dict(hetero_problem(seed=sd))),
                 ("ORL", lambda sd: orl_problem(sd))]
        for dname, pfn in cases:
            print(f"  --- {dname} ---")
            for pat in MISSING_PATTERNS:
                for m in (0.1, 0.3, 0.5, 0.7):
                    for sd in (17, 18, 19):
                        base = mean_impute_row(pfn(sd), m, sd, pattern=pat)
                        base.update(stage="baseline", pattern=pat, dset=dname)
                        out["patterns"].append(base)
                        r = run_one("v3", pfn(sd), m, device, rank_pseudo=1,
                                    seed=sd, pattern=pat, verbose=False,
                                    label=f"{pat}", loss_fn="frobenius",
                                    use_fused_loss=False)
                        r.update(stage="model", pattern=pat, dset=dname, seed=sd)
                        out["patterns"].append(r)
                    save()
                got = [x for x in out["patterns"] if x["dset"] == dname
                       and x["pattern"] == pat and x.get("stage") == "model"]
                bas = [x for x in out["patterns"] if x["dset"] == dname
                       and x["pattern"] == pat and x.get("stage") == "baseline"]
                print(f"    {pat:<8} NMAE {np.nanmean([x['nmae_unobs'] for x in got]):.4f}"
                      f"  (floor {np.nanmean([x['mae_unobs'] for x in bas]):.4f})"
                      f"  clust {np.nanmean([x['cluster_acc'] for x in got]):5.1f}%")
        print()

    if "flatab" in parts:
        print("[flatab] block-diagonal (per-feature) vs fully-connected flattened")
        print("  pseudo-completion, at IDENTICAL parameter counts (2*B*F*r either way).")
        print("  blockdiag: F independent (B,B) sample-mixers, no cross-feature path.")
        print("  full:      one (B*F, B*F) map, rank r — mixes across features too.\n")
        for (Bs, Fs) in ((100, 20), (200, 50)):
            print(f"  --- {Bs}x{Fs} (flattened N={Bs*Fs}) ---")
            for rp in (1, 5, 10):
                for mode in ("blockdiag", "full"):
                    for m in (0.1, 0.3, 0.5, 0.7):
                        for sd in (17, 18, 19):
                            Pf = sized_problem(Bs, F=Fs, seed=sd)
                            r = run_one("v3", Pf, m, device, rank_pseudo=rp,
                                        seed=sd, verbose=False,
                                        label=f"{mode} r={rp}",
                                        pseudo_mode=mode)
                            r.update(mode=mode, rank_pseudo=rp, seed=sd,
                                     Bs=Bs, Fs=Fs)
                            out["flatab"].append(r)
                        save()
                    got = [x for x in out["flatab"]
                           if x["Bs"] == Bs and x["rank_pseudo"] == rp
                           and x["mode"] == mode]
                    print(f"    r={rp:<3} {mode:<10} "
                          f"NMAE {np.nanmean([x['nmae_unobs'] for x in got]):.4f}  "
                          f"clust {np.nanmean([x['cluster_acc'] for x in got]):5.1f}%  "
                          f"{np.mean([x['avg_iter_ms'] for x in got]):5.2f} ms/it")
        print()

    if "orlnorm" in parts:
        P0 = orl_problem()
        print("[orlnorm] ORL: is per-pixel z-scoring required?")
        print(f"  {P0['input_shape']}, encoder {P0['enc_layer_size']}, rank "
              f"{P0['rank']}, lr {P0['lr']}, v3 only, frobenius, rank_pseudo=1")
        print(f"  pixels {P0['data'].min():.0f}..{P0['data'].max():.0f}, "
              f"mean {P0['data'].mean():.1f}, std {P0['data'].std():.1f}")
        print("  BOTH conditions scored in ORIGINAL PIXEL UNITS — the normalized"
              "\n  model's output is mapped back through the transform first, "
              "otherwise\n  z-units and pixels would not be comparable.\n")

        def orl_norm(sd):
            P = orl_problem(sd)
            P["normalize"] = True
            return P

        for m in MISSING_SWEEP:
            for sd in MULTISEED_SEEDS:
                base = mean_impute_row(orl_problem(sd), m, sd)
                base.update(stage="baseline", norm=False)
                out["orlnorm"].append(base)
                for norm, fn in ((False, orl_problem), (True, orl_norm)):
                    r = run_one("v3", fn(sd), m, device, rank_pseudo=1, seed=sd,
                                verbose=False, label=f"norm={norm}",
                                loss_fn="frobenius", use_fused_loss=False)
                    r.update(stage="model", norm=norm, seed=sd)
                    out["orlnorm"].append(r)
                save()
            done = [r for r in out["orlnorm"]
                    if r["missing_pct"] == m and r.get("stage") == "model"]
            f = lambda nb: np.mean([x["mae_unobs_raw"] for x in done
                                    if x["norm"] is nb])
            print(f"  {int(m*100):>3}% missing  MAE(px)  raw={f(False):7.3f}  "
                  f"z-scored={f(True):7.3f}")
        print()

    if "orlloss" in parts:
        P0 = orl_problem()
        print(f"[orlloss] ORL {P0['input_shape']}, encoder {P0['enc_layer_size']} "
              f"kernel {P0['kernel_size']}, rank {P0['rank']}, lr {P0['lr']}")
        print(f"  pixels {P0['data'].min():.0f}..{P0['data'].max():.0f}; MAE is in "
              f"pixel units. No normalization — ORL is single-channel imagery, so "
              f"all\n  features already share a scale (unlike the telemetry-style "
              f"synthetic).")
        print(f"  equal-budget protocol: max_iters={P0['max_iters']}, v3 only, "
              f"rank_pseudo=1\n")
        loss_sweep(orl_problem, device, out["orlloss"], save, rank_pseudo=1)
        print()

    if "hetloss" in parts:
        from deluca_v3 import RECON_LOSSES

        # Resume support: a crash 90 minutes into a 505-run sweep should cost
        # one run, not the whole thing. Rows are keyed by their configuration.
        def _key(stage, kind, mult, miss, sd):
            return (stage, kind, float(mult), round(float(miss), 4), int(sd))

        have = {_key(r.get("stage"), r.get("loss_fn"), r.get("lr_mult", 1.0),
                     r["missing_pct"], r.get("seed", -1))
                for r in out["hetloss"] if r.get("seed") is not None}
        if have:
            print(f"[hetloss] resuming — {len(have)} runs already saved, skipping those")

        h0 = hetero_problem()
        s = h0["feature_scales"]
        print(f"[hetloss] heterogeneous-scale synthetic, {h0['data'].shape}, "
              f"z-scored per feature from observed entries only")
        print(f"  feature scales span {s.min():.3g} .. {s.max():.3g} "
              f"({np.log10(s.max()/s.min()):.1f} decades)")
        print(f"  raw column std ranges {h0['data'].std(0).min():.3g} .. "
              f"{h0['data'].std(0).max():.3g}")
        print(f"  selection metric: MAE on HELD-OUT entries (lower is better); "
              f"observed entries are never pasted back\n")

        # Unnormalized control: shows whether normalization is actually needed.
        print("  [control] no normalization, frobenius, 30% missing")
        for sd in LOSS_SEEDS:
            if _key(0, "frobenius", 1.0, 0.3, sd) in have:
                continue
            Pc = hetero_problem(seed=sd, normalize=False)
            r = run_one("v3", Pc, 0.3, device, rank_pseudo=1, verbose=False,
                        label="frobenius (unnormalized)", loss_fn="frobenius",
                        use_fused_loss=False)
            r.update(loss_fn="frobenius", lr_mult=1.0, seed=sd, stage=0)
            out["hetloss"].append(r)
            save()
        ctl = [r["mae_unobs"] for r in out["hetloss"] if r.get("stage") == 0]
        print(f"    MAE(held-out) = {np.mean(ctl):.4f} +- {np.std(ctl):.4f}\n")

        print("  [1] norm x learning rate, 30% missing, 3 seeds")
        for kind in RECON_LOSSES:
            for mult in LOSS_LR_MULT:
                for sd in LOSS_SEEDS:
                    if _key(1, kind, mult, 0.3, sd) in have:
                        continue
                    Ph = hetero_problem(seed=sd)
                    Ph["lr"] = Ph["lr"] * mult
                    r = run_one("v3", Ph, 0.3, device, rank_pseudo=1,
                                label=f"{kind} lr x{mult:g}", verbose=False,
                                loss_fn=kind, use_fused_loss=False)
                    r.update(loss_fn=kind, lr_mult=mult, seed=sd, stage=1)
                    out["hetloss"].append(r)
                    save()
            b = _best_lr(out["hetloss"], kind, "mae_unobs", minimize=True)
            print(f"    {kind:<10} best lr x{b[0]:g} -> MAE(held-out) {b[1]:.4f}")

        print("\n  [2] each norm at its best lr, across missing rates")
        for kind in RECON_LOSSES:
            mult = _best_lr(out["hetloss"], kind, "mae_unobs", minimize=True)[0]
            for m in LOSS_MISSING:
                for sd in LOSS_SEEDS:
                    if _key(2, kind, mult, m, sd) in have:
                        continue
                    Ph = hetero_problem(seed=sd)
                    Ph["lr"] = Ph["lr"] * mult
                    r = run_one("v3", Ph, m, device, rank_pseudo=1,
                                label=f"{kind}", verbose=False,
                                loss_fn=kind, use_fused_loss=False)
                    r.update(loss_fn=kind, lr_mult=mult, seed=sd, stage=2)
                    out["hetloss"].append(r)
                    save()
            print(f"    {kind:<10} done (lr x{mult:g})")

        # Stage 3: the published baseline on the same data, same held-out metric.
        # v1's loss is hardwired to the Frobenius form, so it is compared as
        # published against v3 running whichever norm stage 2 selected.
        print("\n  [3] v1 baseline (frobenius, as published) on the same data")
        for m in LOSS_MISSING:
            for sd in LOSS_SEEDS:
                if _key(3, "frobenius(v1)", 1.0, m, sd) in have:
                    continue
                Ph = hetero_problem(seed=sd)
                r = run_one("v1", Ph, m, device, verbose=False, label="v1")
                r.update(loss_fn="frobenius(v1)", lr_mult=1.0, seed=sd, stage=3)
                out["hetloss"].append(r)
                save()
            print(f"    v1 @ {int(m*100)}% done")
        print()

    if "loss" in parts:
        from deluca_v3 import RECON_LOSSES
        print("[loss/1] reconstruction norm x learning rate, 30% missing, 3 seeds")
        for kind in RECON_LOSSES:
            for mult in LOSS_LR_MULT:
                for s in LOSS_SEEDS:
                    Pl = dict(synthetic_problem(seed=s))
                    Pl["lr"] = Pl["lr"] * mult
                    r = run_one("v3", Pl, 0.3, device, rank_pseudo=1,
                                label=f"{kind} lr x{mult:g}", verbose=False,
                                loss_fn=kind, use_fused_loss=False)
                    r.update(loss_fn=kind, lr_mult=mult, seed=s, stage=1)
                    out["loss"].append(r)
                    save()
            best = _best_lr(out["loss"], kind)
            print(f"  {kind:<10} best lr x{best[0]:g} -> "
                  f"completion {best[1]:.2f}%")

        print("\n[loss/2] each norm at its best lr, across missing rates")
        for kind in RECON_LOSSES:
            mult = _best_lr(out["loss"], kind)[0]
            for m in LOSS_MISSING:
                for s in LOSS_SEEDS:
                    Pl = dict(synthetic_problem(seed=s))
                    Pl["lr"] = Pl["lr"] * mult
                    r = run_one("v3", Pl, m, device, rank_pseudo=1,
                                label=f"{kind}", verbose=False,
                                loss_fn=kind, use_fused_loss=False)
                    r.update(loss_fn=kind, lr_mult=mult, seed=s, stage=2)
                    out["loss"].append(r)
                    save()
            print(f"  {kind:<10} done (lr x{mult:g})")
        print()

    if "multiseed" in parts:
        print("[multiseed] synthetic, v1 vs v3, 5 seeds x 10-90% missing")
        for seed in MULTISEED_SEEDS:
            for pct in MISSING_SWEEP:
                for version in ("v1", "v3"):
                    Ps = synthetic_problem(seed=seed)
                    r = run_one(version, Ps, pct, device,
                                rank_pseudo=RANK_PSEUDO if version == "v3" else None)
                    r["seed"] = seed
                    out["multiseed"].append(r)
                    save()
            print(f"  --- seed {seed} done ---\n")

    if "rpsweep" in parts:
        print("[rpsweep] pseudo-rank r_p x problem size, 30% missing")
        for B in SIZES:
            for rp in RP_SWEEP:
                if rp > B:
                    continue
                Pb = sized_problem(B)
                out["rpsweep"].append(run_one("v3", Pb, 0.3, device, rank_pseudo=rp,
                                              label=f"B={B} r_p={rp}"))
                save()
            print()

    if "arch" in parts:
        print("[arch] autoencoder shape at B=250, F=50, r_p=1")
        for label, enc, deco, rank in ARCHS:
            Pa = sized_problem(250, enc=enc, deco=deco, rank=rank)
            note = "  [CFS = identity: rank >= bottleneck]" if Pa["cfs_identity"] else ""
            print(f"    {label}  rank={Pa['rank']} bottleneck={Pa['f_enc']}{note}")
            out["arch"].append(run_one("v3", Pa, 0.3, device, rank_pseudo=1,
                                       label=label))
            save()
        print()

    if "scale" in parts:
        print("[scale] VRAM vs sample count, measured then extrapolated")
        out["scale"] = scaling_study(device)
        save()
        print()

    if "opt" in parts:
        print("[opt] cumulative optimization ablation @ synthetic 30% missing")
        # Start from v3 as it stood before the optimization study, then add one
        # change at a time. Each dict is the FULL kwarg set for that variant.
        base_kw = dict(cfs_backend="lowrank", eigh_device="gpu", skip_nan_check=False,
                       track_autoenc_loss=True, defer_output=False, log_every=1)
        steps = [
            ("0 baseline (pre-study v3)", {}),
            ("1 + gram CFS (eigh on GPU)", dict(cfs_backend="gram")),
            ("2 + eigh on CPU", dict(eigh_device="cpu")),
            ("3 + EncoderNoNaN", dict(skip_nan_check=True)),
            ("4 + drop unused autoenc loss", dict(track_autoenc_loss=False)),
            ("5 + defer output copy", dict(defer_output=True)),
            ("6 + no per-iter tensorboard", dict(log_every=0)),
        ]
        kw = dict(base_kw)
        for label, delta in steps:
            kw.update(delta)
            r = run_one("v3", syn, 0.3, device, rank_pseudo=RANK_PSEUDO,
                        label=label, **kw)
            r["cumulative"] = True
            out["opt"].append(r)
            save()
        # Measured-and-rejected alternatives, on top of the fully optimized config.
        rejected = [("x cuSOLVER CFS (v2 kernel)", dict(cfs_backend="cusolver")),
                    ("x lowrank CFS (original)", dict(cfs_backend="lowrank"))]
        for label, delta in rejected:
            alt = dict(kw); alt.update(delta)
            r = run_one("v3", syn, 0.3, device, rank_pseudo=RANK_PSEUDO,
                        label=label, **alt)
            r["cumulative"] = False
            out["opt"].append(r)
            save()
        print()

    if "attrib" in parts:
        print("[5] v3 kernel attribution @ synthetic 30% missing")
        variants = [
            dict(label="v3 fused kernel + svd_lowrank", kw={}),
            dict(label="v3 fused kernel + cuSOLVER CFS", kw=dict(cfs_backend="cusolver")),
        ]
        for v in variants:
            out["attrib"].append(run_one("v3", syn, 0.3, device,
                                         rank_pseudo=RANK_PSEUDO,
                                         label=v["label"], **v["kw"]))
            save()
        # PyTorch-only path: disable the v2 kernel lookup for one run
        import deluca_v3
        saved_loader = deluca_v3._load_pseudo_cuda
        deluca_v3._load_pseudo_cuda = lambda: None
        try:
            out["attrib"].append(run_one("v3", syn, 0.3, device,
                                         rank_pseudo=RANK_PSEUDO,
                                         label="v3 no fused kernel (pure torch)"))
        finally:
            deluca_v3._load_pseudo_cuda = saved_loader
        save()
        print()

    save()
    report(out)
    print(f"\nSaved to {os.path.normpath(RESULTS_FILE)}")


if __name__ == "__main__":
    main()
