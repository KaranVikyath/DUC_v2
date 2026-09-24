"""
CAGI — Cluster-Aware Generative Imputation ("Imputation Meets Clustering:
Exploiting Latent Subgroup Structure for Missing Data Recovery", ECML-PKDD 2026).

The official code (github.com/supercocachii/CAGI, pinned below) is fetched into
SOTA_ROOT/CAGI and imported from there; nothing of it is copied here. The model:
a masked (partial-distance) K-means on the incomplete matrix gives every row a
cluster one-hot, which conditions a GAIN-style generator (x, m, c) -> sigmoid,
trained against a hint discriminator with an alpha-weighted reconstruction loss
on the observed entries and a beta-weighted Sinkhorn term; K-means is refit on
the current imputation every `cluster_update_freq` iterations. It returns both
an imputed matrix and cluster labels.

Protocol around it (fixed in advance, nothing tuned on held-out entries):
  * transductive, as the method allows: train_cagi(X, M) then test_cagi on the
    SAME matrix (their run_experiment.py trains on 4 folds and imputes the 5th;
    we have one matrix and score its held-out entries).
  * per-column min-max scaling to [0, 1] from the OBSERVED entries (their
    pipeline uses sklearn MinMaxScaler; the generator ends in a sigmoid), missing
    entries set to 0 and M = 1 on observed entries, exactly as run_experiment.py
    feeds it. Inverted at the end; observed entries are returned untouched.
  * hyper-parameters = run_experiment.py CONFIG['cagi_params'] with
    hidden_dim = 2 * n_features (as run_experiment.py sets it), except
    n_clusters = the true K (their default of 5 is a dataset constant).
  * a FIXED budget of `iterations` generator/discriminator steps. The official
    early stopping (patience counted in checks every 100 iterations: 500 * 100 =
    50,000) and its ReduceLROnPlateau (patience 200 checks = 20,000 iterations)
    cannot trigger inside 5,000 iterations, and its "best" state_dict is a
    shallow copy (the tensors alias the live parameters), so the model returned
    is the last iterate. We keep that behaviour as published.
  * labels = the method's own masked K-means assignment, cluster_model.predict
    on the final (X, M), i.e. the one-hot that conditioned the returned
    imputation. Note that the refits are invariant: the partial distance and the
    centre update both use only entries with M = 1, which the imputation leaves
    unchanged, so every refit reproduces the first fit (info records this as
    kmeans_label_changes). The harness also clusters X_hat itself (cluster_acc).

Runtime patches (applied to the imported module, the repo files are untouched):
  1. CustomKMeans.fit/predict: the official code materialises (B, K, F) arrays
     (15-50 GB of host RAM on COIL100 / OxfordPet / Flowers). Replaced by the
     same partial distance written as matmuls,
       sum_f M_if Cm_kf (X_if - mu_kf)^2
         = (M X^2) Cm^T - 2 (M X)(Cm mu)^T + M (Cm mu^2)^T,  counts = M Cm^T,
     and the per-cluster nanmean centre update as a sparse one-hot product.
     Same initialisation (the K most complete rows), same sqrt(ss F / count)
     distance, same empty-cluster, NaN-fill, center_mask and tolerance rules;
     float64 throughout.
  2. tqdm -> a silent counting pass-through (5,000-line progress bars in CHTC
     logs; the count is reported as iterations_run).
  3. ImprovedGenerator.forward: a batch of ONE row in training mode (the refit
     pass chunks all B rows by mb_size, and B % 128 == 1 leaves a single row)
     crashes BatchNorm; such a call uses the running statistics instead. Any
     other call is unchanged.
Not applied (listed for completeness): deep-copying the "best" state_dict.
"""

import importlib.util
import os
import sys
import time
import types

import numpy as np
import torch

from . import SOTA_ROOT

REPO_URL = "https://github.com/supercocachii/CAGI"
REPO_DIR = "CAGI"
REPO_COMMIT = "35e543b013f62a979421a2b13e403a8e06da4eac"

# run_experiment.py CONFIG['cagi_params'] (the paper's settings). One training
# run costs ~minutes even at D = 3072 (hidden 6144) on an L40S and the patched
# K-means is ~1 min per fit on Flowers with 4 cores, so the "reduced" profile
# needs no cut: it is the same budget, kept as a separate entry for the contract.
_COMMON = dict(iterations=5000, mb_size=128, p_hint=0.9, alpha=200.0, beta=1.5,
               hidden_mult=2, sinkhorn_freq=1000, cluster_update_freq=500,
               patience=500, min_delta=1e-4, cluster_method="kmeans")
BUDGETS = {
    "full":    dict(_COMMON),
    "reduced": dict(_COMMON),
}

_MOD = None
_STATS = {}          # filled by the patched module during one impute() call


# ----------------------------------------------------------- masked K-means --
def _masked_dists(MX, MX2, M, C, Cm):
    """Official partial distance sqrt(sum_f M Cm (X - C)^2 * F / count), (B, K)."""
    F = M.shape[1]
    CmC = Cm * C
    ss = MX2 @ Cm.T
    ss -= 2.0 * (MX @ CmC.T)
    ss += M @ (CmC * C).T
    np.maximum(ss, 0.0, out=ss)
    cnt = M @ Cm.T
    d = np.sqrt(ss * F / np.maximum(cnt, 1e-8))
    d[cnt == 0] = np.inf
    return d


def _make_fast_kmeans(base):
    import scipy.sparse as sp

    class FastCustomKMeans(base):
        """CAGI's CustomKMeans with the (B, K, F) broadcasts written as matmuls."""

        def fit(self, X, M):
            X = np.asarray(X, dtype=np.float64)
            M = (np.asarray(M) != 0).astype(np.float64)
            n_samples, n_features = X.shape
            K = self.n_clusters

            completeness = np.sum(M, axis=1)
            top_idx = np.argsort(-completeness)[:K]        # same call as the original
            self.centers = X[top_idx].copy()
            center_masks = M[top_idx].copy()

            MX = np.where(M > 0, X, 0.0)
            MX2 = MX * X
            rows = np.arange(n_samples)
            labels = np.zeros(n_samples, dtype=int)
            it = -1
            for it in range(self.max_iter):
                labels = np.argmin(_masked_dists(MX, MX2, M, self.centers, center_masks), axis=1)
                old_centers = self.centers.copy()

                P = sp.csr_matrix((np.ones(n_samples), (labels, rows)), shape=(K, n_samples))
                S = np.asarray(P @ MX)                     # (K, F) sums of observed values
                N = np.asarray(P @ M)                      # (K, F) observed counts
                nonempty = np.bincount(labels, minlength=K) > 0
                has = N > 0
                new = np.where(has, S / np.maximum(N, 1.0), self.centers)   # nanmean, NaN -> old
                self.centers[nonempty] = new[nonempty]
                center_masks[nonempty] = has[nonempty].astype(np.float64)

                if np.sum((old_centers - self.centers) ** 2) < self.tol:
                    break

            self.labels_ = labels
            self.center_masks = center_masks
            fits = _STATS.setdefault("kmeans_fits", [])
            fits.append(dict(iters=it + 1, labels=labels.copy()))
            return self

        def predict(self, X, M):
            X = np.asarray(X, dtype=np.float64)
            M = (np.asarray(M) != 0).astype(np.float64)
            MX = np.where(M > 0, X, 0.0)
            return np.argmin(_masked_dists(MX, MX * X, M, self.centers,
                                           np.asarray(self.center_masks, dtype=np.float64)),
                             axis=1)

    return FastCustomKMeans


# ------------------------------------------------------------------ loading --
def _counting_tqdm(iterable, *args, **kwargs):
    _STATS["iterations_run"] = 0
    for x in iterable:
        _STATS["iterations_run"] += 1
        yield x


def _load():
    """Import SOTA_ROOT/CAGI/CAGI.py under a private name and patch it."""
    global _MOD
    if _MOD is not None:
        return _MOD
    site = os.path.join(SOTA_ROOT, "_site")                  # geomloss lives here
    if os.path.isdir(site) and site not in sys.path:
        sys.path.insert(0, site)
    repo = os.path.join(SOTA_ROOT, REPO_DIR)
    src = os.path.join(repo, "CAGI.py")
    if not os.path.isfile(src):
        raise ImportError(f"CAGI not found at {repo}: git clone {REPO_URL} {repo} && "
                          f"git -C {repo} checkout {REPO_COMMIT}")

    # CAGI.py does `from utils import ...`; "utils" is a generic name that may
    # already be taken in sys.modules, so bind theirs only while CAGI.py runs.
    # It also does `from tqdm import tqdm`, which patch 2 replaces anyway, so a
    # stub stands in if tqdm is not installed.
    uspec = importlib.util.spec_from_file_location("_cagi_utils", os.path.join(repo, "utils.py"))
    umod = importlib.util.module_from_spec(uspec)
    uspec.loader.exec_module(umod)
    bind = {"utils": umod}
    if importlib.util.find_spec("tqdm") is None:
        stub = types.ModuleType("tqdm")
        stub.tqdm = _counting_tqdm
        bind["tqdm"] = stub
    sentinel = object()
    prev = {k: sys.modules.get(k, sentinel) for k in bind}
    sys.modules.update(bind)
    try:
        spec = importlib.util.spec_from_file_location("_cagi_official", src)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_cagi_official"] = mod
        spec.loader.exec_module(mod)
    finally:
        for k, v in prev.items():
            if v is sentinel:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    # patch 1: memory-safe masked K-means (get_cluster_model looks the name up at call time)
    mod.CustomKMeans = _make_fast_kmeans(mod.CustomKMeans)
    # patch 2: silent, counting progress bar
    mod.tqdm = _counting_tqdm
    # patch 3: BatchNorm on a single-row chunk in the refit pass
    gen_forward = mod.ImprovedGenerator.forward

    def forward(self, x, m, c):
        if self.training and x.shape[0] < 2:
            self.bn1.eval(); self.bn2.eval()
            try:
                return gen_forward(self, x, m, c)
            finally:
                self.bn1.train(); self.bn2.train()
        return gen_forward(self, x, m, c)

    mod.ImprovedGenerator.forward = forward
    _MOD = mod
    return mod


# ------------------------------------------------------------------- adapter --
def impute(X, *, seed, K, profile="full", device="cuda"):
    if profile not in BUDGETS:
        raise ValueError(f"cagi: unknown profile {profile!r}; have {sorted(BUDGETS)}")
    bud = BUDGETS[profile]
    mod = _load()

    shape = X.shape
    Xf = np.asarray(X, dtype=np.float64).reshape(shape[0], -1)
    B, F = Xf.shape
    obs = np.isfinite(Xf)
    M = obs.astype(np.float64)

    # per-column min-max on observed entries (MinMaxScaler semantics: range 0 -> 1)
    lo = np.where(obs, Xf, np.inf).min(axis=0)
    hi = np.where(obs, Xf, -np.inf).max(axis=0)
    empty = ~np.isfinite(lo)
    lo[empty], hi[empty] = 0.0, 1.0
    rng = hi - lo
    rng[rng <= 0] = 1.0
    Xs = np.where(obs, (Xf - lo) / rng, 0.0)

    dev = torch.device(device if (str(device).startswith("cuda") and torch.cuda.is_available())
                       else "cpu")
    n_clusters = int(max(1, min(int(K), B)))
    params = dict(mb_size=int(min(bud["mb_size"], B)), p_hint=bud["p_hint"],
                  n_clusters=n_clusters, iterations=bud["iterations"],
                  hidden_dim=int(bud["hidden_mult"] * F), cluster_method=bud["cluster_method"],
                  alpha=bud["alpha"], beta=bud["beta"], patience=bud["patience"],
                  min_delta=bud["min_delta"], sinkhorn_freq=bud["sinkhorn_freq"],
                  cluster_update_freq=bud["cluster_update_freq"])

    _STATS.clear()
    np_state = np.random.get_state()
    cuda_devs = [dev.index or 0] if dev.type == "cuda" else []
    t0 = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=cuda_devs), torch.enable_grad():
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            G, cm = mod.train_cagi(Xs, M, params, dev)
            t_train = time.perf_counter() - t0
            X_imp, _ = mod.test_cagi(G, Xs, M, dev, n_clusters, cm)
        labels = np.asarray(cm.predict(Xs, M), dtype=int)
    finally:
        np.random.set_state(np_state)
    del G
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    X_hat = np.where(obs, Xf, np.asarray(X_imp, dtype=np.float64) * rng + lo)

    fits = _STATS.get("kmeans_fits", [])
    first = fits[0]["labels"] if fits else None
    info = dict(method="CAGI", repo=REPO_URL, commit=REPO_COMMIT, profile=profile,
                budget=dict(bud), n_clusters=n_clusters, hidden_dim=params["hidden_dim"],
                mb_size=params["mb_size"], device=str(dev),
                iterations_run=int(_STATS.get("iterations_run", 0)),
                stopped_early=int(_STATS.get("iterations_run", 0)) < bud["iterations"],
                kmeans_fits=len(fits), kmeans_iters=[f["iters"] for f in fits],
                kmeans_label_changes=int(sum(not np.array_equal(f["labels"], first)
                                             for f in fits[1:])),
                labels_source="CAGI masked K-means, cluster_model.predict on (X, M)",
                scaling="per-column min-max from observed entries",
                train_s=t_train, total_s=time.perf_counter() - t0)
    _STATS.clear()
    return dict(X_hat=X_hat.reshape(shape), labels=labels, info=info)
