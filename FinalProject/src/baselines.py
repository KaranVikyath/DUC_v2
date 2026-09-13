"""
Completion baselines, run through the same protocol as DUC.

Every method takes the masked matrix (NaN = missing) and returns a completed
matrix in the same shape and units. They are scored by bench_v1_v3 with the
identical held-out NMAE / mean-imputation floor, so the numbers are comparable.

Tier 1 — classical, runnable today, and the gating question for the paper:
does DUC beat a 30-line low-rank baseline?

  mean        each feature's observed mean. The floor.
  svd_impute  rank-r iterative SVD (Troyanskaya 2001 / "hard-impute"). THE
              classical low-rank matrix-completion baseline; it assumes one
              global subspace where DUC assumes a union of them.
  soft_impute Mazumder, Hastie & Tibshirani 2010: iterative soft-thresholded
              SVD, the nuclear-norm relaxation.
  knn         sklearn KNNImputer. O(B^2 F), so capped by size.
  mice        sklearn IterativeImputer. Fits one regressor per feature per
              round, so only viable for small F.

None of these cluster. For a clustering number they get the same two-stage
treatment — complete, then spectral-cluster a kNN affinity of the completed
rows — which is exactly the pipeline the DUC paper argues a joint method
should beat. Reported separately as `cluster_acc`.

Memory note: the low-rank methods hold X_hat plus an SVD workspace — O(B*F),
linear in B, the same order as DUC v3. That matters for the comparison: they
are not handicapped by a B**2 term the way v1 and SSC-style methods are.
"""

import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ helpers --
def _flat(X):
    B = X.shape[0]
    return X.reshape(B, -1)


def _col_mean_fill(Xm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = np.nanmean(Xm, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    out = Xm.copy()
    idx = np.isnan(out)
    out[idx] = np.broadcast_to(mu, out.shape)[idx]
    return out, mu


# ----------------------------------------------------------------- methods ---
def mean_impute(X_missing, **kw):
    Xm = _flat(X_missing).astype(np.float64)
    out, _ = _col_mean_fill(Xm)
    return out.reshape(X_missing.shape)


def svd_impute(X_missing, rank=10, max_iter=100, tol=1e-4, device="cuda", **kw):
    """Rank-r hard-impute: X_hat <- P_Omega(X) + P_Omega^c(SVD_r(X_hat))."""
    Xm = _flat(X_missing).astype(np.float32)
    obs = ~np.isnan(Xm)
    fill, _ = _col_mean_fill(Xm)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(fill, device=dev)
    O = torch.as_tensor(obs, device=dev)
    Xobs = torch.where(O, X, torch.zeros_like(X))
    r = int(min(rank, min(X.shape) - 1))
    prev = None
    for _ in range(max_iter):
        U, S, V = torch.svd_lowrank(X, q=r, niter=2)
        low = (U * S) @ V.t()
        X = torch.where(O, Xobs, low)
        if prev is not None:
            d = (low - prev).norm() / (prev.norm() + 1e-12)
            if d < tol:
                break
        prev = low
    return X.cpu().numpy().reshape(X_missing.shape)


def soft_impute(X_missing, rank=None, lam_frac=0.05, max_iter=100, tol=1e-4,
                device="cuda", **kw):
    """Mazumder et al. 2010: soft-threshold the singular values by lam each step.

    lam is set as a fraction of the largest singular value of the mean-filled
    matrix, which is the usual practical choice absent a validation split.
    """
    Xm = _flat(X_missing).astype(np.float32)
    obs = ~np.isnan(Xm)
    fill, _ = _col_mean_fill(Xm)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(fill, device=dev)
    O = torch.as_tensor(obs, device=dev)
    Xobs = torch.where(O, X, torch.zeros_like(X))
    q = int(min(rank or min(X.shape) - 1, min(X.shape) - 1))
    _, S0, _ = torch.svd_lowrank(X, q=q, niter=2)
    lam = lam_frac * S0.max()
    prev = None
    for _ in range(max_iter):
        U, S, V = torch.svd_lowrank(X, q=q, niter=2)
        S = torch.clamp(S - lam, min=0.0)
        low = (U * S) @ V.t()
        X = torch.where(O, Xobs, low)
        if prev is not None:
            d = (low - prev).norm() / (prev.norm() + 1e-12)
            if d < tol:
                break
        prev = low
    return X.cpu().numpy().reshape(X_missing.shape)


def knn_impute(X_missing, n_neighbors=5, **kw):
    from sklearn.impute import KNNImputer
    Xm = _flat(X_missing).astype(np.float64)
    out = KNNImputer(n_neighbors=n_neighbors).fit_transform(Xm)
    return out.reshape(X_missing.shape)


def mice_impute(X_missing, max_iter=10, **kw):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    Xm = _flat(X_missing).astype(np.float64)
    out = IterativeImputer(max_iter=max_iter, random_state=0,
                           sample_posterior=False).fit_transform(Xm)
    return out.reshape(X_missing.shape)


# Size caps: a method is skipped (returns None) beyond these, rather than
# silently taking hours. KNN is O(B^2 F); MICE fits F regressors per round.
BASELINES = {
    #  name          fn            max_B    max_F   needs_rank
    "mean":        (mean_impute,   None,    None,   False),
    "svd_impute":  (svd_impute,    None,    None,   True),
    "soft_impute": (soft_impute,   None,    None,   True),
    "knn":         (knn_impute,    3000,    None,   False),
    "mice":        (mice_impute,   None,    128,    False),
}


def run_baseline(name, X_missing, rank=None):
    """Returns (X_hat, seconds) or (None, reason) if the method is capped out."""
    fn, max_B, max_F, needs_rank = BASELINES[name]
    B = X_missing.shape[0]
    F = int(np.prod(X_missing.shape[1:]))
    if max_B is not None and B > max_B:
        return None, f"B={B} > cap {max_B}"
    if max_F is not None and F > max_F:
        return None, f"F={F} > cap {max_F}"
    t0 = time.perf_counter()
    X_hat = fn(X_missing, rank=rank)
    return X_hat, time.perf_counter() - t0


# ------------------------------------------------ complete-then-cluster ------
def cluster_completed(X_hat, K, n_neighbors=10, pca_dim=100, seed=0):
    """Two-stage baseline clustering: kNN affinity on the completed rows,
    then spectral clustering with the same K. PCA first so the kNN graph is
    not O(B^2 F) on 3072-dim pixel vectors."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import kneighbors_graph
    from sklearn.cluster import SpectralClustering
    Xf = _flat(X_hat).astype(np.float64)
    Xf = Xf - Xf.mean(0, keepdims=True)
    if Xf.shape[1] > pca_dim:
        Xf = PCA(n_components=pca_dim, random_state=seed).fit_transform(Xf)
    A = kneighbors_graph(Xf, n_neighbors=n_neighbors, mode="connectivity",
                         include_self=False)
    A = 0.5 * (A + A.T)
    sc = SpectralClustering(n_clusters=K, affinity="precomputed",
                            assign_labels="discretize", random_state=seed)
    return sc.fit_predict(A) + 1
