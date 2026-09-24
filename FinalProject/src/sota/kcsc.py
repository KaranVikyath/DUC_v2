"""
KC-SC: Kernel Correction + standard spectral clustering on incomplete data.

    Fangchen Yu, Runze Zhao, Zhan Shi, Yiwen Lu, Jicong Fan, Yicheng Zeng,
    Jianfeng Mao, Wenye Li. "Boosting Spectral Clustering on Incomplete Data via
    Kernel Correction and Affinity Learning". Advances in Neural Information
    Processing Systems 36 (NeurIPS 2023).

Independent implementation from the paper. The authors' MATLAB repository
(github.com/SciYu/Spectral-Clustering-on-Incomplete-Data) has no license, so
nothing of it is copied, translated or imported; every step below is written
from the equations and defaults stated in the paper. The method only clusters:
it never estimates the missing entries.

Pipeline (paper Sec. 2.2 Eq. 1, Sec. 3.2 Eq. 5-6 and Algorithm 1, Sec. 5.2):
  1. Naive distance on the co-observed features I_ij of rows i and j (Eq. 1):
         d0_ij = ||x_i(I) - x_j(I)||_2 * sqrt(d / |I|).
     Computed with masked matmuls, where Xc is the input with missing entries
     set to 0 and M the observed mask:
         sum_f M_if M_jf (x_if - x_jf)^2 = (Xc^2) M^T + M (Xc^2)^T - 2 Xc Xc^T,
         |I_ij| = (M M^T)_ij.
  2. Gaussian kernel k0_ij = exp(-d0_ij^2 / sigma^2), sigma = median{d0_ij}
     (Sec. 5.1).
  3. Kernel correction (Eq. 5): the projection of K0 onto
         F = {K >= 0 (PSD), k_ii = 1, k_ij = k_ji in [0, 1]} = F1 n F2,
     by Dykstra's alternating projection (Algorithm 1), with
         P1(K) = U max(Sigma, 0) U^T      (PSD cone, Eq. 6)
         P2(K) = median{0, k_ij, 1}, k_ii = 1                           (Eq. 6)
     X0 = K0, P0 = Q0 = 0;  Y = P2(X + P);  P <- X + P - Y;
     X' = P1(Y + Q);  Q <- Y + Q - X';  stop when X' stops changing.
     The corrected kernel is the last PSD iterate X (Algorithm 1, line 12).
  4. kNN graph on the corrected kernel, k = 10 (Sec. 5.2): each row keeps the
     kernel values of its 10 most similar other rows; the graph is the union
     (w_ij > 0 if j is among i's 10 nearest or i among j's), weighted by the
     corrected kernel.
  5. Normalized cut, Ng-Jordan-Weiss form (the paper's reference [1]): the K
     leading eigenvectors of D^{-1/2} W D^{-1/2}, rows normalized to unit
     length, then k-means. Labels are returned as 1..K.

Protocol choices (fixed in advance; nothing is tuned, nothing looks at the
held-out entries or the labels):
  * Every parameter is a default stated in the paper: sigma = median distance,
    k = 10, KC maxiter = 100, tol = 1e-5, bounds [l, u] = [0, 1]. There is no
    hyper-parameter to select, so the 5% observed hold-out is not needed.
  * The paper also proposes self-expressive affinity learners (KSL-Pp, KSL-Sp,
    AKLSR) on the corrected kernel. They are not run here: their lambda was
    chosen per dataset by clustering performance (paper Appendix F, Table 7),
    i.e. with labels, and AKLSR's rho and initialisation are not given in the
    paper. The standard-spectral-clustering path (paper Table 2) needs no
    label-tuned value.
  * KC stopping test: the paper tests ||X_{t+1} - X_t||_F < tol in absolute
    terms. The kernel has n^2 entries of order 1, so that number grows with n
    and its float32 rounding floor (~1e-7 ||X||_F) is above 1e-5 for any n we
    run; the test could never trigger and the result would depend on the
    arithmetic. We use the scale-free reading ||X_{t+1} - X_t||_F / ||X_t||_F
    < tol with the same tol, and record both numbers in info.
  * Pairs with NO co-observed feature (|I| = 0) have no distance estimate
    (Eq. 1 divides by |I|). The paper does not say what to do with them. We give
    them the largest estimated distance (so the smallest kernel value) and
    leave the rest to KC; they are excluded from the sigma median. The count
    is recorded in info (it is 0 on every image set at <= 90% missing; it
    matters only for low-d tabular data at high missing rates).
  * The kernel is invariant to a global rescaling of the data (sigma rescales
    with it) and distances are invariant to per-column shifts, so the
    harness's units are kept (raw pixels for images, z-scores for tabular),
    as for every other baseline. Columns are centred and globally scaled only
    to keep the float64 distance matmuls well conditioned.
  * Precision: distances in float64, the KC loop in float32 (one eigh of an
    n x n matrix per iteration dominates the cost; if an eigh fails to
    converge it is retried in float64), the final spectral eigh in float64.
  * X_hat: KC-SC produces no completion. X_hat is MEAN IMPUTATION (observed
    entries untouched, missing ones set to the observed column mean), returned
    only to satisfy the contract; its NMAE is the mean baseline's and must not
    be reported as KC-SC's. Only `labels` (cluster_acc_own) is the method's.

Cost: O(n^2 d) for the distances, O(maxiter * n^3) for KC (one symmetric
eigendecomposition per iteration, plus an n x n x m matmul with m <= n/2),
O(n^3) for the spectral step. Memory O(n^2): about eight dense n x n float32
matrices, ~3.4 GB at n = 10,299 (HAR).
"""

import math
import time

import numpy as np
import torch

PAPER = ("Yu, Zhao, Shi, Lu, Fan, Zeng, Mao, Li. Boosting Spectral Clustering on "
         "Incomplete Data via Kernel Correction and Affinity Learning. NeurIPS 2023")

# Paper defaults (Algorithm 1: maxiter 100, tol 1e-5; Sec. 5.1: sigma = median;
# Sec. 5.2: kNN graph with k = 10). kcsc is not in sota.EXPENSIVE, so the
# protocol runs "full" on every dataset; "reduced" exists for the contract only
# and halves the KC iteration cap.
_COMMON = dict(kc_tol=1e-5, kc_low=0.0, kc_high=1.0, knn=10, kmeans_n_init=10,
               kc_dtype="float32", spectral_dtype="float64")
BUDGETS = {
    "full":    dict(_COMMON, kc_maxiter=100),
    "reduced": dict(_COMMON, kc_maxiter=50),
}

_DT = {"float32": torch.float32, "float64": torch.float64}


# --------------------------------------------------------------- distances --
def naive_distances(Xc, M, chunk_elems=1 << 26):
    """Eq. (1) for every pair of rows; NaN where the rows share no feature.

    Xc: (n, d) float64 with missing entries 0; M: (n, d) float64 0/1 mask.
    Returns an (n, n) float64 tensor on Xc's device.
    """
    n, d = Xc.shape
    A2 = Xc * Xc
    D = torch.empty((n, n), dtype=torch.float64, device=Xc.device)
    step = max(1, min(n, chunk_elems // max(n, 1)))
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        ss = A2[r0:r1] @ M.T
        ss += M[r0:r1] @ A2.T
        ss -= 2.0 * (Xc[r0:r1] @ Xc.T)
        ss.clamp_(min=0.0)
        cnt = M[r0:r1] @ M.T                     # |I_ij|, exact integers in float64
        ss.mul_(d).div_(cnt.clamp(min=1.0)).sqrt_()
        ss[cnt < 0.5] = float("nan")
        D[r0:r1] = ss
    D.fill_diagonal_(0.0)
    return D


def _median(v):
    """Exact median of a 1-D tensor (mean of the two middle values if even)."""
    m = v.numel()
    lo = torch.kthvalue(v, (m + 1) // 2).values
    hi = torch.kthvalue(v, m // 2 + 1).values
    return 0.5 * (float(lo) + float(hi))


# ------------------------------------------------------- kernel correction --
def _eigh(S):
    try:
        return torch.linalg.eigh(S)
    except RuntimeError:                         # rare non-convergence in float32
        w, V = torch.linalg.eigh(S.double())
        return w.to(S.dtype), V.to(S.dtype)


def project_psd(S):
    """P1 of Eq. (6): U max(Sigma, 0) U^T. Returns (X, number of negative eigenvalues,
    sum of the negative eigenvalues).

    eigh reads one triangle only, so a slightly asymmetric S is harmless. The
    reconstruction uses whichever eigen-block is smaller: S minus the negative
    part, or the positive part alone (identical in exact arithmetic).
    """
    n = S.shape[0]
    w, V = _eigh(S)                              # ascending
    neg = int((w < 0).sum())
    neg_mass = float(w[:neg].sum()) if neg else 0.0
    if neg == 0:
        X = S.clone()
    elif neg <= n - neg:
        Vn = V[:, :neg]
        X = S - (Vn * w[:neg]) @ Vn.T
    else:
        Vp = V[:, neg:]
        X = (Vp * w[neg:]) @ Vp.T
    del V
    X = 0.5 * (X + X.T)
    return X, neg, neg_mass


def project_box(T, low, high):
    """P2 of Eq. (6): entries clipped to [low, high], unit diagonal."""
    Y = T.clamp(low, high)
    Y.fill_diagonal_(1.0)
    return Y


def kernel_correction(K0, maxiter=100, tol=1e-5, low=0.0, high=1.0):
    """Algorithm 1 (Dykstra's projection onto F1 n F2). Returns (K_hat, stats)."""
    X = K0.clone()
    P = torch.zeros_like(K0)
    Q = torch.zeros_like(K0)
    rel = absd = float("nan")
    negs, t = [], 0
    for t in range(1, maxiter + 1):
        T = X + P
        Y = project_box(T, low, high)
        P = T.sub_(Y)                            # P_{t+1} = X_t + P_t - Y_t
        S = Y.add_(Q)                            # Y_t + Q_t (Y is not needed after)
        Xn, neg, neg_mass = project_psd(S)
        Q = S.sub_(Xn)                           # Q_{t+1} = Y_t + Q_t - X_{t+1}
        absd = float(torch.linalg.norm(Xn - X))
        rel = absd / max(float(torch.linalg.norm(X)), 1e-30)
        negs.append((neg, neg_mass))
        X = Xn
        if rel < tol:
            break
    stats = dict(kc_iters=t, kc_hit_cap=bool(rel >= tol), kc_last_rel_change=rel,
                 kc_last_abs_change=absd,
                 kc_first_neg_eigs=negs[0][0] if negs else 0,
                 kc_first_neg_eig_sum=negs[0][1] if negs else 0.0,
                 kc_last_neg_eigs=negs[-1][0] if negs else 0)
    return X, stats


# ----------------------------------------------------- spectral clustering --
def knn_graph(Kmat, k):
    """Union kNN graph weighted by the kernel (self-loops excluded, weights >= 0)."""
    n = Kmat.shape[0]
    k = int(min(k, n - 1))
    S = Kmat.clone()
    S.fill_diagonal_(-float("inf"))
    vals, idx = torch.topk(S, k, dim=1)
    del S
    A = torch.zeros_like(Kmat)
    A.scatter_(1, idx, vals.clamp(min=0.0))
    return torch.maximum(A, A.T)


def ncut_embedding(W, K, dtype):
    """Ng-Jordan-Weiss: K leading eigenvectors of D^-1/2 W D^-1/2, unit rows."""
    W = W.to(dtype)
    deg = W.sum(1)
    isolated = int((deg <= 0).sum())
    dm = deg.clamp(min=torch.finfo(dtype).tiny).rsqrt()
    Ms = dm[:, None] * W * dm[None, :]
    Ms = 0.5 * (Ms + Ms.T)
    w, V = torch.linalg.eigh(Ms)                 # ascending
    U = V[:, -K:]
    U = U / U.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return U, w[-K:], (float(w[-K - 1]) if w.numel() > K else float("nan")), isolated


# ------------------------------------------------------------------ adapter --
def impute(X, *, seed, K, profile="full", device="cuda"):
    if profile not in BUDGETS:
        raise ValueError(f"kcsc: unknown profile {profile!r}; have {sorted(BUDGETS)}")
    bud = BUDGETS[profile]
    from sklearn.cluster import KMeans

    dev = torch.device(device if (not str(device).startswith("cuda")
                                  or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seed)
    shape = X.shape
    Xf = np.asarray(X, dtype=np.float64).reshape(shape[0], -1)
    n, d = Xf.shape
    obs = np.isfinite(Xf)
    K = int(max(1, min(int(K), n)))

    # X_hat = mean imputation (KC-SC does not complete the matrix).
    cnt_col = obs.sum(0)
    mu = np.where(cnt_col > 0, np.where(obs, Xf, 0.0).sum(0) / np.maximum(cnt_col, 1), np.nan)
    fill = mu.copy()
    fill[~np.isfinite(fill)] = float(np.nanmean(mu)) if np.isfinite(mu).any() else 0.0
    X_hat = np.where(obs, Xf, fill[None, :])

    t0 = time.perf_counter()
    with torch.no_grad():
        # centre columns (distances unchanged) and scale globally (kernel unchanged)
        Xc = np.where(obs, Xf - fill[None, :], 0.0)
        rms = math.sqrt(float((Xc ** 2).sum()) / max(int(obs.sum()), 1)) or 1.0
        Xt = torch.as_tensor(Xc / rms, dtype=torch.float64, device=dev)
        Mt = torch.as_tensor(obs, dtype=torch.float64, device=dev)
        D = naive_distances(Xt, Mt)
        del Xt, Mt
        undefined = torch.isnan(D)
        n_zero_overlap = int(undefined.sum()) // 2
        iu = torch.triu(torch.ones((n, n), dtype=torch.bool, device=dev), diagonal=1)
        vals = D[iu & ~undefined]
        del iu
        if vals.numel() == 0:
            sigma, dmax = 1.0, 1.0
        else:
            sigma = _median(vals)
            dmax = float(vals.max())
        del vals
        if not (sigma > 0 and math.isfinite(sigma)):
            sigma = 1.0
        D[undefined] = dmax
        del undefined
        wdt = _DT[bud["kc_dtype"]]
        K0 = torch.exp(-(D / sigma) ** 2).to(wdt)
        del D
        K0 = 0.5 * (K0 + K0.T)
        t_kernel = time.perf_counter() - t0

        t1 = time.perf_counter()
        Khat, kc_stats = kernel_correction(K0, maxiter=bud["kc_maxiter"], tol=bud["kc_tol"],
                                           low=bud["kc_low"], high=bud["kc_high"])
        kernel_change = float(torch.linalg.norm(Khat - K0) / torch.linalg.norm(K0))
        box_violation = float(torch.relu(bud["kc_low"] - Khat).max()
                              + torch.relu(Khat - bud["kc_high"]).max())
        diag_dev = float((Khat.diagonal() - 1.0).abs().max())
        del K0
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_kc = time.perf_counter() - t1

        t2 = time.perf_counter()
        W = knn_graph(Khat, bud["knn"])
        del Khat
        U, top_eigs, next_eig, isolated = ncut_embedding(W, K, _DT[bud["spectral_dtype"]])
        del W
        U = U.cpu().numpy().astype(np.float64)
    km = KMeans(n_clusters=K, n_init=bud["kmeans_n_init"], random_state=seed).fit(U)
    labels = km.labels_.astype(int) + 1
    t_sc = time.perf_counter() - t2
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    info = dict(method="KC-SC (kernel correction + kNN normalized-cut spectral clustering)",
                paper=PAPER, implementation="independent implementation from the paper",
                profile=profile, budget=dict(bud), device=str(dev), n=n, d=d, K=K,
                sigma_median_scaled=sigma, zero_overlap_pairs=n_zero_overlap,
                zero_overlap_rule="largest estimated distance; excluded from the sigma median",
                kc_stop_rule="||X_t+1 - X_t||_F / ||X_t||_F < tol (paper: absolute)",
                **kc_stats, kc_kernel_rel_change=kernel_change,
                kc_box_violation=box_violation, kc_diag_max_dev=diag_dev,
                spectral_top_eigs_minmax=[float(top_eigs.min()), float(top_eigs.max())],
                spectral_next_eig=next_eig, graph_isolated_nodes=isolated,
                kmeans_inertia=float(km.inertia_),
                labels_source="KC-SC normalized cut on the corrected kernel's kNN graph",
                X_hat="MEAN IMPUTATION, not the method's (KC-SC clusters only; "
                      "do not report its NMAE as KC-SC's)",
                kernel_s=t_kernel, kc_s=t_kc, spectral_s=t_sc,
                total_s=time.perf_counter() - t0)
    return dict(X_hat=X_hat.reshape(shape), labels=labels, info=info)
