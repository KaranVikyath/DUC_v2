"""
Alt PZF-EnSC+gLRMC: alternating projected-zero-filled elastic-net subspace
clustering and per-group low-rank matrix completion.

    C. Lane, R. Boger, C. You, M. C. Tsakiris, B. D. Haeffele and R. Vidal,
    "Classifying and comparing approaches to subspace clustering with missing
    data", ICCV 2019 Workshops (RSL-CV).

This is an independent implementation from the paper (its Algorithm 1 with the
PZF-EnSC affinity and gLRMC completion, Sec. 2.1, and the parameter ranges of
its Table 2). The authors' scmd-comparison repository has no license and was
not copied or translated. The MIT-licensed altmin-scmc repository (Copyright
2018 Connor Lane) was consulted for the solver structure. The component
algorithms come from their own papers: EnSC (You, Li, Robinson and Vidal,
CVPR 2016), Ng-Jordan-Weiss spectral clustering (NIPS 2001), and inexact ALM
matrix completion (Lin, Chen and Ma, arXiv:1009.5055, 2010).

Model. X is D x N with the samples as columns (the transpose of our (B, F)),
Omega is the observed mask and X0 = P_Omega(X) is the zero fill.
  Y^0 = X0.
  PZF-EnSC on Y: the columns of Y are scaled to unit l2 norm, then
      min_C  sum_j lam/2 ||w_j * (Y c_j - y_j)||^2 + gam ||c_j||_1
                                         + (1-gam)/2 ||c_j||^2,   c_jj = 0,
    with w_j = Omega_j in every iteration (the "projection" is kept), and
      lam = lam0 * max_j gam / ||(Y^T (w_j * y_j))_{-j}||_inf
    (Table 2: lam0 > 1 guarantees that no optimal column c_j is zero).
  Affinity |C| + |C|^T, entries below 1e-8 * max dropped. NJW spectral
    clustering: top-K eigenvectors of D^-1/2 A D^-1/2, rows scaled to unit
    norm, k-means with 20 restarts.
  gLRMC: for each cluster, min ||L||_* s.t. P_Omega(L) = P_Omega(X0), solved by
    inexact ALM (unscaled dual, mu0 = 1/(0.2 ||P_Omega(X0_k)||_2), rho = 1.1,
    mu_max = 1e4, tol 1e-5 on ||P_Omega(X0_k - L)||_F / ||P_Omega(X0_k)||_F,
    500 iterations). The observed entries are then restored.
  Alternate for up to 20 rounds and stop when the partition is unchanged up
  to a Hungarian-matched relabelling (Algorithm 1, line 8).

Solver changes, needed to reach N = 8-10k on one GPU:
  * PZF-EnSC is solved for all N columns at once by FISTA with per-column
    steps, a per-column gradient restart (O'Donoghue and Candes 2015), and the
    elastic-net prox (soft threshold / (1 + t (1-gam)), then diag(C) = 0).
    The masked gradient for every column is two GEMMs,
        grad = lam (Y^T (W * (Y C)) - Y^T (W * Y)),
    so an iteration costs 4 D N^2 flops. The authors used exact per-column
    LARS (SPAMS). The step 1/L_j uses L_j = 1.2 * lam * ||A_j v_j||, where
    A_j = Y^T diag(w_j) Y and v_j comes from a batched power iteration.
    Stopping: the relative gradient-mapping norm ||C+ - Z||_F / ||C+||_F falls
    below `ensc_tol`, or `ensc_maxit` is reached. Each alternation is
    warm-started from the previous C.
  * The singular-value thresholding in the ALM uses the Gram matrix of the
    smaller side plus eigh, in float64. Clusters are padded to a common width
    and batched. A padded column counts as observed with value 0, and it stays
    exactly 0, so padding does not change the result.

Protocol around it (fixed in advance, held-out truth never read):
  * no centering. The model is a union of LINEAR subspaces on the zero fill
    (tabular data already arrive z-scored, so for them zero fill = mean fill).
    One global scale s makes the mean squared column norm 1, which matches
    the unit-norm columns of the paper's synthetic data, the regime where
    mu_max = 1e4 was set. Every other step is scale-invariant. Inverted at
    the end.
  * (lam0, gam) is picked from BUDGETS[...]["configs"] by NMAE on a random 5%
    of the OBSERVED entries, which are hidden during the selection fits. The
    paper does the same in its Sec. 3: 10 random configurations from the Table
    2 grid, chosen by completion error, and it recommends held-out observed
    entries for real data. Our 10 configurations were drawn once from that grid
    with numpy default_rng(2019), with the code default (20, 0.9) listed first,
    and are hard-coded below. The model is then refit on all observed entries.
  * labels = the final spectral clustering (1..K). X_hat = the final gLRMC
    completion.
  * A row that is unobserved in every column of a cluster is completed with 0
    (the minimum nuclear norm choice). This is inherent to zero-fill gLRMC and
    becomes visible at high missing rates on images.
"""

import math
import time

import numpy as np
import torch
from scipy import linalg as sla
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from sklearn.cluster import KMeans

PAPER = ("Lane, Boger, You, Tsakiris, Haeffele, Vidal. Classifying and comparing "
         "approaches to subspace clustering with missing data. ICCV Workshops 2019.")

# Table 2 grid: lam0 in {5, 10, 20, ..., 320}, gam in {0.5, 0.6, ..., 0.9, 0.99}.
# The code default (20, 0.9) comes first, then 9 draws without replacement,
# numpy.random.default_rng(2019), uniform over the rest of the grid.
CONFIGS = ((20, 0.9), (40, 0.6), (20, 0.5), (320, 0.9), (320, 0.8),
           (20, 0.6), (10, 0.9), (20, 0.8), (160, 0.6), (5, 0.9))

_COMMON = dict(
    maxit=20,               # outer alternations (Algorithm 1)
    ensc_maxit=3000, ensc_tol=1e-4, power_its=30, lip_safety=1.2,
    lrmc_maxit=500, lrmc_tol=1e-5, lrmc_rho=1.1, lrmc_mu0_frac=0.2, lrmc_mu_max=1e4,
    affinity_thresh=1e-8, km_n_init=20, km_max_iter=1000,
    val_frac=0.05,
    time_limit_s=7.0 * 3600,   # hard wall-clock guard (CHTC jobs get < 8 h)
)
# "full": all 10 configurations, each selection fit is the full algorithm.
# "reduced" (the large datasets): the first 3 configurations; selection fits
# use 5 alternations; the final refit is the full algorithm.
BUDGETS = {
    "full":    dict(_COMMON, configs=CONFIGS, search_maxit=20),
    "reduced": dict(_COMMON, configs=CONFIGS[:3], search_maxit=5, ensc_maxit=2000),
}

_EIGH_CPU_MAX = 512        # gLRMC: Grams up to this size go to LAPACK (latency)
_DENSE_EIG_MAX = 3000      # spectral: dense eigh below this N, Lanczos above


# ---------------------------------------------------------------- PZF-EnSC --
def _column_lipschitz(Yt, Y, W, its, V0=None, gen=None):
    """lambda_max(Y^T diag(w_j) Y) for every column j (batched power iteration).

    Returns the estimate ||A_j v_j|| (a lower bound that converges to the top
    eigenvalue) and the final V, which warm-starts the next call.
    """
    N = Y.shape[1]
    if V0 is None:
        # The masked Grams A_j are all close to a multiple of Y^T Y, so start
        # every column from that matrix's top eigenvector plus a little noise.
        u = torch.ones(N, 1, dtype=Y.dtype, device=Y.device)
        for _ in range(50):
            u = Yt @ (Y @ u)
            u = u / u.norm().clamp_min(1e-30)
        noise = torch.randn(N, N, generator=gen, dtype=Y.dtype).to(Y.device)
        V = u + 0.1 * noise / math.sqrt(N)
    else:
        V = V0
    V = V / V.norm(dim=0, keepdim=True).clamp_min(1e-30)
    nrm = None
    for _ in range(its):
        U = Yt @ (W * (Y @ V))
        nrm = U.norm(dim=0)
        V = U / nrm.clamp_min(1e-30)
    return nrm, V


def pzf_ensc(Y, W, lam0, gam, *, C0=None, V0=None, maxit=3000, tol=1e-4,
             power_its=30, lip_safety=1.2, gen=None, check_every=10):
    """Weighted (projected zero-filled) elastic-net self-expression, all columns.

    Y : (D, N) float tensor with unit-norm columns (zero columns allowed).
    W : (D, N) float tensor, 1 = observed.
    Solves, for every j,  min lam/2 ||w_j*(Y c - y_j)||^2 + gam|c|_1
    + (1-gam)/2 |c|^2  s.t. c_j = 0.  Returns (C, V, info).
    """
    D, N = Y.shape
    Yt = Y.t().contiguous()
    Pm = Yt @ (W * Y)                       # P[:, j] = Y^T (w_j * y_j)
    Pm.fill_diagonal_(0.0)                  # c_jj is fixed to 0 anyway
    m = Pm.abs().amax(dim=0)
    ok = m > 0
    if not bool(ok.any()):
        return torch.zeros(N, N, dtype=Y.dtype, device=Y.device), V0, dict(
            lam=float("nan"), iters=0, res=0.0, note="no correlated columns")
    lam = float(lam0 * gam / m[ok].min())

    lj, V = _column_lipschitz(Yt, Y, W, power_its, V0, gen)
    Lj = (lip_safety * lj).clamp_min(1e-12)          # curvature / lam
    inv_l = (1.0 / Lj).unsqueeze(0)                  # step * lam
    step = inv_l / lam
    thr = step * gam
    shrink = 1.0 / (1.0 + step * (1.0 - gam))

    def prox(Vm):
        out = torch.sign(Vm) * (Vm.abs() - thr).clamp_min(0.0) * shrink
        out.fill_diagonal_(0.0)
        return out

    C = (torch.zeros(N, N, dtype=Y.dtype, device=Y.device) if C0 is None
         else C0.clone())
    C.fill_diagonal_(0.0)
    Z = C.clone()
    t = torch.ones(N, dtype=Y.dtype, device=Y.device)
    res, it, n_restart, retries = float("nan"), 0, 0, 0
    while it < maxit:
        it += 1
        G = Yt @ (W * (Y @ Z))
        G.sub_(Pm)                                   # grad / lam
        Cn = prox(Z - G * inv_l)
        Dn = Cn - C
        restart = ((Z - Cn) * Dn).sum(0) > 0          # gradient restart test
        tn = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
        beta = torch.where(restart, torch.zeros_like(t), (t - 1.0) / tn)
        tn = torch.where(restart, torch.ones_like(t), tn)
        n_restart += int(restart.sum()) if it % check_every == 0 else 0
        if it % check_every == 0 or it == maxit:
            den = float(Cn.norm())
            res = float((Cn - Z).norm()) / max(den, 1e-30)
            if not math.isfinite(res):
                # step too long for some column: halve every step, restart cold
                retries += 1
                if retries > 3:
                    raise FloatingPointError("PZF-EnSC FISTA diverged")
                inv_l, step = inv_l / 2, step / 2
                thr, shrink = step * gam, 1.0 / (1.0 + step * (1.0 - gam))
                C = torch.zeros_like(C); Z = C.clone(); t = torch.ones_like(t)
                continue
        Z = Cn + beta.unsqueeze(0) * Dn
        C, t = Cn, tn
        if it % check_every == 0 and res < tol:
            break
    R = W * (Y @ C - Y)
    obj = (0.5 * lam * float((R * R).sum()) + gam * float(C.abs().sum())
           + 0.5 * (1 - gam) * float((C * C).sum()))
    info = dict(lam=lam, iters=it, res=res, obj=obj, retries=retries,
                nnz_per_col=float((C != 0).sum()) / N,
                lip_median=float(Lj.median()))
    return C, V, info


# -------------------------------------------------------- spectral (NJW) --
def _affinity(C, thresh):
    A = C.abs()
    A = A + A.t()
    A.fill_diagonal_(0.0)
    amax = float(A.max())
    if amax <= 0:
        return sparse.csr_matrix((C.shape[0], C.shape[0]))
    A = torch.where(A >= thresh * amax, A, torch.zeros_like(A))
    idx = torch.nonzero(A, as_tuple=True)
    vals = A[idx].double().cpu().numpy()
    r, c = idx[0].cpu().numpy(), idx[1].cpu().numpy()
    N = C.shape[0]
    return sparse.csr_matrix((vals, (r, c)), shape=(N, N))


def spectral_njw(A, K, seed, n_init, max_iter):
    N = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    isolated = int((deg <= 0).sum())
    dinv = 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0))
    M = sparse.diags(dinv) @ A @ sparse.diags(dinv)
    how = "dense"
    if N <= _DENSE_EIG_MAX:
        _, U = sla.eigh(M.toarray(), subset_by_index=[N - K, N - 1])
    else:
        how = "lanczos"
        v0 = np.random.default_rng([seed, 3]).standard_normal(N)
        try:
            _, U = eigsh(M, k=K, which="LA", v0=v0, tol=1e-8,
                         ncv=min(N, max(2 * K + 1, 40)), maxiter=20 * N)
        except ArpackNoConvergence:
            how = "dense-fallback"
            _, U = sla.eigh(M.toarray(), subset_by_index=[N - K, N - 1])
    rn = np.linalg.norm(U, axis=1, keepdims=True)
    U = U / np.where(rn > 0, rn, 1.0)
    km = KMeans(n_clusters=K, n_init=n_init, max_iter=max_iter, random_state=seed)
    labels = km.fit_predict(U)
    return labels, dict(eig=how, isolated=isolated, nnz=int(A.nnz))


def _same_partition(a, b, K):
    M = np.zeros((K, K))
    np.add.at(M, (a, b), 1)
    r, c = linear_sum_assignment(-M)
    return int(M[r, c].sum()) == a.size


# -------------------------------------------------------------------- gLRMC --
def _eigh(G):
    """Batched symmetric eigh. Small matrices go to LAPACK: cuSOLVER's per-matrix
    latency dominates below a few hundred rows."""
    if G.device.type == "cuda" and G.shape[-1] <= _EIGH_CPU_MAX:
        ev, V = torch.linalg.eigh(G.cpu())
        return ev.to(G.device), V.to(G.device)
    return torch.linalg.eigh(G)


def _svt(Z, tau):
    """Singular value thresholding of every matrix in the batch Z (G, m, n)
    at level tau (G,), through the Gram matrix of the smaller side."""
    _, m, n = Z.shape
    if n <= m:
        ev, V = _eigh(Z.mT @ Z)
    else:
        ev, V = _eigh(Z @ Z.mT)
    s = ev.clamp_min(0.0).sqrt()
    r = torch.where(s > tau[:, None], (s - tau[:, None]) / s.clamp_min(1e-300),
                    torch.zeros_like(s))
    Vr = V * r[:, None, :]
    if n <= m:
        return Z @ (Vr @ V.mT)
    return (Vr @ V.mT) @ Z


def alm_mc_batch(M, Obs, *, maxit, tol, rho, mu0_frac, mu_max):
    """Inexact ALM (Lin, Chen and Ma) for min ||L||_* s.t. P_Obs(L) = P_Obs(M),
    for a batch of matrices M (G, m, n), zero off Obs. Returns (L, iters)."""
    Gn = M.shape[0]
    Oc = ~Obs
    normM = M.flatten(1).norm(dim=1)
    if M.shape[2] <= M.shape[1]:
        sig = torch.linalg.eigvalsh(M.mT @ M)[:, -1].clamp_min(0).sqrt()
    else:
        sig = torch.linalg.eigvalsh(M @ M.mT)[:, -1].clamp_min(0).sqrt()
    active = normM > 0
    mu = torch.where(active, 1.0 / (mu0_frac * sig.clamp_min(1e-300)),
                     torch.ones_like(sig))
    L = torch.zeros_like(M)
    E = torch.zeros_like(M)
    Lam = torch.zeros_like(M)
    iters = torch.zeros(Gn, dtype=torch.long, device=M.device)
    for _ in range(maxit):
        if not bool(active.any()):
            break
        mb = mu[:, None, None]
        Ln = _svt(M - E + Lam / mb, 1.0 / mu)
        En = torch.where(Oc, M - Ln + Lam / mb, torch.zeros_like(M))
        R = M - Ln - En
        Lamn = Lam + mb * R
        a = active[:, None, None]
        L = torch.where(a, Ln, L)
        E = torch.where(a, En, E)
        Lam = torch.where(a, Lamn, Lam)
        iters += active.long()
        res = R.flatten(1).norm(dim=1) / normM.clamp_min(1e-300)
        active = active & ~(res < tol)
        mu = torch.where(active, (mu * rho).clamp_max(mu_max), mu)
    return L, iters


def glrmc(X0, Om, labels, K, P, mem_bytes=1 << 30):
    """Per-cluster nuclear-norm completion. X0: (D, N) float64 zero fill, Om:
    (D, N) bool. Clusters of similar width are padded and solved as a batch."""
    D, N = X0.shape
    Y = X0.clone()
    groups = [np.flatnonzero(labels == k) for k in range(K)]
    groups = sorted([g for g in groups if g.size], key=len)
    buckets, cur = [], []
    for g in groups:
        width = len(g)
        if cur and (width > 1.25 * len(cur[0]) + 4
                    or (len(cur) + 1) * D * width * 8 * 8 > mem_bytes):
            buckets.append(cur)
            cur = []
        cur.append(g)
    if cur:
        buckets.append(cur)
    it_tot, it_max = 0, 0
    for bk in buckets:
        w = len(bk[-1])
        Gn = len(bk)
        M = torch.zeros(Gn, D, w, dtype=X0.dtype, device=X0.device)
        Ob = torch.ones(Gn, D, w, dtype=torch.bool, device=X0.device)
        for i, g in enumerate(bk):
            gi = torch.as_tensor(g, device=X0.device)
            M[i, :, :len(g)] = X0[:, gi]
            Ob[i, :, :len(g)] = Om[:, gi]
        L, its = alm_mc_batch(M, Ob, maxit=P["lrmc_maxit"], tol=P["lrmc_tol"],
                              rho=P["lrmc_rho"], mu0_frac=P["lrmc_mu0_frac"],
                              mu_max=P["lrmc_mu_max"])
        for i, g in enumerate(bk):
            gi = torch.as_tensor(g, device=X0.device)
            Y[:, gi] = torch.where(Om[:, gi], X0[:, gi], L[i, :, :len(g)])
        it_tot += int(its.sum())
        it_max = max(it_max, int(its.max()))
        del M, Ob, L
    return Y, dict(buckets=len(buckets), alm_iters_mean=it_tot / max(len(groups), 1),
                   alm_iters_max=it_max)


# ---------------------------------------------------------------- driver --
def _normalize_cols(Y):
    n = Y.norm(dim=0, keepdim=True)
    return Y / torch.where(n > 0, n, torch.ones_like(n))


def alt_pzf_ensc_glrmc(X0, Om, K, lam0, gam, P, *, maxit, seed, deadline=None):
    """Algorithm 1 with PZF-EnSC and gLRMC. X0: (D, N) float64 zero fill (device),
    Om: (D, N) bool. Returns (Y, labels 0..K-1, info)."""
    W = Om.to(torch.float32)
    gen = torch.Generator().manual_seed(int(np.random.default_rng([seed, 7]).integers(2**62)))
    hist = []

    def sc_step(Y, C0, V0):
        t0 = time.perf_counter()
        Yn = _normalize_cols(Y).to(torch.float32)
        C, V, ei = pzf_ensc(Yn, W, lam0, gam, C0=C0, V0=V0, maxit=P["ensc_maxit"],
                            tol=P["ensc_tol"], power_its=P["power_its"],
                            lip_safety=P["lip_safety"], gen=gen)
        t1 = time.perf_counter()
        A = _affinity(C, P["affinity_thresh"])
        lab, si = spectral_njw(A, K, seed, P["km_n_init"], P["km_max_iter"])
        t2 = time.perf_counter()
        return lab, C, V, dict(ensc=ei, spectral=si, ensc_s=t1 - t0, spectral_s=t2 - t1)

    Y = X0.clone()
    labels, C, V, st = sc_step(Y, None, None)
    hist.append(dict(k=0, **st))
    converged, stop = False, "maxit"
    for k in range(1, maxit + 1):
        if deadline is not None and time.perf_counter() > deadline:
            stop = "time_limit"
            break
        t0 = time.perf_counter()
        Y, gi = glrmc(X0, Om, labels, K, P)
        tg = time.perf_counter() - t0
        new, C, V, st = sc_step(Y, C, V)
        same = _same_partition(labels, new, K)
        hist.append(dict(k=k, glrmc=gi, glrmc_s=tg, unchanged=same, **st))
        labels = new
        if same:
            converged, stop = True, "labels_unchanged"
            break
    if len(hist) == 1:
        # maxit = 0 or out of time before the first completion: still complete
        Y, gi = glrmc(X0, Om, labels, K, P)
    del C, V
    return Y, labels, dict(alternations=len(hist) - 1, converged=converged,
                           stop=stop, history=hist)


def _fit(Xt, obs_t, K, lam0, gam, P, maxit, seed, dev, scale, deadline):
    """Xt: (D, N) float64 numpy with NaN, obs_t: (D, N) bool. Returns X_hat (D, N)."""
    X0 = torch.as_tensor(np.where(obs_t, Xt / scale, 0.0), dtype=torch.float64, device=dev)
    Om = torch.as_tensor(obs_t, device=dev)
    t0 = time.perf_counter()
    Y, labels, info = alt_pzf_ensc_glrmc(X0, Om, K, lam0, gam, P, maxit=maxit,
                                         seed=seed, deadline=deadline)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    info["seconds"] = time.perf_counter() - t0
    Xh = Y.cpu().numpy() * scale
    del X0, Om, Y
    return np.where(obs_t, Xt, Xh), labels, info


def _brief(info):
    h = info["history"]
    return dict(alternations=info["alternations"], stop=info["stop"],
                seconds=round(info["seconds"], 2),
                ensc_iters=[x["ensc"]["iters"] for x in h],
                ensc_res_last=h[-1]["ensc"].get("res"),
                nnz_per_col_last=h[-1]["ensc"].get("nnz_per_col"))


def impute(X, *, seed, K, profile="full", device="cuda"):
    P = BUDGETS[profile]
    dev = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")
    t_start = time.perf_counter()
    deadline = t_start + P["time_limit_s"]
    shape = X.shape
    Xf = np.asarray(X, dtype=np.float64).reshape(shape[0], -1)
    obs = ~np.isnan(Xf)
    B, F = Xf.shape
    K = int(K)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))

    # One global scale: mean squared column norm 1 (estimated from observed entries).
    ms = float((Xf[obs] ** 2).mean()) if obs.any() else 1.0
    scale = math.sqrt(ms * F) if ms > 0 and math.isfinite(ms) else 1.0

    # Paper layout: D x N, samples as columns.
    Xt = Xf.T.copy()
    obs_t = obs.T.copy()

    old_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False     # full fp32 GEMMs in FISTA
    torch.backends.cudnn.allow_tf32 = False
    try:
        configs = list(P["configs"])
        val, val_fit, skipped = {}, {}, []
        if len(configs) > 1:
            # Selection on a random val_frac of the OBSERVED entries only.
            rng = np.random.default_rng([seed, 5])
            oi = np.flatnonzero(obs_t)
            hold = rng.choice(oi, size=max(1, int(round(P["val_frac"] * oi.size))),
                              replace=False)
            fit_obs = obs_t.copy()
            fit_obs.reshape(-1)[hold] = False
            sd = np.nanstd(np.where(fit_obs, Xt, np.nan), axis=1)
            sd = np.where(np.isfinite(sd) & (sd >= 1e-8), sd, 1.0)
            hr, hc = np.unravel_index(hold, Xt.shape)
            search_deadline = t_start + 0.6 * P["time_limit_s"]
            for cfg in configs:
                if val and time.perf_counter() > search_deadline:
                    skipped.append(cfg)
                    continue
                lam0, gam = cfg
                try:
                    Xh, _, inf = _fit(Xt, fit_obs, K, lam0, gam, P, P["search_maxit"],
                                      seed, dev, scale, search_deadline)
                    err = Xh[hr, hc] - Xt[hr, hc]
                    val[cfg] = float(np.mean(np.abs(err) / sd[hr]))
                    val_fit[cfg] = _brief(inf)
                except (FloatingPointError, np.linalg.LinAlgError, RuntimeError) as e:
                    val[cfg] = float("nan")
                    val_fit[cfg] = dict(error=f"{type(e).__name__}: {e}"[:200])
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            # ties -> earlier in the fixed list; a failed fit (nan) is never picked
            order = {c: i for i, c in enumerate(configs)}
            best = min(val, key=lambda c: (not np.isfinite(val[c]), val[c], order[c]))
        else:
            best = configs[0]

        lam0, gam = best
        X_hat, labels, info = _fit(Xt, obs_t, K, lam0, gam, P, P["maxit"], seed, dev,
                                   scale, deadline)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tf32

    X_hat = X_hat.T
    budget = {k: v for k, v in P.items() if k != "configs"}
    out_info = dict(
        method="Alt PZF-EnSC+gLRMC (independent implementation from the paper)",
        paper=PAPER, profile=profile, budget=budget,
        configs=[list(c) for c in P["configs"]],
        val_nmae={f"{c[0]},{c[1]}": v for c, v in val.items()},
        val_fits={f"{c[0]},{c[1]}": v for c, v in val_fit.items()},
        skipped_configs=[list(c) for c in skipped],
        lam0=best[0], gamma=best[1], scale=scale,
        final=dict(_brief(info), converged=info["converged"],
                   history=[{k: v for k, v in h.items()} for h in info["history"]]),
        solver="PZF-EnSC by batched FISTA (per-column steps + restart), "
               "gLRMC by batched inexact ALM with Gram+eigh SVT",
        device=str(dev), total_seconds=time.perf_counter() - t_start)
    return dict(X_hat=X_hat.reshape(shape), labels=np.asarray(labels, dtype=np.int64) + 1,
                info=out_info)
