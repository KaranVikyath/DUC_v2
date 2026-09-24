"""
k-FSC on incomplete data: union-of-subspaces clustering and completion in one
factorization.

Paper: Jicong Fan, "Large-Scale Subspace Clustering via k-Factorization",
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data
Mining (KDD 2021), pp. 342-352, doi:10.1145/3447548.3467267 (arXiv:2012.04345).

This is an independent implementation from the paper, written in PyTorch from
its equations and algorithms (Sec. 2-4, Algorithms 1, 3 and 4). The authors'
MATLAB repository has no license, so none of its code is used, translated,
fetched or imported here.

Model: the paper's eq. (19), Sec. 4.4, with samples as the COLUMNS of X (m x n):

    min_{D in S_D, C}  1/2 ||M o (X - D C)||_F^2 + lam * sum_j ||C^(j)||_{2,1}

D = [D^(1), ..., D^(k)] has d columns per block, and every column lies in the
unit ball (S_D). C = [C^(1); ...; C^(k)] is split into matching row blocks.
||.||_{2,1} sums the Euclidean norms of the columns of each block, so a sample
is pushed to use as few blocks, i.e. subspaces, as possible. M is 1 on the
observed entries and 0 on the missing ones.

Sec. 4.4 leaves the solver as "adapted from Algorithm 4". The adaptation here
replaces the residual X - D C with M o (X - D C) in every gradient:
  * Normalization (Alg. 4, line 1): each sample is scaled to unit l2 norm over
    its observed entries (s_i = ||m_i o x_i||). Missing entries are zero.
  * Initialization (Sec. 3.1): k-means under the cosine distance, 10 replicates
    as in Sec. 6.2, with k-means++ seeding, on the zero-filled unit samples.
    D0^(j) holds the d leading left singular vectors of the d samples closest
    to centre j, and C0 = (D0'D0 + lam_hat I)^-1 D0' X with lam_hat = 1e-5.
  * C step (Algorithm 1): block Gauss-Seidel proximal gradient with
    extrapolation. eta_j = delta * sqrt(tau_{j,t-2} / tau_{j,t-1}), with no
    extrapolation for t <= 2 and delta = 0.95. tau_j = gamma ||D^(j)||_2^2 with
    gamma = 1. The gradient is G^(j) = -D^(j)'(M o (X - Xhat)), and the prox is
    the column-wise group soft threshold Theta_{lam/tau} of eq. (12). Xhat is
    refreshed after every block (line 14). The mask can only lower the
    Lipschitz constant, so tau_j is still a valid step.
  * D step (Algorithm 3): theta = 5 projected gradient steps
    D <- P_Pi(D - G / kappa), with G = -(M o (X - D C)) C' and
    kappa = ||C C'||_2. P_Pi projects each column onto the unit ball (eq. 15).
    The complete-data shortcut (A = X C', B = C C') no longer holds under the
    mask, so the masked residual is recomputed at every inner step.
  * Stopping (Alg. 4, line 8): max(||C_t - C_{t-1}||_F / ||C_{t-1}||_F,
    ||D_t - D_{t-1}||_F / ||D_{t-1}||_F) <= eps = 1e-4, or t = T.
  * Clusters (Alg. 4, line 9: "by (3) or (8)"): a sample with exactly one
    nonzero coefficient group takes that group, rule (3). Any other sample,
    with zero groups or with two or more, takes rule (8), the block with the
    least reconstruction error, computed over its observed entries:
    c = (D^(j)' diag(m_i) D^(j) + lam' I)^-1 D^(j)'(m_i o x_i), lam' = 1e-5,
    and err_j = ||m_i o (x_i - D^(j) c)||.
  * Completion: the paper says "the missing entries can be obtained from DC".
    So xhat_i = s_i (D C)_:i on the missing entries, and the observed entries
    are returned exactly as given. A sample whose C column is entirely zero
    gets s_i D^(j) c from its rule-(8) block instead.

Our choices, which the paper does not fix (all made in advance, none tuned on
held-out truth):
  1. d and lam, the per-dataset hand settings in Sec. 6.2, are chosen on a
     random 5% of the OBSERVED entries, hidden during the selection fits, by
     NMAE on those entries. Stage 1 tries every d in d_grid at lam_ref = 0.2.
     Stage 2 tries each lam in lam_grid, plus the label-free eq. (7) value
     computed from D0, at the chosen d. The final model is then refit on all
     observed entries. d_grid keeps only d <= m/2 and d <= n/k, since a
     subspace per cluster needs fewer dimensions than members.
  2. A lambda guard, motivated by Theorem 2 (lam must not exceed any sample's
     best-block projection, or that sample loses every group). If at least
     10% of samples end a C step with an all-zero coefficient column, lam is
     halved and the fit restarts from (D0, C0) with extrapolation reset. Each
     halving is recorded in info.
  3. kappa = ||C C'||_2 comes from a warm-started power iteration (10 steps
     per outer iteration, 30 on a cold start), not an exact eigendecomposition.
     Each tau_j is exact: one batched eigvalsh of the k d x d Gram matrices.
  4. Rows with no observed entry carry no information. They are left out of
     the fit, filled with the feature means of the observed entries, and put
     in the largest cluster. info counts them.
  5. The main loop runs in float32, fast on the L40S. C0, eq. (7) and rule (8)
     run in float64.

Complexity per outer iteration, with m = features, n = samples and k*d
dictionary atoms: the C step costs about 6 m n k d flops plus k masked
residual updates of size m x n. The D step costs about 4 theta m n k d =
20 m n k d. The total is O(k d m n) time and O(m n + k m d + k d n) memory,
with no n x n matrix, as in the paper's Sec. 3.4. Rule (8) costs
O(k d^2 m n_amb) once, on the ambiguous samples only.
"""

import math
import time

import numpy as np
import torch

PAPER = ("J. Fan, Large-Scale Subspace Clustering via k-Factorization, "
         "KDD 2021, doi:10.1145/3447548.3467267")

# Algorithm 4's defaults are delta = 0.95, gamma = 1, theta = 5, eps = 1e-4,
# and lam_hat = lam' = 1e-5 (Sec. 3.1, eq. 8). Its default T is 200, for
# complete data. The masked problem converges more slowly, so "full" allows up
# to 500 iterations, and the eps test usually stops it earlier. kfsc is a cheap
# method, so the protocol runs "full" everywhere. "reduced" exists for the
# contract: the paper's T = 200 and a smaller selection grid.
_COMMON = dict(T=500, delta=0.95, gamma=1.0, theta=5, eps=1e-4,
               lam_hat=1e-5, lam_prime=1e-5,
               init="kmeans", km_reps=10, km_iters=100,
               d_grid=(5, 10, 20, 30), lam_grid=(0.05, 0.1, 0.2, 0.4),
               lam_ref=0.2, use_eq7=True,
               val_frac=0.05, zero_frac=0.1, max_halvings=8,
               power_iters=10, dtype="float32", chunk=2048)
BUDGETS = {
    "full":    dict(_COMMON),
    "reduced": dict(_COMMON, T=200, d_grid=(10, 20), lam_grid=(0.1, 0.2, 0.4),
                    use_eq7=False),
}

_TINY = 1e-12


# ------------------------------------------------------------ initialization --
def _cosine_kmeans(U, k, reps, iters, gen):
    """k-means with the cosine distance 1 - u'c on the unit rows of U (n x m).

    Centres are the renormalized means of their members, and seeding is
    k-means++ under the same distance. The replicate with the least total
    distance is kept. Returns (cosine similarities to the kept centres, n x k;
    labels; total distance).
    """
    n = U.shape[0]
    best = None
    for _ in range(reps):
        first = int(torch.randint(n, (1,), generator=gen))
        idx = [first]
        dmin = (1.0 - U @ U[first]).clamp_min(0)
        for _ in range(1, k):
            p = dmin.double().cpu()
            tot = float(p.sum())
            nxt = (int(torch.randint(n, (1,), generator=gen)) if tot <= 0
                   else int(torch.multinomial(p / tot, 1, generator=gen)))
            idx.append(nxt)
            dmin = torch.minimum(dmin, (1.0 - U @ U[nxt]).clamp_min(0))
        cen = U[idx].clone()
        prev = None
        for _ in range(iters):
            S = U @ cen.T
            sim, lab = S.max(1)
            if prev is not None and torch.equal(lab, prev):
                break
            prev = lab
            cnt = torch.bincount(lab, minlength=k)
            sums = torch.zeros_like(cen).index_add_(0, lab, U)
            empty = (cnt == 0).nonzero().flatten().tolist()
            if empty:
                # an empty cluster restarts at the worst-fitted samples
                far = torch.argsort(sim)[:len(empty)].tolist()
                for e, f in zip(empty, far):
                    sums[e] = U[f]
            cen = sums / sums.norm(dim=1, keepdim=True).clamp_min(_TINY)
        S = U @ cen.T
        sim, lab = S.max(1)
        cost = float((1.0 - sim).sum())
        if best is None or cost < best[2]:
            best = (S, lab, cost)
    return best


def _dict_from_kmeans(X, S, k, d):
    """Sec. 3.1: D0^(j) = left singular vectors of the d samples closest to centre j."""
    idx = torch.topk(S, d, dim=0).indices              # (d, k): most similar per centre
    blocks = X[:, idx.T.reshape(-1)].reshape(X.shape[0], k, d).permute(1, 0, 2)
    Ub = torch.linalg.svd(blocks.double(), full_matrices=False)[0]   # (k, m, d)
    return Ub.permute(1, 0, 2).reshape(X.shape[0], k * d).to(X.dtype)


def _ridge_codes(D, X, lam_hat):
    """C0 = (D'D + lam_hat I)^-1 D'X, in float64."""
    D64 = D.double()
    G = D64.T @ D64
    G.diagonal().add_(lam_hat)
    L, bad = torch.linalg.cholesky_ex(G)
    rhs = D64.T @ X.double()
    C = torch.cholesky_solve(rhs, L) if int(bad) == 0 else torch.linalg.solve(G, rhs)
    return C.to(X.dtype)


def _lambda_eq7(D, X, k, d):
    """Eq. (7): midpoint of max_i 2nd-best and min_i best block projection norm."""
    Pn = (D.double().T @ X.double()).view(k, d, -1).norm(dim=1)   # (k, n)
    top = torch.topk(Pn, 2, dim=0).values
    return float(0.5 * (top[1].max() + top[0].min()))


# ------------------------------------------------------------------- solver --
def _sq_spec(C, w, iters, gen):
    """||C C'||_2 = ||C||_2^2 by (warm-started) power iteration on C C'."""
    if w is None:
        w = torch.randn(C.shape[0], generator=gen, dtype=torch.float64).to(C)
        iters = 3 * iters
    for _ in range(iters):
        w = C @ (C.T @ w)
        w = w / w.norm().clamp_min(_TINY)
    z = C.T @ w
    return float(z @ z), w


def _objective(X, M, D, C, k, d, lam):
    R = X - M * (D @ C)
    return 0.5 * float((R * R).sum()) + lam * float(C.view(k, d, -1).norm(dim=1).sum())


def kfsc_masked(X, M, D0, C0, k, d, lam, P, gen):
    """Alternating Algorithm 1 (C) / Algorithm 3 (D) on the masked model (19).

    X : (m, n) unit-norm samples, zero at missing entries; M : (m, n) 0/1 mask.
    Returns (D, C, lam_used, info).
    """
    m, n = X.shape
    delta, gamma, theta = P["delta"], P["gamma"], P["theta"]
    D, C = D0.clone(), C0.clone()
    Delta = torch.zeros_like(C)
    tau_prev, w = None, None
    t, halvings, stop = 0, 0, "T"
    dC = dD = float("nan")
    t0 = time.perf_counter()
    while True:
        t += 1
        # ---- C step: Algorithm 1 with the masked residual --------------------
        Db = D.view(m, k, d).permute(1, 0, 2)                  # (k, m, d)
        tau = gamma * torch.linalg.eigvalsh((Db.transpose(1, 2) @ Db).double())[:, -1]
        tau = tau.clamp_min(1e-8)
        if t > 2 and tau_prev is not None:
            eta = (delta * torch.sqrt(tau_prev / tau)).to(C.dtype)
            Chat = C - eta.repeat_interleave(d)[:, None] * Delta
        else:
            Chat = C
        R = X - M * (D @ Chat)                                  # M o (X - Xhat)
        Cn = torch.empty_like(C)
        tau_l = tau.tolist()
        for j in range(k):
            sl = slice(j * d, (j + 1) * d)
            Dj = D[:, sl]
            V = Chat[sl] + (Dj.T @ R) / tau_l[j]                # Chat - G / tau
            nv = V.norm(dim=0)
            Cj = V * ((nv - lam / tau_l[j]).clamp_min(0) / nv.clamp_min(_TINY))
            Cn[sl] = Cj
            R.addcmul_(M, Dj @ (Cj - Chat[sl]), value=-1.0)     # Xhat += Dj (Cj - Chat_j)
        zero_cols = int((Cn.abs().amax(0) == 0).sum())
        if zero_cols >= P["zero_frac"] * n and halvings < P["max_halvings"]:
            lam, halvings = lam / 2.0, halvings + 1             # our guard (Theorem 2)
            D, C = D0.clone(), C0.clone()
            Delta.zero_()
            tau_prev, w, t = None, None, 0
            continue
        # ---- D step: Algorithm 3 with the masked residual --------------------
        kappa, w = _sq_spec(Cn, w, P["power_iters"], gen)
        Dn = D
        if kappa > _TINY:
            CnT = Cn.T
            for _ in range(theta):
                Rd = X - M * (Dn @ Cn)
                Dn = Dn + (Rd @ CnT) / kappa                    # D - G / kappa
                Dn = Dn / Dn.norm(dim=0, keepdim=True).clamp_min(1.0)
        if not (torch.isfinite(Cn).all() and torch.isfinite(Dn).all()):
            stop = "nonfinite"                                  # keep the last finite iterate
            break
        dC = float((Cn - C).norm() / C.norm().clamp_min(_TINY))
        dD = float((Dn - D).norm() / D.norm().clamp_min(_TINY))
        Delta = C - Cn                                          # Alg. 1, line 16
        tau_prev = tau
        C, D = Cn, Dn
        if max(dC, dD) <= P["eps"]:
            stop = "eps"
            break
        if t >= P["T"]:
            break
    zero_cols = int((C.abs().amax(0) == 0).sum())
    info = dict(iters=t, stop=stop, last_dC=dC, last_dD=dD, lam_halvings=halvings,
                zero_cols=zero_cols, objective=_objective(X, M, D, C, k, d, lam),
                solver_seconds=time.perf_counter() - t0)
    return D, C, lam, info


# --------------------------------------------------------- labels/completion --
def _assign_and_complete(X, M, D, C, k, d, lam_prime, chunk):
    """Rule (3) where exactly one group is nonzero, masked rule (8) otherwise.

    Returns (labels 0..k-1, completion D C in the normalized space, counts).
    """
    m, n = X.shape
    blk = C.view(k, d, n).norm(dim=1)                           # (k, n)
    nnz = (blk > 0).sum(0)
    lab = blk.argmax(0)
    Xh = D @ C
    amb = (nnz != 1).nonzero().flatten()
    zero = nnz == 0
    if amb.numel():
        D64 = D.double()
        eye = torch.eye(d, dtype=torch.float64, device=X.device)
        for s in range(0, amb.numel(), chunk):
            idx = amb[s:s + chunk]
            xs, ms = X[:, idx].double(), M[:, idx].double()
            best_err = torch.full((idx.numel(),), float("inf"), dtype=torch.float64,
                                  device=X.device)
            best_j = torch.zeros(idx.numel(), dtype=torch.long, device=X.device)
            best_rec = torch.zeros_like(xs)
            for j in range(k):
                Dj = D64[:, j * d:(j + 1) * d]
                outer = (Dj[:, :, None] * Dj[:, None, :]).reshape(m, d * d)
                G = (ms.T @ outer).view(-1, d, d) + lam_prime * eye   # D_j' diag(m_i) D_j
                b = (xs.T @ Dj).unsqueeze(-1)                   # D_j'(m o x): x is 0 off-mask
                L, bad = torch.linalg.cholesky_ex(G)
                c = (torch.cholesky_solve(b, L) if int(bad.max()) == 0
                     else torch.linalg.solve(G, b)).squeeze(-1)
                rec = Dj @ c.T                                  # (m, nc)
                err = ((xs - ms * rec) ** 2).sum(0)
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_j = torch.where(better, torch.full_like(best_j, j), best_j)
                best_rec[:, better] = rec[:, better]
            lab[idx] = best_j
            zi = zero[idx]
            if bool(zi.any()):
                Xh[:, idx[zi]] = best_rec[:, zi].to(Xh.dtype)
    counts = dict(rule3=int((nnz == 1).sum()), rule8_multi=int((nnz > 1).sum()),
                  rule8_zero=int(zero.sum()),
                  mean_nonzero_groups=float(nnz.double().mean()))
    return lab, Xh, counts


# ------------------------------------------------------------------ adapter --
class _Prepared:
    """Normalized (m, n) problem for one observation mask, plus its k-means."""

    def __init__(self, Xf, obs, k, P, seed, dev, dtype):
        self.obs = obs
        Xz = np.where(obs, Xf, 0.0)
        s = np.sqrt((Xz * Xz).sum(1))
        self.keep = obs.any(1) & (s > _TINY)
        self.s = s[self.keep]
        Xn = Xz[self.keep] / self.s[:, None]
        self.X = torch.as_tensor(Xn.T.copy(), dtype=dtype, device=dev)
        self.M = torch.as_tensor(obs[self.keep].T.astype(np.float64), dtype=dtype, device=dev)
        cnt = obs.sum(0)
        self.mu = np.zeros(Xf.shape[1])
        np.divide(Xz.sum(0), cnt, out=self.mu, where=cnt > 0)
        self.k, self.P, self.seed = k, P, seed
        self.km_cost, self.S = None, None
        if P["init"] == "kmeans":
            gen = torch.Generator().manual_seed(int(seed) * 1000003 + 17)
            t0 = time.perf_counter()
            self.S, _, self.km_cost = _cosine_kmeans(self.X.T.contiguous(), k,
                                                     P["km_reps"], P["km_iters"], gen)
            self.km_seconds = time.perf_counter() - t0

    def init(self, d, gen):
        if self.P["init"] == "kmeans":
            D0 = _dict_from_kmeans(self.X, self.S, self.k, d)
        else:                                                   # Sec. 3.1, random N(0, 1)
            D0 = torch.randn(self.X.shape[0], self.k * d, generator=gen,
                             dtype=torch.float64).to(self.X)
            D0 = D0 / D0.norm(dim=0, keepdim=True).clamp_min(1.0)
        return D0, _ridge_codes(D0, self.X, self.P["lam_hat"])


def _fit(prep, Xf, d, lam, lam_is_eq7=False):
    """One k-FSC fit. Returns X_hat (B, F) in input units, labels (1..K), info."""
    P, k = prep.P, prep.k
    gen = torch.Generator().manual_seed(int(prep.seed) * 7919 + d)
    D0, C0 = prep.init(d, gen)
    lam7 = _lambda_eq7(D0, prep.X, k, d)
    lam0 = lam7 if lam_is_eq7 else lam
    D, C, lam_used, sinfo = kfsc_masked(prep.X, prep.M, D0, C0, k, d, lam0, P, gen)
    lab, Xh, counts = _assign_and_complete(prep.X, prep.M, D, C, k, d,
                                           P["lam_prime"], P["chunk"])
    B, F = Xf.shape
    X_hat = np.broadcast_to(prep.mu, (B, F)).copy()             # empty rows: feature means
    X_hat[prep.keep] = Xh.double().cpu().numpy().T * prep.s[:, None]
    X_hat = np.where(prep.obs, Xf, X_hat)
    lab_np = lab.cpu().numpy()
    labels = np.empty(B, dtype=int)
    labels[prep.keep] = lab_np + 1
    if (~prep.keep).any():
        labels[~prep.keep] = int(np.bincount(lab_np, minlength=k).argmax()) + 1
    info = dict(d=d, lam_start=lam0, lam_used=lam_used, lam_eq7=lam7, **sinfo, **counts,
                clusters_used=int(np.unique(lab_np).size))
    return X_hat, labels, info


def impute(X, *, seed, K, profile="full", device="cuda"):
    if profile not in BUDGETS:
        raise ValueError(f"kfsc: unknown profile {profile!r}; have {sorted(BUDGETS)}")
    P = BUDGETS[profile]
    dev = torch.device(device if (not str(device).startswith("cuda")
                                  or torch.cuda.is_available()) else "cpu")
    dtype = getattr(torch, P["dtype"])
    shape = X.shape
    Xf = np.asarray(X, dtype=np.float64).reshape(shape[0], -1)
    obs = np.isfinite(Xf)
    B, F = Xf.shape
    k = int(max(1, min(int(K), B)))
    t_all = time.perf_counter()

    n_rows = int(obs.any(1).sum())
    d_grid = [d for d in P["d_grid"] if 2 * d <= F and d * k <= n_rows]
    if not d_grid:
        d_grid = [max(1, min(F // 2, n_rows // k))]

    # ---- selection of (d, lam) on a random 5% of the OBSERVED entries --------
    rng = np.random.default_rng([int(seed), 0x6B465343])
    oi = np.flatnonzero(obs)
    hold = rng.choice(oi, size=max(1, int(round(P["val_frac"] * oi.size))), replace=False)
    fit_obs = obs.copy()
    fit_obs.reshape(-1)[hold] = False
    sd = np.nanstd(np.where(fit_obs, Xf, np.nan), axis=0)
    sd = np.where(np.isfinite(sd) & (sd >= 1e-8), sd, 1.0)
    hr, hc = np.unravel_index(hold, Xf.shape)
    prep_v = _Prepared(Xf, fit_obs, k, P, seed, dev, dtype)

    trials = []

    def trial(d, lam, eq7=False):
        t0 = time.perf_counter()
        Xh, _, inf = _fit(prep_v, Xf, d, lam, lam_is_eq7=eq7)
        score = float(np.mean(np.abs(Xh[hr, hc] - Xf[hr, hc]) / sd[hc]))
        trials.append(dict(d=d, lam="eq7" if eq7 else lam, lam_start=inf["lam_start"],
                           lam_used=inf["lam_used"], val_nmae=score, iters=inf["iters"],
                           stop=inf["stop"], halvings=inf["lam_halvings"],
                           seconds=time.perf_counter() - t0))
        return score

    def key(tr):   # a nonfinite score never wins; ties -> smaller d, larger lam
        return (not np.isfinite(tr["val_nmae"]), tr["val_nmae"], tr["d"], -tr["lam_start"])

    for d in d_grid:                                            # stage 1: d at lam_ref
        trial(d, P["lam_ref"])
    d_best = min(trials, key=key)["d"]
    for lam in P["lam_grid"]:                                   # stage 2: lam at d_best
        if lam != P["lam_ref"]:
            trial(d_best, lam)
    if P["use_eq7"]:
        trial(d_best, None, eq7=True)
    best = min((tr for tr in trials if tr["d"] == d_best), key=key)
    del prep_v
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ---- final fit on all observed entries ----------------------------------
    prep = _Prepared(Xf, obs, k, P, seed, dev, dtype)
    X_hat, labels, final = _fit(prep, Xf, d_best, best["lam_start"])
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    final["kmeans_cost"] = prep.km_cost
    final["kmeans_seconds"] = getattr(prep, "km_seconds", None)

    info = dict(method="k-FSC, masked model (19), independent implementation from the paper",
                paper=PAPER, profile=profile, device=str(dev), dtype=P["dtype"],
                budget={a: (list(v) if isinstance(v, tuple) else v) for a, v in P.items()},
                K=k, d_grid_used=d_grid, d_best=d_best, lam_best=best["lam"],
                lam_best_value=best["lam_start"], val_trials=trials,
                empty_rows=int(B - int(prep.keep.sum())),
                final=final, total_seconds=time.perf_counter() - t_all,
                completion="s_i * (D C)_:i on missing entries; observed returned as given",
                labels_rule="(3) if exactly one nonzero group else masked (8)")
    return dict(X_hat=X_hat.reshape(shape), labels=labels, info=info)
