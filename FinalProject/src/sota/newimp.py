"""
NewImp / KnewImp (Chen et al., "Rethinking the Diffusion Models for Numerical
Tabular Data Imputation from the Perspective of Wasserstein Gradient Flow",
NeurIPS 2024): negative-entropy-regularised Wasserstein gradient flow, i.e. an
SVGD-style particle flow on the missing entries driven by a DSM-trained energy
network.

Official code: https://github.com/JustusvLiebig/NewImp (Apache-2.0), fetched
into SOTA_ROOT/NewImp at the pinned COMMIT below and imported from there. We
use its NeuralGradFlowImputer class, score network (ToyMLP + Energy), DSM
Trainer and train_step unchanged; the following are replaced at RUNTIME from
here (their files are never edited):

  * import-time stubs, installed only while their modules import and only when
    the real package is missing: `hyperimpute` (utils/utils.py imports it for
    data amputation we never call), `ot` (POT; used only for a logging
    Wasserstein distance when X_true is given, which we never pass) and
    `torch.utils.tensorboard` (imported by score_tuple/score_trainer, never
    used by the imputer).
  * xRBF kernel (passed through the `kernel_func` argument): the original loops
    over features, keeps the whole n x n kernel, and its median-bandwidth path
    uses `math` without importing it. The replacement is block-wise over rows
    (K is never materialised beyond `_BLOCK_ELEMS` floats), so n = 85k fits.
    Same maths: K = exp(-|x_i-x_j|^2 / (2 s^2)), grad term (x_i sum_j K_ij -
    sum_j K_ij x_j) / s^2.
  * knew_imp_sampling: same Adam-on-the-missing-entries update
    -(K @ score - lambda * dK) / n with the observed entries' gradient masked
    to zero, but K@score, K@X and the row sums of K come from one block-wise
    pass (patched method on the instance).
  * fit_transform: the original calls torch.tensor(X_true) (crashes for
    X_true=None), runs float64 (exper_wgf sets DoubleTensor default), and
    trains the score net through a DataLoader of per-row __getitem__ on a
    batch of size n (one full-batch step per epoch, n Python indexing calls).
    The replacement keeps the loop exactly -- mean + N(0, 0.1^2) init, `niter`
    outer rounds of {fresh Trainer (Adam 1e-3, DSM sigma 0.1) for
    `score_net_epoch` full-batch steps on the current fill, then
    `sampling_step` flow steps with one Adam optimiser shared across rounds} --
    but in float32, with one full-batch train_step per epoch on the tensor
    itself (the row order of a full batch does not change the DSM loss), and
    on the requested device.

Data handling: X is flattened to (B, F) and z-scored per feature from observed
entries (NewImp is designed for StandardScaler'd data; images arrive in raw
pixel units), imputed, and mapped back. Observed entries are returned as given.

Bandwidth. The repo default (0.5) was tuned for low-dimensional UCI tables;
for d >= ~48 z-scored features every off-diagonal kernel entry underflows and
the flow degenerates to plain score ascent. We therefore take the SVGD median
heuristic that the repo's own xRBF implements for bandwidth < 0,
s0^2 = median |x_i - x_j|^2 / (2 log(n + 1)), computed once on (a subsample
of) the mean-imputed data, and choose a multiplier from BUDGETS[...]["bw_mult"]
by MAE on a random 5% of the OBSERVED entries hidden during a selection fit.
The final fit uses all observed entries. Every other hyper-parameter is the
repo default of exper_wgf.py (entropy_reg 10, ode_step 0.1, score_net_epoch
200, iter_time 2, score_lr 1e-3, MLP [128, 128]) and the class default
sampling_step 500; images use the [512, 512] MLP (research notes).

No clustering: labels=None.
"""

import contextlib
import math
import os
import random
import sys
import time
import types

import numpy as np
import torch

from . import SOTA_ROOT

REPO_DIR = os.path.join(SOTA_ROOT, "NewImp")
REPO_URL = "https://github.com/JustusvLiebig/NewImp"
COMMIT = "915acfc7d8854ea3a7baf8e4eeb90d85be10f92c"

# Fixed in advance. "full" = repo defaults (exper_wgf.py + class defaults).
# "reduced" halves the flow and the score training and skips the bandwidth
# selection (NewImp is not in sota.EXPENSIVE, so the job grid never uses it).
BUDGETS = {
    "full": dict(
        entropy_reg=10.0,        # exper_wgf --entropy_reg
        lr=0.1,                  # exper_wgf --ode_step (Adam on the imputations)
        # The paper's 0.1 is for tables of <= ~50 columns. On ORL (1024 pixels)
        # it drove the held-out OBSERVED MAE from 0.81 to 1.50 at every
        # bandwidth, so the step size joins the label-free selection grid.
        lr_grid=(0.1, 0.03, 0.01),
        niter=2,                 # exper_wgf --iter_time
        sampling_step=500,       # NeuralGradFlowImputer default
        score_net_epoch=200,     # exper_wgf --score_net_epoch
        score_net_lr=1e-3,       # exper_wgf --score_lr (Trainer default)
        noise=0.1,               # init noise, class default
        mlp_hidden=[128, 128],   # exper_wgf --mlp_hidden
        mlp_hidden_image=[512, 512],
        bw_mult=(0.25, 0.5, 1.0),  # x median-heuristic bandwidth, picked on 5% observed
        early_stop=True,           # flow step count picked on the same 5% observed
        eval_every=10,
        val_frac=0.05,
        bw_subsample=4000,
    ),
    "reduced": dict(
        entropy_reg=10.0,
        lr=0.1,
        niter=2,
        sampling_step=250,
        score_net_epoch=100,
        score_net_lr=1e-3,
        noise=0.1,
        mlp_hidden=[128, 128],
        mlp_hidden_image=[512, 512],
        bw_mult=(0.5,),
        early_stop=True,
        eval_every=10,
        val_frac=0.05,
        bw_subsample=4000,
    ),
}

# Max floats in one (rows x n) kernel block (256 MB in fp32).
_BLOCK_ELEMS = 1 << 26

_MOD = None


# --------------------------------------------------------------- import -----
@contextlib.contextmanager
def _isolated_import(root, names):
    """Import the repo's top-level packages (`utils`, `score`, `model`) without
    clobbering or keeping any same-named modules: stash them, import, restore."""
    saved = {k: v for k, v in sys.modules.items()
             if k.split(".")[0] in names}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, root)
    try:
        yield
    finally:
        sys.path.remove(root)
        for k in [k for k in sys.modules if k.split(".")[0] in names]:
            del sys.modules[k]
        sys.modules.update(saved)


def _stub_missing(stubs):
    """Install stub modules for packages that are not importable; returns the
    names added so they can be removed after the import."""
    added = []
    for name, attrs in stubs.items():
        try:
            __import__(name)
            continue
        except Exception:
            pass
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            sub = ".".join(parts[:i])
            if sub in sys.modules and i < len(parts):
                continue
            m = types.ModuleType(sub)
            m.__path__ = []
            if i == len(parts):
                for k, v in attrs.items():
                    setattr(m, k, v)
            sys.modules[sub] = m
            added.append(sub)
            if i > 1:
                setattr(sys.modules[".".join(parts[:i - 1])], parts[i - 1], m)
    return added


def _load():
    global _MOD
    if _MOD is not None:
        return _MOD
    if not os.path.isfile(os.path.join(REPO_DIR, "model", "wgf_imp.py")):
        raise RuntimeError(
            f"NewImp not found at {REPO_DIR}: git clone {REPO_URL} there and "
            f"git checkout {COMMIT}")

    def _unused(*a, **k):
        raise RuntimeError("stubbed out by sota/newimp.py")

    class _NoWriter:
        def __init__(self, *a, **k):
            pass

        def add_scalar(self, *a, **k):
            pass

    added = _stub_missing({
        "hyperimpute.plugins.utils.simulate": dict(simulate_nan=_unused),
        "ot": dict(emd2=_unused),
        "torch.utils.tensorboard": dict(SummaryWriter=_NoWriter),
    })
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # functorch.* deprecation notices
            with _isolated_import(REPO_DIR, {"utils", "score", "model"}):
                import model.wgf_imp as wgf
                import score.score_tuple as score_tuple
                from score.score_trainer import Trainer
    finally:
        for k in added:
            sys.modules.pop(k, None)
    _MOD = types.SimpleNamespace(wgf=wgf, score_tuple=score_tuple, Trainer=Trainer)
    return _MOD


# ---------------------------------------------------------- runtime patches --
def _kernel_terms(X, R, sigma, block_elems=_BLOCK_ELEMS):
    """Block-wise K @ R for the RBF kernel K_ij = exp(-|x_i-x_j|^2 / (2 s^2))."""
    n = X.shape[0]
    gamma = 1.0 / (1e-8 + 2.0 * sigma ** 2)
    sq = (X * X).sum(1)
    rows = max(1, min(n, block_elems // max(n, 1)))
    out = torch.empty(n, R.shape[1], device=X.device, dtype=X.dtype)
    for i0 in range(0, n, rows):
        i1 = min(n, i0 + rows)
        D = torch.addmm(sq[None, :], X[i0:i1], X.t(), beta=1.0, alpha=-2.0)
        D.add_(sq[i0:i1, None]).clamp_(min=0.0).mul_(-gamma).exp_()
        out[i0:i1] = D @ R
        del D
    return out


def xRBF_blockwise(sigma):
    """Drop-in for wgf_imp.xRBF(sigma)(X, X): returns (dx_K, K) semantics, but
    only the pieces the sampler needs; K itself is never built in full."""
    def compute(X, Y=None, S=None):
        n, d = X.shape
        R = torch.cat([X, torch.ones(n, 1, device=X.device, dtype=X.dtype)]
                      + ([S] if S is not None else []), 1)
        out = _kernel_terms(X, R, sigma)
        KX, ks = out[:, :d], out[:, d:d + 1]
        dK = (X * ks - KX) / (1.0e-8 + sigma ** 2)
        KS = out[:, d + 1:] if S is not None else None
        return dK, KS
    return compute


def _knew_imp_sampling(self, data, bandwidth, data_number, score_func,
                       grad_optim, iter_steps, mask_matrix):
    """Repo's knew_imp_sampling with the kernel products computed block-wise.
    `mask_matrix` is True on OBSERVED entries (their gradient is zeroed).
    Adapter hooks (not in the repo): `self._monitor(step, data)` records the
    validation curve every `self._eval_every` steps, and `self._stop_at` ends
    the flow after that many steps in total (validation-chosen early stop)."""
    for _ in range(iter_steps):
        if self._stop_at is not None and self._steps_done >= self._stop_at:
            break
        with torch.no_grad():
            eval_score = score_func(data)
            eval_grad_k, KS = self.grad_val_kernel(data.detach(), None, eval_score)
            grad_tensor = -1.0 * (KS - self.entropy_reg * eval_grad_k) / data_number
        if torch.isnan(grad_tensor).any() or torch.isinf(grad_tensor).any():
            self._nan_break = True
            break
        grad_optim.zero_grad()
        data.grad = torch.masked_fill(input=grad_tensor, mask=mask_matrix, value=0.0)
        grad_optim.step()
        self._steps_done += 1
        if self._monitor is not None and self._steps_done % self._eval_every == 0:
            self._monitor(self._steps_done, data)
    return data


def _fit_transform(self, X, fill_mean):
    """Repo's fit_transform loop in float32 without X_true (see module doc).
    X: float32 torch tensor (n, d) with NaN; fill_mean: (d,) column means."""
    M = _load()
    dev = self.device
    n, d = X.shape
    self.mlp_model = M.score_tuple.ToyMLP(input_dim=d, units=self.mlp_hidden).to(dev)
    self.score_net = M.score_tuple.Energy(net=self.mlp_model).to(dev)

    miss = torch.isnan(X)
    X_filled = X.clone()
    init = self.noise * torch.randn(X.shape) + fill_mean[None, :]
    X_filled[miss] = init[miss]
    X_filled = X_filled.to(dev).requires_grad_()
    grad_mask = (~miss).to(dev)
    optimizer = self.opt([X_filled], lr=self.lr)
    self._steps_done, self._nan_break = 0, False
    self._times = dict(score_s=0.0, flow_s=0.0)
    if self._monitor is not None:
        self._monitor(0, X_filled)

    for _ in range(self.niter):
        if self._stop_at is not None and self._steps_done >= self._stop_at:
            break
        t0 = time.perf_counter()
        trainer = M.Trainer(model=self.score_net, learning_rate=self.score_net_lr,
                            loss_type=self.score_loss_type, device=dev)
        data = X_filled.detach()
        for _ in range(self.score_net_epoch):
            M.score_tuple.train_step(data, trainer)
        self._times["score_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        X_filled = self.knew_imp_sampling(
            data=X_filled, data_number=n, score_func=self.score_net.functorch_score,
            bandwidth=self.bandwidth, grad_optim=optimizer,
            iter_steps=self.sampling_step, mask_matrix=grad_mask)
        self._times["flow_s"] += time.perf_counter() - t0
        if self._nan_break:
            break
    return X_filled.detach()


# ------------------------------------------------------------------ helpers --
def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _median_bandwidth(Z, n_sub, seed):
    """SVGD median heuristic of the repo's xRBF (bandwidth < 0 branch):
    s^2 = median |x_i - x_j|^2 / (2 log(n + 1)), on a row subsample."""
    n = Z.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n, n_sub), replace=False)
    S = Z[idx]
    sq = (S * S).sum(1)
    D = (sq[:, None] + sq[None, :] - 2.0 * S @ S.T).clamp_(min=0.0)
    iu = torch.triu_indices(len(idx), len(idx), offset=1, device=D.device)
    med = D[iu[0], iu[1]].median().item()
    return math.sqrt(max(med, 1e-12) / (2.0 * math.log(n + 1.0)))


def _run_one(Z, fill_mean, sigma, cfg, hidden, device, seed,
             monitor=None, stop_at=None):
    M = _load()
    _seed_all(seed)
    imp = M.wgf.NeuralGradFlowImputer(
        entropy_reg=cfg["entropy_reg"], lr=cfg["lr"], niter=cfg["niter"],
        kernel_func=xRBF_blockwise, mlp_hidden=list(hidden),
        score_net_epoch=cfg["score_net_epoch"], score_net_lr=cfg["score_net_lr"],
        score_loss_type="dsm", bandwidth=sigma,
        sampling_step=cfg["sampling_step"], device=device, noise=cfg["noise"])
    imp.knew_imp_sampling = types.MethodType(_knew_imp_sampling, imp)
    imp._monitor, imp._stop_at, imp._eval_every = monitor, stop_at, cfg["eval_every"]
    out = _fit_transform(imp, Z, fill_mean)
    meta = dict(steps=imp._steps_done, nan_break=imp._nan_break,
                score_s=round(imp._times["score_s"], 2),
                flow_s=round(imp._times["flow_s"], 2))
    del imp
    return out, meta


# ------------------------------------------------------------------ adapter --
def impute(X, *, seed, K, profile, device):
    cfg = BUDGETS[profile]
    dev = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available())
                       else "cpu")
    shape = X.shape
    B = shape[0]
    Xf = np.asarray(X, dtype=np.float64).reshape(B, -1)
    F = Xf.shape[1]
    obs = ~np.isnan(Xf)

    # z-score per feature from observed entries (identity-ish for tabular data)
    with np.errstate(all="ignore"):
        cnt = obs.sum(0)
        mu = np.where(cnt > 0, np.nansum(Xf, 0) / np.maximum(cnt, 1), 0.0)
        sd = np.sqrt(np.where(cnt > 0, np.nansum((Xf - mu) ** 2, 0) / np.maximum(cnt, 1), 1.0))
    sd = np.where(np.isfinite(sd) & (sd >= 1e-8), sd, 1.0)
    Z = torch.from_numpy(((Xf - mu) / sd).astype(np.float32))  # NaN stays NaN
    fill_mean = torch.zeros(F)  # column means of z-scored observed data = 0

    image = len(shape) > 2  # (B, H, W[, C]) image datasets; tables are (B, F)
    hidden = cfg["mlp_hidden_image"] if image else cfg["mlp_hidden"]

    # base bandwidth on the mean-imputed data (observed entries + 0 fill)
    Z0 = torch.nan_to_num(Z, nan=0.0).to(dev)
    sigma0 = _median_bandwidth(Z0, cfg["bw_subsample"], seed)
    del Z0

    # Selection on a random val_frac of the OBSERVED entries, hidden during one
    # fit per bandwidth multiplier: pick the multiplier and (if early_stop) the
    # flow step count with the lowest MAE there. Held-out truth is never read.
    t_start = time.perf_counter()
    mults = list(cfg["bw_mult"])
    lrs = list(cfg.get("lr_grid", (cfg["lr"],)))
    total = cfg["niter"] * cfg["sampling_step"]
    best_m, best_lr, best_T, sel = mults[0], lrs[0], total, {}
    if len(mults) > 1 or len(lrs) > 1 or cfg["early_stop"]:
        rng = np.random.default_rng(seed + 7919)
        oi = np.flatnonzero(obs.ravel())
        nv = max(1, int(round(cfg["val_frac"] * oi.size)))
        vi = torch.from_numpy(np.sort(rng.choice(oi, size=nv, replace=False)))
        Zv = Z.clone().view(-1)
        truth = Zv[vi].clone().to(dev)
        Zv[vi] = float("nan")
        Zv = Zv.view(B, F)
        vi = vi.to(dev)
        best_mae = float("inf")
        for m in mults:
          for lr in lrs:
            curve = {}

            def monitor(step, data, curve=curve):
                with torch.no_grad():
                    e = (data.detach().view(-1)[vi] - truth).abs().mean().item()
                curve[step] = e if math.isfinite(e) else float("inf")

            out, meta = _run_one(Zv, fill_mean, sigma0 * m, dict(cfg, lr=lr), hidden,
                                 dev, seed, monitor=monitor)
            del out
            last = max(curve)
            if cfg["early_stop"]:
                T, mae = min(curve.items(), key=lambda kv: (kv[1], kv[0]))
            else:
                T, mae = last, curve[last]
            sel[f"bw{m}_lr{lr}"] = dict(val_mae=round(mae, 5), val_step=T,
                                        val_mae_last=round(curve[last], 5), **meta)
            if mae < best_mae:
                best_mae, best_m, best_lr, best_T = mae, m, lr, T
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        del Zv, truth, vi
    t_sel = time.perf_counter() - t_start

    stop_at = best_T if cfg["early_stop"] else None
    out, meta = _run_one(Z, fill_mean, sigma0 * best_m, dict(cfg, lr=best_lr), hidden,
                         dev, seed, stop_at=stop_at)
    Zhat = out.cpu().numpy().astype(np.float64)
    del out
    Zhat = np.where(np.isfinite(Zhat), Zhat, 0.0)
    X_hat = Zhat * sd + mu
    X_hat = np.where(obs, Xf, X_hat)

    info = dict(
        profile=profile, commit=COMMIT, device=str(dev), image_mlp=image,
        mlp_hidden=list(hidden), entropy_reg=cfg["entropy_reg"], lr=best_lr,
        niter=cfg["niter"], sampling_step=cfg["sampling_step"],
        score_net_epoch=cfg["score_net_epoch"], sigma_median=sigma0,
        bw_mult=best_m, bandwidth=sigma0 * best_m, early_stop=cfg["early_stop"],
        flow_steps=meta["steps"], flow_steps_budget=total,
        selection=sel, selection_s=round(t_sel, 2), final_fit=meta)
    return dict(X_hat=X_hat.reshape(shape), labels=None, info=info)
