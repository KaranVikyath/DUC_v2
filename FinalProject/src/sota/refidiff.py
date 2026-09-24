"""
RefiDiff (Ahamed, Ye & Cheng, "RefiDiff: Progressive Refinement Diffusion for
Efficient Missing Data Imputation", AAAI 2026): an XGBoost per-column
refinement, an EDM denoiser (Mamba blocks) trained on the refined matrix,
RePaint-style conditional sampling of the missing entries averaged over
`num_trials` runs, and a second XGBoost refinement pass on that result.

Official code: https://github.com/Atik-Ahamed/RefiDiff (Apache-2.0), fetched
into SOTA_ROOT/RefiDiff at the pinned COMMIT below and imported from there.
Used unchanged: model.CustomDenoiser / Model / Precond (EDM preconditioning +
EDMLoss), diffusion_utils.impute_mask / sample_step (the sampler, N=10
resampling rounds per step) and diffusion_utils.refinement (the XGBoost
per-column pass), dataset.mean_std. What this adapter changes at RUNTIME
(their files are never edited):

  * `mamba_ssm` is replaced by the pure-PyTorch `Mamba` below, installed in
    sys.modules only while model.py imports. mamba-ssm needs its CUDA
    extension (no Windows wheel; a source build in Docker). The class is a
    line-by-line port of mamba_ssm.modules.mamba_simple.Mamba (v1, as in
    mamba-ssm 2.2.x): same parameters, same initialisation order, same
    maths as its reference path (causal depthwise conv1d, then
    selective_scan_ref). RefiDiff's MambaBlock always calls it on a sequence
    of length 1, where the scan reduces exactly to
    y = softplus(dt) * u * <B, C> + D * u (A never enters); we take that
    shortcut for L == 1 and run the reference recurrence otherwise.
  * `catboost` is stubbed, but only when it is not installed. It is used only
    for categorical columns (col >= len_num); every column here is numeric
    (len_num = F), so that branch is unreachable.
  * diffusion_utils.XGBRegressor (a module global that refinement() calls
    as XGBRegressor()) is replaced by a factory that returns an XGBRegressor
    with the budget's parameters: device=cuda when running on GPU (same
    `hist` algorithm the default CPU regressor uses), random_state=seed, the
    job's thread count, and n_estimators / max_bin from BUDGETS (the repo
    uses the library defaults, 100 / 256, which "full" keeps). The wrapper
    also handles a column with no observed entry (it keeps the zero fill,
    i.e. the observed-mean). XGBoost cannot fit zero rows, and the repo's
    datasets never have such a column.
  * main.py's orchestration is rewritten here for the in-sample
    (transductive) case: the whole matrix is RefiDiff's "train" split and
    there is no test split. We follow its steps in order: zero-filled
    normalised X -> refinement -> train on the refined matrix ->
    `num_trials` x impute_mask from the zero-filled X with the best
    checkpoint -> observed entries restored -> mean over trials ->
    refinement -> x2, de-normalise. Differences, all behaviour-neutral:
    - Batches are drawn with an on-device randperm, not a num_workers=4
      DataLoader (on Windows that respawns 4 processes every epoch).
    - The best checkpoint is kept in memory, not in ckpt/ on disk.
    - impute_mask runs over row chunks of <= _CHUNK_ELEMS floats. Rows are
      independent in the network and the sampler, so only the RNG
      stream changes.
    - The trial mean is a running sum, not a stack of num_trials copies.
    - Everything runs in float32. main.py keeps X in float64, but the network
      is float32 and XGBoost converts its input to float32 anyway.
  * Normalisation: dataset.mean_std on the observed entries (NaN -> 0
    first, since mean_std multiplies by the mask), then (X - mean) / std / 2
    as in main.py. A constant (or empty) column has std 0, which gives
    inf/NaN in the original, so std < 1e-8 is set to 1.

Training follows main.py exactly: Adam(lr 1e-4), ReduceLROnPlateau(0.9, 40)
on the epoch loss, up to `epochs` epochs, stopping after `patience` epochs
without a new best TRAINING loss, and sampling from the best-loss weights.
That loss is computed on the refined matrix, which is built from observed
entries only; no held-out value is ever used, and nothing is tuned.

Budgets (fixed in advance, never tuned on test data). "full" is the paper /
repo default. XGBoost refinement dominates the cost: F regressions, each on
F-1 features, run twice, so it grows as F^2 * B. "reduced" (the expensive-
method grid on COIL100 / Flowers / OxfordPet / HARUS / DSDD / VED) keeps
every diffusion setting and uses 30 trees and 64 histogram bins per column
regressor. Laptop RTX 3060 measurements (GPU shared with other jobs, so
pessimistic) at the worst case, Flowers at 10% missing (7370 x 3071 fits):
100 trees / 256 bins OOMs the 6 GB card (the hist cache alone is ~0.8 GB);
50/64 took 7.8 s per fit and 30/64 took 5.3 s. Over 2 x 3072 fits that is
~9.1 h on the laptop, ~3.6 h at the assumed 2.5x L40S speed-up, against the
8 h limit.

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
import torch.nn as nn
import torch.nn.functional as F

from . import SOTA_ROOT

REPO_DIR = os.path.join(SOTA_ROOT, "RefiDiff")
REPO_URL = "https://github.com/Atik-Ahamed/RefiDiff"
COMMIT = "996aab1fc86d9e92f853b816f42123afc9d070fd"

_DIFFUSION = dict(
    hid_dim=32,          # main.py --hid_dim
    num_trials=10,       # main.py --num_trials
    num_steps=50,        # main.py --num_steps (impute_mask: N=10 resampling rounds, fixed)
    batch_size=8192,     # main.py --batch_size
    epochs=10001,        # main.py --epochs
    patience=500,        # main.py early stop on the epoch TRAINING loss
    lr=1e-4,             # Adam, weight_decay 0
    plateau_factor=0.9,  # ReduceLROnPlateau(factor=0.9, patience=40)
    plateau_patience=40,
)
BUDGETS = {
    # repo defaults; the XGBoost values are the library defaults the repo relies on
    "full": dict(_DIFFUSION, xgb=dict(n_estimators=100, max_bin=256, max_depth=6,
                                      learning_rate=0.3)),
    # expensive-method grid on the large datasets: smaller per-column regressors
    "reduced": dict(_DIFFUSION, xgb=dict(n_estimators=30, max_bin=64, max_depth=6,
                                         learning_rate=0.3)),
}

# Max floats per (rows x F) tensor in one impute_mask call (64 MB in fp32).
_CHUNK_ELEMS = 1 << 24

_MOD = None


# ------------------------------------------------------ pure-PyTorch Mamba --
class Mamba(nn.Module):
    """Port of mamba_ssm.modules.mamba_simple.Mamba (mamba-ssm 2.2.x, v1 block)
    without the CUDA kernels. Parameters, initialisation order and forward
    maths follow the original's reference path (conv1d + selective_scan_ref)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True,
                 layer_idx=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                bias=conv_bias, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, **factory_kwargs)
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
                                **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(self.d_inner, **factory_kwargs)
                       * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device)
        A = A.unsqueeze(0).expand(self.d_inner, -1).contiguous()  # S4D real init
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError("recurrent decoding is not used by RefiDiff")
        batch, seqlen, _ = hidden_states.shape
        xz = F.linear(hidden_states, self.in_proj.weight, self.in_proj.bias)  # (b, l, 2d)
        x, z = xz.chunk(2, dim=-1)
        x = self.act(self.conv1d(x.transpose(1, 2))[..., :seqlen]).transpose(1, 2)  # causal
        x_dbl = self.x_proj(x)
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # delta = softplus(W_dt dt + b_dt): selective_scan(delta_bias=b_dt, delta_softplus=True)
        delta = F.softplus(F.linear(dt, self.dt_proj.weight).float()
                           + self.dt_proj.bias.float())                        # (b, l, d)
        u = x.float()
        Bm, Cm = Bm.float(), Cm.float()
        if seqlen == 1:
            # h_1 = exp(delta A) h_0 + delta B u with h_0 = 0  =>  y = delta u <B, C>
            y = delta * u * (Bm * Cm).sum(-1, keepdim=True)
        else:
            A = -torch.exp(self.A_log.float())                                  # (d, n)
            h = u.new_zeros(batch, self.d_inner, self.d_state)
            ys = []
            for t in range(seqlen):
                dA = torch.exp(delta[:, t, :, None] * A)                        # (b, d, n)
                h = dA * h + (delta[:, t] * u[:, t])[:, :, None] * Bm[:, t, None, :]
                ys.append(torch.einsum("bdn,bn->bd", h, Cm[:, t]))
            y = torch.stack(ys, dim=1)
        y = y + u * self.D.float()
        y = y * F.silu(z.float())
        return self.out_proj(y.to(hidden_states.dtype))


# --------------------------------------------------------------- import -----
@contextlib.contextmanager
def _isolated_import(root, names):
    """Import the repo's top-level modules (model, diffusion_utils, dataset)
    without clobbering or keeping same-named modules of other repos."""
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in names}
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


def _load():
    global _MOD
    if _MOD is not None:
        return _MOD
    if not os.path.isfile(os.path.join(REPO_DIR, "diffusion_utils.py")):
        raise RuntimeError(f"RefiDiff not found at {REPO_DIR}: git clone {REPO_URL} there "
                           f"and git checkout {COMMIT}")

    stubs = {}
    mamba = types.ModuleType("mamba_ssm")          # always the pure-PyTorch port
    mamba.Mamba = Mamba
    stubs["mamba_ssm"] = mamba
    try:
        import catboost  # noqa: F401
    except Exception:
        cb = types.ModuleType("catboost")

        class CatBoostClassifier:                   # categorical columns only
            def __init__(self, *a, **k):
                raise RuntimeError("catboost stubbed by sota/refidiff.py: all columns "
                                   "are numeric, so this branch must be unreachable")
        cb.CatBoostClassifier = CatBoostClassifier
        stubs["catboost"] = cb
    saved = {k: sys.modules[k] for k in stubs if k in sys.modules}
    sys.modules.update(stubs)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _isolated_import(REPO_DIR, {"model", "diffusion_utils", "dataset"}):
                import diffusion_utils as du
                import model as rmodel
                import dataset as rdataset
    finally:
        for k in stubs:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    _MOD = types.SimpleNamespace(du=du, model=rmodel, dataset=rdataset,
                                 XGBRegressor=du.XGBRegressor)
    return _MOD


# ---------------------------------------------------------------- helpers ---
def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _n_threads():
    for var in ("OMP_NUM_THREADS", "NSLOTS"):
        v = os.environ.get(var, "")
        if v.isdigit() and int(v) > 0:
            return int(v)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class _ColumnRegressor:
    """What refinement() gets from `XGBRegressor()`: an XGBRegressor with the
    budget's parameters. A column with no observed row keeps its zero fill
    (the observed mean after normalisation) instead of crashing XGBoost."""

    stats = None  # dict(fits, seconds), set per refinement pass

    def __init__(self, xgb_cls, params):
        self._m = xgb_cls(**params)
        self._const = None

    def fit(self, X, y, verbose=False):
        t0 = time.perf_counter()
        if len(y) == 0:
            self._const = 0.0
        else:
            self._m.fit(X, y, verbose=verbose)
        st = _ColumnRegressor.stats
        if st is not None:
            st["fits"] += 1
            st["seconds"] += time.perf_counter() - t0
            if st["report_every"] and st["fits"] % st["report_every"] == 0:
                print(f"    [refidiff] {st['name']}: {st['fits']}/{st['total']} columns, "
                      f"{time.perf_counter() - st['t0']:.0f}s", flush=True)
        return self

    def predict(self, X):
        if self._const is not None:
            return np.full(len(X), self._const, dtype=np.float32)
        return self._m.predict(X)


def _refine(M, X, obs_f, xgb_params, name):
    """One pass of the repo's refinement() (their per-column loop)."""
    n_cols = int((obs_f == 0).any(0).sum())
    stats = dict(name=name, fits=0, seconds=0.0, total=n_cols, t0=time.perf_counter(),
                 report_every=max(1, n_cols // 4) if n_cols >= 200 else 0)
    _ColumnRegressor.stats = stats
    M.du.XGBRegressor = lambda *a, **k: _ColumnRegressor(M.XGBRegressor, xgb_params)
    try:
        out = M.du.refinement(rec_X=X, mask=obs_f, len_num=X.shape[1])
    finally:
        M.du.XGBRegressor = M.XGBRegressor
        _ColumnRegressor.stats = None
    return out, dict(fits=stats["fits"], xgb_s=round(stats["seconds"], 2),
                     s=round(time.perf_counter() - stats["t0"], 2))


def _train(M, data, cfg, dev, seed):
    """main.py's training loop on the refined matrix (all rows), best-loss weights."""
    n, d = data.shape
    denoise_fn = M.model.CustomDenoiser(d, cfg["hid_dim"]).to(dev)
    model = M.model.Model(denoise_fn=denoise_fn, hid_dim=d).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["plateau_factor"], patience=cfg["plateau_patience"])
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    bs = cfg["batch_size"]
    model.train()
    best_loss, patience, best_state, best_epoch = float("inf"), 0, None, -1
    t0 = time.perf_counter()
    epoch = -1
    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(n, device=dev, generator=gen)
        batch_loss, len_input = 0.0, 0
        for i in range(0, n, bs):
            inputs = data[perm[i:i + bs]]
            loss = model(inputs, None).mean()
            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)
        if curr_loss < best_loss:
            best_loss, patience, best_epoch = curr_loss, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience == cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(epochs=epoch + 1, best_epoch=best_epoch + 1,
                       best_loss=float(best_loss), train_s=round(time.perf_counter() - t0, 2),
                       n_params=sum(p.numel() for p in denoise_fn.parameters()))


# ------------------------------------------------------------------ adapter --
def impute(X, *, seed, K=None, profile="full", device="cuda"):
    if profile not in BUDGETS:
        raise ValueError(f"unknown profile {profile!r}; have {sorted(BUDGETS)}")
    cfg = BUDGETS[profile]
    M = _load()
    t_start = time.perf_counter()
    dev = torch.device(device if (str(device).startswith("cpu") or torch.cuda.is_available())
                       else "cpu")
    _seed_all(seed)

    shape = X.shape
    B = shape[0]
    Xf = np.asarray(X, dtype=np.float64).reshape(B, -1)
    d = Xf.shape[1]
    miss = np.isnan(Xf)
    obs_f = (~miss).astype(np.float32)          # refinement's mask: 1 = observed

    # main.py: mean_std on observed entries, X = (X - mean) / std / 2
    mean_X, std_X = M.dataset.mean_std(np.where(miss, 0.0, Xf), miss)
    std_X = np.where(np.isfinite(std_X) & (std_X >= 1e-8), std_X, 1.0)
    Xn = np.where(miss, 0.0, (Xf - mean_X) / std_X / 2).astype(np.float32)  # = (1-mask) * X

    xgb_params = dict(cfg["xgb"], random_state=int(seed), n_jobs=_n_threads(), verbosity=0,
                      tree_method="hist", device="cuda" if dev.type == "cuda" else "cpu")

    # 1) refinement of the zero-filled matrix -> training data
    X_ref, r1 = _refine(M, Xn, obs_f, xgb_params, "refine-1")

    # 2) EDM denoiser on the refined matrix
    data = torch.from_numpy(np.ascontiguousarray(X_ref, dtype=np.float32)).to(dev)
    model, tr = _train(M, data, cfg, dev, seed)
    del data
    net = model.denoise_fn_D

    # 3) num_trials x RePaint-style sampling from the zero-filled X, observed restored
    t0 = time.perf_counter()
    Xt = torch.from_numpy(Xn).to(dev)
    Mt = torch.from_numpy(miss)                    # 1 = missing, as main.py's mask
    rows = max(1, min(B, _CHUNK_ELEMS // max(d, 1)))
    rec_sum = torch.zeros_like(Xt)
    for trial in range(cfg["num_trials"]):
        for s in range(0, B, rows):
            x = Xt[s:s + rows]
            m = Mt[s:s + rows]
            rec = M.du.impute_mask(net, x, m, x.shape[0], d, cfg["num_steps"], dev)
            mi = m.to(torch.float32).to(dev)
            rec_sum[s:s + rows] += rec * mi + x * (1 - mi)
            del rec, mi
    rec_X = (rec_sum / cfg["num_trials"]).cpu().numpy()
    del rec_sum, Xt, model, net
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    sample_s = round(time.perf_counter() - t0, 2)

    # 4) refinement of the trial mean -> final imputations
    rec_X, r2 = _refine(M, rec_X, obs_f, xgb_params, "refine-2")

    Zhat = rec_X.astype(np.float64) * 2
    X_hat = Zhat * std_X + mean_X
    X_hat = np.where(np.isfinite(X_hat), X_hat, mean_X)
    X_hat = np.where(miss, X_hat, Xf)             # observed entries exactly as given

    info = dict(method="RefiDiff", repo=REPO_URL, commit=COMMIT, profile=profile,
                device=str(dev),
                budget={k: v for k, v in cfg.items()},
                xgb=dict(xgb_params), sample_chunk_rows=rows,
                refine1=r1, train=tr, sample_s=sample_s, refine2=r2,
                seconds=round(time.perf_counter() - t_start, 2))
    return dict(X_hat=X_hat.reshape(shape), labels=None, info=info)
