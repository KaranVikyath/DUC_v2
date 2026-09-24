"""
CACTI (Kim, Rasheed, ... Sankararaman, ICML 2025), run as its no-context
variant CMAE: a transformer masked autoencoder trained with Median-Truncated
Copy Masking (MT-CM). The full CACTI model also needs per-column text
descriptions embedded by a language model; our columns are pixels or
anonymous sensor channels with no descriptions, so the context branch has
nothing to encode and CMAE is the applicable variant (it is the paper's own
"CACTI without context" ablation).

Official code: https://github.com/sriramlab/CACTI (GPL-3.0), fetched at
COMMIT into SOTA_ROOT/CACTI and imported from there; nothing is vendored.

Runtime patches (the repo's files are never edited):
  1. Checkpointing. The official train+eval path (train.py) always keeps the
     epoch with the lowest mean TRAINING loss after warm-up and imputes with
     it. Its CheckpointHandler writes .pth files and an os.symlink, which
     needs admin rights on Windows. It is replaced by an in-memory handler
     with the same selection rule. The training loss is computed on observed
     entries only, so no held-out value is ever used; --pval (a row split) is
     left unset as in the paper runs.
  2. Eval-time token order. masker.CopySegmentModule.contextify builds its
     ordered "noise" with torch.arange(0, 0.99, 0.99/L), which float rounding
     makes L+1 long for many L (50, 100, 200, 400, 800, ...: 302 values of
     L < 6000), and the next line then fails with a shape mismatch, so the
     original crashes at inference on e.g. our 50-feature synthetic data.
     The `torch` global of masker.py (only) is replaced by a proxy whose
     float-step arange returns exactly round((end-start)/step) values
     start + i*step; everything else is forwarded to torch. Inference itself
     is the original CopyMAE.transform (batch size 1: its eval contextify is
     only correct for one row per batch, because the expanded noise tensor is
     shared by all rows of a batch).
  3. tqdm progress bars are disabled (log noise on CHTC); nothing else in the
     training loop is touched.
  4. Min-max scales come from observed entries (as train.py's _getscales);
     a column with no observed entry gets min = max = the global observed
     mean, so it is imputed with that constant instead of NaN.

The model outputs tanh/2 + 1/2 on per-column min-max scaled data, so X_hat
stays in the input units (raw pixels, or the harness's z-scores).
"""

import contextlib
import io
import math
import os
import random
import re
import sys
import tempfile
import time
import types
import warnings

import numpy as np
import torch

from . import SOTA_ROOT

REPO_URL = "https://github.com/sriramlab/CACTI"
REPO_DIR = "CACTI"
COMMIT = "b78bce8505611df982d23f7e4832d0877d09cfe3"   # tag v1.1.1

# Paper / README configuration for the MAE family (CMAE = CACTI without the
# column-context branch): copy-mask ratio 0.9, embed 64, 10 encoder and 4
# decoder blocks, batch 128, AdamW lr 1e-3 (betas 0.9/0.95, wd 1e-3), grad clip
# 5, 300 epochs with a 50-epoch linear warm-up then cosine to min_lr 5e-6.
_PAPER = dict(
    model="CMAE", mask_ratio=0.9, embed_dim=64, nencoder=10, ndecoder=4,
    batch_size=128, epochs=300, warmup_epochs=50,
    lr=1e-3, min_lr=5e-6, weight_decay=1e-3, grad_clip=5.0, num_workers=0,
)

BUDGETS = {
    # Paper default, used on the small datasets.
    "full": dict(_PAPER),
    # Large datasets. Every token attends to every other, so an epoch costs
    # about B*(F+1)^2; the epoch count is cut so that this "attention work"
    # stays under attn_work_cap, never below min_epochs and never above the
    # paper's 300, with warm-up kept at the paper's 1/6 of training.
    # Fixed in advance from shapes only (not from any result):
    #   COIL100 (7200x1024), HARUS, DSDD, VED -> 300 epochs (= paper budget)
    #   Flowers (8189x3072)  -> 38 epochs, warm-up 6
    #   OxfordPet (7349x3072)-> 43 epochs, warm-up 7
    # Measured on the RTX 3060 laptop at F=3072 and 10% missing (the worst
    # rate): ~118 ms/row/epoch, i.e. ~16 min/epoch for Flowers -> 38 epochs is
    # ~10 h there, ~4 h on an L40S.
    "reduced": dict(_PAPER, attn_work_cap=3.0e12, min_epochs=30),
}

_WARMUP_FRAC = 50 / 300

_mods = None


# --------------------------------------------------------------- import ----
def _load():
    """Import the CACTI modules we need, isolated from any other `src` package.

    The repo is a namespace package called `src`, a name other SOTA repos
    also use. Any foreign `src*` modules are set aside while CACTI's are
    imported, CACTI's are then kept under private names, and the originals
    are put back. CACTI's modules bind their dependencies at import time, so
    they keep working after the rename.
    """
    global _mods
    if _mods is not None:
        return _mods
    root = os.path.join(SOTA_ROOT, REPO_DIR)
    if not os.path.isdir(os.path.join(root, "src", "imputers")):
        raise RuntimeError(
            f"CACTI not found at {root}. Fetch it with:\n"
            f"  git clone {REPO_URL} {root} && git -C {root} checkout {COMMIT}")
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == "src" or k.startswith("src.")}
    sys.path.insert(0, root)
    try:
        import importlib
        copymae = importlib.import_module("src.imputers.copymae")
        cmae = importlib.import_module("src.models.cmae")
        ours = {k: sys.modules.pop(k) for k in list(sys.modules)
                if k == "src" or k.startswith("src.")}
    finally:
        try:
            sys.path.remove(root)
        except ValueError:
            pass
        sys.modules.update(saved)
    for k, m in ours.items():
        sys.modules["_cacti_" + k] = m
    _mods = types.SimpleNamespace(copymae=copymae, cmae=cmae)
    _patch_quiet_tqdm(copymae)
    # masker.py's module globals, reached through the class cmae.py calls.
    cmae.CopySegmentModule.contextify.__globals__["torch"] = _ExactArangeTorch()
    return _mods


class _ExactArangeTorch:
    """Stands in for `torch` inside CACTI's masker.py only (patch 2)."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def arange(*a, **kw):
        if len(a) == 3 and isinstance(a[2], float):
            start, end, step = a
            n = int(round((end - start) / step))
            i = torch.arange(n, dtype=kw.get("dtype") or torch.get_default_dtype(),
                             device=kw.get("device"))
            return start + step * i
        return torch.arange(*a, **kw)


def _patch_quiet_tqdm(copymae):
    base = copymae.tqdm

    def quiet(*a, **kw):
        kw["disable"] = True
        return base(*a, **kw)
    copymae.tqdm = quiet


class _MemCheckpoint:
    """In-memory stand-in for CACTI's CheckpointHandler (same selection rule:
    copymae.fit calls save_checkpoint whenever the epoch's mean training loss
    is a new best and epoch >= warmup_epochs)."""

    def __init__(self, model, marker_dir):
        self.model = model
        self.checkpoint_dir = marker_dir    # fit() checks for best.pth here
        self.hyperparameters = {}
        self.best_state = None
        self.best_epoch = None

    def save_checkpoint(self, epoch, optimizer):
        self.best_state = {k: v.detach().clone()
                           for k, v in self.model.state_dict().items()}
        self.best_epoch = int(epoch)
        open(os.path.join(self.checkpoint_dir, "best.pth"), "a").close()


class _Tee(io.TextIOBase):
    def __init__(self, stream):
        self.stream, self.buf = stream, io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self.stream.write(s)

    def flush(self):
        self.stream.flush()


# ------------------------------------------------------------- budgets -----
def _schedule(budget, B, F):
    E, W = int(budget["epochs"]), int(budget["warmup_epochs"])
    cap = budget.get("attn_work_cap")
    if cap:
        E_cap = int(math.floor(cap / (B * (F + 1) ** 2)))
        E_new = max(int(budget.get("min_epochs", 1)), min(E, E_cap))
        if E_new < E:
            E = E_new
            W = max(1, int(round(E * _WARMUP_FRAC)))
    if W >= E:
        W = max(0, E // 6)
    return E, W


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- main -----
def _impute_with(X, budget, *, seed, device):
    mods = _load()
    shape = X.shape
    B = shape[0]
    Xf = np.asarray(X, dtype=np.float64).reshape(B, -1)
    F = Xf.shape[1]
    obs_mask = ~np.isnan(Xf)
    if not obs_mask.any():
        raise ValueError("CACTI: no observed entries")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mins = np.nanmin(Xf, axis=0)
        maxs = np.nanmax(Xf, axis=0)
    empty = ~obs_mask.any(axis=0)
    if empty.any():
        g = float(Xf[obs_mask].mean())
        mins[empty] = g
        maxs[empty] = g

    epochs, warmup = _schedule(budget, B, F)
    dev = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available())
                       else "cpu")
    _seed_all(seed)

    args = types.SimpleNamespace(
        batch_size=int(budget["batch_size"]), min_lr=budget["min_lr"],
        lr=budget["lr"], grad_clip=budget["grad_clip"],
        warmup_epochs=warmup, epochs=epochs, mask_ratio=budget["mask_ratio"],
        device=dev, num_workers=int(budget.get("num_workers", 0)),
        weight_decay=budget["weight_decay"], embed_dim=int(budget["embed_dim"]),
        nencoder=int(budget["nencoder"]), ndecoder=int(budget["ndecoder"]),
        checkpoint_path=None, pval=None, seed=seed)
    feats = [f"f{i}" for i in range(F)]
    obs_t = torch.tensor(Xf, dtype=torch.float32)

    prev_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("medium")      # as train.py's train_eval
    tee = _Tee(sys.stdout)
    try:
        with tempfile.TemporaryDirectory(prefix="cacti_") as tmp, \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            if budget["model"] != "CMAE":
                raise ValueError("only the CMAE variant is wired up")
            with contextlib.redirect_stdout(tee):
                imp = mods.copymae.CopyMAE(args, feats=feats)
                ckpt = _MemCheckpoint(imp.model, tmp)
                imp.checkpoint_handler = ckpt
                imp.min_scale, imp.max_scale = mins, maxs
                t0 = time.perf_counter()
                imp.fit(obs_t)
                fit_s = time.perf_counter() - t0
            if ckpt.best_state is not None:
                imp.model.load_state_dict(ckpt.best_state)
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(tee):
                X_hat = np.asarray(imp.transform(obs_t), dtype=np.float64)
            infer_s = time.perf_counter() - t0
    finally:
        torch.set_float32_matmul_precision(prev_prec)

    X_hat[obs_mask] = Xf[obs_mask]
    bad = ~np.isfinite(X_hat)
    if bad.any():   # should not happen; keep the run scorable and say so
        colmean = np.where(empty, 0.0, np.nanmean(np.where(obs_mask, Xf, np.nan), 0))
        X_hat[bad] = np.broadcast_to(colmean, X_hat.shape)[bad]

    losses = [float(v) for v in re.findall(
        r"Epoch \d+ :: Average RMSE loss: ([0-9.eE+-]+|nan|inf)", tee.buf.getvalue())]
    spe = int(math.ceil(B / args.batch_size))
    info = dict(
        variant="CMAE (CACTI without column context)", commit=COMMIT,
        epochs=epochs, warmup_epochs=warmup, steps=epochs * spe,
        batch_size=args.batch_size, embed_dim=args.embed_dim,
        nencoder=args.nencoder, ndecoder=args.ndecoder,
        mask_ratio=args.mask_ratio, lr=args.lr,
        best_epoch=ckpt.best_epoch,
        selection="min mean training loss after warm-up (observed entries only)"
                  if ckpt.best_epoch is not None else "last epoch (no post-warm-up best)",
        train_rmse_first=losses[0] if losses else None,
        train_rmse_last=losses[-1] if losses else None,
        train_rmse_best=min(losses[warmup:]) if len(losses) > warmup else None,
        fit_s=fit_s, infer_s=infer_s,
        empty_columns=int(empty.sum()), nonfinite_filled=int(bad.sum()),
        device=str(dev),
    )
    del imp, ckpt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return X_hat.reshape(shape), info


def impute(X, *, seed, K=None, profile="full", device="cuda"):
    if profile not in BUDGETS:
        raise ValueError(f"unknown profile {profile!r}; have {sorted(BUDGETS)}")
    X_hat, info = _impute_with(X, BUDGETS[profile], seed=seed, device=device)
    info["profile"] = profile
    return dict(X_hat=X_hat, labels=None, info=info)
