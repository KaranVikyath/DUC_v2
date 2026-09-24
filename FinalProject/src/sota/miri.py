"""MIRI: missing-data imputation by reducing mutual information with rectified
flows (Yu et al., NeurIPS 2025). https://github.com/yujhml/MIRI-Imputation

The official code is imported from SOTA_ROOT/MIRI-Imputation at COMMIT and
driven through its own `src.imputer.rectified_impute`, with every change made
here at runtime (nothing in the clone is edited):

  * `src.imputer` does per round what only its notebooks need. The MMD against
    the ground truth and the MINE estimate of I(X0; M) are diagnostics that
    never feed back into the imputation, and they are replaced by stubs; MMD
    also needs the true data, which an adapter may not have. The per-round
    `torch.save` of (X0, every model) to the working directory is dropped. The
    tqdm bars and prints are silenced (one log line per round instead).
  * `model_list` keeps every round's velocity net alive on the GPU. The model
    factory passed as `modelclass` moves the previous round's net to the CPU
    when the next round starts, and it is also where the wall-clock guard
    lives: it raises KeyboardInterrupt at a round boundary, which
    rectified_impute already catches and turns into "return X0 as of the last
    finished round".
  * The module-level `device` follows the `device` argument.
  * The package is named `src`, which other adapters may share, so it is
    imported under a clean sys.modules and removed again afterwards.
    `src/cnn.py` and `src/cnnrgb.py` run a test forward pass at import, which
    prints a tensor and consumes the global torch RNG: stdout is swallowed and
    the RNG state restored. IPython / matplotlib / tqdm are stubbed only when
    they are not installed (the module imports them but never needs them).

Velocity nets, as in the authors' `imputer_wrapper.impute_now`: the MLP
(3x1001, SiLU) for tabular data; `cnn.CNNNet` for square grayscale images and
`cnnrgb.CNNNet` (CHW flattening) for square RGB. The CNNs hard-code a square
side, so a non-square image (EYaleB 48x42) gets the same layers with a forward
that reshapes to (H, W) instead of (sqrt(d), sqrt(d)).

Data handling follows the paper: tabular features are standardised (from
observed entries only; idempotent on the harness's z-scored data) and missing
entries start as N(0,1) draws; images are scaled to [0,1] with the observed
global min/max, missing pixels start as U(0,1), and the ODE clamps to [0,1].
For tabular data the Euler ODE is kept inside each feature's observed range
(see _sample_ode: without it the ODE diverged to NaN on the synthetic set).
Everything is inverted before returning.

No model selection or early stopping: the number of rounds and epochs is a
fixed budget (BUDGETS), so the held-out entries are never looked at.
"""

import contextlib
import importlib
import importlib.util
import io
import math
import os
import sys
import time
import types

import numpy as np
import torch

from sota import SOTA_ROOT

REPO = "MIRI-Imputation"
REPO_URL = "https://github.com/yujhml/MIRI-Imputation"
COMMIT = "be49f94c2ebab431a2afe40a0228e01b857fc097"

# Fixed in advance, never tuned on held-out entries.
#
# "full" = the authors' demo settings (examples/*.ipynb): tabular (UCI demo)
# 10 rounds x 100 epochs at batch 50; images (CIFAR demo, n=5000) 15 rounds x
# 900 epochs at batch 500. Adam lr 0.01 and 100 Euler steps are hard-coded /
# defaults in rectified_impute.
#
# "reduced" (MIRI on COIL100 / Flowers / OxfordPet / HARUS / DSDD / VED): the
# same nets, batch sizes, lr and ODE, with a per-round cap on SGD steps so the
# cost stops growing with n, and 8 instead of 15 rounds for images.
#
# Cost model (RTX 3060 Laptop, shared with other jobs, so pessimistic):
#   CNN train step ~1.45 ms per 32x32 sample (~3.2 ms at 48x42), ODE ~40 ms per
#   sample per round; MLP step ~10 ms at batch 50 (launch/sync bound, assumed
#   NOT faster on the L40S). With the L40S at 2.5x for the CNNs:
#     full    ORL ~0.9 h, COIL20 ~3.1 h, EYaleB ~6.3 h (the tightest),
#             synthetic < 1 min
#     reduced COIL100 / Pet / Flowers: 8 x 4500 steps x 500 ~ 3.2 h
#             HARUS 10 x 20.6k steps ~ 0.6 h; DSDD / VED 10 x ~60k ~ 1.7 h
#   The image CNNs at batch 500 need ~4.5 GB (32x32) / ~7 GB (48x42) of VRAM.
#
# time_cap_h is a safety net only: a round is never started once the elapsed
# time plus the slowest round so far would pass it, and whatever the last
# finished round imputed is returned (info["hit_time_cap"] records it).
BUDGETS = {
    "full": {
        "tabular": dict(max_rounds=10, batchsize=50, maxepochs=100, odesteps=100,
                        max_steps_per_round=None),
        "image": dict(max_rounds=15, batchsize=500, maxepochs=900, odesteps=100,
                      max_steps_per_round=None),
        "time_cap_h": 7.0,
    },
    "reduced": {
        "tabular": dict(max_rounds=10, batchsize=50, maxepochs=100, odesteps=100,
                        max_steps_per_round=60_000),
        "image": dict(max_rounds=8, batchsize=500, maxepochs=900, odesteps=100,
                      max_steps_per_round=4_500),
        "time_cap_h": 7.0,
    },
}

_MIRI = None          # namespace of the imported, patched modules


# ------------------------------------------------------------------ import --
def _stub_missing(names):
    """Put do-nothing modules in sys.modules for imports the code never uses."""
    added = []
    for name in names:
        top = name.split(".")[0]
        if name in sys.modules:
            continue
        if top not in added and importlib.util.find_spec(top) is not None:
            continue
        mod = types.ModuleType(name)
        if name == "IPython.display":
            mod.clear_output = lambda *a, **k: None
            mod.display = lambda *a, **k: None
        if name == "tqdm":
            mod.tqdm = lambda it=None, *a, **k: it
        sys.modules[name] = mod
        added.append(name)
        if "." in name:
            parent = sys.modules.get(name.rsplit(".", 1)[0])
            if parent is not None:
                setattr(parent, name.rsplit(".", 1)[1], mod)
    return added


class _TorchNoSave(types.ModuleType):
    """`torch` as seen from src.imputer, with the per-round checkpoint disabled."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def save(*args, **kwargs):
        return None


def _load():
    global _MIRI
    if _MIRI is not None:
        return _MIRI
    repo = os.path.join(SOTA_ROOT, REPO)
    if not os.path.isfile(os.path.join(repo, "src", "imputer.py")):
        raise RuntimeError(
            f"MIRI code not found in {repo}. Fetch it with:\n"
            f"  git clone {REPO_URL} {repo} && git -C {repo} checkout {COMMIT}")

    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == "src" or k.startswith("src.")}
    stubs = _stub_missing(["IPython", "IPython.display", "matplotlib",
                           "matplotlib.pyplot", "tqdm"])
    rng = torch.random.get_rng_state()
    sys.path.insert(0, repo)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            imputer = importlib.import_module("src.imputer")
            mlp = importlib.import_module("src.mlp")
            cnn = importlib.import_module("src.cnn")          # runs a test at import
            cnnrgb = importlib.import_module("src.cnnrgb")    # likewise
    finally:
        torch.random.set_rng_state(rng)
        try:
            sys.path.remove(repo)
        except ValueError:
            pass
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            sys.modules.pop(k)
        sys.modules.update(saved)
        for k in stubs:
            sys.modules.pop(k, None)

    # Runtime patches (see the module docstring).
    imputer.mmd = lambda X, Y, *a, **k: 0.0
    imputer.MINE = lambda X, M, *a, **k: (0.0,)
    imputer.tqdm = lambda it=None, *a, **k: it
    imputer.print = lambda *a, **k: None
    imputer.torch = _TorchNoSave("torch")
    imputer.sample_ode = _sample_ode

    _MIRI = types.SimpleNamespace(imputer=imputer, MLP=mlp.MLP, CNN=cnn.CNNNet,
                                  CNNRGB=cnnrgb.CNNNet)
    return _MIRI


_ODE_STATS = {"nonfinite": 0}


@torch.no_grad()
def _sample_ode(model, z0=None, N=None, clamp=False):
    """src.imputer.sample_ode (Euler, N steps) with two safeguards.

    * clamp=True is the authors' [0, 1] clamp of the first block (images),
      unchanged. clamp=(lo, hi), per-feature tensors, is the same idea for
      tabular data: both state blocks are kept inside the observed range of
      each feature. Without it the unclamped Euler ODE blew up on synthetic
      200x50 in round 2 (|z| ~ 4e6, then inf, and (1 - m) * inf = NaN on the
      observed entries), and the NaNs reached every entry by round 3.
    * Any imputation that is still non-finite falls back to the value it had
      when the round started, so one bad row cannot poison the next round.
    """
    d = z0.shape[1] // 3
    dt = 1.0 / N
    start = z0[:, :d].clone()
    box = None
    if isinstance(clamp, tuple):
        box = tuple(b.to(z0.device) for b in clamp)
    for i in range(N):
        t = torch.full((z0.shape[0], 1), i / N, device=z0.device, dtype=z0.dtype)
        pred = model(z0, t)
        z0[:, :d] = z0[:, :d] + pred * dt
        z0[:, d:2 * d] = z0[:, d:2 * d] + pred * dt
        if box is not None:
            z0[:, :d] = torch.minimum(torch.maximum(z0[:, :d], box[0]), box[1])
            z0[:, d:2 * d] = torch.minimum(torch.maximum(z0[:, d:2 * d], box[0]), box[1])
        elif clamp:
            z0[:, :d] = torch.clamp(z0[:, :d], 0, 1)
    out = z0[:, :d]
    bad = ~torch.isfinite(out)
    if bad.any():
        _ODE_STATS["nonfinite"] += int(bad.sum())
        out[bad] = start[bad]
    return out


def _rect_cnn(base, C, H, W):
    """The authors' CNN (same layers) for a C x H x W image that is not square."""

    class RectCNN(base):
        def __init__(self, dimvec):
            super().__init__(dimvec)
            self.C, self.H, self.W = C, H, W

        def forward(self, x, t):
            n, P = x.size(0), self.C * self.H * self.W
            m = x[:, 2 * P:]
            x = x[:, :3 * P].reshape(n, 3 * self.C, self.H, self.W)
            t = t.reshape(n, 1, 1, 1).expand(-1, 1, self.H, self.W)
            out = self.initial_conv(torch.cat([x, t], dim=1))
            out = self.final_conv(self.residual_layers(out))
            return (1 - m) * out.reshape(n, -1)

    RectCNN.__name__ = f"{base.__module__}.{base.__name__}[{C}x{H}x{W}]"
    return RectCNN


class _Rounds:
    """The `modelclass` handed to rectified_impute: called once per round."""

    def __init__(self, cls, cap_s, log):
        self.cls, self.cap_s, self.log = cls, cap_s, log
        self.prev, self.t0, self.t_round = None, time.perf_counter(), None
        self.round_s, self.hit_cap = [], False

    def close_round(self):
        if self.prev is None:
            return
        self.prev.to("cpu")                    # model_list keeps it; free the GPU
        self.prev = None
        dt = time.perf_counter() - self.t_round
        self.round_s.append(dt)
        self.log(f"round {len(self.round_s)} done in {dt:.1f}s "
                 f"(elapsed {time.perf_counter() - self.t0:.0f}s)")

    def __call__(self, d):
        self.close_round()
        if self.round_s and self.cap_s:
            elapsed = time.perf_counter() - self.t0
            if elapsed + 1.05 * max(self.round_s) > self.cap_s:
                self.hit_cap = True
                self.log(f"time cap: stopping after {len(self.round_s)} rounds")
                raise KeyboardInterrupt  # rectified_impute returns X0 as it is
        self.t_round = time.perf_counter()
        self.prev = self.cls(d)
        return self.prev


# ------------------------------------------------------------------ adapter --
def _seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def impute(X, *, seed, K, profile="full", device="cuda"):
    X = np.asarray(X, dtype=np.float64)
    if profile not in BUDGETS:
        raise ValueError(f"profile must be one of {sorted(BUDGETS)}")
    log = lambda s: print(f"    [miri] {s}", flush=True)
    miri = _load()
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    miri.imputer.device = dev

    shape = X.shape
    B = shape[0]
    obs = ~np.isnan(X)
    if obs.all():
        return dict(X_hat=X.copy(), labels=None, info=dict(note="nothing missing"))

    # --- layout and scaling ------------------------------------------------
    is_image = X.ndim >= 3
    if is_image:
        img = X.reshape(B, shape[1], shape[2], -1)            # (B, H, W, C)
        H, W, C = img.shape[1:]
        flat = np.ascontiguousarray(img.transpose(0, 3, 1, 2)).reshape(B, -1)  # CHW
        lo, hi = float(np.nanmin(X)), float(np.nanmax(X))
        span = hi - lo if hi > lo else 1.0
        Z = (flat - lo) / span
        if H == W and C == 1:
            cls, net = miri.CNN, "cnn.CNNNet"
        elif H == W and C == 3:
            cls, net = miri.CNNRGB, "cnnrgb.CNNNet"
        else:
            cls = _rect_cnn(miri.CNNRGB if C == 3 else miri.CNN, C, H, W)
            net = f"{'cnnrgb' if C == 3 else 'cnn'}.CNNNet (rectangular {H}x{W}x{C})"
        kind, clamp = "image", True
    else:
        flat = X.reshape(B, -1)
        mu = np.nanmean(flat, axis=0)
        sd = np.nanstd(flat, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
        Z = (flat - mu) / sd
        cls, net, kind = miri.MLP, "mlp.MLP(3x1001)", "tabular"
        lo_f = np.nan_to_num(np.nanmin(Z, axis=0), nan=0.0)
        hi_f = np.nan_to_num(np.nanmax(Z, axis=0), nan=0.0)
        clamp = (torch.from_numpy(lo_f.astype(np.float32)),
                 torch.from_numpy(hi_f.astype(np.float32)))

    n, d = Z.shape
    Mnp = ~np.isnan(Z)
    bud = dict(BUDGETS[profile][kind])
    cap_h = BUDGETS[profile]["time_cap_h"]
    bs = int(bud["batchsize"])
    steps_per_epoch = math.ceil(n / bs)
    epochs = int(bud["maxepochs"])
    if bud["max_steps_per_round"]:
        epochs = max(1, min(epochs, bud["max_steps_per_round"] // steps_per_epoch))

    # --- initial fill (paper: N(0,1) for standardised data, U(0,1) for images)
    _seed(seed)
    g = torch.Generator().manual_seed(int(seed))
    noise = (torch.rand(n, d, generator=g, dtype=torch.float64) if is_image
             else torch.randn(n, d, generator=g, dtype=torch.float64)).numpy()
    Z0 = np.where(Mnp, Z, noise)
    X0 = torch.from_numpy(Z0.astype(np.float32))
    M = torch.from_numpy(Mnp.astype(np.float32))
    Xstar = X0.clone()                   # only read by the (stubbed) MMD

    log(f"{kind} n={n} d={d} net={net} rounds={bud['max_rounds']} epochs={epochs} "
        f"bs={bs} ({steps_per_epoch * epochs} steps/round) odesteps={bud['odesteps']} "
        f"device={dev}")
    rounds = _Rounds(cls, cap_h * 3600 if cap_h else None, log)
    _ODE_STATS["nonfinite"] = 0
    bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True     # fixed-shape convs; restored below
    try:
        out = miri.imputer.rectified_impute(
            X0, M, Xstar, rounds, max_rounds=int(bud["max_rounds"]), clamp=clamp,
            callback=None, verbose=False, batchsize=bs, maxepochs=epochs,
            odesteps=int(bud["odesteps"]), dsetname="miri")
        rounds.close_round()
        Xi = out[0]
        del out
    finally:
        torch.backends.cudnn.benchmark = bench
        rounds.prev = None
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    Zh = Xi.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(Zh[~Mnp]).all():
        bad = ~np.isfinite(Zh) & ~Mnp
        log(f"{int(bad.sum())} non-finite imputations replaced by the feature mean")
        Zh[bad] = np.broadcast_to(np.nanmean(np.where(Mnp, Z, np.nan), axis=0),
                                  Zh.shape)[bad]
    Zh = np.where(Mnp, Z, Zh)            # observed entries exactly as given

    if is_image:
        Xh = (Zh * span + lo).reshape(B, C, H, W).transpose(0, 2, 3, 1).reshape(shape)
        scale = dict(lo=lo, hi=hi)
    else:
        Xh = (Zh * sd + mu).reshape(shape)
        scale = dict(standardised=True)
    Xh = np.where(obs, X, Xh)

    info = dict(repo=REPO_URL, commit=COMMIT, profile=profile, kind=kind, net=net,
                max_rounds=int(bud["max_rounds"]), rounds_done=len(rounds.round_s),
                epochs_per_round=epochs, batchsize=bs,
                steps_per_round=steps_per_epoch * epochs, odesteps=int(bud["odesteps"]),
                lr=0.01, clamp="[0,1] (authors')" if is_image else "observed per-feature range",
                ode_nonfinite_reverted=_ODE_STATS["nonfinite"],
                time_cap_h=cap_h, hit_time_cap=rounds.hit_cap,
                round_s=[round(s, 2) for s in rounds.round_s], scale=scale,
                device=str(dev))
    return dict(X_hat=Xh, labels=None, info=info)
