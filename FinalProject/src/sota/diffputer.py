"""
DiffPuter — EM-driven diffusion imputation (Zhang et al., ICLR 2025).

Official code: https://github.com/hengruizhang98/DiffPuter (MIT), pinned below
and fetched into SOTA_ROOT/DiffPuter; nothing of it is copied here.

What is theirs and what is ours
-------------------------------
From the repo, used unchanged: the denoiser `model.MLPDiffusion`, the EDM
preconditioning + loss `model.Model` (Precond, EDMLoss), the conditional
sampler `diffusion_utils.impute_mask` (Heun/EDM steps with N=20 RePaint-style
resampling per step), and `dataset.mean_std` (with the epsilon fix below).

`main.py` is an argparse script (it parses sys.argv at import time, reads
their CSV/mask files, writes checkpoints to ./ckpt and results to ./results),
so it cannot be imported; `_diffputer_em` below re-drives exactly its EM loop:

  normalise  X = (X - mean_obs) / std_obs / 2, missing entries set to 0
  for each EM round:
    M-step  fresh MLPDiffusion(F, 1024), Adam(lr=1e-4, wd=0),
            ReduceLROnPlateau(factor=0.9, patience=50) on the epoch train loss,
            mini-batches of 4096, keep the lowest-train-loss weights,
            early stop after 500 epochs without improvement;
            trained on the current completed matrix (round 0: mean-filled)
    E-step  num_trials x impute_mask(...) with the best weights, observed
            entries clamped back, averaged -> the completed matrix
  X_hat = rec_X * 2 * std + mean      (rec_X before main.py's own "* 2")

Runtime deviations from main.py (none of them changes the algorithm):
  - dataset.mean_std: a column whose observed entries are constant (image
    borders) or that has no observed entry gets std 0 and NaNs everywhere;
    such std is floored at 1e-8 (computed in float64, so the centred column
    is exactly 0 and its held-out entries map back to the observed constant).
  - DataLoader(num_workers=4, shuffle=True) -> a torch.randperm over rows of a
    GPU-resident tensor, same 4096 batches incl. the short last one (workers
    fail on Windows and are pure overhead for an in-memory array).
  - best weights kept as an in-memory copy instead of torch.save/torch.load
    to a shared ./ckpt path (which concurrent runs would overwrite); the
    every-1000-epoch snapshots are unused and dropped.
  - the E-step keeps a running sum over trials instead of torch.stack (B x F x
    trials on the GPU), and runs impute_mask on row chunks — rows are
    independent under the MLP and the per-row noise, so this only bounds VRAM.
  - no out-of-sample E-step and no get_eval: the protocol is transductive and
    the harness scores held-out entries itself, in original units.
  - fixed budget (BUDGETS) instead of max_iter=10 / num_trials=20 / 10001
    epochs, which are infeasible on the large datasets (research notes).

Held-out entries are never used: the only stopping rule is DiffPuter's own
train-loss patience, and the EM round count is fixed — the LAST round is
returned, never one picked by error. No clustering (labels=None): scored
two-stage by the harness.
"""

import importlib.util
import os
import random
import sys
import time

import numpy as np
import torch

from . import SOTA_ROOT

REPO = "DiffPuter"
REPO_URL = "https://github.com/hengruizhang98/DiffPuter"
COMMIT = "2fa55373655b9e910146d94820fc1012da0dfd75"

# Fixed before any result was seen; identical across datasets and missing rates
# within a profile. Architecture/optimiser/sampler settings are the repo's.
#   full    — research-notes budget: 4 EM rounds, 10 trials, 50 steps, 3000-epoch
#             cap (repo: 10 / 20 / 50 / 10001). Small datasets.
#   reduced — large datasets (COIL100, Flowers, OxfordPet, HARUS, DSDD, VED):
#             5 trials, 2000-epoch cap (the notes' Sensorless setting, ~4 h on
#             a 3060 -> ~1.6 h on an L40S; VED's 85k rows ~2.3 h).
_COMMON = dict(num_steps=50, hid_dim=1024, batch_size=4096, lr=1e-4,
               weight_decay=0.0, patience=500, lr_factor=0.9, lr_patience=50,
               # Safety net only (should never fire): stop starting new EM
               # rounds / E-step trials once the projected time passes this;
               # the last completed round is returned and info says so.
               time_limit_s=7.0 * 3600)
BUDGETS = {
    "full":    dict(_COMMON, max_iter=4, num_trials=10, max_epochs=3000),
    "reduced": dict(_COMMON, max_iter=4, num_trials=5, max_epochs=2000),
}

# E-step row chunk: keep the widest activation (rows x max(2*hid, F)) at
# <= 2^25 floats (128 MB), so VRAM stays well under 1 GB at any B.
_ESTEP_FLOATS = 2 ** 25

_MODS = None


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load():
    """Import DiffPuter's modules from SOTA_ROOT/DiffPuter.

    Loaded by file path under unique names (their generic names model /
    dataset / diffusion_utils would otherwise collide with other repos or our
    own modules). model.py does `from diffusion_utils import EDMLoss`, so that
    name is bound only while model.py executes.
    """
    global _MODS
    if _MODS is not None:
        return _MODS
    root = os.path.join(SOTA_ROOT, REPO)
    if not os.path.isfile(os.path.join(root, "model.py")):
        raise RuntimeError(
            f"DiffPuter not found at {root}. Fetch it with:\n"
            f"  git clone {REPO_URL} {root} && git -C {root} checkout {COMMIT}")
    du = _load_file("_diffputer_diffusion_utils", os.path.join(root, "diffusion_utils.py"))
    prev = sys.modules.get("diffusion_utils")
    sys.modules["diffusion_utils"] = du
    try:
        md = _load_file("_diffputer_model", os.path.join(root, "model.py"))
    finally:
        if prev is None:
            sys.modules.pop("diffusion_utils", None)
        else:
            sys.modules["diffusion_utils"] = prev
    ds = _load_file("_diffputer_dataset", os.path.join(root, "dataset.py"))
    _MODS = (md, du, ds)
    return _MODS


_STD_EPS = 1e-8


def _mean_std(ds, data, mask):
    """dataset.mean_std with the zero-std fix (constant / all-missing columns).

    Called on float64 data so the centring is exact for a constant column;
    its std is floored at a tiny epsilon (the research notes' "add an
    epsilon"), so whatever the model generates there maps back to the column's
    observed value. Flooring at 1 instead would leave the model's residual
    noise in raw pixel units (~0.5 grey level on 0..255, far more on 0..1).
    """
    mean, std = ds.mean_std(data, mask)
    std = np.where(np.isfinite(std) & (std > _STD_EPS), std, _STD_EPS)
    return mean, std


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _m_step(md, train_data, cfg, device):
    """main.py's M-step: train a fresh denoiser on the current completed matrix."""
    n, in_dim = train_data.shape
    denoise_fn = md.MLPDiffusion(in_dim, cfg["hid_dim"]).to(device)
    model = md.Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg["lr_factor"], patience=cfg["lr_patience"])
    model.train()
    bs = cfg["batch_size"]
    best_loss, patience, best_state, epoch = float("inf"), 0, None, -1
    for epoch in range(cfg["max_epochs"]):
        perm = torch.randperm(n, device=device)
        total = torch.zeros((), dtype=torch.float64, device=device)
        for s in range(0, n, bs):
            inputs = train_data[perm[s:s + bs]]
            loss = model(inputs).mean()
            total += loss.detach().double() * inputs.shape[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
        curr = float(total) / n
        sched.step(curr)
        if curr < best_loss:
            best_loss, patience = curr, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience == cfg["patience"]:
                break
    if best_state is not None:          # NaN loss from epoch 0 leaves it None
        model.load_state_dict(best_state)
    lr_end = float(opt.param_groups[0]["lr"])
    del opt, sched, best_state
    return model, dict(epochs=epoch + 1, best_loss=best_loss, lr_end=lr_end)


def _diffputer_em(X, miss, cfg, device, t_start):
    """EM loop of main.py on a (B, F) float64 array; miss is True = MISSING."""
    md, du, ds = _load()
    X0f = np.where(miss, 0.0, X)                          # float64, no NaN
    mean_X, std_X = _mean_std(ds, X0f, miss)
    Xn = np.where(miss, 0.0, (X0f - mean_X) / std_X / 2).astype(np.float32)

    B, F = Xn.shape
    X0 = torch.from_numpy(Xn).to(device)                 # (1 - mask) * X
    mask = torch.from_numpy(miss).to(device)
    mask_f = mask.float()
    chunk = max(1, min(B, _ESTEP_FLOATS // max(2 * cfg["hid_dim"], F)))

    rounds, cur, guard = [], X0, None
    for it in range(cfg["max_iter"]):
        el = time.perf_counter() - t_start
        if it > 0 and el + rounds[-1]["seconds"] > cfg["time_limit_s"]:
            guard = f"stopped before EM round {it}: {el:.0f}s elapsed"
            break
        t0 = time.perf_counter()
        # ---- M-step (round 0 trains on the mean-filled matrix)
        model, mi = _m_step(md, cur, cfg, device)
        t_m = time.perf_counter() - t0
        net = model.denoise_fn_D
        # ---- E-step: in-sample imputation, averaged over trials
        acc = torch.zeros_like(X0)
        done, t_e0 = 0, time.perf_counter()
        for trial in range(cfg["num_trials"]):
            for s in range(0, B, chunk):
                xs, ms = X0[s:s + chunk], mask[s:s + chunk]
                rec = du.impute_mask(net, xs, ms, xs.shape[0], F,
                                     cfg["num_steps"], device)
                acc[s:s + chunk] += rec * mask_f[s:s + chunk] + xs * (1 - mask_f[s:s + chunk])
            done += 1
            el = time.perf_counter() - t_start
            per = (time.perf_counter() - t_e0) / done
            if trial + 1 < cfg["num_trials"] and el + per > cfg["time_limit_s"]:
                guard = f"EM round {it}: stopped after {done} of {cfg['num_trials']} trials"
                break
        cur = acc / done
        del model, net, acc
        rounds.append(dict(round=it, epochs=mi["epochs"], best_train_loss=mi["best_loss"],
                           lr_end=mi["lr_end"], trials=done, m_step_s=round(t_m, 2),
                           e_step_s=round(time.perf_counter() - t_e0, 2),
                           seconds=round(time.perf_counter() - t0, 2)))
        r = rounds[-1]
        print(f"    [diffputer] EM round {it + 1}/{cfg['max_iter']}: {r['epochs']} epochs "
              f"(loss {r['best_train_loss']:.4f}) {r['m_step_s']:.0f}s, {r['trials']} trials "
              f"{r['e_step_s']:.0f}s", flush=True)
        if guard:
            break

    rec_X = cur.cpu().numpy().astype(np.float64)
    X_hat = rec_X * 2 * std_X + mean_X
    if not np.all(np.isfinite(X_hat)):
        # Non-finite only if training diverged; fall back to the observed mean
        # of the column for those entries rather than returning NaN.
        bad = ~np.isfinite(X_hat)
        X_hat[bad] = np.broadcast_to(mean_X, X_hat.shape)[bad]
    info = dict(em_rounds=len(rounds), rounds=rounds, estep_chunk_rows=int(chunk),
                time_guard=guard)
    return X_hat, info


def impute(X, *, seed, K=None, profile="full", device="cuda"):
    if profile not in BUDGETS:
        raise ValueError(f"unknown profile {profile!r}; have {sorted(BUDGETS)}")
    cfg = dict(BUDGETS[profile])
    t_start = time.perf_counter()
    dev = torch.device(device if (str(device).startswith("cpu")
                                  or torch.cuda.is_available()) else "cpu")
    _seed(seed)

    shape = X.shape
    Xf = np.asarray(X, dtype=np.float64).reshape(shape[0], -1)
    miss = np.isnan(Xf)
    X_hat, info = _diffputer_em(Xf, miss, cfg, dev, t_start)
    X_hat = np.where(miss, X_hat, Xf)           # observed entries exactly as given

    info.update(method="DiffPuter", repo=REPO_URL, commit=COMMIT, profile=profile,
                budget={k: cfg[k] for k in ("max_iter", "num_trials", "num_steps",
                                            "max_epochs", "patience", "hid_dim",
                                            "batch_size", "lr")},
                device=str(dev), seconds=round(time.perf_counter() - t_start, 2))
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return dict(X_hat=X_hat.reshape(shape), labels=None, info=info)
