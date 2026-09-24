"""
State-of-the-art completion / clustering baselines, run through the same
protocol as DeLUCA (same masks, same z-scoring, held-out-only metrics).

Each method lives in sota/<name>.py and exposes

    impute(X, *, seed, K, profile, device) -> dict(
        X_hat  = np.ndarray, same shape as X  (only held-out entries are scored),
        labels = np.ndarray of B ints, or None (methods that cluster themselves),
        info   = dict  (budget actually used, epochs, anything worth recording),
    )

X is float64, shape (B, ...), NaN = missing. Image datasets arrive in raw pixel
units, tabular ones z-scored from observed entries; an adapter does any extra
scaling it needs and inverts it. Adapters must never touch the held-out
entries: any tuning or early stopping uses a slice of the OBSERVED entries.

Third-party code is NOT vendored — several of these repos have no license or
are GPL, and this repository is public. It is fetched at pinned commits into
SOTA_ROOT (the Docker image puts it in /opt/sota; locally FinalProject/
third_party) and patched at runtime by the adapters. Extra Python packages for
local runs can go in SOTA_ROOT/_site (pip install --target).

`profile` selects a budget fixed in advance in each adapter's BUDGETS:
"full" everywhere a method is cheap, "reduced" for the expensive methods on the
large datasets (see EXPENSIVE / LARGE_DATASETS, and generate_jobs.py --exp sota).
"""

import importlib
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
SOTA_ROOT = os.environ.get("SOTA_ROOT",
                           os.path.normpath(os.path.join(_HERE, "..", "..", "third_party")))

# name -> (family, year/venue) — the order is the table order.
METHODS = {
    "diffputer": ("diffusion + EM", "ICLR 2025"),
    "cacti":     ("masked autoencoder (CMAE)", "ICML 2025"),
    "miri":      ("rectified flow", "NeurIPS 2025"),
    "newimp":    ("Wasserstein gradient flow", "NeurIPS 2024"),
    "aemc_ne":   ("autoencoder matrix completion", "ICLR 2024"),
    "refidiff":  ("diffusion refinement", "AAAI 2026"),
    "kfmc":      ("kernel high-rank matrix completion", "CVPR 2019"),
    "cagi":      ("joint imputation + clustering", "ECML-PKDD 2026"),
    "kfsc":      ("union-of-subspaces factorization, clusters + completes", "KDD 2021"),
    "altpzf":    ("subspace clustering + group low-rank completion", "ICCV-W 2019"),
    "kcsc":      ("kernel correction + spectral clustering (clusters only)", "NeurIPS 2023"),
}
# Methods whose per-run cost on the large datasets needs the reduced grid.
EXPENSIVE = {"diffputer", "cacti", "miri", "refidiff", "cagi", "altpzf"}
LARGE_DATASETS = {"COIL100", "Flowers", "OxfordPet", "HARUS", "DSDD", "VED"}
# (method, dataset) pairs that cannot run: both build an n x n self-expressive
# matrix, which is 13.7 GB at DSDD's 58,509 rows (VED's 85k is worse); kcsc
# only clusters, and those two datasets are completion-only.
SKIP = {("altpzf", "DSDD"), ("altpzf", "VED"), ("kcsc", "DSDD"), ("kcsc", "VED")}


def _paths():
    site = os.path.join(SOTA_ROOT, "_site")
    if os.path.isdir(site) and site not in sys.path:
        sys.path.insert(0, site)


def run(name, X, *, seed, K, profile="full", device="cuda"):
    """Returns (X_hat, labels_or_None, seconds, info)."""
    if name not in METHODS:
        raise ValueError(f"unknown SOTA method {name!r}; have {sorted(METHODS)}")
    _paths()
    mod = importlib.import_module(f"sota.{name}")
    t0 = time.perf_counter()
    out = mod.impute(X, seed=seed, K=K, profile=profile, device=device)
    took = time.perf_counter() - t0
    X_hat = out["X_hat"]
    if X_hat.shape != X.shape:
        X_hat = X_hat.reshape(X.shape)
    return X_hat, out.get("labels"), took, dict(out.get("info") or {})
