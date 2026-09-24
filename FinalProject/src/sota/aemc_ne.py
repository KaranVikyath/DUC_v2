"""
AEMC-NE: Neuron-Enhanced AutoEncoder Matrix Completion
(Fan, Chen, Zhang & Ding, ICLR 2024).

Official code: the single file "Codes of AEMC-NE.py" in the ICLR 2024
supplementary zip (no public git repository, no license stated). It is
fetched, not vendored, into SOTA_ROOT/aemc_ne:

    curl -fsSL -o supplementary.zip <ZIP_URL>   (sha256 ZIP_SHA256)
    python -c "import zipfile; zipfile.ZipFile('supplementary.zip').extractall('.')"

Why the file is not imported directly: at module level it hard-codes
device cuda:0, sets the global default dtype, and then RUNS the paper's
synthetic experiment (5 trials x 2000 epochs on a 3000x300 matrix). So this
adapter parses it with `ast` and takes only the model class, `RCAutoRec`,
verbatim (no edits to its body). That class is written once into
SOTA_ROOT/_patched/aemc_ne/rcautorec.py (or a temp dir if SOTA_ROOT is
read-only) and imported from there.

What the adapter adds around the unchanged model, and why:
- Input: X is centred per column and divided by ONE global scale, both from
  the observed entries, so the tanh network sees unit-scale data and a zero
  input equals column-mean fill. The official script zero-fills missing
  entries and uses `!= 0` to find the observed ones; that is wrong for
  z-scored data (genuine zeros), so an explicit mask is passed instead.
- Training is the official full-batch AdamW loop and loss
  (||(X - f(X)) * M||_F^2 / |M|), computed over row chunks with gradient
  accumulation. The gradient is the same as the full-batch one; chunking
  only bounds memory on the large datasets.
- Model selection. The official script prints the minimum TEST error over
  epochs, which peeks at the test set. Here 5% of the OBSERVED entries are
  held out: they are zeroed in the input and dropped from the loss, and
  the weight decay and the number of epochs are chosen by the RMSE on
  them. The "full" profile then refits on all observed entries with the
  chosen pair. "reduced" keeps the best snapshot instead.
- Imputation: forward(X_in, mask = unobserved), exactly as the official
  evaluate() does. The element-wise network is applied to the imputed
  entries. Observed entries are returned unchanged.
"""

import ast
import copy
import hashlib
import importlib.util
import os
import tempfile
import time
import types

import numpy as np
import torch

from . import SOTA_ROOT

ZIP_URL = ("https://proceedings.iclr.cc/paper_files/paper/2024/file/"
           "6cd3ac24cdb789beeaa9f7145670fcae-Supplementary-Conference.zip")
ZIP_SHA256 = "75ab908bca5022fec047cadef9931f24ca31db535dfe891ddd62571d4d817963"
SRC_NAME = "Codes of AEMC-NE.py"
SRC_SHA256 = "a0c0688aecebe82f4f20d5881a375643663339a17f8491cc9e886c6daadcf55c"
REPO_DIR = os.path.join(SOTA_ROOT, "aemc_ne")

# Fixed before any run and never tuned on test entries.
#  main_hidden / enet_hidden / lr / epochs / optimizer are the official
#  script's (Section 5.1): main net 300-100-30-100-300 (tanh), element-wise
#  net 1-20-20-1 (tanh), AdamW lr 1e-2, 2000 full-batch epochs, wd 0.1.
#  wd_grid is centred on the official 0.1. The paper tunes its lambdas on 5%
#  of the training entries (Sec. 5.2/5.3). With AdamW's decoupled decay at
#  lr 1e-2 the weights settle near |w| ~ 1/wd, so values much above 3 only
#  shrink the net to a linear map. 3.0 was added after 1.0 won at the grid
#  edge on the validation entries of ORL (50% missing); test entries played
#  no part.
#  Each grid point trains the full 2000 epochs; the best step is taken
#  afterwards (evaluated every eval_every steps).
#  "reduced" (not normally used: this method is cheap) drops the refit and
#  the extreme grid points.
#  chunk_mfloats: the activation budget per row chunk, in millions of
#  float32. It changes memory only, never the result.
BUDGETS = {
    "full": dict(mod="NE", main_hidden=(300, 100, 30, 100, 300),
                 enet_hidden=(20, 20), lr=1e-2, epochs=2000,
                 wd_grid=(0.01, 0.1, 1.0, 3.0), val_frac=0.05, eval_every=10,
                 refit=True, chunk_mfloats=150),
    "reduced": dict(mod="NE", main_hidden=(300, 100, 30, 100, 300),
                    enet_hidden=(20, 20), lr=1e-2, epochs=2000,
                    wd_grid=(0.1, 1.0), val_frac=0.05, eval_every=10,
                    refit=False, chunk_mfloats=150),
}

_MODEL_CLASS = None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _extract_source():
    """Header + the RCAutoRec class, copied verbatim from the official file."""
    src_path = os.path.join(REPO_DIR, SRC_NAME)
    if not os.path.isfile(src_path):
        raise FileNotFoundError(
            f"AEMC-NE source not found at {src_path}. Fetch it with:\n"
            f"  mkdir -p {REPO_DIR} && cd {REPO_DIR} && curl -fsSL -o supplementary.zip "
            f"{ZIP_URL} && python -c \"import zipfile; "
            f"zipfile.ZipFile('supplementary.zip').extractall('.')\"")
    sha = _sha256(src_path)
    if sha != SRC_SHA256:
        raise RuntimeError(f"{src_path} has sha256 {sha}, expected {SRC_SHA256}")
    with open(src_path, encoding="utf-8") as f:
        text = f.read()
    tree = ast.parse(text)
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RCAutoRec"]
    if len(cls) != 1:
        raise RuntimeError("RCAutoRec class not found in the official AEMC-NE file")
    body = ast.get_source_segment(text, cls[0])
    header = (f"# AUTO-GENERATED by sota/aemc_ne.py from '{SRC_NAME}' "
              f"(sha256 {SRC_SHA256}).\n"
              "# Only the RCAutoRec class is kept, verbatim; the rest of the file\n"
              "# (cuda:0 device, global default dtype, the experiment that runs at\n"
              "# import) is dropped. Do not edit: it is regenerated when stale.\n"
              "import numpy as np\nimport torch\nfrom torch import nn\n\n\n")
    return header + body + "\n", sha


def _load_model_class():
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS
    code, _ = _extract_source()
    mod = None
    for d in (os.path.join(SOTA_ROOT, "_patched", "aemc_ne"),
              os.path.join(tempfile.gettempdir(), "sota_patched", "aemc_ne")):
        path = os.path.join(d, "rcautorec.py")
        try:
            os.makedirs(d, exist_ok=True)
            current = None
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    current = f.read()
            if current != code:
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(code)
                os.replace(tmp, path)
            spec = importlib.util.spec_from_file_location("_sota_aemc_ne_rcautorec", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            break
        except OSError:
            mod = None
    if mod is None:  # nowhere writable: same code, executed in memory
        mod = types.ModuleType("_sota_aemc_ne_rcautorec")
        exec(compile(code, os.path.join(REPO_DIR, SRC_NAME), "exec"), mod.__dict__)
    _MODEL_CLASS = mod.RCAutoRec
    return _MODEL_CLASS


def _seed_all(seed):
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _row_chunks(B, F, hidden, enet_hidden, mfloats):
    # Floats held per row in one training chunk: main-net activations and
    # grads, plus the element-wise net over (at most) all F entries of the row.
    per_row = 3 * sum(hidden) + 6 * F + F * (4 * sum(enet_hidden) + 8)
    rows = max(1, min(B, int(mfloats * 1e6) // per_row))
    return [(a, min(a + rows, B)) for a in range(0, B, rows)]


def _fit(Model, cfg, X_in, M_in, wd, n_steps, chunks, seed, device, val=None):
    """The official full-batch AdamW loop; returns (net, curve, best).

    X_in: (B, F) float32, zero wherever M_in is False. M_in: bool, the
    entries used both as input and in the loss. val: None, or (X_true,
    M_val) with the held-out observed entries (never input) scored every
    eval_every steps; best = (val RMSE, step, state_dict) at the best step.
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    F = X_in.shape[1]
    net = Model(F, list(cfg["main_hidden"]), list(cfg["enet_hidden"]), cfg["mod"])
    net = net.to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=wd)
    n_in = float(M_in.sum().item())
    curve, best = [], (float("inf"), 0, None)
    n_val = float(val[1].sum().item()) if val is not None else 0.0
    for step in range(1, n_steps + 1):
        net.train()
        opt.zero_grad(set_to_none=True)
        for a, b in chunks:
            x, m = X_in[a:b], M_in[a:b]
            pred = net.forward(x, m)
            loss = torch.sum(torch.mul(x - pred, m) ** 2) / n_in
            loss.backward()
        opt.step()
        if val is not None and (step % cfg["eval_every"] == 0 or step == n_steps):
            X_true, M_val = val
            se, ae = 0.0, 0.0
            net.eval()
            with torch.no_grad():
                for a, b in chunks:
                    mv = M_val[a:b]
                    pred = net.forward(X_in[a:b], ~M_in[a:b])
                    d = pred[mv] - X_true[a:b][mv]
                    se += float(torch.sum(d * d))
                    ae += float(torch.sum(torch.abs(d)))
            # The official loop has no divergence guard; a NaN step scores
            # +inf here and so can never be selected.
            rmse = (se / n_val) ** 0.5 if np.isfinite(se) else float("inf")
            mae = ae / n_val if np.isfinite(ae) else float("inf")
            curve.append((step, rmse, mae))
            if rmse < best[0]:
                best = (rmse, step, copy.deepcopy(net.state_dict()))
    return net, curve, best


def _predict(net, X_in, M_pred, chunks):
    out = torch.empty_like(X_in)
    net.eval()
    with torch.no_grad():
        for a, b in chunks:
            out[a:b] = net.forward(X_in[a:b], M_pred[a:b])
    return out


def impute(X, *, seed, K=None, profile="full", device="cuda"):
    t_start = time.perf_counter()
    cfg = BUDGETS[profile]
    Model = _load_model_class()
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    _seed_all(seed)

    shape = X.shape
    Xf = np.asarray(X, dtype=np.float64).reshape(shape[0], -1)
    B, F = Xf.shape
    obs = np.isfinite(Xf)
    if not obs.any():
        raise ValueError("no observed entries")

    # Scaling from observed entries only: per-column centre, one global scale.
    cnt = obs.sum(0)
    mu = np.where(cnt > 0, np.where(obs, Xf, 0.0).sum(0) / np.maximum(cnt, 1), 0.0)
    Xc = np.where(obs, Xf - mu, 0.0)
    scale = float(np.sqrt((Xc ** 2).sum() / obs.sum()))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    Xs = Xc / scale  # zero at unobserved entries

    # 5% of the observed entries held out for selection (never the NaN ones).
    rng = np.random.default_rng(seed)
    obs_idx = np.flatnonzero(obs.ravel())
    n_val = max(1, int(round(cfg["val_frac"] * obs_idx.size)))
    val_idx = rng.choice(obs_idx, size=n_val, replace=False)
    val = np.zeros(B * F, dtype=bool)
    val[val_idx] = True
    val = val.reshape(B, F)
    train = obs & ~val

    chunks = _row_chunks(B, F, cfg["main_hidden"], cfg["enet_hidden"], cfg["chunk_mfloats"])
    to_t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
    X_all = to_t(Xs)
    M_obs = to_t(obs, torch.bool)
    M_tr = to_t(train, torch.bool)
    M_val = to_t(val, torch.bool)
    X_tr = X_all * M_tr  # validation entries are zeroed in the input

    sel = {}
    best_wd, best_step, best_rmse, best_state = None, None, float("inf"), None
    t0 = time.perf_counter()
    for wd in cfg["wd_grid"]:
        _, curve, (rmse, step, state) = _fit(Model, cfg, X_tr, M_tr, wd, cfg["epochs"],
                                            chunks, seed, dev, val=(X_all, M_val))
        at_best = [c for c in curve if c[0] == step]
        sel[str(wd)] = dict(best_val_rmse=rmse, best_step=step,
                            best_val_mae=at_best[0][2] if at_best else float("nan"),
                            final_val_rmse=curve[-1][1] if curve else float("nan"))
        if rmse < best_rmse:
            best_wd, best_step, best_rmse, best_state = wd, step, rmse, state
    t_select = time.perf_counter() - t0
    if best_wd is None:
        raise RuntimeError(f"AEMC-NE diverged for every weight decay: {sel}")

    # Input = every observed entry; the element-wise net goes on the
    # unobserved ones (official evaluate(): forward(train, missing)).
    t0 = time.perf_counter()
    used = "best_snapshot"
    pred = None
    if cfg["refit"]:
        # Same init seed, all observed entries, the chosen wd and length.
        net, _, _ = _fit(Model, cfg, X_all, M_obs, best_wd, best_step, chunks, seed, dev)
        pred = _predict(net, X_all, ~M_obs, chunks).double().cpu().numpy()
        used = "refit"
        if not np.all(np.isfinite(pred[~obs])):  # diverged: keep the selected snapshot
            pred, used = None, "best_snapshot (refit diverged)"
    if pred is None:
        net = Model(F, list(cfg["main_hidden"]), list(cfg["enet_hidden"]), cfg["mod"])
        net = net.to(device=dev, dtype=torch.float32)
        net.load_state_dict(best_state)
        pred = _predict(net, X_all, ~M_obs, chunks).double().cpu().numpy()
    t_final = time.perf_counter() - t0
    del X_all, M_obs, M_tr, M_val, X_tr, net, best_state

    if not np.all(np.isfinite(pred[~obs])):
        raise RuntimeError("AEMC-NE produced non-finite imputations")
    X_hat = np.where(obs, Xf, pred * scale + mu).reshape(shape)

    info = dict(method="AEMC-NE", profile=profile, mod=cfg["mod"],
                main_hidden=list(cfg["main_hidden"]), enet_hidden=list(cfg["enet_hidden"]),
                lr=cfg["lr"], epochs_max=cfg["epochs"], wd_grid=list(cfg["wd_grid"]),
                chosen_wd=best_wd, chosen_steps=best_step, best_val_rmse_scaled=best_rmse,
                selection=sel, refit=cfg["refit"], final_model=used, val_entries=int(n_val),
                n_chunks=len(chunks), device=str(dev), scale=scale,
                select_s=t_select, final_s=t_final,
                total_s=time.perf_counter() - t_start, source_sha256=SRC_SHA256)
    return dict(X_hat=X_hat, labels=None, info=info)
