"""
DeLUCA v3 — low-rank / low-VRAM variant, built on the v2 CUDA stack.

Two memory wins over DeLUCA.py, both targeting the B**2 terms (B = batch/sample count):

  1. LowRankPseudoCompletion:
      The per-feature mixing weight W_f (B, B) is factorized as  W_f = B_mat_f @ A_f  with
      A (F, r_p, B) and B_mat (F, B, r_p), so memory is 2*F*B*r_p instead of F*B**2.

  2. EfficientCFSModule:
      The CFS layer never materializes the dense (B, B) `Coef = V_rank @ V_rank.T`.
      In pure-completion (CFS) mode `Coef` is *not* in the loss or the gradient — `PZ` (fwd)
      and `d_Z` (bwd) both factor through `V_rank (B, rank)`. So we never build the B**2
      matrix during training. It is rebuilt once, on demand, at the very end via
      `coefficient_matrix()` / `cluster()` for the clustering step only.

Everything on the training path is GPU-parallel and reuses the v2 kernels:

  stage                     v3 implementation                              parallel via
  ------------------------  ---------------------------------------------  ---------------
  pseudo-completion matmul  2 x torch.bmm (rank-r_p bottleneck)             cuBLAS batched
  pseudo bias+PReLU+transp  pseudo_completion_cuda.hybrid_forward/backward  v2 fused kernel
  encoder / decoder         Conv2d / Linear (unchanged from v2)             cuDNN / cuBLAS
  CFS projection            Gram + eigh in F_enc-space (see EfficientCFS)   cuBLAS / LAPACK
  masked reconstruction     MaskedLossFn                                    v2 fused kernel

Measured and REJECTED (see bench_v1_v3.py --parts opt for the numbers):
  * v2's cuSOLVER CFS kernel — slower here (its backward recomputes the SVD, so two
    per iteration) and it allocates the dense (B, B) Coef internally.
  * Adam(fused=True) — slower than the default foreach path at these parameter counts.
  * TF32 matmul — no effect; these matmuls are launch-bound, not compute-bound.
  * svd_lowrank(niter=0) — 3e-2 subspace error, unusable.
"""

import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from custom_funcs import convert_nan, thrC, post_proC
from DeLUCA import (Encoder, Decoder, MaskedLossFn, CusolverCFSFn,
                    _load_masked_loss_cuda, _load_pseudo_cuda, _load_cfs_cuda)


RECON_LOSSES = ("frobenius", "mse", "rmse", "mae", "huber", "logcosh", "msre")


def masked_reconstruction_loss(x, Xc, decoded, kind="frobenius"):
    """Three-term masked reconstruction loss under a choice of norm.

    The DUC objective compares three quantities on the OBSERVED entries only:
        d1 = Xc  - x     pseudo-completion vs data
        d2 = Xc  - dec   pseudo-completion vs decoder
        d3 = dec - x     decoder vs data
    v1/v2 sum the Frobenius norm of each. This swaps that norm out while keeping
    the three-term structure identical, so the comparison isolates the norm.

    Mean-based variants divide by the number of OBSERVED entries, not the total,
    so their scale does not drift with the missing-data rate.

    Note these do not share a gradient scale: MAE has bounded gradients, MSE's
    grow with the error, and the Frobenius norm self-normalises by its own value.
    At a fixed learning rate a comparison between them is partly a comparison of
    effective step size — see the `lr` column in the sweep.
    """
    mask = ~torch.isnan(x)
    m = mask.to(Xc.dtype)
    n = m.sum().clamp(min=1.0)
    xo = torch.nan_to_num(x, nan=0.0).to(Xc.dtype)

    d1 = (Xc - xo) * m
    d2 = (Xc - decoded) * m
    d3 = (decoded - xo) * m
    ds = (d1, d2, d3)

    if kind == "frobenius":
        return sum(torch.norm(d, p="fro") for d in ds)
    if kind == "mse":
        return sum((d ** 2).sum() / n for d in ds)
    if kind == "rmse":
        # = frobenius / sqrt(n): same direction, different scale. Included as a
        # control to separate "norm choice" from "effective learning rate".
        return sum(torch.sqrt((d ** 2).sum() / n + 1e-12) for d in ds)
    if kind == "mae":
        return sum(d.abs().sum() / n for d in ds)
    if kind == "huber":
        return sum(torch.nn.functional.smooth_l1_loss(
            d, torch.zeros_like(d), reduction="sum") / n for d in ds)
    if kind == "logcosh":
        # log(cosh z) written stably: |z| + softplus(-2|z|) - log 2
        def _lc(d):
            a = d.abs()
            return (a + torch.nn.functional.softplus(-2.0 * a)
                    - float(np.log(2.0))).sum() / n
        return sum(_lc(d) for d in ds)
    if kind == "msre":
        # Relative to the observed value. The reference for d2 (Xc vs decoder)
        # is not the data, so |x| is used for all three terms for consistency.
        den = xo.abs() + 1.0
        return sum(((d / den) ** 2).sum() / n for d in ds)
    raise ValueError(f"unknown reconstruction loss: {kind}")


class LowRankPseudoCompletionFn(torch.autograd.Function):
    """cuBLAS bmm chain + v2's fused bias/PReLU/transpose kernel.

    forward   tmp = A @ x_t          (F, r, 1)   contract samples -> rank
              pre = B_mat @ tmp      (F, B, 1)   expand rank -> samples
              out = PReLU(pre + bias) transposed to (B, F)   [one fused kernel]

    backward  dz  = PReLU'(.) * grad (F, B)      [one fused kernel]
              d_B_mat = dz  @ tmp^T  (F, B, r)
              d_tmp   = B_mat^T @ dz (F, r, 1)
              d_A     = d_tmp @ x_t^T(F, r, B)
    Every tensor here is O(F*B*r) — no B**2 object is ever allocated.
    """

    @staticmethod
    def forward(ctx, x_clean, A, B_mat, bias, prelu_w):
        x_t = x_clean.t().unsqueeze(2).contiguous()      # (F, B, 1)
        tmp = torch.bmm(A, x_t)                          # (F, r, 1)
        pre = torch.bmm(B_mat, tmp).squeeze(2)           # (F, B)

        ext = _load_pseudo_cuda()
        out, pre_act = ext.hybrid_forward(pre.contiguous(), bias, prelu_w)
        ctx.save_for_backward(x_t, A, B_mat, tmp, prelu_w, pre_act)
        return out                                       # (B, F)

    @staticmethod
    def backward(ctx, grad_out):
        x_t, A, B_mat, tmp, prelu_w, pre_act = ctx.saved_tensors
        ext = _load_pseudo_cuda()
        dz, d_prelu = ext.hybrid_backward(grad_out.contiguous(), prelu_w, pre_act)

        dz3 = dz.unsqueeze(2)                                 # (F, B, 1)
        d_B_mat = torch.bmm(dz3, tmp.transpose(1, 2))         # (F, B, r)
        d_tmp = torch.bmm(B_mat.transpose(1, 2), dz3)         # (F, r, 1)
        d_A = torch.bmm(d_tmp, x_t.transpose(1, 2))           # (F, r, B)
        return None, d_A, d_B_mat, dz, d_prelu


class LowRankPseudoCompletion(nn.Module):
    """Pseudo-completion with low-rank per-feature weight  W_f = B_mat_f @ A_f.

    weight (F, B, B)  ->  A (F, r_p, B) + B_mat (F, B, r_p)   [2*F*B*r_p vs F*B**2]
    """

    def __init__(self, input_shape, flat_layer_size, rank_pseudo):
        super().__init__()
        self.input_shape = input_shape
        self.feature_size = int(np.prod(input_shape[1:]))
        self.batch_size = flat_layer_size[0]
        self.rank_pseudo = int(rank_pseudo)

        F, B, r = self.feature_size, self.batch_size, self.rank_pseudo
        self.A = nn.Parameter(torch.empty(F, r, B))      # contract samples -> rank
        self.B_mat = nn.Parameter(torch.empty(F, B, r))  # expand rank -> samples
        self.bias = nn.Parameter(torch.zeros(F, B))
        self.prelu_weight = nn.Parameter(torch.full((F, 1), 0.25))

        # Match Var(W_ij) of the original kaiming_uniform_ (~2/B):
        #   Var(W) = r * sigma_A**2 * sigma_B**2  =>  sigma = (2/(B*r))**0.25
        std = (2.0 / (B * r)) ** 0.25
        nn.init.normal_(self.A, std=std)
        nn.init.normal_(self.B_mat, std=std)

    def forward(self, x):
        # x: (B, *features) -> (B, F)
        x = x.reshape(self.batch_size, -1).float()
        x = torch.nan_to_num(x, nan=0.0)

        ext = _load_pseudo_cuda()
        if x.is_cuda and ext is not None and hasattr(ext, "hybrid_forward"):
            out = LowRankPseudoCompletionFn.apply(
                x, self.A, self.B_mat, self.bias, self.prelu_weight)
            return out.reshape(self.input_shape)

        # Portable fallback (CPU, or extension not built)
        x_t = x.t().unsqueeze(2)
        tmp = torch.bmm(self.A, x_t)
        out = torch.bmm(self.B_mat, tmp).squeeze(2) + self.bias
        out = torch.clamp(out, min=0) + self.prelu_weight * torch.clamp(out, max=0)
        return out.t().reshape(self.input_shape)


class FullLowRankPseudoCompletion(nn.Module):
    """Rank-r mixing over the ENTIRE flattened matrix, not per feature.

    The default pseudo-completion is block-diagonal in feature space: feature f
    gets its own (B, B) map, so samples mix within a feature but features never
    talk to each other. This variant flattens X to a vector of length N = B*F
    and learns one N x N map, factorized as U V^T:

        out = U (V^T vec(X))        U, V : (N, r)

    Parameter count is 2*N*r = 2*B*F*r — identical to the block-diagonal
    version at the same rank (2*F*B*r_p), so the two are directly comparable at
    equal budget, and both stay linear in B. What differs is the structure:
    r global statistics pooled across everything, versus F independent
    per-feature sample-mixers. The full version can represent cross-feature
    dependencies the block-diagonal one cannot; the block-diagonal version
    spends its capacity on sample mixing, which is what the subspace model
    actually needs. Which matters more is an empirical question.
    """

    def __init__(self, input_shape, flat_layer_size, rank_pseudo):
        super().__init__()
        self.input_shape = input_shape
        self.feature_size = int(np.prod(input_shape[1:]))
        self.batch_size = flat_layer_size[0]
        self.rank_pseudo = int(rank_pseudo)

        F, B, r = self.feature_size, self.batch_size, self.rank_pseudo
        N = B * F
        self.N = N
        self.U = nn.Parameter(torch.empty(N, r))
        self.V = nn.Parameter(torch.empty(N, r))
        self.bias = nn.Parameter(torch.zeros(N))
        self.prelu_weight = nn.Parameter(torch.full((F,), 0.25))

        # Each output pools over N inputs here (vs B in the block-diagonal
        # case), so match Var(W_ij) ~ 2/N rather than 2/B.
        std = (2.0 / (N * r)) ** 0.25
        nn.init.normal_(self.U, std=std)
        nn.init.normal_(self.V, std=std)

    def forward(self, x):
        x = x.reshape(1, -1).float()
        x = torch.nan_to_num(x, nan=0.0)
        t = x @ self.V                       # (1, r)  — pool everything
        out = (t @ self.U.t()).reshape(-1) + self.bias      # (N,)
        out = out.view(self.batch_size, self.feature_size)  # row-major: i = b*F + f
        out = torch.clamp(out, min=0) + self.prelu_weight * torch.clamp(out, max=0)
        return out.reshape(self.input_shape)


class EfficientCFSModule(nn.Module):
    """CFS subspace projection that never builds the dense (B, B) Coef.

    Backends (all give the same subspace; see bench_v1_v3.py --parts cfs for numbers):

      "gram" (default)  Z has far more rows than columns (B >> F_enc), so the whole
                        projection can run in F_enc-space instead of B-space:
                            G  = Z^T Z          (F_enc, F_enc)  -- tiny
                            G  = U S^2 U^T      (symmetric eigendecomposition)
                            PZ = Z U_r U_r^T                    -- no B-sized factor
                        because V = Z U S^-1 gives V V^T Z = Z U U^T. Exact (not
                        randomized), ~2.2x faster than svd_lowrank fwd+bwd, and ~4x
                        more accurate on the subspace it returns.
      "lowrank"         the original torch.svd_lowrank randomized SVD.
      "cusolver"        v2's cuSOLVER gesvdj kernel. Measured slower here (its
                        backward recomputes the SVD, so two per iteration) and it
                        allocates the dense (B, B) Coef internally.

    `eigh_device="cpu"` runs the (F_enc, F_enc) eigendecomposition on the host:
    cuSOLVER is setup-bound at these sizes and the CPU is ~4.5x faster for a 40x40.

    V_rank (B, rank) is never formed during training — only `Z @ U_r` (B, rank) and
    the eigenvalues are kept, and V is reconstructed on demand for clustering.
    """

    # Measured crossover for torch.linalg.eigh on an (n x n) Gram, RTX 3060:
    # CPU wins 8x at n=40 and 1.4x at n=512; GPU wins 2.3x at n=900 and 7.3x
    # at n=3840. Below this size the launch/setup cost of cuSOLVER dominates.
    EIGH_CPU_MAX = 512

    def __init__(self, rank, svd_niter=2, backend="gram", eigh_device="auto"):
        super().__init__()
        if backend not in ("gram", "lowrank", "cusolver"):
            raise ValueError(f"unknown cfs backend: {backend}")
        if eigh_device not in ("auto", "cpu", "gpu"):
            raise ValueError(f"unknown eigh_device: {eigh_device}")
        self.rank = rank
        self.svd_niter = svd_niter
        self.backend = backend
        self.eigh_device = eigh_device
        self._ZU = None      # (B, rank) = Z @ U_r      (F_enc-space path)
        self._lam = None     # (rank,)   top eigenvalues (F_enc-space path)
        self._V = None       # (B, rank) V_rank directly (B-space / other backends)

    def _eigh(self, G):
        n = G.shape[0]
        use_cpu = (self.eigh_device == "cpu"
                   or (self.eigh_device == "auto" and n <= self.EIGH_CPU_MAX))
        if use_cpu and G.is_cuda:
            w, U = torch.linalg.eigh(G.cpu())
            return w.to(G.device), U.to(G.device)
        return torch.linalg.eigh(G)

    def forward(self, Z):
        # Z: (B, F_enc)
        if self.backend == "gram":
            B, Fe = Z.shape
            r = min(self.rank, Fe, B)
            if Fe <= B:
                # Small latent, many samples (telemetry, synthetic): work in
                # F_enc-space so nothing here scales with B.
                G = Z.t() @ Z                   # (F_enc, F_enc)
                w, U = self._eigh(G)            # ascending eigenvalues
                Ur = U[:, -r:]                  # top-r eigenvectors
                ZU = Z @ Ur                     # (B, r)
                PZ = ZU @ Ur.t()                # = Z U_r U_r^T
                self._ZU = ZU.detach()
                self._lam = w[-r:].detach().clamp_min(1e-12)
                self._V = None
            else:
                # Large latent, fewer samples (COIL20's single conv leaves 3840
                # dims for 1440 images): an (F_enc, F_enc) Gram would cost
                # O(F_enc^3) per step — 4.5 s at 3840. The (B, B) Gram is the
                # smaller matrix here, and since B < F_enc it is never larger
                # than Z itself, so this cannot reintroduce the B**2 problem
                # at scale. Its top eigenvectors are V_rank directly.
                G = Z @ Z.t()                   # (B, B), B < F_enc
                w, V = self._eigh(G)
                Vr = V[:, -r:]                  # (B, r) = V_rank
                PZ = Vr @ (Vr.t() @ Z)
                self._V = Vr.detach()
                self._ZU = None
                self._lam = None
            return PZ, None

        if self.backend == "cusolver" and Z.is_cuda and _load_cfs_cuda() is not None:
            PZ, Coef = CusolverCFSFn.apply(Z, self.rank)
            with torch.no_grad():
                _, _, V = torch.svd_lowrank(Z.t(), q=self.rank, niter=self.svd_niter)
            self._V = V.detach()
            return PZ, None

        U, S, V = torch.svd_lowrank(Z.t(), q=self.rank, niter=self.svd_niter)
        # detach: the projection needs the graph, the stored statistic does not.
        # Keeping the graph here would pin the previous iteration's activations.
        self._V = V.detach()
        PZ = V @ (V.t() @ Z)                    # (B, F_enc) — no (B, B) ever formed
        return PZ, None

    @property
    def V_rank(self):
        """(B, rank) orthonormal basis — reconstructed on demand, not each step."""
        if self._V is not None:
            return self._V
        if self._ZU is None:
            return None
        return self._ZU / self._lam.sqrt()

    @torch.no_grad()
    def affinity_tiled(self, tile=4096, threshold_fn=None):
        """Build C = V_rank @ V_rank.T lazily in row-tiles (for clustering only).

        C is the CFS projector P = V V^T. P is symmetric, so this equals the
        `Coef = P.T` that v1/v2 return. Returns a dense (B, B) on CPU.
        Never call this during training — it is the one place a B**2 object appears.
        """
        V = self.V_rank
        if V is None:
            raise RuntimeError("affinity_tiled called before a forward pass")
        B = V.shape[0]
        C = torch.empty(B, B, device="cpu", dtype=torch.float32)
        for i in range(0, B, tile):
            j = min(i + tile, B)
            row = (V[i:j] @ V.t())  # (tile, B) on GPU
            if threshold_fn is not None:
                row = threshold_fn(row)
            C[i:j] = row.cpu()
        return C


class EncoderNoNaN(Encoder):
    """Encoder without the per-iteration convert_nan.

    The encoder's input is Xc, the PReLU output of pseudo-completion, whose own
    input has already been through nan_to_num — so Xc never contains NaN and the
    isnan + two where() kernels are pure overhead (~0.18 ms/iter).
    """

    def forward(self, x):
        h = x.float()
        if not self.kernel_size:
            for layer in self.fc_layers:
                h = layer(h)
            return h
        h = h.permute(0, 3, 1, 2)
        for layer in self.fc_layers:
            h = layer(h)
        return h.permute(0, 2, 3, 1)


class DeLUCAV3(nn.Module):
    """Low-rank, low-VRAM DeLUCA (CFS / pure completion).

    Constructor-compatible with DeLUCA plus:
        rank_pseudo : rank r_p of the factorized pseudo-completion weight
                      (None -> min(B, max(2*rank, 8))). Lower = less VRAM.
        cfs_backend : "gram" (default, exact + fastest), "lowrank", or "cusolver".
        eigh_device : "cpu" (default) or "gpu" for the tiny (F_enc, F_enc) eigh.
        grad_clip   : max grad norm, or None (default) for no clipping.
        skip_nan_check     : use EncoderNoNaN (safe — Xc cannot contain NaN).
        track_autoenc_loss : compute 0.5*||Z-PZ||_F. It is logging-only in CFS mode,
                             so it is off by default.
        defer_output       : keep the decoded output on device during training and
                             copy it to host once, via `completed_data()`.
        log_every          : TensorBoard scalar interval (0 disables logging).

    Neither the coefficient matrix nor the completed output is materialized during
    training. Call `completed_data()`, `coefficient_matrix()` or `cluster()` after
    the loop.
    """

    def __init__(self, input_shape, flat_layer_size, enc_layer_size, deco_layer_size,
                 kernel_size, output_padding, lr, K, rank, reg_const1=1.0, reg_const2=1.0,
                 batch_size=200, model_path=None, logs_path="",
                 cluster_model="CFS", device="CPU",
                 rank_pseudo=None, cfs_backend="gram", eigh_device="cpu",
                 grad_clip=None, skip_nan_check=True, track_autoenc_loss=False,
                 defer_output=True, log_every=1,
                 loss_fn="frobenius", use_fused_loss=True,
                 pseudo_mode="blockdiag"):
        super().__init__()
        if loss_fn not in RECON_LOSSES:
            raise ValueError(f"loss_fn must be one of {RECON_LOSSES}")
        self.loss_fn = loss_fn
        # The v2 CUDA kernel implements the Frobenius form only.
        self.use_fused_loss = use_fused_loss and loss_fn == "frobenius"
        if cluster_model != "CFS":
            raise NotImplementedError("deluca_v3 implements the CFS / completion path only")

        self.input_shape = input_shape
        self.n_features = input_shape[1]
        self.model_path = model_path
        self.iter = 0
        self.batch_size = batch_size
        self.reg1 = reg_const1
        self.reg2 = reg_const2
        self.lr = lr
        self.kernel_size = kernel_size
        self.flat_layer_size = flat_layer_size
        self.enc_layer_size = enc_layer_size
        self.deco_layer_size = deco_layer_size
        self.cluster_model = cluster_model
        self.device = device
        self.rank = rank
        self.K = K
        self.grad_clip = grad_clip
        self.track_autoenc_loss = track_autoenc_loss
        self.defer_output = defer_output
        self.log_every = log_every

        if rank_pseudo is None:
            rank_pseudo = min(batch_size, max(2 * rank, 8))
        self.rank_pseudo = rank_pseudo

        # Layers (Encoder/Decoder reused from v2)
        enc_cls = EncoderNoNaN if skip_nan_check else Encoder
        if pseudo_mode not in ("blockdiag", "full"):
            raise ValueError(f"pseudo_mode must be blockdiag or full: {pseudo_mode}")
        self.pseudo_mode = pseudo_mode
        pseudo_cls = (LowRankPseudoCompletion if pseudo_mode == "blockdiag"
                      else FullLowRankPseudoCompletion)
        self.pseudo = pseudo_cls(input_shape, flat_layer_size, rank_pseudo)
        self.encoder = enc_cls(input_shape, enc_layer_size, kernel_size)
        self.CFS_module = EfficientCFSModule(rank, backend=cfs_backend,
                                             eigh_device=eigh_device)
        self.decoder = Decoder(enc_layer_size[-1], deco_layer_size, kernel_size, output_padding)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # No `verbose=`: deprecated in torch 2.2 and REMOVED in 2.7, so passing it
        # is a TypeError on the CHTC container image even though it still works
        # on 2.6 locally. verbose=False was the default behaviour anyway.
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9,
                                           patience=10)
        self.summary_writer = SummaryWriter(logs_path)

        self._x_src = None      # identity of the cached input array
        self._x_gpu = None      # its device-resident copy
        self._last_decoded = None   # device-resident output, copied out on demand

    def _to_device(self, x):
        """Upload the (constant) input once instead of on every iteration."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if self._x_src is not x:
            self._x_gpu = torch.as_tensor(x).to(self.device)
            self._x_src = x
        return self._x_gpu

    def forward(self, x):
        x = self._to_device(x)
        Xc = self.pseudo(x)
        Z = self.encoder(Xc)
        Z_flat = Z.view(self.batch_size, -1)

        PZ, _V = self.CFS_module(Z_flat)
        PZ = PZ.view(Z.shape)
        decoded = self.decoder(PZ)

        _ml_ext = _load_masked_loss_cuda()
        if self.use_fused_loss and x.is_cuda and _ml_ext is not None:
            reconstruction_loss = MaskedLossFn.apply(x, Xc, decoded)
        else:
            reconstruction_loss = masked_reconstruction_loss(
                x, Xc, decoded, self.loss_fn)
        # Logging-only in CFS mode — it is not part of total_loss, so skip by default.
        autoencoder_loss = (0.5 * torch.norm(Z - PZ, p='fro')
                            if self.track_autoenc_loss else None)

        # CFS / pure completion: clustering coefficient is not part of the loss.
        total_loss = reconstruction_loss
        return decoded, total_loss, autoencoder_loss, reconstruction_loss

    def finetune_fit(self, x):
        """One training step. Returns (None, loss, completed_data_or_None, lr).

        Slot 0 is always None: v1/v2 return the dense (B,B) coefficient matrix here,
        which is precisely the B**2 allocation v3 exists to avoid. Slot 2 is None
        when `defer_output` is set. Use `completed_data()`, `coefficient_matrix()`
        or `cluster()` after the loop.
        """
        self.optimizer.zero_grad()
        decoded, total_loss, autoencoder_loss, reconstruction_loss = self.forward(x)
        total_loss.backward()
        # Clip AFTER backward — v1/v2 call this before backward, where the grads
        # are still None/stale, so their clipping silently never takes effect.
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        self.scheduler.step(total_loss)

        with torch.no_grad():
            self._last_decoded = decoded.detach()
            complete_data = None if self.defer_output else self.completed_data()
            # One device sync per step (the LR schedule needs the value anyway).
            total_loss = total_loss.item()
            lr = self.optimizer.param_groups[0]['lr']

        self.iter += 1
        if self.log_every and self.iter % self.log_every == 0:
            self.summary_writer.add_scalar('Total Loss', total_loss, self.iter)
            if autoencoder_loss is not None:
                self.summary_writer.add_scalar('Autoencoder Loss',
                                               autoencoder_loss.item(), self.iter)
        return None, total_loss, complete_data, lr

    @torch.no_grad()
    def completed_data(self):
        """Host copy of the most recent decoded output."""
        if self._last_decoded is None:
            raise RuntimeError("completed_data called before a training step")
        return self._last_decoded.cpu().numpy()

    # --- clustering: the only place a (B, B) object is ever built --------------
    @torch.no_grad()
    def coefficient_matrix(self, tile=4096):
        """Materialize C = P = V_rank @ V_rank.T once, on CPU, as numpy."""
        return self.CFS_module.affinity_tiled(tile=tile).numpy()

    @torch.no_grad()
    def cluster(self, K=None, d=None, alpha1=1.0, alpha2=1.0, tile=4096):
        """Spectral clustering on C, materializing and releasing C inside."""
        K = self.K if K is None else K
        d = max(1, self.rank // max(K, 1)) if d is None else d
        C = self.coefficient_matrix(tile=tile)
        labels, L = post_proC(thrC(C, alpha1), K, d, alpha2)
        del C
        return labels


# --------------------------------------------------------------------------- #
# Smoke test + VRAM probe (synthetic). Run:  py -3.10 deluca_v3.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from custom_funcs import generate_data, missing_data_generation, err_rate

    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    np.random.seed(17)

    (data, full_data, input_shape, batch_size, total_datapoints, flat_layer_size,
     enc_layer_size, deco_layer_size, kernel_size, output_padding, K, reg1, reg2,
     alpha1, alpha2, d, lr, rank, true_labels) = generate_data(m=50, n=40, r=5, k=5, noise=1)

    missing = missing_data_generation(data, int(0.30 * total_datapoints))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _here = os.path.dirname(os.path.abspath(__file__))
    logs = os.path.join(_here, "..", "data", "logs", "deluca_v3_smoke")

    model = DeLUCAV3(
        input_shape, flat_layer_size, enc_layer_size, deco_layer_size,
        kernel_size, output_padding, lr, K, rank,
        reg_const1=reg1, reg_const2=reg2, batch_size=batch_size,
        logs_path=logs, cluster_model="CFS", device=device,
        rank_pseudo=10)
    model.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    stopping_lr = lr / 10
    data_norm = np.linalg.norm(full_data)
    it = 0
    while True:
        _, cost, _, cur_lr = model.finetune_fit(missing)
        it += 1
        if cur_lr < stopping_lr or it > 2000:
            break

    complete_data = model.completed_data()
    complete_data[~np.isnan(missing)] = missing[~np.isnan(missing)]
    acc = 1 - np.linalg.norm(complete_data - full_data) / data_norm
    labels = model.cluster(K=K, d=d, alpha1=alpha1, alpha2=alpha2)
    cacc = 1 - err_rate(true_labels, labels)
    print(f"iters={it}  completion={acc*100:.2f}%  clustering={cacc*100:.2f}%  "
          f"rank_pseudo={model.rank_pseudo}")
    if device.type == "cuda":
        print(f"peak VRAM = {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
