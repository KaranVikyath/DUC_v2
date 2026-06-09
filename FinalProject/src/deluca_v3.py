"""
DeLUCA v3 — low-rank / low-VRAM variant.

Two memory wins over DeLUCA.py, both targeting the B**2 terms (B = batch/sample count):

  Part 1 — EfficientCFSModule:
      The CFS clustering layer never materializes the dense (B, B) `Coef = V_rank @ V_rank.T`.
      In pure-completion (CFS) mode `Coef` is *not* in the loss or the gradient — `PZ` (fwd)
      and `d_Z` (bwd) both factor through `V_rank (B, rank)`. So we simply never build the
      B**2 matrix. Optionally (need_clustering=True) it can be rebuilt lazily in row-tiles
      at the very end via `affinity_tiled`, never densely during training.

  Part 2 — LowRankPseudoCompletion:
      The per-feature mixing weight W_f (B, B) is factorized as  W_f = B_mat_f @ A_f  with
      A (F, r_p, B) and B_mat (F, B, r_p), so memory is 2*F*B*r_p instead of F*B**2.
      Because the `bmm` treats the feature axis F as its batch dim, every feature is
      independent — so we also chunk along F (`f_chunk`), which is *lossless* (exact) and
      caps peak activation/grad memory at f_chunk*B*r_p.

Part 3 (cross-chunk subspace carry-over for sample-axis / B chunking) is intentionally NOT
implemented here yet — to be discussed before building.

Reuses Encoder, Decoder, the fused masked-loss path, and convert_nan from the existing code.
"""

import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from custom_funcs import convert_nan
from DeLUCA import Encoder, Decoder, MaskedLossFn, _load_masked_loss_cuda


class LowRankPseudoCompletion(nn.Module):
    """Pseudo-completion with low-rank per-feature weight  W_f = B_mat_f @ A_f.

    weight (F, B, B)  ->  A (F, r_p, B) + B_mat (F, B, r_p)   [2*F*B*r_p vs F*B**2]

    Forward (per-feature, all independent):
        tmp = A   @ x_t   -> (F, r_p, 1)   cheap contraction over samples
        out = B_mat @ tmp -> (F, B,   1)   expansion back to samples
        out = PReLU(out + bias)
    `f_chunk` loops feature slices through the two bmms — lossless, lower peak VRAM.
    """

    def __init__(self, input_shape, flat_layer_size, rank_pseudo, f_chunk=None):
        super().__init__()
        self.input_shape = input_shape
        self.feature_size = int(np.prod(input_shape[1:]))
        self.batch_size = flat_layer_size[0]
        self.rank_pseudo = int(rank_pseudo)
        self.f_chunk = f_chunk  # None => no feature chunking (single bmm)

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

    def _prelu(self, out, f0, f1):
        pos = torch.clamp(out, min=0)
        neg = self.prelu_weight[f0:f1] * torch.clamp(out, max=0)
        return pos + neg

    def forward(self, x):
        # x: (B, *features) -> (B, F)
        x = x.reshape(self.batch_size, -1).float()
        x = torch.nan_to_num(x, nan=0.0)
        x_t = x.t().unsqueeze(2)  # (F, B, 1)

        F = self.feature_size
        chunk = self.f_chunk if self.f_chunk else F

        if chunk >= F:
            tmp = torch.bmm(self.A, x_t)              # (F, r, 1)
            out = torch.bmm(self.B_mat, tmp).squeeze(2)  # (F, B)
            out = self._prelu(out + self.bias, 0, F)
            return out.t().reshape(self.input_shape)

        # Lossless feature-axis chunking: features are independent.
        out = torch.empty(F, self.batch_size, device=x.device, dtype=x.dtype)
        for f0 in range(0, F, chunk):
            f1 = min(f0 + chunk, F)
            tmp = torch.bmm(self.A[f0:f1], x_t[f0:f1])       # (fc, r, 1)
            o = torch.bmm(self.B_mat[f0:f1], tmp).squeeze(2)  # (fc, B)
            out[f0:f1] = self._prelu(o + self.bias[f0:f1], f0, f1)
        return out.t().reshape(self.input_shape)


class EfficientCFSModule(nn.Module):
    """CFS subspace projection that never builds the dense (B, B) Coef.

    PZ = V_rank @ (V_rank.T @ Z)  -- O(B*r) memory, fully differentiable through
    torch.svd_lowrank. The B**2 affinity is only ever built on demand via
    `affinity_tiled` (row-tiled), for the case where clustering output is needed.
    """

    def __init__(self, rank, need_clustering=False, svd_niter=2):
        super().__init__()
        self.rank = rank
        self.need_clustering = need_clustering
        self.svd_niter = svd_niter
        self.V_rank = None  # (B, rank) — sufficient statistic for the projection

    def forward(self, Z):
        # Z: (B, F_enc)
        Zt = Z.t()  # (F_enc, B)
        U, S, V = torch.svd_lowrank(Zt, q=self.rank, niter=self.svd_niter)
        self.V_rank = V                 # (B, rank)
        VtZ = V.t() @ Z                 # (rank, F_enc)
        PZ = V @ VtZ                    # (B, F_enc) — no (B, B) ever formed
        return PZ, V

    @torch.no_grad()
    def affinity_tiled(self, tile=4096, threshold_fn=None):
        """Build C = V_rank @ V_rank.T lazily in row-tiles (for clustering only).

        Returns a dense (B, B) on CPU. If `threshold_fn` is given it is applied
        per tile so only the sparsified result is kept. Never call this during
        training — it is the one place a B**2 object appears, and only at the end.
        """
        if self.V_rank is None:
            raise RuntimeError("affinity_tiled called before a forward pass")
        V = self.V_rank
        B = V.shape[0]
        C = torch.empty(B, B, device="cpu", dtype=torch.float32)
        for i in range(0, B, tile):
            j = min(i + tile, B)
            row = (V[i:j] @ V.t())  # (tile, B) on GPU
            if threshold_fn is not None:
                row = threshold_fn(row)
            C[i:j] = row.cpu()
        return C


class DeLUCAV3(nn.Module):
    """Low-rank, low-VRAM DeLUCA (CFS / pure-completion).

    Drop-in constructor-compatible with DeLUCA + 3 extra kwargs:
        rank_pseudo     : rank r_p of the factorized pseudo-completion weight
                          (None -> min(B, max(2*rank, 8))). Lower = less VRAM.
        f_chunk         : feature-axis chunk size for pseudo-completion (lossless).
        need_clustering : if False (default) C is never built.
    """

    def __init__(self, input_shape, flat_layer_size, enc_layer_size, deco_layer_size,
                 kernel_size, output_padding, lr, K, rank, reg_const1=1.0, reg_const2=1.0,
                 batch_size=200, model_path=None, logs_path="",
                 cluster_model="CFS", device="CPU",
                 rank_pseudo=None, f_chunk=None, need_clustering=False):
        super().__init__()
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
        self.need_clustering = need_clustering

        if rank_pseudo is None:
            rank_pseudo = min(batch_size, max(2 * rank, 8))
        self.rank_pseudo = rank_pseudo

        # Layers (Encoder/Decoder reused unchanged)
        self.pseudo = LowRankPseudoCompletion(input_shape, flat_layer_size,
                                              rank_pseudo, f_chunk=f_chunk)
        self.encoder = Encoder(input_shape, enc_layer_size, kernel_size)
        self.CFS_module = EfficientCFSModule(rank, need_clustering=need_clustering)
        self.decoder = Decoder(enc_layer_size[-1], deco_layer_size, kernel_size, output_padding)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9,
                                           patience=10, verbose=False)
        self.summary_writer = SummaryWriter(logs_path)

    def forward(self, x):
        x = torch.tensor(x).to(self.device)
        Xc = self.pseudo(x)
        Z = self.encoder(Xc)
        Z_flat = Z.view(self.batch_size, -1)

        PZ, V_rank = self.CFS_module(Z_flat)
        PZ = PZ.view(Z.shape)
        decoded = self.decoder(PZ)

        _ml_ext = _load_masked_loss_cuda()
        if x.is_cuda and _ml_ext is not None:
            reconstruction_loss = MaskedLossFn.apply(x, Xc, decoded)
        else:
            x_omega, mask_tensor = convert_nan(x)
            Xc_m = Xc * mask_tensor
            dec_m = decoded * mask_tensor
            d1 = Xc_m - x_omega
            d2 = Xc_m - dec_m
            d3 = dec_m - x_omega
            reconstruction_loss = (torch.norm(d1, p='fro')
                                   + torch.norm(d2, p='fro')
                                   + torch.norm(d3, p='fro'))
        autoencoder_loss = 0.5 * torch.norm(Z - PZ, p='fro')

        # CFS / pure completion: clustering coefficient is not part of the loss.
        total_loss = reconstruction_loss
        return decoded, total_loss, autoencoder_loss, reconstruction_loss

    def finetune_fit(self, x):
        self.optimizer.zero_grad()
        decoded, total_loss, autoencoder_loss, reconstruction_loss = self.forward(x)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step(total_loss)

        with torch.no_grad():
            # No dense (B, B) here: C stays None unless clustering is explicitly wanted.
            C = None
            if self.need_clustering:
                C = self.CFS_module.V_rank.detach().cpu().numpy()  # (B, rank) embedding
            reconstruction_loss = reconstruction_loss.item()
            autoencoder_loss = autoencoder_loss.item()
            total_loss = total_loss.item()
            complete_data = decoded.detach().cpu().numpy()
            lr = self.optimizer.param_groups[0]['lr']

        self.iter += 1
        self.summary_writer.add_scalar('Reconstruction Loss', reconstruction_loss, self.iter)
        self.summary_writer.add_scalar('Autoencoder Loss', autoencoder_loss, self.iter)
        self.summary_writer.add_scalar('Total Loss', total_loss, self.iter)
        return C, total_loss, complete_data, lr


# --------------------------------------------------------------------------- #
# Smoke test + VRAM probe (synthetic). Run:  python deluca_v3.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from custom_funcs import generate_data, missing_data_generation

    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    np.random.seed(17)

    (data, full_data, input_shape, batch_size, total_datapoints, flat_layer_size,
     enc_layer_size, deco_layer_size, kernel_size, output_padding, K, reg1, reg2,
     alpha1, alpha2, d, lr, rank, true_labels) = generate_data(m=100, n=100, r=5, k=5, noise=0)

    missing = missing_data_generation(data, int(0.30 * total_datapoints))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _here = os.path.dirname(os.path.abspath(__file__))
    logs = os.path.join(_here, "..", "data", "logs", "deluca_v3_smoke")

    model = DeLUCAV3(
        input_shape, flat_layer_size, enc_layer_size, deco_layer_size,
        kernel_size, output_padding, lr, K, rank,
        reg_const1=reg1, reg_const2=reg2, batch_size=batch_size,
        logs_path=logs, cluster_model="CFS", device=device,
        rank_pseudo=1, f_chunk=None, need_clustering=False)
    model.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    stopping_lr = lr / 10
    data_norm = np.linalg.norm(full_data)
    it = 0
    while True:
        C, cost, complete_data, cur_lr = model.finetune_fit(missing)
        it += 1
        if cur_lr < stopping_lr or it > 2000:
            break

    complete_data[~np.isnan(missing)] = missing[~np.isnan(missing)]
    acc = 1 - np.linalg.norm(complete_data - full_data) / data_norm
    print(f"iters={it}  completion_acc={acc*100:.2f}%  rank_pseudo={model.rank_pseudo}")
    if device.type == "cuda":
        print(f"peak VRAM = {torch.cuda.max_memory_allocated()/1e6:.1f} MB")

    # Lossless feature-chunk check: chunked == unchunked
    x = torch.tensor(np.nan_to_num(missing, nan=0.0)).to(device)
    with torch.no_grad():
        model.pseudo.f_chunk = None
        full = model.pseudo(x).clone()
        model.pseudo.f_chunk = max(1, model.pseudo.feature_size // 3)
        chunked = model.pseudo(x).clone()
        model.pseudo.f_chunk = None
    print(f"feature-chunk allclose: {torch.allclose(full, chunked, atol=1e-5)}")
