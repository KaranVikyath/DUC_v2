"""
Dataset configurations as published in the DUC v1 paper (Table 1).

Transcribed verbatim so v3 can be compared against v1 under identical settings.
`check_configs()` validates each one against the CFS rank constraint:

    the CFS projector spans min(rank, F_enc) latent directions, so when
    rank >= F_enc it spans EVERYTHING, P = I, and the subspace projection
    silently does nothing — the model degenerates to a plain autoencoder.

F_enc is measured by building the actual Encoder and running one forward pass,
rather than hand-computing conv output shapes.

Run:  py -3.10 paper_configs.py
"""

import numpy as np
import torch

# name: (shape, K, d, rank, lr, enc_layer_size, kernel_size, output_padding)
# K/d = None where the paper lists N/A (unlabeled -> completion only).
PAPER_CONFIGS = {
    "COIL20":      ([1440, 32, 32, 1], 20,   12,  240, 1e-5, [15],          [3],       [1]),
    "EYaleB":      ([1280, 48, 42, 1], 20,   10,  200, 5e-2, [10, 20, 30],  [5, 3, 3], [1, (0, 1), 1]),
    "ORL":         ([400, 32, 32, 1],  40,   10,  400, 4e-2, [3, 3, 5],     [3, 3, 3], [1, 1, 1]),
    "Flowers":     ([8189, 32, 32, 3], 102,  2,   204, 1e-3, [32, 64, 128], [3, 3, 3], [1, 1, 1]),
    "OxfordPet":   ([7349, 32, 32, 3], 37,   3,   11,  9e-3, [32, 64, 128], [3, 3, 3], [1, 1, 1]),
    "BostonHousing": ([507, 14],       None, None, 12, 5e-2, [12, 10],      None,      None),
    "HeartDisease":([303, 14],         5,    1,   5,   7e-3, [12],          None,      None),
    "HARUS":       ([10299, 561],      6,    20,  120, 7e-3, [512, 256],    None,      None),
    "PeriodChanger": ([90, 1177],      2,    3,   6,   7e-3, [1024, 512],   None,      None),
}


def latent_dim(name, device="cpu"):
    """F_enc: the flattened encoder output width, measured not derived."""
    from DeLUCA import Encoder
    shape, K, d, rank, lr, enc, ks, opad = PAPER_CONFIGS[name]
    # One sample is enough to measure the width, and keeps this cheap for
    # the 8k-sample image sets.
    probe_shape = [2] + list(shape[1:])
    model = Encoder(tuple(shape), enc, ks).to(device)
    with torch.no_grad():
        z = model(torch.zeros(*probe_shape, device=device))
    return int(np.prod(z.shape[1:]))


def check_configs(device="cpu"):
    print(f"  {'dataset':<15} {'shape':<20} {'K':>4} {'d':>4} {'rank':>5} "
          f"{'F_enc':>6} | CFS")
    print("  " + "-" * 72)
    bad = []
    for name, (shape, K, d, rank, lr, enc, ks, opad) in PAPER_CONFIGS.items():
        try:
            fe = latent_dim(name, device)
        except Exception as e:
            print(f"  {name:<15} {str(shape):<20} -> encoder failed: {type(e).__name__}: {e}")
            continue
        degenerate = rank >= fe
        if degenerate:
            bad.append((name, rank, fe))
        print(f"  {name:<15} {str(shape):<20} {str(K):>4} {str(d):>4} {rank:>5} "
              f"{fe:>6} | {'IDENTITY (P=I)' if degenerate else 'active'}")
    if bad:
        print("\n  DEGENERATE CONFIGS — the subspace projection is a no-op:")
        for name, rank, fe in bad:
            print(f"    {name}: rank={rank} >= F_enc={fe}. "
                  f"Needs rank < {fe} (or a wider bottleneck).")
    return bad


if __name__ == "__main__":
    check_configs()
