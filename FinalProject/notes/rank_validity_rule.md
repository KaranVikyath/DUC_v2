# Rank validity rule (fixed 2026-09-24, before the rerun it governs)

v3's CFS step projects each sample's latent code onto a rank-r subspace, and
that projection is the model's bottleneck: completion works because missing
entries must be explained by the few directions shared across samples. If r is
at least the input width F, or at least the encoder width F_enc, the projection
constrains nothing and the model can reproduce the observed entries of each row
without learning shared structure.

**Validity condition:** r < min(F, F_enc).

Every config the original authors wrote satisfies it. Three configs in this
benchmark did not, and they are exactly the three datasets where v3 imputed
worse than the column mean:

| dataset | old rank | F | F_enc | violated |
|---|---|---|---|---|
| COIL100 | 1200 (= K*d, copied from COIL20's d=12) | 1024 | 3840 | r >= F |
| DSDD | 44 (authors' dataset_params.py) | 48 | 40 | r >= F_enc |
| VED / VED10k | 20 (ours) | 9 | 32 | r >= F |

**Rule:** for a config that violates the condition, r = round(0.2 * min(F, F_enc)).
0.2 is the median of r / min(F, F_enc) over the authors' own configs:

| config | r | min(F, F_enc) | ratio |
|---|---|---|---|
| ORL | 40 | 80 | 0.50 |
| COIL20 | 240 | 1024 | 0.23 |
| EYaleB | 200 | 1080 | 0.19 |
| HARUS | 120 | 256 | 0.47 |
| Flowers | 204 | 2048 | 0.10 |
| OxfordPet | 111 | 2048 | 0.05 |

median = (0.19 + 0.23) / 2 = 0.21 -> 0.2.

New ranks: COIL100 205, DSDD 8, VED 2. Configs that satisfy the condition are
untouched. The same rank feeds the svd_impute / soft_impute baselines, which
were equally unconstrained on these datasets, so their rows are rerun too.

Reporting: both the original-config and rule-corrected results are kept and
shown; the rule was fixed from the authors' configs, not from any held-out
error of the rerun.
