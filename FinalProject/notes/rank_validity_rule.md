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

## Outcome of the rerun (5 seeds, L40S; results/rank_rule_2026-09-24 vs results/original_rank_configs)

NMAE on held-out entries, old -> new:

| dataset, 30 / 50 / 70% | v3 tanh | v3 identity | svd_impute | soft_impute |
|---|---|---|---|---|
| COIL100 | 0.81/0.82/0.68 -> 0.27/0.34/0.63 | 1.86/1.88/1.49 -> 0.37/0.35/0.63 | 0.75/0.75/0.75 -> 0.27/0.35/0.54 | unchanged 0.61/0.67/0.76 |
| DSDD | 0.37/0.44/0.56 -> 0.42/0.62/1.25 | 0.57/0.88/0.94 -> 0.54/1.01/2.36 | 0.58/0.63/0.65 -> 0.41/0.47/0.62 | 0.24/0.29/0.38 -> 0.39/0.43/0.52 |
| VED | 0.76/0.82/0.86 -> 0.87/1.50/3.15 | 1.26/1.76/2.18 -> 1.14/2.30/3.95 | 0.75/0.77/0.78 -> 0.72/0.84/0.95 | 0.64/0.68/0.72 -> 0.70/0.77/0.87 |

COIL100 clustering: v3 tanh 12/10/8 -> 48/43/33 %, identity 5/4/4 -> 50/42/33 %
(svd_impute 60/53/44 -> 61/56/46 %). v3 linear on COIL100 is unstable under
the new rank: 4 of 5 seeds stop on a high-loss plateau.

Reading: the rule repairs COIL100, where the old rank exceeded the pixel
count. On the low-dimensional tables (DSDD 48, VED 9 features) the 0.2 ratio,
taken from 1,000+-dimensional image configs, is too restrictive — it degrades
SoftImpute as much as v3 — and DUC trails SoftImpute/MICE under either config.
Open issue: svd_impute / soft_impute borrow DUC's rank; their rank should be
chosen independently (e.g. on 5% of observed entries) for a fair baseline.
