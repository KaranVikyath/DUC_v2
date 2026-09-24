# VED: DUC config sensitivity (disclosed, not used for selection)

The pre-registered VED config (bench_v1_v3.mat_problem, committed 70a717a:
MLP encoder [32], rank 20, lr 7e-3) is what the CHTC `best` runs used and what
the paper reports. After seeing those results (held-out NMAE worse than mean
imputation for linear/identity), two alternatives were tried locally on VED10k,
30% missing, seed 17. They are reported here so the choice is transparent; none
was adopted, because picking among them by held-out error would be tuning on
the test entries.

| config | linear | CoLA-identity | CoLA-tanh | SoftImpute | mean |
|---|---|---|---|---|---|
| pre-registered: enc [32], rank 20, lr 7e-3 (CHTC, 5 seeds) | 1.121 | 1.273 | 0.778 | 0.641 | 0.769 |
| BostonHousing preset scaled: enc [8,6], rank 3, lr 5e-2 | 0.773 | 0.773 | 1.443 | | |
| same, lr 7e-3 (tabular lr) | 1.463 | 1.427 | 1.213 | | |

(z-space MAE on held-out entries, comparable to NMAE; CHTC row is the 5-seed mean.)

Reading: with 9 features each row has only 6-7 observed values at 30%
missing. DUC's pseudo-completion gives every missing entry a free parameter,
and a flexible decoder can then reproduce each row's few observed values
exactly (training loss 0.1-2.3) without learning cross-row structure, so the
held-out entries are no better than — or worse than — the column mean. On
HARUS (561 features) the same pipeline is competitive (tanh 0.255 vs
SoftImpute 0.267 at 30%), so this is a low-dimensionality limitation of the
model, not a pipeline bug.
