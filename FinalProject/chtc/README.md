# Running DUC experiments on CHTC

Same shape as the SEFT setup: build a tarball from the repo, ship it to a
container job, one HTCondor job per configuration, collect JSON shards.

```
chtc/
├── Dockerfile              torch 2.7/cu12.8 + scipy/sklearn/munkres
├── generate_jobs.py        --exp syn|real|base|rank|scale|main|all  → DAG
├── submit/
│   ├── gpu_small.sub       L40S, short queue — v3, baselines
│   ├── gpu_large.sub       L40S, medium queue — v1 on ORL, big-dataset v3
│   └── gpu_xlarge.sub      96 GB card, long queue — v1 on COIL20/EYaleB + paired v3
├── scripts/
│   ├── build_project_tar.sh  repo → project.tar.gz (embeds git SHA)
│   ├── stage_data.sh         verify .mat files on the submit node
│   └── run_experiment.sh     runs inside the container
└── dag/                    generated DAGs
```

---

## 0. One-time: push the image

Needs Docker locally. From `FinalProject/`:

```bash
docker build -t karanvikyath17/duc-chtc:latest -f chtc/Dockerfile .
docker push karanvikyath17/duc-chtc:latest
```

Only rebuild when dependencies change — code changes ship in the tarball.

## 1. One-time: put the datasets on the submit node

The eight `.mat` files total ~128 MB, but **each job needs exactly one of them**
and the largest is 44 MB — inside HTCondor's <100 MB per-file / <500 MB per-job
transfer limits. So they ride along via `transfer_input_files` from your home
directory. They are gitignored, so `git pull` will not bring them.

**Do not use `/staging`.** That is for files individually larger than 1 GB, and
those directories are provisioned by CHTC staff — `mkdir /staging/$USER` gives
"Permission denied" by design. Email chtc@cs.wisc.edu only if you later need one.

```bash
# on the submit node
mkdir -p ~/duc_data

# from your laptop
scp FinalProject/data/*.mat veerannarupa@ap2002.chtc.wisc.edu:~/duc_data/

# back on the submit node — checks every file is present and prints quota
bash chtc/scripts/stage_data.sh
```

| file | dataset | size |
|---|---|---|
| `ORL_32x32.mat` | ORL | 0.4 MB |
| `COIL20.mat` | COIL20 | 2.9 MB |
| `COIL100.mat` | COIL100 | 4.6 MB |
| `YaleBCrop025.mat` | EYaleB | 7.6 MB |
| `oxford_pet.mat` | OxfordPet | 21.6 MB |
| `flowers.mat` | Flowers | 24.0 MB |
| `Dataset_for_Sensorless_Drive_diagnosis.mat` | DSDD | 21.9 MB |
| `HARUS.mat` | HARUS | 44.2 MB |

## 2. Every run: pull, build, submit

```bash
cd ~/DUC_v2/FinalProject
git pull
bash chtc/scripts/build_project_tar.sh
mkdir -p logs tars results chtc/dag

# 0) confirm the pinned GPU model exists and how many there are (see §Same-GPU)
condor_status -compact -constraint 'GPUs > 0' -af GPUs_DeviceName | sort | uniq -c

# 1) five-job canary — must come back with a .json before anything bigger
python chtc/generate_jobs.py --exp scale > chtc/dag/smoke.dag
condor_submit_dag chtc/dag/smoke.dag
#    ... wait, then:  tar -tzf "$(ls -t tars/*.tar.gz | head -1)"   # expect results/*.json

# 2) the paper: v1 vs v3 (syn + real) and the baselines, submitted together
python chtc/generate_jobs.py --exp main > chtc/dag/main.dag
python chtc/generate_jobs.py --exp base > chtc/dag/base.dag
condor_submit_dag chtc/dag/main.dag
condor_submit_dag chtc/dag/base.dag
```

Monitor:

```bash
condor_q                              # queue
condor_q -dag                         # grouped by DAG node
tail -f chtc/dag/main.dag.dagman.out  # DAG progress
condor_q -hold -af HoldReason         # why anything is stuck
```

The tarball embeds `git rev-parse --short HEAD` and every job logs
`=== code SHA: ... ===`, so any result traces back to a commit. It also prints
`[DIRTY working tree]` if built from uncommitted changes — if you see that in a
log, that result is not reproducible from the repo.

## Same-GPU rule

Per-iteration time and peak VRAM are only comparable across methods when every
job in a table ran on the **same card model**. On a shared cluster that means
pinning one, so each submit file carries a `require_gpus` expression:

| tier | default `gpu_model` | jobs |
|---|---|---|
| `gpu_small` / `gpu_large` | `NVIDIA L40S` (46 GB) | all v3, all baselines, v1 on ORL |
| `gpu_xlarge` | `NVIDIA RTX PRO 6000 Blackwell Server Edition` (96 GB) | v1 on COIL20/EYaleB + their paired v3 |

Two consequences:

- **Everything in the main tables runs on the L40S**, including the classical
  baselines (they are on the GPU queue precisely for this reason, not a CPU one).
- COIL20/EYaleB v1 cannot fit a 46 GB card, so those pairs run on the 96 GB
  tier — same-card *within* the pair, but not on the same card as the rest.
  Report them as a separate row group.

Every shard records `torch.cuda.get_device_name()`. `--aggregate` prints the
set of devices seen and marks any row whose seeds landed on different cards
with `MIX!` — if you see that, those time/VRAM numbers are not reportable.

If a tier sits idle, the pinned string does not match what CHTC advertises.
Check with the `condor_status` line above and override in the DAG VARS, e.g.
`gpu_model="NVIDIA H100 80GB HBM3"` for the xlarge tier.

## 3. Collect

```bash
mkdir -p results/shards
for t in tars/results_*.tar.gz; do tar -xzf "$t" -C results/shards --strip-components=1; done
python src/bench_v1_v3.py --aggregate "results/shards/*.json"
```

`--aggregate` groups by (dataset, version, missing rate) and reports mean ± std
across seeds.

---

## Experiment groups

| `--exp` | jobs | contents |
|---|---|---|
| `syn` | 90 | synthetic 200x50 as published — v1 vs v3, 9 rates x 5 seeds |
| `real` | 324 | v3 on all six image sets; v1 on ORL/COIL20/EYaleB, each paired with a same-card v3 |
| **`base`** | **315** | mean / svd_impute / soft_impute / knn / mice on every dataset, same L40S tier |
| `rank` | 72 | v3 pseudo-rank sweep per dataset |
| `scale` | 5 | synthetic to B=200k, completion only |
| **`main`** | **414** | `syn` + `real` — the v1-vs-v3 tables |
| `all` | 806 | everything |

## Why v1 is queued for three datasets only

v1's pseudo-completion weight is dense `(F, B, B)`. With parameter, gradient,
two Adam moments and one gradient temporary that is `5*F*B^2*4` bytes:

| dataset | B | F | v1 needs | queued? |
|---|---|---|---|---|
| synthetic | 200 | 50 | 0.04 GB | yes |
| **ORL** | 400 | 1024 | **3.1 GB** | **yes** (L40S) |
| **COIL20** | 1440 | 1024 | **39.6 GB** | **yes** (96 GB tier) |
| **EYaleB** | 1280 | 2016 | **61.5 GB** | **yes** (96 GB tier) |
| COIL100 | 7200 | 1024 | 989 GB | no |
| OxfordPet | 7349 | 3072 | 3091 GB | no |
| Flowers | 8189 | 3072 | 3837 GB | no |

v3 at `rank_pseudo=1` needs 24 MB – 1.5 GB for the same datasets. On the three
infeasible sets the comparison is v3 against the classical baselines, and v1's
absence is itself a reported result — the generator refuses to queue those jobs
and prints the table above into every DAG.

Clustering is skipped (`--no-cluster`) above ~20k samples: the coefficient
matrix is a dense `(B, B)` on the host, 149 GB at B=200k.

## Verified on CHTC so far

Image pull, container start (L40S), tarball extraction with SHA, `.mat`
transfer, GPU allocation, and results tarball return have all been exercised.
Six configuration bugs were found and fixed on the way (staging, comma-separated
inputs, the `-` placeholder, torch 2.7 / numpy 2 API removals, warmup
clustering, host memory sizing). Still to confirm on the first full submit:

- The pinned `gpu_model` strings match what CHTC advertises (`condor_status`
  line in §2). A tier that sits idle forever means they do not.
- xlarge availability — those cards are scarce; expect that tier to trail the
  rest by hours.
- Home-directory quota — `stage_data.sh` prints it at the end.
