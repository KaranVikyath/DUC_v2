# Running DUC experiments on CHTC

Same shape as the SEFT setup: build a tarball from the repo, ship it to a
container job, one HTCondor job per configuration, collect JSON shards.

```
chtc/
├── Dockerfile              torch 2.7/cu12.8 + scipy/sklearn/munkres
├── generate_jobs.py        --exp syn|real|rank|scale|main|all  → DAG
├── submit/
│   ├── gpu_small.sub       v3 jobs, 8 GB GPU, short queue
│   └── gpu_large.sub       v1 baselines + big datasets, 24 GB, medium
├── scripts/
│   ├── build_project_tar.sh  repo → project.tar.gz (embeds git SHA)
│   ├── stage_data.sh         .mat files → /staging
│   └── run_experiment.sh     runs inside the container
└── dag/                    generated DAGs
```

---

## 0. One-time: push the image

Needs Docker locally. From `FinalProject/`:

```bash
docker build -t <dockerhub-user>/duc-chtc:latest -f chtc/Dockerfile .
docker push <dockerhub-user>/duc-chtc:latest
```

Only rebuild when dependencies change — code changes ship in the tarball.

## 1. One-time: get the datasets into staging

The eight `.mat` files total ~128 MB, which is past CHTC's guidance for
`transfer_input_files`, so they live in `/staging` and each job copies only the
one it needs. They are **not** in git for the same reason.

From your laptop:

```bash
# ~128 MB, one transfer
scp FinalProject/data/*.mat <netid>@ap2001.chtc.wisc.edu:/staging/<netid>/duc_data/
```

If the directory does not exist yet, on the submit node first:

```bash
mkdir -p /staging/$USER/duc_data
```

Then verify (also prints your staging usage):

```bash
bash chtc/scripts/stage_data.sh $USER
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
git pull
bash chtc/scripts/build_project_tar.sh

python chtc/generate_jobs.py --exp main \
    --image <dockerhub-user>/duc-chtc:latest \
    --staging-user $USER > chtc/dag/main.dag

mkdir -p logs tars results
condor_submit_dag chtc/dag/main.dag
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
| `real` | 279 | ORL, COIL20, COIL100, EYaleB, Flowers, OxfordPet |
| `rank` | 72 | v3 pseudo-rank sweep per dataset |
| `scale` | 5 | synthetic to B=200k, completion only |
| **`main`** | **369** | `syn` + `real` — the paper's two headline tables |
| `all` | 446 | everything |

## Why v1 is queued for ORL only

v1's pseudo-completion weight is dense `(F, B, B)`. With parameter, gradient,
two Adam moments and one gradient temporary that is `5*F*B^2*4` bytes:

| dataset | B | F | v1 needs | queued? |
|---|---|---|---|---|
| synthetic | 200 | 50 | 0.04 GB | yes |
| **ORL** | 400 | 1024 | **3.1 GB** | **yes** |
| COIL20 | 1440 | 1024 | 39.6 GB | no |
| EYaleB | 1280 | 2016 | 61.5 GB | no |
| COIL100 | 7200 | 1024 | 989 GB | no |
| OxfordPet | 7349 | 3072 | 3091 GB | no |
| Flowers | 8189 | 3072 | 3837 GB | no |

v3 at `rank_pseudo=1` needs 24 MB – 1.5 GB for the same datasets. The infeasible
rows belong in the paper as a derivation, not as jobs that OOM — which is why
the generator refuses to queue them and prints the table above into every DAG.

Clustering is skipped (`--no-cluster`) above ~20k samples: the coefficient
matrix is a dense `(B, B)` on the host, 149 GB at B=200k.

## Not yet verified on CHTC

Written against the CHTC docs, not yet run there. Check on first submit:

- GPU Lab availability at `gpus_minimum_memory = 24000` for `gpu_large.sub`.
  If those queue too long, drop it and skip v1-on-ORL.
- `gpus_maximum_capability = 12.0` assumes the cu12.8 image really carries
  Blackwell kernels. On "no kernel image is available", pass `gpu_cap_max="9.0"`
  in the DAG VARS.
- Staging quota — `stage_data.sh` prints `du -sh` at the end.
