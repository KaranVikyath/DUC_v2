"""
Generate an HTCondor DAG for the DUC v1-vs-v3 experiments.

    python chtc/generate_jobs.py --exp main --image karanvikyath17/duc-chtc:latest \
        > chtc/dag/main.dag
    condor_submit_dag chtc/dag/main.dag

Experiment groups
  syn    synthetic 200x50 exactly as published: v1 vs v3, 10-90% missing, 5 seeds
  real   ORL, COIL20, COIL100, EYaleB, Flowers, OxfordPet — all for v3. v1 runs
         on ORL (large tier) and COIL20/EYaleB (xlarge: 80+ GB cards), paired
         with v3 on the same tier so timing is same-hardware. COIL100/Flowers/
         OxfordPet need 1-4 TB for v1 and are reported as infeasible.
  rank   v3 pseudo-rank sweep per dataset
  scale  synthetic scaling to B=200k, completion only
  base   classical completion baselines (mean/svd_impute/soft_impute/knn/mice)
         on every dataset, CPU queue. Same masks and metrics as v3, so the
         rows land in the same aggregate table.
  all    everything above
"""

import argparse
import math

SEEDS = [17, 18, 19, 20, 21]
MISSING = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# (B, F, rank) per dataset, and the memory v1's dense (F, B, B) weight needs:
#   5*F*B^2*4 bytes  (param + grad + 2 Adam moments + one gradient temporary).
# v1 is queued only where that fits a single GPU. The rest are reported in the
# paper as infeasible-by-construction, with the derivation — not as failed jobs.
DATASETS = {
    #            B      F  rank  v1 GB   sub      cluster?
    "ORL":      (400, 1024,   40,    3.1, "large", True),
    "COIL20":  (1440, 1024,  240,   39.6, "small", True),
    "EYaleB":  (1280, 2016,  200,   61.5, "small", True),
    "COIL100": (7200, 1024, 1200,  989.0, "large", True),
    "OxfordPet": (7349, 3072, 111, 3090.6, "large", True),
    "Flowers": (8189, 3072,  204, 3837.4, "large", True),
}
# v1 is queued on three tiers by its (F,B,B) footprint. Above XLARGE nothing
# fits any single GPU: COIL100 needs 989 GB, Flowers 3.8 TB.
V1_LARGE_GB = 20.0              # fits a 24-46 GB card (ORL: 3.1 GB)
V1_XLARGE_GB = 70.0             # needs an 80-96 GB card (COIL20 39.6, EYaleB 61.5)
# Datasets v1 is asked to run on. Everything else is v3-only.
V1_DATASETS = ("ORL", "COIL20", "EYaleB")
# Past ~20k samples the dense (B,B) coefficient matrix is a host-side problem
# (149 GB at B=200k), so clustering is skipped and only completion is reported.
CLUSTER_LIMIT = 20000

# Classical baselines and their size caps (mirrors baselines.BASELINES). Capped
# jobs are simply not queued rather than submitted to exit immediately.
BASELINES = {
    #  name          max_B   max_F
    "mean":        (None,   None),
    "svd_impute":  (None,   None),
    "soft_impute": (None,   None),
    "knn":         (3000,   None),      # O(B^2 F)
    "mice":        (None,   128),       # one regressor per feature per round
}


# Each job needs exactly ONE dataset file, and the largest is 44 MB — inside
# HTCondor's <100 MB per-file / <500 MB per-job transfer limits. /staging is for
# files individually over 1 GB and has to be provisioned by CHTC staff, so plain
# transfer_input_files is the right mechanism here.
MATFILE = {
    "ORL": "ORL_32x32.mat",
    "COIL20": "COIL20.mat",
    "COIL100": "COIL100.mat",
    "EYaleB": "YaleBCrop025.mat",
    "Flowers": "flowers.mat",
    "OxfordPet": "oxford_pet.mat",
    "HARUS": "HARUS.mat",
    "DSDD": "Dataset_for_Sensorless_Drive_diagnosis.mat",
}


def host_mem_gb(B, F, clustering):
    """Host RAM a job needs, in GB — the thing that actually held jobs.

    Clustering is the driver: thrC sorts and argsorts the dense (B,B) and
    allocates a float64 copy (~24*B^2 bytes), then post_proC symmetrises it and
    forms U U^T and its powers (~32*B^2). That is ~56*B^2 bytes, i.e. 5.6 GB at
    B=10k — which overran an 8 GB request. Everything else is a dozen or so
    float64 (B,F) copies plus the torch/CUDA context.
    """
    base = 6.0                    # torch + CUDA context + libs + model, with slack
    dense = 56.0 * B * B / 1e9 if clustering else 0.0
    arrays = 12.0 * B * F * 8 / 1e9
    return int(math.ceil(base + dense + arrays))


def jobs_for(exp):
    out = []

    def add(name, sub, version, dataset, bsize, missing, seed, rp, extra="-",
            B=200, F=50, v1gb=0.0):
        mem = host_mem_gb(B, F, clustering=(extra != "--no-cluster"))
        if version == "v1" and v1gb > V1_LARGE_GB:
            # (F,B,B) is built on the host first, so host RAM must cover it too.
            sub = "xlarge"
            mem = max(mem, int(math.ceil(v1gb * 1.5)))
        elif sub != "xlarge":
            # Pick the submit file from what the job actually needs.
            sub = "large" if (mem > 8 or sub == "large") else "small"
        out.append(dict(name=name, sub=sub, version=version, dataset=dataset,
                        bsize=bsize, missing=missing, seed=seed, rankpseudo=rp,
                        extra=extra, req_mem=f"{mem}GB"))

    if exp in ("syn", "main", "all"):
        # The headline 1-for-1: identical data, mask, and seed for both versions.
        for s in SEEDS:
            for m in MISSING:
                add(f"syn_v1_m{int(m*100)}_s{s}", "small", "v1",
                    "synthetic", 200, m, s, 10, B=200, F=50)
                add(f"syn_v3_m{int(m*100)}_s{s}", "small", "v3",
                    "synthetic", 200, m, s, 1, B=200, F=50)

    if exp in ("real", "main", "all"):
        for ds, (B, F, rank, v1gb, sub, clu) in DATASETS.items():
            extra = "-" if (clu and B <= CLUSTER_LIMIT) else "--no-cluster"
            for s in SEEDS:
                for m in MISSING:
                    add(f"real_v3_{ds}_m{int(m*100)}_s{s}", sub, "v3",
                        ds, "-", m, s, 1, extra, B=B, F=F)
            if ds in V1_DATASETS and v1gb <= V1_XLARGE_GB:
                # v1 is slow (ORL ~1.3 s/iter; COIL20/EYaleB ~13x that), so it
                # gets a reduced grid. The v3 jobs at the SAME (missing, seed)
                # are queued on the SAME submit file so the timing comparison is
                # same-hardware — a v1 on an H100 against a v3 on a 3060 would
                # understate v3 by the hardware gap.
                tier = "xlarge" if v1gb > V1_LARGE_GB else "large"
                for s in SEEDS[:3]:
                    for m in (0.3, 0.5, 0.7):
                        add(f"real_v1_{ds}_m{int(m*100)}_s{s}", tier, "v1",
                            ds, "-", m, s, 10, B=B, F=F, v1gb=v1gb)
                        add(f"pair_v3_{ds}_m{int(m*100)}_s{s}", tier, "v3",
                            ds, "-", m, s, 1, extra, B=B, F=F)

    if exp in ("rank", "all"):
        for ds, (B, F, rank, v1gb, sub, clu) in DATASETS.items():
            extra = "-" if (clu and B <= CLUSTER_LIMIT) else "--no-cluster"
            for rp in (1, 5, 10, 25):
                for s in SEEDS[:3]:
                    add(f"rank_{ds}_rp{rp}_s{s}", sub, "v3", ds, "-", 0.3, s,
                        rp, extra, B=B, F=F)

    if exp in ("base", "all"):
        targets = [("synthetic", 200, 50)] + [(ds, v[0], v[1]) for ds, v in DATASETS.items()]
        for ds, B, F in targets:
            extra = "-" if B <= CLUSTER_LIMIT else "--no-cluster"
            for name, (max_B, max_F) in BASELINES.items():
                if (max_B and B > max_B) or (max_F and F > max_F):
                    continue
                for s in SEEDS:
                    for m in MISSING:
                        bs = 200 if ds == "synthetic" else "-"
                        # Baseline memory is a few (B,F) float64 copies plus the
                        # clustering (B,B) — no torch context, so a lower base.
                        mem = int(math.ceil(2.0 + (56.0 * B * B / 1e9 if extra == "-" else 0)
                                            + 12.0 * B * F * 8 / 1e9))
                        out.append(dict(name=f"base_{name}_{ds}_m{int(m*100)}_s{s}",
                                        sub="cpu", version=name, dataset=ds,
                                        bsize=bs, missing=m, seed=s, rankpseudo=0,
                                        extra=extra, req_mem=f"{max(mem, 4)}GB"))

    if exp in ("scale", "all"):
        for B in (10000, 25000, 50000, 100000, 200000):
            extra = "--no-cluster" if B > CLUSTER_LIMIT else "-"
            add(f"scale_B{B}", "small", "v3",
                "synthetic", B, 0.3, 17, 1, extra, B=B, F=50)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main",
                    choices=["syn", "real", "rank", "scale", "base", "main", "all"])
    ap.add_argument("--image", default="karanvikyath17/duc-chtc:latest",
                    help="docker image (default: %(default)s)")
    ap.add_argument("--data-dir", default="$ENV(HOME)/duc_data",
                    help="where the .mat files live on the submit node")
    ap.add_argument("--retry", type=int, default=2)
    args = ap.parse_args()

    js = jobs_for(args.exp)
    n_v1 = sum(1 for j in js if j["version"] == "v1")
    print(f"# ===== DUC v1-vs-v3 DAG — group '{args.exp}' =====")
    print(f"# {len(js)} jobs ({n_v1} v1, {len(js)-n_v1} v3)")
    print(f"# image {args.image} | datasets {args.data_dir}")
    print("#")
    print("# v1's pseudo-completion weight is (F, B, B); it is queued only where")
    print("# 5*F*B^2*4 bytes fits one GPU:")
    for ds, (B, F, rank, v1gb, sub, clu) in sorted(DATASETS.items()):
        if ds not in V1_DATASETS or v1gb > V1_XLARGE_GB:
            ok = "infeasible" if v1gb > V1_XLARGE_GB else "not requested"
        else:
            ok = "QUEUED (xlarge, 80+ GB card)" if v1gb > V1_LARGE_GB else "QUEUED"
        print(f"#   {ds:<10} B={B:<5} F={F:<5} v1 needs {v1gb:>8.1f} GB   {ok}")
    print(f"#\n# Run: condor_submit_dag chtc/dag/{args.exp}.dag\n")

    for j in js:
        sub = ("chtc/submit/cpu.sub" if j["sub"] == "cpu"
               else f"chtc/submit/gpu_{j['sub']}.sub")
        mf = MATFILE.get(j["dataset"], "")
        # transfer_input_files is COMMA-separated. Build the whole list here so
        # synthetic jobs (no .mat) do not end up with a dangling comma.
        inputs = ("project.tar.gz," + f"{args.data_dir}/{mf}") if mf else "project.tar.gz"
        print(f"JOB {j['name']} {sub}")
        print(f"VARS {j['name']} version=\"{j['version']}\" dataset=\"{j['dataset']}\" "
              f"bsize=\"{j['bsize']}\" missing=\"{j['missing']}\" seed=\"{j['seed']}\" "
              f"rankpseudo=\"{j['rankpseudo']}\" extra_args=\"{j['extra']}\" "
              f"image_name=\"{args.image}\" inputs=\"{inputs}\" "
              f"req_mem=\"{j['req_mem']}\"")
        print(f"RETRY {j['name']} {args.retry}\n")


if __name__ == "__main__":
    main()
