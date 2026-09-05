"""
Generate an HTCondor DAG for the DUC v1-vs-v3 experiments.

    python chtc/generate_jobs.py --exp main --image <user>/duc-chtc:latest \
        > chtc/dag/main.dag
    condor_submit_dag chtc/dag/main.dag

Experiment groups
  syn    synthetic 200x50 exactly as published: v1 vs v3, 10-90% missing, 5 seeds
  real   ORL, COIL20, COIL100, EYaleB, Flowers, OxfordPet. v1 is queued ONLY for
         ORL, because it is the only one whose (F, B, B) weight fits on a GPU.
  rank   v3 pseudo-rank sweep per dataset
  scale  synthetic scaling to B=200k, completion only
  all    everything above
"""

import argparse

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
V1_GPU_LIMIT_GB = 20.0          # largest single-GPU allocation we will request
# Past ~20k samples the dense (B,B) coefficient matrix is a host-side problem
# (149 GB at B=200k), so clustering is skipped and only completion is reported.
CLUSTER_LIMIT = 20000


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


def jobs_for(exp):
    out = []

    def add(name, sub, version, dataset, bsize, missing, seed, rp, extra="-"):
        out.append(dict(name=name, sub=sub, version=version, dataset=dataset,
                        bsize=bsize, missing=missing, seed=seed, rankpseudo=rp,
                        extra=extra))

    if exp in ("syn", "main", "all"):
        # The headline 1-for-1: identical data, mask, and seed for both versions.
        for s in SEEDS:
            for m in MISSING:
                add(f"syn_v1_m{int(m*100)}_s{s}", "small", "v1",
                    "synthetic", 200, m, s, 10)
                add(f"syn_v3_m{int(m*100)}_s{s}", "small", "v3",
                    "synthetic", 200, m, s, 1)

    if exp in ("real", "main", "all"):
        for ds, (B, F, rank, v1gb, sub, clu) in DATASETS.items():
            extra = "-" if (clu and B <= CLUSTER_LIMIT) else "--no-cluster"
            for s in SEEDS:
                for m in MISSING:
                    add(f"real_v3_{ds}_m{int(m*100)}_s{s}", sub, "v3",
                        ds, "-", m, s, 1, extra)
            if v1gb <= V1_GPU_LIMIT_GB:
                # ORL only. ~1.3 s/iter, so keep the v1 grid deliberately small.
                for s in SEEDS[:3]:
                    for m in (0.3, 0.5, 0.7):
                        add(f"real_v1_{ds}_m{int(m*100)}_s{s}", "large", "v1",
                            ds, "-", m, s, 10)

    if exp in ("rank", "all"):
        for ds, (B, F, rank, v1gb, sub, clu) in DATASETS.items():
            extra = "-" if (clu and B <= CLUSTER_LIMIT) else "--no-cluster"
            for rp in (1, 5, 10, 25):
                for s in SEEDS[:3]:
                    add(f"rank_{ds}_rp{rp}_s{s}", sub, "v3", ds, "-", 0.3, s,
                        rp, extra)

    if exp in ("scale", "all"):
        for B in (10000, 25000, 50000, 100000, 200000):
            extra = "--no-cluster" if B > CLUSTER_LIMIT else "-"
            add(f"scale_B{B}", "large" if B >= 50000 else "small", "v3",
                "synthetic", B, 0.3, 17, 1, extra)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main",
                    choices=["syn", "real", "rank", "scale", "main", "all"])
    ap.add_argument("--image", required=True, help="docker image, user/duc-chtc:latest")
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
        ok = "QUEUED" if v1gb <= V1_GPU_LIMIT_GB else "infeasible"
        print(f"#   {ds:<10} B={B:<5} F={F:<5} v1 needs {v1gb:>8.1f} GB   {ok}")
    print(f"#\n# Run: condor_submit_dag chtc/dag/{args.exp}.dag\n")

    for j in js:
        sub = f"chtc/submit/gpu_{j['sub']}.sub"
        mf = MATFILE.get(j["dataset"], "")
        matfile = f"{args.data_dir}/{mf}" if mf else ""
        print(f"JOB {j['name']} {sub}")
        print(f"VARS {j['name']} version=\"{j['version']}\" dataset=\"{j['dataset']}\" "
              f"bsize=\"{j['bsize']}\" missing=\"{j['missing']}\" seed=\"{j['seed']}\" "
              f"rankpseudo=\"{j['rankpseudo']}\" extra_args=\"{j['extra']}\" "
              f"image_name=\"{args.image}\" matfile=\"{matfile}\"")
        print(f"RETRY {j['name']} {args.retry}\n")


if __name__ == "__main__":
    main()
