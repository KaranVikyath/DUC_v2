#!/bin/bash
# Executed inside the DUC container on a CHTC GPU node.
#
#   run_experiment.sh <version> <dataset> <bsize> <missing> <seed> <rank_pseudo> [extra...]
#
# bsize "-" means "use the dataset's own sample count".
set -uo pipefail

VERSION="${1}"; DATASET="${2}"; BSIZE="${3}"; MISSING="${4}"; SEED="${5}"; RP="${6}"
shift 6
EXTRA="${*:-}"

echo "=== DUC experiment runner ==="
echo "version=$VERSION dataset=$DATASET B=$BSIZE missing=$MISSING seed=$SEED r_p=$RP"
echo "extra='$EXTRA'"
echo "host=$(hostname)  date=$(date -u)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no GPU"

# HTCondor drops transferred files in the scratch dir.
if [ -n "${_CONDOR_SCRATCH_DIR:-}" ]; then cd "$_CONDOR_SCRATCH_DIR"; fi

if [ -f project.tar.gz ]; then
    tar -xzf project.tar.gz
    [ -f .build_sha ] && echo "=== code SHA: $(cat .build_sha) ===" \
                      || echo "=== code SHA: UNKNOWN ==="
else
    echo "ERROR: project.tar.gz not found"; ls -la; exit 1
fi

mkdir -p data results data/logs

# --- datasets come from staging, one file per job -------------------------
declare -A MATFILE=(
    [ORL]="ORL_32x32.mat"
    [COIL20]="COIL20.mat"
    [COIL100]="COIL100.mat"
    [EYaleB]="YaleBCrop025.mat"
    [Flowers]="flowers.mat"
    [OxfordPet]="oxford_pet.mat"
    [HARUS]="HARUS.mat"
    [DSDD]="Dataset_for_Sensorless_Drive_diagnosis.mat"
)
if [ "$DATASET" != "synthetic" ]; then
    F="${MATFILE[$DATASET]:-}"
    if [ -z "$F" ]; then echo "ERROR: no .mat mapping for $DATASET"; exit 1; fi
    SRC="/staging/${STAGING_USER:-}/duc_data/$F"
    if [ -f "$SRC" ]; then
        echo "Copying $F from staging"
        cp "$SRC" data/
    elif [ -f "data/$F" ]; then
        echo "Using $F bundled in the tarball"
    else
        echo "ERROR: $F not in staging ($SRC) or tarball. Run stage_data.sh first."
        exit 1
    fi
fi

BFLAG=""
if [ "$BSIZE" != "-" ]; then BFLAG="--B $BSIZE"; fi

OUT="results/${DATASET}_${VERSION}_rp${RP}_m${MISSING}_s${SEED}.json"

cd src
python bench_v1_v3.py --single \
    --version "$VERSION" --dataset "$DATASET" $BFLAG \
    --missing "$MISSING" --seed "$SEED" --rank-pseudo "$RP" \
    --out "../$OUT" $EXTRA
STATUS=$?
cd ..

if [ $STATUS -ne 0 ]; then echo "run failed with status $STATUS"; exit $STATUS; fi

tar -czf results.tar.gz results/
echo "=== done $(date -u) — wrote $OUT ==="
