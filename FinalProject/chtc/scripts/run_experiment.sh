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
# The DAG uses "-" as a placeholder for "no extra flags" because an empty VARS
# value would collapse the argument list. Forwarding it verbatim makes argparse
# choke on a bare "-" positional, which failed nearly every job.
[ "$EXTRA" = "-" ] && EXTRA=""

echo "=== DUC experiment runner ==="
echo "version=$VERSION dataset=$DATASET B=$BSIZE missing=$MISSING seed=$SEED r_p=$RP"
echo "extra='$EXTRA'"
echo "host=$(hostname)  date=$(date -u)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no GPU"

# HTCondor drops transferred files in the scratch dir.
if [ -n "${_CONDOR_SCRATCH_DIR:-}" ]; then cd "$_CONDOR_SCRATCH_DIR"; fi

# transfer_output_files names results.tar.gz unconditionally, so if the script
# dies before creating it HTCondor reports an opaque "transfer output files
# failure" and the real error never leaves the execute node. Guarantee the
# tarball exists on every exit path, with the log inside.
mkdir -p results
trap 'echo "exit=$?" > results/exit_status.txt 2>/dev/null; \
      tar -czf results.tar.gz results 2>/dev/null || true' EXIT

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
    # HTCondor drops transfer_input_files in the scratch dir alongside the tarball.
    if [ -f "$F" ]; then
        echo "Using transferred $F ($(du -h "$F" | cut -f1))"
        mv "$F" data/
    elif [ -f "data/$F" ]; then
        echo "Using $F already in data/"
    elif [ -n "${STAGING_USER:-}" ] && [ -f "/staging/$STAGING_USER/duc_data/$F" ]; then
        echo "Copying $F from staging"          # only if a staging dir was provisioned
        cp "/staging/$STAGING_USER/duc_data/$F" data/
    else
        echo "ERROR: $F was not transferred. Check that the DAG sets matfile= and"
        echo "       that the file exists on the submit node."
        ls -la; exit 1
    fi
fi

BFLAG=""
if [ "$BSIZE" != "-" ]; then BFLAG="--B $BSIZE"; fi

OUT="results/${DATASET}_${VERSION}_rp${RP}_m${MISSING}_s${SEED}.json"

cd src
# Tee the run so a traceback comes back inside results.tar.gz rather than being
# lost on the execute node.
python bench_v1_v3.py --single \
    --version "$VERSION" --dataset "$DATASET" $BFLAG \
    --missing "$MISSING" --seed "$SEED" --rank-pseudo "$RP" \
    --out "../$OUT" $EXTRA 2>&1 | tee "../results/run.log"
STATUS=${PIPESTATUS[0]}
cd ..

if [ "$STATUS" -ne 0 ]; then
    echo "run failed with status $STATUS — see results/run.log"
    exit "$STATUS"          # the EXIT trap still ships results.tar.gz
fi

echo "=== done $(date -u) — wrote $OUT ==="
