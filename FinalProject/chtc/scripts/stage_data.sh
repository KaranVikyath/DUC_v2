#!/bin/bash
# Verify the dataset files are present on the submit node and sized as expected.
#
# NOTE: these do NOT go in /staging. That location is for files individually
# larger than 1 GB, and the directories are provisioned by CHTC staff (trying to
# mkdir one yourself gives "Permission denied"). The largest file here is 44 MB
# and each job needs exactly one of them, which is comfortably inside HTCondor's
# <100 MB per-file / <500 MB per-job transfer limits. So they live in your home
# directory and ride along via transfer_input_files.
#
# Put them there from your laptop with:
#   scp FinalProject/data/*.mat veerannarupa@ap2002.chtc.wisc.edu:~/duc_data/
#
#   bash chtc/scripts/stage_data.sh [data_dir]
set -euo pipefail

DEST="${1:-$HOME/duc_data}"
EXPECTED="ORL_32x32.mat COIL20.mat COIL100.mat YaleBCrop025.mat
          flowers.mat oxford_pet.mat HARUS.mat
          Dataset_for_Sensorless_Drive_diagnosis.mat"

echo "Checking $DEST"
if [ ! -d "$DEST" ]; then
    echo "  missing — create it and copy the .mat files over:"
    echo "    mkdir -p $DEST"
    echo "    scp FinalProject/data/*.mat veerannarupa@ap2002.chtc.wisc.edu:$DEST/"
    exit 1
fi

missing=0
for f in $EXPECTED; do
    if [ -f "$DEST/$f" ]; then
        printf '  %-45s %8s\n' "$f" "$(du -h "$DEST/$f" | cut -f1)"
    else
        printf '  %-45s %8s\n' "$f" "MISSING"
        missing=$((missing + 1))
    fi
done

echo
echo "total: $(du -sh "$DEST" | cut -f1)"
echo "home quota:"
quota -vs 2>/dev/null || df -h "$HOME" | tail -1

if [ "$missing" -gt 0 ]; then
    echo
    echo "$missing file(s) missing — jobs for those datasets will fail."
    exit 1
fi
echo "All datasets present."
