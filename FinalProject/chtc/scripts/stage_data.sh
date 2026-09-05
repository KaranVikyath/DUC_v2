#!/bin/bash
# One-time: copy the .mat/.csv datasets into CHTC staging.
#
# Datasets are ~130 MB total, past the guidance for transfer_input_files, so
# they live in /staging and each job copies only the one it needs.
#
#   bash chtc/scripts/stage_data.sh <staging_user> [source_dir]
set -euo pipefail

USER_NAME="${1:?usage: stage_data.sh <staging_user> [source_dir]}"
SRC="${2:-$(cd "$(dirname "$0")/../.." && pwd)/data}"
DEST="/staging/$USER_NAME/duc_data"

echo "Staging from $SRC -> $DEST"
mkdir -p "$DEST"
shopt -s nullglob
for f in "$SRC"/*.mat "$SRC"/*.csv; do
    echo "  $(basename "$f") ($(du -h "$f" | cut -f1))"
    cp -n "$f" "$DEST/"
done
echo
echo "Staged:"
ls -lh "$DEST"
echo
echo "Quota check (staging is limited; see CHTC docs):"
du -sh "/staging/$USER_NAME" 2>/dev/null || true
