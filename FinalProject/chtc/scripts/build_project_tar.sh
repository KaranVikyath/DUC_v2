#!/bin/bash
# Build project.tar.gz from FinalProject/, excluding data and bulky paths.
#
# Datasets are NOT bundled: they total ~130 MB, which is past CHTC's guidance
# for transfer_input_files. Stage them once with stage_data.sh instead.
#
# Run from anywhere:  bash chtc/scripts/build_project_tar.sh
set -euo pipefail

cd "$(dirname "$0")/../.."          # -> FinalProject/

OUT="${OUT:-project.tar.gz}"
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")
DIRTY=""
if ! git diff --quiet 2>/dev/null; then DIRTY=" [DIRTY working tree]"; fi
echo "Building $OUT from commit $SHA${DIRTY} ..."

# Embed the SHA so the worker can print exactly which code it ran.
echo "$SHA${DIRTY}" > .build_sha
trap 'rm -f .build_sha' EXIT

# --anchored so excludes match from the archive root: without it "data" would
# also match src/data/ and "logs" would match anything named logs/ at any depth.
tar --anchored \
    --exclude-vcs \
    --exclude="./$OUT" \
    --exclude="./data/*.mat" \
    --exclude="./data/*.csv" \
    --exclude="./data/logs" \
    --exclude="./data/shards" \
    --exclude="./build" \
    --exclude="*/__pycache__" \
    --exclude="*.tar.gz" \
    -czf "$OUT" \
    ./src ./chtc ./.build_sha

echo "Created $OUT ($(du -sm "$OUT" | cut -f1) MB) at $(pwd)"
