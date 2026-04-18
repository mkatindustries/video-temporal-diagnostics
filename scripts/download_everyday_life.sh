#!/bin/bash
# Download Everyday Life Aria recordings (≥60 min) from Manifold.
#
# Source: manifold://fair_data/tree/aria/Everyday_Life/<recording_id>/data.mp4
# Destination: datasets/everyday_life/<recording_id>/data.mp4
#
# Duration filter: fetches metadata.json (~17 KB) first to check duration_seconds,
# only downloads data.mp4 if recording meets minimum duration threshold.
#
# Usage:
#   bash scripts/download_everyday_life.sh                       # first 20 recordings ≥60 min
#   bash scripts/download_everyday_life.sh --max-recordings 5    # first 5
#   bash scripts/download_everyday_life.sh --min-duration-min 30 # ≥30 min
#   bash scripts/download_everyday_life.sh --all                 # all recordings ≥60 min

set -euo pipefail

DATASET_ROOT="datasets/everyday_life"
MANIFOLD_BASE="fair_data/tree/aria/Everyday_Life"

# Parse arguments
MAX_RECORDINGS=20
MIN_DURATION_MIN=60
ALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-recordings)
            MAX_RECORDINGS="$2"
            shift 2
            ;;
        --min-duration-min)
            MIN_DURATION_MIN="$2"
            shift 2
            ;;
        --all)
            ALL=true
            shift
            ;;
        *)
            echo "Usage: $0 [--max-recordings N] [--min-duration-min M] [--all]"
            exit 1
            ;;
    esac
done

MIN_DURATION_SEC=$((MIN_DURATION_MIN * 60))

# Check for manifold
if ! command -v manifold &>/dev/null; then
    echo "ERROR: manifold CLI not found."
    echo "This script requires Meta's internal manifold tool."
    exit 1
fi

echo "Discovering Everyday Life recording IDs..."
mapfile -t ALL_IDS < <(manifold ls "$MANIFOLD_BASE" 2>/dev/null | sed 's/^DIR[[:space:]]*//' | sed 's:/*$::' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//' | grep -v '^$')

if [[ ${#ALL_IDS[@]} -eq 0 ]]; then
    echo "ERROR: No recording IDs found at manifold://$MANIFOLD_BASE"
    exit 1
fi

echo "  Found ${#ALL_IDS[@]} total recordings"

if [[ "$ALL" == "true" ]]; then
    echo "Downloading ALL recordings (min duration: ${MIN_DURATION_MIN} min)"
else
    echo "Downloading up to $MAX_RECORDINGS recordings (min duration: ${MIN_DURATION_MIN} min)"
    echo "  Use --all for all recordings"
fi

mkdir -p "$DATASET_ROOT"

downloaded=0
skipped=0
failed=0
filtered=0
total=0

for rec_id in "${ALL_IDS[@]}"; do
    # Stop if we've reached max (unless --all)
    if [[ "$ALL" != "true" ]] && [[ "$downloaded" -ge "$MAX_RECORDINGS" ]]; then
        break
    fi

    total=$((total + 1))
    dest="$DATASET_ROOT/$rec_id/data.mp4"

    if [[ -f "$dest" ]]; then
        echo "[$rec_id] Already exists, skipping"
        skipped=$((skipped + 1))
        downloaded=$((downloaded + 1))
        continue
    fi

    # Fetch metadata.json (~17 KB) to check duration before downloading video
    meta_dir="$DATASET_ROOT/$rec_id"
    meta_file="$meta_dir/metadata.json"
    mkdir -p "$meta_dir"

    if ! manifold get "$MANIFOLD_BASE/$rec_id/metadata.json" "$meta_file" 2>/dev/null; then
        echo "[$rec_id] Failed to fetch metadata, skipping"
        failed=$((failed + 1))
        rm -f "$meta_file"
        rmdir --ignore-fail-on-non-empty "$meta_dir" 2>/dev/null || true
        continue
    fi

    # Extract duration_seconds from metadata.json
    duration=$(python3 -c "import json; print(json.load(open('$meta_file')).get('duration_seconds', 0))" 2>/dev/null || echo "0")
    duration_int=${duration%.*}
    duration_int=${duration_int:-0}
    duration_min=$((duration_int / 60))

    if [[ "$duration_int" -lt "$MIN_DURATION_SEC" ]]; then
        echo "[$rec_id] Too short (${duration_min} min < ${MIN_DURATION_MIN} min), skipping"
        rm -f "$meta_file"
        rmdir --ignore-fail-on-non-empty "$meta_dir" 2>/dev/null || true
        filtered=$((filtered + 1))
        continue
    fi

    echo "[$rec_id] Duration ${duration_min} min — downloading data.mp4..."

    if manifold get "$MANIFOLD_BASE/$rec_id/data.mp4" "$dest" 2>/dev/null; then
        downloaded=$((downloaded + 1))
        echo "[$rec_id] Done"
    else
        echo "[$rec_id] FAILED to download video"
        failed=$((failed + 1))
        rm -f "$dest"
    fi
done

# Summary
echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "  Recordings scanned:  $total"
echo "  Total kept:           $downloaded (skipped: $skipped, new: $((downloaded - skipped)))"
echo "  Filtered (too short): $filtered"
echo "  Failed:               $failed"
echo ""
echo "Dataset at: $DATASET_ROOT/"
