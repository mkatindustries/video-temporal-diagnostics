#!/bin/bash
# Download VCDB core dataset (~7 GB, 528 videos, 28 categories) from Google Drive.
#
# Source: https://drive.google.com/drive/folders/0B-b0CY525pH8NjdxbFNGY0JJdGs
# Requires: gdown (pip install gdown)
#
# Expected structure after download:
#   datasets/vcdb/core_dataset/
#     annotation/         # .txt files with copy pair annotations
#     core_dataset/       # 28 category subdirs with video files
#
# Usage:
#   bash scripts/download_vcdb.sh

set -euo pipefail

DATASET_ROOT="datasets/vcdb/core_dataset"
GDRIVE_FOLDER_ID="0B-b0CY525pH8NjdxbFNGY0JJdGs"

# Check if already downloaded (≥500 video files)
if [[ -d "$DATASET_ROOT" ]]; then
    video_count=$(find "$DATASET_ROOT" -type f \( -name "*.mp4" -o -name "*.flv" -o -name "*.avi" -o -name "*.wmv" -o -name "*.mkv" -o -name "*.webm" \) 2>/dev/null | wc -l)
    if [[ "$video_count" -ge 500 ]]; then
        echo "VCDB already downloaded ($video_count video files found). Skipping."
        exit 0
    fi
    echo "Partial download detected ($video_count video files). Re-downloading..."
fi

# Check for gdown
if ! command -v gdown &>/dev/null; then
    echo "ERROR: gdown not found."
    echo "Install it with: pip install gdown"
    exit 1
fi

mkdir -p "$DATASET_ROOT"

echo "Downloading VCDB core dataset (~7 GB) from Google Drive..."
echo "  Folder: https://drive.google.com/drive/folders/$GDRIVE_FOLDER_ID"
echo "  Destination: $DATASET_ROOT/"
echo ""

gdown --folder "https://drive.google.com/drive/folders/$GDRIVE_FOLDER_ID" -O "$DATASET_ROOT" --remaining-ok

# Summary
echo ""
echo "Done."
video_count=$(find "$DATASET_ROOT" -type f \( -name "*.mp4" -o -name "*.flv" -o -name "*.avi" -o -name "*.wmv" -o -name "*.mkv" -o -name "*.webm" \) 2>/dev/null | wc -l)
ann_count=$(find "$DATASET_ROOT" -name "*.txt" 2>/dev/null | wc -l)
echo "  Video files: $video_count"
echo "  Annotation files: $ann_count"
echo ""
echo "Structure:"
echo "  $DATASET_ROOT/annotation/     - copy pair annotations"
echo "  $DATASET_ROOT/core_dataset/   - 28 category subdirs with videos"
