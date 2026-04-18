#!/bin/bash
# Download MUVR dataset (dance + news partitions) from HuggingFace.
#
# Downloads:
#   - dance partition: videos (~13.5 GB) + annotations
#   - news partition: videos (~15.3 GB) + annotations
#   Total: ~29 GB
#
# Usage:
#   bash scripts/download_muvr.sh [--all]
#
# With --all: downloads all 5 partitions (~91 GB)

set -euo pipefail

DATASET_ROOT="datasets/muvr"
HF_BASE="https://huggingface.co/datasets/debby0527/MUVR/resolve/main"

# Default: dance + news only
PARTITIONS=("dance" "news")

if [[ "${1:-}" == "--all" ]]; then
    PARTITIONS=("dance" "instance" "news" "others" "region")
    echo "Downloading ALL partitions (~91 GB)"
else
    echo "Downloading dance + news partitions (~29 GB)"
    echo "  Use --all for all 5 partitions"
fi

mkdir -p "$DATASET_ROOT/videos" "$DATASET_ROOT/annotations"

# Download and extract video tars
for partition in "${PARTITIONS[@]}"; do
    tar_file="$DATASET_ROOT/videos/${partition}.tar"
    extract_dir="$DATASET_ROOT/videos/${partition}"

    if [[ -d "$extract_dir" ]]; then
        echo "[$partition] Videos already extracted at $extract_dir, skipping"
    else
        if [[ ! -f "$tar_file" ]]; then
            echo "[$partition] Downloading videos..."
            wget -c -O "$tar_file" "$HF_BASE/videos/all_videos/${partition}.tar"
        else
            echo "[$partition] Tar already downloaded: $tar_file"
        fi
        echo "[$partition] Extracting videos..."
        mkdir -p "$extract_dir"
        tar xf "$tar_file" -C "$extract_dir"
        echo "[$partition] Extracted to $extract_dir"
    fi
done

# Download annotations
for partition in "${PARTITIONS[@]}"; do
    ann_dir="$DATASET_ROOT/annotations/${partition}"
    mkdir -p "$ann_dir"

    for file in topics.json videos.json relationships.json queries.json; do
        out="$ann_dir/$file"
        if [[ -f "$out" ]]; then
            echo "[$partition] $file already exists, skipping"
        else
            echo "[$partition] Downloading $file..."
            wget -q -O "$out" "$HF_BASE/annotations/retrieval/${partition}/${file}" 2>/dev/null || \
                echo "[$partition] $file not found (optional)"
        fi
    done
done

echo ""
echo "Done. Dataset at: $DATASET_ROOT/"
echo ""
echo "Structure:"
echo "  $DATASET_ROOT/videos/{dance,news}/   - video files"
echo "  $DATASET_ROOT/annotations/{dance,news}/  - JSON annotations"
