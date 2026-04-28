#!/usr/bin/env bash
# Build NeurIPS 2026 E&D supplementary zip from the current repo working tree.
# E&D is single-blind: author identity is preserved (LICENSE included,
# Citation block in diagnostics README intact). Excludes the paper subdir
# (submitted separately), datasets, and uncommitted paper-editing helpers.
# Emits /tmp/neurips_supp.zip.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="/tmp/supp_build"
OUT="/tmp/neurips_supp.zip"

rm -rf "$STAGE"
mkdir -p "$STAGE/supplementary"

rsync -a \
    --exclude='.git' \
    --exclude='.claude' \
    --exclude='.codex' \
    --exclude='paper' \
    --exclude='texput.log' \
    --exclude='datasets' \
    --exclude='slurm_jobs' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.egg-info' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='build' \
    --exclude='dist' \
    --exclude='.DS_Store' \
    --exclude='*.whl' \
    --exclude='*.ckpt' \
    --exclude='*.pth' \
    --exclude='trajectories.png' \
    --exclude='finalize_paper.py' \
    --exclude='fix_paper.py' \
    --exclude='fix_paper_2.py' \
    --exclude='scripts/build_supp_bundle.sh' \
    "$REPO/" "$STAGE/supplementary/"

rm -f "$OUT"
( cd "$STAGE" && zip -rq "$OUT" supplementary )

echo "--- bundle ---"
ls -lh "$OUT"
echo "--- file count ---"
unzip -l "$OUT" | tail -1
echo "--- attribution check (E&D single-blind: should be present) ---"
unzip -p "$OUT" 2>/dev/null | strings | grep -iE "mkat|talattof|arjang" | head -5 || echo "WARNING: no attribution found — verify LICENSE and diagnostics Citation are intact"
echo "--- key additions present? ---"
unzip -l "$OUT" | grep -E "tara|pl_stitch|gemini|diagnostics|viclip" | head
echo "--- LICENSE present? ---"
unzip -l "$OUT" | grep -E "LICENSE" || echo "WARNING: LICENSE missing"
