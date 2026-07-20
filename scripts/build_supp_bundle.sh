#!/usr/bin/env bash
# Build NeurIPS 2026 E&D supplementary zip from the current repo working tree.
# E&D 2026 is DOUBLE-BLIND for evaluation-methodology submissions. The paper
# directory (submitted separately), standalone license, datasets, and local
# paper-editing helpers are excluded. Emits /tmp/neurips_supp.zip.
# Set ANONYMITY_PATTERN to an extended regex matching private names, emails,
# usernames, and repository identifiers before running the bundle audit.
set -euo pipefail

: "${ANONYMITY_PATTERN:?Set ANONYMITY_PATTERN for the double-blind identity scan}"

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
    --exclude='LICENSE' \
    --exclude='texput.log' \
    --exclude='datasets' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.ruff_cache' \
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
echo "--- identity scan (double-blind: should be EMPTY) ---"
matches="$(
    unzip -p "$OUT" 2>/dev/null \
        | strings \
        | grep -iE -m 5 -- "$ANONYMITY_PATTERN" \
        || true
)"
if [[ -n "$matches" ]]; then
    printf '%s\n' "$matches"
    echo "ERROR: identifying information found in bundle"
    exit 1
fi
echo "OK: no configured identifying patterns found in bundle"
echo "--- key additions present? ---"
unzip -l "$OUT" | grep -E "tara|pl_stitch|gemini|diagnostics|viclip" | head
echo "--- LICENSE absent? (should be excluded under double-blind) ---"
unzip -l "$OUT" | grep -E "LICENSE" && echo "WARNING: LICENSE present — would de-anonymize via copyright" || echo "OK: LICENSE excluded"
