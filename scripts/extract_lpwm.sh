#!/usr/bin/env bash
# Extract the LPWM archives (Panda + Super Mario Bros episodes) into
#   <base>/extracted/{panda_ds,smb_ep}/
# using per-archive temporary staging directories and an atomic rename into the
# final path, so an interrupted extraction can never leave a partial directory
# that looks complete. The source archives are never modified.
#
# Usage: scripts/extract_lpwm.sh [BASE_DIR]
#   BASE_DIR defaults to the shared checkpoint LPWM directory.
set -euo pipefail

BASE="${1:-/checkpoint/dream/arjangt/video_retrieval/datasets/lpwm}"
PANDA_TGZ="$BASE/panda_ds.tar.gz"
SMB_ZIP="$BASE/smb_ep.zip"
DEST="$BASE/extracted"

log() { printf '[%s] %s\n' "extract_lpwm" "$*"; }

[[ -f "$PANDA_TGZ" ]] || { echo "missing $PANDA_TGZ" >&2; exit 1; }
[[ -f "$SMB_ZIP"  ]] || { echo "missing $SMB_ZIP"  >&2; exit 1; }
mkdir -p "$DEST"

# --- Panda: tar.gz whose top-level entry is panda_ds/ ------------------------
if [[ -d "$DEST/panda_ds" ]]; then
    log "panda_ds already present at $DEST/panda_ds; skipping."
else
    STAGE="$BASE/.staging_panda.$$"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    log "extracting $(basename "$PANDA_TGZ") -> staging ..."
    # --no-same-owner: we are not root; do not attempt to restore the archive's
    # original uid/gid (would emit per-file "Operation not permitted" and make
    # GNU tar exit non-zero, aborting the script under `set -e`). Files are
    # extracted owned by the current user, which is what we want.
    tar --no-same-owner -xzf "$PANDA_TGZ" -C "$STAGE"
    [[ -d "$STAGE/panda_ds" ]] || { echo "panda_ds/ not found after extract" >&2; exit 1; }
    n_cfg=$(find "$STAGE/panda_ds" -mindepth 1 -maxdepth 1 -type d | wc -l)
    n_png=$(find "$STAGE/panda_ds" -name 'frame*.png' | wc -l)
    n_meta=$(find "$STAGE/panda_ds" -name 'metadata*.pt' | wc -l)
    log "panda staging: ${n_cfg} config dirs, ${n_png} frame PNGs, ${n_meta} metadata tensors"
    [[ "$n_cfg" -ge 1 && "$n_png" -ge 1 ]] || { echo "panda validation failed" >&2; exit 1; }
    mv "$STAGE/panda_ds" "$DEST/panda_ds"      # atomic rename within same FS
    rmdir "$STAGE"
    log "panda_ds -> $DEST/panda_ds (done)"
fi

# --- Mario: zip whose top-level entries are train/ val/ ----------------------
if [[ -d "$DEST/smb_ep" ]]; then
    log "smb_ep already present at $DEST/smb_ep; skipping."
else
    STAGE="$BASE/.staging_smb.$$"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
    log "extracting $(basename "$SMB_ZIP") -> staging ..."
    unzip -q "$SMB_ZIP" -d "$STAGE"
    [[ -d "$STAGE/train" && -d "$STAGE/val" ]] || { echo "train/ or val/ missing after extract" >&2; exit 1; }
    n_tr=$(find "$STAGE/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
    n_va=$(find "$STAGE/val"   -mindepth 1 -maxdepth 1 -type d | wc -l)
    n_png=$(find "$STAGE" -name '*.png' | wc -l)
    log "mario staging: ${n_tr} train episodes, ${n_va} val episodes, ${n_png} PNGs"
    [[ "$n_tr" -ge 1 && "$n_va" -ge 1 && "$n_png" -ge 1 ]] || { echo "mario validation failed" >&2; exit 1; }
    mv "$STAGE" "$DEST/smb_ep"                  # atomic rename of the whole staging dir
    log "smb_ep -> $DEST/smb_ep (done)"
fi

log "ALL DONE"
find "$DEST" -mindepth 1 -maxdepth 2 -type d 2>/dev/null | sort | head
