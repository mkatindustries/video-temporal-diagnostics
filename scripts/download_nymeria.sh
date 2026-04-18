#!/bin/bash
# Download a subset of Nymeria multi-activity person-sessions from Manifold.
#
# Source: manifold://fair_data/tree/aria/nymeria/<recording_id>/data.mp4
# Destination: datasets/nymeria/<recording_id>/<recording_id>/data.mp4
#   (double-nested to match eval_nymeria_activities.py expectations)
#
# Discovery logic:
#   1. List all recording IDs via manifold ls
#   2. Parse regex to group by person-session
#   3. Keep only sessions with ≥2 activities
#   4. Select first N sessions (sorted alphabetically)
#
# Usage:
#   bash scripts/download_nymeria.sh                  # first 15 sessions
#   bash scripts/download_nymeria.sh --max-sessions 5 # first 5 sessions
#   bash scripts/download_nymeria.sh --all            # all multi-activity sessions

set -euo pipefail

DATASET_ROOT="datasets/nymeria"
MANIFOLD_BASE="fair_data/tree/aria/nymeria"

# Parse arguments
MAX_SESSIONS=30
ALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-sessions)
            MAX_SESSIONS="$2"
            shift 2
            ;;
        --all)
            ALL=true
            shift
            ;;
        *)
            echo "Usage: $0 [--max-sessions N] [--all]"
            exit 1
            ;;
    esac
done

# Check for manifold
if ! command -v manifold &>/dev/null; then
    echo "ERROR: manifold CLI not found."
    echo "This script requires Meta's internal manifold tool."
    exit 1
fi

echo "Discovering Nymeria recording IDs..."
mapfile -t ALL_IDS < <(manifold ls "$MANIFOLD_BASE" 2>/dev/null | sed 's/^DIR[[:space:]]*//' | sed 's:/*$::' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//' | grep -v '^$')

if [[ ${#ALL_IDS[@]} -eq 0 ]]; then
    echo "ERROR: No recording IDs found at manifold://$MANIFOLD_BASE"
    exit 1
fi

echo "  Found ${#ALL_IDS[@]} total recordings"
echo "  Sample IDs: ${ALL_IDS[0]}, ${ALL_IDS[1]:-}, ${ALL_IDS[2]:-}"

# Group by person-session using regex:
#   {date}_{session}_{first}_{last}_{actN}_{hash}
# Pattern matches names like: 20230628_s1_hayley_little_act1_ab12cd
# Uses case-insensitive name parts ([A-Za-z]+) and flexible hash ([A-Za-z0-9]+)
declare -A SESSION_RECORDINGS  # session -> space-separated recording IDs
declare -A SESSION_ACTIVITIES  # session -> space-separated activity IDs

PATTERN='^([0-9]{8}_s[0-9]+_[A-Za-z]+_[A-Za-z]+)_(act[0-9]+)_([A-Za-z0-9]+)$'
unmatched=0

for rec_id in "${ALL_IDS[@]}"; do
    if [[ "$rec_id" =~ $PATTERN ]]; then
        session="${BASH_REMATCH[1]}"
        activity="${BASH_REMATCH[2]}"
        SESSION_RECORDINGS["$session"]="${SESSION_RECORDINGS[$session]:-} $rec_id"
        SESSION_ACTIVITIES["$session"]="${SESSION_ACTIVITIES[$session]:-} $activity"
    else
        unmatched=$((unmatched + 1))
    fi
done

echo "  Matched regex: $(( ${#ALL_IDS[@]} - unmatched )) / ${#ALL_IDS[@]}"
if [[ "$unmatched" -gt 0 ]]; then
    # Show a few unmatched IDs for debugging
    echo "  Unmatched samples:"
    count=0
    for rec_id in "${ALL_IDS[@]}"; do
        if ! [[ "$rec_id" =~ $PATTERN ]]; then
            echo "    '$rec_id'"
            count=$((count + 1))
            if [[ "$count" -ge 5 ]]; then
                break
            fi
        fi
    done
fi

# Filter to multi-activity sessions (≥2 activities)
MULTI_SESSIONS=()
for session in $(echo "${!SESSION_RECORDINGS[@]}" | tr ' ' '\n' | sort); do
    # Count unique activities
    n_acts=$(echo "${SESSION_ACTIVITIES[$session]}" | tr ' ' '\n' | sort -u | grep -c .)
    if [[ "$n_acts" -ge 2 ]]; then
        MULTI_SESSIONS+=("$session")
    fi
done

echo "  Multi-activity sessions (≥2 activities): ${#MULTI_SESSIONS[@]}"

if [[ ${#MULTI_SESSIONS[@]} -eq 0 ]]; then
    echo "ERROR: No multi-activity sessions found."
    exit 1
fi

# Select sessions
if [[ "$ALL" == "true" ]]; then
    SELECTED_SESSIONS=("${MULTI_SESSIONS[@]}")
    echo "Downloading ALL ${#SELECTED_SESSIONS[@]} multi-activity sessions"
else
    SELECTED_SESSIONS=("${MULTI_SESSIONS[@]:0:$MAX_SESSIONS}")
    echo "Downloading first $MAX_SESSIONS multi-activity sessions (use --all for all)"
fi

# Download recordings
downloaded=0
skipped=0
failed=0
total=0

for session in "${SELECTED_SESSIONS[@]}"; do
    rec_ids=(${SESSION_RECORDINGS[$session]})
    acts=$(echo "${SESSION_ACTIVITIES[$session]}" | tr ' ' '\n' | sort -u | tr '\n' ',' | sed 's/,$//')
    echo ""
    echo "[$session] ${#rec_ids[@]} recordings (activities: $acts)"

    for rec_id in "${rec_ids[@]}"; do
        total=$((total + 1))
        dest="$DATASET_ROOT/$rec_id/$rec_id/data.mp4"

        if [[ -f "$dest" ]]; then
            echo "  [$rec_id] Already exists, skipping"
            skipped=$((skipped + 1))
            continue
        fi

        mkdir -p "$(dirname "$dest")"
        echo "  [$rec_id] Downloading..."

        if manifold get "$MANIFOLD_BASE/$rec_id/data.mp4" "$dest" 2>/dev/null; then
            downloaded=$((downloaded + 1))
            echo "  [$rec_id] Done"
        else
            failed=$((failed + 1))
            echo "  [$rec_id] FAILED"
            rm -f "$dest"
        fi
    done
done

# Summary
echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "  Sessions selected: ${#SELECTED_SESSIONS[@]}"
echo "  Total recordings:  $total"
echo "  Downloaded:         $downloaded"
echo "  Skipped (exists):   $skipped"
echo "  Failed:             $failed"
echo ""
echo "Dataset at: $DATASET_ROOT/"
