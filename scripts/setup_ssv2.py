#!/usr/bin/env python3
"""Set up Something-Something V2 for the motion-direction protocol.

SSv2 contains 174 action templates, of which several appear in chiral
pairs (e.g., "Pushing [something] from left to right" vs. "...from right
to left"; "Moving [something] up" vs. "Moving [something] down").
For motion-direction retrieval, we want pairs that share appearance but
oppose motion direction — exactly what chiral SSv2 templates provide.

This script discovers chiral pairs from the SSv2 label file by matching
template-name patterns, samples N videos per template, and writes a
manifest the eval driver consumes.

SSv2 videos are not redistributable; you must download them from the
official source (Qualcomm AI Datasets) under their license. This script
expects the videos and label JSONs already on disk.

Usage:
    python scripts/setup_ssv2.py \\
        --ssv2-dir datasets/ssv2 \\
        --videos-per-template 50 \\
        --split validation \\
        --out datasets/ssv2/spike_manifest.json

Expected layout under --ssv2-dir:
    labels/labels.json                # 174 templates with placeholder slots
    labels/validation.json            # validation set annotations
    labels/train.json                 # training set annotations (optional)
    videos/                           # *.webm files, one per video_id
"""

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


# Chiral-pair patterns. Each entry is (regex_a, regex_b, axis_label).
# When a template name matches regex_a, we look for a corresponding
# template that matches regex_b after substituting the axis tag.
CHIRAL_PATTERNS = [
    # "Pushing X from left to right" <-> "...from right to left"
    (
        r"^(?P<verb>.+?)\s+from\s+left\s+to\s+right(?P<tail>.*)$",
        r"^(?P<verb>.+?)\s+from\s+right\s+to\s+left(?P<tail>.*)$",
        "left_right",
    ),
    # "Moving X up" <-> "Moving X down"
    (
        r"^(?P<verb>.+?)\s+up(?P<tail>.*)$",
        r"^(?P<verb>.+?)\s+down(?P<tail>.*)$",
        "up_down",
    ),
    # "Tilting X with X on it slightly so it doesn't fall down" type pairs
    # are not handled here — we restrict to clean directional axes.
]


@dataclass
class ManifestEntry:
    video_id: str
    template_id: int
    template_name: str
    chiral_axis: str  # e.g. "left_right" or "up_down"
    chiral_direction: str  # e.g. "left_to_right" or "up"
    chiral_pair_id: int  # shared across the two templates in a pair
    video_path: str


def load_labels(labels_path: Path) -> dict[str, int]:
    """Load SSv2 labels.json: maps template_name -> template_id (string)."""
    with open(labels_path) as f:
        labels_raw = json.load(f)
    # SSv2 labels.json maps template_name -> "id_string"; normalize to int.
    return {name: int(tid) for name, tid in labels_raw.items()}


def load_split(split_path: Path) -> list[dict]:
    """Load SSv2 train.json / validation.json: list of {id, template, label, ...}."""
    with open(split_path) as f:
        return json.load(f)


def normalize_template(name: str) -> str:
    """Strip placeholder brackets so 'Pushing [something]' compares to
    'Pushing X' regardless of slot fill."""
    return re.sub(r"\[[^\]]*\]", "X", name).strip()


def find_chiral_pairs(
    label_to_id: dict[str, int],
) -> list[tuple[int, int, str, str, str]]:
    """Return chiral pairs as (id_a, id_b, axis, direction_a, direction_b).

    direction_a corresponds to id_a (e.g. "left_to_right"); direction_b to id_b.
    """
    pairs: list[tuple[int, int, str, str, str]] = []
    used: set[int] = set()
    normalized = {name: normalize_template(name) for name in label_to_id}

    for pattern_a, pattern_b, axis in CHIRAL_PATTERNS:
        regex_a = re.compile(pattern_a, re.IGNORECASE)
        regex_b = re.compile(pattern_b, re.IGNORECASE)

        for name_a, norm_a in normalized.items():
            id_a = label_to_id[name_a]
            if id_a in used:
                continue
            match = regex_a.match(norm_a)
            if not match:
                continue
            verb = match.group("verb").strip()
            tail = match.group("tail").strip()

            # Build the canonical "b" name and search for a matching template.
            for name_b, norm_b in normalized.items():
                id_b = label_to_id[name_b]
                if id_b == id_a or id_b in used:
                    continue
                match_b = regex_b.match(norm_b)
                if not match_b:
                    continue
                verb_b = match_b.group("verb").strip()
                tail_b = match_b.group("tail").strip()
                if verb_b == verb and tail_b == tail:
                    direction_a = (
                        "left_to_right" if axis == "left_right" else "up"
                    )
                    direction_b = (
                        "right_to_left" if axis == "left_right" else "down"
                    )
                    pairs.append((id_a, id_b, axis, direction_a, direction_b))
                    used.add(id_a)
                    used.add(id_b)
                    break

    return pairs


def build_manifest(
    annotations: list[dict],
    label_to_id: dict[str, int],
    chiral_pairs: list[tuple[int, int, str, str, str]],
    videos_dir: Path,
    videos_per_template: int,
    seed: int,
) -> list[ManifestEntry]:
    """Sample N videos per chiral template and verify the file exists."""
    rng = random.Random(seed)
    by_template: dict[int, list[dict]] = defaultdict(list)
    id_to_name = {tid: name for name, tid in label_to_id.items()}

    template_ids = set()
    pair_assignment: dict[int, tuple[int, str, str]] = {}
    for pair_idx, (id_a, id_b, axis, dir_a, dir_b) in enumerate(chiral_pairs):
        template_ids.add(id_a)
        template_ids.add(id_b)
        pair_assignment[id_a] = (pair_idx, axis, dir_a)
        pair_assignment[id_b] = (pair_idx, axis, dir_b)

    for ann in annotations:
        # SSv2 annotation: {"id": "<video_id>", "template": "<name>", ...}
        template_name = ann.get("template")
        if template_name is None:
            continue
        # Normalize template_name placeholders match labels.json placeholders.
        # SSv2 stores templates as e.g. "Pushing [something] from left to right".
        if template_name not in label_to_id:
            continue
        tid = label_to_id[template_name]
        if tid in template_ids:
            by_template[tid].append(ann)

    manifest: list[ManifestEntry] = []
    missing = 0
    for tid, anns in by_template.items():
        rng.shuffle(anns)
        kept = 0
        for ann in anns:
            if kept >= videos_per_template:
                break
            video_id = str(ann["id"])
            video_path = videos_dir / f"{video_id}.webm"
            if not video_path.exists():
                missing += 1
                continue
            pair_idx, axis, direction = pair_assignment[tid]
            manifest.append(
                ManifestEntry(
                    video_id=video_id,
                    template_id=tid,
                    template_name=id_to_name[tid],
                    chiral_axis=axis,
                    chiral_direction=direction,
                    chiral_pair_id=pair_idx,
                    video_path=str(video_path),
                )
            )
            kept += 1

    if missing:
        print(f"  warning: {missing} videos referenced by annotations were not "
              f"found on disk under {videos_dir}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Set up SSv2 motion-direction manifest")
    parser.add_argument(
        "--ssv2-dir",
        type=Path,
        default=Path("datasets/ssv2"),
        help="SSv2 root directory containing labels/ and videos/",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation"],
        default="validation",
        help="SSv2 split to sample from (validation is default; smaller, "
             "still has all 174 templates)",
    )
    parser.add_argument(
        "--videos-per-template",
        type=int,
        default=50,
        help="Number of videos to sample per chiral template",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output manifest path (default: <ssv2-dir>/<split>_manifest.json)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels_path = args.ssv2_dir / "labels" / "labels.json"
    split_path = args.ssv2_dir / "labels" / f"{args.split}.json"
    videos_dir = args.ssv2_dir / "videos"

    if not labels_path.exists():
        raise SystemExit(f"missing {labels_path}; download SSv2 labels first")
    if not split_path.exists():
        raise SystemExit(f"missing {split_path}; expected split JSON")
    if not videos_dir.exists():
        raise SystemExit(f"missing {videos_dir}; expected video files in *.webm")

    print(f"Loading labels from {labels_path} ...")
    label_to_id = load_labels(labels_path)
    print(f"  {len(label_to_id)} templates")

    print("Discovering chiral pairs ...")
    chiral_pairs = find_chiral_pairs(label_to_id)
    if not chiral_pairs:
        raise SystemExit("no chiral pairs found; check CHIRAL_PATTERNS")
    for idx, (id_a, id_b, axis, dir_a, dir_b) in enumerate(chiral_pairs):
        name_a = next(n for n, i in label_to_id.items() if i == id_a)
        name_b = next(n for n, i in label_to_id.items() if i == id_b)
        print(f"  pair {idx} ({axis}): [{id_a}] {name_a}  <-->  [{id_b}] {name_b}")

    print(f"Loading {args.split} annotations from {split_path} ...")
    annotations = load_split(split_path)
    print(f"  {len(annotations)} annotation rows")

    print(f"Sampling {args.videos_per_template} videos per template ...")
    manifest = build_manifest(
        annotations,
        label_to_id,
        chiral_pairs,
        videos_dir,
        args.videos_per_template,
        args.seed,
    )
    print(f"  {len(manifest)} videos kept ({len(set(e.template_id for e in manifest))} templates)")

    out_path = args.out or args.ssv2_dir / f"{args.split}_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "split": args.split,
                "videos_per_template": args.videos_per_template,
                "seed": args.seed,
                "chiral_pairs": [
                    {
                        "pair_id": idx,
                        "axis": axis,
                        "id_a": id_a,
                        "id_b": id_b,
                        "direction_a": dir_a,
                        "direction_b": dir_b,
                    }
                    for idx, (id_a, id_b, axis, dir_a, dir_b) in enumerate(
                        chiral_pairs
                    )
                ],
                "entries": [asdict(e) for e in manifest],
            },
            f,
            indent=2,
        )
    print(f"\nWrote manifest to {out_path}")


if __name__ == "__main__":
    main()
