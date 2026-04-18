#!/usr/bin/env python3
"""Download and prepare nuScenes data for video retrieval experiments.

Extracts metadata, installs nuscenes-devkit if needed, and downloads
front-camera keyframe images for maneuver discrimination experiments.

nuScenes trainval has 850 scenes (~34K keyframes × 6 cameras). For our
experiment we only need the front camera (CAM_FRONT) keyframes, which
are ~5.5GB vs ~300GB for the full blob set.

Prerequisites:
    - nuScenes account at https://www.nuscenes.org/nuscenes
    - v1.0-trainval_meta.tar already at NUSCENES_DIR (metadata only, 2.5GB)
    - CAN bus expansion already at NUSCENES_DIR/can_bus/

Steps:
    1. Extract v1.0-trainval_meta.tar → v1.0-trainval/ + maps/
    2. pip install nuscenes-devkit (if not installed)
    3. Download CAM_FRONT keyframe blobs via AWS (no auth needed for US region)
       OR: user provides pre-downloaded blob tarballs

Usage:
    # Step 1: Extract metadata + install devkit
    python scripts/download_nuscenes.py --nuscenes-dir ./data/nuscenes --setup

    # Step 2: Download front-camera blobs (requires internet)
    python scripts/download_nuscenes.py --nuscenes-dir ./data/nuscenes --download-blobs

    # Step 3: Verify data completeness
    python scripts/download_nuscenes.py --nuscenes-dir ./data/nuscenes --verify
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# nuScenes trainval camera blob URLs (AWS, no auth required)
# Each blob is ~2-4GB. We only need the ones containing CAM_FRONT samples.
# Full list: https://www.nuscenes.org/nuscenes#download
BLOB_URLS = {
    f"v1.0-trainval{i:02d}_blobs.tar": f"https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval{i:02d}_blobs.tar"
    for i in range(1, 11)
}


def extract_metadata(nuscenes_dir: Path) -> None:
    """Extract v1.0-trainval_meta.tar if not already extracted."""
    meta_tar = nuscenes_dir / "v1.0-trainval_meta.tar"
    meta_dir = nuscenes_dir / "v1.0-trainval"

    if meta_dir.exists() and (meta_dir / "scene.json").exists():
        print(f"  Metadata already extracted at {meta_dir}")
        return

    if not meta_tar.exists():
        print(f"  ERROR: {meta_tar} not found")
        print("  Download from https://www.nuscenes.org/nuscenes#download")
        sys.exit(1)

    print(f"  Extracting {meta_tar}...")
    with tarfile.open(meta_tar, "r") as tf:
        tf.extractall(path=nuscenes_dir)
    print(f"  Extracted to {meta_dir}")


def install_devkit() -> None:
    """Install nuscenes-devkit if not available."""
    try:
        import nuscenes  # type: ignore[import]  # noqa: F401

        print(f"  nuscenes-devkit already installed: {nuscenes.__file__}")
    except ImportError:
        print("  Installing nuscenes-devkit...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "nuscenes-devkit"],
            stdout=subprocess.DEVNULL,
        )
        print("  Installed nuscenes-devkit")


def setup(nuscenes_dir: Path) -> None:
    """Extract metadata and install devkit."""
    print("\n=== Setup ===")
    extract_metadata(nuscenes_dir)
    install_devkit()

    # Verify
    meta_dir = nuscenes_dir / "v1.0-trainval"
    expected = [
        "scene.json",
        "sample.json",
        "sample_data.json",
        "ego_pose.json",
        "sensor.json",
        "calibrated_sensor.json",
    ]
    missing = [f for f in expected if not (meta_dir / f).exists()]
    if missing:
        print(f"  WARNING: Missing metadata files: {missing}")
    else:
        print("  All metadata files present")

    # Count scenes
    with open(meta_dir / "scene.json") as f:
        scenes = json.load(f)
    print(f"  Scenes: {len(scenes)}")

    # Check CAN bus
    can_dir = nuscenes_dir / "can_bus" / "can_bus"
    if can_dir.exists():
        n_can = len([f for f in os.listdir(can_dir) if f.endswith("_pose.json")])
        print(f"  CAN bus scenes: {n_can}")
    else:
        print("  WARNING: CAN bus data not found")


def download_blobs(nuscenes_dir: Path) -> None:
    """Download camera blob tarballs and extract them."""
    print("\n=== Downloading camera blobs ===")

    samples_dir = nuscenes_dir / "samples" / "CAM_FRONT"
    if samples_dir.exists():
        n_existing = len(list(samples_dir.glob("*.jpg")))
        if n_existing > 30000:
            print(f"  CAM_FRONT already has {n_existing} images, skipping download")
            return

    blob_dir = nuscenes_dir / "blobs"
    blob_dir.mkdir(exist_ok=True)

    for blob_name, url in BLOB_URLS.items():
        blob_path = blob_dir / blob_name
        if blob_path.exists():
            print(f"  {blob_name} already downloaded")
            continue

        print(f"  Downloading {blob_name}...")
        try:
            urlretrieve(url, blob_path, reporthook=_download_progress)
            print()
        except Exception as e:
            print(f"\n  ERROR downloading {blob_name}: {e}")
            print("  You may need to download manually from:")
            print(f"    {url}")
            continue

    # Extract blobs
    print("\n  Extracting blob tarballs...")
    for blob_name in sorted(BLOB_URLS.keys()):
        blob_path = blob_dir / blob_name
        if not blob_path.exists():
            continue
        print(f"  Extracting {blob_name}...")
        with tarfile.open(blob_path, "r") as tf:
            tf.extractall(path=nuscenes_dir)

    # Check result
    if samples_dir.exists():
        n_images = len(list(samples_dir.glob("*.jpg")))
        print(f"  CAM_FRONT images: {n_images}")
    else:
        print("  WARNING: CAM_FRONT directory not created")
        print("  Blobs may contain a different directory structure")
        print(f"  Check: ls {nuscenes_dir}/samples/")


def _download_progress(block_num, block_size, total_size):
    """Progress callback for urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)", end="", flush=True)


def verify(nuscenes_dir: Path) -> None:
    """Verify nuScenes data completeness for video retrieval experiments."""
    print("\n=== Verification ===")

    # Metadata
    meta_dir = nuscenes_dir / "v1.0-trainval"
    if not meta_dir.exists():
        print("  FAIL: Metadata not extracted (run --setup first)")
        return

    with open(meta_dir / "scene.json") as f:
        scenes = json.load(f)
    print(f"  Scenes: {len(scenes)} (expected: 850)")

    with open(meta_dir / "sample.json") as f:
        samples = json.load(f)
    print(f"  Keyframe samples: {len(samples)} (expected: ~34149)")

    # Front camera images
    cam_dir = nuscenes_dir / "samples" / "CAM_FRONT"
    if cam_dir.exists():
        n_images = len(list(cam_dir.glob("*.jpg")))
        print(f"  CAM_FRONT images: {n_images}")
        if n_images >= 34000:
            print("  OK: Sufficient front-camera keyframes for experiments")
        else:
            print(f"  WARNING: Expected ~34149, got {n_images}")
    else:
        print("  FAIL: No CAM_FRONT images (run --download-blobs)")

    # CAN bus
    can_dir = nuscenes_dir / "can_bus" / "can_bus"
    if can_dir.exists():
        n_pose = len([f for f in os.listdir(can_dir) if f.endswith("_pose.json")])
        print(f"  CAN bus pose files: {n_pose}")
    else:
        print("  FAIL: No CAN bus data")

    # devkit
    try:
        import nuscenes  # type: ignore[import]  # noqa: F401

        print("  nuscenes-devkit: installed")
    except ImportError:
        print("  FAIL: nuscenes-devkit not installed (run --setup)")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare nuScenes data")
    parser.add_argument(
        "--nuscenes-dir",
        type=str,
        default=None,
        required=True,
        help="Path to nuScenes data directory",
    )
    parser.add_argument(
        "--setup", action="store_true", help="Extract metadata + install devkit"
    )
    parser.add_argument(
        "--download-blobs",
        action="store_true",
        help="Download and extract camera blob tarballs",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify data completeness"
    )
    args = parser.parse_args()

    nuscenes_dir = Path(args.nuscenes_dir)
    if not nuscenes_dir.exists():
        print(f"ERROR: {nuscenes_dir} does not exist")
        sys.exit(1)

    if not (args.setup or args.download_blobs or args.verify):
        print("No action specified. Use --setup, --download-blobs, or --verify")
        parser.print_help()
        sys.exit(1)

    if args.setup:
        setup(nuscenes_dir)
    if args.download_blobs:
        download_blobs(nuscenes_dir)
    if args.verify:
        verify(nuscenes_dir)


if __name__ == "__main__":
    main()
