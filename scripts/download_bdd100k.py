#!/usr/bin/env python3
"""
BDD100K Download Manager - Automatically test sources and download from first working one.

Usage:
    python download_bdd100k.py --output-dir ./bdd100k --source auto
    python download_bdd100k.py --output-dir ./bdd100k --source huggingface-bitmind
    python download_bdd100k.py --output-dir ./bdd100k --source huggingface-dgural
    python download_bdd100k.py --output-dir ./bdd100k --source comma2k19
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import Optional, Tuple
import time


def check_network_connectivity() -> bool:
    """Quick check if we have network access."""
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except Exception:
        return False


def test_berkeley_server() -> bool:
    """Test if Berkeley official server is accessible."""
    print("Testing Berkeley official server...")
    try:
        import requests
        response = requests.head(
            "https://bdd-data.berkeley.edu/",
            timeout=5,
            allow_redirects=True
        )
        if response.status_code == 200:
            print("  ✓ Berkeley server is ONLINE")
            return True
        else:
            print(f"  ✗ Berkeley server returned HTTP {response.status_code}")
            return False
    # pyrefly: ignore [unbound-name]
    except requests.Timeout:
        print("  ✗ Berkeley server TIMEOUT (not responding)")
        return False
    except Exception as e:
        print(f"  ✗ Berkeley server unreachable: {e}")
        return False


def download_from_huggingface_bitmind(output_dir: str) -> bool:
    """Download BDD100K from HuggingFace bitmind mirror (FULL DATASET WITH GPS)."""
    print("\nDownloading from HuggingFace (bitmind/bdd100k-real)...")

    try:
        # Check if huggingface-hub is installed
        import importlib.util
        if importlib.util.find_spec("huggingface_hub") is None:
            print("  Installing huggingface-hub...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "huggingface-hub"],
                check=True,
                capture_output=True
            )

        # Import after potential installation
        from huggingface_hub import snapshot_download

        print(f"  Downloading to {output_dir}...")
        print("  This may take several minutes to hours depending on file size and connection...")

        # pyrefly: ignore [no-matching-overload]
        snapshot_download(
            repo_id="bitmind/bdd100k-real",
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        # Verify download
        info_dir = Path(output_dir) / "info"
        if info_dir.exists():
            num_files = len(list(info_dir.glob("*.json")))
            print(f"\n  ✓ Download successful! Found {num_files} info files")
            return True
        else:
            print("\n  ✗ Download failed: info directory not found")
            return False

    except ImportError as e:
        print(f"  ✗ Failed to import: {e}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Installation failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def download_from_huggingface_dgural(output_dir: str) -> bool:
    """Download BDD100K images from HuggingFace dgural (IMAGES ONLY, NO GPS)."""
    print("\nDownloading from HuggingFace (dgural/bdd100k - IMAGES ONLY)...")
    print("  WARNING: This includes only images, not GPS data")

    try:
        from huggingface_hub import snapshot_download

        # pyrefly: ignore [no-matching-overload]
        snapshot_download(
            repo_id="dgural/bdd100k",
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )

        print(f"\n  ✓ Download successful!")
        print("  Note: This is images + detection labels only, no GPS data")
        return True

    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def download_comma2k19_fallback(output_dir: str) -> bool:
    """Download comma2k19 as fallback (highway routes with GPS)."""
    print("\nDownloading comma2k19 (fallback with verified GPS data)...")
    print("  Note: Highway-only dataset (~10 GB)")

    try:
        output_path = Path(output_dir).parent / "comma2k19"
        print(f"  Cloning to {output_path}...")

        subprocess.run(
            ["git", "clone", "https://github.com/commaai/comma2k19.git", str(output_path)],
            check=True,
            timeout=3600
        )

        # List routes
        routes = list((output_path / "data").glob("20*"))
        print(f"\n  ✓ Downloaded {len(routes)} routes with GPS data")
        print(f"  Directory: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"  ✗ Git clone failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def verify_download(output_dir: str) -> bool:
    """Verify that downloaded data includes GPS information."""
    print("\nVerifying downloaded data...")

    output_path = Path(output_dir)

    # Check for info files (BDD100K)
    info_dir = output_path / "info"
    if info_dir.exists():
        json_files = list(info_dir.glob("*.json"))
        print(f"  Found {len(json_files)} BDD100K info files")

        if json_files:
            # Spot check first file
            try:
                with open(json_files[0]) as f:
                    data = json.load(f)
                if "gps" in data:
                    print(f"  ✓ GPS data confirmed in info files")
                    gps = data["gps"]
                    print(f"    Sample: lat={gps.get('latitude', [None])[0]}, "
                          f"lon={gps.get('longitude', [None])[0]}")
                    return True
                else:
                    print(f"  ⚠ GPS data NOT found in first file")
                    return False
            except Exception as e:
                print(f"  ✗ Error reading info file: {e}")
                return False

    # Check for comma2k19 routes
    data_dir = output_path / "data"
    if data_dir.exists():
        routes = list(data_dir.glob("20*"))
        if routes:
            print(f"  Found {len(routes)} comma2k19 routes with GPS data")
            return True

    print(f"  ⚠ Could not verify GPS data in {output_dir}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download BDD100K GPS/info data from working sources"
    )
    parser.add_argument(
        "--output-dir",
        default="./bdd100k",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--source",
        choices=[
            "auto",
            "huggingface-bitmind",
            "huggingface-dgural",
            "berkeley",
            "comma2k19"
        ],
        default="auto",
        help="Download source (auto = try bitmind first)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing download, don't download"
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)

    # Check network
    if not check_network_connectivity():
        print("✗ No network connectivity detected")
        sys.exit(1)

    # If verify only
    if args.verify_only:
        if verify_download(output_dir):
            sys.exit(0)
        else:
            sys.exit(1)

    # Auto mode: try sources in order
    if args.source == "auto":
        print("=== BDD100K Download Manager (Auto Mode) ===\n")

        # Try HuggingFace bitmind first (full dataset with GPS)
        print("1. Trying HuggingFace bitmind (full dataset with GPS)...")
        if download_from_huggingface_bitmind(output_dir):
            if verify_download(output_dir):
                print("\n✓ SUCCESS: BDD100K with GPS data downloaded")
                sys.exit(0)

        # Try HuggingFace dgural (images only)
        print("\n2. Trying HuggingFace dgural (images only)...")
        if download_from_huggingface_dgural(output_dir):
            print("\n⚠ Partial success: Downloaded images but no GPS data")
            print("   For GPS data, try --source comma2k19 for fallback")
            sys.exit(0)

        # Try Berkeley
        print("\n3. Checking Berkeley official server...")
        if test_berkeley_server():
            print("   Berkeley is online! Visit https://bdd-data.berkeley.edu/")
            sys.exit(0)

        # Fallback: comma2k19
        print("\n4. Using comma2k19 fallback (highway routes with GPS)...")
        if download_comma2k19_fallback(output_dir):
            print("\n✓ Fallback successful: comma2k19 downloaded with GPS data")
            sys.exit(0)

        print("\n✗ All download sources failed. Check your network connection.")
        sys.exit(1)

    # Specific source
    elif args.source == "huggingface-bitmind":
        if download_from_huggingface_bitmind(output_dir):
            verify_download(output_dir)
            sys.exit(0)
        sys.exit(1)

    elif args.source == "huggingface-dgural":
        if download_from_huggingface_dgural(output_dir):
            verify_download(output_dir)
            sys.exit(0)
        sys.exit(1)

    elif args.source == "berkeley":
        if test_berkeley_server():
            print("Berkeley server is online. Visit https://bdd-data.berkeley.edu/")
            sys.exit(0)
        else:
            print("Berkeley server is currently unavailable")
            sys.exit(1)

    elif args.source == "comma2k19":
        if download_comma2k19_fallback(output_dir):
            verify_download(output_dir)
            sys.exit(0)
        sys.exit(1)


if __name__ == "__main__":
    main()
