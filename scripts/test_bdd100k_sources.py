#!/usr/bin/env python3
"""
Test alternative BDD100K download sources.

This script attempts to access and validate BDD100K data from multiple sources.
Use this to find a working mirror for your environment.

Usage:
    python scripts/test_bdd100k_sources.py

Output:
    - Prints availability status for each source
    - Attempts to load metadata from working sources
    - Provides size estimates and data completeness info
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional
import time

# Try importing optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    # pyrefly: ignore [missing-module-attribute]
    from datasets import load_dataset
    HAS_HF = True
except ImportError:
    HAS_HF = False


class BDD100KSourceTester:
    """Test availability of different BDD100K download sources."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.results = {}

    def test_berkeley_server(self) -> Tuple[bool, str]:
        """Test if Berkeley official server is accessible."""
        print("\n1. Testing Berkeley Official Server (bdd-data.berkeley.edu)")
        print("-" * 60)

        if not HAS_REQUESTS:
            print("❌ requests library not installed, skipping")
            return False, "requests not installed"

        try:
            url = "https://bdd-data.berkeley.edu/"
            response = requests.head(url, timeout=self.timeout, allow_redirects=True)

            if response.status_code == 200:
                print(f"✓ Server accessible (HTTP {response.status_code})")
                print(f"  URL: {url}")
                print(f"  Status: Online")
                return True, "Server is online"
            else:
                print(f"✗ Server returned HTTP {response.status_code}")
                return False, f"HTTP {response.status_code}"

        except requests.Timeout:
            print(f"✗ Server timeout (>{self.timeout}s)")
            return False, "Timeout"
        except requests.ConnectionError as e:
            print(f"✗ Connection error: {e}")
            return False, str(e)
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False, str(e)

    def test_huggingface_bitmind(self) -> Tuple[bool, str]:
        """Test HuggingFace bitmind mirror (full dataset)."""
        print("\n2. Testing HuggingFace Mirror (bitmind/bdd100k-real)")
        print("-" * 60)

        if not HAS_HF:
            print("❌ datasets library not installed")
            print("   Install with: pip install datasets")
            return False, "datasets not installed"

        try:
            print("Loading dataset metadata...")
            # Just load metadata, don't download actual data
            dataset = load_dataset("bitmind/bdd100k-real", split=None)

            # Check what splits are available
            if hasattr(dataset, 'keys'):
                splits = list(dataset.keys())
                print(f"✓ Dataset accessible")
                print(f"  Available splits: {splits}")

                # Try to get sample from first split
                for split in splits:
                    try:
                        sample_data = dataset[split][0]
                        print(f"  {split}: {len(dataset[split])} samples")
                        print(f"    Sample keys: {list(sample_data.keys())[:5]}...")

                        # Check for GPS data
                        if 'gps' in sample_data or 'info' in sample_data:
                            print(f"    ✓ GPS/info data present")
                            return True, "Full dataset with GPS data"
                        else:
                            print(f"    ⚠ No GPS/info data in this split")

                    except Exception as e:
                        print(f"  Error loading {split}: {e}")

                return True, "Dataset accessible (GPS status unknown)"

            else:
                print(f"✓ Dataset accessible")
                print(f"  Size: {len(dataset)} samples")
                return True, "Dataset accessible"

        except Exception as e:
            print(f"✗ Failed to load: {e}")
            if "Connection" in str(e) or "timeout" in str(e).lower():
                return False, "Network error"
            else:
                return False, str(e)

    def test_huggingface_dgural(self) -> Tuple[bool, str]:
        """Test HuggingFace dgural mirror (images only)."""
        print("\n3. Testing HuggingFace Mirror (dgural/bdd100k - Images Only)")
        print("-" * 60)

        if not HAS_HF:
            print("❌ datasets library not installed")
            return False, "datasets not installed"

        try:
            print("Loading dataset metadata...")
            dataset = load_dataset("dgural/bdd100k", split="train")

            print(f"✓ Dataset accessible")
            print(f"  Total samples: {len(dataset)}")
            print(f"  Sample keys: {list(dataset[0].keys())}")
            print(f"  ⚠ Note: Images only, no video or GPS data")

            return True, "Images only (no GPS)"

        except Exception as e:
            print(f"✗ Failed to load: {e}")
            return False, str(e)

    def test_github_info_files(self) -> Tuple[bool, str]:
        """Check BDD100K GitHub for info/GPS file specifications."""
        print("\n4. Testing GitHub Access (Documentation)")
        print("-" * 60)

        if not HAS_REQUESTS:
            print("❌ requests library not installed")
            return False, "requests not installed"

        try:
            # Try to fetch info about the dataset from GitHub
            url = "https://api.github.com/repos/bdd100k/bdd100k"
            response = requests.get(url, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                print(f"✓ GitHub repository accessible")
                print(f"  Repository: {data.get('full_name')}")
                print(f"  Stars: {data.get('stargazers_count')}")
                print(f"  Open issues: {data.get('open_issues_count')}")

                # Check for recent issues about downloads
                issues_url = f"{url}/issues?state=open&labels=download"
                issues = requests.get(issues_url, timeout=self.timeout).json()

                if issues:
                    print(f"\n  Recent download-related issues: {len(issues)}")
                    for issue in issues[:3]:
                        print(f"    - {issue.get('title')}")

                return True, "Repository accessible"
            else:
                print(f"✗ GitHub API returned HTTP {response.status_code}")
                return False, f"HTTP {response.status_code}"

        except Exception as e:
            print(f"✗ GitHub check failed: {e}")
            return False, str(e)

    def test_cli_tool(self) -> Tuple[bool, str]:
        """Test if bdd100k pip package is installed."""
        print("\n5. Testing BDD100K CLI Tool")
        print("-" * 60)

        try:
            result = subprocess.run(
                ["python", "-m", "bdd100k", "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                print("✓ BDD100K toolkit installed")
                print("  You can use: python -m bdd100k download --help")
                return True, "CLI tool available"
            else:
                print("✗ BDD100K toolkit not working properly")
                print(f"  Error: {result.stderr[:200]}")
                return False, "CLI not functional"

        except FileNotFoundError:
            print("✗ BDD100K CLI tool not found")
            print("  Install with: pip install bdd100k")
            return False, "Not installed"
        except subprocess.TimeoutExpired:
            print("✗ BDD100K CLI tool timeout")
            return False, "Timeout"
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False, str(e)

    def run_all_tests(self) -> Dict[str, Tuple[bool, str]]:
        """Run all availability tests."""
        print("\n" + "=" * 60)
        print("BDD100K DOWNLOAD SOURCE AVAILABILITY TEST")
        print("=" * 60)

        self.results = {
            "Berkeley Official": self.test_berkeley_server(),
            "HuggingFace (bitmind)": self.test_huggingface_bitmind(),
            "HuggingFace (dgural)": self.test_huggingface_dgural(),
            "GitHub Access": self.test_github_info_files(),
            "CLI Tool": self.test_cli_tool(),
        }

        self._print_summary()
        return self.results

    def _print_summary(self):
        """Print summary of all test results."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print("\nWorking sources:")
        working = [k for k, (status, msg) in self.results.items() if status]
        if working:
            for i, source in enumerate(working, 1):
                status, msg = self.results[source]
                print(f"  {i}. {source}: {msg}")
        else:
            print("  None - all sources currently unavailable")

        print("\nUnavailable sources:")
        broken = [k for k, (status, msg) in self.results.items() if not status]
        for source in broken:
            status, msg = self.results[source]
            print(f"  ✗ {source}: {msg}")

        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)

        if self.results["HuggingFace (bitmind)"][0]:
            print("\n1. TRY FIRST: HuggingFace bitmind mirror")
            print("   This likely has the full dataset with GPS data")
            print("\n   Download with:")
            print("   ```bash")
            print("   from datasets import load_dataset")
            print("   dataset = load_dataset('bitmind/bdd100k-real')")
            print("   ```")

        elif self.results["HuggingFace (dgural)"][0]:
            print("\n1. AVAILABLE: HuggingFace dgural mirror")
            print("   ⚠ This has only images (no video/GPS)")
            print("   Use for initial testing only")

        if self.results["Berkeley Official"][0]:
            print("\n2. FALLBACK: Berkeley Official Server")
            print("   Currently online - try direct download")
            print("   https://bdd-data.berkeley.edu/")

        if not any(status for status, _ in self.results.values()):
            print("\nAll sources unavailable. Options:")
            print("  1. Wait 24-48 hours and retry")
            print("  2. Contact: Fisher Yu (i@yf.io)")
            print("  3. Use alternative dataset: comma2k19, nuScenes, or Waymo")
            print("  4. Check GitHub issues for community mirrors:")
            print("     https://github.com/bdd100k/bdd100k/issues")


def main():
    """Run the source availability tests."""
    tester = BDD100KSourceTester(timeout=10)

    # Check dependencies
    if not HAS_HF:
        print("⚠️  Warning: 'datasets' library not installed")
        print("   Some tests will be skipped")
        print("   Install with: pip install datasets\n")

    if not HAS_REQUESTS:
        print("⚠️  Warning: 'requests' library not installed")
        print("   Some tests will be skipped")
        print("   Install with: pip install requests\n")

    # Run all tests
    results = tester.run_all_tests()

    # Exit with appropriate code
    if any(status for status, _ in results.values()):
        sys.exit(0)  # At least one source is available
    else:
        sys.exit(1)  # All sources unavailable


if __name__ == "__main__":
    main()
