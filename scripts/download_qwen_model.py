#!/usr/bin/env python3
"""Pre-download Qwen3-VL-7B-Instruct for local use.

Uses huggingface_hub.snapshot_download for reliability — ensures all files,
configs, and tokenizers are pulled. Compute nodes cannot access HuggingFace
(proxy 403), so models must be pre-downloaded.

Usage:
    python scripts/download_qwen_model.py
    python scripts/download_qwen_model.py --output-dir /path/to/models
    python scripts/download_qwen_model.py --model-id Qwen/Qwen3-VL-7B-Instruct
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download Qwen3-VL model for cluster use"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-7B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        required=True,
        help="Parent directory for model download",
    )
    args = parser.parse_args()

    model_name = args.model_id.split("/")[-1]
    output_path = Path(args.output_dir) / model_name

    print(f"Downloading {args.model_id} -> {output_path}")
    print("  Using huggingface_hub.snapshot_download (all files + configs)")

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(output_path),
    )

    print(f"\n=== Done ===")
    print(f"  Model downloaded to: {output_path}")
    print(f"\n  Use with eval script:")
    print(f"    --qwen-model {output_path}")
    print(f"\n  Verify with:")
    print(
        f'    python -c "from transformers import AutoProcessor; '
        f"AutoProcessor.from_pretrained('{output_path}', local_files_only=True); "
        f"print('OK')\""
    )


if __name__ == "__main__":
    main()
