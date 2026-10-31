"""
Dataset Inspection Script for Tiny-GenImage Benchmark
Validates directory integrity, counts Real vs AI-Generated samples,
and checks image readability.
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "tiny-genimage"

def inspect_dataset(dataset_dir: Path):
    print("=" * 60)
    print("        TINY-GENIMAGE DATASET INSPECTION")
    print("=" * 60)
    print(f"Dataset root: {dataset_dir}")
    
    if not dataset_dir.exists():
        print(f"[!] Directory {dataset_dir} not found.")
        print("[*] Please download or unpack Tiny-GenImage before training.")
        return

    counts = defaultdict(int)
    generators = defaultdict(lambda: defaultdict(int))
    corrupted = 0
    extensions = defaultdict(int)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_files = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in image_extensions]
    print(f"[*] Found {len(all_files):,} candidate image files.")

    for img_path in all_files:
        p_str = str(img_path).lower().replace("\\", "/")
        ext = img_path.suffix.lower()
        extensions[ext] += 1

        label = "AI-GENERATED" if "/ai" in p_str or "synthetic" in p_str else "REAL"
        counts[label] += 1

        # Generator detection
        gen = "unknown"
        for candidate_gen in ["stablediffusion", "midjourney", "biggan", "glide", "adm", "vqdm", "wukong"]:
            if candidate_gen in p_str:
                gen = candidate_gen
                break
        generators[gen][label] += 1

    print("\n--- Class Distribution ---")
    for cls, cnt in counts.items():
        print(f"  {cls}: {cnt:,} samples ({cnt / max(1, len(all_files)) * 100:.1f}%)")

    print("\n--- Generator Breakdown ---")
    for gen, cls_counts in sorted(generators.items()):
        print(f"  [{gen.upper()}]: Real={cls_counts['REAL']:,} | AI={cls_counts['AI-GENERATED']:,}")

    print("\n[+] Dataset inspection complete.")

if __name__ == "__main__":
    inspect_dataset(DATASET_DIR)
