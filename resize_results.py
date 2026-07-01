#!/usr/bin/env python3
"""
Resize result images in results_exp44/ from 640x480 to 224x224
using the same resize_with_pad method as OpenPI.

Usage:
    python resize_results.py
    python resize_results.py --input_dir results_exp44 --output_dir results_exp44_224
"""

import argparse
from pathlib import Path
from PIL import Image


def resize_with_pad(image: Image.Image, height: int, width: int) -> Image.Image:
    cur_width, cur_height = image.size
    ratio = max(cur_width / width, cur_height / height)
    resized_width = int(cur_width / ratio)
    resized_height = int(cur_height / ratio)
    resized = image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    canvas = Image.new(image.mode, (width, height), 0)
    pad_w = (width - resized_width) // 2
    pad_h = (height - resized_height) // 2
    canvas.paste(resized, (pad_w, pad_h))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  type=str, default="results_exp44")
    parser.add_argument("--output_dir", type=str, default="results_exp44_224")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width",  type=int, default=224)
    args = parser.parse_args()

    src = Path(args.input_dir)
    dst = Path(args.output_dir)
    dst.mkdir(exist_ok=True)

    img_paths = sorted(src.glob("*.png")) + sorted(src.glob("*.jpg"))
    print(f"Found {len(img_paths)} images in '{src}/'")

    for p in img_paths:
        img = Image.open(p).convert("RGB")
        out = resize_with_pad(img, args.height, args.width)
        out.save(dst / p.name)
        print(f"  {p.name}: {img.size} -> {out.size}")

    print(f"\nDone. Saved to '{dst}/'")


if __name__ == "__main__":
    main()
