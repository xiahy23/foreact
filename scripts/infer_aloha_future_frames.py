#!/usr/bin/env python3
"""Sample Aloha future-frame predictions from a trained ForeAct checkpoint.

For each sampled item this script saves:
  - current source image
  - ground-truth image at source + target_frame_offset
  - ForeAct prediction
  - a side-by-side panel for quick visual inspection
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.dataset_finetune import ImagePairDataset
from pipeline import VisualForesightPipeline
from utils.trainer_utils import find_newest_checkpoint


def parse_indices(raw: str) -> list[int]:
    if not raw:
        return []
    indices: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            chunks = [int(x) for x in part.split(":")]
            if len(chunks) == 2:
                start, stop = chunks
                step = 1
            elif len(chunks) == 3:
                start, stop, step = chunks
            else:
                raise ValueError(f"Bad index range: {part}")
            indices.extend(list(range(start, stop, step)))
        else:
            indices.append(int(part))
    return indices


def choose_indices(dataset_len: int, num_samples: int, seed: int, explicit: list[int]) -> list[int]:
    if explicit:
        return [idx for idx in explicit if 0 <= idx < dataset_len]
    rng = random.Random(seed)
    if num_samples >= dataset_len:
        return list(range(dataset_len))
    return sorted(rng.sample(range(dataset_len), num_samples))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def draw_label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    pad = 8
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0] + pad * 2
    height = bbox[3] - bbox[1] + pad * 2
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return out


def make_panel(source: Image.Image, target: Image.Image, pred: Image.Image, caption: str) -> Image.Image:
    source = draw_label(source, "source")
    target = draw_label(target, "gt +30 frames")
    pred = draw_label(pred, "prediction")

    w, h = source.size
    header_h = 34
    panel = Image.new("RGB", (w * 3, h + header_h), (255, 255, 255))
    panel.paste(source, (0, header_h))
    panel.paste(target, (w, header_h))
    panel.paste(pred, (w * 2, header_h))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((8, 10), caption[:220], fill=(0, 0, 0), font=font)
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ForeAct inference on Aloha future-frame samples.")
    parser.add_argument("--checkpoint-path", required=True, help="Checkpoint dir or parent dir.")
    parser.add_argument("--dataset-root", required=True, help="LeRobot-style Aloha dataset root.")
    parser.add_argument("--camera-key", default="observation.images.cam_high")
    parser.add_argument("--output-dir", default="outputs/aloha_infer")
    parser.add_argument("--target-frame-offset", type=int, default=30)
    parser.add_argument("--source-frame-stride", type=int, default=30)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--indices", default="", help="Comma-separated indices or ranges, e.g. 0,10,20:30:2.")
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--num-images-per-prompt", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImagePairDataset(
        root=args.dataset_root,
        camera_key=args.camera_key,
        target_frame_offset=args.target_frame_offset,
        source_frame_stride=args.source_frame_stride,
    )
    indices = choose_indices(len(dataset), args.num_samples, args.seed, parse_indices(args.indices))
    if not indices:
        raise RuntimeError("No valid sample indices selected.")

    checkpoint = find_newest_checkpoint(args.checkpoint_path)
    print(f"Loading checkpoint: {checkpoint}")
    pipe = VisualForesightPipeline.from_pretrained(
        checkpoint,
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device=args.device, dtype=torch.bfloat16)
    pipe.eval()

    manifest = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(args.dataset_root),
        "camera_key": args.camera_key,
        "target_frame_offset": args.target_frame_offset,
        "source_frame_stride": args.source_frame_stride,
        "seed": args.seed,
        "samples": [],
    }

    for rank, idx in enumerate(indices):
        sample = dataset[idx]
        source = sample["source_image"].convert("RGB")
        target = sample["target_image"].convert("RGB")
        caption = str(sample["caption"])
        generator = torch.Generator(device=args.device if args.device.startswith("cuda") else "cpu")
        generator.manual_seed(args.seed + rank)

        with torch.no_grad():
            pred = pipe(
                image=source,
                caption=caption,
                guidance_scale=args.guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_images_per_prompt,
                generator=generator,
                enable_progress_bar=True,
            ).images[0].convert("RGB")

        stem = f"sample_{rank:02d}_idx_{idx:06d}"
        source_path = output_dir / f"{stem}_source.png"
        target_path = output_dir / f"{stem}_gt_future.png"
        pred_path = output_dir / f"{stem}_pred.png"
        panel_path = output_dir / f"{stem}_panel.png"
        source.save(source_path)
        target.save(target_path)
        pred.save(pred_path)
        make_panel(source, target, pred, caption).save(panel_path)

        manifest["samples"].append(
            {
                "dataset_index": idx,
                "caption": caption,
                "source": str(source_path),
                "gt_future": str(target_path),
                "prediction": str(pred_path),
                "panel": str(panel_path),
            }
        )
        print(f"Saved {panel_path}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
