#!/usr/bin/env python3
"""Generate an HTML prediction report for obstacle left/right ForeAct checkpoints."""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.dataset_finetune import ImagePairDataset
from pipeline import VisualForesightPipeline
from utils.trainer_utils import find_newest_checkpoint


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    checkpoint_parent: Path


@dataclass
class DatasetHandle:
    split: str
    repo_id: str
    root: Path
    dataset: ImagePairDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_name(value: str) -> str:
    keep: list[str] = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "item"


def load_episode_filter(path: Path) -> dict[str, set[int]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected per-repo episode filter dict at {path}")
    return {str(repo_id): {int(x) for x in indices} for repo_id, indices in raw.items()}


def split_counts(total: int, names: list[str]) -> dict[str, int]:
    base = total // len(names)
    extra = total % len(names)
    return {name: base + (1 if i < extra else 0) for i, name in enumerate(names)}


def draw_label(image: Image.Image, label: str) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    pad_x = 8
    pad_y = 6
    bbox = draw.textbbox((0, 0), label, font=font)
    width = bbox[2] - bbox[0] + pad_x * 2
    height = bbox[3] - bbox[1] + pad_y * 2
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
    draw.text((pad_x, pad_y), label, fill=(255, 255, 255), font=font)
    return out


def make_panel(images: list[tuple[str, Image.Image]], caption: str) -> Image.Image:
    labeled = [draw_label(image, label) for label, image in images]
    widths = [image.width for image in labeled]
    heights = [image.height for image in labeled]
    gap = 4
    header_h = 46
    panel_w = sum(widths) + gap * (len(labeled) - 1)
    panel_h = max(heights) + header_h
    panel = Image.new("RGB", (panel_w, panel_h), (246, 248, 251))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 12), caption[:300], fill=(20, 25, 35), font=ImageFont.load_default())
    x = 0
    for image in labeled:
        panel.paste(image, (x, header_h))
        x += image.width + gap
    return panel


def build_dataset_handles(
    split: str,
    split_root: Path,
    filter_path: Path,
    args: argparse.Namespace,
) -> list[DatasetHandle]:
    episode_filter = load_episode_filter(filter_path)
    handles: list[DatasetHandle] = []
    for repo_id in sorted(episode_filter):
        root = split_root / repo_id
        print(
            f"[dataset] loading {split}/{repo_id}: root={root}, "
            f"episodes={len(episode_filter[repo_id])}",
            flush=True,
        )
        dataset = ImagePairDataset(
            root=root,
            camera_key=args.camera_key,
            allowed_episode_indices=episode_filter[repo_id],
            target_frame_offset=args.target_frame_offset,
            source_frame_stride=args.source_frame_stride,
            min_source_frame_index=args.min_source_frame_index,
            trajectory_motion_filter=args.trajectory_motion_filter,
            trajectory_key=args.trajectory_key,
            trajectory_motion_start_threshold=args.trajectory_motion_start_threshold,
            trajectory_motion_start_padding=args.trajectory_motion_start_padding,
            min_trajectory_delta=args.min_trajectory_delta,
        )
        print(f"[dataset] ready {split}/{repo_id}: {len(dataset)} indexed frame pairs", flush=True)
        handles.append(DatasetHandle(split=split, repo_id=repo_id, root=root, dataset=dataset))
    return handles


def sample_from_handles(
    handles: list[DatasetHandle],
    num_samples: int,
    seed: int,
    split: str,
) -> list[tuple[DatasetHandle, int]]:
    pool: list[tuple[int, int]] = []
    for handle_idx, handle in enumerate(handles):
        pool.extend((handle_idx, dataset_index) for dataset_index in range(len(handle.dataset)))
    if not pool:
        raise RuntimeError(f"No frame pairs available for split {split}")

    rng = random.Random(f"{seed}:{split}")
    selected = pool if num_samples >= len(pool) else rng.sample(pool, num_samples)
    return [(handles[handle_idx], dataset_index) for handle_idx, dataset_index in selected]


def index_metadata(dataset: ImagePairDataset, dataset_index: int) -> dict[str, Any]:
    ep_pos, source_frame, target_frame, index_caption = dataset._index[dataset_index]
    ep = dataset._episodes[ep_pos]
    episode_index = int(ep["episode_index"])
    return {
        "episode_index": episode_index,
        "source_frame": int(source_frame),
        "target_frame": int(target_frame),
        "source_time_s": float(source_frame) / float(dataset.fps),
        "target_time_s": float(target_frame) / float(dataset.fps),
        "fps": int(dataset.fps),
        "index_caption": str(index_caption),
    }


def collect_samples(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    sample_dir = output_dir / "assets" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    left_handles = build_dataset_handles("left_first", args.left_root, args.left_filter, args)
    right_handles = build_dataset_handles("right_first", args.right_root, args.right_filter, args)
    handles_by_split = {"left_first": left_handles, "right_first": right_handles}
    counts = split_counts(args.num_samples, ["left_first", "right_first"])

    selected: list[tuple[str, DatasetHandle, int]] = []
    for split, handles in handles_by_split.items():
        for handle, dataset_index in sample_from_handles(handles, counts[split], args.seed, split):
            selected.append((split, handle, dataset_index))

    samples: list[dict[str, Any]] = []
    for rank, (split, handle, dataset_index) in enumerate(selected):
        meta = index_metadata(handle.dataset, dataset_index)
        item = handle.dataset[dataset_index]
        caption = str(item["caption"])
        sample_id = (
            f"{rank:03d}_{sanitize_name(split)}_{sanitize_name(handle.repo_id)}"
            f"_ep{meta['episode_index']:06d}_src{meta['source_frame']:06d}"
        )
        source_path = sample_dir / f"{sample_id}_source.jpg"
        target_path = sample_dir / f"{sample_id}_gt_future.jpg"
        item["source_image"].convert("RGB").save(source_path, quality=92)
        item["target_image"].convert("RGB").save(target_path, quality=92)

        sample = {
            "sample_id": sample_id,
            "split": split,
            "repo_id": handle.repo_id,
            "dataset_root": str(handle.root),
            "dataset_index": int(dataset_index),
            "episode_index": meta["episode_index"],
            "source_frame": meta["source_frame"],
            "target_frame": meta["target_frame"],
            "source_time_s": meta["source_time_s"],
            "target_time_s": meta["target_time_s"],
            "fps": meta["fps"],
            "caption": caption,
            "source": str(source_path.relative_to(output_dir)),
            "gt_future": str(target_path.relative_to(output_dir)),
            "predictions": {},
        }
        samples.append(sample)
        print(
            f"[sample] {rank + 1:03d}/{len(selected):03d} {sample_id} "
            f"{handle.repo_id} idx={dataset_index} frame={meta['source_frame']}->{meta['target_frame']}",
            flush=True,
        )

    return samples


def load_pipeline(checkpoint_parent: Path, device: str) -> tuple[VisualForesightPipeline, str]:
    checkpoint = find_newest_checkpoint(str(checkpoint_parent))
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_parent}")
    print(f"[ckpt] loading {checkpoint}", flush=True)
    pipe = VisualForesightPipeline.from_pretrained(
        str(checkpoint),
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device=device, dtype=torch.bfloat16)
    pipe.eval()
    return pipe, str(checkpoint)


def unload_pipeline(pipe: VisualForesightPipeline) -> None:
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def run_checkpoint(
    spec: CheckpointSpec,
    samples: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    pred_dir = output_dir / "assets" / "predictions" / sanitize_name(spec.label)
    pred_dir.mkdir(parents=True, exist_ok=True)
    pipe, checkpoint = load_pipeline(spec.checkpoint_parent, args.device)

    start = time.time()
    for rank, sample in enumerate(samples):
        pred_path = pred_dir / f"{sample['sample_id']}_pred.jpg"
        if pred_path.exists() and not args.overwrite:
            sample["predictions"][spec.label] = str(pred_path.relative_to(output_dir))
            print(f"[pred] {spec.label} {rank + 1:03d}/{len(samples):03d} cached {sample['sample_id']}", flush=True)
            continue

        source = Image.open(output_dir / sample["source"]).convert("RGB")
        caption = str(sample["caption"])
        generator_device = args.device if args.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(args.seed + rank)

        try:
            with torch.no_grad():
                pred = pipe(
                    image=source,
                    caption=caption,
                    guidance_scale=args.guidance_scale,
                    image_guidance_scale=args.image_guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    num_images_per_prompt=1,
                    generator=generator,
                    enable_progress_bar=False,
                ).images[0].convert("RGB")
        except RuntimeError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        pred.save(pred_path, quality=92)
        sample["predictions"][spec.label] = str(pred_path.relative_to(output_dir))
        print(f"[pred] {spec.label} {rank + 1:03d}/{len(samples):03d} saved {sample['sample_id']}", flush=True)

    elapsed_s = time.time() - start
    unload_pipeline(pipe)
    return {"label": spec.label, "checkpoint": checkpoint, "elapsed_s": elapsed_s}


def write_panels(samples: list[dict[str, Any]], labels: list[str], output_dir: Path, target_frame_offset: int) -> None:
    panel_dir = output_dir / "assets" / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        images = [
            ("source", Image.open(output_dir / sample["source"]).convert("RGB")),
            (f"gt +{target_frame_offset} frames", Image.open(output_dir / sample["gt_future"]).convert("RGB")),
        ]
        for label in labels:
            images.append((label, Image.open(output_dir / sample["predictions"][label]).convert("RGB")))
        caption = (
            f"{sample['split']} | {sample['repo_id']} | ep={sample['episode_index']} | "
            f"frame={sample['source_frame']}->{sample['target_frame']} | {sample['caption']}"
        )
        panel = make_panel(images, caption)
        panel_path = panel_dir / f"{sample['sample_id']}_panel.jpg"
        panel.save(panel_path, quality=92)
        sample["panel"] = str(panel_path.relative_to(output_dir))


def write_html(
    samples: list[dict[str, Any]],
    checkpoint_results: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    checkpoint_items = "\n".join(
        f"<li><strong>{html.escape(result['label'])}</strong>: "
        f"<code>{html.escape(result['checkpoint'])}</code></li>"
        for result in checkpoint_results
    )
    rows: list[str] = []
    labels = [str(result["label"]) for result in checkpoint_results]
    for i, sample in enumerate(samples):
        pred_cells = []
        for label in labels:
            pred_rel = html.escape(sample["predictions"][label])
            pred_cells.append(
                f'<td><img src="{pred_rel}" loading="lazy">'
                f'<div class="img-label">{html.escape(label)}</div></td>'
            )

        rows.append(
            f"""
            <section class="sample">
              <header>
                <div>
                  <h2>{html.escape(sample["split"])} / {html.escape(sample["repo_id"])} <span>#{i + 1}</span></h2>
                  <p class="meta">
                    episode {sample["episode_index"]} · dataset index {sample["dataset_index"]} ·
                    frame {sample["source_frame"]} -> {sample["target_frame"]}
                  </p>
                </div>
                <p class="caption">{html.escape(sample["caption"])}</p>
              </header>
              <table>
                <tr>
                  <td><img src="{html.escape(sample["source"])}" loading="lazy"><div class="img-label">source</div></td>
                  <td><img src="{html.escape(sample["gt_future"])}" loading="lazy"><div class="img-label">gt +{args.target_frame_offset} frames</div></td>
                  {''.join(pred_cells)}
                </tr>
              </table>
              <details>
                <summary>combined panel</summary>
                <img class="panel" src="{html.escape(sample["panel"])}" loading="lazy">
              </details>
            </section>
            """
        )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ForeAct Obstacle Prediction Report</title>
  <style>
    body {{
      margin: 0;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #17202a;
      background: #f4f6f8;
    }}
    main {{
      max-width: 1540px;
      margin: 0 auto;
      padding: 28px 22px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.2;
    }}
    .summary {{
      background: #ffffff;
      border: 1px solid #d8dee6;
      border-radius: 8px;
      padding: 18px 20px;
      margin: 18px 0 22px;
    }}
    .summary p {{
      margin: 4px 0 10px;
      color: #52606d;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      word-break: break-all;
    }}
    ul {{
      margin: 8px 0 0;
      padding-left: 20px;
    }}
    li {{
      margin: 4px 0;
    }}
    .sample {{
      background: #ffffff;
      border: 1px solid #d8dee6;
      border-radius: 8px;
      margin: 18px 0;
      overflow: hidden;
    }}
    .sample header {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      padding: 14px 16px;
      border-bottom: 1px solid #e5e9ef;
      background: #fbfcfd;
    }}
    h2 {{
      margin: 0;
      font-size: 17px;
    }}
    h2 span {{
      color: #697586;
      font-weight: 500;
    }}
    .meta {{
      margin: 4px 0 0;
      color: #697586;
      font-size: 13px;
    }}
    .caption {{
      max-width: 760px;
      margin: 0;
      color: #243447;
      font-weight: 600;
      text-align: right;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    td {{
      vertical-align: top;
      padding: 10px;
      border-right: 1px solid #edf0f3;
    }}
    td:last-child {{
      border-right: 0;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      background: #eef2f6;
      border-radius: 4px;
    }}
    .img-label {{
      text-align: center;
      color: #52606d;
      font-size: 13px;
      margin-top: 6px;
      font-weight: 600;
    }}
    details {{
      padding: 0 16px 14px;
    }}
    summary {{
      cursor: pointer;
      color: #2f6fed;
      font-weight: 600;
      margin: 4px 0 10px;
    }}
    .panel {{
      max-width: 100%;
      width: auto;
      border: 1px solid #d8dee6;
    }}
    @media (max-width: 900px) {{
      .sample header {{
        display: block;
      }}
      .caption {{
        text-align: left;
        margin-top: 10px;
      }}
      table, tr, td {{
        display: block;
        width: 100%;
      }}
      td {{
        border-right: 0;
        border-bottom: 1px solid #edf0f3;
      }}
    }}
  </style>
</head>
<body>
<main>
  <h1>ForeAct Obstacle Prediction Report</h1>
  <div class="summary">
    <p>Generated at {html.escape(generated_at)}. Samples: {len(samples)} total, balanced across left_first/right_first. Seed: {args.seed}.</p>
    <p>Recipe parameters: camera <code>{html.escape(args.camera_key)}</code>, source stride {args.source_frame_stride}, target +{args.target_frame_offset} frames, min trajectory delta {args.min_trajectory_delta}.</p>
    <p>Filters: <code>{html.escape(str(args.left_filter))}</code> and <code>{html.escape(str(args.right_filter))}</code>.</p>
    <h3>Checkpoints</h3>
    <ul>{checkpoint_items}</ul>
  </div>
  {''.join(rows)}
</main>
</body>
</html>
"""
    html_path = output_dir / "index.html"
    html_path.write_text(doc, encoding="utf-8")
    return html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare obstacle left/right ForeAct checkpoints on random frames.")
    parser.add_argument("--ckpt-left", type=Path, default=Path("checkpoints/F-exp28-obstacle-left-first_motion_delta005_s5_t2s"))
    parser.add_argument("--ckpt-right", type=Path, default=Path("checkpoints/F-exp29-obstacle-right-first_motion_delta005_s5_t2s"))
    parser.add_argument("--label-left", default="F-exp28 left-first")
    parser.add_argument("--label-right", default="F-exp29 right-first")
    parser.add_argument("--left-root", type=Path, default=Path("datasets/obstacle_foreact_left_first"))
    parser.add_argument("--right-root", type=Path, default=Path("datasets/obstacle_foreact_right_first"))
    parser.add_argument(
        "--left-filter",
        type=Path,
        default=Path("datasets/obstacle_foreact_split/left_first_filtered_episodes.json"),
    )
    parser.add_argument(
        "--right-filter",
        type=Path,
        default=Path("datasets/obstacle_foreact_split/right_first_filtered_episodes.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/F-exp28_F-exp29_obstacle_prediction_report_50"))
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-key", default="observation.images.cam_high")
    parser.add_argument("--target-frame-offset", type=int, default=60)
    parser.add_argument("--source-frame-stride", type=int, default=5)
    parser.add_argument("--min-source-frame-index", type=int, default=0)
    parser.add_argument("--trajectory-motion-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-key", default="observation.state")
    parser.add_argument("--trajectory-motion-start-threshold", type=float, default=0.05)
    parser.add_argument("--trajectory-motion-start-padding", type=int, default=5)
    parser.add_argument("--min-trajectory-delta", type=float, default=0.05)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_samples(args, args.output_dir)

    specs = [
        CheckpointSpec(label=args.label_left, checkpoint_parent=args.ckpt_left),
        CheckpointSpec(label=args.label_right, checkpoint_parent=args.ckpt_right),
    ]

    checkpoint_results = []
    for spec in specs:
        checkpoint_results.append(run_checkpoint(spec, samples, args.output_dir, args))

    write_panels(samples, [spec.label for spec in specs], args.output_dir, args.target_frame_offset)

    manifest = {
        "args": {
            "num_samples": args.num_samples,
            "seed": args.seed,
            "device": args.device,
            "camera_key": args.camera_key,
            "target_frame_offset": args.target_frame_offset,
            "source_frame_stride": args.source_frame_stride,
            "min_source_frame_index": args.min_source_frame_index,
            "trajectory_motion_filter": args.trajectory_motion_filter,
            "trajectory_key": args.trajectory_key,
            "trajectory_motion_start_threshold": args.trajectory_motion_start_threshold,
            "trajectory_motion_start_padding": args.trajectory_motion_start_padding,
            "min_trajectory_delta": args.min_trajectory_delta,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "image_guidance_scale": args.image_guidance_scale,
            "left_root": str(args.left_root),
            "right_root": str(args.right_root),
            "left_filter": str(args.left_filter),
            "right_filter": str(args.right_filter),
        },
        "checkpoints": checkpoint_results,
        "samples": samples,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = write_html(samples, checkpoint_results, args.output_dir, args)

    print(f"[done] wrote manifest: {manifest_path}", flush=True)
    print(f"[done] wrote HTML report: {html_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
