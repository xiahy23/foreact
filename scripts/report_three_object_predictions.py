#!/usr/bin/env python3
"""Generate an HTML prediction report for a ForeAct three-object checkpoint."""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.dataset_finetune import ImagePairDataset
from models.visualforesight import VisualForesightConfig
from pipeline import VisualForesightPipeline
from utils.trainer_utils import find_newest_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_name(value: str) -> str:
    chars = [c if c.isalnum() or c in ("-", "_", ".") else "_" for c in value]
    return "".join(chars).strip("_") or "item"


def choose_indices(dataset_len: int, num_samples: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if num_samples >= dataset_len:
        return list(range(dataset_len))
    return sorted(rng.sample(range(dataset_len), num_samples))


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


def make_panel(source: Image.Image, target: Image.Image, pred: Image.Image, caption: str) -> Image.Image:
    labeled = [
        draw_label(source, "source"),
        draw_label(target, "gt +30 frames"),
        draw_label(pred, "prediction"),
    ]
    w, h = labeled[0].size
    gap = 4
    header_h = 42
    panel = Image.new("RGB", (w * 3 + gap * 2, h + header_h), (245, 247, 250))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 13), caption[:240], fill=(20, 25, 35), font=ImageFont.load_default())
    x = 0
    for image in labeled:
        panel.paste(image, (x, header_h))
        x += w + gap
    return panel


def rel(path: Path, base: Path) -> str:
    return str(path.relative_to(base))


def collect_samples(args: argparse.Namespace, output_dir: Path) -> tuple[ImagePairDataset, list[dict]]:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImagePairDataset(
        root=args.dataset_root,
        camera_key=args.camera_key,
        target_frame_offset=args.target_frame_offset,
        source_frame_stride=args.source_frame_stride,
        trajectory_motion_filter=args.trajectory_motion_filter,
        trajectory_key="observation.state",
        trajectory_motion_start_threshold=args.trajectory_motion_start_threshold,
        trajectory_motion_start_padding=args.trajectory_motion_start_padding,
        min_trajectory_delta=args.min_trajectory_delta,
    )
    indices = choose_indices(len(dataset), args.num_samples, args.seed)
    samples: list[dict] = []
    for rank, dataset_index in enumerate(indices):
        item = dataset[dataset_index]
        sample_id = f"sample_{rank:02d}_idx_{dataset_index:06d}"
        source_path = sample_dir / f"{sample_id}_source.png"
        target_path = sample_dir / f"{sample_id}_gt_future.png"
        item["source_image"].convert("RGB").save(source_path)
        item["target_image"].convert("RGB").save(target_path)
        samples.append(
            {
                "sample_id": sample_id,
                "dataset_index": dataset_index,
                "caption": str(item["caption"]),
                "source_path": str(source_path),
                "gt_future_path": str(target_path),
                "source": rel(source_path, output_dir),
                "gt_future": rel(target_path, output_dir),
            }
        )
        print(f"[sample] {sample_id}: {item['caption']}", flush=True)
    return dataset, samples


def run_predictions(args: argparse.Namespace, output_dir: Path, samples: list[dict]) -> str:
    checkpoint = find_newest_checkpoint(str(args.checkpoint_path))
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found under {args.checkpoint_path}")
    print(f"[model] loading checkpoint: {checkpoint}", flush=True)
    config = None
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
        input_size = (
            int(yaml_config["target_image_size"][0]) // int(yaml_config["vae_downsample_f"]),
            int(yaml_config["target_image_size"][1]) // int(yaml_config["vae_downsample_f"]),
        )
        config = VisualForesightConfig(
            input_size=input_size,
            mllm_id=yaml_config["mllm_id"],
            diffusion_model_id=yaml_config["diffusion_model_id"],
            vae_id=yaml_config["vae_id"],
            noise_scheduler_id=yaml_config["noise_scheduler_id"],
            scheduler_id=yaml_config["scheduler_id"],
            vae_downsample_f=yaml_config["vae_downsample_f"],
            in_channels=yaml_config["in_channels"],
            system_prompt=yaml_config["system_prompt"],
            _gradient_checkpointing=False,
            modules_to_freeze=tuple(yaml_config.get("modules_to_freeze") or ()),
            modules_to_unfreeze=tuple(yaml_config.get("modules_to_unfreeze") or ()),
        )
    pipe = VisualForesightPipeline.from_pretrained(
        checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device=args.device, dtype=torch.bfloat16)
    pipe.eval()

    pred_dir = output_dir / "predictions"
    panel_dir = output_dir / "panels"
    pred_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    generator_device = args.device if str(args.device).startswith("cuda") else "cpu"
    for rank, sample in enumerate(samples):
        source = Image.open(sample["source_path"]).convert("RGB")
        target = Image.open(sample["gt_future_path"]).convert("RGB")
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(args.seed + rank)

        with torch.no_grad():
            pred = pipe(
                image=source,
                caption=sample["caption"],
                guidance_scale=args.guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=1,
                generator=generator,
                enable_progress_bar=True,
            ).images[0].convert("RGB")

        pred_path = pred_dir / f"{sample['sample_id']}_prediction.png"
        panel_path = panel_dir / f"{sample['sample_id']}_panel.png"
        pred.save(pred_path)
        make_panel(source, target, pred, sample["caption"]).save(panel_path)
        sample["prediction"] = rel(pred_path, output_dir)
        sample["panel"] = rel(panel_path, output_dir)
        print(f"[pred] saved {pred_path}", flush=True)

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return str(checkpoint)


def write_html(args: argparse.Namespace, output_dir: Path, checkpoint: str, dataset_len: int, samples: list[dict]) -> Path:
    rows = []
    for i, sample in enumerate(samples):
        rows.append(
            f"""
            <section class="sample">
              <header>
                <div>
                  <h2>sample #{i + 1}</h2>
                  <p class="meta">dataset index {sample["dataset_index"]}</p>
                </div>
                <p class="caption">{html.escape(sample["caption"])}</p>
              </header>
              <table>
                <tr>
                  <td><img src="{html.escape(sample["source"])}" loading="lazy"><div class="img-label">source</div></td>
                  <td><img src="{html.escape(sample["gt_future"])}" loading="lazy"><div class="img-label">gt +{args.target_frame_offset} frames</div></td>
                  <td><img src="{html.escape(sample["prediction"])}" loading="lazy"><div class="img-label">ForeAct prediction</div></td>
                </tr>
              </table>
              <details>
                <summary>side-by-side panel</summary>
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
  <title>{html.escape(args.report_title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #17202a;
      background: #f4f6f8;
    }}
    main {{
      max-width: 1360px;
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
  <h1>{html.escape(args.report_title)}</h1>
  <div class="summary">
    <p>Generated at {html.escape(generated_at)}. Samples: {len(samples)} / {dataset_len}. Seed: {args.seed}. Inference steps: {args.num_inference_steps}.</p>
    <p>Each row shows the source frame, ground-truth future frame, and ForeAct prediction.</p>
    <p>Checkpoint: <code>{html.escape(checkpoint)}</code></p>
    <p>Dataset: <code>{html.escape(str(args.dataset_root))}</code></p>
    <p>Recipe: camera <code>{html.escape(args.camera_key)}</code>, source stride {args.source_frame_stride}, target +{args.target_frame_offset} frames, min trajectory delta {args.min_trajectory_delta}.</p>
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
    parser = argparse.ArgumentParser(description="Generate ForeAct three-object prediction HTML report.")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/F-exp31-three-object-lerobot-binary_motion_delta005_s5"))
    parser.add_argument("--config-file", type=Path, default=Path("configs/F-exp31-three-object-lerobot-binary_motion_delta005_s5.yaml"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/aloha_fexp_roots/F-exp31-three-object-lerobot-binary_motion_delta005_s5/three_object_lerobot_binary"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/F-exp31_three_object_prediction_report"))
    parser.add_argument("--report-title", default="ForeAct Prediction Report")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-key", default="observation.images.cam_high")
    parser.add_argument("--target-frame-offset", type=int, default=30)
    parser.add_argument("--source-frame-stride", type=int, default=5)
    parser.add_argument("--trajectory-motion-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-motion-start-threshold", type=float, default=0.05)
    parser.add_argument("--trajectory-motion-start-padding", type=int, default=5)
    parser.add_argument("--min-trajectory-delta", type=float, default=0.05)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset, samples = collect_samples(args, args.output_dir)
    checkpoint = run_predictions(args, args.output_dir, samples)
    manifest = {
        "checkpoint": checkpoint,
        "dataset_root": str(args.dataset_root),
        "dataset_len": len(dataset),
        "args": {
            "num_samples": args.num_samples,
            "seed": args.seed,
            "camera_key": args.camera_key,
            "target_frame_offset": args.target_frame_offset,
            "source_frame_stride": args.source_frame_stride,
            "trajectory_motion_filter": args.trajectory_motion_filter,
            "min_trajectory_delta": args.min_trajectory_delta,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "image_guidance_scale": args.image_guidance_scale,
        },
        "samples": samples,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = write_html(args, args.output_dir, checkpoint, len(dataset), samples)
    print(f"[done] wrote manifest: {manifest_path}", flush=True)
    print(f"[done] wrote HTML report: {html_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
