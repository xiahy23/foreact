#!/usr/bin/env python3
"""Generate an HTML comparison report for two ForeAct checkpoints."""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
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


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint_parent: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_name(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "item"


def choose_indices(dataset_len: int, num_samples: int, seed: int, dataset_name: str) -> list[int]:
    rng = random.Random(f"{seed}:{dataset_name}")
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


def make_panel(images: list[tuple[str, Image.Image]], caption: str) -> Image.Image:
    labeled = [draw_label(image, label) for label, image in images]
    widths = [image.width for image in labeled]
    heights = [image.height for image in labeled]
    gap = 4
    header_h = 44
    panel_w = sum(widths) + gap * (len(labeled) - 1)
    panel_h = max(heights) + header_h
    panel = Image.new("RGB", (panel_w, panel_h), (245, 247, 250))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 12), caption[:260], fill=(20, 25, 35), font=ImageFont.load_default())
    x = 0
    for image in labeled:
        panel.paste(image, (x, header_h))
        x += image.width + gap
    return panel


def load_pipeline(checkpoint_parent: str, device: str) -> tuple[VisualForesightPipeline, str]:
    checkpoint = find_newest_checkpoint(checkpoint_parent)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_parent}")
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    pipe = VisualForesightPipeline.from_pretrained(
        checkpoint,
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
        torch.cuda.ipc_collect()


def run_model_predictions(
    spec: ModelSpec,
    samples: list[dict],
    output_dir: Path,
    device: str,
    seed: int,
    guidance_scale: float,
    image_guidance_scale: float,
    num_inference_steps: int,
) -> str:
    pipe, checkpoint = load_pipeline(spec.checkpoint_parent, device)
    model_dir = output_dir / sanitize_name(spec.label)
    model_dir.mkdir(parents=True, exist_ok=True)

    for rank, sample in enumerate(samples):
        source = Image.open(sample["source_path"]).convert("RGB")
        caption = sample["caption"]
        generator_device = device if device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(seed + rank)

        with torch.no_grad():
            pred = pipe(
                image=source,
                caption=caption,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                generator=generator,
                enable_progress_bar=True,
            ).images[0].convert("RGB")

        pred_name = f"{sample['sample_id']}_pred.png"
        pred_path = model_dir / pred_name
        pred.save(pred_path)
        sample["predictions"][spec.label] = str(pred_path.relative_to(output_dir))
        print(f"[{spec.label}] saved {pred_path}", flush=True)

    unload_pipeline(pipe)
    return checkpoint


def collect_samples(
    dataset_roots: list[Path],
    output_dir: Path,
    samples_per_dataset: int,
    seed: int,
    camera_key: str,
    target_frame_offset: int,
    source_frame_stride: int,
    min_trajectory_delta: float,
    trajectory_motion_filter: bool,
) -> list[dict]:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []

    for root in dataset_roots:
        dataset_name = root.name
        print(f"Loading dataset {dataset_name}: {root}", flush=True)
        dataset = ImagePairDataset(
            root=root,
            camera_key=camera_key,
            target_frame_offset=target_frame_offset,
            source_frame_stride=source_frame_stride,
            trajectory_motion_filter=trajectory_motion_filter,
            trajectory_key="observation.state",
            trajectory_motion_start_threshold=0.05,
            trajectory_motion_start_padding=5,
            min_trajectory_delta=min_trajectory_delta,
        )
        indices = choose_indices(len(dataset), samples_per_dataset, seed, dataset_name)
        for local_rank, dataset_index in enumerate(indices):
            item = dataset[dataset_index]
            caption = str(item["caption"])
            sample_id = f"{sanitize_name(dataset_name)}_{local_rank:02d}_idx_{dataset_index:06d}"
            source_path = sample_dir / f"{sample_id}_source.png"
            target_path = sample_dir / f"{sample_id}_gt_future.png"
            item["source_image"].convert("RGB").save(source_path)
            item["target_image"].convert("RGB").save(target_path)
            samples.append(
                {
                    "sample_id": sample_id,
                    "dataset_name": dataset_name,
                    "dataset_root": str(root),
                    "dataset_index": dataset_index,
                    "caption": caption,
                    "source_path": str(source_path),
                    "gt_future_path": str(target_path),
                    "source": str(source_path.relative_to(output_dir)),
                    "gt_future": str(target_path.relative_to(output_dir)),
                    "predictions": {},
                }
            )
            print(f"Selected {sample_id}: {caption}", flush=True)

    return samples


def write_panels(samples: list[dict], model_labels: list[str], output_dir: Path) -> None:
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        images = [
            ("source", Image.open(output_dir / sample["source"]).convert("RGB")),
            ("gt +30 frames", Image.open(output_dir / sample["gt_future"]).convert("RGB")),
        ]
        for label in model_labels:
            pred_rel = sample["predictions"][label]
            images.append((label, Image.open(output_dir / pred_rel).convert("RGB")))
        panel = make_panel(
            images,
            f"{sample['dataset_name']} | idx={sample['dataset_index']} | {sample['caption']}",
        )
        panel_path = panel_dir / f"{sample['sample_id']}_panel.png"
        panel.save(panel_path)
        sample["panel"] = str(panel_path.relative_to(output_dir))


def write_html(
    samples: list[dict],
    model_specs: list[ModelSpec],
    checkpoints: dict[str, str],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    rows = []
    for i, sample in enumerate(samples):
        pred_cells = []
        for spec in model_specs:
            rel = html.escape(sample["predictions"][spec.label])
            pred_cells.append(
                f'<td><img src="{rel}" loading="lazy"><div class="img-label">{html.escape(spec.label)}</div></td>'
            )

        rows.append(
            f"""
            <section class="sample">
              <header>
                <div>
                  <h2>{html.escape(sample["dataset_name"])} <span>#{i + 1}</span></h2>
                  <p class="meta">dataset index {sample["dataset_index"]}</p>
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
                <summary>panel</summary>
                <img class="panel" src="{html.escape(sample["panel"])}" loading="lazy">
              </details>
            </section>
            """
        )

    ckpt_items = "\n".join(
        f"<li><strong>{html.escape(label)}</strong>: <code>{html.escape(path)}</code></li>"
        for label, path in checkpoints.items()
    )
    dataset_items = "\n".join(
        f"<li><code>{html.escape(str(root))}</code></li>" for root in args.dataset_roots
    )
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ForeAct Two-Checkpoint Prediction Report</title>
  <style>
    body {{
      margin: 0;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #17202a;
      background: #f4f6f8;
    }}
    main {{
      max-width: 1520px;
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
  <h1>ForeAct Two-Checkpoint Prediction Report</h1>
  <div class="summary">
    <p>Generated at {html.escape(generated_at)}. Samples: {len(samples)}. Seed: {args.seed}. Inference steps: {args.num_inference_steps}.</p>
    <p>Each row shows source, ground-truth future frame, and predictions from both latest checkpoints.</p>
    <h3>Checkpoints</h3>
    <ul>{ckpt_items}</ul>
    <h3>Datasets</h3>
    <ul>{dataset_items}</ul>
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
    parser = argparse.ArgumentParser(description="Compare two ForeAct checkpoints on sampled LeRobot frames.")
    parser.add_argument(
        "--ckpt-a",
        default="checkpoints/F-exp23-eggplant-potato-gripper-binary_motion_delta005_s5",
    )
    parser.add_argument("--label-a", default="F-exp23 eggplant/potato")
    parser.add_argument(
        "--ckpt-b",
        default="checkpoints/F-exp24-aloha+eggplant/potato-mixed_motion_delta005_s5",
        help="Second checkpoint parent. Defaults to F-exp24 if the shorthand path does not exist.",
    )
    parser.add_argument("--label-b", default="F-exp24 mixed")
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        type=Path,
        default=[
            Path("datasets/aloha_fexp_roots/F-exp24-aloha-banana-fix-cube-eggplant-potato_motion_delta005_s5/aloha_banana"),
            Path("datasets/aloha_fexp_roots/F-exp24-aloha-banana-fix-cube-eggplant-potato_motion_delta005_s5/aloha_banana_fix"),
            Path("datasets/aloha_fexp_roots/F-exp24-aloha-banana-fix-cube-eggplant-potato_motion_delta005_s5/aloha_cube"),
            Path("datasets/aloha_fexp_roots/F-exp24-aloha-banana-fix-cube-eggplant-potato_motion_delta005_s5/eggplant_potato_gripper_binary"),
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/F-exp23_F-exp24_prediction_report"))
    parser.add_argument("--samples-per-dataset", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-key", default="observation.images.cam_high")
    parser.add_argument("--target-frame-offset", type=int, default=30)
    parser.add_argument("--source-frame-stride", type=int, default=5)
    parser.add_argument("--trajectory-motion-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-trajectory-delta", type=float, default=0.05)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    args = parser.parse_args()

    if args.ckpt_b == "checkpoints/F-exp24-aloha+eggplant/potato-mixed_motion_delta005_s5":
        args.ckpt_b = "checkpoints/F-exp24-aloha-banana-fix-cube-eggplant-potato_motion_delta005_s5"
    return args


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = [
        ModelSpec(args.label_a, args.ckpt_a),
        ModelSpec(args.label_b, args.ckpt_b),
    ]

    samples = collect_samples(
        dataset_roots=args.dataset_roots,
        output_dir=args.output_dir,
        samples_per_dataset=args.samples_per_dataset,
        seed=args.seed,
        camera_key=args.camera_key,
        target_frame_offset=args.target_frame_offset,
        source_frame_stride=args.source_frame_stride,
        min_trajectory_delta=args.min_trajectory_delta,
        trajectory_motion_filter=args.trajectory_motion_filter,
    )

    checkpoints: dict[str, str] = {}
    for spec in model_specs:
        checkpoints[spec.label] = run_model_predictions(
            spec=spec,
            samples=samples,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            num_inference_steps=args.num_inference_steps,
        )

    write_panels(samples, [spec.label for spec in model_specs], args.output_dir)

    manifest = {
        "args": {
            "samples_per_dataset": args.samples_per_dataset,
            "seed": args.seed,
            "camera_key": args.camera_key,
            "target_frame_offset": args.target_frame_offset,
            "source_frame_stride": args.source_frame_stride,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "image_guidance_scale": args.image_guidance_scale,
        },
        "checkpoints": checkpoints,
        "samples": samples,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = write_html(samples, model_specs, checkpoints, args.output_dir, args)
    print(f"Wrote manifest: {manifest_path}", flush=True)
    print(f"Wrote HTML report: {html_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
