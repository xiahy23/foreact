#!/usr/bin/env python3
"""Run all banana-trained ForeAct checkpoints on aloha_banana_fix and write an HTML report."""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.dataset_finetune import ImagePairDataset
from pipeline import VisualForesightPipeline
from utils.trainer_utils import find_newest_checkpoint


@dataclass
class CheckpointInfo:
    name: str
    config_path: Path
    checkpoint_path: Path
    group: str
    learning_rate: Any
    source_frame_stride: Any
    target_frame_offset: Any
    trajectory_motion_filter: Any
    min_trajectory_delta: Any
    num_train_epochs: Any


@dataclass
class SampleInfo:
    sample_id: str
    dataset_index: int
    episode_index: int
    source_frame: int
    target_frame: int
    caption: str
    source_path: Path
    target_path: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numeric_checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def newest_checkpoint_dir(run_dir: Path) -> Path | None:
    if not run_dir.is_dir():
        return None
    checkpoint = find_newest_checkpoint(str(run_dir))
    if checkpoint:
        return Path(checkpoint)
    candidates = sorted(run_dir.glob("checkpoint-*"), key=numeric_checkpoint_step)
    return candidates[-1] if candidates else None


def infer_group(config_path: Path, cfg: dict[str, Any]) -> str:
    name = config_path.stem.lower()
    data_path = str(cfg.get("data_path", "")).lower()
    run_name = str(cfg.get("run_name", "")).lower()
    joined = " ".join([name, data_path, run_name])
    if "aloha-banana" in joined or "aloha_banana" in joined:
        return "banana"
    if "aloha-mixed" in joined or "mixed" in joined:
        return "mixed"
    return "other"


def discover_checkpoints(config_dir: Path, checkpoints_dir: Path, include_mixed: bool) -> tuple[list[CheckpointInfo], list[dict[str, str]]]:
    found: list[CheckpointInfo] = []
    skipped: list[dict[str, str]] = []
    accepted_groups = {"banana", "mixed"} if include_mixed else {"banana"}

    for config_path in sorted(config_dir.glob("F-exp*-aloha*.yaml")):
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        group = infer_group(config_path, cfg)
        if group not in accepted_groups:
            skipped.append({"config": str(config_path), "reason": f"group={group}"})
            continue

        run_name = str(cfg.get("run_name") or config_path.stem)
        checkpoint = newest_checkpoint_dir(checkpoints_dir / run_name)
        if checkpoint is None:
            skipped.append({"config": str(config_path), "run_name": run_name, "reason": "no checkpoint"})
            continue

        found.append(
            CheckpointInfo(
                name=run_name,
                config_path=config_path,
                checkpoint_path=checkpoint,
                group=group,
                learning_rate=cfg.get("learning_rate"),
                source_frame_stride=cfg.get("source_frame_stride"),
                target_frame_offset=cfg.get("target_frame_offset"),
                trajectory_motion_filter=cfg.get("trajectory_motion_filter", False),
                min_trajectory_delta=cfg.get("min_trajectory_delta", 0.0),
                num_train_epochs=cfg.get("num_train_epochs"),
            )
        )

    found.sort(key=lambda c: (0 if c.group == "banana" else 1, c.name))
    return found, skipped


def choose_evenly(items: list[int], count: int) -> list[int]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return items
    if count == 1:
        return [items[len(items) // 2]]
    # Use interior quantiles so the default 2 samples avoid the usually static first frame.
    positions = [(i + 1) * (len(items) - 1) / (count + 1) for i in range(count)]
    chosen = sorted({items[int(round(pos))] for pos in positions})
    cursor = 0
    while len(chosen) < count and cursor < len(items):
        if items[cursor] not in chosen:
            chosen.append(items[cursor])
        cursor += 1
    return sorted(chosen[:count])


def select_samples(dataset: ImagePairDataset, samples_per_episode: int) -> list[int]:
    by_episode_pos: dict[int, list[int]] = {}
    for dataset_index, item in enumerate(dataset._index):
        episode_pos = int(item[0])
        by_episode_pos.setdefault(episode_pos, []).append(dataset_index)

    selected: list[int] = []
    for episode_pos in range(len(dataset._episodes)):
        candidates = by_episode_pos.get(episode_pos, [])
        selected.extend(choose_evenly(candidates, samples_per_episode))
    return selected


def save_reference_images(dataset: ImagePairDataset, indices: list[int], assets_dir: Path) -> list[SampleInfo]:
    refs_dir = assets_dir / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    samples: list[SampleInfo] = []

    for rank, dataset_index in enumerate(indices):
        ep_pos, source_frame, target_frame, caption = dataset._index[dataset_index]
        ep = dataset._episodes[ep_pos]
        episode_index = int(ep["episode_index"])
        sample_id = f"ep{episode_index:06d}_src{source_frame:05d}_tgt{target_frame:05d}"

        sample = dataset[dataset_index]
        source = sample["source_image"].convert("RGB")
        target = sample["target_image"].convert("RGB")
        caption = str(sample.get("caption") or caption or "")

        source_path = refs_dir / f"{sample_id}_source.jpg"
        target_path = refs_dir / f"{sample_id}_gt.jpg"
        if not source_path.exists():
            source.save(source_path, quality=92)
        if not target_path.exists():
            target.save(target_path, quality=92)

        samples.append(
            SampleInfo(
                sample_id=sample_id,
                dataset_index=dataset_index,
                episode_index=episode_index,
                source_frame=int(source_frame),
                target_frame=int(target_frame),
                caption=caption,
                source_path=source_path,
                target_path=target_path,
            )
        )
        print(f"[refs] {rank + 1:04d}/{len(indices):04d} {sample_id}")

    return samples


def rel(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def run_checkpoint(
    checkpoint: CheckpointInfo,
    dataset: ImagePairDataset,
    samples: list[SampleInfo],
    assets_dir: Path,
    device: str,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
    overwrite: bool,
) -> dict[str, Any]:
    pred_dir = assets_dir / "predictions" / checkpoint.name
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ckpt] loading {checkpoint.name}: {checkpoint.checkpoint_path}")
    pipe = VisualForesightPipeline.from_pretrained(
        str(checkpoint.checkpoint_path),
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device=device, dtype=torch.bfloat16)
    pipe.eval()

    predictions: dict[str, str] = {}
    start_time = time.time()
    for rank, sample_info in enumerate(samples):
        pred_path = pred_dir / f"{sample_info.sample_id}_pred.jpg"
        if pred_path.exists() and not overwrite:
            predictions[sample_info.sample_id] = str(pred_path)
            print(f"[pred] {checkpoint.name} {rank + 1:04d}/{len(samples):04d} cached {sample_info.sample_id}")
            continue

        sample = dataset[sample_info.dataset_index]
        source = sample["source_image"].convert("RGB")
        caption = str(sample["caption"])
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
                enable_progress_bar=False,
            ).images[0].convert("RGB")

        pred.save(pred_path, quality=92)
        predictions[sample_info.sample_id] = str(pred_path)
        print(f"[pred] {checkpoint.name} {rank + 1:04d}/{len(samples):04d} saved {sample_info.sample_id}")

    elapsed_s = time.time() - start_time
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": checkpoint.name,
        "config_path": str(checkpoint.config_path),
        "checkpoint_path": str(checkpoint.checkpoint_path),
        "group": checkpoint.group,
        "learning_rate": checkpoint.learning_rate,
        "source_frame_stride": checkpoint.source_frame_stride,
        "target_frame_offset": checkpoint.target_frame_offset,
        "trajectory_motion_filter": checkpoint.trajectory_motion_filter,
        "min_trajectory_delta": checkpoint.min_trajectory_delta,
        "num_train_epochs": checkpoint.num_train_epochs,
        "elapsed_s": elapsed_s,
        "predictions": predictions,
    }


def write_report(
    report_path: Path,
    manifest_path: Path,
    output_dir: Path,
    dataset_root: Path,
    camera_key: str,
    target_frame_offset: int,
    samples: list[SampleInfo],
    checkpoints: list[CheckpointInfo],
    checkpoint_results: dict[str, dict[str, Any]],
    skipped: list[dict[str, str]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    for sample in samples:
        cells = [
            f"<td class='meta'>ep {sample.episode_index}<br>src {sample.source_frame}<br>gt {sample.target_frame}</td>",
            f"<td><img loading='lazy' src='{html.escape(rel(sample.source_path, output_dir))}'></td>",
            f"<td><img loading='lazy' src='{html.escape(rel(sample.target_path, output_dir))}'></td>",
        ]
        for checkpoint in checkpoints:
            result = checkpoint_results.get(checkpoint.name, {})
            pred_path = result.get("predictions", {}).get(sample.sample_id)
            if pred_path:
                src = html.escape(rel(Path(pred_path), output_dir))
                cells.append(f"<td><img loading='lazy' src='{src}'></td>")
            else:
                cells.append("<td class='missing'>missing</td>")
        sample_rows.append("<tr>" + "".join(cells) + "</tr>")

    checkpoint_cards = []
    for checkpoint in checkpoints:
        step = numeric_checkpoint_step(checkpoint.checkpoint_path)
        result = checkpoint_results.get(checkpoint.name, {})
        elapsed = result.get("elapsed_s")
        elapsed_text = f"{elapsed / 60:.1f} min" if isinstance(elapsed, (int, float)) else "pending"
        checkpoint_cards.append(
            "<tr>"
            f"<td>{html.escape(checkpoint.name)}</td>"
            f"<td>{html.escape(checkpoint.group)}</td>"
            f"<td>{step}</td>"
            f"<td>{html.escape(str(checkpoint.learning_rate))}</td>"
            f"<td>{html.escape(str(checkpoint.source_frame_stride))}</td>"
            f"<td>{html.escape(str(checkpoint.trajectory_motion_filter))}</td>"
            f"<td>{html.escape(str(checkpoint.min_trajectory_delta))}</td>"
            f"<td>{html.escape(elapsed_text)}</td>"
            "</tr>"
        )

    skipped_rows = []
    for item in skipped:
        skipped_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('config', '')))}</td>"
            f"<td>{html.escape(str(item.get('run_name', '')))}</td>"
            f"<td>{html.escape(str(item.get('reason', '')))}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ForeAct aloha_banana_fix report</title>
<style>
:root {{
  color-scheme: light;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f6f7f9;
  color: #1f2933;
}}
body {{ margin: 0; }}
header {{
  padding: 24px 28px 18px;
  background: #ffffff;
  border-bottom: 1px solid #d8dee6;
}}
h1 {{ margin: 0 0 10px; font-size: 22px; letter-spacing: 0; }}
p {{ margin: 4px 0; color: #52606d; font-size: 14px; }}
main {{ padding: 18px 28px 32px; }}
section {{ margin: 0 0 22px; }}
h2 {{ margin: 0 0 10px; font-size: 16px; letter-spacing: 0; }}
.table-wrap {{
  overflow: auto;
  border: 1px solid #d8dee6;
  background: #ffffff;
}}
table {{ border-collapse: collapse; width: max-content; min-width: 100%; }}
th, td {{
  border-right: 1px solid #e2e8f0;
  border-bottom: 1px solid #e2e8f0;
  padding: 8px;
  vertical-align: top;
  font-size: 12px;
}}
th {{
  position: sticky;
  top: 0;
  z-index: 2;
  background: #eef2f7;
  text-align: left;
  white-space: nowrap;
}}
td.meta {{
  position: sticky;
  left: 0;
  z-index: 1;
  background: #ffffff;
  min-width: 92px;
  color: #334e68;
  line-height: 1.45;
}}
img {{
  display: block;
  width: 240px;
  height: auto;
  background: #e5e7eb;
}}
.summary table {{ width: 100%; }}
.summary th, .summary td {{ white-space: nowrap; }}
.missing {{ color: #9b1c1c; background: #fff5f5; text-align: center; }}
.caption {{ max-width: 520px; white-space: normal; color: #52606d; }}
</style>
</head>
<body>
<header>
  <h1>ForeAct aloha_banana_fix future-frame report</h1>
  <p>Dataset: {html.escape(str(dataset_root))}</p>
  <p>Camera: {html.escape(camera_key)} | Target offset: {target_frame_offset} frames | Samples: {len(samples)} from {len({s.episode_index for s in samples})} videos | Checkpoints: {len(checkpoints)}</p>
  <p>Manifest: {html.escape(rel(manifest_path, output_dir))}</p>
</header>
<main>
  <section class="summary">
    <h2>Checkpoints</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Run</th><th>Group</th><th>Step</th><th>LR</th><th>Train stride</th><th>Motion filter</th><th>Min delta</th><th>Runtime</th></tr></thead>
        <tbody>{''.join(checkpoint_cards)}</tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Predictions</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Sample</th><th>Source</th><th>GT +30</th>{''.join(f'<th>{html.escape(c.name)}</th>' for c in checkpoints)}
          </tr>
        </thead>
        <tbody>{''.join(sample_rows)}</tbody>
      </table>
    </div>
  </section>
  <section class="summary">
    <h2>Skipped Configs</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Config</th><th>Run</th><th>Reason</th></tr></thead>
        <tbody>{''.join(skipped_rows) or '<tr><td colspan="3">None</td></tr>'}</tbody>
      </table>
    </div>
  </section>
</main>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")
    print(f"[report] wrote {report_path}")


def write_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    samples: list[SampleInfo],
    checkpoints: list[CheckpointInfo],
    checkpoint_results: dict[str, dict[str, Any]],
    skipped: list[dict[str, str]],
) -> None:
    data = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "args": vars(args),
        "samples": [
            {
                "sample_id": s.sample_id,
                "dataset_index": s.dataset_index,
                "episode_index": s.episode_index,
                "source_frame": s.source_frame,
                "target_frame": s.target_frame,
                "caption": s.caption,
                "source_path": str(s.source_path),
                "target_path": str(s.target_path),
            }
            for s in samples
        ],
        "checkpoints": [
            {
                "name": c.name,
                "config_path": str(c.config_path),
                "checkpoint_path": str(c.checkpoint_path),
                "group": c.group,
                "learning_rate": c.learning_rate,
                "source_frame_stride": c.source_frame_stride,
                "target_frame_offset": c.target_frame_offset,
                "trajectory_motion_filter": c.trajectory_motion_filter,
                "min_trajectory_delta": c.min_trajectory_delta,
                "num_train_epochs": c.num_train_epochs,
            }
            for c in checkpoints
        ],
        "checkpoint_results": checkpoint_results,
        "skipped": skipped,
    }
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] wrote {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="datasets/aloha_banana_fix")
    parser.add_argument("--camera-key", default="observation.images.cam_high")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="outputs/aloha_banana_fix_report")
    parser.add_argument("--run-id", default="", help="Subdirectory name under output-dir. Defaults to timestamp.")
    parser.add_argument("--target-frame-offset", type=int, default=30)
    parser.add_argument("--source-frame-stride", type=int, default=30)
    parser.add_argument("--samples-per-episode", type=int, default=2)
    parser.add_argument("--include-mixed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-name", action="append", default=[], help="Optional run name filter. Can be repeated.")
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_root = Path(args.dataset_root)
    config_dir = Path(args.config_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "index.html"

    checkpoints, skipped = discover_checkpoints(config_dir, checkpoints_dir, args.include_mixed)
    if args.checkpoint_name:
        allowed = set(args.checkpoint_name)
        skipped.extend(
            {"config": str(c.config_path), "run_name": c.name, "reason": "filtered by --checkpoint-name"}
            for c in checkpoints
            if c.name not in allowed
        )
        checkpoints = [c for c in checkpoints if c.name in allowed]
    if args.max_checkpoints > 0:
        skipped.extend(
            {"config": str(c.config_path), "run_name": c.name, "reason": "filtered by --max-checkpoints"}
            for c in checkpoints[args.max_checkpoints :]
        )
        checkpoints = checkpoints[: args.max_checkpoints]

    if not checkpoints:
        raise RuntimeError("No banana-trained checkpoints found.")

    dataset = ImagePairDataset(
        root=dataset_root,
        camera_key=args.camera_key,
        target_frame_offset=args.target_frame_offset,
        source_frame_stride=args.source_frame_stride,
    )
    selected_indices = select_samples(dataset, args.samples_per_episode)
    if not selected_indices:
        raise RuntimeError("No valid source/target frame pairs found.")

    print(f"[setup] dataset={dataset_root} samples={len(selected_indices)} checkpoints={len(checkpoints)}")
    for checkpoint in checkpoints:
        print(f"[setup] checkpoint {checkpoint.name} -> {checkpoint.checkpoint_path}")

    samples = save_reference_images(dataset, selected_indices, assets_dir)
    checkpoint_results: dict[str, dict[str, Any]] = {}

    if not args.dry_run:
        for rank, checkpoint in enumerate(checkpoints):
            print(f"[setup] running checkpoint {rank + 1}/{len(checkpoints)}")
            checkpoint_results[checkpoint.name] = run_checkpoint(
                checkpoint=checkpoint,
                dataset=dataset,
                samples=samples,
                assets_dir=assets_dir,
                device=args.device,
                seed=args.seed + rank * 100000,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                overwrite=args.overwrite,
            )
            write_manifest(manifest_path, args, samples, checkpoints, checkpoint_results, skipped)
            write_report(
                report_path=report_path,
                manifest_path=manifest_path,
                output_dir=output_dir,
                dataset_root=dataset_root,
                camera_key=args.camera_key,
                target_frame_offset=args.target_frame_offset,
                samples=samples,
                checkpoints=checkpoints,
                checkpoint_results=checkpoint_results,
                skipped=skipped,
            )
    else:
        print("[dry-run] inference skipped")

    write_manifest(manifest_path, args, samples, checkpoints, checkpoint_results, skipped)
    write_report(
        report_path=report_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        dataset_root=dataset_root,
        camera_key=args.camera_key,
        target_frame_offset=args.target_frame_offset,
        samples=samples,
        checkpoints=checkpoints,
        checkpoint_results=checkpoint_results,
        skipped=skipped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
