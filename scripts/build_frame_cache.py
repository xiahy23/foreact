#!/usr/bin/env python3
"""Build a decoded frame cache for LeRobot-style video datasets."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
from PIL import Image


def load_episode_lengths(dataset_root: Path) -> dict[int, int]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing episodes metadata: {episodes_path}")

    lengths: dict[int, int] = {}
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            episode_index = int(item["episode_index"])
            lengths[episode_index] = int(item.get("length", item.get("num_frames", 0)))
    return lengths


def load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset info: {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def video_path_for_episode(dataset_root: Path, info: dict, camera_key: str, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    template = info.get("video_path")
    chunk = episode_chunk(episode_index, chunks_size)
    if template:
        rel = template.format(
            episode_chunk=chunk,
            video_key=camera_key,
            episode_index=episode_index,
        )
    else:
        rel = f"videos/chunk-{chunk:03d}/{camera_key}/episode_{episode_index:06d}.mp4"
    return dataset_root / rel


def cache_dir_for_episode(cache_dataset_root: Path, camera_key: str, chunks_size: int, episode_index: int) -> Path:
    chunk = episode_chunk(episode_index, chunks_size)
    return cache_dataset_root / f"chunk-{chunk:03d}" / camera_key / f"episode_{episode_index:06d}"


def is_complete(cache_dir: Path, expected_frames: int, ext: str) -> bool:
    if expected_frames <= 0:
        return False
    first = cache_dir / f"frame_{0:06d}.{ext}"
    last = cache_dir / f"frame_{expected_frames - 1:06d}.{ext}"
    done = cache_dir / ".complete"
    return first.is_file() and last.is_file() and done.is_file()


def decode_episode(
    dataset_root: str,
    cache_dataset_root: str,
    camera_key: str,
    episode_index: int,
    expected_frames: int,
    chunks_size: int,
    video_rel_path: str,
    ext: str,
    quality: int,
    overwrite: bool,
) -> tuple[int, int, str]:
    video_path = Path(dataset_root) / video_rel_path
    cache_dir = cache_dir_for_episode(Path(cache_dataset_root), camera_key, chunks_size, episode_index)

    if not overwrite and is_complete(cache_dir, expected_frames, ext):
        return episode_index, expected_frames, "skip"

    cache_dir.mkdir(parents=True, exist_ok=True)

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    count = 0
    for packet in container.demux(stream):
        for frame in packet.decode():
            image = frame.to_image().convert("RGB")
            out_path = cache_dir / f"frame_{count:06d}.{ext}"
            if ext.lower() in {"jpg", "jpeg"}:
                image.save(out_path, quality=quality, subsampling=0)
            else:
                image.save(out_path)
            count += 1

    container.close()

    if expected_frames > 0 and count < expected_frames:
        raise RuntimeError(
            f"Decoded too few frames for episode {episode_index}: got {count}, expected {expected_frames}"
        )

    (cache_dir / ".complete").write_text(str(count), encoding="utf-8")
    return episode_index, count, "done"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode LeRobot videos into an image frame cache.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--camera-key", default="observation.images.cam_high")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "png"])
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--episode-indices", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    info = load_info(args.dataset_root)
    lengths = load_episode_lengths(args.dataset_root)
    chunks_size = int(info.get("chunks_size", 1000))
    template = info.get("video_path")
    cache_dataset_root = args.cache_root / args.dataset_name
    cache_dataset_root.mkdir(parents=True, exist_ok=True)

    selected_episode_indices: set[int] | None = None
    if args.episode_indices.strip():
        selected_episode_indices = {
            int(item.strip())
            for item in args.episode_indices.split(",")
            if item.strip()
        }

    tasks = []
    for episode_index, expected_frames in sorted(lengths.items()):
        if selected_episode_indices is not None and episode_index not in selected_episode_indices:
            continue
        chunk = episode_chunk(episode_index, chunks_size)
        if template:
            video_rel = template.format(
                episode_chunk=chunk,
                video_key=args.camera_key,
                episode_index=episode_index,
            )
        else:
            video_rel = f"videos/chunk-{chunk:03d}/{args.camera_key}/episode_{episode_index:06d}.mp4"
        video_path = args.dataset_root / video_rel
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing video: {video_path}")
        tasks.append((episode_index, expected_frames, video_rel))
        if args.max_episodes and len(tasks) >= args.max_episodes:
            break

    print(
        f"Building frame cache: dataset={args.dataset_root}, camera={args.camera_key}, "
        f"episodes={len(tasks)}, cache={cache_dataset_root}, workers={args.workers}, ext={args.ext}",
        flush=True,
    )

    completed = 0
    skipped = 0
    frames = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                decode_episode,
                str(args.dataset_root),
                str(cache_dataset_root),
                args.camera_key,
                episode_index,
                expected_frames,
                chunks_size,
                video_rel,
                args.ext,
                args.quality,
                args.overwrite,
            )
            for episode_index, expected_frames, video_rel in tasks
        ]
        for future in as_completed(futures):
            episode_index, count, status = future.result()
            frames += count
            if status == "skip":
                skipped += 1
            else:
                completed += 1
            done = completed + skipped
            if done % 10 == 0 or done == len(tasks):
                print(
                    f"[{done}/{len(tasks)}] completed={completed} skipped={skipped} "
                    f"frames={frames} last_episode={episode_index:06d}",
                    flush=True,
                )

    print(
        f"Done: completed={completed}, skipped={skipped}, frames={frames}, cache={cache_dataset_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
