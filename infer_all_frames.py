"""
Batch inference on all video frames, saving predicted frames as videos
alongside the original videos.

Output structure:
  datasets/bridge_orig_lerobot/videos/chunk-XXX/observation.images.image_0_foreact/episode_NNNNNN.mp4

Usage:
    bash run_infer_all.sh
"""
import argparse
import json
import numpy as np
import os
import sys
import torch
import time

from PIL import Image
from torchvision.io import read_video, write_video
from pipeline import VisualForesightPipeline


def get_video_path(base_dir, episode_id):
    chunk_idx = episode_id // 1000
    return os.path.join(
        base_dir,
        f"chunk-{chunk_idx:03d}",
        "observation.images.image_0",
        f"episode_{episode_id:06d}.mp4",
    )


def get_output_video_path(base_dir, episode_id):
    chunk_idx = episode_id // 1000
    return os.path.join(
        base_dir,
        f"chunk-{chunk_idx:03d}",
        "observation.images.image_0_foreact",
        f"episode_{episode_id:06d}.mp4",
    )


def get_prompt_for_frame(episode_data, frame_idx):
    step_to_subtask_index = episode_data["step_to_subtask_index"]
    all_subtasks = episode_data["all_subtasks"]
    max_cot_step = max(int(k) for k in step_to_subtask_index.keys())
    frame_key = str(frame_idx)
    if frame_key not in step_to_subtask_index:
        return all_subtasks[-1]
    current_subtask_idx = step_to_subtask_index[frame_key]
    last_step_of_subtask = frame_idx
    for s in range(frame_idx, max_cot_step + 1):
        if str(s) in step_to_subtask_index and step_to_subtask_index[str(s)] == current_subtask_idx:
            last_step_of_subtask = s
        else:
            break
    distance_to_end = last_step_of_subtask - frame_idx
    if distance_to_end < 2 and current_subtask_idx + 1 < len(all_subtasks):
        return all_subtasks[current_subtask_idx + 1]
    return all_subtasks[current_subtask_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,
                        default="")
    parser.add_argument("--video_dir", type=str,
                        default="./datasets/bridge_orig_lerobot/videos")
    parser.add_argument("--cot_path", type=str,
                        default="./datasets/cot_by_episode.json")
    parser.add_argument("--worker_id", type=int, required=True,
                        help="Global worker ID (0-indexed)")
    parser.add_argument("--num_workers", type=int, required=True,
                        help="Total number of workers across all GPUs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    args = parser.parse_args()

    tag = f"[W{args.worker_id}]"

    print(f"{tag} Loading COT data...", flush=True)
    with open(args.cot_path, "r") as f:
        cot_data = json.load(f)

    all_episode_ids = sorted(int(k) for k in cot_data.keys())
    print(f"{tag} Total episodes with COT: {len(all_episode_ids)}", flush=True)
    all_episode_ids = all_episode_ids  # for testing

    my_episode_ids = [eid for i, eid in enumerate(all_episode_ids) if i % args.num_workers == args.worker_id]
    print(f"{tag} Assigned {len(my_episode_ids)} episodes", flush=True)

    print(f"{tag} Loading model from {args.checkpoint_path}...", flush=True)
    pipeline = VisualForesightPipeline.from_pretrained(
        args.checkpoint_path,
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
    )
    pipeline = pipeline.to(device="cuda", dtype=torch.bfloat16)
    print(f"{tag} Model loaded.", flush=True)

    total_episodes = len(my_episode_ids)
    total_frames_processed = 0
    episodes_done = 0
    start_time = time.time()

    for ep_i, episode_id in enumerate(my_episode_ids):
        output_video_path = get_output_video_path(args.video_dir, episode_id)

        if os.path.exists(output_video_path):
            episodes_done += 1
            continue

        source_video_path = get_video_path(args.video_dir, episode_id)
        if not os.path.exists(source_video_path):
            print(f"{tag} Video not found: {source_video_path}, skipping", flush=True)
            continue

        episode_data = cot_data[str(episode_id)]

        try:
            video_tensor, _, info = read_video(source_video_path, pts_unit="sec")
        except Exception as e:
            print(f"{tag} Error reading {source_video_path}: {e}", flush=True)
            continue

        num_frames = video_tensor.shape[0]
        if num_frames == 0:
            continue

        fps = int(info.get("video_fps", 5))

        pred_frames = []
        for frame_idx in range(num_frames):
            input_image = Image.fromarray(video_tensor[frame_idx].numpy())
            prompt = get_prompt_for_frame(episode_data, frame_idx)

            with torch.no_grad():
                output = pipeline(
                    image=input_image,
                    caption=prompt,
                    negative_prompt="",
                    guidance_scale=args.guidance_scale,
                    image_guidance_scale=args.image_guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    num_images_per_prompt=1,
                    generator=torch.Generator(device="cuda").manual_seed(args.seed),
                )

            pred_np = np.array(output.images[0], dtype=np.uint8)
            pred_frames.append(pred_np)
            total_frames_processed += 1

        pred_tensor = torch.from_numpy(np.stack(pred_frames, axis=0))
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        tmp_path = output_video_path + ".tmp.mp4"
        write_video(tmp_path, pred_tensor, fps=fps, video_codec="libx264")
        os.rename(tmp_path, output_video_path)

        episodes_done += 1
        elapsed = time.time() - start_time
        eps = episodes_done / elapsed if elapsed > 0 else 0
        remaining = total_episodes - ep_i - 1
        eta = remaining / eps if eps > 0 else 0
        print(
            f"{tag} Episode {episode_id} done "
            f"({ep_i+1}/{total_episodes}, {num_frames}f, "
            f"total {total_frames_processed}f, "
            f"ETA: {eta/3600:.1f}h)",
            flush=True,
        )

    print(f"{tag} All done! {episodes_done} episodes, {total_frames_processed} frames.", flush=True)


if __name__ == "__main__":
    main()
