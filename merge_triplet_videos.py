
"""
Merge same-named videos from three folders into one side-by-side video.

Example:
  python merge_triplet_videos.py \
    --dir_a datasets/bridge_orig_lerobot/videos/chunk-000/observation.images.image_0 \
    --dir_b datasets/bridge_orig_lerobot/videos/chunk-000/observation.images.image_0_foreact \
    --dir_c datasets/bridge_orig_lerobot/videos/chunk-000/observation.images.image_0_foreact_mx \
    --output_dir datasets/bridge_orig_lerobot/videos/chunk-000/observation.images.image_0_triplet
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_video, write_video


def list_mp4_files(folder: str) -> List[str]:
    return sorted([f for f in os.listdir(folder) if f.endswith(".mp4")])


def validate_and_collect_triplets(dir_a: str, dir_b: str, dir_c: str) -> List[Tuple[str, str, str, str]]:
    names_a = set(list_mp4_files(dir_a))
    names_b = set(list_mp4_files(dir_b))
    names_c = set(list_mp4_files(dir_c))

    common = sorted(names_a & names_b & names_c)
    if not common:
        raise RuntimeError("No common .mp4 filenames across the three folders.")

    return [
        (
            name,
            os.path.join(dir_a, name),
            os.path.join(dir_b, name),
            os.path.join(dir_c, name),
        )
        for name in common
    ]


def resize_video_to_height(video_thwc: torch.Tensor, target_h: int) -> torch.Tensor:
    """Resize each frame to target height, keeping aspect ratio."""
    # THWC -> TCHW for interpolation
    t, h, w, c = video_thwc.shape
    if h == target_h:
        return video_thwc

    new_w = max(1, int(round(w * (target_h / float(h)))))

    video_tchw = video_thwc.permute(0, 3, 1, 2).float()
    resized = F.interpolate(video_tchw, size=(target_h, new_w), mode="bilinear", align_corners=False)
    # Back to THWC uint8
    resized = resized.clamp(0, 255).byte().permute(0, 2, 3, 1).contiguous()
    return resized


def merge_three_videos(path_a: str, path_b: str, path_c: str, out_path: str) -> None:
    va, _, info_a = read_video(path_a, pts_unit="sec")
    vb, _, info_b = read_video(path_b, pts_unit="sec")
    vc, _, info_c = read_video(path_c, pts_unit="sec")

    if va.numel() == 0 or vb.numel() == 0 or vc.numel() == 0:
        raise RuntimeError("At least one input video has no frames.")

    # Keep output duration unchanged by using min frame count.
    t = min(va.shape[0], vb.shape[0], vc.shape[0])
    if t <= 0:
        raise RuntimeError("No overlapping frames among the three videos.")

    va = va[:t]
    vb = vb[:t]
    vc = vc[:t]

    # Align heights to avoid concat shape mismatch.
    target_h = min(va.shape[1], vb.shape[1], vc.shape[1])
    va = resize_video_to_height(va, target_h)
    vb = resize_video_to_height(vb, target_h)
    vc = resize_video_to_height(vc, target_h)

    merged = torch.cat([va, vb, vc], dim=2)  # THWC concat on width

    fps = info_a.get("video_fps", info_b.get("video_fps", info_c.get("video_fps", 5)))
    fps = int(round(float(fps))) if fps else 5

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp.mp4"
    write_video(tmp_path, merged, fps=fps, video_codec="libx264")
    os.replace(tmp_path, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Horizontally stitch same-name videos from 3 folders.")
    parser.add_argument("--dir_a", type=str, required=True)
    parser.add_argument("--dir_b", type=str, required=True)
    parser.add_argument("--dir_c", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_videos", type=int, default=0, help="0 means process all")
    args = parser.parse_args()

    triplets = validate_and_collect_triplets(args.dir_a, args.dir_b, args.dir_c)
    if args.max_videos > 0:
        triplets = triplets[: args.max_videos]

    print(f"Found {len(triplets)} matching videos to process.")

    ok = 0
    for i, (name, pa, pb, pc) in enumerate(triplets, start=1):
        out_path = os.path.join(args.output_dir, name)
        try:
            merge_three_videos(pa, pb, pc, out_path)
            ok += 1
            print(f"[{i}/{len(triplets)}] Done: {name}")
        except Exception as e:
            print(f"[{i}/{len(triplets)}] Failed: {name} | {e}")

    print(f"Completed: {ok}/{len(triplets)}")


if __name__ == "__main__":
    main()
