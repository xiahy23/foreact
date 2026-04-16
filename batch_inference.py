"""
Batch inference script for ForeAct with per-video prompts.

python batch_inference.py \
    --checkpoint_path ./checkpoints/finetuned_bridge2/run_finetune/checkpoint-2000 \
    --test_dir ./test \
    --output_dir ./batch_results_2
"""
import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from pipeline import VisualForesightPipeline
from utils.trainer_utils import find_newest_checkpoint

# Video filename (sorted) -> prompt mapping
VIDEO_PROMPTS = {
    "failure_obj_episode_0_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_False_src_on_target_False": "stack green cube on yellow cube",
    "failure_obj_episode_0_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_False": "put eggplant in basket",
    "failure_obj_episode_3_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_False": "put spoon on cloth",
    "failure_obj_episode_45_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_False_consecutive_grasp_False_src_on_target_False": "put carrot on plate",
    "success_obj_episode_1_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_True": "put eggplant in basket",
    "success_obj_episode_36_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_True": "put carrot on plate",
    "success_obj_episode_43_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_True": "put spoon on cloth",
    "success_obj_episode_60_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_True": "stack green cube on yellow cube",
}


def extract_frames(video_path, frame_interval=10, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_indices = list(range(0, total_frames, frame_interval))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frames.append((idx, pil_image))
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="Batch inference for ForeAct")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./batch_results")
    parser.add_argument("--frame_interval", type=int, default=6)
    parser.add_argument("--max_frames_per_video", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print("Loading model...")
    checkpoint = find_newest_checkpoint(args.checkpoint_path)
    print(f"Using checkpoint: {checkpoint}")
    pipeline = VisualForesightPipeline.from_pretrained(
        checkpoint,
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
    )
    pipeline = pipeline.to(device="cuda", dtype=torch.bfloat16)
    print("Model loaded successfully!\n")

    video_files = sorted(glob.glob(os.path.join(args.test_dir, "*.mp4")))
    if not video_files:
        print(f"No video files found in {args.test_dir}")
        return

    print(f"Found {len(video_files)} videos\n")
    os.makedirs(args.output_dir, exist_ok=True)

    results_meta = []

    for vid_idx, video_path in enumerate(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        short_name = video_name[:80] + "..." if len(video_name) > 80 else video_name

        # Get per-video prompt
        prompt = VIDEO_PROMPTS.get(video_name, "")
        if not prompt:
            print(f"WARNING: No prompt found for {video_name}, using empty prompt.")

        print(f"\n[{vid_idx+1}/{len(video_files)}] Processing: {short_name}")
        print(f"  Prompt: \"{prompt}\"")

        video_output_dir = os.path.join(args.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        frames = extract_frames(video_path, args.frame_interval, args.max_frames_per_video)
        if not frames:
            print(f"  No frames extracted, skipping.")
            continue

        print(f"  Extracted {len(frames)} frames")

        video_results = []
        for frame_idx, (original_frame_idx, input_image) in enumerate(frames):
            print(f"  Frame {frame_idx+1}/{len(frames)} (video frame #{original_frame_idx})...", end=" ", flush=True)

            with torch.no_grad():
                output = pipeline(
                    image=input_image,
                    caption=prompt,
                    negative_prompt="",
                    guidance_scale=args.guidance_scale,
                    image_guidance_scale=args.image_guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    num_images_per_prompt=1,
                    generator=torch.Generator().manual_seed(args.seed + frame_idx),
                )

            predicted_image = output.images[0]

            input_path = os.path.join(video_output_dir, f"frame_{original_frame_idx:04d}_input.png")
            pred_path = os.path.join(video_output_dir, f"frame_{original_frame_idx:04d}_predicted.png")

            input_image.save(input_path)
            predicted_image.save(pred_path)

            video_results.append({
                "frame_idx": original_frame_idx,
                "input_path": os.path.relpath(input_path, args.output_dir),
                "pred_path": os.path.relpath(pred_path, args.output_dir),
            })

            print("done")

        results_meta.append({
            "video_name": video_name,
            "short_name": short_name,
            "prompt": prompt,
            "frames": video_results,
        })

    generate_html(results_meta, args.output_dir)
    print(f"\n{'='*60}")
    print(f"All done! Results saved to: {args.output_dir}")
    print(f"Open {os.path.join(args.output_dir, 'index.html')} in a browser to view.")
    print(f"Or run: python -m http.server 8888 --directory {args.output_dir}")
    print(f"{'='*60}")


def generate_html(results_meta, output_dir):
    total_frames = sum(len(v["frames"]) for v in results_meta)

    sections = []
    for vid_idx, video in enumerate(results_meta):
        is_success = video["video_name"].startswith("success")
        badge_class = "badge-success" if is_success else "badge-failure"
        badge_text = "SUCCESS" if is_success else "FAILURE"
        prompt = video.get("prompt", "")

        frames_html = ""
        for fr in video["frames"]:
            frames_html += f'''
            <div class="frame-card">
                <div class="frame-label">Frame #{fr['frame_idx']}</div>
                <div class="side-by-side">
                    <div>
                        <img src="{fr['input_path']}" onclick="openModal(this.src)" title="Input frame">
                        <div class="side-label">Input</div>
                    </div>
                    <div>
                        <img src="{fr['pred_path']}" onclick="openModal(this.src)" title="Predicted frame">
                        <div class="side-label">ForeAct Prediction</div>
                    </div>
                </div>
            </div>
'''

        section = f'''
    <div class="video-section">
        <div class="video-header" onclick="toggleSection({vid_idx})">
            <div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;">
                <span class="toggle-arrow" id="arrow-{vid_idx}">&#9654;</span>
                <span class="video-title">{video['short_name']}</span>
                <span class="prompt-badge">Prompt: "{prompt}"</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
                <span class="video-badge {badge_class}">{badge_text}</span>
                <span class="video-badge" style="background:#1a1a3e;color:#aaa;border:1px solid #2a2a5a;">{len(video['frames'])} frames</span>
            </div>
        </div>
        <div class="video-content" id="content-{vid_idx}">
            <div class="frames-grid">
                {frames_html}
            </div>
        </div>
    </div>
'''
        sections.append(section)

    sections_str = "\n".join(sections)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ForeAct Inference Results</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: #0f0f23;
        color: #e0e0e0;
        padding: 20px;
    }}
    h1 {{
        text-align: center;
        color: #00d4ff;
        margin-bottom: 10px;
        font-size: 2em;
    }}
    .subtitle {{
        text-align: center;
        color: #888;
        margin-bottom: 30px;
        font-size: 0.95em;
    }}
    .stats {{
        text-align: center;
        margin-bottom: 30px;
        color: #aaa;
    }}
    .stats span {{
        background: #1a1a3e;
        padding: 5px 15px;
        border-radius: 20px;
        margin: 0 5px;
        font-size: 0.9em;
    }}
    .video-section {{
        background: #1a1a2e;
        border-radius: 12px;
        margin-bottom: 30px;
        overflow: hidden;
        border: 1px solid #2a2a4a;
    }}
    .video-header {{
        background: linear-gradient(135deg, #16213e, #0f3460);
        padding: 15px 20px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
        user-select: none;
        gap: 10px;
    }}
    .video-header:hover {{
        background: linear-gradient(135deg, #1a2a4e, #134070);
    }}
    .video-title {{
        font-size: 1.05em;
        font-weight: 600;
        color: #e0e0e0;
        word-break: break-all;
    }}
    .prompt-badge {{
        background: #2a1a4e;
        color: #c084fc;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 600;
        border: 1px solid #7c3aed;
        white-space: nowrap;
    }}
    .video-badge {{
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        flex-shrink: 0;
    }}
    .badge-success {{
        background: #0a4a2a;
        color: #4ade80;
        border: 1px solid #166534;
    }}
    .badge-failure {{
        background: #4a0a0a;
        color: #f87171;
        border: 1px solid #991b1b;
    }}
    .video-content {{
        padding: 20px;
        display: none;
    }}
    .video-content.active {{
        display: block;
    }}
    .frames-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(550px, 1fr));
        gap: 15px;
    }}
    .frame-card {{
        background: #16213e;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #2a2a5a;
        transition: transform 0.2s;
    }}
    .frame-card:hover {{
        transform: scale(1.02);
        border-color: #00d4ff;
    }}
    .frame-label {{
        padding: 8px 12px;
        background: #0f3460;
        color: #00d4ff;
        font-size: 0.85em;
        font-weight: 600;
    }}
    .frame-card img {{
        width: 100%;
        display: block;
        cursor: pointer;
    }}
    .toggle-arrow {{
        font-size: 1.2em;
        transition: transform 0.3s;
        color: #00d4ff;
    }}
    .toggle-arrow.open {{
        transform: rotate(90deg);
    }}
    .side-by-side {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 3px;
    }}
    .side-by-side img {{
        width: 100%;
        display: block;
    }}
    .side-label {{
        text-align: center;
        padding: 4px;
        font-size: 0.8em;
        color: #aaa;
    }}
    .modal {{
        display: none;
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0,0,0,0.9);
        z-index: 1000;
        cursor: pointer;
        justify-content: center;
        align-items: center;
    }}
    .modal.active {{
        display: flex;
    }}
    .modal img {{
        max-width: 95%;
        max-height: 95%;
        object-fit: contain;
    }}
    .expand-all {{
        background: #1a1a3e;
        color: #00d4ff;
        border: 1px solid #2a2a5a;
        padding: 8px 20px;
        border-radius: 20px;
        cursor: pointer;
        font-size: 0.9em;
        margin: 0 5px;
    }}
    .expand-all:hover {{
        background: #0f3460;
    }}
    .view-controls {{
        text-align: center;
        margin-bottom: 20px;
    }}
</style>
</head>
<body>

<h1>ForeAct Visual Foresight Results</h1>
<p class="subtitle">Input frames vs. predicted future frames (with task prompts)</p>

<div class="stats">
    <span>Total: {len(results_meta)} videos</span>
    <span>Total: {total_frames} frame pairs</span>
</div>

<div class="view-controls">
    <button class="expand-all" onclick="toggleAll()">Expand / Collapse All</button>
</div>

<div class="modal" id="modal" onclick="closeModal()">
    <img id="modal-img" src="">
</div>

{sections_str}

<script>
function toggleSection(id) {{
    var content = document.getElementById('content-' + id);
    var arrow = document.getElementById('arrow-' + id);
    content.classList.toggle('active');
    arrow.classList.toggle('open');
}}

function toggleAll() {{
    var contents = document.querySelectorAll('.video-content');
    var arrows = document.querySelectorAll('.toggle-arrow');
    var anyOpen = false;
    for (var i = 0; i < contents.length; i++) {{
        if (contents[i].classList.contains('active')) {{ anyOpen = true; break; }}
    }}
    for (var i = 0; i < contents.length; i++) {{
        if (anyOpen) {{ contents[i].classList.remove('active'); }}
        else {{ contents[i].classList.add('active'); }}
    }}
    for (var i = 0; i < arrows.length; i++) {{
        if (anyOpen) {{ arrows[i].classList.remove('open'); }}
        else {{ arrows[i].classList.add('open'); }}
    }}
}}

function openModal(src) {{
    document.getElementById('modal-img').src = src;
    document.getElementById('modal').classList.add('active');
}}

function closeModal() {{
    document.getElementById('modal').classList.remove('active');
}}

document.addEventListener('keydown', function(e) {{
    if (e.key === 'Escape') closeModal();
}});
</script>
</body>
</html>'''

    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\nHTML viewer saved to: {html_path}")


if __name__ == "__main__":
    main()