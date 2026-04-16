"""
Batch inference using Doubao Seedream 5.0 for visual foresight.
Extracts frames from test videos, sends each frame + prompt to Doubao image generation API,
saves input-prediction pairs and generates an HTML viewer.
"""
import argparse
import base64
import glob
import io
import json
import os
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
from volcenginesdkarkruntime import Ark


# ─── Video-to-prompt mapping (same order as sorted test filenames) ───
VIDEO_PROMPTS = {
    "failure_obj_episode_0_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_False_src_on_target_False":
        "stack green cube on yellow cube",
    "failure_obj_episode_0_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_False":
        "put eggplant in basket",
    "failure_obj_episode_3_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_False":
        "put spoon on cloth",
    "failure_obj_episode_45_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_False_consecutive_grasp_False_src_on_target_False":
        "put carrot on plate",
    "success_obj_episode_1_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_True":
        "put eggplant in basket",
    "success_obj_episode_36_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_True":
        "put carrot on plate",
    "success_obj_episode_43_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_True":
        "put spoon on cloth",
    "success_obj_episode_60_moved_correct_obj_True_moved_wrong_obj_False_is_src_obj_grasped_True_consecutive_grasp_True_src_on_target_True":
        "stack green cube on yellow cube",
}


def pil_to_base64_uri(img, fmt="JPEG", quality=90):
    """Convert PIL image to data URI string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def extract_frames_ffmpeg(video_path, frame_interval=6, max_frames=8):
    """Extract frames using ffmpeg (handles AV1 etc.)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True, timeout=30
    )
    parts = probe.stdout.strip().split(",")
    w, h = int(parts[0]), int(parts[1])

    # Try to get total frames
    try:
        total = int(parts[2])
    except (ValueError, IndexError):
        total = 200  # fallback

    frame_indices = list(range(0, total, frame_interval))[:max_frames]
    frames = []

    for idx in frame_indices:
        cmd = [
            "ffmpeg", "-v", "error", "-i", video_path,
            "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1",
            "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0 or len(result.stdout) < w * h * 3:
            continue
        arr = np.frombuffer(result.stdout[:w * h * 3], dtype=np.uint8).reshape(h, w, 3)
        frames.append((idx, Image.fromarray(arr)))

    return frames


def extract_frames_cv2(video_path, frame_interval=6, max_frames=8):
    """Extract frames using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return extract_frames_ffmpeg(video_path, frame_interval, max_frames)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = list(range(0, total, frame_interval))[:max_frames]
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((idx, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))

    cap.release()
    if not frames:
        return extract_frames_ffmpeg(video_path, frame_interval, max_frames)
    return frames


def doubao_predict(client, model, input_image, prompt, seed=42, retry=3):
    """Call Doubao Seedream API for image-to-image prediction."""
    img_uri = pil_to_base64_uri(input_image)
    w, h = input_image.size

    full_prompt = (
        f"这是一张机械臂操作场景图，当前总体任务是：“{prompt}”。\n"
        f"请基于当前画面，推理并仅生成【下一个微小分解动作】。\n"
        f"【动作推演逻辑】：\n"
        f"- 若夹爪正在靠近物体：仅让夹爪移动到物体上方或侧面。\n"
        f"- 若夹爪已靠近但未对齐：必须先【旋转夹爪】，使其双指开口方向与目标物体的长轴或几何形状完美平行对齐。\n"
        f"- 若夹爪已完全对齐：仅让金属双指从两侧向内闭合，直到表面刚好贴紧目标物体。\n"
        f"- 若夹爪已夹紧物体：仅让夹爪带着物体朝终点移动一小段距离。\n"
        f"【强制物理约束条件（极其重要）】：\n"
        f"1. 刚体防穿模：夹爪和物体都是不可形变的坚硬刚体！绝对禁止穿模、交叉或融合！夹爪金属表面只能【贴紧】物体表面，绝对不能穿透进物体内部（No clipping or intersecting）。\n"
        f"2. 严格保持夹爪的机械形态、双叉结构、材质和颜色完全不变，不能变异出人手或多余部件。\n"
        f"3. 保持背景、光影、桌面以及未被夹紧的物体完全不变，不能出现物体悬空。\n"
    )

    for attempt in range(retry):
        try:
            result = client.images.generate(
                model=model,
                prompt=full_prompt,
                image=img_uri,
                seed=seed,
                watermark=False,
                size="2304x1728",
                # guidance_scale=2.5,
            )
            # Decode result
            if result.data and len(result.data) > 0:
                b64 = result.data[0].b64_json
                if b64:
                    img_bytes = base64.b64decode(b64)
                    pred_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    pred_img = pred_img.resize((w, h), resample=Image.BICUBIC)
                    return pred_img
                elif result.data[0].url:
                    import urllib.request
                    resp = urllib.request.urlopen(result.data[0].url, timeout=30)
                    pred_img = Image.open(io.BytesIO(resp.read())).convert("RGB")
                    pred_img = pred_img.resize((w, h), resample=Image.BICUBIC)
                    return pred_img
            print(f"    Warning: Empty result on attempt {attempt+1}")
        except Exception as e:
            print(f"    Error on attempt {attempt+1}: {e}")
            if attempt < retry - 1:
                time.sleep(2 ** attempt)

    return None


def generate_html(results_meta, output_dir):
    """Generate HTML viewer."""
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
                        <img src="{fr['input_path']}" onclick="openModal(this.src)">
                        <div class="side-label">Input</div>
                    </div>
                    <div>
                        <img src="{fr['pred_path']}" onclick="openModal(this.src)">
                        <div class="side-label">Doubao Prediction</div>
                    </div>
                </div>
            </div>'''

        section = f'''
    <div class="video-section">
        <div class="video-header" onclick="toggleSection({vid_idx})">
            <div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;">
                <span class="toggle-arrow" id="arrow-{vid_idx}">&#9654;</span>
                <span class="video-title">{video['short_name']}</span>
                <span class="prompt-badge">"{prompt}"</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
                <span class="video-badge {badge_class}">{badge_text}</span>
                <span class="video-badge" style="background:#1a1a3e;color:#aaa;border:1px solid #2a2a5a;">{len(video['frames'])} frames</span>
            </div>
        </div>
        <div class="video-content" id="content-{vid_idx}">
            <div class="frames-grid">{frames_html}
            </div>
        </div>
    </div>'''
        sections.append(section)

    sections_str = "\n".join(sections)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Doubao Seedream Visual Foresight</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', sans-serif; background: #0f0f23; color: #e0e0e0; padding: 20px; }}
    h1 {{ text-align: center; color: #ff6b35; margin-bottom: 10px; font-size: 2em; }}
    .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
    .stats {{ text-align: center; margin-bottom: 30px; color: #aaa; }}
    .stats span {{ background: #1a1a3e; padding: 5px 15px; border-radius: 20px; margin: 0 5px; font-size: 0.9em; }}
    .video-section {{ background: #1a1a2e; border-radius: 12px; margin-bottom: 30px; overflow: hidden; border: 1px solid #2a2a4a; }}
    .video-header {{ background: linear-gradient(135deg, #2a1a0e, #3a2010); padding: 15px 20px; cursor: pointer; display: flex; align-items: center; justify-content: space-between; user-select: none; gap: 10px; }}
    .video-header:hover {{ background: linear-gradient(135deg, #3a2a1e, #4a3020); }}
    .video-title {{ font-size: 1.05em; font-weight: 600; color: #e0e0e0; word-break: break-all; }}
    .prompt-badge {{ background: #2a1a4e; color: #c084fc; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; border: 1px solid #7c3aed; }}
    .video-badge {{ padding: 4px 12px; border-radius: 12px; font-size: 0.8em; font-weight: bold; flex-shrink: 0; }}
    .badge-success {{ background: #0a4a2a; color: #4ade80; border: 1px solid #166534; }}
    .badge-failure {{ background: #4a0a0a; color: #f87171; border: 1px solid #991b1b; }}
    .video-content {{ padding: 20px; display: none; }}
    .video-content.active {{ display: block; }}
    .frames-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(550px, 1fr)); gap: 15px; }}
    .frame-card {{ background: #16213e; border-radius: 8px; overflow: hidden; border: 1px solid #2a2a5a; transition: transform 0.2s; }}
    .frame-card:hover {{ transform: scale(1.02); border-color: #ff6b35; }}
    .frame-label {{ padding: 8px 12px; background: #3a2010; color: #ff6b35; font-size: 0.85em; font-weight: 600; }}
    .frame-card img {{ width: 100%; display: block; cursor: pointer; }}
    .toggle-arrow {{ font-size: 1.2em; transition: transform 0.3s; color: #ff6b35; }}
    .toggle-arrow.open {{ transform: rotate(90deg); }}
    .side-by-side {{ display: grid; grid-template-columns: 1fr 1fr; gap: 3px; }}
    .side-by-side img {{ width: 100%; display: block; }}
    .side-label {{ text-align: center; padding: 4px; font-size: 0.8em; color: #aaa; }}
    .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 1000; cursor: pointer; justify-content: center; align-items: center; }}
    .modal.active {{ display: flex; }}
    .modal img {{ max-width: 95%; max-height: 95%; object-fit: contain; }}
    .expand-all {{ background: #1a1a3e; color: #ff6b35; border: 1px solid #2a2a5a; padding: 8px 20px; border-radius: 20px; cursor: pointer; font-size: 0.9em; }}
    .expand-all:hover {{ background: #3a2010; }}
    .view-controls {{ text-align: center; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>Doubao Seedream 5.0 Visual Foresight</h1>
<p class="subtitle">Input frames vs. Doubao predicted future frames</p>
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
    var c = document.getElementById('content-' + id);
    var a = document.getElementById('arrow-' + id);
    c.classList.toggle('active');
    a.classList.toggle('open');
}}
function toggleAll() {{
    var cs = document.querySelectorAll('.video-content');
    var as_ = document.querySelectorAll('.toggle-arrow');
    var open = false;
    for (var i = 0; i < cs.length; i++) if (cs[i].classList.contains('active')) {{ open = true; break; }}
    for (var i = 0; i < cs.length; i++) {{ if (open) cs[i].classList.remove('active'); else cs[i].classList.add('active'); }}
    for (var i = 0; i < as_.length; i++) {{ if (open) as_[i].classList.remove('open'); else as_[i].classList.add('open'); }}
}}
function openModal(src) {{ document.getElementById('modal-img').src = src; document.getElementById('modal').classList.add('active'); }}
function closeModal() {{ document.getElementById('modal').classList.remove('active'); }}
document.addEventListener('keydown', function(e) {{ if (e.key === 'Escape') closeModal(); }});
</script>
</body>
</html>'''

    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\nHTML viewer saved to: {html_path}")


def main():
    parser = argparse.ArgumentParser(description="Doubao Seedream batch inference")
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./batch_results_doubao")
    parser.add_argument("--model", type=str, default="doubao-seedream-5-0-260128")
    parser.add_argument("--frame_interval", type=int, default=15)
    parser.add_argument("--max_frames_per_video", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("VOLCENKEY")
    if not api_key:
        print("ERROR: Set VOLCENKEY environment variable")
        return

    client = Ark(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3")

    # Find videos
    video_files = sorted(glob.glob(os.path.join(args.test_dir, "*.mp4")))
    if not video_files:
        print(f"No videos found in {args.test_dir}")
        return

    print(f"Found {len(video_files)} videos")
    print(f"Model: {args.model}")
    os.makedirs(args.output_dir, exist_ok=True)

    results_meta = []

    for vid_idx, video_path in enumerate(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        short_name = video_name[:80] + "..." if len(video_name) > 80 else video_name
        prompt = VIDEO_PROMPTS.get(video_name, "")

        if not prompt:
            print(f"\n[{vid_idx+1}/{len(video_files)}] {short_name}: no prompt, skipping")
            continue

        print(f"\n[{vid_idx+1}/{len(video_files)}] {short_name}")
        print(f"  Prompt: \"{prompt}\"")

        video_output_dir = os.path.join(args.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        frames = extract_frames_cv2(video_path, args.frame_interval, args.max_frames_per_video)
        if not frames:
            print("  No frames extracted, skipping")
            continue

        print(f"  Extracted {len(frames)} frames")

        video_results = []
        for fi, (frame_idx, input_image) in enumerate(frames):
            print(f"  Frame {fi+1}/{len(frames)} (#{frame_idx})...", end=" ", flush=True)

            pred_image = doubao_predict(
                client, args.model, input_image, prompt, seed=args.seed + fi
            )

            input_path = os.path.join(video_output_dir, f"frame_{frame_idx:04d}_input.png")
            pred_path = os.path.join(video_output_dir, f"frame_{frame_idx:04d}_predicted.png")

            input_image.save(input_path)
            if pred_image:
                pred_image.save(pred_path)
                print("done")
            else:
                # Save a placeholder
                placeholder = Image.new("RGB", input_image.size, (128, 0, 0))
                placeholder.save(pred_path)
                print("FAILED (placeholder saved)")

            video_results.append({
                "frame_idx": frame_idx,
                "input_path": os.path.relpath(input_path, args.output_dir),
                "pred_path": os.path.relpath(pred_path, args.output_dir),
            })

            # Rate limiting: small sleep between API calls
            time.sleep(0.5)

        results_meta.append({
            "video_name": video_name,
            "short_name": short_name,
            "prompt": prompt,
            "frames": video_results,
        })

    generate_html(results_meta, args.output_dir)
    print(f"\n{'='*60}")
    print(f"All done! Results saved to: {args.output_dir}")
    print(f"Open {os.path.join(args.output_dir, 'index.html')} in a browser")
    print(f"Or run: python -m http.server 8890 --directory {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
