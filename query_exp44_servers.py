#!/usr/bin/env python3
"""
Send all images in origin/ to all 5 exp44 foreact servers simultaneously.

Usage:
    python query_exp44_servers.py
    python query_exp44_servers.py --host localhost
"""

import argparse
import asyncio
import functools
import logging
import os
from pathlib import Path

import msgpack
import numpy as np
import websockets.asyncio.client
from PIL import Image, ImageDraw

# ── Config ────────────────────────────────────────────────────────────────────

ORIGIN_DIR = "origin"
TASK = "Put the geometry into the corresponding slot."
OUTPUT_DIR = "results_exp44"

SERVERS = [
    {"step": 400,  "port": 5100},
    {"step": 800,  "port": 5101},
    {"step": 1200, "port": 5102},
    {"step": 1600, "port": 5103},
    {"step": 2000, "port": 5104},
]

# ── msgpack numpy support ─────────────────────────────────────────────────────

def _pack_array(obj):
    if isinstance(obj, np.ndarray) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)

# ── resize_with_pad (replicates VLA preprocessing) ───────────────────────────

def _resize_with_pad(image: Image.Image, height: int, width: int) -> Image.Image:
    """Keep aspect ratio, scale down, pad with black to (width, height)."""
    cur_width, cur_height = image.size
    ratio = max(cur_width / width, cur_height / height)
    resized_width = int(cur_width / ratio)
    resized_height = int(cur_height / ratio)
    resized = image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    canvas = Image.new(image.mode, (width, height), 0)
    pad_w = (width - resized_width) // 2
    pad_h = (height - resized_height) // 2
    canvas.paste(resized, (pad_w, pad_h))
    return canvas

# ── Per-server async query ────────────────────────────────────────────────────

async def query_server(host, port, step, image_arr, results):
    uri = f"ws://{host}:{port}"
    try:
        async with websockets.asyncio.client.connect(uri, compression=None, max_size=None) as ws:
            packer = _Packer()
            _unpackb(await ws.recv())  # consume ready metadata
            req = {
                "type": "predict",
                "request_id": f"ckpt{step}",
                "image": image_arr,
                "task_description": TASK,
            }
            await ws.send(packer.pack(req))
            resp = _unpackb(await ws.recv())
            if resp.get("status") == "ok":
                subgoal = resp["data"]["subgoal_image"]
                latency = resp["data"].get("latency", 0.0)
                logging.info(f"  [ckpt-{step}] OK in {latency:.2f}s")
                results[step] = subgoal
            else:
                logging.error(f"  [ckpt-{step}] Server error: {resp.get('error')}")
                results[step] = None
    except Exception as e:
        logging.error(f"  [ckpt-{step}] Failed: {e}")
        results[step] = None


async def query_all(host, image_arr):
    results = {}
    await asyncio.gather(*[
        query_server(host, s["port"], s["step"], image_arr, results)
        for s in SERVERS
    ])
    return results

# ── Comparison image ──────────────────────────────────────────────────────────

def make_comparison(input_image, results, steps):
    w, h = input_image.size
    label_h = 28
    n = 1 + len(steps)
    canvas = Image.new("RGB", (w * n, h + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    canvas.paste(input_image, (0, label_h))
    draw.text((4, 6), "Input", fill=(255, 255, 255))

    for i, step in enumerate(steps):
        x = (i + 1) * w
        arr = results.get(step)
        if arr is not None:
            img = Image.fromarray(arr.astype(np.uint8)).resize((w, h))
            canvas.paste(img, (x, label_h))
            draw.text((x + 4, 6), f"ckpt-{step}", fill=(180, 255, 180))
        else:
            draw.rectangle([x, label_h, x + w - 1, h + label_h - 1], fill=(60, 0, 0))
            draw.text((x + 4, 6), f"ckpt-{step} FAIL", fill=(255, 80, 80))

    return canvas

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    img_paths = sorted(Path(ORIGIN_DIR).glob("*.png")) + \
                sorted(Path(ORIGIN_DIR).glob("*.jpg")) + \
                sorted(Path(ORIGIN_DIR).glob("*.jpeg"))
    if not img_paths:
        print(f"No images found in '{ORIGIN_DIR}/'")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    steps = [s["step"] for s in SERVERS]

    logging.info(f"Task: {TASK}")
    logging.info(f"Found {len(img_paths)} images in '{ORIGIN_DIR}/'")
    logging.info(f"Output dir: {OUTPUT_DIR}/")

    for img_path in img_paths:
        stem = img_path.stem
        logging.info(f"\n[{stem}] Querying 5 servers ...")

        input_image = Image.open(img_path).convert("RGB")
        image_arr = np.array(input_image, dtype=np.uint8)

        results = asyncio.run(query_all(args.host, image_arr))

        # Save individual results
        for step in steps:
            arr = results.get(step)
            if arr is not None:
                out = os.path.join(OUTPUT_DIR, f"{stem}_ckpt{step}.png")
                Image.fromarray(arr.astype(np.uint8)).save(out)

        # Save comparison
        comparison = make_comparison(input_image, results, steps)
        cmp_path = os.path.join(OUTPUT_DIR, f"{stem}_comparison.png")
        comparison.save(cmp_path)

        succeeded = [s for s in steps if results.get(s) is not None]
        failed    = [s for s in steps if results.get(s) is None]
        logging.info(f"[{stem}] Saved {len(succeeded)}/5 results -> {cmp_path}"
                     + (f"  FAILED: {failed}" if failed else ""))

    print(f"\nAll done. Results in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
