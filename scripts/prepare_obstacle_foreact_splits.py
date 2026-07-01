#!/usr/bin/env python3
"""Prepare left-first/right-first obstacle splits for ForeAct training.

The split is based on which arm first exceeds a small motion threshold in
`observation.state`. The output filter JSON uses the dataloader's per-dataset
format:

    {"foreact_obstacle": [...], "aloha_banana_obstacle_gripper_binary": [...]}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_ROOTS = [
    Path("datasets/foreact_obstacle"),
    Path("datasets/aloha_banana_obstacle_gripper_binary"),
]
OUT_ROOT = Path("datasets/obstacle_foreact_split")
LEFT_DATA_ROOT = Path("datasets/obstacle_foreact_left_first")
RIGHT_DATA_ROOT = Path("datasets/obstacle_foreact_right_first")
MOTION_THRESHOLD = 0.05
TIE_BREAK_WINDOW = 90


def load_episodes(root: Path) -> list[dict]:
    episodes = []
    with (root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def classify_episode(parquet_path: Path) -> tuple[str, int | None, int | None, float, float]:
    df = pd.read_parquet(parquet_path, columns=["observation.state"])
    states = np.asarray([np.asarray(x, dtype=np.float32) for x in df["observation.state"].to_numpy()])

    left = states[:, :7]
    right = states[:, 7:14]
    left_delta = np.linalg.norm(left - left[0], axis=1)
    right_delta = np.linalg.norm(right - right[0], axis=1)

    left_hits = np.flatnonzero(left_delta >= MOTION_THRESHOLD)
    right_hits = np.flatnonzero(right_delta >= MOTION_THRESHOLD)
    left_first = int(left_hits[0]) if left_hits.size else None
    right_first = int(right_hits[0]) if right_hits.size else None

    window = min(len(states), TIE_BREAK_WINDOW)
    left_max = float(left_delta[:window].max())
    right_max = float(right_delta[:window].max())

    if left_first is None and right_first is None:
        side = "unknown"
    elif right_first is None or (left_first is not None and left_first < right_first):
        side = "left_first"
    elif left_first is None or right_first < left_first:
        side = "right_first"
    else:
        side = "left_first" if left_max >= right_max else "right_first"

    return side, left_first, right_first, left_max, right_max


def contiguous_segments(rows: list[dict]) -> list[dict]:
    segments = []
    prev = None
    start = None
    last = None
    for row in rows:
        side = row["side"]
        if side != prev:
            if prev is not None:
                segments.append({"start": start, "end": last, "side": prev})
            start = row["episode_index"]
            prev = side
        last = row["episode_index"]
    if prev is not None:
        segments.append({"start": start, "end": last, "side": prev})
    return segments


def ensure_symlink(link: Path, target: Path) -> None:
    resolved_target = (link.parent / target).resolve()
    if link.is_symlink() and link.resolve() == resolved_target:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"{link} already exists and does not point to {target}")
    link.symlink_to(target)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LEFT_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RIGHT_DATA_ROOT.mkdir(parents=True, exist_ok=True)

    filters = {"left_first": {}, "right_first": {}}
    summary = {}

    for root in DATASET_ROOTS:
        repo_id = root.name
        rows = []
        for ep in load_episodes(root):
            episode_index = int(ep["episode_index"])
            parquet_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
            side, left_first, right_first, left_max, right_max = classify_episode(parquet_path)
            row = {
                "episode_index": episode_index,
                "source_episode_index": int(ep.get("source_episode_index", episode_index)),
                "side": side,
                "left_first_frame": left_first,
                "right_first_frame": right_first,
                "left_max_motion_first_90": round(left_max, 6),
                "right_max_motion_first_90": round(right_max, 6),
                "length": int(ep["length"]),
            }
            rows.append(row)
            if side in filters:
                filters[side].setdefault(repo_id, []).append(episode_index)

        summary[repo_id] = {
            "num_episodes": len(rows),
            "counts": {side: len(filters[side].get(repo_id, [])) for side in filters},
            "segments": contiguous_segments(rows),
        }
        (OUT_ROOT / f"{repo_id}_side_classification.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )

        ensure_symlink(LEFT_DATA_ROOT / repo_id, Path("..") / repo_id)
        ensure_symlink(RIGHT_DATA_ROOT / repo_id, Path("..") / repo_id)

    for side, mapping in filters.items():
        for repo_id in mapping:
            mapping[repo_id] = sorted(mapping[repo_id])
        (OUT_ROOT / f"{side}_filtered_episodes.json").write_text(
            json.dumps(mapping, indent=2), encoding="utf-8"
        )

    (OUT_ROOT / "classification_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
