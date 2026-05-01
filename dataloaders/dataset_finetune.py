import glob
import json
import math
import os
import re
import time
import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import v2
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from utils.video_utils import decode_video_frames, get_safe_default_codec


class ImagePairDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        camera_key: str = "observation.images.head_left_rgb",
        tolerance_s: float = 1e-4,
        video_backend: Optional[str] = None,
        cot_data: Optional[Dict[str, Any]] = None,
        allowed_episode_indices: Optional[set] = None,
        target_frame_offset: int = 0,
        source_frame_stride: int = 0,
        min_source_frame_index: int = 0,
        trajectory_motion_filter: bool = False,
        trajectory_key: str = "observation.state",
        trajectory_motion_start_threshold: float = 0.05,
        trajectory_motion_start_padding: int = 0,
        min_trajectory_delta: float = 0.0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.camera_key = camera_key
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        print(f"video_backend: {self.video_backend}")

        # Load metadata
        info_path = self.root / "meta" / "info.json"
        episodes_path = self.root / "meta" / "episodes.jsonl"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing info.json at: {info_path}")
        if not episodes_path.is_file():
            raise FileNotFoundError(f"Missing episodes.jsonl at: {episodes_path}")

        with open(info_path, "r", encoding="utf-8") as f:
            self.info: Dict[str, Any] = json.load(f)

        self.fps: int = int(self.info.get("fps", 30))
        self.chunks_size: int = int(self.info.get("chunks_size", 1000))
        self.video_path_template: Optional[str] = self.info.get("video_path")
        self.data_path_template: Optional[str] = self.info.get("data_path")
        self.features: Dict[str, Dict[str, Any]] = self.info.get("features", {})
        self._cot_data = cot_data  # keyed by str(episode_index)
        self._allowed_episode_indices = allowed_episode_indices  # set of ints, None = all
        self._target_frame_offset = target_frame_offset
        self._source_frame_stride = int(source_frame_stride) if source_frame_stride else 0
        self._min_source_frame_index = max(0, int(min_source_frame_index))
        self._trajectory_motion_filter = bool(trajectory_motion_filter)
        self._trajectory_key = trajectory_key
        self._trajectory_motion_start_threshold = float(trajectory_motion_start_threshold)
        self._trajectory_motion_start_padding = max(0, int(trajectory_motion_start_padding))
        self._min_trajectory_delta = max(0.0, float(min_trajectory_delta))
        self._trajectory_cache: Dict[int, Optional[np.ndarray]] = {}
        self._motion_start_by_episode: Dict[int, int] = {}

        # Validate camera key
        if self.camera_key not in self.features:
            raise KeyError(
                f"Camera key '{self.camera_key}' not found in features. Available: {list(self.features.keys())}"
            )
        if self.features[self.camera_key].get("dtype") != "video":
            raise ValueError(
                f"Camera key '{self.camera_key}' is not stored as video. dtype={self.features[self.camera_key].get('dtype')}"
            )

        # Load episodes list
        self._episodes: List[Dict[str, Any]] = []
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ep = json.loads(line)
                # Ensure required fields exist
                if "episode_index" not in ep or "length" not in ep:
                    continue
                if ep.get("tasks") == [""]:
                    continue
                self._episodes.append(ep)

        if self._allowed_episode_indices is not None:
            self._episodes = [
                ep for ep in self._episodes
                if int(ep["episode_index"]) in self._allowed_episode_indices
            ]
        if len(self._episodes) == 0:
            raise RuntimeError("No episodes loaded. Check 'episodes' filter or dataset contents.")

        # Build global index -> (episode_pos, source_fi, target_fi, subtask_name)
        self._index: List[Tuple[int, int, int, str]] = []
        for i, ep in enumerate(self._episodes):
            length = int(ep["length"])  # number of frames
            episode_index = int(ep["episode_index"])
            ep_key = str(episode_index)

            # Get episode-level task description (used for all modes)
            ep_caption = self._episode_caption(ep)

            if self._target_frame_offset > 0:
                # Fixed-offset mode: target = source + offset, every frame is used
                source_stride = self._source_frame_stride or self.fps
                min_fi = self._episode_min_source_frame(episode_index)
                traj = self._trajectory_for_episode(episode_index) if self._min_trajectory_delta > 0 else None
                for fi in range(0, length, source_stride):
                    if fi < min_fi:
                        continue
                    target_fi = fi + self._target_frame_offset
                    if target_fi >= length:
                        continue  # strict: skip when target frame doesn't exist
                    if (
                        traj is not None
                        and target_fi < len(traj)
                        and self._trajectory_delta(traj, fi, target_fi) < self._min_trajectory_delta
                    ):
                        continue
                    self._index.append((i, fi, target_fi, ep_caption))
            elif self._cot_data and ep_key in self._cot_data:
                cot = self._cot_data[ep_key]
                step_to_si: Dict[int, int] = {
                    int(k): int(v) for k, v in cot["step_to_subtask_index"].items()
                }
                all_subtasks_cot: List[str] = cot["all_subtasks"]

                # Compute the last frame index for each subtask
                subtask_last_frame: Dict[int, int] = {}
                for step, si in step_to_si.items():
                    if si not in subtask_last_frame or step > subtask_last_frame[si]:
                        subtask_last_frame[si] = step

                for fi in range(length):
                    if fi % self.fps != 0:
                        continue
                    si = step_to_si.get(fi)
                    if si is not None and si < len(all_subtasks_cot):
                        target_fi = subtask_last_frame.get(si, length - 1)
                        subtask_name = all_subtasks_cot[si]
                    else:
                        target_fi = length - 1
                        subtask_name = ""
                    self._index.append((i, fi, target_fi, subtask_name))
            else:
                # Fallback: target is always the last frame of the episode
                self._index.extend(
                    [(i, fi, length - 1, "") for fi in range(length) if fi % self.fps == 0]
                )

    def __len__(self) -> int:
        return len(self._index)

    @staticmethod
    def _episode_caption(ep: Dict[str, Any]) -> str:
        tasks = ep.get("tasks", None)
        if isinstance(tasks, list):
            caption = "; ".join([str(t) for t in tasks]) if len(tasks) > 0 else ""
        elif tasks is not None:
            caption = str(tasks)
        else:
            caption = str(ep.get("task", ""))
        return caption

    def _episode_chunk(self, episode_index: int) -> int:
        return episode_index // self.chunks_size

    def _episode_min_source_frame(self, episode_index: int) -> int:
        min_fi = self._min_source_frame_index
        if not self._trajectory_motion_filter:
            return min_fi
        if episode_index not in self._motion_start_by_episode:
            self._motion_start_by_episode[episode_index] = self._find_motion_start_frame(episode_index)
        motion_start = max(0, self._motion_start_by_episode[episode_index] - self._trajectory_motion_start_padding)
        return max(min_fi, motion_start)

    def _video_path_for_episode(self, episode_index: int) -> Path:
        if not self.video_path_template:
            chunk = self._episode_chunk(episode_index)
            rel = f"videos/chunk-{chunk:03d}/{self.camera_key}/episode_{episode_index:06d}.mp4"
            return self.root / rel
        fpath = self.video_path_template.format(
            episode_chunk=self._episode_chunk(episode_index),
            video_key=self.camera_key,
            episode_index=episode_index,
        )
        return self.root / fpath

    def _data_path_for_episode(self, episode_index: int) -> Path:
        if not self.data_path_template:
            chunk = self._episode_chunk(episode_index)
            rel = f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
            return self.root / rel
        fpath = self.data_path_template.format(
            episode_chunk=self._episode_chunk(episode_index),
            episode_index=episode_index,
        )
        return self.root / fpath

    def _trajectory_for_episode(self, episode_index: int) -> Optional[np.ndarray]:
        if episode_index in self._trajectory_cache:
            return self._trajectory_cache[episode_index]

        data_path = self._data_path_for_episode(episode_index)
        if not data_path.is_file():
            self._trajectory_cache[episode_index] = None
            return None

        try:
            import pandas as pd

            df = pd.read_parquet(data_path, columns=[self._trajectory_key])
            values = np.asarray([np.asarray(v, dtype=np.float32) for v in df[self._trajectory_key].to_numpy()])
            if values.ndim != 2 or len(values) == 0:
                values = None
        except Exception as exc:
            print(f"Failed to load trajectory {self._trajectory_key} from {data_path}: {exc}")
            values = None

        self._trajectory_cache[episode_index] = values
        return values

    @staticmethod
    def _trajectory_delta(traj: np.ndarray, source_fi: int, target_fi: int) -> float:
        if source_fi >= len(traj) or target_fi >= len(traj):
            return math.inf
        return float(np.linalg.norm(traj[target_fi] - traj[source_fi]))

    def _find_motion_start_frame(self, episode_index: int) -> int:
        traj = self._trajectory_for_episode(episode_index)
        if traj is None or len(traj) <= 1:
            return 0

        deltas = np.linalg.norm(traj - traj[0], axis=1)
        moving = np.flatnonzero(deltas >= self._trajectory_motion_start_threshold)
        if moving.size == 0:
            return 0
        return int(moving[0])

    # ---------- core decoding ----------
    def _decode_pair(
        self, video_path: Path, ts_source: float, ts_target: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frames = decode_video_frames(
            str(video_path),
            [ts_source, ts_target],
            self.tolerance_s,
            self.video_backend,
        )
        source, target = frames[0], frames[1]

        return source, target

    _to_pil = v2.ToPILImage()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_pos, local_fi, target_fi, subtask_name = self._index[idx]
        ep = self._episodes[ep_pos]
        episode_index: int = int(ep["episode_index"])

        ts_source = local_fi / float(self.fps)
        ts_target = target_fi / float(self.fps)

        video_path = self._video_path_for_episode(episode_index)
        source_img, target_img = self._decode_pair(video_path, ts_source, ts_target)

        caption = subtask_name if subtask_name else self._episode_caption(ep)

        return {
            "source_image": self._to_pil(source_img),
            "target_image": self._to_pil(target_img),
            "caption": caption,
        }


class SubtaskVideoDataset(Dataset):
    """Dataset for the subtask_dataset format with per-episode annotation JSONs
    and standalone video files organized by task name."""

    def __init__(
        self,
        root: str | Path,
        fps: int = 5,
        tolerance_s: float = 1e-4,
        video_backend: Optional[str] = None,
        target_frame_offset: int = 0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.fps = fps
        self.tolerance_s = tolerance_s
        self._target_frame_offset = target_frame_offset
        self.video_backend = video_backend if video_backend else get_safe_default_codec()

        ann_root = self.root / "annotations"
        vid_root = self.root / "videos"
        if not ann_root.is_dir():
            raise FileNotFoundError(f"Missing annotations dir: {ann_root}")
        if not vid_root.is_dir():
            raise FileNotFoundError(f"Missing videos dir: {vid_root}")

        # Build index: (video_path, source_fi, target_fi, subtask_description)
        self._index: List[Tuple[str, int, int, str]] = []

        for task_name in sorted(os.listdir(str(ann_root))):
            task_ann_dir = ann_root / task_name
            task_vid_dir = vid_root / task_name
            if not task_ann_dir.is_dir():
                continue

            # Build episode_idx -> video_path mapping
            vid_map: Dict[int, str] = {}
            for vp in sorted(glob.glob(str(task_vid_dir / "*.mp4"))):
                m = re.search(r'episode_(\d+)_', os.path.basename(vp))
                if m:
                    vid_map[int(m.group(1))] = vp

            # Process each annotation
            for ann_path in sorted(glob.glob(str(task_ann_dir / "*.json"))):
                with open(ann_path, "r", encoding="utf-8") as f:
                    ann = json.load(f)

                ep_idx = int(ann["episode_idx"])
                if ep_idx not in vid_map:
                    continue
                video_path = vid_map[ep_idx]
                total_frames = int(ann["total_frames"])
                subtask_segments = ann.get("subtask_segments", [])
                frame_labels = ann.get("frame_labels", [])

                # Build label -> segment mapping and ordered segment list
                label_to_seg: Dict[int, Dict] = {}
                for seg in subtask_segments:
                    label_to_seg[int(seg["subtask_label"])] = seg

                # Order segments by start_frame for next-subtask lookup
                ordered_segs = sorted(subtask_segments, key=lambda s: int(s["start_frame"]))
                # Map label -> index in ordered list
                label_to_order: Dict[int, int] = {
                    int(s["subtask_label"]): i for i, s in enumerate(ordered_segs)
                }
                last_seg_order = len(ordered_segs) - 1

                # Get task-level description for offset mode
                task_desc = ann.get("task_description", "")

                if self._target_frame_offset > 0:
                    # Fixed-offset mode
                    for fi in range(total_frames):
                        if fi % self.fps != 0:
                            continue
                        target_fi = fi + self._target_frame_offset
                        if target_fi >= total_frames:
                            continue
                        self._index.append((video_path, fi, target_fi, task_desc))
                else:
                    for fi in range(total_frames):
                        if fi % self.fps != 0:
                            continue
                        # Find which subtask this frame belongs to
                        if fi < len(frame_labels):
                            lbl = int(frame_labels[fi])
                            seg = label_to_seg.get(lbl)
                            if seg:
                                seg_end = int(seg["end_frame"])
                                seg_order = label_to_order.get(lbl, -1)
                                is_last_subtask = (seg_order == last_seg_order)
                                near_end = (fi >= seg_end - 2)  # last 2 frames of subtask

                                if near_end and is_last_subtask:
                                    # Last 2 frames of final subtask -> skip
                                    continue
                                elif near_end and not is_last_subtask:
                                    # Last 2 frames of non-final subtask ->
                                    # use next subtask's end as target
                                    next_seg = ordered_segs[seg_order + 1]
                                    target_fi = max(0, int(next_seg["end_frame"]) - 1)
                                    desc = next_seg.get("subtask_description", "")
                                else:
                                    target_fi = max(0, seg_end - 1)
                                    desc = seg.get("subtask_description", "")
                            else:
                                target_fi = total_frames - 1
                                desc = ""
                        else:
                            target_fi = total_frames - 1
                            desc = ""
                        self._index.append((video_path, fi, target_fi, desc))

        if len(self._index) == 0:
            raise RuntimeError("No samples loaded from SubtaskVideoDataset.")
        print(f"SubtaskVideoDataset: loaded {len(self._index)} samples from {self.root}")

    def __len__(self) -> int:
        return len(self._index)

    _to_pil = v2.ToPILImage()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path, source_fi, target_fi, desc = self._index[idx]
        ts_source = source_fi / float(self.fps)
        ts_target = target_fi / float(self.fps)

        frames = decode_video_frames(
            video_path,
            [ts_source, ts_target],
            self.tolerance_s,
            self.video_backend,
        )
        source_img, target_img = frames[0], frames[1]

        return {
            "source_image": self._to_pil(source_img),
            "target_image": self._to_pil(target_img),
            "caption": desc,
        }


class CustomVideoDataset(Dataset):
    """Dataset for custom video-based tasks with fixed-offset frame pairs.

    Expected directory layout:
        root/
            task_1/
                task_info.json      # {"task_name": "put apple on plate"}
                episode_0.mp4
                episode_1.mp4
                ...
            task_2/
                task_info.json
                ...

    Each video is an episode. Source frames are sampled every `fps` frames.
    Target frame = source frame + target_frame_offset (clamped to video length).
    """

    def __init__(
        self,
        root: str | Path,
        fps: int = 5,
        target_frame_offset: int = 6,
        source_frame_stride: int = 1,
        tolerance_s: float = 1e-4,
        video_backend: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.fps = fps
        self._target_frame_offset = target_frame_offset
        self._source_frame_stride = max(1, int(source_frame_stride))
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend if video_backend else get_safe_default_codec()

        # Build index: (video_path, source_fi, target_fi, task_name)
        self._index: List[Tuple[str, int, int, str]] = []

        for task_dir in sorted(self.root.iterdir()):
            if not task_dir.is_dir():
                continue

            # Read task name from task_info.json
            task_info_path = task_dir / "task_info.json"
            task_name = ""
            if task_info_path.is_file():
                with open(task_info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                task_name = str(info.get("task_name", ""))

            # Find all mp4 videos in this task folder
            video_files = sorted(glob.glob(str(task_dir / "*.mp4")))
            for video_path in video_files:
                # Probe video to get total number of frames
                total_frames = self._get_video_frame_count(video_path)
                if total_frames <= 1:
                    continue

                for fi in range(0, total_frames, self._source_frame_stride):
                    target_fi = fi + self._target_frame_offset
                    if target_fi >= total_frames:
                        continue
                    self._index.append((video_path, fi, target_fi, task_name))

        if len(self._index) == 0:
            raise RuntimeError(f"No samples loaded from CustomVideoDataset at {self.root}")
        print(f"CustomVideoDataset: loaded {len(self._index)} samples from {self.root}")

    @staticmethod
    def _get_video_frame_count(video_path: str) -> int:
        """Get total frame count by probing with torchvision or ffprobe."""
        try:
            from torchvision.io import VideoReader
            reader = VideoReader(video_path, "video")
            metadata = reader.get_metadata()
            duration = metadata["video"]["duration"][0]
            fps = metadata["video"]["fps"][0]
            return int(duration * fps)
        except Exception:
            pass
        # Fallback: ffprobe
        try:
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-count_frames", "-show_entries", "stream=nb_read_frames",
                 "-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=30,
            )
            return int(result.stdout.strip())
        except Exception:
            return 0

    def __len__(self) -> int:
        return len(self._index)

    _to_pil = v2.ToPILImage()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path, source_fi, target_fi, task_name = self._index[idx]
        ts_source = source_fi / float(self.fps)
        ts_target = target_fi / float(self.fps)

        frames = decode_video_frames(
            video_path,
            [ts_source, ts_target],
            self.tolerance_s,
            self.video_backend,
        )
        source_img, target_img = frames[0], frames[1]

        return {
            "source_image": self._to_pil(source_img),
            "target_image": self._to_pil(target_img),
            "caption": task_name,
        }


class BalancedConcatDataset(Dataset):
    """Wraps two datasets and ensures 50/50 sampling by oversampling the smaller one.

    Each side's access order for one epoch is a per-epoch sampling schedule of length
    ``_half``: ``floor(_half / n)`` random permutations of its indices concatenated
    with a random ``_half % n``-sized subset drawn without replacement. Call
    ``set_epoch(epoch)`` at each epoch start so the "remainder" entries change across
    epochs instead of always hitting the first ``_half % n`` indices.
    """

    def __init__(self, dataset_a: Dataset, dataset_b: Dataset, base_seed: int = 0) -> None:
        super().__init__()
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.len_a = len(dataset_a)
        self.len_b = len(dataset_b)
        # Total length = 2 * max(len_a, len_b) so each contributes 50%
        self._half = max(self.len_a, self.len_b)
        self._base_seed = int(base_seed)
        self._schedule_a: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._schedule_b: torch.Tensor = torch.empty(0, dtype=torch.long)
        self.set_epoch(0)
        print(f"BalancedConcatDataset: dataset_a={self.len_a}, dataset_b={self.len_b}, "
              f"effective_half={self._half}, total={2 * self._half}, base_seed={self._base_seed}")

    def _build_schedule(self, n: int, generator: torch.Generator) -> torch.Tensor:
        full_cycles, remainder = divmod(self._half, n)
        parts = [torch.randperm(n, generator=generator) for _ in range(full_cycles)]
        if remainder > 0:
            parts.append(torch.randperm(n, generator=generator)[:remainder])
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)

    def set_epoch(self, epoch: int) -> None:
        g = torch.Generator()
        g.manual_seed(self._base_seed * 1_000_003 + int(epoch))
        self._schedule_a = self._build_schedule(self.len_a, g)
        self._schedule_b = self._build_schedule(self.len_b, g)

    def __len__(self) -> int:
        return 2 * self._half

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pos = idx // 2
        if idx % 2 == 0:
            return self.dataset_a[int(self._schedule_a[pos].item())]
        else:
            return self.dataset_b[int(self._schedule_b[pos].item())]


def _collate_fn_imagepair(batch, tokenize_func, tokenizer, source_transform, target_transform):
    captions = [example["caption"] for example in batch]
    source_images = [example["source_image"] for example in batch]
    target_images = [example["target_image"] for example in batch]

    rand_probs = torch.rand((len(source_images), 1))
    null_caption_mask = rand_probs < 0.2
    null_image_mask = (rand_probs >= 0.1) & (rand_probs < 0.3)

    captions = [
        caption if not null_caption_mask[i] else ""
        for i, caption in enumerate(captions)
    ]
    source_images = [
        (
            Image.new("RGB", (image.width, image.height))
            if (image is not None and null_image_mask[i])
            else image
        )
        for i, image in enumerate(source_images)
    ]

    sources = [source_transform(image) for image in source_images]
    targets = [target_transform(image) for image in target_images]

    return_dict = {"source": torch.stack(sources), "target": torch.stack(targets)}
    (
        return_dict["input_ids"],
        return_dict["attention_mask"],
    ) = tokenize_func(tokenizer, captions)

    return return_dict


def _load_single_dataset(
    repo_id,
    root,
    camera_key,
    cot_data=None,
    allowed_episode_indices=None,
    target_frame_offset=0,
    source_frame_stride=0,
    min_source_frame_index=0,
    trajectory_motion_filter=False,
    trajectory_key="observation.state",
    trajectory_motion_start_threshold=0.05,
    trajectory_motion_start_padding=0,
    min_trajectory_delta=0.0,
):
    try:
        dataset = ImagePairDataset(
            root=root,
            camera_key=camera_key,
            cot_data=cot_data,
            allowed_episode_indices=allowed_episode_indices,
            target_frame_offset=target_frame_offset,
            source_frame_stride=source_frame_stride,
            min_source_frame_index=min_source_frame_index,
            trajectory_motion_filter=trajectory_motion_filter,
            trajectory_key=trajectory_key,
            trajectory_motion_start_threshold=trajectory_motion_start_threshold,
            trajectory_motion_start_padding=trajectory_motion_start_padding,
            min_trajectory_delta=min_trajectory_delta,
        )
        print(f"✓ Loaded dataset: {repo_id}")
        return repo_id, dataset
    except Exception as e:
        print(f"✗ Failed to load dataset {repo_id}: {e}")
        return repo_id, None


def get_train_datasets(data_args, tokenize_func, tokenizer, base_seed: int = 0):
    train_datasets = {}
    
    # Prepare dataset loading tasks
    dataset_tasks = []
    
    data_path = data_args.data_path
    camera_key = data_args.camera_key
    for dir_name in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, dir_name)):
            dataset_tasks.append((
                dir_name,
                os.path.join(data_path, dir_name),
                camera_key,
            ))
    # Load filtered episode indices if specified
    allowed_episode_indices = None
    filtered_episodes_path = getattr(data_args, "filtered_episodes_path", None)
    if filtered_episodes_path and Path(filtered_episodes_path).is_file():
        print(f"Loading filtered episode list from {filtered_episodes_path} ...")
        with open(filtered_episodes_path, "r", encoding="utf-8") as f:
            allowed_episode_indices = set(json.load(f))
        print(f"Filtering to {len(allowed_episode_indices)} allowed episodes")

    # Load COT subtask data if specified
    cot_data = None
    cot_json_path = getattr(data_args, "cot_json_path", None)
    if cot_json_path and Path(cot_json_path).is_file():
        print(f"Loading COT data from {cot_json_path} ...")
        with open(cot_json_path, "r", encoding="utf-8") as f:
            cot_data = json.load(f)
        print(f"Loaded COT data for {len(cot_data)} episodes")

    # Read target_frame_offset from data_args
    target_frame_offset = getattr(data_args, "target_frame_offset", 0)
    if target_frame_offset > 0:
        print(f"Using fixed target_frame_offset={target_frame_offset} (ignoring COT/subtask logic)")
    source_frame_stride = getattr(data_args, "source_frame_stride", 0)
    if source_frame_stride:
        print(f"Using source_frame_stride={source_frame_stride}")
    min_source_frame_index = getattr(data_args, "min_source_frame_index", 0)
    trajectory_motion_filter = getattr(data_args, "trajectory_motion_filter", False)
    trajectory_key = getattr(data_args, "trajectory_key", "observation.state")
    trajectory_motion_start_threshold = getattr(data_args, "trajectory_motion_start_threshold", 0.05)
    trajectory_motion_start_padding = getattr(data_args, "trajectory_motion_start_padding", 0)
    min_trajectory_delta = getattr(data_args, "min_trajectory_delta", 0.0)
    if min_source_frame_index:
        print(f"Skipping source frames before frame {min_source_frame_index}")
    if trajectory_motion_filter:
        print(
            "Using trajectory_motion_filter: "
            f"key={trajectory_key}, threshold={trajectory_motion_start_threshold}, "
            f"padding={trajectory_motion_start_padding}"
        )
    if min_trajectory_delta:
        print(f"Using min_trajectory_delta={min_trajectory_delta} with key={trajectory_key}")

    # Load datasets in parallel using ThreadPoolExecutor
    print(f"Loading {len(dataset_tasks)} datasets in parallel...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=min(32, len(dataset_tasks))) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                _load_single_dataset,
                repo_id,
                root,
                camera_key,
                cot_data,
                allowed_episode_indices,
                target_frame_offset,
                source_frame_stride,
                min_source_frame_index,
                trajectory_motion_filter,
                trajectory_key,
                trajectory_motion_start_threshold,
                trajectory_motion_start_padding,
                min_trajectory_delta,
            ): (repo_id, root, camera_key)
            for repo_id, root, camera_key in dataset_tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            repo_id, dataset = future.result()
            if dataset is not None:
                train_datasets[repo_id] = dataset
    
    elapsed_time = time.time() - start_time
    print(f"Loaded {len(train_datasets)}/{len(dataset_tasks)} datasets in {elapsed_time:.2f} seconds")

    # ---- Load subtask_dataset (new format) if specified ----
    subtask_dataset = None
    subtask_data_path = getattr(data_args, "subtask_data_path", None)
    if subtask_data_path and Path(subtask_data_path).is_dir():
        print(f"Loading SubtaskVideoDataset from {subtask_data_path} ...")
        try:
            subtask_dataset = SubtaskVideoDataset(root=subtask_data_path, target_frame_offset=target_frame_offset)
            print(f"✓ Loaded SubtaskVideoDataset: {len(subtask_dataset)} samples")
        except Exception as e:
            print(f"✗ Failed to load SubtaskVideoDataset: {e}")

    source_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
        ]
    )
    
    target_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
        ]
    )

    # Use a custom collate function for the torch Dataset
    collate_fn = partial(
        _collate_fn_imagepair,
        tokenize_func=tokenize_func,
        tokenizer=tokenizer,
        source_transform=source_transform,
        target_transform=target_transform,
    )

    # ---- Load custom dataset if specified ----
    custom_dataset = None
    custom_data_path = getattr(data_args, "custom_data_path", None)
    if custom_data_path and Path(custom_data_path).is_dir():
        print(f"Loading CustomVideoDataset from {custom_data_path} ...")
        try:
            custom_dataset = CustomVideoDataset(
                root=custom_data_path,
                target_frame_offset=target_frame_offset,
                source_frame_stride=source_frame_stride or 1,
            )
            print(f"✓ Loaded CustomVideoDataset: {len(custom_dataset)} samples")
        except Exception as e:
            print(f"✗ Failed to load CustomVideoDataset: {e}")

    # ---- Assemble final dataset ----
    all_datasets = list(train_datasets.values())
    if subtask_dataset is not None:
        all_datasets.append(subtask_dataset)

    balance_datasets = getattr(data_args, "balance_datasets", False)

    if balance_datasets and custom_dataset is not None and len(all_datasets) > 0:
        # Use BalancedConcatDataset for 1:1 ratio between lerobot data and custom data
        base_dataset = ConcatDataset(all_datasets) if len(all_datasets) > 1 else all_datasets[0]
        train_dataset = BalancedConcatDataset(base_dataset, custom_dataset, base_seed=base_seed)
        print(f"Final training dataset (balanced): {len(train_dataset)} samples "
              f"(base={len(base_dataset)}, custom={len(custom_dataset)})")
    else:
        if custom_dataset is not None:
            all_datasets.append(custom_dataset)
        if len(all_datasets) == 0:
            raise RuntimeError("No datasets loaded. Check data_path and other dataset configs.")
        train_dataset = ConcatDataset(all_datasets)
        print(f"Final training dataset: {len(train_dataset)} samples from {len(all_datasets)} dataset(s)")

    return train_dataset, collate_fn
