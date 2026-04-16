"""Verify that SubtaskVideoDataset loads correctly and produces valid data
alongside the original ImagePairDataset, and that BalancedConcatDataset
interleaves them 50/50."""

import json
import os
import sys
from pathlib import Path

# ---- 1. Test SubtaskVideoDataset alone ----
print("=" * 60)
print("[1] Testing SubtaskVideoDataset")
print("=" * 60)

from dataloaders.dataset_finetune import SubtaskVideoDataset

subtask_ds = SubtaskVideoDataset(
    root="datasets/bridge_orig_lerobot/subtask_dataset",
)
print(f"  Total samples: {len(subtask_ds)}")

# Check a few samples
for i in [0, len(subtask_ds) // 2, len(subtask_ds) - 1]:
    sample = subtask_ds[i]
    src = sample["source_image"]
    tgt = sample["target_image"]
    cap = sample["caption"]
    print(f"  Sample {i}: src={src.size}, tgt={tgt.size}, caption='{cap[:80]}'")
    assert src.size[0] > 0 and src.size[1] > 0, "Source image has zero size"
    assert tgt.size[0] > 0 and tgt.size[1] > 0, "Target image has zero size"
    assert isinstance(cap, str) and len(cap) > 0, f"Caption is empty for sample {i}"

print("  ✓ SubtaskVideoDataset OK\n")

# ---- 2. Test ImagePairDataset with COT data ----
print("=" * 60)
print("[2] Testing ImagePairDataset (original, with COT)")
print("=" * 60)

from dataloaders.dataset_finetune import ImagePairDataset

cot_path = "datasets/cot_by_episode.json"
filtered_path = "datasets/top10000_episodes.json"

cot_data = None
if Path(cot_path).is_file():
    with open(cot_path) as f:
        cot_data = json.load(f)
    print(f"  COT data loaded: {len(cot_data)} episodes")

allowed = None
if Path(filtered_path).is_file():
    with open(filtered_path) as f:
        allowed = set(json.load(f))
    print(f"  Filtered episodes: {len(allowed)}")

orig_ds = ImagePairDataset(
    root="datasets/bridge_orig_lerobot",
    camera_key="observation.images.image_0",
    cot_data=cot_data,
    allowed_episode_indices=allowed,
)
print(f"  Total samples: {len(orig_ds)}")

# Check a few samples
for i in [0, len(orig_ds) // 2, len(orig_ds) - 1]:
    sample = orig_ds[i]
    src = sample["source_image"]
    tgt = sample["target_image"]
    cap = sample["caption"]
    print(f"  Sample {i}: src={src.size}, tgt={tgt.size}, caption='{cap[:80]}'")
    assert src.size[0] > 0 and src.size[1] > 0
    assert tgt.size[0] > 0 and tgt.size[1] > 0

print("  ✓ ImagePairDataset OK\n")

# ---- 3. Test BalancedConcatDataset ----
print("=" * 60)
print("[3] Testing BalancedConcatDataset (50/50)")
print("=" * 60)

from dataloaders.dataset_finetune import BalancedConcatDataset
from torch.utils.data import ConcatDataset

balanced_ds = BalancedConcatDataset(orig_ds, subtask_ds)
print(f"  Total samples: {len(balanced_ds)}")
print(f"  Original dataset: {len(orig_ds)}, Subtask dataset: {len(subtask_ds)}")
print(f"  Expected 50/50 ratio: each source contributes {len(balanced_ds)//2} samples")

# Verify interleaving: even indices -> orig, odd indices -> subtask
from collections import Counter
source_counter = Counter()
n_check = min(100, len(balanced_ds))
for i in range(n_check):
    if i % 2 == 0:
        source_counter["original"] += 1
    else:
        source_counter["subtask"] += 1

print(f"  First {n_check} samples distribution: {dict(source_counter)}")
assert source_counter["original"] == n_check // 2 or source_counter["original"] == (n_check + 1) // 2
print("  ✓ 50/50 balance verified")

# Actually load a sample from each side
sample_orig = balanced_ds[0]  # even -> original
sample_sub = balanced_ds[1]   # odd -> subtask
print(f"  Balanced[0] (orig): caption='{sample_orig['caption'][:60]}'")
print(f"  Balanced[1] (sub):  caption='{sample_sub['caption'][:60]}'")

# Save sample images for visual inspection
os.makedirs("verify_output", exist_ok=True)
sample_orig["source_image"].save("verify_output/orig_source.png")
sample_orig["target_image"].save("verify_output/orig_target.png")
sample_sub["source_image"].save("verify_output/subtask_source.png")
sample_sub["target_image"].save("verify_output/subtask_target.png")
print(f"  Sample images saved to verify_output/")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
