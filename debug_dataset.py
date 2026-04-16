"""
Debug script to verify:
1. ImagePairDataset returns correct (source, target) frame pairs
2. filtered_episodes_path filtering is working correctly
3. COT subtask target frames are correctly set

Usage:
    conda run -n foreact python debug_dataset.py
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataloaders.dataset_finetune import ImagePairDataset

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
DATA_ROOT          = "datasets/bridge_orig_lerobot"
CAMERA_KEY         = "observation.images.image_0"
FILTERED_JSON      = "datasets/top10000_episodes.json"
COT_JSON           = "datasets/cot_by_episode.json"
OUT_IMG            = "debug_dataset_pairs.png"
N_SAMPLES          = 6     # how many pairs to visualize (pick diverse episodes)

# -----------------------------------------------------------------------
# Helper: load datasets under different modes
# -----------------------------------------------------------------------

def load_dataset(with_filter=False, with_cot=False, max_ep_override=None):
    """Load dataset from single root, with optional filter / COT."""
    allowed = None
    if with_filter:
        print(f"  Loading filter list from {FILTERED_JSON}")
        with open(FILTERED_JSON) as f:
            indices = json.load(f)
        if max_ep_override:
            indices = indices[:max_ep_override]
        allowed = set(indices)
        print(f"  Filter contains {len(allowed)} episode indices")

    cot_data = None
    if with_cot:
        print(f"  Loading COT data from {COT_JSON}")
        with open(COT_JSON) as f:
            cot_data = json.load(f)
        print(f"  COT data loaded ({len(cot_data)} episodes)")

    ds = ImagePairDataset(
        root=DATA_ROOT,
        camera_key=CAMERA_KEY,
        cot_data=cot_data,
        allowed_episode_indices=allowed,
    )
    return ds

# -----------------------------------------------------------------------
# Test 1: Filter sanity check
# -----------------------------------------------------------------------
print("=" * 60)
print("TEST 1: Filter mechanism")
print("=" * 60)

print("\n[No filter]")
ds_full = load_dataset(with_filter=False, with_cot=False)
full_ep_set = set(ds._episodes[ep_pos]["episode_index"]
                  for ep_pos, *_ in ds_full._index) if False else None
# Collect loaded episode indices
full_ep_indices = set(int(ep["episode_index"]) for ep in ds_full._episodes)
print(f"  Episodes loaded: {len(ds_full._episodes)}")
print(f"  Total index entries (samples): {len(ds_full)}")

print("\n[With filter: top-10000]")
ds_filt = load_dataset(with_filter=True, with_cot=False)
filt_ep_indices = set(int(ep["episode_index"]) for ep in ds_filt._episodes)
print(f"  Episodes loaded: {len(ds_filt._episodes)}")
print(f"  Total index entries (samples): {len(ds_filt)}")

# Verify: every episode in filtered dataset must be in the allowed set
with open(FILTERED_JSON) as f:
    allowed_set = set(json.load(f))
leaked = filt_ep_indices - allowed_set
if leaked:
    print(f"  [FAIL] {len(leaked)} episodes leaked through filter! e.g. {list(leaked)[:5]}")
    sys.exit(1)
else:
    print(f"  [PASS] All {len(filt_ep_indices)} episodes are within the allowed set")

# Verify: filtered episodes are a strict subset of full
extra = filt_ep_indices - full_ep_indices
if extra:
    print(f"  [FAIL] Filtered set has episodes not in full set: {list(extra)[:5]}")
else:
    print(f"  [PASS] Filtered episodes are a subset of full dataset")

# -----------------------------------------------------------------------
# Test 2: No-COT target frame check (must be last frame of episode)
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 2: No-COT target = last frame of episode")
print("=" * 60)

# Build a quick lookup: episode_index -> ep_len from meta
import json as _json
ep_len_map = {}
with open(Path(DATA_ROOT) / "meta" / "episodes.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ep = _json.loads(line)
        if "episode_index" in ep and "length" in ep:
            ep_len_map[int(ep["episode_index"])] = int(ep["length"])

fps = ds_filt.fps
checked = 0
fails = 0
for i, entry in enumerate(ds_filt._index[:2000]):
    ep_pos, src_fi, tgt_fi, subtask_name = entry
    ep = ds_filt._episodes[ep_pos]
    ep_idx = int(ep["episode_index"])
    ep_len = ep_len_map.get(ep_idx)
    if ep_len is None:
        continue
    expected_tgt = ep_len - 1
    if subtask_name == "":  # no COT for this episode
        if tgt_fi != expected_tgt:
            print(f"  [FAIL] ep={ep_idx} src_fi={src_fi} tgt_fi={tgt_fi} expected={expected_tgt}")
            fails += 1
        checked += 1
if fails == 0:
    print(f"  [PASS] Checked {checked} no-COT samples, all have correct last-frame target")
else:
    print(f"  [FAIL] {fails} samples have wrong target frame")

# -----------------------------------------------------------------------
# Test 3: COT target frame check
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 3: COT mode - target = last frame of current subtask")
print("=" * 60)

ds_cot = load_dataset(with_filter=True, with_cot=True)
print(f"  Dataset with COT: {len(ds_cot)} samples")

with open(COT_JSON) as f:
    cot_raw = json.load(f)

checked_cot = 0
fails_cot = 0
for entry in ds_cot._index[:3000]:
    ep_pos, src_fi, tgt_fi, subtask_name = entry
    ep = ds_cot._episodes[ep_pos]
    ep_idx_str = str(int(ep["episode_index"]))

    if subtask_name == "" or ep_idx_str not in cot_raw:
        continue

    cot = cot_raw[ep_idx_str]
    step_to_si = {int(k): int(v) for k, v in cot["step_to_subtask_index"].items()}
    all_subtasks = cot["all_subtasks"]

    si = step_to_si.get(src_fi)
    if si is None or si >= len(all_subtasks):
        continue

    # Expected: last step that belongs to si
    expected_tgt = max(s for s, idx in step_to_si.items() if idx == si)
    if tgt_fi != expected_tgt:
        fails_cot += 1
        if fails_cot <= 3:
            print(f"  [FAIL] ep={ep_idx_str} src={src_fi} subtask_idx={si} "
                  f"tgt={tgt_fi} expected={expected_tgt}")
    checked_cot += 1

if fails_cot == 0:
    print(f"  [PASS] Checked {checked_cot} COT samples, all have correct subtask-end target")
else:
    print(f"  [FAIL] {fails_cot}/{checked_cot} COT samples have wrong target frame")

# -----------------------------------------------------------------------
# Test 4: Visual inspection — render N sample pairs to PNG
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 4: Visual inspection of image pairs")
print("=" * 60)

# Pick N samples: spread across the index, preferring COT ones
import random
random.seed(42)
total = len(ds_cot)
cot_indices = [i for i, e in enumerate(ds_cot._index) if e[3] != ""]
nocot_indices = [i for i, e in enumerate(ds_cot._index) if e[3] == ""]

# Try to show half COT, half no-COT
n_cot = min(N_SAMPLES // 2, len(cot_indices))
n_nocot = N_SAMPLES - n_cot
sample_ids = (random.sample(cot_indices, n_cot) +
              random.sample(nocot_indices, min(n_nocot, len(nocot_indices))))
random.shuffle(sample_ids)

print(f"  Sampling {len(sample_ids)} pairs "
      f"({n_cot} COT + {N_SAMPLES - n_cot} no-COT) ...")

fig, axes = plt.subplots(len(sample_ids), 3,
                         figsize=(12, 3.5 * len(sample_ids)))
if len(sample_ids) == 1:
    axes = [axes]

for row, idx in enumerate(sample_ids):
    ep_pos, src_fi, tgt_fi, subtask_name = ds_cot._index[idx]
    ep = ds_cot._episodes[ep_pos]
    ep_idx = int(ep["episode_index"])
    ep_len = ep_len_map.get(ep_idx, "?")

    sample = ds_cot[idx]
    src_img = sample["source_image"]
    tgt_img = sample["target_image"]
    caption = sample["caption"]

    axes[row][0].imshow(src_img)
    axes[row][0].set_title(f"SOURCE\nep={ep_idx}  frame={src_fi}", fontsize=8)
    axes[row][0].axis("off")

    axes[row][1].imshow(tgt_img)
    axes[row][1].set_title(f"TARGET\nframe={tgt_fi}  ep_len={ep_len}", fontsize=8)
    axes[row][1].axis("off")

    # Text panel
    axes[row][2].axis("off")
    mode = "COT" if subtask_name else "no-COT"
    info = (f"mode: {mode}\n\n"
            f"src_fi:  {src_fi}\n"
            f"tgt_fi:  {tgt_fi}\n"
            f"ep_len:  {ep_len}\n\n"
            f"caption:\n{caption[:120]}")
    axes[row][2].text(0.02, 0.95, info, transform=axes[row][2].transAxes,
                      fontsize=7.5, va="top", family="monospace",
                      bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.suptitle("ImagePairDataset debug — source | target | metadata", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(OUT_IMG, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved visual inspection to {OUT_IMG}")
print("\nAll tests complete.")
