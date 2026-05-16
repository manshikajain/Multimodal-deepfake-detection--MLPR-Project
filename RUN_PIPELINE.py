# ================================================================
# RUN_PIPELINE_3000dataset.py
# Dataset  : 500 real + 2500 fake = 3000 total
# Test set : 100 real + 100 fake  = 200  (locked first, never touched)
# Train    : 400 real + 2400 fake available
#            → oversample real 400 → 1250 (with replacement)
#            → random sample 1250 fake from 2400
#            → final train: 1250 real + 1250 fake = 2500
# Batch    : 16 (8 real + 8 fake) via WeightedRandomSampler
# ================================================================

import os
import random

# ================================================================
# CONFIG
# ================================================================
REAL_ROOT  = "/content/drive/MyDrive/deepfake_project/RealVideo-RealAudio"
FAKE_ROOT  = "/content/drive/MyDrive/deepfake_project/FakeVideo-FakeAudio"
TRAIN_SAVE = "features_train_3000dataset.npz"
TEST_SAVE  = "features_test_3000dataset.npz"

# How many UNIQUE real/fake videos go to test (locked away first)
TEST_REAL  = 100
TEST_FAKE  = 400

# Target train size after oversampling
TARGET_TRAIN_REAL = 1250   # oversampled from ~400 unique real train videos
TARGET_TRAIN_FAKE = 1250   # randomly sampled from ~2400 fake train videos

RANDOM_STATE = 42

# ================================================================
# COLLECT VIDEOS
# ================================================================
def collect_videos(root):
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov")):
                paths.append(os.path.join(r, f))
    return paths

print("Collecting videos...")
random.seed(RANDOM_STATE)

real_vids = collect_videos(REAL_ROOT)
fake_vids = collect_videos(FAKE_ROOT)

print(f"Total real videos found : {len(real_vids)}")
print(f"Total fake videos found : {len(fake_vids)}")

# ================================================================
# STEP 1 — LOCK TEST SET AWAY FIRST (before any oversampling)
# Only unique, original videos in test — no duplicates ever
# ================================================================
random.shuffle(real_vids)
random.shuffle(fake_vids)

assert len(real_vids) >= TEST_REAL, f"Need {TEST_REAL} real videos for test, only have {len(real_vids)}"
assert len(fake_vids) >= TEST_FAKE, f"Need {TEST_FAKE} fake videos for test, only have {len(fake_vids)}"

# Lock test — these paths are NEVER oversampled
test_real = real_vids[:TEST_REAL]        # 100 unique real
test_fake = fake_vids[:TEST_FAKE]        # 100 unique fake

# Remaining pool available for training
train_real_pool = real_vids[TEST_REAL:]  # ~400 unique real videos
train_fake_pool = fake_vids[TEST_FAKE:]  # ~2400 unique fake videos

print(f"\nTest set locked:")
print(f"  Real : {len(test_real)}")
print(f"  Fake : {len(test_fake)}")
print(f"\nTraining pool (before oversampling):")
print(f"  Real pool : {len(train_real_pool)} unique videos")
print(f"  Fake pool : {len(train_fake_pool)} unique videos")

# ================================================================
# STEP 2 — OVERSAMPLE REAL TRAINING VIDEOS
# random.choices samples WITH replacement — duplicates allowed
# This is what lets 400 unique real videos become 1250
# The feature extractor will process each path including duplicates
# so the model genuinely sees the same video multiple times across
# different batches → acts like data augmentation at the video level
# ================================================================
train_real_oversampled = random.choices(train_real_pool, k=TARGET_TRAIN_REAL)

# Randomly pick 1250 fake from the 2400 available (no replacement needed,
# we have more than enough fake videos)
train_fake_sampled = random.sample(train_fake_pool, TARGET_TRAIN_FAKE)

print(f"\nAfter oversampling:")
print(f"  Real train : {len(train_real_oversampled)} (oversampled from {len(train_real_pool)} unique)")
print(f"  Fake train : {len(train_fake_sampled)}")

# ================================================================
# BUILD PAIRS
# label 0 = real, label 1 = fake
# ================================================================
train_pairs = [(p, 0) for p in train_real_oversampled] + \
              [(p, 1) for p in train_fake_sampled]
test_pairs  = [(p, 0) for p in test_real] + \
              [(p, 1) for p in test_fake]

random.shuffle(train_pairs)
random.shuffle(test_pairs)

print(f"\n{'='*50}")
print(f"FINAL SPLIT SUMMARY")
print(f"{'='*50}")
print(f"TRAIN : {len(train_pairs)} videos  (Real={TARGET_TRAIN_REAL}, Fake={TARGET_TRAIN_FAKE})")
print(f"TEST  : {len(test_pairs)} videos   (Real={TEST_REAL}, Fake={TEST_FAKE})")
print(f"{'='*50}")
print(f"NOTE  : Test set has {len(test_real)} UNIQUE real videos — no oversampling, no leakage")

# ================================================================
# STEP 3 — FEATURE EXTRACTION
# The extractor processes each path in train_pairs
# Duplicate real paths are processed again (model sees them multiple times)
# Checkpointing is based on path — so duplicate paths WILL be reprocessed
# which is correct behaviour for oversampling
# ================================================================
from FEATURE_PIPELINE_v4 import (
    load_yunet,
    load_feature_extractor,
    extract_and_save_features
)

print("\nLoading models...")
yunet  = load_yunet()
effnet = load_feature_extractor(bottleneck_dim=128)

# --- TRAIN ---
print("\nProcessing TRAIN set (2500 videos, real oversampled)...")
extract_and_save_features(
    train_pairs,
    yunet,
    effnet,
    save_path=TRAIN_SAVE,
    checkpoint_every=50
)

# --- TEST ---
print("\nProcessing TEST set (200 videos, balanced, no oversampling)...")
extract_and_save_features(
    test_pairs,
    yunet,
    effnet,
    save_path=TEST_SAVE,
    checkpoint_every=50
)

# ================================================================
# STEP 4 — VERIFY SAVED FEATURES
# ================================================================
import numpy as np

print("\nVerifying saved features...")
for save_path, name in [(TRAIN_SAVE, "TRAIN"), (TEST_SAVE, "TEST")]:
    data   = np.load(save_path, allow_pickle=True)
    labels = data["labels"]
    print(f"\n{name} ({save_path}):")
    print(f"  Total videos : {len(labels)}")
    print(f"  Real (0)     : {(labels == 0).sum()}")
    print(f"  Fake (1)     : {(labels == 1).sum()}")

# ================================================================
# STEP 5 — WEIGHTED SAMPLER FOR BALANCED BATCHES (8 real + 8 fake)
# Use this in your training script when building the DataLoader
# Since train set is already 1250/1250 balanced, weights are equal
# but keeping the sampler is good practice and makes it easy to
# adjust ratios later without changing the dataset
# ================================================================
print("""
================================================================
BALANCED DATALOADER SETUP (copy into your training script)
================================================================

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

# After loading your train dataset:
#   train_labels = list of 0s and 1s for each sample

# Count per class
n_real = sum(1 for l in train_labels if l == 0)
n_fake = sum(1 for l in train_labels if l == 1)

# Weight = inverse of class frequency
weight_real = 1.0 / n_real
weight_fake = 1.0 / n_fake

# Assign weight to each sample
sample_weights = [
    weight_real if label == 0 else weight_fake
    for label in train_labels
]

sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True      # allows oversampled real videos to appear in any batch
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,        # 8 real + 8 fake per batch (approx)
    sampler=sampler,      # replaces shuffle=True
    num_workers=2
)
================================================================
""")

print("\nAll done! Send ane2_350_150.py and I'll update that too.")
