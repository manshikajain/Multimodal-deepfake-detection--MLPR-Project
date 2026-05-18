# ================================================================
# RUN_PIPELINE.py
# Dataset setup:
#   Real videos : 500
#   Fake videos : 2500+ available
#
# Test set:
#   100 real + 400 fake = 500 videos
#   Locked before oversampling to avoid leakage
#
# Training set:
#   Real videos are oversampled to 1250
#   Fake videos are under-sampled to 1250
#   Final train set = 1250 real + 1250 fake = 2500 videos
# ================================================================

import os
import random
import numpy as np

# ================================================================
# CONFIG
# ================================================================

# For GitHub use relative paths.
# Put your dataset folders inside:
# ./data/RealVideo-RealAudio
# ./data/FakeVideo-FakeAudio

REAL_ROOT = "./data/RealVideo-RealAudio"
FAKE_ROOT = "./data/FakeVideo-FakeAudio"

TRAIN_SAVE = "features_train_3000dataset.npz"
TEST_SAVE = "features_test_3000dataset.npz"

TEST_REAL = 100
TEST_FAKE = 400

TARGET_TRAIN_REAL = 1250
TARGET_TRAIN_FAKE = 1250

RANDOM_STATE = 42

# ================================================================
# COLLECT VIDEOS
# ================================================================

def collect_videos(root):
    paths = []

    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                paths.append(os.path.join(r, f))

    return paths


print("Collecting videos...")

random.seed(RANDOM_STATE)

real_vids = collect_videos(REAL_ROOT)
fake_vids = collect_videos(FAKE_ROOT)

print(f"Total real videos found : {len(real_vids)}")
print(f"Total fake videos found : {len(fake_vids)}")

if len(real_vids) == 0 or len(fake_vids) == 0:
    raise ValueError(
        "No videos found. Please check that dataset folders exist at "
        "'./data/RealVideo-RealAudio' and './data/FakeVideo-FakeAudio'."
    )

# ================================================================
# STEP 1 — LOCK TEST SET FIRST
# This prevents data leakage because test videos are never oversampled.
# ================================================================

random.shuffle(real_vids)
random.shuffle(fake_vids)

assert len(real_vids) >= TEST_REAL, (
    f"Need {TEST_REAL} real videos for test, only found {len(real_vids)}"
)

assert len(fake_vids) >= TEST_FAKE + TARGET_TRAIN_FAKE, (
    f"Need at least {TEST_FAKE + TARGET_TRAIN_FAKE} fake videos, "
    f"only found {len(fake_vids)}"
)

test_real = real_vids[:TEST_REAL]
test_fake = fake_vids[:TEST_FAKE]

train_real_pool = real_vids[TEST_REAL:]
train_fake_pool = fake_vids[TEST_FAKE:]

print("\nTest set locked:")
print(f"  Real : {len(test_real)}")
print(f"  Fake : {len(test_fake)}")

print("\nTraining pool before sampling:")
print(f"  Real pool : {len(train_real_pool)} unique videos")
print(f"  Fake pool : {len(train_fake_pool)} unique videos")

# ================================================================
# STEP 2 — BALANCE TRAINING SET
# Real class is oversampled with replacement.
# Fake class is under-sampled without replacement.
# ================================================================

if len(train_real_pool) == 0:
    raise ValueError("No real videos left for training after locking test set.")

train_real_oversampled = random.choices(
    train_real_pool,
    k=TARGET_TRAIN_REAL
)

train_fake_sampled = random.sample(
    train_fake_pool,
    TARGET_TRAIN_FAKE
)

print("\nAfter training-set balancing:")
print(
    f"  Real train : {len(train_real_oversampled)} "
    f"(oversampled from {len(train_real_pool)} unique videos)"
)
print(
    f"  Fake train : {len(train_fake_sampled)} "
    f"(sampled from {len(train_fake_pool)} unique videos)"
)

# ================================================================
# STEP 3 — BUILD LABELLED PAIRS
# label 0 = real
# label 1 = fake
# ================================================================

train_pairs = [(p, 0) for p in train_real_oversampled] + \
              [(p, 1) for p in train_fake_sampled]

test_pairs = [(p, 0) for p in test_real] + \
             [(p, 1) for p in test_fake]

random.shuffle(train_pairs)
random.shuffle(test_pairs)

print("\n" + "=" * 60)
print("FINAL SPLIT SUMMARY")
print("=" * 60)
print(
    f"TRAIN : {len(train_pairs)} videos "
    f"(Real={TARGET_TRAIN_REAL}, Fake={TARGET_TRAIN_FAKE})"
)
print(
    f"TEST  : {len(test_pairs)} videos "
    f"(Real={TEST_REAL}, Fake={TEST_FAKE})"
)
print("NOTE  : Test set is locked before oversampling to prevent leakage.")
print("=" * 60)

# ================================================================
# STEP 4 — FEATURE EXTRACTION
# ================================================================

from FEATURE_PIPELINE_v4 import (
    load_yunet,
    load_feature_extractor,
    extract_and_save_features
)

print("\nLoading models...")

yunet = load_yunet()
effnet = load_feature_extractor(bottleneck_dim=128)

print("\nProcessing TRAIN set...")
extract_and_save_features(
    train_pairs,
    yunet,
    effnet,
    save_path=TRAIN_SAVE,
    checkpoint_every=50
)

print("\nProcessing TEST set...")
extract_and_save_features(
    test_pairs,
    yunet,
    effnet,
    save_path=TEST_SAVE,
    checkpoint_every=50
)

# ================================================================
# STEP 5 — VERIFY SAVED FEATURES
# Some videos may be skipped if valid face frames cannot be extracted.
# ================================================================

print("\nVerifying saved features...")

for save_path, split_name in [(TRAIN_SAVE, "TRAIN"), (TEST_SAVE, "TEST")]:
    if not os.path.exists(save_path):
        print(f"{split_name}: {save_path} not found.")
        continue

    data = np.load(save_path, allow_pickle=True)
    labels = data["labels"].astype(int)

    print(f"\n{split_name} ({save_path}):")
    print(f"  Total extracted videos : {len(labels)}")
    print(f"  Real (0)               : {(labels == 0).sum()}")
    print(f"  Fake (1)               : {(labels == 1).sum()}")

print("\nFeature extraction complete.")
