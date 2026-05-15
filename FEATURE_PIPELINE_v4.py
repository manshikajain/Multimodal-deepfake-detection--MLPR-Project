# =========================================================
# FEATURE PIPELINE v4
# Changes from v3:
# 1. MediaPipe landmark-based lip + eye crops (not positional)
# 2. Temporal delta features for lips and eyes
# 3. Eye Aspect Ratio (EAR) + blink rate per video
# 4. Lip opening distance per frame (mouth aperture)
# 5. Audio-visual cross-correlation sync score
# 6. All bugs from v3 preserved as fixed
# 7. Checkpointing + .npz saving preserved
# =========================================================

import os
import cv2
import numpy as np
import librosa
import mediapipe as mp

from moviepy.editor import VideoFileClip
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models

# -------------------------------------------------------
# MediaPipe face mesh — init once at module load
# -------------------------------------------------------
_mp_face_mesh = mp.solutions.face_mesh
_face_mesh    = _mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# MediaPipe landmark indices
# Lips outer boundary
LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]
# Upper + lower lip centre for mouth opening distance
LIP_TOP    = 13   # upper lip centre
LIP_BOTTOM = 14   # lower lip centre

# Eye landmarks (left eye + right eye)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


# ===============================
# LOAD YUNET FACE DETECTOR
# ===============================
def load_yunet(model_path="face_detection_yunet_2023mar.onnx"):
    return cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(320, 320),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    )


# ===============================
# LOAD PRETRAINED EFFICIENTNET
# ===============================
def load_feature_extractor(bottleneck_dim=128):
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    inputs = layers.Input(shape=(224, 224, 3))
    x      = base_model(inputs)
    x      = layers.Dense(bottleneck_dim, activation="relu", name="fc_bottleneck")(x)
    model  = models.Model(inputs=inputs, outputs=x)
    print(f"Feature extractor output dim: {bottleneck_dim}")
    return model


# ===============================
# CONSERVATIVE FACE CROP (unchanged from v3)
# ===============================
def conservative_crop(frame, face_box, margin_scale=1.3):
    h, w, _ = frame.shape
    x, y, box_w, box_h = face_box.astype(int)
    cx = x + box_w // 2
    cy = y + box_h // 2
    nw = int(box_w * margin_scale)
    nh = int(box_h * margin_scale)
    x1 = max(cx - nw // 2, 0)
    y1 = max(cy - nh // 2, 0)
    x2 = min(cx + nw // 2, w)
    y2 = min(cy + nh // 2, h)
    return frame[y1:y2, x1:x2]


# ===============================
# EAR — Eye Aspect Ratio
# Standard formula from Soukupova & Cech (2016)
# EAR < 0.2 = blink
# ===============================
def _eye_aspect_ratio(landmarks, eye_indices, img_h, img_w):
    pts = np.array([
        [landmarks[i].x * img_w, landmarks[i].y * img_h]
        for i in eye_indices
    ])
    # vertical distances
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    # horizontal distance
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


# ===============================
# LIP OPENING DISTANCE
# Vertical distance between upper and lower lip centre
# ===============================
def _lip_opening(landmarks, img_h, img_w):
    top    = np.array([landmarks[LIP_TOP].x * img_w,
                       landmarks[LIP_TOP].y * img_h])
    bottom = np.array([landmarks[LIP_BOTTOM].x * img_w,
                       landmarks[LIP_BOTTOM].y * img_h])
    return float(np.linalg.norm(top - bottom))


# ===============================
# LANDMARK-BASED CROP
# Uses MediaPipe to get precise bounding box for a region
# region_indices: list of landmark indices defining the region
# ===============================
def _landmark_crop(frame, landmarks, region_indices, pad=0.2):
    h, w = frame.shape[:2]
    xs = [landmarks[i].x * w for i in region_indices]
    ys = [landmarks[i].y * h for i in region_indices]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # add padding
    pw = (x_max - x_min) * pad
    ph = (y_max - y_min) * pad

    x1 = max(int(x_min - pw), 0)
    y1 = max(int(y_min - ph), 0)
    x2 = min(int(x_max + pw), w)
    y2 = min(int(y_max + ph), h)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


# ===============================
# VIDEO PREPROCESSING
# Returns uint8 face crops + per-frame behavioural signals
# ===============================
def preprocess_video(
    video_path,
    yunet,
    mode="dense",
    sample_rate=10,
    sequence_length=30,
    resize_dim=224
):
    cap         = cv2.VideoCapture(video_path)
    face_frames = []          # uint8 full face crops (224,224,3)
    ear_series  = []          # EAR per frame (float)
    lip_open    = []          # lip opening distance per frame (float)
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        process = (mode == "dense") or (mode == "sparse" and frame_index % sample_rate == 0)

        if process:
            h, w, _ = frame.shape
            yunet.setInputSize((w, h))
            _, faces = yunet.detect(frame)

            if faces is not None:
                face_box  = faces[0][:4]
                face_crop = conservative_crop(frame, face_box)
                face_crop_resized = cv2.resize(face_crop, (resize_dim, resize_dim))

                # --- MediaPipe on the face crop ---
                rgb = cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2RGB)
                result = _face_mesh.process(rgb)

                if result.multi_face_landmarks:
                    lm  = result.multi_face_landmarks[0].landmark
                    fh, fw = face_crop_resized.shape[:2]

                    ear_l = _eye_aspect_ratio(lm, LEFT_EYE,  fh, fw)
                    ear_r = _eye_aspect_ratio(lm, RIGHT_EYE, fh, fw)
                    ear   = (ear_l + ear_r) / 2.0

                    lip_dist = _lip_opening(lm, fh, fw)

                    ear_series.append(ear)
                    lip_open.append(lip_dist)
                else:
                    # fallback if mediapipe misses landmarks
                    ear_series.append(0.3)   # neutral EAR
                    lip_open.append(0.0)

                face_frames.append(face_crop_resized.astype(np.uint8))

        frame_index += 1

    cap.release()

    if len(face_frames) < sequence_length:
        return None

    face_frames = np.array(face_frames[:sequence_length])   # (30,224,224,3)
    ear_series  = np.array(ear_series[:sequence_length])    # (30,)
    lip_open    = np.array(lip_open[:sequence_length])      # (30,)

    return {
        "frames":    face_frames,
        "ear":       ear_series,
        "lip_open":  lip_open
    }


# ===============================
# SPATIAL FEATURES — full face
# output: (30, 128)
# ===============================
def extract_spatial_features(frames, feature_extractor):
    seq = frames.astype("float32")
    seq = preprocess_input(seq)
    return feature_extractor.predict(seq, verbose=0)


# ===============================
# LANDMARK-BASED LIP FEATURES
# Uses MediaPipe lip landmarks for precise crop
# output: (30, 128)
# ===============================
def extract_lip_features_landmark(frames, feature_extractor):
    mouth_frames = []

    for frame in frames:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm   = result.multi_face_landmarks[0].landmark
            crop = _landmark_crop(frame, lm, LIP_LANDMARKS, pad=0.3)
            if crop is not None and crop.size > 0:
                crop = cv2.resize(crop, (224, 224))
                mouth_frames.append(crop)
                continue

        # fallback: bottom 50% positional crop
        h = frame.shape[0]
        fallback = cv2.resize(frame[h // 2:h, :], (224, 224))
        mouth_frames.append(fallback)

    mouth_arr = np.array(mouth_frames, dtype="float32")
    mouth_arr = preprocess_input(mouth_arr)
    return feature_extractor.predict(mouth_arr, verbose=0)   # (30,128)


# ===============================
# LANDMARK-BASED EYE FEATURES
# Uses MediaPipe eye landmarks for precise crop
# output: (30, 128)
# ===============================
def extract_eye_features_landmark(frames, feature_extractor):
    # combined left+right eye landmark indices for bounding box
    EYE_ALL = LEFT_EYE + RIGHT_EYE + [
        # eyebrow region
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
        336, 296, 334, 293, 300, 276, 283, 282, 295, 285
    ]
    eye_frames = []

    for frame in frames:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm   = result.multi_face_landmarks[0].landmark
            crop = _landmark_crop(frame, lm, EYE_ALL, pad=0.4)
            if crop is not None and crop.size > 0:
                crop = cv2.resize(crop, (224, 224))
                eye_frames.append(crop)
                continue

        # fallback: top 40% positional crop
        h = frame.shape[0]
        fallback = cv2.resize(frame[0:int(h * 0.4), :], (224, 224))
        eye_frames.append(fallback)

    eye_arr = np.array(eye_frames, dtype="float32")
    eye_arr = preprocess_input(eye_arr)
    return feature_extractor.predict(eye_arr, verbose=0)   # (30,128)


# ===============================
# OPTICAL FLOW — unchanged from v3
# output: (29,)
# ===============================
def extract_optical_flow(frames):
    flows = []
    prev  = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flows.append(float(np.mean(mag)))
        prev = curr

    return np.array(flows)   # (29,)


# ===============================
# TEMPORAL DELTA FEATURES
# Frame-to-frame change in feature vectors
# Captures HOW FAST features are changing — key behavioural signal
# output: (29, feat_dim)
# ===============================
def extract_temporal_delta(features_seq):
    # features_seq: (30, feat_dim)
    deltas = np.diff(features_seq, axis=0)   # (29, feat_dim)
    return deltas


# ===============================
# BLINK FEATURES from EAR series
# output: dict with scalar behavioural stats
# ===============================
def extract_blink_features(ear_series, blink_threshold=0.2):
    blinks       = 0
    was_blinking = False

    for ear in ear_series:
        if ear < blink_threshold:
            if not was_blinking:
                blinks      += 1
                was_blinking = True
        else:
            was_blinking = False

    n_frames     = len(ear_series)
    blink_rate   = blinks / (n_frames / 30.0 + 1e-6)   # blinks per second approx
    mean_ear     = float(np.mean(ear_series))
    std_ear      = float(np.std(ear_series))
    min_ear      = float(np.min(ear_series))
    ear_temporal = extract_temporal_delta(ear_series.reshape(-1, 1)).flatten()  # (29,)

    return {
        "blink_rate":    blink_rate,
        "mean_ear":      mean_ear,
        "std_ear":       std_ear,
        "min_ear":       min_ear,
        "ear_temporal":  ear_temporal   # (29,) — how fast eyes are opening/closing
    }


# ===============================
# LIP BEHAVIOURAL FEATURES from lip_open series
# output: dict with scalar + temporal stats
# ===============================
def extract_lip_behavioural(lip_open_series):
    mean_open    = float(np.mean(lip_open_series))
    std_open     = float(np.std(lip_open_series))
    max_open     = float(np.max(lip_open_series))
    lip_temporal = extract_temporal_delta(lip_open_series.reshape(-1, 1)).flatten()  # (29,)

    return {
        "mean_lip_open":  mean_open,
        "std_lip_open":   std_open,
        "max_lip_open":   max_open,
        "lip_temporal":   lip_temporal   # (29,) — how fast mouth is opening/closing
    }


# ===============================
# AUDIO FEATURES — unchanged from v3 + adds delta
# output: dict
# ===============================
def extract_audio_features(video_path, n_mfcc=40, target_len=30):
    temp_audio = "temp_audio.wav"
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            video.close()
            mfcc_aligned = np.zeros((target_len, n_mfcc))
        else:
            video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
            video.close()
            y_audio, sr = librosa.load(temp_audio, sr=None)
            mfcc        = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc).T
            indices     = np.linspace(0, len(mfcc) - 1, target_len).astype(int)
            mfcc_aligned = mfcc[indices]   # (30, 40)

        mfcc_delta = extract_temporal_delta(mfcc_aligned)   # (29, 40)

        return {
            "mfcc":       mfcc_aligned,   # (30, 40)
            "mfcc_delta": mfcc_delta      # (29, 40) — audio rate of change
        }

    except Exception as e:
        print(f"  Audio failed: {e}")
        return {
            "mfcc":       np.zeros((target_len, n_mfcc)),
            "mfcc_delta": np.zeros((target_len - 1, n_mfcc))
        }
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)


# ===============================
# AV SYNC SCORE
# Cross-correlation between lip opening and audio energy
# Real videos: lips move in sync with sound
# Fake videos: often have a lag or no correlation
# output: scalar sync_score (higher = better sync = more likely real)
# ===============================
def compute_av_sync_score(lip_open_series, mfcc_aligned):
    # use RMS of MFCC as proxy for audio energy per frame
    audio_energy = np.sqrt(np.mean(mfcc_aligned ** 2, axis=1))   # (30,)
    lip_norm     = (lip_open_series - np.mean(lip_open_series)) / (np.std(lip_open_series) + 1e-6)
    audio_norm   = (audio_energy - np.mean(audio_energy)) / (np.std(audio_energy) + 1e-6)

    # cross-correlation at lag 0 and ±2 frames, take max
    scores = []
    for lag in [-2, -1, 0, 1, 2]:
        if lag == 0:
            corr = float(np.corrcoef(lip_norm, audio_norm)[0, 1])
        elif lag > 0:
            corr = float(np.corrcoef(lip_norm[lag:], audio_norm[:-lag])[0, 1])
        else:
            corr = float(np.corrcoef(lip_norm[:lag], audio_norm[-lag:])[0, 1])
        if not np.isnan(corr):
            scores.append(corr)

    return float(np.max(scores)) if scores else 0.0


# ===============================
# PROCESS SINGLE VIDEO — main wrapper
# ===============================
def process_video(video_path, yunet, feature_extractor):
    try:
        # --- Step 1: video preprocessing ---
        prep = preprocess_video(video_path, yunet, mode="dense", sequence_length=30)
        if prep is None:
            return None

        frames      = prep["frames"]       # (30,224,224,3) uint8
        ear_series  = prep["ear"]          # (30,)
        lip_open_s  = prep["lip_open"]     # (30,)

        # --- Step 2: deep features ---
        spatial    = extract_spatial_features(frames, feature_extractor)         # (30,128)
        lips_deep  = extract_lip_features_landmark(frames, feature_extractor)    # (30,128)
        eyes_deep  = extract_eye_features_landmark(frames, feature_extractor)    # (30,128)

        # --- Step 3: temporal deltas of deep features ---
        spatial_delta = extract_temporal_delta(spatial)     # (29,128)
        lips_delta    = extract_temporal_delta(lips_deep)   # (29,128)
        eyes_delta    = extract_temporal_delta(eyes_deep)   # (29,128)

        # --- Step 4: optical flow ---
        flow = extract_optical_flow(frames)   # (29,)

        # --- Step 5: behavioural signals ---
        blink_feats = extract_blink_features(ear_series)
        lip_feats   = extract_lip_behavioural(lip_open_s)

        # --- Step 6: audio ---
        audio = extract_audio_features(video_path)
        mfcc       = audio["mfcc"]         # (30,40)
        mfcc_delta = audio["mfcc_delta"]   # (29,40)

        # --- Step 7: AV sync score ---
        sync_score = compute_av_sync_score(lip_open_s, mfcc)

        # --- Step 8: joint AV (lips + audio) ---
        joint_av = np.concatenate([lips_deep, mfcc], axis=1)   # (30,168)

        return {
            # deep spatial features
            "spatial":        spatial,          # (30,128)
            "lips":           lips_deep,         # (30,128)
            "eyes":           eyes_deep,         # (30,128)

            # temporal deltas — KEY new addition
            "spatial_delta":  spatial_delta,    # (29,128)
            "lips_delta":     lips_delta,        # (29,128)
            "eyes_delta":     eyes_delta,        # (29,128)

            # motion
            "optical_flow":   flow,              # (29,)

            # audio
            "mfcc":           mfcc,              # (30,40)
            "mfcc_delta":     mfcc_delta,        # (29,40)

            # joint
            "joint_av":       joint_av,          # (30,168)

            # behavioural scalars — KEY new addition
            "blink_rate":     blink_feats["blink_rate"],       # scalar
            "mean_ear":       blink_feats["mean_ear"],          # scalar
            "std_ear":        blink_feats["std_ear"],           # scalar
            "min_ear":        blink_feats["min_ear"],           # scalar
            "ear_temporal":   blink_feats["ear_temporal"],      # (29,)
            "mean_lip_open":  lip_feats["mean_lip_open"],       # scalar
            "std_lip_open":   lip_feats["std_lip_open"],        # scalar
            "max_lip_open":   lip_feats["max_lip_open"],        # scalar
            "lip_temporal":   lip_feats["lip_temporal"],        # (29,)
            "av_sync_score":  sync_score,                       # scalar
        }

    except Exception as e:
        print(f"  ERROR {os.path.basename(video_path)}: {e}")
        return None


# ================================================================
# BATCH EXTRACTION WITH CHECKPOINTING
# ================================================================
def extract_and_save_features(
    video_label_pairs,
    yunet,
    feature_extractor,
    save_path="features_v4.npz",
    checkpoint_every=50
):
    all_ = {k: [] for k in [
        "spatial", "lips", "eyes",
        "spatial_delta", "lips_delta", "eyes_delta",
        "optical_flow", "mfcc", "mfcc_delta", "joint_av",
        "ear_temporal", "lip_temporal",
        "blink_rate", "mean_ear", "std_ear", "min_ear",
        "mean_lip_open", "std_lip_open", "max_lip_open",
        "av_sync_score",
        "labels", "paths"
    ]}

    processed_paths = set()

    # --- load checkpoint ---
    if os.path.exists(save_path):
        print(f"Checkpoint found — loading '{save_path}'...")
        data = np.load(save_path, allow_pickle=True)
        for k in all_:
            if k in data:
                all_[k] = list(data[k])
        processed_paths = set(all_["paths"])
        print(f"Resuming from {len(processed_paths)} already done.")

    total  = len(video_label_pairs)
    failed = 0

    for i, (vpath, label) in enumerate(video_label_pairs):
        if vpath in processed_paths:
            continue

        print(f"[{i+1}/{total}] {os.path.basename(vpath)}", end=" ... ")
        feats = process_video(vpath, yunet, feature_extractor)

        if feats is None:
            print("SKIPPED")
            failed += 1
            continue

        for k in ["spatial", "lips", "eyes", "spatial_delta", "lips_delta",
                  "eyes_delta", "optical_flow", "mfcc", "mfcc_delta",
                  "joint_av", "ear_temporal", "lip_temporal"]:
            all_[k].append(feats[k])

        for k in ["blink_rate", "mean_ear", "std_ear", "min_ear",
                  "mean_lip_open", "std_lip_open", "max_lip_open", "av_sync_score"]:
            all_[k].append(feats[k])

        all_["labels"].append(label)
        all_["paths"].append(vpath)
        print("OK")

        if len(all_["labels"]) % checkpoint_every == 0:
            _save(save_path, all_)
            print(f"  >>> Checkpoint: {len(all_['labels'])} done <<<")

    _save(save_path, all_)
    print(f"\nDONE — {len(all_['labels'])} videos saved to '{save_path}' | Failed: {failed}")
    return save_path


def _save(path, all_):
    arrays = {}
    for k, v in all_.items():
        try:
            arrays[k] = np.array(v)
        except Exception:
            arrays[k] = np.array(v, dtype=object)
    np.savez_compressed(path, **arrays)


# ================================================================
# LOAD SAVED FEATURES
# ================================================================
def load_features(save_path="features_v4.npz"):
    print(f"Loading from '{save_path}'...")
    data = np.load(save_path, allow_pickle=True)
    out  = {k: data[k] for k in data.files}
    n    = len(out["labels"])
    print(f"Loaded {n} videos — Real: {(out['labels']==0).sum()} | Fake: {(out['labels']==1).sum()}")
    return out
