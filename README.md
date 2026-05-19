# Behaviour-Aware Multimodal Deepfake Detection

A multimodal deepfake detection framework that moves beyond visual appearance and analyzes behavioural inconsistencies across facial dynamics, temporal motion, and audio-visual synchronization.

---

## Overview

Recent advances in generative AI have enabled highly realistic deepfake videos capable of mimicking facial expressions, speech, and human behaviour. Traditional visual-only detection methods often fail when synthetic media becomes photorealistic.

This Machine Learning & Pattern Recognition project proposes a behaviour-aware multimodal deepfake detection pipeline that focuses on:

* Eye-blink behaviour
* Lip movement consistency
* Audio-visual synchronization
* Temporal motion irregularities
* Frame-level behavioural dynamics

The framework combines spatial, temporal, motion, and audio features into a unified multimodal representation and classifies videos using a hyperparameter-tuned LightGBM ensemble model.

---

# Pipeline Architecture

## 1. Frame & Audio Extraction

* Dense frame sampling (30 continuous valid frames per video)
* Audio extraction aligned with visual sequences

## 2. Face Detection & Landmark Localization

* YuNet face detector
* MediaPipe FaceMesh landmarks
* Lip and eye region extraction

## 3. Multimodal Feature Extraction

### Spatial Features

* EfficientNetB0 embeddings from:

  * Full face
  * Lip region
  * Eye region

### Temporal Features

* Blink rate
* Eye Aspect Ratio (EAR)
* Lip motion
* Optical flow motion statistics

### Audio Features

* MFCC features
* Delta-MFCC features

### Audio-Visual Features

* Joint AV synchronization embeddings

---

# Temporal Pooling & Feature Fusion

Sequential features are aggregated using:

* Mean pooling
* Standard deviation pooling
* Max pooling

This converts variable-length sequential data into fixed-size video-level embeddings.

All pooled multimodal representations are concatenated into a final:

```text
3065-dimensional fused behavioural embedding
```

---

# ML Methodology

## Final Classifier

* LightGBM (Gradient Boosted Decision Trees)

### Why LightGBM?

* Efficient for high-dimensional embeddings
* Handles nonlinear multimodal interactions
* Scalable ensemble learning
* Strong generalization performance
* Compatible with SHAP explainability

---

# Explainability

The framework integrates SHAP (SHapley Additive Explanations) to identify influential behavioural cues contributing to fake predictions.

Top contributing signals included:

* Lip-sync mismatch
* Audio irregularities
* Temporal instability
* Delta-MFCC variations

---

# Dataset

## FakeAVCeleb

Multimodal audiovisual deepfake dataset containing:

* Real videos
* Fake videos
* Multiple ethnicities
* Audio-synchronized manipulated content

---

# Final Performance

| Metric            | Value  |
| ----------------- | ------ |
| Accuracy          | 98.39% |
| Balanced Accuracy | 96.75% |
| F1-score          | 0.9900 |
| AUC-ROC           | 0.9994 |

---

# PCA & Feature Space Analysis

* PCA was used for multimodal feature-space visualization.
* Larger datasets produced significantly more overlap in low-dimensional projections, indicating complex nonlinear behavioural distributions.
* 354, 533, and 937 PCA components were required to explain 90%, 95%, and 99% of the variance respectively.

---

# Repository Structure

```text
FEATURE_PIPELINE_v4.py
RUN_PIPELINE.py
analyse_and_explain.py
face_detection_yunet_2023mar.onnx
requirements.txt
README.md
```

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

## Step 1 — Feature Extraction

```bash
python FEATURE_PIPELINE_v4.py
```

## Step 2 — Dataset Processing

```bash
python RUN_PIPELINE.py
```

## Step 3 — Training & Evaluation

```bash
python analyse_and_explain.py
```

---

# Challenges Addressed

* Runtime disconnections during large-scale extraction
* GPU memory limitations
* Failed face detections
* High-dimensional multimodal fusion
* Class imbalance
* Overfitting prevention
* Cross-library dependency conflicts

---

# Future Work

* Cross-dataset generalization testing
* Real-world deployment evaluation
* Temporal transformers for sequential modelling
* Real-time inference optimization
* Robustness against unseen generative models

---

# Author

Manshika Jain

MLPR Project - Multimodal Deepfake Detection
