# 🚗 Egocentric Traffic Anomaly Detection & Anticipation System

## A Tri-Expert Ensemble with Cross-Attention MS-TCN for Dashcam Collision Prediction

> **Undergraduate Project** — Optimised for NVIDIA RTX 4050 (6 GB VRAM)  
> **Primary Dataset:** [Nexar Dashcam Collision Prediction Dataset](https://www.kaggle.com/competitions/nexar-collision-prediction)

---

## Abstract

This project implements a high-performance **Video Anomaly Detection and Anticipation** system for egocentric (dashcam) video.  The system targets the **Nexar Dashcam Collision Prediction Challenge**, leveraging its frame-level **Time-of-Alert (TOA)** and **Time-of-Event (TOE)** metadata to optimise two critical safety metrics:

| Metric | Definition | Our Target |
|--------|-----------|------------|
| **STAUC** | Spatio-Temporal Area Under the Curve — joint accuracy of *when* and *where* an anomaly is detected | ≥ 0.75 |
| **TTA** | Time-to-Accident — how far in advance (ms) the system raises a sustained alert before collision | ≥ 3 000 ms |

The core innovation is a **Tri-Expert Ensemble** architecture with an **Offline Feature Extraction** ("Feature Bottlenecking") strategy that enables training on a consumer-grade RTX 4050 GPU with only 6 GB of VRAM — a constraint that normally prohibits end-to-end video model training.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    OFFLINE FEATURE EXTRACTION                    │
│              (one expert loaded at a time, ≤ 2 GB)              │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Expert A:       │  │  Expert B:       │  │  Expert C:       │ │
│  │  VideoMAE-Base   │  │  YOLOv12-Nano   │  │  EfficientNet   │ │
│  │  (4-bit NF4)     │  │  + ByteTrack    │  │  V2-S           │ │
│  │                  │  │                  │  │                  │ │
│  │  Global Temporal │  │  Local Spatial   │  │  Environmental  │ │
│  │  Motion Dynamics │  │  Object Tracks   │  │  Scene Context  │ │
│  │                  │  │                  │  │                  │ │
│  │  Output:         │  │  Output:         │  │  Output:         │ │
│  │  (32, 768)       │  │  (32, 5, 10)     │  │  (32, 1280)     │ │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬────────┘ │
│           │                     │                      │          │
│           └─────────┬───────────┴──────────┬───────────┘          │
│                     ▼                      ▼                      │
│              Saved as .npy          (compressed NumPy)            │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                ONLINE FUSION HEAD TRAINING                       │
│            (~2.5 M params, < 4 GB VRAM total)                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Cross-Attention MS-TCN                        │  │
│  │                                                            │  │
│  │   Q = Motion (global scene)  ──┐                          │  │
│  │   K,V = Spatial (tracklets)  ──┼──► Cross-Attention       │  │
│  │                                │    "Which vehicle is     │  │
│  │                                │     deviating from the   │  │
│  │                                │     global traffic flow?"│  │
│  │                                ▼                          │  │
│  │   ┌──────────┐    ┌──────────────────────┐               │  │
│  │   │ Context  │───►│  3-Stage Dilated TCN  │               │  │
│  │   │ Embedding│    │  (causal, residual)   │               │  │
│  │   └──────────┘    └──────────┬───────────┘               │  │
│  │                              │                            │  │
│  │              ┌───────────────┼───────────────┐            │  │
│  │              ▼                               ▼            │  │
│  │   ┌──────────────────┐           ┌──────────────────┐    │  │
│  │   │  Branch A:       │           │  Branch B:       │    │  │
│  │   │  Frame Score     │           │  Spatial Weight  │    │  │
│  │   │  (B, 32) ∈ [0,1] │           │  Mask (B, 32, 5) │    │  │
│  │   └──────────────────┘           └──────────────────┘    │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Why This Ensemble Maximises STAUC

STAUC requires the model to answer **two questions simultaneously**:

1. **WHEN** — temporal detection accuracy (is this frame anomalous?)
2. **WHERE** — spatial localisation (which object is causing it?)

A single backbone cannot excel at both.  Our ensemble divides the problem:

| Expert | Role | STAUC Contribution |
|--------|------|-------------------|
| **VideoMAE-Base** | Captures *global* temporal dynamics — sudden camera shakes, erratic optical flow, abrupt deceleration patterns | Provides the **WHEN** signal via global scene understanding |
| **YOLOv12-Nano + ByteTrack** | Tracks individual vehicles — bounding boxes, velocities, trajectory deviations | Provides the **WHERE** signal via per-object spatial localisation |
| **EfficientNet-V2-S** | Classifies environmental context — rain, night, fog, intersections | Provides **risk priors** that modulate anomaly thresholds |

The **Cross-Attention layer** is the key fusion mechanism: it uses the global motion descriptor (VideoMAE) as the **Query** to attend over individual vehicle tracklets (YOLO) as **Key/Value**.  This explicitly computes *which tracked objects deviate most from the expected global traffic flow* — the exact signal needed for high STAUC.

---

## Mathematical Framework

### STAUC (Spatio-Temporal Area Under the Curve)

For a set of IoU thresholds $\tau \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$:

$$\text{STAUC} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \text{AUC}_\tau$$

Where $\text{AUC}_\tau$ is the ROC-AUC computed over all (video, frame) pairs:

- **Positive frame** (GT anomaly exists): score = $s_t \cdot \mathbb{1}[\max_j \text{IoU}(\hat{b}_t, b_t^j) \geq \tau]$
- **Negative frame** (no GT anomaly): score = $s_t$

This penalises models that detect anomalies at the wrong spatial location.

### TTA (Time-to-Accident)

$$\text{TTA} = (t_{\text{TOE}} - t_{\text{alert}}) \times 1000 \quad \text{[milliseconds]}$$

Where $t_{\text{alert}}$ is the timestamp of the first frame in a run of $\geq 3$ consecutive frames where $s_t \geq 0.5$.

### Focal Loss

$$\mathcal{L}_{\text{FL}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

With $\alpha = 0.75$ (upweighting crash frames) and $\gamma = 2.0$ (focusing on hard examples).  This is critical because in a typical 40-second Nexar clip at 32 sampled frames, only 2–5 frames fall within the TOA–TOE alert window.

---

## Hardware Constraints Management

### The Feature Bottlenecking Strategy

The RTX 4050 has **6 GB VRAM**.  End-to-end video training with even a small VideoMAE model requires 8–12 GB.  Our solution:

```
Phase 1 (Offline — run once):
    Load VideoMAE (4-bit NF4)  →  ~800 MB VRAM  →  extract  →  free
    Load YOLOv12-Nano          →  ~200 MB VRAM  →  extract  →  free
    Load EfficientNet-V2-S     →  ~400 MB VRAM  →  extract  →  free

Phase 2 (Training — iterative):
    Load Fusion Head only      →  ~10 MB params
    + Optimizer states (8-bit) →  ~20 MB
    + Batch activations (AMP)  →  ~500 MB
    ─────────────────────────────────────
    Total:                     →  < 1 GB VRAM ✓
```

### 4-bit NF4 Quantisation (bitsandbytes)

VideoMAE-Base has 86M parameters.  At FP32: 344 MB.  At NF4 with double quantisation:

$$\text{VRAM} = \frac{86 \times 10^6 \times 4\text{-bit}}{8} + \text{scales} \approx 55 \text{ MB}$$

Combined with the FP16 compute dtype, inference VRAM peaks at ~800 MB — safely within the 6 GB budget.

### Mixed Precision + Gradient Accumulation

```python
# AMP keeps forward/backward in float16 (halves activation memory)
with torch.amp.autocast("cuda", dtype=torch.float16):
    out = model(motion, spatial, context)
    loss = focal_loss(out["frame_logits"], labels) / 4  # accum steps

# Gradient Accumulation: batch_size=8 × 4 = effective batch of 32
# without ever holding 32 samples in VRAM simultaneously
```

---

## Data Stratification Matrix

### The Data Leakage Problem

A naive `train_test_split` on individual feature files (or frames) causes **severe data leakage** because:

1. **Temporal continuity** — Frames 1–16 and 17–32 of the same 40-second clip share identical vehicles, weather, and road geometry.
2. **Tracklet persistence** — ByteTrack IDs persist across the entire clip. If any frames from video `X` appear in training, the model has already "seen" the accident trajectory.
3. **Contextual bleed** — EfficientNet scene embeddings are near-identical within a single clip.

### Our Solution: Stratified Group Shuffle Split

| Property | Implementation |
|----------|---------------|
| **Grouping key** | `video_id` — all 32 frames from one clip stay in the same split |
| **Stratification** | `target × weather × light_conditions` — preserves crash ratio and environment diversity |
| **Ratios** | 70% Train / 15% Validation / 15% Test |
| **Overlap** | Verified zero video-ID intersection via assertion checks |

```
Split Distribution (example for 1000 clips):
┌─────────┬───────┬───────┬────────┬──────────────────────────┐
│ Split   │ Total │ Crash │ Normal │ Environment coverage     │
├─────────┼───────┼───────┼────────┼──────────────────────────┤
│ Train   │  700  │  105  │   595  │ day-clear, night-rain, … │
│ Val     │  150  │   23  │   127  │ proportionally matched   │
│ Test    │  150  │   22  │   128  │ proportionally matched   │
└─────────┴───────┴───────┴────────┴──────────────────────────┘
```

The split is executed in `preprocess.py` via `create_stratified_splits()` using `sklearn.model_selection.GroupShuffleSplit`.

---

## Installation & Usage

### 1. Environment Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Dataset Placement

```
DL-Project/
├── data/nexar/
│   ├── train.csv              # Full Nexar metadata CSV
│   └── videos/
│       ├── vid_001.mp4
│       └── ...
├── preprocess.py
├── model.py
├── train.py
├── eval.py
├── run_pipeline.py            # ← Master orchestrator
├── requirements.txt
└── README.md
```

**CSV Schema:**
```
video_id, target, event_type, time_of_alert, time_of_event, light_conditions, weather, scene
```

### 3. Automated Pipeline (Recommended)

Run the entire lifecycle with a single command:

```bash
# Full pipeline: split → extract → train → evaluate
python run_pipeline.py --data_root ./data/nexar --csv ./data/nexar/train.csv

# Skip extraction if features already exist
python run_pipeline.py --skip_extract

# Evaluation only (with demo video)
python run_pipeline.py --eval_only --demo_video data/nexar/videos/clip.mp4

# Custom hyperparameters
python run_pipeline.py --epochs 80 --batch_size 4 --lr 1e-4
```

### 4. Manual Stage-by-Stage Execution

```bash
# Stage 2: Stratified split (creates metadata/train.csv, val.csv, test.csv)
python preprocess.py --csv ./data/nexar/train.csv --do_split

# Stage 3: Feature extraction per split
python preprocess.py --data_root ./data/nexar --csv metadata/train.csv --split train
python preprocess.py --data_root ./data/nexar --csv metadata/val.csv --split val
python preprocess.py --data_root ./data/nexar --csv metadata/test.csv --split test

# Stage 4: Training
python train.py --train_dir features/train --val_dir features/val --epochs 50

# Stage 5: Evaluation
python eval.py --feature_dir features/test --checkpoint checkpoints/best.pt

# Stage 6: Demo video
python eval.py --visualize --video data/nexar/videos/clip.mp4 \
               --checkpoint checkpoints/best.pt --output demo.mp4
```

---

## Hardware Profiling — RTX 4050 (6 GB VRAM)

### VRAM Budget Per Stage

| Stage | Component | VRAM Usage | Fits 6 GB? |
|-------|-----------|-----------|------------|
| **Feature Extraction** | VideoMAE-Base (NF4 4-bit) | ~800 MB | ✅ |
| | YOLOv12-Nano | ~200 MB | ✅ |
| | EfficientNet-V2-S | ~400 MB | ✅ |
| | *Max at any time* | *~800 MB* | ✅ |
| **Training** | Fusion Head params (2.5M) | ~10 MB | ✅ |
| | Optimizer states (8-bit) | ~20 MB | ✅ |
| | Batch activations (AMP f16) | ~500 MB | ✅ |
| | Gradient accumulation buffer | ~200 MB | ✅ |
| | *Peak training VRAM* | *~730 MB* | ✅ |
| **Inference** | Fusion Head + one video | ~50 MB | ✅ |

Key optimisations that keep us under budget:
- **Feature Bottlenecking**: Backbones never loaded during training
- **Sequential loading**: Only one expert in VRAM at a time during extraction
- **NF4 quantisation**: VideoMAE-Base 86M params → ~55 MB instead of 344 MB
- **AMP float16**: Halves activation memory during forward/backward
- **8-bit AdamW**: Halves optimizer state memory
- **Gradient accumulation ×4**: Effective batch=32 without holding 32 samples

---

## File Structure & Data Flow

```
preprocess.py                          model.py
─────────────                          ────────
VideoMAE  → motion  (32, 768)  ─┐      ┌─ Projector → Cross-Attention ─┐
YOLO+BT   → spatial (32, 5,10) ─┼─.npy─┤                              ├─ TCN → Scores
EfficientNet→context(32,1280)  ─┘      └─ Projector ──────────────────┘

train.py                              eval.py
────────                              ───────
Loads .npy → FocalLoss + AMP          Loads .npy → STAUC + TTA metrics
OneCycleLR + GradAccum(4)             OR: raw .mp4 → full pipeline → demo.mp4

run_pipeline.py
───────────────
Orchestrates: env check → dirs → split → extract → train → eval → demo
```

---

## Citation

```bibtex
@thesis{traffic_anomaly_2026,
    title   = {Tri-Expert Ensemble with Cross-Attention MS-TCN for
               Egocentric Traffic Anomaly Detection and Anticipation},
    year    = {2026},
    note    = {Undergraduate Thesis. Nexar Dashcam Collision Prediction Dataset.
               Optimised for NVIDIA RTX 4050 (6 GB VRAM).}
}
```

---

## License

This project is for academic and research purposes.  The Nexar dataset is subject to its own license terms from the Kaggle competition.
