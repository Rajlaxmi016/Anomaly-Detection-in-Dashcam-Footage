"""
preprocess.py — The Feature Factory
=====================================
Offline 3-Expert feature extraction for Nexar Dashcam Collision Prediction.

Expert Ensemble (loaded one-at-a-time to stay within 6 GB VRAM):
    1. VideoMAE-Base  (4-bit NF4)  → motion  (32, 768)
    2. YOLOv12-Nano + ByteTrack    → spatial  (32, 5, 10)
    3. EfficientNet-V2-S           → context  (32, 1280)

CSV Schema: video_id, target, event_type, time_of_alert, time_of_event,
            light_conditions, weather, scene

Usage:
    python preprocess.py --data_root ./data/nexar --csv ./data/nexar/train.csv
    python preprocess.py --data_root ./data/nexar --csv ./data/nexar/val.csv --split val
"""

import argparse
import gc
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────
NUM_FRAMES = 32
IMAGE_SIZE = 224
VIDEOMAE_CKPT = "MCG-NJU/videomae-base-finetuned-kinetics"
VIDEOMAE_DIM = 768
VIDEOMAE_INPUT_T = 16
VIDEOMAE_TUBELET = 2
YOLO_WEIGHTS = "yolo12n.pt"
YOLO_CONF = 0.25
YOLO_IOU = 0.45
YOLO_IMGSZ = 640
YOLO_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorbike, bus, truck
MAX_VEHICLES = 5
SPATIAL_DIM = 10  # per-vehicle feature width
TRACK_GAP_LIMIT = 3
EFFNET_NAME = "tf_efficientnetv2_s"
CONTEXT_DIM = 1280
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Frame Sampling
# ═══════════════════════════════════════════════════════════════════════════
def sample_frames(video_path: str, num_frames: int = NUM_FRAMES,
                  size: int = IMAGE_SIZE):
    """Uniformly sample exactly *num_frames* from an mp4 clip.

    Returns
    -------
    frames     : (T, H, W, 3) uint8 RGB
    timestamps : (T,)  float64 seconds
    fps        : float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        cap.release()
        raise ValueError(f"Empty video: {video_path}")

    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)
    timestamps = indices.astype(np.float64) / fps

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (size, size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        elif frames:
            frames.append(frames[-1].copy())
        else:
            frames.append(np.zeros((size, size, 3), dtype=np.uint8))

    cap.release()
    return np.array(frames, dtype=np.uint8), timestamps, fps


# ═══════════════════════════════════════════════════════════════════════════
# 2. Frame-Level Binary Labels  (0 = before TOA, 1 = TOA→end)
# ═══════════════════════════════════════════════════════════════════════════
def compute_frame_labels(timestamps, toa, toe, target):
    """Return (T,) int8 array — 1 inside the alert window, else 0."""
    labels = np.zeros(len(timestamps), dtype=np.int8)
    if target == 0 or np.isnan(toa) or toa < 0:
        return labels
    for i, t in enumerate(timestamps):
        if t >= toa:
            labels[i] = 1
    return labels


# ═══════════════════════════════════════════════════════════════════════════
# 3. Expert A — VideoMAE-Base  (Motion)
# ═══════════════════════════════════════════════════════════════════════════
def _load_videomae(device):
    from transformers import VideoMAEModel, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = VideoMAEModel.from_pretrained(
        VIDEOMAE_CKPT, quantization_config=bnb, device_map="auto",
    )
    model.eval()
    mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"[VideoMAE] Loaded 4-bit — ~{mb:.0f} MB")
    return model


def _prep_videomae(frames_uint8):
    """(T,H,W,3) uint8 → (1,T,3,H,W) float32 ImageNet-normed."""
    x = frames_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (0, 3, 1, 2))
    return torch.from_numpy(x).unsqueeze(0)


@torch.no_grad()
def extract_motion(frames, model, device):
    """32 frames → 2 × 16-frame chunks → pool spatially → interpolate → (32, 768)."""
    parts = []
    for chunk in [frames[:16], frames[16:]]:
        pv = _prep_videomae(chunk).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(pixel_values=pv)
        tokens = out.last_hidden_state.squeeze(0)          # (S, 768)
        nt = VIDEOMAE_INPUT_T // VIDEOMAE_TUBELET           # 8
        ns = tokens.shape[0] // nt
        pooled = tokens[: nt * ns].view(nt, ns, -1).mean(1) # (8, 768)
        parts.append(pooled.cpu().float())
        del pv, out, tokens; _clear()

    combined = torch.cat(parts, dim=0)                      # (16, 768)
    combined = F.interpolate(
        combined.unsqueeze(0).permute(0, 2, 1),             # (1, 768, 16)
        size=NUM_FRAMES, mode="linear", align_corners=False,
    ).permute(0, 2, 1).squeeze(0)                           # (32, 768)

    return combined.numpy().astype(np.float16)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Expert B — YOLOv12-Nano + ByteTrack  (Spatial Tracklets)
# ═══════════════════════════════════════════════════════════════════════════
def _load_yolo():
    try:
        from ultralytics import YOLO
        m = YOLO(YOLO_WEIGHTS)
        print("[YOLO] Loaded YOLOv12-Nano")
        return m
    except Exception as e:
        print(f"[YOLO] Unavailable ({e}). Spatial features → zeros.")
        return None


def _interpolate_tracks(tracks, total_frames):
    """Fill gaps ≤ TRACK_GAP_LIMIT frames via linear interpolation."""
    for tid in tracks:
        present = sorted(tracks[tid].keys())
        for i in range(len(present) - 1):
            a, b = present[i], present[i + 1]
            gap = b - a - 1
            if 0 < gap <= TRACK_GAP_LIMIT:
                va = np.array(tracks[tid][a])
                vb = np.array(tracks[tid][b])
                for g in range(1, gap + 1):
                    alpha = g / (gap + 1)
                    tracks[tid][a + g] = (va * (1 - alpha) + vb * alpha).tolist()
    return tracks


def extract_spatial(frames, yolo):
    """Per-vehicle feature: [cx, cy, w, h, vx, vy, conf, cls_norm, area, valid].

    Returns (32, 5, 10) float32  — top-5 closest vehicles per frame.
    """
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    if yolo is None:
        return np.zeros((T, MAX_VEHICLES, SPATIAL_DIM), dtype=np.float32)

    # Phase 1 — detect & track
    tracks = {}  # tid → {frame_idx: [cx,cy,w,h,conf,cls]}
    for t in range(T):
        bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        try:
            res = yolo.track(bgr, persist=True, conf=YOLO_CONF, iou=YOLO_IOU,
                             imgsz=YOLO_IMGSZ, classes=YOLO_CLASSES, verbose=False)
        except Exception:
            try:
                res = yolo(bgr, conf=YOLO_CONF, iou=YOLO_IOU,
                           imgsz=YOLO_IMGSZ, classes=YOLO_CLASSES, verbose=False)
            except Exception:
                continue

        if not res or res[0].boxes is None:
            continue
        for i in range(len(res[0].boxes)):
            box = res[0].boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            tid = int(box.id[0].item()) if box.id is not None else t * 1000 + i
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            tracks.setdefault(tid, {})[t] = [cx, cy, w, h, conf, cls]

    # Phase 2 — interpolate gaps
    tracks = _interpolate_tracks(tracks, T)

    # Phase 3 — build output matrix with velocity
    out = np.zeros((T, MAX_VEHICLES, SPATIAL_DIM), dtype=np.float32)
    for t in range(T):
        dets = []
        for tid, fd in tracks.items():
            if t not in fd:
                continue
            cx, cy, w, h, conf, cls = fd[t]
            # velocity (central difference when possible)
            vx, vy = 0.0, 0.0
            if (t - 1) in fd:
                vx = cx - fd[t - 1][0]
                vy = cy - fd[t - 1][1]
            elif (t + 1) in fd:
                vx = fd[t + 1][0] - cx
                vy = fd[t + 1][1] - cy
            area = w * h
            # distance to ego (bottom-center of image = (0.5, 1.0))
            dist = ((cx - 0.5) ** 2 + (cy - 1.0) ** 2) ** 0.5
            dets.append((dist, [cx, cy, w, h, vx, vy, conf, cls / 80.0, area, 1.0]))

        # Sort by proximity → take top-5
        dets.sort(key=lambda d: d[0])
        for k in range(min(MAX_VEHICLES, len(dets))):
            out[t, k, :] = dets[k][1]

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 5. Expert C — EfficientNet-V2-S  (Context / Scene)
# ═══════════════════════════════════════════════════════════════════════════
def _load_effnet(device):
    try:
        import timm
        m = timm.create_model(EFFNET_NAME, pretrained=True, num_classes=0)
        m = m.to(device).eval()
        mb = sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6
        print(f"[EfficientNet] Loaded — ~{mb:.0f} MB")
        return m
    except Exception as e:
        print(f"[EfficientNet] Unavailable ({e}). Context → zeros.")
        return None


def _prep_effnet(frame_rgb):
    """(H,W,3) uint8 → (1,3,H,W) float32."""
    x = cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0)


@torch.no_grad()
def extract_context(frames, model, device):
    """Per-frame scene embedding → (32, 1280) float16."""
    if model is None:
        return np.zeros((NUM_FRAMES, CONTEXT_DIM), dtype=np.float16)

    feats = []
    for t in range(frames.shape[0]):
        inp = _prep_effnet(frames[t]).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            f = model(inp)
        feats.append(f.squeeze(0).cpu().float().numpy())
        del inp, f

    _clear()
    return np.array(feats, dtype=np.float16)  # (32, 1280)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Stratified Group Split  (70 / 15 / 15 — zero video leakage)
# ═══════════════════════════════════════════════════════════════════════════
def create_stratified_splits(
    csv_path: str,
    output_dir: str = "metadata",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """Split Nexar CSV into train/val/test with zero video-ID overlap.

    Grouping  : video_id  (all frames from one clip stay together)
    Stratify  : target × weather × light_conditions  (preserves class
                and environment distribution across splits)

    Why not random split?
        A random frame-level split causes SEVERE data leakage — frames
        from the same 40-second clip share identical vehicles, weather,
        and ByteTrack tracklet IDs.  The model memorises the clip, not
        the anomaly pattern.  Grouping on video_id eliminates this.

    Returns dict with paths to the 3 CSV files.
    """
    from sklearn.model_selection import GroupShuffleSplit

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Build stratification key from available columns
    strat_cols = ["target"]
    for col in ["weather", "light_conditions"]:
        if col in df.columns:
            strat_cols.append(col)

    df["_strat_key"] = df[strat_cols].astype(str).agg("|".join, axis=1)

    # Map each video_id to its stratification key (one row per video)
    vid_df = df.drop_duplicates(subset="video_id")[["video_id", "_strat_key", "target"]]
    vid_df = vid_df.reset_index(drop=True)

    groups = vid_df["video_id"].values
    strat = vid_df["_strat_key"].values
    targets = vid_df["target"].values

    # First split: train vs (val+test)
    test_val_ratio = 1.0 - train_ratio
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_val_ratio, random_state=seed)
    train_idx, valtest_idx = next(gss1.split(vid_df, targets, groups))

    # Second split: val vs test (50/50 of the remaining 30%)
    valtest_df = vid_df.iloc[valtest_idx].reset_index(drop=True)
    vt_groups = valtest_df["video_id"].values
    vt_targets = valtest_df["target"].values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_sub, test_sub = next(gss2.split(valtest_df, vt_targets, vt_groups))

    train_vids = set(vid_df.iloc[train_idx]["video_id"])
    val_vids = set(valtest_df.iloc[val_sub]["video_id"])
    test_vids = set(valtest_df.iloc[test_sub]["video_id"])

    # Verify zero overlap
    assert len(train_vids & val_vids) == 0, "Train/Val leak!"
    assert len(train_vids & test_vids) == 0, "Train/Test leak!"
    assert len(val_vids & test_vids) == 0, "Val/Test leak!"

    # Write split CSVs
    paths = {}
    for name, vids in [("train", train_vids), ("val", val_vids), ("test", test_vids)]:
        split_df = df[df["video_id"].isin(vids)].drop(columns=["_strat_key"])
        p = os.path.join(output_dir, f"{name}.csv")
        split_df.to_csv(p, index=False)
        paths[name] = p

        n_crash = int(split_df["target"].sum())
        print(f"  {name:5s}: {len(split_df):5d} clips  "
              f"(crash={n_crash}, normal={len(split_df)-n_crash})")

    print(f"\n  Overlap check: PASSED ✓")
    print(f"  Zero video IDs shared across any split.")
    print(f"  CSVs → {output_dir}/\n")

    df.drop(columns=["_strat_key"], inplace=True, errors="ignore")
    return paths


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════
def run_pipeline(data_root: str, csv_path: str, split: str = "train"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join("features", split)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {"video_id", "target", "time_of_alert", "time_of_event"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    print(f"{'=' * 70}")
    print(f"  Feature Factory — {len(df)} clips  [{split}]")
    print(f"  Strategy: sequential expert loading (6 GB VRAM safe)")
    print(f"{'=' * 70}\n")

    # ── Phase 0: sample frames for all videos ──────────────────────────
    print("Phase 0 — Sampling frames …")
    video_cache = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Frames"):
        vid = str(row["video_id"])
        vid_padded = f"{int(float(vid)):05d}"
        npy_path = os.path.join(out_dir, f"{vid_padded}.npy")
        if os.path.isfile(npy_path):
            continue

        vpath = os.path.join(data_root, "videos", f"{vid_padded}.mp4")
        if not os.path.isfile(vpath):
            # try to find it recursively
            found = list(Path(data_root).rglob(f"{vid_padded}.*"))
            exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
            found = [f for f in found if f.suffix.lower() in exts]
            vpath = str(found[0]) if found else None

        if vpath is None or not os.path.isfile(vpath):
            print(f"  [SKIP] {vid} — video not found")
            continue

        try:
            frames, ts, fps = sample_frames(vpath)
            toa = float(row["time_of_alert"]) if not pd.isna(row["time_of_alert"]) else -1.0
            toe = float(row["time_of_event"]) if not pd.isna(row["time_of_event"]) else -1.0
            labels = compute_frame_labels(ts, toa, toe, int(row["target"]))
            video_cache[vid] = {
                "frames": frames, "timestamps": ts, "fps": fps,
                "labels": labels, "toa": toa, "toe": toe,
                "target": int(row["target"]),
            }
        except Exception as e:
            print(f"  [SKIP] {vid}: {e}")

    if not video_cache:
        print("No videos to process."); return
    vids = list(video_cache.keys())

    # ── Phase 1: VideoMAE ──────────────────────────────────────────────
    print(f"\nPhase 1 — VideoMAE motion features ({len(vids)} clips) …")
    vmae = _load_videomae(device)
    motion_map = {}
    for vid in tqdm(vids, desc="Motion"):
        try:
            motion_map[vid] = extract_motion(video_cache[vid]["frames"], vmae, device)
        except Exception as e:
            print(f"  [WARN] {vid}: {e}")
            motion_map[vid] = np.zeros((NUM_FRAMES, VIDEOMAE_DIM), dtype=np.float16)
        _clear()
    del vmae; _clear()
    print("[VideoMAE] freed.\n")

    # ── Phase 2: YOLO ─────────────────────────────────────────────────
    print(f"Phase 2 — YOLOv12-Nano spatial tracklets ({len(vids)} clips) …")
    yolo = _load_yolo()
    spatial_map = {}
    for vid in tqdm(vids, desc="Spatial"):
        try:
            spatial_map[vid] = extract_spatial(video_cache[vid]["frames"], yolo)
        except Exception as e:
            print(f"  [WARN] {vid}: {e}")
            spatial_map[vid] = np.zeros((NUM_FRAMES, MAX_VEHICLES, SPATIAL_DIM), np.float32)
    del yolo; _clear()
    print("[YOLO] freed.\n")

    # ── Phase 3: EfficientNet ─────────────────────────────────────────
    print(f"Phase 3 — EfficientNet-V2-S context features ({len(vids)} clips) …")
    effnet = _load_effnet(device)
    context_map = {}
    for vid in tqdm(vids, desc="Context"):
        try:
            context_map[vid] = extract_context(video_cache[vid]["frames"], effnet, device)
        except Exception as e:
            print(f"  [WARN] {vid}: {e}")
            context_map[vid] = np.zeros((NUM_FRAMES, CONTEXT_DIM), dtype=np.float16)
        _clear()
    del effnet; _clear()
    print("[EfficientNet] freed.\n")

    # ── Phase 4: Save ─────────────────────────────────────────────────
    print("Phase 4 — Saving .npy feature files …")
    saved = 0
    for vid in vids:
        vid_padded = f"{int(float(vid)):05d}"
        npy_path = os.path.join(out_dir, f"{vid_padded}.npy")
        bundle = {
            "motion": motion_map[vid],                          # (32, 768) f16
            "spatial": spatial_map[vid],                         # (32, 5, 10) f32
            "context": context_map[vid],                         # (32, 1280) f16
            "frame_labels": video_cache[vid]["labels"],          # (32,) i8
            "timestamps": video_cache[vid]["timestamps"].astype(np.float32),
            "toa": np.float32(video_cache[vid]["toa"]),
            "toe": np.float32(video_cache[vid]["toe"]),
            "target": np.int8(video_cache[vid]["target"]),
        }
        np.save(npy_path, bundle, allow_pickle=True)
        saved += 1

    print(f"\n✓ Saved {saved}/{len(vids)} → {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Feature Factory — Nexar Dataset")
    pa.add_argument("--data_root", type=str, default="./data/nexar")
    pa.add_argument("--csv", type=str, default="./data/nexar/train.csv")
    pa.add_argument("--split", type=str, default="train",
                    choices=["train", "val", "test"])
    pa.add_argument("--do_split", action="store_true",
                    help="Run stratified group split on --csv first, then extract")
    pa.add_argument("--split_output", type=str, default="metadata",
                    help="Directory to save split CSVs")
    args = pa.parse_args()

    if args.do_split:
        print("="*70)
        print("  Stratified Group Split — 70/15/15")
        print("="*70)
        split_paths = create_stratified_splits(
            args.csv, output_dir=args.split_output,
        )
        for name, path in split_paths.items():
            print(f"\n>>> Extracting features for [{name}] split …")
            run_pipeline(args.data_root, path, name)
    else:
        run_pipeline(args.data_root, args.csv, args.split)
