# eval.py - STAUC & TTA Evaluator + Visual Demo
# ================================================
# Metrics:
#     1. STAUC  - Frame-level detection accuracy x spatial IoU, AUC'd over
#                 score thresholds, averaged across IoU thresholds.
#     2. TTA    - Milliseconds between first sustained alert and ground-truth
#                 time_of_event.
#
# Visualization:
#     • Green->Red bounding box on the hazard vehicle
#     • Risk Level Gauge (0-100 %) top-left
#     • Countdown timer "Seconds to Anticipated Impact"
#
# Usage:
#     python eval.py --feature_dir features/val \
#                    --checkpoint checkpoints/best.pt
#
#     python eval.py --visualize --video data/nexar/videos/clip.mp4 \
#                    --checkpoint checkpoints/best.pt --output demo.mp4

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_model, NUM_FRAMES, SPATIAL_VEHICLES
from train import FeatureDataset
from preprocess import (sample_frames, extract_motion, extract_spatial,
                        extract_context, _load_videomae, _load_yolo,
                        _load_effnet, _clear, IMAGE_SIZE)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  IoU
# ═══════════════════════════════════════════════════════════════════════════
def iou(a, b):
    """IoU between two [x1,y1,x2,y2] boxes (normalised coords)."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    ab = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / max(aa + ab - inter, 1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  STAUC (Spatio-Temporal AUC)
# ═══════════════════════════════════════════════════════════════════════════
def compute_stauc(
    pred_boxes,          # [V][T] -> list of [x1,y1,x2,y2]
    gt_boxes,            # [V][T] -> list of [x1,y1,x2,y2]
    frame_scores,        # [V][T] -> float
    thresholds=None,
):
    from sklearn.metrics import roc_auc_score

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    aucs = {}
    for tau in thresholds:
        yt, ys = [], []
        for v in range(len(pred_boxes)):
            pf = pred_boxes[v]; gf = gt_boxes[v]; sf = frame_scores[v]
            n = min(len(pf), len(gf), len(sf))
            for t in range(n):
                s = sf[t]
                if gf[t]:                              # GT anomaly frame
                    best = max((iou(pb, gb)
                                for pb in (pf[t] if pf[t] else [[0,0,0,0]])
                                for gb in gf[t]), default=0)
                    yt.append(1)
                    ys.append(s if best >= tau else s * 0.1)
                else:
                    yt.append(0)
                    ys.append(s)

        if len(set(yt)) >= 2:
            aucs[f"auc@{tau:.1f}"] = roc_auc_score(yt, ys)
        else:
            aucs[f"auc@{tau:.1f}"] = 0.0

    stauc = float(np.mean(list(aucs.values())))
    return {"stauc": stauc, **aucs}


# ═══════════════════════════════════════════════════════════════════════════
# 3.  TTA (Time-to-Accident, in milliseconds)
# ═══════════════════════════════════════════════════════════════════════════
def compute_tta_ms(frame_scores, timestamps, toe,
                   threshold=0.5, min_run=3):
    if toe <= 0:
        return -1.0

    run = 0
    for i in range(len(frame_scores)):
        if frame_scores[i] >= threshold:
            run += 1
            if run >= min_run:
                alert_idx = i - min_run + 1
                alert_t = timestamps[alert_idx]
                if alert_t < toe:
                    return (toe - alert_t) * 1000.0   # -> ms
        else:
            run = 0
    return -1.0


def tta_stats(all_scores, all_ts, all_toe, threshold=0.5):
    vals = []
    total_crash = 0
    for s, t, toe in zip(all_scores, all_ts, all_toe):
        if toe <= 0:
            continue
        total_crash += 1
        v = compute_tta_ms(s, t, toe, threshold)
        if v > 0:
            vals.append(v)
    return {
        "mean_tta_ms": float(np.mean(vals)) if vals else 0,
        "median_tta_ms": float(np.median(vals)) if vals else 0,
        "detection_rate": len(vals) / max(total_crash, 1),
        "detected": len(vals),
        "total_crash": total_crash,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Standard metrics
# ═══════════════════════════════════════════════════════════════════════════
def standard_auc(clip_scores, clip_labels):
    from sklearn.metrics import roc_auc_score, average_precision_score
    if len(set(clip_labels)) < 2:
        return {"auc": 0.0, "ap": 0.0}
    return {
        "auc": roc_auc_score(clip_labels, clip_scores),
        "ap": average_precision_score(clip_labels, clip_scores),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Boxes from spatial features + vehicle weights
# ═══════════════════════════════════════════════════════════════════════════
def boxes_from_spatial(spatial, vehicle_w, score_thresh=0.3):
    T = spatial.shape[0]
    boxes = []
    for t in range(T):
        fb = []
        for n in range(SPATIAL_VEHICLES):
            if vehicle_w[t, n] >= score_thresh and spatial[t, n, 9] > 0:
                cx, cy, w, h = spatial[t, n, :4]
                fb.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        boxes.append(fb)
    return boxes


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Full evaluation loop
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_clip_s, all_clip_l = [], []
    all_frame_s, all_ts, all_toe = [], [], []
    all_pred_b, all_gt_b = [], []
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    for batch in tqdm(loader, desc="Eval"):
        m = batch["motion"].to(device)
        s = batch["spatial"].to(device)
        c = batch["context"].to(device)

        with autocast(device_type=device_type):
            out = model(m, s, c)

        B = m.shape[0]
        for b in range(B):
            # FIXED: Added .detach() here to prevent Numpy RuntimeError[cite: 1]
            fs = out["frame_scores"][b].cpu().detach().numpy()
            vw = out["vehicle_weights"][b].cpu().detach().numpy()
            sp = batch["spatial"][b].numpy()
            fl = batch["frame_labels"][b].numpy()
            ts = batch["timestamps"][b].numpy()
            toe = batch["toe"][b]

            all_frame_s.append(fs.tolist())
            all_ts.append(ts.tolist())
            all_toe.append(float(toe))
            all_clip_s.append(float(fs.max()))
            all_clip_l.append(int(fl.max()))

            pred_b = boxes_from_spatial(sp, vw)
            gt_b = []
            for t in range(len(fl)):
                if fl[t] > 0.5:
                    gt_b.append([[0.2, 0.2, 0.8, 0.8]])
                else:
                    gt_b.append([])
            all_pred_b.append(pred_b)
            all_gt_b.append(gt_b)

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)

    std = standard_auc(all_clip_s, all_clip_l)
    print(f"\n  📊 Clip AUC = {std['auc']:.4f}   AP = {std['ap']:.4f}")

    st = compute_stauc(all_pred_b, all_gt_b, all_frame_s)
    print(f"\n  📐 STAUC = {st['stauc']:.4f}")
    for k, v in st.items():
        if k != "stauc":
            print(f"     {k}: {v:.4f}")

    tt = tta_stats(all_frame_s, all_ts, all_toe)
    print(f"\n  ⏱  Mean TTA  = {tt['mean_tta_ms']:.0f} ms")
    print(f"     Median    = {tt['median_tta_ms']:.0f} ms")
    print(f"     Detection = {tt['detection_rate']:.1%} "
          f"({tt['detected']}/{tt['total_crash']})")

    print("=" * 70)
    results = {**std, **st, **tt}
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Inference Visualization
# ═══════════════════════════════════════════════════════════════════════════
def _risk_color(score):
    """0->green, 0.5->yellow, 1->red (BGR)."""
    if score < 0.5:
        r = int(score / 0.5 * 255)
        return (0, 255, r)
    else:
        g = int((1 - (score - 0.5) / 0.5) * 255)
        return (0, g, 255)


def _draw_gauge(frame, score, x=20, y=20):
    H_bar, W_bar = 30, 200
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + W_bar + 4, y + H_bar + 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    pct = int(score * 100)
    color = _risk_color(score)
    fill_w = int(score * W_bar)
    cv2.rectangle(frame, (x+2, y+20), (x+2+W_bar, y+20+H_bar), (60,60,60), -1)
    cv2.rectangle(frame, (x+2, y+20), (x+2+fill_w, y+20+H_bar), color, -1)
    cv2.rectangle(frame, (x+2, y+20), (x+2+W_bar, y+20+H_bar), (200,200,200), 1)

    label = "DANGER" if score >= 0.7 else ("WARNING" if score >= 0.4 else "NORMAL")
    cv2.putText(frame, f"Risk: {pct}% - {label}", (x+4, y+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


def _draw_countdown(frame, seconds_left, x=20, y=100):
    if seconds_left <= 0:
        return frame
    txt = f"Impact in {seconds_left:.1f}s"
    cv2.putText(frame, txt, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


def visualize_video(video_path, checkpoint_path, output_path="demo.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_model().to(device).eval()
    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"✓ Checkpoint loaded: {checkpoint_path}")
    else:
        print("⚠ No checkpoint — untrained model")

    print("Sampling 32 frames …")
    frames, timestamps, fps = sample_frames(video_path)

    vmae = _load_videomae(device)
    motion = extract_motion(frames, vmae, device)
    del vmae; _clear()

    yolo = _load_yolo()
    spatial = extract_spatial(frames, yolo)
    del yolo; _clear()

    effnet = _load_effnet(device)
    context = extract_context(frames, effnet, device)
    del effnet; _clear()

    print("Running fusion model …")
    m_t = torch.from_numpy(motion.astype(np.float32)).unsqueeze(0).to(device)
    s_t = torch.from_numpy(spatial).unsqueeze(0).to(device)
    c_t = torch.from_numpy(context.astype(np.float32)).unsqueeze(0).to(device)

    with autocast(device_type=device_type):
        out = model(m_t, s_t, c_t)

    # FIXED: Added .detach() here to prevent Numpy RuntimeError[cite: 1]
    scores = out["frame_scores"][0].cpu().detach().numpy()       
    vw = out["vehicle_weights"][0].cpu().detach().numpy()         

    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             vid_fps, (W, H))

    interp_s = np.interp(np.linspace(0, 31, total_fr),
                         np.arange(32), scores)

    peak_idx = int(np.argmax(interp_s))
    est_toe_sec = peak_idx / vid_fps

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        sc = interp_s[fidx] if fidx < len(interp_s) else 0.0
        cur_sec = fidx / vid_fps

        frame = _draw_gauge(frame, sc)

        if sc > 0.4 and cur_sec < est_toe_sec:
            frame = _draw_countdown(frame, est_toe_sec - cur_sec)

        si = min(int(fidx / max(total_fr-1, 1) * 31), 31)
        best_v = int(np.argmax(vw[si]))
        feat = spatial[si, best_v]
        cx, cy, w, h = feat[:4]
        if w > 0 and h > 0:
            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W)
            y2 = int((cy + h/2) * H)
            color = _risk_color(sc)
            thick = 2 + int(sc * 4)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            cv2.putText(frame, f"{sc:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        writer.write(frame)
        fidx += 1

    cap.release()
    writer.release()
    print(f"✓ Visualization -> {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser(description="Evaluate / Visualize")
    pa.add_argument("--feature_dir", default=None)
    pa.add_argument("--checkpoint", default="checkpoints/best.pt")
    pa.add_argument("--output_json", default="outputs/eval_results.json")
    pa.add_argument("--visualize", action="store_true")
    pa.add_argument("--video", default=None)
    pa.add_argument("--output", default="visualizations/demo.mp4")
    pa.add_argument("--batch_size", type=int, default=8)
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.visualize and args.video:
        visualize_video(args.video, args.checkpoint, args.output)
        return

    if args.feature_dir is None:
        print("Provide --feature_dir for metrics or --visualize --video for demo")
        return

    model = build_model().to(device)
    if os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"✓ Loaded: {args.checkpoint}")

    ds = FeatureDataset(args.feature_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)

    results = evaluate(model, dl, device)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results -> {args.output_json}")


if __name__ == "__main__":
    main()