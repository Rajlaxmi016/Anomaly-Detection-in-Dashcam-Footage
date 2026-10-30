"""
train.py — Hardware-Aware Training Loop
=========================================
Trains the lightweight FusionMSTCN head on cached .npy features.
No backbone in VRAM — peak usage < 4 GB on RTX 4050.

Key features:
    • Focal Loss  (handles massive normal-vs-crash imbalance)
    • AMP float16  (torch.amp.autocast)
    • Gradient Accumulation  (4 steps → effective batch = 32)
    • OneCycleLR  scheduler
    • 8-bit AdamW  via bitsandbytes (halves optimizer state)

Usage:
    python train.py --train_dir features/train --val_dir features/val
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from model import build_model, NUM_FRAMES, MOTION_DIM, SPATIAL_VEHICLES, SPATIAL_FEAT, CONTEXT_DIM

# ── Hyperparameters ──────────────────────────────────────────────────────
LR = 3e-4
WEIGHT_DECAY = 0.01
EPOCHS = 50
BATCH_SIZE = 8
ACCUM_STEPS = 4           # effective batch = 32
GRAD_CLIP = 1.0
FOCAL_ALPHA = 0.75        # upweight crash frames
FOCAL_GAMMA = 2.0


# ═══════════════════════════════════════════════════════════════════════════
# 1. Dataset — loads .npy feature files
# ═══════════════════════════════════════════════════════════════════════════
class FeatureDataset(Dataset):
    """Maps saved .npy feature bundles to tensors for the fusion head.

    Each .npy (saved with allow_pickle) is a dict:
        motion       (32, 768) f16
        spatial      (32, 5, 10) f32
        context      (32, 1280) f16
        frame_labels (32,) i8       ← 0 before TOA, 1 from TOA onward
        toa, toe     float32
        target       int8
    """

    def __init__(self, feature_dir: str):
        self.paths = sorted(Path(feature_dir).glob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No .npy in {feature_dir}")

        # quick label scan
        labels = []
        for p in self.paths:
            d = np.load(p, allow_pickle=True).item()
            labels.append(int(d.get("target", 0)))
        n1 = sum(labels)
        print(f"[Dataset] {len(self.paths)} clips  |  crash={n1}  normal={len(labels)-n1}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        d = np.load(self.paths[idx], allow_pickle=True).item()
        motion = torch.from_numpy(d["motion"].astype(np.float32))        # (32,768)
        spatial = torch.from_numpy(d["spatial"].astype(np.float32))      # (32,5,10)
        context = torch.from_numpy(d["context"].astype(np.float32))      # (32,1280)
        frame_labels = torch.from_numpy(d["frame_labels"].astype(np.float32))  # (32,)
        toa = float(d.get("toa", -1.0))
        toe = float(d.get("toe", -1.0))
        timestamps = torch.from_numpy(d.get("timestamps",
                                             np.arange(NUM_FRAMES, dtype=np.float32)))
        return {
            "motion": motion,
            "spatial": spatial,
            "context": context,
            "frame_labels": frame_labels,
            "toa": toa,
            "toe": toe,
            "timestamps": timestamps,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Focal Loss
# ═══════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    r"""Focal Loss for binary classification.

    FL(p_t) = -α_t (1 − p_t)^γ  log(p_t)

    α = 0.75 upweights the rare crash frames.
    γ = 2.0  down-weights easy normal frames.
    """

    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits, targets: same shape — e.g. (B, 32)."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.sigmoid(logits)
        pt = pt * targets + (1 - pt) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Optimizer builder
# ═══════════════════════════════════════════════════════════════════════════
def make_optimizer(model, lr=LR, wd=WEIGHT_DECAY):
    params = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=wd)
        print("[Optim] 8-bit AdamW (bitsandbytes)")
    except ImportError:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        print("[Optim] standard AdamW")
    return opt


# ═══════════════════════════════════════════════════════════════════════════
# 4. Checkpointing
# ═══════════════════════════════════════════════════════════════════════════
def save_ckpt(model, optimizer, scheduler, epoch, metrics, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
    }, path)
    # keep only last 3 periodic checkpoints
    folder = Path(path).parent
    for old in sorted(folder.glob("epoch_*.pt"))[:-3]:
        old.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Training epoch
# ═══════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, scheduler, criterion, scaler,
                device, epoch):
    model.train()
    total_loss = 0.0
    n = 0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}", leave=False)

    for step, batch in enumerate(pbar):
        m = batch["motion"].to(device)
        s = batch["spatial"].to(device)
        c = batch["context"].to(device)
        labels = batch["frame_labels"].to(device)

        with autocast("cuda", dtype=torch.float16):
            out = model(m, s, c)
            loss = criterion(out["frame_logits"], labels) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * ACCUM_STEPS
        n += 1
        pbar.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.4f}")

    torch.cuda.empty_cache()
    return total_loss / max(n, 1)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Validation
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def validate(model, loader, device):
    from sklearn.metrics import roc_auc_score
    model.eval()
    all_s, all_l = [], []

    for batch in loader:
        m = batch["motion"].to(device)
        s = batch["spatial"].to(device)
        c = batch["context"].to(device)
        labels = batch["frame_labels"]

        with autocast("cuda", dtype=torch.float16):
            out = model(m, s, c)

        scores = out["frame_scores"].cpu()
        for b in range(scores.shape[0]):
            all_s.extend(scores[b].tolist())
            all_l.extend(labels[b].tolist())

    # clip-level AUC (max score per clip)
    if len(set(int(l) for l in all_l)) >= 2:
        auc = roc_auc_score([int(l) for l in all_l], all_s)
    else:
        auc = 0.0

    torch.cuda.empty_cache()
    return {"frame_auc": auc}


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser(description="Train Fusion Head")
    pa.add_argument("--train_dir", default="features/train")
    pa.add_argument("--val_dir", default=None)
    pa.add_argument("--epochs", type=int, default=EPOCHS)
    pa.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    pa.add_argument("--lr", type=float, default=LR)
    pa.add_argument("--ckpt_dir", default="checkpoints")
    pa.add_argument("--resume", default=None)
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"  Fusion Head Training  |  eff. batch={args.batch_size * ACCUM_STEPS}")
    print("=" * 70)

    model = build_model().to(device)

    train_ds = FeatureDataset(args.train_dir)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)

    val_dl = None
    if args.val_dir and os.path.isdir(args.val_dir):
        val_ds = FeatureDataset(args.val_dir)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    optimizer = make_optimizer(model, lr=args.lr)
    total_steps = len(train_dl) * args.epochs // ACCUM_STEPS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(total_steps, 1),
        pct_start=0.1, anneal_strategy="cos",
    )
    criterion = FocalLoss()
    scaler = GradScaler()

    start = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start = ckpt["epoch"] + 1
        print(f"  ✓ Resumed from epoch {ckpt['epoch']}")

    best_auc = 0.0
    for epoch in range(start, args.epochs + 1):
        t0 = time.time()

        loss = train_epoch(model, train_dl, optimizer, scheduler,
                           criterion, scaler, device, epoch)

        val = {"frame_auc": 0.0}
        if val_dl:
            val = validate(model, val_dl, device)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}  |  loss={loss:.4f}  "
              f"val_AUC={val['frame_auc']:.4f}  "
              f"lr={lr_now:.2e}  |  {time.time()-t0:.1f}s")

        if epoch % 5 == 0 or val["frame_auc"] > best_auc:
            p = os.path.join(args.ckpt_dir, f"epoch_{epoch:03d}.pt")
            save_ckpt(model, optimizer, scheduler, epoch,
                      {"loss": loss, **val}, p)

            if val["frame_auc"] > best_auc:
                best_auc = val["frame_auc"]
                bp = os.path.join(args.ckpt_dir, "best.pt")
                save_ckpt(model, optimizer, scheduler, epoch,
                          {"loss": loss, **val, "best_auc": best_auc}, bp)
                print(f"  🏆 Best AUC: {best_auc:.4f}")

        torch.cuda.empty_cache()

    print(f"\n✓ Done. Best frame AUC = {best_auc:.4f}")


if __name__ == "__main__":
    main()
