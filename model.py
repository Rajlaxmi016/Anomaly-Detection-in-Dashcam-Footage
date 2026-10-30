"""
model.py — Spatio-Temporal Fusion Head (Cross-Attention MS-TCN)
================================================================
Input tensors (from preprocess.py .npy files):
    motion  : (B, 32, 768)   — VideoMAE-Base global temporal descriptors
    spatial : (B, 32, 5, 10) — YOLOv12 top-5 vehicle tracklets
    context : (B, 32, 1280)  — EfficientNet-V2-S scene embeddings

Architecture:
    1. Per-expert linear projectors → shared 256-dim
    2. Cross-Attention: Q=Motion(global scene), K/V=Spatial(tracklets)
       → identifies which vehicles deviate from expected traffic flow
    3. Concatenation → 3-stage Dilated Causal TCN with residual connections
    4. Output Branch A: frame-level sigmoid anomaly score  (B, 32)
       Output Branch B: spatial weight mask over tracklets (B, 32, 5)

All parameters: ~2.5 M  →  ~10 MB VRAM (fits trivially on RTX 4050).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Dimensions (must match preprocess.py) ────────────────────────────────
MOTION_DIM = 768
SPATIAL_VEHICLES = 5
SPATIAL_FEAT = 10
CONTEXT_DIM = 1280
NUM_FRAMES = 32
HIDDEN = 256


# ═══════════════════════════════════════════════════════════════════════════
# 1. Dilated Causal Temporal Conv Block
# ═══════════════════════════════════════════════════════════════════════════
class DilatedResBlock(nn.Module):
    """Conv1D (dilated, causal) → BN → ReLU → Dropout → Residual."""

    def __init__(self, channels: int, kernel: int = 5,
                 dilation: int = 1, dropout: float = 0.3):
        super().__init__()
        self.pad = (kernel - 1) * dilation          # causal left-pad
        self.conv = nn.Conv1d(channels, channels, kernel,
                              dilation=dilation, padding=self.pad)
        self.bn = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, C, T)"""
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]             # remove future leak
        return x + self.drop(F.relu(self.bn(out)))


# ═══════════════════════════════════════════════════════════════════════════
# 2. TCN Stage (stack of exponentially dilated blocks)
# ═══════════════════════════════════════════════════════════════════════════
class TCNStage(nn.Module):
    def __init__(self, channels: int, num_layers: int = 4,
                 kernel: int = 5, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            DilatedResBlock(channels, kernel, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# 3. Cross-Attention  (Q = Motion-global, K/V = Spatial-tracklets)
# ═══════════════════════════════════════════════════════════════════════════
class TrackletCrossAttention(nn.Module):
    """Multi-head cross-attention that finds anomalously-deviating vehicles.

    Q : global scene motion   (B, T, D)
    K,V : per-vehicle spatial  (B, T, N, D_v) → projected to D

    Returns:
        attended : (B, T, D)  — motion refined by tracklet context
        weights  : (B, T, N)  — per-vehicle importance (for Branch B)
    """

    def __init__(self, d_model: int = HIDDEN, d_vehicle: int = SPATIAL_FEAT,
                 nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.dk = d_model // nhead

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_vehicle, d_model)
        self.wv = nn.Linear(d_vehicle, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_vehicle)
        self.drop = nn.Dropout(dropout)

    def forward(self, motion, tracklets):
        B, T, D = motion.shape
        N = tracklets.shape[2]

        q = self.wq(self.ln_q(motion))                             # (B,T,D)
        kv_flat = self.ln_kv(tracklets).reshape(B, T * N, -1)
        k = self.wk(kv_flat)                                      # (B,T*N,D)
        v = self.wv(kv_flat)

        # reshape → (B, nhead, seq, dk)
        q = q.view(B, T, self.nhead, self.dk).transpose(1, 2)
        k = k.view(B, T * N, self.nhead, self.dk).transpose(1, 2)
        v = v.view(B, T * N, self.nhead, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)   # (B,H,T,T*N)
        attn = self.drop(F.softmax(scores, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.wo(out)

        # per-vehicle importance: avg over heads, diagonal temporal slice
        attn_avg = attn.mean(dim=1)                                # (B,T,T*N)
        vehicle_w = torch.zeros(B, T, N, device=motion.device)
        for t in range(T):
            s = t * N
            e = s + N
            if e <= attn_avg.shape[2]:
                vehicle_w[:, t, :] = attn_avg[:, t, s:e]

        return motion + out, vehicle_w


# ═══════════════════════════════════════════════════════════════════════════
# 4. Expert Projectors
# ═══════════════════════════════════════════════════════════════════════════
class Projector(nn.Module):
    """Linear → LayerNorm → GELU → Dropout."""
    def __init__(self, inp, out, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, out), nn.LayerNorm(out),
            nn.GELU(), nn.Dropout(drop),
        )
    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Full Fusion Head
# ═══════════════════════════════════════════════════════════════════════════
class FusionMSTCN(nn.Module):
    """Cross-Attention MS-TCN Fusion Head.

    Outputs
    -------
    frame_logits : (B, 32) — raw logits for per-frame anomaly
    frame_scores : (B, 32) — sigmoid probabilities
    vehicle_weights : (B, 32, 5) — which tracklet is most anomalous
    """

    def __init__(self, hidden: int = HIDDEN, n_stages: int = 3,
                 tcn_layers: int = 4, tcn_kernel: int = 5, dropout: float = 0.3):
        super().__init__()

        # Expert projectors
        self.proj_motion = Projector(MOTION_DIM, hidden)
        self.proj_context = Projector(CONTEXT_DIM, hidden)

        # Cross-Attention (motion ↔ spatial)
        self.cross_attn = TrackletCrossAttention(
            d_model=hidden, d_vehicle=SPATIAL_FEAT,
            nhead=4, dropout=dropout,
        )

        # Fuse attended motion + context → TCN input
        self.fuse_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden), nn.ReLU(inplace=True),
        )

        # Multi-Scale TCN (3 stages, averaged)
        self.tcn_stages = nn.ModuleList([
            TCNStage(hidden, tcn_layers, tcn_kernel, dropout)
            for _ in range(n_stages)
        ])

        # Branch A — frame-level anomaly score
        self.head_frame = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

        # Branch B — spatial weight mask (which of 5 vehicles is the hazard)
        self.head_spatial = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden // 2, SPATIAL_VEHICLES),
        )

    def forward(self, motion, spatial, context):
        """
        motion  : (B, 32, 768)
        spatial : (B, 32, 5, 10)
        context : (B, 32, 1280)
        """
        m = self.proj_motion(motion)                           # (B,32,H)
        c = self.proj_context(context)                         # (B,32,H)

        # Cross-attention: motion attends to raw spatial tracklets
        m_att, vehicle_w = self.cross_attn(m, spatial)         # (B,32,H), (B,32,5)

        # Concat attended motion + context
        fused = self.fuse_proj(torch.cat([m_att, c], dim=-1))  # (B,32,H)

        # TCN (B,H,T) format
        x = fused.permute(0, 2, 1)
        stage_outs = [stage(x) for stage in self.tcn_stages]
        x = torch.stack(stage_outs).mean(dim=0)                # (B,H,T)
        x = x.permute(0, 2, 1)                                # (B,32,H)

        # Temporal smoothing via 1-D average filter (σ=3 frames)
        # Applied after TCN to stabilize predictions
        x_smooth = x  # smoothing applied on scores below

        # Branch A — frame anomaly logits
        frame_logits = self.head_frame(x).squeeze(-1)          # (B,32)
        # Temporal smoothing on probabilities
        frame_scores = torch.sigmoid(frame_logits)
        # 1-D Gaussian-like smoothing kernel (size=5)
        if frame_scores.shape[1] >= 5:
            kernel = torch.tensor([0.06, 0.24, 0.40, 0.24, 0.06],
                                  device=frame_scores.device).view(1, 1, 5)
            padded = F.pad(frame_scores.unsqueeze(1), (2, 2), mode="reflect")
            frame_scores = F.conv1d(padded, kernel).squeeze(1)

        # Branch B — spatial hazard weights
        spatial_logits = self.head_spatial(x)                  # (B,32,5)
        spatial_weights = torch.softmax(spatial_logits, dim=-1)

        # Combine cross-attention importance + head weights
        combined_vehicle_w = 0.5 * vehicle_w + 0.5 * spatial_weights

        return {
            "frame_logits": frame_logits,
            "frame_scores": frame_scores,
            "vehicle_weights": combined_vehicle_w,
        }


def build_model() -> FusionMSTCN:
    """Build and summarize the fusion head."""
    m = FusionMSTCN()
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[FusionMSTCN] params={total:,}  trainable={train:,}  "
          f"~{train * 4 / 1e6:.1f} MB")
    return m


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model = build_model()
    B = 2
    m = torch.randn(B, NUM_FRAMES, MOTION_DIM)
    s = torch.randn(B, NUM_FRAMES, SPATIAL_VEHICLES, SPATIAL_FEAT)
    c = torch.randn(B, NUM_FRAMES, CONTEXT_DIM)
    out = model(m, s, c)
    for k, v in out.items():
        print(f"  {k:20s}: {v.shape}")
