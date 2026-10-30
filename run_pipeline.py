"""
run_pipeline.py — Master Execution Orchestrator
=================================================
Automates the entire project lifecycle:
    Stage 0: Environment & dependency validation
    Stage 1: Directory structure verification
    Stage 2: Stratified group split (70/15/15)
    Stage 3: Feature extraction (preprocess.py)
    Stage 4: Model training (train.py)
    Stage 5: Evaluation — STAUC/TTA metrics (eval.py)
    Stage 6: Demo video generation (eval.py --visualize)

Usage:
    python run_pipeline.py --data_root ./data/nexar --csv ./data/nexar/train.csv
    python run_pipeline.py --skip_extract   # skip stage 3 if features exist
"""

import argparse
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# Stage 0: Environment & Dependency Validation
# ═══════════════════════════════════════════════════════════════════════════
REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("bitsandbytes", "bitsandbytes"),
    ("ultralytics", "ultralytics"),
    ("timm", "timm"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
]


def check_environment():
    """Verify all dependencies and CUDA availability."""
    print("=" * 70)
    print("  STAGE 0: Environment & Dependency Validation")
    print("=" * 70)

    ok = True
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            print(f"  ✓ {pip_name:20s} {ver}")
        except ImportError:
            print(f"  ✗ {pip_name:20s} MISSING — pip install {pip_name}")
            ok = False

    # CUDA check
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n  🟢 CUDA available: {gpu} ({mem:.1f} GB)")
        else:
            print("\n  🟡 CUDA not available — will run on CPU (very slow)")
    except Exception as e:
        print(f"\n  🔴 CUDA check failed: {e}")

    if not ok:
        print("\n  ⚠  Missing packages detected. Install with:")
        print("     pip install -r requirements.txt")
        return False

    print("\n  ✓ All dependencies satisfied.\n")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Directory Structure
# ═══════════════════════════════════════════════════════════════════════════
DIRS = [
    "data",
    "metadata",
    "features/train",
    "features/val",
    "features/test",
    "checkpoints",
    "outputs",
    "visualizations",
]


def ensure_directories():
    """Create all required directories."""
    print("=" * 70)
    print("  STAGE 1: Directory Structure Verification")
    print("=" * 70)

    for d in DIRS:
        os.makedirs(d, exist_ok=True)
        status = "exists" if Path(d).exists() else "created"
        print(f"  ✓ {d:30s} [{status}]")

    print()
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Stratified Group Split
# ═══════════════════════════════════════════════════════════════════════════
def run_split(csv_path: str):
    """Run the stratified group split from preprocess.py."""
    print("=" * 70)
    print("  STAGE 2: Stratified Group Split (70 / 15 / 15)")
    print("=" * 70)

    # Check if splits already exist
    split_files = {
        "train": "metadata/train.csv",
        "val": "metadata/val.csv",
        "test": "metadata/test.csv",
    }
    all_exist = all(os.path.isfile(p) for p in split_files.values())

    if all_exist:
        print("  Split CSVs already exist in metadata/. Skipping.")
        for name, path in split_files.items():
            import pandas as pd
            df = pd.read_csv(path)
            n_crash = int(df["target"].sum())
            print(f"    {name:5s}: {len(df)} clips  "
                  f"(crash={n_crash}, normal={len(df)-n_crash})")
        print()
        return split_files

    if not os.path.isfile(csv_path):
        print(f"  ⚠  CSV not found: {csv_path}")
        print("  Place your Nexar train.csv at this path and re-run.")
        return None

    try:
        from preprocess import create_stratified_splits
        paths = create_stratified_splits(csv_path, output_dir="metadata")
        return paths
    except Exception as e:
        print(f"  ✗ Split failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════
def run_extraction(data_root: str, skip: bool = False):
    """Run preprocess.py for each split."""
    print("=" * 70)
    print("  STAGE 3: Offline Feature Extraction (3-Expert Ensemble)")
    print("=" * 70)

    if skip:
        print("  --skip_extract flag set. Checking existing features...")
        for split in ["train", "val", "test"]:
            d = f"features/{split}"
            n = len(list(Path(d).glob("*.npy"))) if os.path.isdir(d) else 0
            print(f"    {split}: {n} .npy files")
        print()
        return True

    splits = {
        "train": "metadata/train.csv",
        "val": "metadata/val.csv",
        "test": "metadata/test.csv",
    }

    for split_name, csv_path in splits.items():
        if not os.path.isfile(csv_path):
            print(f"  [SKIP] {csv_path} not found")
            continue

        # Check if already extracted
        out_dir = f"features/{split_name}"
        import pandas as pd
        expected = len(pd.read_csv(csv_path))
        existing = len(list(Path(out_dir).glob("*.npy"))) if os.path.isdir(out_dir) else 0

        if existing >= expected and expected > 0:
            print(f"  ✓ {split_name}: {existing}/{expected} already extracted. Skipping.")
            continue

        print(f"\n  >>> Extracting [{split_name}] ({expected} clips) ...")
        t0 = time.time()
        try:
            from preprocess import run_pipeline
            run_pipeline(data_root, csv_path, split_name)
            elapsed = time.time() - t0
            print(f"  ✓ {split_name} done in {elapsed:.0f}s")
        except Exception as e:
            print(f"  ✗ {split_name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print()
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4: Training
# ═══════════════════════════════════════════════════════════════════════════
def run_training(epochs: int = 50, batch_size: int = 8, lr: float = 3e-4):
    """Launch training via train.py."""
    print("=" * 70)
    print("  STAGE 4: Fusion Head Training")
    print("=" * 70)

    train_dir = "features/train"
    val_dir = "features/val"

    n_train = len(list(Path(train_dir).glob("*.npy"))) if os.path.isdir(train_dir) else 0
    n_val = len(list(Path(val_dir).glob("*.npy"))) if os.path.isdir(val_dir) else 0

    if n_train == 0:
        print("  ✗ No training features found. Run Stage 3 first.")
        return False

    print(f"  Training:   {n_train} clips")
    print(f"  Validation: {n_val} clips")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size} × 4 (grad accum) = {batch_size * 4}")

    cmd = [
        sys.executable, "train.py",
        "--train_dir", train_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
    ]
    if n_val > 0:
        cmd.extend(["--val_dir", val_dir])

    t0 = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        print(f"\n  ✓ Training complete in {elapsed:.0f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Training FAILED (exit code {e.returncode})")
        return False
    except Exception as e:
        print(f"  ✗ Training FAILED: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5: Evaluation
# ═══════════════════════════════════════════════════════════════════════════
def run_evaluation():
    """Compute STAUC and TTA on the test set."""
    print("=" * 70)
    print("  STAGE 5: Evaluation — STAUC & TTA Metrics")
    print("=" * 70)

    test_dir = "features/test"
    ckpt = "checkpoints/best.pt"

    if not os.path.isdir(test_dir) or len(list(Path(test_dir).glob("*.npy"))) == 0:
        # Fall back to val
        test_dir = "features/val"
        print("  (No test features — evaluating on validation set)")

    n = len(list(Path(test_dir).glob("*.npy"))) if os.path.isdir(test_dir) else 0
    if n == 0:
        print("  ✗ No features for evaluation.")
        return False

    if not os.path.isfile(ckpt):
        print(f"  ⚠  Checkpoint not found: {ckpt}")
        print("  Looking for any checkpoint...")
        ckpts = sorted(Path("checkpoints").glob("*.pt"))
        if ckpts:
            ckpt = str(ckpts[-1])
            print(f"  Using: {ckpt}")
        else:
            print("  ✗ No checkpoints found. Run Stage 4 first.")
            return False

    cmd = [
        sys.executable, "eval.py",
        "--feature_dir", test_dir,
        "--checkpoint", ckpt,
        "--output_json", "outputs/eval_results.json",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\n  ✓ Evaluation complete. Results → outputs/eval_results.json")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Evaluation FAILED (exit code {e.returncode})")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Stage 6: Demo Video
# ═══════════════════════════════════════════════════════════════════════════
def run_demo(video_path: str = None):
    """Generate annotated demo video."""
    print("=" * 70)
    print("  STAGE 6: Demo Video Generation")
    print("=" * 70)

    if video_path is None:
        # Try to find a video in data/
        candidates = list(Path("data").rglob("*.mp4"))
        if candidates:
            video_path = str(candidates[0])
            print(f"  Auto-selected: {video_path}")
        else:
            print("  ✗ No video found for demo. Provide --demo_video.")
            return False

    ckpt = "checkpoints/best.pt"
    if not os.path.isfile(ckpt):
        ckpts = sorted(Path("checkpoints").glob("*.pt"))
        ckpt = str(ckpts[-1]) if ckpts else None

    if ckpt is None:
        print("  ✗ No checkpoint. Run training first.")
        return False

    output = "visualizations/demo.mp4"
    cmd = [
        sys.executable, "eval.py",
        "--visualize",
        "--video", video_path,
        "--checkpoint", ckpt,
        "--output", output,
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n  ✓ Demo video → {output}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Demo generation FAILED (exit code {e.returncode})")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser(
        description="Master Pipeline Orchestrator — Nexar VAD Project"
    )
    pa.add_argument("--data_root", default="./data/nexar")
    pa.add_argument("--csv", default="./data/nexar/train.csv",
                    help="Master CSV with all video annotations")
    pa.add_argument("--epochs", type=int, default=50)
    pa.add_argument("--batch_size", type=int, default=8)
    pa.add_argument("--lr", type=float, default=3e-4)
    pa.add_argument("--skip_extract", action="store_true",
                    help="Skip feature extraction if features already exist")
    pa.add_argument("--skip_train", action="store_true")
    pa.add_argument("--eval_only", action="store_true")
    pa.add_argument("--demo_video", default=None,
                    help="Path to .mp4 for demo visualization")
    args = pa.parse_args()

    t_start = time.time()

    print("\n" + "═" * 70)
    print("  🚗  NEXAR DASHCAM COLLISION PREDICTION — MASTER PIPELINE")
    print("  📐  Target: STAUC ≥ 0.75  |  TTA ≥ 3000 ms")
    print("  🖥️   Hardware: NVIDIA RTX 4050 (6 GB VRAM)")
    print("═" * 70 + "\n")

    # Stage 0
    if not check_environment():
        print("Fix dependencies and re-run.")
        return

    # Stage 1
    ensure_directories()

    if args.eval_only:
        run_evaluation()
        if args.demo_video:
            run_demo(args.demo_video)
        return

    # Stage 2
    splits = run_split(args.csv)
    if splits is None:
        print("Cannot proceed without split CSVs.")
        return

    # Stage 3
    run_extraction(args.data_root, skip=args.skip_extract)

    # Stage 4
    if not args.skip_train:
        run_training(args.epochs, args.batch_size, args.lr)

    # Stage 5
    run_evaluation()

    # Stage 6
    if args.demo_video:
        run_demo(args.demo_video)

    elapsed = time.time() - t_start
    print("\n" + "═" * 70)
    print(f"  ✓ PIPELINE COMPLETE — {elapsed / 60:.1f} minutes total")
    print("═" * 70)


if __name__ == "__main__":
    main()
