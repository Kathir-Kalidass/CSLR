"""
iSign Dataset Preprocessor
===========================
Converts the raw iSign v1.1 dataset into the training-ready format
used by the existing CSLR backend pipeline.

Input layout (after merge_isign_parts.py):
  <isign-dir>/
    iSign_v1.1.csv                 # metadata: video_id, sentence
    videos/                        # extracted video files  *.mp4
    poses/                         # extracted pose files   *.npy  (per-frame keypoints)

Output layout:
  <out-dir>/
    frames/
      <video_id>/
        frame_0000.jpg
        frame_0001.jpg
        ...
    poses/
      <video_id>.npy               # shape (T, K*2) float32
    vocab.json                     # sorted word list  ["<blank>", "hello", ...]
    train.json                     # list of annotation dicts
    val.json
    test.json

Each annotation dict:
  {
    "video_id": "...",
    "sentence": "...",
    "gloss_tokens": ["WORD1", "WORD2", ...],
    "frame_dir":  "frames/<video_id>",
    "pose_file":  "poses/<video_id>.npy",
    "num_frames": 64
  }

Usage
-----
python scripts/preprocess_isign.py \\
    --isign-dir dataset/isign \\
    --out-dir   dataset/isign_processed \\
    --max-frames 64 \\
    --val-ratio 0.1 \\
    --test-ratio 0.1 \\
    --workers 4

Tip: if pose files are already extracted, frame extraction can be
     skipped with --skip-frames (useful to save time on re-runs).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess_isign")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLANK_TOKEN = "<blank>"
UNK_TOKEN   = "<unk>"
SPECIAL_TOKENS = [BLANK_TOKEN, UNK_TOKEN]

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize_sentence(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9 ']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentence_to_gloss(sentence: str) -> List[str]:
    """Convert a raw sentence to word-level gloss tokens (upper-case)."""
    return [w.upper() for w in normalize_sentence(sentence).split() if w]


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: Path,
    out_dir: Path,
    max_frames: int,
    target_size: Tuple[int, int] = (224, 224),
) -> int:
    """Extract up to *max_frames* uniformly-sampled JPEG frames.

    Returns the number of frames saved (0 on failure).
    """
    try:
        import cv2
    except ImportError:
        log.error("opencv-python not installed.  Run:  pip install opencv-python")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("Cannot open video: %s", video_path)
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Unknown length — read all
        all_frames: List[Any] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        frames_to_save = all_frames
    else:
        cap.release()
        indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        frames_to_save = []
        cap = cv2.VideoCapture(str(video_path))
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames_to_save.append(frame)
        cap.release()

    # Uniform sub-sample if needed
    if len(frames_to_save) > max_frames:
        indices2 = np.linspace(0, len(frames_to_save) - 1, max_frames, dtype=int)
        frames_to_save = [frames_to_save[i] for i in indices2]

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames_to_save):
        frame_resized = cv2.resize(frame, target_size)
        cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"), frame_resized)

    return len(frames_to_save)


# ---------------------------------------------------------------------------
# Pose loading helpers
# ---------------------------------------------------------------------------

def load_pose_npy(pose_path: Path) -> Optional[np.ndarray]:
    """Load a single pose .npy file.

    Returns float32 array of shape (T, D) or None on failure.
    """
    try:
        arr = np.load(str(pose_path)).astype(np.float32)
        if arr.ndim == 2:
            return arr          # already (T, D)
        if arr.ndim == 3:
            # (T, K, 2 or 3)  → flatten last two dims
            return arr.reshape(arr.shape[0], -1)
        log.warning("Unexpected pose array shape %s in %s", arr.shape, pose_path)
        return None
    except Exception as exc:
        log.warning("Failed to load pose %s: %s", pose_path, exc)
        return None


def find_pose_file(poses_root: Path, video_id: str) -> Optional[Path]:
    """Search for pose file matching *video_id* (various layouts)."""
    candidates = [
        poses_root / f"{video_id}.npy",
        poses_root / video_id / "pose.npy",
        poses_root / f"{video_id}_pose.npy",
    ]
    for c in candidates:
        if c.exists():
            return c
    # glob fallback
    matches = list(poses_root.glob(f"**/{video_id}*.npy"))
    if matches:
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# Worker for parallel frame extraction
# ---------------------------------------------------------------------------

def _worker_extract(args: Tuple[Path, Path, int]) -> Tuple[str, int]:
    video_path, frame_dir, max_frames = args
    n = extract_frames(video_path, frame_dir, max_frames)
    return str(video_path), n


# ---------------------------------------------------------------------------
# Build vocabulary
# ---------------------------------------------------------------------------

def build_vocabulary(annotations: List[Dict]) -> List[str]:
    from collections import Counter
    counter: Counter = Counter()
    for ann in annotations:
        counter.update(ann["gloss_tokens"])
    vocab = SPECIAL_TOKENS + sorted(counter.keys())
    return vocab


# ---------------------------------------------------------------------------
# Main preprocessing logic
# ---------------------------------------------------------------------------

def preprocess(args: argparse.Namespace) -> None:
    import pandas as pd

    isign_dir  = Path(args.isign_dir).resolve()
    out_dir    = Path(args.out_dir).resolve()
    videos_dir = isign_dir / "videos"
    poses_dir  = isign_dir / "poses"

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_root = out_dir / "frames"
    poses_out   = out_dir / "poses"
    poses_out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load metadata CSV
    # ------------------------------------------------------------------
    csv_path = isign_dir / "iSign_v1.1.csv"
    if not csv_path.exists():
        log.error("Metadata CSV not found: %s", csv_path)
        sys.exit(1)

    log.info("Loading metadata from %s …", csv_path)
    df = pd.read_csv(str(csv_path))
    log.info("  Loaded %d rows.  Columns: %s", len(df), list(df.columns))

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "video_id" not in df.columns:
        # Try first column as video_id
        df = df.rename(columns={df.columns[0]: "video_id"})
    if "sentence" not in df.columns:
        candidates = [c for c in df.columns if "sentence" in c or "text" in c or "label" in c]
        if candidates:
            df = df.rename(columns={candidates[0]: "sentence"})
        else:
            log.error("Cannot identify 'sentence' column.  Available: %s", list(df.columns))
            sys.exit(1)

    df = df[["video_id", "sentence"]].dropna()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["sentence"] = df["sentence"].astype(str).str.strip()
    log.info("  After cleaning: %d samples", len(df))

    # Limit samples for quick testing
    if args.max_samples and args.max_samples > 0:
        df = df.head(args.max_samples)
        log.info("  Limited to %d samples (--max-samples)", len(df))

    # ------------------------------------------------------------------
    # 2. Build initial annotation list
    # ------------------------------------------------------------------
    annotations: List[Dict] = []
    missing_videos = 0
    missing_poses  = 0

    for _, row in df.iterrows():
        vid_id   = row["video_id"]
        sentence = row["sentence"]
        glosses  = sentence_to_gloss(sentence)
        if not glosses:
            continue

        # Locate video file
        video_path: Optional[Path] = None
        if videos_dir.exists():
            for ext in VIDEO_EXTS:
                cand = videos_dir / f"{vid_id}{ext}"
                if cand.exists():
                    video_path = cand
                    break
            if video_path is None:
                missing_videos += 1

        # Locate pose file
        pose_path: Optional[Path] = None
        if poses_dir.exists():
            pose_path = find_pose_file(poses_dir, vid_id)
            if pose_path is None:
                missing_poses += 1

        ann: Dict[str, Any] = {
            "video_id":    vid_id,
            "sentence":    sentence,
            "gloss_tokens": glosses,
            "frame_dir":   f"frames/{vid_id}",
            "pose_file":   f"poses/{vid_id}.npy",
            "video_path":  str(video_path) if video_path else None,
            "pose_src":    str(pose_path)  if pose_path  else None,
            "num_frames":  args.max_frames,
        }
        annotations.append(ann)

    log.info(
        "Annotations built: %d total, %d missing videos, %d missing poses",
        len(annotations), missing_videos, missing_poses,
    )

    # ------------------------------------------------------------------
    # 3. Extract frames from videos (optional)
    # ------------------------------------------------------------------
    if not args.skip_frames and videos_dir.exists():
        log.info("Extracting frames (workers=%d, max_frames=%d) …", args.workers, args.max_frames)
        tasks = []
        for ann in annotations:
            if ann["video_path"] is None:
                continue
            frame_dir = frames_root / ann["video_id"]
            if frame_dir.exists() and len(list(frame_dir.glob("*.jpg"))) >= 4:
                continue  # already extracted
            tasks.append((Path(ann["video_path"]), frame_dir, args.max_frames))

        if tasks:
            log.info("  %d videos to extract …", len(tasks))
            t0 = time.time()
            if args.workers > 1:
                with mp.Pool(args.workers) as pool:
                    results = pool.map(_worker_extract, tasks)
            else:
                results = [_worker_extract(t) for t in tasks]
            ok = sum(1 for _, n in results if n > 0)
            log.info(
                "  Extracted %d/%d videos in %.0f s",
                ok, len(tasks), time.time() - t0,
            )
        else:
            log.info("  All frames already extracted — skipping")

        # Update num_frames from extracted counts
        for ann in annotations:
            frame_dir = frames_root / ann["video_id"]
            count = len(list(frame_dir.glob("*.jpg"))) if frame_dir.exists() else 0
            if count > 0:
                ann["num_frames"] = count
    else:
        log.info("Skipping frame extraction (--skip-frames or no video directory)")

    # ------------------------------------------------------------------
    # 4. Copy / link pose files into out_dir
    # ------------------------------------------------------------------
    if poses_dir.exists():
        log.info("Processing pose files …")
        copied = 0
        for ann in annotations:
            if ann["pose_src"] is None:
                continue
            dest = poses_out / f"{ann['video_id']}.npy"
            if dest.exists():
                copied += 1
                continue
            # Load, validate, resave
            arr = load_pose_npy(Path(ann["pose_src"]))
            if arr is not None:
                np.save(str(dest), arr)
                copied += 1
        log.info("  Pose files ready: %d/%d", copied, len(annotations))
    else:
        log.info("Pose directory not found — skipping pose copy")

    # ------------------------------------------------------------------
    # 5. Build vocabulary
    # ------------------------------------------------------------------
    log.info("Building vocabulary …")
    vocab = build_vocabulary(annotations)
    word2idx = {w: i for i, w in enumerate(vocab)}
    vocab_path = out_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    log.info("  Vocabulary size: %d  →  %s", len(vocab), vocab_path)

    # Convert gloss tokens to indices in each annotation
    for ann in annotations:
        ann["gloss_ids"] = [word2idx.get(t, word2idx[UNK_TOKEN]) for t in ann["gloss_tokens"]]

    # ------------------------------------------------------------------
    # 6. Split into train / val / test
    # ------------------------------------------------------------------
    random.seed(args.seed)
    random.shuffle(annotations)
    n = len(annotations)
    n_test  = max(1, int(n * args.test_ratio))
    n_val   = max(1, int(n * args.val_ratio))
    n_train = n - n_val - n_test

    split_data = {
        "train": annotations[:n_train],
        "val":   annotations[n_train : n_train + n_val],
        "test":  annotations[n_train + n_val :],
    }

    for split, data in split_data.items():
        path = out_dir / f"{split}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info("  %s: %d samples  →  %s", split, len(data), path)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Preprocessing complete!")
    log.info("  Output dir : %s", out_dir)
    log.info("  Vocab size : %d", len(vocab))
    log.info("  Train      : %d", n_train)
    log.info("  Val        : %d", n_val)
    log.info("  Test       : %d", n_test)
    log.info("=" * 60)
    log.info("Next step:")
    log.info(
        "  python scripts/train_isign.py "
        "--data-dir %s --vocab %s",
        out_dir, vocab_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess the iSign dataset into CSLR training format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--isign-dir",   default="dataset/isign",           help="Root directory of downloaded + extracted iSign data")
    p.add_argument("--out-dir",     default="dataset/isign_processed", help="Output directory for processed data")
    p.add_argument("--max-frames",  type=int, default=64,              help="Maximum frames to sample per video")
    p.add_argument("--val-ratio",   type=float, default=0.1,           help="Fraction of data for validation")
    p.add_argument("--test-ratio",  type=float, default=0.1,           help="Fraction of data for testing")
    p.add_argument("--workers",     type=int, default=4,               help="Parallel workers for frame extraction")
    p.add_argument("--seed",        type=int, default=42,              help="Random seed for split")
    p.add_argument("--max-samples", type=int, default=0,               help="Limit total samples (0 = no limit, useful for testing)")
    p.add_argument("--skip-frames", action="store_true",               help="Skip frame extraction (use pre-extracted frames)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        log.error("pandas is not installed.  Run:  pip install pandas")
        sys.exit(1)
    preprocess(args)


if __name__ == "__main__":
    main()
