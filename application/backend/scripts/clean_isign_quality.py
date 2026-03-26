"""
iSign Data Quality Cleaner
==========================
Detects corrupted videos, low-motion clips, and outlier pose files.
Generates a report and optional cleaned metadata CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _video_motion_score(video_path: Path, sample_frames: int = 24) -> Tuple[float, str]:
    try:
        import cv2
    except Exception:
        return 0.0, "opencv-missing"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0, "video-open-failed"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 1:
        cap.release()
        return 0.0, "video-too-short"

    idxs = np.linspace(0, total - 1, min(sample_frames, total), dtype=int)
    prev = None
    diffs: List[float] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if prev is not None:
            diffs.append(float(np.mean(np.abs(gray - prev))))
        prev = gray
    cap.release()

    if not diffs:
        return 0.0, "video-read-failed"
    return float(np.mean(diffs)), "ok"


def _pose_quality_score(pose_path: Path) -> Tuple[float, str]:
    if not pose_path.exists():
        return 0.0, "pose-missing"
    try:
        arr = np.load(str(pose_path)).astype(np.float32)
    except Exception:
        return 0.0, "pose-load-failed"

    if arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0, "pose-shape-invalid"

    finite_ratio = float(np.isfinite(arr).mean())
    if finite_ratio < 0.98:
        return finite_ratio, "pose-nan-outlier"

    motion = np.diff(arr, axis=0)
    motion_score = float(np.mean(np.abs(motion)))
    return motion_score, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and clean low-quality iSign samples")
    parser.add_argument("--isign-dir", default="dataset/isign", help="Raw iSign directory")
    parser.add_argument("--metadata", default=None, help="Metadata CSV path (default: <isign-dir>/iSign_v1.1.csv)")
    parser.add_argument("--out-report", default="dataset/isign/quality_report.json", help="Quality report output")
    parser.add_argument("--out-clean-csv", default="dataset/isign/iSign_v1.1.cleaned.csv", help="Filtered metadata CSV output")
    parser.add_argument("--min-video-motion", type=float, default=0.012, help="Minimum video motion threshold")
    parser.add_argument("--min-pose-motion", type=float, default=0.004, help="Minimum pose motion threshold")
    parser.add_argument("--allow-missing-pose", action="store_true", help="Keep samples with missing pose")
    args = parser.parse_args()

    isign_dir = Path(args.isign_dir).resolve()
    metadata = Path(args.metadata).resolve() if args.metadata else isign_dir / "iSign_v1.1.csv"
    videos_dir = isign_dir / "videos"
    poses_dir = isign_dir / "poses"

    if not metadata.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata}")

    df = pd.read_csv(str(metadata))
    cols = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.columns = cols
    if "video_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "video_id"})

    keep_rows = []
    quality_rows = []

    for _, row in df.iterrows():
        vid = str(row["video_id"]).strip()

        video_path = None
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            p = videos_dir / f"{vid}{ext}"
            if p.exists():
                video_path = p
                break

        pose_path = poses_dir / f"{vid}.npy"

        v_score, v_reason = (0.0, "video-missing") if video_path is None else _video_motion_score(video_path)
        p_score, p_reason = _pose_quality_score(pose_path)

        video_ok = (v_reason == "ok") and (v_score >= args.min_video_motion)
        pose_ok = (p_reason == "ok") and (p_score >= args.min_pose_motion)
        if args.allow_missing_pose and p_reason == "pose-missing":
            pose_ok = True

        keep = video_ok and pose_ok
        quality_rows.append(
            {
                "video_id": vid,
                "video_motion": v_score,
                "video_reason": v_reason,
                "pose_motion": p_score,
                "pose_reason": p_reason,
                "keep": keep,
            }
        )

        if keep:
            keep_rows.append(row)

    report = {
        "total_samples": int(len(df)),
        "kept_samples": int(len(keep_rows)),
        "dropped_samples": int(len(df) - len(keep_rows)),
        "min_video_motion": args.min_video_motion,
        "min_pose_motion": args.min_pose_motion,
        "allow_missing_pose": args.allow_missing_pose,
        "rows": quality_rows,
    }

    out_report = Path(args.out_report).resolve()
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    out_clean_csv = Path(args.out_clean_csv).resolve()
    out_clean_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(keep_rows).to_csv(out_clean_csv, index=False)

    print(f"Quality report: {out_report}")
    print(f"Cleaned CSV   : {out_clean_csv}")
    print(f"Kept {len(keep_rows)}/{len(df)} samples")


if __name__ == "__main__":
    main()
