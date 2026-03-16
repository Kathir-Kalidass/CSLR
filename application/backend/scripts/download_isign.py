"""
iSign Dataset Downloader
========================
Downloads the iSign v1.1 Indian Sign Language dataset from HuggingFace.

Dataset: https://huggingface.co/datasets/Exploration-Lab/iSign
~228 GB total  |  pose parts (~170 GB)  |  video parts (~58 GB)

Usage
-----
# Full download (228 GB — requires sufficient storage)
python scripts/download_isign.py --target dataset/isign

# Metadata + poses only  (no raw video, ~170 GB)
python scripts/download_isign.py --target dataset/isign --no-videos

# Metadata only (tiny, use for quick testing)
python scripts/download_isign.py --target dataset/isign --metadata-only

# Resume an interrupted download
python scripts/download_isign.py --target dataset/isign --resume
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_isign")

# ---------------------------------------------------------------------------
# HuggingFace repo details
# ---------------------------------------------------------------------------
REPO_ID   = "Exploration-Lab/iSign"
REPO_TYPE = "dataset"

# All known files on the Hub (update if the dataset is revised)
METADATA_FILES = [
    "iSign_v1.1.csv",
    "word-description-dataset_v1.1.csv",
    "word-presence-dataset_v1.1.csv",
    "README.md",
]

VIDEO_PARTS = [
    "iSign-videos_v1.1_part_aa",
    "iSign-videos_v1.1_part_ab",
]

POSE_PARTS = [
    "iSign-poses_v1.1_part_aa",
    "iSign-poses_v1.1_part_ab",
    "iSign-poses_v1.1_part_ac",
    "iSign-poses_v1.1_part_ad",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_hf_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        log.error("huggingface_hub is not installed.  Run:  pip install huggingface-hub")
        sys.exit(1)


def _check_login() -> None:
    """Warn if no HF token is found."""
    try:
        from huggingface_hub import whoami
        whoami()
    except Exception:
        log.warning(
            "No HuggingFace login detected. "
            "Run: hf auth login"
        )

def _human_size(path: Path) -> str:
    """Return human-readable size of a file."""
    if not path.exists():
        return "–"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _free_space_gb(path: Path) -> float:
    import shutil
    total, used, free = shutil.disk_usage(path)
    return free / 1024**3


# ---------------------------------------------------------------------------
# Download individual file
# ---------------------------------------------------------------------------

def download_file(
    filename: str,
    local_dir: Path,
    resume: bool = True,
) -> Path:
    """Download a single file from the Hub into *local_dir*.

    Uses hf_hub_download which supports cache / resume automatically.
    """
    from huggingface_hub import hf_hub_download

    dest = local_dir / filename
    if dest.exists() and not resume:
        log.info("  skip (exists): %s", filename)
        return dest

    log.info("  downloading: %s …", filename)
    t0 = time.time()
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=filename,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=resume,
    )
    elapsed = time.time() - t0
    log.info("  done in %.0f s  (%s)  →  %s", elapsed, _human_size(Path(path)), path)
    return Path(path)


# ---------------------------------------------------------------------------
# Download groups
# ---------------------------------------------------------------------------

def download_metadata(local_dir: Path, resume: bool) -> None:
    log.info("=== Downloading metadata files ===")
    for fname in METADATA_FILES:
        try:
            download_file(fname, local_dir, resume)
        except Exception as exc:
            log.warning("  failed to download %s: %s", fname, exc)


def download_poses(local_dir: Path, resume: bool) -> None:
    log.info("=== Downloading pose parts (~170 GB) ===")
    free = _free_space_gb(local_dir)
    if free < 175:
        log.warning(
            "Only %.1f GB free — pose data requires ~170 GB.  "
            "Proceeding anyway; download may fail.", free
        )
    for part in POSE_PARTS:
        try:
            download_file(part, local_dir, resume)
        except Exception as exc:
            log.error("  FAILED: %s — %s", part, exc)


def download_videos(local_dir: Path, resume: bool) -> None:
    log.info("=== Downloading video parts (~58 GB) ===")
    free = _free_space_gb(local_dir)
    if free < 60:
        log.warning(
            "Only %.1f GB free — video data requires ~58 GB.  "
            "Proceeding anyway; download may fail.", free
        )
    for part in VIDEO_PARTS:
        try:
            download_file(part, local_dir, resume)
        except Exception as exc:
            log.error("  FAILED: %s — %s", part, exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download iSign v1.1 dataset from HuggingFace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--target",
        default="dataset/isign",
        help="Local directory to save the dataset",
    )
    p.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only the CSV metadata files (fast, <20 MB)",
    )
    p.add_argument(
        "--no-videos",
        action="store_true",
        help="Skip the raw video archives (~58 GB)",
    )
    p.add_argument(
        "--no-poses",
        action="store_true",
        help="Skip the pose archives (~170 GB)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume interrupted downloads (default: True)",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Restart downloads from scratch",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _check_hf_hub()
    _check_login()

    local_dir = Path(args.target).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    log.info("Target directory: %s", local_dir)
    log.info("Free disk space : %.1f GB", _free_space_gb(local_dir))

    # Always get metadata
    download_metadata(local_dir, args.resume)

    if args.metadata_only:
        log.info("--metadata-only set. Finished.")
        return

    if not args.no_poses:
        download_poses(local_dir, args.resume)

    if not args.no_videos:
        download_videos(local_dir, args.resume)

    log.info("Download complete. Files saved to: %s", local_dir)
    log.info(
        "Next step: python scripts/merge_isign_parts.py --src %s --out %s",
        local_dir, local_dir,
    )


if __name__ == "__main__":
    main()
