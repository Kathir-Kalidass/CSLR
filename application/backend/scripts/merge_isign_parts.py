"""
iSign Split-File Merger
=======================
Merges the multi-part binary archives (split with `split -b`) into single
tar archives and then extracts them.

The iSign dataset ships as:

  iSign-videos_v1.1_part_aa   ─┐
  iSign-videos_v1.1_part_ab   ─┴─► isign_videos.tar  →  extract

  iSign-poses_v1.1_part_aa    ─┐
  iSign-poses_v1.1_part_ab    ─┤
  iSign-poses_v1.1_part_ac    ─┤
  iSign-poses_v1.1_part_ad    ─┴─► isign_poses.tar   →  extract

Usage
-----
# Merge + extract both archives (default)
python scripts/merge_isign_parts.py --src dataset/isign --out dataset/isign

# Only merge (skip extraction)
python scripts/merge_isign_parts.py --src dataset/isign --out dataset/isign --no-extract

# Only poses
python scripts/merge_isign_parts.py --src dataset/isign --out dataset/isign --poses-only

# Only videos
python scripts/merge_isign_parts.py --src dataset/isign --out dataset/isign --videos-only
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("merge_isign")

# ---------------------------------------------------------------------------
# Part definitions
# ---------------------------------------------------------------------------

GROUPS: dict[str, dict] = {
    "poses": {
        "parts": [
            "iSign-poses_v1.1_part_aa",
            "iSign-poses_v1.1_part_ab",
            "iSign-poses_v1.1_part_ac",
            "iSign-poses_v1.1_part_ad",
        ],
        "merged_name": "isign_poses.tar",
        "extract_dir": "poses",
    },
    "videos": {
        "parts": [
            "iSign-videos_v1.1_part_aa",
            "iSign-videos_v1.1_part_ab",
        ],
        "merged_name": "isign_videos.tar",
        "extract_dir": "videos",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024  # type: ignore[assignment]
    return f"{n_bytes:.1f} PB"


def _free_space_gb(path: Path) -> float:
    total, used, free = shutil.disk_usage(path)
    return free / 1024**3


def _cat_parts_python(parts: list[Path], dest: Path) -> None:
    """Pure-Python fallback cat (slower for very large files)."""
    log.info("  Merging %d parts → %s (Python)", len(parts), dest.name)
    chunk = 1024 * 1024 * 64  # 64 MB
    with open(dest, "wb") as out:
        for part in parts:
            log.info("    appending %s  (%s)", part.name, _human_size(part.stat().st_size))
            with open(part, "rb") as inp:
                while True:
                    data = inp.read(chunk)
                    if not data:
                        break
                    out.write(data)
    log.info("  Merged file: %s  (%s)", dest.name, _human_size(dest.stat().st_size))


def _cat_parts_system(parts: list[Path], dest: Path) -> None:
    """Use `cat` system command — much faster on Linux/macOS."""
    cmd = ["cat"] + [str(p) for p in parts]
    log.info("  Merging %d parts → %s (cat)", len(parts), dest.name)
    with open(dest, "wb") as out:
        subprocess.check_call(cmd, stdout=out)
    log.info("  Merged file: %s  (%s)", dest.name, _human_size(dest.stat().st_size))


def merge_parts(parts: list[Path], dest: Path) -> None:
    """Merge binary split parts into a single file."""
    if dest.exists():
        log.info("  Already merged: %s — skipping", dest.name)
        return

    missing = [p for p in parts if not p.exists()]
    if missing:
        log.error("  Missing parts: %s", [p.name for p in missing])
        raise FileNotFoundError(f"Missing split parts: {missing}")

    total = sum(p.stat().st_size for p in parts)
    free  = _free_space_gb(dest.parent) * 1024**3
    if free < total * 1.1:
        log.warning(
            "  Low disk space (need ~%s, have ~%s) — proceeding anyway",
            _human_size(int(total * 1.1)),
            _human_size(int(free)),
        )

    t0 = time.time()
    if shutil.which("cat") and sys.platform != "win32":
        _cat_parts_system(parts, dest)
    else:
        _cat_parts_python(parts, dest)
    log.info("  Merge done in %.0f s", time.time() - t0)


def extract_tar(tar_path: Path, extract_dir: Path) -> None:
    """Extract a tar archive into *extract_dir*."""
    if extract_dir.exists() and any(extract_dir.iterdir()):
        log.info("  Already extracted: %s — skipping", extract_dir)
        return

    extract_dir.mkdir(parents=True, exist_ok=True)
    log.info("  Extracting %s → %s …", tar_path.name, extract_dir)
    t0 = time.time()

    # Try system tar first (faster, supports progress)
    if shutil.which("tar") and sys.platform != "win32":
        cmd = ["tar", "-xf", str(tar_path), "-C", str(extract_dir)]
        subprocess.check_call(cmd)
    else:
        with tarfile.open(tar_path, "r:*") as tf:
            tf.extractall(path=extract_dir)

    log.info("  Extraction done in %.0f s", time.time() - t0)


# ---------------------------------------------------------------------------
# Process a single group (poses or videos)
# ---------------------------------------------------------------------------

def process_group(
    group_name: str,
    src_dir: Path,
    out_dir: Path,
    do_extract: bool,
) -> None:
    cfg = GROUPS[group_name]
    parts = [src_dir / p for p in cfg["parts"]]
    merged = out_dir / cfg["merged_name"]
    extract_dir = out_dir / cfg["extract_dir"]

    # Check if any parts exist
    available = [p for p in parts if p.exists()]
    if not available:
        log.warning(
            "No parts found for '%s' in %s — skipping",
            group_name, src_dir,
        )
        return

    if len(available) < len(parts):
        log.warning(
            "Only %d/%d parts found for '%s'.  "
            "Merging with available parts only.",
            len(available), len(parts), group_name,
        )
        parts = available

    log.info("=== Processing: %s ===", group_name.upper())
    merge_parts(parts, merged)

    if do_extract:
        extract_tar(merged, extract_dir)
        log.info("  Contents in: %s", extract_dir)
    else:
        log.info("  --no-extract set; merged archive at: %s", merged)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge iSign split archives and extract them",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", default="dataset/isign", help="Directory containing the downloaded split parts")
    p.add_argument("--out", default="dataset/isign", help="Output directory for merged + extracted files")
    p.add_argument("--no-extract", action="store_true", help="Merge only; do not extract")
    p.add_argument("--poses-only", action="store_true", help="Process pose archives only")
    p.add_argument("--videos-only", action="store_true", help="Process video archives only")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    src_dir = Path(args.src).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    do_extract = not args.no_extract

    groups_to_run: list[str] = []
    if args.poses_only:
        groups_to_run = ["poses"]
    elif args.videos_only:
        groups_to_run = ["videos"]
    else:
        groups_to_run = ["poses", "videos"]

    for g in groups_to_run:
        process_group(g, src_dir, out_dir, do_extract)

    log.info("All done.  Next step:")
    log.info(
        "  python scripts/preprocess_isign.py "
        "--isign-dir %s --out-dir dataset/isign_processed",
        out_dir,
    )


if __name__ == "__main__":
    main()
