"""
Build an LMDB pose store for iSign pose tensors.

This packs the pose `.npy` files referenced by train/val/test annotations into a
single LMDB database keyed by each annotation's `pose_file` value.
"""

from __future__ import annotations

import argparse
import os
import json
import logging
import shutil
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Set

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_pose_lmdb")


def _available_cpu_workers() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _default_builder_workers() -> int:
    return max(1, min(6, _available_cpu_workers()))


@dataclass
class PosePayload:
    ref: str
    value: bytes | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build an LMDB database for iSign pose .npy files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="dataset/isign_processed", help="Processed iSign dataset directory")
    p.add_argument("--output", default=None, help="Output LMDB path (default: <data-dir>/poses.lmdb)")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"], help="Splits to scan for pose references")
    p.add_argument("--workers", type=int, default=_default_builder_workers(), help="Concurrent pose reader workers; use 0 or 1 for fully serial builds")
    p.add_argument("--batch-size", type=int, default=16, help="How many read tasks to submit to the worker pool at a time")
    p.add_argument("--prefetch", type=int, default=16, help="Maximum number of in-flight pose reads waiting on the writer")
    p.add_argument("--commit-every", type=int, default=2048, help="LMDB commit interval")
    p.add_argument("--map-size-gb", type=float, default=0.0, help="Optional explicit LMDB map size in GB; 0 auto-estimates from .npy file sizes")
    p.add_argument("--value-format", choices=["npy", "raw-f32"], default="npy", help="LMDB value format; raw-f32 uses a compact [int32 T, int32 D, float32 payload] layout")
    p.add_argument("--writemap", action="store_true", help="Enable LMDB writemap for faster bulk builds")
    p.add_argument("--durable", action="store_true", help="Use fully durable LMDB commits during the build (slower, safer on crashes)")
    p.add_argument("--overwrite", action="store_true", help="Delete an existing LMDB output path before rebuilding")
    return p.parse_args()


def load_pose_refs(data_dir: Path, splits: List[str]) -> List[str]:
    refs: Set[str] = set()
    for split in splits:
        ann_path = data_dir / f"{split}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")
        with open(ann_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        for ann in annotations:
            pose_file = ann.get("pose_file")
            if pose_file:
                refs.add(str(pose_file))
    return sorted(refs)


def infer_map_size_bytes(data_dir: Path, pose_refs: List[str]) -> int:
    total_bytes = 0
    for ref in pose_refs:
        path = data_dir / ref
        if path.exists():
            total_bytes += path.stat().st_size
    cushion = max(1 << 30, int(total_bytes * 0.2))
    return total_bytes + cushion


def normalize_pose_array(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=False)
    if out.ndim == 3:
        out = out.reshape(out.shape[0], -1)
    elif out.ndim == 1:
        out = out.reshape(1, -1)
    elif out.ndim == 0:
        out = out.reshape(1, 1)
    return out


def pack_raw_f32_pose(arr: np.ndarray) -> bytes:
    if arr.ndim != 2:
        raise ValueError(f"raw-f32 format expects 2D pose array, got shape={arr.shape}")
    t = int(arr.shape[0])
    d = int(arr.shape[1])
    header = np.array([t, d], dtype=np.int32).tobytes(order="C")
    payload = np.ascontiguousarray(arr, dtype=np.float32).tobytes(order="C")
    return header + payload


def infer_pose_dim(data_dir: Path, pose_refs: List[str]) -> int:
    for ref in pose_refs:
        pose_path = data_dir / ref
        if not pose_path.exists():
            continue
        try:
            arr = normalize_pose_array(np.load(pose_path, allow_pickle=False, mmap_mode="r"))
        except Exception:
            continue
        if arr.ndim == 2 and arr.shape[1] > 0:
            return int(arr.shape[1])
        return 1
    return 0


def load_pose_payload(data_dir: Path, ref: str, value_format: str) -> PosePayload:
    pose_path = data_dir / ref
    if not pose_path.exists():
        return PosePayload(ref=ref, value=None)

    if value_format == "raw-f32":
        arr = normalize_pose_array(np.load(pose_path, allow_pickle=False))
        return PosePayload(ref=ref, value=pack_raw_f32_pose(arr))

    return PosePayload(ref=ref, value=pose_path.read_bytes())


def iter_pose_payloads(
    data_dir: Path,
    pose_refs: List[str],
    value_format: str,
    workers: int,
    batch_size: int,
    prefetch: int,
) -> Iterator[PosePayload]:
    if workers <= 1:
        for ref in pose_refs:
            yield load_pose_payload(data_dir, ref, value_format)
        return

    max_pending = max(1, int(prefetch))
    submit_batch = max(1, int(batch_size))
    refs_iter = iter(pose_refs)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        pending: Deque[Future[PosePayload]] = deque()

        def fill_pending() -> None:
            submitted = 0
            while len(pending) < max_pending and submitted < submit_batch:
                try:
                    ref = next(refs_iter)
                except StopIteration:
                    break
                pending.append(executor.submit(load_pose_payload, data_dir, ref, value_format))
                submitted += 1

        fill_pending()
        while pending:
            payload = pending.popleft().result()
            yield payload
            if len(pending) < max_pending:
                fill_pending()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_path = Path(args.output) if args.output else data_dir / "poses.lmdb"

    try:
        import lmdb
    except ImportError as exc:
        raise RuntimeError("Please install `lmdb` before building the pose store.") from exc

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to rebuild.")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    pose_refs = load_pose_refs(data_dir, list(args.splits))
    if not pose_refs:
        raise RuntimeError("No pose_file entries were found in the requested splits.")

    pose_dim = infer_pose_dim(data_dir, pose_refs)
    map_size = (
        int(args.map_size_gb * (1024 ** 3))
        if args.map_size_gb and args.map_size_gb > 0
        else infer_map_size_bytes(data_dir, pose_refs)
    )
    log.info(
        "Building pose LMDB at %s from %d unique pose references (map_size=%.2f GB, workers=%d, batch=%d, prefetch=%d, durable=%s, writemap=%s)",
        output_path,
        len(pose_refs),
        map_size / (1024 ** 3),
        max(1, int(args.workers)),
        max(1, int(args.batch_size)),
        max(1, int(args.prefetch)),
        bool(args.durable),
        bool(args.writemap),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(output_path),
        map_size=map_size,
        subdir=True,
        meminit=False,
        map_async=not args.durable,
        writemap=bool(args.writemap),
        sync=bool(args.durable),
        metasync=bool(args.durable),
    )

    written = 0
    missing = 0
    start = time.time()

    txn = env.begin(write=True)
    try:
        for idx, payload in enumerate(
            iter_pose_payloads(
                data_dir=data_dir,
                pose_refs=pose_refs,
                value_format=str(args.value_format),
                workers=int(args.workers),
                batch_size=int(args.batch_size),
                prefetch=int(args.prefetch),
            ),
            start=1,
        ):
            if payload.value is None:
                missing += 1
                continue

            txn.put(payload.ref.encode("utf-8"), payload.value)
            written += 1

            if idx % max(1, int(args.commit_every)) == 0:
                txn.commit()
                txn = env.begin(write=True)
                elapsed = max(time.time() - start, 1e-6)
                speed = written / elapsed
                log.info(
                    "Progress %d/%d | written=%d | missing=%d | speed=%.1f poses/s",
                    idx,
                    len(pose_refs),
                    written,
                    missing,
                    speed,
                )

        meta: Dict[str, object] = {
            "num_entries": written,
            "num_missing": missing,
            "pose_dim": pose_dim,
            "value_format": args.value_format,
            "splits": list(args.splits),
            "created_at_unix": time.time(),
        }
        txn.put(b"__meta__", json.dumps(meta).encode("utf-8"))
        txn.commit()
        env.sync()
    except Exception:
        txn.abort()
        raise
    finally:
        env.close()

    elapsed = max(time.time() - start, 1e-6)
    log.info(
        "Done. Wrote %d pose entries to %s (missing=%d, pose_dim=%d, elapsed=%.1fs)",
        written,
        output_path,
        missing,
        pose_dim,
        elapsed,
    )


if __name__ == "__main__":
    main()
