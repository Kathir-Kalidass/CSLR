"""
iSign Dataset Loader
====================
PyTorch Dataset class that reads the preprocessed iSign data produced by
  scripts/preprocess_isign.py

Expected directory layout:
  <data_dir>/
    train.json
    val.json
    test.json
    vocab.json
    frames/
      <video_id>/
        frame_0000.jpg  ...
    poses/
      <video_id>.npy          # shape (T, D)  float32

Each JSON entry contains:
  {
    "video_id"    : str,
    "sentence"    : str,
    "gloss_tokens": [str, ...],
    "gloss_ids"   : [int, ...],
    "frame_dir"   : "frames/<video_id>",
    "pose_file"   : "poses/<video_id>.npy",
    "num_frames"  : int
  }
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

# LMDB environments cannot always be reopened multiple times in the same process
# with identical paths/settings. Cache one handle per-process and path.
_LMDB_ENV_CACHE: Dict[Tuple[str, bool, bool], Any] = {}
_LMDB_ENV_CACHE_PID = os.getpid()
# ---------------------------------------------------------------------------
# ImageNet normalisation constants (same as the rest of the pipeline)
# ---------------------------------------------------------------------------
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ISignDataset(Dataset):
    """
    Loads RGB frames + pose keypoints for iSign samples.

    Args:
        data_dir    : Root directory of the processed dataset.
        split       : "train", "val", or "test".
        num_frames  : Number of frames to sample uniformly per clip.
        frame_size  : (H, W) to resize each frame.
        augment     : Apply random horizontal flip during training.
        use_poses   : Load pose keypoints (returns zero tensor if .npy missing).
        use_frames  : Load RGB frames (returns zero tensor if frames missing).
        max_samples : Limit dataset size (0 = use all; useful for smoke-tests).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_frames: int = 64,
        frame_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
        use_poses: bool = True,
        use_frames: bool = True,
        max_samples: int = 0,
        hard_negative_prob: float = 0.0,
        temporal_jitter: int = 2,
        frame_drop_prob: float = 0.05,
        brightness_jitter: float = 0.15,
        blur_prob: float = 0.10,
        noise_std: float = 0.02,
        pose_jitter_std: float = 0.01,
        use_albumentations: bool = True,
        albumentations_prob: float = 0.35,
        motion_blur_prob: float = 0.10,
        coarse_dropout_prob: float = 0.08,
        preload_n: int = 0,
        pose_backend: str = "auto",
        pose_lmdb_path: Optional[str] = None,
        pose_lmdb_readahead: bool = False,
    ) -> None:
        self.data_dir   = Path(data_dir)
        self.split      = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment    = augment and (split == "train")
        self.use_poses  = use_poses
        self.use_frames = use_frames
        self.hard_negative_prob = float(max(0.0, hard_negative_prob)) if split == "train" else 0.0
        self.temporal_jitter = int(max(0, temporal_jitter)) if split == "train" else 0
        self.frame_drop_prob = float(max(0.0, frame_drop_prob)) if split == "train" else 0.0
        self.brightness_jitter = float(max(0.0, brightness_jitter)) if split == "train" else 0.0
        self.blur_prob = float(max(0.0, blur_prob)) if split == "train" else 0.0
        self.noise_std = float(max(0.0, noise_std)) if split == "train" else 0.0
        self.pose_jitter_std = float(max(0.0, pose_jitter_std)) if split == "train" else 0.0
        self.use_albumentations = bool(use_albumentations) and self.augment
        self.albumentations_prob = float(np.clip(albumentations_prob, 0.0, 1.0)) if self.augment else 0.0
        self.motion_blur_prob = float(np.clip(motion_blur_prob, 0.0, 1.0)) if self.augment else 0.0
        self.coarse_dropout_prob = float(np.clip(coarse_dropout_prob, 0.0, 1.0)) if self.augment else 0.0
        self._albumentations = None
        self._frame_transform = self._build_frame_transform()

        # Load annotations
        ann_path = self.data_dir / f"{split}.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}\n"
                "Run  scripts/preprocess_isign.py  first."
            )
        with open(ann_path, "r", encoding="utf-8") as f:
            self.annotations: List[Dict[str, Any]] = json.load(f)

        if max_samples and max_samples > 0:
            self.annotations = self.annotations[:max_samples]

        self.pose_backend = str(pose_backend).lower()
        self.pose_lmdb_path = Path(pose_lmdb_path) if pose_lmdb_path else self.data_dir / "poses.lmdb"
        self.pose_lmdb_readahead = bool(pose_lmdb_readahead)
        self._pose_lmdb_env = None
        self._pose_lmdb_env_pid: Optional[int] = None
        self._pose_lmdb_meta: Dict[str, Any] = {}
        self._pose_lmdb_value_format = "npy"
        self._configure_pose_backend()
        self.pose_dim = self._infer_pose_dim() if self.use_poses else 1
        self.preload_n = max(0, min(int(preload_n), len(self.annotations)))
        self._sample_cache: Dict[int, Dict[str, Any]] = {}

        # Load vocabulary
        vocab_path = self.data_dir / "vocab.json"
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab: List[str] = json.load(f)
            self.vocab    = vocab
            self.word2idx = {w: i for i, w in enumerate(vocab)}
            self.idx2word = {i: w for i, w in enumerate(vocab)}
            self.blank_id = self.word2idx.get("<blank>", 0)
        else:
            log.warning("vocab.json not found at %s — gloss IDs may be missing", vocab_path)
            self.vocab    = []
            self.word2idx = {}
            self.idx2word = {}
            self.blank_id = 0

        log.info(
            "[ISignDataset] split=%s  samples=%d  vocab=%d  pose_dim=%d  preload_n=%d  pose_backend=%s",
            split,
            len(self),
            len(self.vocab),
            self.pose_dim,
            self.preload_n,
            self.pose_backend,
        )
        if self.preload_n > 0 and self.split == "train" and (
            self.augment
            or self.hard_negative_prob > 0
            or self.temporal_jitter > 0
            or self.frame_drop_prob > 0
            or self.pose_jitter_std > 0
        ):
            log.warning(
                "[ISignDataset] train preload caches the first sampled augmentations for the first %d samples",
                self.preload_n,
            )
        self._preload_samples()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.annotations)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _configure_pose_backend(self) -> None:
        if not self.use_poses:
            self.pose_backend = "npy"
            return

        if self.pose_backend not in {"auto", "npy", "lmdb"}:
            raise ValueError(f"Unsupported pose_backend: {self.pose_backend}")

        if self.pose_backend == "auto":
            self.pose_backend = "lmdb" if self.pose_lmdb_path.exists() else "npy"

        if self.pose_backend == "lmdb" and not self.pose_lmdb_path.exists():
            raise FileNotFoundError(
                f"Pose LMDB not found at {self.pose_lmdb_path}. "
                "Build it first or use --pose-backend npy."
            )

    def _get_pose_lmdb_env(self):
        global _LMDB_ENV_CACHE_PID

        current_pid = os.getpid()
        if _LMDB_ENV_CACHE_PID != current_pid:
            # Child workers (fork/spawn) must not reuse parent process handles.
            _LMDB_ENV_CACHE.clear()
            _LMDB_ENV_CACHE_PID = current_pid

        if self._pose_lmdb_env is not None and self._pose_lmdb_env_pid == current_pid:
            return self._pose_lmdb_env

        if self._pose_lmdb_env_pid != current_pid:
            self._pose_lmdb_env = None

        if self._pose_lmdb_env is not None:
            return self._pose_lmdb_env

        try:
            import lmdb
        except ImportError as exc:
            raise RuntimeError(
                "LMDB pose backend requested but `lmdb` is not installed. "
                "Install it with `pip install lmdb`."
            ) from exc

        lmdb_key = (
            str(self.pose_lmdb_path.resolve()),
            bool(self.pose_lmdb_path.is_dir()),
            bool(self.pose_lmdb_readahead),
        )

        shared_env = _LMDB_ENV_CACHE.get(lmdb_key)
        if shared_env is None:
            shared_env = lmdb.open(
                str(self.pose_lmdb_path),
                readonly=True,
                lock=False,
                readahead=self.pose_lmdb_readahead,
                meminit=False,
                subdir=self.pose_lmdb_path.is_dir(),
                max_readers=512,
            )
            _LMDB_ENV_CACHE[lmdb_key] = shared_env

        self._pose_lmdb_env = shared_env
        self._pose_lmdb_env_pid = current_pid
        with self._pose_lmdb_env.begin(write=False) as txn:
            meta_raw = txn.get(b"__meta__")
        if meta_raw:
            try:
                self._pose_lmdb_meta = json.loads(meta_raw.decode("utf-8"))
            except Exception:
                self._pose_lmdb_meta = {}
        self._pose_lmdb_value_format = str(self._pose_lmdb_meta.get("value_format", "npy")).lower()
        if self._pose_lmdb_value_format not in {"npy", "raw-f32"}:
            self._pose_lmdb_value_format = "npy"
        return self._pose_lmdb_env

    def _load_pose_array_from_lmdb(self, pose_key: str) -> Optional[np.ndarray]:
        env = self._get_pose_lmdb_env()
        with env.begin(write=False) as txn:
            raw = txn.get(pose_key.encode("utf-8"))
        if raw is None:
            return None
        if self._pose_lmdb_value_format == "raw-f32":
            try:
                shape = np.frombuffer(raw, dtype=np.int32, count=2)
                if shape.size != 2:
                    return None
                t = int(shape[0])
                d = int(shape[1])
                if t <= 0 or d <= 0:
                    return None
                arr = np.frombuffer(raw, dtype=np.float32, offset=8)
                if arr.size != t * d:
                    return None
                return arr.reshape(t, d)
            except Exception:
                return None
        try:
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            return arr
        except Exception:
            return None

    def _infer_pose_dim(self) -> int:
        """Find a stable pose feature width from the dataset annotations."""
        if self.pose_backend == "lmdb":
            meta_dim = int(self._pose_lmdb_meta.get("pose_dim", 0) or 0)
            if meta_dim > 0:
                return meta_dim
            for ann in self.annotations:
                pose_file = ann.get("pose_file")
                if not pose_file:
                    continue
                arr = self._load_pose_array_from_lmdb(str(pose_file))
                if arr is None or arr.ndim == 0:
                    continue
                if arr.ndim == 1:
                    return int(arr.shape[0]) if arr.shape[0] > 0 else 1
                return int(np.prod(arr.shape[1:])) if np.prod(arr.shape[1:]) > 0 else 1
            return 1

        for ann in self.annotations:
            pose_file = ann.get("pose_file")
            if not pose_file:
                continue
            pose_path = self.data_dir / pose_file
            if not pose_path.exists():
                continue
            try:
                arr = np.load(str(pose_path), mmap_mode="r")
            except Exception:
                continue
            if arr.ndim == 0:
                continue
            if arr.ndim == 1:
                return int(arr.shape[0]) if arr.shape[0] > 0 else 1
            return int(np.prod(arr.shape[1:])) if np.prod(arr.shape[1:]) > 0 else 1
        return 1

    def _zeros_pose(self) -> torch.Tensor:
        return torch.zeros(self.num_frames, self.pose_dim, dtype=torch.float32)

    def _empty_rgb(self) -> torch.Tensor:
        return torch.empty(0, self.num_frames, *self.frame_size, dtype=torch.float32)

    def _clone_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rgb": sample["rgb"].clone(),
            "pose": sample["pose"].clone(),
            "labels": sample["labels"].clone(),
            "length": int(sample["length"]),
            "name": str(sample["name"]),
            "sentence": str(sample["sentence"]),
        }

    def _preload_samples(self) -> None:
        """Warm a small prefix of samples into RAM to reduce HDD reads."""
        if self.preload_n <= 0:
            return

        for idx in range(self.preload_n):
            self._sample_cache[idx] = self._load_sample(idx)

        log.info(
            "[ISignDataset] split=%s preloaded %d samples into RAM",
            self.split,
            len(self._sample_cache),
        )

    def _match_pose_dim(self, arr: np.ndarray) -> np.ndarray:
        """Pad or trim pose features to the dataset-wide canonical width."""
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)
        current_dim = int(arr.shape[1]) if arr.ndim == 2 else 0
        if current_dim == self.pose_dim:
            return arr
        if current_dim > self.pose_dim:
            return arr[:, : self.pose_dim]
        pad_width = self.pose_dim - current_dim
        return np.pad(arr, ((0, 0), (0, pad_width)), mode="constant")

    def _build_frame_transform(self):
        if not self.use_albumentations:
            return None

        try:
            import albumentations as A
            import cv2
        except ImportError:
            log.warning("Albumentations unavailable; falling back to built-in RGB augmentations")
            self.use_albumentations = False
            return None

        transforms: List[Any] = []
        if self.albumentations_prob > 0:
            transforms.append(A.HorizontalFlip(p=0.5))
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=0.03,
                    scale_limit=0.05,
                    rotate_limit=5,
                    border_mode=cv2.BORDER_REPLICATE,
                    p=self.albumentations_prob,
                )
            )
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=self.brightness_jitter,
                    contrast_limit=min(0.2, self.brightness_jitter),
                    p=self.albumentations_prob,
                )
            )
        if self.motion_blur_prob > 0:
            transforms.append(
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5, p=1.0),
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    ],
                    p=self.motion_blur_prob,
                )
            )
        if self.coarse_dropout_prob > 0:
            transforms.append(
                A.CoarseDropout(
                    max_holes=4,
                    min_holes=1,
                    max_height=max(8, int(self.frame_size[0] * 0.18)),
                    min_height=max(4, int(self.frame_size[0] * 0.08)),
                    max_width=max(8, int(self.frame_size[1] * 0.18)),
                    min_width=max(4, int(self.frame_size[1] * 0.08)),
                    fill_value=0,
                    p=self.coarse_dropout_prob,
                )
            )

        if not transforms:
            return None

        self._albumentations = A
        return A.ReplayCompose(transforms)

    def _apply_consistent_frame_transform(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if self._frame_transform is None or not frames:
            return frames

        replay: Optional[Dict[str, Any]] = None
        transformed: List[np.ndarray] = []
        for frame in frames:
            if replay is None:
                result = self._frame_transform(image=frame)
                replay = result["replay"]
            else:
                result = self._albumentations.ReplayCompose.replay(replay, image=frame)
            transformed.append(result["image"])
        return transformed

    def _load_frames(self, frame_dir: Path) -> torch.Tensor:
        """
        Load JPEG frames, sample *num_frames* uniformly, normalise.

        Returns:
            Tensor  (C, T, H, W)  float32
        """
        try:
            import cv2
        except ImportError:
            return torch.zeros(3, self.num_frames, *self.frame_size, dtype=torch.float32)

        jpgs = sorted(frame_dir.glob("*.jpg"))
        if not jpgs:
            return torch.zeros(3, self.num_frames, *self.frame_size, dtype=torch.float32)

        # Uniform index sampling
        T    = len(jpgs)
        idxs = np.linspace(0, T - 1, self.num_frames, dtype=int)
        if self.temporal_jitter > 0:
            jitter = np.random.randint(-self.temporal_jitter, self.temporal_jitter + 1, size=len(idxs))
            idxs = np.clip(idxs + jitter, 0, T - 1)
        frames = []
        for i in idxs:
            img = cv2.imread(str(jpgs[i]))
            if img is None:
                img = np.zeros((*self.frame_size, 3), dtype=np.uint8)
            img = cv2.resize(img, (self.frame_size[1], self.frame_size[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        frames = self._apply_consistent_frame_transform(frames)
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)

        # Frame drop augmentation
        if self.frame_drop_prob > 0:
            keep = np.random.rand(arr.shape[0]) > self.frame_drop_prob
            if not np.any(keep):
                keep[np.random.randint(0, arr.shape[0])] = True
            kept = arr[keep]
            resample_idxs = np.linspace(0, len(kept) - 1, self.num_frames, dtype=int)
            arr = kept[resample_idxs]

        # Fallback RGB augmentations when albumentations is disabled.
        if self._frame_transform is None:
            if self.brightness_jitter > 0:
                scale = 1.0 + np.random.uniform(-self.brightness_jitter, self.brightness_jitter)
                arr = np.clip(arr * scale, 0.0, 1.0)
            if self.noise_std > 0:
                arr = np.clip(arr + np.random.normal(0, self.noise_std, size=arr.shape).astype(np.float32), 0.0, 1.0)

            if self.blur_prob > 0:
                try:
                    import cv2
                    if np.random.rand() < self.blur_prob:
                        blurred = []
                        for fr in arr:
                            fr_u8 = (fr * 255.0).astype(np.uint8)
                            fr_u8 = cv2.GaussianBlur(fr_u8, (3, 3), sigmaX=0.8)
                            blurred.append(fr_u8.astype(np.float32) / 255.0)
                        arr = np.stack(blurred, axis=0)
                except Exception:
                    pass

        arr = (arr - _MEAN) / _STD

        # Random horizontal flip augmentation
        if self._frame_transform is None and self.augment and np.random.rand() < 0.5:
            arr = arr[:, :, ::-1, :].copy()

        # (T, H, W, C) → (C, T, H, W)
        arr = arr.transpose(3, 0, 1, 2)
        return torch.tensor(arr, dtype=torch.float32)

    def _load_pose(self, pose_ref: str | Path) -> torch.Tensor:
        """
        Load pose keypoints and sample *num_frames* uniformly.

        Returns:
            Tensor  (T, D)  float32
        """
        arr: Optional[np.ndarray]
        if self.pose_backend == "lmdb":
            pose_key = str(pose_ref)
            arr = self._load_pose_array_from_lmdb(pose_key)
            if arr is None:
                return self._zeros_pose()
        else:
            pose_path = pose_ref if isinstance(pose_ref, Path) else self.data_dir / str(pose_ref)
            if not pose_path.exists():
                return self._zeros_pose()

            try:
                arr = np.load(str(pose_path)).astype(np.float32)
            except Exception:
                return self._zeros_pose()

        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim == 0:
            return self._zeros_pose()

        T = arr.shape[0]
        idxs = np.linspace(0, T - 1, self.num_frames, dtype=int)
        if self.temporal_jitter > 0:
            jitter = np.random.randint(-self.temporal_jitter, self.temporal_jitter + 1, size=len(idxs))
            idxs = np.clip(idxs + jitter, 0, T - 1)
        arr  = arr[idxs]                     # (num_frames, D)
        arr = self._match_pose_dim(arr)
        if self.pose_jitter_std > 0:
            arr = arr + np.random.normal(0, self.pose_jitter_std, size=arr.shape).astype(np.float32)
        return torch.tensor(arr, dtype=torch.float32)

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]

        if self.use_frames:
            frame_dir = self.data_dir / ann["frame_dir"]
            rgb = self._load_frames(frame_dir)
        else:
            rgb = self._empty_rgb()

        if self.use_poses:
            pose = self._load_pose(str(ann["pose_file"]))
        else:
            pose = self._zeros_pose()

        gloss_ids = ann.get("gloss_ids", [])
        labels = torch.tensor(gloss_ids, dtype=torch.long)

        return {
            "rgb": rgb,
            "pose": pose,
            "labels": labels,
            "length": self.num_frames,
            "name": ann["video_id"],
            "sentence": ann["sentence"],
        }

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]

        if self.hard_negative_prob > 0 and np.random.rand() < self.hard_negative_prob:
            rgb = self._empty_rgb() if not self.use_frames else torch.zeros(
                3, self.num_frames, *self.frame_size, dtype=torch.float32
            )
            pose = self._zeros_pose()
            labels = torch.zeros(0, dtype=torch.long)
            return {
                "rgb": rgb,
                "pose": pose,
                "labels": labels,
                "length": self.num_frames,
                "name": f"neg_{ann['video_id']}",
                "sentence": "",
            }

        cached = self._sample_cache.get(idx)
        if cached is not None:
            return self._clone_sample(cached)

        return self._load_sample(idx)


# ---------------------------------------------------------------------------
# Collate function (handles variable-length label sequences)
# ---------------------------------------------------------------------------

def isign_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of samples from ISignDataset into a padded batch."""
    rgb_list    = [s["rgb"]   for s in batch]
    pose_list   = [s["pose"]  for s in batch]
    label_list  = [s["labels"] for s in batch]
    lengths     = [s["length"] for s in batch]
    names       = [s["name"]   for s in batch]
    sentences   = [s["sentence"] for s in batch]

    # Stack RGB
    rgb_batch = torch.stack(rgb_list, dim=0)    # (B, C, T, H, W)

    # Pad pose to max D in batch
    max_d = max(p.shape[-1] for p in pose_list)
    padded_poses = []
    for p in pose_list:
        if p.shape[-1] < max_d:
            pad_width = max_d - p.shape[-1]
            p = torch.nn.functional.pad(p, (0, pad_width))
        padded_poses.append(p)
    pose_batch = torch.stack(padded_poses, dim=0)   # (B, T, D)

    # Pad labels
    max_label_len = max(len(l) for l in label_list)
    padded_labels = torch.zeros(len(batch), max_label_len, dtype=torch.long)
    label_lengths = torch.tensor([len(l) for l in label_list], dtype=torch.long)
    for i, l in enumerate(label_list):
        if len(l) > 0:
            padded_labels[i, : len(l)] = l

    return {
        "rgb":            rgb_batch,
        "pose":           pose_batch,
        "labels":         padded_labels,
        "label_lengths":  label_lengths,
        "lengths":        torch.tensor(lengths, dtype=torch.long),
        "names":          names,
        "sentences":      sentences,
    }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    num_frames: int = 64,
    frame_size: Tuple[int, int] = (224, 224),
    num_workers: int = 2,
    use_poses: bool = True,
    use_frames: bool = True,
    max_samples: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    hard_negative_prob: float = 0.0,
    temporal_jitter: int = 2,
    frame_drop_prob: float = 0.05,
    brightness_jitter: float = 0.15,
    blur_prob: float = 0.10,
    noise_std: float = 0.02,
    pose_jitter_std: float = 0.01,
    use_albumentations: bool = True,
    albumentations_prob: float = 0.35,
    motion_blur_prob: float = 0.10,
    coarse_dropout_prob: float = 0.08,
    preload_n: int = 0,
    pose_backend: str = "auto",
    pose_lmdb_path: Optional[str] = None,
    pose_lmdb_readahead: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build train / val / test DataLoaders plus vocabulary list.

    Returns:
        (train_loader, val_loader, test_loader, vocab)
    """
    common: Dict[str, Any] = dict(
        data_dir=data_dir,
        num_frames=num_frames,
        frame_size=frame_size,
        use_poses=use_poses,
        use_frames=use_frames,
        max_samples=max_samples,
        hard_negative_prob=hard_negative_prob,
        temporal_jitter=temporal_jitter,
        frame_drop_prob=frame_drop_prob,
        brightness_jitter=brightness_jitter,
        blur_prob=blur_prob,
        noise_std=noise_std,
        pose_jitter_std=pose_jitter_std,
        use_albumentations=use_albumentations,
        albumentations_prob=albumentations_prob,
        motion_blur_prob=motion_blur_prob,
        coarse_dropout_prob=coarse_dropout_prob,
        preload_n=preload_n,
        pose_backend=pose_backend,
        pose_lmdb_path=pose_lmdb_path,
        pose_lmdb_readahead=pose_lmdb_readahead,
    )

    train_ds = ISignDataset(split="train", augment=True, **common)
    val_ds   = ISignDataset(split="val",   augment=False, **common)
    test_ds  = ISignDataset(split="test",  augment=False, **common)

    loader_kwargs: Dict[str, Any] = dict(
        collate_fn=isign_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = max(2, int(prefetch_factor))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_ds.vocab
