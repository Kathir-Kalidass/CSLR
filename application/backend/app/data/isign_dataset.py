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

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

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
    ) -> None:
        self.data_dir   = Path(data_dir)
        self.split      = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment    = augment and (split == "train")
        self.use_poses  = use_poses
        self.use_frames = use_frames

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

        log.info("[ISignDataset] split=%s  samples=%d  vocab=%d", split, len(self), len(self.vocab))

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
        frames = []
        for i in idxs:
            img = cv2.imread(str(jpgs[i]))
            if img is None:
                img = np.zeros((*self.frame_size, 3), dtype=np.uint8)
            img = cv2.resize(img, (self.frame_size[1], self.frame_size[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)
        arr = (arr - _MEAN) / _STD

        # Random horizontal flip augmentation
        if self.augment and np.random.rand() < 0.5:
            arr = arr[:, :, ::-1, :].copy()

        # (T, H, W, C) → (C, T, H, W)
        arr = arr.transpose(3, 0, 1, 2)
        return torch.from_numpy(arr)

    def _load_pose(self, pose_path: Path) -> torch.Tensor:
        """
        Load pose keypoints and sample *num_frames* uniformly.

        Returns:
            Tensor  (T, D)  float32
        """
        if not pose_path.exists():
            return torch.zeros(self.num_frames, 1, dtype=torch.float32)

        try:
            arr = np.load(str(pose_path)).astype(np.float32)
        except Exception:
            return torch.zeros(self.num_frames, 1, dtype=torch.float32)

        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)

        T = arr.shape[0]
        idxs = np.linspace(0, T - 1, self.num_frames, dtype=int)
        arr  = arr[idxs]                     # (num_frames, D)
        return torch.from_numpy(arr)

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]

        # RGB frames
        if self.use_frames:
            frame_dir = self.data_dir / ann["frame_dir"]
            rgb = self._load_frames(frame_dir)
        else:
            rgb = torch.zeros(3, self.num_frames, *self.frame_size, dtype=torch.float32)

        # Pose
        if self.use_poses:
            pose_path = self.data_dir / ann["pose_file"]
            pose = self._load_pose(pose_path)
        else:
            pose = torch.zeros(self.num_frames, 1, dtype=torch.float32)

        # Labels
        gloss_ids = ann.get("gloss_ids", [])
        labels    = torch.tensor(gloss_ids, dtype=torch.long)

        return {
            "rgb":      rgb,            # (C, T, H, W)
            "pose":     pose,           # (T, D)
            "labels":   labels,         # (L,)
            "length":   self.num_frames,
            "name":     ann["video_id"],
            "sentence": ann["sentence"],
        }


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
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build train / val / test DataLoaders plus vocabulary list.

    Returns:
        (train_loader, val_loader, test_loader, vocab)
    """
    common = dict(
        data_dir=data_dir,
        num_frames=num_frames,
        frame_size=frame_size,
        use_poses=use_poses,
        use_frames=use_frames,
        max_samples=max_samples,
    )

    train_ds = ISignDataset(split="train", augment=True, **common)
    val_ds   = ISignDataset(split="val",   augment=False, **common)
    test_ds  = ISignDataset(split="test",  augment=False, **common)

    loader_kwargs = dict(
        collate_fn=isign_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_ds.vocab
