"""
Video Dataset for CSLR Training
Handles video loading, augmentation, and batching
"""

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypedDict, cast

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from app.core.logging import logger


class DatasetSample(TypedDict):
    rgb: torch.Tensor
    pose: torch.Tensor
    labels: torch.Tensor
    length: int
    name: str


class BatchSample(TypedDict):
    rgb: torch.Tensor
    pose: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor
    names: List[str]


class CSLRVideoDataset(Dataset[DatasetSample]):
    """
    Dataset for Sign Language Recognition
    Loads RGB videos and pose keypoints
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        video_size: Tuple[int, int] = (224, 224),
        num_frames: int = 64,
        vocab_file: Optional[str] = None,
        augment: bool = True,
    ):
        """
        Args:
            data_dir: Root data directory
            split: Dataset split (train/val/test)
            video_size: Target video frame size
            num_frames: Number of frames to sample
            vocab_file: Path to vocabulary JSON
            augment: Apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.video_size = video_size
        self.num_frames = num_frames
        self.augment = augment and split == "train"
        
        # Load annotations
        ann_file = self.data_dir / f"{split}.json"
        with open(ann_file, 'r') as f:
            self.annotations: List[dict[str, Any]] = json.load(f)
        
        # Load vocabulary
        if vocab_file:
            with open(vocab_file, 'r') as f:
                self.vocab = json.load(f)
            self.word2idx = {w: i for i, w in enumerate(self.vocab)}
            self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        else:
            self.vocab = None
            self.word2idx = None
        
        logger.info(f"Loaded {len(self.annotations)} samples for {split}")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def load_video(self, video_path: Path) -> torch.Tensor:
        """
        Load and preprocess RGB video
        
        Returns:
            Tensor of shape (C, T, H, W)
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize
            frame = cv2.resize(frame, self.video_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            logger.warning(f"Empty video: {video_path}")
            frames = [np.zeros((*self.video_size, 3), dtype=np.uint8)]
        
        # Sample frames
        frames = self.sample_frames(frames, self.num_frames)
        
        # Convert to tensor (T, H, W, C) -> (C, T, H, W)
        video = np.stack(frames, axis=0)
        video = torch.from_numpy(video).permute(3, 0, 1, 2).float()
        
        # Normalize to [0, 1]
        video = video / 255.0
        
        # Augmentation
        if self.augment:
            video = self.augment_video(video)
        
        return video
    
    def load_pose(self, pose_path: Path) -> torch.Tensor:
        """
        Load pose keypoints
        
        Returns:
            Tensor of shape (T, N, D) where N=75 landmarks, D=2 (x,y)
        """
        if not pose_path.exists():
            # Return zeros if pose not available
            return torch.zeros(self.num_frames, 75, 2)
        
        # Load from numpy or JSON
        if pose_path.suffix == '.npy':
            keypoints = np.load(pose_path)
        elif pose_path.suffix == '.json':
            with open(pose_path, 'r') as f:
                data = json.load(f)
            keypoints = np.array(data['keypoints'])
        else:
            return torch.zeros(self.num_frames, 75, 2)
        
        # Sample frames
        keypoints = self.sample_frames(list(keypoints), self.num_frames)
        keypoints = np.stack(keypoints, axis=0)
        
        return torch.from_numpy(keypoints).float()
    
    def sample_frames(self, frames: List, target_len: int) -> List:
        """Uniformly sample frames"""
        total = len(frames)
        
        if total >= target_len:
            # Sample uniformly
            indices = np.linspace(0, total - 1, target_len, dtype=int)
        else:
            # Repeat last frame
            indices = list(range(total)) + [total - 1] * (target_len - total)
        
        return [frames[i] for i in indices]
    
    def augment_video(self, video: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation"""
        # Random horizontal flip
        if np.random.rand() < 0.5:
            video = torch.flip(video, dims=[3])
        
        # Random brightness/contrast
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-0.1, 0.1)
            video = video * alpha + beta
            video = torch.clamp(video, 0, 1)
        
        return video
    
    def encode_labels(self, glosses: List[str]) -> Tuple[torch.Tensor, int]:
        """
        Encode gloss labels to indices
        
        Returns:
            (label_tensor, length)
        """
        if self.word2idx is None:
            raise ValueError("Vocabulary not loaded")
        
        indices = [self.word2idx.get(g, 0) for g in glosses]
        return torch.tensor(indices, dtype=torch.long), len(indices)
    
    def __getitem__(self, idx: int) -> DatasetSample:
        """Get one sample"""
        ann = self.annotations[idx]
        
        # Paths
        video_path = self.data_dir / ann['video_path']
        pose_path = self.data_dir / ann.get('pose_path', 'poses/' + ann['video_path'].replace('.mp4', '.npy'))
        
        # Load data
        rgb = self.load_video(video_path)
        pose = self.load_pose(Path(pose_path))
        
        # Encode labels
        glosses = cast(List[str], ann['glosses'])
        if self.word2idx:
            labels, length = self.encode_labels(glosses)
        else:
            labels = torch.zeros(1, dtype=torch.long)
            length = 0
        
        return {
            'rgb': rgb,
            'pose': pose,
            'labels': labels,
            'length': length,
            'name': str(ann.get('name', f"sample_{idx}")),
        }


def collate_fn(batch: List[DatasetSample]) -> BatchSample:
    """
    Collate function for DataLoader
    Handles variable-length sequences
    """
    rgb = torch.stack([b['rgb'] for b in batch])
    pose = torch.stack([b['pose'] for b in batch])
    
    # Pad labels to same length
    max_len = max(b['length'] for b in batch)
    labels = torch.zeros(len(batch), max_len, dtype=torch.long)
    lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
    
    for i, b in enumerate(batch):
        labels[i, :b['length']] = b['labels']
    
    return {
        'rgb': rgb,
        'pose': pose,
        'labels': labels,
        'lengths': lengths,
        'names': [b['name'] for b in batch],
    }
