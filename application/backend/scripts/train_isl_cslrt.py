"""
Train a CSLR model on ISL_CSLRT_Corpus with RGB + Pose + CTC.

This script is aligned with project docs:
- Dual-stream features (RGB + Pose)
- Gated/attention fusion
- Temporal sequence modeling (BiLSTM/Transformer)
- CTC training and decoding
"""

from __future__ import annotations

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import csv
import difflib
import hashlib
import importlib
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from typing import Union

try:
    from torch.amp.grad_scaler import GradScaler  # PyTorch 2.x correct location
except ImportError:
    from torch.cuda.amp import GradScaler  # fallback for PyTorch < 2.0

try:
    import cv2
except Exception:  # pragma: no cover - validated at runtime
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - handled at runtime
    YOLO = None


BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.pipeline.module2_feature.fusion import FeatureFusion  # noqa: E402
from app.pipeline.module2_feature.pose_stream import PoseStream  # noqa: E402
from app.pipeline.module2_feature.rgb_stream import RGBStream  # noqa: E402
from app.pipeline.module3_sequence.ctc_layer import CTCLayer  # noqa: E402
from app.pipeline.module3_sequence.temporal_model import TemporalModel  # noqa: E402

LOGGER = logging.getLogger("isl_cslrt_train")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FRAME_NUMBER_RE = re.compile(r"\d+")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4", ".AVI", ".MOV", ".MKV", ".WEBM")
POSE_KEYPOINTS = 17
POSE_FEATURE_DIM = POSE_KEYPOINTS * 2

# Known naming mismatches between CSV and directory names.
SENTENCE_ALIASES = {
    "which collegeschool are you from": "which college school are you from",
}

def make_grad_scaler(enabled: bool):
    return GradScaler(enabled=enabled)


def get_autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def log_gpu() -> None:
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**3
        max_mem = torch.cuda.max_memory_allocated() / 1024**3
        LOGGER.info("GPU memory used: %.2f GB (max %.2f GB)", mem, max_mem)


def gpu_utilization() -> float:
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        if total_memory > 0:
            return (torch.cuda.memory_allocated() / total_memory) * 100.0
    return 0.0


def compute_grad_norm(model: nn.Module) -> float:
    total_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_norm_sq += grad_norm * grad_norm
    return total_norm_sq ** 0.5


@dataclass
class SequenceSample:
    sample_id: str
    sentence: str
    sentence_norm: str
    sequence_id: str
    frame_paths: List[str]
    gloss_tokens: List[str]
    video_path: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_sentence(text: str) -> str:
    normalized = text.strip().lower()
    normalized = normalized.replace("’", "'").replace("`", "'")
    normalized = normalize_whitespace(normalized)
    normalized = normalized.replace("collegeschool", "college school")
    return normalized


def tokenize_gloss(gloss_text: str) -> List[str]:
    cleaned = gloss_text.strip().upper()
    cleaned = cleaned.replace(",", " ").replace(".", " ")
    cleaned = normalize_whitespace(cleaned)
    return [token for token in cleaned.split(" ") if token]


def parse_pose_hidden_dims(value: str) -> List[int]:
    dims = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not dims:
        raise ValueError("Pose hidden dimensions cannot be empty.")
    return dims


def load_sentence_gloss_map(csv_path: Path) -> Dict[str, List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Gloss CSV not found: {csv_path}")

    sentence_map: Dict[str, List[str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sentence_raw = normalize_whitespace(row.get("Sentence", ""))
            gloss_raw = normalize_whitespace(row.get("SIGN GLOSSES", ""))
            if not sentence_raw or not gloss_raw:
                continue
            sentence_map[normalize_sentence(sentence_raw)] = tokenize_gloss(gloss_raw)

    if not sentence_map:
        raise RuntimeError(f"No sentence-gloss pairs loaded from {csv_path}")
    return sentence_map


def frame_sort_key(path: Path) -> Tuple[int, str]:
    matches = FRAME_NUMBER_RE.findall(path.stem)
    frame_idx = int(matches[-1]) if matches else -1
    return frame_idx, path.name


def resolve_gloss_tokens(
    sentence_name: str,
    sentence_map: Dict[str, List[str]],
) -> Optional[List[str]]:
    normalized = normalize_sentence(sentence_name)

    if normalized in sentence_map:
        return sentence_map[normalized]

    alias = SENTENCE_ALIASES.get(normalized)
    if alias and alias in sentence_map:
        return sentence_map[alias]

    de_punct = normalize_whitespace(re.sub(r"[^a-z0-9()'\s]", " ", normalized))
    if de_punct in sentence_map:
        return sentence_map[de_punct]

    close = difflib.get_close_matches(
        normalized, sentence_map.keys(), n=1, cutoff=0.92
    )
    if close:
        return sentence_map[close[0]]
    return None


def collect_sentence_samples(
    corpus_root: Path,
    sentence_map: Dict[str, List[str]],
) -> Tuple[List[SequenceSample], List[str]]:
    sentence_root = corpus_root / "Frames_Sentence_Level"
    video_root = corpus_root / "Videos_Sentence_Level"
    if not sentence_root.exists():
        raise FileNotFoundError(f"Sentence-level frame folder not found: {sentence_root}")

    samples: List[SequenceSample] = []
    unmatched_sentences: List[str] = []

    sentence_dirs = [p for p in sentence_root.iterdir() if p.is_dir()]
    sentence_dirs.sort(key=lambda p: p.name.lower())

    for sentence_dir in sentence_dirs:
        gloss_tokens = resolve_gloss_tokens(sentence_dir.name, sentence_map)
        if gloss_tokens is None:
            unmatched_sentences.append(sentence_dir.name)
            continue

        sequence_dirs = [p for p in sentence_dir.iterdir() if p.is_dir()]
        sequence_dirs.sort(key=lambda p: p.name)
        for seq_dir in sequence_dirs:
            frame_files = sorted(seq_dir.glob("*.jpg"), key=frame_sort_key)
            if not frame_files:
                continue

            sample = SequenceSample(
                sample_id=f"{sentence_dir.name}/{seq_dir.name}",
                sentence=sentence_dir.name,
                sentence_norm=normalize_sentence(sentence_dir.name),
                sequence_id=seq_dir.name,
                frame_paths=[str(path) for path in frame_files],
                gloss_tokens=gloss_tokens,
            )
            samples.append(sample)

    if video_root.exists():
        video_sentence_dirs = [p for p in video_root.iterdir() if p.is_dir()]
        video_sentence_dirs.sort(key=lambda p: p.name.lower())

        for sentence_dir in video_sentence_dirs:
            gloss_tokens = resolve_gloss_tokens(sentence_dir.name, sentence_map)
            if gloss_tokens is None:
                unmatched_sentences.append(sentence_dir.name)
                continue

            video_files = [
                p for p in sentence_dir.iterdir()
                if p.is_file() and p.suffix in VIDEO_EXTENSIONS
            ]
            video_files.sort(key=lambda p: p.name.lower())
            for video_file in video_files:
                samples.append(
                    SequenceSample(
                        sample_id=f"{sentence_dir.name}/video::{video_file.stem}",
                        sentence=sentence_dir.name,
                        sentence_norm=normalize_sentence(sentence_dir.name),
                        sequence_id=f"video::{video_file.name}",
                        frame_paths=[],
                        gloss_tokens=gloss_tokens,
                        video_path=str(video_file),
                    )
                )

    if not samples:
        raise RuntimeError("No usable samples found in the sentence-level dataset.")
    return samples, unmatched_sentences


def stratified_split(
    samples: Sequence[SequenceSample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[SequenceSample], List[SequenceSample], List[SequenceSample]]:
    by_class: Dict[str, List[SequenceSample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.sentence_norm].append(sample)

    rng = random.Random(seed)
    train_samples: List[SequenceSample] = []
    val_samples: List[SequenceSample] = []
    test_samples: List[SequenceSample] = []

    for class_samples in by_class.values():
        class_list = list(class_samples)
        rng.shuffle(class_list)
        n_total = len(class_list)

        n_val = 0
        n_test = 0
        if val_ratio > 0 and n_total >= 3:
            n_val = max(1, int(round(n_total * val_ratio)))
        if test_ratio > 0 and n_total >= 5:
            n_test = max(1, int(round(n_total * test_ratio)))

        while n_total - n_val - n_test < 1:
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break

        val_samples.extend(class_list[:n_val])
        test_samples.extend(class_list[n_val : n_val + n_test])
        train_samples.extend(class_list[n_val + n_test :])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    return train_samples, val_samples, test_samples


def build_vocab(samples: Sequence[SequenceSample]) -> Tuple[Dict[str, int], Dict[int, str]]:
    tokens = sorted({token for sample in samples for token in sample.gloss_tokens})
    token_to_id = {token: idx + 1 for idx, token in enumerate(tokens)}  # blank=0
    id_to_token = {0: "<blank>"}
    id_to_token.update({idx + 1: token for idx, token in enumerate(tokens)})
    return token_to_id, id_to_token


def compute_dataset_report(
    corpus_root: Path,
    sentence_map: Dict[str, List[str]],
    samples: Sequence[SequenceSample],
    unmatched_sentences: Sequence[str],
) -> Dict[str, object]:
    sentence_root = corpus_root / "Frames_Sentence_Level"
    video_root = corpus_root / "Videos_Sentence_Level"
    word_root = corpus_root / "Frames_Word_Level"

    sentence_dirs = [p for p in sentence_root.iterdir() if p.is_dir()] if sentence_root.exists() else []
    video_dirs = [p for p in video_root.iterdir() if p.is_dir()] if video_root.exists() else []
    sentence_names = sorted({p.name for p in sentence_dirs}.union({p.name for p in video_dirs}))
    sentence_norm_set = {normalize_sentence(name) for name in sentence_names}

    csv_only = sorted(norm for norm in sentence_map.keys() if norm not in sentence_norm_set)
    used_sentence_classes = sorted({sample.sentence for sample in samples})

    frame_counts_by_class: Counter[str] = Counter()
    seq_counts_by_class: Counter[str] = Counter()
    sequence_lengths: List[int] = []
    for sample in samples:
        frame_count = len(sample.frame_paths)
        frame_counts_by_class[sample.sentence] += frame_count
        seq_counts_by_class[sample.sentence] += 1
        sequence_lengths.append(frame_count)

    top_frame_classes = [
        {"sentence": sent, "frames": frame_counts_by_class[sent], "sequences": seq_counts_by_class[sent]}
        for sent, _ in frame_counts_by_class.most_common(12)
    ]

    report = {
        "sentence_classes_in_dataset": len(sentence_names),
        "sentence_classes_with_gloss_mapping": len(used_sentence_classes),
        "word_classes_in_dataset": len([p for p in word_root.iterdir() if p.is_dir()]) if word_root.exists() else 0,
        "total_sentence_sequences": len(samples),
        "total_sentence_frames": int(sum(sequence_lengths)),
        "avg_frames_per_sequence": float(np.mean(sequence_lengths)) if sequence_lengths else 0.0,
        "min_frames_per_sequence": int(min(sequence_lengths)) if sequence_lengths else 0,
        "max_frames_per_sequence": int(max(sequence_lengths)) if sequence_lengths else 0,
        "csv_rows": len(sentence_map),
        "csv_entries_missing_in_dataset": csv_only,
        "dataset_entries_missing_in_csv": sorted(set(unmatched_sentences)),
        "top_classes_by_frames": top_frame_classes,
    }
    return report


class PoseXYExtractor:
    """
    YOLOv8-pose extractor that outputs normalized (17,2) flattened to 34 dims.
    """

    def __init__(self) -> None:
        self.model = self._load_model()

    @staticmethod
    def _load_model():
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Install with: pip install ultralytics")
        try:
            return YOLO("yolov8n-pose.pt")
        except Exception as exc:
            raise RuntimeError(
                f"YOLOv8 pose model initialization failed: {exc}. "
                "Try: pip install ultralytics"
            )

    @staticmethod
    def _normalize(points17: np.ndarray) -> np.ndarray:
        left_shoulder = points17[5]
        right_shoulder = points17[6]
        center = (left_shoulder + right_shoulder) / 2.0
        torso_scale = float(np.linalg.norm(left_shoulder - right_shoulder))
        if torso_scale < 1e-4:
            torso_scale = 1.0
        normalized = (points17 - center) / torso_scale
        normalized = np.clip(normalized, -1.5, 1.5)
        return normalized.astype(np.float32)

    @staticmethod
    def _extract_xy(result: Any) -> np.ndarray:
        if result is None or getattr(result, "keypoints", None) is None:
            return np.zeros((POSE_KEYPOINTS, 2), dtype=np.float32)
        keypoints_xy = result.keypoints.xy
        if keypoints_xy is None:
            return np.zeros((POSE_KEYPOINTS, 2), dtype=np.float32)

        points = keypoints_xy[0]
        if torch.is_tensor(points):
            points = points.detach().cpu().numpy()
        else:
            points = np.asarray(points)
        points = points.astype(np.float32)

        padded = np.zeros((POSE_KEYPOINTS, 2), dtype=np.float32)
        valid = min(POSE_KEYPOINTS, points.shape[0])
        padded[:valid] = points[:valid]
        return padded

    def extract(self, frame_rgb: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return np.zeros((POSE_FEATURE_DIM,), dtype=np.float32)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        try:
            results = self.model.predict(source=frame_bgr, verbose=False, max_det=1)
        except Exception:
            return np.zeros((POSE_FEATURE_DIM,), dtype=np.float32)

        if not results:
            return np.zeros((POSE_FEATURE_DIM,), dtype=np.float32)

        points17 = self._extract_xy(results[0])
        normalized = self._normalize(points17)
        return normalized.reshape(-1)

    def close(self) -> None:
        return None

    def __del__(self) -> None:
        self.close()


class ISLCSLRTDataset(Dataset):
    """
    Dataset for sentence-level CSLR with on-demand pose extraction and caching.
    """

    def __init__(
        self,
        samples: Sequence[SequenceSample],
        token_to_id: Dict[str, int],
        num_frames: Optional[int] = None,
        image_size: int = 224,
        augment: bool = False,
        use_pose: bool = True,
        pose_cache_dir: Optional[Path] = None,
        target_fps: int = 8,
        min_video_frames: int = 8,
    ) -> None:
        self.samples = list(samples)
        self.token_to_id = token_to_id
        self.num_frames = num_frames
        self.target_fps = max(1, int(target_fps))
        self.min_video_frames = max(1, int(min_video_frames))
        self.image_size = image_size
        self.augment = augment
        self.use_pose = use_pose
        self.pose_cache_dir = pose_cache_dir
        if self.pose_cache_dir is not None:
            self.pose_cache_dir.mkdir(parents=True, exist_ok=True)
        self._pose_extractor: Optional[PoseXYExtractor] = None
        self._pose_unavailable_warned = False

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pose_extractor"] = None
        return state

    def _get_pose_extractor(self) -> PoseXYExtractor:
        if self._pose_extractor is None:
            self._pose_extractor = PoseXYExtractor()
        return self._pose_extractor

    @staticmethod
    def _sample_indices(total_frames: int, target_frames: Optional[int]) -> List[int]:
        if total_frames <= 0:
            if target_frames is None:
                return [0]
            return [0] * target_frames
        if target_frames is None:
            return list(range(total_frames))
        if total_frames >= target_frames:
            return list(np.linspace(0, total_frames - 1, target_frames, dtype=int))
        indices = list(range(total_frames))
        indices.extend([total_frames - 1] * (target_frames - total_frames))
        return indices

    @staticmethod
    def _safe_read_rgb(path: Path) -> np.ndarray:
        if cv2 is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        frame_bgr = cv2.imread(str(path))
        if frame_bgr is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _load_video_rgb_frames(self, video_path: Path) -> List[np.ndarray]:
        if cv2 is None:
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.min_video_frames)]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.min_video_frames)]

        original_fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = (total_frames / max(original_fps, 1.0)) if total_frames > 0 else 0.0

        target_frames = max(self.min_video_frames, int(duration_sec * self.target_fps))
        if self.num_frames is not None:
            target_frames = self.num_frames

        indices = self._sample_indices(total_frames if total_frames > 0 else target_frames, target_frames)
        frames: List[np.ndarray] = []
        last_valid = np.zeros((224, 224, 3), dtype=np.uint8)

        for frame_idx in indices:
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                frames.append(last_valid)
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            last_valid = frame_rgb
            frames.append(frame_rgb)

        cap.release()
        if len(frames) < target_frames:
            frames.extend([last_valid] * (target_frames - len(frames)))
        return frames[:target_frames]

    def _augment_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        frame = frame_rgb
        if random.random() < 0.40:
            alpha = random.uniform(0.85, 1.15)  # contrast
            beta = random.uniform(-12.0, 12.0)  # brightness
            frame = np.clip(frame.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if cv2 is None:
            return frame
        if random.random() < 0.20:
            k = random.choice((3, 5))
            frame = cv2.GaussianBlur(frame, (k, k), 0)
        return frame

    def _rgb_to_tensor(self, frame_rgb: np.ndarray) -> torch.Tensor:
        if cv2 is None:
            raise RuntimeError("OpenCV is required. Install with: pip install opencv-python")
        resized = cv2.resize(
            frame_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        normed = resized.astype(np.float32) / 255.0
        normed = (normed - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(normed, (2, 0, 1))
        return torch.from_numpy(chw)

    def _pose_cache_path(self, sample: SequenceSample) -> Optional[Path]:
        if self.pose_cache_dir is None:
            return None
        digest = hashlib.sha1(sample.sample_id.encode("utf-8")).hexdigest()[:16]
        if self.num_frames is None:
            cache_tag = f"varfps{self.target_fps}"
        else:
            cache_tag = f"t{self.num_frames}"
        return self.pose_cache_dir / f"{digest}_{cache_tag}_d{POSE_FEATURE_DIM}.npy"

    def _compute_pose_array(
        self,
        selected_paths: Optional[Sequence[Path]] = None,
        selected_frames: Optional[Sequence[np.ndarray]] = None,
    ) -> np.ndarray:
        seq_len = 0
        if selected_frames is not None:
            seq_len = len(selected_frames)
        elif selected_paths is not None:
            seq_len = len(selected_paths)
        elif self.num_frames is not None:
            seq_len = self.num_frames
        seq_len = max(1, seq_len)

        if not self.use_pose:
            return np.zeros((seq_len, POSE_FEATURE_DIM), dtype=np.float32)

        try:
            extractor = self._get_pose_extractor()
        except Exception as exc:
            if not self._pose_unavailable_warned:
                LOGGER.warning(
                    "Pose extraction unavailable (%s). Falling back to zero pose vectors.",
                    exc,
                )
                self._pose_unavailable_warned = True
            return np.zeros((seq_len, POSE_FEATURE_DIM), dtype=np.float32)

        poses: List[np.ndarray] = []
        frame_iterable: List[np.ndarray] = []
        if selected_frames is not None:
            frame_iterable = list(selected_frames)
        elif selected_paths is not None:
            frame_iterable = [self._safe_read_rgb(path) for path in selected_paths]

        for frame_rgb in frame_iterable:
            try:
                pose_vec = extractor.extract(frame_rgb)
            except Exception:
                pose_vec = np.zeros((POSE_FEATURE_DIM,), dtype=np.float32)
            poses.append(pose_vec)
        if not poses:
            poses = [np.zeros((POSE_FEATURE_DIM,), dtype=np.float32) for _ in range(seq_len)]
        return np.stack(poses, axis=0).astype(np.float32)

    def _load_or_extract_pose(
        self,
        sample: SequenceSample,
        selected_paths: Optional[Sequence[Path]] = None,
        selected_frames: Optional[Sequence[np.ndarray]] = None,
    ) -> np.ndarray:
        seq_len = len(selected_frames) if selected_frames is not None else len(selected_paths or [])
        seq_len = max(1, seq_len)
        if not self.use_pose:
            return np.zeros((seq_len, POSE_FEATURE_DIM), dtype=np.float32)

        cache_path = self._pose_cache_path(sample)
        if cache_path is not None and cache_path.exists():
            cached = np.load(str(cache_path))
            if cached.shape == (seq_len, POSE_FEATURE_DIM):
                return cached.astype(np.float32)

        pose_array = self._compute_pose_array(selected_paths=selected_paths, selected_frames=selected_frames)
        if cache_path is not None:
            np.save(str(cache_path), pose_array)
        return pose_array

    def prepare_pose_cache(self, log_every: int = 25) -> int:
        if not self.use_pose:
            LOGGER.info("Pose cache skipped because --no-pose was used.")
            return 0
        if self.pose_cache_dir is None:
            raise ValueError("Pose cache directory is not configured.")

        created = 0
        for idx, sample in enumerate(self.samples, start=1):
            cache_path = self._pose_cache_path(sample)
            if cache_path is None or cache_path.exists():
                continue
            if sample.video_path:
                rgb_frames = self._load_video_rgb_frames(Path(sample.video_path))
                pose_array = self._compute_pose_array(selected_frames=rgb_frames)
            else:
                frame_paths = [Path(p) for p in sample.frame_paths]
                indices = self._sample_indices(len(frame_paths), self.num_frames)
                selected = [frame_paths[i] for i in indices]
                pose_array = self._compute_pose_array(selected_paths=selected)
            np.save(str(cache_path), pose_array)
            created += 1
            if idx % log_every == 0:
                LOGGER.info("Pose cache progress: %d/%d", idx, len(self.samples))
        return created

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        selected_paths: List[Path] = []
        selected_rgb_frames: List[np.ndarray] = []
        if sample.video_path:
            selected_rgb_frames = self._load_video_rgb_frames(Path(sample.video_path))
        else:
            frame_paths = [Path(p) for p in sample.frame_paths]
            indices = self._sample_indices(len(frame_paths), self.num_frames)
            selected_paths = [frame_paths[i] for i in indices]
            selected_rgb_frames = [self._safe_read_rgb(frame_path) for frame_path in selected_paths]

        rgb_frames: List[torch.Tensor] = []
        for frame_rgb in selected_rgb_frames:
            if self.augment:
                frame_rgb = self._augment_rgb(frame_rgb)
            rgb_frames.append(self._rgb_to_tensor(frame_rgb))
        rgb_tensor = torch.stack(rgb_frames, dim=0).float()  # (T, C, H, W)

        pose_array = self._load_or_extract_pose(
            sample,
            selected_paths=selected_paths if selected_paths else None,
            selected_frames=selected_rgb_frames,
        )
        pose_tensor = torch.from_numpy(pose_array).float()  # (T, 34)

        token_ids = [self.token_to_id[token] for token in sample.gloss_tokens if token in self.token_to_id]
        target_tensor = torch.tensor(token_ids, dtype=torch.long)

        return {
            "rgb": rgb_tensor,
            "pose": pose_tensor,
            "target_ids": target_tensor,
            "target_list": token_ids,
            "target_text": sample.gloss_tokens,
            "target_length": len(token_ids),
            "input_length": int(rgb_tensor.shape[0]),
            "sample_id": sample.sample_id,
        }

    def __del__(self) -> None:
        if self._pose_extractor is not None:
            self._pose_extractor.close()


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    max_len = max(int(item["rgb"].shape[0]) for item in batch)  # type: ignore[index]

    rgb_batch: List[torch.Tensor] = []
    pose_batch: List[torch.Tensor] = []
    for item in batch:
        rgb_item = item["rgb"]  # type: ignore[index]
        pose_item = item["pose"]  # type: ignore[index]
        assert isinstance(rgb_item, torch.Tensor)
        assert isinstance(pose_item, torch.Tensor)

        pad_len = max_len - int(rgb_item.shape[0])
        if pad_len > 0:
            rgb_item = torch.cat([rgb_item, rgb_item[-1:].repeat(pad_len, 1, 1, 1)], dim=0)
            pose_item = torch.cat([pose_item, pose_item[-1:].repeat(pad_len, 1)], dim=0)

        rgb_batch.append(rgb_item)
        pose_batch.append(pose_item)

    rgb = torch.stack(rgb_batch, dim=0)
    pose = torch.stack(pose_batch, dim=0)

    target_lengths = torch.tensor(
        [int(item["target_length"]) for item in batch], dtype=torch.long  # type: ignore[arg-type]
    )
    input_lengths = torch.tensor(
        [int(item["rgb"].shape[0]) for item in batch], dtype=torch.long  # type: ignore[index]
    )

    max_target = int(target_lengths.max().item()) if len(batch) > 0 else 0
    targets = torch.zeros((len(batch), max_target), dtype=torch.long)
    target_lists: List[List[int]] = []
    target_texts: List[List[str]] = []
    sample_ids: List[str] = []

    for i, item in enumerate(batch):
        target_ids = item["target_ids"]  # type: ignore[index]
        assert isinstance(target_ids, torch.Tensor)
        targets[i, : target_ids.numel()] = target_ids
        target_lists.append(list(item["target_list"]))  # type: ignore[index]
        target_texts.append(list(item["target_text"]))  # type: ignore[index]
        sample_ids.append(str(item["sample_id"]))

    return {
        "rgb": rgb,
        "pose": pose,
        "targets": targets,
        "target_lengths": target_lengths,
        "input_lengths": input_lengths,
        "target_lists": target_lists,
        "target_texts": target_texts,
        "sample_ids": sample_ids,
    }


class CSLRCTCModel(nn.Module):
    """
    RGB + Pose dual-stream network for CTC sequence training.
    """

    def __init__(
        self,
        vocab_size: int,
        backbone: str = "resnet18",
        pose_hidden_dims: Optional[List[int]] = None,
        feature_dim: int = 512,
        fusion_type: str = "gated_attention",
        temporal_model: str = "bilstm",
        temporal_hidden_dim: int = 256,
        temporal_layers: int = 2,
        dropout: float = 0.2,
        pretrained_rgb: bool = True,
        rgb_backbone_chunk_size: int = 0,
    ) -> None:
        super().__init__()

        pose_hidden_dims = pose_hidden_dims or [512, 256]

        self.rgb_stream = RGBStream(
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=pretrained_rgb,
            freeze_backbone=False,
            dropout=dropout,
            backbone_chunk_size=rgb_backbone_chunk_size,
        )
        self.pose_stream = PoseStream(
            input_dim=POSE_FEATURE_DIM,
            hidden_dims=pose_hidden_dims,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        self.fusion = FeatureFusion(
            rgb_dim=feature_dim,
            pose_dim=feature_dim,
            fusion_dim=feature_dim,
            fusion_type=fusion_type,
        )
        self.temporal = TemporalModel(
            input_dim=feature_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_layers,
            vocab_size=vocab_size,
            model_type=temporal_model,
            dropout=dropout,
        )

    def forward(self, rgb: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, T, C, H, W)
            pose: (B, T, 34)
        Returns:
            logits: (B, T, vocab_size + 1)
        """
        rgb_features = self.rgb_stream(rgb)
        pose_features = self.pose_stream(pose)
        fused, _, _ = self.fusion(rgb_features, pose_features)
        logits = self.temporal(fused)
        return logits


def levenshtein_distance(a: Sequence[int], b: Sequence[int]) -> int:
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            curr[j] = min(
                prev[j] + 1,      # delete
                curr[j - 1] + 1,  # insert
                prev[j - 1] + cost,  # substitute
            )
        prev = curr
    return prev[-1]


def compute_batch_metrics(
    predicted: Sequence[Sequence[int]],
    references: Sequence[Sequence[int]],
) -> Tuple[float, float]:
    if not references:
        return 0.0, 0.0

    wer_sum = 0.0
    exact = 0
    for pred, ref in zip(predicted, references):
        dist = levenshtein_distance(ref, pred)
        wer_sum += dist / max(1, len(ref))
        if list(pred) == list(ref):
            exact += 1
    n = len(references)
    return wer_sum / n, exact / n


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ctc_layer: CTCLayer,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    grad_clip: float,
    log_interval: int,
    grad_accumulation: int,
    show_progress: bool,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    start_time = time.time()
    accumulation_steps = max(1, grad_accumulation)
    grad_norm_sum = 0.0
    grad_norm_steps = 0

    progress = tqdm(data_loader, desc="Training", leave=True, disable=not show_progress)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress, start=1):
        try:
            rgb = batch["rgb"].to(device, non_blocking=True)
            pose = batch["pose"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            input_lengths = batch["input_lengths"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=get_autocast_dtype(device),
                enabled=amp_enabled,
            ):
                logits = model(rgb, pose)
                raw_loss = ctc_layer(
                    logits=logits,
                    targets=targets,
                    input_lengths=input_lengths,
                    target_lengths=target_lengths,
                )
                loss = raw_loss / accumulation_steps

            scaler.scale(loss).backward()

            should_step = (step % accumulation_steps == 0) or (step == len(data_loader))
            if should_step:
                scaler.unscale_(optimizer)
                curr_grad_norm = compute_grad_norm(model)
                grad_norm_sum += curr_grad_norm
                grad_norm_steps += 1
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(raw_loss.item())

            if show_progress:
                progress.set_postfix({
                    "loss": f"{running_loss/step:.4f}",
                    "gpu_mem": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB"
                })

        except RuntimeError as exc:
            if device.type == "cuda" and "out of memory" in str(exc).lower():
                LOGGER.warning("CUDA OOM at train step %d/%d; skipping batch.", step, len(data_loader))
                optimizer.zero_grad(set_to_none=True)
                scaler = make_grad_scaler(enabled=amp_enabled)
                torch.cuda.empty_cache()
                continue
            raise

        if step % log_interval == 0:
            LOGGER.info(
                "Train step %d/%d | loss=%.4f",
                step,
                len(data_loader),
                running_loss / step,
            )

    elapsed = time.time() - start_time
    return {
        "loss": running_loss / max(1, len(data_loader)),
        "time_sec": elapsed,
        "grad_norm": grad_norm_sum / max(1, grad_norm_steps),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    ctc_layer: CTCLayer,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_wer = 0.0
    total_exact = 0.0
    total_samples = 0

    for batch in data_loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        pose = batch["pose"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        ref_lists = batch["target_lists"]

        with torch.autocast(
            device_type=device.type,
            dtype=get_autocast_dtype(device),
            enabled=amp_enabled,
        ):
            logits = model(rgb, pose)
            loss = ctc_layer(
                logits=logits,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )

        total_loss += float(loss.item())
        pred_lists = ctc_layer.decode_greedy(logits.detach().cpu())
        if pred_lists and ref_lists and random.random() < 0.05:
            LOGGER.info("Pred: %s", pred_lists[0])
            LOGGER.info("True: %s", ref_lists[0])
        batch_wer, batch_exact = compute_batch_metrics(pred_lists, ref_lists)
        batch_size = len(ref_lists)

        total_wer += batch_wer * batch_size
        total_exact += batch_exact * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / max(1, len(data_loader)),
        "wer": total_wer / max(1, total_samples),
        "exact_match": total_exact / max(1, total_samples),
    }


def summarize_samples(samples: Sequence[SequenceSample]) -> Dict[str, object]:
    class_counts = Counter(sample.sentence for sample in samples)
    frame_counts = [len(sample.frame_paths) for sample in samples]
    return {
        "num_samples": len(samples),
        "num_classes": len(class_counts),
        "avg_frames": float(np.mean(frame_counts)) if frame_counts else 0.0,
        "min_frames": int(min(frame_counts)) if frame_counts else 0,
        "max_frames": int(max(frame_counts)) if frame_counts else 0,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, payload: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def setup_file_logging(output_dir: Path, filename: str = "train_full_run.log") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / filename
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == log_path.resolve():
            return log_path

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)
    return log_path


def save_split_manifest(path: Path, samples: Sequence[SequenceSample]) -> None:
    records = []
    for sample in samples:
        records.append(
            {
                "sample_id": sample.sample_id,
                "sentence": sample.sentence,
                "sequence_id": sample.sequence_id,
                "num_frames_in_sequence": len(sample.frame_paths),
                "gloss_tokens": sample.gloss_tokens,
            }
        )
    write_json(path, records)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: Any,
    best_val_wer: float,
    metrics: Dict[str, float],
    args: argparse.Namespace,
    token_to_id: Dict[str, int],
    id_to_token: Dict[int, str],
    no_improve_epochs: int = 0,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_wer": best_val_wer,
        "metrics": metrics,
        "args": vars(args),
        "token_to_id": token_to_id,
        "id_to_token": {str(k): v for k, v in id_to_token.items()},
        "no_improve_epochs": no_improve_epochs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: Any,
    device: torch.device,
) -> Tuple[int, float, int]:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    optimizer.defaults["foreach"] = False
    for group in optimizer.param_groups:
        group["foreach"] = False
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler_state = checkpoint.get("scaler_state")
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_wer = float(checkpoint.get("best_val_wer", 1.0))
    no_improve_epochs = int(checkpoint.get("no_improve_epochs", 0))
    return start_epoch, best_val_wer, no_improve_epochs


def export_module_weights(model: CSLRCTCModel, output_dir: Path) -> None:
    torch.save(model.rgb_stream.state_dict(), output_dir / "rgb_stream_best.pt")
    torch.save(model.pose_stream.state_dict(), output_dir / "pose_stream_best.pt")
    torch.save(model.fusion.state_dict(), output_dir / "fusion_best.pt")
    torch.save(model.temporal.state_dict(), output_dir / "temporal_best.pt")


def _prune_best_checkpoints(checkpoint_dir: Path, keep: int) -> None:
    """Delete oldest best_epoch_*.pt files, keeping only the `keep` most recent."""
    if keep <= 0:
        return  # 0 = keep all
    pattern = list(checkpoint_dir.glob("best_epoch_*.pt"))
    # Extract epoch number from filename and sort ascending
    def _epoch_num(p: Path) -> int:
        m = re.search(r"best_epoch_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1
    pattern.sort(key=_epoch_num)
    to_delete = pattern[: max(0, len(pattern) - keep)]
    for old_ckpt in to_delete:
        try:
            old_ckpt.unlink()
            LOGGER.info("Pruned old checkpoint: %s", old_ckpt.name)
        except OSError as exc:
            LOGGER.warning("Could not delete %s: %s", old_ckpt.name, exc)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    corpus_root = Path(args.corpus_root).resolve()
    gloss_csv = Path(args.gloss_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    active_log_path = setup_file_logging(output_dir)
    LOGGER.info("Training log file: %s", active_log_path)

    tensorboard_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    sentence_map = load_sentence_gloss_map(gloss_csv)
    samples, unmatched_sentences = collect_sentence_samples(corpus_root, sentence_map)
    report = compute_dataset_report(corpus_root, sentence_map, samples, unmatched_sentences)

    write_json(output_dir / "dataset_analysis.json", report)
    LOGGER.info("Dataset report saved: %s", output_dir / "dataset_analysis.json")
    LOGGER.info(
        "Dataset summary | classes=%d mapped=%d sequences=%d frames=%d",
        report["sentence_classes_in_dataset"],
        report["sentence_classes_with_gloss_mapping"],
        report["total_sentence_sequences"],
        report["total_sentence_frames"],
    )
    if report["dataset_entries_missing_in_csv"]:
        LOGGER.warning(
            "Unmatched sentence folders (skipped): %s",
            report["dataset_entries_missing_in_csv"],
        )

    if args.analyze_only:
        LOGGER.info("Analyze-only run complete.")
        return

    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for training. Install dependencies with "
            "`pip install -r requirements.txt` from application/backend."
        )

    token_to_id, id_to_token = build_vocab(samples)
    vocab_tokens = [id_to_token[idx] for idx in range(1, len(id_to_token))]

    train_samples, val_samples, test_samples = stratified_split(
        samples=samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]

    split_summary = {
        "train": summarize_samples(train_samples),
        "val": summarize_samples(val_samples),
        "test": summarize_samples(test_samples),
        "vocab_size_without_blank": len(token_to_id),
    }
    write_json(output_dir / "split_summary.json", split_summary)
    write_json(output_dir / "vocab_tokens.json", vocab_tokens)
    write_json(output_dir / "token_to_id.json", token_to_id)
    save_split_manifest(output_dir / "manifests/train_manifest.json", train_samples)
    save_split_manifest(output_dir / "manifests/val_manifest.json", val_samples)
    save_split_manifest(output_dir / "manifests/test_manifest.json", test_samples)

    LOGGER.info(
        "Split sizes | train=%d val=%d test=%d vocab=%d",
        len(train_samples),
        len(val_samples),
        len(test_samples),
        len(token_to_id),
    )

    use_pose = not args.no_pose
    pose_cache_dir = Path(args.pose_cache_dir).resolve() if args.pose_cache_dir else None
    if pose_cache_dir and use_pose:
        pose_cache_dir.mkdir(parents=True, exist_ok=True)

    effective_num_frames: Optional[int] = None if args.num_frames <= 0 else args.num_frames

    train_dataset = ISLCSLRTDataset(
        samples=train_samples,
        token_to_id=token_to_id,
        num_frames=effective_num_frames,
        image_size=args.image_size,
        augment=True,
        use_pose=use_pose,
        pose_cache_dir=pose_cache_dir,
        target_fps=args.target_fps,
        min_video_frames=args.min_video_frames,
    )
    val_dataset = ISLCSLRTDataset(
        samples=val_samples,
        token_to_id=token_to_id,
        num_frames=effective_num_frames,
        image_size=args.image_size,
        augment=False,
        use_pose=use_pose,
        pose_cache_dir=pose_cache_dir,
        target_fps=args.target_fps,
        min_video_frames=args.min_video_frames,
    )
    test_dataset = ISLCSLRTDataset(
        samples=test_samples,
        token_to_id=token_to_id,
        num_frames=effective_num_frames,
        image_size=args.image_size,
        augment=False,
        use_pose=use_pose,
        pose_cache_dir=pose_cache_dir,
        target_fps=args.target_fps,
        min_video_frames=args.min_video_frames,
    )

    if args.prepare_pose_cache or args.prepare_pose_cache_only:
        if not use_pose:
            LOGGER.warning("Pose cache requested but pose is disabled (--no-pose).")
        elif pose_cache_dir is None:
            LOGGER.warning("Pose cache requested but --pose-cache-dir is not set.")
        else:
            LOGGER.info("Preparing pose cache for train split...")
            created_train = train_dataset.prepare_pose_cache()
            LOGGER.info("Preparing pose cache for val split...")
            created_val = val_dataset.prepare_pose_cache()
            LOGGER.info("Preparing pose cache for test split...")
            created_test = test_dataset.prepare_pose_cache()
            LOGGER.info(
                "Pose cache created files | train=%d val=%d test=%d",
                created_train,
                created_val,
                created_test,
            )
        if args.prepare_pose_cache_only:
            LOGGER.info("Pose-cache-only run complete.")
            return

    loader_kwargs: Dict[str, Any] = {}
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        collate_fn=collate_batch,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        collate_fn=collate_batch,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        collate_fn=collate_batch,
        **loader_kwargs,
    )

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    amp_enabled = (not args.no_amp) and device.type == "cuda"

    pose_hidden_dims = parse_pose_hidden_dims(args.pose_hidden_dims)
    model = CSLRCTCModel(
        vocab_size=len(token_to_id),
        backbone=args.backbone,
        pose_hidden_dims=pose_hidden_dims,
        feature_dim=args.feature_dim,
        fusion_type=args.fusion_type,
        temporal_model=args.temporal_model,
        temporal_hidden_dim=args.temporal_hidden_dim,
        temporal_layers=args.temporal_layers,
        dropout=args.dropout,
        pretrained_rgb=not args.no_pretrained_rgb,
        rgb_backbone_chunk_size=args.rgb_backbone_chunk_size,
    ).to(device)

    try:
        dummy_len = effective_num_frames if effective_num_frames is not None else args.min_video_frames
        dummy_rgb = torch.zeros(1, dummy_len, 3, args.image_size, args.image_size).to(device)
        dummy_pose = torch.zeros(1, dummy_len, POSE_FEATURE_DIM).to(device)
        writer.add_graph(model, (dummy_rgb, dummy_pose))
    except Exception as exc:
        LOGGER.warning("TensorBoard graph logging skipped: %s", exc)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.998),
        foreach=False,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1)
    scaler = make_grad_scaler(enabled=amp_enabled)
    ctc_layer = CTCLayer(blank_idx=0, vocab_size=len(token_to_id))

    start_epoch = 1
    best_val_wer = float("inf")
    no_improve_epochs = 0
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint to resume from not found: {resume_path}")
        start_epoch, best_val_wer, no_improve_epochs = load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        LOGGER.info(
            "Resumed from %s | start_epoch=%d best_val_wer=%.4f no_improve_epochs=%d",
            resume_path, start_epoch, best_val_wer, no_improve_epochs,
        )

    LOGGER.info(
        "Training start | device=%s amp=%s epochs=%d batch_size=%d",
        device,
        amp_enabled,
        args.epochs,
        args.batch_size,
    )

    history: List[Dict[str, float]] = []
    history_log_path = output_dir / "training_history.jsonl"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    early_stop_patience = max(1, args.early_stop_patience)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()

        LOGGER.info("Epoch %d/%d", epoch, args.epochs)
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            ctc_layer=ctc_layer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            grad_accumulation=args.grad_accumulation,
            show_progress=args.progress_bar,
        )

        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            ctc_layer=ctc_layer,
            device=device,
            amp_enabled=amp_enabled,
        )
        scheduler.step()

        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_wer": val_metrics["wer"],
            "val_exact_match": val_metrics["exact_match"],
            "epoch_time_sec": train_metrics["time_sec"],
        }
        history.append(epoch_metrics)
        LOGGER.info(
            "Epoch %d summary | train_loss=%.4f val_loss=%.4f val_wer=%.4f val_exact=%.4f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["wer"],
            val_metrics["exact_match"],
        )

        # ✅ TensorBoard logging
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("WER/val", val_metrics["wer"], epoch)
        writer.add_scalar("ExactMatch/val", val_metrics["exact_match"], epoch)
        writer.add_scalar("GradNorm", train_metrics.get("grad_norm", 0.0), epoch)
        writer.add_scalar("GPU/util_percent", gpu_utilization(), epoch)

        # GPU stats
        if torch.cuda.is_available():
            writer.add_scalar(
                "GPU/memory_allocated_GB",
                torch.cuda.memory_allocated() / 1024**3,
                epoch,
            )

        # Learning rate
        writer.add_scalar(
            "LR",
            scheduler.get_last_lr()[0],
            epoch,
        )

        # ✅ CSV logging
        csv_path = output_dir / "training_metrics.csv"

        file_exists = csv_path.exists()

        with open(csv_path, "a", newline="") as f:
            writer_csv = csv.writer(f)

            if not file_exists:
                writer_csv.writerow([
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_wer",
                    "val_exact",
                    "gpu_memory_GB",
                    "learning_rate"
                ])

            writer_csv.writerow([
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["wer"],
                val_metrics["exact_match"],
                torch.cuda.memory_allocated()/1024**3 if torch.cuda.is_available() else 0,
                scheduler.get_last_lr()[0]
            ])
        
        # ✅ Terminal GPU logging
        if torch.cuda.is_available():
            LOGGER.info(
                "GPU memory | allocated=%.2fGB reserved=%.2fGB",
                torch.cuda.memory_allocated()/1024**3,
                torch.cuda.memory_reserved()/1024**3
            )

        if args.checkpoint_strategy == "all" and args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                path=checkpoint_dir / f"epoch_{epoch:03d}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_wer=best_val_wer,
                metrics=epoch_metrics,
                args=args,
                token_to_id=token_to_id,
                id_to_token=id_to_token,
                no_improve_epochs=no_improve_epochs,
            )

        if val_metrics["wer"] < best_val_wer:
            best_val_wer = val_metrics["wer"]
            no_improve_epochs = 0
            best_path = checkpoint_dir / "best.pt"
            backup_path = checkpoint_dir / f"best_epoch_{epoch}.pt"
            save_checkpoint(
                path=best_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_wer=best_val_wer,
                metrics=epoch_metrics,
                args=args,
                token_to_id=token_to_id,
                id_to_token=id_to_token,
                no_improve_epochs=0,
            )
            save_checkpoint(
                path=backup_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_wer=best_val_wer,
                metrics=epoch_metrics,
                args=args,
                token_to_id=token_to_id,
                id_to_token=id_to_token,
                no_improve_epochs=0,
            )
            LOGGER.info("New best checkpoint at epoch %d (val_wer=%.4f)", epoch, best_val_wer)
            LOGGER.info("Best model saved at epoch %d with WER %.4f", epoch, best_val_wer)
            _prune_best_checkpoints(checkpoint_dir, keep=args.max_best_checkpoints)
        else:
            no_improve_epochs += 1

        # Always save last.pt so resume always continues from the latest epoch
        save_checkpoint(
            path=checkpoint_dir / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_wer=best_val_wer,
            metrics=epoch_metrics,
            args=args,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            no_improve_epochs=no_improve_epochs,
        )

        write_json(output_dir / "history.json", history)
        append_jsonl(history_log_path, epoch_metrics)

        epoch_time = time.time() - epoch_start_time
        remaining_epochs = args.epochs - epoch

        LOGGER.info(
            "Epoch time: %.1f sec | ETA: %.1f min",
            epoch_time,
            (epoch_time * remaining_epochs) / 60
        )
        log_gpu()

        if no_improve_epochs >= early_stop_patience:
            LOGGER.info("Early stopping triggered.")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_ckpt = checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        checkpoint = torch.load(str(best_ckpt), map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if args.export_module_weights:
            export_module_weights(model, output_dir)
        LOGGER.info("Loaded best checkpoint for final test evaluation.")
    else:
        LOGGER.warning("Best checkpoint not found; evaluating using final model state.")

    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        ctc_layer=ctc_layer,
        device=device,
        amp_enabled=amp_enabled,
    )
    final_report = {
        "best_val_wer": best_val_wer,
        "test_loss": test_metrics["loss"],
        "test_wer": test_metrics["wer"],
        "test_exact_match": test_metrics["exact_match"],
    }
    write_json(output_dir / "final_metrics.json", final_report)
    LOGGER.info(
        "Training complete | best_val_wer=%.4f test_wer=%.4f test_exact=%.4f",
        best_val_wer,
        test_metrics["wer"],
        test_metrics["exact_match"],
    )

    writer.close()


def build_parser() -> argparse.ArgumentParser:
    default_corpus_root = BACKEND_ROOT / "dataset" / "ISL_CSLRT_Corpus"
    default_gloss_csv = (
        default_corpus_root
        / "corpus_csv_files"
        / "ISL Corpus sign glosses.csv"
    )
    default_output_dir = BACKEND_ROOT / "checkpoints" / "isl_cslrt_experiment"
    default_pose_cache = BACKEND_ROOT / "dataset" / ".pose_cache_isl_cslrt"

    parser = argparse.ArgumentParser(
        description="Train CSLR model on ISL_CSLRT_Corpus"
    )
    parser.add_argument("--corpus-root", type=str, default=str(default_corpus_root))
    parser.add_argument("--gloss-csv", type=str, default=str(default_gloss_csv))
    parser.add_argument("--output-dir", type=str, default=str(default_output_dir))
    parser.add_argument("--pose-cache-dir", type=str, default=str(default_pose_cache))

    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--prepare-pose-cache", action="store_true")
    parser.add_argument("--prepare-pose-cache-only", action="store_true")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--grad-accumulation", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--checkpoint-strategy",
        type=str,
        choices=["best_only", "best_and_last", "all"],
        default="best_only",
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--max-best-checkpoints",
        type=int,
        default=3,
        help="Keep only the N most recent best_epoch_*.pt files (0 = keep all).",
    )

    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument("--target-fps", type=int, default=8)
    parser.add_argument("--min-video-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=0)

    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--rgb-backbone-chunk-size", type=int, default=8)
    parser.add_argument("--pose-hidden-dims", type=str, default="512,256")
    parser.add_argument("--fusion-type", type=str, default="gated_attention")
    parser.add_argument("--temporal-model", type=str, default="bilstm")
    parser.add_argument("--temporal-hidden-dim", type=int, default=256)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained-rgb", action="store_true")
    parser.add_argument("--no-pose", action="store_true")
    parser.add_argument("--export-module-weights", action="store_true")

    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
