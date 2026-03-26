"""
Train CSLR model on iSign v1.1 Dataset
=======================================
End-to-end training script for the iSign Indian Sign Language Recognition
dataset.  Integrates with the existing project pipeline modules.

Architecture:
  RGB frames   →  EfficientNet-B0  →┐
                                     ├─ Weighted Fusion → Causal TCN → CTC Loss
  Pose keypts  →  Pose MLP         →┘

Usage
-----
# Quick smoke-test (10 samples, 2 epochs, CPU)
python scripts/train_isign.py \\
    --data-dir dataset/isign_processed \\
    --epochs 2 --batch-size 2 --max-samples 10

# Full training with GPU
python scripts/train_isign.py \\
    --data-dir dataset/isign_processed \\
    --epochs 50 --batch-size 8 --lr 1e-4

# Real-time focused model (fixed architecture)
python scripts/train_isign.py \\
    --data-dir dataset/isign_processed \\
    --pose-fusion-weight 0.7

# Deployment note
# Train with 64 frames for accuracy.
# For lower-latency inference, sample 32 frames from the 64-frame clip.
# On GPU inference, FP16 (`model.half()`) can provide an extra speedup.

# Resume from checkpoint
python scripts/train_isign.py \\
    --data-dir dataset/isign_processed \\
    --resume checkpoints/isign/best_model.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

try:
    from torch.amp import GradScaler, autocast

    def amp_autocast(device_type: str, enabled: bool):
        return autocast(device_type=device_type, enabled=enabled)

    def make_grad_scaler(device_type: str, enabled: bool) -> GradScaler:
        return GradScaler(device=device_type, enabled=enabled)

except ImportError:  # pragma: no cover - compatibility for older PyTorch
    from torch.cuda.amp import autocast as cuda_autocast
    from torch.cuda.amp import GradScaler as CudaGradScaler

    GradScaler = CudaGradScaler  # type: ignore[misc,assignment]

    def amp_autocast(device_type: str, enabled: bool):
        if device_type != "cuda":
            return nullcontext()
        return cuda_autocast(enabled=enabled)

    def make_grad_scaler(device_type: str, enabled: bool) -> GradScaler:
        return GradScaler(enabled=enabled and device_type == "cuda")

# ---------------------------------------------------------------------------
# Resolve backend root so internal imports work when calling from any cwd
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.data.isign_dataset import build_dataloaders  # noqa: E402
from app.utils.ctc_decoder import CTCDecoder as BeamCTCDecoder  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_isign")


def _available_cpu_workers() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _default_workers() -> int:
    cores = _available_cpu_workers()
    if cores <= 2:
        return cores
    return max(2, min(8, cores // 2))


def resolve_dataloader_workers(requested_workers: int) -> int:
    if requested_workers < 0:
        raise ValueError("--workers must be >= 0")

    available_workers = _available_cpu_workers()
    resolved_workers = min(int(requested_workers), available_workers)
    if resolved_workers != requested_workers:
        log.warning(
            "Reducing DataLoader workers from %d to %d to match available CPU worker slots (%d).",
            requested_workers,
            resolved_workers,
            available_workers,
        )
    return resolved_workers


def find_resume_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Pick a safe automatic resume checkpoint from checkpoint directory."""
    last_ckpt = ckpt_dir / "last.pt"
    if last_ckpt.exists():
        return last_ckpt

    epoch_ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if epoch_ckpts:
        return epoch_ckpts[-1]

    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        return best_ckpt
    return None


def build_epoch_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
):
    """Warm up linearly, then decay with cosine annealing."""
    total_epochs = max(1, int(total_epochs))
    warmup_epochs = max(0, min(int(warmup_epochs), total_epochs))
    eta_min = base_lr * 0.01

    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)

    warmup = LinearLR(
        optimizer,
        start_factor=max(1e-3, 1.0 / float(warmup_epochs)),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    if warmup_epochs >= total_epochs:
        return warmup

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=eta_min,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def ids_to_tokens(ids: Sequence[int], vocab: Sequence[str], blank_id: int = 0) -> List[str]:
    tokens: List[str] = []
    for idx in ids:
        j = int(idx)
        if j == blank_id or not (0 <= j < len(vocab)):
            continue
        tok = str(vocab[j]).strip()
        if tok:
            tokens.append(tok)
    return tokens


def export_lm_corpus(
    data_dir: Path,
    output_path: Path,
    vocab: Sequence[str],
    splits: Sequence[str],
    blank_id: int = 0,
) -> int:
    """Export gloss-token text corpus for N-gram LM training."""
    lines: List[str] = []
    for split in splits:
        ann_path = data_dir / f"{split}.json"
        if not ann_path.exists():
            continue
        with open(ann_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        for ann in annotations:
            gloss_tokens = ann.get("gloss_tokens")
            if not gloss_tokens:
                gloss_ids = ann.get("gloss_ids", [])
                gloss_tokens = ids_to_tokens(gloss_ids, vocab, blank_id=blank_id)
            cleaned = [str(tok).strip().upper() for tok in gloss_tokens if str(tok).strip()]
            if cleaned:
                lines.append(" ".join(cleaned))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    return len(lines)


def decode_sequences(
    log_probs: torch.Tensor,
    blank_id: int,
    vocab: Optional[Sequence[str]] = None,
    decoder: Optional[BeamCTCDecoder] = None,
) -> List[List[Any]]:
    """Decode a batch of CTC log-probs with either greedy or beam/LM decoding."""
    if decoder is not None:
        seq_logits = log_probs.permute(1, 0, 2).detach().cpu()
        return cast(List[List[Any]], decoder.beam_search_decode(seq_logits))

    decoded_ids = ctc_greedy_decode(log_probs, blank_id)
    if vocab is None:
        return cast(List[List[Any]], decoded_ids)
    return [ids_to_tokens(seq, vocab, blank_id=blank_id) for seq in decoded_ids]


# Dataset-guided dynamic inference framing.
# Derived from application/backend/dataset/isign_processed/.preprocess_annotations_cache.jsonl
# (68,393 samples; average gloss length 10.24; 7-12 tokens is the largest bucket).
AVG_GLOSS_TOKENS_PER_64_FRAMES = 10.24
TOKEN_FRAME_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (2, 16),   # single words / very short phrases
    (6, 32),   # short sentences
    (12, 48),  # medium sentences
    (10_000, 64),  # long sentences
)


def infer_dynamic_frame_count(num_gloss_tokens: int) -> int:
    """Map an expected gloss/token length to a latency-aware frame budget."""
    tokens = max(1, int(num_gloss_tokens))
    for max_tokens, frames in TOKEN_FRAME_BUCKETS:
        if tokens <= max_tokens:
            return frames
    return 64


def uniform_frame_indices(total_frames: int, target_frames: int) -> np.ndarray:
    """Sample a target number of frames uniformly from a clip."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    target = max(1, min(int(target_frames), int(total_frames)))
    return np.linspace(0, total_frames - 1, target, dtype=int)


def estimate_gloss_tokens_from_frames(total_frames: int) -> int:
    """Estimate token count from raw clip length using dataset-level priors."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    est = round((total_frames / 64.0) * AVG_GLOSS_TOKENS_PER_64_FRAMES)
    return max(1, int(est))


def sliding_window_indices(
    total_frames: int,
    num_gloss_tokens: int,
    stride: int = 1,
) -> List[np.ndarray]:
    """
    Build low-latency sliding windows for streaming inference.

    Example:
      1-32, 2-33, 3-34, ...
    """
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    window = min(infer_dynamic_frame_count(num_gloss_tokens), total_frames)
    step = max(1, int(stride))

    if total_frames <= window:
        return [uniform_frame_indices(total_frames, window)]

    windows: List[np.ndarray] = []
    for start in range(0, total_frames - window + 1, step):
        windows.append(np.arange(start, start + window, dtype=int))

    last = np.arange(total_frames - window, total_frames, dtype=int)
    if not windows or not np.array_equal(windows[-1], last):
        windows.append(last)
    return windows


# ===========================================================================
# Model components
# ===========================================================================

class PoseEncoder(nn.Module):
    """MLP to project pose keypoints → hidden_dim."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)  →  (B, T, hidden_dim)
        return self.net(x)


class RGBEncoder(nn.Module):
    """
    EfficientNet-B0 applied per-frame to extract spatial features,
    then average-pools across spatial dims.
    """

    def __init__(
        self,
        hidden_dim: int,
        pretrained: bool = True,
        freeze_bn: bool = True,
        freeze_stages: int = 4,
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = efficientnet_b0(weights=weights)
        except TypeError:
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0(pretrained=pretrained)
        self.features = model.features  # (B, 1280, h, w)
        in_dim = 1280

        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.proj     = nn.Linear(in_dim, hidden_dim)

        if freeze_bn:
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad_(False)

        if freeze_stages > 0:
            blocks = list(self.features.children())
            for block in blocks[: min(int(freeze_stages), len(blocks))]:
                for p in block.parameters():
                    p.requires_grad_(False)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb: (B, C, T, H, W)
        B, C, T, H, W = rgb.shape
        x = rgb.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)   # (B*T, C, H, W)
        x = self.pool(self.features(x)).squeeze(-1).squeeze(-1)
        x = self.proj(x)                                           # (B*T, hidden_dim)
        return x.reshape(B, T, -1)                                 # (B, T, hidden_dim)


class CausalConv1d(nn.Module):
    """1D causal convolution that preserves sequence length."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.pad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.pad > 0:
            y = y[:, :, :-self.pad]
        return y


class CausalTCN(nn.Module):
    """Streaming-friendly causal TCN for real-time decoding."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            CausalConv1d(input_dim, hidden_dim, kernel_size=3, dilation=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, D, T) -> (B, T, D)
        y = self.blocks(x.permute(0, 2, 1))
        return y.permute(0, 2, 1)


class ISignCSLRModel(nn.Module):
    """
    Fixed iSign CSLR model for accurate low-latency inference/training.

    Streams:
        - RGB  →  EfficientNet-B0 per-frame encoder
        - Pose →  MLP encoder

    Fusion:
        - Pose-weighted RGB+pose fusion

    Sequence:
        - Causal TCN

    Head:
        - Linear → vocab_size  (used with CTC loss)

    Notes:
        - The pose/RGB fusion weight is learnable and initialized from
          --pose-fusion-weight.
        - Temporal attention and frame weighting improve long-range focus
          without changing the CTC interface.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        pose_input_dim: int = 34,
        dropout: float = 0.2,
        pretrained_cnn: bool = True,
        pose_fusion_weight: float = 0.7,
        attention_heads: int = 4,
        freeze_rgb_stages: int = 4,
        use_rgb: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads != 0:
            raise ValueError("--hidden-dim must be divisible by --attention-heads")
        self.use_rgb = bool(use_rgb)
        if self.use_rgb:
            init_alpha = float(np.clip(pose_fusion_weight, 1e-4, 1.0 - 1e-4))
            self.fusion_alpha = nn.Parameter(
                torch.tensor(np.log(init_alpha / (1.0 - init_alpha)), dtype=torch.float32)
            )
            self.rgb_enc = RGBEncoder(
                hidden_dim,
                pretrained=pretrained_cnn,
                freeze_stages=freeze_rgb_stages,
            )
            self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.register_parameter("fusion_alpha", None)
            self.rgb_enc = None

        self.pose_enc = PoseEncoder(pose_input_dim, hidden_dim, dropout)
        self.temporal = CausalTCN(hidden_dim, hidden_dim, dropout)
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.frame_score = nn.Linear(hidden_dim, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def pose_fusion_weight(self) -> torch.Tensor:
        if not self.use_rgb or self.fusion_alpha is None:
            return self.head.weight.new_tensor(1.0)
        return torch.sigmoid(self.fusion_alpha)

    def forward(self, rgb: Optional[torch.Tensor], pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb  : (B, C, T, H, W)
            pose : (B, T, D)

        Returns:
            log_probs : (T, B, vocab_size)  —  CTC-compatible format
        """
        pose_feat = self.pose_enc(pose)
        if self.use_rgb:
            if rgb is None or rgb.ndim != 5 or rgb.shape[1] == 0:
                raise ValueError("RGB input is required unless --pose-only is enabled")
            rgb_feat = self.rgb_enc(rgb)
            p = torch.clamp(self.pose_fusion_weight(), 0.1, 0.9)
            x = torch.cat([p * pose_feat, (1.0 - p) * rgb_feat], dim=-1)
            x = torch.relu(self.fusion_proj(x)) + (0.1 * pose_feat)  # (B, T, hidden_dim)
        else:
            x = pose_feat
        x = self.dropout(x)
        x = self.temporal(x)                     # (B, T, hidden_dim)
        attn_out, _ = self.temporal_attn(x, x, x, need_weights=False)
        x = self.attn_norm(x + attn_out)
        frame_weights = torch.softmax(self.frame_score(x), dim=1)
        x = x * frame_weights
        x = self.norm(x)
        x = self.dropout(x)
        x = self.head(x)                         # (B, T, vocab_size)
        log_p = self.log_softmax(x)              # (B, T, vocab_size)
        return log_p.permute(1, 0, 2)            # (T, B, vocab_size)  for CTC


def _confidence_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Get sequence confidence proxy from frame-wise max probs."""
    probs = log_probs.exp().permute(1, 0, 2)  # (B, T, V)
    conf = probs.max(dim=-1).values.mean(dim=-1)
    return conf.clamp(1e-6, 1 - 1e-6)


def confidence_penalty(log_probs: torch.Tensor) -> torch.Tensor:
    """CTC-safe confidence regularizer; higher weight reduces overconfidence."""
    probs = log_probs.exp()
    return (probs * log_probs).sum(dim=-1).mean()


def gpu_stats() -> str:
    """Return allocated GPU memory against total device memory."""
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{mem:.2f}/{total:.1f} GB"
    return "CPU"


def calibrate_confidence_temperature(
    model: ISignCSLRModel,
    loader,
    device: torch.device,
    blank_id: int,
    vocab: Optional[Sequence[str]] = None,
    decoder: Optional[BeamCTCDecoder] = None,
) -> Dict[str, float]:
    """Fit confidence temperature on validation set using sequence correctness."""
    model.eval()
    confs: List[float] = []
    labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            pose = batch["pose"].to(device, non_blocking=True)
            gt = batch["labels"].to(device, non_blocking=True)
            gt_lens = batch["label_lengths"].to(device, non_blocking=True)

            log_probs = model(rgb, pose)
            conf = _confidence_from_log_probs(log_probs).cpu().numpy().tolist()

            hyps = decode_sequences(log_probs, blank_id, vocab=vocab, decoder=decoder)
            if vocab is None:
                refs = [gt[i, : gt_lens[i]].tolist() for i in range(gt.shape[0])]
            else:
                refs = [
                    ids_to_tokens(gt[i, : gt_lens[i]].tolist(), vocab, blank_id=blank_id)
                    for i in range(gt.shape[0])
                ]
            for c, h, r in zip(conf, hyps, refs):
                confs.append(float(c))
                labels.append(1 if h == r else 0)

    if not confs:
        return {"temperature": 1.0, "nll_before": 0.0, "nll_after": 0.0}

    conf_t = torch.tensor(confs, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    logit = torch.logit(conf_t)

    def nll(temp: float) -> float:
        p = torch.sigmoid(logit / max(temp, 1e-3))
        return float(F.binary_cross_entropy(p, y).item())

    temps = np.linspace(0.4, 3.0, 120)
    best_t = 1.0
    best_nll = nll(best_t)
    for t in temps:
        curr = nll(float(t))
        if curr < best_nll:
            best_nll = curr
            best_t = float(t)

    return {
        "temperature": best_t,
        "nll_before": nll(1.0),
        "nll_after": best_nll,
        "samples": len(confs),
    }


# ===========================================================================
# CTC Greedy decoder (for WER evaluation)
# ===========================================================================

def ctc_greedy_decode(
    log_probs: torch.Tensor,       # (T, B, V)
    blank_id: int = 0,
) -> List[List[int]]:
    """Greedy CTC decode — returns list of decoded id sequences (one per sample)."""
    preds = log_probs.argmax(dim=-1).permute(1, 0)  # (B, T)
    results = []
    for seq in preds.cpu().numpy():
        decoded: List[int] = []
        prev = -1
        for val in seq:
            v = int(val)
            if v != blank_id and v != prev:
                decoded.append(v)
            prev = v
        results.append(decoded)
    return results


@torch.no_grad()
def streaming_predict_with_early_exit(
    model: ISignCSLRModel,
    rgb: Optional[torch.Tensor],
    pose: torch.Tensor,
    blank_id: int = 0,
    expected_gloss_tokens: Optional[int] = None,
    stride: int = 1,
    confidence_threshold: float = 0.90,
    vocab: Optional[Sequence[str]] = None,
    decoder: Optional[BeamCTCDecoder] = None,
) -> Dict[str, Any]:
    """
    Sliding-window inference with confidence-based early exit.

    Args:
        rgb: (C, T, H, W) when RGB is enabled, else None
        pose: (T, D)
    """
    if pose.ndim != 2:
        raise ValueError("pose must have shape (T, D)")
    if model.use_rgb:
        if rgb is None or rgb.ndim != 4:
            raise ValueError("rgb must have shape (C, T, H, W)")

    total_frames = int(pose.shape[0])
    token_hint = expected_gloss_tokens or estimate_gloss_tokens_from_frames(total_frames)
    windows = sliding_window_indices(total_frames, token_hint, stride=stride)

    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    best: Optional[Dict[str, Any]] = None
    for idxs in windows:
        if model.use_rgb:
            rgb_window = rgb[:, idxs, :, :].unsqueeze(0).to(device=device)
        else:
            rgb_window = None
        pose_window = pose[idxs, :].unsqueeze(0).to(device=device)
        if model_dtype == torch.float16:
            if rgb_window is not None:
                rgb_window = rgb_window.half()
            pose_window = pose_window.half()

        log_probs = model(rgb_window, pose_window)
        confidence = float(_confidence_from_log_probs(log_probs).item())
        prediction = decode_sequences(log_probs, blank_id, vocab=vocab, decoder=decoder)[0]
        current = {
            "prediction": prediction,
            "confidence": confidence,
            "window_start": int(idxs[0]),
            "window_end": int(idxs[-1]),
            "window_size": int(len(idxs)),
            "exited_early": confidence >= confidence_threshold,
        }
        if best is None or confidence > float(best["confidence"]):
            best = current
        if confidence >= confidence_threshold:
            return current

    if best is None:
        raise RuntimeError("No inference windows were generated")
    return best


def word_error_rate(hypotheses: Sequence[Sequence[Any]], references: Sequence[Sequence[Any]]) -> float:
    """Compute WER over a batch."""
    total_err = 0
    total_ref = 0
    for hyp, ref in zip(hypotheses, references):
        # Simple edit-distance-based WER
        n, m = len(ref), len(hyp)
        dp   = list(range(n + 1))
        for i in range(1, m + 1):
            new_dp = [i] + [0] * n
            for j in range(1, n + 1):
                if hyp[i - 1] == ref[j - 1]:
                    new_dp[j] = dp[j - 1]
                else:
                    new_dp[j] = 1 + min(dp[j], dp[j - 1], new_dp[j - 1])
            dp = new_dp
        total_err += dp[n]
        total_ref += n if n > 0 else 1
    return total_err / max(total_ref, 1)


# ===========================================================================
# Training loop
# ===========================================================================

def train_epoch(
    model: ISignCSLRModel,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    clip_grad: float,
    confidence_penalty_weight: float,
    log_interval: int,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss  = 0.0
    total_steps = 0
    total_batches = len(loader)
    start_time = time.time()
    log_every = max(1, int(log_interval))

    for step, batch in enumerate(loader, 1):
        rgb    = batch["rgb"].to(device, non_blocking=True)
        pose   = batch["pose"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        label_lens = batch["label_lengths"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp_autocast(device.type, enabled=use_amp):
            log_probs = model(rgb, pose)          # (T, B, V)
            T, B, _   = log_probs.shape
            input_lens = torch.full((B,), T, dtype=torch.long, device=device)
            ctc_loss = criterion(log_probs, labels, input_lens, label_lens)
            reg_loss = confidence_penalty(log_probs)
            loss = ctc_loss + (confidence_penalty_weight * reg_loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss  += loss.item()
        total_steps += 1

        if step % log_every == 0 or step == total_batches:
            elapsed = max(time.time() - start_time, 1e-6)
            processed_samples = step * rgb.size(0)
            speed = processed_samples / elapsed
            remaining_steps = total_batches - step
            eta_seconds = remaining_steps * (elapsed / step)
            log.info(
                "[Train][Epoch %03d/%03d] Step %d/%d | Loss=%.4f | CTC=%.4f | Reg=%.4f | Speed=%.2f samples/s | GPU=%s | ETA=%.1f min",
                epoch,
                total_epochs,
                step,
                total_batches,
                loss.item(),
                ctc_loss.item(),
                reg_loss.item(),
                speed,
                gpu_stats(),
                eta_seconds / 60.0,
            )

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate(
    model: ISignCSLRModel,
    loader,
    criterion: nn.CTCLoss,
    device: torch.device,
    blank_id: int,
    vocab: Optional[Sequence[str]] = None,
    decoder: Optional[BeamCTCDecoder] = None,
) -> Tuple[float, float]:
    model.eval()
    total_loss  = 0.0
    total_steps = 0
    all_hyps: List[List[str]] = []
    all_refs: List[List[str]] = []

    for batch in loader:
        rgb        = batch["rgb"].to(device, non_blocking=True)
        pose       = batch["pose"].to(device, non_blocking=True)
        labels     = batch["labels"].to(device, non_blocking=True)
        label_lens = batch["label_lengths"].to(device, non_blocking=True)

        log_probs = model(rgb, pose)
        T, B, _   = log_probs.shape
        input_lens = torch.full((B,), T, dtype=torch.long, device=device)
        loss = criterion(log_probs, labels, input_lens, label_lens)
        total_loss  += loss.item()
        total_steps += 1

        hyps = decode_sequences(log_probs, blank_id, vocab=vocab, decoder=decoder)
        if vocab is None:
            refs = [labels[i, : label_lens[i]].tolist() for i in range(B)]
        else:
            refs = [
                ids_to_tokens(labels[i, : label_lens[i]].tolist(), vocab, blank_id=blank_id)
                for i in range(B)
            ]
        all_hyps.extend(hyps)
        all_refs.extend(refs)

    avg_loss = total_loss / max(total_steps, 1)
    wer      = word_error_rate(all_hyps, all_refs)
    return avg_loss, wer


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def save_checkpoint(
    path: Path,
    epoch: int,
    model: ISignCSLRModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    best_wer: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_wer":  best_wer,
            "args":      vars(args),
        },
        str(path),
    )


def save_emergency_resume_checkpoint(
    ckpt_dir: Path,
    epoch_to_resume: int,
    model: ISignCSLRModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    best_wer: float,
    args: argparse.Namespace,
    reason: str,
) -> Path:
    """
    Save a resumable checkpoint after an interruption/failure.

    `epoch_to_resume` is the epoch number that should be run next.
    Since the loader resumes from `ckpt['epoch'] + 1`, store `resume-1`.
    """
    resume_epoch = max(1, int(epoch_to_resume))
    stored_epoch = max(0, resume_epoch - 1)
    path = ckpt_dir / "last.pt"
    save_checkpoint(
        path,
        stored_epoch,
        model,
        optimizer,
        scheduler,
        scaler,
        best_wer,
        args,
    )
    log.warning(
        "Emergency checkpoint saved to %s (next resume epoch: %d, reason: %s)",
        path,
        resume_epoch,
        reason,
    )
    return path


def load_checkpoint(
    path: Path,
    model: ISignCSLRModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[GradScaler] = None,
) -> Dict:
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(path), map_location="cpu")
    saved_args = ckpt.get("args") or {}
    saved_pose_only = saved_args.get("pose_only")
    current_pose_only = not getattr(model, "use_rgb", True)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        if saved_pose_only is not None and bool(saved_pose_only) != bool(current_pose_only):
            if bool(saved_pose_only) and not bool(current_pose_only):
                missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
                ckpt["_warmstart"] = True
                ckpt["_warmstart_reason"] = (
                    "Loaded pose-only checkpoint into RGB+pose model; RGB/fusion weights were initialized fresh."
                )
                ckpt["_missing_keys"] = list(missing)
                ckpt["_unexpected_keys"] = list(unexpected)
            else:
                raise RuntimeError(
                    f"Incompatible checkpoint at {path}: checkpoint pose_only={bool(saved_pose_only)} "
                    f"but current run pose_only={bool(current_pose_only)}. "
                    "Use a matching --ckpt-dir/--resume, or disable auto-resume for a fresh run."
                ) from exc
        else:
            raise
    if optimizer and "optimizer" in ckpt and not ckpt.get("_warmstart"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt and not ckpt.get("_warmstart"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and "scaler" in ckpt and not ckpt.get("_warmstart"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def _read_resume_args(ckpt_dir: Path, resume_path: Path) -> Dict:
    config_path = ckpt_dir / "train_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            log.warning("Could not read checkpoint config from %s: %s", config_path, exc)

    try:
        ckpt = torch.load(str(resume_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(resume_path), map_location="cpu")
    except Exception as exc:
        log.warning("Could not inspect checkpoint metadata from %s: %s", resume_path, exc)
        return {}

    args = ckpt.get("args")
    return args if isinstance(args, dict) else {}


def _resume_is_compatible(current_args: argparse.Namespace, saved_args: Dict) -> Tuple[bool, str]:
    if not saved_args:
        return True, ""

    current_pose_only = bool(current_args.pose_only)
    saved_pose_only = saved_args.get("pose_only")
    if saved_pose_only is not None and bool(saved_pose_only) != current_pose_only:
        if bool(saved_pose_only) and not current_pose_only:
            return (
                True,
                f"warm-start from pose-only checkpoint (checkpoint={bool(saved_pose_only)}, current={current_pose_only})",
            )
        return (
            False,
            f"pose_only mismatch (checkpoint={bool(saved_pose_only)}, current={current_pose_only})",
        )

    return True, ""


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train CSLR model on iSign dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data-dir",    default="dataset/isign_processed",  help="Preprocessed dataset directory")
    p.add_argument("--vocab",       default=None,                       help="vocab.json path (default: <data-dir>/vocab.json)")
    p.add_argument("--export-lm-corpus", default=None, help="Write gloss corpus text for KenLM/N-gram training")
    p.add_argument("--export-lm-corpus-only", action="store_true", help="Export the LM corpus and exit without training")
    p.add_argument("--lm-corpus-splits", nargs="+", default=["train"], choices=["train", "val", "test"], help="Dataset splits included when exporting the LM corpus")
    p.add_argument("--num-frames",  type=int, default=64,               help="Frames per clip")
    p.add_argument("--frame-size",  type=int, nargs=2, default=[224, 224], metavar=("H", "W"), help="Frame resize dimensions")
    p.add_argument("--max-samples", type=int, default=0,                help="Limit samples (0=all; useful for testing)")
    p.add_argument("--pose-backend", choices=["auto", "npy", "lmdb"], default="auto", help="Pose storage backend; auto prefers <data-dir>/poses.lmdb when present")
    p.add_argument("--pose-lmdb-path", default=None, help="Optional LMDB path for pose tensors (default: <data-dir>/poses.lmdb)")
    p.add_argument("--pose-lmdb-readahead", action="store_true", help="Enable LMDB filesystem readahead for sequential/local-disk workloads")
    p.add_argument("--no-pretrained", action="store_true",              help="Do not use pretrained CNN weights")
    # Model
    p.add_argument("--hidden-dim",  type=int, default=256,              help="Encoder / temporal hidden dimension")
    p.add_argument("--dropout",     type=float, default=0.2,            help="Dropout rate")
    p.add_argument("--attention-heads", type=int, default=4, help="Temporal self-attention heads after Causal TCN")
    p.add_argument("--freeze-rgb-stages", type=int, default=4, help="Number of early EfficientNet feature blocks to freeze")
    p.add_argument(
        "--pose-fusion-weight",
        type=float,
        default=0.7,
        help="Relative weight for pose features during RGB+pose fusion.",
    )
    p.add_argument("--pose-only", action="store_true", help="Train and evaluate using pose .npy features only; skip RGB loading and EfficientNet")
    p.add_argument("--hard-negative-prob", type=float, default=0.10, help="Probability of synthetic no-sign negatives in train split")
    p.add_argument("--temporal-jitter", type=int, default=2, help="Temporal index jitter for data augmentation")
    p.add_argument("--frame-drop-prob", type=float, default=0.05, help="Random frame drop probability")
    p.add_argument("--brightness-jitter", type=float, default=0.15, help="Brightness jitter strength")
    p.add_argument("--blur-prob", type=float, default=0.10, help="Gaussian blur augmentation probability")
    p.add_argument("--noise-std", type=float, default=0.02, help="Additive RGB noise std")
    p.add_argument("--pose-jitter-std", type=float, default=0.01, help="Pose jitter std")
    p.add_argument("--use-albumentations", action="store_true", default=True, help="Use video-safe Albumentations RGB augmentations during training")
    p.add_argument("--no-albumentations", dest="use_albumentations", action="store_false", help="Disable Albumentations and use the built-in RGB augmentations only")
    p.add_argument("--albumentations-prob", type=float, default=0.35, help="Probability for geometric and photometric Albumentations transforms")
    p.add_argument("--motion-blur-prob", type=float, default=0.10, help="Probability of blur-style Albumentations transforms")
    p.add_argument("--coarse-dropout-prob", type=float, default=0.08, help="Probability of coarse dropout occlusion augmentation")
    p.add_argument("--calibrate-confidence", action="store_true", default=True, help="Fit confidence temperature on validation set")
    p.add_argument("--no-calibrate-confidence", dest="calibrate_confidence", action="store_false", help="Disable confidence temperature fitting")
    # Training
    p.add_argument("--epochs",      type=int, default=50,               help="Total training epochs")
    p.add_argument("--batch-size",  type=int, default=2,                help="Batch size")
    p.add_argument("--lr",          type=float, default=1e-4,           help="Learning rate")
    p.add_argument("--weight-decay",type=float, default=1e-4,           help="AdamW weight decay")
    p.add_argument("--confidence-penalty-weight", type=float, default=0.02, help="CTC-safe confidence regularization weight")
    p.add_argument("--clip-grad",   type=float, default=5.0,            help="Gradient clipping norm")
    p.add_argument("--warmup-epochs", type=int, default=3,              help="LR warmup epochs")
    p.add_argument("--no-amp",      action="store_true",                help="Disable automatic mixed precision")
    p.add_argument("--early-exit-threshold", type=float, default=0.90, help="Confidence threshold for streaming early-exit inference helper")
    p.add_argument("--eval-beam-width", type=int, default=1, help="Beam width for validation/test decoding; 1 uses greedy decode")
    p.add_argument("--lm-arpa", default=None, help="Optional KenLM ARPA/binary model path for validation/test rescoring")
    p.add_argument("--lm-weight", type=float, default=0.0, help="Language model weight for beam rescoring")
    p.add_argument("--lm-token-bonus", type=float, default=0.0, help="Per-token bonus during LM rescoring")
    p.add_argument("--lm-candidates", type=int, default=20, help="Number of acoustic beam candidates kept for LM rescoring")
    # Logging / checkpoints
    p.add_argument("--ckpt-dir",    default="checkpoints/isign_fast_v2", help="Checkpoint save directory")
    p.add_argument("--save-every-epoch", action="store_true", default=True, help="Save an epoch_{N}.pt checkpoint every epoch")
    p.add_argument("--no-save-every-epoch", dest="save_every_epoch", action="store_false", help="Disable per-epoch checkpoint files")
    p.add_argument("--log-interval",type=int, default=20,               help="Log every N batches")
    p.add_argument("--resume",      default=None,                       help="Path to checkpoint to resume from")
    p.add_argument("--auto-resume", action="store_true", default=True, help="Auto-resume from checkpoint directory when available")
    p.add_argument("--no-auto-resume", dest="auto_resume", action="store_false", help="Disable auto-resume behavior")
    p.add_argument("--workers",     type=int, default=_default_workers(), help="DataLoader workers (0 runs loading in the main process)")
    p.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when workers > 0")
    p.add_argument("--persistent-workers", action="store_true", default=True, help="Keep DataLoader workers alive across epochs")
    p.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false", help="Disable DataLoader persistent workers")
    p.add_argument("--preload-n", type=int, default=0, help="Preload the first N samples of each split into RAM to reduce disk reads")
    p.add_argument("--seed",        type=int, default=42,               help="Random seed")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Training device")
    p.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is not available")
    p.add_argument("--allow-tf32", action="store_true", default=True, help="Enable TF32 matmul/cudnn on Ampere+")
    p.add_argument("--no-allow-tf32", dest="allow_tf32", action="store_false", help="Disable TF32")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    if args.preload_n < 0:
        raise ValueError("--preload-n must be >= 0")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")
    if args.attention_heads < 1:
        raise ValueError("--attention-heads must be >= 1")
    if args.freeze_rgb_stages < 0:
        raise ValueError("--freeze-rgb-stages must be >= 0")
    if not 0.0 <= args.pose_fusion_weight <= 1.0:
        raise ValueError("--pose-fusion-weight must be between 0 and 1")
    if args.confidence_penalty_weight < 0:
        raise ValueError("--confidence-penalty-weight must be >= 0")
    if not 0.0 < args.early_exit_threshold < 1.0:
        raise ValueError("--early-exit-threshold must be between 0 and 1")
    if args.eval_beam_width < 1:
        raise ValueError("--eval-beam-width must be >= 1")
    if args.lm_weight < 0:
        raise ValueError("--lm-weight must be >= 0")
    if args.lm_candidates < 1:
        raise ValueError("--lm-candidates must be >= 1")
    if not 0.0 <= args.albumentations_prob <= 1.0:
        raise ValueError("--albumentations-prob must be between 0 and 1")
    if not 0.0 <= args.motion_blur_prob <= 1.0:
        raise ValueError("--motion-blur-prob must be between 0 and 1")
    if not 0.0 <= args.coarse_dropout_prob <= 1.0:
        raise ValueError("--coarse-dropout-prob must be between 0 and 1")

    # Data
    data_dir = Path(args.data_dir)
    vocab_path = Path(args.vocab) if args.vocab else data_dir / "vocab.json"
    if not vocab_path.exists():
        log.error("vocab.json not found at %s — run preprocess_isign.py first", vocab_path)
        sys.exit(1)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_for_export = json.load(f)
    blank_id_from_vocab = vocab_for_export.index("<blank>") if "<blank>" in vocab_for_export else 0

    if args.export_lm_corpus:
        corpus_path = Path(args.export_lm_corpus)
        num_lines = export_lm_corpus(
            data_dir=data_dir,
            output_path=corpus_path,
            vocab=vocab_for_export,
            splits=args.lm_corpus_splits,
            blank_id=blank_id_from_vocab,
        )
        log.info("Exported %d gloss lines to %s", num_lines, corpus_path)
        if args.export_lm_corpus_only:
            return

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = args.device

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Remove --require-cuda or fix CUDA setup.")
    if resolved == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")

    device  = torch.device(resolved)
    use_amp = not args.no_amp and device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    args.workers = resolve_dataloader_workers(args.workers)
    if args.workers == 0:
        args.persistent_workers = False

    log.info("Device: %s  |  AMP: %s", device, use_amp)
    if device.type == "cuda":
        log.info("GPU: %s  |  VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)
    log.info("GPU Memory: %s", gpu_stats())
    log.info(
        "DataLoader config: workers=%d | persistent_workers=%s | prefetch_factor=%d | cpu_slots=%d",
        args.workers,
        args.persistent_workers,
        args.prefetch_factor,
        _available_cpu_workers(),
    )

    log.info("Loading dataloaders from %s …", data_dir)
    use_rgb = not args.pose_only

    resume_path: Optional[Path] = Path(args.resume) if args.resume else None
    if resume_path is None and args.auto_resume:
        candidate_path = find_resume_checkpoint(ckpt_dir)
        if candidate_path is not None:
            saved_args = _read_resume_args(ckpt_dir, candidate_path)
            compatible, reason = _resume_is_compatible(args, saved_args)
            if compatible:
                resume_path = candidate_path
                log.info("Auto-resume checkpoint found: %s", resume_path)
            else:
                log.warning(
                    "Skipping auto-resume checkpoint %s because it is incompatible with the current run: %s",
                    candidate_path,
                    reason,
                )

    train_loader, val_loader, test_loader, vocab = build_dataloaders(
        data_dir       = str(data_dir),
        batch_size     = args.batch_size,
        num_frames     = args.num_frames,
        frame_size     = tuple(args.frame_size),
        num_workers    = args.workers,
        use_poses      = True,
        use_frames     = use_rgb,
        max_samples    = args.max_samples,
        pin_memory     = device.type == "cuda",
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        hard_negative_prob=args.hard_negative_prob,
        temporal_jitter=args.temporal_jitter,
        frame_drop_prob=args.frame_drop_prob,
        brightness_jitter=args.brightness_jitter,
        blur_prob=args.blur_prob,
        noise_std=args.noise_std,
        pose_jitter_std=args.pose_jitter_std,
        use_albumentations=args.use_albumentations,
        albumentations_prob=args.albumentations_prob,
        motion_blur_prob=args.motion_blur_prob,
        coarse_dropout_prob=args.coarse_dropout_prob,
        preload_n=args.preload_n,
        pose_backend=args.pose_backend,
        pose_lmdb_path=args.pose_lmdb_path,
        pose_lmdb_readahead=args.pose_lmdb_readahead,
    )
    vocab_size = len(vocab)
    blank_id   = vocab.index("<blank>") if "<blank>" in vocab else 0
    log.info("Vocab size: %d  |  Blank id: %d", vocab_size, blank_id)

    eval_decoder: Optional[BeamCTCDecoder] = None
    if args.eval_beam_width > 1 or (args.lm_arpa and args.lm_weight > 0):
        eval_decoder = BeamCTCDecoder(
            labels=vocab,
            blank_idx=blank_id,
            beam_width=args.eval_beam_width,
            lm_path=args.lm_arpa,
            lm_weight=args.lm_weight,
            lm_token_bonus=args.lm_token_bonus,
            lm_candidates=args.lm_candidates,
        )
        if args.lm_arpa:
            if eval_decoder.has_language_model:
                log.info(
                    "Eval decoder: beam=%d | LM=%s | weight=%.3f | token_bonus=%.3f | candidates=%d",
                    args.eval_beam_width,
                    args.lm_arpa,
                    args.lm_weight,
                    args.lm_token_bonus,
                    args.lm_candidates,
                )
            elif eval_decoder.language_model_loaded:
                log.info("Eval decoder: beam=%d | LM loaded from %s but disabled because --lm-weight=0", args.eval_beam_width, args.lm_arpa)
            else:
                log.warning("LM path provided but KenLM could not be loaded from %s; using beam-only evaluation", args.lm_arpa)
        else:
            log.info("Eval decoder: beam=%d without LM", args.eval_beam_width)

    # Prefer the dataset-wide pose width so missing-pose batches stay compatible.
    pose_dim = 34  # default fallback (17 keypoints x 2)
    train_dataset_pose_dim = getattr(train_loader.dataset, "pose_dim", None)
    if isinstance(train_dataset_pose_dim, int) and train_dataset_pose_dim > 0:
        pose_dim = train_dataset_pose_dim
        log.info("Pose dimension (dataset-wide): %d", pose_dim)
    else:
        try:
            sample_batch = next(iter(train_loader))
            pose_dim = int(sample_batch["pose"].shape[-1])
            log.info("Pose dimension (fallback batch inference): %d", pose_dim)
        except StopIteration:
            pass

    # Model
    model = ISignCSLRModel(
        vocab_size     = vocab_size,
        hidden_dim     = args.hidden_dim,
        pose_input_dim = pose_dim,
        dropout        = args.dropout,
        pretrained_cnn = not args.no_pretrained,
        pose_fusion_weight=args.pose_fusion_weight,
        attention_heads=args.attention_heads,
        freeze_rgb_stages=args.freeze_rgb_stages,
        use_rgb=use_rgb,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %.2fM", total_params / 1e6)
    log.info(
        "Model config: rgb_backbone=%s | temporal=causal_tcn+attn | use_rgb=%s | use_pose=True | init_pose_fusion_weight=%.2f | attn_heads=%d | freeze_rgb_stages=%d | conf_penalty=%.3f | early_exit=%.2f | albumentations=%s",
        "efficientnet_b0" if use_rgb else "disabled",
        use_rgb,
        args.pose_fusion_weight,
        args.attention_heads,
        args.freeze_rgb_stages,
        args.confidence_penalty_weight,
        args.early_exit_threshold,
        args.use_albumentations,
    )

    # Optimiser & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_epoch_scheduler(
        optimizer=optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        base_lr=args.lr,
    )

    # CTC loss
    criterion = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)

    # AMP
    scaler = make_grad_scaler(device.type, enabled=use_amp)

    # Resume
    start_epoch = 1
    best_wer    = float("inf")
    if resume_path is not None and resume_path.exists():
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, scaler=scaler)
        start_epoch = ckpt.get("epoch", 0) + 1
        if ckpt.get("_warmstart"):
            best_wer = float("inf")
            log.info("%s", ckpt.get("_warmstart_reason", "Warm-started from compatible checkpoint."))
            log.info(
                "Warm-start details: resumed epoch numbering from %d, restored %d shared tensors, left %d tensors freshly initialized.",
                start_epoch - 1,
                len(ckpt["model"]) - len(ckpt.get("_unexpected_keys", [])),
                len(ckpt.get("_missing_keys", [])),
            )
        else:
            best_wer = ckpt.get("best_wer", float("inf"))
            log.info("Resumed from epoch %d  (best WER: %.4f)", start_epoch - 1, best_wer)

    history_jsonl = ckpt_dir / "history.jsonl"
    history_csv = ckpt_dir / "history.csv"
    train_config_path = ckpt_dir / "train_config.json"

    with open(train_config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    csv_header = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_wer",
        "fusion_alpha",
        "learning_rate",
        "epoch_seconds",
        "is_best",
    ]
    if not history_csv.exists():
        with open(history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_header)
            writer.writeheader()

    # Training loop
    log.info("Starting training for %d epochs …", args.epochs)
    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            t0 = time.time()

            train_loss = train_epoch(
                model, train_loader, optimizer, criterion,
                device, scaler, use_amp, args.clip_grad, args.confidence_penalty_weight,
                args.log_interval, epoch, args.epochs,
            )
            val_loss, val_wer = evaluate(
                model,
                val_loader,
                criterion,
                device,
                blank_id,
                vocab=vocab,
                decoder=eval_decoder,
            )
            scheduler.step()

            elapsed = time.time() - t0
            lr_now = float(optimizer.param_groups[0]["lr"])
            fusion_weight = float(model.pose_fusion_weight().detach().cpu().item())
            log.info(
                "[Epoch %03d/%03d] Train=%.4f | Val=%.4f | WER=%.4f | Fusion=%.3f | LR=%.2e | GPU=%s | Time=%.0fs",
                epoch, args.epochs,
                train_loss, val_loss, val_wer, fusion_weight,
                lr_now,
                gpu_stats(),
                elapsed,
            )

            # Save best
            is_best = False
            if val_wer < best_wer:
                best_wer = val_wer
                is_best = True
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    epoch, model, optimizer, scheduler, scaler, best_wer, args,
                )
                log.info("  *** New best WER: %.4f — checkpoint saved ***", best_wer)

            history_row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_wer": float(val_wer),
                "fusion_alpha": fusion_weight,
                "learning_rate": lr_now,
                "epoch_seconds": float(elapsed),
                "is_best": is_best,
            }
            with open(history_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(history_row) + "\n")
            with open(history_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_header)
                writer.writerow(history_row)

            if args.save_every_epoch:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch:03d}.pt",
                    epoch, model, optimizer, scheduler, scaler, best_wer, args,
                )

            # Save latest
            save_checkpoint(
                ckpt_dir / "last.pt",
                epoch, model, optimizer, scheduler, scaler, best_wer, args,
            )
    except KeyboardInterrupt:
        save_emergency_resume_checkpoint(
            ckpt_dir=ckpt_dir,
            epoch_to_resume=current_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_wer=best_wer,
            args=args,
            reason="keyboard_interrupt",
        )
        log.warning("Training interrupted. Re-run the same command to auto-resume from last.pt.")
        raise
    except Exception as exc:
        save_emergency_resume_checkpoint(
            ckpt_dir=ckpt_dir,
            epoch_to_resume=current_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_wer=best_wer,
            args=args,
            reason=type(exc).__name__,
        )
        raise

    # Final test evaluation
    log.info("=== Final evaluation on test set ===")
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model)
    test_loss, test_wer = evaluate(
        model,
        test_loader,
        criterion,
        device,
        blank_id,
        vocab=vocab,
        decoder=eval_decoder,
    )
    log.info("Test loss: %.4f  |  Test WER: %.4f", test_loss, test_wer)

    calibration = {"temperature": 1.0, "nll_before": 0.0, "nll_after": 0.0, "samples": 0}
    if args.calibrate_confidence:
        log.info("Calibrating confidence temperature on validation set …")
        calibration = calibrate_confidence_temperature(
            model,
            val_loader,
            device,
            blank_id,
            vocab=vocab,
            decoder=eval_decoder,
        )
        calib_path = ckpt_dir / "confidence_calibration.json"
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2)
        log.info("Confidence calibration saved to %s (T=%.3f)", calib_path, calibration["temperature"])

    # Save results summary
    results = {
        "test_loss": test_loss,
        "test_wer":  test_wer,
        "best_val_wer": best_wer,
        "epochs": args.epochs,
        "vocab_size": vocab_size,
        "learned_pose_fusion_weight": float(model.pose_fusion_weight().detach().cpu().item()),
        "history_jsonl": str(history_jsonl),
        "history_csv": str(history_csv),
        "train_config": str(train_config_path),
        "confidence_calibration": calibration,
    }
    results_path = ckpt_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results written to %s", results_path)


if __name__ == "__main__":
    main()
