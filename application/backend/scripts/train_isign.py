"""
Train CSLR model on iSign v1.1 Dataset
=======================================
End-to-end training script for the iSign Indian Sign Language Recognition
dataset.  Integrates with the existing project pipeline modules.

Architecture:
  RGB frames   →  CNN (ResNet-18)  →┐
                                    ├─ Fusion → BiLSTM/Transformer → CTC Loss
  Pose keypts  →  Pose MLP        →┘

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

# Poses only (no frames, faster)
python scripts/train_isign.py \\
    --data-dir dataset/isign_processed \\
    --no-rgb --epochs 30 --batch-size 16

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch.amp import autocast, GradScaler
    _AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
    _AMP_DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Resolve backend root so internal imports work when calling from any cwd
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.data.isign_dataset import build_dataloaders  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_isign")


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
    Lightweight CNN applied per-frame to extract spatial features,
    then average-pools across spatial dims.

    Uses torchvision ResNet-18 backbone (pretrained or random init).
    """

    def __init__(self, hidden_dim: int, pretrained: bool = True, freeze_bn: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
        except TypeError:
            from torchvision.models import resnet18
            backbone = resnet18(pretrained=pretrained)

        # Remove the final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # (B, 512, h, w)
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.proj     = nn.Linear(512, hidden_dim)

        if freeze_bn:
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad_(False)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb: (B, C, T, H, W)
        B, C, T, H, W = rgb.shape
        x = rgb.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)   # (B*T, C, H, W)
        x = self.pool(self.features(x)).squeeze(-1).squeeze(-1)   # (B*T, 512)
        x = self.proj(x)                                           # (B*T, hidden_dim)
        return x.reshape(B, T, -1)                                 # (B, T, hidden_dim)


class ISignCSLRModel(nn.Module):
    """
    Dual-stream CSLR model for iSign training.

    Streams:
        - RGB  →  ResNet-18 per-frame encoder
        - Pose →  MLP encoder

    Fusion:
        - Concatenate + linear projection  OR  pose-only  OR  rgb-only

    Sequence:
        - 2-layer bidirectional LSTM

    Head:
        - Linear → vocab_size  (used with CTC loss)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int    = 256,
        pose_input_dim: int = 34,
        num_layers: int    = 2,
        dropout: float     = 0.3,
        use_rgb: bool      = True,
        use_pose: bool     = True,
        pretrained_cnn: bool = True,
        temporal_type: str = "bilstm",
        transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        self.use_rgb  = use_rgb
        self.use_pose = use_pose
        self.temporal_type = temporal_type.lower()

        fusion_in = 0

        if use_rgb:
            self.rgb_enc = RGBEncoder(hidden_dim, pretrained=pretrained_cnn)
            fusion_in += hidden_dim

        if use_pose:
            self.pose_enc = PoseEncoder(pose_input_dim, hidden_dim, dropout)
            fusion_in += hidden_dim

        if fusion_in == 0:
            raise ValueError("At least one of use_rgb or use_pose must be True")

        self.fusion_proj = nn.Linear(fusion_in, hidden_dim)
        if self.temporal_type == "transformer":
            enc = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=transformer_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.temporal = nn.TransformerEncoder(enc, num_layers=num_layers)
            head_in = hidden_dim
        else:
            self.temporal = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            head_in = hidden_dim * 2

        self.head     = nn.Linear(head_in, vocab_size)
        self.dropout  = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, rgb: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb  : (B, C, T, H, W)
            pose : (B, T, D)

        Returns:
            log_probs : (T, B, vocab_size)  —  CTC-compatible format
        """
        feats: List[torch.Tensor] = []

        if self.use_rgb:
            feats.append(self.rgb_enc(rgb))      # (B, T, hidden_dim)

        if self.use_pose:
            feats.append(self.pose_enc(pose))    # (B, T, hidden_dim)

        x = torch.cat(feats, dim=-1)             # (B, T, fusion_in)
        x = torch.relu(self.fusion_proj(x))      # (B, T, hidden_dim)
        x = self.dropout(x)

        if self.temporal_type == "transformer":
            x = self.temporal(x)                 # (B, T, hidden_dim)
        else:
            x, _ = self.temporal(x)              # (B, T, 2*hidden_dim)
        x = self.dropout(x)
        x = self.head(x)                         # (B, T, vocab_size)
        log_p = self.log_softmax(x)              # (B, T, vocab_size)
        return log_p.permute(1, 0, 2)            # (T, B, vocab_size)  for CTC


def _confidence_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    """Get sequence confidence proxy from frame-wise max probs."""
    probs = log_probs.exp().permute(1, 0, 2)  # (B, T, V)
    conf = probs.max(dim=-1).values.mean(dim=-1)
    return conf.clamp(1e-6, 1 - 1e-6)


def calibrate_confidence_temperature(
    model: ISignCSLRModel,
    loader,
    device: torch.device,
    blank_id: int,
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

            hyps = ctc_greedy_decode(log_probs, blank_id)
            refs = [gt[i, : gt_lens[i]].tolist() for i in range(gt.shape[0])]
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


def word_error_rate(hypotheses: List[List[int]], references: List[List[int]]) -> float:
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
) -> float:
    model.train()
    total_loss  = 0.0
    total_steps = 0

    for batch in loader:
        rgb    = batch["rgb"].to(device, non_blocking=True)
        pose   = batch["pose"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        label_lens = batch["label_lengths"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(_AMP_DEVICE, enabled=use_amp):
            log_probs = model(rgb, pose)          # (T, B, V)
            T, B, _   = log_probs.shape
            input_lens = torch.full((B,), T, dtype=torch.long, device=device)
            loss = criterion(log_probs, labels, input_lens, label_lens)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss  += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate(
    model: ISignCSLRModel,
    loader,
    criterion: nn.CTCLoss,
    device: torch.device,
    blank_id: int,
) -> Tuple[float, float]:
    model.eval()
    total_loss  = 0.0
    total_steps = 0
    all_hyps: List[List[int]] = []
    all_refs: List[List[int]] = []

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

        hyps = ctc_greedy_decode(log_probs, blank_id)
        refs = [labels[i, : label_lens[i]].tolist() for i in range(B)]
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
            "best_wer":  best_wer,
            "args":      vars(args),
        },
        str(path),
    )


def load_checkpoint(
    path: Path,
    model: ISignCSLRModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> Dict:
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


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
    p.add_argument("--num-frames",  type=int, default=64,               help="Frames per clip")
    p.add_argument("--frame-size",  type=int, nargs=2, default=[224, 224], metavar=("H", "W"), help="Frame resize dimensions")
    p.add_argument("--max-samples", type=int, default=0,                help="Limit samples (0=all; useful for testing)")
    # Streams
    p.add_argument("--no-rgb",      action="store_true",                help="Disable RGB stream (pose only)")
    p.add_argument("--no-pose",     action="store_true",                help="Disable pose stream (RGB only)")
    p.add_argument("--no-pretrained", action="store_true",              help="Do not use pretrained CNN weights")
    # Model
    p.add_argument("--hidden-dim",  type=int, default=256,              help="LSTM / encoder hidden dimension")
    p.add_argument("--num-layers",  type=int, default=2,                help="LSTM layers")
    p.add_argument("--dropout",     type=float, default=0.3,            help="Dropout rate")
    p.add_argument("--temporal-type", choices=["bilstm", "transformer"], default="bilstm", help="Temporal encoder type")
    p.add_argument("--transformer-heads", type=int, default=4, help="Attention heads for transformer temporal encoder")
    p.add_argument("--hard-negative-prob", type=float, default=0.10, help="Probability of synthetic no-sign negatives in train split")
    p.add_argument("--temporal-jitter", type=int, default=2, help="Temporal index jitter for data augmentation")
    p.add_argument("--frame-drop-prob", type=float, default=0.05, help="Random frame drop probability")
    p.add_argument("--brightness-jitter", type=float, default=0.15, help="Brightness jitter strength")
    p.add_argument("--blur-prob", type=float, default=0.10, help="Gaussian blur augmentation probability")
    p.add_argument("--noise-std", type=float, default=0.02, help="Additive RGB noise std")
    p.add_argument("--pose-jitter-std", type=float, default=0.01, help="Pose jitter std")
    p.add_argument("--calibrate-confidence", action="store_true", default=True, help="Fit confidence temperature on validation set")
    p.add_argument("--no-calibrate-confidence", dest="calibrate_confidence", action="store_false", help="Disable confidence temperature fitting")
    # Training
    p.add_argument("--epochs",      type=int, default=50,               help="Total training epochs")
    p.add_argument("--batch-size",  type=int, default=4,                help="Batch size")
    p.add_argument("--lr",          type=float, default=1e-4,           help="Learning rate")
    p.add_argument("--weight-decay",type=float, default=1e-4,           help="AdamW weight decay")
    p.add_argument("--clip-grad",   type=float, default=5.0,            help="Gradient clipping norm")
    p.add_argument("--warmup-epochs", type=int, default=3,              help="LR warmup epochs")
    p.add_argument("--no-amp",      action="store_true",                help="Disable automatic mixed precision")
    # Logging / checkpoints
    p.add_argument("--ckpt-dir",    default="checkpoints/isign_fast_v2", help="Checkpoint save directory")
    p.add_argument("--save-every-epoch", action="store_true", default=True, help="Save an epoch_{N}.pt checkpoint every epoch")
    p.add_argument("--no-save-every-epoch", dest="save_every_epoch", action="store_false", help="Disable per-epoch checkpoint files")
    p.add_argument("--log-interval",type=int, default=50,               help="Log every N batches")
    p.add_argument("--resume",      default=None,                       help="Path to checkpoint to resume from")
    p.add_argument("--workers",     type=int, default=2,                help="DataLoader workers")
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
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")
    if args.no_rgb and args.no_pose:
        raise ValueError("Cannot disable both streams. Use at least one of RGB or pose.")

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

    log.info("Device: %s  |  AMP: %s", device, use_amp)
    if device.type == "cuda":
        log.info("GPU: %s  |  VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Data
    data_dir = Path(args.data_dir)
    vocab_path = Path(args.vocab) if args.vocab else data_dir / "vocab.json"
    if not vocab_path.exists():
        log.error("vocab.json not found at %s — run preprocess_isign.py first", vocab_path)
        sys.exit(1)

    log.info("Loading dataloaders from %s …", data_dir)
    train_loader, val_loader, test_loader, vocab = build_dataloaders(
        data_dir       = str(data_dir),
        batch_size     = args.batch_size,
        num_frames     = args.num_frames,
        frame_size     = tuple(args.frame_size),
        num_workers    = args.workers,
        use_poses      = not args.no_pose,
        use_frames     = not args.no_rgb,
        max_samples    = args.max_samples,
        pin_memory     = device.type == "cuda",
        hard_negative_prob=args.hard_negative_prob,
        temporal_jitter=args.temporal_jitter,
        frame_drop_prob=args.frame_drop_prob,
        brightness_jitter=args.brightness_jitter,
        blur_prob=args.blur_prob,
        noise_std=args.noise_std,
        pose_jitter_std=args.pose_jitter_std,
    )
    vocab_size = len(vocab)
    blank_id   = vocab.index("<blank>") if "<blank>" in vocab else 0
    log.info("Vocab size: %d  |  Blank id: %d", vocab_size, blank_id)

    # Infer pose dimension from first validation batch
    pose_dim = 34  # default (17 keypoints × 2)
    try:
        sample_batch = next(iter(val_loader))
        pose_dim = sample_batch["pose"].shape[-1]
        log.info("Pose dimension (inferred): %d", pose_dim)
    except StopIteration:
        pass

    # Model
    model = ISignCSLRModel(
        vocab_size     = vocab_size,
        hidden_dim     = args.hidden_dim,
        pose_input_dim = pose_dim,
        num_layers     = args.num_layers,
        dropout        = args.dropout,
        use_rgb        = not args.no_rgb,
        use_pose       = not args.no_pose,
        pretrained_cnn = not args.no_pretrained,
        temporal_type  = args.temporal_type,
        transformer_heads=args.transformer_heads,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %.2fM", total_params / 1e6)

    # Optimiser & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # CTC loss
    criterion = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)

    # AMP
    scaler = GradScaler(enabled=use_amp)

    # Resume
    start_epoch = 1
    best_wer    = float("inf")
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), model, optimizer, scheduler)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_wer    = ckpt.get("best_wer", float("inf"))
        log.info("Resumed from epoch %d  (best WER: %.4f)", start_epoch - 1, best_wer)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
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
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # LR warmup
        if epoch <= args.warmup_epochs:
            lr_scale = epoch / max(args.warmup_epochs, 1)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_scale

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, scaler, use_amp, args.clip_grad,
        )
        val_loss, val_wer = evaluate(model, val_loader, criterion, device, blank_id)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = float(optimizer.param_groups[0]["lr"])
        log.info(
            "Epoch %03d/%03d  |  train_loss=%.4f  |  val_loss=%.4f  |  "
            "val_WER=%.4f  |  lr=%.2e  |  %.0fs",
            epoch, args.epochs,
            train_loss, val_loss, val_wer,
            lr_now,
            elapsed,
        )

        # Save best
        is_best = False
        if val_wer < best_wer:
            best_wer = val_wer
            is_best = True
            save_checkpoint(
                ckpt_dir / "best.pt",
                epoch, model, optimizer, scheduler, best_wer, args,
            )
            log.info("  *** New best WER: %.4f — checkpoint saved ***", best_wer)

        history_row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_wer": float(val_wer),
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
                epoch, model, optimizer, scheduler, best_wer, args,
            )

        # Save latest
        save_checkpoint(
            ckpt_dir / "last.pt",
            epoch, model, optimizer, scheduler, best_wer, args,
        )

    # Final test evaluation
    log.info("=== Final evaluation on test set ===")
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model)
    test_loss, test_wer = evaluate(model, test_loader, criterion, device, blank_id)
    log.info("Test loss: %.4f  |  Test WER: %.4f", test_loss, test_wer)

    calibration = {"temperature": 1.0, "nll_before": 0.0, "nll_after": 0.0, "samples": 0}
    if args.calibrate_confidence:
        log.info("Calibrating confidence temperature on validation set …")
        calibration = calibrate_confidence_temperature(model, val_loader, device, blank_id)
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
