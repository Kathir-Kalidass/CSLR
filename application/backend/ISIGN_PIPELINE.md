# iSign Dataset — Full Pipeline Guide

End-to-end instructions for downloading, preprocessing, and training a CSLR
model on the **iSign v1.1** Indian Sign Language dataset (~228 GB).

---

## Hardware Requirements

| Component  | Minimum              | Recommended          |
|------------|----------------------|----------------------|
| Storage    | 230 GB free          | 400 GB (SSD preferred)|
| RAM        | 16 GB                | 32+ GB               |
| GPU        | —  (CPU possible)    | 8+ GB VRAM           |
| Python     | 3.9+                 | 3.10+                |

> **18 GB RAM / 200 GB storage tip**: Download metadata + poses only to stay
> under ~180 GB, skip video extraction with `--no-videos`, and use
> `--max-samples` to train on a subset first.

---

## Step 0 — Install Dependencies

```bash
cd application/backend
pip install -r requirements.txt
```

---

## Step 1 — HuggingFace Login

Get a token from <https://huggingface.co/settings/tokens> then:

```bash
huggingface-cli login
# paste token when prompted
```

---

## Step 2 — Download the Dataset

```bash
# From application/backend/
# Full download (~228 GB)
python scripts/download_isign.py --target dataset/isign

# Poses only (~170 GB, no raw video)
python scripts/download_isign.py --target dataset/isign --no-videos

# Metadata only (< 20 MB, for quick testing)
python scripts/download_isign.py --target dataset/isign --metadata-only
```

Downloaded files land in `dataset/isign/`:
```
dataset/isign/
  iSign_v1.1.csv
  word-description-dataset_v1.1.csv
  word-presence-dataset_v1.1.csv
  iSign-poses_v1.1_part_aa
  iSign-poses_v1.1_part_ab
  iSign-poses_v1.1_part_ac
  iSign-poses_v1.1_part_ad
  iSign-videos_v1.1_part_aa
  iSign-videos_v1.1_part_ab
```

---

## Step 3 — Merge Split Archives

The dataset is split with Unix `split`.  Merge and extract:

```bash
# Merge + extract both poses and videos
python scripts/merge_isign_parts.py \
    --src dataset/isign \
    --out dataset/isign

# Poses only
python scripts/merge_isign_parts.py \
    --src dataset/isign --out dataset/isign --poses-only

# Merge without extracting (save ~170+ GB of duplicated data)
python scripts/merge_isign_parts.py \
    --src dataset/isign --out dataset/isign --no-extract
```

After this step:
```
dataset/isign/
  isign_poses.tar       # merged pose archive
  isign_videos.tar      # merged video archive
  poses/                # extracted pose .npy files
  videos/               # extracted video .mp4 files
```

---

## Step 4 — Preprocess

Reads `iSign_v1.1.csv`, extracts video frames, validates poses, and creates
the `train.json / val.json / test.json / vocab.json` needed for training.

```bash
# Full preprocessing
python scripts/preprocess_isign.py \
    --isign-dir dataset/isign \
    --out-dir   dataset/isign_processed \
    --max-frames 64 \
    --workers 4

# Quick test on 500 samples
python scripts/preprocess_isign.py \
    --isign-dir dataset/isign \
    --out-dir   dataset/isign_processed_small \
    --max-samples 500 \
    --max-frames 32

# Poses only (skip frame extraction if no videos downloaded)
python scripts/preprocess_isign.py \
    --isign-dir dataset/isign \
    --out-dir   dataset/isign_processed \
    --skip-frames
```

Output:
```
dataset/isign_processed/
  vocab.json        # ["<blank>", "<unk>", "A", "ABOUT", ...]
  train.json
  val.json
  test.json
  frames/
    <video_id>/
      frame_0000.jpg ...
  poses/
    <video_id>.npy  # (T, D) float32
```

---

## Step 5 — Train

```bash
# Smoke-test (CPU, 10 samples, 2 epochs)
python scripts/train_isign.py \
    --data-dir dataset/isign_processed \
    --epochs 2 --batch-size 2 --max-samples 10

# Pose-only training (faster, less VRAM)
python scripts/train_isign.py \
    --data-dir dataset/isign_processed \
    --no-rgb --epochs 50 --batch-size 16 --lr 1e-4

# Full dual-stream training (GPU recommended)
python scripts/train_isign.py \
    --data-dir dataset/isign_processed \
    --epochs 100 --batch-size 8 --lr 1e-4 \
    --hidden-dim 512 --num-layers 3

# Resume from checkpoint
python scripts/train_isign.py \
    --data-dir dataset/isign_processed \
    --resume checkpoints/isign/best_model.pt
```

Checkpoints are saved to `checkpoints/isign/`:
- `best_model.pt`  — lowest validation WER
- `latest.pt`      — most recent epoch
- `results.json`   — final test metrics

---

## Memory-Saving Tips (18 GB RAM)

| Tip                         | Flag / setting                    |
|-----------------------------|-----------------------------------|
| Train on poses only         | `--no-rgb`                        |
| Reduce batch size           | `--batch-size 2`                  |
| Reduce frame count          | `--max-frames 32`                 |
| Use small hidden dim        | `--hidden-dim 128`                |
| Limit dataset subset        | `--max-samples 2000`              |
| Reduce DataLoader workers   | `--workers 0`                     |
| Disable AMP (CPU)           | `--no-amp` (auto on CPU)          |

---

## Dataset Loader API

```python
from app.data.isign_dataset import ISignDataset, build_dataloaders

# Direct dataset
ds = ISignDataset(
    data_dir="dataset/isign_processed",
    split="train",
    num_frames=64,
    use_poses=True,
    use_frames=True,
)
sample = ds[0]
# sample["rgb"]    → (C, T, H, W) tensor
# sample["pose"]   → (T, D) tensor
# sample["labels"] → (L,) gloss id tensor

# Convenience factory
train_loader, val_loader, test_loader, vocab = build_dataloaders(
    data_dir="dataset/isign_processed",
    batch_size=8,
    num_workers=4,
)
```

---

## File Overview

| File                               | Purpose                              |
|------------------------------------|--------------------------------------|
| `scripts/download_isign.py`        | Download from HuggingFace Hub        |
| `scripts/merge_isign_parts.py`     | Merge split archives + extract       |
| `scripts/preprocess_isign.py`      | Build frames / poses / JSON splits   |
| `app/data/isign_dataset.py`        | PyTorch Dataset + DataLoader factory |
| `scripts/train_isign.py`           | Dual-stream CSLR training + eval    |
