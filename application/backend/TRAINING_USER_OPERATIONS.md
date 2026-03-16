# iSign Training User Operations

This guide provides user-friendly operations for preparing data, starting training, and viewing detailed history.

## 1. Dataset Readiness Checks

Use API endpoints before running training:

- `GET /api/v1/training/dataset-status`
- `GET /api/v1/training/ready`

What to verify:

- `metadata_ready = true`
- `processed_dataset_ready = true`
- `cuda_available = true` when using GPU

## 2. Safe Data Preparation (Current Partial Pose Download)

If pose parts are still downloading, use RGB mode for now.

### Merge downloaded parts

```bash
python scripts/merge_isign_parts.py --src dataset/isign --out dataset/isign
```

### Preprocess with strict filtering

```bash
python scripts/preprocess_isign.py \
  --isign-dir dataset/isign \
  --out-dir dataset/isign_processed \
  --modality-mode rgb \
  --min-frames 8 \
  --min-gloss-tokens 1 \
  --max-gloss-tokens 30
```

## 3. Start Training (GPU, User-Friendly)

```bash
python scripts/train_isign.py \
  --data-dir dataset/isign_processed \
  --device cuda \
  --require-cuda \
  --allow-tf32 \
  --epochs 30 \
  --batch-size 8 \
  --ckpt-dir checkpoints/isign_fast_v2 \
  --save-every-epoch
```

Outputs saved automatically:

- `best.pt` (best validation WER)
- `last.pt` (latest epoch)
- `epoch_XXX.pt` (every epoch)
- `history.jsonl` (detailed epoch log)
- `history.csv` (spreadsheet-friendly)
- `results.json` (final summary)
- `train_config.json` (exact run config)

## 4. View Detailed Training History

### API view

- `GET /api/v1/training/history`
- `GET /api/v1/training/history?limit=50`
- `GET /api/v1/training/artifacts`

### Local files

- `checkpoints/isign_fast_v2/history.jsonl`
- `checkpoints/isign_fast_v2/history.csv`

## 5. Full Multimodal Mode (After Pose Download Completes)

Re-run preprocessing with multimodal constraints:

```bash
python scripts/preprocess_isign.py \
  --isign-dir dataset/isign \
  --out-dir dataset/isign_processed \
  --modality-mode multimodal \
  --min-frames 12
```

Then train with both streams:

```bash
python scripts/train_isign.py \
  --data-dir dataset/isign_processed \
  --device cuda \
  --require-cuda \
  --epochs 60 \
  --batch-size 8 \
  --ckpt-dir checkpoints/isign_fast_v2_multimodal
```

## 6. Preset Profiles

Preset templates are provided in:

- `configs/isign_training_profiles.json`

Profiles include:

- `quick_debug_cpu`
- `balanced_gpu_rgb_only`
- `full_multimodal_gpu`

## 7. Advanced Filtering And Decoding Controls

Set these in your environment for stronger noise suppression and better sign stability:

- `CONFIDENCE_THRESHOLD=0.72`
- `ENABLE_GLOSS_FILTER=true`
- `ENABLE_TEMPORAL_GLOSS_VOTING=true`
- `GLOSS_VOTE_WINDOW=5`
- `GLOSS_MIN_VOTES=2`
- `CTC_MIN_TOKEN_RUN=2`
- `CTC_MIN_TOKEN_MARGIN=0.04`
- `CTC_LENGTH_NORM_ALPHA=0.35`
- `CTC_REPETITION_PENALTY=0.15`

Recommended blocklist tuning:

- `GLOSS_BLOCKLIST=<blank>,<unk>,<pad>,sil,noise`

What this gives you:

- Better rejection of unwanted signs
- More stable outputs over time
- Reduced random token spikes in low-confidence frames

## 8. Adaptive Filtering Mode (Auto Strictness)

Adaptive mode increases strictness when frames are noisy and relaxes it when confidence is stable/high.

Enable and tune:

- `ENABLE_ADAPTIVE_FILTERING=true`
- `ADAPTIVE_STRICTNESS_STEP_UP=0.12`
- `ADAPTIVE_STRICTNESS_STEP_DOWN=0.08`
- `ADAPTIVE_NOISE_VAR_THRESHOLD=0.03`
- `ADAPTIVE_CONF_LOW=0.55`
- `ADAPTIVE_CONF_HIGH=0.85`
- `ADAPTIVE_THRESHOLD_BOOST_MAX=0.18`
- `ADAPTIVE_MAXTOK_REDUCTION=0.40`
- `ADAPTIVE_VOTES_BONUS_MAX=2`

Recommended use:

- Keep this enabled for real-time camera streams and noisy backgrounds.
- For clean studio videos, lower strictness by reducing `ADAPTIVE_THRESHOLD_BOOST_MAX`.
