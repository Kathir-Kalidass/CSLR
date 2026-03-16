# ISL_CSLRT Corpus Training Guide

This guide uses the new trainer script:

`application/backend/scripts/train_isl_cslrt.py`

It is built for this dataset layout:
- `dataset/ISL_CSLRT_Corpus/Frames_Sentence_Level/<sentence>/<sequence>/*.jpg`
- `dataset/ISL_CSLRT_Corpus/corpus_csv_files/ISL Corpus sign glosses.csv`

The model follows the project workflow:
- RGB stream + Pose stream
- Attention/gated fusion
- Temporal model (BiLSTM/Transformer)
- CTC training + decoding

## 1) Analyze the dataset

```bash
cd /home/kathir/CSLR/application/backend
python scripts/train_isl_cslrt.py --analyze-only
```

Output:
- `checkpoints/isl_cslrt_experiment/dataset_analysis.json`

## 2) (Recommended) Precompute pose cache

MediaPipe extraction is expensive. Cache once before training:

```bash
python scripts/train_isl_cslrt.py --prepare-pose-cache-only
```

## 3) Start training

Baseline run:

```bash
python scripts/train_isl_cslrt.py \
  --epochs 60 \
  --batch-size 4 \
  --num-frames 64 \
  --image-size 224 \
  --learning-rate 1e-4 \
  --workers 4 \
  --checkpoint-strategy best_only
```

Checkpoint strategy options:
- `best_only` (default): saves only `checkpoints/best.pt`
- `best_and_last`: saves `best.pt` and `last.pt`
- `all`: saves `best.pt`, `last.pt`, and periodic `epoch_XXX.pt`

## 4) Resume training

```bash
python scripts/train_isl_cslrt.py \
  --resume checkpoints/isl_cslrt_experiment/checkpoints/best.pt \
  --checkpoint-strategy best_only \
  --epochs 100
```

## 5) Useful outputs

In `checkpoints/isl_cslrt_experiment/`:
- `dataset_analysis.json`
- `split_summary.json`
- `history.json`
- `training_history.jsonl` (append-only epoch log)
- `train_full_run.log` (append-only run log)
- `final_metrics.json`
- `checkpoints/best.pt` (default)
- `checkpoints/last.pt` (only with `--checkpoint-strategy best_and_last` or `all`)
- `checkpoints/epoch_XXX.pt` (only with `--checkpoint-strategy all`)
- `rgb_stream_best.pt`, `pose_stream_best.pt`, `fusion_best.pt`, `temporal_best.pt` (only with `--export-module-weights`)
- `vocab_tokens.json`, `token_to_id.json`
- `manifests/*.json`

## 6) Accuracy-focused suggestions

- Keep `--num-frames 64` for continuous signing.
- Use `--backbone resnet18` first for speed; try `resnet34` for better accuracy.
- Train at least `60-100` epochs.
- Do not disable pose unless testing ablations.
- If GPU memory is tight: reduce `--batch-size` to `2`.
