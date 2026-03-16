# Architecture Diagrams & Module Mapping

## Visual Documentation Reference

This document maps the architecture diagrams in `report_pages/architecture_diagram/` to the implementation workflow and code modules.

---

## Available Architecture Diagrams

### 1. sign_archi-Architecture.png
**Purpose:** Overall system architecture (end-to-end pipeline)

**Expected Content:**
- High-level flow: Video Input → Processing → Output (Text + Speech)
- Major components: Preprocessing, Feature Extraction, Fusion, Temporal Modeling, Decoding, Translation, TTS
- Data flow arrows showing information movement

**Maps to Workflow:**
- See `DETAILED_WORKFLOW.md` Section: "Complete Data Flow Diagram"
- Corresponds to all 7 stages

**Implementation Files:**
- Overall pipeline: `Online/CSLR/`, `Online/CTC_fusion/`
- Orchestration: Custom integration scripts (TBD)

---

### 2. sign_archi-Module1.png
**Purpose:** Video Preprocessing & Feature Extraction

**Expected Content:**
- Input: Raw video frames
- Preprocessing: Frame sampling, normalization, ROI detection
- Pose estimation: MediaPipe/OpenPose keypoint extraction
- Feature extraction: RGB (CNN) and Pose (GCN/MLP) encoders
- Output: RGB features (T×D_rgb) and Pose features (T×D_pose)

**Maps to Workflow:**
- Stage 1: Video Preprocessing (Section in `DETAILED_WORKFLOW.md`)
- Stage 2: Multi-Feature Extraction

**From Base Paper:**
- Dual-stream architecture (RGB + Pose)
- CNN backbone for RGB (ResNet-18/50)
- Pose encoder options (ST-GCN, MLP, RNN)

**Implementation Files:**
```
references/NLA-SLR/
├── gen_pose.py              → Pose keypoint extraction
└── dataset/
    ├── Dataloader.py        → Data loading (RGB + Pose)
    └── Dataset.py           → Dataset classes

TwoStreamNetwork/
├── extract_feature.py       → Feature extraction utilities
├── preprocess/              → Video preprocessing
└── modelling/
    ├── rgb_stream.py        → CNN backbone implementation
    └── pose_stream.py       → Pose encoder (GCN/RNN)
```

**Key Hyperparameters:**
- Input resolution: 224×224 or 256×256
- Frame sampling: 32–64 frames per clip
- RGB backbone: ResNet-18 (fast) or ResNet-50 (accurate)
- Pose keypoints: 33 body + 21×2 hands + 468 face (MediaPipe Holistic)
- Feature dimensions: D_rgb = 512–2048, D_pose = 256–512

---

### 3. sign_archi-Module2.png
**Purpose:** Attention-Based Fusion & Temporal Modeling

**Expected Content:**
- Input: RGB features (T×D_rgb) and Pose features (T×D_pose)
- Temporal attention: Self-attention per modality
- Cross-modal fusion: Attention-weighted combination
- Temporal modeling: BiLSTM or Transformer encoder
- CTC layer: Frame-level gloss predictions
- Output: Sequence embeddings (T × D_model)

**Maps to Workflow:**
- Stage 3: Feature Fusion with Attention Mechanism
- Stage 4: Temporal Sequence Modeling

**From Base Paper (Core Innovation):**
- **Efficient Multi-Feature Attention Mechanism**
  - Temporal self-attention per stream
  - Cross-modal attention fusion: F_fused = α × F_rgb + β × F_pose
  - Adaptive weighting learned per time step
  - Result: +5–8% accuracy over concatenation

**Implementation Files:**
```
TwoStreamNetwork/modelling/
├── fusion.py                → Attention-based fusion module
├── temporal_model.py        → BiLSTM/Transformer encoder
└── ctc_decoder.py           → CTC loss and decoding

references/NLA-SLR/modelling/
├── attention.py             → Multi-head attention
└── sequence_model.py        → Temporal modeling layers
```

**Key Components:**
1. **Temporal Self-Attention**
   - Input: Features (T × D)
   - Query, Key, Value projections
   - Multi-head attention (8 heads)
   - Output: Attended features (T × D)

2. **Cross-Modal Fusion**
   - Option A: Attention-based (base paper)
   - Option B: Concatenation + MLP (baseline)
   - Fusion dimension: D_fusion = 512–1024

3. **Temporal Encoder**
   - BiLSTM: 2 layers, hidden size 512–1024
   - Transformer: 6–8 layers, D_model = 512
   - Dropout: 0.2–0.3

4. **CTC Layer**
   - Output: Probability distribution over vocabulary per frame
   - Vocabulary size: |V| + 1 (blank token)

**Training Loss:**
```
L_total = L_CTC + λ × L_attention
```
Where L_attention encourages balanced modality weighting.

---

### 4. sign_archi-module3.png
**Purpose:** Decoding & Language-Level Correction

**Expected Content:**
- Input: Frame-level gloss probabilities (T × |V|)
- CTC decoding: Greedy or beam search
- Gloss sequence buffering: Handle partial predictions
- Language translation: Seq2Seq (ISL gloss → English text)
- Grammar correction: Post-processing for fluency
- Output: Corrected English sentence

**Maps to Workflow:**
- Stage 5: Decoding & Prediction
- Stage 6: Language-Level Correction

**Implementation Files:**
```
Online/CTC_fusion/
├── ctc_decoder.py           → Greedy/beam search decoding
└── buffer.py                → Caption buffering and reordering

Online/SLT/
├── translator.py            → Gloss-to-text Seq2Seq model
└── grammar_correction.py    → Post-processing for grammar

references/NLA-SLR/
└── prediction.py            → End-to-end inference pipeline
```

**Decoding Workflow:**

1. **CTC Decoding**
   - **Greedy (Fast):** Select argmax per frame → collapse repetitions → remove blanks
   - **Beam Search (Accurate):** Maintain top-K hypotheses with LM priors
   - Beam width: 5–10
   - Language model weight: 0.1–0.5

2. **Caption Buffering** (Real-Time Only)
   - Sliding window: 3–5 second clips
   - Confidence thresholding: Commit predictions above 0.8
   - Reorder when longer context increases confidence

3. **Gloss-to-Text Translation**
   - Input: ISL gloss sequence (e.g., "ME BOOK READ")
   - Model: Seq2Seq (LSTM encoder-decoder) or Transformer
   - Training data: iSign DB parallel corpus (ISL glosses ↔ English sentences)
   - Output: English sentence ("I read a book")

4. **Grammar Correction**
   - Rule-based: Fix common ISL→English patterns (SOV → SVO)
   - LM-based: T5/BART fine-tuned on grammar correction
   - Tools: LanguageTool, lightweight GPT-based models

**Key Metrics:**
- WER (Word Error Rate): Measure recognition accuracy
- BLEU Score: Measure translation quality
- Latency: Decoding + translation time

---

### 5. sign_archi-module4.png
**Purpose:** Output Generation (Text Display + Text-to-Speech)

**Expected Content:**
- Input: Corrected English sentence
- Text output: Display in UI (console, web app, mobile app)
- TTS synthesis: Convert text to audio waveform
- Audio output: Stream to speaker or save to file
- Feedback loop: User confirmation or correction (optional)

**Maps to Workflow:**
- Stage 7: Text-to-Speech (TTS) Synthesis
- Output integration and UI

**Implementation Files:**
```
Online/
├── tts_module.py            → TTS integration (gTTS, pyttsx3, Cloud APIs)
└── ui/
    ├── console_ui.py        → Text output (terminal)
    ├── web_app.py           → Web-based interface (Flask/FastAPI)
    └── mobile_app/          → Mobile UI (optional)
```

**TTS Options:**

1. **Offline TTS (Fast, Lower Quality)**
   - **gTTS (Google TTS):** Text → MP3, requires internet
   - **pyttsx3:** Offline, cross-platform, basic voices
   - Latency: 50–200ms
   - Use case: Rapid prototyping, low-resource environments

2. **Cloud TTS (Natural, Higher Latency)**
   - **Google Cloud TTS:** Neural voices, 40+ languages
   - **AWS Polly:** High-quality voices, streaming support
   - **Azure Speech Service:** Real-time synthesis
   - Latency: 200–500ms
   - Use case: Production systems with internet

3. **Neural TTS (Best Quality, GPU Required)**
   - **Tacotron 2 + WaveGlow:** End-to-end neural synthesis
   - **VITS:** Variational Inference TTS (state-of-the-art)
   - Latency: 500ms–2s (batched inference)
   - Use case: High-quality offline applications

**Real-Time Pipeline:**
```
Corrected Text → TTS Synthesis → Audio Buffer → Speaker Output
      ↓                ↓                ↓              ↓
   Queue (FIFO)   Async Worker    Audio Stream    Real-time
```

**Latency Budget (Total < 500ms target):**
- Video capture: 30ms
- Feature extraction: 50–100ms
- Temporal modeling + CTC: 100–150ms
- Decoding: 20–50ms
- Translation: 50–100ms
- TTS: 50–200ms (offline) or 200–500ms (cloud)

**Optimization Strategies:**
- Pre-generate common phrases (e.g., "Hello", "Thank you")
- Asynchronous processing: Overlap stages
- Model quantization: FP16/INT8 for faster inference
- Batching: Process multiple frames simultaneously

---

## Module Dependencies & Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Module 1: Preprocessing & Feature Extraction                   │
│  Files: gen_pose.py, extract_feature.py, rgb_stream.py,         │
│         pose_stream.py                                           │
│  Output: RGB features (T×512) + Pose features (T×256)           │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 2: Attention Fusion & Temporal Modeling                 │
│  Files: fusion.py, attention.py, temporal_model.py,             │
│         ctc_decoder.py                                           │
│  Output: Frame-level gloss probabilities (T × |V|)              │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 3: Decoding & Language Correction                        │
│  Files: ctc_decoder.py, translator.py, grammar_correction.py    │
│  Output: Corrected English sentence                              │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Module 4: Text-to-Speech & Output                               │
│  Files: tts_module.py, console_ui.py, web_app.py                │
│  Output: Text display + Audio waveform                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Files Mapping

Each module has associated configuration files in `references/NLA-SLR/configs/` and `TwoStreamNetwork/experiments/`.

### ASL Experiments (Pretraining)
```
references/NLA-SLR/configs/
├── nla_slr_msasl_100.yaml     → MS-ASL 100 glosses
├── nla_slr_msasl_500.yaml     → MS-ASL 500 glosses
├── nla_slr_msasl_1000.yaml    → MS-ASL 1000 glosses
├── nla_slr_wlasl_100.yaml     → WLASL 100 glosses
├── nla_slr_wlasl_1000.yaml    → WLASL 1000 glosses
└── nla_slr_wlasl_2000.yaml    → WLASL 2000 glosses
```

### Feature Type Experiments
```
references/NLA-SLR/configs/
├── rgb_frame32.yaml           → RGB-only, 32 frames
├── rgb_frame64.yaml           → RGB-only, 64 frames
├── pose_frame32.yaml          → Pose-only, 32 frames
├── pose_frame64.yaml          → Pose-only, 64 frames
├── two_frame32.yaml           → RGB+Pose (dual-stream), 32 frames
├── two_frame64.yaml           → RGB+Pose (dual-stream), 64 frames
```

### ISL Experiments (Fine-Tuning)
```
references/NLA-SLR/configs/
└── nla_slr_nmf.yaml           → Adapt this for iSign DB (ISL)
```

**To Create ISL Config:**
1. Copy `nla_slr_nmf.yaml` → `isl_isign_db.yaml`
2. Update:
   - `dataset_path`: Path to iSign DB
   - `vocabulary_size`: ISL gloss count
   - `num_classes`: Same as vocabulary_size
   - `pretrained_model`: Path to ASL pretrained weights
   - `learning_rate`: Reduce to 1e-5 (fine-tuning)
   - `freeze_backbone`: True (first 50% epochs), then False

---

## Training Commands (Examples)

### Module 1+2: Feature Extraction + Fusion + Temporal Model

**ASL Pretraining:**
```bash
# Train on MS-ASL 1000 glosses with dual-stream (RGB+Pose)
python references/NLA-SLR/training.py \
    --config references/NLA-SLR/configs/nla_slr_msasl_1000.yaml \
    --gpu 0 \
    --batch_size 16 \
    --epochs 100

# Or using TwoStreamNetwork
python TwoStreamNetwork/training.py \
    --config TwoStreamNetwork/experiments/msasl_dual_stream.yaml
```

**ISL Fine-Tuning:**
```bash
# Fine-tune on iSign DB (ISL)
python references/NLA-SLR/training.py \
    --config references/NLA-SLR/configs/isl_isign_db.yaml \
    --pretrained checkpoints/msasl_best.pth \
    --gpu 0 \
    --batch_size 8 \
    --epochs 50 \
    --learning_rate 1e-5
```

### Module 3: Translation Model Training

```bash
# Train ISL gloss → English Seq2Seq translator
python Online/SLT/train_translator.py \
    --data data/iSign_parallel_corpus.csv \
    --model transformer \
    --epochs 50 \
    --batch_size 32
```

### Module 4: End-to-End Inference

```bash
# Real-time inference with TTS
python Online/CSLR/real_time_inference.py \
    --model checkpoints/isl_model.pth \
    --translator checkpoints/isl_translator.pth \
    --tts gtts \
    --camera 0
```

---

## Ablation Study Mapping (To Prove Base Paper Contributions)

Test each component's impact:

| Experiment | Config Modifications | Expected Result |
|------------|---------------------|-----------------|
| RGB-only | Use `rgb_frame64.yaml` | Baseline: ~70% accuracy |
| Pose-only | Use `pose_frame64.yaml` | Lower: ~65% (less info) |
| RGB+Pose (concat) | Disable attention in fusion | ~73% (simple fusion) |
| **RGB+Pose (attention)** | **Base paper config** | **~78% (+5% from attention)** |
| BiLSTM temporal | `sequence_model: bilstm` | ~78% (good for real-time) |
| Transformer temporal | `sequence_model: transformer` | ~81% (best accuracy) |
| Greedy decoding | `decoder: greedy` | Fast, ~78% |
| Beam search | `decoder: beam`, `beam_width: 5` | +2-4% accuracy |
| No language correction | Skip Module 3 translation | Low BLEU (~15) |
| With language correction | Full pipeline | High BLEU (~35) |

**Key Takeaway:** Base paper's attention fusion provides significant accuracy boost.

---

## Summary: Architecture → Code Mapping

| Diagram | Workflow Stage | Key Files | Base Paper Contribution |
|---------|----------------|-----------|------------------------|
| Module 1 | Preprocessing + Feature Extraction | `gen_pose.py`, `rgb_stream.py`, `pose_stream.py` | Dual-stream (RGB + Pose) |
| Module 2 | Fusion + Temporal | `fusion.py`, `temporal_model.py` | **Attention-based fusion** |
| Module 3 | Decoding + Translation | `ctc_decoder.py`, `translator.py` | Extension (new) |
| Module 4 | Output (TTS) | `tts_module.py`, `web_app.py` | Extension (new) |

**Visual Guide:** Use architecture diagrams in presentations; refer to this document for implementation details.

---

**Last Updated:** January 24, 2026  
**Use With:** `DETAILED_WORKFLOW.md`, `CSLR_Project_Report.md`, `QUICK_REFERENCE.md`
