# Real-Time Vision-Based Continuous ISL Recognition & Translation System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A deep learning-based system for real-time Indian Sign Language (ISL) recognition and translation using multi-feature attention mechanisms**

## 🎯 Project Overview

This project implements a comprehensive **continuous sign language recognition and translation system** that converts ISL video streams into grammatically correct English text and speech. The system employs a dual-stream architecture with attention-based fusion, leveraging both RGB visual features and pose/landmark information.

### Key Features

- ✅ **Dual-Stream Architecture**: RGB + Pose feature extraction and fusion
- ✅ **Attention-Based Fusion**: Adaptive weighting of complementary modalities
- ✅ **Continuous Recognition**: Handles unsegmented sign sequences using CTC alignment
- ✅ **Real-Time Performance**: Target latency < 500ms end-to-end
- ✅ **Language Translation**: Seq2Seq gloss-to-text with grammar correction
- ✅ **Text-to-Speech**: Integrated TTS for audio output
- ✅ **Transfer Learning**: ASL pretraining → ISL fine-tuning

## 📋 Table of Contents

- [Installation](#installation)
- [Documentation](#documentation)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)
- [Contributing](#contributing)

## 🚀 Installation

### Training Environment (Google Colab)

**The model training and evaluation use Google Colab with GPU acceleration:**
- GPU: Tesla T4/V100/A100 (provided by Colab)
- CUDA 12.1+ support
- Sliding-window inference for fast processing
- Mount Google Drive for dataset access

### Local Deployment Prerequisites

- Python 3.12 or higher
- FFmpeg (for video processing)
- 8GB+ RAM
- Webcam (for real-time inference)
- **Note:** Real-time webcam deployment runs locally due to browser sandbox limitations

### Setup

#### For Training on Google Colab

1. **Open in Colab**
   - Navigate to the training notebook in `references/NLA-SLR/`
   - Click "Open in Colab" badge
   
2. **Setup Colab environment**
   ```python
   # Mount Google Drive for dataset access
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Clone repository
   !git clone https://github.com/Kathir-Kalidass/CLSR.git
   %cd CLSR
   
   # Install dependencies (PyTorch pre-installed in Colab)
   !pip install -r application/requirements.txt
   ```

3. **Enable GPU acceleration**
   - Runtime → Change runtime type → GPU (T4/V100/A100)
   - Verify: `!nvidia-smi`

#### For Local Deployment (Webcam Inference)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kathir-Kalidass/CLSR.git
   cd CLSR
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   # CPU-only (sufficient for inference)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install -r application/requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import mediapipe; print('MediaPipe: OK')"
   python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
   ```

## 📚 Documentation

Comprehensive documentation is available in the [`doc/`](doc/) directory:

- **[01_project_overview_merged.md](doc/01_project_overview_merged.md)** - Project objectives, problem statement, and base paper analysis
- **[02_architecture_overview.md](doc/02_architecture_overview.md)** - High-level system architecture and design principles
- **[03_complete_modular_flow.md](doc/03_complete_modular_flow.md)** - Detailed module-by-module implementation guide
- **[QUICK_REFERENCE.md](doc/QUICK_REFERENCE.md)** - Quick answers for reviewers and presentations
- **[DETAILED_WORKFLOW.md](doc/DETAILED_WORKFLOW.md)** - Complete end-to-end workflow description

## 🏗️ Architecture

### System Pipeline

```
ISL Video → Preprocessing → Feature Extraction → Temporal Modeling → 
CTC Decoding → Translation → Grammar Correction → TTS → Text + Speech
```

### Core Components

1. **Module 1: Preprocessing**
   - Video capture and frame extraction
   - MediaPipe Holistic pose estimation
   - RGB normalization and augmentation

2. **Module 2: Feature Extraction & Fusion**
   - RGB Stream: CNN (ResNet-18/I3D)
   - Pose Stream: Keypoint encoding (GCN/MLP)
   - Attention-based fusion mechanism ⭐

3. **Module 3: Temporal Modeling**
   - BiLSTM or Transformer encoder
   - CTC alignment for continuous recognition
   - Beam search decoding

4. **Module 4: Language Processing**
   - Gloss-to-text translation (T5/BART)
   - Grammar correction
   - Text-to-speech synthesis

### Architecture Diagrams

<p align="center">
  <img src="report_pages/architecture_diagram/sign_archi-Architecture.png" width="800" alt="System Architecture">
</p>

> See [`report_pages/architecture_diagram/`](report_pages/architecture_diagram/) for detailed module diagrams

## 📊 Datasets

### Primary Dataset: iSign DB
- **118,000+ videos** with 1,000+ ISL glosses
- Sentence-level continuous annotations
- Multiple signers with real-world conditions

### ASL Pretraining
- **MS-ASL**: 25K videos, 1000 glosses
- **WLASL**: 21K videos, 2000 glosses
- Transfer learning for robust feature extractors

### Dataset Strategy
1. **Pretrain** on large-scale ASL datasets
2. **Adapt** vocabulary layer for ISL
3. **Fine-tune** on ISL datasets (iSign DB)
4. **Evaluate** on held-out ISL test set

## 🎯 Usage

### Training (Google Colab)

**All model training is performed on Google Colab with GPU acceleration:**

```python
# In Colab notebook
# Train dual-stream model with attention fusion
!python references/NLA-SLR/training.py --config configs/nla_slr_wlasl_1000.yaml --gpu 0

# Fine-tune on ISL with sliding-window inference
!python references/NLA-SLR/training.py \
    --config configs/nla_slr_isl.yaml \
    --pretrained checkpoints/asl_model.pth \
    --sliding-window --window-size 64
```

**Features:**
- GPU acceleration (T4/V100/A100)
- Sliding-window inference for continuous sign sequences
- Checkpoints auto-saved to Google Drive
- TensorBoard logging

### Evaluation (Google Colab)

```python
# Compute WER/BLEU metrics on test set
!python references/NLA-SLR/evaluation.py \
    --test-set /content/drive/MyDrive/iSign_DB/test/ \
    --model checkpoints/best_model.pth \
    --sliding-window
```

### Inference (Local - Webcam)

**Real-time webcam deployment runs locally due to browser sandbox limitations:**

```bash
# Download trained model from Colab/Drive
# Place in checkpoints/

# Run real-time webcam inference
python application/realtime_inference.py --camera 0 --model checkpoints/best_model.pth

# Or process pre-recorded video
python references/NLA-SLR/prediction.py --video path/to/video.mp4 --model checkpoints/best_model.pth
```

## 📁 Project Structure

```
CLSR/
├── application/           # Main application code
│   └── requirements.txt   # Python dependencies (Python 3.12+)
├── doc/                   # Comprehensive documentation
│   ├── 01_project_overview_merged.md
│   ├── 02_architecture_overview.md
│   ├── 03_complete_modular_flow.md
│   └── QUICK_REFERENCE.md
├── references/            # Reference implementations
│   ├── NLA-SLR/          # Multi-feature attention baseline
│   ├── Online/           # Online CSLR & SLT
│   └── TwoStreamNetwork/ # Two-stream architecture
├── report_pages/          # Reports and diagrams
│   ├── architecture_diagram/
│   ├── conference_journels_std/
│   └── Our_report_ppt/
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 📖 References

### Base Paper
**"Deep Learning-Based Sign Language Recognition Using Efficient Multi-Feature Attention Mechanism"**
- Located in: `report_pages/conference_journels_std/`
- Core contribution: Dual-stream architecture with attention-based fusion

### Additional Papers
1. **"Real-time Vision-based Indian Sign Language Translation Using Deep Learning Techniques"**
2. **"Toward Real-Time Recognition of Continuous Indian Sign Language: A Multi-Modal Approach Using RGB and Pose"**
3. **"iSign: A Benchmark for Indian Sign Language Processing"**

### Reference Implementations
- [NLA-SLR](https://github.com/FangyunWei/SLRT)
- [Online CSLR/SLT](https://github.com/FangyunWei/SLRT)
- [TwoStreamNetwork](https://github.com/FangyunWei/SLRT)

## 🎓 Key Metrics & Targets

| Metric | Target | Description |
|--------|--------|-------------|
| **WER** | < 20% | Word Error Rate for gloss recognition |
| **BLEU** | > 30 | Translation quality (ISL→English) |
| **ROUGE-L** | > 0.5 | Longest common subsequence score |
| **Latency** | < 500ms | End-to-end processing time |
| **Throughput** | > 20 FPS | Real-time video processing |

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Kathir Kalidass** - Project Lead & Implementation
- Research Supervisor: [Name]
- Institution: [University/Organization]

## 🙏 Acknowledgments

- Base paper authors for the attention-based fusion mechanism
- iSign DB team for the ISL dataset
- Reference implementation authors
- MediaPipe team for pose estimation tools

## 📧 Contact

For questions or collaboration:
- GitHub: [@Kathir-Kalidass](https://github.com/Kathir-Kalidass)
- Email: [your.email@example.com]

---

**⭐ Star this repository if you find it helpful!**

*Last Updated: January 30, 2026*
