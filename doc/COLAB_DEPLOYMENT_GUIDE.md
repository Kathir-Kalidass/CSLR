# Google Colab Training & Local Deployment Guide

## Overview

This guide explains the complete training and deployment workflow:
- **Training:** Google Colab with GPU acceleration
- **Deployment:** Local system with webcam access

---

## Part 1: Training on Google Colab

### Why Google Colab?
- ✅ Free GPU access (Tesla T4/V100/A100)
- ✅ Pre-installed PyTorch and CUDA
- ✅ No local hardware requirements
- ✅ Easy dataset sharing via Google Drive
- ✅ Collaborative notebook environment

### Setup Colab Environment

#### Step 1: Create Training Notebook

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Clone repository
!git clone https://github.com/Kathir-Kalidass/CLSR.git
%cd CLSR

# Cell 3: Verify GPU
!nvidia-smi
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 4: Install dependencies
!pip install -r application/requirements.txt

# Cell 5: Setup paths
import sys
sys.path.append('/content/CLSR')
```

#### Step 2: Prepare Dataset on Google Drive

**Option A: Upload to Google Drive**
```
Google Drive/
└── CSLR_Datasets/
    ├── iSign_DB/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── MS-ASL/
    └── WLASL/
```

**Option B: Download directly in Colab**
```python
# Download iSign DB
!wget -O /content/drive/MyDrive/CSLR_Datasets/iSign_DB.zip [DATASET_URL]
!unzip /content/drive/MyDrive/CSLR_Datasets/iSign_DB.zip -d /content/drive/MyDrive/CSLR_Datasets/
```

#### Step 3: Configure Training

```python
# config/train_config.yaml
model:
  rgb_encoder: "resnet18"
  pose_encoder: "gcn"
  temporal_model: "bilstm"
  num_classes: 1000
  attention: true

training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  sliding_window: true
  window_size: 64
  window_overlap: 0.5
  
paths:
  dataset: "/content/drive/MyDrive/CSLR_Datasets/iSign_DB"
  checkpoints: "/content/drive/MyDrive/CSLR_Checkpoints"
  logs: "/content/drive/MyDrive/CSLR_Logs"

gpu:
  device: "cuda:0"
  mixed_precision: true
```

### Training with Sliding-Window Approach

#### What is Sliding-Window Inference?

```
Continuous Sign Sequence: 300 frames
├── Window 1: [frames 0-63]     → Process → Output segment 1
├── Window 2: [frames 32-95]    → Process → Output segment 2  (50% overlap)
├── Window 3: [frames 64-127]   → Process → Output segment 3
├── Window 4: [frames 96-159]   → Process → Output segment 4
└── ...

Merge overlapping predictions → Final continuous gloss sequence
```

**Benefits:**
- ✅ Memory efficient (no need to buffer entire sequence)
- ✅ Faster inference (parallel processing possible)
- ✅ Handles variable-length sequences
- ✅ Reduces latency for real-time processing

#### Training Code

```python
# train.py
from torch.utils.data import DataLoader
from models.dual_stream import DualStreamAttentionModel
from utils.sliding_window import SlidingWindowDataset
import torch
import wandb

# Initialize model
model = DualStreamAttentionModel(
    rgb_encoder='resnet18',
    pose_encoder='gcn',
    num_classes=1000,
    attention=True
).cuda()

# Setup sliding-window dataset
train_dataset = SlidingWindowDataset(
    data_path='/content/drive/MyDrive/CSLR_Datasets/iSign_DB/train',
    window_size=64,
    window_stride=32,  # 50% overlap
    augment=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CTCLoss()

for epoch in range(50):
    model.train()
    for batch_idx, (rgb_windows, pose_windows, targets, lengths) in enumerate(train_loader):
        rgb_windows = rgb_windows.cuda()
        pose_windows = pose_windows.cuda()
        targets = targets.cuda()
        
        # Forward pass
        outputs = model(rgb_windows, pose_windows)
        loss = criterion(outputs, targets, lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Save checkpoint to Google Drive
    if epoch % 5 == 0:
        torch.save(
            model.state_dict(),
            f'/content/drive/MyDrive/CSLR_Checkpoints/model_epoch_{epoch}.pth'
        )
```

#### Run Training

```python
# In Colab notebook cell
!python train.py --config config/train_config.yaml --gpu 0
```

### Monitor Training

```python
# TensorBoard
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/CSLR_Logs

# Weights & Biases (optional)
import wandb
wandb.init(project="cslr-isl")
wandb.watch(model)
```

### Evaluation on Colab

```python
# evaluate.py
from utils.metrics import compute_wer, compute_bleu

model.eval()
total_wer = 0
total_bleu = 0

with torch.no_grad():
    for rgb_windows, pose_windows, ground_truth in test_loader:
        rgb_windows = rgb_windows.cuda()
        pose_windows = pose_windows.cuda()
        
        # Sliding-window inference
        predictions = model.sliding_inference(rgb_windows, pose_windows, window_size=64)
        
        # Compute metrics
        wer = compute_wer(predictions, ground_truth)
        bleu = compute_bleu(predictions, ground_truth)
        
        total_wer += wer
        total_bleu += bleu

print(f"Average WER: {total_wer / len(test_loader):.2%}")
print(f"Average BLEU: {total_bleu / len(test_loader):.2f}")
```

---

## Part 2: Local Deployment (Webcam)

### Why Local Deployment?

**Google Colab Limitations:**
- ❌ Cannot access local webcam (browser sandbox)
- ❌ No direct hardware access
- ❌ Session timeouts (12-hour limit)

**Local System Advantages:**
- ✅ Direct webcam access
- ✅ No session limits
- ✅ CPU inference sufficient (<500ms latency)
- ✅ Offline operation

### Setup Local Environment

#### Step 1: Install Dependencies

```bash
# Clone repository (if not already done)
git clone https://github.com/Kathir-Kalidass/CLSR.git
cd CLSR

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install CPU-only PyTorch (lighter, faster)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r application/requirements.txt
```

#### Step 2: Download Trained Model

**From Google Drive:**
1. Navigate to `/content/drive/MyDrive/CSLR_Checkpoints/`
2. Download `best_model.pth`
3. Place in local `checkpoints/` directory

**Or download via command line:**
```bash
# Install gdown
pip install gdown

# Download from Google Drive (get shareable link)
gdown https://drive.google.com/uc?id=YOUR_MODEL_FILE_ID -O checkpoints/best_model.pth
```

#### Step 3: Verify Setup

```bash
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import mediapipe; print('MediaPipe: OK')"

# Test webcam
python -c "import cv2; cap = cv2.VideoCapture(0); ret, _ = cap.read(); print(f'Webcam: {'OK' if ret else 'FAIL'}'); cap.release()"
```

### Real-Time Webcam Inference

#### Inference Code

```python
# application/realtime_inference.py
import cv2
import torch
import numpy as np
from collections import deque
from models.dual_stream import DualStreamAttentionModel
from utils.preprocessing import preprocess_frame, extract_pose
from utils.translation import translate_glosses

# Load trained model
model = DualStreamAttentionModel(num_classes=1000)
model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location='cpu'))
model.eval()

# Sliding window buffer
window_size = 64
window_stride = 32
frame_buffer = deque(maxlen=window_size)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Starting real-time ISL recognition...")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    rgb_frame = preprocess_frame(frame)
    pose_keypoints = extract_pose(frame)
    
    # Add to buffer
    frame_buffer.append({
        'rgb': rgb_frame,
        'pose': pose_keypoints
    })
    
    # Process when buffer is full
    if len(frame_buffer) == window_size:
        # Prepare input tensors
        rgb_tensor = torch.stack([f['rgb'] for f in frame_buffer]).unsqueeze(0)
        pose_tensor = torch.stack([f['pose'] for f in frame_buffer]).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            gloss_logits = model(rgb_tensor, pose_tensor)
            predicted_glosses = decode_ctc(gloss_logits)
        
        # Translate to English
        english_text = translate_glosses(predicted_glosses)
        
        # Display results
        cv2.putText(frame, f"Glosses: {predicted_glosses}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"English: {english_text}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Slide window (remove old frames)
        for _ in range(window_stride):
            frame_buffer.popleft()
    
    # Show frame
    cv2.imshow('ISL Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### Run Inference

```bash
python application/realtime_inference.py --camera 0 --model checkpoints/best_model.pth
```

### Performance Optimization

#### CPU Optimization

```python
# Enable CPU optimizations
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

# Use INT8 quantization for faster inference
from torch.quantization import quantize_dynamic
model_quantized = quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
)
```

#### Reduce Latency

```python
# Smaller window size (trade accuracy for speed)
window_size = 32  # Instead of 64

# Larger stride (less overlap)
window_stride = 32  # Same as window_size (no overlap)

# Lower webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

---

## Part 3: Complete Workflow Summary

### Training Workflow (Google Colab)

```
1. Setup Colab → Mount Drive → Verify GPU
2. Clone repo → Install dependencies
3. Prepare dataset on Drive
4. Configure training (sliding-window enabled)
5. Train model with GPU acceleration
6. Monitor with TensorBoard/W&B
7. Evaluate on test set (WER, BLEU)
8. Save best checkpoint to Drive
```

### Deployment Workflow (Local)

```
1. Setup local environment → Install CPU PyTorch
2. Download trained model from Drive
3. Verify webcam access
4. Run real-time inference script
5. Process frames with sliding-window
6. Display glosses + English translation
7. Optional: Add TTS for audio output
```

### Data Flow

```
[TRAINING: Google Colab]
iSign DB (Drive) → Sliding-Window Dataset → 
GPU Training → Model Checkpoint → Save to Drive

[DEPLOYMENT: Local System]
Download Checkpoint → Load Model (CPU) → 
Webcam Capture → Sliding-Window Buffer → 
CPU Inference → Display Results
```

---

## Troubleshooting

### Colab Issues

**GPU Out of Memory:**
```python
# Reduce batch size
batch_size = 8  # Instead of 16

# Enable gradient accumulation
accumulation_steps = 4
```

**Session Timeout:**
```python
# Keep session alive
from IPython.display import Javascript
Javascript('setInterval(() => { document.querySelector("colab-toolbar-button#connect").click() }, 60000)')
```

### Local Issues

**Webcam Not Found:**
```bash
# List available cameras
ls /dev/video*

# Try different camera index
python application/realtime_inference.py --camera 1
```

**Slow Inference:**
```python
# Profile bottlenecks
import time
start = time.time()
output = model(rgb, pose)
print(f"Inference time: {time.time() - start:.3f}s")

# Use quantization (see CPU Optimization above)
```

---

## Best Practices

### Training
- ✅ Use mixed precision training (`torch.cuda.amp`)
- ✅ Save checkpoints regularly to Drive
- ✅ Log metrics to TensorBoard/W&B
- ✅ Use sliding-window for memory efficiency
- ✅ Validate on ISL test set, not just ASL

### Deployment
- ✅ Use CPU-only PyTorch (lighter)
- ✅ Quantize model for faster inference
- ✅ Test with different window sizes
- ✅ Add error handling for webcam failures
- ✅ Display FPS and latency metrics

---

## Example Notebooks

### Training Notebook (Colab)
```
notebooks/
└── train_isl_colab.ipynb
```

### Inference Script (Local)
```
application/
├── realtime_inference.py
├── video_inference.py
└── batch_inference.py
```

---

## Performance Targets

| Metric | Target | Achieved (After Training) |
|--------|--------|---------------------------|
| **WER** | <20% | _(Fill after training)_ |
| **BLEU** | >30 | _(Fill after training)_ |
| **Latency** | <500ms | _(Fill after deployment)_ |
| **FPS** | >20 | _(Fill after deployment)_ |

---

## References

- [Google Colab Documentation](https://colab.research.google.com/)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)

---

*Last Updated: January 30, 2026*
