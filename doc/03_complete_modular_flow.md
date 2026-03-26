# Complete Modular Flow Description

## Document Purpose
This document provides **detailed, box-by-box** descriptions of every module in the system architecture, mapping directly to the architecture diagrams in `report_pages/architecture_diagram/`. Each section explains inputs, processing logic, outputs, and implementation considerations.

---

## Training & Deployment Environments

### Training Pipeline (Google Colab)
**Purpose:** Train and evaluate models with GPU acceleration

**Environment Setup:**
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify GPU
!nvidia-smi

# Install dependencies
!pip install -r application/requirements.txt
```

**Training Features:**
- **GPU Acceleration:** Tesla T4/V100/A100 (CUDA 12.1+)
- **Sliding-Window Training:** Process continuous sequences with 64-frame windows
- **Batch Processing:** Multiple windows per sequence for efficiency
- **Checkpointing:** Auto-save to Google Drive every N epochs
- **Monitoring:** TensorBoard logging (loss, WER, BLEU)

**Sliding-Window Inference:**
```
Continuous Sign Sequence (300 frames)
    ↓
[Window 1: frames 0-63]   → Model → Output 1
[Window 2: frames 32-95]  → Model → Output 2  (50% overlap)
[Window 3: frames 64-127] → Model → Output 3
    ...
    ↓
Merge overlapping predictions → Final gloss sequence
```

**Benefits:**
- No need to buffer entire sequence (memory efficient)
- Faster inference than full-sequence processing
- Handles variable-length sequences naturally

### Deployment Pipeline (Local System)
**Purpose:** Real-time webcam inference with trained model

**Environment Requirements:**
- Python 3.12+ with CPU-only PyTorch
- Webcam (cv2.VideoCapture)
- Downloaded model checkpoint from Colab/Drive
- <500ms latency target

**Why Local Deployment?**
- Google Colab **cannot access local webcam** (browser sandbox restrictions)
- Real-time capture requires direct hardware access
- CPU inference sufficient for sliding-window approach

**Deployment Flow:**
```
Local Webcam → Frame Capture → Preprocessing → 
    ↓
Sliding-Window Buffer (64 frames) → 
    ↓
Trained Model (from Colab) → Gloss Prediction → 
    ↓
Translation → TTS → Output
```

---

## Architecture Diagrams Reference

**Available Diagrams:**
- `sign_archi-Architecture.png` — Overall system
- `sign_archi-Module1.png` — Preprocessing & Feature Extraction
- `sign_archi-Module2.png` — Fusion & Temporal Modeling
- `sign_archi-module3.png` — Decoding & Translation
- `sign_archi-module4.png` — Output Generation

---

## MODULE 1: Video Acquisition & Preprocessing

### 1.1 Input Source Layer

**Components:**
```
ISL Video Stream
    ↓
[Camera/Dataset Selector]
    ↓
[FPS Control] → [Resolution Control] → [Sync Control]
```

**Purpose:** Ensure stable, consistent video input

**Input:**
- Live camera feed (webcam, USB camera)
- OR recorded ISL video file (MP4, AVI)

**Processing:**
1. **Source Selection**
   - If camera: Initialize video capture device
   - If file: Open video file stream
   - Validate source availability

2. **FPS Control**
   - Target FPS: 30 (configurable: 15-60)
   - Frame drop handling if source FPS ≠ target
   - Timestamp synchronization

3. **Resolution Control**
   - Input resolution: Variable
   - Target resolution: 224×224 or 256×256
   - Aspect ratio preservation option

4. **Sync Control**
   - Frame buffer management
   - Timestamp alignment
   - Audio-video sync (if audio present)

**Output:**
- Synchronized video frame stream
- Metadata: FPS, resolution, timestamp per frame

**Implementation Notes:**
- Use OpenCV `cv2.VideoCapture` for reliability
- Buffer size: 5-10 frames for smoothing
- Error handling: Source unavailable → retry or fallback

---

### 1.2 RGB Frame Processing Pipeline

```
Video Decoder
    ↓
Frame Extraction
    ↓
Frame Sampling
    ↓
Frame Resizing
    ↓
RGB Normalization
    ↓
[RGB Frame Tensor]
```

#### 1.2.1 Video Decoder
**Input:** Raw video stream (codec-encoded)  
**Process:** Decode video codec (H.264, MPEG-4) to raw frames  
**Output:** Uncompressed RGB frames  
**Tools:** FFmpeg, OpenCV

#### 1.2.2 Frame Extraction
**Input:** Decoded video stream  
**Process:**
- Extract individual frames sequentially
- Store in memory buffer (FIFO queue)
- Track frame index and timestamp

**Output:** Frame sequence (list of numpy arrays)

#### 1.2.3 Frame Sampling
**Input:** All extracted frames  
**Process:**
- **Uniform sampling:** Every Nth frame (e.g., N=2 for 15 FPS from 30 FPS)
- **Adaptive sampling:** Motion-based (more frames during fast signing)
- **Fixed-length clips:** Pad or trim to target length (e.g., 64 frames)

**Output:** Sampled frame sequence (T frames)

**Hyperparameters:**
- T (clip length): 32, 64, or 128 frames
- Sampling stride: 1 (dense), 2 (half), 4 (sparse)

#### 1.2.4 Frame Resizing
**Input:** Sampled frames (variable H×W)  
**Process:**
- Resize to target dimensions (224×224 or 256×256)
- Interpolation: Bilinear or Bicubic
- Optional: Crop center region to preserve hand/face

**Output:** Uniform-sized frames (T × H × W × 3)

#### 1.2.5 RGB Normalization
**Input:** Resized frames (pixel values 0-255)  
**Process:**
- **Pixel normalization:** Divide by 255 → [0, 1]
- **Mean-std normalization:** (pixel - mean) / std
  - Mean: [0.485, 0.456, 0.406] (ImageNet stats)
  - Std: [0.229, 0.224, 0.225]
- **Optional:** Histogram equalization for lighting correction

**Output:** Normalized RGB tensor (T × 3 × H × W) in PyTorch format

**Implementation:**
```python
# Pseudo-code
frames = load_video(video_path)
frames = sample_frames(frames, target_length=64)
frames = resize_frames(frames, size=(224, 224))
frames = normalize(frames, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
rgb_tensor = torch.tensor(frames).permute(0, 3, 1, 2)  # T×H×W×3 → T×3×H×W
```

---

### 1.3 Pose & Landmark Extraction Pipeline

```
Pose Estimation
    ↓
Hand Landmark Detection
    ↓
Landmark Normalization
    ↓
[Pose & Landmark Tensor]
```

#### 1.3.1 Pose Estimation
**Input:** RGB frames (T × H × W × 3)  
**Process:**
- Detect human body keypoints per frame
- Methods:
  - **MediaPipe Holistic:** 33 body + 21×2 hands + 468 face landmarks
  - **OpenPose:** 25 body + 21×2 hands keypoints
  - **HRNet:** High-resolution pose estimation

**Output:** Body keypoints (T × 33 × 2) in (x, y) coordinates

**Confidence Scores:** Per-keypoint confidence (0-1)

#### 1.3.2 Hand Landmark Detection
**Input:** RGB frames + body keypoints  
**Process:**
- Crop hand regions using body wrist keypoints
- Detect 21 landmarks per hand (knuckles, fingertips)
- Left hand + Right hand

**Output:**
- Left hand keypoints: (T × 21 × 2)
- Right hand keypoints: (T × 21 × 2)

#### 1.3.3 Landmark Normalization
**Input:** Raw keypoints (pixel coordinates)  
**Process:**
1. **Coordinate Normalization:**
   - Center at origin (subtract mean)
   - Scale to unit range (divide by frame dimensions)

2. **Temporal Smoothing:**
   - Apply Gaussian smoothing to reduce jitter
   - Window size: 3-5 frames

3. **Missing Keypoint Handling:**
   - Interpolate from neighboring frames
   - OR use zero vector with confidence=0

**Output:** Normalized pose tensor (T × N_keypoints × 2/3)
- N_keypoints: 33 (body) + 42 (hands) = 75 total
- Dimension: 2 (x,y) or 3 (x,y,confidence)

**Implementation:**
```python
# Pseudo-code
keypoints = []
for frame in rgb_frames:
    result = mediapipe_holistic(frame)
    body_kp = result.pose_landmarks  # 33 points
    left_hand_kp = result.left_hand_landmarks  # 21 points
    right_hand_kp = result.right_hand_landmarks  # 21 points
    keypoints.append(concat([body_kp, left_hand_kp, right_hand_kp]))

keypoints = normalize_keypoints(keypoints)  # Center and scale
pose_tensor = torch.tensor(keypoints)  # T × 75 × 2
```

---

### 1.4 Temporal Standardization

```
Key-Frame Selection
    ↓
Sequence Normalization
    ↓
Frame Count Validation
    ↓
    Valid? ──YES→ [Forward to Module 2]
      │
      NO
      ↓
[Re-capture Request]
```

#### 1.4.1 Key-Frame Selection
**Purpose:** Identify most informative frames

**Methods:**
- **Motion-based:** Select frames with high optical flow
- **Pose-based:** Select frames with distinct hand positions
- **Uniform:** Select evenly spaced frames

**Output:** Subset of frames (reduced T if needed)

#### 1.4.2 Sequence Normalization
**Purpose:** Ensure consistent clip length

**Process:**
- If T < target: Pad with last frame or zeros
- If T > target: Trim or downsample
- Target lengths: 32, 64, 128 frames

**Output:** Fixed-length sequence

#### 1.4.3 Frame Count Validation
**Decision Node:**
```
Is frame count valid?
├─ YES: T within acceptable range → Forward to Module 2
└─ NO: T too short/long or corrupt → Re-capture request
```

**Validation Rules:**
- Minimum T: 16 frames (~0.5 seconds at 30 FPS)
- Maximum T: 256 frames (~8 seconds)
- Corrupt frame detection: All-black or all-white frames

**Error Handling:**
- If re-capture fails 3 times → Log error, skip video

---

### MODULE 1 Summary

**Inputs:**
- Raw ISL video (camera or file)

**Outputs:**
- RGB Frame Tensor: (T × 3 × 224 × 224)
- Pose & Landmark Tensor: (T × 75 × 2)

**Key Parameters:**
- Target FPS: 30
- Frame dimensions: 224×224
- Clip length: 64 frames
- Normalization: ImageNet stats

**Implementation Files:**
- `references/NLA-SLR/gen_pose.py` — Pose extraction
- `TwoStreamNetwork/preprocess/video_processing.py` — Frame processing

---

## MODULE 2: Feature Extraction & Attention-Based Fusion

### 2.1 Preprocessed Input Junction

```
[RGB Frame Tensor] ──┐
                     ├──→ [Feature Extraction]
[Pose Tensor] ───────┘
```

**Purpose:** Synchronize inputs for parallel feature extraction

**Validation:**
- Check tensor shapes match (same T)
- Verify no missing data
- Align timestamps

---

### 2.2 RGB Feature Stream

```
RGB Tensors
    ↓
CNN Backbone (ResNet/I3D)
    ↓
Feature Maps Extraction
    ↓
Global Pooling
    ↓
[RGB Feature Vector]
```

#### 2.2.1 CNN Backbone
**Architecture Options:**

**A. ResNet-18/50 (2D CNN)**
- Pretrained on ImageNet
- Extract per-frame features
- Fast inference
- Feature dim: 512 (ResNet-18), 2048 (ResNet-50)

**B. I3D / C3D (3D CNN)**
- Spatiotemporal convolutions
- Pretrained on Kinetics-400
- Captures motion directly
- Feature dim: 1024

**C. SlowFast Networks**
- Dual pathway (slow + fast)
- State-of-the-art for video
- Higher computational cost

**Recommendation:** ResNet-18 for real-time, I3D for accuracy

#### 2.2.2 Feature Maps Extraction
**Process:**
- Forward pass through CNN
- Extract intermediate layer activations
- For ResNet: Use `layer4` output (before FC layer)
- Shape: (T × C × H' × W') where H', W' are reduced dimensions

#### 2.2.3 Global Pooling
**Methods:**
- **Global Average Pooling (GAP):** Average over spatial dims
- **Global Max Pooling:** Max over spatial dims
- **Adaptive Pooling:** Fixed output size

**Output:** RGB feature vector per frame (T × D_rgb)
- D_rgb: 512 (ResNet-18), 2048 (ResNet-50)

**Implementation:**
```python
# Pseudo-code
class RGBStream(nn.Module):
    def __init__(self):
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove classification head
    
    def forward(self, rgb_tensor):  # T×3×224×224
        features = []
        for frame in rgb_tensor:
            feat = self.backbone(frame.unsqueeze(0))  # 1×512
            features.append(feat)
        return torch.stack(features, dim=0)  # T×512
```

---

### 2.3 Pose Feature Stream

```
Pose Tensors
    ↓
Pose Encoder (GCN/MLP/RNN)
    ↓
Temporal Pose Encoding
    ↓
[Pose Feature Vector]
```

#### 2.3.1 Pose Encoder Options

**A. Spatial-Temporal Graph Convolutional Network (ST-GCN)**
- Models skeletal connectivity as graph
- Spatial conv: Within-frame joint relationships
- Temporal conv: Across-frame joint trajectories
- Best for capturing pose dynamics

**B. Multi-Layer Perceptron (MLP)**
- Flatten keypoints per frame
- Pass through fully connected layers
- Simple and fast
- Less expressive than GCN

**C. Recurrent Neural Network (RNN/LSTM)**
- Process keypoint sequence temporally
- Captures motion trajectories
- Good for gesture dynamics

**Recommendation:** ST-GCN for accuracy, MLP for speed

#### 2.3.2 Temporal Pose Encoding
**Process:**
- Encode keypoint sequences over time
- Learn motion patterns (velocity, acceleration)
- Output: Pose embeddings per frame

**Output:** Pose feature vector (T × D_pose)
- D_pose: 256-512

**Implementation:**
```python
# Pseudo-code (MLP encoder)
class PoseStream(nn.Module):
    def __init__(self, num_keypoints=75, feat_dim=256):
        self.mlp = nn.Sequential(
            nn.Linear(num_keypoints * 2, 512),
            nn.ReLU(),
            nn.Linear(512, feat_dim)
        )
    
    def forward(self, pose_tensor):  # T×75×2
        pose_flat = pose_tensor.reshape(pose_tensor.size(0), -1)  # T×150
        features = self.mlp(pose_flat)  # T×256
        return features
```

---

### 2.4 Attention & Fusion Unit

```
[RGB Features (T×D_rgb)] ──┐
                           ├──→ Attention Input Merge
[Pose Features (T×D_pose)]─┘
                ↓
    Attention Input Generator
                ↓
    Feature Importance Validation
                ↓
          Valid? ──YES→ Feature Weighting
            │                   ↓
            NO              Feature Fusion
            ↓                   ↓
    Weight Recalibration ───→ [Fused Features]
```

#### 2.4.1 Attention Input Merge
**Process:**
- Concatenate or stack RGB and pose features
- Align temporal dimensions (ensure same T)

**Intermediate:** (T × (D_rgb + D_pose)) or two separate streams

#### 2.4.2 Attention Input Generator
**Purpose:** Compute attention scores for each modality

**Method (from Base Paper):**
1. **Temporal Self-Attention (per modality):**
   - Query, Key, Value projections
   - Multi-head attention (8 heads)
   - Identifies important time steps

2. **Cross-Modal Attention:**
   - Compute similarity between RGB and Pose
   - Learn adaptive weights α(t), β(t)

**Formula:**
```
α(t) = softmax(W_rgb · [F_rgb(t) || F_pose(t)])
β(t) = softmax(W_pose · [F_rgb(t) || F_pose(t)])
```

Where || denotes concatenation, W are learned weight matrices.

#### 2.4.3 Feature Importance Validation
**Decision Node:**
```
Are attention weights valid?
├─ YES: Weights sum to 1, no NaN → Proceed to weighting
└─ NO: Invalid weights → Recalibration
```

**Validation Checks:**
- α(t) + β(t) ≈ 1 (normalized)
- No NaN or Inf values
- Weights in reasonable range (e.g., [0.1, 0.9])

#### 2.4.4 Weight Recalibration
**If validation fails:**
- Reset to uniform weights (α=β=0.5)
- OR use exponential moving average of past weights
- Log warning for debugging

#### 2.4.5 Feature Fusion
**Process:**
```
F_fused(t) = α(t) × F_rgb(t) + β(t) × F_pose(t)
```

**Alternative (Concatenation + MLP):**
```
F_concat(t) = [F_rgb(t) || F_pose(t)]
F_fused(t) = MLP(F_concat(t))
```

**Output:** Fused feature representation (T × D_fusion)
- D_fusion: 512-1024

**Implementation:**
```python
# Pseudo-code (attention-based fusion)
class AttentionFusion(nn.Module):
    def __init__(self, d_rgb, d_pose, d_fusion):
        self.attn_rgb = nn.Linear(d_rgb + d_pose, 1)
        self.attn_pose = nn.Linear(d_rgb + d_pose, 1)
        self.proj = nn.Linear(d_rgb + d_pose, d_fusion)
    
    def forward(self, rgb_feat, pose_feat):  # T×D_rgb, T×D_pose
        concat = torch.cat([rgb_feat, pose_feat], dim=-1)  # T×(D_rgb+D_pose)
        
        # Compute attention weights
        alpha = torch.sigmoid(self.attn_rgb(concat))  # T×1
        beta = torch.sigmoid(self.attn_pose(concat))   # T×1
        
        # Normalize
        total = alpha + beta
        alpha = alpha / total
        beta = beta / total
        
        # Weighted fusion
        fused = alpha * rgb_feat + beta * pose_feat  # Broadcasting
        fused = self.proj(torch.cat([rgb_feat, pose_feat], dim=-1))
        return fused  # T×D_fusion
```

---

### MODULE 2 Summary

**Inputs:**
- RGB Frame Tensor: (T × 3 × 224 × 224)
- Pose Tensor: (T × 75 × 2)

**Outputs:**
- Fused Feature Representation: (T × D_fusion)

**Key Components:**
- CNN Backbone: ResNet-18/50 or I3D
- Pose Encoder: ST-GCN or MLP
- Attention Fusion: Adaptive weighting (base paper contribution)

**Parameters:**
- D_rgb: 512 (ResNet-18), 2048 (ResNet-50)
- D_pose: 256-512
- D_fusion: 512-1024
- Attention heads: 8

**Implementation Files:**
- `TwoStreamNetwork/modelling/rgb_stream.py`
- `TwoStreamNetwork/modelling/pose_stream.py`
- `TwoStreamNetwork/modelling/fusion.py`

---

## MODULE 3: Continuous Sign Recognition

### 3.1 Feature Input Interface

```
[Fused Features (T×D_fusion)]
    ↓
[Temporal Modeling]
```

**Validation:** Check sequence length T within limits (16-256)

---

### 3.2 Temporal Modeling

```
Fused Features
    ↓
┌────────────────────────────┐
│  BiLSTM Encoder            │
│  OR                        │
│  Transformer Encoder       │
└────────────┬───────────────┘
             ↓
[Temporal Feature Maps]
```

#### 3.2.1 BiLSTM Encoder
**Architecture:**
- 2-4 layers of Bidirectional LSTM
- Hidden size: 512-1024
- Dropout: 0.2-0.3

**Process:**
- Input: Fused features (T × D_fusion)
- Forward LSTM: Processes t=0 to t=T-1
- Backward LSTM: Processes t=T-1 to t=0
- Concatenate forward + backward hidden states

**Output:** Temporal features (T × 2×hidden_size)

**Advantages:**
- Captures bidirectional context
- Lower latency than Transformer
- Proven for sequence tasks

**Implementation:**
```python
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=True, dropout=0.3)
    
    def forward(self, fused_feat):  # T×D_fusion
        output, (h_n, c_n) = self.lstm(fused_feat.unsqueeze(0))  # 1×T×(2*hidden)
        return output.squeeze(0)  # T×(2*hidden)
```

#### 3.2.2 Transformer Encoder
**Architecture:**
- 6-8 Transformer encoder layers
- Hidden size (d_model): 512
- Attention heads: 8
- FFN dimension: 2048
- Dropout: 0.1

**Process:**
- Input: Fused features + positional encoding
- Multi-head self-attention (capture dependencies)
- Feed-forward network (non-linearity)
- Layer norm + residual connections

**Output:** Contextualized embeddings (T × d_model)

**Advantages:**
- Better long-range dependencies
- Parallel processing (faster training)
- State-of-the-art accuracy

**Implementation:**
```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, fused_feat):  # T×D_fusion
        x = self.pos_encoder(fused_feat)  # Add positional info
        x = self.transformer(x)  # T×d_model
        return x
```

---

### 3.3 CTC Alignment Layer

```
Temporal Features
    ↓
[CTC Layer]
    ↓
[Gloss Probability Distribution (T × |Vocab|)]
```

#### Purpose
- Align frame-level predictions to gloss-level labels
- Handle variable-length sequences
- No need for frame-level annotations

#### CTC Layer Architecture
**Components:**
- Linear layer: (d_model → |Vocab| + 1)
  - |Vocab|: Number of glosses (e.g., 1000)
  - +1: Blank token for CTC

**Process:**
1. Input: Temporal features (T × d_model)
2. Linear projection: (T × |Vocab| + 1)
3. Log-softmax: Probability distribution per frame

**Output:** Log probabilities (T × |Vocab| + 1)

#### CTC Loss (Training)
**Formula:**
```
L_CTC = -log P(Y | X)
```

Where:
- X: Input sequence (video frames)
- Y: Target gloss sequence

**CTC Properties:**
- Allows repetitions: "H-H-E-L-L-O" → "HELLO"
- Blank token (-): Handles transitions
- Example alignment:
  - Target: "HELLO WORLD"
  - CTC path: "HH-EE-LL-LL-OO- -WW-OO-RR-LL-DD"

**Implementation:**
```python
class CTCLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.fc = nn.Linear(d_model, vocab_size + 1)  # +1 for blank
    
    def forward(self, temporal_feat):  # T×d_model
        logits = self.fc(temporal_feat)  # T×(vocab+1)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

# Training loss
def ctc_loss(log_probs, target_glosses, input_lengths, target_lengths):
    return nn.CTCLoss()(log_probs, target_glosses, input_lengths, target_lengths)
```

---

### 3.4 Decoding Strategy

```
CTC Output (Log Probs)
    ↓
Decoder Strategy Selection
    ↓
┌──────────────┬──────────────┐
│ Greedy       │ Beam Search  │
└──────┬───────┴──────┬───────┘
       ↓              ↓
[Gloss Sequence]
```

#### 3.4.1 Greedy Decoding (Fast)
**Process:**
1. Select argmax gloss per frame: `g(t) = argmax P(g|t)`
2. Collapse consecutive duplicates: "HHH-EEE-LLL" → "HEL"
3. Remove blank tokens: "H-E-L-L-O" → "HELLO"

**Advantages:**
- Very fast (no search)
- Low latency (<10ms)

**Disadvantages:**
- Suboptimal (may miss best path)
- No language model integration

**Implementation:**
```python
def greedy_decode(log_probs):  # T×(vocab+1)
    # Select best token per frame
    preds = torch.argmax(log_probs, dim=-1)  # T
    
    # Collapse repetitions
    collapsed = [preds[0]]
    for i in range(1, len(preds)):
        if preds[i] != preds[i-1]:
            collapsed.append(preds[i])
    
    # Remove blanks (assume blank_id=0)
    glosses = [g for g in collapsed if g != 0]
    return glosses
```

#### 3.4.2 Beam Search Decoding (Accurate)
**Process:**
1. Maintain top-K hypotheses (beams) at each step
2. Expand each hypothesis with possible next tokens
3. Score using: CTC prob + Language Model prob
4. Keep top-K after each step
5. Return best hypothesis after full sequence

**Parameters:**
- Beam width (K): 5-10
- Language model weight (λ): 0.1-0.5
- Length penalty (α): Optional

**Advantages:**
- More accurate (+2-4% WER reduction)
- Can integrate language model
- Better handling of ambiguous frames

**Disadvantages:**
- Slower (50-200ms)
- Higher memory usage

**Implementation:**
```python
def beam_search_decode(log_probs, beam_width=5, lm_weight=0.3):
    # Initialize beam
    beams = [([], 0.0)]  # (path, score)
    
    for t in range(log_probs.size(0)):  # For each frame
        new_beams = []
        for path, score in beams:
            # Expand with all possible tokens
            for token in range(log_probs.size(1)):
                new_path = path + [token]
                new_score = score + log_probs[t, token].item()
                
                # Optional: Add language model score
                if lm_weight > 0 and len(new_path) > 1:
                    lm_score = language_model(new_path[-2], new_path[-1])
                    new_score += lm_weight * lm_score
                
                new_beams.append((new_path, new_score))
        
        # Keep top-K beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
    
    # Collapse best beam
    best_path, best_score = beams[0]
    glosses = collapse_and_remove_blanks(best_path)
    return glosses
```

---

### 3.5 Output Validation

```
Decoded Gloss Sequence
    ↓
Confidence Validation
    ↓
Valid? ──YES→ [Final ISL Gloss Sequence]
  │
  NO
  ↓
[Re-decode / Error Flag]
```

#### Confidence Validation Rules
**Checks:**
1. **Sequence Length:** 1 ≤ length ≤ 50 (reasonable sentence)
2. **Average Confidence:** Mean log-prob > threshold (e.g., -2.0)
3. **No All-Blank:** At least one non-blank token
4. **Vocabulary Check:** All tokens in valid gloss vocabulary

**Actions:**
- If valid → Forward to Module 4
- If invalid:
  - Try beam search (if greedy was used)
  - Reduce confidence threshold
  - OR flag as uncertain output

**Implementation:**
```python
def validate_output(glosses, log_probs, threshold=-2.0):
    if len(glosses) == 0:
        return False, "Empty sequence"
    
    if len(glosses) > 50:
        return False, "Too long"
    
    avg_conf = log_probs.mean().item()
    if avg_conf < threshold:
        return False, f"Low confidence: {avg_conf}"
    
    return True, "Valid"
```

---

### MODULE 3 Summary

**Inputs:**
- Fused Feature Representation: (T × D_fusion)

**Outputs:**
- Final ISL Gloss Sequence: List of gloss tokens

**Key Components:**
- Temporal Encoder: BiLSTM or Transformer
- CTC Layer: Frame-to-gloss alignment
- Decoder: Greedy (fast) or Beam Search (accurate)

**Parameters:**
- Hidden size: 512-1024 (BiLSTM), 512 (Transformer)
- Num layers: 2-4 (BiLSTM), 6-8 (Transformer)
- Vocab size: 500-2000 (ISL glosses)
- Beam width: 5-10
- LM weight: 0.1-0.5

**Implementation Files:**
- `references/NLA-SLR/modelling/sequence_model.py`
- `TwoStreamNetwork/modelling/temporal_model.py`
- `Online/CTC_fusion/ctc_decoder.py`

---

## MODULE 4: Language Processing & Audio Output

### 4.1 Token Input Interface

```
[ISL Gloss Sequence]
    ↓
[Token Junction]
```

**Process:**
- Receive gloss tokens from Module 3
- Synchronize token stream
- Handle partial predictions (real-time mode)

---

### 4.2 Caption Buffer & Text Handling

```
Token Junction
    ↓
Token Accumulator
    ↓
Duplicate Filter
    ↓
Temporal Ordering
    ↓
Caption Merge Junction
    ↓
[Buffered Caption]
```

#### 4.2.1 Token Accumulator
**Purpose:** Collect gloss tokens over sliding window

**Process:**
- Maintain buffer of recent tokens (e.g., last 10 glosses)
- In real-time mode: Accumulate tokens as they arrive
- In offline mode: Process full gloss sequence

**Buffer Size:** 5-15 tokens (adjustable)

#### 4.2.2 Duplicate Filter
**Purpose:** Remove redundant gloss tokens

**Rules:**
- Remove consecutive duplicates: "HELLO HELLO" → "HELLO"
- Keep repetitions if separated by other tokens
- Preserve intentional repetitions (if marked in data)

**Implementation:**
```python
def filter_duplicates(glosses):
    if not glosses:
        return []
    
    filtered = [glosses[0]]
    for i in range(1, len(glosses)):
        if glosses[i] != glosses[i-1]:
            filtered.append(glosses[i])
    return filtered
```

#### 4.2.3 Temporal Ordering
**Purpose:** Ensure correct gloss sequence order

**Process:**
- Sort by timestamp (if available)
- Handle out-of-order arrivals (real-time mode)
- Reorder based on context if needed

#### 4.2.4 Caption Merge Junction
**Purpose:** Combine partial captions into complete sentences

**Process:**
- Detect sentence boundaries (e.g., pause in signing)
- Merge glosses into sentence-level sequences
- Add punctuation hints (optional)

**Output:** Sentence-level gloss sequence

---

### 4.3 AI-Based Sentence Correction

```
Buffered Caption
    ↓
Sentence Encoder
    ↓
Language Model Inference
    ↓
Sentence Refinement
    ↓
Sentence Quality Validation
    ↓
Valid? ──YES→ [Final Caption]
  │
  NO
  ↓
[Rephrase Request Loop]
```

#### 4.3.1 Sentence Encoder
**Purpose:** Convert gloss sequence to embeddings

**Architecture:**
- Embedding layer: Gloss → Vector (d=256)
- Positional encoding (if Transformer)
- Encode full gloss sequence

**Output:** Gloss embeddings (N × d)

#### 4.3.2 Language Model Inference
**Options:**

**A. Seq2Seq LSTM**
- Encoder: LSTM on gloss sequence
- Decoder: LSTM generates English tokens
- Attention mechanism between encoder-decoder

**B. Transformer (T5, BART)**
- Pretrained on large text corpus
- Fine-tuned on ISL gloss → English pairs
- Prompt: "Translate ISL: HELLO MY NAME JOHN"

**C. Rule-Based + Template**
- Simple gloss-to-text mappings
- Template filling for common patterns
- Fast but limited coverage

**Recommendation:** Fine-tuned T5 for best results

**Implementation:**
```python
# Pseudo-code (T5-based)
from transformers import T5ForConditionalGeneration, T5Tokenizer

class GlossToTextTranslator:
    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        # Fine-tune on ISL gloss-English pairs
    
    def translate(self, gloss_sequence):
        # Format as prompt
        prompt = "Translate ISL to English: " + " ".join(gloss_sequence)
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
        
        # Generate
        outputs = self.model.generate(input_ids, max_length=50)
        english_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return english_text
```

#### 4.3.3 Sentence Refinement
**Purpose:** Improve grammar and fluency

**Methods:**

**A. Grammar Correction Model**
- Use LanguageTool or Grammarly API
- OR fine-tuned BART on grammar correction

**B. Rule-Based Post-Processing**
- Fix common ISL→English patterns:
  - "ME BOOK READ" → "I read a book"
  - "YESTERDAY I GO SCHOOL" → "Yesterday I went to school"
- Add articles (a, an, the)
- Fix verb tenses

**C. Fluency Enhancement**
- Paraphrase model for naturalness
- Remove awkward phrasings

**Implementation:**
```python
def refine_sentence(raw_text):
    # Rule-based corrections
    text = raw_text.replace("ME ", "I ")
    text = add_articles(text)
    text = fix_verb_tense(text)
    
    # Grammar correction (optional)
    text = grammar_corrector(text)
    
    return text
```

#### 4.3.4 Sentence Quality Validation
**Decision Node:**
```
Is sentence valid?
├─ YES: Grammatical, fluent → Final caption
└─ NO: Poor grammar, nonsensical → Rephrase request
```

**Validation Checks:**
1. **Grammar Check:** No major errors
2. **Semantic Check:** Makes sense (not gibberish)
3. **Length Check:** 3-100 words
4. **Profanity Filter:** (optional)

**Rephrase Request Loop:**
- If validation fails, retry translation with:
  - Different decoding strategy
  - Adjusted gloss sequence
  - Fallback to simpler model
- Maximum retries: 2-3

---

### 4.4 Output Generation

```
Final Caption
    ├────→ [Caption Display]
    └────→ [Text-to-Speech Engine]
                ↓
          [Speech Audio Output]
```

#### 4.4.1 Caption Display
**Process:**
- Render text on UI (console, web app, mobile app)
- Optional: Add timestamp, confidence score
- Optional: Show intermediate gloss sequence

**UI Options:**
- **Console:** Print to terminal (simple)
- **Web App:** Display in browser (Flask/FastAPI)
- **Mobile App:** Android/iOS native UI

#### 4.4.2 Text-to-Speech Engine
**Options:**

**A. Offline TTS**
- **gTTS (Google TTS):** Text → MP3, requires internet
- **pyttsx3:** Offline, cross-platform, basic voices
- Latency: 50-200ms

**B. Cloud TTS**
- **Google Cloud TTS:** Neural voices, 40+ languages
- **AWS Polly:** High-quality, streaming support
- **Azure Speech Service:** Real-time synthesis
- Latency: 200-500ms

**C. Neural TTS**
- **Tacotron 2 + WaveGlow:** End-to-end neural
- **VITS:** State-of-the-art quality
- Latency: 500ms-2s (GPU required)

**Recommendation:**
- Real-time: gTTS or pyttsx3
- Production: Google Cloud TTS
- Offline high-quality: Neural TTS (if GPU available)

#### 4.4.3 Speech Audio Output
**Process:**
- TTS engine generates audio waveform
- Play audio through speaker
- Optional: Save to file (WAV/MP3)

**Implementation:**
```python
# Pseudo-code (gTTS)
from gtts import gTTS
import pygame

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    
    # Play audio
    pygame.mixer.init()
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()
```

---

### MODULE 4 Summary

**Inputs:**
- ISL Gloss Sequence: List of gloss tokens

**Outputs:**
- English Text: Grammatically correct sentence
- Speech Audio: Synthesized audio waveform

**Key Components:**
- Caption Buffer: Token accumulation and filtering
- Gloss-to-Text Translator: Seq2Seq or T5-based
- Grammar Correction: Rule-based + LM-based
- TTS Engine: gTTS, Cloud TTS, or Neural TTS

**Parameters:**
- Buffer size: 5-15 tokens
- Translation model: T5-small fine-tuned on ISL
- TTS engine: gTTS (default), configurable
- Max rephrase attempts: 2-3

**Implementation Files:**
- `Online/CTC_fusion/buffer.py`
- `Online/SLT/translator.py`
- `Online/SLT/grammar_correction.py`
- `Online/tts_module.py`

---

## Cross-Module Data Flow Summary

**Complete Pipeline:**
```
ISL Video
  → Module 1: RGB (T×3×224×224) + Pose (T×75×2)
  → Module 2: Fused Features (T×512)
  → Module 3: Gloss Sequence ["HELLO", "MY", "NAME", "JOHN"]
  → Module 4: English Text "Hello, my name is John." + Audio
```

**Latency Breakdown (Target <500ms):**
| Module | Component | Latency |
|--------|-----------|---------|
| Module 1 | Frame capture + preprocessing | 30-50ms |
| Module 1 | Pose estimation | 20-40ms |
| Module 2 | CNN feature extraction | 50-100ms |
| Module 2 | Pose encoding + fusion | 10-20ms |
| Module 3 | Temporal modeling | 50-100ms |
| Module 3 | CTC + decoding | 20-50ms |
| Module 4 | Translation + correction | 50-100ms |
| Module 4 | TTS | 50-200ms |
| **Total** | **End-to-end** | **280-660ms** |

**Optimization for <500ms:**
- Use ResNet-18 (not ResNet-50)
- Use BiLSTM (not Transformer)
- Use greedy decoding (not beam search)
- Use gTTS (not neural TTS)
- Asynchronous processing (parallel stages)

---

## Related Documents

**For architectural overview:** [02_architecture_overview.md](02_architecture_overview.md)  
**For algorithmic details:** [04_algorithmic_design.md](04_algorithmic_design.md)  
**For model specifications:** [05_model_architecture_details.md](05_model_architecture_details.md)  
**For training:** [06_training_pipeline.md](06_training_pipeline.md)  
**For evaluation:** [07_evaluation_metrics.md](07_evaluation_metrics.md)

---

**Document Version:** 1.0  
**Last Updated:** January 24, 2026  
**Purpose:** Detailed modular flow descriptions
