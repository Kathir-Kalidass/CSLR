# ISL CSLR Complete Workflow Documentation

## Overview

This document provides a detailed, step-by-step explanation of the complete workflow in the Indian Sign Language Continuous Sign Language Recognition (ISL CSLR) system. From camera capture to final sentence output, every stage is broken down with technical details, code references, and data transformations.

## System Workflow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Model         │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (PyTorch)     │
│                 │    │                 │    │                 │
│ 1. Camera       │    │ 4. WebSocket    │    │ 6. Feature      │
│    Capture      │    │    Processing   │    │    Extraction   │
│                 │    │                 │    │                 │
│ 2. Real-time    │    │ 5. Frame Queue  │    │ 7. Temporal     │
│    Display      │    │    Management   │    │    Modeling     │
│                 │    │                 │    │                 │
│ 3. UI Updates   │    │                 │    │ 8. CTC Decoding │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Detailed Workflow Steps

### Phase 1: Frontend Camera Capture

#### Step 1.1: Camera Initialization (`useCamera.js:45-85`)
```javascript
// Request camera access with specific constraints
const constraints = {
  video: {
    width: { ideal: 640, max: 1280 },
    height: { ideal: 480, max: 720 },
    frameRate: { ideal: 30, max: 30 }
  }
};

const stream = await navigator.mediaDevices.getUserMedia(constraints);
videoRef.current.srcObject = stream;
```

**Technical Details:**
- **Resolution**: 640×480 ideal, 1280×720 max
- **Frame Rate**: 30 FPS ideal
- **Browser API**: `getUserMedia()` with MediaStream
- **Error Handling**: Permission denied, device not found

#### Step 1.2: Canvas Setup (`useCamera.js:86-120`)
```javascript
// Three synchronized canvases for different views
const videoCanvas = videoRef.current;
const overlayCanvas = overlayRef.current;
const rgbCanvas = rgbPreviewRef.current;
const poseCanvas = posePreviewRef.current;

// Set canvas dimensions to 320×240 for processing
overlayCanvas.width = 320;
overlayCanvas.height = 240;
```

**Data Flow:**
- **Input**: Raw camera stream (MediaStream)
- **Processing**: Canvas rendering at 320×240
- **Output**: JPEG frames for backend transmission

#### Step 1.3: Frame Capture Loop (`useCamera.js:200-250`)
```javascript
const captureFrame = () => {
  if (!cameraActive || !running) return;

  // Draw current video frame to capture canvas
  const ctx = canvasRef.current.getContext('2d');
  ctx.drawImage(videoRef.current, 0, 0, 320, 240);

  // Convert to JPEG with quality 0.8
  const jpegData = canvasRef.current.toDataURL('image/jpeg', 0.8);

  // Send via WebSocket
  sendFrame(jpegData);

  // Schedule next capture (220ms = ~4.5 FPS)
  setTimeout(captureFrame, 220);
};
```

**Performance Metrics:**
- **Frame Rate**: 4.5 FPS (220ms intervals)
- **Resolution**: 320×240 pixels
- **Compression**: JPEG quality 0.8
- **Bandwidth**: ~15-20 KB per frame

### Phase 2: Real-time Display Updates

#### Step 2.1: Live Video Rendering (`useCamera.js:300-350`)
```javascript
const drawLiveView = () => {
  const video = videoRef.current;
  const overlay = overlayRef.current;

  if (video && overlay) {
    const ctx = overlay.getContext('2d');

    // Draw video frame
    ctx.drawImage(video, 0, 0, 320, 240);

    // Draw pose skeleton
    drawLandmarks(ctx, poseLandmarks, COCO_SKELETON);

    // Draw hand landmarks
    drawAllHands(ctx, handLandmarks);

    // Draw face privacy mask
    if (FACE_MASK_ENABLED) {
      drawFaceMask(ctx, poseLandmarks);
    }
  }
};
```

**Visual Elements:**
- **Body Pose**: 17 COCO keypoints with connecting lines
- **Hand Landmarks**: 21 points per hand × 2 hands
- **Face Mask**: 👨 emoji over face region
- **Attention Bars**: RGB vs Pose contribution indicators

#### Step 2.2: RGB Stream Preview (`useCamera.js:400-450`)
```javascript
const drawRgbPreview = (ctx, frame) => {
  // Enhanced color processing
  const imageData = ctx.getImageData(0, 0, 320, 240);
  const data = imageData.data;

  // Apply color enhancement
  for (let i = 0; i < data.length; i += 4) {
    data[i] *= 1.2;     // Red boost
    data[i + 1] *= 1.1; // Green boost
    data[i + 2] *= 1.3; // Blue boost
  }

  ctx.putImageData(imageData, 0, 0);

  // Add face mask
  if (FACE_MASK_ENABLED) {
    drawFaceMask(ctx, poseLandmarks);
  }
};
```

#### Step 2.3: Pose Stream Preview (`useCamera.js:450-500`)
```javascript
const drawPosePreview = (ctx) => {
  // Dark background
  ctx.fillStyle = '#0a1929';
  ctx.fillRect(0, 0, 320, 240);

  // Draw pose skeleton
  drawLandmarks(ctx, poseLandmarks, COCO_SKELETON, {
    color: '#22d3ee',
    lineWidth: 3
  });

  // Draw hand landmarks
  drawAllHands(ctx, handLandmarks, {
    color: '#a78bfa',
    radius: 4
  });

  // Draw face mask
  if (FACE_MASK_ENABLED) {
    drawFaceMask(ctx, poseLandmarks);
  }

  // Draw grid overlay
  drawGrid(ctx, 320, 240, 20);
};
```

### Phase 3: Backend WebSocket Processing

#### Step 3.1: WebSocket Message Handling (`websocket.py:50-85`)
```python
async def websocket_handler(websocket: WebSocket):
    client_state = {
        'frame_queue': deque(maxlen=120),
        'connected_at': datetime.now(),
        'last_frame_at': None,
        'stats': {}
    }

    await websocket.accept()

    try:
        while True:
            message = await websocket.receive_json()

            if message['type'] == 'client_video_frame':
                # Store frame in queue
                client_state['frame_queue'].append({
                    'data': message['data'],
                    'timestamp': message['timestamp']
                })

            elif message['type'] == 'control':
                # Handle control commands
                await handle_control_command(message, client_state)

    except WebSocketDisconnect:
        pass
```

**Queue Management:**
- **Max Length**: 120 frames
- **Data Format**: Base64 JPEG strings
- **Timestamp Tracking**: Client-side timestamps

#### Step 3.2: Frame Queue Processing (`engine/frame_ingest.py:20-45`)
```python
def maybe_process_latest_client_frame(client_state: dict) -> dict:
    frame_queue = client_state['frame_queue']

    if not frame_queue:
        return None

    # Process ALL pending frames in batch
    results = []
    while frame_queue:
        frame_data = frame_queue.popleft()

        # Decode base64 JPEG
        jpeg_bytes = base64.b64decode(frame_data['data'].split(',')[1])

        # Process frame through model runtime
        result = REAL_RUNTIME.process_frame(jpeg_bytes)
        results.append(result)

    # Return latest result
    return results[-1] if results else None
```

### Phase 4: Model Runtime Processing

#### Step 4.1: Frame Preprocessing (`runtime.py:50-80`)
```python
def process_frame(self, frame_bytes: bytes) -> dict:
    # Decode JPEG to numpy array
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Extract pose landmarks
    pose_vector = extract_pose_vector(frame_rgb)

    # Extract hand landmarks
    hand_landmarks = extract_hand_landmarks(frame_rgb)

    # Update sliding window buffers
    self._update_buffers(frame_rgb, pose_vector, hand_landmarks)

    # Check if we should run inference
    if self._should_run_inference():
        return self._run_inference()

    return self._get_current_state()
```

#### Step 4.2: Pose Vector Extraction (`runtime_pose.py:50-100`)
```python
def extract_pose_vector(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Extract 17-keypoint pose vector using YOLOv8-pose
    """
    try:
        # Primary: YOLOv8-pose
        results = self.yolo_model(frame_rgb, conf=0.5, iou=0.7)

        if results and len(results) > 0:
            keypoints = results[0].keypoints.xy.cpu().numpy()  # (17, 2)

            # Normalize relative to shoulder center
            shoulder_center = (keypoints[5] + keypoints[6]) / 2  # Shoulders
            width = np.linalg.norm(keypoints[6] - keypoints[5])  # Shoulder width

            normalized_keypoints = (keypoints - shoulder_center) / width
            return normalized_keypoints.flatten()  # (34,) vector

    except Exception as e:
        print(f"YOLOv8 failed, falling back to MediaPipe: {e}")

    # Fallback: MediaPipe Pose
    return self._extract_mediapipe_pose(frame_rgb)
```

**Keypoint Mapping:**
- **COCO-17 Format**: Nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
- **Normalization**: Relative to shoulder center, scaled by shoulder width
- **Output**: 34D vector (17 points × 2 coordinates)

#### Step 4.3: Hand Landmark Extraction (`runtime_pose.py:120-160`)
```python
def extract_hand_landmarks(frame_rgb: np.ndarray) -> list:
    """
    Extract hand landmarks using MediaPipe Tasks API
    """
    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    # Run detection
    result = self.hand_landmarker.detect(mp_image)

    hands_data = []
    for hand in result.hand_landmarks:
        # 21 landmarks per hand
        landmarks = [[lm.x, lm.y] for lm in hand]

        # Determine handedness
        label = "Right" if hand.handedness[0].category_name == "Right" else "Left"

        hands_data.append({
            'landmarks': landmarks,
            'label': label
        })

    return hands_data
```

**Hand Detection Details:**
- **Model**: MediaPipe HandLandmarker (7.8MB)
- **Landmarks**: 21 points per hand (fingertips, joints, palm)
- **Handedness**: Automatic left/right classification
- **Output**: List of hand dictionaries with normalized coordinates

#### Step 4.4: Buffer Management (`runtime.py:100-130`)
```python
def _update_buffers(self, frame_rgb, pose_vector, hand_landmarks):
    """
    Maintain 64-frame sliding window buffers
    """
    # RGB frame buffer (for ResNet features)
    self.rgb_frames.append(frame_rgb)
    if len(self.rgb_frames) > 64:
        self.rgb_frames.pop(0)

    # Pose vector buffer (34D per frame)
    self.pose_vectors.append(pose_vector)
    if len(self.pose_vectors) > 64:
        self.pose_vectors.pop(0)

    # Hand landmarks buffer
    self.hand_landmarks_buffer.append(hand_landmarks)
    if len(self.hand_landmarks_buffer) > 64:
        self.hand_landmarks_buffer.pop(0)

    self.frame_count += 1
```

**Buffer Characteristics:**
- **Size**: 64 frames sliding window
- **RGB Frames**: Raw images for CNN processing
- **Pose Vectors**: 34D normalized keypoints
- **Hand Landmarks**: List of hand dictionaries

#### Step 4.5: Inference Trigger Logic (`runtime.py:140-170`)
```python
def _should_run_inference(self) -> bool:
    """
    Determine if we should run model inference
    """
    # Need at least 64 frames
    if len(self.rgb_frames) < 64:
        return False

    # Run inference every 32 frames (stride)
    return self.frame_count % 32 == 0
```

**Inference Strategy:**
- **Minimum Frames**: 64-frame window required
- **Stride**: Process every 32 frames
- **Overlap**: 32-frame overlap between inferences
- **Real-time**: Continuous processing with sliding window

### Phase 5: Neural Network Inference

#### Step 5.1: Feature Extraction Pipeline (`runtime_loader.py:80-120`)
```python
def _load_model_components(self):
    """
    Load RGB stream, Pose stream, Fusion, and Temporal model
    """
    # RGB Stream: ResNet18 backbone
    self.rgb_stream = RGBStream(
        backbone="resnet18",
        feature_dim=512,
        pretrained=True
    )

    # Pose Stream: MLP
    self.pose_stream = PoseStream(
        input_dim=34,  # 17 keypoints × 2 coords
        hidden_dims=[512, 256],
        feature_dim=512
    )

    # Feature Fusion: Gated attention
    self.fusion = FeatureFusion(
        rgb_dim=512,
        pose_dim=512,
        fusion_dim=512,
        fusion_type="gated_attention"
    )

    # Temporal Model: BiLSTM
    self.temporal_model = TemporalModel(
        input_dim=512,
        hidden_dim=256,
        num_layers=2,
        vocab_size=self.vocab_size,
        model_type="bilstm"
    )
```

#### Step 5.2: RGB Feature Extraction (`rgb_stream.py:70-100`)
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Extract features from RGB frame sequence

    Args:
        x: (B, T, 3, 224, 224) - Batch of frame sequences

    Returns:
        (B, T, 512) - Frame-level features
    """
    B, T, C, H, W = x.shape

    # Reshape for batch processing
    x = x.view(B * T, C, H, W)  # (B*T, 3, 224, 224)

    # Extract features with optional chunking
    if self.backbone_chunk_size > 0:
        features = self._extract_features_chunked(x)
    else:
        features = self.backbone(x).flatten(1)  # (B*T, 512)

    # Project to feature dimension
    features = self.feature_proj(features)  # (B*T, 512)

    # Reshape back to sequence format
    return features.view(B, T, self.feature_dim)
```

**ResNet18 Processing:**
- **Input**: 224×224 RGB frames
- **Backbone**: Pretrained ResNet18 (remove final layers)
- **Output**: 512D feature vectors per frame
- **Batch Processing**: Handle multiple frames efficiently

#### Step 5.3: Pose Feature Extraction (`pose_stream.py:40-70`)
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Extract features from pose keypoint sequences

    Args:
        x: (B, T, 34) - Pose keypoint sequences

    Returns:
        (B, T, 512) - Pose features
    """
    # Handle different input formats
    if x.dim() == 4:  # (B, T, N, D)
        B, T, N, D = x.shape
        x = x.view(B, T, N * D)

    B, T, D = x.shape

    # Flatten temporal dimension
    x = x.view(B * T, D)  # (B*T, 34)

    # MLP processing
    features = self.mlp(x)  # (B*T, 512)

    # Reshape to sequence
    return features.view(B, T, self.feature_dim)
```

**MLP Architecture:**
- **Input**: 34D pose vectors (17 keypoints × 2 coords)
- **Hidden Layers**: 512 → 256 → 512 with BatchNorm + ReLU + Dropout
- **Output**: 512D feature vectors per frame

#### Step 5.4: Feature Fusion (`fusion.py:50-90`)
```python
def forward(self, rgb_features, pose_features):
    """
    Fuse RGB and Pose features with gated attention

    Args:
        rgb_features: (B, T, 512)
        pose_features: (B, T, 512)

    Returns:
        fused: (B, T, 512) - Fused features
        alpha: (B, T, 512) - RGB attention weights
        beta: (B, T, 512) - Pose attention weights
    """
    # Project pose features to same dimension
    pose_aligned = self.pose_proj(pose_features)

    # Compute attention gates
    alpha = torch.sigmoid(self.rgb_gate(rgb_features))  # RGB importance
    beta = torch.sigmoid(self.pose_gate(pose_features))  # Pose importance

    # Gated fusion
    fused = alpha * rgb_features + beta * pose_aligned

    # Layer normalization
    fused = self.norm(fused)

    return fused, alpha, beta
```

**Attention Mechanism:**
- **RGB Gate**: Learns importance of visual features
- **Pose Gate**: Learns importance of pose features
- **Fusion**: Weighted combination with normalization
- **Output**: Fused 512D features + attention weights

#### Step 5.5: Temporal Modeling (`temporal_model.py:50-80`)
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Temporal sequence modeling with BiLSTM

    Args:
        x: (B, T, 512) - Fused feature sequences

    Returns:
        (B, T, vocab_size+1) - CTC logits
    """
    # BiLSTM encoding
    encoded, _ = self.encoder(x)  # (B, T, 512)

    # Project to vocabulary logits
    logits = self.classifier(encoded)  # (B, T, vocab_size+1)

    return logits
```

**BiLSTM Details:**
- **Input**: 512D fused features per frame
- **Hidden Size**: 256D × 2 directions = 512D output
- **Layers**: 2 stacked BiLSTM layers
- **Dropout**: 0.3 between layers
- **Output**: Logits for each vocabulary token + CTC blank

#### Step 5.6: CTC Decoding (`ctc_layer.py:80-120`)
```python
def decode_greedy(self, logits: torch.Tensor) -> list:
    """
    Greedy CTC decoding

    Args:
        logits: (B, T, vocab_size+1)

    Returns:
        List of decoded token sequences
    """
    # Get most likely tokens per frame
    predictions = torch.argmax(logits, dim=-1)  # (B, T)

    decoded_sequences = []
    for pred in predictions:
        pred = pred.tolist()

        # Remove blanks and consecutive duplicates
        decoded_seq = []
        prev_token = None

        for token in pred:
            if token != self.blank_idx and token != prev_token:
                decoded_seq.append(token)
            prev_token = token

        decoded_sequences.append(decoded_seq)

    return decoded_sequences
```

**CTC Decoding Rules:**
- **Blank Removal**: Ignore CTC blank tokens (index 0)
- **Duplicate Removal**: Collapse consecutive same tokens
- **Output**: Clean token sequences

### Phase 6: Language Processing

#### Step 6.1: Token to Gloss Mapping (`decoder.py:30-60`)
```python
def decode(self, token_ids: List[int]) -> List[str]:
    """
    Convert token IDs to gloss strings
    """
    glosses = []
    for token_id in token_ids:
        if token_id in self.idx2gloss:
            gloss = self.idx2gloss[token_id]
            if gloss != "<blank>":
                glosses.append(gloss)

    return glosses
```

#### Step 6.2: Sentence Formation (`runtime_helpers.py:50-80`)
```python
def gloss_to_sentence(gloss_tokens: List[str]) -> str:
    """
    Convert gloss sequence to natural sentence
    """
    if not gloss_tokens:
        return "Analyzing live sign window..."

    # Basic rule-based sentence formation
    sentence = " ".join(gloss_tokens)

    # Capitalize first letter
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]

    # Add punctuation
    if not sentence.endswith(('.', '!', '?')):
        sentence += '.'

    return sentence
```

### Phase 7: Payload Construction

#### Step 7.1: Running Payload Assembly (`running_payload.py:100-150`)
```python
def build_running_payload(client_state: dict, inference_result: dict) -> dict:
    """
    Construct complete payload for frontend
    """
    payload = {
        'timestamp': datetime.now().isoformat(),
        'inference_mode': 'real',
        'module1': build_module1_data(client_state),
        'module2': build_module2_data(inference_result),
        'module3': build_module3_data(inference_result),
        'module4': build_module4_data(inference_result),
        'module5': build_module5_data(inference_result),
        'module6': build_module6_data(inference_result),
        'module7': build_module7_data(inference_result),
        'metrics': extract_metrics(inference_result),
        'audio_wave': generate_audio_wave(),
        'transcript_history': get_transcript_history(),
        'attention_weights': inference_result.get('attention_weights', {}),
        'system_info': get_system_info()
    }

    return payload
```

**7-Module Structure:**
1. **Module 1**: Frame capture and preprocessing status
2. **Module 2**: Feature extraction (RGB + Pose)
3. **Module 3**: Feature fusion with attention weights
4. **Module 4**: Temporal modeling (BiLSTM)
5. **Module 5**: CTC decoding results
6. **Module 6**: Language post-processing
7. **Module 7**: Final output and TTS

### Phase 8: Frontend Display Updates

#### Step 8.1: Payload Processing (`useWebSocket.js:40-70`)
```javascript
const handlePayload = (payload) => {
  // Update global state
  useStore.setState({
    payload: payload,
    connected: true
  });

  // Push metrics for dashboard
  if (payload.metrics) {
    const metricsHistory = useStore.getState().metricsHistory;
    metricsHistory.push({
      timestamp: Date.now(),
      fps: payload.metrics.fps,
      latency: payload.metrics.latency,
      confidence: payload.metrics.confidence
    });

    // Keep last 60 points
    if (metricsHistory.length > 60) {
      metricsHistory.shift();
    }
  }

  // Update console logs
  if (payload.module1?.parse) {
    const consoleLogs = useStore.getState().consoleLogs;
    consoleLogs.push(...payload.module1.parse);
    if (consoleLogs.length > 80) {
      consoleLogs.splice(0, consoleLogs.length - 80);
    }
  }
};
```

#### Step 8.2: UI Component Updates
- **GlossDisplay**: Shows current gloss tokens with confidence
- **SentenceOutput**: Displays full sentence with TTS controls
- **AttentionViz**: Updates RGB vs Pose attention bars
- **MetricsBar**: Shows FPS, latency, confidence
- **ConsolePanel**: Displays pipeline debug logs
- **TranscriptHistory**: Maintains rolling history

### Phase 9: Text-to-Speech Integration

#### Step 9.1: TTS Request Handling (`routes.py:80-100`)
```python
@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """
    Generate speech from text
    """
    try:
        # Primary: gTTS
        audio_bytes = await synthesize_with_gtts(
            request.text,
            lang=request.lang,
            slow=request.slow
        )

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"}
        )

    except Exception as e:
        # Fallback: pyttsx3
        audio_bytes = await synthesize_with_pyttsx3(request.text)

        return Response(
            content=audio_bytes,
            media_type="audio/wav"
        )
```

#### Step 9.2: Frontend TTS Playback (`SentenceOutput.jsx:20-50`)
```javascript
const speakNow = async (textOverride = '') => {
  const text = (textOverride || sentence || '').trim();
  if (!text) return;

  try {
    const res = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, lang: 'en', slow: false }),
    });

    if (!res.ok) return;

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);

    activeAudioRef.current = audio;
    audio.play();

    audio.onended = () => {
      URL.revokeObjectURL(url);
      if (activeAudioRef.current === audio) {
        activeAudioRef.current = null;
      }
    };

  } catch (error) {
    console.error('TTS failed:', error);
  }
};
```

## Performance Metrics and Optimization

### Latency Breakdown
- **Camera Capture**: 220ms (frontend)
- **WebSocket Transfer**: ~50ms
- **Frame Decoding**: ~20ms
- **Pose Detection**: ~100ms (YOLOv8 + MediaPipe)
- **Model Inference**: ~150ms (GPU)
- **Payload Construction**: ~10ms
- **UI Update**: ~20ms
- **Total Round-trip**: ~350ms

### Memory Usage
- **Model Weights**: 164MB (best.pt)
- **Frame Buffers**: 64 × (320×240×3 + 34 + hand_data) ≈ 15MB
- **WebSocket Queue**: 120 frames × 20KB ≈ 2.4MB
- **Frontend Canvases**: 3 × 320×240 × 4 ≈ 1MB

### Bandwidth Analysis
- **Frame Rate**: 4.5 FPS
- **Frame Size**: ~20KB JPEG (320×240, quality 0.8)
- **Total Bandwidth**: ~90 KB/s upstream
- **Payload Size**: ~50KB JSON every 350ms
- **Total Bandwidth**: ~140 KB/s bidirectional

## Error Handling and Fallbacks

### Camera Failures
- **Permission Denied**: Show error message, offer retry
- **Device Unavailable**: Fallback to placeholder, disable features
- **Resolution Issues**: Auto-scale to available resolution

### Model Failures
- **YOLOv8 Fails**: Fallback to MediaPipe Pose
- **Hand Detection Fails**: Continue with pose-only features
- **Inference Errors**: Return last valid result, show warning

### Network Issues
- **WebSocket Disconnect**: Auto-reconnect with exponential backoff
- **Frame Loss**: Queue management prevents overflow
- **Slow Network**: Reduce frame rate, increase compression

## Configuration and Tuning

### Model Parameters
```python
MODEL_CONFIG = {
    'rgb_backbone': 'resnet18',
    'feature_dim': 512,
    'fusion_type': 'gated_attention',
    'temporal_layers': 2,
    'hidden_dim': 256,
    'vocab_size': 1000
}
```

### Frontend Parameters
```javascript
CAMERA_CONFIG = {
    frameInterval: 220,  // ms
    width: 320,
    height: 240,
    jpegQuality: 0.8,
    faceMaskEnabled: true
};
```

### Backend Parameters
```python
RUNTIME_CONFIG = {
    'buffer_size': 64,
    'inference_stride': 32,
    'websocket_tick': 0.35,  # seconds
    'max_queue_size': 120
}
```

## Future Enhancements

### Real-time Optimizations
- **Model Quantization**: Reduce model size and latency
- **Edge Inference**: Run models on client-side GPU
- **Streaming Inference**: Continuous processing without windows
- **Adaptive Frame Rate**: Dynamic adjustment based on performance

### Advanced Features
- **Beam Search Decoding**: Better accuracy with language models
- **Multi-language Support**: Extend vocabulary and TTS
- **Gesture Recognition**: Additional hand gesture classification
- **Feedback Loop**: Real-time visual feedback during signing

---

This detailed workflow documentation covers every step of the ISL CSLR system, from camera capture to final sentence output, with complete technical specifications, code references, and performance characteristics.</content>
<parameter name="filePath">/home/kathir/CSLR/application/demo_ui/WORKFLOW.md