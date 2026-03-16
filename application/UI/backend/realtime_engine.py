"""
Real-time CSLR Engine - Complete 4-Module Pipeline
Integrates: Camera → Preprocessing → Feature Extraction → Recognition → Translation
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import torchvision.transforms as transforms

# ============================================================================
# MODULE 1: Video Preprocessing
# ============================================================================

@dataclass
class PreprocessingStats:
    fps: float
    processing_time_ms: float
    motion_score: float
    frames_kept: int
    frames_discarded: int
    buffer_fill: int
    buffer_capacity: int
    roi_detected: bool
    pose_detected: bool


class VideoPreprocessor:
    """Module 1: Smart Video Preprocessing with Motion Filtering & ROI"""
    
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        process_every_n_frame: int = 2,
        motion_threshold: float = 5.0,
        adaptive_motion: bool = True,
        draw_landmarks: bool = True
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.process_every_n_frame = process_every_n_frame
        self.motion_threshold = motion_threshold
        self.adaptive_motion = adaptive_motion
        self.draw_landmarks = draw_landmarks
        
        # Motion filtering state
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_count = 0
        self._recent_motion: deque = deque(maxlen=30)
        self._frames_kept = 0
        self._frames_discarded = 0
        
        # MediaPipe Holistic
        # MediaPipe exposes `solutions` dynamically; use getattr for static type checkers.
        mp_solutions: Any = getattr(mp, "solutions")
        self.mp_holistic = mp_solutions.holistic
        self.mp_drawing = mp_solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
        )
        
        # Transform for RGB normalization
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        # FPS tracking
        self._last_time = time.time()
        self._fps_history = deque(maxlen=30)
    
    def _calculate_motion_score(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Calculate motion score and decide if frame should be kept."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self._prev_gray is None:
            self._prev_gray = gray
            return True, 255.0
        
        # Compute motion
        diff = cv2.absdiff(self._prev_gray, gray)
        motion_score = float(np.mean(diff))
        self._prev_gray = gray
        
        # Adaptive threshold
        threshold = self.motion_threshold
        if self.adaptive_motion and self._recent_motion:
            mean_motion = float(np.mean(self._recent_motion))
            threshold = max(3.0, min(12.0, 0.6 * mean_motion + 2.0))
        
        self._recent_motion.append(motion_score)
        
        return motion_score > threshold, motion_score
    
    def _extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract 75 keypoints using MediaPipe Holistic."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        if results.pose_landmarks is None:
            return None
        
        # Extract pose (33) + left hand (21) + right hand (21) = 75 points
        pose_xy = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark], dtype=np.float32)
        
        left_hand = np.zeros((21, 2), dtype=np.float32)
        if results.left_hand_landmarks:
            left_hand = np.array([[lm.x, lm.y] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
        
        right_hand = np.zeros((21, 2), dtype=np.float32)
        if results.right_hand_landmarks:
            right_hand = np.array([[lm.x, lm.y] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
        
        # Combine: (33 + 21 + 21) × 2 = 75 × 2
        all_points = np.vstack([pose_xy, left_hand, right_hand])
        
        # Normalize relative to shoulder center and torso width
        left_shoulder = pose_xy[11]
        right_shoulder = pose_xy[12]
        center = (left_shoulder + right_shoulder) / 2.0
        torso_width = float(np.linalg.norm(left_shoulder - right_shoulder))
        if torso_width < 1e-4:
            torso_width = 0.2
        
        normalized = (all_points - center) / torso_width
        normalized = np.clip(normalized, -1.5, 1.5).astype(np.float32)
        
        # Draw landmarks if enabled
        if self.draw_landmarks and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        
        return normalized
    
    def _extract_roi(self, frame: np.ndarray, pose_xy: np.ndarray) -> np.ndarray:
        """Extract ROI focused on upper body and hands."""
        h, w = frame.shape[:2]
        
        # Shoulders = indices 11, 12
        left_shoulder = pose_xy[11] * [w, h]
        right_shoulder = pose_xy[12] * [w, h]
        
        if np.all(left_shoulder == 0) or np.all(right_shoulder == 0):
            return frame
        
        center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
        shoulder_width = max(40, int(np.linalg.norm(left_shoulder - right_shoulder)))
        
        roi_w = int(2.5 * shoulder_width)
        roi_h = int(2.8 * shoulder_width)
        y_center = int((left_shoulder[1] + right_shoulder[1]) / 2 + 0.5 * shoulder_width)
        
        x_min = max(center_x - roi_w // 2, 0)
        x_max = min(center_x + roi_w // 2, w)
        y_min = max(y_center - roi_h // 2, 0)
        y_max = min(y_center + roi_h // 2, h)
        
        if x_max <= x_min or y_max <= y_min:
            return frame
        
        return frame[y_min:y_max, x_min:x_max]
    
    def process(self, frame: np.ndarray) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], np.ndarray, PreprocessingStats]:
        """
        Process single frame through Module 1 pipeline.
        
        Returns:
            (rgb_tensor, pose_tensor, display_frame, stats)
        """
        start_time = time.time()
        self._frame_count += 1
        
        # Stage 1: Temporal subsampling
        if self._frame_count % self.process_every_n_frame != 0:
            self._frames_discarded += 1
            stats = PreprocessingStats(
                fps=self._get_fps(),
                processing_time_ms=0,
                motion_score=0,
                frames_kept=self._frames_kept,
                frames_discarded=self._frames_discarded,
                buffer_fill=0,
                buffer_capacity=64,
                roi_detected=False,
                pose_detected=False
            )
            return None, None, frame, stats
        
        # Stage 2: Motion filtering
        keep_frame, motion_score = self._calculate_motion_score(frame)
        if not keep_frame:
            self._frames_discarded += 1
            stats = PreprocessingStats(
                fps=self._get_fps(),
                processing_time_ms=(time.time() - start_time) * 1000,
                motion_score=motion_score,
                frames_kept=self._frames_kept,
                frames_discarded=self._frames_discarded,
                buffer_fill=0,
                buffer_capacity=64,
                roi_detected=False,
                pose_detected=False
            )
            return None, None, frame, stats
        
        # Stage 3: Pose extraction
        pose_normalized = self._extract_landmarks(frame)
        if pose_normalized is None:
            self._frames_discarded += 1
            stats = PreprocessingStats(
                fps=self._get_fps(),
                processing_time_ms=(time.time() - start_time) * 1000,
                motion_score=motion_score,
                frames_kept=self._frames_kept,
                frames_discarded=self._frames_discarded,
                buffer_fill=0,
                buffer_capacity=64,
                roi_detected=False,
                pose_detected=False
            )
            return None, None, frame, stats
        
        # Stage 4: ROI extraction
        roi_frame = self._extract_roi(frame, pose_normalized[:33])
        
        # Stage 5: RGB preprocessing
        transformed = self.transform(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
        if not isinstance(transformed, torch.Tensor):
            raise TypeError("RGB transform did not return a torch.Tensor")
        rgb_tensor = transformed
        
        # Stage 6: Pose tensor (flatten to 150-dim)
        pose_tensor: torch.Tensor = torch.from_numpy(pose_normalized.reshape(-1)).float()
        
        self._frames_kept += 1
        
        processing_time = (time.time() - start_time) * 1000
        
        stats = PreprocessingStats(
            fps=self._get_fps(),
            processing_time_ms=processing_time,
            motion_score=motion_score,
            frames_kept=self._frames_kept,
            frames_discarded=self._frames_discarded,
            buffer_fill=0,
            buffer_capacity=64,
            roi_detected=True,
            pose_detected=True
        )
        
        return rgb_tensor, pose_tensor, frame, stats
    
    def _get_fps(self) -> float:
        """Calculate current FPS."""
        current_time = time.time()
        fps = 1.0 / (current_time - self._last_time + 1e-6)
        self._last_time = current_time
        self._fps_history.append(fps)
        return float(np.mean(self._fps_history)) if self._fps_history else 0.0
    
    def close(self):
        """Cleanup resources."""
        if self.holistic:
            self.holistic.close()


# ============================================================================
# MODULE 2: Dual-Stream Feature Extraction
# ============================================================================

class RGBStreamExtractor(nn.Module):
    """RGB feature extractor using ResNet18."""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        except:
            from torchvision.models import resnet18
            backbone = resnet18(pretrained=pretrained)
        
        # Remove final FC layer
        self.feature_net = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = 512
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, T, 3, 224, 224)
        Output: (B, T, 512)
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.feature_net(x).flatten(1)
        return feat.view(b, t, self.out_dim)


class PoseStreamExtractor(nn.Module):
    """MLP-based pose feature extractor."""
    
    def __init__(self, in_dim: int = 150, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, T, 150)
        Output: (B, T, 256)
        """
        b, t, d = x.shape
        x = x.view(b * t, d)
        feat = self.net(x)
        return feat.view(b, t, self.out_dim)


class AttentionFusion(nn.Module):
    """Attention-based fusion of RGB and Pose streams."""
    
    def __init__(self, rgb_dim: int = 512, pose_dim: int = 256, fused_dim: int = 512):
        super().__init__()
        self.pose_proj = nn.Linear(pose_dim, fused_dim)
        self.rgb_gate = nn.Linear(rgb_dim, fused_dim)
        self.pose_gate = nn.Linear(pose_dim, fused_dim)
        self.norm = nn.LayerNorm(fused_dim)
    
    def forward(self, rgb_feat: torch.Tensor, pose_feat: torch.Tensor) -> torch.Tensor:
        """
        Input: rgb_feat (B, T, 512), pose_feat (B, T, 256)
        Output: fused (B, T, 512)
        """
        pose_aligned = self.pose_proj(pose_feat)
        alpha = torch.sigmoid(self.rgb_gate(rgb_feat))
        beta = torch.sigmoid(self.pose_gate(pose_feat))
        fused = self.norm(alpha * rgb_feat + beta * pose_aligned)
        return fused


# ============================================================================
# MODULE 3: Temporal Recognition
# ============================================================================

class TemporalRecognizer(nn.Module):
    """BiLSTM-based temporal model with CTC output."""
    
    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 20,
        dropout: float = 0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes + 1)  # +1 for CTC blank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, T, 512)
        Output: (B, T, num_classes + 1)
        """
        out, _ = self.lstm(x)
        logits = self.classifier(out)
        return torch.log_softmax(logits, dim=-1)


class CTCGreedyDecoder:
    """CTC greedy decoder."""
    
    def __init__(self, vocabulary: List[str]):
        self.vocabulary = vocabulary
        self.blank_idx = 0
    
    def decode(self, logits: torch.Tensor) -> Tuple[List[str], float]:
        """
        Decode CTC output to gloss sequence.
        
        Args:
            logits: (T, num_classes + 1)
        
        Returns:
            (gloss_tokens, confidence)
        """
        probs = torch.exp(logits)
        confidence = float(torch.max(probs, dim=-1).values.mean().item())
        pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        
        # Collapse repeats and remove blanks
        tokens = []
        prev = None
        for idx in pred:
            if idx != self.blank_idx and idx != prev:
                tokens.append(idx)
            prev = idx
        
        # Map to vocabulary
        decoded = [self.vocabulary[i - 1] for i in tokens if 1 <= i <= len(self.vocabulary)]
        
        return decoded, confidence


# ============================================================================
# MODULE 4: Translation & Grammar
# ============================================================================

class GrammarCorrector:
    """Rule-based gloss-to-sentence conversion."""
    
    def __init__(self):
        self.patterns = {
            "ME GO SCHOOL": "I am going to school.",
            "ME NEED WATER": "I need water.",
            "WHAT YOUR NAME": "What is your name?",
            "YOU FINE": "Are you fine?",
            "HELLO": "Hello.",
            "THANK-YOU": "Thank you.",
            "THANKS": "Thanks.",
            "PLEASE": "Please.",
            "YES": "Yes.",
            "NO": "No.",
        }
    
    def correct(self, gloss_tokens: List[str]) -> str:
        """Convert gloss sequence to grammatical English."""
        if not gloss_tokens:
            return ""
        
        text = " ".join(gloss_tokens)
        
        # Exact match
        if text in self.patterns:
            return self.patterns[text]
        
        # Rule-based conversion
        if text.startswith("ME "):
            tail = text[3:].replace("GO", "am going to").replace("NEED", "need").replace("WANT", "want")
            return f"I {tail.lower()}."
        
        if text.startswith("YOU "):
            tail = text[4:].replace("FINE", "fine").replace("GO", "are going")
            return f"You {tail.lower()}."
        
        # Fallback: capitalize first word, lowercase rest
        words = [w.lower() if i > 0 else w.capitalize() for i, w in enumerate(gloss_tokens)]
        return " ".join(words) + "."


# ============================================================================
# SLIDING WINDOW BUFFER
# ============================================================================

class SlidingWindowBuffer:
    """Maintains rolling window with configurable stride."""
    
    def __init__(self, window_size: int = 64, stride: int = 32):
        self.window_size = window_size
        self.stride = stride
        self.rgb_buffer = deque(maxlen=window_size)
        self.pose_buffer = deque(maxlen=window_size)
        self._frames_since_emit = 0
    
    def add(self, rgb_frame: torch.Tensor, pose_frame: torch.Tensor):
        """Add frame to buffer."""
        self.rgb_buffer.append(rgb_frame)
        self.pose_buffer.append(pose_frame)
    
    def is_ready(self) -> bool:
        """Check if window is ready for processing."""
        if len(self.rgb_buffer) < self.window_size:
            return False
        
        self._frames_since_emit += 1
        if self._frames_since_emit < self.stride:
            return False
        
        return True
    
    def get_window(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current window."""
        rgb_window = torch.stack(list(self.rgb_buffer), dim=0)
        pose_window = torch.stack(list(self.pose_buffer), dim=0)
        return rgb_window, pose_window
    
    def advance(self):
        """Advance window by stride."""
        self._frames_since_emit = 0
    
    def get_fill(self) -> int:
        """Get current buffer fill level."""
        return len(self.rgb_buffer)
    
    def reset(self):
        """Clear buffer."""
        self.rgb_buffer.clear()
        self.pose_buffer.clear()
        self._frames_since_emit = 0


# ============================================================================
# COMPLETE CSLR PIPELINE
# ============================================================================

class CSLRPipeline:
    """Complete 4-module CSLR pipeline."""
    
    def __init__(self, vocabulary: List[str], use_gpu: bool = True, demo_mode: bool = False):
        self.vocabulary = vocabulary
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.demo_mode = demo_mode
        
        # Module 2: Feature extractors
        self.rgb_extractor = RGBStreamExtractor(pretrained=True).to(self.device)
        self.pose_extractor = PoseStreamExtractor().to(self.device)
        self.fusion = AttentionFusion().to(self.device)
        
        # Module 3: Temporal recognizer
        self.temporal_model = TemporalRecognizer(num_classes=len(vocabulary)).to(self.device)
        self.decoder = CTCGreedyDecoder(vocabulary)
        
        # Module 4: Grammar corrector
        self.grammar = GrammarCorrector()
        
        # Set to eval mode (for demo without trained weights)
        self.rgb_extractor.eval()
        self.pose_extractor.eval()
        self.fusion.eval()
        self.temporal_model.eval()
    
    @torch.no_grad()
    def process_window(
        self,
        rgb_window: torch.Tensor,
        pose_window: torch.Tensor
    ) -> Tuple[List[str], str, float]:
        """
        Process complete window through pipeline.
        
        Args:
            rgb_window: (T, 3, 224, 224)
            pose_window: (T, 150)
        
        Returns:
            (glosses, sentence, confidence)
        """
        # Add batch dimension
        rgb_window = rgb_window.unsqueeze(0).to(self.device)  # (1, T, 3, 224, 224)
        pose_window = pose_window.unsqueeze(0).to(self.device)  # (1, T, 150)
        
        # Module 2: Feature extraction
        rgb_feat = self.rgb_extractor(rgb_window)  # (1, T, 512)
        pose_feat = self.pose_extractor(pose_window)  # (1, T, 256)
        fused_feat = self.fusion(rgb_feat, pose_feat)  # (1, T, 512)
        
        # Module 3: Temporal recognition
        logits = self.temporal_model(fused_feat)  # (1, T, num_classes + 1)
        
        # Module 3: CTC decoding
        glosses, confidence = self.decoder.decode(logits.squeeze(0))  # Remove batch dim
        
        # Module 4: Grammar correction
        sentence = self.grammar.correct(glosses)
        
        # Fallback for demo mode (untrained weights produce random output)
        if self.demo_mode and (not glosses or confidence < 0.5):
            # Use heuristic based on input motion
            pose_mean = pose_window.mean().item()
            idx = abs(int(pose_mean * 1000)) % len(self.vocabulary)
            glosses = [self.vocabulary[idx]]
            sentence = self.grammar.correct(glosses)
            confidence = 0.75
        
        return glosses, sentence, confidence


# ============================================================================
# MAIN REALTIME ENGINE
# ============================================================================

class RealtimeCSLREngine:
    """Complete real-time CSLR engine with camera integration."""
    
    def __init__(self, vocabulary: List[str], demo_mode: bool = False):
        self.vocabulary = vocabulary
        self.demo_mode = demo_mode
        
        # Module 1: Preprocessor
        self.preprocessor: Optional[VideoPreprocessor] = None
        self.camera: Optional[cv2.VideoCapture] = None
        
        # Modules 2-4: Pipeline
        self.pipeline = CSLRPipeline(vocabulary, use_gpu=True, demo_mode=demo_mode)
        
        # Sliding window
        self.window_buffer = SlidingWindowBuffer(window_size=64, stride=32)
        
        # History
        self.transcripts = deque(maxlen=20)
    
    def start_camera(self, camera_id: int = 0) -> bool:
        """Initialize camera and preprocessor."""
        try:
            self.camera = cv2.VideoCapture(camera_id)
            if not self.camera.isOpened():
                return False
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 20)
            
            self.preprocessor = VideoPreprocessor(
                frame_width=640,
                frame_height=480,
                process_every_n_frame=2,
                motion_threshold=5.0,
                adaptive_motion=True,
                draw_landmarks=True
            )
            
            return True
        except Exception as e:
            print(f"Camera init failed: {e}")
            return False
    
    def stop_camera(self):
        """Release camera resources."""
        if self.camera:
            self.camera.release()
            self.camera = None
        if self.preprocessor:
            self.preprocessor.close()
            self.preprocessor = None
        self.clear_runtime()

    def clear_runtime(self):
        """Clear runtime buffers/transcripts without reinitializing models."""
        self.window_buffer.reset()
        self.transcripts.clear()
    
    def process_frame(self) -> Tuple[Optional[str], Optional[str], float, PreprocessingStats, np.ndarray]:
        """
        Process single camera frame through complete pipeline.
        
        Returns:
            (glosses_str, sentence, confidence, stats, display_frame)
        """
        if not self.camera or not self.preprocessor:
            raise RuntimeError("Camera not initialized")
        
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to read frame")
        
        # Module 1: Preprocessing
        rgb_tensor, pose_tensor, display_frame, stats = self.preprocessor.process(frame)
        
        # Update buffer fill stat
        stats.buffer_fill = self.window_buffer.get_fill()
        
        if rgb_tensor is not None and pose_tensor is not None:
            # Add to window buffer
            self.window_buffer.add(rgb_tensor, pose_tensor)
            
            # Check if window is ready
            if self.window_buffer.is_ready():
                # Get window
                rgb_window, pose_window = self.window_buffer.get_window()
                
                # Process through pipeline
                glosses, sentence, confidence = self.pipeline.process_window(rgb_window, pose_window)
                
                # Advance window
                self.window_buffer.advance()
                
                # Update history
                if sentence and sentence not in self.transcripts:
                    self.transcripts.append(sentence)
                
                glosses_str = " ".join(glosses) if glosses else ""
                return glosses_str, sentence, confidence, stats, display_frame
        
        return None, None, 0.0, stats, display_frame
