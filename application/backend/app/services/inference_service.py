"""
Inference Service
Orchestrates the full ML pipeline with all 4 modules connected.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from app.core.config import settings
from app.core.logging import logger
from app.monitoring.metrics import InferenceMetrics, global_metrics
from app.pipeline import (
    CTCLayer,
    Decoder,
    FeatureFusion,
    FrameSampler,
    PoseExtractor,
    PoseStream,
    PostProcessor,
    RGBStream,
    TemporalModel,
    Translator,
    VideoLoader,
)
from app.services.translation_service import TranslationService
from app.utils.ctc_decoder import CTCDecoder
from app.utils.grammar_correction import GrammarCorrector as UtilGrammarCorrector
from app.utils.math_utils import moving_average
from app.utils.sliding_window import SlidingWindowBuffer
from app.utils.tensor_utils import to_tensor
from app.utils.video_preprocessing import VideoPreprocessor
from app.services.checkpoint_runtime import CheckpointRuntime


class InferenceService:
    """
    Main inference service - connects all 4 modules.
    Full end-to-end pipeline for CSLR.
    """

    def __init__(self, vocab_file: Optional[str] = None):
        self.device = settings.DEVICE
        self.use_amp = settings.USE_AMP
        self.metrics = global_metrics
        self.checkpoint_runtime = CheckpointRuntime()

        # Module 1: Preprocessing
        logger.info("Initializing Module 1: Preprocessing")
        self.video_loader = VideoLoader()
        self.pose_extractor = PoseExtractor()
        self.frame_sampler = FrameSampler(target_fps=25)
        self.video_preprocessor = VideoPreprocessor()

        # Module 2: Feature Extraction
        logger.info("Initializing Module 2: Feature Extraction")
        self.rgb_stream = RGBStream(
            backbone="resnet18",
            pretrained=True,
            freeze_backbone=False,
            dropout=0.1,
        ).to(self.device)

        self.pose_stream = PoseStream(
            input_dim=150,  # 75 keypoints * 2 (x, y)
            hidden_dims=[512, 256],
            feature_dim=512,
        ).to(self.device)

        self.fusion = FeatureFusion(
            rgb_dim=512,
            pose_dim=512,
            fusion_dim=512,
            fusion_type="gated_attention",
        ).to(self.device)

        # Module 3: Sequence Modeling
        logger.info("Initializing Module 3: Sequence Modeling")
        self.temporal_model = TemporalModel(
            input_dim=512,
            hidden_dim=256,
            num_layers=2,
            vocab_size=2000,
            model_type="bilstm",
            dropout=0.3,
        ).to(self.device)

        self.ctc_layer = CTCLayer(blank_idx=0, vocab_size=2000)

        # Load vocab
        if vocab_file and Path(vocab_file).exists():
            with open(vocab_file, "r") as f:
                import json

                vocab = json.load(f)
        else:
            vocab = [f"WORD_{i}" for i in range(2000)]

        self.decoder = Decoder(labels=vocab)
        self.ctc_decoder = CTCDecoder(labels=vocab, beam_width=settings.BEAM_WIDTH)

        # Module 4: Language Processing
        logger.info("Initializing Module 4: Language Processing")
        self.translator = Translator()
        self.translation_service = TranslationService()
        self.grammar_corrector = UtilGrammarCorrector()
        self.post_processor = PostProcessor()

        # Sliding window buffer for streaming
        self.sliding_window = SlidingWindowBuffer(window_size=64, stride=32)

        # Set to eval mode
        self.rgb_stream.eval()
        self.pose_stream.eval()
        self.fusion.eval()
        self.temporal_model.eval()

        logger.info("All modules initialized successfully")

    async def _build_sentence(self, gloss_tokens: List[str]) -> str:
        if settings.USE_TRANSLATION_SERVICE:
            sentence = await self.translation_service.translate_with_grammar(gloss_tokens)
            return self.post_processor.apply(sentence) if sentence else ""
        sentence = self.grammar_corrector.gloss_to_sentence(gloss_tokens)
        return self.post_processor.apply(sentence)

    def _record_metrics(
        self,
        total_start: float,
        module1_time: float,
        module2_time: float,
        module3_time: float,
        module4_time: float,
        num_frames: int,
        confidence: float,
        frame_confidences: Optional[List[float]] = None,
    ) -> None:
        if not settings.ENABLE_METRICS:
            return

        total_time = time.time() - total_start
        fps_value = num_frames / total_time if total_time > 0 else 0.0
        smoothed = moving_average(frame_confidences or [], window_size=5)
        metrics_conf = float(np.mean(smoothed)) if smoothed else confidence
        self.metrics.record_inference(
            InferenceMetrics(
                total_time=total_time,
                module1_time=module1_time,
                module2_time=module2_time,
                module3_time=module3_time,
                module4_time=module4_time,
                fps=fps_value,
                num_frames=num_frames,
                confidence=metrics_conf,
            )
        )

    async def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process full video file through complete pipeline.
        """
        if self.checkpoint_runtime.available:
            frames, _ = self.video_loader.load_video(video_path)
            result = self.checkpoint_runtime.process_frames(
                [cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) for frame_rgb in frames]
            )
            return {
                "gloss": result.get("gloss", []),
                "sentence": result.get("sentence", ""),
                "confidence": result.get("confidence", 0.0),
                "frame_count": len(frames),
                "fps": 25.0,
                "pose_landmarks": result.get("pose_landmarks", []),
                "runtime_status": self.checkpoint_runtime.reason,
            }

        logger.info(f"Processing video: {video_path}")

        total_start = time.time()
        module1_time = module2_time = module3_time = module4_time = 0.0

        try:
            # Module 1: Load and preprocess video
            if settings.ENABLE_METRICS:
                self.metrics.start_timer("module1")
            frames, fps = self.video_loader.load_video(video_path)
            frames = self.frame_sampler.sample_frames(
                frames, original_fps=fps, target_fps=25
            )

            rgb_frames: List[torch.Tensor] = []
            pose_frames: List[torch.Tensor] = []

            for frame_rgb in frames:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                result = self.video_preprocessor.process(frame_bgr)
                if result.kept and result.rgb_tensor is not None and result.pose_tensor is not None:
                    rgb_frames.append(result.rgb_tensor)
                    pose_frames.append(result.pose_tensor)

            if not rgb_frames:
                return {
                    "gloss": [],
                    "sentence": "",
                    "confidence": 0.0,
                    "error": "No valid frames extracted",
                }
            if settings.ENABLE_METRICS:
                module1_time = self.metrics.stop_timer("module1")

            # Stack to tensors
            rgb_tensor = to_tensor(torch.stack(rgb_frames), device=self.device).unsqueeze(0)
            pose_tensor = to_tensor(torch.stack(pose_frames), device=self.device).unsqueeze(0)

            # Module 2: Feature extraction
            if settings.ENABLE_METRICS:
                self.metrics.start_timer("module2")
            with torch.no_grad():
                rgb_features = self.rgb_stream(rgb_tensor)
                pose_features = self.pose_stream(pose_tensor)
                fused_features, alpha, beta = self.fusion(rgb_features, pose_features)
            if settings.ENABLE_METRICS:
                module2_time = self.metrics.stop_timer("module2")

            # Module 3: Sequence modeling
            if settings.ENABLE_METRICS:
                self.metrics.start_timer("module3")
            with torch.no_grad():
                logits = self.temporal_model(fused_features)
            if settings.ENABLE_METRICS:
                module3_time = self.metrics.stop_timer("module3")

            # CTC Decoding
            decode_result = self.ctc_decoder.beam_search_decode(
                logits.squeeze(0).cpu(),
                beam_width=settings.BEAM_WIDTH,
            )
            gloss_tokens = decode_result.gloss_tokens
            confidence = decode_result.confidence

            # Module 4: Language processing
            if settings.ENABLE_METRICS:
                self.metrics.start_timer("module4")
            sentence = await self._build_sentence(gloss_tokens)
            if settings.ENABLE_METRICS:
                module4_time = self.metrics.stop_timer("module4")

            self._record_metrics(
                total_start=total_start,
                module1_time=module1_time,
                module2_time=module2_time,
                module3_time=module3_time,
                module4_time=module4_time,
                num_frames=len(frames),
                confidence=confidence,
                frame_confidences=decode_result.frame_confidences,
            )

            logger.info(
                f"Inference complete: {len(gloss_tokens)} glosses, confidence={confidence:.2f}"
            )
            return {
                "gloss": gloss_tokens,
                "sentence": sentence,
                "confidence": confidence,
                "frame_count": len(frames),
                "fps": 25.0,
                "attention": {
                    "alpha": alpha.cpu().numpy().tolist() if alpha is not None else None,
                    "beta": beta.cpu().numpy().tolist() if beta is not None else None,
                },
            }
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return {"gloss": [], "sentence": "", "confidence": 0.0, "error": str(e)}

    async def process_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Process sequence of frames through pipeline.
        """
        if self.checkpoint_runtime.available:
            result = self.checkpoint_runtime.process_frames(frames)
            return {
                "gloss": result.get("gloss", []),
                "sentence": result.get("sentence", ""),
                "confidence": result.get("confidence", 0.0),
                "fps": 25.0,
                "pose_landmarks": result.get("pose_landmarks", []),
                "runtime_status": self.checkpoint_runtime.reason,
            }

        logger.info(f"Processing {len(frames)} frames")

        total_start = time.time()
        module1_time = module2_time = module3_time = module4_time = 0.0

        try:
            if settings.ENABLE_METRICS:
                self.metrics.start_timer("module1")

            rgb_frames: List[torch.Tensor] = []
            pose_frames: List[torch.Tensor] = []

            for frame in frames:
                result = self.video_preprocessor.process(frame)
                if result.kept and result.rgb_tensor is not None and result.pose_tensor is not None:
                    rgb_frames.append(result.rgb_tensor)
                    pose_frames.append(result.pose_tensor)

            if not rgb_frames:
                return {"gloss": [], "sentence": "", "confidence": 0.0, "partial": True}
            if settings.ENABLE_METRICS:
                module1_time = self.metrics.stop_timer("module1")

            rgb_tensor = to_tensor(torch.stack(rgb_frames), device=self.device).unsqueeze(0)
            pose_tensor = to_tensor(torch.stack(pose_frames), device=self.device).unsqueeze(0)

            if settings.ENABLE_METRICS:
                self.metrics.start_timer("module2")
            with torch.no_grad():
                rgb_features = self.rgb_stream(rgb_tensor)
                pose_features = self.pose_stream(pose_tensor)
                fused_features, _, _ = self.fusion(rgb_features, pose_features)
            if settings.ENABLE_METRICS:
                module2_time = self.metrics.stop_timer("module2")

            if settings.ENABLE_METRICS:
                self.metrics.start_timer("module3")
            with torch.no_grad():
                logits = self.temporal_model(fused_features)
            if settings.ENABLE_METRICS:
                module3_time = self.metrics.stop_timer("module3")

            decode_result = self.ctc_decoder.ctc_greedy_decode(logits.squeeze(0).cpu())
            gloss_tokens = decode_result.gloss_tokens

            if settings.ENABLE_METRICS:
                self.metrics.start_timer("module4")
            sentence = await self._build_sentence(gloss_tokens)
            if settings.ENABLE_METRICS:
                module4_time = self.metrics.stop_timer("module4")

            self._record_metrics(
                total_start=total_start,
                module1_time=module1_time,
                module2_time=module2_time,
                module3_time=module3_time,
                module4_time=module4_time,
                num_frames=len(frames),
                confidence=decode_result.confidence,
                frame_confidences=decode_result.frame_confidences,
            )

            return {
                "gloss": gloss_tokens,
                "sentence": sentence,
                "confidence": decode_result.confidence,
                "fps": 25.0,
            }
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return {"gloss": [], "sentence": "", "confidence": 0.0, "error": str(e)}

    async def process_frame_stream(
        self, frame: np.ndarray, state: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process single frame in streaming mode with sliding window.
        """
        if self.checkpoint_runtime.available:
            result = self.checkpoint_runtime.process_stream_frame(frame, state)
            return {
                "gloss": result.get("gloss", []),
                "sentence": result.get("sentence", ""),
                "confidence": result.get("confidence", 0.0),
                "fps": 25.0,
                "pose_landmarks": result.get("pose_landmarks", []),
                "hand_landmarks": [],
                "partial": result.get("partial", True),
                "buffer_fill": result.get("buffer_fill", 0),
                "state": result.get("state"),
                "runtime_status": self.checkpoint_runtime.reason,
            }

        if state is None:
            state = {
                "buffer": SlidingWindowBuffer(window_size=64, stride=32),
                "last_gloss": [],
                "last_sentence": "",
                "last_pose_landmarks": [],
                "last_hand_landmarks": [],
            }

        try:
            result = self.video_preprocessor.process(frame)
            if not result.kept:
                return {
                    "gloss": state["last_gloss"],
                    "sentence": state["last_sentence"],
                    "confidence": 0.0,
                    "pose_landmarks": state.get("last_pose_landmarks", []),
                    "hand_landmarks": state.get("last_hand_landmarks", []),
                    "partial": True,
                    "state": state,
                }

            if result.rgb_tensor is None or result.pose_tensor is None:
                return {
                    "gloss": state["last_gloss"],
                    "sentence": state["last_sentence"],
                    "confidence": 0.0,
                    "pose_landmarks": state.get("last_pose_landmarks", []),
                    "hand_landmarks": state.get("last_hand_landmarks", []),
                    "partial": True,
                    "state": state,
                }

            state["last_pose_landmarks"] = result.pose_landmarks
            state["last_hand_landmarks"] = result.hand_landmarks

            window = state["buffer"].add(result.rgb_tensor, result.pose_tensor)
            if window is None:
                return {
                    "gloss": state["last_gloss"],
                    "sentence": state["last_sentence"],
                    "confidence": 0.0,
                    "pose_landmarks": state.get("last_pose_landmarks", []),
                    "hand_landmarks": state.get("last_hand_landmarks", []),
                    "partial": True,
                    "state": state,
                }

            rgb_window, pose_window = window
            rgb_tensor = rgb_window.unsqueeze(0).to(self.device)
            pose_tensor = pose_window.unsqueeze(0).to(self.device)

            with torch.no_grad():
                rgb_features = self.rgb_stream(rgb_tensor)
                pose_features = self.pose_stream(pose_tensor)
                fused_features, _, _ = self.fusion(rgb_features, pose_features)
                logits = self.temporal_model(fused_features)

            decode_result = self.ctc_decoder.ctc_greedy_decode(logits.squeeze(0).cpu())
            gloss_tokens = decode_result.gloss_tokens
            sentence = await self._build_sentence(gloss_tokens)

            state["last_gloss"] = gloss_tokens
            state["last_sentence"] = sentence

            return {
                "gloss": gloss_tokens,
                "sentence": sentence,
                "confidence": decode_result.confidence,
                "fps": 25.0,
                "pose_landmarks": state.get("last_pose_landmarks", []),
                "hand_landmarks": state.get("last_hand_landmarks", []),
                "partial": state["buffer"].counts() < state["buffer"].window_size,
                "state": state,
            }
        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            return {
                "gloss": state.get("last_gloss", []),
                "sentence": state.get("last_sentence", ""),
                "confidence": 0.0,
                "pose_landmarks": state.get("last_pose_landmarks", []),
                "hand_landmarks": state.get("last_hand_landmarks", []),
                "partial": True,
                "state": state,
                "error": str(e),
            }
