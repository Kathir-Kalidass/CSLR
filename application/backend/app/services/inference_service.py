"""
Inference Service
Orchestrates the full ML pipeline with all 4 modules connected.
"""

from __future__ import annotations

import json
import math
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
from app.utils.ctc_decoder import CTCDecoder, CaptionPostProcessor
from app.utils.grammar_correction import GrammarCorrector as UtilGrammarCorrector
from app.utils.math_utils import moving_average
from app.utils.sliding_window import SlidingWindowBuffer
from app.utils.tensor_utils import to_tensor
from app.utils.video_preprocessing import VideoPreprocessor


class InferenceService:
    """
    Main inference service - connects all 4 modules.
    Full end-to-end pipeline for CSLR.
    """

    def __init__(self, vocab_file: Optional[str] = None):
        self.device = settings.DEVICE
        self.use_amp = settings.USE_AMP
        self.metrics = global_metrics
        self.blocked_glosses = {
            tok.strip().upper()
            for tok in settings.GLOSS_BLOCKLIST.split(",")
            if tok.strip()
        }

        vocab = self._load_vocab(vocab_file)
        self.vocab_size = len(vocab)

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
            vocab_size=self.vocab_size,
            model_type="bilstm",
            dropout=0.3,
        ).to(self.device)

        self.ctc_layer = CTCLayer(blank_idx=0, vocab_size=self.vocab_size)

        self.decoder = Decoder(labels=vocab)
        self.ctc_decoder = CTCDecoder(
            labels=vocab,
            beam_width=settings.BEAM_WIDTH,
            min_token_run=settings.CTC_MIN_TOKEN_RUN,
            min_token_margin=settings.CTC_MIN_TOKEN_MARGIN,
            length_norm_alpha=settings.CTC_LENGTH_NORM_ALPHA,
            repetition_penalty=settings.CTC_REPETITION_PENALTY,
        )
        self.caption_filter = CaptionPostProcessor(
            max_history=settings.GLOSS_HISTORY_SIZE,
            min_confidence=settings.CONFIDENCE_THRESHOLD,
            vote_window=settings.GLOSS_VOTE_WINDOW,
            min_votes=settings.GLOSS_MIN_VOTES,
        )

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

        self._load_pipeline_checkpoints()
        self._load_confidence_calibration()

        logger.info("All modules initialized successfully")

    def _load_confidence_calibration(self) -> None:
        self.confidence_temperature = settings.CONFIDENCE_TEMPERATURE
        if not settings.ENABLE_CONFIDENCE_CALIBRATION:
            return
        path = Path(settings.CONFIDENCE_CALIBRATION_FILE)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            t = float(obj.get("temperature", self.confidence_temperature))
            if t > 0:
                self.confidence_temperature = t
                logger.info("Loaded confidence temperature calibration from %s (T=%.3f)", path, t)
        except Exception as exc:
            logger.warning("Failed to load confidence calibration from %s: %s", path, exc)

    def _apply_confidence_temperature(self, confidence: float) -> float:
        c = float(np.clip(confidence, 1e-6, 1 - 1e-6))
        if not settings.ENABLE_CONFIDENCE_CALIBRATION:
            return c
        logit = math.log(c / (1.0 - c))
        p = 1.0 / (1.0 + math.exp(-(logit / max(self.confidence_temperature, 1e-6))))
        return float(np.clip(p, 0.0, 1.0))

    def _decode_with_ensemble(self, logits: torch.Tensor) -> Dict[str, Any]:
        """Decode with beam+greedy agreement checks for improved precision."""
        beam = self.ctc_decoder.beam_search_decode(logits.squeeze(0).cpu(), beam_width=settings.BEAM_WIDTH)
        if not settings.ENABLE_ENSEMBLE_DECODE:
            return {
                "tokens": beam.gloss_tokens,
                "confidence": self._apply_confidence_temperature(beam.confidence),
                "frame_confidences": beam.frame_confidences,
                "agreement": 1.0,
            }

        greedy = self.ctc_decoder.ctc_greedy_decode(logits.squeeze(0).cpu())
        b = beam.gloss_tokens
        g = greedy.gloss_tokens
        max_len = max(1, max(len(b), len(g)))
        overlap = sum(1 for i in range(min(len(b), len(g))) if b[i] == g[i])
        agreement = overlap / max_len

        if agreement >= settings.ENSEMBLE_MIN_AGREEMENT:
            tokens = b if beam.confidence >= greedy.confidence else g
            conf = max(beam.confidence, greedy.confidence)
        else:
            # Conservative fallback to common prefix when models disagree.
            common_prefix: List[str] = []
            for bi, gi in zip(b, g):
                if bi == gi:
                    common_prefix.append(bi)
                else:
                    break
            tokens = common_prefix
            conf = min(beam.confidence, greedy.confidence) * agreement

        return {
            "tokens": tokens,
            "confidence": self._apply_confidence_temperature(conf),
            "frame_confidences": beam.frame_confidences or greedy.frame_confidences,
            "agreement": agreement,
        }

    def _init_adaptive_state(self) -> Dict[str, float]:
        return {
            "strictness": 0.0,
            "ema_noise": 0.0,
            "ema_conf": settings.CONFIDENCE_THRESHOLD,
        }

    def _update_adaptive_strictness(
        self,
        confidence: float,
        frame_confidences: Optional[List[float]],
        adaptive_state: Optional[Dict[str, float]],
    ) -> float:
        if not settings.ENABLE_ADAPTIVE_FILTERING or adaptive_state is None:
            return 0.0

        arr = np.array(frame_confidences or [confidence], dtype=np.float32)
        mean_conf = float(np.mean(arr)) if arr.size else confidence
        var_conf = float(np.var(arr)) if arr.size else 0.0
        noise = var_conf + max(0.0, settings.CONFIDENCE_THRESHOLD - mean_conf)

        alpha = 0.35
        adaptive_state["ema_noise"] = (1.0 - alpha) * adaptive_state.get("ema_noise", 0.0) + alpha * noise
        adaptive_state["ema_conf"] = (1.0 - alpha) * adaptive_state.get("ema_conf", settings.CONFIDENCE_THRESHOLD) + alpha * confidence

        strictness = float(adaptive_state.get("strictness", 0.0))
        if confidence < settings.ADAPTIVE_CONF_LOW or adaptive_state["ema_noise"] > settings.ADAPTIVE_NOISE_VAR_THRESHOLD:
            strictness += settings.ADAPTIVE_STRICTNESS_STEP_UP
        elif confidence > settings.ADAPTIVE_CONF_HIGH and adaptive_state["ema_noise"] < settings.ADAPTIVE_NOISE_VAR_THRESHOLD * 0.7:
            strictness -= settings.ADAPTIVE_STRICTNESS_STEP_DOWN

        strictness = float(np.clip(strictness, 0.0, 1.0))
        adaptive_state["strictness"] = strictness
        return strictness

    def _adaptive_filter_params(self, strictness: float) -> Dict[str, float]:
        threshold = settings.CONFIDENCE_THRESHOLD + settings.ADAPTIVE_THRESHOLD_BOOST_MAX * strictness
        max_tokens = int(max(3, round(settings.GLOSS_MAX_TOKENS * (1.0 - settings.ADAPTIVE_MAXTOK_REDUCTION * strictness))))
        min_votes = int(settings.GLOSS_MIN_VOTES + round(settings.ADAPTIVE_VOTES_BONUS_MAX * strictness))
        return {
            "threshold": float(np.clip(threshold, 0.0, 0.99)),
            "max_tokens": max_tokens,
            "min_votes": min_votes,
        }

    def _filter_gloss_tokens(
        self,
        gloss_tokens: List[str],
        confidence: float,
        frame_confidences: Optional[List[float]] = None,
        adaptive_state: Optional[Dict[str, float]] = None,
        caption_filter: Optional[CaptionPostProcessor] = None,
    ) -> List[str]:
        """Filter noisy tokens to reduce unwanted sign outputs."""
        strictness = self._update_adaptive_strictness(confidence, frame_confidences, adaptive_state)
        adaptive = self._adaptive_filter_params(strictness)

        if confidence < adaptive["threshold"]:
            return []

        if not settings.ENABLE_GLOSS_FILTER:
            return gloss_tokens

        filtered: List[str] = []
        prev = None
        for token in gloss_tokens:
            token_up = token.strip().upper()
            if not token_up:
                continue
            if token_up in self.blocked_glosses:
                continue
            if prev == token_up:
                continue
            filtered.append(token_up)
            prev = token_up
            if len(filtered) >= adaptive["max_tokens"]:
                break
        if settings.ENABLE_TEMPORAL_GLOSS_VOTING:
            target_filter = caption_filter or self.caption_filter
            return target_filter.update(
                filtered,
                confidence,
                min_votes_override=adaptive["min_votes"],
            )
        return filtered

    def _load_vocab(self, vocab_file: Optional[str]) -> List[str]:
        candidates = []
        if vocab_file:
            candidates.append(Path(vocab_file))
        candidates.append(Path(settings.ISIGN_VOCAB_FILE))

        for candidate in candidates:
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        vocab = json.load(f)
                    if isinstance(vocab, list) and vocab:
                        logger.info("Loaded vocabulary from %s (%d tokens)", candidate, len(vocab))
                        return vocab
                except Exception as exc:
                    logger.warning("Failed to read vocab from %s: %s", candidate, exc)

        logger.warning(
            "No vocab file found. Falling back to synthetic vocab of size %d",
            settings.INFERENCE_DEFAULT_VOCAB_SIZE,
        )
        return [f"WORD_{i}" for i in range(settings.INFERENCE_DEFAULT_VOCAB_SIZE)]

    def _load_state_dict(self, module: torch.nn.Module, checkpoint_path: str, module_name: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            logger.info("%s checkpoint not found at %s (skipping)", module_name, path)
            return

        try:
            checkpoint = torch.load(str(path), map_location=self.device)
            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get("model_state_dict")
                    or checkpoint.get("state_dict")
                    or checkpoint.get("model")
                    or checkpoint
                )
            missing, unexpected = module.load_state_dict(state_dict, strict=False)
            logger.info(
                "Loaded %s checkpoint from %s (missing=%d, unexpected=%d)",
                module_name,
                path,
                len(missing),
                len(unexpected),
            )
        except Exception as exc:
            logger.warning("Failed to load %s checkpoint from %s: %s", module_name, path, exc)

    def _load_pipeline_checkpoints(self) -> None:
        self._load_state_dict(self.rgb_stream, settings.RGB_MODEL_PATH, "RGB stream")
        self._load_state_dict(self.pose_stream, settings.POSE_MODEL_PATH, "Pose stream")
        self._load_state_dict(self.fusion, settings.FUSION_MODEL_PATH, "Fusion")
        self._load_state_dict(self.temporal_model, settings.SEQUENCE_MODEL_PATH, "Temporal model")

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
        logger.info(f"Processing video: {video_path}")

        total_start = time.time()
        module1_time = module2_time = module3_time = module4_time = 0.0

        try:
            self.caption_filter.reset()
            adaptive_state = self._init_adaptive_state()
            # Module 1: Load and preprocess video
            if settings.ENABLE_METRICS:
                self.metrics.start_timer("module1")
            frames, fps = self.video_loader.load_video(video_path)
            frames = self.frame_sampler.sample_frames(
                frames, original_fps=fps, target_fps=settings.FPS_TARGET
            )

            rgb_frames: List[torch.Tensor] = []
            pose_frames: List[torch.Tensor] = []

            for frame_rgb in frames:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                result = self.video_preprocessor.process(frame_bgr)
                if result.kept and result.rgb_tensor is not None and result.pose_tensor is not None:
                    rgb_frames.append(result.rgb_tensor)
                    pose_frames.append(result.pose_tensor)

            if len(rgb_frames) < settings.MIN_VALID_FRAMES:
                return {
                    "gloss": [],
                    "sentence": "",
                    "confidence": 0.0,
                    "error": "Not enough valid frames extracted",
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
            decoded = self._decode_with_ensemble(logits)
            gloss_tokens = self._filter_gloss_tokens(
                decoded["tokens"],
                decoded["confidence"],
                frame_confidences=decoded["frame_confidences"],
                adaptive_state=adaptive_state,
            )
            confidence = decoded["confidence"]

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
                frame_confidences=decoded["frame_confidences"],
            )

            logger.info(
                f"Inference complete: {len(gloss_tokens)} glosses, confidence={confidence:.2f}"
            )
            return {
                "gloss": gloss_tokens,
                "sentence": sentence,
                "confidence": confidence,
                "frame_count": len(frames),
                "fps": float(settings.FPS_TARGET),
                "attention": {
                    "alpha": alpha.cpu().numpy().tolist() if alpha is not None else None,
                    "beta": beta.cpu().numpy().tolist() if beta is not None else None,
                },
                "decode_agreement": decoded.get("agreement", 1.0),
            }
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return {"gloss": [], "sentence": "", "confidence": 0.0, "error": str(e)}

    async def process_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Process sequence of frames through pipeline.
        """
        logger.info(f"Processing {len(frames)} frames")

        total_start = time.time()
        module1_time = module2_time = module3_time = module4_time = 0.0

        try:
            self.caption_filter.reset()
            adaptive_state = self._init_adaptive_state()
            if settings.ENABLE_METRICS:
                self.metrics.start_timer("module1")

            rgb_frames: List[torch.Tensor] = []
            pose_frames: List[torch.Tensor] = []

            for frame in frames:
                result = self.video_preprocessor.process(frame)
                if result.kept and result.rgb_tensor is not None and result.pose_tensor is not None:
                    rgb_frames.append(result.rgb_tensor)
                    pose_frames.append(result.pose_tensor)

            if len(rgb_frames) < settings.MIN_VALID_FRAMES:
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

            decoded = self._decode_with_ensemble(logits)
            gloss_tokens = self._filter_gloss_tokens(
                decoded["tokens"],
                decoded["confidence"],
                frame_confidences=decoded["frame_confidences"],
                adaptive_state=adaptive_state,
            )

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
                confidence=decoded["confidence"],
                frame_confidences=decoded["frame_confidences"],
            )

            return {
                "gloss": gloss_tokens,
                "sentence": sentence,
                "confidence": decoded["confidence"],
                "fps": float(settings.FPS_TARGET),
                "decode_agreement": decoded.get("agreement", 1.0),
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
        if state is None:
            state = {
                "buffer": SlidingWindowBuffer(window_size=64, stride=32),
                "last_gloss": [],
                "last_sentence": "",
                "adaptive": self._init_adaptive_state(),
                "caption_filter": CaptionPostProcessor(
                    max_history=settings.GLOSS_HISTORY_SIZE,
                    min_confidence=settings.CONFIDENCE_THRESHOLD,
                    vote_window=settings.GLOSS_VOTE_WINDOW,
                    min_votes=settings.GLOSS_MIN_VOTES,
                ),
            }

        try:
            result = self.video_preprocessor.process(frame)
            if not result.kept:
                return {
                    "gloss": state["last_gloss"],
                    "sentence": state["last_sentence"],
                    "confidence": 0.0,
                    "partial": True,
                    "state": state,
                }

            if result.rgb_tensor is None or result.pose_tensor is None:
                return {
                    "gloss": state["last_gloss"],
                    "sentence": state["last_sentence"],
                    "confidence": 0.0,
                    "partial": True,
                    "state": state,
                }

            window = state["buffer"].add(result.rgb_tensor, result.pose_tensor)
            if window is None:
                return {
                    "gloss": state["last_gloss"],
                    "sentence": state["last_sentence"],
                    "confidence": 0.0,
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

            decoded = self._decode_with_ensemble(logits)
            gloss_tokens = [
                tok for tok in self._filter_gloss_tokens(
                    decoded["tokens"],
                    decoded["confidence"],
                    frame_confidences=decoded["frame_confidences"],
                    adaptive_state=state.get("adaptive"),
                    caption_filter=state.get("caption_filter"),
                )
                if tok not in self.blocked_glosses
            ]
            sentence = await self._build_sentence(gloss_tokens)

            state["last_gloss"] = gloss_tokens
            state["last_sentence"] = sentence

            return {
                "gloss": gloss_tokens,
                "sentence": sentence,
                "confidence": decoded["confidence"],
                "fps": float(settings.FPS_TARGET),
                "decode_agreement": decoded.get("agreement", 1.0),
                "partial": state["buffer"].counts() < state["buffer"].window_size,
                "state": state,
            }
        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            return {
                "gloss": state.get("last_gloss", []),
                "sentence": state.get("last_sentence", ""),
                "confidence": 0.0,
                "partial": True,
                "state": state,
                "error": str(e),
            }
