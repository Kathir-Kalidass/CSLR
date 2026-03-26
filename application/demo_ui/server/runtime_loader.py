from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .deps import YOLO, cv2, mp, np, torch, transforms
from .runtime_helpers import build_id_to_token, infer_temporal_layers, sub_state


def initialize_runtime(runtime: Any, checkpoint_path: Path) -> tuple[bool, str, str]:
    if not checkpoint_path.exists():
        return False, "checkpoint_missing", ""

    if any(dep is None for dep in (cv2, np, torch, transforms)):
        return False, "runtime_dependencies_missing", ""

    try:
        backend_root = PROJECT_ROOT / "application" / "backend"
        backend_app_root = backend_root / "app"
        if str(backend_app_root) not in sys.path:
            sys.path.insert(0, str(backend_app_root))

        def _load_symbol(module_path: Path, symbol_name: str) -> Any:
            module_name = f"demo_runtime_{module_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, str(module_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load module from {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, symbol_name):
                raise AttributeError(f"{symbol_name} not found in {module_path}")
            return getattr(module, symbol_name)

        pipeline_root = backend_app_root / "pipeline"
        FeatureFusion = _load_symbol(pipeline_root / "module2_feature" / "fusion.py", "FeatureFusion")
        PoseStream = _load_symbol(pipeline_root / "module2_feature" / "pose_stream.py", "PoseStream")
        RGBStream = _load_symbol(pipeline_root / "module2_feature" / "rgb_stream.py", "RGBStream")
        TemporalModel = _load_symbol(pipeline_root / "module3_sequence" / "temporal_model.py", "TemporalModel")

        runtime.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=runtime.device)
        model_state = checkpoint.get("model_state", {})
        if not isinstance(model_state, dict) or not model_state:
            return False, "invalid_checkpoint_state", ""

        # Infer architecture config from checkpoint tensors.
        feature_dim = int(model_state["fusion.rgb_gate.weight"].shape[1])
        runtime.pose_input_dim = int(model_state["pose_stream.mlp.0.weight"].shape[1])
        pose_hidden_1 = int(model_state["pose_stream.mlp.0.weight"].shape[0])
        pose_hidden_2 = int(model_state["pose_stream.mlp.4.weight"].shape[0])
        temporal_classifier_out = int(model_state["temporal.classifier.weight"].shape[0])
        temporal_hidden = int(model_state["temporal.classifier.weight"].shape[1] // 2)
        vocab_size = temporal_classifier_out - 1
        temporal_layers = infer_temporal_layers(model_state)

        runtime.rgb_stream = RGBStream(
            backbone="resnet18",
            feature_dim=feature_dim,
            pretrained=False,
            freeze_backbone=False,
            dropout=0.0,
            backbone_chunk_size=0,
        ).to(runtime.device)

        runtime.pose_stream = PoseStream(
            input_dim=runtime.pose_input_dim,
            hidden_dims=[pose_hidden_1, pose_hidden_2],
            feature_dim=feature_dim,
            dropout=0.0,
        ).to(runtime.device)

        runtime.fusion = FeatureFusion(
            rgb_dim=feature_dim,
            pose_dim=feature_dim,
            fusion_dim=feature_dim,
            fusion_type="gated_attention",
        ).to(runtime.device)

        runtime.temporal = TemporalModel(
            input_dim=feature_dim,
            hidden_dim=temporal_hidden,
            num_layers=temporal_layers,
            vocab_size=vocab_size,
            model_type="bilstm",
            dropout=0.0,
        ).to(runtime.device)

        runtime.rgb_stream.load_state_dict(sub_state(model_state, "rgb_stream"), strict=True)
        runtime.pose_stream.load_state_dict(sub_state(model_state, "pose_stream"), strict=True)
        runtime.fusion.load_state_dict(sub_state(model_state, "fusion"), strict=True)
        runtime.temporal.load_state_dict(sub_state(model_state, "temporal"), strict=True)

        runtime.rgb_stream.eval()
        runtime.pose_stream.eval()
        runtime.fusion.eval()
        runtime.temporal.eval()

        runtime.id_to_token = build_id_to_token(checkpoint, vocab_size=vocab_size)

        yolo_pose_path = PROJECT_ROOT / "application" / "backend" / "yolov8n-pose.pt"
        if YOLO is not None and yolo_pose_path.exists():
            runtime._pose_model = YOLO(str(yolo_pose_path))
        elif mp is not None and getattr(mp, "solutions", None) is not None:
            runtime._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        # MediaPipe Hands for finger landmark detection (Tasks API for 0.10+)
        hand_model_path = Path(__file__).parent / "hand_landmarker.task"
        if mp is not None and hand_model_path.exists():
            try:
                BaseOptions = mp.tasks.BaseOptions
                HandLandmarker = mp.tasks.vision.HandLandmarker
                HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode
                options = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(hand_model_path)),
                    running_mode=VisionRunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.3,
                    min_hand_presence_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
                runtime._mp_hands = HandLandmarker.create_from_options(options)
            except Exception:
                runtime._mp_hands = None

        runtime._rgb_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        return True, "ready", ""
    except Exception as exc:  # pragma: no cover - runtime dependent
        return False, f"init_error:{type(exc).__name__}", str(exc)
