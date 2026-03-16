"""
iSign Training API Endpoints
Launches preprocess/train scripts with modular hyperparameter control.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.core.config import settings
from app.core.logging import logger

router = APIRouter(prefix="/training", tags=["training"])


class ISignPreprocessConfig(BaseModel):
    isign_dir: str = Field(default_factory=lambda: settings.ISIGN_DATA_DIR)
    out_dir: str = Field(default_factory=lambda: settings.ISIGN_PROCESSED_DIR)
    max_frames: int = 64
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    workers: int = 4
    seed: int = 42
    max_samples: int = 0
    skip_frames: bool = False
    min_frames: int = 8
    min_gloss_tokens: int = 1
    max_gloss_tokens: int = 40
    modality_mode: str = "auto"

    @model_validator(mode="after")
    def _validate(self):
        if not (0.0 < self.val_ratio < 0.5):
            raise ValueError("val_ratio must be in (0, 0.5)")
        if not (0.0 < self.test_ratio < 0.5):
            raise ValueError("test_ratio must be in (0, 0.5)")
        if self.val_ratio + self.test_ratio >= 0.9:
            raise ValueError("val_ratio + test_ratio must be < 0.9")
        if self.max_frames < 8:
            raise ValueError("max_frames must be >= 8")
        if self.min_frames < 1:
            raise ValueError("min_frames must be >= 1")
        if self.min_gloss_tokens < 1:
            raise ValueError("min_gloss_tokens must be >= 1")
        if self.max_gloss_tokens < self.min_gloss_tokens:
            raise ValueError("max_gloss_tokens must be >= min_gloss_tokens")
        if self.modality_mode not in {"auto", "multimodal", "rgb", "pose"}:
            raise ValueError("modality_mode must be one of auto, multimodal, rgb, pose")
        if self.workers < 0:
            raise ValueError("workers must be >= 0")
        return self


class ISignTrainingConfig(BaseModel):
    data_dir: str = Field(default_factory=lambda: settings.ISIGN_PROCESSED_DIR)
    vocab: Optional[str] = None
    num_frames: int = 64
    frame_size_h: int = 224
    frame_size_w: int = 224
    max_samples: int = 0
    use_rgb: bool = True
    use_pose: bool = True
    auto_disable_pose_if_missing: bool = True
    pretrained_cnn: bool = True
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    clip_grad: float = 5.0
    warmup_epochs: int = 3
    use_amp: bool = True
    ckpt_dir: str = Field(default_factory=lambda: settings.ISIGN_TRAINING_OUTPUT_DIR)
    save_every_epoch: bool = True
    workers: int = 2
    seed: int = 42
    device: str = "auto"
    require_cuda: bool = False
    allow_tf32: bool = True
    resume: Optional[str] = None
    preprocess_first: bool = False
    preprocess: ISignPreprocessConfig = Field(default_factory=ISignPreprocessConfig)

    @model_validator(mode="after")
    def _validate(self):
        if not self.use_rgb and not self.use_pose:
            raise ValueError("At least one modality must be enabled (use_rgb or use_pose)")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not (0.0 <= self.dropout < 0.9):
            raise ValueError("dropout must be in [0.0, 0.9)")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, cpu")
        return self


class TrainingStatus(BaseModel):
    status: str
    phase: str
    pid: int
    started_at: float
    config: Dict[str, Any]
    message: str


training_state: Dict[str, Any] = {
    "active": False,
    "phase": "idle",
    "process": None,
    "started_at": 0.0,
    "config": {},
}


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dataset_download_status(isign_dir: Path) -> Dict[str, Any]:
    pending_ext = settings.ISIGN_PENDING_EXT
    pose_parts = sorted(isign_dir.glob("iSign-poses_v1.1_part_*"))
    pending_parts = sorted(isign_dir.glob(f"*{pending_ext}"))
    poses_dir = isign_dir / "poses"
    videos_dir = isign_dir / "videos"
    pose_npy_count = len(list(poses_dir.glob("**/*.npy"))) if poses_dir.exists() else 0
    video_count = 0
    if videos_dir.exists():
        video_count = sum(len(list(videos_dir.glob(f"*{ext}"))) for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"])

    metadata_ready = (isign_dir / "iSign_v1.1.csv").exists()

    return {
        "isign_dir": str(isign_dir),
        "metadata_ready": metadata_ready,
        "pose_parts_present": len(pose_parts),
        "pending_download_parts": [p.name for p in pending_parts],
        "poses_extracted": poses_dir.exists() and pose_npy_count > 0,
        "pose_file_count": pose_npy_count,
        "videos_extracted": videos_dir.exists() and video_count > 0,
        "video_file_count": video_count,
        "cuda_available": __import__("torch").cuda.is_available(),
    }


def _to_preprocess_command(cfg: ISignPreprocessConfig) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/preprocess_isign.py",
        "--isign-dir",
        cfg.isign_dir,
        "--out-dir",
        cfg.out_dir,
        "--max-frames",
        str(cfg.max_frames),
        "--val-ratio",
        str(cfg.val_ratio),
        "--test-ratio",
        str(cfg.test_ratio),
        "--workers",
        str(cfg.workers),
        "--seed",
        str(cfg.seed),
        "--min-frames",
        str(cfg.min_frames),
        "--min-gloss-tokens",
        str(cfg.min_gloss_tokens),
        "--max-gloss-tokens",
        str(cfg.max_gloss_tokens),
        "--modality-mode",
        cfg.modality_mode,
    ]
    if cfg.max_samples > 0:
        cmd.extend(["--max-samples", str(cfg.max_samples)])
    if cfg.skip_frames:
        cmd.append("--skip-frames")
    return cmd


def _to_training_command(cfg: ISignTrainingConfig) -> List[str]:
    cmd = [
        sys.executable,
        "scripts/train_isign.py",
        "--data-dir",
        cfg.data_dir,
        "--num-frames",
        str(cfg.num_frames),
        "--frame-size",
        str(cfg.frame_size_h),
        str(cfg.frame_size_w),
        "--hidden-dim",
        str(cfg.hidden_dim),
        "--num-layers",
        str(cfg.num_layers),
        "--dropout",
        str(cfg.dropout),
        "--epochs",
        str(cfg.epochs),
        "--batch-size",
        str(cfg.batch_size),
        "--lr",
        str(cfg.learning_rate),
        "--weight-decay",
        str(cfg.weight_decay),
        "--clip-grad",
        str(cfg.clip_grad),
        "--warmup-epochs",
        str(cfg.warmup_epochs),
        "--ckpt-dir",
        cfg.ckpt_dir,
        "--save-every-epoch" if cfg.save_every_epoch else "--no-save-every-epoch",
        "--workers",
        str(cfg.workers),
        "--seed",
        str(cfg.seed),
        "--device",
        cfg.device,
    ]
    if cfg.vocab:
        cmd.extend(["--vocab", cfg.vocab])
    if cfg.max_samples > 0:
        cmd.extend(["--max-samples", str(cfg.max_samples)])
    if not cfg.use_rgb:
        cmd.append("--no-rgb")
    if not cfg.use_pose:
        cmd.append("--no-pose")
    if not cfg.pretrained_cnn:
        cmd.append("--no-pretrained")
    if not cfg.use_amp:
        cmd.append("--no-amp")
    if cfg.require_cuda:
        cmd.append("--require-cuda")
    if not cfg.allow_tf32:
        cmd.append("--no-allow-tf32")
    if cfg.resume:
        cmd.extend(["--resume", cfg.resume])
    return cmd


def _launch_process(cmd: List[str], phase: str, config_dump: Dict[str, Any]) -> int:
    backend_root = _backend_root()
    logger.info("Launching %s job: %s", phase, " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(backend_root))
    training_state["active"] = True
    training_state["phase"] = phase
    training_state["process"] = proc
    training_state["started_at"] = time.time()
    training_state["config"] = config_dump
    return proc.pid


def _refresh_state() -> None:
    proc = training_state.get("process")
    if not proc:
        return
    code = proc.poll()
    if code is None:
        return
    training_state["active"] = False
    training_state["phase"] = "completed" if code == 0 else "failed"


@router.post("/preprocess")
async def start_preprocess(config: ISignPreprocessConfig):
    if training_state["active"]:
        raise HTTPException(status_code=400, detail="Another training/preprocess job is already running")
    cmd = _to_preprocess_command(config)
    pid = _launch_process(cmd, phase="preprocess", config_dump=config.model_dump())
    return {
        "status": "started",
        "phase": "preprocess",
        "pid": pid,
        "message": "iSign preprocessing started",
    }


@router.post("/start")
async def start_training(config: ISignTrainingConfig):
    if training_state["active"]:
        raise HTTPException(status_code=400, detail="Training already in progress")

    backend_root = _backend_root()
    isign_dir = (backend_root / settings.ISIGN_DATA_DIR).resolve()
    ds_status = _dataset_download_status(isign_dir)

    if settings.ISIGN_STRICT_DATA_CHECK and not ds_status["metadata_ready"]:
        raise HTTPException(status_code=400, detail=f"Missing iSign metadata CSV in {isign_dir}")

    if ds_status["pending_download_parts"]:
        logger.warning(
            "Detected pending iSign downloads: %s",
            ds_status["pending_download_parts"],
        )

    if config.use_pose and config.auto_disable_pose_if_missing and ds_status["pose_file_count"] < settings.ISIGN_MIN_READY_POSE_FILES:
        logger.warning("Poses are not ready yet. Auto-switching to RGB-only training for now.")
        config = config.model_copy(update={"use_pose": False})

    if not config.preprocess_first:
        data_dir = (backend_root / config.data_dir).resolve()
        required = [data_dir / "train.json", data_dir / "val.json", data_dir / "test.json", data_dir / "vocab.json"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Processed dataset is incomplete. Missing files: "
                    + ", ".join(missing)
                    + ". Run /training/preprocess first or set preprocess_first=true."
                ),
            )

    if config.preprocess_first:
        preprocess_cmd = _to_preprocess_command(config.preprocess)
        logger.info("Running preprocess step before training")
        preprocess_result = subprocess.run(preprocess_cmd, cwd=str(backend_root), check=False)
        if preprocess_result.returncode != 0:
            raise HTTPException(status_code=500, detail="Preprocess step failed. Check server logs.")

    cmd = _to_training_command(config)
    pid = _launch_process(cmd, phase="train", config_dump=config.model_dump())
    return {
        "status": "started",
        "phase": "train",
        "pid": pid,
        "dataset_status": ds_status,
        "effective_use_pose": config.use_pose,
        "message": f"iSign training started for {config.epochs} epochs",
    }


@router.get("/dataset-status")
async def get_dataset_status(isign_dir: Optional[str] = None):
    backend_root = _backend_root()
    target = (backend_root / (isign_dir or settings.ISIGN_DATA_DIR)).resolve()
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"iSign directory not found: {target}")
    return _dataset_download_status(target)


@router.get("/ready")
async def get_training_readiness(data_dir: Optional[str] = None):
    backend_root = _backend_root()
    processed_dir = (backend_root / (data_dir or settings.ISIGN_PROCESSED_DIR)).resolve()
    required = [processed_dir / "train.json", processed_dir / "val.json", processed_dir / "test.json", processed_dir / "vocab.json"]
    missing = [str(p) for p in required if not p.exists()]

    import torch

    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "processed_dataset_dir": str(processed_dir),
        "processed_dataset_ready": len(missing) == 0,
        "missing_processed_files": missing,
        "recommended_output_dir": settings.ISIGN_TRAINING_OUTPUT_DIR,
    }


@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    _refresh_state()
    proc = training_state.get("process")
    if not proc:
        return TrainingStatus(
            status="idle",
            phase="idle",
            pid=0,
            started_at=0.0,
            config={},
            message="No active job",
        )

    status = "running" if training_state["active"] else training_state["phase"]
    return TrainingStatus(
        status=status,
        phase=training_state["phase"],
        pid=proc.pid,
        started_at=training_state["started_at"],
        config=training_state["config"],
        message=f"Current phase: {training_state['phase']}",
    )


@router.post("/stop")
async def stop_training():
    _refresh_state()
    proc = training_state.get("process")
    if not proc or not training_state["active"]:
        raise HTTPException(status_code=400, detail="No active training job")

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    training_state["active"] = False
    training_state["phase"] = "stopped"
    return {"status": "stopped", "message": "Training job stopped"}


@router.get("/checkpoints")
async def list_checkpoints(save_dir: str = "checkpoints/isign"):
    ckpt_dir = _backend_root() / save_dir
    if not ckpt_dir.exists():
        return {"checkpoints": [], "count": 0}

    checkpoints = sorted(
        [str(p.relative_to(_backend_root())) for p in ckpt_dir.rglob("*.pt")],
    )
    return {"checkpoints": checkpoints, "count": len(checkpoints)}


@router.get("/history")
async def get_training_history(
    save_dir: Optional[str] = None,
    limit: int = 200,
):
    backend_root = _backend_root()
    target_dir = backend_root / (save_dir or settings.ISIGN_TRAINING_OUTPUT_DIR)
    history_path = target_dir / "history.jsonl"
    if not history_path.exists():
        raise HTTPException(status_code=404, detail=f"Training history not found: {history_path}")

    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")

    rows: List[Dict[str, Any]] = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(rows)
    rows = rows[-limit:]
    best_row = min(rows, key=lambda r: r.get("val_wer", 1e9)) if rows else None

    return {
        "history_file": str(history_path),
        "total_epochs_logged": total,
        "returned_epochs": len(rows),
        "best_in_returned": best_row,
        "rows": rows,
    }


@router.get("/artifacts")
async def list_user_friendly_artifacts(
    save_dir: Optional[str] = None,
):
    backend_root = _backend_root()
    target_dir = backend_root / (save_dir or settings.ISIGN_TRAINING_OUTPUT_DIR)
    ops_guide = backend_root / "TRAINING_USER_OPERATIONS.md"
    presets = backend_root / "configs" / "isign_training_profiles.json"

    files = {
        "checkpoint_dir": str(target_dir),
        "best": str(target_dir / "best.pt"),
        "last": str(target_dir / "last.pt"),
        "history_jsonl": str(target_dir / "history.jsonl"),
        "history_csv": str(target_dir / "history.csv"),
        "results_json": str(target_dir / "results.json"),
        "train_config": str(target_dir / "train_config.json"),
        "operations_guide": str(ops_guide),
        "training_presets": str(presets),
    }

    return {
        "files": files,
        "exists": {k: Path(v).exists() for k, v in files.items()},
    }
