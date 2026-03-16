"""
iSign Training API Endpoints
Launches preprocess/train scripts with modular hyperparameter control.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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


class ISignTrainingConfig(BaseModel):
    data_dir: str = Field(default_factory=lambda: settings.ISIGN_PROCESSED_DIR)
    vocab: Optional[str] = None
    num_frames: int = 64
    frame_size_h: int = 224
    frame_size_w: int = 224
    max_samples: int = 0
    use_rgb: bool = True
    use_pose: bool = True
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
    ckpt_dir: str = "checkpoints/isign"
    workers: int = 2
    seed: int = 42
    resume: Optional[str] = None
    preprocess_first: bool = False
    preprocess: ISignPreprocessConfig = Field(default_factory=ISignPreprocessConfig)


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
        "--workers",
        str(cfg.workers),
        "--seed",
        str(cfg.seed),
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
        "message": f"iSign training started for {config.epochs} epochs",
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
