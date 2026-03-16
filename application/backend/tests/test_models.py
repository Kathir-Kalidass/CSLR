"""
Test Model Loading and Management
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path


def test_find_latest_checkpoint(tmp_path):
    """Test checkpoint detection"""
    from app.models.load_model import find_latest_checkpoint
    
    # Create dummy checkpoint files
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    
    # Create some checkpoint files
    (checkpoint_dir / "rgb_epoch_1.pth").touch()
    (checkpoint_dir / "rgb_epoch_2.pth").touch()
    (checkpoint_dir / "rgb_best.pth").touch()
    
    # Find latest checkpoint
    latest = find_latest_checkpoint(str(checkpoint_dir), "rgb")
    
    assert latest is not None
    assert "rgb" in latest


def test_find_latest_checkpoint_no_files(tmp_path):
    """Test checkpoint detection with no files"""
    from app.models.load_model import find_latest_checkpoint
    
    checkpoint_dir = tmp_path / "empty_checkpoints"
    checkpoint_dir.mkdir()
    
    result = find_latest_checkpoint(str(checkpoint_dir), "rgb")
    
    assert result is None


def test_find_latest_checkpoint_best_priority(tmp_path):
    """Test that _best.pth is found when no timestamped checkpoints"""
    from app.models.load_model import find_latest_checkpoint
    
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    
    # Only create best checkpoint
    (checkpoint_dir / "pose_best.pth").touch()
    
    result = find_latest_checkpoint(str(checkpoint_dir), "pose")
    
    assert result is not None
    assert "pose_best.pth" in result


def test_load_model_checkpoint():
    """Test loading model checkpoint"""
    from app.models.load_model import load_model_checkpoint
    
    # Create temporary checkpoint
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        tmp_path = tmp.name
        
        # Save dummy checkpoint
        checkpoint = {
            'epoch': 10,
            'model_state_dict': {},
            'optimizer_state_dict': {}
        }
        torch.save(checkpoint, tmp_path)
    
    try:
        # Load checkpoint
        loaded = load_model_checkpoint(tmp_path, device='cpu')
        
        assert 'epoch' in loaded
        assert loaded['epoch'] == 10
        assert 'model_state_dict' in loaded
    
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_load_model_checkpoint_file_not_found():
    """Test loading non-existent checkpoint"""
    from app.models.load_model import load_model_checkpoint
    
    with pytest.raises(FileNotFoundError):
        load_model_checkpoint("/non/existent/path.pth")


def test_load_rgb_model():
    """Test RGB model loading"""
    from app.models.load_model import load_rgb_model
    
    # Load model without checkpoint
    model = load_rgb_model(device='cpu', checkpoint_path=None)
    
    assert model is not None
    assert hasattr(model, 'forward')


def test_load_pose_model():
    """Test Pose model loading"""
    from app.models.load_model import load_pose_model
    
    model = load_pose_model(device='cpu', checkpoint_path=None)
    
    assert model is not None
    assert hasattr(model, 'forward')


def test_load_fusion_model():
    """Test Fusion model loading"""
    from app.models.load_model import load_fusion_model
    
    model = load_fusion_model(device='cpu', checkpoint_path=None)
    
    assert model is not None
    assert hasattr(model, 'forward')


def test_load_sequence_model():
    """Test Sequence model loading"""
    from app.models.load_model import load_sequence_model
    
    model = load_sequence_model(device='cpu', vocab_size=100, checkpoint_path=None)
    
    assert model is not None
    assert hasattr(model, 'forward')


def test_model_eval_mode():
    """Test that loaded models are in eval mode"""
    from app.models.load_model import load_rgb_model
    
    model = load_rgb_model(device='cpu')
    
    # Model should be in eval mode
    assert not model.training
