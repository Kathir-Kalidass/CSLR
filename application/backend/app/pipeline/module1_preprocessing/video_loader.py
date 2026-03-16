"""
Video Loader
Loads and decodes video files from paths or byte streams
"""

import cv2
import numpy as np
import tempfile
import os
from typing import List, Tuple, Optional
from app.core.logging import logger


class VideoLoader:
    """Loads video and extracts frames"""
    
    def __init__(self, target_fps: int = 25):
        self.target_fps = target_fps
    
    def load_video(self, video_path: str) -> Tuple[List[np.ndarray], float]:
        """
        Load video and extract frames
        
        Args:
            video_path: Path to video file
        
        Returns:
            Tuple of (frames list, original fps)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        logger.info(f"Loaded {len(frames)} frames at {original_fps} FPS")
        
        return frames, original_fps
    
    def load_from_bytes(self, video_bytes: bytes) -> Tuple[List[np.ndarray], float]:
        """
        Load video from byte stream (for uploaded files)
        
        Args:
            video_bytes: Video file as bytes
        
        Returns:
            Tuple of (frames list, original fps)
        """
        temp_path = None
        try:
            # Write bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(video_bytes)
                temp_path = tmp.name
            
            # Load from temp file
            frames, fps = self.load_video(temp_path)
            
            logger.info(f"Loaded {len(frames)} frames from byte stream")
            return frames, fps
        
        except Exception as e:
            logger.error(f"Failed to load video from bytes: {e}")
            raise
        
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_path}: {e}")
