"""
Streaming Service
Manages WebSocket streaming and frame buffering
"""

from typing import Dict, List, Deque
from collections import deque
import asyncio
from app.core.logging import logger
from app.core.config import settings


class StreamingService:
    """
    Manages streaming inference
    Handles frame buffering, temporal context, etc.
    """
    
    def __init__(self):
        self.clip_length = settings.CLIP_LENGTH
        self.active_streams: Dict[str, StreamState] = {}
    
    def create_stream(self, client_id: str) -> 'StreamState':
        """Create new streaming state for client"""
        state = StreamState(clip_length=self.clip_length)
        self.active_streams[client_id] = state
        logger.info(f"Created stream state for client: {client_id}")
        return state
    
    def get_stream(self, client_id: str) -> 'StreamState':
        """Get existing stream state"""
        if client_id not in self.active_streams:
            return self.create_stream(client_id)
        return self.active_streams[client_id]
    
    def close_stream(self, client_id: str):
        """Close and cleanup stream state"""
        if client_id in self.active_streams:
            del self.active_streams[client_id]
            logger.info(f"Closed stream for client: {client_id}")


class StreamState:
    """Maintains state for a single streaming session"""
    
    def __init__(self, clip_length: int = 32):
        self.clip_length = clip_length
        self.frame_buffer: Deque = deque(maxlen=clip_length)
        self.gloss_buffer: List[str] = []
        self.last_prediction = None
        self.frame_count = 0
        self.pipeline_state = None
    
    def add_frame(self, frame):
        """Add frame to buffer"""
        self.frame_buffer.append(frame)
        self.frame_count += 1
    
    def is_ready(self) -> bool:
        """Check if buffer has enough frames for inference"""
        return len(self.frame_buffer) >= self.clip_length
    
    def reset(self):
        """Reset state"""
        self.frame_buffer.clear()
        self.gloss_buffer.clear()
        self.last_prediction = None
        self.frame_count = 0
