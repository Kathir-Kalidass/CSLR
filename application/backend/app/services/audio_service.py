"""
Audio Service (Enhanced)
Non-blocking Text-to-Speech with queue-based processing
"""

import queue
import threading
from typing import Optional
from app.core.config import settings
from app.core.logging import logger


class AudioService:
    """
    Enhanced Text-to-Speech service (non-blocking)
    Converts translated text to audio with threaded queue processing
    """
    
    def __init__(self):
        self.enabled = True
        self.rate = settings.TTS_RATE
        self.volume = settings.TTS_VOLUME
        
        # Queue-based non-blocking TTS
        self._q: queue.Queue[str] = queue.Queue(maxsize=16)
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._engine = None
        
        if self.enabled:
            self._start_worker()
    
    def _start_worker(self) -> None:
        """Start TTS worker thread"""
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()
        logger.info("TTS worker thread started")
    
    def _loop(self) -> None:
        """Worker thread loop for processing TTS queue"""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)
            logger.info(f"TTS engine initialized: rate={self.rate}, volume={self.volume}")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self._engine = None
        
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            
            if not text:
                continue
            
            if self._engine is None:
                logger.warning(f"[TTS disabled] {text}")
                continue
            
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS synthesis failed: {e}, text: {text}")
    
    async def synthesize(self, text: str, save_path: Optional[str] = None) -> bool:
        """
        Synthesize speech from text (non-blocking)
        
        Args:
            text: Input text
            save_path: Optional path to save audio file (not supported in non-blocking mode)
        
        Returns:
            Success status
        """
        if not text or not self.enabled:
            return False
        
        try:
            # Drop oldest if queue is full
            if self._q.full():
                try:
                    _ = self._q.get_nowait()
                except queue.Empty:
                    pass
            
            self._q.put_nowait(text)
            return True
        except Exception as e:
            logger.error(f"Failed to queue TTS: {e}")
            return False
    
    async def synthesize_streaming(self, text: str):
        """Synthesize speech for streaming (non-blocking)"""
        return await self.synthesize(text)
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable TTS"""
        self.enabled = enabled
        logger.info(f"TTS {'enabled' if enabled else 'disabled'}")
    
    def close(self) -> None:
        """Cleanup TTS resources"""
        self._stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)
        logger.info("TTS worker thread stopped")
