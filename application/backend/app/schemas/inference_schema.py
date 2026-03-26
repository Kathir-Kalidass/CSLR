"""
Inference Request/Response Schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class InferenceRequest(BaseModel):
    """Request schema for frame-based inference"""
    frames: List[str] = Field(..., description="List of base64-encoded frames")
    fps: Optional[float] = Field(30.0, description="Frames per second")
    
    class Config:
        json_schema_extra = {
            "example": {
                "frames": ["base64_encoded_frame1", "base64_encoded_frame2"],
                "fps": 30.0
            }
        }


class InferenceResponse(BaseModel):
    """Response schema for inference results"""
    gloss: List[str] = Field(..., description="Recognized gloss sequence")
    sentence: str = Field(..., description="Translated sentence")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    fps: float = Field(..., description="Processing FPS")
    processing_time: float = Field(..., description="Total processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "gloss": ["HELLO", "WORLD", "HOW", "ARE", "YOU"],
                "sentence": "Hello world, how are you?",
                "confidence": 0.92,
                "fps": 28.5,
                "processing_time": 1.2
            }
        }


class StreamingInferenceResponse(BaseModel):
    """Response schema for streaming inference"""
    gloss: List[str]
    sentence: str
    confidence: float
    timestamp: float
    fps: float
    module_timings: Optional[dict] = None
