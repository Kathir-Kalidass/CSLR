"""
Performance Metrics
Tracks inference performance metrics
"""

import time
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class InferenceMetrics:
    """Stores metrics for a single inference"""
    total_time: float
    module1_time: float = 0.0
    module2_time: float = 0.0
    module3_time: float = 0.0
    module4_time: float = 0.0
    fps: float = 0.0
    num_frames: int = 0
    confidence: float = 0.0


class MetricsCollector:
    """
    Collects and aggregates performance metrics
    """
    
    def __init__(self):
        self.metrics_history: List[InferenceMetrics] = []
        self.current_metrics: Optional[Dict] = None
        self.timers: Dict[str, float] = {}
    
    def start_timer(self, name: str):
        """Start a named timer"""
        self.timers[f"{name}_start"] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and return elapsed time
        
        Args:
            name: Timer name
        
        Returns:
            Elapsed time in seconds
        """
        start_key = f"{name}_start"
        if start_key not in self.timers:
            return 0.0
        
        elapsed = time.time() - self.timers[start_key]
        del self.timers[start_key]
        
        return elapsed
    
    def record_inference(self, metrics: InferenceMetrics):
        """Record inference metrics"""
        self.metrics_history.append(metrics)
        
        # Keep only last 100
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
    
    def get_average_metrics(self, last_n: int = 10) -> Dict[str, float]:
        """
        Get average metrics over last N inferences
        
        Args:
            last_n: Number of recent inferences to average
        
        Returns:
            Dictionary of average metrics
        """
        if not self.metrics_history:
            return {}
        
        recent = self.metrics_history[-last_n:]
        
        return {
            "avg_total_time": sum(m.total_time for m in recent) / len(recent),
            "avg_module1_time": sum(m.module1_time for m in recent) / len(recent),
            "avg_module2_time": sum(m.module2_time for m in recent) / len(recent),
            "avg_module3_time": sum(m.module3_time for m in recent) / len(recent),
            "avg_module4_time": sum(m.module4_time for m in recent) / len(recent),
            "avg_fps": sum(m.fps for m in recent) / len(recent),
            "avg_confidence": sum(m.confidence for m in recent) / len(recent)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall metrics summary"""
        if not self.metrics_history:
            return {"total_inferences": 0}
        
        return {
            "total_inferences": len(self.metrics_history),
            "recent_averages": self.get_average_metrics(10),
            "all_time_averages": self.get_average_metrics(len(self.metrics_history))
        }
    
    def reset(self):
        """Reset all metrics"""
        self.metrics_history.clear()
        self.timers.clear()


# Global metrics collector
global_metrics = MetricsCollector()
