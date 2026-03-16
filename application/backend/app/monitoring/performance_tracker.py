"""
Performance Tracker
Tracks detailed performance across pipeline
"""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PerformanceRecord:
    """Single performance measurement"""
    name: str
    duration: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


class PerformanceTracker:
    """
    Tracks performance across the entire pipeline
    Provides detailed timing breakdowns
    """
    
    def __init__(self):
        self.records: List[PerformanceRecord] = []
        self.active_timers: Dict[str, float] = {}
        self.aggregated_stats: Dict[str, List[float]] = defaultdict(list)
    
    @contextmanager
    def track(self, name: str, metadata: Optional[Dict] = None):
        """
        Context manager for tracking code blocks
        
        Usage:
            with tracker.track("preprocessing"):
                # code to track
                pass
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.record(name, duration, metadata or {})
    
    def record(self, name: str, duration: float, metadata: Optional[Dict] = None):
        """Record a performance measurement"""
        record = PerformanceRecord(
            name=name,
            duration=duration,
            metadata=metadata or {}
        )
        
        self.records.append(record)
        self.aggregated_stats[name].append(duration)
        
        # Keep only last 1000 records
        if len(self.records) > 1000:
            self.records.pop(0)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a specific operation
        
        Args:
            name: Operation name
        
        Returns:
            Dictionary with min, max, avg, count
        """
        if name not in self.aggregated_stats:
            return {}
        
        timings = self.aggregated_stats[name]
        
        return {
            "count": len(timings),
            "min_ms": min(timings) * 1000,
            "max_ms": max(timings) * 1000,
            "avg_ms": (sum(timings) / len(timings)) * 1000,
            "total_s": sum(timings)
        }
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get stats for all tracked operations"""
        return {
            name: self.get_stats(name)
            for name in self.aggregated_stats.keys()
        }
    
    def get_recent(self, n: int = 10) -> List[PerformanceRecord]:
        """Get N most recent records"""
        return self.records[-n:]
    
    def reset(self):
        """Reset all tracking data"""
        self.records.clear()
        self.aggregated_stats.clear()
        self.active_timers.clear()
    
    def print_summary(self):
        """Print performance summary"""
        print("\n=== Performance Summary ===")
        for name, stats in self.get_all_stats().items():
            print(f"\n{name}:")
            print(f"  Count: {stats['count']}")
            print(f"  Avg: {stats['avg_ms']:.2f} ms")
            print(f"  Min: {stats['min_ms']:.2f} ms")
            print(f"  Max: {stats['max_ms']:.2f} ms")


# Global tracker instance
performance_tracker = PerformanceTracker()
