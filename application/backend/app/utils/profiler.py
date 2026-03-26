"""
Profiler
Code profiling utilities
"""

import time
import functools
from typing import Callable
from app.core.logging import logger


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile function execution time
    
    Usage:
        @profile_function
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        logger.debug(f"{func.__name__} took {elapsed*1000:.2f}ms")
        
        return result
    
    return wrapper


class CodeProfiler:
    """
    Context manager for profiling code blocks
    
    Usage:
        with CodeProfiler("my_operation"):
            # code to profile
            pass
    """
    
    def __init__(self, name: str, log: bool = True):
        self.name = name
        self.log = log
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            
            if self.log:
                logger.debug(f"{self.name} took {self.elapsed*1000:.2f}ms")
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds"""
        return self.elapsed if self.elapsed else 0.0


class Timer:
    """
    Simple timer class
    
    Usage:
        timer = Timer()
        timer.start()
        # do work
        print(f"Elapsed: {timer.stop()}")
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start timer"""
        self.start_time = time.time()
        return self
    
    def stop(self) -> float:
        """
        Stop timer and return elapsed time
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        
        self.end_time = time.time()
        return self.end_time - self.start_time
    
    def reset(self):
        """Reset timer"""
        self.start_time = None
        self.end_time = None
    
    def elapsed(self) -> float:
        """Get elapsed time without stopping"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
