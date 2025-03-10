import time
import functools
from collections import defaultdict
ENABLE_PROFILING = True
class TimeProfiler:
    """
    A context manager for accurate time profiling that can be toggled on/off.
    """
    def __init__(self, time_profile, name):
        self.time_profile = time_profile
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        if ENABLE_PROFILING:
            self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if ENABLE_PROFILING and self.start_time is not None:
            end_time = time.time()
            self.time_profile[self.name] += (end_time - self.start_time)

def time_profile_decorator(time_profile_dict):
    """
    A decorator for timing functions.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            time_profile_dict[f"{func.__name__}"] += (end_time - start_time)
            return result
        return wrapper
    return decorator