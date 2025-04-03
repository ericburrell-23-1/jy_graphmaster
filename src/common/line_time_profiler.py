import line_profiler
import inspect
import os
import io
from functools import wraps
from operator import itemgetter

class LineTimeProfiler:
    def __init__(self, output_filename):
        self.filename = output_filename
        if '.' in self.filename:
            self.filename = self.filename.split('.')[0]
        self.line_profiler = line_profiler.LineProfiler()
        self.functions_to_profile = []
        self.unit = 1.0  # Changed to 1.0 for seconds instead of microseconds
    
    def add_function(self, func):
        """Add a function to be profiled."""
        self.line_profiler.add_function(func)
        self.functions_to_profile.append(func)
        return func
    
    def add_module(self, module):
        """Profile all functions in a module."""
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                self.add_function(obj)
    
    def add_class(self, cls):
        """Profile all methods in a class."""
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            self.add_function(method)
    
    def __enter__(self):
        # Enable the profiler
        self.line_profiler.enable_by_count()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # Disable the profiler
        self.line_profiler.disable_by_count()
        
        # Get the stats
        stats = self.line_profiler.get_stats()
        
        # Custom formatting of results with time in seconds
        self.write_sorted_stats(stats.timings)
        
        print(f"Line profiling results written to {self.filename}.txt")
    
    def write_sorted_stats(self, timings):
        """Write stats to file with time in seconds and sorted by total time."""
        with open(self.filename + '.txt', 'w+') as f:
            # Process each function's stats
            all_lines = []
            
            for (fn, lineno, name), timings_dict in timings.items():
                # Get the source lines
                with open(fn, 'r') as src_file:
                    all_src_lines = src_file.readlines()
                
                # Calculate total time for this function (in seconds)
                func_total = sum(item[2] for item in timings_dict) / 1000000.0  # Convert microseconds to seconds
                
                f.write(f"Timer unit: seconds\n\n")
                f.write(f"File: {fn}\n")
                f.write(f"Function: {name} at line {lineno}\n")
                f.write(f"Total time: {func_total:.6f} s\n\n")
                
                # Prepare the line data
                line_data = []
                for line_stats in timings_dict:
                    line_no, hits, time_us = line_stats
                    if line_no > 0 and line_no <= len(all_src_lines):
                        line_content = all_src_lines[line_no-1].rstrip()
                        time_s = time_us / 1000000.0  # Convert to seconds
                        per_hit_s = time_s / hits if hits > 0 else 0
                        percent = (time_s / func_total * 100) if func_total > 0 else 0
                        
                        line_data.append((line_no, hits, time_s, per_hit_s, percent, line_content))
                
                # Sort by total time (descending)
                line_data.sort(key=itemgetter(2), reverse=True)
                
                # Write header
                f.write("Line #      Hits         Time(s)  Per Hit(s)   % Time  Line Contents\n")
                f.write("==============================================================\n")
                
                # Write the sorted line data
                for line_no, hits, time_s, per_hit_s, percent, line_content in line_data:
                    f.write(f"{line_no:8d} {hits:9d} {time_s:12.6f} {per_hit_s:10.6f} {percent:8.1f}  {line_content}\n")
                
                f.write("\n\n")

# Decorator for profiling specific functions
def profile_lines(profiler):
    def decorator(func):
        profiler.add_function(func)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator