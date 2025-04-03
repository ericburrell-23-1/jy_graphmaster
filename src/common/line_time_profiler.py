import sys
import inspect
import time
import os
from functools import wraps
from collections import defaultdict

class HierarchicalProfiler:
    def __init__(self, output_filename, project_path=None):
        """
        Initialize the profiler.
        
        Args:
            output_filename: Base filename for the output
            project_path: Path to your project root. Used to distinguish your code from libraries.
                If None, uses current working directory.
        """
        self.filename = output_filename
        if '.' in self.filename:
            self.filename = self.filename.split('.')[0]
            
        self.project_path = project_path or os.getcwd()
        self.project_path = self.project_path.lower()  # Case-insensitive comparison
        
        # Maps (filename, line_no) -> hierarchical timing data (in seconds)
        self.line_times = defaultdict(float)
        
        # Maps (filename, line_no) -> call hierarchy information
        self.call_tree = {}
        
        # Current call stack during profiling
        self.call_stack = []
        
        # Tracks visited files to avoid redundant checks
        self.is_project_file_cache = {}
        
        # Store source code lines for all files
        self.source_lines = {}
        
        # Store previous trace function to properly handle nested calls
        self.prev_trace_func = None
        
        # Use a global start time for the entire profiling session
        self.profiling_start_time = time.time()
        
    def is_project_file(self, filename):
        """Check if a file is part of the project (not a library)."""
        if filename in self.is_project_file_cache:
            return self.is_project_file_cache[filename]
        
        result = False
        try:
            norm_filename = os.path.normpath(filename).lower()
            result = norm_filename.startswith(self.project_path)
        except:
            pass
            
        self.is_project_file_cache[filename] = result
        return result
        
    def profile_function(self, func):
        """Profile a specific function."""
        wrapped = self._wrap_function(func)
        return wrapped
    
    def _get_source_lines(self, filename):
        """Get source lines for a file, with caching."""
        if filename not in self.source_lines:
            try:
                with open(filename, 'r') as f:
                    self.source_lines[filename] = f.readlines()
            except:
                self.source_lines[filename] = []
        return self.source_lines[filename]
        
    def _wrap_function(self, func):
        """Wrap a function to track execution with hierarchical timing."""
        profiler = self  # Capture self reference
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function information
            try:
                filename = inspect.getfile(func)
                func_name = func.__name__
            except:
                # Handle built-in or otherwise non-inspectable functions
                filename = "<unknown>"
                func_name = str(func)
                
            # Don't profile if not a project file
            if not profiler.is_project_file(filename):
                return func(*args, **kwargs)
            
            # Get function line range to identify calls within this function
            try:
                source_lines, start_line_no = inspect.getsourcelines(func)
                end_line_no = start_line_no + len(source_lines) - 1
            except:
                start_line_no = 0
                end_line_no = 0
                
            # Save the current trace function to restore later
            old_trace = sys.gettrace()
            
            # Function-specific call stack to avoid issues with recursion and nested calls
            local_call_stack = []
            
            # Create a trace function to track line execution
            def trace(frame, event, arg):
                if event not in ('line', 'return', 'call'):
                    return trace
                    
                current_file = frame.f_code.co_filename
                
                # Only trace project files
                if not profiler.is_project_file(current_file):
                    return trace
                
                current_line = frame.f_lineno
                now = time.time()  # Time is recorded in seconds
                
                if event == 'line':
                    # Record timing for the previous line if there's something in the local stack
                    if local_call_stack:
                        prev_key, prev_time = local_call_stack[-1]
                        elapsed = now - prev_time  # Elapsed time in seconds
                        profiler.line_times[prev_key] += elapsed
                    
                    # Update the current line    
                    line_key = (current_file, current_line)
                    local_call_stack.append((line_key, now))
                    
                    # Update call tree
                    if len(local_call_stack) > 1:
                        caller_key = local_call_stack[-2][0]
                        if line_key not in profiler.call_tree:
                            profiler.call_tree[line_key] = {
                                'callers': set([caller_key]),
                                'is_function_start': False
                            }
                        else:
                            profiler.call_tree[line_key]['callers'].add(caller_key)
                    
                elif event == 'return':
                    # Handle return from a function
                    if local_call_stack:
                        last_key, last_time = local_call_stack.pop()
                        elapsed = now - last_time  # Elapsed time in seconds
                        profiler.line_times[last_key] += elapsed
                
                return trace
            
            # Add function entry to call stack
            func_entry_time = time.time()
            line_key = (filename, start_line_no)
            
            # Initialize local stack
            local_call_stack = [(line_key, func_entry_time)]
            
            # Mark this line as a function start
            if line_key not in profiler.call_tree:
                profiler.call_tree[line_key] = {
                    'callers': set(),
                    'is_function_start': True
                }
            else:
                profiler.call_tree[line_key]['is_function_start'] = True
                
            # Set up tracing
            sys.settrace(trace)
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                return result
            finally:
                # Clean up tracing
                sys.settrace(old_trace)
                
                # Handle any remaining items in the stack
                now = time.time()
                while local_call_stack:
                    line_key, start_time = local_call_stack.pop()
                    elapsed = now - start_time  # Elapsed time in seconds
                    profiler.line_times[line_key] += elapsed
        
        return wrapper
    
    def profile_method(self, instance, method_name):
        """Profile a specific method on an instance."""
        method = getattr(instance, method_name)
        wrapped = self.profile_function(method)
        setattr(instance, method_name, wrapped)
        return wrapped
        
    def write_results(self):
        """Write profiling results to a file with hierarchical structure."""
        with open(f"{self.filename}.txt", 'w') as f:
            f.write("Hierarchical Cumulative Time Profile (Sorted by Time)\n")
            f.write("================================================\n\n")
            
            # Calculate total time - this is the elapsed time since profiling started
            total_time = time.time() - self.profiling_start_time
            
            # Calculate sum of all line times for percentage calculations
            sum_line_times = sum(self.line_times.values())
            
            f.write(f"Total execution time: {total_time:.4f} seconds\n\n")
            
            # Group by file
            by_file = defaultdict(list)
            for (filename, line_no), time_taken in self.line_times.items():
                by_file[filename].append((line_no, time_taken))
            
            # Process each file
            for filename, lines in sorted(by_file.items(), 
                                         key=lambda x: sum(l[1] for l in x[1]), 
                                         reverse=True):
                file_total = sum(time for _, time in lines)
                file_percent = (file_total / sum_line_times * 100) if sum_line_times > 0 else 0
                
                # Skip files with very small contribution
                if file_percent < 0.1:
                    continue
                    
                f.write(f"File: {filename}\n")
                f.write(f"Total time: {file_total:.4f} seconds ({file_percent:.1f}%)\n\n")
                
                # Get source lines if available
                source_lines = self._get_source_lines(filename)
                
                # Identify function boundaries
                func_starts = {}
                for (file, line), info in self.call_tree.items():
                    if file == filename and info['is_function_start']:
                        func_starts[line] = True
                
                # Sort lines by time (highest first)
                lines.sort(key=lambda x: x[1], reverse=True)
                
                # Write header
                f.write("Line #      Time(s)    % Total   % File    Line Contents\n")
                f.write("================================================================\n")
                
                # Group unknown lines (if any)
                unknown_time = 0.0
                known_lines = []
                
                for line_no, time_taken in lines:
                    if line_no < 1 or (source_lines and line_no > len(source_lines)):
                        unknown_time += time_taken
                    else:
                        known_lines.append((line_no, time_taken))
                
                # Write unknown lines total if any
                if unknown_time > 0:
                    percent_total = (unknown_time / sum_line_times * 100) if sum_line_times > 0 else 0
                    percent_file = (unknown_time / file_total * 100) if file_total > 0 else 0
                    f.write(f"Unknown {unknown_time:10.4f} {percent_total:8.1f} {percent_file:8.1f}  All Unknown Lines (combined)\n")
                
                # Write each known line
                for line_no, time_taken in known_lines:
                    # Determine if this is a function start
                    is_func_start = line_no in func_starts
                    
                    # Calculate percentages
                    percent_total = (time_taken / sum_line_times * 100) if sum_line_times > 0 else 0
                    percent_file = (time_taken / file_total * 100) if file_total > 0 else 0
                    
                    # Get line content
                    line_content = ""
                    if 0 <= line_no - 1 < len(source_lines):
                        line_content = source_lines[line_no - 1].rstrip()
                        if not line_content:
                            line_content = "<empty line>"
                    
                    # Add an indicator for function starts
                    prefix = "DEF " if is_func_start else "    "
                    
                    # Format line - time is already in seconds, just format for display
                    f.write(f"{line_no:8d} {time_taken:10.4f} {percent_total:8.1f} {percent_file:8.1f}  {prefix}{line_content}\n")
                
                f.write("\n\n")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.write_results()
        print(f"Profiling results written to {self.filename}.txt")